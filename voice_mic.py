"""Microphone capture via Apple's VoiceProcessingIO AudioUnit (macOS 10.14+).

Replaces sounddevice InputStream for the mic channel. Voice processing performs
hardware acoustic echo cancellation at the audio layer — the captured stream
has system playback bleed removed before it leaves the unit, so downstream code
no longer needs transcript-level dedup.

Uses AVAudioEngine.installTap with setVoiceProcessingEnabled. Apple's official
high-level API; the alternative (ctypes + AudioUnitRender + render-notify
callbacks) starves the GIL because every audio callback grabs it.

The tap delivers 3 channels: [0] = AEC-processed mic, [1] = raw mic (pre-AEC),
[2] = reference signal (what was sent to the speakers). We keep only [0].

Setup
-----
    pip install pyobjc-framework-AVFoundation
"""

import sys
import threading
import time
from typing import List, Optional

import numpy as np

try:
    import AVFoundation
    import objc
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


# Target sample rate the rest of the pipeline expects. The AVAudioEngine tap
# delivers at the hardware's preferred rate (~24 kHz on Mac built-in mic);
# we resample to this rate on a background thread so callers always see 16 kHz.
_DEFAULT_TARGET_RATE = 16000


class VPMicCapture:
    """AVAudioEngine-based mic capture with hardware acoustic echo cancellation.

    The callback signature matches sounddevice.InputStream for drop-in use:
        callback(indata: np.ndarray, frames: int, time, status)
        indata shape: (N, 1), dtype float32, sample_rate=16000.

    Usage
    -----
        vp = VPMicCapture(sample_rate=16000, callback=my_fn)
        vp.start()
        ...
        vp.read_chunk(timeout=0.1)   # → (frames, overflow_flag)
        ...
        vp.stop()
    """

    def __init__(self, sample_rate: int = _DEFAULT_TARGET_RATE, callback=None):
        if not _AVAILABLE:
            raise RuntimeError(
                "pyobjc-framework-AVFoundation not installed.\n"
                "Fix:  pip install pyobjc-framework-AVFoundation"
            )
        self._target_rate = sample_rate
        self._callback = callback

        self._engine = None
        self._input_node: "AVFoundation.AVAudioInputNode | None" = None
        self._tap_bus = 0
        self._source_rate = 0

        # Resampled (16 kHz) mono frames, ready for the rest of the pipeline.
        self._frames: List[np.ndarray] = []
        self._frames_lock = threading.Lock()

        self._started = False
        self._stopped = False
        self._overflow = 0
        self._err: "str | None" = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Set up the AVAudioEngine, enable voice processing, install the tap, start."""
        if self._started:
            return
        self._engine = AVFoundation.AVAudioEngine.alloc().init()
        self._input_node = self._engine.inputNode()

        # Enable Apple's hardware acoustic echo cancellation. This is the entire
        # reason this module exists — without it, mic picks up the speakers'
        # playback and ASR transcribes both.
        ok, err = self._input_node.setVoiceProcessingEnabled_error_(True, None)
        if not ok:
            err_str = err.localizedDescription() if err is not None else "unknown"
            raise RuntimeError(f"setVoiceProcessingEnabled failed: {err_str}")

        # Tap on the input bus. Apple's VPIO returns 3 channels:
        #   ch0: AEC-processed mic (this is what we want)
        #   ch1: raw mic (pre-AEC, for diagnostics)
        #   ch2: reference signal (what the speakers played)
        # We extract ch0 in the tap callback.
        fmt = self._input_node.outputFormatForBus_(self._tap_bus)
        if fmt is None:
            raise RuntimeError("inputNode.outputFormatForBus returned nil")
        self._source_rate = int(fmt.sampleRate())

        self._input_node.installTapOnBus_bufferSize_format_block_(
            self._tap_bus,
            4096,           # 4096-frame buffers (~170 ms at 24 kHz)
            fmt,
            self._tap_block,
        )

        self._engine.prepare()
        ok, err = self._engine.startAndReturnError_(None)
        if not ok:
            err_str = err.localizedDescription() if err is not None else "unknown"
            raise RuntimeError(f"AVAudioEngine.start failed: {err_str}")
        self._started = True
        self._stopped = False

    def stop(self) -> None:
        """Stop the engine and remove the tap. Idempotent and thread-safe."""
        if self._stopped:
            return
        self._stopped = True
        if self._input_node is not None:
            try:
                self._input_node.removeTapOnBus_(self._tap_bus)
            except Exception:
                pass
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass
        self._engine = None
        self._input_node = None

    def read_chunk(self, timeout: float = 0.1):
        """Return the next buffered block of 16 kHz mono audio.

        Returns (frames, overflow). frames has shape (N,). Returns (None, False)
        on timeout. May return short frames.
        """
        if not self._started:
            raise RuntimeError("VPMicCapture not started — call start() first")
        deadline = time.time() + timeout
        with self._frames_lock:
            if self._frames:
                chunk = self._frames.pop(0)
                return chunk, False
        remaining = max(0.0, deadline - time.time())
        if remaining > 0:
            time.sleep(remaining)
        with self._frames_lock:
            if self._frames:
                chunk = self._frames.pop(0)
                return chunk, False
        return None, False

    def drain(self) -> np.ndarray:
        """Drain all buffered frames and return as a single 1-D array."""
        with self._frames_lock:
            if not self._frames:
                return np.zeros(0, dtype=np.float32)
            out = np.concatenate(self._frames).astype(np.float32)
            self._frames.clear()
            return out

    @property
    def overflow(self) -> int:
        return self._overflow

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ── Internal: tap callback (block) ────────────────────────────────────────

    def _tap_block(self, buf, when):
        """Block called by AVAudioEngine on a real-time audio thread.

        We extract channel 0 (the AEC-processed mic), resample to target rate,
        and hand off to the consumer via the callback or the read buffer.
        No blocking work is done here.
        """
        try:
            if buf is None:
                return
            frame_count = int(buf.frameLength())
            if frame_count <= 0:
                return

            # floatChannelData gives us per-channel float32 buffers. VPIO
            # delivers 3 channels: [0] AEC-processed, [1] raw, [2] reference.
            # We want [0].
            data_ptrs = buf.floatChannelData()
            ch0 = data_ptrs[0]
            # ch0 is a PyObjC "varlist" wrapping an UnsafePointer<Float>.
            # as_buffer(n_bytes) gives a memoryview of the first n_bytes —
            # which is exactly frame_count × sizeof(Float32).
            raw = np.frombuffer(
                ch0.as_buffer(frame_count * 4),
                dtype=np.float32,
                count=frame_count,
            ).copy()

            # Resample source rate → 16 kHz.
            if self._source_rate != self._target_rate:
                out = _linear_resample(raw, self._source_rate, self._target_rate)
            else:
                out = raw.astype(np.float32, copy=True)

            with self._frames_lock:
                self._frames.append(out)

            if self._callback is not None:
                try:
                    self._callback(out.reshape(-1, 1), out.shape[0], None, None)
                except Exception as e:
                    # Callback runs in audio thread — never let user code crash us.
                    sys.stderr.write(f"vp-mic callback error: {e}\n")
        except Exception as e:
            sys.stderr.write(f"vp-mic tap error: {e}\n")


# ── Resampler ────────────────────────────────────────────────────────────────

def _linear_resample(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resample (mono float32).

    AVAudioEngine's VPIO has already bandlimited the output to voice frequencies
    (≤8 kHz) — we're just re-spacing the samples. Linear interpolation is
    audibly transparent at this stage and ~10× cheaper than polyphase.
    """
    if x.size == 0 or src_rate == dst_rate:
        return x.astype(np.float32, copy=True)
    n_out = int(round(x.size * dst_rate / src_rate))
    if n_out == 0:
        return np.zeros(0, dtype=np.float32)
    t_in = np.arange(x.size, dtype=np.float64)
    t_out = np.linspace(0, x.size - 1, n_out, dtype=np.float64)
    return np.interp(t_out, t_in, x).astype(np.float32)
