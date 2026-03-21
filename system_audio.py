"""System audio capture via ScreenCaptureKit (macOS 12.3+).

Replaces BlackHole as the system audio source — no virtual device needed,
no Multi-Output Device setup, volume keys work normally.

Requirements
------------
    pip install pyobjc-framework-ScreenCaptureKit

On first use macOS shows a one-time Screen Recording permission dialog.
After granting it the capture starts automatically on every subsequent run.
"""

import ctypes
import threading

import numpy as np

try:
    import objc
    from Foundation import NSObject
    import ScreenCaptureKit as _SCKit
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


# ── ScreenCaptureKit helpers (metadata overrides + CoreMedia ctypes) ─────────

if _AVAILABLE:
    # PyObjC metadata fix: the runtime calls this block with 3 args
    # (content, error, extra_flag) but the framework stub declares only 2.
    # Arg 3 is not an ObjC object — declare as ^v (void*) to avoid retain crash.
    objc.registerMetaDataForSelector(
        b"SCShareableContent",
        b"getShareableContentWithCompletionHandler:",
        {
            "arguments": {
                2: {  # index 2 = first real param (0=self, 1=_cmd, 2=handler)
                    "callable": {
                        "retval": {"type": b"v"},
                        "arguments": {
                            0: {"type": b"^v"},  # block literal pointer (hidden)
                            1: {"type": b"@"},   # SCShareableContent
                            2: {"type": b"@"},   # NSError
                            3: {"type": b"^v"},  # extra arg (raw int, not an object)
                        },
                    }
                }
            }
        },
    )

    # PyObjC metadata fix: the framework declares the CMSampleBufferRef arg as
    # ^{opaqueCMSampleBuffer=} which PyObjC can't bridge.  Declaring it as ^v
    # (void*) gives us the raw pointer as a Python integer for ctypes access.
    objc.registerMetaDataForSelector(
        b"NSObject",
        b"stream:didOutputSampleBuffer:ofType:",
        {
            "required": False,
            "retval": {"type": b"v"},
            "arguments": {
                2: {"type": b"@"},   # SCStream
                3: {"type": b"^v"},  # CMSampleBufferRef → raw pointer
                4: {"type": b"q"},   # SCStreamOutputType (NSInteger)
            },
        },
    )

    # ── CoreMedia ctypes bindings ────────────────────────────────────────────

    _CM = ctypes.CDLL("/System/Library/Frameworks/CoreMedia.framework/CoreMedia")

    # CMBlockBufferRef CMSampleBufferGetDataBuffer(CMSampleBufferRef sbuf)
    _CM.CMSampleBufferGetDataBuffer.restype = ctypes.c_void_p

    # OSStatus CMBlockBufferGetDataPointer(
    #     CMBlockBufferRef buf, size_t offset,
    #     size_t *lengthAtOffsetOut,   <- pass None to skip
    #     size_t *totalLengthOut,
    #     char   **dataPointerOut)
    _CM.CMBlockBufferGetDataPointer.restype  = ctypes.c_int32
    _CM.CMBlockBufferGetDataPointer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_char_p),
    ]

    def _pcm_from_sample_buffer(sample_buffer: int) -> "np.ndarray | None":
        """Return a float32 numpy array from a CMSampleBufferRef pointer, or None on failure."""
        sb_ptr = ctypes.c_void_p(sample_buffer)
        bb = _CM.CMSampleBufferGetDataBuffer(sb_ptr)
        if not bb:
            return None

        total_len = ctypes.c_size_t(0)
        data_ptr  = ctypes.c_char_p()
        if _CM.CMBlockBufferGetDataPointer(
            ctypes.c_void_p(bb), 0, None,
            ctypes.byref(total_len),
            ctypes.byref(data_ptr),
        ) != 0 or not data_ptr or total_len.value == 0:
            return None

        # Copy immediately — the block buffer memory is owned by SCKit and is
        # only valid for the duration of this callback invocation.
        return np.frombuffer(
            ctypes.string_at(data_ptr, total_len.value), dtype=np.float32
        ).copy()

    # ── SCStreamOutput delegate ──────────────────────────────────────────────

    class _AudioOut(NSObject):
        """Receives audio sample buffers from SCStream."""
        _capture_ref = objc.ivar()

        # Belt-and-suspenders: explicit ObjC type encoding for the delegate method.
        # The registerMetaDataForSelector above handles dispatch, but this ensures
        # the selector is registered with the correct signature on the class itself.
        @objc.typedSelector(b"v@:@^vq")
        def stream_didOutputSampleBuffer_ofType_(self_d, stream, sb, kind):
            capture = self_d._capture_ref
            if capture is None or capture._stop.is_set() or kind != 1:
                return
            pcm = _pcm_from_sample_buffer(sb)
            if pcm is None or len(pcm) == 0:
                return
            if capture._callback:
                try:
                    capture._callback(pcm.reshape(-1, 1), len(pcm), None, None)
                except Exception:
                    pass  # absorb sd.CallbackAbort and other signals


# ── Public helpers ────────────────────────────────────────────────────────────

def is_available() -> bool:
    """True if pyobjc-framework-ScreenCaptureKit is installed and importable."""
    return _AVAILABLE


# ── Capture class ─────────────────────────────────────────────────────────────

class SystemAudioCapture:
    """Capture all system audio using ScreenCaptureKit.  No BlackHole needed.

    The callback signature matches sounddevice.InputStream:
        callback(indata: np.ndarray, frames: int, time, status)
        indata shape: (N, 1), dtype float32, sample_rate as configured.

    Usage
    -----
        sck = SystemAudioCapture(sample_rate=16000, callback=my_fn)
        sck.start()   # blocks until streaming begins (or raises on error)
        ...
        sck.stop()
    """

    def __init__(self, sample_rate: int = 16000, callback=None):
        if not _AVAILABLE:
            raise RuntimeError(
                "pyobjc-framework-ScreenCaptureKit not installed.\n"
                "Fix:  pip install pyobjc-framework-ScreenCaptureKit"
            )
        self._rate     = sample_rate
        self._callback = callback
        self._stream   = None
        self._delegate = None   # held to prevent garbage collection
        self._ready    = threading.Event()
        self._stop     = threading.Event()
        self._err: "str | None" = None

    # ── Public interface ──────────────────────────────────────────────────────

    def start(self):
        """Set up ScreenCaptureKit and block until audio streaming begins."""
        threading.Thread(target=self._setup, daemon=True).start()
        if not self._ready.wait(timeout=10):
            raise RuntimeError(
                "ScreenCaptureKit timed out.\n"
                "Grant Screen Recording permission: "
                "System Settings → Privacy & Security → Screen Recording"
            )
        if self._err:
            raise RuntimeError(f"ScreenCaptureKit: {self._err}")

    def stop(self):
        """Stop capture and release the SCStream."""
        self._stop.set()
        if self._stream:
            done = threading.Event()
            self._stream.stopCaptureWithCompletionHandler_(lambda _: done.set())
            done.wait(timeout=2)  # give SCKit up to 2s to clean up
            self._stream = None
        self._delegate = None  # release delegate

    # ── Internal setup ────────────────────────────────────────────────────────

    def _setup(self):
        delegate = _AudioOut.alloc().init()
        delegate._capture_ref = self
        self._delegate = delegate      # prevent GC

        # ── Step 1: get shareable content (async, GCD callback) ───────────────
        # Use objc.lookUpClass() to bypass the framework's metadict — our
        # registerMetaDataForSelector override is only consulted via lookUpClass.
        _SCShareableContent = objc.lookUpClass("SCShareableContent")

        def on_content(content, error, _):
            if error:
                self._err = error.localizedDescription()
                self._ready.set()
                return

            if content is None:
                self._err = "no SCShareableContent returned"
                self._ready.set()
                return

            displays = content.displays()
            if not displays or len(displays) == 0:
                self._err = "no display found (SCContentFilter requires a display)"
                self._ready.set()
                return

            # ── Step 2: configure stream ──────────────────────────────────────
            filt = _SCKit.SCContentFilter.alloc().initWithDisplay_excludingWindows_(
                displays[0], []
            )
            cfg = _SCKit.SCStreamConfiguration.alloc().init()
            cfg.setCapturesAudio_(True)
            cfg.setExcludesCurrentProcessAudio_(False)
            cfg.setSampleRate_(float(self._rate))
            cfg.setChannelCount_(1)

            stream = _SCKit.SCStream.alloc().initWithFilter_configuration_delegate_(
                filt, cfg, None
            )

            # addStreamOutput:type:sampleHandlerQueue:error:
            # Framework metadata does NOT mark error: as an out-param,
            # so we must pass all 4 args explicitly (returns just BOOL).
            ok = stream.addStreamOutput_type_sampleHandlerQueue_error_(
                delegate,
                1,      # SCStreamOutputTypeAudio
                None,   # sampleHandlerQueue — None uses SCKit's internal queue
                None,   # error — not an out-param in this framework's metadata
            )
            if not ok:
                self._err = "addStreamOutput failed"
                self._ready.set()
                return

            self._stream = stream

            # ── Step 3: start the stream ──────────────────────────────────────
            def on_started(err):
                if err is not None and hasattr(err, "localizedDescription"):
                    self._err = err.localizedDescription()
                self._ready.set()

            stream.startCaptureWithCompletionHandler_(on_started)

        _SCShareableContent.getShareableContentWithCompletionHandler_(on_content)

        # Keep this thread alive until stop() is called.
        # All SCKit callbacks fire on GCD queues — no NSRunLoop needed.
        self._stop.wait()
