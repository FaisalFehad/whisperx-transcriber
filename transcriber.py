#!/usr/bin/env python3
"""
Interview Transcriber — Record, transcribe, and save to Obsidian.

Commands:
    transcribe rec                  Start recording (auto-stops on silence)
    transcribe run [file]           Transcribe a recording
    transcribe live                 Record + transcribe simultaneously
    transcribe watch                Auto-record meetings from calendar
    transcribe list                 List available recordings
    transcribe enroll               Save your voice for auto-recognition
    transcribe setup                Audio device setup guide
"""

import argparse
import gc
import json
import os
import re
import select
import subprocess
import sys
import termios
import traceback
import threading
import time
import tty
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

# Suppress torchcodec warnings
warnings.filterwarnings("ignore", message=".*torchcodec.*")


# ─── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"

DEFAULT_CONFIG = {
    # ── General ──────────────────────────────────────────────────────────
    # Whisper model: tiny/base/small/medium/large-v3
    # Recommended: "small" — best balance of speed and accuracy on M1 16GB
    "default_model": "small",
    # Language code for transcription (e.g. "en", "ar", "fr")
    # Set to avoid misdetection — auto-detect can pick wrong language
    "language": "en",
    # Quantization: "int8" recommended for Apple Silicon (fastest, low RAM)
    # Options: "int8", "float16", "float32"
    "compute_type": "int8",
    # Processing device: "cpu" for Apple Silicon (MPS not yet supported by WhisperX)
    "device": "cpu",

    # ── Batch sizes per model ────────────────────────────────────────────
    # Higher = faster but more RAM. These are tuned for M1 16GB.
    # Reduce if you get memory errors; increase on machines with more RAM.
    "batch_sizes": {
        "tiny": 32, "base": 24, "small": 16, "medium": 8, "large-v3": 4
    },

    # ── File paths ───────────────────────────────────────────────────────
    "paths": {
        # Root folder in Obsidian vault for all interview data
        "obsidian_base": "~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Interviews",
        # Subfolders under obsidian_base (created automatically)
        "recordings_subdir": "Recordings",
        "scripts_subdir": "Scripts",
    },

    # ── Recording settings ───────────────────────────────────────────────
    "recording": {
        # Sample rate in Hz — 16000 is what Whisper was trained on (don't change)
        "sample_rate": 16000,
        # RMS level below this = silence (0.005 works for most mics)
        # Lower = more sensitive, higher = ignores quiet speech
        "silence_threshold": 0.005,
        # Minutes of continuous silence before auto-stopping recording
        "silence_timeout_minutes": 10,
        # Virtual audio device name for system audio capture
        # Change if using BlackHole 16ch or another virtual driver
        "blackhole_device": "BlackHole 2ch",
        # Seconds of low mic level before showing warning
        "mic_low_warning_seconds": 10,
    },

    # Accepted audio file types for transcription and listing
    "audio_formats": [".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"],

    # ── Speaker diarization (who said what) ──────────────────────────────
    "diarization": {
        # Set to false to always skip diarization (faster, no speaker labels)
        "enabled": True,
        # Pyannote model — requires accepting terms at huggingface.co
        "model": "pyannote/speaker-diarization-3.1",
    },

    # ── Live mode (record + transcribe simultaneously) ───────────────────
    "live": {
        # Seconds between transcription chunks during recording
        # Lower = more real-time but more CPU. Recommended: 120
        "chunk_interval_seconds": 120,
        # Adaptive scaling bounds (auto-adjusted based on CPU load)
        "min_chunk_interval": 60,       # fastest: every 1 min
        "max_chunk_interval": 300,      # slowest: every 5 min
        # Overlap between chunks for context continuity (seconds)
        "chunk_overlap_seconds": 5,
        # Minimum new audio (seconds) before a chunk is worth transcribing
        "min_chunk_seconds": 30,
        # Adaptive thresholds — ratio of processing_time / audio_time
        # > struggle_ratio: slow down chunking + reduce batch size
        "struggle_ratio": 0.7,
        # > rec_only_ratio: stop transcribing entirely, just record
        # Transcription runs after call ends instead
        "rec_only_ratio": 1.5,
    },

    # ── Speaker memory (voice recognition) ───────────────────────────────
    "speaker_memory": {
        # Auto-match voices against saved profiles
        "enabled": True,
        # Cosine similarity threshold (0-1). Higher = stricter matching.
        # 0.75 recommended — avoids false matches while catching most voices
        "similarity_threshold": 0.75,
        # Pyannote embedding model for voice fingerprinting
        "embedding_model": "pyannote/embedding",
        # How long to record during 'transcribe enroll' (seconds)
        "enrollment_duration_seconds": 15,
    },

    # ── Calendar watch (auto-record meetings) ────────────────────────────
    "watch": {
        # Silence timeout for watch recordings (same as recording default)
        "silence_timeout_minutes": 10,
        # Discard recordings shorter than this (catches non-meeting events)
        "min_recording_minutes": 2,
        # Model to use for watch auto-transcription
        "model": "small",
        # Only watch these calendars (empty = all calendars)
        # Example: ["Work", "Personal"]
        "calendars": [],
        # Keep recording N minutes after calendar event ends
        # Catches meetings that run over
        "end_buffer_minutes": 2,
        # Re-read calendar every N hours (catches newly added meetings)
        "refresh_hours": 3,
    },

    # ── List command ─────────────────────────────────────────────────────
    "list": {
        # Max recordings to show in 'transcribe list'
        "max_recordings": 20,
    },

    # Your name — set via 'transcribe enroll' or manually here
    # Used as the display name when your voice is recognized
    "user_name": "",

    # ── Model definitions ────────────────────────────────────────────────
    # speed_factor: multiplier for time estimation (audio_duration × factor)
    # Lower = faster model. These are calibrated for M1 Apple Silicon.
    "models": {
        "tiny":     {"speed_factor": 0.1,  "description": "fastest, least accurate",    "size": "~75MB"},
        "base":     {"speed_factor": 0.2,  "description": "fast, good for clear audio", "size": "~140MB"},
        "small":    {"speed_factor": 0.5,  "description": "balanced (recommended)",     "size": "~460MB"},
        "medium":   {"speed_factor": 1.5,  "description": "slow, more accurate",        "size": "~1.5GB"},
        "large-v3": {"speed_factor": 4.0,  "description": "slowest, most accurate",     "size": "~3GB"},
    },
}


def _deep_merge(base, override):
    """Merge override dict into base dict recursively."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config():
    """Load config from config.json, falling back to defaults."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                user_config = json.load(f)
            return _deep_merge(DEFAULT_CONFIG, user_config)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ⚠  Error reading config.json: {e}")
            print("     Using default settings.")
    return DEFAULT_CONFIG.copy()


# Load config and apply
config = load_config()

HF_TOKEN = os.environ.get("HF_TOKEN", "")
OBSIDIAN_BASE = Path(os.path.expanduser(config["paths"]["obsidian_base"]))
RECORDINGS_DIR = OBSIDIAN_BASE / config["paths"]["recordings_subdir"]
SCRIPTS_DIR = OBSIDIAN_BASE / config["paths"]["scripts_subdir"]

SAMPLE_RATE = config["recording"]["sample_rate"]
SILENCE_THRESHOLD = config["recording"]["silence_threshold"]
SILENCE_TIMEOUT = config["recording"]["silence_timeout_minutes"] * 60

AUDIO_FORMATS = tuple(config["audio_formats"])
BLACKHOLE_DEVICE_NAME = config["recording"]["blackhole_device"]
MIC_LOW_WARNING_SECONDS = config["recording"]["mic_low_warning_seconds"]
DEVICE = config.get("device", "cpu")
SPEAKERS_DIR = SCRIPT_DIR / "speakers"

MODEL_INFO = config["models"]
STEP_NAMES = ["Loading model", "Transcribing", "Aligning timestamps", "Identifying speakers", "Saving"]


# ─── Display Helpers ──────────────────────────────────────────────────────────

def clear_line():
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


def format_duration(seconds):
    """Format seconds as HH:MM:SS or MM:SS."""
    if seconds < 0:
        return "--:--"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def level_meter(rms, width=20):
    """Create a visual audio level meter."""
    level = min(1.0, rms * 15)
    filled = int(level * width)
    return "█" * filled + "░" * (width - filled)


def spinner_char(tick):
    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    return chars[tick % len(chars)]


def quality_label(rms):
    """Return audio quality label based on RMS level."""
    if rms > 0.1:
        return "Hot!  "
    elif rms > 0.02:
        return "Good  "
    elif rms > 0.005:
        return "Low   "
    elif rms > 0.001:
        return "Quiet "
    else:
        return "Silent"


# ─── Live Panel ──────────────────────────────────────────────────────────────

class LivePanel:
    """Interactive terminal UI with keyboard controls for live recording."""

    HELP_TEXT = [
        "  ┌─── Keyboard Shortcuts ──────────────────────┐",
        "  │  [Space]  Pause / Resume transcription       │",
        "  │  [b]      Drop bookmark at current time      │",
        "  │  [!]      Flag moment for review             │",
        "  │  [+]      Increase chunk frequency           │",
        "  │  [-]      Decrease chunk frequency           │",
        "  │  [i]      Toggle info panel                  │",
        "  │  [c]      Show recent transcript             │",
        "  │  [n]      Name a speaker                     │",
        "  │  [q]      Stop recording                     │",
        "  │  [?]      Toggle this help                   │",
        "  └──────────────────────────────────────────────┘",
    ]

    def __init__(self, source_label, mic_name, bh_name, model_name,
                 model_info, adaptive, output_path, silence_timeout):
        self.source_label = source_label
        self.mic_name = mic_name
        self.bh_name = bh_name or "N/A"
        self.model_name = model_name
        self.model_size = model_info.get("size", "?")
        self.adaptive = adaptive
        self.output_path = output_path
        self.silence_timeout = silence_timeout

        self.paused = False
        self.show_info = True
        self.show_help = False
        self.show_transcript = False
        self.bookmarks = []
        self.flags = []
        self.speaker_names_override = []
        self.should_stop = False
        self._low_mic_since = None  # track sustained low mic

        self._prev_line_count = 0
        self._old_term = None
        self._raw = False

    def start(self):
        """Enter cbreak mode for single-char key reads."""
        try:
            self._old_term = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self._raw = True
            sys.stdout.write("\033[?25l")  # hide cursor
            sys.stdout.flush()
        except Exception:
            pass

    def stop(self):
        """Restore terminal to normal mode."""
        sys.stdout.write("\033[?25h\n")  # show cursor
        sys.stdout.flush()
        if self._old_term:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_term)
            except Exception:
                pass
        self._raw = False

    def poll_key(self):
        """Non-blocking single key read. Returns key char or None."""
        if not self._raw:
            return None
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
        except Exception:
            pass
        return None

    def handle_key(self, key, elapsed):
        """Process a keypress."""
        if key == ' ':
            self.paused = not self.paused
        elif key == 'q':
            self.should_stop = True
        elif key == 'b':
            self.bookmarks.append(elapsed)
        elif key == '!':
            self.flags.append(elapsed)
        elif key == 'i':
            self.show_info = not self.show_info
            self._prev_line_count = 0  # force full redraw
        elif key == 'c':
            self.show_transcript = not self.show_transcript
            self._prev_line_count = 0
        elif key == '?':
            self.show_help = not self.show_help
            self._prev_line_count = 0
        elif key == '+':
            self.adaptive["interval"] = max(
                self.adaptive["min_interval"],
                self.adaptive["interval"] - 30
            )
        elif key == '-':
            self.adaptive["interval"] = min(
                self.adaptive["max_interval"],
                self.adaptive["interval"] + 30
            )
        elif key == 'n':
            self._input_speaker_name()

    def _input_speaker_name(self):
        """Switch to cooked mode for text input, then back to cbreak."""
        if self._old_term:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_term)
        sys.stdout.write("\033[?25h")  # show cursor
        try:
            name = input("\n  Speaker name (Enter to cancel): ").strip()
            if name:
                self.speaker_names_override.append(name)
                print(f"  ✅ '{name}' will be applied to transcript")
                time.sleep(0.5)
        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            sys.stdout.write("\033[?25l")
            if self._old_term:
                tty.setcbreak(sys.stdin.fileno())
        self._prev_line_count = 0  # force full redraw

    def render(self, elapsed, mic_rms, bh_rms, silence_elapsed,
              chunks_done, chunk_in_progress, adaptive,
              transcribed_segments, rec_size_bytes):
        """Render the live control panel to terminal."""
        lines = []

        # ── Header ─────────────────────────────────────────
        time_str = format_duration(elapsed)
        pause_tag = "  ⏸ PAUSED" if self.paused else ""
        rec_only_tag = "  ⚠ REC ONLY" if adaptive.get("rec_only") else ""

        lines.append("  ┌─────────────────────────────────────────────────────┐")
        tag = f"{pause_tag}{rec_only_tag}"
        lines.append(f"  │  🎙  Live  {time_str}{tag:<32} [?]  │")
        lines.append("  ├─────────────────────────────────────────────────────┤")

        # ── Audio quality meters ───────────────────────────
        mic_meter = level_meter(mic_rms, width=12)
        mic_qual = quality_label(mic_rms)
        sil_str = f"Sil: {format_duration(silence_elapsed)}/{format_duration(self.silence_timeout)}"
        lines.append(f"  │  🎙  {mic_meter} {mic_qual} {sil_str:<20}│")

        if bh_rms is not None:
            bh_meter = level_meter(bh_rms, width=12)
            bh_qual = quality_label(bh_rms)
            bh_label = self.bh_name[:19]
            lines.append(f"  │  🔊  {bh_meter} {bh_qual} ({bh_label:<19})│")

        # ── Audio quality warning ──────────────────────────
        if mic_rms < 0.001:
            if self._low_mic_since is None:
                self._low_mic_since = time.time()
            elif time.time() - self._low_mic_since > 10:
                lines.append("  │  ⚠  Mic very low — check connection!              │")
        else:
            self._low_mic_since = None

        # ── Info panel (toggle with 'i') ───────────────────
        if self.show_info:
            chunk_str = f"Chunks: {chunks_done}"
            if chunk_in_progress:
                chunk_str += " ⟳"
            if adaptive.get("rec_only"):
                chunk_str = "REC only"

            interval_str = f"Every {adaptive['interval']}s"
            batch_str = f"Batch: {adaptive['batch_size']}"
            size_mb = rec_size_bytes / (1024 * 1024) if rec_size_bytes else 0

            lines.append(f"  │  📊  {self.model_name} {self.model_size:<8}│ {batch_str:<10}│ {interval_str:<9}│")
            lines.append(f"  │  💾  {size_mb:>5.1f}MB recorded  │ {chunk_str:<23}│")

            # Bookmarks & flags
            markers = []
            if self.bookmarks:
                markers.append(f"📌 {len(self.bookmarks)} bookmarks")
            if self.flags:
                markers.append(f"⚠️  {len(self.flags)} flags")
            if markers:
                marker_str = "  ".join(markers)
                lines.append(f"  │  {marker_str:<50}│")

        lines.append("  └─────────────────────────────────────────────────────┘")

        # ── Help panel ─────────────────────────────────────
        if self.show_help:
            for help_line in self.HELP_TEXT:
                lines.append(help_line)

        # ── Recent transcript (toggle with 'c') ───────────
        if self.show_transcript and transcribed_segments:
            lines.append("  ── Recent transcript ──────────────────────────────")
            recent = transcribed_segments[-5:]
            for seg in recent:
                ts = format_duration(seg.get("start", 0))
                text = seg.get("text", "").strip()
                if len(text) > 50:
                    text = text[:47] + "..."
                spk = seg.get("speaker", "?")
                lines.append(f"  [{ts}] {spk}: {text}")
            lines.append("")

        # ── Render to terminal (overwrite previous) ────────
        if self._prev_line_count > 0:
            sys.stdout.write(f"\033[{self._prev_line_count}A")

        output = ""
        for line in lines:
            output += f"\033[K{line}\n"

        sys.stdout.write(output)
        sys.stdout.flush()
        self._prev_line_count = len(lines)


# ─── Audio Device Detection ───────────────────────────────────────────────────

def get_default_mic():
    """Get the current default input device (mic, earphones, etc.)."""
    import sounddevice as sd
    default_idx = sd.default.device[0]
    if default_idx is not None and default_idx >= 0:
        info = sd.query_devices(default_idx)
        return default_idx, info["name"]
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and BLACKHOLE_DEVICE_NAME not in d["name"]:
            return i, d["name"]
    return None, "Unknown"


def find_blackhole():
    """Check if BlackHole is available."""
    import sounddevice as sd
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and BLACKHOLE_DEVICE_NAME in d["name"]:
            return i, d["name"]
    return None, None


def find_audio_device():
    """
    Find the best recording input.
    Returns: (blackhole_id, mic_id, mic_name)
    """
    bh_id, _ = find_blackhole()
    mic_id, mic_name = get_default_mic()

    if bh_id is not None:
        if mic_name and BLACKHOLE_DEVICE_NAME in mic_name:
            import sounddevice as sd
            for i, d in enumerate(sd.query_devices()):
                if (d["max_input_channels"] > 0
                        and BLACKHOLE_DEVICE_NAME not in d["name"]
                        and "Aggregate" not in d["name"]
                        and "Multi-Output" not in d["name"]):
                    mic_id, mic_name = i, d["name"]
                    break
        return bh_id, mic_id, mic_name

    print("  ⚠  BlackHole not installed — recording from mic only.")
    print("     Zoom/Teams audio won't be captured.")
    print("     Install with: brew install blackhole-2ch && reboot")
    print()
    return None, mic_id, mic_name


def cmd_setup(args):
    """Check audio setup status."""
    import sounddevice as sd

    bh_id, _ = find_blackhole()
    mic_id, mic_name = get_default_mic()

    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  🔧 Audio Setup                                 │")
    print("  ├─────────────────────────────────────────────────┤")

    if bh_id is not None:
        print("  │  ✅ BlackHole 2ch — system audio capture       │")
    else:
        print("  │  ❌ BlackHole 2ch — not installed               │")
        print("  │     brew install blackhole-2ch && reboot        │")

    print(f"  │  🎙  Mic: {mic_name:<38}│")
    print("  └─────────────────────────────────────────────────┘")

    if bh_id is not None:
        print()
        print("  ✅ Ready! The script auto-captures system audio + mic.")
        print("     No manual Audio MIDI Setup needed.")
        print()
        print("  Before a Zoom/Teams call, set your Mac sound output to")
        print("  include BlackHole so the script can capture it:")
        print()
        print("  1. Open 'Audio MIDI Setup' (Spotlight → Audio MIDI)")
        print("  2. Click '+' → 'Create Multi-Output Device'")
        print("  3. Check: your speakers/earphones + BlackHole 2ch")
        print("  4. System Settings → Sound → Output → Multi-Output Device")
        print()
        print("  This only needs to be done once. After that, just:")
        print("     transcribe rec")

    print()
    print("  Input devices:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            markers = []
            if BLACKHOLE_DEVICE_NAME in d["name"]:
                markers.append("system audio")
            if i == mic_id:
                markers.append("active mic")
            tag = f" ← {', '.join(markers)}" if markers else ""
            print(f"    [{i}] {d['name']} ({d['max_input_channels']}ch){tag}")
    print()


# ─── Recording ────────────────────────────────────────────────────────────────

def cmd_record(args):
    """Record audio with silence auto-stop. Captures system + mic by default."""
    import sounddevice as sd
    import soundfile as sf

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    bh_id, mic_id, mic_name = find_audio_device()
    dual_mode = bh_id is not None
    source_label = f"System + {mic_name}" if dual_mode else mic_name

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = RECORDINGS_DIR / f"{timestamp}.wav"

    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  🎙  Interview Recorder                          │")
    print("  │  Press Ctrl+C to stop recording                  │")
    print(f"  │  Auto-stops after {SILENCE_TIMEOUT // 60}min of silence                  │")
    print(f"  │  Source: {source_label:<38}│")
    print("  └─────────────────────────────────────────────────┘")
    print()

    mic_frames = []
    bh_frames = []
    recording = True
    silence_start = None
    current_rms = 0.0
    start_time = time.time()
    lock = threading.Lock()

    def mic_callback(indata, frame_count, time_info, status):
        nonlocal silence_start, current_rms, recording
        if not recording:
            raise sd.CallbackAbort()
        mono = indata.mean(axis=1, keepdims=True) if indata.shape[1] > 1 else indata
        with lock:
            mic_frames.append(mono.copy())
            current_rms = float(np.sqrt(np.mean(mono ** 2)))
            if current_rms < SILENCE_THRESHOLD:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_TIMEOUT:
                    recording = False
                    raise sd.CallbackAbort()
            else:
                silence_start = None

    def bh_callback(indata, frame_count, time_info, status):
        nonlocal silence_start, current_rms
        if not recording:
            raise sd.CallbackAbort()
        mono = indata.mean(axis=1, keepdims=True) if indata.shape[1] > 1 else indata
        with lock:
            bh_frames.append(mono.copy())
            bh_rms = float(np.sqrt(np.mean(mono ** 2)))
            if bh_rms > SILENCE_THRESHOLD:
                silence_start = None
                current_rms = max(current_rms, bh_rms)

    blocksize = int(SAMPLE_RATE * 0.5)

    mic_stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        device=mic_id, blocksize=blocksize, callback=mic_callback,
    )
    bh_stream = None
    if dual_mode:
        bh_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=2, dtype="float32",
            device=bh_id, blocksize=blocksize, callback=bh_callback,
        )

    auto_stopped = False
    try:
        mic_stream.start()
        if bh_stream:
            bh_stream.start()
        mic_low_since_rec = None
        while recording:
            # Fix #4: read shared state under lock
            with lock:
                rms = current_rms
                sil = silence_start

            elapsed = time.time() - start_time
            silence_elapsed = time.time() - sil if sil else 0
            meter = level_meter(rms)

            # Quality indicator
            if rms > 0.01:
                q = "🟢"
            elif rms > 0.003:
                q = "🟡"
            else:
                q = "🔴"

            # Mic low warning
            mic_warn = ""
            if rms < 0.001:
                if mic_low_since_rec is None:
                    mic_low_since_rec = time.time()
                elif time.time() - mic_low_since_rec > MIC_LOW_WARNING_SECONDS:
                    mic_warn = " ⚠ Mic very low!"
            else:
                mic_low_since_rec = None

            status_line = (
                f"  ⏺  {format_duration(elapsed)} │ "
                f"{meter} {q} │ "
                f"Silence: {format_duration(silence_elapsed)}/{format_duration(SILENCE_TIMEOUT)}"
                f"{mic_warn}"
            )
            clear_line()
            sys.stdout.write(status_line)
            sys.stdout.flush()
            time.sleep(0.3)

        auto_stopped = True

    except KeyboardInterrupt:
        recording = False
    finally:
        mic_stream.stop()
        mic_stream.close()
        if bh_stream:
            bh_stream.stop()
            bh_stream.close()

    clear_line()
    elapsed = time.time() - start_time

    if not mic_frames:
        print("  ⚠  No audio recorded.")
        return

    audio_data = _mix_audio(mic_frames, bh_frames, dual_mode)
    sf.write(str(output_path), audio_data, SAMPLE_RATE, subtype="PCM_16")

    print()
    if auto_stopped:
        print(f"  ⏹  Auto-stopped after {SILENCE_TIMEOUT // 60}min of silence")
    else:
        print(f"  ⏹  Recording stopped")

    print(f"  📁 Saved: {output_path.name}")
    print(f"  ⏱  Duration: {format_duration(elapsed)}")
    print(f"  📂 Location: {RECORDINGS_DIR}")
    print()

    prompt_transcribe(str(output_path))


def _mix_audio(mic_frames, bh_frames, dual_mode):
    """Mix mic + system audio into mono. Returns numpy array."""
    if not mic_frames:
        return np.array([], dtype=np.float32)
    mic_audio = np.concatenate(mic_frames)
    if dual_mode and bh_frames:
        bh_audio = np.concatenate(bh_frames)
        min_len = min(len(mic_audio), len(bh_audio))
        mic_audio = mic_audio[:min_len]
        bh_audio = bh_audio[:min_len]
        return (mic_audio + bh_audio) / 2.0
    return mic_audio


def prompt_transcribe(audio_path):
    """Ask user if they want to transcribe now."""
    try:
        answer = input("  Transcribe now? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  OK, you can transcribe later with: transcribe run")
        return

    if answer in ("", "y", "yes"):
        model = prompt_model_selection()
        speakers = input("  Speaker names (comma-separated, or Enter to skip): ").strip()
        title = input("  Title (or Enter for auto): ").strip()

        run_args = argparse.Namespace(
            audio=audio_path,
            model=model,
            speakers=speakers or None,
            title=title or None,
            language=config["language"],
            no_diarize=False,
        )
        cmd_run(run_args)
    else:
        print("  OK, you can transcribe later with:")
        print(f"    transcribe run \"{audio_path}\"")


def prompt_model_selection():
    """Interactive model selection menu."""
    models = list(MODEL_INFO.keys())
    default_model = config["default_model"]
    default_idx = models.index(default_model) + 1 if default_model in models else 3

    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  Choose transcription model:                    │")
    print("  ├─────────────────────────────────────────────────┤")
    for i, (name, info) in enumerate(MODEL_INFO.items(), 1):
        marker = " ★" if name == default_model else "  "
        desc = info.get("description", "")
        line = f"  │  {i}) {name:<10} {desc:<28}{marker}│"
        print(line)
    print("  └─────────────────────────────────────────────────┘")

    while True:
        try:
            choice = input(f"  Select [1-{len(models)}] (default: {default_idx} — {default_model}): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return default_model

        if choice == "":
            return default_model
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            selected = models[int(choice) - 1]
            print(f"  → Using {selected}")
            return selected
        print("  Invalid choice, try again.")


# ─── Live Record + Transcribe ────────────────────────────────────────────────

def cmd_live(args):
    """Record and transcribe simultaneously — transcript ready shortly after recording stops."""
    import whisperx
    import sounddevice as sd
    import soundfile as sf

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    model_name = args.model or config["default_model"]
    language = args.language or config["language"]
    batch_size = config["batch_sizes"].get(model_name, 8)
    chunk_interval = config["live"]["chunk_interval_seconds"]
    no_diarize = getattr(args, "no_diarize", False)
    speakers_str = getattr(args, "speakers", None)
    title = getattr(args, "title", None)

    # ── Pre-load whisper model ────────────────────────────────────────────
    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  🎙  Live Record + Transcribe                   │")
    print("  └─────────────────────────────────────────────────┘")
    print()
    print(f"  Loading '{model_name}' model...")
    model = whisperx.load_model(
        model_name, DEVICE, compute_type=config["compute_type"], language=language
    )
    print(f"  ✅ Model ready — starting recording")

    # ── Find audio devices ────────────────────────────────────────────────
    bh_id, mic_id, mic_name = find_audio_device()
    dual_mode = bh_id is not None
    source_label = f"System + {mic_name}" if dual_mode else mic_name

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = RECORDINGS_DIR / f"{timestamp}.wav"

    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  Press Ctrl+C to stop recording                 │")
    print(f"  │  Auto-stops after {SILENCE_TIMEOUT // 60}min silence                 │")
    print(f"  │  Source: {source_label:<38}│")
    print(f"  │  Transcribing every {chunk_interval}s in background       │")
    print("  └─────────────────────────────────────────────────┘")
    print()

    # ── Recording state ───────────────────────────────────────────────────
    mic_frames = []
    bh_frames = []
    recording = True
    silence_start = None
    current_rms = 0.0
    mic_rms_val = 0.0
    bh_rms_val = 0.0
    mic_low_since = None  # timestamp when mic went quiet
    start_time = time.time()
    lock = threading.Lock()

    # ── Live transcription state ──────────────────────────────────────────
    transcribed_segments = []
    last_transcribed_sample = 0
    chunks_done = 0
    chunk_in_progress = False
    transcribe_lock = threading.Lock()
    stop_event = threading.Event()

    # ── Adaptive chunking state ───────────────────────────────────────────
    live_cfg = config.get("live", {})
    adaptive = {
        "interval": chunk_interval,
        "batch_size": batch_size,
        "original_batch_size": batch_size,
        "min_interval": live_cfg.get("min_chunk_interval", 60),
        "max_interval": live_cfg.get("max_chunk_interval", 300),
        "struggle_ratio": live_cfg.get("struggle_ratio", 0.7),
        "rec_only_ratio": live_cfg.get("rec_only_ratio", 1.5),
        "rec_only": False,
        "status_msg": "",
    }

    def mic_callback(indata, frame_count, time_info, status):
        nonlocal silence_start, current_rms, mic_rms_val, recording
        if not recording:
            raise sd.CallbackAbort()
        mono = indata.mean(axis=1, keepdims=True) if indata.shape[1] > 1 else indata
        with lock:
            mic_frames.append(mono.copy())
            mic_rms_val = float(np.sqrt(np.mean(mono ** 2)))
            current_rms = mic_rms_val
            if current_rms < SILENCE_THRESHOLD:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_TIMEOUT:
                    recording = False
                    raise sd.CallbackAbort()
            else:
                silence_start = None

    def bh_callback(indata, frame_count, time_info, status):
        nonlocal silence_start, current_rms, bh_rms_val
        if not recording:
            raise sd.CallbackAbort()
        mono = indata.mean(axis=1, keepdims=True) if indata.shape[1] > 1 else indata
        with lock:
            bh_frames.append(mono.copy())
            bh_rms_val = float(np.sqrt(np.mean(mono ** 2)))
            if bh_rms_val > SILENCE_THRESHOLD:
                silence_start = None
                current_rms = max(current_rms, bh_rms_val)

    def _get_mixed_snapshot():
        """Get current mixed audio snapshot (copies frame lists under lock)."""
        with lock:
            if not mic_frames:
                return None
            mic_copy = list(mic_frames)
            bh_copy = list(bh_frames) if dual_mode else []
        audio = np.concatenate(mic_copy).flatten()
        if bh_copy:
            bh = np.concatenate(bh_copy).flatten()
            min_len = min(len(audio), len(bh))
            audio = (audio[:min_len] + bh[:min_len]) / 2.0
        return audio

    def _adapt_chunking(audio_seconds, process_seconds):
        """Adjust chunk interval and batch size based on processing speed."""
        if audio_seconds <= 0:
            return

        ratio = process_seconds / audio_seconds

        if ratio > adaptive["rec_only_ratio"]:
            # Can't keep up — switch to record-only
            adaptive["rec_only"] = True
            adaptive["status_msg"] = "⚠  System busy — record only (will transcribe after)"
        elif ratio > adaptive["struggle_ratio"]:
            # Struggling — grow interval, shrink batch size
            adaptive["interval"] = min(
                adaptive["max_interval"],
                int(adaptive["interval"] * 1.5)
            )
            adaptive["batch_size"] = max(2, adaptive["batch_size"] // 2)
            adaptive["status_msg"] = f"↑ interval={adaptive['interval']}s batch={adaptive['batch_size']}"
        elif ratio < 0.3:
            # Lots of headroom — shrink interval, restore batch size
            adaptive["interval"] = max(
                adaptive["min_interval"],
                int(adaptive["interval"] * 0.75)
            )
            adaptive["batch_size"] = min(
                adaptive["original_batch_size"],
                adaptive["batch_size"] * 2
            )
            adaptive["status_msg"] = f"↓ interval={adaptive['interval']}s batch={adaptive['batch_size']}"
        else:
            # Healthy — clear status
            adaptive["status_msg"] = ""

    def transcription_worker():
        """Background thread: transcribe chunks during recording with adaptive pacing."""
        nonlocal last_transcribed_sample, chunks_done, chunk_in_progress
        live_cfg = config.get("live", {})
        overlap_seconds = live_cfg.get("chunk_overlap_seconds", 5)
        min_chunk_secs = live_cfg.get("min_chunk_seconds", 30)
        overlap_samples = int(SAMPLE_RATE * overlap_seconds)

        while not stop_event.is_set():
            stop_event.wait(timeout=adaptive["interval"])
            if stop_event.is_set():
                break

            # Skip if we've switched to record-only mode
            if adaptive["rec_only"]:
                continue

            audio = _get_mixed_snapshot()
            if audio is None:
                continue

            total_samples = len(audio)
            new_samples = total_samples - last_transcribed_sample
            if new_samples < SAMPLE_RATE * min_chunk_secs:
                continue

            chunk_start = max(0, last_transcribed_sample - overlap_samples)
            chunk = audio[chunk_start:total_samples].astype(np.float32)
            offset = chunk_start / SAMPLE_RATE
            overlap_boundary = last_transcribed_sample / SAMPLE_RATE
            audio_seconds = len(chunk) / SAMPLE_RATE

            chunk_in_progress = True
            process_start = time.time()
            try:
                result = model.transcribe(chunk, batch_size=adaptive["batch_size"])
                process_seconds = time.time() - process_start

                segments = result.get("segments", [])
                for seg in segments:
                    seg["start"] = seg.get("start", 0) + offset
                    seg["end"] = seg.get("end", 0) + offset
                # Keep segments whose midpoint is past the overlap boundary
                segments = [
                    s for s in segments
                    if (s.get("start", 0) + s.get("end", 0)) / 2 > overlap_boundary
                ]
                with transcribe_lock:
                    transcribed_segments.extend(segments)
                last_transcribed_sample = total_samples
                chunks_done += 1

                # Adapt based on how fast we processed this chunk
                _adapt_chunking(audio_seconds, process_seconds)

            except Exception as e:
                # Log but don't crash — recording continues even if a chunk fails
                print(f"\n  ⚠ Chunk transcription error: {e}", file=sys.stderr)
            finally:
                chunk_in_progress = False

    # ── Start recording + background transcription ────────────────────────
    blocksize = int(SAMPLE_RATE * 0.5)
    mic_stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        device=mic_id, blocksize=blocksize, callback=mic_callback,
    )
    bh_stream = None
    if dual_mode:
        bh_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=2, dtype="float32",
            device=bh_id, blocksize=blocksize, callback=bh_callback,
        )

    transcription_thread = threading.Thread(target=transcription_worker, daemon=True)
    auto_stopped = False
    _last_adaptive_msg = [""]  # mutable container so inner scope can read it

    try:
        mic_stream.start()
        if bh_stream:
            bh_stream.start()
        transcription_thread.start()

        while recording:
            with lock:
                rms = current_rms
                sil = silence_start
                m_rms = mic_rms_val
                b_rms = bh_rms_val

            elapsed = time.time() - start_time
            silence_elapsed = time.time() - sil if sil else 0

            # ── Quality indicators (green/yellow/red) ──
            if m_rms > 0.01:
                mic_status = "🟢"
            elif m_rms > 0.003:
                mic_status = "🟡"
            else:
                mic_status = "🔴"

            if dual_mode:
                if b_rms > 0.01:
                    sys_status = "🟢"
                elif b_rms > 0.003:
                    sys_status = "🟡"
                else:
                    sys_status = "🔴"
                quality = f"Mic{mic_status} Sys{sys_status}"
            else:
                quality = f"Mic{mic_status}"

            # ── Mic low warning (10+ seconds) ──
            mic_warn = ""
            if m_rms < 0.001:
                if mic_low_since is None:
                    mic_low_since = time.time()
                elif time.time() - mic_low_since > MIC_LOW_WARNING_SECONDS:
                    mic_warn = " ⚠ Mic very low!"
            else:
                mic_low_since = None

            # ── Chunk status ──
            if adaptive["rec_only"]:
                chunk_status = " │ REC only"
            else:
                chunk_status = f" │ Chunks: {chunks_done}"
                if chunk_in_progress:
                    chunk_status += " ⟳"

            meter = level_meter(rms)
            status_line = (
                f"  ⏺  {format_duration(elapsed)} │ "
                f"{meter} │ {quality}"
                f"{chunk_status}{mic_warn}"
            )

            # Show adaptive warning on a separate line
            if adaptive["status_msg"] and adaptive["status_msg"] != _last_adaptive_msg[0]:
                clear_line()
                print(f"  {adaptive['status_msg']}")
                _last_adaptive_msg[0] = adaptive["status_msg"]
            clear_line()
            sys.stdout.write(status_line)
            sys.stdout.flush()
            time.sleep(0.3)

        auto_stopped = True

    except KeyboardInterrupt:
        recording = False
    finally:
        mic_stream.stop()
        mic_stream.close()
        if bh_stream:
            bh_stream.stop()
            bh_stream.close()

    clear_line()
    recording_end = time.time()
    elapsed = recording_end - start_time

    # Signal transcription worker to stop and wait for in-progress chunk
    stop_event.set()
    if chunk_in_progress:
        print("  ⏳ Waiting for in-progress chunk to finish...")
    transcription_thread.join(timeout=120)

    if not mic_frames:
        print("  ⚠  No audio recorded.")
        del model
        gc.collect()
        return

    print()
    if auto_stopped:
        print(f"  ⏹  Auto-stopped after {SILENCE_TIMEOUT // 60}min silence")
    else:
        print(f"  ⏹  Recording stopped ({format_duration(elapsed)})")
    if adaptive["rec_only"]:
        print(f"  📊 {chunks_done} chunks done before switching to record-only")
    else:
        print(f"  📊 {chunks_done} chunks pre-transcribed during recording")

    # ── Save audio file ───────────────────────────────────────────────────
    audio_data = _mix_audio(mic_frames, bh_frames, dual_mode)
    sf.write(str(output_path), audio_data, SAMPLE_RATE, subtype="PCM_16")
    print(f"  📁 Saved: {output_path.name}")

    # ── Transcribe remaining audio ────────────────────────────────────────
    total_samples = len(audio_data.flatten())
    remaining = total_samples - last_transcribed_sample

    if adaptive["rec_only"] and last_transcribed_sample == 0:
        # Never managed to transcribe anything — do full file transcription
        remaining_duration = total_samples / SAMPLE_RATE
        print(f"  ⏳ Transcribing full recording ({format_duration(remaining_duration)})...")
        full_audio = audio_data.flatten().astype(np.float32)
        try:
            result = model.transcribe(full_audio, batch_size=adaptive["original_batch_size"])
            segments = result.get("segments", [])
            with transcribe_lock:
                transcribed_segments.extend(segments)
        except Exception as e:
            print(f"  ⚠  Transcription failed: {e}")

    overlap_seconds = config.get("live", {}).get("chunk_overlap_seconds", 5)
    if remaining > SAMPLE_RATE * overlap_seconds:
        remaining_duration = remaining / SAMPLE_RATE
        print(f"  ⏳ Transcribing final {format_duration(remaining_duration)}...")
        post_overlap = int(SAMPLE_RATE * overlap_seconds)
        chunk_start = max(0, last_transcribed_sample - post_overlap)
        chunk = audio_data.flatten()[chunk_start:total_samples].astype(np.float32)
        offset = chunk_start / SAMPLE_RATE
        overlap_boundary = last_transcribed_sample / SAMPLE_RATE

        try:
            result = model.transcribe(chunk, batch_size=adaptive["original_batch_size"])
            segments = result.get("segments", [])
            for seg in segments:
                seg["start"] = seg.get("start", 0) + offset
                seg["end"] = seg.get("end", 0) + offset
            segments = [
                s for s in segments
                if (s.get("start", 0) + s.get("end", 0)) / 2 > overlap_boundary
            ]
            with transcribe_lock:
                transcribed_segments.extend(segments)
        except Exception as e:
            print(f"  ⚠  Final chunk failed: {e}")

    # Free whisper model
    del model
    gc.collect()

    all_segments = sorted(transcribed_segments, key=lambda s: s.get("start", 0))
    if not all_segments:
        print("  ⚠  No speech detected.")
        return

    result = {"segments": all_segments, "language": language}

    # ── Alignment ─────────────────────────────────────────────────────────
    print("  ⏳ Aligning timestamps...")
    try:
        audio_np = whisperx.load_audio(str(output_path))
        align_model, metadata = whisperx.load_align_model(
            language_code=language, device=DEVICE
        )
        result = whisperx.align(
            result["segments"], align_model, metadata, audio_np, DEVICE,
            return_char_alignments=False,
        )
        del align_model
        gc.collect()
    except Exception as e:
        print(f"  ⚠  Alignment failed: {e}")

    # ── Diarization ───────────────────────────────────────────────────────
    diarize_enabled = config["diarization"]["enabled"] and not no_diarize
    if HF_TOKEN and diarize_enabled:
        print("  ⏳ Identifying speakers...")
        try:
            from whisperx.diarize import DiarizationPipeline
            diarize_model = DiarizationPipeline(
                model_name=config["diarization"]["model"],
                token=HF_TOKEN, device=DEVICE
            )
            diarize_segments = diarize_model(str(output_path))
            result = whisperx.assign_word_speakers(diarize_segments, result)
            del diarize_model
            gc.collect()
        except Exception as e:
            print(f"  ⚠  Diarization failed: {e}")

    # ── Save transcript ───────────────────────────────────────────────────
    speaker_names = resolve_speaker_names(result, str(output_path), speakers_str)

    if not title:
        title = Path(output_path).stem.replace("_", " ").replace("-", " ").title()

    date_str = extract_recording_date(str(output_path))
    markdown = format_transcript(result, title, speaker_names, date_str)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in title)
    transcript_file = SCRIPTS_DIR / f"{date_str} {safe_title}.md"
    transcript_file.write_text(markdown, encoding="utf-8")

    post_time = time.time() - recording_end

    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  ✅ Live transcription complete!                │")
    print("  ├─────────────────────────────────────────────────┤")
    print(f"  │  Recording:       {format_duration(elapsed):<30}│")
    print(f"  │  Post-processing: {format_duration(post_time):<30}│")
    print(f"  │  Chunks done:     {chunks_done:<30}│")
    print(f"  │  Speakers:        {len(speaker_names):<30}│")
    print(f"  │  Segments:        {len(result.get('segments', [])):<30}│")
    print(f"  │  Saved: Scripts/{transcript_file.name:<32}│")
    print("  └─────────────────────────────────────────────────┘")
    print()


# ─── Transcription with Progress ─────────────────────────────────────────────

class ProgressTracker:
    """Track and display transcription progress with smooth time-based bar."""

    def __init__(self, audio_duration, model_name):
        self.audio_duration = audio_duration
        self.model_name = model_name
        speed_factor = MODEL_INFO.get(model_name, {}).get("speed_factor", 1.0)
        self.estimated_total = max(10, audio_duration * speed_factor)
        self.start_time = time.time()
        self.current_step = 0
        self.running = True
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._display_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=1)
        clear_line()

    def set_step(self, step_index):
        self.current_step = step_index

    def _display_loop(self):
        tick = 0
        while self.running:
            elapsed = time.time() - self.start_time
            step_name = STEP_NAMES[self.current_step] if self.current_step < len(STEP_NAMES) else "Processing"

            # Smooth time-based progress (no more stuck-at-5%)
            total_progress = min(0.95, elapsed / self.estimated_total)

            bar_width = 25
            filled = int(total_progress * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            pct = int(total_progress * 100)

            # ETA: self-correcting based on elapsed time
            if total_progress > 0.05:
                eta = (elapsed / total_progress) * (1 - total_progress)
            else:
                eta = max(0, self.estimated_total - elapsed)

            spin = spinner_char(tick)
            status = (
                f"  {spin} {step_name}... "
                f"[{bar}] {pct}% │ "
                f"Elapsed: {format_duration(elapsed)} │ "
                f"ETA: ~{format_duration(eta)}"
            )
            clear_line()
            sys.stdout.write(status)
            sys.stdout.flush()

            tick += 1
            time.sleep(0.3)


def cmd_run(args):
    """Transcribe an audio file with progress tracking."""
    import whisperx

    audio_path = args.audio

    if not audio_path:
        audio_path = get_last_recording()
        if not audio_path:
            print("  ❌ No recordings found. Record first with: transcribe rec")
            return

    if not os.path.exists(audio_path):
        print(f"  ❌ File not found: {audio_path}")
        return

    model_name = getattr(args, "model", None) or config["default_model"]
    if not getattr(args, "model", None) and sys.stdin.isatty():
        model_name = prompt_model_selection()

    language = getattr(args, "language", None) or config["language"]
    speakers_str = getattr(args, "speakers", None)
    title = getattr(args, "title", None)
    no_diarize = getattr(args, "no_diarize", False)
    batch_size = config["batch_sizes"].get(model_name, 8)

    if not title:
        title = Path(audio_path).stem.replace("_", " ").replace("-", " ").title()

    if not HF_TOKEN:
        print()
        print("  ⚠  HF_TOKEN not set — speaker diarization will be skipped.")
        print("  Set it with: export HF_TOKEN=your_token_here")
        print()

    audio_duration = get_audio_duration(audio_path)
    if audio_duration <= 0:
        print("  ⚠  Couldn't determine audio duration — progress estimates may be off.")
        audio_duration = 300  # assume 5 minutes as fallback

    speed_factor = MODEL_INFO.get(model_name, {}).get("speed_factor", 1.0)
    estimated_time = audio_duration * speed_factor

    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  📝 Transcription                               │")
    print("  ├─────────────────────────────────────────────────┤")
    print(f"  │  File:     {Path(audio_path).name:<38}│")
    print(f"  │  Duration: {format_duration(audio_duration):<38}│")
    print(f"  │  Model:    {model_name:<38}│")
    print(f"  │  Language: {language:<38}│")
    print(f"  │  Est time: ~{format_duration(estimated_time):<37}│")
    print(f"  │  Started:  {datetime.now().strftime('%H:%M:%S'):<38}│")
    print("  └─────────────────────────────────────────────────┘")
    print()

    progress = ProgressTracker(audio_duration, model_name)
    progress.start()

    try:
        # Step 0: Load model
        progress.set_step(0)
        model = whisperx.load_model(
            model_name, DEVICE, compute_type=config["compute_type"], language=language
        )

        # Step 1: Transcribe
        progress.set_step(1)
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=batch_size)
        detected_language = result.get("language", language)
        del model
        gc.collect()

        # Step 2: Align
        progress.set_step(2)
        align_model, metadata = whisperx.load_align_model(
            language_code=detected_language, device=DEVICE
        )
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, DEVICE,
            return_char_alignments=False,
        )
        del align_model
        gc.collect()

        # Step 3: Diarize
        progress.set_step(3)
        diarize_enabled = config["diarization"]["enabled"] and not no_diarize
        if HF_TOKEN and diarize_enabled:
            from whisperx.diarize import DiarizationPipeline
            diarize_model = DiarizationPipeline(
                model_name=config["diarization"]["model"],
                token=HF_TOKEN, device=DEVICE
            )
            diarize_segments = diarize_model(audio_path)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            del diarize_model
            gc.collect()

        # Step 4: Save
        progress.set_step(4)

    except Exception as e:
        progress.stop()
        print(f"\n\n  ❌ Transcription failed: {e}")
        return

    progress.stop()

    # Resolve speaker names (auto-match from saved profiles + -s flag)
    speaker_names = resolve_speaker_names(result, audio_path, speakers_str)

    # Save transcript
    date_str = extract_recording_date(audio_path)
    markdown = format_transcript(result, title, speaker_names, date_str)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in title)
    output_file = SCRIPTS_DIR / f"{date_str} {safe_title}.md"
    output_file.write_text(markdown, encoding="utf-8")

    total_time = time.time() - progress.start_time

    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  ✅ Transcription complete!                     │")
    print("  ├─────────────────────────────────────────────────┤")
    print(f"  │  Speakers:  {len(speaker_names):<37}│")
    print(f"  │  Segments:  {len(result.get('segments', [])):<37}│")
    print(f"  │  Time:      {format_duration(total_time):<37}│")
    print(f"  │  Saved to:  Scripts/{output_file.name:<27}│")
    print("  └─────────────────────────────────────────────────┘")
    print()


# ─── List Recordings ─────────────────────────────────────────────────────────

def cmd_list(args):
    """List available recordings with transcript status."""
    if not RECORDINGS_DIR.exists():
        print("  No recordings yet. Start with: transcribe rec")
        return

    recordings = list_recordings(RECORDINGS_DIR)
    if not recordings:
        print("  No recordings found.")
        return

    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  📂 Recordings                                  │")
    print("  ├─────────────────────────────────────────────────┤")

    max_list = config.get("list", {}).get("max_recordings", 20)
    for i, rec in enumerate(recordings[:max_list], 1):
        duration = get_audio_duration(str(rec))
        size_mb = rec.stat().st_size / (1024 * 1024)

        # Check if transcript exists for this recording
        date_prefix = rec.stem[:10]  # "2026-03-19"
        has_transcript = SCRIPTS_DIR.exists() and any(SCRIPTS_DIR.glob(f"{date_prefix}*.md"))
        status = "✅" if has_transcript else "  "

        line = f"  │ {status} {i:>2}) {rec.stem}  ({format_duration(duration)}, {size_mb:.1f}MB)"
        print(f"{line:<52}│")

    print("  └─────────────────────────────────────────────────┘")
    print()
    print(f"  📂 {RECORDINGS_DIR}")
    print(f"  ✅ = transcript exists")
    print()

    if recordings:
        try:
            choice = input("  Transcribe one? Enter number (or Enter to skip): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(recordings):
                selected = recordings[int(choice) - 1]
                prompt_transcribe(str(selected))
        except (EOFError, KeyboardInterrupt):
            print()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def extract_recording_date(audio_path):
    """Extract date from recording filename or file metadata."""
    stem = Path(audio_path).stem
    match = re.match(r"(\d{4}-\d{2}-\d{2})", stem)
    if match:
        return match.group(1)
    try:
        mtime = os.path.getmtime(audio_path)
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")


def list_recordings(directory):
    """List all audio files in directory, sorted newest first."""
    files = []
    if directory.exists():
        for f in directory.iterdir():
            if f.suffix.lower() in AUDIO_FORMATS:
                files.append(f)
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def get_last_recording():
    """Get the most recent recording file."""
    recordings = list_recordings(RECORDINGS_DIR)
    return str(recordings[0]) if recordings else None


def get_audio_duration(audio_path):
    """Get audio duration in seconds. Tries soundfile first, then ffprobe."""
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", audio_path],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
        return 0.0
    except Exception:
        return 0.0


def format_transcript(result, title, speaker_names, date_str):
    """Format transcription result as Obsidian-friendly Markdown."""
    lines = []

    # YAML frontmatter
    lines.append("---")
    lines.append(f"title: \"{title}\"")
    lines.append(f"date: {date_str}")
    lines.append("type: interview-transcript")
    speaker_list = ", ".join(speaker_names.values())
    lines.append(f"speakers: [{speaker_list}]")
    lines.append("---")
    lines.append("")
    lines.append(f"# {title}")
    lines.append(f"**Date:** {date_str}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Group consecutive segments by speaker
    segments = result.get("segments", [])
    single_speaker = len(speaker_names) <= 1
    current_speaker = None
    current_text = []
    current_start = 0.0

    for seg in segments:
        speaker_id = seg.get("speaker", "Unknown")
        speaker_label = speaker_names.get(speaker_id, speaker_id)
        text = seg.get("text", "").strip()

        if not single_speaker and speaker_label == current_speaker:
            # Multi-speaker: merge consecutive segments from same speaker
            current_text.append(text)
        else:
            if current_speaker is not None:
                timestamp = format_duration(current_start)
                joined = " ".join(current_text)
                if single_speaker:
                    lines.append(f"**[{timestamp}]** {joined}")
                else:
                    lines.append(f"**[{timestamp}] {current_speaker}:** {joined}")
                lines.append("")

            current_speaker = speaker_label
            current_text = [text]
            current_start = seg.get("start", 0.0)

    if current_speaker is not None:
        timestamp = format_duration(current_start)
        joined = " ".join(current_text)
        if single_speaker:
            lines.append(f"**[{timestamp}]** {joined}")
        else:
            lines.append(f"**[{timestamp}] {current_speaker}:** {joined}")
        lines.append("")

    return "\n".join(lines)


# ─── Speaker Memory ──────────────────────────────────────────────────────────

def load_embedding_model():
    """Load pyannote embedding model for speaker recognition."""
    memory_cfg = config.get("speaker_memory", {})
    model_name = memory_cfg.get("embedding_model", "pyannote/embedding")
    try:
        from pyannote.audio import Inference
        try:
            model = Inference(model_name, window="whole", use_auth_token=HF_TOKEN)
        except TypeError:
            model = Inference(model_name, window="whole", token=HF_TOKEN)
        return model
    except Exception as e:
        error_msg = str(e).lower()
        if "403" in str(e) or "gated" in error_msg or "access" in error_msg:
            print(f"  ⚠  Speaker embedding model requires HuggingFace terms acceptance.")
            print(f"     Accept at: https://huggingface.co/{model_name}")
        else:
            print(f"  ⚠  Failed to load embedding model: {e}")
        return None


def extract_speaker_embedding(embedding_model, audio_path, segments):
    """Extract average embedding for a speaker from their audio segments."""
    from pyannote.core import Segment

    embeddings = []
    total_duration = 0

    for start, end in segments:
        duration = end - start
        if duration < 1.0:
            continue
        total_duration += duration
        try:
            emb = embedding_model.crop(audio_path, Segment(start, end))
            emb = np.array(emb).flatten()
            if emb.shape[0] > 0:
                embeddings.append(emb)
        except Exception:
            continue

    if not embeddings or total_duration < 2.0:
        return None

    # Average and L2-normalize
    avg = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg = avg / norm
    return avg


def load_speaker_profiles():
    """Load all saved voice profiles from speakers/ directory."""
    profiles = {}
    if not SPEAKERS_DIR.exists():
        return profiles
    for f in SPEAKERS_DIR.glob("*.npz"):
        try:
            data = np.load(f, allow_pickle=True)
            profiles[f.stem] = {
                "name": str(data["name"]),
                "embedding": data["embedding"],
            }
        except Exception:
            continue
    return profiles


def save_speaker_profile(name, embedding):
    """Save a voice profile to speakers/ directory."""
    SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = SPEAKERS_DIR / f"{name.lower()}.npz"
    np.savez(profile_path, name=name, embedding=embedding,
             enrolled_at=datetime.now().strftime("%Y-%m-%d"))
    return profile_path


def match_speakers(speaker_embeddings, saved_profiles, threshold=0.75):
    """Match diarized speakers against saved profiles using cosine similarity.

    Returns: dict {speaker_id: (display_name, similarity_score)}
    """
    if not saved_profiles or not speaker_embeddings:
        return {}

    scores = []
    for spk_id, spk_emb in speaker_embeddings.items():
        if spk_emb is None:
            continue
        for prof_key, profile in saved_profiles.items():
            prof_emb = profile["embedding"]
            similarity = float(np.dot(spk_emb, prof_emb) / (
                np.linalg.norm(spk_emb) * np.linalg.norm(prof_emb) + 1e-8
            ))
            scores.append((similarity, spk_id, profile["name"]))

    # Greedy assignment: best scores first, each profile/speaker used once
    scores.sort(reverse=True)
    matched = {}
    used_profiles = set()
    used_speakers = set()

    for similarity, spk_id, name in scores:
        if similarity < threshold:
            break
        if spk_id in used_speakers or name in used_profiles:
            continue
        matched[spk_id] = (name, similarity)
        used_speakers.add(spk_id)
        used_profiles.add(name)

    return matched


def resolve_speaker_names(result, audio_path, speakers_str=None):
    """Resolve speaker IDs to display names.

    Priority: -s flag > saved profile match > Person N
    """
    segments = result.get("segments", [])
    speaker_ids = sorted(set(seg.get("speaker", "Unknown") for seg in segments))

    if not speaker_ids:
        return {}

    speaker_names = {}
    matched = {}
    memory_cfg = config.get("speaker_memory", {})
    memory_enabled = memory_cfg.get("enabled", True)

    # Step 1: Match against saved voice profiles
    if memory_enabled and HF_TOKEN:
        profiles = load_speaker_profiles()
        if profiles:
            print("  🔍 Matching voices...")
            embedding_model = load_embedding_model()
            if embedding_model is not None:
                # Group segments by speaker
                speaker_segments = {}
                for seg in segments:
                    sid = seg.get("speaker", "Unknown")
                    if sid not in speaker_segments:
                        speaker_segments[sid] = []
                    speaker_segments[sid].append(
                        (seg.get("start", 0), seg.get("end", 0))
                    )

                # Extract embedding per speaker
                speaker_embeddings = {}
                for sid, segs in speaker_segments.items():
                    speaker_embeddings[sid] = extract_speaker_embedding(
                        embedding_model, audio_path, segs
                    )

                threshold = memory_cfg.get("similarity_threshold", 0.75)
                matched = match_speakers(speaker_embeddings, profiles, threshold)
                for sid, (name, _score) in matched.items():
                    speaker_names[sid] = name

                del embedding_model
                gc.collect()

    # Step 2: -s flag overrides
    if speakers_str:
        names = [n.strip() for n in speakers_str.split(",")]
        for i, sid in enumerate(speaker_ids):
            if i < len(names) and names[i]:
                speaker_names[sid] = names[i]

    # Step 3: Fill remaining with Person N
    person_counter = 1
    for sid in speaker_ids:
        if sid not in speaker_names:
            speaker_names[sid] = f"Person {person_counter}"
            person_counter += 1

    # Print results
    if matched or speakers_str:
        print("  🔊 Speakers:")
        for sid in speaker_ids:
            name = speaker_names[sid]
            if sid in matched:
                _, score = matched[sid]
                print(f"     {sid} → {name} ({int(score * 100)}% match)")
            else:
                print(f"     {sid} → {name}")

    return speaker_names


def cmd_enroll(args):
    """Record a voice sample and save profile for auto-recognition."""
    import sounddevice as sd
    import soundfile as sf

    user_name = getattr(args, "name", None) or config.get("user_name", "")
    if not user_name:
        try:
            user_name = input("  Enter your name: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
    if not user_name:
        print("  ❌ Name required.")
        return

    duration = config.get("speaker_memory", {}).get("enrollment_duration_seconds", 15)

    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  🎤 Voice Enrollment                            │")
    print("  ├─────────────────────────────────────────────────┤")
    print(f"  │  Name: {user_name:<42}│")
    print("  │                                                 │")
    print(f"  │  Speak naturally for {duration} seconds.{' ' * (16 - len(str(duration)))}│")
    print("  │  Read something aloud or describe your day.     │")
    print("  └─────────────────────────────────────────────────┘")
    print()

    mic_id, mic_name = get_default_mic()
    print(f"  🎙  Using: {mic_name}")

    try:
        input("  Press Enter to start recording...")
    except (EOFError, KeyboardInterrupt):
        print()
        return
    frames = []
    recording = True
    start_time = time.time()

    def callback(indata, frame_count, time_info, status):
        nonlocal recording
        if not recording:
            raise sd.CallbackAbort()
        mono = indata.mean(axis=1, keepdims=True) if indata.shape[1] > 1 else indata
        frames.append(mono.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        device=mic_id, blocksize=int(SAMPLE_RATE * 0.5), callback=callback,
    )

    try:
        stream.start()
        while recording:
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            if remaining <= 0:
                recording = False
                break
            bar_width = 25
            progress = min(1.0, elapsed / duration)
            filled = int(progress * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            clear_line()
            sys.stdout.write(f"  ⏺  Recording... [{bar}] {int(remaining)}s remaining")
            sys.stdout.flush()
            time.sleep(0.2)
    except KeyboardInterrupt:
        recording = False
    finally:
        stream.stop()
        stream.close()
    clear_line()

    if not frames:
        print("  ❌ No audio captured.")
        return

    audio = np.concatenate(frames)
    actual_duration = len(audio) / SAMPLE_RATE

    if actual_duration < 3:
        print(f"  ❌ Too short ({format_duration(actual_duration)}). Need at least 3 seconds.")
        return

    # Save temp file for embedding extraction
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, audio, SAMPLE_RATE, subtype="PCM_16")

    print(f"  ⏳ Extracting voice profile...")
    embedding_model = load_embedding_model()
    if embedding_model is None:
        os.unlink(tmp_path)
        return

    embedding = extract_speaker_embedding(
        embedding_model, tmp_path, [(0, actual_duration)]
    )
    os.unlink(tmp_path)
    del embedding_model
    gc.collect()

    if embedding is None:
        print("  ❌ Could not extract voice profile. Try speaking louder or longer.")
        return

    # Check existing profile
    profile_path = SPEAKERS_DIR / f"{user_name.lower()}.npz"
    if profile_path.exists():
        try:
            answer = input(f"  Profile for '{user_name}' exists. Overwrite? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if answer not in ("y", "yes"):
            print("  ─  Keeping existing profile.")
            return

    save_speaker_profile(user_name, embedding)

    # Save user_name to config if not set
    if not config.get("user_name"):
        config["user_name"] = user_name
        try:
            with open(CONFIG_PATH, "w") as f:
                json.dump(config, f, indent=4)
            print(f"  ✅ Saved name '{user_name}' to config.json")
        except Exception:
            print(f"  Tip: add \"user_name\": \"{user_name}\" to config.json")

    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  ✅ Voice profile saved!                        │")
    print("  ├─────────────────────────────────────────────────┤")
    print(f"  │  Name:     {user_name:<38}│")
    print(f"  │  Duration: {format_duration(actual_duration):<38}│")
    print(f"  │  Profile:  speakers/{profile_path.name:<29}│")
    print("  │                                                 │")
    print("  │  Your voice will be auto-recognized in future   │")
    print("  │  transcripts. Everyone else = Person N.         │")
    print("  └─────────────────────────────────────────────────┘")
    print()


# ─── Calendar Watch ──────────────────────────────────────────────────────────

def get_today_events(calendars=None):
    """Read today's calendar events using macOS Calendar via osascript.

    Returns list of dicts: [{"title": str, "start": datetime, "end": datetime}, ...]
    """
    # Build calendar filter
    if calendars:
        cal_filter = " or ".join(f'name of its calendar is "{c}"' for c in calendars)
        cal_condition = f" and ({cal_filter})"
    else:
        cal_condition = ""

    script = f'''
    tell application "Calendar"
        set today to current date
        set time of today to 0
        set tomorrow to today + 1 * days
        set output to ""
        repeat with cal in calendars
            repeat with evt in (every event of cal whose start date ≥ today and start date < tomorrow and allday event is false{cal_condition})
                set evtStart to start date of evt
                set evtEnd to end date of evt
                set evtTitle to summary of evt
                -- Format: title|||YYYY-MM-DD HH:MM|||YYYY-MM-DD HH:MM
                set y to year of evtStart as text
                set mo to text -2 thru -1 of ("0" & ((month of evtStart) as integer))
                set d to text -2 thru -1 of ("0" & (day of evtStart))
                set h to text -2 thru -1 of ("0" & (hours of evtStart))
                set mi to text -2 thru -1 of ("0" & (minutes of evtStart))
                set startStr to y & "-" & mo & "-" & d & " " & h & ":" & mi

                set y2 to year of evtEnd as text
                set mo2 to text -2 thru -1 of ("0" & ((month of evtEnd) as integer))
                set d2 to text -2 thru -1 of ("0" & (day of evtEnd))
                set h2 to text -2 thru -1 of ("0" & (hours of evtEnd))
                set mi2 to text -2 thru -1 of ("0" & (minutes of evtEnd))
                set endStr to y2 & "-" & mo2 & "-" & d2 & " " & h2 & ":" & mi2

                set output to output & evtTitle & "|||" & startStr & "|||" & endStr & linefeed
            end repeat
        end repeat
        return output
    end tell
    '''

    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return []
    except (subprocess.TimeoutExpired, OSError):
        return []

    events = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split("|||")
        if len(parts) != 3:
            continue
        title, start_str, end_str = parts
        try:
            start_dt = datetime.strptime(start_str.strip(), "%Y-%m-%d %H:%M")
            end_dt = datetime.strptime(end_str.strip(), "%Y-%m-%d %H:%M")
            events.append({"title": title.strip(), "start": start_dt, "end": end_dt})
        except ValueError:
            continue

    events.sort(key=lambda e: e["start"])
    return events


WATCH_LOG = SCRIPT_DIR / "watch.log"
LAUNCHD_LABEL = "com.transcriber.watch"
LAUNCHD_PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def _log(msg, also_print=True):
    """Write to watch.log and optionally print."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    try:
        with open(WATCH_LOG, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass
    if also_print:
        print(f"  {msg}")


def cmd_watch(args):
    """Watch calendar and auto-record meetings."""
    import signal
    from datetime import timedelta

    watch_cfg = config.get("watch", {})
    silence_timeout = watch_cfg.get("silence_timeout_minutes", 10)
    min_recording = watch_cfg.get("min_recording_minutes", 2)
    model_name = watch_cfg.get("model", config.get("default_model", "small"))
    end_buffer = watch_cfg.get("end_buffer_minutes", 2)
    calendars = watch_cfg.get("calendars", []) or []
    refresh_hours = watch_cfg.get("refresh_hours", 3)

    _log("📅 Calendar Watch started")
    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  📅 Calendar Watch                               │")
    print("  │  Auto-records meetings from your calendar        │")
    print("  ├─────────────────────────────────────────────────┤")
    print(f"  │  Model:    {model_name:<38}│")
    print(f"  │  Silence:  {silence_timeout}min → auto-stop{' ' * 24}│")
    print(f"  │  Min rec:  {min_recording}min (discard if shorter){' ' * 13}│")
    print(f"  │  Buffer:   {end_buffer}min after meeting end{' ' * 16}│")
    print(f"  │  Refresh:  every {refresh_hours}h{' ' * 30}│")
    print(f"  │  Log:      watch.log{' ' * 28}│")
    print("  └─────────────────────────────────────────────────┘")

    while True:
        # ── Read today's events ────────────────────────────────
        events = get_today_events(calendars if calendars else None)
        now = datetime.now()
        last_refresh = now

        # Filter to future events only
        upcoming = [e for e in events if e["end"] > now]

        _log(f"📅 {len(upcoming)} meetings remaining today")
        if upcoming:
            for e in upcoming:
                start_str = e["start"].strftime("%H:%M")
                end_str = e["end"].strftime("%H:%M")
                duration = int((e["end"] - e["start"]).total_seconds() / 60)
                _log(f"   {start_str}-{end_str}  {e['title']}  ({duration}min)")

        # ── Process events or sleep ────────────────────────────
        if not upcoming:
            # Sleep until midnight or next refresh, whichever is sooner
            tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow += timedelta(days=1)
            next_refresh = now + timedelta(hours=refresh_hours)
            wake_at = min(tomorrow, next_refresh)
            wait = (wake_at - datetime.now()).total_seconds()

            if wake_at == tomorrow:
                _log(f"💤 No meetings — sleeping until midnight")
            else:
                _log(f"💤 No meetings — refreshing in {refresh_hours}h")
            try:
                time.sleep(max(1, wait))
            except KeyboardInterrupt:
                _log("👋 Watch stopped by user")
                return
            continue

        next_event = upcoming[0]
        wait_seconds = (next_event["start"] - datetime.now()).total_seconds()

        if wait_seconds > 0:
            # Sleep until meeting starts, but wake for refresh if needed
            refresh_seconds = refresh_hours * 3600
            sleep_time = min(wait_seconds, refresh_seconds)

            if sleep_time == wait_seconds:
                _log(f"💤 Next: \"{next_event['title']}\" at {next_event['start'].strftime('%H:%M')} ({format_duration(wait_seconds)})")
            else:
                _log(f"💤 Refreshing calendar in {refresh_hours}h (before \"{next_event['title']}\")")

            try:
                time.sleep(max(1, sleep_time))
            except KeyboardInterrupt:
                _log("👋 Watch stopped by user")
                return

            # If we woke up for refresh (not for the meeting), loop back
            if sleep_time < wait_seconds:
                _log("🔄 Refreshing calendar...")
                continue

        # ── Meeting time — start recording ─────────────────────
        _log(f"⏺  \"{next_event['title']}\" — auto-recording started")

        stop_time = next_event["end"] + timedelta(minutes=end_buffer)

        # Start recording as a subprocess
        safe_title = next_event["title"]
        proc = subprocess.Popen(
            [sys.executable, __file__,
             "live",
             "--model", model_name,
             "--title", safe_title,
             "--no-diarize"],
            stdin=subprocess.DEVNULL,
        )

        # Wait for meeting to end or process to finish
        try:
            while datetime.now() < stop_time:
                ret = proc.poll()
                if ret is not None:
                    _log(f"⏹  Recording ended on its own (silence timeout)")
                    break
                time.sleep(10)

            # If still running after meeting end + buffer, stop it
            if proc.poll() is None:
                _log(f"⏹  \"{next_event['title']}\" ended + {end_buffer}min buffer — stopping")
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=300)

        except KeyboardInterrupt:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=300)
            _log("👋 Watch stopped by user")
            return

        # ── Check if recording was too short ───────────────────
        if proc.returncode == 0:
            recordings = list_recordings(RECORDINGS_DIR)
            if recordings:
                latest = recordings[0]
                duration = get_audio_duration(str(latest))
                speech_minutes = duration / 60

                if speech_minutes < min_recording:
                    _log(f"🗑  \"{next_event['title']}\" too short ({format_duration(duration)}) — discarded")
                    try:
                        latest.unlink()
                    except OSError:
                        pass
                else:
                    _log(f"✅ \"{next_event['title']}\" recorded ({format_duration(duration)})")

        time.sleep(5)


def cmd_install_daemon(args):
    """Install launchd agent so watch runs automatically on login."""
    script_path = Path(__file__).resolve()
    venv_python = SCRIPT_DIR / ".venv" / "bin" / "python3"

    if not venv_python.exists():
        print("  ❌ Virtual environment not found. Run install.sh first.")
        return

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LAUNCHD_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{venv_python}</string>
        <string>{script_path}</string>
        <string>watch</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{SCRIPT_DIR}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>{SCRIPT_DIR}/watch-stdout.log</string>
    <key>StandardErrorPath</key>
    <string>{SCRIPT_DIR}/watch-stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
        <key>HF_TOKEN</key>
        <string>{HF_TOKEN}</string>
    </dict>
</dict>
</plist>
"""

    # Create LaunchAgents directory if needed
    LAUNCHD_PLIST.parent.mkdir(parents=True, exist_ok=True)

    # Stop existing daemon if running
    subprocess.run(["launchctl", "unload", str(LAUNCHD_PLIST)],
                   capture_output=True)

    # Write plist
    LAUNCHD_PLIST.write_text(plist_content)

    # Load daemon
    result = subprocess.run(["launchctl", "load", str(LAUNCHD_PLIST)],
                           capture_output=True, text=True)

    if result.returncode == 0:
        print()
        print("  ┌─────────────────────────────────────────────────┐")
        print("  │  ✅ Calendar Watch daemon installed!             │")
        print("  ├─────────────────────────────────────────────────┤")
        print("  │  Starts automatically on login                  │")
        print("  │  Restarts if it crashes                         │")
        print("  │  Runs silently in background                    │")
        print("  ├─────────────────────────────────────────────────┤")
        print(f"  │  Log:    watch.log{' ' * 30}│")
        print(f"  │  Plist:  ~/Library/LaunchAgents/{' ' * 16}│")
        print("  │                                                 │")
        print("  │  Commands:                                      │")
        print("  │    transcribe watch-status   Check status        │")
        print("  │    transcribe uninstall-daemon  Remove           │")
        print("  └─────────────────────────────────────────────────┘")
        print()
    else:
        print(f"  ❌ Failed to load daemon: {result.stderr.strip()}")


def cmd_uninstall_daemon(args):
    """Remove the launchd agent."""
    if not LAUNCHD_PLIST.exists():
        print("  ❌ Daemon not installed.")
        return

    subprocess.run(["launchctl", "unload", str(LAUNCHD_PLIST)],
                   capture_output=True)

    try:
        LAUNCHD_PLIST.unlink()
    except OSError as e:
        print(f"  ⚠  Could not remove plist: {e}")
        return

    print()
    print("  ✅ Calendar Watch daemon removed.")
    print("     It will no longer start on login.")
    print()


def cmd_watch_status(args):
    """Check if the watch daemon is running."""
    result = subprocess.run(
        ["launchctl", "list", LAUNCHD_LABEL],
        capture_output=True, text=True
    )

    print()
    if result.returncode == 0:
        print("  ✅ Calendar Watch daemon is running")
    else:
        print("  ⏹  Calendar Watch daemon is not running")

    if LAUNCHD_PLIST.exists():
        print(f"  📄 Plist: {LAUNCHD_PLIST}")
    else:
        print("  📄 Plist: not installed")

    if WATCH_LOG.exists():
        print(f"  📋 Log:   {WATCH_LOG}")
        # Show last 5 log lines
        try:
            lines = WATCH_LOG.read_text().strip().split("\n")
            print()
            print("  ── Recent activity ──")
            for line in lines[-5:]:
                print(f"  {line}")
        except OSError:
            pass
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interview Transcriber — Record, transcribe, save to Obsidian",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    transcribe rec                              Record an interview
    transcribe run                              Transcribe last recording
    transcribe run ~/Downloads/interview.mp3    Transcribe a specific file
    transcribe live                             Record + transcribe simultaneously
    transcribe list                             List & manage recordings
        """,
    )
    subparsers = parser.add_subparsers(dest="command")

    # rec
    subparsers.add_parser("rec", help="Start recording")

    # run
    run_parser = subparsers.add_parser("run", help="Transcribe a recording")
    run_parser.add_argument("audio", nargs="?", default=None, help="Audio file path")
    run_parser.add_argument("--model", "-m", default=None, help="Model: tiny/base/small/medium/large-v3")
    run_parser.add_argument("--speakers", "-s", default=None, help="Speaker names (comma-separated)")
    run_parser.add_argument("--title", "-t", default=None, help="Transcript title")
    run_parser.add_argument("--language", "-l", default=None, help="Language code (default: from config)")
    run_parser.add_argument("--no-diarize", action="store_true", help="Skip speaker identification (faster)")

    # live
    live_parser = subparsers.add_parser("live", help="Record + transcribe simultaneously")
    live_parser.add_argument("--model", "-m", default=None, help="Model (default: from config)")
    live_parser.add_argument("--speakers", "-s", default=None, help="Speaker names (comma-separated)")
    live_parser.add_argument("--title", "-t", default=None, help="Transcript title")
    live_parser.add_argument("--language", "-l", default=None, help="Language code (default: from config)")
    live_parser.add_argument("--no-diarize", action="store_true", help="Skip speaker identification (faster)")

    # enroll
    enroll_parser = subparsers.add_parser("enroll", help="Save your voice for auto-recognition")
    enroll_parser.add_argument("--name", "-n", default=None, help="Your name (default: from config)")

    # watch
    watch_parser = subparsers.add_parser("watch", help="Auto-record meetings from calendar")
    watch_parser.add_argument("--refresh", action="store_true", help="Force calendar refresh")

    # daemon management
    subparsers.add_parser("install-daemon", help="Install watch as background service")
    subparsers.add_parser("uninstall-daemon", help="Remove background service")
    subparsers.add_parser("watch-status", help="Check if watch daemon is running")

    # list
    subparsers.add_parser("list", help="List recordings")

    # setup
    subparsers.add_parser("setup", help="Set up audio devices for recording")

    args = parser.parse_args()

    if args.command == "rec":
        cmd_record(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "live":
        cmd_live(args)
    elif args.command == "enroll":
        cmd_enroll(args)
    elif args.command == "watch":
        cmd_watch(args)
    elif args.command == "install-daemon":
        cmd_install_daemon(args)
    elif args.command == "uninstall-daemon":
        cmd_uninstall_daemon(args)
    elif args.command == "watch-status":
        cmd_watch_status(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "setup":
        cmd_setup(args)
    else:
        parser.print_help()
        print()
        print("  Quick start:")
        print("    transcribe enroll — Save your voice (one time)")
        print("    transcribe setup  — Set up audio (first time)")
        print("    transcribe rec    — Record an interview")
        print("    transcribe run    — Transcribe a recording")
        print("    transcribe live   — Record + transcribe at once")
        print("    transcribe watch  — Auto-record calendar meetings")
        print("    transcribe list   — List recordings")
        print()


if __name__ == "__main__":
    main()
