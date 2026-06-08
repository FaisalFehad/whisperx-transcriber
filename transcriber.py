#!/usr/bin/env python3
"""
Transcriber — Record, transcribe, and save as Markdown.

Commands:
    transcribe rec                  Start recording (auto-stops on silence)
    transcribe run [file]           Transcribe a recording
    transcribe live                 Record + transcribe simultaneously
    transcribe watch                Auto-record meetings from calendar
    transcribe list                 List available recordings
    transcribe setup                Audio device setup guide
"""

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # suppress tqdm "Fetching N files" bars

import argparse
import copy
import gc
import json
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ui import format_duration

try:
    from system_audio import SystemAudioCapture as _SystemAudioCapture
except ImportError:
    _SystemAudioCapture = None

try:
    from voice_mic import VPMicCapture
    _VP_MIC_AVAILABLE = True
except (ImportError, RuntimeError):
    VPMicCapture = None  # type: ignore
    _VP_MIC_AVAILABLE = False


# ─── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"

DEFAULT_CONFIG = {
    # ── General ──────────────────────────────────────────────────────────
    # Transcription model: "parakeet" (recommended) or Whisper variants
    # Parakeet: CTC model — no hallucinations, proper punctuation, 2.5GB
    # Whisper: small.en, medium, large-v3
    "default_model": "parakeet",
    # Language code for transcription (e.g. "en", "ar", "fr")
    "language": "en",
    # Read HF_TOKEN from macOS keychain instead of environment variable
    "keychain": False,

    # ── File paths ───────────────────────────────────────────────────────
    "paths": {
        # Root folder for recordings and transcripts
        "base": "~/Transcriptions",
        # Subfolders under base (created automatically)
        "recordings_subdir": "Recordings",
        "scripts_subdir": "Scripts",
    },

    # ── Recording settings ───────────────────────────────────────────────
    "recording": {
        # Sample rate in Hz — 16000 is what Whisper was trained on (don't change)
        "sample_rate": 16000,
        # RMS level below this = silence (0.005 works for most mics)
        # Lower = more sensitive, higher = ignores quiet speech
        "silence_threshold": 0.010,
        # Minutes of continuous silence before auto-stopping recording
        "silence_timeout_minutes": 10,
        # Seconds of low mic level before showing warning
        "mic_low_warning_seconds": 10,
        # Virtual audio device names to exclude from mic selection
        # Add your virtual audio driver here if it shows up as a mic
        "virtual_device_names": ["Aggregate", "Multi-Output"],
    },

    # Audio file management after transcription
    "audio": {
        # Keep the source WAV after successful transcription (True) or delete it (False)
        "keep_recording": True,
        # Use Apple VoiceProcessingIO for hardware AEC during recording.
        # Removes system audio bleed from mic at the audio layer.
        "voice_processing": False,
    },

    # Accepted audio file types for transcription and listing
    "audio_formats": [".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".qta"],

    # ── Auto-title ─────────────────────────────────────────────────────
    # If true, uses current calendar event title as transcript name
    # Falls back to date-only if no event is happening or calendar unavailable
    "auto_title_from_calendar": True,

    # ── History logging ────────────────────────────────────────────────
    # Log each transcription run to history.jsonl for benchmarking.
    # Disabled by default — enable in config.json to track performance.
    "history": False,

    # ── Whisper-specific settings ────────────────────────────────────────
    # Only used when default_model is a Whisper model (not parakeet)
    "whisper": {
        # Skip silent segments where hallucination is detected (seconds).
        # Set to None to disable. Requires word_timestamps=True internally.
        "hallucination_silence_threshold": 2.0,
    },

    # ── Speaker diarisation (who said what) ──────────────────────────────
    "diarization": {
        # Set to false to always skip diarisation (faster, no speaker labels)
        "enabled": True,
        # Sortformer model (max 4 speakers per channel)
        "mlx_model": "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16",
        # Chunk duration in seconds for streaming mode (higher = more acoustic context)
        "mlx_chunk_duration": 20.0,
        # Speaker activity threshold (0-1). Lower = catch more speech, higher = fewer false positives
        "threshold": 0.5,
        # Ignore speaker segments shorter than this (seconds). Filters micro-segments
        "min_duration": 0.1,
        # Merge segments from same speaker closer than this gap (seconds). Reduces fragmentation
        "merge_gap": 0.2,
        # AOSC speaker cache size in frames (188 frames = ~15s). Larger = better speaker tracking
        "spkcache_max": 752,
        # FIFO buffer size in frames (188 frames = ~15s). Larger = more recent context
        "fifo_max": 752,
    },

    # ── Live mode (record + transcribe simultaneously) ───────────────────
    "live": {
        # Model for live mode — parakeet is fast enough for real-time (15x real-time on M1)
        "model": "parakeet",
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

    # ── Calendar watch (auto-record meetings) ────────────────────────────
    "watch": {
        # Silence timeout for watch recordings (same as recording default)
        "silence_timeout_minutes": 10,
        # Discard recordings shorter than this (catches non-meeting events)
        "min_recording_minutes": 2,
        # Model to use for watch auto-transcription
        "model": "parakeet",
        # Only watch these calendars (empty = all calendars)
        # Example: ["Work", "Personal"]
        "calendars": [],
        # Keep recording N minutes after calendar event ends
        # Catches meetings that run over
        "end_buffer_minutes": 2,
        # Re-read calendar every N hours (catches newly added meetings)
        "refresh_hours": 3,
        # Record only during meeting, transcribe after it ends.
        # Uses less CPU/GPU during the call and produces better results
        # since the full model runs on the complete audio.
        "record_only": True,
    },

    # ── Audio normalisation ──────────────────────────────────────────────
    # Scales to 0.05 RMS with peak headroom to prevent clipping.
    # Stereo channels normalised independently. Set to False to disable.
    "normalise": True,

    # ── Audio denoising ─────────────────────────────────────────────────
    "denoise": {
        # Spectral subtraction to reduce background noise before transcription.
        # Disabled by default — eval tests showed all models handle noise well
        # without preprocessing (+6-14% WER increase with denoise on).
        # Enable with --denoise flag for recordings with heavy constant noise
        # that start with silence (first N seconds used as noise profile).
        "enabled": True,
        # Over-subtraction factor: how aggressively to remove noise.
        # 1.0 = conservative (recommended). Higher = more removal but risks speech distortion.
        "factor": 1.0,
        # Seconds of audio to use as noise profile.
        "noise_profile_seconds": 3,
        # FFT window size and hop for spectral subtraction (advanced)
        "n_fft": 2048,
        "hop_length": 512,
    },

    # ── Parakeet-specific settings ─────────────────────────────────────
    "parakeet": {
        # Chunk duration in seconds for processing long audio files
        # Lower = less RAM but slightly slower. 30s works well for most files.
        "chunk_duration": 30.0,
    },

    # ── List command ─────────────────────────────────────────────────────
    "list": {
        # Max recordings to show in 'transcribe list'
        "max_recordings": 20,
    },

    # ── Model definitions ────────────────────────────────────────────────
    # speed_factor: multiplier for time estimation (audio_duration × factor)
    # Lower = faster model. These are calibrated for M1 Apple Silicon.
    "models": {
        "parakeet":  {"speed_factor": 0.5, "size": "~2.5GB", "ram_gb": 3.0, "description": "Best accuracy and speed (English) ★"},
        "small.en":  {"speed_factor": 0.5, "size": "~460MB", "ram_gb": 1.0, "description": "Fast, low RAM, English-only"},
        "medium":    {"speed_factor": 1.5, "size": "~1.5GB", "ram_gb": 2.0, "description": "Balanced multilingual"},
        "turbo":     {"speed_factor": 1.5, "size": "~800MB", "ram_gb": 1.5, "description": "Whisper turbo (slow on MLX)"},
        "large-v3":  {"speed_factor": 4.0, "size": "~3GB",   "ram_gb": 3.5, "description": "Most capable multilingual — slow"},
    },

    # ── MLX model repos (HuggingFace) ────────────────────────────────
    # Maps model names to mlx-community HF repos (auto-downloaded on first use)
    "mlx_models": {
        "parakeet":  "mlx-community/parakeet-tdt-0.6b-v3",
        "small.en":  "mlx-community/whisper-small.en-mlx",
        "medium":    "mlx-community/whisper-medium-mlx",
        "large-v3":  "mlx-community/whisper-large-v3-mlx",
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
    return copy.deepcopy(DEFAULT_CONFIG)


# Load config and apply
config = load_config()

if config.get("keychain"):
    _r = subprocess.run(
        ["security", "find-generic-password", "-s", "HF_TOKEN", "-a", os.environ.get("USER", ""), "-w"],
        capture_output=True, text=True
    )
    HF_TOKEN = _r.stdout.strip() if _r.returncode == 0 else ""
else:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
BASE_DIR = Path(config["paths"]["base"]).expanduser()
RECORDINGS_DIR = BASE_DIR / config["paths"]["recordings_subdir"]
SCRIPTS_DIR = BASE_DIR / config["paths"]["scripts_subdir"]

SAMPLE_RATE = config["recording"]["sample_rate"]
SILENCE_THRESHOLD = config["recording"]["silence_threshold"]
SILENCE_TIMEOUT = config["recording"]["silence_timeout_minutes"] * 60

AUDIO_FORMATS = tuple(config["audio_formats"])
MIC_LOW_WARNING_SECONDS = config["recording"]["mic_low_warning_seconds"]

MODEL_INFO = config["models"]

# ─── Debug Logging ────────────────────────────────────────────────────────────

_DEBUG = config.get("debug", False)

def _log(tag, **kwargs):
    """Pipeline debug log. Prints to stderr when debug=true or --debug flag."""
    if not _DEBUG:
        return
    parts = " ".join(f"{k}={v}" for k, v in kwargs.items())
    print(f"  [{tag}] {parts}", file=sys.stderr, flush=True)


# ─── Safe File Writing ────────────────────────────────────────────────────────

FALLBACK_DIR = Path.home() / "Downloads"


def _safe_filename(stem, suffix, max_bytes=200):
    """Truncate stem if needed so the full filename stays under max_bytes.

    macOS allows 255 bytes per filename. We leave room for suffix + " (99)".
    """
    # Reserve space for suffix and potential " (99)" dedup
    reserved = len(suffix.encode("utf-8")) + 6
    limit = max_bytes - reserved
    encoded = stem.encode("utf-8")
    if len(encoded) <= limit:
        return stem
    # Truncate at character boundary
    truncated = encoded[:limit].decode("utf-8", errors="ignore").rstrip()
    return truncated


def _unique_path(path):
    """Return path if it doesn't exist, otherwise append (2), (3), etc.
    Also ensures filename doesn't exceed macOS's 255-byte limit.

    Example: "2026-03-20 Meeting.md" → "2026-03-20 Meeting (2).md"
    """
    path = Path(path)
    stem = _safe_filename(path.stem, path.suffix)
    path = path.parent / f"{stem}{path.suffix}"
    if not path.exists():
        return path
    suffix = path.suffix
    parent = path.parent
    n = 2
    while True:
        candidate = parent / f"{stem} ({n}){suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def _atomic_write_text(path, content):
    """Write text via temp file + rename to prevent partial files on crash."""
    path = Path(path)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=path.suffix, prefix=".tmp_"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, str(path))
    except Exception:
        # Clean up temp file if rename fails
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _diagnose_write_error(err):
    """Return a human-readable reason and fix suggestion for a write error."""
    msg = str(err)
    if "Operation not permitted" in msg:
        return ("Permission denied — macOS is blocking access to this folder.",
                "System Settings → Privacy & Security → Full Disk Access → add your Terminal app, then restart terminal.")
    if "No space left" in msg:
        return ("Disk is full.",
                "Free up disk space and try again.")
    if "Read-only file system" in msg:
        return ("The destination is on a read-only volume.",
                "Check if the drive is mounted read-only or if iCloud sync is paused.")
    if "No such file or directory" in msg:
        return ("The destination folder doesn't exist and couldn't be created.",
                "Check the paths.base setting in config.json.")
    return (str(err), "Check the error above and fix the issue.")


def _ask_retry(description, err, target_path):
    """Show error diagnosis and ask user to retry or save to Downloads."""
    reason, fix = _diagnose_write_error(err)
    print(f"\n  ⚠️  Could not save {description} to:")
    print(f"      {target_path}")
    print(f"\n  Reason: {reason}")
    print(f"  Fix:    {fix}")
    print()
    while True:
        try:
            choice = input("  [R]etry  |  [S]ave to ~/Downloads  |  [Q]uit without saving: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = "s"
        if choice in ("r", "retry"):
            return "retry"
        if choice in ("s", "save", ""):
            return "fallback"
        if choice in ("q", "quit"):
            return "quit"
        print("  Please enter R, S, or Q.")


def _safe_write_text(target_path, content, description="file"):
    """Write text to target_path with retry prompt and ~/Downloads fallback.
    Never overwrites existing files — appends (2), (3), etc.

    Returns the path that was actually written to.
    """
    target_path = _unique_path(target_path)
    while True:
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(target_path, content)
            return target_path
        except Exception as primary_err:
            choice = _ask_retry(description, primary_err, target_path)
            if choice == "retry":
                continue
            if choice == "quit":
                # Last resort: dump to stdout so user can copy-paste
                print(f"\n  ── {description.upper()} CONTENT (copy this!) {'─' * 30}")
                print(content)
                print(f"  ── END {description.upper()} {'─' * 40}\n")
                return None
            # fallback to ~/Downloads
            fallback = _unique_path(FALLBACK_DIR / target_path.name)
            try:
                _atomic_write_text(fallback, content)
                print(f"  ✅ Saved to: {fallback}")
                return fallback
            except Exception as fallback_err:
                print(f"\n  ❌ Fallback also failed: {fallback_err}")
                print(f"\n  ── {description.upper()} CONTENT (copy this!) {'─' * 30}")
                print(content)
                print(f"  ── END {description.upper()} {'─' * 40}\n")
                return None


def _write_wav(path, audio, sr):
    """Write float32 [-1, 1] array as 16-bit mono WAV (stdlib)."""
    audio_int16 = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())


def _write_audio_wav(path, audio_data, sample_rate):
    """Write audio WAV and return the path; raise on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_wav(path, audio_data, sample_rate)
    return path


def _remove_empty_file(path):
    """Best-effort delete of a zero-byte file (a partial write artifact)."""
    try:
        if path.exists() and path.stat().st_size == 0:
            path.unlink()
    except OSError:
        pass


def _try_fallback_save(audio_data, sample_rate, original_path):
    """Try writing to ~/Downloads. Returns the fallback path or None on failure."""
    fallback = _unique_path(FALLBACK_DIR / original_path.name)
    try:
        _write_audio_wav(fallback, audio_data, sample_rate)
        print(f"  ✅ Saved to: {fallback}")
        return fallback
    except Exception as fallback_err:
        _remove_empty_file(fallback)
        print(f"\n  ❌ Fallback also failed: {fallback_err}")
        return None


def _safe_write_audio(target_path, audio_data, sample_rate, description="audio"):
    """Write audio to target_path with retry prompt and ~/Downloads fallback.
    Never overwrites existing files — appends (2), (3), etc.
    Cleans up partial files on failure.

    Returns the path that was actually written to.
    """
    target_path = _unique_path(target_path)
    while True:
        try:
            return _write_audio_wav(target_path, audio_data, sample_rate)
        except Exception as primary_err:
            _remove_empty_file(target_path)
            choice = _ask_retry(description, primary_err, target_path)
            if choice == "retry":
                continue
            if choice == "quit":
                print(f"\n  ❌ {description} not saved. Audio data lost.")
                return None
            return _try_fallback_save(audio_data, sample_rate, target_path)


def _save_checkpoint(result, audio_path, title=None):
    """Save raw transcription result as JSON checkpoint.

    Called immediately after transcription completes, before formatting.
    If a later step (speaker resolve, format, save) crashes, the raw data
    survives and can be recovered.

    Returns the checkpoint path, or None if save failed.
    """
    checkpoint_dir = SCRIPT_DIR / ".checkpoints"
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = Path(audio_path).stem[:80]
        cp_path = checkpoint_dir / f"{timestamp}_{safe_name}.json"

        checkpoint = {
            "audio_path": str(audio_path),
            "title": title,
            "timestamp": timestamp,
            "segments": result.get("segments", []),
            "language": result.get("language", ""),
        }
        # Segments may contain numpy types — convert to plain Python
        cp_json = json.dumps(checkpoint, default=str, ensure_ascii=False, indent=2)
        cp_path.write_text(cp_json, encoding="utf-8")
        return cp_path
    except Exception:
        return None


def _remove_checkpoint(cp_path):
    """Remove checkpoint after successful save."""
    if cp_path:
        try:
            Path(cp_path).unlink()
        except OSError:
            pass


# ─── History Logging ─────────────────────────────────────────────────────────

HISTORY_LOG = SCRIPT_DIR / "history.jsonl"


def _log_run(metadata, step_times=None):
    """Append a transcription run to history.jsonl (if history is enabled in config)."""
    if not config.get("history", False):
        return
    entry = {
        "timestamp": datetime.now().isoformat(),
        **metadata,
    }
    if step_times:
        entry["steps"] = {name: round(t, 1) for name, t in step_times.items()}
    try:
        with open(HISTORY_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def _load_history_runs():
    """Parse history.jsonl into a list of dicts (skips blank/bad lines)."""
    runs = []
    with open(HISTORY_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return runs


def _format_history_row(run):
    """Format one run as a fixed-width table row."""
    date = run.get("timestamp", "")[:10]
    model = run.get("model", "?")[:9]
    mode = run.get("mode", "run")[:5]
    audio_dur = run.get("audio_duration", 0)
    proc_time = run.get("processing_time", 0)
    speed = f"{audio_dur / proc_time:.1f}x" if proc_time > 0 else "?"
    segments = run.get("segments", "?")
    speakers = run.get("speakers", "?")
    title = run.get("title", run.get("audio_file", "?"))
    if len(title) > 25:
        title = title[:22] + "..."
    return (f"  {date:<12} {model:<10} {mode:<6} "
            f"{format_duration(audio_dur):>7} {format_duration(proc_time):>7} "
            f"{speed:>7} {segments:>5} {speakers:>4}  {title}")


def _per_model_speeds(runs):
    """Group speed ratios (audio_dur / proc_time) by model name."""
    stats = defaultdict(list)
    for run in runs:
        model = run.get("model")
        proc_time = run.get("processing_time", 0)
        if not model or proc_time <= 0:
            continue
        stats[model].append(run.get("audio_duration", 0) / proc_time)
    return stats


def _print_model_stats_table(model_stats):
    """Print the per-model speed summary table."""
    print()
    print(f"  {'Model':<10} {'Runs':>5} {'Avg Speed':>10} {'Min':>7} {'Max':>7}")
    print(f"  {'─'*10} {'─'*5} {'─'*10} {'─'*7} {'─'*7}")
    for model, speeds in sorted(model_stats.items()):
        avg = sum(speeds) / len(speeds)
        print(f"  {model:<10} {len(speeds):>5} {avg:>9.1f}x "
              f"{min(speeds):>6.1f}x {max(speeds):>6.1f}x")


def cmd_history(args):
    """Show transcription run history with performance stats."""
    if not HISTORY_LOG.exists():
        print("  No history yet.")
        if not config.get("history", False):
            print("  History is disabled. Enable with: \"history\": true in config.json")
        return

    runs = _load_history_runs()
    if not runs:
        print("  No history yet.")
        return

    print()
    print(f"  {'Date':<12} {'Model':<10} {'Mode':<6} {'Audio':>7} {'Time':>7} {'Speed':>7} {'Seg':>5} {'Spk':>4}  Title")
    print(f"  {'─'*12} {'─'*10} {'─'*6} {'─'*7} {'─'*7} {'─'*7} {'─'*5} {'─'*4}  {'─'*25}")
    for run in runs:
        print(_format_history_row(run))

    model_stats = _per_model_speeds(runs)
    if model_stats:
        _print_model_stats_table(model_stats)
    print()


# ─── Display Helpers ──────────────────────────────────────────────────────────

def clear_line():
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


def level_meter(rms, width=20):
    """Create a visual audio level meter."""
    level = min(1.0, rms * 15)
    filled = int(level * width)
    return "█" * filled + "░" * (width - filled)


def spinner_char(tick):
    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    return chars[tick % len(chars)]


def _rms_emoji(rms):
    """Return a colored emoji for an RMS level."""
    from ui import rms_colour
    return {"green": "🟢", "yellow": "🟡", "red": "🔴"}[rms_colour(rms)]


def get_available_ram_gb():
    """Get available RAM in GB (macOS)."""
    try:
        # Use vm_stat to estimate free memory
        vm = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=5
        )
        free_pages = 0
        for line in vm.stdout.splitlines():
            if "free" in line.lower() or "inactive" in line.lower():
                nums = re.findall(r"(\d+)", line)
                if nums:
                    free_pages += int(nums[-1])
        # Each page is 16KB on Apple Silicon
        available = (free_pages * 16384) / (1024 ** 3)
        return available
    except Exception:
        return -1  # unknown


def check_ram_for_model(model_name):
    """Check if there's enough RAM for the model. Returns warning message or None."""
    available = get_available_ram_gb()
    if available < 0:
        return None  # can't determine

    needed = MODEL_INFO.get(model_name, {}).get("ram_gb", 2.0)

    if available < needed:
        return f"⚠ Low RAM ({available:.1f}GB free, model needs ~{needed:.0f}GB) — may be slow or crash"
    elif available < needed + 2:
        return f"⚠ Limited RAM ({available:.1f}GB free) — close to model requirements"
    return None


# ─── Audio Device Detection ───────────────────────────────────────────────────


def _cleanup_temp_files(*paths):
    """Safe cleanup of multiple temporary files."""
    for p in paths:
        if p:
            try:
                os.unlink(p)
            except OSError:
                pass


def _transcribe_audio(audio_path, model_name, language="en", **kwargs):
    """Transcribe audio using appropriate engine (Parakeet or mlx-whisper)."""
    if _is_parakeet(model_name):
        return _transcribe_parakeet(audio_path, model_name)
    return _transcribe_mlx(audio_path, model_name, language, **kwargs)


def _setup_audio_devices():
    """Get recording configuration. Returns (mic_name, dual_mode, source_label, output_path) or raises.

    Mic is captured via VoiceProcessingIO (Apple's hardware AEC). No device
    enumeration is needed — VPIO opens the system's default input directly.
    System audio is captured via ScreenCaptureKit.
    """
    if not _VP_MIC_AVAILABLE:
        print("  ❌ VoiceProcessingIO not available.")
        print("     Install: pip install pyobjc-framework-AVFoundation")
        raise RuntimeError("VoiceProcessingIO not available")
    mic_name = "Mac mic (VPIO + AEC)"
    dual_mode = _SystemAudioCapture is not None
    source_label = f"System + {mic_name}" if dual_mode else mic_name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = _unique_path(RECORDINGS_DIR / f"{timestamp}.wav")
    return mic_name, dual_mode, source_label, output_path


def _save_transcript(result, audio_path, title, model_name, language,
                     audio_duration, processing_time, cp_path=None):
    """Save transcription result as Markdown. Returns (output_file, speaker_names)."""
    if not title:
        title = Path(audio_path).stem.replace("_", " ").replace("-", " ").title()
    speaker_names = _rename_speakers(result)
    date_str = extract_recording_date(str(audio_path))
    metadata = {
        "model": model_name,
        "engine": "mlx",
        "language": language,
        "audio_duration": audio_duration,
        "processing_time": processing_time,
        "audio_file": Path(audio_path).name,
    }
    markdown = format_transcript(result, title, speaker_names, date_str, metadata)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in title).strip()
    if not safe_title:
        safe_title = "Untitled"
    output_file = SCRIPTS_DIR / f"{date_str} {safe_title}.md"
    saved_to = _safe_write_text(output_file, markdown, description="transcript")
    if saved_to:
        output_file = saved_to

    if cp_path:
        _remove_checkpoint(cp_path)

    return output_file, speaker_names





def cmd_setup(args):
    """Check audio setup status."""
    sck_status = "✅ ScreenCaptureKit (system audio)" if _SystemAudioCapture is not None else "❌ pip install pyobjc-framework-ScreenCaptureKit"
    vp_status = "✅ VoiceProcessingIO (mic + hardware AEC)" if _VP_MIC_AVAILABLE else "❌ pip install pyobjc-framework-AVFoundation"

    from ui import info_panel
    print()
    info_panel("🔧 Audio Setup", [
        ("System:", sck_status),
        ("Mic:", vp_status),
    ])

    if _SystemAudioCapture is not None and _VP_MIC_AVAILABLE:
        print()
        print("  ✅ Ready! Mic captured via VoiceProcessingIO (hardware AEC).")
        print("     System audio captured via ScreenCaptureKit.")
        print("     No virtual audio drivers needed.")
        print("     Volume keys work normally.")
    else:
        print()
        if not _VP_MIC_AVAILABLE:
            print("  ⚠  VoiceProcessingIO not available.")
            print("     pip install pyobjc-framework-AVFoundation")
        if not _SystemAudioCapture:
            print("  ⚠  System audio not available. Only mic will be recorded.")
            print("     pip install pyobjc-framework-ScreenCaptureKit")

    print()


# ─── Recording ────────────────────────────────────────────────────────────────

def _mic_pump(vp_capture, state):
    """Pump mic frames from VPMicCapture into `state["mic_frames"]`.

    Shared by `cmd_record` (called from the main recording loop) and
    `cmd_live` (called from a dedicated `_mic_pump_thread`).

    Returns True if recording should continue, False to signal stop.
    """
    chunk, _overflow = vp_capture.read_chunk(timeout=0.1)
    if chunk is None or len(chunk) == 0:
        return True
    if state.get("paused", False):
        return True
    rms = float(np.sqrt(np.mean(chunk ** 2)))
    mono2d = chunk.reshape(-1, 1).astype(np.float32)
    with state["lock"]:
        state["mic_frames"].append(mono2d)
        if "mic_rms" in state:
            state["mic_rms"] = rms
        if "current_rms" in state:
            state["current_rms"] = max(state["current_rms"], rms)
        if rms < SILENCE_THRESHOLD:
            if state["silence_start"] is None:
                state["silence_start"] = time.time()
            elif time.time() - state["silence_start"] >= SILENCE_TIMEOUT:
                state["recording"] = False
                return False
        else:
            state["silence_start"] = None
    return True


def _mic_pump_thread(vp_capture, state, stop_event):
    """Dedicated mic-pump thread used by `cmd_live`.

    Polls VPMicCapture and appends to state['mic_frames'] until either
    `stop_event` is set or the auto-stop silence threshold trips.
    """
    while not stop_event.is_set() and state.get("recording", True):
        if not _mic_pump(vp_capture, state):
            return


def _sys_audio_callback(indata, _frames, _time, _status, *, state):
    """SCK callback for the system channel — fills sys_frames + resets silence timer."""
    if not state.get("recording", True):
        return
    if state.get("paused", False):
        return
    mono = indata.mean(axis=1, keepdims=True) if indata.shape[1] > 1 else indata
    with state["lock"]:
        state["sys_frames"].append(mono.copy())
        if "sys_rms" in state:
            state["sys_rms"] = float(np.sqrt(np.mean(mono ** 2)))
        if state.get("sys_rms", 0) > SILENCE_THRESHOLD:
            state["silence_start"] = None


def _toggle_pause(key, paused, silence_ref):
    """Toggle pause on 'p' key. Pause is a state, not silence — clear the timer."""
    if key != "p":
        return paused, silence_ref
    new_paused = not paused
    return new_paused, None if new_paused else silence_ref


def _mic_low_warning(mic_rms, mic_low_since, paused=False, prefix=""):
    """Return (warning_text, updated_mic_low_since) for the mic-too-quiet warning.

    In record mode, paused is a toggle; in live mode, paused is always False.
    prefix is prepended to the warning text (e.g. " ⚠" for live mode warnings
    that appear inline in the status line).
    """
    no_warning = "" if prefix else None
    if paused or mic_rms >= 0.001:
        return no_warning, None
    if mic_low_since is None:
        return no_warning, time.time()
    if time.time() - mic_low_since > MIC_LOW_WARNING_SECONDS:
        return f"{prefix}Mic very low!", mic_low_since
    return no_warning, mic_low_since


def _maybe_system_health_warning(last_check, current_warning):
    """Run check_system_health() at most every 30s; merge with current warning."""
    from ui import check_system_health
    if time.time() - last_check <= 30:
        return last_check, current_warning
    health = check_system_health()
    if health:
        merged = health if not current_warning else f"{current_warning} │ {health}"
        return time.time(), merged
    return time.time(), current_warning


def _run_recording_loop(state, vp_capture, display, dual_mode):
    """Main display/keyboard loop. Polls VPMicCapture and SCK, drives silence detection."""
    from ui import raw_keys, poll_key
    last_health_check = 0
    mic_low_since_rec = None
    with raw_keys(), display:
        while state["recording"]:
            paused, _sil_ref = _toggle_pause(poll_key(), state["paused"], state["silence_start"])
            state["paused"] = paused
            if paused:
                state["silence_start"] = None

            # Pump mic (VPIO) — returns False when silence timeout trips
            if not _mic_pump(vp_capture, state):
                state["recording"] = False
                break

            # System audio is filled by SCK's callback (see cmd_record / cmd_live).

            with state["lock"]:
                mic_rms = state["mic_rms"]
                sys_rms_val = state["sys_rms"]
                sil = state["silence_start"]

            rms = max(mic_rms, sys_rms_val)
            elapsed = time.time() - state["start_time"]
            silence_elapsed = time.time() - sil if sil else 0

            mic_low_warn, mic_low_since_rec = _mic_low_warning(mic_rms, mic_low_since_rec, paused=paused)
            last_health_check, warning = _maybe_system_health_warning(last_health_check, mic_low_warn)

            display.update(rms, elapsed, silence_elapsed, paused=paused,
                           mic_rms=mic_rms if dual_mode else None,
                           sys_rms=sys_rms_val if dual_mode else None)
            time.sleep(0.1)
    return True


def _print_record_complete(output_path, elapsed, auto_stopped):
    """Print the success panel after a recording finishes."""
    from ui import success_panel
    stop_reason = f"Auto-stopped after {SILENCE_TIMEOUT // 60}min silence" if auto_stopped else "Recording stopped"
    file_size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0
    print()
    success_panel(f"⏹ {stop_reason}", [
        ("File:", output_path.name),
        ("Duration:", format_duration(elapsed)),
        ("Size:", f"{file_size_mb:.1f} MB"),
        ("Location:", str(output_path.parent)),
    ])
    print()


def cmd_record(args):
    """Record audio with silence auto-stop. Captures system + mic by default."""
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        _mic_name, dual_mode, source_label, output_path = _setup_audio_devices()
    except RuntimeError:
        return

    from ui import info_panel
    print()
    info_panel("🎙 Recording", [
        ("Source:", source_label),
        ("Silence:", f"Auto-stops after {SILENCE_TIMEOUT // 60}min"),
        ("Controls:", "P = pause, Ctrl+C = stop"),
    ])
    print()

    state = {
        "mic_frames": [],
        "sys_frames": [],
        "recording": True,
        "paused": False,
        "silence_start": None,
        "mic_rms": 0.0,
        "sys_rms": 0.0,
        "start_time": time.time(),
        "lock": threading.Lock(),
        "mic_switch_frames": [],
    }

    vp_capture = VPMicCapture(sample_rate=SAMPLE_RATE,
                               voice_processing=config["audio"].get("voice_processing", False))

    def sys_callback(indata, frames, time_info, status):
        _sys_audio_callback(indata, frames, time_info, status, state=state)

    sck_capture = None
    if _SystemAudioCapture is not None:
        try:
            sck_capture = _SystemAudioCapture(sample_rate=SAMPLE_RATE, callback=sys_callback)
        except RuntimeError as e:
            print(f"  ⚠  System audio unavailable: {e}")
            dual_mode = False

    from ui import RecordingDisplay
    auto_stopped = False
    display = RecordingDisplay()
    try:
        vp_capture.start()
        if sck_capture:
            sck_capture.start()
        _run_recording_loop(state, vp_capture, display, dual_mode)
        auto_stopped = True
    except KeyboardInterrupt:
        state["recording"] = False
    finally:
        try:
            vp_capture.stop()
        except Exception:
            pass
        if sck_capture:
            sck_capture.stop()
    elapsed = time.time() - state["start_time"]

    if not state["mic_frames"]:
        print("  ⚠  No audio recorded.")
        return

    audio_data = _mix_audio(state["mic_frames"], state["sys_frames"], dual_mode,
                            normalise=config.get("normalise", True),
                            mic_switch_points=state["mic_switch_frames"] or None)
    saved_to = _safe_write_audio(output_path, audio_data, SAMPLE_RATE, description="recording")
    if not saved_to:
        print("  ⚠  Recording not saved.")
        return
    output_path = saved_to

    _print_record_complete(output_path, elapsed, auto_stopped)
    prompt_transcribe(str(output_path))


def _concat_mic_segments(mic_frames, mic_switch_points):
    """Concatenate mic frames, optionally normalising per device segment."""
    if not (mic_switch_points):
        return np.concatenate(mic_frames)
    boundaries = [0] + list(mic_switch_points) + [len(mic_frames)]
    parts = []
    for i in range(len(boundaries) - 1):
        seg = mic_frames[boundaries[i]:boundaries[i + 1]]
        if seg:
            parts.append(_normalise(np.concatenate(seg)))
    return np.concatenate(parts) if parts else np.array([], dtype=np.float32)


def _normalise_mic(mic_frames, normalise, mic_switch_points):
    """Concatenate + (optionally) normalise mic frames into a single 1-D array."""
    if not mic_frames:
        return np.array([], dtype=np.float32)
    if normalise and mic_switch_points:
        return _concat_mic_segments(mic_frames, mic_switch_points)
    mic_audio = np.concatenate(mic_frames)
    return _normalise(mic_audio) if normalise else mic_audio


def _pad_to_length(audio, target_len):
    """Right-pad `audio` with zeros so its length matches target_len."""
    if len(audio) < target_len:
        return np.pad(audio, (0, target_len - len(audio)))
    return audio



def _mix_audio(mic_frames, sys_frames, dual_mode, normalise=True, mic_switch_points=None):
    """Combine mic + system audio. Returns stereo (N,2) when dual, mono (N,) when single.

    Stereo format: channel 0 = mic, channel 1 = system audio.
    Each channel is independently normalised before combining (unless normalise=False).
    If mic_switch_points is provided, normalises each mic segment independently
    to handle volume differences between devices (e.g. AirPods vs MacBook mic).
    """
    if not mic_frames:
        return np.array([], dtype=np.float32)

    mic_audio = _normalise_mic(mic_frames, normalise, mic_switch_points)
    if dual_mode and sys_frames:
        sys_audio = np.concatenate(sys_frames) if sys_frames else np.array([], dtype=np.float32)
        if normalise and sys_audio.size:
            sys_audio = _normalise(sys_audio)
        mic_audio = np.asarray(mic_audio).reshape(-1)
        sys_audio = np.asarray(sys_audio).reshape(-1)
        max_len = max(len(mic_audio), len(sys_audio))
        mic_audio = _pad_to_length(mic_audio, max_len)
        sys_audio = _pad_to_length(sys_audio, max_len)
        _log("mix", mode="stereo", normalise=normalise, samples=max_len,
             mic_rms=f"{np.sqrt(np.mean(mic_audio**2)):.5f}",
             sys_rms=f"{np.sqrt(np.mean(sys_audio**2)):.5f}")
        return np.column_stack([mic_audio, sys_audio])

    mic_audio = np.asarray(mic_audio).reshape(-1)
    _log("mix", mode="mono", normalise=normalise, samples=len(mic_audio),
         rms=f"{np.sqrt(np.mean(mic_audio**2)):.5f}")
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
        title = input("  Title (or Enter for auto): ").strip()

        run_args = argparse.Namespace(
            audio=audio_path,
            model=model,
            title=title or None,
            language=config["language"],
            no_diarise=False,
            denoise=False,
            no_denoise=False,
            debug=False,
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

    from ui import console
    from rich.table import Table
    from rich.panel import Panel

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("num", style="dim", width=3)
    table.add_column("name", style="bold", width=11)
    table.add_column("desc")
    table.add_column("star", width=1)
    for i, (name, info) in enumerate(MODEL_INFO.items(), 1):
        star = "★" if name == default_model else ""
        table.add_row(f"{i})", name, info.get("description", ""), f"[yellow]{star}[/yellow]")
    print()
    console.print(Panel(table, title="[bold]Choose transcription model[/bold]", border_style="dim", expand=False, width=54))

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


# ─── Shared Helpers ──────────────────────────────────────────────────────────

def _validate_model(model_name):
    """Check model name is valid. Returns True if valid, prints error and returns False if not."""
    valid_models = set(MODEL_INFO.keys()) | set(config["mlx_models"].keys())
    if model_name not in valid_models:
        print(f"  ❌ Unknown model '{model_name}'")
        print(f"  Available: {', '.join(sorted(valid_models))}")
        return False
    return True


def _check_diarise(no_diarise):
    """Check if diarisation is enabled; warn if HF_TOKEN is missing. Returns bool."""
    wanted = config["diarization"]["enabled"] and not no_diarise
    if wanted and not HF_TOKEN:
        print("  ⚠  Skipping speaker ID — HF_TOKEN not set (see README)")
        return False
    return wanted



def _offset_and_filter_segments(segments, offset, overlap_boundary):
    """Apply time offset to segments and keep only those whose midpoint exceeds overlap_boundary."""
    result = []
    for seg in segments:
        start = seg.get("start", 0) + offset
        end = seg.get("end", 0) + offset
        seg["start"] = start
        seg["end"] = end
        # Midpoint filter: avoids cutting segments that straddle the boundary
        if (start + end) / 2 > overlap_boundary:
            result.append(seg)
    return result


# ─── Live Record + Transcribe ────────────────────────────────────────────────

def _preload_live_model(model_name):
    """Pre-load model and print the live-mode banner. Returns False on failure."""
    from ui import console
    from rich.panel import Panel
    print()
    console.print(Panel("[bold]🎙 Live Record + Transcribe[/bold]", border_style="dim", expand=False, width=54))
    print()
    print(f"  Loading '{model_name}' model...")
    try:
        if _is_parakeet(model_name):
            from mlx_audio.stt import load as load_stt
            load_stt(config["mlx_models"].get(model_name, "mlx-community/parakeet-tdt-0.6b-v3"))
        else:
            pass
    except Exception as e:
        print(f"  ❌ Failed to load model '{model_name}': {e}")
        print("  Check your internet connection — models download on first use.")
        return False
    print("  ✅ Model ready — starting recording")
    return True




def _get_mixed_snapshot(lock, mic_frames, sys_frames, dual_mode):
    """Get a normalized mono mix of the current frame buffer (None if empty)."""
    with lock:
        if not mic_frames:
            return None
        mic_copy = list(mic_frames)
        sys_copy = list(sys_frames) if dual_mode else []
    mic_audio = _normalise(np.concatenate(mic_copy).flatten())
    if sys_copy:
        sys_audio = _normalise(np.concatenate(sys_copy).flatten())
        min_len = min(len(mic_audio), len(sys_audio))
        return (mic_audio[:min_len] + sys_audio[:min_len]) / 2.0
    return mic_audio


def _adapt_chunking(adaptive, audio_seconds, process_seconds):
    """Tune chunk interval / mode based on processing-speed ratio."""
    if audio_seconds <= 0:
        return
    ratio = process_seconds / audio_seconds
    if ratio > adaptive["rec_only_ratio"]:
        adaptive["rec_only"] = True
        adaptive["status_msg"] = "⚠  System busy — record only (will transcribe after)"
    elif ratio > adaptive["struggle_ratio"]:
        adaptive["interval"] = min(adaptive["max_interval"], int(adaptive["interval"] * 1.5))
        adaptive["status_msg"] = f"↑ interval={adaptive['interval']}s"
    elif ratio < 0.3:
        adaptive["interval"] = max(adaptive["min_interval"], int(adaptive["interval"] * 0.75))
        adaptive["status_msg"] = f"↓ interval={adaptive['interval']}s"
    else:
        adaptive["status_msg"] = ""


def _transcription_worker(state, model_name, language, live_cfg, adaptive, stop_event,
                          transcribe_lock, transcribed_segments, get_snapshot):
    """Background thread: transcribe chunks during recording with adaptive pacing.

    Mutates state['last_transcribed_sample'], state['chunks_done'], state['chunk_in_progress'].
    """
    overlap_seconds = live_cfg["chunk_overlap_seconds"]
    min_chunk_secs = live_cfg["min_chunk_seconds"]
    overlap_samples = int(SAMPLE_RATE * overlap_seconds)

    while not stop_event.is_set():
        stop_event.wait(timeout=adaptive["interval"])
        if stop_event.is_set() or adaptive["rec_only"]:
            if stop_event.is_set():
                break
            continue

        audio = get_snapshot()
        if audio is None:
            continue

        total_samples = len(audio)
        new_samples = total_samples - state["last_transcribed_sample"]
        if new_samples < SAMPLE_RATE * min_chunk_secs:
            continue

        chunk_start = max(0, state["last_transcribed_sample"] - overlap_samples)
        chunk = audio[chunk_start:total_samples].astype(np.float32)
        offset = chunk_start / SAMPLE_RATE
        overlap_boundary = state["last_transcribed_sample"] / SAMPLE_RATE
        audio_seconds = len(chunk) / SAMPLE_RATE

        state["chunk_in_progress"] = True
        process_start = time.time()
        try:
            result = _transcribe_chunk(chunk, model_name, language)
            process_seconds = time.time() - process_start

            segments = _offset_and_filter_segments(
                result.get("segments", []), offset, overlap_boundary)
            with transcribe_lock:
                transcribed_segments.extend(segments)
            state["last_transcribed_sample"] = total_samples
            state["chunks_done"] += 1

            _adapt_chunking(adaptive, audio_seconds, process_seconds)
        except Exception as e:
            print(f"\n  ⚠ Chunk transcription error: {e}", file=sys.stderr)
        finally:
            state["chunk_in_progress"] = False


def _build_live_status_line(rms, m_rms, sys_rms, dual_mode, elapsed, chunks_done, chunk_in_progress, rec_only, mic_warn):
    """Compose the single-line status shown during live recording."""
    quality = f"Mic{_rms_emoji(m_rms)}"
    if dual_mode:
        quality += f" Sys{_rms_emoji(sys_rms)}"
    if rec_only:
        chunk_status = " │ REC only"
    else:
        chunk_status = f" │ Chunks: {chunks_done}"
        if chunk_in_progress:
            chunk_status += " ⟳"
    meter = level_meter(rms)
    return (
        f"  ⏺  {format_duration(elapsed)} │ "
        f"{meter} │ {quality}"
        f"{chunk_status}{mic_warn}"
    )


def _run_live_display_loop(state, dual_mode):
    """UI / keyboard / display loop for live recording. Mutates state['mic_low_since']."""
    last_adaptive_msg = ""
    while state["recording"]:
        with state["lock"]:
            rms = state["current_rms"]
            m_rms = state["mic_rms"]
            sys_rms = state["sys_rms"]

        elapsed = time.time() - state["start_time"]

        mic_warn, state["mic_low_since"] = _mic_low_warning(
            m_rms, state["mic_low_since"], prefix=" ⚠")

        status_line = _build_live_status_line(
            rms, m_rms, sys_rms, dual_mode, elapsed,
            state["chunks_done"], state["chunk_in_progress"],
            state["adaptive"]["rec_only"], mic_warn,
        )

        if state["adaptive"]["status_msg"] and state["adaptive"]["status_msg"] != last_adaptive_msg:
            clear_line()
            print(f"  {state['adaptive']['status_msg']}")
            last_adaptive_msg = state["adaptive"]["status_msg"]
        clear_line()
        sys.stdout.write(status_line)
        sys.stdout.flush()
        time.sleep(0.3)


def _post_process_stereo(audio_data, model_name, language, live_cfg,
                        last_transcribed_sample, rec_only, transcribed_segments):
    """Run stereo post-processing (split + per-channel ASR + merge). Returns segments."""
    tmp_fd, stereo_tmp = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        _write_wav(stereo_tmp, audio_data, SAMPLE_RATE)
        overlap_seconds = live_cfg["chunk_overlap_seconds"]
        remaining = len(audio_data) - last_transcribed_sample
        if rec_only and last_transcribed_sample == 0:
            return _stereo_full_transcribe(stereo_tmp, model_name, language)
        if remaining > SAMPLE_RATE * overlap_seconds:
            return _stereo_tail_transcribe(audio_data, last_transcribed_sample,
                                           overlap_seconds, model_name, language,
                                           base_segments=transcribed_segments)
        return transcribed_segments
    except Exception as e:
        print(f"  ⚠  Stereo post-processing failed: {e}")
        return transcribed_segments
    finally:
        _cleanup_temp_files(stereo_tmp)


def _stereo_full_transcribe(stereo_tmp, model_name, language):
    """Transcribe a full stereo file by splitting into mic+sys and interleaving."""
    remaining_segments = []
    print("  ⏳ Transcribing full recording (stereo)...")
    mic_tmp, sys_tmp = _split_stereo(stereo_tmp)
    try:
        mic_r = _transcribe_audio(mic_tmp, model_name, language)
        sys_r = _transcribe_audio(sys_tmp, model_name, language)
        merged = _interleave_by_timestamp(mic_r, sys_r)
        remaining_segments = merged.get("segments", [])
    except Exception as e:
        print(f"  ⚠  Transcription failed: {e}")
    finally:
        _cleanup_temp_files(mic_tmp, sys_tmp)
    return remaining_segments


def _stereo_tail_transcribe(audio_data, last_transcribed_sample, overlap_seconds,
                            model_name, language, *, base_segments):
    """Transcribe the final overlapping tail as a stereo pair and interleave."""
    remaining = len(audio_data) - last_transcribed_sample
    remaining_duration = remaining / SAMPLE_RATE
    print(f"  ⏳ Transcribing final {format_duration(remaining_duration)} (stereo)...")
    post_overlap = int(SAMPLE_RATE * overlap_seconds)
    chunk_start = max(0, last_transcribed_sample - post_overlap)
    mic_chunk = audio_data[:, 0][chunk_start:]
    sys_chunk = audio_data[:, 1][chunk_start:]
    offset = chunk_start / SAMPLE_RATE
    try:
        mic_r = _transcribe_chunk(mic_chunk, model_name, language)
        sys_r = _transcribe_chunk(sys_chunk, model_name, language)
        _shift_segments(mic_r.get("segments", []), offset)
        _shift_segments(sys_r.get("segments", []), offset)
        merged = _interleave_by_timestamp(mic_r, sys_r)
        base_segments.extend(merged.get("segments", []))
    except Exception as e:
        print(f"  ⚠  Final chunk failed: {e}")
    return base_segments


def _shift_segments(segments, offset):
    """Add `offset` (seconds) to every segment's start/end in place."""
    for seg in segments:
        seg["start"] = seg.get("start", 0) + offset
        seg["end"] = seg.get("end", 0) + offset


def _post_process_mono(output_path, audio_data, model_name, language, live_cfg,
                      last_transcribed_sample, rec_only, transcribed_segments,
                      transcribe_lock):
    """Run mono post-processing (single-channel tail or full). Mutates transcribed_segments."""
    flat_audio = audio_data.flatten().astype(np.float32)
    total_samples = len(flat_audio)
    remaining = total_samples - last_transcribed_sample
    overlap_seconds = live_cfg["chunk_overlap_seconds"]
    if rec_only and last_transcribed_sample == 0:
        remaining_duration = total_samples / SAMPLE_RATE
        print(f"  ⏳ Transcribing full recording ({format_duration(remaining_duration)})...")
        try:
            result = _transcribe_audio(str(output_path), model_name, language)
            with transcribe_lock:
                transcribed_segments.extend(result.get("segments", []))
        except Exception as e:
            print(f"  ⚠  Transcription failed: {e}")
        return transcribed_segments
    if remaining <= SAMPLE_RATE * overlap_seconds:
        return transcribed_segments
    remaining_duration = remaining / SAMPLE_RATE
    print(f"  ⏳ Transcribing final {format_duration(remaining_duration)}...")
    post_overlap = int(SAMPLE_RATE * overlap_seconds)
    chunk_start = max(0, last_transcribed_sample - post_overlap)
    chunk = flat_audio[chunk_start:total_samples]
    offset = chunk_start / SAMPLE_RATE
    overlap_boundary = last_transcribed_sample / SAMPLE_RATE
    try:
        result = _transcribe_chunk(chunk, model_name, language)
        segments = _offset_and_filter_segments(
            result.get("segments", []), offset, overlap_boundary)
        with transcribe_lock:
            transcribed_segments.extend(segments)
    except Exception as e:
        print(f"  ⚠  Final chunk failed: {e}")
    return transcribed_segments


def _run_live_diarisation(result, output_path, no_diarise):
    """Diarise a live-mode result in place (stereo or mono path)."""
    if not _check_diarise(no_diarise):
        return
    print("  ⏳ Identifying speakers...")
    try:
        if _is_stereo(str(output_path)):
            _diarise_stereo_live(result, str(output_path))
        else:
            speaker_turns = _diarise_standalone(str(output_path))
            result["segments"] = _assign_speakers_to_segments(result.get("segments", []), speaker_turns)
    except Exception as e:
        print(f"  ⚠  Diarisation failed: {e}")


def _diarise_stereo_live(result, output_path):
    """Diarise a stereo live result by splitting channels and reassigning per side.

    Handles three segment types:
    - _source_channel="system" — from per-channel tail/full transcribe
    - _source_channel="mic"    — from per-channel tail/full transcribe
    - no _source_channel       — from mixed-mono chunks transcribed during recording

    Mixed-mono segments are assigned speakers by overlap against all diarised
    turns combined (sys + mic). This is approximate but prevents data loss.
    """
    mic_tmp, sys_tmp = _split_stereo(output_path)
    try:
        if not ("segments" in result and result["segments"]):
            return
        sys_turns = _diarise_standalone(sys_tmp)
        mic_turns = _diarise_standalone(mic_tmp)
        sys_segs = [s for s in result["segments"] if s.get("_source_channel") == "system"]
        mic_segs = [s for s in result["segments"] if s.get("_source_channel") == "mic"]
        other_segs = [s for s in result["segments"] if not s.get("_source_channel")]
        if sys_segs:
            sys_segs = _assign_speakers_to_segments(sys_segs, sys_turns)
            for seg in sys_segs:
                label = seg["speaker"]
                if label == "Unknown":
                    label = "Remote"
                seg["speaker"] = f"sys:{label}"
        if mic_segs and mic_turns:
            mic_segs = _assign_speakers_to_segments(mic_segs, mic_turns)
            for seg in mic_segs:
                label = seg["speaker"]
                if label == "Unknown":
                    label = config.get("user_name", "You") or "You"
                seg["speaker"] = f"mic:{label}"
        if other_segs:
            all_turns = (sys_turns or []) + (mic_turns or [])
            if all_turns:
                _assign_speakers_to_segments(other_segs, all_turns)
        result["segments"] = sorted(sys_segs + mic_segs + other_segs, key=lambda s: s.get("start", 0))
    finally:
        _cleanup_temp_files(mic_tmp, sys_tmp)



def _save_live_transcript(result, output_path, title, model_name, language, elapsed, cp_path, recording_end, chunks_done):
    """Save the live transcript, print the success panel, open the folder. Returns ok."""
    post_time = time.time() - recording_end
    try:
        transcript_file, speaker_names = _save_transcript(
            result, str(output_path), title, model_name, language,
            elapsed, post_time, cp_path,
        )
    except Exception as e:
        print(f"\n  ❌ Post-processing failed: {e}")
        if cp_path:
            print(f"  ✅ Raw transcription saved to: {cp_path}")
            print("  💡 Your data is safe. Re-run the command to retry formatting.")
        return None

    from ui import success_panel
    print()
    success_panel("✅ Live transcription complete!", [
        ("Recording:", format_duration(elapsed)),
        ("Post-proc:", format_duration(post_time)),
        ("Chunks:", str(chunks_done)),
        ("Speakers:", str(len(speaker_names))),
        ("Segments:", str(len(result.get("segments", [])))),
        ("Saved to:", f"Scripts/{transcript_file.name}"),
    ])
    print()
    subprocess.run(["open", str(transcript_file.parent)], check=False)
    return post_time, transcript_file, speaker_names


def _print_live_intro(source_label, chunk_interval):
    """Print the live-mode info panel."""
    from ui import info_panel
    print()
    info_panel("🎙 Recording", [
        ("Source:", source_label),
        ("Silence:", f"Auto-stops after {SILENCE_TIMEOUT // 60}min"),
        ("Transcribe:", f"Every {chunk_interval}s in background"),
        ("Controls:", "Ctrl+C to stop"),
    ])
    print()


def _build_live_state(chunk_interval):
    """Initialise the live-recording state dict (frames, locks, counters, adaptive)."""
    live_cfg = config["live"]
    return {
        "mic_frames": [], "sys_frames": [],
        "recording": True, "silence_start": None,
        "current_rms": 0.0, "mic_rms": 0.0, "sys_rms": 0.0,
        "start_time": time.time(), "lock": threading.Lock(),
        "last_transcribed_sample": 0, "chunks_done": 0, "chunk_in_progress": False,
        "mic_low_since": None,
        "adaptive": {
            "interval": chunk_interval,
            "min_interval": live_cfg["min_chunk_interval"],
            "max_interval": live_cfg["max_chunk_interval"],
            "struggle_ratio": live_cfg["struggle_ratio"],
            "rec_only_ratio": live_cfg["rec_only_ratio"],
            "rec_only": False, "status_msg": "",
        },
    }


def _open_sys_capture(sys_callback):
    """Start ScreenCaptureKit capture if available. Returns (capture, error_str|None)."""
    if _SystemAudioCapture is None:
        return None, None
    try:
        return _SystemAudioCapture(sample_rate=SAMPLE_RATE, callback=sys_callback), None
    except RuntimeError as e:
        return None, str(e)


def _print_stop_summary(auto_stopped, elapsed, rec_only, chunks_done):
    """Print the post-recording stop summary lines."""
    print()
    if auto_stopped:
        print(f"  ⏹  Auto-stopped after {SILENCE_TIMEOUT // 60}min silence")
    else:
        print(f"  ⏹  Recording stopped ({format_duration(elapsed)})")
    if rec_only:
        print(f"  📊 {chunks_done} chunks done before switching to record-only")
    else:
        print(f"  📊 {chunks_done} chunks pre-transcribed during recording")


def _run_live_session(vp_capture, sck_capture, transcription_thread, mic_pump_thread,
                     state, dual_mode):
    """Start VPIO, SCK, transcription + mic poll threads; run the display loop.

    Returns True if the loop ended normally (auto-stop), False if user hit Ctrl-C.
    """
    auto_stopped = False
    try:
        vp_capture.start()
        if sck_capture:
            sck_capture.start()
        transcription_thread.start()
        mic_pump_thread.start()
        _run_live_display_loop(state, dual_mode)
        auto_stopped = True
    except KeyboardInterrupt:
        state["recording"] = False
    finally:
        try:
            vp_capture.stop()
        except Exception:
            pass
        if sck_capture:
            sck_capture.stop()
    return auto_stopped


def _drain_transcription_thread(stop_event, chunk_in_progress, thread, timeout=120):
    """Signal the transcription worker to stop and wait for any in-flight chunk."""
    stop_event.set()
    if chunk_in_progress:
        print("  ⏳ Waiting for in-progress chunk to finish...")
    thread.join(timeout=timeout)


def _setup_live_recording(state, model_name, language, dual_mode, get_snapshot):
    """Create VPIO capture, system capture, transcription thread + mic poll thread.

    Returns dict: vp_capture, sck_capture, dual_mode, transcription_thread,
    mic_poll_thread, transcribed_segments, transcribe_lock, stop_event,
    live_cfg, adaptive.
    """
    def sys_callback(indata, frames, time_info, status):
        _sys_audio_callback(indata, frames, time_info, status, state=state)

    sck_capture, sck_error = _open_sys_capture(sys_callback=sys_callback)
    if sck_error:
        print(f"  ⚠  System audio unavailable: {sck_error}")
        dual_mode = False

    transcribed_segments = []
    transcribe_lock = threading.Lock()
    stop_event = threading.Event()
    live_cfg = config["live"]
    adaptive = state["adaptive"]

    transcription_thread = threading.Thread(
        target=_transcription_worker,
        args=(state, model_name, language, live_cfg, adaptive, stop_event,
              transcribe_lock, transcribed_segments, get_snapshot),
        daemon=True,
    )

    vp_capture = VPMicCapture(sample_rate=SAMPLE_RATE,
                               voice_processing=config["audio"].get("voice_processing", False))
    mic_poll_thread = threading.Thread(
        target=_mic_pump_thread,
        args=(vp_capture, state, stop_event),
        daemon=True,
    )

    return {
        "vp_capture": vp_capture, "sck_capture": sck_capture, "dual_mode": dual_mode,
        "transcription_thread": transcription_thread, "mic_poll_thread": mic_poll_thread,
        "transcribed_segments": transcribed_segments, "transcribe_lock": transcribe_lock,
        "stop_event": stop_event, "live_cfg": live_cfg, "adaptive": adaptive,
    }


def _post_process_live_audio(audio_data, output_path, model_name, language, live_cfg,
                             state, adaptive, transcribed_segments, transcribe_lock):
    """Run stereo or mono post-processing depending on channel count."""
    if audio_data.ndim == 2:
        return _post_process_stereo(
            audio_data, model_name, language, live_cfg,
            state["last_transcribed_sample"], adaptive["rec_only"], transcribed_segments)
    return _post_process_mono(
        output_path, audio_data, model_name, language, live_cfg,
        state["last_transcribed_sample"], adaptive["rec_only"],
        transcribed_segments, transcribe_lock)


def _finalize_live_audio(state, dual_mode, output_path):
    """Mix + save the recorded audio. Returns (saved_path, audio_data) or (None, None)."""
    audio_data = _mix_audio(state["mic_frames"], state["sys_frames"], dual_mode,
                            normalise=config.get("normalise", True))
    saved_to = _safe_write_audio(output_path, audio_data, SAMPLE_RATE, description="recording")
    if not saved_to:
        print("  ⚠  Recording not saved.")
        return None, None
    print(f"  📁 Saved: {saved_to.name}")
    return saved_to, audio_data


def cmd_live(args):
    """Record and transcribe simultaneously — transcript ready shortly after recording stops."""
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    model_name = args.model or config["live"]["model"]
    if not _validate_model(model_name):
        return

    language = args.language or config["language"]
    chunk_interval = config["live"]["chunk_interval_seconds"]
    no_diarise = args.no_diarise
    title = args.title

    if not _preload_live_model(model_name):
        return

    try:
        _mic_name, dual_mode, source_label, output_path = _setup_audio_devices()
    except RuntimeError:
        return

    _print_live_intro(source_label, chunk_interval)
    state = _build_live_state(chunk_interval)

    def get_snapshot():
        return _get_mixed_snapshot(state["lock"], state["mic_frames"],
                                   state["sys_frames"], dual_mode)

    setup = _setup_live_recording(
        state, model_name, language, dual_mode, get_snapshot,
    )
    dual_mode = setup["dual_mode"]
    transcribed_segments = setup["transcribed_segments"]
    transcribe_lock = setup["transcribe_lock"]
    stop_event = setup["stop_event"]
    live_cfg = setup["live_cfg"]
    adaptive = setup["adaptive"]

    auto_stopped = _run_live_session(
        setup["vp_capture"], setup["sck_capture"], setup["transcription_thread"],
        setup["mic_poll_thread"], state, dual_mode)

    clear_line()
    recording_end = time.time()
    elapsed = recording_end - state["start_time"]

    _drain_transcription_thread(stop_event, state["chunk_in_progress"], setup["transcription_thread"])
    setup["mic_poll_thread"].join(timeout=2)

    if not state["mic_frames"]:
        print("  ⚠  No audio recorded.")
        return

    _print_stop_summary(auto_stopped, elapsed, adaptive["rec_only"], state["chunks_done"])
    output_path, audio_data = _finalize_live_audio(state, dual_mode, output_path)
    if output_path is None:
        return

    remaining_segments = _post_process_live_audio(
        audio_data, output_path, model_name, language, live_cfg,
        state, adaptive, transcribed_segments, transcribe_lock)

    gc.collect()
    all_segments = sorted(remaining_segments, key=lambda s: s.get("start", 0))
    if not all_segments:
        print("  ⚠  No speech detected.")
        return

    result = {"segments": all_segments, "language": language}
    _run_live_diarisation(result, output_path, no_diarise)

    cp_path = _save_checkpoint(result, str(output_path), title)
    save_outcome = _save_live_transcript(
        result, output_path, title, model_name, language,
        elapsed, cp_path, recording_end, state["chunks_done"])
    if save_outcome is None:
        return
    post_time, transcript_file, speaker_names = save_outcome

    _log_run({
        "model": model_name, "mode": "live", "title": title,
        "audio_duration": round(elapsed, 1),
        "processing_time": round(post_time, 1),
        "segments": len(result.get("segments") or []),
        "speakers": len(speaker_names),
        "audio_file": output_path.name,
    })


# ─── Transcription with Progress ─────────────────────────────────────────────

class ProgressTracker:
    """Track and display transcription progress as a step list.

    Shows completed steps with ✓ and duration, current step with spinner.
    Uses Rich Live for in-place rendering (no scrolling).
    """

    def __init__(self):
        from ui import console
        from rich.live import Live
        self.start_time = time.time()
        self.step_start_time = time.time()
        self.current_step = 0
        self.running = True
        self._thread = None
        self._live = Live("", console=console, auto_refresh=False)
        self._completed = []  # [(name, duration_str)]
        self._step_times = {}
        self.step_names = ["Transcribing", "Saving"]  # overridden by callers

    def start(self):
        self._step_times.clear()
        self._live.__enter__()
        self._thread = threading.Thread(target=self._display_loop, daemon=True)
        self._thread.start()

    def stop(self):
        # Record final step
        if self.current_step not in self._step_times:
            self._step_times[self.current_step] = time.time() - self.step_start_time
            name = self.step_names[self.current_step] if self.current_step < len(self.step_names) else "Step"
            self._completed.append((name, format_duration(self._step_times[self.current_step])))
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
        # Final render: all steps completed — this becomes permanent output
        self._render_final()
        self._live.__exit__(None, None, None)

    def set_step(self, step_index):
        if self.current_step < step_index:
            prev_elapsed = time.time() - self.step_start_time
            self._step_times[self.current_step] = prev_elapsed
            prev_name = self.step_names[self.current_step] if self.current_step < len(self.step_names) else "Step"
            self._completed.append((prev_name, format_duration(prev_elapsed)))
        self.current_step = step_index
        self.step_start_time = time.time()

    def _render(self, tick=0):
        """Build the display: completed steps + current step with spinner."""
        from rich.text import Text
        lines = [f"  [green]✓[/green] {name} [dim]({dur})[/dim]" for name, dur in self._completed]

        if self.running:
            total_steps = len(self.step_names)
            step_name = self.step_names[self.current_step] if self.current_step < total_steps else "Processing"
            step_elapsed = time.time() - self.step_start_time
            spin = spinner_char(tick)
            lines.append(f"  {spin} {step_name}... [dim]{format_duration(step_elapsed)}[/dim]")

        self._live.update(Text.from_markup("\n".join(lines)))
        self._live.refresh()

    def _render_final(self):
        """Render all steps as completed — becomes permanent output when Live exits."""
        self._render()

    def _display_loop(self):
        tick = 0
        while self.running:
            self._render(tick)
            tick += 1
            time.sleep(0.3)


# ─── Audio Loading & Processing ───────────────────────────────────────────────
def _is_stereo(audio_path):
    """Check channel count via ffprobe (format-agnostic)."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "stream=channels",
             "-of", "csv=p=0", str(audio_path)],
            capture_output=True, text=True, timeout=10)
        return result.returncode == 0 and result.stdout.strip() == "2"
    except Exception:
        return False


def _decode_stereo(audio_path):
    """Decode any stereo audio to (mic, sys, sample_rate) via ffmpeg."""
    sr_result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "stream=sample_rate",
         "-of", "csv=p=0", str(audio_path)],
        capture_output=True, text=True, timeout=10)
    sr = int(sr_result.stdout.strip()) if sr_result.returncode == 0 else SAMPLE_RATE
    result = subprocess.run([
        "ffmpeg", "-i", str(audio_path),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ar", str(sr), "-ac", "2", "-v", "quiet", "-"
    ], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode {audio_path}: {result.stderr.decode()[:200]}")
    audio = np.frombuffer(result.stdout, dtype=np.float32).reshape(-1, 2)
    return audio[:, 0], audio[:, 1], sr


def _split_stereo(audio_path):
    """Split a stereo file into two mono temp files (mic, system).

    Returns (mic_path, sys_path) as temp file paths. Caller must clean up.
    """
    mic, sys_audio, sr = _decode_stereo(audio_path)
    mic_fd, mic_path = tempfile.mkstemp(suffix="_mic.wav")
    sys_fd, sys_path = tempfile.mkstemp(suffix="_sys.wav")
    os.close(mic_fd)
    os.close(sys_fd)
    try:
        _write_wav(mic_path, mic, sr)
        _write_wav(sys_path, sys_audio, sr)
    except Exception:
        _cleanup_temp_files(mic_path, sys_path)
        raise
    _log("split", duration=f"{len(mic)/sr:.1f}s",
         mic_rms=f"{np.sqrt(np.mean(mic**2)):.5f}",
         sys_rms=f"{np.sqrt(np.mean(sys_audio**2)):.5f}")
    return mic_path, sys_path















def _interleave_by_timestamp(mic_result, sys_result):
    """Combine mic + system transcripts by timestamp.

    With VoiceProcessingIO AEC at the audio layer, the mic channel has no
    system-audio bleed by the time it reaches ASR, so transcript-level
    dedup is no longer needed. We just sort all segments by start time,
    stamp each with its source channel for display, and return.

    Both inputs should already have speaker labels assigned.
    """
    mic_segs = mic_result.get("segments", [])
    sys_segs = sys_result.get("segments", [])
    for seg in mic_segs:
        seg["_source_channel"] = "mic"
    for seg in sys_segs:
        seg["_source_channel"] = "system"
    final = sorted(mic_segs + sys_segs, key=lambda s: s.get("start", 0))
    _log("interleave", mic_segs=len(mic_segs), sys_segs=len(sys_segs), total=len(final))
    return {
        "segments": final,
        "language": mic_result.get("language", sys_result.get("language", "en")),
    }


def _normalise(audio, target_rms=0.05, peak_headroom_db=1.0):
    """Normalise audio to target RMS with peak headroom to prevent clipping.

    Scales to target_rms unless that would push peaks above the ceiling
    (-peak_headroom_db from full scale). In that case, gain is reduced
    so peaks stay below the ceiling — no hard clipping, no distortion.
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-6:
        return audio
    peak = float(np.max(np.abs(audio)))
    # Gain needed to reach target RMS
    rms_gain = target_rms / rms
    # Max gain before peaks hit ceiling (e.g. -1 dBFS = 0.891)
    peak_ceiling = 10 ** (-peak_headroom_db / 20)  # linear
    peak_gain = peak_ceiling / peak if peak > 1e-6 else rms_gain
    # Use the lesser — reach target RMS or stay under ceiling
    gain = min(rms_gain, peak_gain)
    audio = audio * gain
    actual_rms = rms * gain
    actual_peak = peak * gain
    limited = "peak-limited" if gain < rms_gain else "rms-target"
    _log("norm", rms_in=f"{rms:.5f}", rms_out=f"{actual_rms:.5f}",
         gain=f"{gain:.2f}x", peak=f"{actual_peak:.3f}", mode=limited)
    return audio


def _load_audio(path, sr=16000):
    """Load any audio format via ffmpeg, return mono float32 numpy array at sr Hz."""
    cmd = [
        "ffmpeg", "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ar", str(sr), "-ac", "1",
        "-v", "quiet", "-"
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to load {path}: {result.stderr.decode()[:200]}")
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if len(audio) == 0:
        raise RuntimeError(f"ffmpeg produced empty audio for {path}")
    _log("load", file=Path(path).name, sr=sr, duration=f"{len(audio)/sr:.1f}s",
         rms=f"{np.sqrt(np.mean(audio**2)):.5f}", peak=f"{np.max(np.abs(audio)):.3f}")
    return audio


def _stft(audio, n_fft=2048, hop_length=512):
    """Short-time Fourier Transform using numpy. Returns complex array (freq_bins, frames)."""
    if len(audio) < n_fft:
        # Pad short audio to n_fft length
        audio = np.pad(audio, (0, n_fft - len(audio)))
    window = np.hanning(n_fft)
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    frames = np.stack([audio[i * hop_length:i * hop_length + n_fft] * window for i in range(n_frames)])
    return np.fft.rfft(frames, axis=1).T


def _istft(stft_matrix, hop_length=512, length=None):
    """Inverse STFT with overlap-add reconstruction. Returns float32 audio array."""
    n_fft = (stft_matrix.shape[0] - 1) * 2
    n_frames = stft_matrix.shape[1]
    window = np.hanning(n_fft)
    output_length = n_fft + hop_length * (n_frames - 1)
    output = np.zeros(output_length)
    window_sum = np.zeros(output_length)
    for i in range(n_frames):
        frame = np.fft.irfft(stft_matrix[:, i], n=n_fft)
        start = i * hop_length
        output[start:start + n_fft] += frame * window
        window_sum[start:start + n_fft] += window ** 2
    # Normalise by window overlap
    nonzero = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]
    if length is not None:
        output = output[:length]
    return output.astype(np.float32)


# ─── Audio Denoising ─────────────────────────────────────────────────────────

def _denoise_audio(audio_path):
    """Apply spectral subtraction denoising to an audio file.

    Finds the quietest N-second window in the audio as a noise profile,
    then subtracts scaled noise power from the entire signal. Falls back
    to the first N seconds if the audio is shorter than 2x the profile.

    Writes a denoised WAV to a temp file and returns its path.
    The caller is responsible for cleanup.

    Returns (denoised_path, elapsed_seconds) or (original_path, 0) if
    denoising is disabled or fails.
    """
    denoise_cfg = config["denoise"]

    factor = denoise_cfg["factor"]
    profile_secs = denoise_cfg["noise_profile_seconds"]
    sr = SAMPLE_RATE

    try:
        t0 = time.time()
        audio = _load_audio(audio_path, sr=sr)

        # Find the quietest window for noise profile instead of blindly
        # using the first N seconds (which may contain speech)
        profile_samples = sr * profile_secs
        if len(audio) > profile_samples * 2:
            # Slide a window across the audio, find lowest RMS
            step = sr  # check every 1 second
            best_rms = float("inf")
            best_start = 0
            for start in range(0, len(audio) - profile_samples, step):
                window = audio[start:start + profile_samples]
                rms = float(np.sqrt(np.mean(window ** 2)))
                if rms < best_rms:
                    best_rms = rms
                    best_start = start
            noise_clip = audio[best_start:best_start + profile_samples]
            _log("denoise", profile=f"quietest window at {best_start/sr:.1f}s (rms={best_rms:.5f})")
        else:
            noise_clip = audio[:profile_samples]

        n_fft = denoise_cfg["n_fft"]
        hop = denoise_cfg["hop_length"]

        noise_stft = _stft(noise_clip, n_fft=n_fft, hop_length=hop)
        noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1, keepdims=True)

        audio_stft = _stft(audio, n_fft=n_fft, hop_length=hop)
        audio_power = np.abs(audio_stft) ** 2

        clean_power = np.maximum(audio_power - factor * noise_power, 0.0)
        gain = np.sqrt(clean_power / (audio_power + 1e-10))
        clean_stft = audio_stft * gain

        denoised = _istft(clean_stft, hop_length=hop, length=len(audio))
        denoised = denoised.astype(np.float32)

        # Write to temp file next to the original
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix=".denoise_")
        os.close(tmp_fd)
        _write_wav(tmp_path, denoised, sr)

        elapsed = time.time() - t0
        noise_rms = float(np.sqrt(np.mean(noise_clip ** 2)))
        in_rms = float(np.sqrt(np.mean(audio ** 2)))
        out_rms = float(np.sqrt(np.mean(denoised ** 2)))
        _log("denoise", factor=factor, noise_rms=f"{noise_rms:.5f}",
             rms_in=f"{in_rms:.5f}", rms_out=f"{out_rms:.5f}", time=f"{elapsed:.1f}s")
        return tmp_path, elapsed

    except Exception as e:
        _log("denoise", error=str(e))
        print(f"  ⚠  Denoising failed ({e}), using original audio")
        # Clean up partial temp file if it was created
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return audio_path, 0


# ─── MLX Engine Helpers ──────────────────────────────────────────────────────

def _get_mlx_repo(model_name):
    """Get HuggingFace repo ID for an MLX whisper model."""
    return config["mlx_models"].get(model_name, f"mlx-community/whisper-{model_name}-mlx")


def _transcribe_mlx(audio_path, model_name, language="en", word_timestamps=True):
    """Transcribe audio using mlx-whisper (Apple Silicon GPU)."""
    import mlx_whisper
    repo = _get_mlx_repo(model_name)

    # hallucination_silence_threshold skips silent segments where hallucination
    # is detected. Requires word_timestamps=True to work.
    hst = config["whisper"]["hallucination_silence_threshold"]
    if hst is not None:
        word_timestamps = True  # Required for HST

    t0 = time.time()
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=repo,
        word_timestamps=word_timestamps,
        language=language,
        hallucination_silence_threshold=hst,
    )
    segs = result.get("segments", [])
    words = sum(len(s.get("text", "").split()) for s in segs)
    speech_dur = sum(s.get("end", 0) - s.get("start", 0) for s in segs)
    _log("asr", model=model_name, segments=len(segs), words=words,
         speech=f"{speech_dur:.1f}s", time=f"{time.time()-t0:.1f}s")
    return result


def _is_parakeet(model_name):
    """Check if model_name refers to a Parakeet model."""
    return model_name.startswith("parakeet")


def _transcribe_parakeet(audio_path, model_name="parakeet"):
    """Transcribe audio using Parakeet TDT (CTC model, Apple Silicon GPU).

    Returns result in same format as _transcribe_mlx() for pipeline compatibility.
    Parakeet doesn't hallucinate on silence (CTC architecture) and produces
    properly punctuated, capitalized text.
    """
    from mlx_audio.stt import load as load_stt

    repo = config["mlx_models"].get(model_name, "mlx-community/parakeet-tdt-0.6b-v3")
    chunk_dur = config["parakeet"]["chunk_duration"]
    model = load_stt(repo)
    result = model.generate(audio_path, chunk_duration=chunk_dur, verbose=False)

    # Convert Parakeet sentences to Whisper-compatible segments
    segments = [
        {"id": i, "start": s.start, "end": s.end, "text": s.text}
        for i, s in enumerate(result.sentences)
    ]

    words = sum(len(s.get("text", "").split()) for s in segments)
    speech_dur = sum(s.get("end", 0) - s.get("start", 0) for s in segments)
    _log("asr", model=model_name, segments=len(segments), words=words,
         speech=f"{speech_dur:.1f}s")

    del model
    gc.collect()

    return {"segments": segments, "language": "en", "text": result.text}


def _transcribe_chunk(chunk, model_name, language="en"):
    """Transcribe a numpy audio chunk. Writes to temp file, transcribes, cleans up.

    Works with both Parakeet and Whisper models.
    Returns the transcription result dict.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            _write_wav(tmp.name, chunk, SAMPLE_RATE)
            tmp_path = tmp.name
        return _transcribe_audio(tmp_path, model_name, language, word_timestamps=True)
    finally:
        _cleanup_temp_files(tmp_path)


def _diarise_mlx_audio(audio_path):
    """Speaker diarisation using Sortformer's native streaming (AOSC).

    Uses generate_stream() which maintains speaker identity across chunks
    via the model's Arrival-Order Speaker Cache. More accurate than manual
    chunking + Hungarian matching.
    """
    from mlx_audio.vad import load as load_vad
    from mlx_audio.vad.models.sortformer.sortformer import DiarizationSegment

    diar_config = config["diarization"]
    model = load_vad(diar_config["mlx_model"])

    all_segs = []
    for res in model.generate_stream(
        audio_path,
        sample_rate=16000,
        chunk_duration=diar_config["mlx_chunk_duration"],
        threshold=diar_config["threshold"],
        min_duration=diar_config["min_duration"],
        merge_gap=diar_config["merge_gap"],
        spkcache_max=diar_config["spkcache_max"],
        fifo_max=diar_config["fifo_max"],
    ):
        all_segs.extend(res.segments)

    all_segs.sort(key=lambda s: (s.start, s.speaker))
    mg = diar_config["merge_gap"]
    merged = []
    for seg in all_segs:
        if not merged or merged[-1].speaker != seg.speaker:
            merged.append(seg)
        elif seg.start - merged[-1].end <= mg:
            merged[-1] = DiarizationSegment(
                start=merged[-1].start, end=seg.end, speaker=merged[-1].speaker)
        else:
            merged.append(seg)

    spk_dur = {}
    for seg in merged:
        spk_dur[seg.speaker] = spk_dur.get(seg.speaker, 0.0) + (seg.end - seg.start)
    total_dur = sum(spk_dur.values()) or 1.0
    min_dur = min(2.0, total_dur * 0.01)
    active = sorted(s for s, d in spk_dur.items() if d >= min_dur)
    remap = {old: new for new, old in enumerate(active)}
    if len(active) < len(spk_dur):
        dropped = sorted(set(spk_dur) - set(active))
        _log("diar_filter", dropped=[f"Speaker {s} ({spk_dur[s]:.1f}s)" for s in dropped],
             kept=[f"Speaker {s} ({spk_dur[s]:.1f}s)" for s in active])
        merged = [DiarizationSegment(start=s.start, end=s.end, speaker=remap[s.speaker])
                  for s in merged if s.speaker in remap]

    turns = [{"start": s.start, "end": s.end, "speaker": f"Speaker {s.speaker}"} for s in merged]

    labels = sorted({t["speaker"] for t in turns})
    _log("diar", speakers=len(labels), turns=len(turns), labels=labels)

    del model
    _mx_cleanup()
    return turns


def _diarise_standalone(audio_path):
    """Run speaker diarisation using MLX Sortformer."""
    t0 = time.time()
    turns = _diarise_mlx_audio(audio_path)
    _log("diar_standalone", file=Path(audio_path).name, time=f"{time.time()-t0:.1f}s")
    return turns


def _assign_speakers_to_segments(segments, speaker_turns):
    """Match transcription segments to speaker turns by timestamp overlap.

    Sort turns by start so the break is safe (starts are monotonic).
    Modifies segments in-place.
    """
    speaker_turns = sorted(speaker_turns, key=lambda t: t["start"])
    for seg in segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        best_speaker = "Unknown"
        best_overlap = 0.0
        for turn in speaker_turns:
            if turn["start"] >= seg_end:
                break
            if turn["end"] <= seg_start:
                continue
            overlap = min(seg_end, turn["end"]) - max(seg_start, turn["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]
        seg["speaker"] = best_speaker
    return segments


def _resolve_run_audio_path(audio_path_arg):
    """Return the audio path to run, falling back to the latest recording."""
    if audio_path_arg:
        return audio_path_arg
    latest = get_last_recording()
    if not latest:
        print("  ❌ No recordings found. Record first with: transcribe rec")
    return latest


def _validate_audio_file(audio_path):
    """Print an error and return False if the file is missing or empty."""
    if not os.path.exists(audio_path):
        print(f"  ❌ File not found: {audio_path}")
        return False
    if os.path.getsize(audio_path) == 0:
        print(f"  ❌ File is empty: {audio_path}")
        return False
    return True


def _resolve_run_model_name(args):
    """Pick the model name from --model, default, or interactive prompt."""
    model_name = args.model or config["default_model"]
    if not args.model and sys.stdin.isatty():
        model_name = prompt_model_selection()
    return model_name


def _resolve_run_title(audio_path, title_arg):
    """Pick the transcript title from --title, calendar, or filename."""
    if title_arg:
        return title_arg
    if config["auto_title_from_calendar"]:
        title = get_current_event_title()
        if title:
            return title
    return Path(audio_path).stem.replace("_", " ").replace("-", " ").title()


def _resolve_denoise_mode(force_denoise, no_denoise):
    """Resolve the effective denoise flag from CLI + config."""
    if force_denoise:
        return True
    if no_denoise:
        return False
    return config["denoise"]["enabled"]


def _estimate_run_time(audio_duration, model_name, diarise_enabled):
    """Estimate wall-clock time, including diarisation overhead (~3s/min)."""
    speed_factor = MODEL_INFO.get(model_name, {}).get("speed_factor", 1.0)
    speed_factor *= 0.15  # MLX GPU acceleration
    estimate = audio_duration * speed_factor
    if diarise_enabled:
        estimate += audio_duration * 0.05
    return estimate


def _build_run_steps_preview(is_stereo, diarise, denoise):
    """Return a short human-readable pipeline summary for the info panel."""
    denoise_prefix = "Denoise → " if denoise else ""
    if is_stereo and diarise:
        return f"{denoise_prefix}Diarise mic + system → Transcribe mic + system → Save"
    if is_stereo:
        return f"{denoise_prefix}Transcribe mic + system → Save"
    if diarise:
        return f"{denoise_prefix}Diarise → Transcribe → Save"
    return f"{denoise_prefix}Transcribe → Save"



def _build_run_step_names(diarise_enabled, denoise_enabled):
    """Return ordered pipeline step labels for the progress display."""
    names = ["Identifying speakers", "Transcribing", "Saving"] if diarise_enabled else ["Transcribing", "Saving"]
    return (["Denoising audio"] + names) if denoise_enabled else names


def _run_denoise_step(audio_path, is_stereo, denoise_enabled, progress, step):
    """Run the (optional) denoise step. Returns (denoised_path_or_none, next_step)."""
    if not (denoise_enabled and not is_stereo):
        return None, step
    progress.set_step(step)
    denoised_path, _ = _denoise_audio(audio_path)
    return denoised_path, step + 1


def _run_normalise_step(transcribe_path, is_stereo, normalise_enabled):
    """Normalise a mono file to a temp wav. Returns (normalised_path_or_none)."""
    if not (normalise_enabled and not is_stereo):
        return None
    audio_data = _load_audio(transcribe_path, sr=SAMPLE_RATE)
    audio_data = _normalise(audio_data)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix=".norm_")
    os.close(tmp_fd)
    _write_wav(tmp_path, audio_data, SAMPLE_RATE)
    return tmp_path


def _diarise_stereo_channels(mic_path, sys_path, diarise_enabled, progress, step):
    """Diarise both stereo channels (system + mic).

    With VPIO at the recording layer, the mic channel is now clean enough to
    always diarise — there's no reason to skip it. The pre-VPIO `diarise_mic`
    config flag was a workaround for when mic contained system bleed that
    confused Sortformer.
    """
    if not diarise_enabled:
        return None, None, step
    progress.set_step(step)
    sys_turns = _diarise_standalone(sys_path)
    _mx_cleanup()
    mic_turns = _diarise_standalone(mic_path)
    _mx_cleanup()
    return sys_turns, mic_turns, step + 1


def _mx_cleanup():
    """Release MLX memory after model operations."""
    gc.collect()
    import mlx.core as mx
    mx.clear_cache()


def _label_mic_segments(mic_result, mic_turns):
    """Apply mic-side speaker labels (diarise if available, else default 'You')."""
    segments = mic_result.get("segments", [])
    if mic_turns:
        mic_result["segments"] = _assign_speakers_to_segments(segments, mic_turns)
        for seg in mic_result["segments"]:
            label = seg["speaker"]
            if label == "Unknown":
                label = config.get("user_name", "You") or "You"
            seg["speaker"] = f"mic:{label}"
    else:
        mic_label = config.get("user_name", "You") or "You"
        for seg in segments:
            seg["speaker"] = mic_label


def _label_sys_segments(sys_result, sys_turns):
    """Apply sys-side speaker labels or default to 'Remote' when no diarisation."""
    segments = sys_result.get("segments", [])
    if sys_turns:
        sys_result["segments"] = _assign_speakers_to_segments(segments, sys_turns)
        for seg in sys_result["segments"]:
            label = seg["speaker"]
            if label == "Unknown":
                label = "Remote"
            seg["speaker"] = f"sys:{label}"
    else:
        for seg in segments:
            seg["speaker"] = "Remote"


def _normalise_stereo_channels(mic_path, sys_path):
    """Normalise each split stereo channel in place."""
    for ch_path in (mic_path, sys_path):
        ch_audio = _load_audio(ch_path, sr=SAMPLE_RATE)
        ch_audio = _normalise(ch_audio)
        _write_wav(ch_path, ch_audio, SAMPLE_RATE)



def _run_stereo_pipeline(transcribe_path, model_name, language, normalise_enabled,
                         diarise_enabled, progress, step):
    """Run the per-channel stereo ASR pipeline + speaker labels + interleave.

    No echo dedup — VPIO at the recording layer has already removed system
    bleed from the mic channel, so transcript-level dedup would be a no-op.
    """
    mic_path, sys_path = _split_stereo(transcribe_path)
    if normalise_enabled:
        _normalise_stereo_channels(mic_path, sys_path)
    try:
        sys_turns, mic_turns, step = _diarise_stereo_channels(
            mic_path, sys_path, diarise_enabled, progress, step)
        progress.set_step(step)
        mic_result = _transcribe_audio(mic_path, model_name, language)
        _mx_cleanup()
        sys_result = _transcribe_audio(sys_path, model_name, language)
        _label_sys_segments(sys_result, sys_turns)
        _label_mic_segments(mic_result, mic_turns)
        return _interleave_by_timestamp(mic_result, sys_result), step + 1
    finally:
        _cleanup_temp_files(mic_path, sys_path)


def _run_mono_pipeline(transcribe_path, model_name, language, diarise_enabled, progress, step):
    """Run the mono ASR pipeline (single channel). Returns (result, next_step)."""
    speaker_turns = None
    if diarise_enabled:
        progress.set_step(step)
        speaker_turns = _diarise_standalone(transcribe_path)
        _mx_cleanup()
        step += 1
    progress.set_step(step)
    result = _transcribe_audio(transcribe_path, model_name, language,
                               word_timestamps=not diarise_enabled)
    if speaker_turns is not None:
        result["segments"] = _assign_speakers_to_segments(
            result.get("segments", []), speaker_turns)
    return result, step + 1


def _print_run_success_panel(audio_duration, model_name, speaker_names, result, total_time,
                            output_file, step_names, step_times):
    """Print the success panel after a successful run."""
    from ui import success_panel
    speed_ratio = audio_duration / total_time if total_time > 0 else 0
    rows = [
        ("Audio:", f"{format_duration(audio_duration)} ({speed_ratio:.1f}x real-time)"),
        ("Model:", model_name),
        ("Speakers:", str(len(speaker_names))),
        ("Segments:", str(len(result.get("segments", [])))),
        ("Time:", format_duration(total_time)),
    ]
    if step_times:
        parts = [f"{step_names[i]}: {format_duration(step_times[i])}" for i in range(len(step_names)) if i in step_times]
        if parts:
            rows.append(("Steps:", parts[0]))
            rows.extend(("", part) for part in parts[1:])
    rows.append(("Saved to:", f"Scripts/{output_file.name}"))
    print()
    success_panel("✅ Transcription complete!", rows)
    print()


def _maybe_remove_source_recording(audio_path):
    """Delete the source recording if keep_recording is False."""
    if not config["audio"]["keep_recording"]:
        try:
            Path(audio_path).unlink()
        except OSError:
            pass


def _log_run_entry(opts, result, speaker_names, total_time, step_names, step_times):
    """Append this run to the history log."""
    step_log = {step_names[i]: step_times[i] for i in range(len(step_names)) if i in step_times}
    _log_run({
        "model": opts["model_name"],
        "mode": "run",
        "title": opts["title"],
        "audio_duration": round(opts["audio_duration"], 1),
        "processing_time": round(total_time, 1),
        "segments": len(result.get("segments") or []),
        "speakers": len(speaker_names),
        "audio_file": Path(opts["audio_path"]).name,
        "denoise": opts["denoise_enabled"],
        "normalise": opts["normalise_enabled"],
        "diarise": opts["diarise_enabled"],
        "stereo": opts["is_stereo_input"],
    }, step_log)



def _resolve_run_options(args, audio_path):
    """Bundle all resolved options for a run into a single context dict."""
    model_name = _resolve_run_model_name(args)
    if not _validate_model(model_name):
        return None

    diarise_enabled = _check_diarise(args.no_diarise)
    denoise_enabled = _resolve_denoise_mode(args.denoise, args.no_denoise)
    audio_duration = get_audio_duration(audio_path)
    if audio_duration <= 0:
        print("  ⚠  Couldn't determine audio duration — progress estimates may be off.")
        audio_duration = 300

    is_parakeet_model = _is_parakeet(model_name)
    is_stereo_input = _is_stereo(audio_path)
    engine_label = "mlx-audio (GPU)" if is_parakeet_model else "mlx-whisper (GPU)"
    model_size = MODEL_INFO.get(model_name, {}).get("size", "?")
    diar_label = "Sortformer (GPU)" if diarise_enabled else "off"
    denoise_factor = config["denoise"]["factor"]
    denoise_label = f"spectral sub {denoise_factor}x" if denoise_enabled else "off"
    channels_label = "stereo (mic + system)" if is_stereo_input else "mono"
    normalise_enabled = config.get("normalise", True)

    ram_msg = check_ram_for_model(model_name)
    if ram_msg:
        print(f"  {ram_msg}")

    return {
        "audio_path": audio_path,
        "model_name": model_name,
        "language": args.language or config["language"],
        "title": _resolve_run_title(audio_path, args.title),
        "normalise_enabled": normalise_enabled,
        "denoise_enabled": denoise_enabled,
        "diarise_enabled": diarise_enabled,
        "audio_duration": audio_duration,
        "is_stereo_input": is_stereo_input,
        "is_parakeet_model": is_parakeet_model,
        "engine_label": engine_label,
        "model_size": model_size,
        "diar_label": diar_label,
        "denoise_label": denoise_label,
        "channels_label": channels_label,
        "estimated_time": _estimate_run_time(audio_duration, model_name, diarise_enabled),
        "steps_preview": _build_run_steps_preview(is_stereo_input, diarise_enabled, denoise_enabled),
    }


def _print_run_info_rows(opts):
    """Print the run info panel from a resolved options dict."""
    from ui import info_panel
    rows = [
        ("File:", Path(opts["audio_path"]).name),
        ("Duration:", format_duration(opts["audio_duration"])),
        ("Channels:", opts["channels_label"]),
        ("Engine:", opts["engine_label"]),
        ("Model:", f"{opts['model_name']} ({opts['model_size']})"),
        ("Language:", opts["language"]),
        ("Denoise:", opts["denoise_label"]),
        ("Normalise:", "0.05 RMS" if opts["normalise_enabled"] else "off"),
        ("Diarise:", opts["diar_label"]),
        ("Pipeline:", opts["steps_preview"]),
        ("Est time:", f"~{format_duration(opts['estimated_time'])}"),
        ("Started:", datetime.now().strftime("%H:%M:%S")),
    ]
    print()
    info_panel("📝 Transcription", rows)
    print()


def _execute_pipeline(opts, progress, step):
    """Run denoise → normalise → transcribe pipeline. Returns (result, step, paths).

    paths = (denoised_path, normalised_path) for cleanup.
    Raises on transcription failure.
    """
    denoised_path, step = _run_denoise_step(
        opts["audio_path"], opts["is_stereo_input"],
        opts["denoise_enabled"], progress, step)
    transcribe_path = denoised_path or opts["audio_path"]
    normalised_path = _run_normalise_step(
        transcribe_path, opts["is_stereo_input"], opts["normalise_enabled"])
    if normalised_path:
        transcribe_path = normalised_path

    if opts["is_stereo_input"]:
        result, step = _run_stereo_pipeline(
            transcribe_path, opts["model_name"], opts["language"],
            opts["normalise_enabled"], opts["diarise_enabled"],
            progress, step)
    else:
        result, step = _run_mono_pipeline(
            transcribe_path, opts["model_name"], opts["language"],
            opts["diarise_enabled"], progress, step)

    return result, step, denoised_path, normalised_path


def _handle_pipeline_failure(e, progress, result, opts):
    """Stop progress, log failure, and try to save a partial checkpoint."""
    progress.stop()
    print(f"\n\n  ❌ Transcription failed: {e}")
    if result is None:
        return
    cp_path = _save_checkpoint(result, opts["audio_path"], opts["title"])
    if cp_path:
        print(f"  ✅ Partial transcription saved to: {cp_path}")
        print("  💡 Re-run with --no-diarise to use the transcription without speakers.")


def _finalize_run(result, opts, progress, step_names):
    """Save, render success panel, open in Finder, log entry, handle errors."""
    cp_path = _save_checkpoint(result, opts["audio_path"], opts["title"])
    try:
        total_time = time.time() - progress.start_time
        step_times = getattr(progress, "_step_times", {})
        output_file, speaker_names = _save_transcript(
            result, opts["audio_path"], opts["title"], opts["model_name"], opts["language"],
            opts["audio_duration"], total_time, cp_path,
        )
        _print_run_success_panel(
            opts["audio_duration"], opts["model_name"], speaker_names, result, total_time,
            output_file, step_names, step_times,
        )
        subprocess.run(["open", str(output_file.parent)], check=False)
        _maybe_remove_source_recording(opts["audio_path"])
        _log_run_entry(opts, result, speaker_names, total_time, step_names, step_times)
    except Exception as e:
        print(f"\n  ❌ Post-processing failed: {e}")
        if cp_path:
            print(f"  ✅ Raw transcription saved to: {cp_path}")
            print("  💡 Your data is safe. Re-run the command to retry formatting.")


def _cleanup_temp_artifacts(audio_path, denoised_path, normalised_path):
    """Remove intermediate denoised/normalised files we won't need again."""
    if denoised_path and denoised_path != audio_path:
        _cleanup_temp_files(denoised_path, normalised_path)
    elif normalised_path:
        _cleanup_temp_files(normalised_path)


def cmd_run(args):
    """Transcribe an audio file with progress tracking."""
    global _DEBUG
    if args.debug:
        _DEBUG = True

    audio_path = _resolve_run_audio_path(args.audio)
    if not audio_path or not _validate_audio_file(audio_path):
        return

    opts = _resolve_run_options(args, audio_path)
    if opts is None:
        return

    _print_run_info_rows(opts)

    step_names = _build_run_step_names(opts["diarise_enabled"], opts["denoise_enabled"])
    progress = ProgressTracker()
    progress.step_names = step_names
    progress.start()

    result = None
    denoised_path = normalised_path = None
    try:
        result, _step, denoised_path, normalised_path = _execute_pipeline(opts, progress, 0)
    except Exception as e:
        _handle_pipeline_failure(e, progress, result, opts)
        return
    finally:
        _cleanup_temp_artifacts(audio_path, denoised_path, normalised_path)

    progress.stop()
    _finalize_run(result, opts, progress, step_names)


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

    from ui import console
    from rich.table import Table
    from rich.panel import Panel

    max_list = config["list"]["max_recordings"]
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("st", width=2)
    table.add_column("num", style="dim", width=3)
    table.add_column("name")
    table.add_column("info", style="dim")
    # Pre-load transcript date prefixes to avoid re-globbing per recording
    transcript_dates = {md.stem[:10] for md in SCRIPTS_DIR.glob("*.md")} if SCRIPTS_DIR.exists() else set()
    for i, rec in enumerate(recordings[:max_list], 1):
        duration = get_audio_duration(str(rec))
        size_mb = rec.stat().st_size / (1024 * 1024)
        has_transcript = rec.stem[:10] in transcript_dates
        status = "✅" if has_transcript else ""
        table.add_row(status, f"{i})", rec.stem, f"{format_duration(duration)}, {size_mb:.1f}MB")
    print()
    console.print(Panel(table, title="[bold]📂 Recordings[/bold]", border_style="dim", expand=False, width=60))
    console.print(f"  [dim]📂 {RECORDINGS_DIR}[/dim]")
    console.print("  [dim]✅ = transcript exists[/dim]")
    print()

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
    files = [f for f in directory.iterdir() if f.suffix.lower() in AUDIO_FORMATS] if directory.exists() else []
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def get_last_recording():
    """Get the most recent recording file."""
    recordings = list_recordings(RECORDINGS_DIR)
    return str(recordings[0]) if recordings else None


def get_audio_duration(audio_path):
    """Get audio duration in seconds via ffprobe."""
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


def _format_yaml_frontmatter(title, date_str, speaker_names, meta):
    """Build the YAML frontmatter block (including closing --- and a blank line)."""
    safe_title = title.replace(chr(34), chr(39))  # escape double quotes
    speaker_list = ", ".join(speaker_names.values())
    lines = [
        "---",
        f"title: \"{safe_title}\"",
        f"date: {date_str}",
        "type: transcript",
        f"speakers: [{speaker_list}]",
    ]
    if meta.get("model"):
        lines.append(f"model: {meta['model']}")
    if meta.get("language"):
        lines.append(f"language: {meta['language']}")
    if meta.get("audio_duration"):
        lines.append(f"audio_duration: {meta['audio_duration']:.1f}")
    lines.append("---")
    lines.append("")
    return lines


def _format_detail_line(meta, segments, speaker_names):
    """Build the 'Duration | Model | ...' details line, or None if empty."""
    parts = []
    if meta.get("audio_duration"):
        parts.append(f"**Duration:** {format_duration(meta['audio_duration'])}")
    if meta.get("model"):
        parts.append(f"**Model:** {meta['model']} ({meta.get('engine', 'unknown')})")
    if meta.get("processing_time"):
        parts.append(f"**Processing time:** {format_duration(meta['processing_time'])}")
    parts.append(f"**Segments:** {len(segments)}")
    parts.append(f"**Speakers:** {len(speaker_names)}")
    if meta.get("audio_file"):
        parts.append(f"**Source:** {meta['audio_file']}")
    if not parts:
        return None
    return " | ".join(parts)


def _format_segment_turn(timestamp, speaker, text, single_speaker):
    """Format one speaker turn (one or more merged segments)."""
    if single_speaker:
        return f"**[{timestamp}]** {text}"
    return f"**[{timestamp}] {speaker}:** {text}"


def _group_segments_into_turns(segments, speaker_names):
    """Group consecutive segments by speaker, yielding (timestamp, speaker, text) tuples."""
    single_speaker = len(speaker_names) <= 1
    current_speaker = None
    current_text = []
    current_start = 0.0

    for seg in segments:
        speaker_id = seg.get("speaker", "Unknown")
        speaker_label = speaker_names.get(speaker_id, speaker_id)
        text = seg.get("text", "").strip()

        if not single_speaker and speaker_label == current_speaker:
            current_text.append(text)
        else:
            if current_speaker is not None:
                yield (format_duration(current_start), current_speaker,
                       " ".join(current_text), single_speaker)
            current_speaker = speaker_label
            current_text = [text]
            current_start = seg.get("start", 0.0)

    if current_speaker is not None:
        yield (format_duration(current_start), current_speaker,
               " ".join(current_text), single_speaker)


def format_transcript(result, title, speaker_names, date_str, metadata=None):
    """Format transcription result as Markdown.

    metadata: optional dict with keys like model, engine, audio_duration,
              processing_time, language, diarisation, audio_file.
    """
    meta = metadata or {}
    segments = result.get("segments") or []
    lines = _format_yaml_frontmatter(title, date_str, speaker_names, meta)
    lines.append(f"# {title}")
    lines.append(f"**Date:** {date_str}")
    lines.append("")

    detail = _format_detail_line(meta, segments, speaker_names)
    if detail:
        lines.append(detail)
        lines.append("")

    lines.append("---")
    lines.append("")

    for timestamp, speaker, text, single_speaker in _group_segments_into_turns(
            segments, speaker_names):
        lines.append(_format_segment_turn(timestamp, speaker, text, single_speaker))
        lines.append("")

    return "\n".join(lines)


def _rename_speakers(result):
    """Rename raw diarisation IDs (SPEAKER_00 or "Speaker 0") to "Speaker 1..N".

    Preserves human-readable labels set by the stereo pipeline (You, Remote,
    Local N, Unknown, or a configured user_name).

    Handles channel-prefixed labels ("mic:Speaker 0", "sys:Speaker 0") from the
    stereo diarisation pipeline — these must be treated as distinct speakers
    because Sortformer runs independently per channel.
    """
    raw_pat = re.compile(r"^(SPEAKER_\d+|Speaker \d+)$")
    chan_pat = re.compile(r"^(mic|sys):(Speaker \d+|SPEAKER_\d+)$")
    segments = result.get("segments") or []
    speaker_ids = sorted(set(seg.get("speaker", "Unknown") for seg in segments))
    name_map = {}
    counter = 1
    for sid in speaker_ids:
        if chan_pat.match(sid) or raw_pat.match(sid):
            name_map[sid] = f"Speaker {counter}"
            counter += 1
        else:
            name_map[sid] = sid
    if len(name_map) > 1:
        print(f"  🔊 {len(name_map)} speakers detected")
    if len(name_map) >= 4:
        print("  ⚠  Sortformer supports max 4 speakers per channel — some speakers may be merged")
    _log("speakers", count=len(name_map), labels=list(name_map.values()))
    return name_map


# ─── Calendar Watch ──────────────────────────────────────────────────────────

def get_today_events(calendars=None):
    """Read today's calendar events using macOS Calendar via osascript.

    Returns list of dicts: [{"title": str, "start": datetime, "end": datetime}, ...]
    """
    # Build calendar filter — escape quotes and backslashes to block AppleScript injection.
    if calendars:
        def _esc(s):
            return s.replace("\\", "\\\\").replace('"', '\\"')
        cal_filter = " or ".join(f'name of its calendar is "{_esc(c)}"' for c in calendars)
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


def get_current_event_title():
    """Return the title of the calendar event happening right now, or None."""
    try:
        events = get_today_events(config["watch"]["calendars"])
        now = datetime.now()
        for event in events:
            if event["start"] <= now <= event["end"]:
                return event["title"]
    except Exception:
        pass
    return None


WATCH_LOG = SCRIPT_DIR / "watch.log"
LAUNCHD_LABEL = "com.transcriber.watch"
LAUNCHD_PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def _watch_log(msg, also_print=True):
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


def _print_watch_intro(watch_cfg):
    """Print the watch-mode intro panel from config."""
    from ui import info_panel
    model_name = watch_cfg["model"]
    silence_timeout = watch_cfg["silence_timeout_minutes"]
    min_recording = watch_cfg["min_recording_minutes"]
    end_buffer = watch_cfg["end_buffer_minutes"]
    refresh_hours = watch_cfg["refresh_hours"]
    record_only = watch_cfg["record_only"]
    mode_label = "record only → transcribe after" if record_only else "live (record + transcribe)"

    _watch_log("📅 Calendar Watch started")
    print()
    info_panel("📅 Calendar Watch", [
        ("Mode:", mode_label),
        ("Model:", model_name),
        ("Silence:", f"{silence_timeout}min → auto-stop"),
        ("Min rec:", f"{min_recording}min (discard if shorter)"),
        ("Buffer:", f"{end_buffer}min after meeting end"),
        ("Refresh:", f"every {refresh_hours}h"),
        ("Log:", "watch.log"),
    ], subtitle="Auto-records meetings from your calendar")


def _log_today_events(upcoming):
    """Log the day's remaining meetings."""
    _watch_log(f"📅 {len(upcoming)} meetings remaining today")
    for e in upcoming:
        start_str = e["start"].strftime("%H:%M")
        end_str = e["end"].strftime("%H:%M")
        duration = int((e["end"] - e["start"]).total_seconds() / 60)
        _watch_log(f"   {start_str}-{end_str}  {e['title']}  ({duration}min)")


def _sleep_until_idle_or_refresh(refresh_hours):
    """Sleep until midnight or the next calendar refresh, whichever comes first.

    Returns False if interrupted by the user.
    """
    now = datetime.now()
    tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    next_refresh = now + timedelta(hours=refresh_hours)
    wake_at = min(tomorrow, next_refresh)
    wait = (wake_at - datetime.now()).total_seconds()

    if wake_at == tomorrow:
        _watch_log("💤 No meetings — sleeping until midnight")
    else:
        _watch_log(f"💤 No meetings — refreshing in {refresh_hours}h")
    try:
        time.sleep(max(1, wait))
    except KeyboardInterrupt:
        _watch_log("👋 Watch stopped by user")
        return False
    return True


def _sleep_until_event_or_refresh(next_event, refresh_hours):
    """Sleep until the event starts, or until refresh time if that's sooner.

    Returns True if we woke up only to refresh (and should re-read the calendar).
    """
    wait_seconds = (next_event["start"] - datetime.now()).total_seconds()
    if wait_seconds <= 0:
        return False
    refresh_seconds = refresh_hours * 3600
    sleep_time = min(wait_seconds, refresh_seconds)

    if sleep_time == wait_seconds:
        _watch_log(f"💤 Next: \"{next_event['title']}\" at {next_event['start'].strftime('%H:%M')} ({format_duration(wait_seconds)})")
    else:
        _watch_log(f"💤 Refreshing calendar in {refresh_hours}h (before \"{next_event['title']}\")")

    try:
        time.sleep(max(1, sleep_time))
    except KeyboardInterrupt:
        _watch_log("👋 Watch stopped by user")
        return None  # signal: user-stop
    return sleep_time < wait_seconds


def _start_recording_subprocess(title, model_name, record_only):
    """Spawn the recorder (rec or live) as a detached subprocess."""
    if record_only:
        return subprocess.Popen(
            [sys.executable, __file__, "rec"],
            stdin=subprocess.DEVNULL,
        )
    return subprocess.Popen(
        [sys.executable, __file__,
         "live",
         "--model", model_name,
         "--title", title,
         "--no-diarise"],
        stdin=subprocess.DEVNULL,
    )


def _stop_subprocess(proc, timeout=30):
    """SIGINT a subprocess, escalating to SIGKILL on timeout."""
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _wait_for_meeting_end(proc, stop_time):
    """Poll the recording subprocess until stop_time or it exits on its own.

    Stops the process if it's still running when the meeting window closes.
    """
    while datetime.now() < stop_time:
        if proc.poll() is not None:
            _watch_log("⏹  Recording ended on its own (silence timeout)")
            return
        time.sleep(10)
    if proc.poll() is None:
        _watch_log("⏹  Meeting end + buffer reached — stopping")
        _stop_subprocess(proc, timeout=30)


def _post_process_recording(proc, event, model_name, record_only, min_recording):
    """Discard too-short recordings, or transcribe in record-only mode."""
    if proc.returncode is None or proc.returncode == -signal.SIGKILL:
        return  # force-killed or still running — nothing usable

    recordings = list_recordings(RECORDINGS_DIR)
    if not recordings:
        return
    latest = recordings[0]
    duration = get_audio_duration(str(latest))

    if duration / 60 < min_recording:
        _watch_log(f"🗑  \"{event['title']}\" too short ({format_duration(duration)}) — discarded")
        try:
            latest.unlink()
        except OSError:
            pass
        return

    _watch_log(f"✅ \"{event['title']}\" recorded ({format_duration(duration)})")
    if not record_only:
        return

    _watch_log(f"⏳ Transcribing \"{event['title']}\"...")
    try:
        subprocess.run(
            [sys.executable, __file__,
             "run", str(latest),
             "--model", model_name,
             "--title", event["title"],
             "--no-diarise"],
            stdin=subprocess.DEVNULL,
            timeout=3600,
        )
        _watch_log(f"✅ \"{event['title']}\" transcribed")
    except subprocess.TimeoutExpired:
        _watch_log("⚠  Transcription timed out (1h limit)")
    except Exception as e:
        _watch_log(f"⚠  Transcription failed: {e}")


def _record_one_event(event, model_name, record_only, end_buffer, min_recording):
    """Run a recording subprocess for one event, wait, then post-process.

    Returns False if the user hit Ctrl-C (watch should exit).
    """
    _watch_log(f"⏺  \"{event['title']}\" — auto-recording started")
    stop_time = event["end"] + timedelta(minutes=end_buffer)
    proc = _start_recording_subprocess(event["title"], model_name, record_only)

    try:
        _wait_for_meeting_end(proc, stop_time)
    except KeyboardInterrupt:
        if proc.poll() is None:
            _stop_subprocess(proc, timeout=10)
        _watch_log("👋 Watch stopped by user")
        return False

    _post_process_recording(proc, event, model_name, record_only, min_recording)
    time.sleep(5)
    return True


def cmd_watch(args):
    """Watch calendar and auto-record meetings."""
    watch_cfg = config["watch"]
    model_name = watch_cfg["model"]
    end_buffer = watch_cfg["end_buffer_minutes"]
    calendars = watch_cfg["calendars"] or []
    refresh_hours = watch_cfg["refresh_hours"]
    record_only = watch_cfg["record_only"]
    min_recording = watch_cfg["min_recording_minutes"]

    _print_watch_intro(watch_cfg)

    while True:
        events = get_today_events(calendars if calendars else None)
        upcoming = [e for e in events if e["end"] > datetime.now()]
        _log_today_events(upcoming)

        if not upcoming:
            if not _sleep_until_idle_or_refresh(refresh_hours):
                return
            continue

        refreshed = _sleep_until_event_or_refresh(upcoming[0], refresh_hours)
        if refreshed is None:
            return
        if refreshed:
            _watch_log("🔄 Refreshing calendar...")
            continue

        if not _record_one_event(upcoming[0], model_name, record_only,
                                 end_buffer, min_recording):
            return


def cmd_install_daemon(args):
    """Install launchd agent so watch runs automatically on login."""
    script_path = Path(__file__).resolve()
    venv_python = SCRIPT_DIR / ".venv" / "bin" / "python3"

    if not venv_python.exists():
        print("  ❌ Virtual environment not found. Run install.sh first.")
        return

    if not HF_TOKEN:
        print("  ⚠  HF_TOKEN not set — diarisation won't work in background daemon.")
        print("     Set it with: export HF_TOKEN=your_token_here")
        print("     Then re-run: transcribe install-daemon")
        print()

    import plistlib
    env_vars = {"PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"}
    # Prefer keychain over embedding the token in a world-readable plist.
    if not config.get("keychain") and HF_TOKEN:
        env_vars["HF_TOKEN"] = HF_TOKEN
    plist_data = {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": [str(venv_python), str(script_path), "watch"],
        "WorkingDirectory": str(SCRIPT_DIR),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "StandardOutPath": str(SCRIPT_DIR / "watch-stdout.log"),
        "StandardErrorPath": str(SCRIPT_DIR / "watch-stderr.log"),
        "EnvironmentVariables": env_vars,
    }

    # Create LaunchAgents directory if needed
    LAUNCHD_PLIST.parent.mkdir(parents=True, exist_ok=True)

    # Stop existing daemon if running
    subprocess.run(["launchctl", "unload", str(LAUNCHD_PLIST)],
                   capture_output=True)

    # Write plist (0600 — plist may contain HF_TOKEN if keychain is not enabled)
    with open(LAUNCHD_PLIST, "wb") as f:
        plistlib.dump(plist_data, f)
    os.chmod(LAUNCHD_PLIST, 0o600)

    # Load daemon
    result = subprocess.run(["launchctl", "load", str(LAUNCHD_PLIST)],
                           capture_output=True, text=True)

    if result.returncode == 0:
        from ui import success_panel
        print()
        success_panel("✅ Calendar Watch daemon installed!", [
            ("Startup:", "Starts automatically on login"),
            ("Recovery:", "Restarts if it crashes"),
            ("Mode:", "Runs silently in background"),
            ("Log:", "watch.log"),
            ("Plist:", "~/Library/LaunchAgents/"),
            ("Status:", "transcribe watch-status"),
            ("Remove:", "transcribe uninstall-daemon"),
        ])
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
            with open(WATCH_LOG) as f:
                tail = deque(f, maxlen=5)
            print()
            print("  ── Recent activity ──")
            for line in tail:
                print(f"  {line.rstrip()}")
        except OSError:
            pass
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Transcriber — Record, transcribe, and save as Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    transcribe rec                              Record audio
    transcribe run                              Transcribe last recording
    transcribe run ~/Downloads/recording.mp3    Transcribe a specific file
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
    run_parser.add_argument("--model", "-m", default=None, help="Model: parakeet/small.en/medium/large-v3")
    run_parser.add_argument("--title", "-t", default=None, help="Transcript title")
    run_parser.add_argument("--language", "-l", default=None, help="Language code (default: from config)")
    run_parser.add_argument("--no-diarise", action="store_true", help="Skip speaker identification (faster)")
    run_parser.add_argument("--denoise", action="store_true", help="Enable denoising (off by default)")
    run_parser.add_argument("--no-denoise", action="store_true", help="Force denoising OFF (overrides config)")
    run_parser.add_argument("--debug", action="store_true", help="Enable pipeline debug logging")

    # live
    live_parser = subparsers.add_parser("live", help="Record + transcribe simultaneously")
    live_parser.add_argument("--model", "-m", default=None, help="Model (default: from config)")
    live_parser.add_argument("--title", "-t", default=None, help="Transcript title")
    live_parser.add_argument("--language", "-l", default=None, help="Language code (default: from config)")
    live_parser.add_argument("--no-diarise", action="store_true", help="Skip speaker identification (faster)")

    # watch
    watch_parser = subparsers.add_parser("watch", help="Auto-record meetings from calendar")
    watch_parser.add_argument("--refresh", action="store_true", help="Force calendar refresh")

    # daemon management
    subparsers.add_parser("install-daemon", help="Install watch as background service")
    subparsers.add_parser("uninstall-daemon", help="Remove background service")
    subparsers.add_parser("watch-status", help="Check if watch daemon is running")

    # list
    subparsers.add_parser("list", help="List recordings")

    # history
    subparsers.add_parser("history", help="Show transcription history and model stats")

    # setup
    subparsers.add_parser("setup", help="Set up audio devices for recording")

    args = parser.parse_args()

    commands = {
        "rec": cmd_record,
        "run": cmd_run,
        "live": cmd_live,
        "watch": cmd_watch,
        "install-daemon": cmd_install_daemon,
        "uninstall-daemon": cmd_uninstall_daemon,
        "watch-status": cmd_watch_status,
        "list": cmd_list,
        "history": cmd_history,
        "setup": cmd_setup,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        print()
        print("  Quick start:")
        print("    transcribe setup  — Check audio setup")
        print("    transcribe rec    — Record audio")
        print("    transcribe run    — Transcribe a recording")
        print("    transcribe live   — Record + transcribe at once")
        print("    transcribe watch  — Auto-record calendar meetings")
        print("    transcribe list   — List recordings")
        print()


if __name__ == "__main__":
    main()
