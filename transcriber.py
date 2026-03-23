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
import gc
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from system_audio import SystemAudioCapture as _SystemAudioCapture, is_available as _sck_is_available
    _SCK_AVAILABLE = _sck_is_available()
except ImportError:
    _SystemAudioCapture = None
    _SCK_AVAILABLE = False



# ─── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"

DEFAULT_CONFIG = {
    # ── General ──────────────────────────────────────────────────────────
    # Transcription model: "parakeet" (recommended) or Whisper variants
    # Parakeet: CTC model — no hallucinations, proper punctuation, 2.5GB
    # Whisper: small.en, medium, turbo, large-v3
    "default_model": "parakeet",
    # Language code for transcription (e.g. "en", "ar", "fr")
    "language": "en",

    # ── Batch sizes per model ────────────────────────────────────────────
    # Higher = faster but more RAM. These are tuned for M1 16GB.
    # Reduce if you get memory errors; increase on machines with more RAM.
    "batch_sizes": {
        "small.en": 16, "medium": 8, "turbo": 4, "large-v3": 4
    },

    # ── File paths ───────────────────────────────────────────────────────
    "paths": {
        # Root folder for recordings and transcripts
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
        "silence_threshold": 0.010,
        # Minutes of continuous silence before auto-stopping recording
        "silence_timeout_minutes": 10,
        # Seconds of low mic level before showing warning
        "mic_low_warning_seconds": 10,
    },

    # Audio file management after transcription
    "audio": {
        # Keep the source WAV after successful transcription (True) or delete it (False)
        "keep_recording": True,
        # Diarize mic channel (for conference room with multiple local speakers)
        # False (default): mic = "You" — fastest, correct when alone at desk
        # True: run Sortformer on mic channel too to separate local speakers
        "diarize_mic": False,
    },

    # Accepted audio file types for transcription and listing
    "audio_formats": [".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"],

    # ── Auto-title ─────────────────────────────────────────────────────
    # If true, uses current calendar event title as transcript name
    # Falls back to date-only if no event is happening or calendar unavailable
    "auto_title_from_calendar": True,

    # ── Whisper-specific settings ────────────────────────────────────────
    # Only used when default_model is a Whisper model (not parakeet)
    "whisper": {
        # Skip silent segments where hallucination is detected (seconds).
        # Set to None to disable. Requires word_timestamps=True internally.
        "hallucination_silence_threshold": 2.0,
    },

    # ── Speaker diarization (who said what) ──────────────────────────────
    "diarization": {
        # Set to false to always skip diarization (faster, no speaker labels)
        "enabled": True,
        # Sortformer model (max 4 speakers per channel)
        "mlx_model": "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16",
        # Chunk duration in seconds for streaming mode (lower = less RAM)
        "mlx_chunk_duration": 10.0,
        # Speaker activity threshold (0-1). Lower = catch more speech, higher = fewer false positives
        "threshold": 0.5,
        # Ignore speaker segments shorter than this (seconds). Filters micro-segments
        "min_duration": 0.0,
        # Merge segments from same speaker closer than this gap (seconds). Reduces fragmentation
        "merge_gap": 0.0,
    },

    # ── Live mode (record + transcribe simultaneously) ───────────────────
    "live": {
        # Model for live mode — use a lighter model for real-time processing
        # The heavier default_model is used for post-recording transcription (cmd_run)
        "model": "small.en",
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
        "model": "small",
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

    # ── Audio normalization ──────────────────────────────────────────────
    "normalize": {
        # Normalize audio volume before transcription.
        # Scales to target RMS with peak headroom to prevent clipping.
        # Each stereo channel is normalized independently.
        "enabled": True,
        # Target RMS level (linear). 0.05 ≈ -26 dBFS — good for all models.
        "target_rms": 0.05,
        # Peak headroom in dB. Prevents peaks from exceeding -N dBFS.
        # 1.0 dB keeps peaks at -1 dBFS (0.891 linear) — no clipping.
        "peak_headroom_db": 1.0,
    },

    # ── Audio denoising ─────────────────────────────────────────────────
    "denoise": {
        # Spectral subtraction to reduce background noise before transcription.
        # Disabled by default — eval tests showed all models handle noise well
        # without preprocessing (+6-14% WER increase with denoise on).
        # Enable with --denoise flag for recordings with heavy constant noise
        # that start with silence (first N seconds used as noise profile).
        "enabled": False,
        # Over-subtraction factor: how aggressively to remove noise.
        # 2.0 = balanced (recommended). Higher = more removal but risks speech distortion.
        "factor": 2.0,
        # Seconds of audio to use as noise profile (from start of file).
        "noise_profile_seconds": 10,
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
        "parakeet":  {"speed_factor": 0.5, "description": "best accuracy + speed, no hallucinations ★", "size": "~2.5GB"},
        "small.en":  {"speed_factor": 0.5, "description": "fast, low RAM, English-only",               "size": "~460MB"},
        "medium":    {"speed_factor": 1.5, "description": "balanced, multilingual (99 languages)",      "size": "~1.5GB"},
        "turbo":     {"speed_factor": 0.8, "description": "fast + accurate, multilingual",              "size": "~1.6GB"},
        "large-v3":  {"speed_factor": 4.0, "description": "most capable Whisper (slow, high RAM)",      "size": "~3GB"},
    },

    # ── MLX model repos (HuggingFace) ────────────────────────────────
    # Maps model names to mlx-community HF repos (auto-downloaded on first use)
    "mlx_models": {
        "parakeet":  "mlx-community/parakeet-tdt-0.6b-v3",
        "small.en":  "mlx-community/whisper-small.en-mlx",
        "medium":    "mlx-community/whisper-medium-mlx",
        "turbo":     "mlx-community/whisper-large-v3-turbo",
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
VIRTUAL_DEVICE_NAMES = ("BlackHole", "Aggregate", "Multi-Output")  # filtered from mic selection
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
STEP_NAMES = ["Transcribing", "Saving"]  # default; overridden by cmd_run with actual pipeline steps


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
                "Check the paths.obsidian_base setting in config.json.")
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


def _safe_write_audio(target_path, audio_data, sample_rate, description="audio"):
    """Write audio to target_path with retry prompt and ~/Downloads fallback.
    Never overwrites existing files — appends (2), (3), etc.
    Cleans up partial files on failure.

    Returns the path that was actually written to.
    """
    import soundfile as sf

    target_path = _unique_path(target_path)
    while True:
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(target_path), audio_data, sample_rate, subtype="PCM_16")
            return target_path
        except Exception as primary_err:
            # Remove partial file left by failed write
            try:
                if target_path.exists() and target_path.stat().st_size == 0:
                    target_path.unlink()
            except OSError:
                pass
            choice = _ask_retry(description, primary_err, target_path)
            if choice == "retry":
                continue
            if choice == "quit":
                print(f"\n  ❌ {description} not saved. Audio data lost.")
                return None
            # fallback to ~/Downloads
            fallback = _unique_path(FALLBACK_DIR / target_path.name)
            try:
                sf.write(str(fallback), audio_data, sample_rate, subtype="PCM_16")
                print(f"  ✅ Saved to: {fallback}")
                return fallback
            except Exception as fallback_err:
                try:
                    if fallback.exists() and fallback.stat().st_size == 0:
                        fallback.unlink()
                except OSError:
                    pass
                print(f"\n  ❌ Fallback also failed: {fallback_err}")
                return None


def _safe_write_binary(target_path, write_fn, description="file"):
    """Run write_fn(path) with retry prompt and ~/Downloads fallback.
    Never overwrites existing files — appends (2), (3), etc.

    write_fn receives the path to write to. Returns the path actually written.
    """
    target_path = _unique_path(target_path)
    while True:
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            write_fn(target_path)
            return target_path
        except Exception as primary_err:
            choice = _ask_retry(description, primary_err, target_path)
            if choice == "retry":
                continue
            if choice == "quit":
                return None
            # fallback to ~/Downloads
            fallback = _unique_path(FALLBACK_DIR / target_path.name)
            try:
                write_fn(fallback)
                print(f"  ✅ Saved to: {fallback}")
                return fallback
            except Exception as fallback_err:
                print(f"\n  ❌ Fallback also failed: {fallback_err}")
                return None


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


# ─── Run History ─────────────────────────────────────────────────────────────

HISTORY_LOG = SCRIPT_DIR / "history.jsonl"


def _get_system_info():
    """Collect system info for debugging: RAM, chip, GPU memory pressure."""
    info = {}
    try:
        import platform
        info["os"] = platform.mac_ver()[0]
        info["chip"] = platform.processor() or subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        pass
    try:
        # Total RAM in GB
        mem = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5
        )
        if mem.returncode == 0:
            info["ram_gb"] = round(int(mem.stdout.strip()) / (1024**3), 1)
    except Exception:
        pass
    try:
        # Available RAM at time of run
        import psutil
        vm = psutil.virtual_memory()
        info["ram_available_gb"] = round(vm.available / (1024**3), 1)
        info["ram_percent_used"] = vm.percent
    except ImportError:
        pass
    try:
        # GPU memory pressure (macOS)
        gpu = subprocess.run(
            ["sysctl", "-n", "iogpu.wired_limit_mb"],
            capture_output=True, text=True, timeout=5
        )
        if gpu.returncode == 0 and gpu.stdout.strip():
            info["gpu_wired_limit_mb"] = int(gpu.stdout.strip())
    except Exception:
        pass
    return info


def _log_run(metadata, step_times=None):
    """Append a transcription run to history.jsonl for benchmarking."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        **metadata,
    }
    if step_times:
        entry["steps"] = {
            name: round(t, 1) for name, t in step_times.items()
        }
    entry["system"] = _get_system_info()
    try:
        with open(HISTORY_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def cmd_history(args):
    """Show transcription run history with model performance stats."""
    if not HISTORY_LOG.exists():
        print("  No history yet. Run a transcription first.")
        return

    runs = []
    with open(HISTORY_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not runs:
        print("  No history yet.")
        return

    # Summary table
    print()
    print(f"  {'Date':<12} {'Model':<14} {'Audio':>7} {'Time':>7} {'Speed':>7} {'Seg':>5} {'Spk':>5}  File")
    print(f"  {'─'*12} {'─'*14} {'─'*7} {'─'*7} {'─'*7} {'─'*5} {'─'*5}  {'─'*20}")

    for run in runs:
        date = run.get("timestamp", "")[:10]
        model = run.get("model", "?")[:13]
        audio_dur = run.get("audio_duration", 0)
        proc_time = run.get("processing_time", 0)
        speed = f"{audio_dur / proc_time:.1f}x" if proc_time > 0 else "?"
        segments = run.get("segments", "?")
        speakers = run.get("speakers", "?")
        audio_file = run.get("audio_file", "?")
        if len(audio_file) > 30:
            audio_file = audio_file[:27] + "..."

        print(f"  {date:<12} {model:<14} {format_duration(audio_dur):>7} {format_duration(proc_time):>7} {speed:>7} {segments:>5} {speakers:>5}  {audio_file}")

    # Per-model averages
    from collections import defaultdict
    model_stats = defaultdict(list)
    for run in runs:
        m = run.get("model")
        audio_dur = run.get("audio_duration", 0)
        proc_time = run.get("processing_time", 0)
        if m and proc_time > 0:
            model_stats[m].append(audio_dur / proc_time)

    if model_stats:
        print()
        print(f"  {'Model':<14} {'Runs':>5} {'Avg Speed':>10} {'Min':>7} {'Max':>7}")
        print(f"  {'─'*14} {'─'*5} {'─'*10} {'─'*7} {'─'*7}")
        for model, speeds in sorted(model_stats.items()):
            avg = sum(speeds) / len(speeds)
            print(f"  {model:<14} {len(speeds):>5} {avg:>9.1f}x {min(speeds):>6.1f}x {max(speeds):>6.1f}x")

    print()


# ─── Display Helpers ──────────────────────────────────────────────────────────

def clear_line():
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


from ui import format_duration  # shared with ui.py


def level_meter(rms, width=20):
    """Create a visual audio level meter."""
    level = min(1.0, rms * 15)
    filled = int(level * width)
    return "█" * filled + "░" * (width - filled)


def spinner_char(tick):
    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    return chars[tick % len(chars)]


def get_available_ram_gb():
    """Get available RAM in GB (macOS)."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
        )
        total = int(result.stdout.strip()) / (1024 ** 3)
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


def adaptive_batch_size(batch_size, model_name):
    """Scale batch_size based on available RAM. Returns (adjusted_size, message)."""
    available = get_available_ram_gb()
    if available < 0:
        return batch_size, None  # can't determine, use default

    if available < 2:
        adjusted = max(1, batch_size // 4)
        return adjusted, f"⚠ Low RAM ({available:.1f}GB free) — batch_size {batch_size}→{adjusted}"
    elif available < 4:
        adjusted = max(2, batch_size // 2)
        return adjusted, f"⚠ Limited RAM ({available:.1f}GB free) — batch_size {batch_size}→{adjusted}"
    else:
        return batch_size, None  # plenty of RAM



# ─── Audio Device Detection ───────────────────────────────────────────────────

def get_default_mic():
    """Get the best available real input device, preferring AirPods over MacBook mic."""
    import sounddevice as sd

    # Pass 1 — trust the system default if it's a real (non-virtual) device
    default_idx = sd.default.device[0]
    if default_idx is not None and default_idx >= 0:
        try:
            info = sd.query_devices(default_idx)
            name = info.get("name", "")
            if not any(v in name for v in VIRTUAL_DEVICE_NAMES):
                return default_idx, name
        except Exception:
            pass

    # Pass 2 — system default is virtual or unset; find best real mic
    # Priority 1: Bluetooth/AirPods; Priority 2: any other real input
    airpods_result = None
    fallback_result = None
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and not any(v in d["name"] for v in VIRTUAL_DEVICE_NAMES):
            if airpods_result is None and any(k in d["name"] for k in ("AirPods", "Bluetooth", "Wireless", "Pro")):
                airpods_result = (i, d["name"])
            elif fallback_result is None:
                fallback_result = (i, d["name"])
    return airpods_result or fallback_result or (None, "Unknown")




def cmd_setup(args):
    """Check audio setup status."""
    import sounddevice as sd

    mic_id, mic_name = get_default_mic()
    sck_status = "✅ ScreenCaptureKit (system audio)" if _SCK_AVAILABLE else "❌ pip install pyobjc-framework-ScreenCaptureKit"

    from ui import info_panel
    print()
    info_panel("🔧 Audio Setup", [
        ("System:", sck_status),
        ("Mic:", mic_name),
    ])

    if _SCK_AVAILABLE:
        print()
        print("  ✅ Ready! System audio is captured via ScreenCaptureKit.")
        print("     No virtual audio drivers needed.")
        print("     Volume keys work normally.")
    else:
        print()
        print("  ⚠  System audio not available. Only mic will be recorded.")
        print("     To capture Zoom/Teams audio:")
        print("     pip install pyobjc-framework-ScreenCaptureKit")

    print()
    print("  Input devices:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            markers = []
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

    mic_id, mic_name = get_default_mic()
    dual_mode = _SystemAudioCapture is not None
    source_label = f"System + {mic_name}" if dual_mode else mic_name

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = RECORDINGS_DIR / f"{timestamp}.wav"

    from ui import info_panel
    print()
    info_panel("🎙 Recording", [
        ("Source:", source_label),
        ("Silence:", f"Auto-stops after {SILENCE_TIMEOUT // 60}min"),
        ("Controls:", "P = pause, Ctrl+C = stop"),
    ])
    print()

    mic_frames = []
    sys_frames = []
    recording = True
    paused = False
    silence_start = None
    current_mic_rms = 0.0
    current_sys_rms = 0.0
    start_time = time.time()
    lock = threading.Lock()

    def mic_callback(indata, frame_count, time_info, status):
        nonlocal silence_start, current_mic_rms, recording
        if not recording:
            raise sd.CallbackAbort()
        if paused:
            return
        mono = indata.mean(axis=1, keepdims=True) if indata.shape[1] > 1 else indata
        with lock:
            mic_frames.append(mono.copy())
            current_mic_rms = float(np.sqrt(np.mean(mono ** 2)))
            if current_mic_rms < SILENCE_THRESHOLD:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_TIMEOUT:
                    recording = False
                    raise sd.CallbackAbort()
            else:
                silence_start = None

    def sys_callback(indata, frame_count, time_info, status):
        nonlocal silence_start, current_sys_rms
        if not recording:
            raise sd.CallbackAbort()
        if paused:
            return
        mono = indata.mean(axis=1, keepdims=True) if indata.shape[1] > 1 else indata
        with lock:
            sys_frames.append(mono.copy())
            current_sys_rms = float(np.sqrt(np.mean(mono ** 2)))
            if current_sys_rms > SILENCE_THRESHOLD:
                silence_start = None

    blocksize = int(SAMPLE_RATE * 0.5)

    mic_stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        device=mic_id, blocksize=blocksize, callback=mic_callback,
    )
    sck_capture = None
    if _SystemAudioCapture is not None:
        try:
            sck_capture = _SystemAudioCapture(sample_rate=SAMPLE_RATE, callback=sys_callback)
        except RuntimeError as e:
            print(f"  ⚠  System audio unavailable: {e}")
            dual_mode = False

    from ui import RecordingDisplay, raw_keys, poll_key, check_system_health

    auto_stopped = False
    display = RecordingDisplay()
    last_health_check = 0
    try:
        mic_stream.start()
        if sck_capture:
            sck_capture.start()
        mic_low_since_rec = None
        with raw_keys(), display:
            while recording:
                # Handle keyboard
                key = poll_key()
                if key == "p":
                    paused = not paused
                    if paused:
                        silence_start = None  # don't count pause as silence

                with lock:
                    mic_rms = current_mic_rms
                    sys_rms_val = current_sys_rms
                    sil = silence_start

                rms = max(mic_rms, sys_rms_val)  # combined for quality indicator
                elapsed = time.time() - start_time
                silence_elapsed = time.time() - sil if sil else 0

                # Warnings: mic low + system health
                warning = None
                if not paused and mic_rms < 0.001:
                    if mic_low_since_rec is None:
                        mic_low_since_rec = time.time()
                    elif time.time() - mic_low_since_rec > MIC_LOW_WARNING_SECONDS:
                        warning = "Mic very low!"
                else:
                    mic_low_since_rec = None

                # Check system health every 30s
                now = time.time()
                if now - last_health_check > 30:
                    last_health_check = now
                    health = check_system_health()
                    if health:
                        warning = health if not warning else f"{warning} │ {health}"

                display.set_warning(warning)

                display.update(rms, elapsed, silence_elapsed, paused=paused,
                              mic_rms=mic_rms if dual_mode else None,
                              sys_rms=sys_rms_val if dual_mode else None)
                time.sleep(0.25)

        auto_stopped = True

    except KeyboardInterrupt:
        recording = False
    finally:
        mic_stream.stop()
        mic_stream.close()
        if sck_capture:
            sck_capture.stop()
    elapsed = time.time() - start_time

    if not mic_frames:
        print("  ⚠  No audio recorded.")
        return

    audio_data = _mix_audio(mic_frames, sys_frames, dual_mode)
    saved_to = _safe_write_audio(output_path, audio_data, SAMPLE_RATE, description="recording")
    if saved_to:
        output_path = saved_to

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

    prompt_transcribe(str(output_path))


def _mix_audio(mic_frames, sys_frames, dual_mode, normalize=True):
    """Combine mic + system audio. Returns stereo (N,2) when dual, mono (N,) when single.

    Stereo format: channel 0 = mic, channel 1 = system audio.
    Each channel is independently normalized before combining (unless normalize=False).
    """
    if not mic_frames:
        return np.array([], dtype=np.float32)
    mic_audio = np.concatenate(mic_frames).flatten()
    if normalize:
        mic_audio = _normalize(mic_audio)
    if dual_mode and sys_frames:
        sys_audio = np.concatenate(sys_frames).flatten()
        if normalize:
            sys_audio = _normalize(sys_audio)
        min_len = min(len(mic_audio), len(sys_audio))
        _log("mix", mode="stereo", normalize=normalize, samples=min_len,
             mic_rms=f"{np.sqrt(np.mean(mic_audio[:min_len]**2)):.5f}",
             sys_rms=f"{np.sqrt(np.mean(sys_audio[:min_len]**2)):.5f}")
        return np.column_stack([mic_audio[:min_len], sys_audio[:min_len]])
    _log("mix", mode="mono", normalize=normalize, samples=len(mic_audio),
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
            no_diarize=False,
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


# ─── Live Record + Transcribe ────────────────────────────────────────────────

def cmd_live(args):
    """Record and transcribe simultaneously — transcript ready shortly after recording stops."""
    import sounddevice as sd
    import soundfile as sf

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    model_name = args.model or config["live"].get("model", config["default_model"])
    language = args.language or config["language"]
    batch_size = config["batch_sizes"].get(model_name, 8)
    chunk_interval = config["live"]["chunk_interval_seconds"]
    no_diarize = getattr(args, "no_diarize", False)
    title = getattr(args, "title", None)

    # ── Pre-load whisper model ────────────────────────────────────────────
    from ui import console
    from rich.panel import Panel
    print()
    console.print(Panel("[bold]🎙 Live Record + Transcribe[/bold]", border_style="dim", expand=False, width=54))
    print()
    print(f"  Loading '{model_name}' model...")
    import mlx_whisper
    mlx_repo = _get_mlx_repo(model_name)
    model = None  # MLX loads on first transcribe() call
    print(f"  ✅ Model ready — starting recording")

    # ── Find audio devices ────────────────────────────────────────────────
    mic_id, mic_name = get_default_mic()
    dual_mode = _SystemAudioCapture is not None
    source_label = f"System + {mic_name}" if dual_mode else mic_name

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = RECORDINGS_DIR / f"{timestamp}.wav"

    from ui import info_panel
    print()
    info_panel("🎙 Recording", [
        ("Source:", source_label),
        ("Silence:", f"Auto-stops after {SILENCE_TIMEOUT // 60}min"),
        ("Transcribe:", f"Every {chunk_interval}s in background"),
        ("Controls:", "Ctrl+C to stop"),
    ])
    print()

    # ── Recording state ───────────────────────────────────────────────────
    mic_frames = []
    sys_frames = []
    recording = True
    silence_start = None
    current_rms = 0.0
    mic_rms_val = 0.0
    sys_rms_val = 0.0
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

    def sys_callback(indata, frame_count, time_info, status):
        nonlocal silence_start, current_rms, sys_rms_val
        if not recording:
            raise sd.CallbackAbort()
        mono = indata.mean(axis=1, keepdims=True) if indata.shape[1] > 1 else indata
        with lock:
            sys_frames.append(mono.copy())
            sys_rms_val = float(np.sqrt(np.mean(mono ** 2)))
            if sys_rms_val > SILENCE_THRESHOLD:
                silence_start = None
                current_rms = max(current_rms, sys_rms_val)

    def _get_mixed_snapshot():
        """Get current mixed audio snapshot as mono (for live chunk transcription)."""
        with lock:
            if not mic_frames:
                return None
            mic_copy = list(mic_frames)
            sys_copy = list(sys_frames) if dual_mode else []
        mic_audio = _normalize(np.concatenate(mic_copy).flatten())
        if sys_copy:
            sys_audio = _normalize(np.concatenate(sys_copy).flatten())
            min_len = min(len(mic_audio), len(sys_audio))
            return (mic_audio[:min_len] + sys_audio[:min_len]) / 2.0
        return mic_audio

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
                result = _transcribe_mlx_chunk(chunk, mlx_repo, language)
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
    sck_capture = None
    if _SystemAudioCapture is not None:
        try:
            sck_capture = _SystemAudioCapture(sample_rate=SAMPLE_RATE, callback=sys_callback)
        except RuntimeError as e:
            print(f"  ⚠  System audio unavailable: {e}")
            dual_mode = False

    transcription_thread = threading.Thread(target=transcription_worker, daemon=True)
    auto_stopped = False
    _last_adaptive_msg = [""]  # mutable container so inner scope can read it

    try:
        mic_stream.start()
        if sck_capture:
            sck_capture.start()
        transcription_thread.start()

        while recording:
            with lock:
                rms = current_rms
                sil = silence_start
                m_rms = mic_rms_val
                b_rms = sys_rms_val

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
        if sck_capture:
            sck_capture.stop()

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
    audio_data = _mix_audio(mic_frames, sys_frames, dual_mode)
    saved_to = _safe_write_audio(output_path, audio_data, SAMPLE_RATE, description="recording")
    if saved_to:
        output_path = saved_to
    print(f"  📁 Saved: {output_path.name}")

    # ── Transcribe remaining audio ────────────────────────────────────────
    # For live post-processing, use mono downmix (per-channel is done by cmd_run)
    if audio_data.ndim == 2:
        flat_audio = audio_data.mean(axis=1).astype(np.float32)
    else:
        flat_audio = audio_data.flatten().astype(np.float32)
    total_samples = len(flat_audio)
    remaining = total_samples - last_transcribed_sample

    if adaptive["rec_only"] and last_transcribed_sample == 0:
        # Never managed to transcribe anything — do full file transcription
        remaining_duration = total_samples / SAMPLE_RATE
        print(f"  ⏳ Transcribing full recording ({format_duration(remaining_duration)})...")
        try:
            result = _transcribe_mlx(str(output_path), model_name, language)
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
        chunk = flat_audio[chunk_start:total_samples]
        offset = chunk_start / SAMPLE_RATE
        overlap_boundary = last_transcribed_sample / SAMPLE_RATE

        try:
            result = _transcribe_mlx_chunk(chunk, mlx_repo, language)
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
    if model is not None:
        del model
    gc.collect()

    all_segments = sorted(transcribed_segments, key=lambda s: s.get("start", 0))
    if not all_segments:
        print("  ⚠  No speech detected.")
        return

    result = {"segments": all_segments, "language": language}

    # ── Diarization ─────────────────────────────────────────────────────
    diarize_enabled = config["diarization"]["enabled"] and not no_diarize
    if diarize_enabled and not HF_TOKEN:
        print("  ⚠  Skipping speaker ID — HF_TOKEN not set (see README)")
    if HF_TOKEN and diarize_enabled:
        print("  ⏳ Identifying speakers...")
        try:
            # For stereo files, diarize system channel only
            if _is_stereo(str(output_path)):
                mic_tmp, sys_tmp = _split_stereo(str(output_path))
                try:
                    speaker_turns = _diarize_standalone(sys_tmp)
                finally:
                    for p in (mic_tmp, sys_tmp):
                        if p and os.path.exists(p):
                            os.unlink(p)
            else:
                speaker_turns = _diarize_standalone(str(output_path))
            result["segments"] = _assign_speakers_to_segments(result.get("segments", []), speaker_turns)
        except Exception as e:
            print(f"  ⚠  Diarization failed: {e}")

    # ── Save transcript ───────────────────────────────────────────────────
    # Checkpoint: save raw result immediately so downstream failures don't lose it
    cp_path = _save_checkpoint(result, str(output_path), title)

    try:
        speaker_names = _rename_speakers(result)

        if not title:
            title = Path(output_path).stem.replace("_", " ").replace("-", " ").title()

        post_time = time.time() - recording_end
        date_str = extract_recording_date(str(output_path))
        metadata = {
            "model": model_name,
            "engine": "mlx",
            "language": language,
            "audio_duration": elapsed,
            "processing_time": post_time,
            "audio_file": output_path.name,
        }
        markdown = format_transcript(result, title, speaker_names, date_str, metadata)
        SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

        safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in title)
        transcript_file = SCRIPTS_DIR / f"{date_str} {safe_title}.md"
        saved_to = _safe_write_text(transcript_file, markdown, description="transcript")
        if saved_to:
            transcript_file = saved_to

        _remove_checkpoint(cp_path)

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

        # Open transcript folder in Finder
        subprocess.run(["open", str(transcript_file.parent)], check=False)

        # Log run to history for benchmarking
        _log_run({
            "model": model_name,
            "engine": "mlx",
            "language": language,
            "audio_duration": round(elapsed, 1),
            "processing_time": round(post_time, 1),
            "segments": len(result.get("segments") or []),
            "speakers": len(speaker_names),
            "audio_file": output_path.name,
            "mode": "live",
        })

    except Exception as e:
        print(f"\n  ❌ Post-processing failed: {e}")
        if cp_path:
            print(f"  ✅ Raw transcription saved to: {cp_path}")
            print(f"  💡 Your data is safe. Re-run the command to retry formatting.")


# ─── Transcription with Progress ─────────────────────────────────────────────

class ProgressTracker:
    """Track and display transcription progress as a step list.

    Shows completed steps with ✓ and duration, current step with spinner.
    Uses Rich Live for in-place rendering (no scrolling).
    """

    def __init__(self, audio_duration, model_name):
        from ui import console
        from rich.live import Live
        self.audio_duration = audio_duration
        self.model_name = model_name
        self.start_time = time.time()
        self.step_start_time = time.time()
        self.current_step = 0
        self.running = True
        self._thread = None
        self._live = Live("", console=console, auto_refresh=False)
        self._completed = []  # [(name, duration_str)]
        self.step_names = STEP_NAMES  # can be overridden before start()
        # Keep step_weights for compatibility (used by callers for estimated_total)
        # MLX + Sortformer: transcription dominates time
        self.step_weights = [0.05, 0.65, 0.10, 0.15, 0.05]
        speed_factor = MODEL_INFO.get(model_name, {}).get("speed_factor", 1.0)
        speed_factor *= 0.15  # MLX GPU acceleration
        diar_time = audio_duration * 0.05  # Sortformer is fast
        self.estimated_total = max(10, audio_duration * speed_factor + diar_time)

    def start(self):
        self._step_times = {}
        self._live.__enter__()
        self._thread = threading.Thread(target=self._display_loop, daemon=True)
        self._thread.start()

    def stop(self):
        # Record final step
        if hasattr(self, '_step_times') and self.current_step not in self._step_times:
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
        if hasattr(self, '_step_times') and self.current_step < step_index:
            prev_elapsed = time.time() - self.step_start_time
            self._step_times[self.current_step] = prev_elapsed
            prev_name = self.step_names[self.current_step] if self.current_step < len(self.step_names) else "Step"
            self._completed.append((prev_name, format_duration(prev_elapsed)))
        self.current_step = step_index
        self.step_start_time = time.time()

    def _render(self, tick=0):
        """Build the display: completed steps + current step with spinner."""
        from rich.text import Text
        lines = []
        for name, dur in self._completed:
            lines.append(f"  [green]✓[/green] {name} [dim]({dur})[/dim]")

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
        from rich.text import Text
        lines = []
        for name, dur in self._completed:
            lines.append(f"  [green]✓[/green] {name} [dim]({dur})[/dim]")
        self._live.update(Text.from_markup("\n".join(lines)))
        self._live.refresh()

    def _display_loop(self):
        tick = 0
        while self.running:
            self._render(tick)
            tick += 1
            time.sleep(0.3)


# ─── Audio Loading & Processing ───────────────────────────────────────────────

def _is_stereo(audio_path):
    """Check if an audio file is stereo (2 channels)."""
    import soundfile as sf
    info = sf.info(str(audio_path))
    return info.channels == 2


def _split_stereo(audio_path, sr=16000):
    """Split a stereo WAV into two mono temp files (mic, system).

    Returns (mic_path, sys_path) as temp file paths. Caller must clean up.
    """
    import soundfile as sf
    data, file_sr = sf.read(str(audio_path), dtype="float32")
    if data.ndim == 1:
        return None, None  # mono file
    mic = data[:, 0]
    sys_audio = data[:, 1]
    mic_path = tempfile.mktemp(suffix="_mic.wav")
    sys_path = tempfile.mktemp(suffix="_sys.wav")
    sf.write(mic_path, mic, file_sr, subtype="PCM_16")
    sf.write(sys_path, sys_audio, file_sr, subtype="PCM_16")
    _log("split", duration=f"{len(mic)/file_sr:.1f}s",
         mic_rms=f"{np.sqrt(np.mean(mic**2)):.5f}",
         sys_rms=f"{np.sqrt(np.mean(sys_audio**2)):.5f}")
    return mic_path, sys_path


def _merge_transcripts(mic_result, sys_result):
    """Merge two transcripts (mic + system) by timestamp into one result.

    Both inputs should already have speaker labels assigned.
    """
    mic_segs = mic_result.get("segments", [])
    sys_segs = sys_result.get("segments", [])
    all_segments = mic_segs + sys_segs
    all_segments.sort(key=lambda s: s.get("start", 0))

    # Count overlapping segments (crosstalk)
    overlaps = 0
    for i in range(len(all_segments) - 1):
        if all_segments[i].get("end", 0) > all_segments[i + 1].get("start", 0):
            overlaps += 1

    _log("merge", mic_segs=len(mic_segs), sys_segs=len(sys_segs),
         total=len(all_segments), overlaps=overlaps)

    return {
        "segments": all_segments,
        "language": mic_result.get("language", sys_result.get("language", "en")),
    }


def _normalize(audio, target_rms=0.05, peak_headroom_db=1.0):
    """Normalize audio to target RMS with peak headroom to prevent clipping.

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
    actual_rms = float(np.sqrt(np.mean(audio ** 2)))
    actual_peak = float(np.max(np.abs(audio)))
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
    _log("load", file=Path(path).name, sr=sr, duration=f"{len(audio)/sr:.1f}s",
         rms=f"{np.sqrt(np.mean(audio**2)):.5f}", peak=f"{np.max(np.abs(audio)):.3f}")
    return audio


def _stft(audio, n_fft=2048, hop_length=512):
    """Short-time Fourier Transform using numpy. Returns complex array (freq_bins, frames)."""
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
    # Normalize by window overlap
    nonzero = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]
    if length is not None:
        output = output[:length]
    return output.astype(np.float32)


# ─── Audio Denoising ─────────────────────────────────────────────────────────

def _denoise_audio(audio_path):
    """Apply spectral subtraction denoising to an audio file.

    Uses the first N seconds as a noise profile, then subtracts scaled noise
    power from the entire signal.  Writes a denoised WAV to a temp file and
    returns its path.  The caller is responsible for cleanup.

    Returns (denoised_path, elapsed_seconds) or (original_path, 0) if
    denoising is disabled or fails.
    """
    denoise_cfg = config.get("denoise", {})
    if not denoise_cfg.get("enabled", True):
        return audio_path, 0

    import soundfile as sf

    factor = denoise_cfg.get("factor", 2.0)
    profile_secs = denoise_cfg.get("noise_profile_seconds", 10)
    sr = SAMPLE_RATE

    try:
        t0 = time.time()
        audio = _load_audio(audio_path, sr=sr)

        noise_clip = audio[: sr * profile_secs]
        n_fft, hop = 2048, 512

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
        sf.write(tmp_path, denoised, sr)

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
        return audio_path, 0


# ─── MLX Engine Helpers ──────────────────────────────────────────────────────

def _get_mlx_repo(model_name):
    """Get HuggingFace repo ID for an MLX whisper model."""
    mlx_models = config.get("mlx_models", {})
    return mlx_models.get(model_name, f"mlx-community/whisper-{model_name}-mlx")


def _transcribe_mlx(audio_path, model_name, language="en", word_timestamps=True):
    """Transcribe audio using mlx-whisper (Apple Silicon GPU)."""
    import mlx_whisper
    repo = _get_mlx_repo(model_name)

    # hallucination_silence_threshold skips silent segments where hallucination
    # is detected. Requires word_timestamps=True to work.
    hst = config.get("whisper", {}).get("hallucination_silence_threshold", 2.0)
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

    repo = config.get("mlx_models", {}).get(model_name, "mlx-community/parakeet-tdt-0.6b-v3")
    model = load_stt(repo)
    result = model.generate(audio_path, chunk_duration=30.0, verbose=False)

    # Convert Parakeet sentences to Whisper-compatible segments
    segments = []
    for i, sentence in enumerate(result.sentences):
        segments.append({
            "id": i,
            "start": sentence.start,
            "end": sentence.end,
            "text": sentence.text,
        })

    words = sum(len(s.get("text", "").split()) for s in segments)
    speech_dur = sum(s.get("end", 0) - s.get("start", 0) for s in segments)
    _log("asr", model=model_name, segments=len(segments), words=words,
         speech=f"{speech_dur:.1f}s")

    del model
    gc.collect()

    return {"segments": segments, "language": "en", "text": result.text}


def _transcribe_mlx_chunk(chunk, mlx_repo, language="en"):
    """Transcribe a numpy audio chunk via mlx-whisper (writes to temp file).

    Returns the transcription result dict.
    """
    import mlx_whisper
    import soundfile as sf

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, chunk, SAMPLE_RATE)
            tmp_path = tmp.name
        return mlx_whisper.transcribe(
            tmp_path, path_or_hf_repo=mlx_repo,
            word_timestamps=True, language=language,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _diarize_mlx_audio(audio_path):
    """Run speaker diarization using MLX Sortformer (fast, Apple GPU).

    Uses streaming mode to keep RAM usage low on 16GB machines.
    Max 4 speakers (architectural limit of Sortformer model).
    """
    from mlx_audio.vad import load as load_vad
    import mlx.core as mx

    diar_config = config["diarization"]
    model_id = diar_config.get("mlx_model", "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16")
    chunk_duration = diar_config.get("mlx_chunk_duration", 10.0)
    threshold = diar_config.get("threshold", 0.5)
    min_duration = diar_config.get("min_duration", 0.0)
    merge_gap = diar_config.get("merge_gap", 0.0)

    model = load_vad(model_id)

    # Streaming mode — processes in chunks, uses much less RAM
    turns = []
    for chunk_result in model.generate_stream(
        audio_path,
        chunk_duration=chunk_duration,
        threshold=threshold,
        min_duration=min_duration,
        merge_gap=merge_gap,
        verbose=False,
    ):
        for seg in chunk_result.segments:
            turns.append({
                "start": seg.start,
                "end": seg.end,
                "speaker": f"Speaker {seg.speaker}",
            })

    speakers = set(t["speaker"] for t in turns)
    _log("diar", speakers=len(speakers), turns=len(turns),
         labels=sorted(speakers))

    del model
    gc.collect()
    mx.clear_cache()
    return turns


def _diarize_standalone(audio_path):
    """Run speaker diarization using MLX Sortformer."""
    t0 = time.time()
    turns = _diarize_mlx_audio(audio_path)
    _log("diar_standalone", file=Path(audio_path).name, time=f"{time.time()-t0:.1f}s")
    return turns


def _assign_speakers_to_segments(segments, speaker_turns):
    """Match transcription segments to speaker turns by timestamp overlap.

    Modifies segments in-place and returns a dict with segments key.
    """
    for seg in segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        best_speaker = "Unknown"
        best_overlap = 0.0
        for turn in speaker_turns:
            overlap_start = max(seg_start, turn["start"])
            overlap_end = min(seg_end, turn["end"])
            overlap = max(0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]
        seg["speaker"] = best_speaker
    return segments


def cmd_run(args):
    """Transcribe an audio file with progress tracking."""
    global _DEBUG
    if getattr(args, "debug", False):
        _DEBUG = True

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

    title = getattr(args, "title", None)
    no_diarize = getattr(args, "no_diarize", False)
    force_denoise = getattr(args, "denoise", False)
    no_denoise = getattr(args, "no_denoise", False)
    normalize_cfg = config.get("normalize", {})
    normalize_enabled = normalize_cfg.get("enabled", True)
    batch_size = config["batch_sizes"].get(model_name, 8)
    batch_size, ram_msg = adaptive_batch_size(batch_size, model_name)
    if ram_msg:
        print(f"  {ram_msg}")

    if not title and config.get("auto_title_from_calendar", True):
        title = get_current_event_title()
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
    speed_factor *= 0.15  # MLX GPU acceleration
    estimated_time = audio_duration * speed_factor
    diarize_enabled = config["diarization"]["enabled"] and not no_diarize and HF_TOKEN
    if diarize_enabled:
        estimated_time += audio_duration * 0.05  # Sortformer: ~3s per minute

    is_parakeet_model = _is_parakeet(model_name)
    if force_denoise:
        denoise_enabled = True
    elif no_denoise:
        denoise_enabled = False
    else:
        denoise_enabled = config.get("denoise", {}).get("enabled", False)
    denoise_factor = config.get("denoise", {}).get("factor", 2.0)

    is_stereo_input = _is_stereo(audio_path)
    engine_label = "parakeet (GPU)" if is_parakeet_model else "mlx (GPU)"
    model_size = MODEL_INFO.get(model_name, {}).get("size", "?")
    diar_label = "Sortformer (GPU)" if diarize_enabled else "off"
    denoise_label = f"spectral sub {denoise_factor}x" if denoise_enabled else "off"
    channels_label = "stereo (mic + system)" if is_stereo_input else "mono"

    denoise_prefix = "Denoise → " if denoise_enabled else ""
    if is_stereo_input:
        if diarize_enabled:
            steps_preview = f"{denoise_prefix}Diarize system → Transcribe mic + system → Save"
        else:
            steps_preview = f"{denoise_prefix}Transcribe mic + system → Save"
    elif diarize_enabled:
        steps_preview = f"{denoise_prefix}Diarize → Transcribe → Save"
    else:
        steps_preview = f"{denoise_prefix}Transcribe → Save"

    from ui import info_panel
    rows = [
        ("File:", Path(audio_path).name),
        ("Duration:", format_duration(audio_duration)),
        ("Channels:", channels_label),
        ("Engine:", engine_label),
        ("Model:", f"{model_name} ({model_size})"),
        ("Language:", language),
        ("Denoise:", denoise_label),
        ("Normalize:", f"{normalize_cfg.get('target_rms', 0.05)} RMS" if normalize_enabled else "off"),
        ("Diarize:", diar_label),
        ("Pipeline:", steps_preview),
        ("Est time:", f"~{format_duration(estimated_time)}"),
        ("Started:", datetime.now().strftime("%H:%M:%S")),
    ]
    print()
    info_panel("📝 Transcription", rows)
    print()

    # Pipeline steps
    if diarize_enabled:
        step_names = ["Identifying speakers", "Transcribing", "Saving"]
        step_weights = [0.15, 0.80, 0.05]
    else:
        step_names = ["Transcribing", "Saving"]
        step_weights = [0.90, 0.10]

    # Prepend denoising step if enabled
    if denoise_enabled:
        step_names = ["Denoising audio"] + step_names
        # Denoising is fast (~2s) — give it a small weight
        step_weights = [0.02] + [w * 0.98 for w in step_weights]

    progress = ProgressTracker(audio_duration, model_name)
    progress.step_names = step_names
    progress.step_weights = step_weights
    progress.start()

    denoised_path = None  # Track temp file for cleanup
    normalized_path = None  # Track temp file for cleanup
    result = None
    try:
        step = 0

        # ── Denoise (off by default, enable with --denoise flag, mono only) ──
        is_stereo_file = _is_stereo(audio_path)
        if denoise_enabled and not is_stereo_file:
            progress.set_step(step)
            denoised_path, denoise_time = _denoise_audio(audio_path)
            transcribe_path = denoised_path
            step += 1
        else:
            transcribe_path = audio_path

        # ── Normalize ──
        target_rms = normalize_cfg.get("target_rms", 0.05)
        headroom = normalize_cfg.get("peak_headroom_db", 1.0)
        if normalize_enabled and not is_stereo_file:
            import soundfile as sf
            audio_data = _load_audio(transcribe_path, sr=SAMPLE_RATE)
            audio_data = _normalize(audio_data, target_rms=target_rms, peak_headroom_db=headroom)
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix=".norm_")
            os.close(tmp_fd)
            sf.write(tmp_path, audio_data, SAMPLE_RATE)
            normalized_path = tmp_path
            transcribe_path = normalized_path

        import mlx.core as mx

        if is_stereo_file:
            # ── Stereo pipeline: transcribe each channel independently ──
            mic_path, sys_path = _split_stereo(transcribe_path)

            # Normalize each channel independently
            if normalize_enabled:
                import soundfile as sf
                for ch_path in (mic_path, sys_path):
                    ch_audio = _load_audio(ch_path, sr=SAMPLE_RATE)
                    ch_audio = _normalize(ch_audio, target_rms=target_rms, peak_headroom_db=headroom)
                    sf.write(ch_path, ch_audio, SAMPLE_RATE)

            diarize_mic = config.get("audio", {}).get("diarize_mic", False)
            try:
                # Diarize system channel (remote speakers) — always when enabled
                sys_speaker_turns = None
                mic_speaker_turns = None
                if diarize_enabled:
                    progress.set_step(step)  # Identifying speakers
                    sys_speaker_turns = _diarize_standalone(sys_path)
                    if diarize_mic:
                        gc.collect()
                        mx.clear_cache()
                        mic_speaker_turns = _diarize_standalone(mic_path)
                    gc.collect()
                    mx.clear_cache()
                    step += 1

                progress.set_step(step)  # Transcribing

                # Transcribe mic channel
                if _is_parakeet(model_name):
                    mic_result = _transcribe_parakeet(mic_path, model_name)
                else:
                    mic_result = _transcribe_mlx(mic_path, model_name, language)

                gc.collect()
                mx.clear_cache()

                # Transcribe system channel
                if _is_parakeet(model_name):
                    sys_result = _transcribe_parakeet(sys_path, model_name)
                else:
                    sys_result = _transcribe_mlx(sys_path, model_name, language)

                # Assign speaker labels — system channel
                if sys_speaker_turns:
                    sys_result["segments"] = _assign_speakers_to_segments(
                        sys_result.get("segments", []), sys_speaker_turns)
                else:
                    for seg in sys_result.get("segments", []):
                        seg["speaker"] = "Remote"

                # Assign speaker labels — mic channel
                if mic_speaker_turns:
                    # Conference room: diarize mic to separate local speakers
                    mic_result["segments"] = _assign_speakers_to_segments(
                        mic_result.get("segments", []), mic_speaker_turns)
                    # Prefix local speakers to distinguish from remote
                    for seg in mic_result.get("segments", []):
                        if seg.get("speaker", "").startswith("SPEAKER_"):
                            num = int(seg["speaker"].split("_")[1]) + 1
                            seg["speaker"] = f"Local {num}"
                else:
                    # Default: mic = "You"
                    mic_label = config.get("user_name", "You") or "You"
                    for seg in mic_result.get("segments", []):
                        seg["speaker"] = mic_label

                # Merge both channels by timestamp
                result = _merge_transcripts(mic_result, sys_result)
            finally:
                for p in (mic_path, sys_path):
                    if p and os.path.exists(p):
                        os.unlink(p)

            step += 1
            progress.set_step(step)  # Saving

        else:
            # ── Mono pipeline: original behavior ──
            speaker_turns = None

            if diarize_enabled:
                progress.set_step(step)  # Identifying speakers
                speaker_turns = _diarize_standalone(audio_path)
                gc.collect()
                mx.clear_cache()
                step += 1

            progress.set_step(step)  # Transcribing
            if _is_parakeet(model_name):
                result = _transcribe_parakeet(transcribe_path, model_name)
            else:
                result = _transcribe_mlx(transcribe_path, model_name, language,
                                         word_timestamps=not diarize_enabled)

            if speaker_turns is not None:
                result["segments"] = _assign_speakers_to_segments(
                    result.get("segments", []), speaker_turns)

            step += 1
            progress.set_step(step)  # Saving

    except Exception as e:
        progress.stop()
        print(f"\n\n  ❌ Transcription failed: {e}")
        # If we got partial results (e.g., transcription OK but diarization crashed),
        # checkpoint what we have
        if result is not None:
            cp_path = _save_checkpoint(result, audio_path, title)
            if cp_path:
                print(f"  ✅ Partial transcription saved to: {cp_path}")
                print(f"  💡 Re-run with --no-diarize to use the transcription without speakers.")
        return
    finally:
        # Clean up denoised temp file
        if denoised_path and denoised_path != audio_path:
            try:
                os.unlink(denoised_path)
            except OSError:
                pass
        # Clean up normalized temp file
        if normalized_path:
            try:
                os.unlink(normalized_path)
            except OSError:
                pass

    progress.stop()

    # Checkpoint: save raw result immediately so downstream failures don't lose it
    cp_path = _save_checkpoint(result, audio_path, title)

    try:
        # Rename speaker IDs to "Speaker 1", "Speaker 2", etc.
        speaker_names = _rename_speakers(result)

        # Save transcript
        total_time = time.time() - progress.start_time
        date_str = extract_recording_date(audio_path)
        metadata = {
            "model": model_name,
            "engine": "mlx",
            "language": language,
            "audio_duration": audio_duration,
            "processing_time": total_time,
            "audio_file": Path(audio_path).name,
        }
        markdown = format_transcript(result, title, speaker_names, date_str, metadata)
        SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

        safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in title)
        output_file = SCRIPTS_DIR / f"{date_str} {safe_title}.md"
        saved_to = _safe_write_text(output_file, markdown, description="transcript")
        if saved_to:
            output_file = saved_to

        _remove_checkpoint(cp_path)

        # Speed ratio: how fast compared to audio length
        speed_ratio = audio_duration / total_time if total_time > 0 else 0
        speed_label = f"{speed_ratio:.1f}x real-time"

        from ui import success_panel
        rows = [
            ("Audio:", f"{format_duration(audio_duration)} ({speed_label})"),
            ("Model:", model_name),
            ("Speakers:", str(len(speaker_names))),
            ("Segments:", str(len(result.get("segments", [])))),
            ("Time:", format_duration(total_time)),
        ]
        if hasattr(progress, "_step_times") and progress._step_times:
            parts = []
            for i, name in enumerate(step_names):
                if i in progress._step_times:
                    parts.append(f"{name}: {format_duration(progress._step_times[i])}")
            if parts:
                rows.append(("Steps:", parts[0]))
                for part in parts[1:]:
                    rows.append(("", part))
        rows.append(("Saved to:", f"Scripts/{output_file.name}"))
        print()
        success_panel("✅ Transcription complete!", rows)
        print()

        # Open transcript folder in Finder
        subprocess.run(["open", str(output_file.parent)], check=False)

        # Delete source recording if configured
        if not config.get("audio", {}).get("keep_recording", True):
            try:
                Path(audio_path).unlink()
            except OSError:
                pass

        # Log run to history for benchmarking
        step_log = {}
        if hasattr(progress, '_step_times'):
            for i, name in enumerate(step_names):
                if i in progress._step_times:
                    step_log[name] = progress._step_times[i]
        _log_run({
            "model": model_name,
            "engine": "mlx",
            "language": language,
            "audio_duration": round(audio_duration, 1),
            "processing_time": round(total_time, 1),
            "segments": len(result.get("segments") or []),
            "speakers": len(speaker_names),
            "audio_file": Path(audio_path).name,
        }, step_log)

    except Exception as e:
        print(f"\n  ❌ Post-processing failed: {e}")
        if cp_path:
            print(f"  ✅ Raw transcription saved to: {cp_path}")
            print(f"  💡 Your data is safe. Re-run the command to retry formatting.")


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

    max_list = config.get("list", {}).get("max_recordings", 20)
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("st", width=2)
    table.add_column("num", style="dim", width=3)
    table.add_column("name")
    table.add_column("info", style="dim")
    for i, rec in enumerate(recordings[:max_list], 1):
        duration = get_audio_duration(str(rec))
        size_mb = rec.stat().st_size / (1024 * 1024)
        date_prefix = rec.stem[:10]
        has_transcript = SCRIPTS_DIR.exists() and any(SCRIPTS_DIR.glob(f"{date_prefix}*.md"))
        status = "✅" if has_transcript else ""
        table.add_row(status, f"{i})", rec.stem, f"{format_duration(duration)}, {size_mb:.1f}MB")
    print()
    console.print(Panel(table, title="[bold]📂 Recordings[/bold]", border_style="dim", expand=False, width=60))
    console.print(f"  [dim]📂 {RECORDINGS_DIR}[/dim]")
    console.print("  [dim]✅ = transcript exists[/dim]")
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


def format_transcript(result, title, speaker_names, date_str, metadata=None):
    """Format transcription result as Obsidian-friendly Markdown.

    metadata: optional dict with keys like model, engine, audio_duration,
              processing_time, language, diarization, audio_file.
    """
    meta = metadata or {}
    lines = []

    # YAML frontmatter
    lines.append("---")
    lines.append(f"title: \"{title}\"")
    lines.append(f"date: {date_str}")
    lines.append("type: transcript")
    speaker_list = ", ".join(speaker_names.values())
    lines.append(f"speakers: [{speaker_list}]")
    if meta.get("model"):
        lines.append(f"model: {meta['model']}")
    if meta.get("language"):
        lines.append(f"language: {meta['language']}")
    if meta.get("audio_duration"):
        lines.append(f"audio_duration: {meta['audio_duration']:.1f}")
    lines.append("---")
    lines.append("")
    lines.append(f"# {title}")
    lines.append(f"**Date:** {date_str}")
    lines.append("")

    # Details block
    detail_lines = []
    if meta.get("audio_duration"):
        detail_lines.append(f"**Duration:** {format_duration(meta['audio_duration'])}")
    if meta.get("model"):
        detail_lines.append(f"**Model:** {meta['model']} ({meta.get('engine', 'unknown')})")
    if meta.get("processing_time"):
        detail_lines.append(f"**Processing time:** {format_duration(meta['processing_time'])}")
    segments = result.get("segments") or []
    detail_lines.append(f"**Segments:** {len(segments)}")
    detail_lines.append(f"**Speakers:** {len(speaker_names)}")
    if meta.get("audio_file"):
        detail_lines.append(f"**Source:** {meta['audio_file']}")

    if detail_lines:
        lines.append(" | ".join(detail_lines))
        lines.append("")

    lines.append("---")
    lines.append("")

    # Group consecutive segments by speaker
    segments = result.get("segments") or []
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



def _rename_speakers(result):
    """Rename diarization speaker IDs (SPEAKER_00 → Speaker 1)."""
    segments = result.get("segments") or []
    speaker_ids = sorted(set(seg.get("speaker", "Unknown") for seg in segments))
    name_map = {}
    for i, sid in enumerate(speaker_ids):
        name_map[sid] = f"Speaker {i + 1}"
    if len(name_map) > 1:
        print(f"  🔊 {len(name_map)} speakers detected")
    _log("speakers", count=len(name_map), labels=list(name_map.values()))
    return name_map


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


def get_current_event_title():
    """Return the title of the calendar event happening right now, or None."""
    try:
        events = get_today_events(config.get("watch", {}).get("calendars", []))
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
    record_only = watch_cfg.get("record_only", True)

    _watch_log("📅 Calendar Watch started")
    from ui import info_panel
    mode_label = "record only → transcribe after" if record_only else "live (record + transcribe)"
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

    while True:
        # ── Read today's events ────────────────────────────────
        events = get_today_events(calendars if calendars else None)
        now = datetime.now()
        last_refresh = now

        # Filter to future events only
        upcoming = [e for e in events if e["end"] > now]

        _watch_log(f"📅 {len(upcoming)} meetings remaining today")
        if upcoming:
            for e in upcoming:
                start_str = e["start"].strftime("%H:%M")
                end_str = e["end"].strftime("%H:%M")
                duration = int((e["end"] - e["start"]).total_seconds() / 60)
                _watch_log(f"   {start_str}-{end_str}  {e['title']}  ({duration}min)")

        # ── Process events or sleep ────────────────────────────
        if not upcoming:
            # Sleep until midnight or next refresh, whichever is sooner
            tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow += timedelta(days=1)
            next_refresh = now + timedelta(hours=refresh_hours)
            wake_at = min(tomorrow, next_refresh)
            wait = (wake_at - datetime.now()).total_seconds()

            if wake_at == tomorrow:
                _watch_log(f"💤 No meetings — sleeping until midnight")
            else:
                _watch_log(f"💤 No meetings — refreshing in {refresh_hours}h")
            try:
                time.sleep(max(1, wait))
            except KeyboardInterrupt:
                _watch_log("👋 Watch stopped by user")
                return
            continue

        next_event = upcoming[0]
        wait_seconds = (next_event["start"] - datetime.now()).total_seconds()

        if wait_seconds > 0:
            # Sleep until meeting starts, but wake for refresh if needed
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
                return

            # If we woke up for refresh (not for the meeting), loop back
            if sleep_time < wait_seconds:
                _watch_log("🔄 Refreshing calendar...")
                continue

        # ── Meeting time — start recording ─────────────────────
        _watch_log(f"⏺  \"{next_event['title']}\" — auto-recording started")

        stop_time = next_event["end"] + timedelta(minutes=end_buffer)

        # Start recording as a subprocess
        safe_title = next_event["title"]
        if record_only:
            # Record only — no transcription during meeting
            proc = subprocess.Popen(
                [sys.executable, __file__, "rec"],
                stdin=subprocess.DEVNULL,
            )
        else:
            # Live mode — record + transcribe simultaneously
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
                    _watch_log(f"⏹  Recording ended on its own (silence timeout)")
                    break
                time.sleep(10)

            # If still running after meeting end + buffer, stop it
            if proc.poll() is None:
                _watch_log(f"⏹  \"{next_event['title']}\" ended + {end_buffer}min buffer — stopping")
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=300)

        except KeyboardInterrupt:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=300)
            _watch_log("👋 Watch stopped by user")
            return

        # ── Check if recording was too short ───────────────────
        if proc.returncode == 0:
            recordings = list_recordings(RECORDINGS_DIR)
            if recordings:
                latest = recordings[0]
                duration = get_audio_duration(str(latest))
                speech_minutes = duration / 60

                if speech_minutes < min_recording:
                    _watch_log(f"🗑  \"{next_event['title']}\" too short ({format_duration(duration)}) — discarded")
                    try:
                        latest.unlink()
                    except OSError:
                        pass
                else:
                    _watch_log(f"✅ \"{next_event['title']}\" recorded ({format_duration(duration)})")

                    # Record-only mode: transcribe after recording
                    if record_only:
                        _watch_log(f"⏳ Transcribing \"{next_event['title']}\"...")
                        try:
                            subprocess.run(
                                [sys.executable, __file__,
                                 "run", str(latest),
                                 "--model", model_name,
                                 "--title", safe_title,
                                 "--no-diarize"],
                                stdin=subprocess.DEVNULL,
                                timeout=3600,
                            )
                            _watch_log(f"✅ \"{next_event['title']}\" transcribed")
                        except subprocess.TimeoutExpired:
                            _watch_log(f"⚠  Transcription timed out (1h limit)")
                        except Exception as e:
                            _watch_log(f"⚠  Transcription failed: {e}")

        time.sleep(5)


def cmd_install_daemon(args):
    """Install launchd agent so watch runs automatically on login."""
    script_path = Path(__file__).resolve()
    venv_python = SCRIPT_DIR / ".venv" / "bin" / "python3"

    if not venv_python.exists():
        print("  ❌ Virtual environment not found. Run install.sh first.")
        return

    if not HF_TOKEN:
        print("  ⚠  HF_TOKEN not set — diarization won't work in background daemon.")
        print("     Set it with: export HF_TOKEN=your_token_here")
        print("     Then re-run: transcribe install-daemon")
        print()

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
    run_parser.add_argument("--model", "-m", default=None, help="Model: parakeet/small.en/medium/turbo/large-v3")
    run_parser.add_argument("--title", "-t", default=None, help="Transcript title")
    run_parser.add_argument("--language", "-l", default=None, help="Language code (default: from config)")
    run_parser.add_argument("--no-diarize", action="store_true", help="Skip speaker identification (faster)")
    run_parser.add_argument("--denoise", action="store_true", help="Enable denoising (off by default)")
    run_parser.add_argument("--no-denoise", action="store_true", help="Force denoising OFF (overrides config)")
    run_parser.add_argument("--debug", action="store_true", help="Enable pipeline debug logging")

    # live
    live_parser = subparsers.add_parser("live", help="Record + transcribe simultaneously")
    live_parser.add_argument("--model", "-m", default=None, help="Model (default: from config)")
    live_parser.add_argument("--title", "-t", default=None, help="Transcript title")
    live_parser.add_argument("--language", "-l", default=None, help="Language code (default: from config)")
    live_parser.add_argument("--no-diarize", action="store_true", help="Skip speaker identification (faster)")

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
    subparsers.add_parser("history", help="Show transcription run history and model benchmarks")

    # setup
    subparsers.add_parser("setup", help="Set up audio devices for recording")

    args = parser.parse_args()

    if args.command == "rec":
        cmd_record(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "live":
        cmd_live(args)
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
    elif args.command == "history":
        cmd_history(args)
    elif args.command == "setup":
        cmd_setup(args)
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
