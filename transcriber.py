#!/usr/bin/env python3
"""
Transcriber — Record, transcribe, and save as Markdown.

Commands:
    transcribe rec                  Start recording (auto-stops on silence)
    transcribe run [file]           Transcribe a recording
    transcribe live                 Record + transcribe simultaneously
    transcribe watch                Auto-record meetings from calendar
    transcribe list                 List available recordings
    transcribe enroll               Save your voice for auto-recognition
    transcribe setup                Audio device setup guide
"""

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # suppress tqdm "Fetching N files" bars

import argparse
import gc
import json
import os
import re
import select
import subprocess
import sys
import tempfile
import termios
import threading
import time
import tty
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from system_audio import SystemAudioCapture as _SystemAudioCapture, is_available as _sck_is_available
    _SCK_AVAILABLE = _sck_is_available()
except ImportError:
    _SystemAudioCapture = None
    _SCK_AVAILABLE = False

# Suppress torchcodec/pyannote/torchaudio deprecation warnings
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", message=".*torchaudio.*deprecated.*")
warnings.filterwarnings("ignore", message=".*speechbrain.*")


# ─── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"

DEFAULT_CONFIG = {
    # ── General ──────────────────────────────────────────────────────────
    # Engine: "mlx" (Apple Silicon GPU — fast) or "whisperx" (CPU — compatible)
    # mlx uses Apple's Metal GPU via MLX framework — ~7x faster on Apple Silicon
    # whisperx uses faster-whisper on CPU — works everywhere, has built-in alignment
    "engine": "mlx",
    # Transcription model: "parakeet" (recommended) or Whisper variants
    # Parakeet: CTC model — no hallucinations, proper punctuation, 2.5GB
    # Whisper: tiny/base/small/medium/turbo/large-v3 (.en variants for English)
    # Recommended: "parakeet" for accuracy, "small.en" for low RAM
    "default_model": "parakeet",
    # Language code for transcription (e.g. "en", "ar", "fr")
    # Set to avoid misdetection — auto-detect can pick wrong language
    "language": "en",
    # Quantization: "int8" recommended for Apple Silicon (fastest, low RAM)
    # Options: "int8", "float16", "float32" — only used by whisperx engine
    "compute_type": "int8",
    # Processing device: "cpu" for Apple Silicon (MPS not yet supported by WhisperX)
    # Only used by whisperx engine — mlx always uses Apple GPU
    "device": "cpu",

    # ── Batch sizes per model ────────────────────────────────────────────
    # Higher = faster but more RAM. These are tuned for M1 16GB.
    # Reduce if you get memory errors; increase on machines with more RAM.
    "batch_sizes": {
        "tiny": 32, "tiny.en": 32, "base": 24, "base.en": 24,
        "small": 16, "small.en": 16, "medium": 8, "medium.en": 8,
        "turbo": 4, "large-v3": 4, "large-v3.en": 4
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
        # Engine: "mlx-audio" (fast, Apple GPU, max 4 speakers)
        #         "pyannote" (slow CPU, unlimited speakers)
        "engine": "mlx-audio",
        # Pyannote model — used when engine is "pyannote"
        "model": "pyannote/speaker-diarization-3.1",
        # MLX Sortformer model — used when engine is "mlx-audio"
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
        # Record only during meeting, transcribe after it ends.
        # Uses less CPU/GPU during the call and produces better results
        # since the full model runs on the complete audio.
        "record_only": True,
    },

    # ── Audio denoising (Whisper only — Parakeet skips this) ────────────
    "denoise": {
        # Spectral subtraction to reduce background noise before Whisper transcription.
        # Parakeet doesn't need this (CTC architecture can't hallucinate on silence).
        # Uses first N seconds as noise profile — works best when audio starts
        # with silence or hold music (before speech begins).
        "enabled": True,
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

    # Your name — set via 'transcribe enroll' or manually here
    # Used as the display name when your voice is recognized
    "user_name": "",

    # ── Model definitions ────────────────────────────────────────────────
    # speed_factor: multiplier for time estimation (audio_duration × factor)
    # Lower = faster model. These are calibrated for M1 Apple Silicon.
    "models": {
        "tiny":     {"speed_factor": 0.1,  "description": "fastest, least accurate",       "size": "~75MB"},
        "tiny.en":  {"speed_factor": 0.1,  "description": "fastest, English-only",        "size": "~75MB"},
        "base":     {"speed_factor": 0.2,  "description": "fast, good for clear audio",   "size": "~140MB"},
        "base.en":  {"speed_factor": 0.2,  "description": "fast, English-only",           "size": "~140MB"},
        "small":    {"speed_factor": 0.5,  "description": "balanced, multilingual",       "size": "~460MB"},
        "small.en": {"speed_factor": 0.5,  "description": "balanced, English-only (recommended)", "size": "~460MB"},
        "medium":   {"speed_factor": 1.5,  "description": "slow, multilingual",           "size": "~1.5GB"},
        "medium.en":{"speed_factor": 1.5,  "description": "slow, English-only",           "size": "~1.5GB"},
        "turbo":    {"speed_factor": 0.8,  "description": "fast + accurate (best value)", "size": "~1.6GB"},
        "large-v3": {"speed_factor": 4.0,  "description": "slowest, most accurate",      "size": "~3GB"},
        "large-v3.en": {"speed_factor": 4.0, "description": "most accurate, English-only", "size": "~3GB"},
        "parakeet":     {"speed_factor": 0.5, "description": "CTC — no hallucinations, best accuracy", "size": "~2.5GB"},
    },

    # ── MLX model repos (HuggingFace) ────────────────────────────────
    # Maps model names to mlx-community HF repos (auto-downloaded on first use)
    "mlx_models": {
        "tiny":      "mlx-community/whisper-tiny-mlx",
        "tiny.en":   "mlx-community/whisper-tiny.en-mlx",
        "base":      "mlx-community/whisper-base-mlx",
        "base.en":   "mlx-community/whisper-base.en-mlx",
        "small":     "mlx-community/whisper-small-mlx",
        "small.en":  "mlx-community/whisper-small.en-mlx",
        "medium":    "mlx-community/whisper-medium-mlx",
        "medium.en": "mlx-community/whisper-medium.en-mlx",
        "turbo":      "mlx-community/whisper-large-v3-turbo",
        "large-v3":   "mlx-community/whisper-large-v3-mlx",
        "large-v3.en":"mlx-community/whisper-large-v3-mlx",
        "parakeet":   "mlx-community/parakeet-tdt-0.6b-v3",
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
VIRTUAL_DEVICE_NAMES = ("BlackHole", "Aggregate", "Multi-Output")
MIC_LOW_WARNING_SECONDS = config["recording"]["mic_low_warning_seconds"]
DEVICE = config.get("device", "cpu")
ENGINE = config.get("engine", "mlx")
SPEAKERS_DIR = SCRIPT_DIR / "speakers"

MODEL_INFO = config["models"]
STEP_NAMES = ["Loading model", "Transcribing", "Aligning timestamps", "Identifying speakers", "Saving"]


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

    if not _SCK_AVAILABLE:
        print("  ⚠  BlackHole not installed — recording from mic only.")
        print("     Zoom/Teams audio won't be captured.")
        print("     Install with: brew install blackhole-2ch && reboot")
        print()
    return None, mic_id, mic_name


# ─── Audio Routing & Volume ──────────────────────────────────────────────────

def find_multi_output_device():
    """Find a Multi-Output Device in the output device list (for BlackHole routing)."""
    import sounddevice as sd
    for d in sd.query_devices():
        if d["max_output_channels"] > 0 and "Multi-Output" in d["name"]:
            return d["name"]
    return None


def get_current_output_device():
    """Get current system audio output device name via SwitchAudioSource."""
    result = subprocess.run(
        ["SwitchAudioSource", "-c", "-t", "output"],
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def set_output_device(device_name):
    """Set system audio output device by name via SwitchAudioSource."""
    subprocess.run(
        ["SwitchAudioSource", "-s", device_name, "-t", "output"],
        capture_output=True, check=True,
    )


def get_system_volume():
    """Get current system output volume (0–100)."""
    result = subprocess.run(
        ["osascript", "-e", "get volume settings"],
        capture_output=True, text=True, check=True,
    )
    parts = result.stdout.strip().split(",")[0]
    return int(parts.split(":")[1].strip())


def set_system_volume(percent):
    """Set system output volume (0–100). Works even when Multi-Output Device is active."""
    percent = max(0, min(100, int(percent)))
    subprocess.run(
        ["osascript", "-e", f"set volume output volume {percent}"],
        check=True,
    )


def _route_to_multi_output():
    """Switch system output to Multi-Output Device for BlackHole capture.

    Returns the previous device name so it can be restored, or None if no
    switch was made (already on Multi-Output, or SwitchAudioSource not installed).
    """
    target = find_multi_output_device()
    if not target:
        return None
    try:
        current = get_current_output_device()
        if current and current != target:
            set_output_device(target)
            return current
    except FileNotFoundError:
        pass  # SwitchAudioSource not installed — silent skip
    except Exception:
        pass
    return None


def _restore_output(previous):
    """Restore system output to a previously saved device name. No-op if None."""
    if previous:
        try:
            set_output_device(previous)
        except Exception:
            pass


def cmd_setup(args):
    """Check audio setup status."""
    import sounddevice as sd

    bh_id, _ = find_blackhole()
    mic_id, mic_name = get_default_mic()

    from ui import info_panel
    bh_status = "✅ System audio capture" if bh_id is not None else "❌ Not installed (brew install blackhole-2ch)"
    print()
    info_panel("🔧 Audio Setup", [
        ("BlackHole:", bh_status),
        ("Mic:", mic_name),
    ])

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

    _, mic_id, mic_name = find_audio_device()
    dual_mode = _SystemAudioCapture is not None
    source_label = f"System + {mic_name}" if dual_mode else mic_name

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = RECORDINGS_DIR / f"{timestamp}.wav"

    from ui import info_panel
    print()
    info_panel("🎙 Recording", [
        ("Source:", source_label),
        ("Silence:", f"Auto-stops after {SILENCE_TIMEOUT // 60}min"),
        ("Controls:", "Ctrl+C to stop"),
    ])
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
    sck_capture = None
    if _SystemAudioCapture is not None:
        try:
            sck_capture = _SystemAudioCapture(sample_rate=SAMPLE_RATE, callback=bh_callback)
        except RuntimeError as e:
            print(f"  ⚠  System audio unavailable: {e}")
            dual_mode = False

    from ui import RecordingDisplay

    auto_stopped = False
    display = RecordingDisplay()
    try:
        mic_stream.start()
        if sck_capture:
            sck_capture.start()
        mic_low_since_rec = None
        with display:
            while recording:
                with lock:
                    rms = current_rms
                    sil = silence_start

                elapsed = time.time() - start_time
                silence_elapsed = time.time() - sil if sil else 0

                # Mic low warning
                if rms < 0.001:
                    if mic_low_since_rec is None:
                        mic_low_since_rec = time.time()
                    elif time.time() - mic_low_since_rec > MIC_LOW_WARNING_SECONDS:
                        display.set_warning("Mic very low!")
                else:
                    mic_low_since_rec = None
                    display.set_warning(None)

                display.update(rms, elapsed, silence_elapsed)
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

    audio_data = _mix_audio(mic_frames, bh_frames, dual_mode)
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
        speakers = input("  Speaker names [optional, comma-separated — Enter to skip]: ").strip()
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

    use_mlx = ENGINE == "mlx"

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    model_name = args.model or config["live"].get("model", config["default_model"])
    language = args.language or config["language"]
    batch_size = config["batch_sizes"].get(model_name, 8)
    chunk_interval = config["live"]["chunk_interval_seconds"]
    no_diarize = getattr(args, "no_diarize", False)
    speakers_str = getattr(args, "speakers", None)
    title = getattr(args, "title", None)

    engine_label = "mlx (GPU)" if use_mlx else "whisperx (CPU)"

    # ── Pre-load whisper model ────────────────────────────────────────────
    from ui import console
    from rich.panel import Panel
    print()
    console.print(Panel("[bold]🎙 Live Record + Transcribe[/bold]", border_style="dim", expand=False, width=54))
    print()
    print(f"  Engine: {engine_label}")
    print(f"  Loading '{model_name}' model...")
    if use_mlx:
        import mlx_whisper
        # MLX loads model on first transcribe() call — just verify repo exists
        mlx_repo = _get_mlx_repo(model_name)
        model = None  # MLX doesn't pre-load
    else:
        import whisperx
        model = whisperx.load_model(
            model_name, DEVICE, compute_type=config["compute_type"], language=language
        )
    print(f"  ✅ Model ready — starting recording")

    # ── Find audio devices ────────────────────────────────────────────────
    _, mic_id, mic_name = find_audio_device()
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
                if use_mlx:
                    result = _transcribe_mlx_chunk(chunk, mlx_repo, language)
                else:
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
    sck_capture = None
    if _SystemAudioCapture is not None:
        try:
            sck_capture = _SystemAudioCapture(sample_rate=SAMPLE_RATE, callback=bh_callback)
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
    audio_data = _mix_audio(mic_frames, bh_frames, dual_mode)
    saved_to = _safe_write_audio(output_path, audio_data, SAMPLE_RATE, description="recording")
    if saved_to:
        output_path = saved_to
    print(f"  📁 Saved: {output_path.name}")

    # ── Transcribe remaining audio ────────────────────────────────────────
    flat_audio = audio_data.flatten().astype(np.float32)
    total_samples = len(flat_audio)
    remaining = total_samples - last_transcribed_sample

    if adaptive["rec_only"] and last_transcribed_sample == 0:
        # Never managed to transcribe anything — do full file transcription
        remaining_duration = total_samples / SAMPLE_RATE
        print(f"  ⏳ Transcribing full recording ({format_duration(remaining_duration)})...")
        try:
            if use_mlx:
                result = _transcribe_mlx(str(output_path), model_name, language)
            else:
                result = model.transcribe(flat_audio, batch_size=adaptive["original_batch_size"])
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
            if use_mlx:
                result = _transcribe_mlx_chunk(chunk, mlx_repo, language)
            else:
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
    if model is not None:
        del model
    gc.collect()

    all_segments = sorted(transcribed_segments, key=lambda s: s.get("start", 0))
    if not all_segments:
        print("  ⚠  No speech detected.")
        return

    result = {"segments": all_segments, "language": language}

    # ── Alignment (whisperx only — mlx has word timestamps built in) ─────
    if not use_mlx:
        print("  ⏳ Aligning timestamps...")
        try:
            import whisperx
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

    # ── Diarization (always on unless --no-diarize) ─────────────────────
    diarize_enabled = config["diarization"]["enabled"] and not no_diarize
    if diarize_enabled and not HF_TOKEN:
        print("  ⚠  Skipping speaker ID — HF_TOKEN not set (see README)")
    if HF_TOKEN and diarize_enabled:
        print("  ⏳ Identifying speakers...")
        try:
            if use_mlx:
                speaker_turns = _diarize_standalone(str(output_path))
                result["segments"] = _assign_speakers_to_segments(result.get("segments", []), speaker_turns)
            else:
                import whisperx
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
    # Checkpoint: save raw result immediately so downstream failures don't lose it
    cp_path = _save_checkpoint(result, str(output_path), title)

    try:
        speaker_names = resolve_speaker_names(result, str(output_path), speakers_str)

        if not title:
            title = Path(output_path).stem.replace("_", " ").replace("-", " ").title()

        post_time = time.time() - recording_end
        date_str = extract_recording_date(str(output_path))
        metadata = {
            "model": model_name,
            "engine": ENGINE,
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

        # Log run to history for benchmarking
        _log_run({
            "model": model_name,
            "engine": ENGINE,
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
        diar_engine = config["diarization"].get("engine", "mlx-audio")
        if diar_engine == "mlx-audio":
            self.step_weights = [0.05, 0.65, 0.10, 0.15, 0.05]
        else:
            self.step_weights = [0.05, 0.30, 0.05, 0.55, 0.05]
        speed_factor = MODEL_INFO.get(model_name, {}).get("speed_factor", 1.0)
        if ENGINE == "mlx":
            speed_factor *= 0.15
        diar_time = audio_duration * 0.05 if diar_engine == "mlx-audio" else audio_duration * 1.2
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

    import librosa
    import soundfile as sf

    factor = denoise_cfg.get("factor", 2.0)
    profile_secs = denoise_cfg.get("noise_profile_seconds", 10)
    sr = SAMPLE_RATE  # 16 kHz — what Whisper expects

    try:
        t0 = time.time()
        audio, _ = librosa.load(audio_path, sr=sr)

        noise_clip = audio[: sr * profile_secs]
        n_fft, hop = 2048, 512

        noise_stft = librosa.stft(noise_clip, n_fft=n_fft, hop_length=hop)
        noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1, keepdims=True)

        audio_stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
        audio_power = np.abs(audio_stft) ** 2

        clean_power = np.maximum(audio_power - factor * noise_power, 0.0)
        gain = np.sqrt(clean_power / (audio_power + 1e-10))
        clean_stft = audio_stft * gain

        denoised = librosa.istft(clean_stft, hop_length=hop, length=len(audio))
        denoised = denoised.astype(np.float32)

        # Write to temp file next to the original
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix=".denoise_")
        os.close(tmp_fd)
        sf.write(tmp_path, denoised, sr)

        elapsed = time.time() - t0
        return tmp_path, elapsed

    except Exception as e:
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

    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=repo,
        word_timestamps=word_timestamps,
        language=language,
        hallucination_silence_threshold=hst,
    )
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

    del model
    gc.collect()
    mx.clear_cache()  # Free GPU memory for next model
    return turns


def _diarize_pyannote(audio_path):
    """Run pyannote speaker diarization (slower CPU, unlimited speakers).

    Pre-loads audio with torchaudio to bypass broken torchcodec in pyannote 4.x.
    """
    import torch
    import torchaudio
    from pyannote.audio import Pipeline as PyannotePipeline

    pipeline = PyannotePipeline.from_pretrained(
        config["diarization"]["model"],
        token=HF_TOKEN,
    )

    # Load audio ourselves — avoids pyannote's broken torchcodec path
    waveform, sample_rate = torchaudio.load(audio_path)
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    diarization = pipeline(audio_input)

    # pyannote 4.x wraps result in DiarizeOutput — unwrap to get Annotation
    if hasattr(diarization, "speaker_diarization"):
        diarization = diarization.speaker_diarization

    # Build speaker turn list
    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})

    del pipeline
    gc.collect()
    return turns


def _diarize_standalone(audio_path):
    """Route to the configured diarization engine."""
    engine = config["diarization"].get("engine", "mlx-audio")
    if engine == "mlx-audio":
        return _diarize_mlx_audio(audio_path)
    else:
        return _diarize_pyannote(audio_path)


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
    no_denoise = getattr(args, "no_denoise", False)
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
    if ENGINE == "mlx":
        speed_factor *= 0.15  # MLX is ~7x faster than CPU
    estimated_time = audio_duration * speed_factor
    # Add diarization time estimate
    diar_engine = config["diarization"].get("engine", "mlx-audio")
    diarize_enabled = config["diarization"]["enabled"] and not no_diarize
    if diarize_enabled:
        if diar_engine == "mlx-audio":
            estimated_time += audio_duration * 0.05  # ~3s per minute
        else:
            estimated_time += audio_duration * 1.2  # pyannote is ~1.2x audio length on CPU

    # Parakeet doesn't need denoising (CTC can't hallucinate on silence)
    is_parakeet_model = _is_parakeet(model_name)
    denoise_enabled = config.get("denoise", {}).get("enabled", True) and not no_denoise and not is_parakeet_model
    denoise_factor = config.get("denoise", {}).get("factor", 2.0)

    engine_label = "parakeet (GPU)" if is_parakeet_model else ("mlx (GPU)" if ENGINE == "mlx" else "whisperx (CPU)")
    model_size = MODEL_INFO.get(model_name, {}).get("size", "?")
    model_desc = MODEL_INFO.get(model_name, {}).get("description", "")
    diar_label = "off"
    if diarize_enabled:
        diar_label = f"{diar_engine} (GPU)" if diar_engine == "mlx-audio" else f"{diar_engine} (CPU)"
    denoise_label = f"spectral sub {denoise_factor}x" if denoise_enabled else "off"

    # Determine pipeline steps for display
    denoise_prefix = "Denoise → " if denoise_enabled else ""
    if ENGINE == "mlx":
        if diarize_enabled:
            steps_preview = f"{denoise_prefix}Diarize → Transcribe → Save"
        else:
            steps_preview = f"{denoise_prefix}Load model → Transcribe → Save"
    else:
        if diarize_enabled:
            steps_preview = f"{denoise_prefix}Load → Transcribe → Align → Diarize → Save"
        else:
            steps_preview = f"{denoise_prefix}Load → Transcribe → Align → Save"

    from ui import info_panel
    rows = [
        ("File:", Path(audio_path).name),
        ("Duration:", format_duration(audio_duration)),
        ("Engine:", engine_label),
        ("Model:", f"{model_name} ({model_size})"),
    ]
    if ENGINE != "mlx":
        rows.append(("Batch:", str(batch_size)))
    rows += [
        ("Language:", language),
        ("Denoise:", denoise_label),
        ("Diarize:", diar_label),
        ("Pipeline:", steps_preview),
        ("Est time:", f"~{format_duration(estimated_time)}"),
        ("Started:", datetime.now().strftime("%H:%M:%S")),
    ]
    print()
    info_panel("📝 Transcription", rows)
    print()

    use_mlx = ENGINE == "mlx"
    diarize_enabled = config["diarization"]["enabled"] and not no_diarize and HF_TOKEN

    # Pipeline steps differ by engine and diarization:
    #   MLX skips alignment (word_timestamps built-in)
    #   MLX diarizes FIRST to avoid GPU memory contention
    PIPELINE_STEPS = {
        ("mlx", True):  (["Identifying speakers", "Transcribing", "Saving"],
                         [0.15, 0.80, 0.05]),
        ("mlx", False): (["Loading model", "Transcribing", "Saving"],
                         [0.05, 0.90, 0.05]),
        ("cpu", True):  (["Loading model", "Transcribing", "Aligning timestamps",
                          "Identifying speakers", "Saving"],
                         [0.05, 0.45, 0.10, 0.35, 0.05]),
        ("cpu", False): (["Loading model", "Transcribing", "Aligning timestamps", "Saving"],
                         [0.05, 0.70, 0.20, 0.05]),
    }
    engine_key = "mlx" if use_mlx else "cpu"
    step_names, step_weights = PIPELINE_STEPS[(engine_key, bool(diarize_enabled))]

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
    result = None
    try:
        step = 0

        # ── Denoise ──
        if denoise_enabled:
            progress.set_step(step)
            denoised_path, denoise_time = _denoise_audio(audio_path)
            transcribe_path = denoised_path
            step += 1
        else:
            transcribe_path = audio_path

        if use_mlx:
            import mlx.core as mx
            # ── MLX engine (Apple Silicon GPU) ──
            # Diarize FIRST with small Sortformer model (~100MB),
            # free GPU memory, THEN transcribe with large Whisper model.
            # Metal GPU crashes if two models run simultaneously.
            speaker_turns = None

            if diarize_enabled:
                progress.set_step(step)  # Identifying speakers
                # Diarize on original audio (denoising can distort speaker embeddings)
                speaker_turns = _diarize_standalone(audio_path)
                gc.collect()
                mx.clear_cache()  # Free Sortformer GPU memory before loading Whisper
                step += 1

            progress.set_step(step)  # Transcribing
            if _is_parakeet(model_name):
                result = _transcribe_parakeet(transcribe_path, model_name)
            else:
                # Skip word_timestamps when diarizing — we only need segment-level
                result = _transcribe_mlx(transcribe_path, model_name, language,
                                         word_timestamps=not diarize_enabled)

            if speaker_turns is not None:
                result["segments"] = _assign_speakers_to_segments(
                    result.get("segments", []), speaker_turns)

            step += 1
            progress.set_step(step)  # Saving
        else:
            # ── WhisperX engine (CPU) ──
            import whisperx

            progress.set_step(step)  # Loading model
            model = whisperx.load_model(
                model_name, DEVICE, compute_type=config["compute_type"], language=language
            )

            step += 1
            progress.set_step(step)  # Transcribing
            audio = whisperx.load_audio(transcribe_path)
            result = model.transcribe(audio, batch_size=batch_size)
            detected_language = result.get("language", language)
            del model
            gc.collect()

            step += 1
            progress.set_step(step)  # Aligning timestamps
            align_model, metadata = whisperx.load_align_model(
                language_code=detected_language, device=DEVICE
            )
            result = whisperx.align(
                result["segments"], align_model, metadata, audio, DEVICE,
                return_char_alignments=False,
            )
            del align_model
            gc.collect()

            if diarize_enabled:
                step += 1
                progress.set_step(step)  # Identifying speakers
                from whisperx.diarize import DiarizationPipeline
                diarize_model = DiarizationPipeline(
                    model_name=config["diarization"]["model"],
                    token=HF_TOKEN, device=DEVICE
                )
                diarize_segments = diarize_model(audio_path)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                del diarize_model
                gc.collect()

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

    progress.stop()

    # Checkpoint: save raw result immediately so downstream failures don't lose it
    cp_path = _save_checkpoint(result, audio_path, title)

    try:
        # Resolve speaker names (auto-match from saved profiles + -s flag)
        speaker_names = resolve_speaker_names(result, audio_path, speakers_str)

        # Save transcript
        total_time = time.time() - progress.start_time
        date_str = extract_recording_date(audio_path)
        metadata = {
            "model": model_name,
            "engine": ENGINE,
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

        # Log run to history for benchmarking
        step_log = {}
        if hasattr(progress, '_step_times'):
            for i, name in enumerate(step_names):
                if i in progress._step_times:
                    step_log[name] = progress._step_times[i]
        _log_run({
            "model": model_name,
            "engine": ENGINE,
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


# ─── Speaker Memory ──────────────────────────────────────────────────────────

def load_embedding_model():
    """Load pyannote embedding model for speaker recognition."""
    memory_cfg = config.get("speaker_memory", {})
    model_name = memory_cfg.get("embedding_model", "pyannote/embedding")
    try:
        from pyannote.audio import Inference
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
            with np.load(f, allow_pickle=True) as data:
                if "name" not in data or "embedding" not in data:
                    continue
                profiles[f.stem] = {
                    "name": str(data["name"]),
                    "embedding": np.array(data["embedding"]),
                }
        except Exception:
            continue
    return profiles


def save_speaker_profile(name, embedding):
    """Save a voice profile to speakers/ directory."""
    SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = SPEAKERS_DIR / f"{name.lower()}.npz"

    def _write(path):
        np.savez(path, name=name, embedding=embedding,
                 enrolled_at=datetime.now().strftime("%Y-%m-%d"))

    saved_to = _safe_write_binary(profile_path, _write, description="speaker profile")
    return saved_to or profile_path


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
            prof_emb = profile.get("embedding")
            if prof_emb is None:
                continue
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
    segments = result.get("segments") or []
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

    from ui import info_panel
    print()
    info_panel("🎤 Voice Enrollment", [
        ("Name:", user_name),
        ("Duration:", f"Speak naturally for {duration} seconds"),
        ("Tip:", "Read something aloud or describe your day"),
    ])
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

    from ui import success_panel
    print()
    success_panel("✅ Voice profile saved!", [
        ("Name:", user_name),
        ("Duration:", format_duration(actual_duration)),
        ("Profile:", f"speakers/{profile_path.name}"),
        ("Note:", "Your voice will be auto-recognized in future transcripts"),
    ])
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
    record_only = watch_cfg.get("record_only", True)

    _log("📅 Calendar Watch started")
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

                    # Record-only mode: transcribe after recording
                    if record_only:
                        _log(f"⏳ Transcribing \"{next_event['title']}\"...")
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
                            _log(f"✅ \"{next_event['title']}\" transcribed")
                        except subprocess.TimeoutExpired:
                            _log(f"⚠  Transcription timed out (1h limit)")
                        except Exception as e:
                            _log(f"⚠  Transcription failed: {e}")

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
    run_parser.add_argument("--model", "-m", default=None, help="Model: tiny/base/small/medium/large-v3")
    run_parser.add_argument("--speakers", "-s", default=None, help="Speaker names (comma-separated)")
    run_parser.add_argument("--title", "-t", default=None, help="Transcript title")
    run_parser.add_argument("--language", "-l", default=None, help="Language code (default: from config)")
    run_parser.add_argument("--no-diarize", action="store_true", help="Skip speaker identification (faster)")
    run_parser.add_argument("--no-denoise", action="store_true", help="Skip audio denoising")

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
    elif args.command == "history":
        cmd_history(args)
    elif args.command == "setup":
        cmd_setup(args)
    else:
        parser.print_help()
        print()
        print("  Quick start:")
        print("    transcribe enroll — Save your voice (one time)")
        print("    transcribe setup  — Set up audio (first time)")
        print("    transcribe rec    — Record audio")
        print("    transcribe run    — Transcribe a recording")
        print("    transcribe live   — Record + transcribe at once")
        print("    transcribe watch  — Auto-record calendar meetings")
        print("    transcribe list   — List recordings")
        print()


if __name__ == "__main__":
    main()
