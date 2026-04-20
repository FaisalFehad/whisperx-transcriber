#!/bin/bash
# Transcriber — Uninstall Script
#
# Removes:
#   - Watch daemon (launchd agent)
#   - Speaker voice profiles
#   - Log files
#   - Virtual environment (.venv)
#   - Cached models (~/.cache/huggingface)
#   - Shell alias from .zshrc
#   - Config file
#   - Optionally: recordings and transcripts

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHELL_RC="$HOME/.zshrc"
# Read output paths from config.json if it exists, otherwise use defaults.
# Path passed via argv (not interpolated) so SCRIPT_DIR cannot break out of the string.
DEFAULT_BASE="$HOME/Transcriptions"
read_cfg() {
    python3 -c '
import json, os, sys
cfg_path, key, default = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(cfg_path) as f:
        val = json.load(f).get("paths", {}).get(key, default)
except Exception:
    val = default
print(os.path.expanduser(val))
' "$1" "$2" "$3" 2>/dev/null
}
if [ -f "$SCRIPT_DIR/config.json" ] && command -v python3 &>/dev/null; then
    CONFIG_BASE=$(read_cfg "$SCRIPT_DIR/config.json" "base" "$DEFAULT_BASE") || CONFIG_BASE="$DEFAULT_BASE"
    [ -z "$CONFIG_BASE" ] && CONFIG_BASE="$DEFAULT_BASE"
    REC_SUBDIR=$(read_cfg "$SCRIPT_DIR/config.json" "recordings_subdir" "Recordings") || REC_SUBDIR="Recordings"
    [ -z "$REC_SUBDIR" ] && REC_SUBDIR="Recordings"
    SCRIPTS_SUBDIR=$(read_cfg "$SCRIPT_DIR/config.json" "scripts_subdir" "Scripts") || SCRIPTS_SUBDIR="Scripts"
    [ -z "$SCRIPTS_SUBDIR" ] && SCRIPTS_SUBDIR="Scripts"
else
    CONFIG_BASE="$DEFAULT_BASE"
    REC_SUBDIR="Recordings"
    SCRIPTS_SUBDIR="Scripts"
fi
RECORDINGS_DIR="$CONFIG_BASE/$REC_SUBDIR"
TRANSCRIPTS_DIR="$CONFIG_BASE/$SCRIPTS_SUBDIR"
LAUNCHD_PLIST="$HOME/Library/LaunchAgents/com.transcriber.watch.plist"

echo ""
echo "┌─────────────────────────────────────────────┐"
echo "│  Transcriber — Uninstaller                  │"
echo "└─────────────────────────────────────────────┘"
echo ""

# ─── Stop and remove watch daemon ─────────────────────────────────────────────

if [ -f "$LAUNCHD_PLIST" ]; then
    launchctl unload "$LAUNCHD_PLIST" 2>/dev/null || true
    rm -f "$LAUNCHD_PLIST"
    echo "  ✅ Stopped and removed watch daemon"
else
    echo "  ─  Watch daemon not installed"
fi

# ─── Remove speaker profiles ─────────────────────────────────────────────────

if [ -d "$SCRIPT_DIR/speakers" ]; then
    rm -rf "$SCRIPT_DIR/speakers"
    echo "  ✅ Removed speaker voice profiles"
else
    echo "  ─  No speaker profiles found"
fi

# ─── Remove log files ────────────────────────────────────────────────────────

removed_logs=false
for logfile in "$SCRIPT_DIR/watch.log" "$SCRIPT_DIR/watch-stdout.log" "$SCRIPT_DIR/watch-stderr.log"; do
    if [ -f "$logfile" ]; then
        rm -f "$logfile"
        removed_logs=true
    fi
done
if [ "$removed_logs" = true ]; then
    echo "  ✅ Removed log files"
else
    echo "  ─  No log files found"
fi

# ─── Remove config and history files ─────────────────────────────────────────

if [ -f "$SCRIPT_DIR/config.json" ]; then
    rm -f "$SCRIPT_DIR/config.json"
    echo "  ✅ Removed config.json"
else
    echo "  ─  No config.json found"
fi

if [ -f "$SCRIPT_DIR/history.jsonl" ]; then
    rm -f "$SCRIPT_DIR/history.jsonl"
    echo "  ✅ Removed history.jsonl"
fi

# ─── Remove virtual environment ──────────────────────────────────────────────

if [ -d "$SCRIPT_DIR/.venv" ]; then
    rm -rf "$SCRIPT_DIR/.venv"
    echo "  ✅ Removed .venv"
else
    echo "  ─  .venv not found (already removed)"
fi

# ─── Remove cached models ────────────────────────────────────────────────────

echo ""
read -p "  Remove cached AI models from ~/.cache? [y/N] " remove_cache
if [ "$remove_cache" = "y" ] || [ "$remove_cache" = "Y" ]; then
    # Transcription and diarisation models
    if [ -d "$HOME/.cache/huggingface/hub" ]; then
        find "$HOME/.cache/huggingface/hub" -maxdepth 1 -type d \( \
            -name "*whisper*" -o \
            -name "*parakeet*" -o \
            -name "*sortformer*" \
        \) -exec rm -rf {} + 2>/dev/null || true
        echo "  ✅ Removed cached transcription/diarisation models"
    fi

    # CTranslate2 whisper models
    if find "$HOME/.cache/huggingface/hub" -maxdepth 1 -type d -name "models--Systran--faster-whisper-*" 2>/dev/null | grep -q .; then
        find "$HOME/.cache/huggingface/hub" -maxdepth 1 -type d -name "models--Systran--faster-whisper-*" -exec rm -rf {} + 2>/dev/null || true
        echo "  ✅ Removed cached faster-whisper models"
    fi
else
    echo "  ─  Keeping cached models"
fi

# ─── Remove alias from .zshrc ────────────────────────────────────────────────

echo ""
if grep -q "alias transcribe=" "$SHELL_RC" 2>/dev/null; then
    sed -i '' '/# Transcriber$/d' "$SHELL_RC"
    sed -i '' '/alias transcribe=/d' "$SHELL_RC"
    echo "  ✅ Removed 'transcribe' alias from $SHELL_RC"
else
    echo "  ─  No alias found in $SHELL_RC"
fi

# ─── Remove recordings and transcripts ───────────────────────────────────────

echo ""
if [ -d "$RECORDINGS_DIR" ] || [ -d "$TRANSCRIPTS_DIR" ]; then
    [ -d "$RECORDINGS_DIR" ] && echo "  ·  $RECORDINGS_DIR"
    [ -d "$TRANSCRIPTS_DIR" ] && echo "  ·  $TRANSCRIPTS_DIR"
    read -p "  Remove the above recordings and transcripts? [y/N] " remove_files
    if [ "$remove_files" = "y" ] || [ "$remove_files" = "Y" ]; then
        [ -d "$RECORDINGS_DIR" ] && rm -rf "$RECORDINGS_DIR" && echo "  ✅ Removed $RECORDINGS_DIR"
        [ -d "$TRANSCRIPTS_DIR" ] && rm -rf "$TRANSCRIPTS_DIR" && echo "  ✅ Removed $TRANSCRIPTS_DIR"
    else
        echo "  ─  Keeping recordings and transcripts"
    fi
else
    echo "  ─  No recordings/transcripts found"
fi

# ─── Done ─────────────────────────────────────────────────────────────────────

echo ""
echo "┌─────────────────────────────────────────────┐"
echo "│  ✅ Uninstall complete                      │"
echo "├─────────────────────────────────────────────┤"
echo "│                                             │"
echo "│  Reload your shell:  source ~/.zshrc        │"
echo "│                                             │"
echo "│  To fully remove, delete this folder:       │"
echo "└─────────────────────────────────────────────┘"
echo "    rm -rf $SCRIPT_DIR"
echo ""
