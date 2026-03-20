#!/bin/bash
# Transcriber — Uninstall Script
#
# Removes:
#   - Watch daemon (launchd agent)
#   - Speaker voice profiles
#   - Log files
#   - Virtual environment (.venv)
#   - Cached models (~/.cache/huggingface, torch)
#   - Shell alias from .zshrc
#   - Config file
#   - Optionally: recordings and transcripts from Obsidian

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHELL_RC="$HOME/.zshrc"
# Read output paths from config.json if it exists, otherwise use defaults
DEFAULT_BASE="$HOME/Library/Mobile Documents/iCloud~md~obsidian/Documents/Interviews"
if [ -f "$SCRIPT_DIR/config.json" ] && command -v python3 &>/dev/null; then
    CONFIG_BASE=$(python3 -c "
import json, os
with open('$SCRIPT_DIR/config.json') as f:
    c = json.load(f)
base = c.get('paths', {}).get('obsidian_base', '$DEFAULT_BASE')
print(os.path.expanduser(base))
" 2>/dev/null) || CONFIG_BASE="$DEFAULT_BASE"
else
    CONFIG_BASE="$DEFAULT_BASE"
fi
REC_SUBDIR=$(python3 -c "
import json
try:
    with open('$SCRIPT_DIR/config.json') as f:
        print(json.load(f).get('paths', {}).get('recordings_subdir', 'Recordings'))
except: print('Recordings')
" 2>/dev/null) || REC_SUBDIR="Recordings"
SCRIPTS_SUBDIR=$(python3 -c "
import json
try:
    with open('$SCRIPT_DIR/config.json') as f:
        print(json.load(f).get('paths', {}).get('scripts_subdir', 'Scripts'))
except: print('Scripts')
" 2>/dev/null) || SCRIPTS_SUBDIR="Scripts"
OBSIDIAN_RECORDINGS="$CONFIG_BASE/$REC_SUBDIR"
OBSIDIAN_SCRIPTS="$CONFIG_BASE/$SCRIPTS_SUBDIR"
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

# ─── Remove config file ──────────────────────────────────────────────────────

if [ -f "$SCRIPT_DIR/config.json" ]; then
    rm -f "$SCRIPT_DIR/config.json"
    echo "  ✅ Removed config.json"
else
    echo "  ─  No config.json found"
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
    # Transcription, diarization, and embedding models
    if [ -d "$HOME/.cache/huggingface/hub" ]; then
        find "$HOME/.cache/huggingface/hub" -maxdepth 1 -type d \( \
            -name "*whisper*" -o \
            -name "*parakeet*" -o \
            -name "*sortformer*" -o \
            -name "*pyannote*" -o \
            -name "*wav2vec2*" -o \
            -name "*embedding*" \
        \) -exec rm -rf {} + 2>/dev/null
        echo "  ✅ Removed cached transcription/diarization models"
    fi

    # CTranslate2 whisper models
    if ls "$HOME/.cache/huggingface/hub/models--Systran--faster-whisper-"* 1>/dev/null 2>&1; then
        rm -rf "$HOME/.cache/huggingface/hub/models--Systran--faster-whisper-"*
        echo "  ✅ Removed cached faster-whisper models"
    fi
else
    echo "  ─  Keeping cached models"
fi

# ─── Remove alias from .zshrc ────────────────────────────────────────────────

echo ""
if grep -q "alias transcribe=" "$SHELL_RC" 2>/dev/null; then
    sed -i '' '/# Transcriber$/d' "$SHELL_RC"
    sed -i '' '/# Interview Transcriber/d' "$SHELL_RC"
    sed -i '' '/alias transcribe=/d' "$SHELL_RC"
    echo "  ✅ Removed 'transcribe' alias from $SHELL_RC"
else
    echo "  ─  No alias found in $SHELL_RC"
fi

# ─── Remove recordings and transcripts ───────────────────────────────────────

echo ""
if [ -d "$OBSIDIAN_RECORDINGS" ] || [ -d "$OBSIDIAN_SCRIPTS" ]; then
    read -p "  Remove recordings and transcripts from Obsidian? [y/N] " remove_obsidian
    if [ "$remove_obsidian" = "y" ] || [ "$remove_obsidian" = "Y" ]; then
        [ -d "$OBSIDIAN_RECORDINGS" ] && rm -rf "$OBSIDIAN_RECORDINGS" && echo "  ✅ Removed $OBSIDIAN_RECORDINGS"
        [ -d "$OBSIDIAN_SCRIPTS" ] && rm -rf "$OBSIDIAN_SCRIPTS" && echo "  ✅ Removed $OBSIDIAN_SCRIPTS"
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
echo "│    rm -rf $SCRIPT_DIR"
echo "│                                             │"
echo "└─────────────────────────────────────────────┘"
echo ""
