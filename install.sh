#!/bin/bash
# Transcriber — Install Script
#
# Usage:
#   git clone https://github.com/FaisalFehad/local-transcribe.git
#   cd local-transcribe
#   ./install.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHELL_RC="$HOME/.zshrc"

echo ""
echo "┌─────────────────────────────────────────────┐"
echo "│  Transcriber — Installer                    │"
echo "└─────────────────────────────────────────────┘"
echo ""

# ─── Check prerequisites ─────────────────────────────────────────────────────

# Python 3.10+
PYTHON=""
for p in python3.12 python3.11 python3.10 python3; do
    if command -v "$p" &>/dev/null; then
        version=$("$p" -c "import sys; print(sys.version_info[:2])" 2>/dev/null)
        major=$("$p" -c "import sys; print(sys.version_info[0])")
        minor=$("$p" -c "import sys; print(sys.version_info[1])")
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON="$p"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "  ❌ Python 3.10+ required but not found."
    echo "     Install with: brew install python@3.12"
    exit 1
fi
echo "  ✅ Python: $($PYTHON --version)"

# ffmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "  ❌ ffmpeg required but not found."
    echo "     Install with: brew install ffmpeg"
    exit 1
fi
echo "  ✅ ffmpeg: $(ffmpeg -version 2>&1 | head -1)"

# ─── Create virtual environment ──────────────────────────────────────────────

echo ""
echo "  Setting up virtual environment..."
$PYTHON -m venv "$SCRIPT_DIR/.venv"
source "$SCRIPT_DIR/.venv/bin/activate"

echo "  Installing dependencies (this may take a few minutes)..."
pip install --upgrade pip -q
pip install -e "$SCRIPT_DIR" -q

echo "  ✅ Dependencies installed"

# ─── Make shell wrapper executable ────────────────────────────────────────────

chmod +x "$SCRIPT_DIR/transcribe"

# ─── Add alias to shell ──────────────────────────────────────────────────────

ALIAS_LINE="alias transcribe='$SCRIPT_DIR/transcribe'"

if grep -q "alias transcribe=" "$SHELL_RC" 2>/dev/null; then
    echo "  ✅ Shell alias already exists in $SHELL_RC"
else
    echo "" >> "$SHELL_RC"
    echo "# Transcriber" >> "$SHELL_RC"
    echo "$ALIAS_LINE" >> "$SHELL_RC"
    echo "  ✅ Added 'transcribe' alias to $SHELL_RC"
fi

# ─── Check HF_TOKEN ──────────────────────────────────────────────────────────

if grep -q "HF_TOKEN" "$SHELL_RC" 2>/dev/null; then
    echo "  ✅ HF_TOKEN found in $SHELL_RC"
else
    echo ""
    echo "  ⚠  HF_TOKEN not set (needed for speaker diarisation)"
    echo "     1. Get a free token at: https://huggingface.co/settings/tokens"
    echo "     2. Add to your shell: echo 'export HF_TOKEN=your_token' >> $SHELL_RC"
fi

# ─── System Audio (ScreenCaptureKit) ─────────────────────────────────────────

echo ""
echo "  ℹ  System audio capture uses ScreenCaptureKit (macOS 12.3+)"
echo "     On first recording, macOS will prompt for Screen Recording permission"
echo "     No virtual audio driver needed"

# ─── Done ─────────────────────────────────────────────────────────────────────

echo ""
echo "┌─────────────────────────────────────────────┐"
echo "│  ✅ Installation complete!                  │"
echo "├─────────────────────────────────────────────┤"
echo "│                                             │"
echo "│  Reload your shell:  source ~/.zshrc        │"
echo "│                                             │"
echo "│  Quick start:                               │"
echo "│    transcribe setup   — Audio setup guide   │"
echo "│    transcribe rec     — Record audio        │"
echo "│    transcribe run     — Transcribe          │"
echo "│    transcribe list    — Browse recordings   │"
echo "│                                             │"
echo "└─────────────────────────────────────────────┘"
echo ""
