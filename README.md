# Local Transcriber

Fast, private speech-to-text for macOS. Record meetings, transcribe with speaker labels — everything runs on your Mac's GPU. No cloud, no subscriptions, no data leaves your machine.

## Get Started in 60 Seconds

```bash
git clone https://github.com/FaisalFehad/local-transcribe.git
cd local-transcribe
./install.sh
source ~/.zshrc
```

That's it. Now transcribe anything:

```bash
transcribe rec                               # Record a meeting (Ctrl+C to stop)
transcribe run                               # Transcribe it → Markdown with speaker labels
```

Or transcribe an existing file:

```bash
transcribe run ~/Downloads/recording.mp3
```

**Requirements:** macOS with Apple Silicon, Python 3.10+, ffmpeg (`brew install ffmpeg`)

## How It Works

```
You speak → ScreenCaptureKit captures system + mic audio
         → Parakeet CTC model transcribes on GPU (~15x real-time)
         → Sortformer identifies who said what
         → Markdown file with timestamps and speaker labels
```

A 1-hour meeting transcribes in ~4 minutes. The transcript is ready before you finish your coffee.

## Features

- **Parakeet CTC engine** — default model, cannot hallucinate, proper punctuation out of the box
- **Whisper models** — 3 alternatives for multilingual or low-RAM setups (99 languages)
- **System audio + mic** — captures both sides of Zoom/Teams/Meet
- **Speaker diarisation** — identifies who said what (up to 4 speakers)
- **Live mode** — transcribe during the call, transcript ready when you hang up
- **Calendar watch** — auto-record meetings as a background daemon
- **Smart mic switching** — auto-detects when you connect/disconnect AirPods
- **Dual-channel meters** — live waveform + level bars for mic and system audio
- **Markdown output** — timestamps, speaker labels, YAML frontmatter
- **Fully private** — no cloud, no telemetry, no accounts

## Commands

| Command | What it does |
|---------|-------------|
| `transcribe rec` | Record system audio + mic (auto-stops on silence) |
| `transcribe run [file]` | Transcribe a recording or audio file |
| `transcribe live` | Record + transcribe simultaneously |
| `transcribe list` | Browse past recordings |
| `transcribe watch` | Auto-record meetings from your calendar |
| `transcribe setup` | Check audio device setup |

### Options

| Flag | Example | What it does |
|------|---------|-------------|
| `-m` | `-m medium` | Use a different model (default: parakeet) |
| `-t` | `-t "Team Meeting"` | Set the transcript title |
| `-l` | `-l ar` | Set language (default: en) |
| `--no-diarise` | | Skip speaker identification (faster) |

## Models

| Model | Speed | RAM | Best for |
|-------|-------|-----|----------|
| **parakeet** (default) | **~15x RT** | **~2.5GB** | **General use — best accuracy, no hallucinations** |
| small.en | ~15x RT | ~460MB | Low RAM, English-only |
| medium | ~7x RT | ~1.5GB | Multilingual (99 languages) |
| large-v3 | ~4x RT | ~3GB | Most capable Whisper (slow) |

**Why Parakeet?** CTC architecture maps audio directly to text — **cannot hallucinate**. No "Thank Thank Thank..." on silence. Proper punctuation and capitalisation out of the box.

## Example Output

```markdown
---
title: "Q1 Review"
date: 2026-03-19
speakers: [Alice, Bob]
model: parakeet
---

**[00:00] Alice:** Let's start with the Q1 numbers...
**[01:23] Bob:** Revenue was up 12% compared to last quarter...
```

## Setup

### Speaker Diarisation (optional)

To identify who said what, add a free [Hugging Face token](https://huggingface.co/settings/tokens):

```bash
echo 'export HF_TOKEN=your_token_here' >> ~/.zshrc
```

Without it, transcription still works — you just won't get speaker labels.

### System Audio

On first recording, macOS will prompt for **Screen Recording permission** — needed for capturing Zoom/Teams/Meet audio. No virtual audio drivers needed, volume keys work normally.

## Configuration

```bash
cp config.example.json config.json
```

Any value you omit uses the built-in default. See [`config.example.json`](config.example.json) for all options.

| Setting | Default | What it does |
|---------|---------|--------------|
| `default_model` | `"parakeet"` | Transcription model |
| `language` | `"en"` | Language code |
| `normalise` | `true` | Volume normalisation before transcription |
| `diarization.enabled` | `true` | Speaker identification |
| `paths.base` | `"~/Transcriptions"` | Where to save recordings and transcripts |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| No audio recorded | Run `transcribe setup` to check your mic |
| System audio not captured | Grant Screen Recording permission in System Settings |
| No speaker labels | Set `HF_TOKEN` (see Setup above) |

## Uninstall

```bash
./uninstall.sh
```

## License

[MIT](LICENSE)

## Acknowledgements

Built with [Parakeet TDT](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#parakeet-tdt) (NVIDIA NeMo), [Sortformer](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/configs.html) (NVIDIA NeMo), [mlx-audio](https://github.com/Blaizzy/mlx-audio), [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), and [ScreenCaptureKit](https://developer.apple.com/documentation/screencapturekit).
