# Interview Transcriber

Record interviews (in-person or Zoom/Teams), transcribe with speaker diarization, and save as Markdown to your Obsidian vault.

Built on [WhisperX](https://github.com/m-bain/whisperX) (OpenAI Whisper + pyannote speaker diarization). Runs locally on Apple Silicon — no cloud APIs needed.

## Features

- **Record** system audio + mic simultaneously (Zoom, Teams, Meet, etc.)
- **Transcribe** with word-level timestamps and speaker labels
- **Live mode** — transcribe during the call so results are ready when you hang up
- **Speaker memory** — enroll your voice once, get auto-recognized in every transcript
- **Calendar watch** — auto-record meetings from macOS Calendar (runs as background daemon)
- **Adaptive scaling** — live mode auto-adjusts chunk size based on CPU load
- **Silence detection** — auto-stop recording when the meeting ends
- **Obsidian-ready** — saves Markdown with YAML frontmatter, timestamps, and speaker labels

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ffmpeg (`brew install ffmpeg`)
- [BlackHole](https://github.com/ExistentialAudio/BlackHole) for system audio capture (Zoom/Teams)

## Install

```bash
git clone https://github.com/faisalfs/whisperx-transcriber.git
cd whisperx-transcriber
./install.sh
source ~/.zshrc
```

The installer will:
- Create a Python virtual environment
- Install WhisperX and dependencies
- Add the `transcribe` alias to your shell
- Check for BlackHole and HF_TOKEN setup

### Hugging Face Token (for speaker diarization)

Speaker identification requires a free Hugging Face token:

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Accept terms for these models:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [pyannote/embedding](https://huggingface.co/pyannote/embedding) (for voice recognition)
3. Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Add to your shell:
   ```bash
   echo 'export HF_TOKEN=your_token_here' >> ~/.zshrc
   ```

### Audio Setup (for Zoom/Teams recording)

To capture system audio from video calls, you need BlackHole:

```bash
brew install blackhole-2ch
# Reboot your Mac
transcribe setup    # Follow the guided setup
```

The setup guide walks you through creating a Multi-Output Device in Audio MIDI Setup so both you and BlackHole hear the call audio.

## Usage

### Quick Start

```bash
transcribe rec                    # Record (system audio + mic)
transcribe run                    # Transcribe last recording
transcribe live                   # Record + transcribe simultaneously
```

### All Commands

| Command | Description |
|---------|-------------|
| `transcribe rec` | Record audio (auto-stops after silence) |
| `transcribe run [file]` | Transcribe a recording |
| `transcribe live` | Record + transcribe at the same time |
| `transcribe list` | Browse past recordings |
| `transcribe enroll` | Save your voice for auto-recognition |
| `transcribe watch` | Auto-record meetings from calendar |
| `transcribe install-daemon` | Run calendar watch as background service |
| `transcribe uninstall-daemon` | Remove background service |
| `transcribe watch-status` | Check if watch daemon is running |
| `transcribe setup` | Audio device setup guide |

### Transcription Options

```bash
transcribe run file.mp3 -m medium         # Choose model (default: small)
transcribe run file.mp3 -s "Alice,Bob"    # Name the speakers
transcribe run file.mp3 -t "Interview"    # Set title
transcribe run file.mp3 -l en             # Set language
transcribe run file.mp3 --no-diarize      # Skip speaker ID (faster)
```

### Live Mode

Record and transcribe at the same time. The whisper model is pre-loaded before recording starts, and audio is transcribed in background chunks during the call. When you stop, only alignment and speaker identification remain.

```bash
transcribe live                              # Uses config defaults
transcribe live -m medium                    # Use a specific model
transcribe live -t "Team Standup"            # Set title
transcribe live -s "Alice,Bob" --no-diarize  # Name speakers manually
```

**Adaptive scaling:** Live mode monitors how fast each chunk is processed relative to the audio length. If your machine is struggling, it automatically:
- Increases the interval between chunks
- Reduces batch size
- Falls back to record-only mode if needed (transcribes after the call ends)

Audio quality indicators show during recording:
- Green/yellow/red for mic and system audio levels
- Warning if mic stays very low for 10+ seconds

### Voice Enrollment

Enroll your voice once, and your name appears automatically in every transcript:

```bash
transcribe enroll                  # Records 15 seconds of your voice
transcribe enroll -n "Alice"       # Set name directly
```

Everyone else shows as `Person 1`, `Person 2`, etc. You can still name others with `-s` when needed.

### Calendar Watch

Auto-record meetings from your macOS Calendar:

```bash
transcribe watch                   # Watch calendar, auto-record meetings
transcribe install-daemon          # Run as background service (survives terminal close)
transcribe watch-status            # Check daemon status
transcribe uninstall-daemon        # Remove background service
```

**How it works:**
1. Reads your macOS Calendar on launch and every few hours
2. Sleeps until the next meeting (zero CPU while waiting)
3. Starts recording when a meeting begins
4. Stops recording on silence timeout or when the meeting ends + buffer
5. Discards recordings shorter than 2 minutes (catches non-meeting events)

You can filter which calendars to watch in `config.json`.

## Models

| Model | Speed | Accuracy | RAM | Size |
|-------|-------|----------|-----|------|
| tiny | Fastest | Low | ~1GB | ~75MB |
| base | Fast | Good | ~1.5GB | ~140MB |
| **small** | **Balanced** | **Recommended** | **~3GB** | **~460MB** |
| medium | Slow | High | ~5GB | ~1.5GB |
| large-v3 | Slowest | Highest | ~10GB | ~3GB |

Speed depends on your hardware. On M1 16GB, `small` processes about 2x faster than realtime.

## Output

Transcripts are saved as Obsidian-compatible Markdown:

```
Obsidian/Interviews/
├── Recordings/          # Audio files
│   └── 2026-03-19_14-30-00.wav
└── Scripts/             # Transcripts
    └── 2026-03-19 Interview Title.md
```

Each transcript includes YAML frontmatter, timestamps, and speaker labels:

```markdown
---
title: "Interview with Jane"
date: 2026-03-19
type: interview-transcript
speakers: [Jane, Faisal]
---

**[00:00] Jane:** So tell me about your experience...
**[01:23] Faisal:** I've been working in care for about five years...
```

When diarization is off (or only one speaker), segments show timestamps without speaker labels:

```markdown
**[00:00]** Welcome everyone to today's session...
**[02:15]** Let's start with the first topic...
```

## Configuration

Copy the example config and change only what you need:

```bash
cp config.example.json config.json
```

Any value you omit uses the built-in default. Here's what's available:

| Key | Default | Description |
|-----|---------|-------------|
| `default_model` | `"small"` | Whisper model to use |
| `language` | `"en"` | Language code |
| `compute_type` | `"int8"` | Quantization (int8 recommended for CPU) |
| `device` | `"cpu"` | Processing device |
| `user_name` | `""` | Your name for voice recognition |

### Paths

| Key | Default | Description |
|-----|---------|-------------|
| `paths.obsidian_base` | `"~/Library/Mobile Documents/..."` | Root folder for recordings and transcripts |
| `paths.recordings_subdir` | `"Recordings"` | Subfolder for audio files |
| `paths.scripts_subdir` | `"Scripts"` | Subfolder for transcripts |

### Recording

| Key | Default | Description |
|-----|---------|-------------|
| `recording.silence_threshold` | `0.005` | RMS level below this = silence |
| `recording.silence_timeout_minutes` | `10` | Auto-stop after N minutes silence |
| `recording.blackhole_device` | `"BlackHole 2ch"` | Virtual audio device name |
| `recording.mic_low_warning_seconds` | `10` | Seconds before mic-low warning |

### Live Mode

| Key | Default | Description |
|-----|---------|-------------|
| `live.chunk_interval_seconds` | `120` | Seconds between transcription chunks |
| `live.min_chunk_interval` | `60` | Fastest chunk interval (adaptive) |
| `live.max_chunk_interval` | `300` | Slowest chunk interval (adaptive) |
| `live.struggle_ratio` | `0.7` | Ratio threshold to slow down |
| `live.rec_only_ratio` | `1.5` | Ratio threshold to stop transcribing |

### Speaker Memory

| Key | Default | Description |
|-----|---------|-------------|
| `speaker_memory.enabled` | `true` | Auto-match voices against profiles |
| `speaker_memory.similarity_threshold` | `0.75` | Cosine similarity threshold (0-1) |
| `speaker_memory.enrollment_duration_seconds` | `15` | Recording length for enrollment |

### Calendar Watch

| Key | Default | Description |
|-----|---------|-------------|
| `watch.calendars` | `[]` | Filter calendars (empty = all) |
| `watch.silence_timeout_minutes` | `10` | Silence timeout for auto-recordings |
| `watch.min_recording_minutes` | `2` | Discard recordings shorter than this |
| `watch.end_buffer_minutes` | `2` | Keep recording N min after event ends |
| `watch.refresh_hours` | `3` | Re-read calendar every N hours |

### Batch Sizes

Per-model batch sizes (higher = faster, more RAM). Tuned for M1 16GB:

```json
"batch_sizes": { "tiny": 32, "base": 24, "small": 16, "medium": 8, "large-v3": 4 }
```

Reduce if you get memory errors. Increase on machines with more RAM.

## Troubleshooting

**"No audio recorded"** — Check that your mic is working: `transcribe setup`

**BlackHole not capturing system audio** — You need a Multi-Output Device in Audio MIDI Setup. Run `transcribe setup` for step-by-step instructions. Reboot after installing BlackHole.

**Speaker diarization not working** — Make sure `HF_TOKEN` is set and you've accepted the model terms on Hugging Face (see install instructions above).

**Live mode too slow** — The adaptive system will handle this automatically by reducing chunk frequency. You can also use a smaller model (`-m tiny`) or increase `live.chunk_interval_seconds` in config.

**Calendar watch doesn't see my meetings** — Make sure macOS Calendar has permission to be accessed by Terminal. You can test with: `osascript -e 'tell application "Calendar" to return name of every calendar'`

**Watch daemon not starting** — Run `transcribe watch-status` to check. If issues persist, try `transcribe uninstall-daemon` then `transcribe install-daemon`.

## Uninstall

```bash
./uninstall.sh
```

Removes: virtual environment, shell alias, watch daemon, speaker profiles, log files, and config. Optionally removes cached AI models and Obsidian data.

To fully remove, delete the project folder afterward.

## License

[MIT](LICENSE) — do whatever you want with it.

## Acknowledgements

- [WhisperX](https://github.com/m-bain/whisperX) — fast Whisper with word-level timestamps
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — speaker diarization and voice embeddings
- [BlackHole](https://github.com/ExistentialAudio/BlackHole) — virtual audio driver for macOS
