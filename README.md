# Local Transcriber

Record recordings (in-person or Zoom/Teams), transcribe with speaker diarization, and save as Markdown to your Obsidian vault.

Built on [Parakeet TDT](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3) (NVIDIA NeMo, CTC) for transcription and [Sortformer](https://huggingface.co/mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16) for speaker diarization — both running on Apple Silicon GPU via [MLX](https://github.com/ml-explore/mlx). Falls back to [WhisperX](https://github.com/m-bain/whisperX) (CPU) if needed. Runs entirely locally — no cloud APIs.

## Features

- **Record** system audio + mic simultaneously (Zoom, Teams, Meet, etc.)
- **Transcribe** with word-level timestamps and speaker labels
- **Live mode** — transcribe during the call so results are ready when you hang up
- **Speaker memory** — enroll your voice once, get auto-recognized in every transcript
- **Calendar watch** — auto-record meetings from macOS Calendar (runs as background daemon)
- **MLX GPU acceleration** — runs on Apple Silicon GPU (M1/M2/M3/M4)
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
git clone https://github.com/FaisalFehad/whisperx-transcriber.git
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

### Transcribe an Audio File

Convert any audio file to a transcript. Supported formats: **MP3, WAV, M4A, OGG, FLAC, WebM**.

```bash
transcribe run ~/Downloads/audio.mp3
```

That's it — the transcript is saved as Markdown to your Obsidian vault.

**Options:**

| Flag | Example | What it does |
|------|---------|-------------|
| `-m` | `-m turbo` | Use a different model (default: parakeet) |
| `-t` | `-t "Team Meeting"` | Set the transcript title |
| `-s` | `-s "Alice,Bob"` | Name the speakers |
| `-l` | `-l ar` | Set language (default: en) — [99 languages supported](https://github.com/openai/whisper#available-models-and-languages) |
| `--no-diarize` | | Skip speaker identification (faster) |
| `--no-denoise` | | Skip audio denoising (Whisper only) |

**Examples:**

```bash
# Basic — just transcribe
transcribe run ~/Downloads/audio.mp3

# With title and speaker names
transcribe run ~/Downloads/audio.mp3 -t "Session with Jane" -s "Jane,Faisal"

# Fast mode — no speaker ID
transcribe run ~/Downloads/audio.mp3 --no-diarize
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

The default model is **Parakeet TDT 0.6B** — a CTC-based model from NVIDIA NeMo that produces properly punctuated, capitalized text without hallucinations. Whisper models are available as alternatives.

| Model | Type | Speed | Size | Notes |
|-------|------|-------|------|-------|
| **parakeet** | **CTC** | **~17x RT** | **~2.5GB** | **Default — best accuracy, no hallucinations** |
| small.en | Whisper | ~16x RT | ~460MB | Low RAM alternative |
| turbo | Whisper | ~12x RT | ~1.6GB | Good for technical jargon |
| medium.en | Whisper | ~8x RT | ~1.5GB | Higher accuracy Whisper |
| large-v3 | Whisper | ~4x RT | ~3GB | Highest accuracy Whisper |

Speed measured on M1 16GB with MLX (GPU). RT = realtime (17x RT means a 1-hour file transcribes in ~3.5 min).

### Why Parakeet over Whisper?

Parakeet uses a CTC (Connectionist Temporal Classification) architecture that maps audio frames directly to text. Unlike Whisper's autoregressive decoder, CTC **cannot hallucinate** — it has no feedback loop to generate runaway tokens like "Thank Thank Thank..." or "..." during silence.

### Accuracy

Evaluated on a 37-minute session recording (4 speakers, background noise, ~60s silence at start). Scored independently by two human evaluators listening to the reference audio. All times measured on M1 16GB.

| Rank | Score /10 | Model | Speed | Hallucinations | Stability | Notes |
|------|-----------|-------|-------|----------------|-----------|-------|
| 1 | **9.5** | **Parakeet + diarize** | **129s (~17x RT)** | **None** | **Stable** | Best fidelity, proper punctuation and caps |
| 2 | 9.0 | Parakeet, no diarize | 129s (~17x RT) | None | Stable | Same text quality, no speaker labels |
| 3 | 7.5 | Whisper medium | ~275s (~8x RT) | Moderate | OK | Lexical substitutions ("flip-flops", "CHI") |
| 4 | 6.5 | Whisper + denoising | ~170s (~13x RT) | Some at start | OK | "DAVID" hallucinations in opening silence |
| 5 | 6.0 | Whisper large-v3.en | ~550s (~4x RT) | Low | Unstable | Skipped first 6.5 min entirely |
| 6 | 5.5 | Whisper turbo | ~180s (~12x RT) | Severe at start | Unstable | First minutes corrupted by looping tokens |
| 7 | 4.5 | Whisper small | ~140s (~16x RT) | Severe | Unstable | Got stuck in hallucination loop on raw audio |

**Key columns explained:**
- **Speed** — Wall-clock time for the 37-min file (transcription only, excludes diarization). RT = realtime.
- **Hallucinations** — Whisper generates fake text ("Thank Thank Thank...", "...", "DAVID") during silence. Parakeet's CTC architecture makes this impossible.
- **Stability** — Whether the model reliably finishes. Whisper turbo/small can enter infinite hallucination loops on noisy audio and never complete. Parakeet always finishes.

**When to choose what:**

| Use case | Model | Why |
|----------|-------|-----|
| **General use** | `parakeet` (default) | Best accuracy, no hallucinations, stable |
| **Low RAM (< 8GB free)** | `small.en` | Only 460MB vs 2.5GB, needs `--no-denoise` off |
| **Non-English** | `small` or `turbo` | Parakeet is English-only |
| **Technical jargon** | `turbo` | Larger Whisper vocabulary, but needs denoising |

When using Whisper models, `hallucination_silence_threshold` (HST) is enabled by default to skip hallucinated segments during silence. Spectral subtraction denoising is also applied automatically (disable with `--no-denoise`).

## Output

Transcripts are saved as Obsidian-compatible Markdown:

```
Obsidian/Sessions/
├── Recordings/          # Audio files
│   └── 2026-03-19_14-30-00.wav
└── Scripts/             # Transcripts
    └── 2026-03-19 Session Title.md
```

Each transcript includes YAML frontmatter, timestamps, and speaker labels:

```markdown
---
title: "Session with Jane"
date: 2026-03-19
type: session-transcript
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
| `engine` | `"mlx"` | `"mlx"` (Apple GPU, fast) or `"whisperx"` (CPU, compatible) |
| `default_model` | `"parakeet"` | Transcription model (`"parakeet"` or Whisper variants) |
| `language` | `"en"` | Language code — [99 languages supported](https://github.com/openai/whisper#available-models-and-languages) |
| `compute_type` | `"int8"` | Quantization — whisperx engine only |
| `device` | `"cpu"` | Processing device — whisperx engine only |
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

- [Parakeet TDT](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#parakeet-tdt) — CTC speech recognition from NVIDIA NeMo
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) — MLX bindings for Parakeet, Sortformer, and Whisper on Apple Silicon
- [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) — Whisper on Apple Silicon GPU via MLX
- [WhisperX](https://github.com/m-bain/whisperX) — fast Whisper with word-level timestamps (CPU fallback)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — speaker diarization and voice embeddings
- [BlackHole](https://github.com/ExistentialAudio/BlackHole) — virtual audio driver for macOS
