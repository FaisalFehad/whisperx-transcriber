# Transcriber

Local speech-to-text for macOS. Record meetings, lectures, podcasts, or any audio — get accurate transcripts with speaker labels, saved as Markdown.

Everything runs on your Mac's GPU. No audio leaves your machine.

## Why Local?

Cloud transcription services (Otter, Rev, Fireflies) upload your audio to remote servers. That's a problem for:
- **Confidential conversations** — legal, medical, HR, financial discussions
- **NDA-bound meetings** — client calls, partner discussions, internal strategy
- **Personal recordings** — therapy sessions, journaling, private conversations

This tool processes everything on-device using Apple Silicon GPU acceleration. Your audio stays on your disk, your transcripts stay in your vault. No accounts, no subscriptions, no data sharing.

## Features

### Transcription
- **Parakeet CTC engine** — default model that cannot hallucinate (no "Thank Thank Thank..." on silence)
- **Proper punctuation and capitalization** — out of the box, no post-processing needed
- **Whisper fallback** — 5 Whisper models available for non-English or low-RAM setups
- **Spectral subtraction denoising** — automatic background noise removal for Whisper models
- **Hallucination protection** — silence threshold detection skips fake text in Whisper output
- **99 languages** — via Whisper models (Parakeet is English-only)
- **Multiple formats** — MP3, WAV, M4A, OGG, FLAC, WebM

### Recording
- **System audio + mic** — captures both sides of Zoom, Teams, Meet, and other calls via BlackHole
- **Live mode** — transcribe during the call so the transcript is ready when you hang up
- **Silence detection** — auto-stops recording when the meeting ends or audio goes quiet
- **Adaptive scaling** — live mode adjusts chunk size and frequency based on system load
- **Audio level indicators** — green/yellow/red display for mic and system audio during recording

### Speaker Identification
- **Speaker diarization** — identifies who said what using Sortformer (GPU) or pyannote (CPU)
- **Voice enrollment** — record 15 seconds of your voice, get auto-recognized in every transcript
- **Up to 4 speakers** — via Sortformer on GPU (unlimited with pyannote CPU fallback)
- **Configurable sensitivity** — tune detection threshold, minimum duration, and merge gap

### Automation
- **Calendar watch** — auto-record meetings from macOS Calendar
- **Background daemon** — runs as a launchd service, survives terminal close and reboots
- **Smart filtering** — watch specific calendars, discard recordings under 2 minutes
- **Auto-title** — pulls meeting title from your calendar event

### Output
- **Markdown with YAML frontmatter** — ready for Obsidian, Logseq, or any Markdown tool
- **Timestamps and speaker labels** — `**[00:00] Alice:** ...` format
- **Configurable paths** — save to any folder, not just Obsidian

### Privacy
- **Fully local** — all processing on Apple Silicon GPU, no cloud APIs
- **No accounts** — no sign-ups, no API keys (except free HF token for diarization)
- **No telemetry** — nothing phones home
- **Your data stays yours** — audio and transcripts never leave your machine

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ffmpeg (`brew install ffmpeg`)
- [BlackHole](https://github.com/ExistentialAudio/BlackHole) for system audio capture (optional — only needed for Zoom/Teams recording)

## Install

```bash
git clone https://github.com/FaisalFehad/whisperx-transcriber.git
cd whisperx-transcriber
./install.sh
source ~/.zshrc
```

The installer creates a virtual environment, installs dependencies, and adds the `transcribe` command to your shell.

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

### Audio Setup (for system audio capture)

To capture audio from Zoom, Teams, or other apps, you need BlackHole:

```bash
brew install blackhole-2ch
# Reboot your Mac
transcribe setup    # Follow the guided setup
```

The setup guide walks you through creating a Multi-Output Device in Audio MIDI Setup so both you and BlackHole receive the call audio.

## Usage

### Quick Start

```bash
transcribe run ~/Downloads/recording.mp3    # Transcribe any audio file
transcribe rec                               # Record system audio + mic
transcribe run                               # Transcribe last recording
transcribe live                              # Record + transcribe simultaneously
```

### Commands

| Command | Description |
|---------|-------------|
| `transcribe run [file]` | Transcribe an audio file or last recording |
| `transcribe rec` | Record audio (auto-stops after silence) |
| `transcribe live` | Record + transcribe at the same time |
| `transcribe list` | Browse past recordings |
| `transcribe enroll` | Save your voice for auto-recognition |
| `transcribe watch` | Auto-record meetings from calendar |
| `transcribe install-daemon` | Run calendar watch as background service |
| `transcribe uninstall-daemon` | Remove background service |
| `transcribe watch-status` | Check if watch daemon is running |
| `transcribe setup` | Audio device setup guide |

### Transcribe an Audio File

Supported formats: **MP3, WAV, M4A, OGG, FLAC, WebM**.

```bash
transcribe run ~/Downloads/recording.mp3
```

The transcript is saved as Markdown to your configured output folder.

**Options:**

| Flag | Example | What it does |
|------|---------|-------------|
| `-m` | `-m turbo` | Use a different model (default: parakeet) |
| `-t` | `-t "Team Meeting"` | Set the transcript title |
| `-s` | `-s "Alice,Bob"` | Name the speakers |
| `-l` | `-l ar` | Set language (default: en) — [99 languages supported](https://github.com/openai/whisper#available-models-and-languages) |
| `--no-diarize` | | Skip speaker identification (faster) |
| `--no-denoise` | | Skip audio denoising (Whisper models only) |

**Examples:**

```bash
# Transcribe a podcast
transcribe run ~/Downloads/episode.mp3 -t "Episode 42" --no-diarize

# Meeting with named speakers
transcribe run ~/Downloads/meeting.m4a -t "Q1 Review" -s "Alice,Bob,Carol"

# Lecture in Arabic
transcribe run ~/Downloads/lecture.wav -m small -l ar

# Fast mode — no speaker identification
transcribe run ~/Downloads/call.mp3 --no-diarize
```

### Live Mode

Record and transcribe at the same time. Audio is transcribed in background chunks during the call. When you stop, only alignment and speaker identification remain.

```bash
transcribe live                              # Uses config defaults
transcribe live -m medium                    # Use a specific model
transcribe live -t "Team Standup"            # Set title
transcribe live -s "Alice,Bob" --no-diarize  # Name speakers manually
```

**Adaptive scaling:** If your machine is struggling, live mode automatically increases the interval between chunks, reduces batch size, or falls back to record-only mode (transcribes after the call ends).

Audio quality indicators show during recording — green/yellow/red for mic and system audio levels.

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
5. Discards recordings shorter than 2 minutes

Filter which calendars to watch in `config.json`.

## Models

The default model is **Parakeet TDT 0.6B** — a CTC-based model from NVIDIA NeMo. Whisper models are available as alternatives.

| Model | Type | Speed | Size | Notes |
|-------|------|-------|------|-------|
| **parakeet** | **CTC** | **~17x RT** | **~2.5GB** | **Default — best accuracy, no hallucinations** |
| small.en | Whisper | ~16x RT | ~460MB | Low RAM alternative |
| turbo | Whisper | ~12x RT | ~1.6GB | Good for technical jargon |
| medium.en | Whisper | ~8x RT | ~1.5GB | Higher accuracy Whisper |
| large-v3 | Whisper | ~4x RT | ~3GB | Highest accuracy Whisper |

Speed measured on M1 16GB. RT = realtime (17x means a 1-hour file transcribes in ~3.5 min).

### Why Parakeet over Whisper?

Parakeet uses CTC (Connectionist Temporal Classification) — it maps audio frames directly to text. Unlike Whisper's autoregressive decoder, CTC **cannot hallucinate**. There's no feedback loop to generate runaway tokens like "Thank Thank Thank..." or "..." during silence.

Parakeet also produces properly punctuated, capitalized text out of the box.

### Accuracy

Evaluated on a 37-minute recording (4 speakers, background noise, ~60s silence at start). Scored independently by two human evaluators listening to the reference audio. All times on M1 16GB.

| Rank | Score /10 | Model | Speed | Hallucinations | Stability | Notes |
|------|-----------|-------|-------|----------------|-----------|-------|
| 1 | **9.5** | **Parakeet + diarize** | **129s (~17x RT)** | **None** | **Stable** | Best fidelity, proper punctuation and caps |
| 2 | 9.0 | Parakeet, no diarize | 129s (~17x RT) | None | Stable | Same text quality, no speaker labels |
| 3 | 7.5 | Whisper medium | ~275s (~8x RT) | Moderate | OK | Lexical substitutions ("flip-flops", "CHI") |
| 4 | 6.5 | Whisper + denoising | ~170s (~13x RT) | Some at start | OK | "DAVID" hallucinations in opening silence |
| 5 | 6.0 | Whisper large-v3.en | ~550s (~4x RT) | Low | Unstable | Skipped first 6.5 min entirely |
| 6 | 5.5 | Whisper turbo | ~180s (~12x RT) | Severe at start | Unstable | First minutes corrupted by looping tokens |
| 7 | 4.5 | Whisper small | ~140s (~16x RT) | Severe | Unstable | Got stuck in hallucination loop on raw audio |

**Key columns:**
- **Hallucinations** — Whisper generates fake text during silence/noise. Parakeet's CTC architecture makes this impossible.
- **Stability** — Whisper turbo/small can enter infinite hallucination loops on noisy audio and never finish. Parakeet always completes.

**When to choose what:**

| Use case | Model | Why |
|----------|-------|-----|
| **General use** | `parakeet` (default) | Best accuracy, no hallucinations, always stable |
| **Low RAM (< 8GB free)** | `small.en` | Only 460MB vs 2.5GB |
| **Non-English** | `small` or `turbo` | Parakeet is English-only |
| **Technical jargon** | `turbo` | Larger vocabulary, but needs denoising |

When using Whisper models, hallucination mitigation is enabled by default (silence threshold + spectral subtraction denoising). Disable denoising with `--no-denoise`.

## Output

Transcripts are saved as Markdown with YAML frontmatter:

```
Output Folder/
├── Recordings/          # Audio files
│   └── 2026-03-19_14-30-00.wav
└── Scripts/             # Transcripts
    └── 2026-03-19 Meeting Title.md
```

Example transcript:

```markdown
---
title: "Q1 Review"
date: 2026-03-19
type: transcript
speakers: [Alice, Bob]
model: parakeet
language: en
audio_duration: 1847.3
---

**[00:00] Alice:** Let's start with the Q1 numbers...
**[01:23] Bob:** Revenue was up 12% compared to last quarter...
```

Without diarization, segments show timestamps only:

```markdown
**[00:00]** Welcome to episode 42...
**[02:15]** Today we're discussing...
```

## Configuration

Copy the example config and change only what you need:

```bash
cp config.example.json config.json
```

Any value you omit uses the built-in default.

### General

| Key | Default | What it does |
|-----|---------|--------------|
| `engine` | `"mlx"` | `"mlx"` runs on Apple GPU (fast). `"whisperx"` runs on CPU (compatible with any machine) |
| `default_model` | `"parakeet"` | Which model to use. `"parakeet"` for best accuracy, or any Whisper model (`"small.en"`, `"turbo"`, `"medium.en"`, `"large-v3"`) |
| `language` | `"en"` | Language code. Set explicitly to avoid misdetection. [99 languages supported](https://github.com/openai/whisper#available-models-and-languages) |
| `user_name` | `""` | Your name — used in speaker recognition to label your voice in transcripts |
| `auto_title_from_calendar` | `true` | When recording during a calendar event, use the event name as the transcript title |

### Whisper Settings

Only applies when using a Whisper model (not parakeet).

| Key | Default | What it does |
|-----|---------|--------------|
| `whisper.hallucination_silence_threshold` | `2.0` | Seconds of silence before Whisper's output is checked for hallucination. When detected, the segment is skipped. Set to `null` to disable |

### Audio Denoising

Spectral subtraction pre-processing. Only applies to Whisper models — Parakeet skips this automatically since CTC can't hallucinate on silence.

| Key | Default | What it does |
|-----|---------|--------------|
| `denoise.enabled` | `true` | Whether to denoise audio before Whisper transcription. Subtracts background noise using the first N seconds as a noise profile |
| `denoise.factor` | `2.0` | How aggressively to remove noise. Higher = more removal but risks distorting speech. 2.0 is a good balance |
| `denoise.noise_profile_seconds` | `10` | How many seconds from the start of the audio to use as a noise sample. Works best when the recording starts with silence or hold music |

### Paths

| Key | Default | What it does |
|-----|---------|--------------|
| `paths.obsidian_base` | `"~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Interviews"` | Root folder where recordings and transcripts are saved |
| `paths.recordings_subdir` | `"Recordings"` | Subfolder for audio files within the base path |
| `paths.scripts_subdir` | `"Scripts"` | Subfolder for transcript Markdown files |

### Recording

| Key | Default | What it does |
|-----|---------|--------------|
| `recording.silence_threshold` | `0.005` | RMS audio level below which the signal is considered silence. Lower = more sensitive |
| `recording.silence_timeout_minutes` | `10` | Stop recording after this many minutes of continuous silence |
| `recording.blackhole_device` | `"BlackHole 2ch"` | Name of the virtual audio device for system audio capture |
| `recording.mic_low_warning_seconds` | `10` | Show a warning if mic input stays very low for this many seconds |

### Speaker Diarization

Controls who-said-what detection.

| Key | Default | What it does |
|-----|---------|--------------|
| `diarization.enabled` | `true` | Whether to identify individual speakers in the transcript |
| `diarization.engine` | `"mlx-audio"` | `"mlx-audio"` uses Sortformer on GPU (fast, max 4 speakers). `"pyannote"` uses CPU (slower, unlimited speakers) |
| `diarization.threshold` | `0.5` | Speaker detection sensitivity (0-1). Lower catches more speech but may produce false detections |
| `diarization.min_duration` | `0.0` | Ignore speaker segments shorter than this (seconds). Filters micro-segments and blips |
| `diarization.merge_gap` | `0.0` | Merge segments from the same speaker that are closer than this gap (seconds). Reduces fragmentation |

### Live Mode

| Key | Default | What it does |
|-----|---------|--------------|
| `live.model` | `"small.en"` | Model for real-time transcription during recording. Uses a lighter model than post-recording for speed |
| `live.chunk_interval_seconds` | `120` | How often to transcribe accumulated audio (seconds) |
| `live.min_chunk_interval` | `60` | Shortest interval the adaptive system will use (seconds) |
| `live.max_chunk_interval` | `300` | Longest interval before the adaptive system gives up and switches to record-only |
| `live.chunk_overlap_seconds` | `5` | Seconds of overlap between consecutive chunks to avoid cutting words at boundaries |
| `live.min_chunk_seconds` | `30` | Minimum audio length before a chunk is worth transcribing |
| `live.struggle_ratio` | `0.7` | If transcription takes longer than this ratio of the chunk duration, increase the interval |
| `live.rec_only_ratio` | `1.5` | If transcription takes longer than this ratio, stop transcribing and just record |

### Speaker Memory

| Key | Default | What it does |
|-----|---------|--------------|
| `speaker_memory.enabled` | `true` | Auto-match voices in new recordings against enrolled voice profiles |
| `speaker_memory.similarity_threshold` | `0.75` | How similar a voice must be to a profile to match (0-1). Lower = more lenient matching |
| `speaker_memory.enrollment_duration_seconds` | `15` | How many seconds of speech to record when enrolling a new voice |

### Calendar Watch

| Key | Default | What it does |
|-----|---------|--------------|
| `watch.model` | `"small"` | Model to use for auto-transcription after a watched recording finishes |
| `watch.calendars` | `[]` | Only watch these calendar names. Empty = watch all calendars |
| `watch.silence_timeout_minutes` | `10` | Stop auto-recording after this many minutes of silence |
| `watch.min_recording_minutes` | `2` | Discard auto-recordings shorter than this. Catches non-meeting calendar events |
| `watch.end_buffer_minutes` | `2` | Keep recording for N minutes after the calendar event ends. Catches meetings that run over |
| `watch.refresh_hours` | `3` | Re-read the calendar every N hours to pick up newly added meetings |
| `watch.record_only` | `true` | `true` = record during the meeting, transcribe after. `false` = live-transcribe during the meeting |

### Batch Sizes

Per-model batch sizes for the WhisperX CPU engine (higher = faster, more RAM). Tuned for M1 16GB:

```json
"batch_sizes": { "tiny": 32, "base": 24, "small": 16, "medium": 8, "large-v3": 4 }
```

Reduce if you get memory errors. Increase on machines with more RAM. Not used by the MLX engine.

## Troubleshooting

**"No audio recorded"** — Check that your mic is working: `transcribe setup`

**BlackHole not capturing system audio** — You need a Multi-Output Device in Audio MIDI Setup. Run `transcribe setup` for step-by-step instructions. Reboot after installing BlackHole.

**Speaker diarization not working** — Make sure `HF_TOKEN` is set and you've accepted the model terms on Hugging Face (see install instructions above).

**Live mode too slow** — The adaptive system handles this automatically. You can also use a smaller model (`-m tiny`) or increase `live.chunk_interval_seconds`.

**Calendar watch doesn't see my meetings** — Make sure macOS Calendar has permission to be accessed by Terminal: `osascript -e 'tell application "Calendar" to return name of every calendar'`

**Watch daemon not starting** — Run `transcribe watch-status` to check. Try `transcribe uninstall-daemon` then `transcribe install-daemon`.

## Uninstall

```bash
./uninstall.sh
```

Removes: virtual environment, shell alias, watch daemon, speaker profiles, log files, and config. Optionally removes cached AI models and output data.

## License

[MIT](LICENSE)

## Acknowledgements

- [Parakeet TDT](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#parakeet-tdt) — CTC speech recognition from NVIDIA NeMo
- [Sortformer](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/configs.html) — neural speaker diarization from NVIDIA NeMo
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) — MLX bindings for Parakeet, Sortformer, and Whisper on Apple Silicon
- [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) — Whisper on Apple Silicon GPU via MLX
- [WhisperX](https://github.com/m-bain/whisperX) — fast Whisper with word-level timestamps (CPU fallback)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — speaker diarization and voice embeddings
- [BlackHole](https://github.com/ExistentialAudio/BlackHole) — virtual audio driver for macOS
