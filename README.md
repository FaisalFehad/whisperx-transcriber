# Local Transcriber

Fast, private speech-to-text for macOS. Record meetings, lectures, podcasts, or any audio — get accurate transcripts with speaker labels. Everything runs on your Mac's GPU. No cloud, no subscriptions, no data leaves your machine.

## Features

### Transcription

- **Parakeet CTC engine** — default model that cannot hallucinate (no "Thank Thank Thank..." on silence)
- **Proper punctuation and capitalization** — out of the box, no post-processing needed
- **Whisper fallback** — 4 Whisper models available for multilingual or low-RAM setups
- **Audio normalization** — peak-headroom-aware volume normalization before transcription (configurable)
- **Hallucination protection** — silence threshold detection skips fake text in Whisper output
- **99 languages** — via Whisper models (Parakeet supports English + 25 European languages)
- **Multiple formats** — MP3, WAV, M4A, OGG, FLAC, WebM

### Recording

- **System audio + mic** — captures both sides of Zoom, Teams, Meet via ScreenCaptureKit
- **No virtual devices** — volume keys work normally, no Audio MIDI Setup configuration
- **Live mode** — transcribe during the call so the transcript is ready when you hang up
- **Pause/resume** — press P to pause recording (audio is discarded while paused)
- **Silence detection** — auto-stops recording when the meeting ends or audio goes quiet
- **Adaptive scaling** — live mode adjusts chunk frequency based on system load
- **Waveform display** — live unicode waveform with quality indicators during recording
- **System health monitoring** — warns when CPU or RAM usage exceeds 85%

### Speaker Identification

- **Speaker diarization** — identifies who said what using Sortformer on Apple GPU
- **Up to 4 speakers** per channel — automatically separated and labeled
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
- Screen Recording permission (macOS prompts on first use)

## Install

```bash
git clone https://github.com/FaisalFehad/local-transcribe.git
cd local-transcribe
./install.sh
source ~/.zshrc
```

The installer creates a virtual environment, installs dependencies, and adds the `transcribe` command to your shell.

### Hugging Face Token (for speaker diarization)

Speaker identification requires a free Hugging Face token:

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Add to your shell:

   ```bash
   echo 'export HF_TOKEN=your_token_here' >> ~/.zshrc
   ```

### Audio Setup (for system audio capture)

System audio capture uses **ScreenCaptureKit** (macOS 12.3+) — no virtual audio driver needed. On first use, macOS will prompt for Screen Recording permission:

**System Settings → Privacy & Security → Screen Recording** → enable your terminal app.

## Usage

### Quick Start

```bash
transcribe run ~/Downloads/recording.mp3     # Transcribe any audio file
transcribe rec                               # Record system audio + mic
transcribe run                               # Transcribe last recording
transcribe live                              # Record + transcribe simultaneously
```

### Commands

| Command                       | Description                                |
| ----------------------------- | ------------------------------------------ |
| `transcribe run [file]`       | Transcribe an audio file or last recording |
| `transcribe rec`              | Record audio (auto-stops after silence)    |
| `transcribe live`             | Record + transcribe at the same time       |
| `transcribe list`             | Browse past recordings                     |
| `transcribe watch`            | Auto-record meetings from calendar         |
| `transcribe install-daemon`   | Run calendar watch as background service   |
| `transcribe uninstall-daemon` | Remove background service                  |
| `transcribe watch-status`     | Check if watch daemon is running           |
| `transcribe setup`            | Audio device setup guide                   |

### Transcribe an Audio File

Supported formats: **MP3, WAV, M4A, OGG, FLAC, WebM**.


```bash
# To transcribe an existing file
transcribe run ~/Downloads/recording.mp3
```

The transcript is saved as Markdown to your configured output folder.

**Options:**

| Flag           | Example             | What it does                                                                                                            |
| -------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `-m`           | `-m medium`         | Use a different model (default: parakeet)                                                                               |
| `-t`           | `-t "Team Meeting"` | Set the transcript title                                                                                                |
| `-l`           | `-l ar`             | Set language (default: en) — [99 languages supported](https://github.com/openai/whisper#available-models-and-languages) |
| `--no-diarize` |                     | Skip speaker identification (faster)                                                                                    |
| `--denoise`    |                     | Enable spectral subtraction denoising (off by default)                                                                  |

**Examples:**

```bash
# Transcribe a podcast
transcribe run ~/Downloads/episode.mp3 -t "Episode 42" --no-diarize

# Meeting with title
transcribe run ~/Downloads/meeting.m4a -t "Q1 Review"

# Lecture in Arabic
transcribe run ~/Downloads/lecture.wav -m medium -l ar

# Noisy recording — enable denoising
transcribe run ~/Downloads/noisy_call.mp3 --denoise
```

### Live Mode

Record and transcribe at the same time. Audio is transcribed in background chunks during the call. When you stop, only alignment and speaker identification remain.

```bash
transcribe live                              # Uses config defaults
transcribe live -m medium                    # Use a specific model
transcribe live -t "Team Standup"            # Set title
transcribe live --no-diarize                 # Skip speaker identification
```

**Adaptive scaling:** If your machine is struggling, live mode automatically increases the interval between chunks or falls back to record-only mode (transcribes after the call ends).

Audio quality indicators show during recording — green/yellow/red for mic and system audio levels.

### Calendar Watch

Auto-record meetings from your macOS Calendar:

```bash
transcribe watch                   # Watch calendar, auto-record meetings
transcribe install-daemon          # Run as background service (survives terminal close)
transcribe watch-status            # Check daemon status
transcribe uninstall-daemon        # Remove background service
```

**Apple Calendar integration (Optional):**

1. Reads your macOS Calendar on launch and every few hours
2. Sleeps until the next meeting (zero CPU while waiting)
3. Starts recording when a meeting begins
4. Stops recording on silence timeout or when the meeting ends + buffer
5. Discards recordings shorter than 2 minutes

Filter which calendars to watch in `config.json`.

## Models

The default model is **Parakeet TDT 0.6B** — a CTC-based model from NVIDIA NeMo. Whisper models are available as alternatives.

| Model        | Type    | Speed       | RAM        | Notes                                                  |
| ------------ | ------- | ----------- | ---------- | ------------------------------------------------------ |
| **parakeet** | **CTC** | **~15x RT** | **~2.5GB** | **Default — best accuracy + speed, no hallucinations** |
| small.en     | Whisper | ~15x RT     | ~460MB     | Low RAM, English-only                                  |
| medium       | Whisper | ~7x RT      | ~1.5GB     | Balanced, multilingual (99 languages)                  |
| turbo        | Whisper | ~3x RT      | ~800MB     | Slow on MLX despite the name                           |
| large-v3     | Whisper | ~4x RT      | ~3GB       | Most capable Whisper (slow, high RAM)                  |

Speed measured on M1 16GB. RT = realtime (15x means a 1-hour file transcribes in ~4 min). RAM is auto-checked — you'll get a warning if your free memory is close to what the model needs.

### Why Parakeet over Whisper?

Parakeet uses CTC (Connectionist Temporal Classification) — it maps audio frames directly to text. Unlike Whisper's autoregressive decoder, CTC **cannot hallucinate**. There's no feedback loop to generate runaway tokens like "Thank Thank Thank..." or "..." during silence.

Parakeet also produces properly punctuated, capitalized text out of the box.

### When to choose what

| Use case                 | Model                | Why                                           |
| ------------------------ | -------------------- | --------------------------------------------- |
| **General use**          | `parakeet` (default) | Best accuracy, blazing fast no hallucinations |
| **Low RAM (< 8GB free)** | `small.en`           | Only 460MB — decent alternative to parakeet   |
| **Non-English**          | `medium`             | 99 languages via Whisper                      |

## Output

Transcripts are saved as Markdown with YAML frontmatter:

```
~/Transcriptions/
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

| Key                        | Default      | What it does                                                                                               |
| -------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------- |
| `default_model`            | `"parakeet"` | Which model to use                                                                                         |
| `language`                 | `"en"`       | Language code ([99 languages supported](https://github.com/openai/whisper#available-models-and-languages)) |
| `auto_title_from_calendar` | `true`       | Use calendar event name as transcript title                                                                |

### Audio Normalization

Volume normalization before transcription. Scales audio to consistent level with peak headroom to prevent clipping. Stereo channels are normalized independently.

| Key                          | Default | What it does                                 |
| ---------------------------- | ------- | -------------------------------------------- |
| `normalize.enabled`          | `true`  | Normalize audio volume before transcription  |
| `normalize.target_rms`       | `0.05`  | Target RMS level (~-26 dBFS)                 |
| `normalize.peak_headroom_db` | `1.0`   | Keep peaks below -1 dBFS to prevent clipping |

### Audio Denoising

Spectral subtraction pre-processing. **Off by default** — eval tests showed all models handle noise well without it. Enable with `--denoise` flag for recordings with heavy constant background noise that start with silence.

| Key                             | Default | What it does                                                                                    |
| ------------------------------- | ------- | ----------------------------------------------------------------------------------------------- |
| `denoise.enabled`               | `false` | Enable denoising (or use `--denoise` flag)                                                      |
| `denoise.factor`                | `2.0`   | Noise subtraction aggressiveness. Higher = more removal but risks distortion                    |
| `denoise.noise_profile_seconds` | `3`     | Seconds of audio used as noise sample. Automatically finds the quietest window in the file |

### Paths

| Key                       | Default                            | What it does                               |
| ------------------------- | ---------------------------------- | ------------------------------------------ |
| `paths.base`              | `"~/Transcriptions"`               | Root folder for recordings and transcripts |
| `paths.recordings_subdir` | `"Recordings"`                     | Subfolder for audio files                  |
| `paths.scripts_subdir`    | `"Scripts"`                        | Subfolder for transcript Markdown files    |

### Recording

| Key                                 | Default                         | What it does                                                    |
| ----------------------------------- | ------------------------------- | --------------------------------------------------------------- |
| `recording.silence_threshold`       | `0.005`                         | RMS level below which signal is silence. Lower = more sensitive |
| `recording.silence_timeout_minutes` | `10`                            | Stop recording after this many minutes of silence               |
| `recording.mic_low_warning_seconds` | `10`                            | Warn if mic stays very low for this long                        |
| `recording.virtual_device_names`    | `["Aggregate", "Multi-Output"]` | Virtual audio devices to exclude from mic selection             |

### Speaker Diarization

| Key                        | Default | What it does                                               |
| -------------------------- | ------- | ---------------------------------------------------------- |
| `diarization.enabled`      | `true`  | Identify individual speakers                               |
| `diarization.threshold`    | `0.5`   | Speaker detection sensitivity (0-1)                        |
| `diarization.min_duration` | `0.0`   | Ignore speaker segments shorter than this (seconds)        |
| `diarization.merge_gap`    | `0.0`   | Merge same-speaker segments closer than this gap (seconds) |

### Live Mode

| Key                           | Default      | What it does                                                        |
| ----------------------------- | ------------ | ------------------------------------------------------------------- |
| `live.model`                  | `"parakeet"` | Model for real-time transcription                                   |
| `live.chunk_interval_seconds` | `120`        | How often to transcribe accumulated audio (seconds)                 |
| `live.struggle_ratio`         | `0.7`        | If transcription takes longer than this ratio, slow down            |
| `live.rec_only_ratio`         | `1.5`        | If transcription takes longer than this ratio, stop and just record |

### Calendar Watch

| Key                             | Default      | What it does                              |
| ------------------------------- | ------------ | ----------------------------------------- |
| `watch.model`                   | `"parakeet"` | Model for auto-transcription              |
| `watch.calendars`               | `[]`         | Calendar names to watch. Empty = all      |
| `watch.silence_timeout_minutes` | `10`         | Stop recording after silence              |
| `watch.min_recording_minutes`   | `2`          | Discard recordings shorter than this      |
| `watch.end_buffer_minutes`      | `2`          | Keep recording N minutes after event ends |
| `watch.record_only`             | `true`       | Record during meeting, transcribe after   |

## Troubleshooting

**"No audio recorded"** — Check that your mic is working: `transcribe setup`

**System audio not captured** — Grant Screen Recording permission: System Settings → Privacy & Security → Screen Recording → enable your terminal app.

**Speaker diarization not working** — Make sure `HF_TOKEN` is set and you've accepted the model terms on Hugging Face (see install instructions above).

**Live mode too slow** — The adaptive system handles this automatically. You can also use a smaller model (`-m small.en`) or increase `live.chunk_interval_seconds`.

**Calendar watch doesn't see my meetings** — Make sure macOS Calendar has permission to be accessed by Terminal: `osascript -e 'tell application "Calendar" to return name of every calendar'`

**Watch daemon not starting** — Run `transcribe watch-status` to check. Try `transcribe uninstall-daemon` then `transcribe install-daemon`.

## Uninstall

```bash
./uninstall.sh
```

Removes: virtual environment, shell alias, watch daemon, log files, and config. Optionally removes cached AI models and output data.

## License

[MIT](LICENSE)

## Acknowledgements

- [Parakeet TDT](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#parakeet-tdt) — CTC speech recognition from NVIDIA NeMo
- [Sortformer](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/configs.html) — neural speaker diarization from NVIDIA NeMo
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) — MLX bindings for Parakeet, Sortformer, and Whisper on Apple Silicon
- [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) — Whisper on Apple Silicon GPU via MLX
- [ScreenCaptureKit](https://developer.apple.com/documentation/screencapturekit) — macOS system audio capture
