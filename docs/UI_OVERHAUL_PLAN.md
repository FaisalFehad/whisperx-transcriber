# UI Overhaul Plan — whisperx-transcriber

**Created**: 2026-03-21
**Status**: PAUSED — separate channels pipeline is higher priority

## Completed So Far
- [x] Phase 0: Rich installed, ui.py created, speaker prompt fixed
- [x] Phase 1: All 15 manual boxes migrated to Rich Panel/Table
- [x] Phase 2a-b: RecordingDisplay with waveform (cmd_record)
- [x] Phase 3: ProgressTracker rewritten with Rich Live
- [x] Pause recording (P key) with reusable raw_keys()/poll_key()
- [x] Auto-open Finder after transcription
- [x] Save source audio config (audio.keep_recording)
- [x] System health warnings (CPU/RAM at 85%)
- [x] Code cleanup: removed LivePanel (220 lines), BlackHole legacy (120 lines), dead code
- [x] Models trimmed to 5 (removed tiny, base, medium.en, large-v3.en)
- [x] README updated for ScreenCaptureKit, new models, diarization/denoise clarification

## Remaining (on pause)
- [ ] Phase 2c: Wire cmd_live to RecordingDisplay
- [ ] Phase 5: Calendar meeting info (--title, --end-time from cmd_watch)
- [ ] Phase 6: Calendar attendee names (extend AppleScript)

## Overview

Replace hand-rolled ANSI terminal UI with Rich library. Add waveform visualization, monitoring warnings, interactive controls, and calendar integration.

## Architecture

```
ui.py (NEW)
├── console = Console()
├── RecordingDisplay(Live)        — shared by cmd_record + cmd_live
│   ├── waveform renderer
│   ├── warning bar
│   ├── meeting countdown
│   └── keyboard handler (termios/tty)
├── box helpers (Panel wrappers)
└── post_completion_menu()

transcriber.py (EXISTING — modified)
├── cmd_record → uses RecordingDisplay
├── cmd_live  → uses RecordingDisplay
├── cmd_run   → uses Rich Progress
├── LivePanel → thin wrapper over RecordingDisplay
└── audio callbacks → feed data to AudioState

system_audio.py (DONE)
└── ScreenCaptureKit capture
```

**Thread safety**: Audio callbacks write to a shared `AudioState` dataclass (RMS, waveform history, warnings) protected by a single lock. `RecordingDisplay` reads it during each Rich `Live` refresh.

---

## Phase 0: Foundation

- [ ] **0a** Install Rich + silero-vad-lite (5 min)
  - `pip install rich silero-vad-lite`
  - Rich: 11MB (terminal UI). Silero: 31MB (speech detection)

- [ ] **0b** Create `ui.py` module (30 min)
  - Shared `console = Console()`
  - Box helper functions wrapping Rich Panel/Table
  - Waveform renderer from RMS history ring buffer
  - Warning bar component
  - Keep transcriber.py from growing beyond 3700 lines

- [ ] **0c** Speaker prompt fix (5 min)
  - Change to: `Speaker names [optional — Enter to skip]:`
  - One-line edit in `prompt_transcribe()` line ~1375

## Phase 1: Replace All Boxes (1.5 hours)

- [ ] **1** Migrate 15 manual box locations to Rich Panel + Table
  - Locations: lines 915, 1151, 1211, 1399, 1448, 1475, 1924, 2390, 2583, 2648, 3040, 3155, 3292, 3520
  - Delete ~300 lines of manual padding math
  - Auto-aligned columns, terminal-width aware
  - Test each box after migration

## Phase 2: Recording Display (2.75 hours)

- [ ] **2a** `RecordingDisplay` class in `ui.py` (1.5 hours)
  - Rich `Live` context for the meter
  - Rolling waveform (`▁▂▃▅▇█▇▅▃▂`) from 60-sample RMS ring buffer
  - Header panel with source, controls, meeting info
  - Warning bar below waveform
  - Shared between cmd_record AND cmd_live

- [ ] **2b** Wire `cmd_record` to `RecordingDisplay` (30 min)
  - Replace scrolling `while recording:` loop (lines 1276-1313)
  - Add keyboard handler: Space=pause, Ctrl+C=stop
  - Test: verify single-line updates, no scrolling

- [ ] **2c** Wire `cmd_live` to `RecordingDisplay` (45 min)
  - Replace `LivePanel.render()` calls
  - Preserve all keyboard controls (Space, B, !, N, C, I, Q, ?)
  - LivePanel becomes thin wrapper over RecordingDisplay

## Phase 3: Transcription Progress (45 min)

- [ ] **3** Replace `ProgressTracker` with Rich Progress
  - Built-in spinner, bar, ETA, step labels
  - Thread-safe (Rich handles the lock)
  - Delete ~100 lines of display code from ProgressTracker
  - Completed steps show ✓ with duration

## Phase 4: Interactive Features (1.75 hours)

- [ ] **4a** Post-completion menu (30 min)
  - `[O]pen transcript  [F]older  [Enter] done`
  - Reusable `post_completion_menu()` in ui.py
  - Used by cmd_run and cmd_live

- [ ] **4b** Live transcription toggle in cmd_record (1 hour)
  - Press T to start streaming ASR
  - Reuse cmd_live's transcription worker thread
  - Show segments below waveform, toggle on/off

- [ ] **4c** Speaker name input during recording (15 min)
  - Already in LivePanel — wire into RecordingDisplay
  - Press N, type name, Enter to confirm

## Phase 5: Config & File Management (30 min)

- [ ] **5** Save source audio config
  - Add to DEFAULT_CONFIG (JSON):
    ```json
    "audio": {
      "save_source": true,
      "source_save_path": ""
    }
    ```
  - Empty path = same folder as transcript
  - If false, delete WAV after successful transcription
  - Guard: only delete after confirmed successful save

## Phase 6: Monitoring & Warnings (1.25 hours)

- [ ] **6a** Clipping + silence warnings (10 min)
  - `np.abs(chunk).max() > 0.99` → clipping flag
  - Escalate 🔴 to text warning after 30s silence

- [ ] **6b** CPU + RAM warnings (15 min)
  - `pip install psutil`
  - Check every 5s: `psutil.cpu_percent()`, `psutil.virtual_memory()`
  - Show in warning bar: `RAM: 78% │ CPU: 45%`

- [ ] **6c** Speech quality via Silero VAD (30 min)
  - `silero-vad-lite` per chunk — speech probability 0.0-1.0
  - Accepts float32 @ 16kHz directly (our format)
  - Warn if sustained low probability: `⚠ Noisy environment`

- [ ] **6d** Spectral SNR estimate (15 min)
  - `np.fft.rfft` per chunk
  - Ratio of speech band (300-3kHz) to total energy
  - Show as `SNR: Good/Fair/Poor`

## Phase 7: Calendar Integration (1.25 hours)

- [ ] **7a** Meeting timer + countdown (30 min)
  - Pass end time from cmd_watch to cmd_record
  - Show in header: `Recording: Weekly Standup`
  - Show in meter: `⏱ -23:41`

- [ ] **7b** Attendee names from calendar (45 min)
  - Extend osascript in `get_today_events()` to fetch attendee names
  - Pre-populate speaker names: `Speakers from calendar: [Alice, Bob] — Enter to accept:`

---

## Target Recording UI

```
┌─ Recording: Weekly Standup ──────────── -23:41 ─┐
│  Source: System + AirPods Pro                    │
│  Speakers: Alice, Bob, Charlie                   │
│  Space=Pause  T=Transcript  N=Names  Ctrl+C=Stop│
└──────────────────────────────────────────────────┘

  ⏺ 06:19  ▁▂▃▅▇▇█▇▅▃▂▁▁▂▄▆▇▅▃▂  🟢 Good
            SNR: Good │ RAM: 4.2GB │ CPU: 23%

  ─── Transcript (T to hide) ──────────────────────
  [Alice]  Yeah I think we should ship it Friday
  [Bob]    What about the API changes?
```

## Noise Detection Stack

| Layer | Package | Size | What it detects |
|-------|---------|------|-----------------|
| Numpy (free) | already installed | 0 | Clipping, silence, SNR estimate, wind |
| Silero VAD Lite | silero-vad-lite | 31MB | Speech vs noise (87.7% accuracy) |
| Echo detection | deferred | — | Needs AEC algorithm, future version |

## Dependencies Added

| Package | Size | Purpose |
|---------|------|---------|
| rich | 11 MB | Terminal UI (panels, tables, live display, progress) |
| silero-vad-lite | 31 MB | Speech detection (ONNX, no PyTorch) |
| psutil | 1 MB | CPU/RAM monitoring |

## Total Estimated Effort

~10 hours across all phases. Phases 0-2 are the critical path (~4 hours).
