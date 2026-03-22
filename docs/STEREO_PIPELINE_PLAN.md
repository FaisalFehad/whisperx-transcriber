# Stereo Pipeline Plan — Separate Mic + System Audio Channels

**Created**: 2026-03-22
**Status**: Planning
**Priority**: High — improves transcription accuracy, simplifies diarization

---

## Phase 0: Go All-MLX (remove torch + whisperx + pyannote + librosa)

Drop the CPU engine path, voice enrollment, and heavy dependencies. MLX only.
Speaker auto-matching kept via ONNX (onnxruntime already installed for mlx_audio).

### What gets removed (~940 MB)

**torch/whisperx/pyannote chain (~720 MB):**

| Package | Size | Was used for | Replaced by |
|---------|------|-------------|-------------|
| torch | 328 MB | Tensor ops for whisperx + pyannote | mlx |
| scipy | 78 MB | librosa dependency | numpy.fft |
| onnxruntime stays | — | Required by mlx_audio (Sortformer) | — |
| transformers | 52 MB | whisperx model loading | mlx_whisper |
| av | 45 MB | torchaudio video decoding | removed |
| pandas | 44 MB | pyannote data handling | removed |
| sympy | 37 MB | torch symbolic math | removed |
| grpc | 37 MB | telemetry (check if huggingface_hub still needs) | TBD |
| sklearn | 30 MB | speechbrain/pyannote ML | removed |
| matplotlib | 19 MB | pyannote plotting | removed |
| whisperx | 17 MB | CPU Whisper ASR + alignment | mlx_whisper |
| numba | 15 MB | librosa JIT acceleration | removed |
| PIL | 13 MB | transformers image processing | removed |
| torchcodec | 13 MB | torchaudio dependency | removed |
| sqlalchemy | 9 MB | transitive dependency | removed |
| networkx | 7 MB | torch dependency | removed |
| nltk | 5 MB | speechbrain dependency | removed |
| ctranslate2 | 5 MB | faster-whisper backend | removed |
| torchaudio | 4 MB | pyannote audio loading | removed |
| speechbrain | 3 MB | pyannote dependency | removed |
| torchmetrics | 3 MB | speechbrain dependency | removed |
| lightning | 3 MB | speechbrain dependency | removed |
| tiktoken | 3 MB | whisperx tokenizer | removed |
| sentencepiece | 3 MB | transformers dependency | removed |
| pyannote | 2 MB | CPU diarization + embeddings | Sortformer |

**librosa chain (~220 MB):**

| Package | Size | Was used for | Replaced by |
|---------|------|-------------|-------------|
| llvmlite | 112 MB | numba JIT compiler | removed |
| scipy | 78 MB | librosa signal processing | numpy.fft |
| librosa | 15 MB | Load MP3/M4A + STFT denoising | ffmpeg subprocess + numpy.fft |
| numba | 15 MB | librosa JIT acceleration | removed |

### What stays (~280 MB)

| Package | Size | Purpose |
|---------|------|---------|
| mlx | 153 MB | Apple ML framework — Parakeet, Whisper, Sortformer |
| onnxruntime | 63 MB | Required by mlx_audio (Sortformer/Silero) |
| numpy | 22 MB | Audio math, everywhere |
| mlx_audio | 5 MB | Parakeet ASR + Sortformer diarization |
| mlx_whisper | ~3 MB | Whisper ASR on MLX |
| huggingface_hub | 3 MB | Download models from HF |
| rich | ~4 MB | Terminal UI |
| sounddevice | ~2 MB | Mic recording |
| soundfile | ~2 MB | WAV I/O |
| pyobjc | ~2 MB | ScreenCaptureKit bridge |
| psutil | ~1 MB | CPU/RAM monitoring |

### Features dropped
- **Voice enrollment** (`transcribe enroll`) — removed. Could re-add via ONNX later.
- **CPU engine** (`engine: "whisperx"`) — removed. Apple Silicon only.
- **pyannote diarization fallback** — Sortformer only (max 4 speakers).

### Audio loading replacement
Replace `librosa.load()` with ffmpeg subprocess (ffmpeg is already a requirement):
```python
def _load_audio(path, sr=16000):
    """Load any audio format via ffmpeg, return mono float32 numpy array."""
    cmd = ["ffmpeg", "-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le",
           "-ar", str(sr), "-ac", "1", "-v", "quiet", "-"]
    result = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(result.stdout, dtype=np.float32)
```

### Denoising replacement
Replace `librosa.stft/istft` with `numpy.fft.rfft/irfft`:
```python
# Current: librosa.stft(audio, n_fft=2048, hop_length=512)
# Replace: manual STFT with numpy
def _stft(audio, n_fft=2048, hop_length=512):
    window = np.hanning(n_fft)
    frames = [audio[i:i+n_fft] * window for i in range(0, len(audio)-n_fft, hop_length)]
    return np.array([np.fft.rfft(f) for f in frames]).T
```

### Code changes

**Step 0a: Remove engine dual-path**
- [ ] Remove `engine` config key and `ENGINE` constant
- [ ] Remove `DEVICE` constant (`config.get("device", "cpu")`)
- [ ] Remove `compute_type` config key
- [ ] Delete all `if use_mlx: ... else: # whisperx` branches in cmd_run (~44 lines)
- [ ] Delete all `if use_mlx: ... else: # whisperx` branches in cmd_live
- [ ] Simplify: MLX path becomes the only path (remove the `if`, keep the body)

**Step 0b: Remove pyannote diarization**
- [ ] Delete `_diarize_pyannote()` function (~30 lines)
- [ ] Simplify `_diarize_standalone()` to just call `_diarize_mlx_audio()` directly
- [ ] Remove `diarization.engine` config (hard-code mlx-audio)
- [ ] Remove `diarization.model` config (pyannote model path)
- [ ] Remove `warnings.filterwarnings("ignore", module="pyannote")`

**Step 0c: Replace speaker embeddings (pyannote → ONNX)**
- [ ] Rewrite `load_embedding_model()`: ONNX InferenceSession instead of pyannote Inference
- [ ] Rewrite `extract_speaker_embedding()`: load audio slice → fbank features → ONNX → 256-dim vector
- [ ] Keep `load_speaker_profiles()`, `save_speaker_profile()`, `match_speakers()` (pure numpy)
- [ ] Keep `resolve_speaker_names()` (rewire to use new embedding functions)
- [ ] Download ONNX model: `Alkd/speaker-embedding-onnx` (~26 MB) via huggingface_hub
- [ ] Fbank feature extraction: use numpy (mel filterbank + log, ~15 lines)

**Step 0d: Remove cmd_enroll**
- [ ] Delete `cmd_enroll()` function (~140 lines)
- [ ] Remove `enroll` subcommand from argparse
- [ ] Remove enrollment-specific config keys (`enrollment_duration_seconds`)
- [ ] Update README: remove enrollment section

**Step 0e: Replace librosa with ffmpeg + numpy.fft**
- [ ] New `_load_audio(path, sr=16000)`: ffmpeg subprocess → numpy float32 array
- [ ] Replace `librosa.load()` in `_denoise_audio()` with `_load_audio()`
- [ ] Replace `librosa.stft()` with numpy windowed FFT (~8 lines)
- [ ] Replace `librosa.istft()` with numpy overlap-add (~8 lines)
- [ ] Remove `--no-denoise` CLI flag (keep denoise config — it still works, just different backend)

Actually wait — we said keep denoising. So keep `--no-denoise` flag too.

**Step 0f: Update dependencies**
- [ ] pyproject.toml: remove `whisperx` dependency
- [ ] pyproject.toml: keep `sounddevice`, `soundfile`, `numpy`, `rich`, `psutil`, `pyobjc-framework-ScreenCaptureKit`
- [ ] pyproject.toml: add `mlx-audio`, `mlx-whisper` (were implicit via whisperx)
- [ ] Remove all dead imports from transcriber.py

**Step 0g: Dead config cleanup**
- [ ] Remove: `engine`, `compute_type`, `device`
- [ ] Remove: `diarization.engine`, `diarization.model`
- [ ] Remove: `speaker_memory.embedding_model` (replace with ONNX model path)
- [ ] Remove: `denoise` section? NO — keep it (feature stays, backend changes)
- [ ] Keep: everything else (paths, recording, batch_sizes, live, watch, diarization.enabled/threshold/etc)

**Step 0h: Update README + test**
- [ ] Update README: remove CPU engine, voice enrollment, update deps section
- [ ] Test: `transcribe rec` → transcribe → verify Parakeet + Sortformer pipeline
- [ ] Test: `transcribe run file.mp3` → verify ffmpeg audio loading
- [ ] Test: `transcribe run file.wav -m small.en` → verify denoising with numpy.fft
- [ ] Test: speaker auto-matching with ONNX embeddings
- [ ] Clean venv install test: verify ~280 MB total

---

## Problem

Currently we mix mic + system audio into mono: `(mic + system) / 2.0`. This:
- Degrades ASR accuracy (two voices in one signal)
- Makes diarization harder (untangle voices from a mix)
- Prevents independent volume control per channel
- Loses the free speaker identity info (mic = you, system = them)

## Solution

Keep mic and system audio as **separate channels** through the entire pipeline. Transcribe each independently, merge by timestamp.

## Architecture

```
Recording:
  Mic (sounddevice)      → channel 1 ─┐
  System (ScreenCaptureKit) → channel 2 ─┤→ save as stereo WAV
                                         │
Transcription (2-person call):           │
  Channel 1 → Parakeet ASR → "You"    ─┐│
  Channel 2 → Parakeet ASR → "Remote"  ─┤→ merge by timestamp → output
                                         │  (NO diarization needed)
                                         │
Transcription (3+ person call):          │
  Channel 1 → Parakeet ASR → "You"    ─┐│
  Channel 2 → Parakeet ASR             ─┤│
            + Sortformer on ch2 only    ─┤→ merge by timestamp → output
              (N-1 speakers)             │
```

## Benefits

| Benefit | Impact |
|---------|--------|
| Skip diarization for 2-person calls | Faster + eliminates diarization errors |
| ASR accuracy improves | Each pass gets clean single-speaker audio |
| Crosstalk handled gracefully | Both talking? Each channel still clear |
| Independent volume/gain per channel | Boost quiet remote speaker without affecting mic |
| 3+ calls: easier diarization | Sortformer separates N-1 speakers on cleaner system audio |
| WhisperX alignment improves | Aligning against clean per-speaker audio |

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Speaker bleed without headphones | Medium | Mic picks up speakers — still better than 50/50 mix. Could detect headphone state via CoreAudio |
| Sync drift between streams | Medium | Both use CMSampleBuffer timestamps. Monitor and insert silence for gaps |
| ~1.5x ASR processing time | Low | Offset by skipping diarization for 2-person. Each pass is faster on cleaner audio |
| Breaking change to saved WAV format | Low | Old mono recordings still work (backward compatible path in cmd_run) |

## Mic Channel Diarization Strategy

The mic captures the user 90% of the time. But occasionally others are in the room too.

**Config key**: `"audio": {"diarize_mic": false}`

| Setting | Behavior | Use case |
|---------|----------|----------|
| `false` (default) | Mic = "You", no diarization on mic channel | Alone at desk (90% of calls) |
| `true` | Run Sortformer on mic channel too, separate local speakers | Conference room, shared laptop |

When `diarize_mic: false`:
- Mic channel → all speech labeled as "You" (or `user_name` from config)
- System channel → always diarized (Sortformer auto-detects 1-4 speakers)
- Fastest path, correct for solo use

When `diarize_mic: true`:
- Mic channel → Sortformer separates local speakers
- System channel → Sortformer separates remote speakers
- Enrolled voice auto-labels "You", others get "Local Speaker 1", etc.
- Two Sortformer runs, but each on clean audio with fewer speakers ≈ one run on messy mix

## Implementation Steps

### Phase 1: Save stereo (backward compatible)

- [ ] **1a** Modify `_mix_audio()` → return stereo numpy array (ch1=mic, ch2=system) instead of mono average
- [ ] **1b** Save as stereo WAV via soundfile (just change channels param)
- [ ] **1c** `cmd_run` detects mono vs stereo input — mono = old recordings (existing pipeline), stereo = new pipeline
- [ ] **1d** Test: record a call, verify stereo WAV has separate channels

### Phase 2: Per-channel transcription

- [ ] **2a** Add `_transcribe_stereo()` function: splits stereo → two mono tracks, runs Parakeet on each
- [ ] **2b** Each channel's transcript gets labeled: ch1 = "You" (or user_name from config), ch2 = "Remote"
- [ ] **2c** Merge function: interleave segments from both transcripts by start timestamp, group consecutive same-speaker segments
- [ ] **2d** Test: compare output quality vs current mono pipeline on same recording

### Phase 3: Channel-aware diarization

- [ ] **3a** System channel (always): run Sortformer, auto-detects 1-4 remote speakers
- [ ] **3b** Mic channel (default: skip): label all mic speech as "You" / `user_name`
- [ ] **3c** Config `audio.diarize_mic: true`: run Sortformer on mic channel too for conference room use
- [ ] **3d** When `diarize_mic: true` + enrolled voice: auto-label matching speaker as "You", others as "Local Speaker N"
- [ ] **3e** Test: solo call (1 mic speaker), conference room (2+ mic speakers), verify both paths

### Phase 4: Per-channel volume control

- [ ] **4a** Auto-normalize each channel independently to target RMS before transcription
- [ ] **4b** Show per-channel levels in RecordingDisplay (mic level + system level separately)
- [ ] **4c** Optional: detect headphone state via CoreAudio, warn if speakers may cause bleed

## Key Design Decisions

1. **Stereo WAV format**: Channel 1 = mic, Channel 2 = system. Standard PCM 16-bit stereo @ 16kHz.
2. **Backward compatible**: `cmd_run` checks channel count. Mono = old pipeline. Stereo = new pipeline. Old recordings keep working.
3. **No new dependencies**: numpy handles all channel manipulation. Parakeet/Whisper run on mono — we just call them twice.
4. **Sequential, not parallel ASR**: Run mic then system. Same peak RAM as current (one Parakeet instance). Could parallel later if needed.
5. **Merge is simple**: Sorted interleave by start timestamp. Overlapping timestamps = crosstalk (both shown).

## Models: All Mono Only

| Model | Input | How we use it |
|-------|-------|---------------|
| Parakeet | Mono 16kHz | Run twice: once per channel |
| Whisper | Mono 16kHz | Same |
| Sortformer | Mono 16kHz | Always on system channel. On mic channel only if `diarize_mic: true` |
| pyannote | Mono 16kHz | Same — fallback for either channel |
| WhisperX align | Mono 16kHz | Run twice: align each channel's transcript against its own audio |
