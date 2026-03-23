# Eval Harness Plan — Model Optimization & Accuracy Testing

**Created**: 2026-03-22
**Status**: Ready to implement
**Purpose**: Grade each model and find optimal settings for each under all conditions.

## Audio Sources (15 min each, from YouTube)

| Source | URL | Type |
|--------|-----|------|
| Clean interview 1 | youtube.com/watch?v=M7K87rGoGps | 2+ speakers, studio |
| Clean interview 2 | youtube.com/watch?v=bc6uFV9CJGg | Different speakers/accent |
| Rain/ambient noise | youtube.com/watch?v=757G_El3ABI | Noise track for mixing |
| Cafe/crowd noise | youtube.com/watch?v=wIBnaNuhuCQ | Noise track for mixing |
| Real noisy recording | youtube.com/watch?v=KVjrpBWtDH0 | People talking in noise |
| Real call 1 | Your Recordings folder | Actual Zoom/Teams from app |
| Real call 2 | Your Recordings folder | Different conditions |

## Test Audio Files

Base file: Interview 1 (15 min) + Interview 2 (15 min) stitched = 30 min.
Noise is overlaid ON TOP — timestamps stay the same across files 1-7.

| # | File | Content | Duration | Ground truth |
|---|------|---------|----------|-------------|
| 1 | test_clean.wav | Base file, no modifications | 30 min | YouTube subtitles |
| 2 | test_rain_light.wav | Base + rain noise at 0.1 | 30 min | Same subtitles |
| 3 | test_rain_heavy.wav | Base + rain noise at 0.5 | 30 min | Same subtitles |
| 4 | test_cafe_light.wav | Base + cafe noise at 0.1 | 30 min | Same subtitles |
| 5 | test_cafe_heavy.wav | Base + cafe noise at 0.5 | 30 min | Same subtitles |
| 6 | test_quiet.wav | Base at 0.1x volume | 30 min | Same subtitles |
| 7 | test_loud.wav | Base at 2.0x volume | 30 min | Same subtitles |
| 8 | test_real_noisy.wav | YouTube noisy clip (as-is) | 15 min | YouTube subtitles |
| 9 | test_real_call_1.wav | Actual Zoom/Teams recording | ~10-15 min | AI grading only |
| 10 | test_real_call_2.wav | Another real call | ~10-15 min | AI grading only |

Files 1-7: same speech, direct comparison. File 8: different YouTube source. Files 9-10: real calls, no subtitles.

## Test Matrix — Full Cross Product

Every setting combo on every file. No assumptions about what affects what.

**Settings:**
- 4 models: parakeet, small.en, medium, turbo
- 2 denoise: on, off (config toggle, works on ALL models including Parakeet)
- 2 normalize: on (0.05 RMS), off (config toggle)
- = 16 setting combos per file

**10 files × 16 combos = 160 tests. Estimated runtime: ~5 hours (run overnight).**

### Naming convention

`{file}_{model}_dn-{on|off}_nm-{on|off}`

Examples:
- `clean_parakeet_dn-off_nm-on`
- `rain_heavy_small.en_dn-on_nm-off`
- `real_call_1_turbo_dn-off_nm-on`

### Full matrix

| File | pk dn✓nm✓ | pk dn✓nm✗ | pk dn✗nm✓ | pk dn✗nm✗ | sm dn✓nm✓ | sm dn✓nm✗ | sm dn✗nm✓ | sm dn✗nm✗ | md dn✓nm✓ | md dn✓nm✗ | md dn✗nm✓ | md dn✗nm✗ | tb dn✓nm✓ | tb dn✓nm✗ | tb dn✗nm✓ | tb dn✗nm✗ |
|------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| rain_light | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| rain_heavy | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| cafe_light | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| cafe_heavy | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| quiet | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| loud | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| real_noisy | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| real_call_1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| real_call_2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

pk=parakeet, sm=small.en, md=medium, tb=turbo, dn=denoise, nm=normalize

## Output Structure

Each test produces a self-contained folder:

```
tests/results/
├── clean_parakeet_dn-off_nm-on/
│   ├── audio.wav           ← copy of test file (not symlink — needed for AI grading upload)
│   ├── transcript.txt      ← plain text only (for AI grading)
│   ├── transcript_full.md  ← full markdown with speakers/timestamps
│   └── info.json           ← all metadata + metrics
├── rain_heavy_small.en_dn-on_nm-off/
│   ├── ...
└── ...
```

### transcript.txt
Plain text only. Stripped of all formatting:
- No YAML frontmatter
- No markdown (`**`, `#`, etc.)
- No timestamps (`[00:01]`)
- No speaker labels (`Speaker 1:`)
- Just the words, one paragraph per segment

### info.json
```json
{
  "test_id": "rain_heavy_small.en_dn-on_nm-on",
  "file": "test_rain_heavy.wav",
  "model": "small.en",
  "denoise": true,
  "normalize": true,
  "duration_seconds": 1800,
  "word_count": 4231,
  "segment_count": 342,
  "speaker_count": 4,
  "processing_time_seconds": 185,
  "wer": 0.258,
  "wer_available": true
}
```

Note: `wer` is null and `wer_available` is false for real call files (no subtitles).

## Metrics Per Test

| Metric | Source | Available for |
|--------|--------|--------------|
| WER | Edit distance vs YouTube subtitles | Files 1-8 only |
| AI accuracy score | AI model grading (1-10) | All files |
| Word count | From transcript | All |
| Segment count | From transcript | All |
| Speaker count | From diarization | All |
| Processing time | Wall clock seconds | All |
| Label switches/min | Speaker change frequency | All |

## Evaluation Methods

### Primary: AI Model Grading
Feed audio.wav + transcript.txt pairs to an AI model with audio support (Gemini, etc.):
```
"Listen to this audio. Read this transcript.
 Rate accuracy 1-10. Assess:
 - Missing speech (words said but not in transcript)
 - Hallucinated text (words in transcript but not said)
 - Wrong words (misheard)
 - Speaker label errors (wrong person attributed)
 - Punctuation and capitalization quality
 - Natural readability (does it flow as real speech?)
 - Completeness (any sections of the conversation missing?)
 Give an overall accuracy percentage."
```
Works for ALL test files including real calls.
Save rating + error list to grading.json per test.

### Secondary: Automated WER vs YouTube subtitles
Compare transcript.txt against downloaded subtitles. Runs during test suite.
Only for files 1-8. Imperfect reference — use for relative comparison only.

### Validation: Human spot-check (one-time)
Listen to 2-3 minutes of one clean file. Verify transcript + AI grading are sensible.

## Reports (generated from results)

### Report 1: Full Results Table
Raw WER/AI score for every test. The big matrix.

### Report 2: Model Scorecard
```
Model      Clean   Avg     Worst   Std    Time    Best settings
parakeet   12.3%   18.2%   28.1%   5.4    42s     dn✗ nm✓
small.en   15.8%   22.4%   38.7%   8.1    28s     dn✓ nm✓
medium     14.2%   20.1%   35.1%   7.2    65s     dn✓ nm✓
turbo      14.1%   20.8%   36.2%   7.5    35s     dn✓ nm✓
```

### Report 3: Setting Impact
```
Setting         Avg impact    Helps on            Hurts on
denoise ON      -2.1% WER    noisy files (+4.2%)  clean (-0.2%)
normalize ON    -3.8% WER    quiet files (+15%)   loud (-0.5%)
```

### Report 4: Condition Difficulty
```
Condition      Avg WER    Hardest for          Easiest for
clean          13.5%      small.en (15.8%)     parakeet (12.3%)
cafe_heavy     32.1%      small.en (38.7%)     parakeet (28.1%)
quiet          28.4%      small.en (45.3%)     parakeet (16.1%)
```

### Report 5: Recommendations
```
Best overall:           parakeet, denoise off, normalize on
Best for noisy audio:   parakeet, denoise off, normalize on
Best for low RAM:       small.en, denoise on, normalize on
Best multilingual:      medium, denoise on, normalize on
Fastest:                turbo, denoise off, normalize on
```

## How Settings Are Changed Per Test

CLI flags — no config file mutation. The test script loops through all combos automatically:

```bash
for file in "${FILES[@]}"; do
    for model in parakeet small.en medium turbo; do
        for denoise in "--denoise" "--no-denoise"; do
            for normalize in "" "--no-normalize"; do
                transcribe run "$file" -m "$model" $denoise $normalize --no-diarize
            done
        done
    done
done
```

| Flag | What | Default |
|------|------|---------|
| `--denoise` | Force denoise ON (even on Parakeet) | Auto (on for Whisper, off for Parakeet) |
| `--no-denoise` | Force denoise OFF | Already exists |
| `--no-normalize` | Disable audio normalization | Normalize on by default |

## Code Changes Required Before Running

| Change | What | Why |
|--------|------|-----|
| Add `--denoise` flag | Force denoise ON for any model including Parakeet | Test denoise on all models |
| Add `--no-normalize` flag | Skip `_normalize()` in `_mix_audio()` | Test normalize on/off |
| Remove `not is_parakeet_model` guard | Let `--denoise` work on Parakeet | Full cross product |
| Plain text export | Function to strip markdown → plain text for transcript.txt | AI grading input |
| Debug logging | `_log()` function gated by `debug` config | Investigate specific problems |

## Pre-run Setup

| Step | What | Why |
|------|------|-----|
| Download models | Run `transcribe run` once with each model on a short file | Avoid download time during tests |
| Download YouTube audio | `./tests/eval_pipeline.sh --download-only` | Get test audio |
| Verify subtitles | Check both interviews have YouTube subtitles | Ground truth for WER |
| Copy real calls | Copy 2 recordings from your Recordings folder to tests/audio/ | Real-world test data |
| Clear GPU memory | Close other GPU-using apps | Consistent timing |

## Error Handling

- If a test crashes (OOM, model error): log error to info.json, continue to next test
- If a model hasn't been downloaded: skip tests for that model, log warning
- If a file is missing: skip all tests for that file, log warning

## Notes

- Files 1-7 share same speech content — timestamps align, direct comparison
- Per-interview accuracy available by splitting at 15-minute mark in the results
- File 8 and files 9-10 reported separately (different content/source)
- Results should be deterministic (temperature=0) but diarization may vary slightly
- Stereo pipeline not tested here — separate test if needed
