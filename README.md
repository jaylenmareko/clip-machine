# Clip Machine

Longform video in → vertical, captioned, clip-ready shorts out. Runs entirely on your machine. No API keys, no subscription, no per-video cost.

## Install

```bash
pip install -r requirements.txt
```

`ffmpeg` and `yt-dlp` must be on PATH (both already are on this machine). The Whisper model downloads itself once on first run (~250MB) and is cached after.

## Use

```bash
cd projects/business/clip-machine
python -m clip_machine.pipeline "https://youtube.com/watch?v=..." --max-clips 6
python -m clip_machine.pipeline "C:/path/to/local/video.mp4"
python -m clip_machine.pipeline "<source>" --no-llm      # heuristics only, skip Ollama
```

Output lands in `artifacts/clips/<video-id>/`:
- `clip_01.mp4` … — 1080×1920, captions burned in
- `manifest.json` — start/end, score, hook line, and full transcript per clip

## Pipeline

| Stage | Tool | Notes |
|---|---|---|
| Download | `yt-dlp` | Skipped if you pass a local file |
| Transcribe | `faster-whisper` (local, CPU) | Word-level timestamps. The slow stage — cached to `artifacts/_work/<id>.words.json` |
| Score | heuristics + optional Ollama | See below |
| Cut + crop + caption | `ffmpeg`, single pass | 9:16 center crop, one encode |

## Picking WHAT kind of moment (modes)

Pass `--mode` (CLI) or click a mode in the UI. Every mode still has to pass
base quality — self-contained, clean ending, sane length — the mode only
changes what gets a bonus on top.

| Mode | Hunts for |
|---|---|
| `balanced` | General-purpose, clean and self-contained (default) |
| `exciting` | High energy, big claims, raised voice |
| `controversial` | Contrarian takes, called-out consensus |
| `debate` | Podcast conflict — money/relationships/status takes, pushback |
| `funny` | Laughter and comedic beats |
| `story` | Personal narrative with a payoff |
| `educational` | Clear explanation, steps, concrete numbers |

```bash
python -m clip_machine.pipeline "<source>" --mode debate
```

### Two signal families

1. **Lexical** — regexes over the transcript, per mode (see `modes.py`).
2. **Acoustic** — loudness percentiles from `audio_features.py`. This is the
   one a keyword list cannot replicate: excitement is physical, and someone
   raising their voice or speeding up is measurable straight from ffmpeg.
   Every reading is a percentile against *that video's own* baseline, so
   there are no tuned constants that only hold for one recording.

Each clip records which signals fired (`signals` in manifest.json, chips in
the UI) so picks stay explainable and the weights can be tuned against real
output rather than guesswork.

**Limits worth knowing:** the scorer always returns N clips, even when
nothing genuinely matches the mode — it ranks relatively, it does not have a
quality floor. And `funny` depends on Whisper emitting laughter tokens; on
content with no laughter it quietly degrades to base quality + audio.

### Optional local LLM rerank

If Ollama is running, it reranks the ~25 shortlisted candidates and is told
which mode it's hunting for — that's where real semantic judgment of
"controversial" happens, versus the regex lexicon catching surface phrasing.
It only ever sees the shortlist, never the full transcript, which is what
makes it feasible without a GPU. Not installed? The pipeline says so and
uses the heuristic ranking.

```bash
winget install Ollama.Ollama
ollama pull llama3.2:3b
```

## Tuning

Env vars (or a `.env` in this folder):

| Var | Default | Notes |
|---|---|---|
| `CLIP_WHISPER_MODEL` | `small.en` | `base.en` is ~2-3x faster and noticeably sloppier; `medium.en` is better and much slower |
| `CLIP_WHISPER_THREADS` | `8` | |
| `CLIP_MIN_SECONDS` / `CLIP_MAX_SECONDS` | `20` / `90` | Clip length bounds |
| `CLIP_OLLAMA_MODEL` | `llama3.2:3b` | |

If clip picks are weak, tune `heuristic_score()` in `src/clip_machine/score.py` — that function is the whole "AI virality score," and it's 40 readable lines.
