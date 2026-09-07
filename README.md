# ClipYourVideos.com

Longform video in → vertical, clip-ready shorts out. Runs entirely on your machine. No API keys, no subscription, no per-video cost.

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
python -m clip_machine.pipeline "<source>" --layout split --mode debate
```

Output lands in `artifacts/clips/<video-id>/`:
- `clip_01.mp4` … — 720×1280 vertical
- `manifest.json` — start/end, score, hook line, and full transcript per clip

## Pipeline

| Stage | Tool | Notes |
|---|---|---|
| Download | `yt-dlp` | Skipped if you pass a local file. **Never use `--download-sections` for a deep offset in a long live VOD** — it forces serial ffmpeg streaming from t=0. Grab a direct `https` DASH itag with `-N 8` and slice locally instead |
| Transcribe | `faster-whisper` (local, CPU) | ~8x realtime at `tiny.en`. Cached to `artifacts/_work/<id>.words.json` |
| Score | heuristics + optional Ollama | See below |
| Cut + reframe | `ffmpeg`, single pass | One encode, 720x1280 |

## Speed

Both the transcript and the speaker turns are cached per video, so re-running
to retune selection skips straight to scoring + render.

Measured on a 15-min two-person source (i5-1235U, no GPU):

| | time |
|---|---|
| Transcribe (`tiny.en`, first run) | ~113s |
| Speaker turns (first run) | ~130s |
| Audio energy | ~4s |
| Render 6 clips (~8 min of output) | ~115s |
| **Re-run with both caches warm** | **~186s** |

What was tried and did NOT help:
- **`h264_qsv` hardware encoding** — 9.5s vs libx264 veryfast's 10.1s. The
  filter graph runs on the CPU either way and QSV pays frame upload/download.
- **Parallel rendering** — 186s / 212s / 157s for 1 / 2 / 3 workers, i.e.
  non-monotonic noise, because libx264 already saturates the CPU. Default is
  sequential; raise `CLIP_RENDER_WORKERS` only on a machine with headroom.

What DID help: dropping output from 1080x1920 to **720x1280** (2x faster, and
the source is ~480p so 1080 was pure upscaling) and turning captions off
(~13%).

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
| `CLIP_WHISPER_MODEL` | `tiny.en` | Transcript is an index, not output — see Models below |
| `CLIP_BATCH_SIZE` | `8` | Batched inference, ~2.5x faster than sequential |
| `CLIP_WHISPER_THREADS` | `8` | |
| `CLIP_MIN_SECONDS` / `CLIP_MAX_SECONDS` | `20` / `150` | Outer bounds, **not a target** — see below |
| `CLIP_OUT_WIDTH` / `CLIP_OUT_HEIGHT` | `720` / `1280` | 1080 doubles render time for nothing on a 480p source |
| `CLIP_RENDER_WORKERS` | `1` | Parallel renders; no measured win on a CPU-bound machine |
| `CLIP_OLLAMA_MODEL` | `llama3.2:3b` | |

If clip picks are weak, tune `heuristic_score()` in `src/clip_machine/score.py` — that function is the whole "AI virality score," and it's 40 readable lines.
