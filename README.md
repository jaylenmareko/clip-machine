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

## How clips get picked

1. **Heuristic prefilter** — scores every candidate window on self-containedness (heavily penalizes clips that open on "and/so/but/he/this" — the #1 cause of clips that make no sense alone), hook-phrase match, clean ending, 30-60s length sweet spot, and speech density. Narrows hundreds of windows to ~25.
2. **Optional local LLM rerank** — if Ollama is running, it ranks the shortlist. It only ever sees the short candidate text, never the full transcript, which is what makes this feasible without a GPU. Not installed? The pipeline says so and uses the heuristic ranking.

To enable the rerank:
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
