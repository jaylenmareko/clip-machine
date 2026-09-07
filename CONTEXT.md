# ClipYourVideos.com — CONTEXT.md

## Stage
MVP build — pipeline written, render path verified, transcription verified. Not yet run end-to-end on a real longform video.

## Done criteria for this stage
- Run `python -m clip_machine.pipeline <url-or-path>` on one real longform video
- Get N vertical (9:16) clips out with burned-in captions and a manifest.json
- Manually post one clip to confirm quality is postable as-is

## Next stage
Face-tracking crop (replace static center crop) once the static-crop clips prove the loop works. Upload automation only if manual posting becomes the bottleneck.

## What this replaces
Paid auto-clippers (Opus Clip, Klap, Vizard). **Runs entirely locally — no API keys, no per-video cost, no subscription.**

## Architecture (high-level, per CLAUDE.md design-pass gate)
- **Pattern:** monolith CLI script. No server, no DB, no auth — single-user local tool.
- **External integrations:** none at runtime. `yt-dlp` (download), `ffmpeg` (cut/crop/burn), `faster-whisper` (local transcription), optional local Ollama (clip reranking). The only network calls are the video download and a one-time Whisper model fetch.
- **Storage:** flat files — `artifacts/clips/<video-id>/clip_NN.mp4` + `manifest.json`. No tables, so the 1NF/2NF/3NF gate doesn't apply.

## Stack
- Python 3.12, deps: `faster-whisper`, `requests` (installed 2026-09-05)
- ffmpeg / yt-dlp — already on PATH
- `faster-whisper small.en`, int8, CPU — model cached at `~/.cache/huggingface`
- Ollama (`llama3.2:3b`) — **optional, not installed.** Pipeline falls back to heuristic ranking without it.

## How clip selection works (the part paid tools charge for)
Two stages, because feeding a full transcript to a CPU LLM is not viable on a GPU-less machine:
1. **Heuristic prefilter** over every candidate window — self-containedness (penalizes clips opening on "and/so/but/he/this"), hook-phrase match, terminal punctuation, 30-60s length sweet spot, speech density. Instant, no model.
2. **Optional local LLM rerank** — Ollama sees only the ~25 shortlisted candidates' text, never the whole transcript. Skipped automatically if Ollama isn't running.

## Hardware reality (this machine: i5-1235U, 16GB, no discrete GPU)
Transcription is the bottleneck and it is CPU-bound. Measured 2026-09-05 — see `reference/benchmark.md`. Transcripts are cached to `artifacts/_work/<id>.words.json`, so re-running scoring/rendering on an already-transcribed video is instant.

## File map
```
src/clip_machine/
  config.py       # model choices + paths, all env-overridable
  download.py     # yt-dlp wrapper, or local file passthrough
  transcribe.py   # ffmpeg audio extract + local faster-whisper -> word list
  score.py        # heuristic prefilter + optional Ollama rerank
  caption.py      # word-timed .ass subtitles (3 words/line, bold, centered)
  cut.py          # single ffmpeg pass: cut + 9:16 crop + caption burn
  pipeline.py     # CLI entrypoint, writes manifest.json
artifacts/clips/  # output, gitignored
artifacts/_work/  # downloads, audio, cached transcripts, gitignored
```

## Current priority
Run it on one real longform video, judge clip-pick quality. If picks are weak, the fix is `score.py` heuristics or installing Ollama — not the rest of the pipeline.

## Known limitations (v1, by design)
- Static center crop — fine for single-speaker talking-head, bad for wide two-person shots
- No upload automation — clips land in `artifacts/clips/`, post manually
- No "already clipped this video" tracking
- Sentence splitting relies on Whisper's punctuation; a model that under-punctuates produces long run-on candidates
