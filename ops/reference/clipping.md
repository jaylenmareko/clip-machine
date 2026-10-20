# Step 2 — Clip

**Does:** pulls the queue (highest-view first) and clips each VOD it hasn't done
yet. One at a time — transcription is the memory bottleneck.

**Run:** `python -m clip_machine.runner 3` (clips up to 3 per invocation; use `1`
on a tight machine). `python -m clip_machine.runner --plan` shows the queue +
free RAM without doing work.

**Guards built in:**
- Refuses to start a VOD under ~2.5 GB free RAM (prints `STOP:` and bails rather
  than OOM). Free RAM and re-run.
- **Framing is auto-detected per clip** (center / split / left / right) — no
  manual layout needed. A clip's detected layout is saved in its manifest.

**Writes:** `artifacts/clips/<video-id>/` — `clip_NN.mp4` + `clip_NN.jpg` +
`manifest.json` (start/end, title, description, transcript, layout per clip).

**Check:** VOD gets a folder with clips + a manifest. A VOD that fails to
download (yt-dlp error) is skipped and the runner moves to the next — see
troubleshooting if a specific one keeps failing.
