# Clip-Machine Playbook

How the tool works and what makes clips perform. No credentials here — API keys
and YouTube OAuth live in `~/.config/clip-machine/` and are set up by Jaylen
separately; this file is only the workflow and the content strategy.

---

## What the tool is
Longform YouTube video in → vertical 9:16 clips out, each with a title,
description, and hashtags, ready to post to the clips channel. It's a Python
CLI plus a small local web UI. One job at a time (transcription is the memory
bottleneck).

## Running it
- Install: `pip install -r requirements.txt`; `ffmpeg` and `yt-dlp` on PATH.
  The Whisper model auto-downloads once.
- **Make clips:** `python -m clip_machine.pipeline "<youtube-url>" --mode debate --max-clips 10`
  → output in `artifacts/clips/<video-id>/`: `clip_NN.mp4` + `manifest.json`
  (start/end, title, description, transcript per clip).
- **Web UI:** `python -m clip_machine.server` → http://127.0.0.1:8420
  (`/api/run`, `/api/library`, `/api/refit` to change layout, `/api/retitle`,
  `/api/upload` to post one clip).

## Hosting note
Runs best on a machine with **open network egress** and **~3 GB+ free RAM**:
- Uploads can stall on a restricted network (the YouTube upload endpoint hangs
  even when auth succeeds).
- Long transcriptions OOM-crash under ~3 GB free.

---

## Content strategy (what drives views)
- **Source:** longform debate/interview VODs from `@TheSaintandTheSinner`
  (Marquett Burton, 266K subs).
- **The winning vein:** Israel / rabbi / anti-Zionist debate clips. On the clips
  channel this theme is ~67% of uploads but **~84% of all views**; the one
  breakout (14K+) was news-pegged to Iran. Prioritize:
  - the anti-Zionist rabbi angle (Judaism ≠ Zionism),
  - Marquett out-arguing a pro-Israel debater,
  - a current-news peg (Iran, Gaza) where it fits.
- **Deprioritize** dating / money / masculinity clips — about ⅓ the median views.
- Keep watching the source channel for NEW Israel/debate VODs and clip those first.

## Framing — always eyeball; the auto-detector is unreliable
After each batch, check one frame per clip and refit to the right layout:
- `center` — single full-frame camera (speaker fills the frame)
- `split` — genuine two-person debate (both stacked top/bottom)
- `left` / `right` — camera in one half + live-chat panel in the other; crop to
  the speaker's half and drop the chat
A single source can switch composition mid-video, so framing is per-clip.

## Titles
- Name **"Marquett"** when HE is the one making the point; use a descriptor
  ("Rabbi", "Pro-Palestinian debater") only when it's the other person.
- In a two-person clip, confirm who actually said the quoted line before
  attributing it — don't put the guest's words in Marquett's mouth.
- Exactly one ALL-CAPS word for punch (CORNERS, EXPOSES, FOOLISH).
- 3–6 hashtags: identity tags (#judaism #palestine) + reach tags
  (#debate #geopolitics). ~40–95 chars before hashtags.

## Posting cadence
- Cap **6 uploads/day** via the API.
- Schedule **staggered** (≥5 min apart; the API needs ≥15 min lead time).
  Scheduled posts upload private and flip public at the set time — nothing is
  public early, and spacing helps distribution.
- Clip length: 35–90s is the sweet spot; over ~110s hurts completion.
