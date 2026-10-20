# Troubleshooting

**A VOD won't download (yt-dlp error).** The runner skips it and moves on.
Usually a live/members/age-restricted video. Don't hand over browser cookies to
fix one video — just skip it and clip the next most-popular one.

**Transcription OOM-crashes / job dies with no output.** Free RAM was too low.
Close other apps to get >3 GB free, then re-run. The runner's RAM gate prevents
most of this, but a very long VOD (2–4 h) can still grow past it mid-run — clip
shorter VODs when memory is tight.

**Upload hangs for minutes, then fails.** The network is blocking YouTube's
upload endpoint (seen on restricted/institutional Wi-Fi). Auth still works, only
the media PUT stalls. Fix: run on an open connection. Nothing posted on a hung
upload — no partial video is created.

**Duplicate posts appeared.** Something called `youtube.upload` directly instead
of `poster`. Only post through `poster` (see posting.md). The token can't delete
videos (upload-only scope) — duplicates must be removed by hand in YouTube Studio.

**Uploads suddenly all fail with auth errors after ~a week.** The OAuth consent
is in "Testing" mode, so its refresh token expired. The channel owner must
re-authorize (a browser consent only they can do). Ask them to re-run the
YouTube auth; then uploads resume.

**Titles have no names.** Intended — point-based titles avoid fabricating who
said what (no speaker labels in the transcript). Not a bug; don't "fix" it.
