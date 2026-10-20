# Step 4 — Post (guarded)

**THE ONE RULE: post only through `clip_machine.poster`. Never `youtube.upload`
directly.** The poster checks `posted_pairs()` first and refuses a clip that's
already up; a direct upload skips that and double-posts (it has happened — 5
duplicates in one run).

**Post one clip:**
```python
from clip_machine import poster
poster.post_clip(video_id, "clip_03.mp4", when=<tz-aware datetime | None>)
# -> ("posted", url) | ("skip","already posted") | ("quota","cap reached") | ("missing",...)
```

**Post a reviewed batch, staggered:**
```python
poster.post_batch([(vid,"clip_01.mp4"), (vid,"clip_03.mp4"), ...],
                  start_offset_min=16, spacing_min=5)
```
First go-live ≥15 min out (YouTube's minimum); each next +5 min. Only successful
posts count against quota.

**Limits:**
- **≤6 uploads/day**, counted on the US/Pacific calendar day (where Google
  counts quota). `youtube.uploads_today()` tells you how many are used.
- Scheduled posts upload **private** and YouTube flips them public at the set
  time — nothing is public before then.

**Network:** uploading needs open egress. If an upload hangs for minutes, the
network is blocking YouTube's upload endpoint — see troubleshooting.
