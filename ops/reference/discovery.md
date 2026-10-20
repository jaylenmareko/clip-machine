# Step 1 — Discover

**Does:** finds the channel's most-popular longform VODs not yet clipped, ranks
by views, writes the queue. Read-only (YouTube Data API) — cheap, safe to run often.

**Run:**
- Daily: `python -m clip_machine.discover` — recent uploads.
- One-time / to re-stock deep: `python -m clip_machine.discover --backlog` —
  searches the whole catalog by view count (the channel's all-time biggest).

**Writes:** `~/.config/clip-machine/work/discovery-queue.json` — a list of
`{id, url, title, duration_min, views, published}`, highest views first,
already-clipped VODs excluded.

**Check:** the queue should be non-empty and topped by high-view VODs. If it's
empty, everything popular is already clipped — fine, wait for new uploads.
