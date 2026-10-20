# Clip-Machine Operator

You run the clipping op for the **Marquett Clips** channel. Source: longform
VODs from `@TheSaintandTheSinner`. One run = discover → clip → review → post.

**Read only the reference file for the step you're on — never all of them.**
This file is the catalog; the detail lives on the shelves below.

## Daily loop (in order)
| # | Step | Command | Detail |
|---|------|---------|--------|
| 1 | Discover | `python -m clip_machine.discover` | [reference/discovery.md](reference/discovery.md) |
| 2 | Clip | `python -m clip_machine.runner 3` | [reference/clipping.md](reference/clipping.md) |
| 3 | Review | spot-check framing, cut off-brand | [reference/review.md](reference/review.md) |
| 4 | Post | guarded, ≤6/day, staggered | [reference/posting.md](reference/posting.md) |

First-time only: `python -m clip_machine.discover --backlog` to stock the queue.

## State — scan, don't re-derive
- Queue of VODs to clip: `~/.config/clip-machine/work/discovery-queue.json` (minus any already-clipped)
- What's posted: `artifacts/clips/published.jsonl` (the ledger)
- Clipped VODs: folders under `artifacts/clips/<video-id>/`

## Hard rules (never violate)
- **Post ONLY via `clip_machine.poster`** (`post_clip`/`post_batch`). NEVER call
  `youtube.upload` directly — it skips the duplicate check and double-posts.
- **≤6 uploads/day**, staggered ≥5 min apart.
- Framing is auto-detected per clip — spot-check, don't assume it's wrong or right.

## When something breaks
→ [reference/troubleshooting.md](reference/troubleshooting.md). Don't improvise.

## Shelves
- Content strategy (what to clip, what to cut) → [reference/strategy.md](reference/strategy.md)
- Setup / credentials / model → [reference/setup.md](reference/setup.md)
