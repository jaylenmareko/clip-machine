# Step 3 — Review (human/agent gate before posting)

Framing is auto-detected and usually right, but spot-check before anything
posts. This is the one judgment step.

**Per clip, confirm two things:**
1. **Framing** — open the clip (or its `clip_NN.jpg` thumbnail). The speaker
   should be cleanly framed, not half-cut, not duplicated, no black bar or chat
   panel filling half the screen. If it's wrong, refit:
   `POST /api/refit {video_id, layout}` or re-render with the right layout —
   `center` (one full camera), `split` (two people stacked), `left`/`right`
   (camera on one side + chat/dead space on the other → crop to the speaker).
2. **On-brand** — cut anything per [strategy.md](strategy.md) (slurs, explicit
   white-nationalist framing, pure meta). When unsure, skip it.

Titles are point-based by design — leave them unless one is inaccurate to the
clip's content.

**Output:** the set of clips you've OK'd to post. Nothing moves to posting until
it's passed this gate.
