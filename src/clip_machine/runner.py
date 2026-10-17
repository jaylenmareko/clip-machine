"""Queue runner: clip the discovered VODs, one at a time, RAM-gated.

The auto-clip half of the pipeline. Pulls the discovery queue (highest-view
first), and for each VOD it hasn't clipped yet, checks free RAM and runs the
clip pipeline. One at a time on purpose - transcription is the memory
bottleneck, and this machine OOMs when it's starved - so it refuses to start a
job under MIN_FREE_GB rather than crash mid-run. Clipped VODs drop out
automatically next pass (the output folder is the done-marker).
"""
import json

from . import config, discover, pipeline, transcribe

MIN_FREE_GB = 2.5   # transcription needs ~3GB; below this it OOMs


def _pending() -> list[dict]:
    if not discover.QUEUE_FILE.exists():
        return []
    queue = json.loads(discover.QUEUE_FILE.read_text(encoding="utf-8"))
    return [v for v in queue if not discover.already_clipped(v["id"])]


def plan() -> None:
    """Show the queue, RAM status, and what the next run would clip. No work."""
    pending = _pending()
    free = transcribe.free_ram_gb() or 0
    print(f"free RAM: {free:.1f} GB (need >= {MIN_FREE_GB})")
    print(f"pending (unclipped) in queue: {len(pending)}")
    for i, v in enumerate(pending[:10]):
        mark = "-> next" if i == 0 and free >= MIN_FREE_GB else ""
        t = v["title"].encode("ascii", "ignore").decode()[:52]
        print(f"  {v['views']:>7,} | {v['duration_min']:>4}min | {v['id']} | {t} {mark}")
    if free < MIN_FREE_GB:
        print(f"\nWOULD SKIP: only {free:.1f} GB free — close apps / free RAM, then run.")


def run_queue(limit: int = 1, max_clips: int = 10) -> int:
    """Clip up to `limit` VODs from the queue. Returns how many it clipped."""
    done = 0
    for v in _pending():
        if done >= limit:
            break
        free = transcribe.free_ram_gb() or 0
        if free < MIN_FREE_GB:
            print(f"STOP: {free:.1f} GB free (< {MIN_FREE_GB}). Free RAM and re-run.")
            break
        t = v["title"].encode("ascii", "ignore").decode()[:50]
        print(f"CLIPPING {v['id']} ({v['duration_min']}min, {v['views']:,} views): {t}", flush=True)
        try:
            out = pipeline.run(v["url"], max_clips=max_clips, mode="debate")
            print(f"  DONE -> {out}", flush=True)
            done += 1
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:150]}", flush=True)
    print(f"clipped {done} VOD(s) this run")
    return done


if __name__ == "__main__":
    import sys
    if "--plan" in sys.argv:
        plan()
    else:
        n = 1
        for a in sys.argv[1:]:
            if a.isdigit():
                n = int(a)
        run_queue(limit=n)
