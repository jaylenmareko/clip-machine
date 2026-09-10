"""The publish bundle and the publish ledger.

Two things live here, and they are deliberately the same shape whether a
human or a worker does the posting.

WHY A BUNDLE AND NOT DIRECT POSTING. Posting from this machine cannot work
for three specific reasons, none of which are about the API calls:

  - Instagram does not accept bytes. You hand Meta a public HTTPS URL and
    it fetches the file, so a local path is unreachable by definition.
  - TikTok's OAuth redirect must be HTTPS, and direct (non-private) posting
    needs the app audited.
  - The laptop has to be awake at 9am. A scheduler that only fires when the
    lid is open is not a scheduler.

YouTube is the exception - direct upload, localhost OAuth - but its quota
is ~6 uploads a day, so it does not carry the whole thing either.

So the bundle is what leaves this machine: clip files plus a queue.json
describing what to post where. Today a person feeds it to YouTube Studio,
Meta Business Suite or Buffer, all of which schedule natively for free with
no app review. Later a cron worker reads THE SAME queue.json out of object
storage. That is the only reason this is a file format and not a function
call - the format is the seam between the two, so option A is not thrown
away to get option B.

CLIP IDENTITY. Entries are keyed by clip_id = "<video_id>:<start>", never
by filename. Clip filenames are positional (clip_03.mp4) and a re-run
overwrites them - that already silently re-pointed five of Jaylen's
removals at unrelated clips once. A ledger with that bug would double-post
or skip real clips.
"""
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config

SCHEMA = 1

# Where a clip can go. Data, not hardcoded branches, so adding a platform is
# a list entry plus a worker handler.
PLATFORMS = ("youtube", "instagram", "tiktok")

# One line per publish attempt, appended forever. Append-only because a
# worker killed mid-post must not corrupt the record, and because the useful
# question is "what went out and when", which is history, not current state.
LEDGER = config.CLIPS_DIR / "published.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clip_id(video_id: str, start: float) -> str:
    """Stable identity for a clip: the source plus where it starts."""
    return f"{video_id}:{float(start):.1f}"


def split_title(text: str) -> tuple[str, str]:
    """('the title', '#tags #here') - hashtags are one field on an upload
    but separate columns are what a scheduler's paste field wants."""
    raw = (text or "").strip()
    m = re.search(r"(^|\s)#\S", raw)
    if not m:
        return raw, ""
    cut = m.start() if m.group(1) == "" else m.start() + 1
    return raw[:cut].strip(), raw[cut:].strip()


# --------------------------------------------------------------------------
# ledger
# --------------------------------------------------------------------------
def record(video_id: str, cid: str, file: str, platform: str, status: str,
           url: str | None = None, note: str = "") -> dict:
    """Append one publish event. Returns the row written."""
    row = {"at": _now(), "video_id": video_id, "clip_id": cid, "file": file,
           "platform": platform, "status": status, "url": url, "note": note}
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def history() -> list[dict]:
    """Every ledger row. A bad line is skipped, not fatal."""
    if not LEDGER.exists():
        return []
    out = []
    for line in LEDGER.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def posted_pairs() -> set[tuple[str, str]]:
    """{(clip_id, platform)} that reached 'posted'.

    The idempotency check. An unattended scheduler that loses its place
    double-posts, and a double post is publicly visible and not undoable.
    """
    return {(r.get("clip_id"), r.get("platform"))
            for r in history() if r.get("status") == "posted"}


# --------------------------------------------------------------------------
# bundle
# --------------------------------------------------------------------------
def build_queue(video_id: str, clips: list[dict], source: str = "",
                platforms=PLATFORMS) -> dict:
    """The publish plan. This is the file a worker would read."""
    already = posted_pairs()
    items = []
    for c in clips:
        cid = clip_id(video_id, c.get("start", -1))
        title, tags = split_title(c.get("title", ""))
        # Never re-offer something already posted to a platform.
        targets = [p for p in platforms if (cid, p) not in already]
        items.append({
            "clip_id": cid,
            "file": f"clips/{c['file']}",
            "start": round(float(c.get("start", 0)), 2),
            "duration": round(float(c.get("duration", 0)), 2),
            "title": title,
            "hashtags": tags,
            "description": c.get("description", ""),
            "platforms": targets,
            "status": "pending",
            # Left null on purpose: the app does not pick posting times.
            # Cadence is the user's call under option A and the worker's
            # config under option B; inventing times here would be a
            # decision pretending to be data.
            "scheduled_for": None,
        })
    return {"schema": SCHEMA, "video_id": video_id, "source": source,
            "created": _now(), "platforms": list(platforms), "items": items}


def csv_rows(queue: dict) -> str:
    """Human-readable half of the bundle - paste into a scheduler."""
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["file", "title", "hashtags", "description",
                "duration_seconds", "source_timestamp", "platforms"])
    for it in queue["items"]:
        mins, secs = divmod(int(it["start"]), 60)
        w.writerow([Path(it["file"]).name, it["title"], it["hashtags"],
                    it["description"], f"{it['duration']:.0f}",
                    f"{mins}:{secs:02d}", " ".join(it["platforms"])])
    return buf.getvalue()


_README = """ClipYourVideos.com - publish bundle
===================================

{n} clips, ready to schedule.

WHAT IS IN HERE
  clips/         the video files
  publish.csv    titles, hashtags and descriptions - open in a spreadsheet
  queue.json     the same thing for software to read

TO SCHEDULE THEM BY HAND (free, no setup)
  YouTube    Studio > Create > Upload, then set Visibility to Scheduled
  Instagram  Meta Business Suite > Create Reel > Schedule
  TikTok     tiktok.com/upload on desktop, then Schedule

  All three schedule natively at no cost. Copy the title and description
  for each clip from publish.csv.

  Hashtags belong at the end of the title or the caption - they are split
  into their own column only so you can paste them separately if you want.

A NOTE ON WHY THIS IS A FOLDER AND NOT AUTOMATIC
  Instagram will not accept an uploaded file - it fetches the video from a
  public HTTPS URL, which a local machine does not have. TikTok needs an
  HTTPS OAuth redirect and an audited app to post publicly. And any
  schedule that runs on this laptop only fires while the laptop is awake.

  queue.json is the same format a cloud worker will read later, so nothing
  here is wasted when that exists.

Generated {when} from:
{source}
"""


def readme(queue: dict) -> str:
    return _README.format(n=len(queue["items"]), when=queue["created"],
                          source=queue.get("source") or "(local file)")


def bundle(video_id: str, clips: list[dict], source: str,
           out_dir: Path) -> Path:
    """Write the zip a person downloads. Records the export in the ledger."""
    import zipfile

    queue = build_queue(video_id, clips, source)
    stamp = time.strftime("%Y-%m-%d")
    path = out_dir / f"{video_id}-publish-{stamp}.zip"
    clip_dir = config.CLIPS_DIR / video_id

    # Text compresses; H.264 does not. Measured on 30MB of clips: DEFLATE
    # took 1.1s to save 0.9%, STORED took 0.1s. So the videos are stored
    # and only the three text files are deflated.
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        z.writestr("queue.json", json.dumps(queue, indent=2),
                   compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("publish.csv", csv_rows(queue),
                   compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("README.txt", readme(queue),
                   compress_type=zipfile.ZIP_DEFLATED)
        for it in queue["items"]:
            src = clip_dir / Path(it["file"]).name
            if src.exists():
                z.write(src, it["file"])

    for it in queue["items"]:
        record(video_id, it["clip_id"], Path(it["file"]).name,
               "export", "exported",
               note=f"bundle {path.name}")
    return path


def by_file(video_id: str) -> dict:
    """{clip filename: latest publish state} for one project.

    Feeds the card badges. The ledger is append-only and per EVENT, so a
    clip can have a failure and then a success; the last row per
    (file, platform) wins, which is the state that is actually true now.
    """
    out: dict[str, dict] = {}
    for r in history():
        if r.get("video_id") != video_id or r.get("platform") == "export":
            continue
        f = r.get("file")
        if not f:
            continue
        # publishAt was recorded inside the free-text note; parse it back
        # rather than migrate the file, so older rows still read correctly.
        when = None
        note = r.get("note") or ""
        if "publishAt=" in note:
            when = note.split("publishAt=", 1)[1].split()[0]
        out[f] = {"platform": r.get("platform"), "status": r.get("status"),
                  "url": r.get("url"), "publish_at": when, "at": r.get("at")}
    return out
