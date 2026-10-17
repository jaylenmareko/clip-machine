"""Idempotent, quota-aware posting — the guard that stops the duplicate bug.

Every post goes through post_clip(), which refuses to re-upload a clip_id that's
already on YouTube (the check a direct youtube.upload() skips, which is how the
10/17 duplicates happened) and refuses once the daily cap is hit. post_batch()
schedules a list staggered, counting only what actually went up against quota.
"""
import datetime
import json

from . import config, publish, youtube


def post_clip(video_id: str, file: str, when=None) -> tuple[str, str]:
    """Post one clip unless it's already up or over quota. (status, url_or_reason)."""
    mpath = config.CLIPS_DIR / video_id / "manifest.json"
    if not mpath.exists():
        return ("missing", "no manifest")
    man = json.loads(mpath.read_text(encoding="utf-8-sig"))
    clip = next((c for c in man["clips"] if c["file"] == file), None)
    if not clip:
        return ("missing", "no such clip")

    cid = publish.clip_id(video_id, clip.get("start", -1))
    # IDEMPOTENCY GUARD — the whole point of this module.
    if (cid, "youtube") in publish.posted_pairs():
        return ("skip", "already posted")
    if youtube.uploads_today() >= youtube.MAX_UPLOADS_PER_DAY:
        return ("quota", f"daily cap {youtube.MAX_UPLOADS_PER_DAY} reached")

    res = youtube.upload(
        config.CLIPS_DIR / video_id / file,
        publish.youtube_title(clip["title"]),
        publish.full_description(clip, man.get("footer", "")),
        privacy="public", video_id=video_id, clip_id=cid,
        tags=publish.hashtag_list(clip["title"]), publish_at=when)
    return ("posted", res.get("url") or "")


def post_batch(items: list[tuple[str, str]], start_offset_min: int = 16,
               spacing_min: int = 5) -> list[tuple]:
    """Schedule (video_id, file) pairs staggered. Idempotent + quota-aware.

    First go-live is start_offset_min out (YouTube needs >=15), each subsequent
    spacing_min later. Only successfully-posted clips advance the clock.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    results, i = [], 0
    for video_id, file in items:
        when = now + datetime.timedelta(minutes=start_offset_min + spacing_min * i)
        status, info = post_clip(video_id, file, when)
        results.append((video_id, file, status, info))
        if status == "posted":
            i += 1
        elif status == "quota":
            break
    return results
