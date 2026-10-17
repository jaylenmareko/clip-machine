"""Auto-discovery: find the most popular VODs from the source channel to clip.

Removes the manual "go find the longform content" step. Queries the source
channel, keeps the longform it hasn't clipped yet, ranks by views (most popular
first, any topic), and writes a queue the runner can pull from. Read-only
against YouTube (Data API key), so it's safe to schedule — the memory-heavy work
is the clip pipeline, which the runner gates separately.
"""
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

from . import config

CHANNEL_HANDLE = "TheSaintandTheSinner"

MIN_DURATION_S = 15 * 60          # longform only; shorts/reels aren't sources
QUEUE_FILE = config.WORK_DIR / "discovery-queue.json"
API = "https://www.googleapis.com/youtube/v3/"


def _api_key() -> str:
    """The Data API key, from the same place the rest of the tool reads it."""
    env = Path.home() / ".config" / "clip-machine" / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("YOUTUBE_API_KEY="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("YOUTUBE_API_KEY", "")


def _slug(raw_id: str) -> str:
    """Mirror download._slugify so we can tell what's already been clipped."""
    text = re.sub(r"[^\w\-]+", "-", raw_id.strip().lower())
    return re.sub(r"-+", "-", text).strip("-")[:60] or "video"


def _get(endpoint: str, **params) -> dict:
    params["key"] = _api_key()
    url = API + endpoint + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def _duration_s(iso: str) -> int:
    secs = 0
    for n, u in re.findall(r"(\d+)([HMS])", iso):
        secs += int(n) * {"H": 3600, "M": 60, "S": 1}[u]
    return secs


def already_clipped(video_id: str) -> bool:
    return (config.CLIPS_DIR / _slug(video_id)).exists()


def discover(scan: int = 60) -> list[dict]:
    """On-vein longform VODs not yet clipped, ranked by views. Writes the queue.

    `scan` is how many of the channel's most recent uploads to look through -
    60 catches new drops plus the recent backlog; bump it to sweep deeper for a
    one-time backlog clear.
    """
    ch = _get("channels", part="contentDetails", forHandle=CHANNEL_HANDLE)["items"][0]
    uploads = ch["contentDetails"]["relatedPlaylists"]["uploads"]

    ids, token = [], None
    while len(ids) < scan:
        page = _get("playlistItems", part="contentDetails", playlistId=uploads,
                    maxResults=50, **({"pageToken": token} if token else {}))
        ids += [i["contentDetails"]["videoId"] for i in page["items"]]
        token = page.get("nextPageToken")
        if not token:
            break
    ids = ids[:scan]

    out = []
    for i in range(0, len(ids), 50):
        items = _get("videos", part="snippet,contentDetails,statistics",
                     id=",".join(ids[i:i + 50]))["items"]
        for v in items:
            title = v["snippet"]["title"]
            dur = _duration_s(v["contentDetails"]["duration"])
            if dur >= MIN_DURATION_S and not already_clipped(v["id"]):
                out.append({
                    "id": v["id"],
                    "url": f"https://youtu.be/{v['id']}",
                    "title": title,
                    "duration_min": round(dur / 60),
                    "views": int(v["statistics"].get("viewCount", 0)),
                    "published": v["snippet"]["publishedAt"][:10],
                })
    out.sort(key=lambda x: -x["views"])
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE_FILE.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def backlog(pages: int = 3) -> list[dict]:
    """The channel's MOST-VIEWED longform VODs not yet clipped, any topic.

    discover() only reaches recent uploads; this channel posts many per day, so
    its biggest hits fall out of range. search.list ordered by viewCount returns
    the top videos across the whole channel - a one-time backlog sweep for the
    most popular source material, regardless of subject.
    """
    ch = _get("channels", part="id", forHandle=CHANNEL_HANDLE)["items"][0]
    chid = ch["id"]
    hits: dict[str, None] = {}
    token = None
    for _ in range(pages):
        res = _get("search", part="id", channelId=chid, type="video",
                   maxResults=50, order="viewCount",
                   **({"pageToken": token} if token else {}))
        for it in res.get("items", []):
            hits[it["id"]["videoId"]] = None
        token = res.get("nextPageToken")
        if not token:
            break

    out = []
    ids = list(hits)
    for i in range(0, len(ids), 50):
        items = _get("videos", part="snippet,contentDetails,statistics",
                     id=",".join(ids[i:i + 50]))["items"]
        for v in items:
            title = v["snippet"]["title"]
            dur = _duration_s(v["contentDetails"]["duration"])
            if dur >= MIN_DURATION_S and not already_clipped(v["id"]):
                out.append({
                    "id": v["id"],
                    "url": f"https://youtu.be/{v['id']}",
                    "title": title,
                    "duration_min": round(dur / 60),
                    "views": int(v["statistics"].get("viewCount", 0)),
                    "published": v["snippet"]["publishedAt"][:10],
                })
    out.sort(key=lambda x: -x["views"])
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE_FILE.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    import sys
    found = backlog() if "--backlog" in sys.argv else discover()
    print(f"{len(found)} VOD(s) to clip, most popular first (queue -> {QUEUE_FILE}):\n")
    for v in found:
        t = v["title"].encode("ascii", "ignore").decode()[:60]
        print(f"{v['views']:>8,} | {v['duration_min']:>4}min | {v['published']} | {v['id']} | {t}")
