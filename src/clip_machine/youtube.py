"""Upload a clip to YouTube as a Short, and record it in the ledger.

YouTube is the one platform that works from this machine with no domain and
no storage: it takes the bytes directly, and OAuth can redirect to
localhost. So it is the first real posting path, and proving it validates
the whole loop - auth, upload, ledger row, confirmed URL - before any cloud
infrastructure exists.

TWO THINGS TO KNOW BEFORE TRUSTING IT

1. QUOTA. A videos.insert costs 1600 units against a default 10,000/day.
   That is SIX uploads a day, and the quota resets on Pacific time, not
   yours. check_quota_budget() below turns that into a plain refusal
   instead of a 403 halfway through a batch.

2. PRIVACY. The worry was that an app whose OAuth consent screen Google
   has not verified would have uploads forced to `private` regardless of
   what was requested. MEASURED 2026-09-09 on an unverified app in testing
   mode: asked for `private`, got `private` - so the response honours the
   request and public posting does not appear to require verification
   first. Still worth reading back rather than assuming, because that is a
   policy that can change without notice, so upload() records the privacy
   status the API RETURNED, not the one it was given. A ledger row saying
   `private` after asking for `public` would be the verification gate
   appearing, not a bug here.

Secrets: the client secrets JSON and the stored token live under
~/.config/clip-machine/, outside the repo and outside OneDrive, because the
school tenant syncs everything in the project tree.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config, publish

# Uploading is a "sensitive" scope; this is the narrowest one that works.
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

CONFIG_DIR = Path.home() / ".config" / "clip-machine"
CLIENT_SECRETS = CONFIG_DIR / "youtube_client_secret.json"
TOKEN_FILE = CONFIG_DIR / "youtube_token.json"

# videos.insert cost and the default daily allowance, both from the API's
# published quota table.
UPLOAD_COST = 1600
DAILY_QUOTA = 10000
MAX_UPLOADS_PER_DAY = DAILY_QUOTA // UPLOAD_COST      # 6

# Shorts are just vertical uploads under this length. Nothing to opt into,
# but a clip over it silently becomes a regular video, which is worth
# refusing rather than discovering on the channel.
SHORTS_MAX_SECONDS = 180


class NotConfigured(RuntimeError):
    """No client secrets on disk yet."""


def configured() -> bool:
    return CLIENT_SECRETS.exists()


def _credentials():
    """Load stored credentials, refreshing or re-authorising as needed."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    if not CLIENT_SECRETS.exists():
        raise NotConfigured(
            f"Put your Google OAuth client secrets JSON at {CLIENT_SECRETS}\n"
            "  console.cloud.google.com -> enable YouTube Data API v3 ->\n"
            "  Credentials -> OAuth client ID -> Desktop app -> download JSON"
        )

    creds = None
    if TOKEN_FILE.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
        except Exception:
            creds = None

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save(creds)
            return creds
        except Exception as e:
            # A consent screen still in "Testing" expires refresh tokens
            # after about a week, so this path is normal, not exceptional.
            print(f"      (stored token could not refresh: {type(e).__name__}"
                  f" - asking for consent again)")
            creds = None

    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS), SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent")
    _save(creds)
    return creds


def _save(creds) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    try:                        # best effort; POSIX only
        TOKEN_FILE.chmod(0o600)
    except Exception:
        pass


def uploads_today() -> int:
    """How many YouTube uploads the ledger already records for today.

    Quota is per calendar day in US/Pacific, which is where Google counts
    it - using local midnight would let a late-evening batch spend
    tomorrow's quota and fail mid-run.
    """
    from datetime import datetime, timedelta, timezone

    pacific = timezone(timedelta(hours=-8))   # PST; an hour out during DST
    today = datetime.now(pacific).date().isoformat()
    return sum(1 for r in publish.history()
               if r.get("platform") == "youtube"
               and r.get("status") == "posted"
               and str(r.get("at", "")).startswith(today))


def check_quota_budget(n: int) -> tuple[bool, str]:
    """Can n more uploads fit in today's quota? (ok, human explanation)"""
    used = uploads_today()
    left = MAX_UPLOADS_PER_DAY - used
    if n <= left:
        return True, f"{used} of {MAX_UPLOADS_PER_DAY} uploads used today"
    return False, (
        f"YouTube allows about {MAX_UPLOADS_PER_DAY} uploads a day on the "
        f"default quota and {used} are already done. {left} left, "
        f"{n} requested - the rest have to wait for the reset."
    )


def upload(clip_path: Path, title: str, description: str,
           privacy: str = "private", video_id: str = "",
           clip_id: str = "", tags: list[str] | None = None,
           publish_at=None) -> dict:
    """Upload one clip. Records the outcome in the ledger either way.

    Defaults to `private` on purpose: a mistake that is visible to the
    world is not undoable, and flipping a video public afterwards takes one
    click. Nothing here posts publicly unless asked to.

    `publish_at` (a timezone-aware datetime) is how scheduling actually
    works, and it is why YouTube needs no cloud worker: the video uploads
    private now and YOUTUBE flips it public at that moment, so this machine
    only has to be awake long enough to upload. The API requires
    privacyStatus stay `private` for publishAt to be honoured - set
    anything else and the schedule is silently dropped, so it is forced
    here rather than trusted to the caller.
    """
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    cid = clip_id or publish.clip_id(video_id or "unknown", 0)

    if not clip_path.exists():
        publish.record(video_id, cid, clip_path.name, "youtube", "failed",
                       note="file missing")
        raise FileNotFoundError(clip_path)

    body = {
        "snippet": {
            "title": title[:100],          # hard API limit, silently rejected over
            "description": description[:5000],
            "tags": tags or [],
            "categoryId": "22",            # People & Blogs
        },
        "status": {
            "privacyStatus": "private" if publish_at else privacy,
            "selfDeclaredMadeForKids": False,
        },
    }
    if publish_at is not None:
        # RFC 3339, UTC. A naive datetime would be interpreted as UTC by
        # the API and publish at the wrong hour, so require an offset.
        if publish_at.tzinfo is None:
            raise ValueError("publish_at must be timezone-aware")
        body["status"]["publishAt"] = (
            publish_at.astimezone(timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"))

    try:
        youtube = build("youtube", "v3", credentials=_credentials(),
                        cache_discovery=False)
        media = MediaFileUpload(str(clip_path), chunksize=-1, resumable=True,
                                mimetype="video/mp4")
        req = youtube.videos().insert(part="snippet,status", body=body,
                                      media_body=media)
        response = req.execute()
    except HttpError as e:
        detail = getattr(e, "reason", "") or str(e)
        publish.record(video_id, cid, clip_path.name, "youtube", "failed",
                       note=f"HttpError {e.status_code}: {detail}"[:300])
        raise
    except Exception as e:
        publish.record(video_id, cid, clip_path.name, "youtube", "failed",
                       note=f"{type(e).__name__}: {e}"[:300])
        raise

    vid = response.get("id", "")
    url = f"https://youtube.com/shorts/{vid}" if vid else None
    scheduled = (response.get("status") or {}).get("publishAt")
    # What Google ACTUALLY set, not what we asked for - an unverified app
    # can have this forced to private, and the ledger should say so.
    got = (response.get("status") or {}).get("privacyStatus", "unknown")
    note = f"privacy={got}"
    if scheduled:
        note += f" publishAt={scheduled}"
    publish.record(video_id, cid, clip_path.name, "youtube", "posted",
                   url=url, note=note)
    return {"id": vid, "url": url, "privacy": got, "asked_for": privacy,
            "publish_at": scheduled}


# --------------------------------------------------------------------------
# scheduling
#
# The quota limit is on UPLOADS, not on publish times: six videos can leave
# this machine per day, but their publishAt can be spread over as many days
# as you like. So a 14-clip batch is three days of uploading and two weeks
# of posting - and it is worth saying that out loud rather than letting the
# seventh upload fail.
# --------------------------------------------------------------------------
DEFAULT_HOUR = 9           # first slot; later ones spread across the day
DEFAULT_PER_DAY = 3


SPAN_HOURS = 12     # same-day slots spread across this many hours


def slots_per_day(gap_hours: float, span_hours: int = SPAN_HOURS) -> int:
    """How many posts fit in a day at this spacing."""
    if gap_hours >= span_hours:
        return 1
    return int(span_hours // gap_hours) + (0 if span_hours % gap_hours else 0)         or 1


def plan_times(n: int, gap_hours: float = 4.0, hour: int = DEFAULT_HOUR,
               minute: int = 0, start_days: int = 1, tz=None,
               span_hours: int = SPAN_HOURS) -> list:
    """Publish times for n clips, `gap_hours` apart, starting tomorrow.

    The GAP is the input now, and posts-per-day falls out of it. It was the
    other way round - pick a count, have the spacing derived - which meant
    the one thing a person actually wants to control was read-only.

    Slots march forward by the gap until they would leave a `span_hours`
    daytime window, then wrap to the same start time the next day. Without
    that wrap, "every 4 hours" from 9am would schedule posts at 1am and
    5am, which is technically correct spacing and nobody's intent.

    Starts tomorrow, not today: a 9am slot requested at 11am is a time in
    the past, which the API rejects, and publishing it immediately instead
    would be worse than waiting a day.
    """
    tz = tz or datetime.now().astimezone().tzinfo
    now = datetime.now(tz)
    gap = max(0.25, float(gap_hours))
    per_day = slots_per_day(gap, span_hours)
    base = (now + timedelta(days=start_days)).replace(
        hour=hour, minute=minute, second=0, microsecond=0)
    out = []
    for i in range(n):
        day, slot = divmod(i, per_day)
        t = base + timedelta(days=day, hours=slot * gap)
        if t <= now:
            t = now + timedelta(minutes=20)
        out.append(t)
    return out


def schedule_batch(clips: list[dict], clip_dir: Path, video_id: str,
                   gap_hours: float = 4.0, hour: int = DEFAULT_HOUR,
                   minute: int = 0, privacy_after: str = "public") -> dict:
    """Upload as many clips as today's quota allows, each with a publish time.

    Returns what happened, including what did NOT fit - a scheduler that
    quietly drops half a batch is worse than one that says so.
    """
    already = publish.posted_pairs()
    todo = []
    for c in clips:
        cid = publish.clip_id(video_id, c.get("start", -1))
        if (cid, "youtube") in already:
            continue                      # never post the same clip twice
        todo.append((cid, c))

    used = uploads_today()
    room = max(0, MAX_UPLOADS_PER_DAY - used)
    batch, deferred = todo[:room], todo[room:]

    times = plan_times(len(batch), gap_hours=gap_hours, hour=hour,
                       minute=minute)
    done, failed = [], []
    for (cid, c), when in zip(batch, times):
        path = clip_dir / c["file"]
        try:
            res = upload(path, c.get("title", "")[:100],
                         c.get("description", ""), privacy="private",
                         video_id=video_id, clip_id=cid, publish_at=when)
            done.append({"file": c["file"], "url": res["url"],
                         "publish_at": res.get("publish_at"),
                         "title": c.get("title", "")})
        except Exception as e:
            failed.append({"file": c["file"],
                           "error": f"{type(e).__name__}: {e}"[:200]})

    return {
        "scheduled": done,
        "failed": failed,
        "deferred": [c["file"] for _, c in deferred],
        "already_posted": len(clips) - len(todo),
        "quota": f"{used + len(done)} of {MAX_UPLOADS_PER_DAY} uploads used today",
    }


def plan(clips: list[dict], video_id: str, gap_hours: float = 4.0,
         hour: int = DEFAULT_HOUR, minute: int = 0) -> dict:
    """What schedule_batch WOULD do. Uploads nothing.

    Exists so the confirm dialog can show real dates before anything
    irreversible happens - "Fri 11 Sep 9:00 AM" explains the cadence better
    than any label describing what "3 a day" means.
    """
    already = publish.posted_pairs()
    todo = [c for c in clips
            if (publish.clip_id(video_id, c.get("start", -1)), "youtube")
            not in already]
    used = uploads_today()
    room = max(0, MAX_UPLOADS_PER_DAY - used)
    now_batch = todo[:room]
    times = plan_times(len(now_batch), gap_hours=gap_hours, hour=hour,
                       minute=minute)
    # The gap is derived, not chosen: slots spread across a 12-hour span, so
    # it falls out of per_day. Worth returning because "3 a day" does not
    # tell you whether that means every 4 hours or three in a row.
    per_day = slots_per_day(gap_hours)
    # When the deferred ones become uploadable: the quota resets at midnight
    # US/Pacific, which is a different clock from the user's, so give the
    # actual local time rather than saying "tomorrow" and being wrong.
    pacific = timezone(timedelta(hours=-8))
    now_pac = datetime.now(pacific)
    reset_local = ((now_pac + timedelta(days=1))
                   .replace(hour=0, minute=0, second=0, microsecond=0)
                   .astimezone())
    return {
        "quota_resets": reset_local.isoformat(),
        "per_day": per_day,
        "gap_hours": gap_hours,
        "queued": len(clips),
        "already_posted": len(clips) - len(todo),
        "will_upload": len(now_batch),
        "deferred": max(0, len(todo) - room),
        "used_today": used,
        "max_per_day": MAX_UPLOADS_PER_DAY,
        "items": [
            {"file": c["file"],
             "title": (c.get("title") or "").split("#")[0].strip(),
             "when": t.isoformat()}
            for c, t in zip(now_batch, times)
        ],
    }
