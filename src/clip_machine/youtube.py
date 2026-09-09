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
           clip_id: str = "", tags: list[str] | None = None) -> dict:
    """Upload one clip. Records the outcome in the ledger either way.

    Defaults to `private` on purpose: a mistake that is visible to the
    world is not undoable, and flipping a video public afterwards takes one
    click. Nothing here posts publicly unless asked to.
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
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

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
    # What Google ACTUALLY set, not what we asked for - an unverified app
    # can have this forced to private, and the ledger should say so.
    got = (response.get("status") or {}).get("privacyStatus", "unknown")
    publish.record(video_id, cid, clip_path.name, "youtube", "posted",
                   url=url, note=f"privacy={got}")
    return {"id": vid, "url": url, "privacy": got, "asked_for": privacy}
