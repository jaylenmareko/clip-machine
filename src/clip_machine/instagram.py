"""Post a clip to Instagram, and record it in the ledger.

Instagram's Graph API will not accept bytes - it fetches the video from a
public HTTPS URL you give it. So posting here is two calls, not one:
upload the clip to Cloudflare R2 first (an S3-compatible bucket, public
read), then hand Meta that URL.

Secrets: the long-lived token lives at ~/.config/clip-machine/, outside the
repo and outside OneDrive, same as youtube_token.json. R2 and Meta app
credentials come from ~/.config/clip-machine/.env, same as every other key
this project uses (config.py loads it).
"""
import json
import time
from pathlib import Path

import requests

from . import config, publish

CONFIG_DIR = Path.home() / ".config" / "clip-machine"
TOKEN_FILE = CONFIG_DIR / "instagram_token.json"

GRAPH = "https://graph.facebook.com/v19.0"


class NotConfigured(RuntimeError):
    """R2 or Meta credentials aren't set, or there's no token yet."""


def configured() -> bool:
    return bool(config.R2_BUCKET and config.META_APP_ID and config.IG_USER_ID)


def _r2_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=f"https://{config.R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=config.R2_ACCESS_KEY_ID,
        aws_secret_access_key=config.R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def upload_to_r2(clip_path: Path) -> str:
    """Push a clip to R2, return its public URL.

    The key includes a timestamp so re-posting after a Fix framing re-cut
    (same filename, new bytes) doesn't serve a stale cached copy from
    whatever edge cache sits in front of the public URL.
    """
    if not config.R2_BUCKET:
        raise NotConfigured("R2 isn't configured - see .env")
    key = f"{int(time.time())}-{clip_path.name}"
    _r2_client().upload_file(
        str(clip_path), config.R2_BUCKET, key,
        ExtraArgs={"ContentType": "video/mp4"})
    base = config.R2_PUBLIC_BASE_URL.rstrip("/")
    return f"{base}/{key}"


def _load_token() -> dict | None:
    if not TOKEN_FILE.exists():
        return None
    try:
        return json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_token(access_token: str, expires_in_seconds: int) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(json.dumps({
        "access_token": access_token,
        "expires_at": time.time() + expires_in_seconds,
    }), encoding="utf-8")
    try:
        TOKEN_FILE.chmod(0o600)
    except Exception:
        pass


def seed_token(access_token: str) -> None:
    """One-time: hand it the long-lived token from Phase A's manual steps."""
    _save_token(access_token, expires_in_seconds=60 * 24 * 3600)


def _refresh(access_token: str) -> dict:
    resp = requests.get(f"{GRAPH}/oauth/access_token", params={
        "grant_type": "fb_exchange_token",
        "client_id": config.META_APP_ID,
        "client_secret": config.META_APP_SECRET,
        "fb_exchange_token": access_token,
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _access_token() -> str:
    """A valid token, refreshing if it's within 3 days of expiring.

    Checked inline on every post attempt rather than on a schedule - there
    is no scheduler in this MVP, and a token only needs to be valid at the
    moment it's used.
    """
    tok = _load_token()
    if not tok:
        raise NotConfigured(
            f"No Instagram token yet. Run instagram.seed_token(TOKEN) once "
            f"with the long-lived token from Meta's Graph API Explorer.")
    if tok["expires_at"] - time.time() > 3 * 24 * 3600:
        return tok["access_token"]
    try:
        data = _refresh(tok["access_token"])
        _save_token(data["access_token"], data.get("expires_in", 60 * 24 * 3600))
        return data["access_token"]
    except Exception:
        # Refresh failed - the stored token might still be good for a few
        # more days. Use it rather than fail a post over a flaky refresh.
        return tok["access_token"]


POLL_INTERVAL_S = 5
POLL_TIMEOUT_S = 180     # Reels can take a couple minutes to process


def post(clip_path: Path, caption: str, video_id: str = "",
         clip_id: str = "") -> dict:
    """Upload a clip to R2, then post it to Instagram as a Reel.

    Records the outcome in the ledger either way - same contract as
    youtube.upload(): a caller can trust publish.by_file() afterward
    without separately tracking what happened here.
    """
    cid = clip_id or publish.clip_id(video_id or "unknown", 0)

    if not configured():
        publish.record(video_id, cid, clip_path.name, "instagram", "failed",
                       note="not configured")
        raise NotConfigured("Instagram isn't configured - see .env")
    if not clip_path.exists():
        publish.record(video_id, cid, clip_path.name, "instagram", "failed",
                       note="file missing")
        raise FileNotFoundError(clip_path)

    try:
        token = _access_token()
        video_url = upload_to_r2(clip_path)

        create = requests.post(
            f"{GRAPH}/{config.IG_USER_ID}/media",
            data={"media_type": "REELS", "video_url": video_url,
                 "caption": caption[:2200], "access_token": token},
            timeout=60)
        create.raise_for_status()
        container_id = create.json()["id"]

        deadline = time.time() + POLL_TIMEOUT_S
        status = None
        while time.time() < deadline:
            check = requests.get(f"{GRAPH}/{container_id}", params={
                "fields": "status_code", "access_token": token}, timeout=30)
            check.raise_for_status()
            status = check.json().get("status_code")
            if status == "FINISHED":
                break
            if status == "ERROR":
                raise RuntimeError("Instagram failed to process the video")
            time.sleep(POLL_INTERVAL_S)
        else:
            raise TimeoutError(
                f"Instagram didn't finish processing within {POLL_TIMEOUT_S}s")

        publish_resp = requests.post(
            f"{GRAPH}/{config.IG_USER_ID}/media_publish",
            data={"creation_id": container_id, "access_token": token},
            timeout=60)
        publish_resp.raise_for_status()
        media_id = publish_resp.json().get("id", "")

    except Exception as e:
        publish.record(video_id, cid, clip_path.name, "instagram", "failed",
                       note=f"{type(e).__name__}: {e}"[:300])
        raise

    url = f"https://www.instagram.com/reel/{media_id}/" if media_id else None
    publish.record(video_id, cid, clip_path.name, "instagram", "posted",
                   url=url, note=f"container={container_id}")
    return {"id": media_id, "url": url}
