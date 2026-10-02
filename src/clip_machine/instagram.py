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
