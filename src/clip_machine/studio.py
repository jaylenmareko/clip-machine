"""Upload to YouTube through Studio in a real Chrome window.

The fallback for when the API's six uploads a day are spent. Studio uploads
are not metered by the Data API quota, so this keeps a card's Upload button
working after the sixth.

WHY A SEPARATE CHROME, DRIVEN OVER CDP. Google refuses sign-in in a browser
that Playwright launched itself ("This browser or app may not be secure") -
it reads the automation flags. So this starts ordinary Google Chrome with its
own profile and a debugging port, the person signs in by hand like any other
browser, and Playwright only ATTACHES afterwards. Credentials are never typed
by this code.

  - The profile lives in ~/.config/clip-machine/studio-chrome, outside
    OneDrive, because it holds a signed-in Google session.
  - The debugging port listens on 127.0.0.1 only. Anything on this machine
    that can reach it can drive that session while that Chrome is open.
    It is NOT closed after an upload: the same window turned out to hold
    the person's own tabs, and killing those would be worse. Each upload
    uses its own tab and closes only that tab, and only on success - a
    failed one stays open so the person can see where it stopped.

SELECTORS were mapped on 2026-09-13 against the live Studio, signed in, with
a real private upload - not recalled from older uploader scripts, several of
which use ids (#create-icon) that no longer exist.

KNOWN COSTS, accepted knowingly: YouTube's terms prohibit automated access,
so this runs visibly, one clip at a time, only on a click; and Studio's page
changes without notice, which breaks selectors. The API path is always used
first while quota remains.
"""
import os
import re
import socket
import subprocess
import time
from pathlib import Path

PROFILE = Path.home() / ".config" / "clip-machine" / "studio-chrome"
PORT = int(os.environ.get("CLIP_STUDIO_PORT", "9333"))
STUDIO = "https://studio.youtube.com"

_CHROME_CANDIDATES = [
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Google/Chrome/Application/chrome.exe",
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Google/Chrome/Application/chrome.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
]


class StudioError(RuntimeError):
    """Something a person needs to see - message is written for them."""


def chrome_path() -> Path:
    for p in _CHROME_CANDIDATES:
        if p.exists():
            return p
    raise StudioError("Google Chrome isn't installed, and Studio uploads need it.")


def _port_open() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def ensure_chrome(url: str = STUDIO, wait: float = 25.0) -> None:
    """Start the dedicated Chrome if it isn't already listening."""
    if _port_open():
        return
    PROFILE.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(
        [str(chrome_path()),
         f"--remote-debugging-port={PORT}",
         "--remote-debugging-address=127.0.0.1",
         f"--user-data-dir={PROFILE}",
         "--no-first-run", "--no-default-browser-check",
         url],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + wait
    while time.time() < deadline:
        if _port_open():
            return
        time.sleep(0.3)
    raise StudioError("Chrome didn't start. Close any stuck Chrome windows and try again.")


def connect(p):
    """(browser, context) attached to the dedicated Chrome."""
    browser = p.chromium.connect_over_cdp(f"http://127.0.0.1:{PORT}")
    ctx = browser.contexts[0] if browser.contexts else browser.new_context()
    return browser, ctx


def signed_in(page) -> bool:
    """Studio redirects a signed-out browser to accounts.google.com."""
    page.goto(STUDIO, wait_until="domcontentloaded")
    try:
        page.wait_for_url(lambda u: "studio.youtube.com" in u or "accounts.google" in u,
                          timeout=20000)
    except Exception:
        pass
    return "studio.youtube.com" in page.url and "accounts.google" not in page.url


UPLOAD_WAIT_SECONDS = 15 * 60
_UPLOADING = re.compile(r"Uploading\s*\.{0,3}\s*\d+\s*%|Upload(?:ing)?\s+\d+\s*%", re.I)


def _say(on_step, msg: str) -> None:
    if on_step:
        on_step(msg)


def _type_into(page, box, text: str, what: str) -> None:
    """Replace a Studio text box's contents, then prove it took."""
    box.click()
    page.keyboard.press("Control+A")
    page.keyboard.press("Delete")
    page.keyboard.insert_text(text)
    page.wait_for_timeout(300)
    if box.inner_text().strip() != text.strip():
        raise StudioError(f"Studio didn't accept the {what}.")


def _wait_for_upload(page, dialog) -> None:
    """Block until the file has finished uploading.

    Saving mid-upload is allowed by Studio, but the bytes are still coming
    from this Chrome - close the tab too early and the video never arrives.
    """
    deadline = time.time() + UPLOAD_WAIT_SECONDS
    while time.time() < deadline:
        text = dialog.first.inner_text() if dialog.count() else ""
        if not _UPLOADING.search(text):
            return
        page.wait_for_timeout(2000)
    raise StudioError("The upload took longer than 15 minutes. Check the Chrome window.")


def upload(clip_path: Path, title: str, description: str, tags: list[str],
           publish_at=None, on_step=None) -> dict:
    """Upload one clip through Studio. Returns {"url": ...}.

    publish_at None means public now; a timezone-aware datetime schedules it
    (Studio keeps it private until then). Every value typed is read back, and
    anything Studio did not accept stops the upload with a plain message
    rather than saving something half-filled.
    """
    from playwright.sync_api import sync_playwright

    clip_path = Path(clip_path)
    if not clip_path.exists():
        raise StudioError("That clip's video file is missing.")

    ensure_chrome()
    with sync_playwright() as p:
        _, ctx = connect(p)
        page = ctx.new_page()
        page.set_default_timeout(30000)
        ok = False
        try:
            _say(on_step, "Opening YouTube Studio…")
            if not signed_in(page):
                raise StudioError("Sign in to YouTube in the Chrome window that "
                                  "opened, then click Upload again.")
            page.bring_to_front()
            dialog = page.locator("ytcp-uploads-dialog")

            page.locator('button[aria-label="Create"]').first.click()
            page.locator('tp-yt-paper-item[test-id="upload"]').first.click()
            _say(on_step, "Sending the video to YouTube…")
            dialog.locator('input[name="Filedata"]').first.set_input_files(str(clip_path))

            title_box = dialog.locator('#textbox[aria-label^="Add a title"]').first
            title_box.wait_for(state="visible", timeout=120000)
            _say(on_step, "Filling in the title and description…")
            _type_into(page, title_box, title[:100], "title")
            _type_into(page, dialog.locator('#textbox[aria-label^="Tell viewers"]').first,
                       description[:5000], "description")

            kids = dialog.locator('tp-yt-paper-radio-button[name="VIDEO_MADE_FOR_KIDS_NOT_MFK"]').first
            kids.click()
            if kids.get_attribute("aria-checked") != "true":
                raise StudioError("Couldn't mark the video as not made for kids.")

            if tags:
                dialog.locator("#toggle-button").first.click()
                box = dialog.locator('input[aria-label="Tags"]').first
                box.wait_for(state="visible", timeout=15000)
                # Commas turn each entry into a tag chip - verified on the
                # live page. Studio caps tags at 500 characters in total.
                box.fill(",".join(tags)[:480] + ",")

            for _ in range(3):
                dialog.locator("#next-button").first.click()
                page.wait_for_timeout(1500)

            if publish_at is None:
                _say(on_step, "Setting it to public…")
                public = dialog.locator('tp-yt-paper-radio-button[name="PUBLIC"]').first
                if not public.is_visible():
                    # The Schedule section, once opened, collapses this one.
                    dialog.locator("#first-container-expand-button").first.click()
                public.click()
                if public.get_attribute("aria-checked") != "true":
                    raise StudioError("Couldn't set the video to public.")
            else:
                _say(on_step, "Setting the schedule…")
                # Studio's picker defaults to this computer's time zone.
                local = publish_at.astimezone()
                trigger = dialog.locator("#datepicker-trigger").first
                if not trigger.is_visible():
                    dialog.locator("#second-container-expand-button").first.click()
                    trigger.wait_for(state="visible", timeout=10000)
                trigger.click()
                date_text = f"{local:%b} {local.day}, {local.year}"
                date_box = page.locator("ytcp-date-picker tp-yt-paper-input input").first
                date_box.wait_for(state="visible", timeout=10000)
                date_box.fill(date_text)
                page.keyboard.press("Enter")
                page.wait_for_timeout(800)
                if date_text not in trigger.inner_text():
                    raise StudioError(f"Studio didn't accept the date {date_text}.")
                time_text = local.strftime("%I:%M %p").lstrip("0")
                time_box = dialog.locator("#time-of-day-container input").first
                time_box.fill(time_text)
                page.keyboard.press("Enter")
                page.wait_for_timeout(800)
                shown = time_box.input_value().replace(" ", " ").strip().upper()
                if shown != time_text.upper():
                    raise StudioError(f"Studio didn't accept the time {time_text}.")

            _say(on_step, "Waiting for the upload to finish…")
            _wait_for_upload(page, dialog)

            link = dialog.locator('a[href*="/shorts/"], a[href*="youtu.be/"]')
            url = link.first.get_attribute("href") if link.count() else None

            _say(on_step, "Saving…")
            dialog.locator("#done-button").first.click()
            page.locator("ytcp-uploads-dialog tp-yt-paper-dialog").first.wait_for(
                state="hidden", timeout=60000)
            # Publishing public can raise a "video published" share sheet.
            share_close = page.locator('ytcp-video-share-dialog [aria-label="Close"]')
            if share_close.count() and share_close.first.is_visible():
                share_close.first.click()
            ok = True
            return {"url": url}
        except StudioError:
            raise
        except Exception as e:
            raise StudioError("Studio upload stopped partway - check the Chrome "
                              f"window. ({type(e).__name__})") from e
        finally:
            if ok:
                try:
                    page.close()
                except Exception:
                    pass


def close_chrome() -> None:
    """Close the dedicated Chrome so the debugging port isn't left open."""
    if not _port_open():
        return
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser, _ = connect(p)
            browser.close()
    except Exception:
        pass


if __name__ == "__main__":
    # python -m clip_machine.studio signin
    #   Opens the dedicated Chrome on Studio so a person can sign in once.
    ensure_chrome()
    print(f"Chrome is open on {STUDIO} (profile: {PROFILE}). Sign in there.")
