"""Double-click entry point for the Desktop shortcut.

The server has to be running before the page is any use, and until now
that meant asking Claude to start it - it gets killed by low memory, or a
person just closes the console window it was minimized to, and either way
nobody's typing `python -m clip_machine.server` themselves.

This checks whether something is already answering on the port; if not it
starts the server as its own DETACHED background process (so it outlives
this launcher) and waits for it to come up, then opens the browser either
way. Safe to double-click as many times as you like - it never starts a
second server on a port that is already bound.
"""
import subprocess
import sys
import time
import urllib.request
import webbrowser

from .server import PORT

URL = f"http://127.0.0.1:{PORT}"
START_TIMEOUT_S = 30


def _up() -> bool:
    try:
        urllib.request.urlopen(URL, timeout=1)
        return True
    except Exception:
        return False


def main() -> None:
    if not _up():
        # DETACHED_PROCESS + CREATE_NO_WINDOW: survives this launcher
        # exiting, and never flashes a console window - the whole point of
        # going through pythonw for this rather than a visible python.exe.
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
        subprocess.Popen(
            [sys.executable, "-m", "clip_machine.server"],
            creationflags=flags,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        deadline = time.time() + START_TIMEOUT_S
        while time.time() < deadline and not _up():
            time.sleep(0.5)
    webbrowser.open(URL)


if __name__ == "__main__":
    main()
