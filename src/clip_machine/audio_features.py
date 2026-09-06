"""Acoustic energy timeline — the signal a keyword list can't see.

Excitement is physical: the speaker gets louder and faster. We get that
for free from ffmpeg's astats filter (no ML, no extra deps, and ffmpeg
does the arithmetic so there's no per-sample Python loop).
"""
import re
import subprocess
from pathlib import Path

# astats resets every N audio frames; at 8kHz a frame is 1024 samples,
# so reset=4 gives roughly one RMS reading every ~0.5s.
_RESET_FRAMES = 4

_PTS = re.compile(r"pts_time:([0-9.]+)")
_RMS = re.compile(r"lavfi\.astats\.Overall\.RMS_level=(-?[0-9.]+|-inf)")


def energy_timeline(audio_path: Path) -> list[tuple[float, float]]:
    """Return [(timestamp_seconds, rms_db), ...]. Empty list on failure."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(audio_path),
        "-af", f"aresample=8000,astats=metadata=1:reset={_RESET_FRAMES},"
               f"ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
        "-f", "null", "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900).stdout
    except Exception:
        return []

    timeline: list[tuple[float, float]] = []
    t = None
    for line in out.splitlines():
        if m := _PTS.search(line):
            t = float(m.group(1))
        elif (m := _RMS.search(line)) and t is not None:
            raw = m.group(1)
            if raw != "-inf":
                timeline.append((t, float(raw)))
            t = None
    return timeline


def _pstdev(vals: list[float]) -> float:
    avg = sum(vals) / len(vals)
    return (sum((v - avg) ** 2 for v in vals) / len(vals)) ** 0.5


def _percentile_rank(sorted_vals: list[float], x: float) -> float:
    """Fraction of the distribution below x — 0..1."""
    lo, hi = 0, len(sorted_vals)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_vals[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo / max(len(sorted_vals), 1)


class EnergyProfile:
    """Normalizes a clip's loudness against the speaker's own baseline.

    Absolute dB is meaningless across recordings — what matters is whether
    THIS moment is hot relative to how this person normally talks. Every
    feature is a percentile against the same video, so there are no tuned
    constants that only hold for one recording.
    """

    _GRID = 30.0  # window size used to build the reference distributions

    def __init__(self, timeline: list[tuple[float, float]]):
        levels = sorted(db for _, db in timeline)
        self.ok = len(levels) >= 40
        if not self.ok:
            self.timeline = timeline
            self.speech = []
            return

        # Drop silence: it drags averages down and inflates variance, so
        # it would otherwise read as "dynamic range" rather than dead air.
        median_all = levels[len(levels) // 2]
        floor = median_all - 20.0
        self.speech = [(t, db) for t, db in timeline if db >= floor]
        self.timeline = timeline

        sp = sorted(db for _, db in self.speech)
        if len(sp) < 40:
            self.ok = False
            return
        self.median = sp[len(sp) // 2]
        self.p90 = sp[int(len(sp) * 0.90)]
        self.spread = max(self.p90 - self.median, 0.5)

        # Reference distributions from this video's own windows.
        span = self.speech[-1][0]
        raw_level, raw_dyn = [], []
        t = 0.0
        while t + self._GRID <= span + self._GRID:
            vals = self.window(t, t + self._GRID)
            if len(vals) >= 8:
                raw_level.append(sum(vals) / len(vals))
                raw_dyn.append(_pstdev(vals))
            t += self._GRID / 2  # 50% overlap for a denser reference
        self._level_dist = sorted(raw_level)
        self._dyn_dist = sorted(raw_dyn)

    def window(self, start: float, end: float) -> list[float]:
        return [db for t, db in self.speech if start <= t <= end]

    def excitement(self, start: float, end: float) -> float:
        """0..1 — how hot this window runs vs. the speaker's own baseline.

        Blends sustained level (is the whole clip energetic) with peak
        (does it contain a spike), so a flat-loud clip and a clip with one
        big moment both register.
        """
        if not self.ok:
            return 0.0
        vals = self.window(start, end)
        if len(vals) < 4:
            return 0.0
        avg = sum(vals) / len(vals)
        peak = max(vals)
        sustained = _percentile_rank(self._level_dist, avg)
        spike = max(0.0, min(1.0, (peak - self.p90) / self.spread + 0.5))
        return max(0.0, min(1.0, 0.7 * sustained + 0.3 * spike))

    def dynamics(self, start: float, end: float) -> float:
        """0..1 — loudness variation vs. this video's own windows.

        Monotone delivery ranks low, animated delivery ranks high. This is
        a weak signal on its own (measured spread across a talk was narrow)
        so callers should weight it lightly.
        """
        if not self.ok:
            return 0.0
        vals = self.window(start, end)
        if len(vals) < 8:
            return 0.0
        return _percentile_rank(self._dyn_dist, _pstdev(vals))
