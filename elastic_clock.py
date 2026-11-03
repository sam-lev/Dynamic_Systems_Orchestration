# elastic_clock.py -- Audible Lava / DSO v4.2
# Elastic (projective) hierarchical clock.
#
# A rigid clock quantizes: an onset at 0.37 s into a 1 s beat snaps to 0.5 s.
# This clock PULLS onsets partway toward the nearest grid line:
#     t' = t + k * (grid(t) - t),   k = elastic strength in [0, 1]
# with k governed externally (the Kuramoto synchronization order parameter of
# the blobs' vertical oscillations: locked blobs -> gridded music, free blobs
# -> events fall where the physics put them). Hasty's projective meter is the
# warrant: meter is a renegotiated expectation, not a lattice.
#
# Hierarchy: bar > beat > subdivision. One pattern cycle = one bar.
# London's perceptual bounds clamp everything: tactus 0.4-2.0 s, subdivision
# >= 0.1 s, bar <= 6 s.
#
# v4.9.3: set_period() preserves BAR phase (and therefore beat phase) rather
# than re-anchoring bar 0 to the current beat every call; see its docstring.

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ClockParams:
    period_s: float = 1.0        # tactus (beat) period
    beats_per_bar: int = 4       # meter cardinality
    subdiv: int = 4              # subdivisions per beat
    accent: tuple = (1.0, 0.6, 0.8, 0.6)   # per-beat accent profile
    elastic: float = 0.5         # pull strength k
    swing: float = 0.0           # 0..1 delay of off-subdivisions


class ElasticClock:
    TACTUS_MIN, TACTUS_MAX = 0.40, 2.00     # London: ~0.5-1.5 Hz felt pulse
    SUBDIV_MIN_S = 0.10
    BAR_MAX_S = 6.0

    def __init__(self, params: ClockParams | None = None, t0: float = 0.0):
        self.p = params or ClockParams()
        self.t0 = float(t0)             # seconds at which bar 0 / beat 0 started
        self._phase_offset_s = 0.0
        self.projection_s = self.p.period_s  # Hasty: last realized duration projects the next
        self.clamp()

    # -- parameterization (called by governors; smooth changes in period keep
    #    phase continuity by re-anchoring t0) --------------------------------
    def clamp(self) -> None:
        p = self.p
        p.period_s = float(min(self.TACTUS_MAX, max(self.TACTUS_MIN, p.period_s)))
        p.beats_per_bar = int(max(1, min(12, p.beats_per_bar)))
        p.subdiv = int(max(1, min(8, p.subdiv)))
        while p.subdiv > 1 and p.period_s / p.subdiv < self.SUBDIV_MIN_S:
            p.subdiv -= 1
        while p.beats_per_bar > 1 and p.period_s * p.beats_per_bar > self.BAR_MAX_S:
            p.beats_per_bar -= 1
        p.elastic = float(min(1.0, max(0.0, p.elastic)))
        if len(p.accent) != p.beats_per_bar:
            p.accent = tuple([1.0] + [0.6 if i % 2 else 0.8 for i in range(1, p.beats_per_bar)])

    def set_period(self, period_s: float, now_s: float) -> None:
        """Change tempo continuously: keep the current BAR phase fixed.

        v4.9.3: previously this re-anchored t0 to the start of the current
        BEAT (t0 = now - beat_phase * period). The governor calls it every
        frame, so bar 0 restarted every beat: bar_phase never left
        [0, 1/beats_per_bar), beat_index was always 0 (every beat read as the
        downbeat, so the accent profile and is_strong_beat were constant),
        and a pattern cycle -- one bar -- never advanced past its first beat,
        which is why the hit layer could only play a pattern's opening slots
        over and over. Anchoring on the bar phase keeps the beat phase fixed
        as well (beat_phase = frac(bar_phase * beats_per_bar)), so nothing
        that relied on beat continuity changes.
        """
        ph_bar = self.bar_phase(now_s)
        self.p.period_s = float(period_s)
        self.clamp()
        self.t0 = now_s - ph_bar * self.bar_len_s()
        self.projection_s = self.p.period_s

    def set_elastic(self, k: float) -> None:
        self.p.elastic = float(min(1.0, max(0.0, k)))

    def set_meter(self, beats_per_bar: int, accent: tuple | None = None) -> None:
        self.p.beats_per_bar = int(beats_per_bar)
        if accent is not None:
            self.p.accent = tuple(accent)
        self.clamp()

    # -- positions ---------------------------------------------------------
    def bar_len_s(self) -> float:
        return self.p.period_s * self.p.beats_per_bar

    def beat_phase(self, t_s: float) -> float:
        return ((t_s - self.t0) / self.p.period_s) % 1.0

    def bar_phase(self, t_s: float) -> float:
        return ((t_s - self.t0) / self.bar_len_s()) % 1.0

    def beat_index(self, t_s: float) -> int:
        return int(math.floor((t_s - self.t0) / self.p.period_s)) % self.p.beats_per_bar

    def accent_at(self, t_s: float) -> float:
        return self.p.accent[self.beat_index(t_s)]

    def seconds_to_cycles(self, t_s: float) -> float:
        return (t_s - self.t0) / self.bar_len_s()

    def cycles_to_seconds(self, c: float) -> float:
        return self.t0 + c * self.bar_len_s()

    def nearest_grid(self, t_s: float) -> float:
        step = self.p.period_s / self.p.subdiv
        n = round((t_s - self.t0) / step)
        g = self.t0 + n * step
        if self.p.swing and int(n) % 2 == 1:
            g += self.p.swing * step * 0.5
        return g

    def pull(self, t_s: float, strength: float | None = None) -> float:
        """Elastic attraction toward the nearest subdivision grid line."""
        k = self.p.elastic if strength is None else float(min(1.0, max(0.0, strength)))
        return t_s + k * (self.nearest_grid(t_s) - t_s)

    def is_strong_beat(self, t_s: float, tol: float = 0.12) -> bool:
        """True when t is within `tol` of a beat with accent >= 0.8
        (used by time-span rule (f): events on strong beats are heads)."""
        ph = self.beat_phase(t_s)
        near = min(ph, 1.0 - ph) < tol
        return near and self.accent_at(t_s) >= 0.8

    # -- Hasty projection ----------------------------------------------------
    def realize(self, ioi_s: float, now_s: float, weight: float = 0.3) -> float:
        """An observed inter-onset interval realizes (or denies) the current
        projection; the projection is nudged toward it. Returns realization
        score in [0,1] (1 = perfectly realized)."""
        if ioi_s <= 0:
            return 0.0
        ratio = ioi_s / self.projection_s
        real = max(0.0, 1.0 - abs(math.log2(ratio)))
        self.projection_s += weight * real * (ioi_s - self.projection_s)
        return real
