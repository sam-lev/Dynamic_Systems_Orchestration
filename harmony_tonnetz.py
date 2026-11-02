# harmony_tonnetz.py -- Audible Lava / DSO v4.2
# Neo-Riemannian harmonic layer: the harmonic field is a walk on the Tonnetz.
#
#   Triad(root_pc, major)          24 triads; pcs() gives its 3 pitch classes
#   P, L, R                        the three parsimonious transformations
#                                  (each moves one voice by 1-2 semitones)
#   tonnetz_distance / path        shortest P/L/R path between two triads
#   TonnetzWalker                  governed walk: WHEN to step (topological
#                                  change rate), WHICH move (tension/surprisal),
#                                  which DIRECTION (toward/away from home per
#                                  the distance-from-equilibrium gradient), and
#                                  the chord CARDINALITY (conditional entropy).
#   roughness_proxy                Plomp-Levelt-style sensory dissonance of a
#                                  pitch-class set (the non-tonal "dissonance").
#
# Lewin: governors emit transformations ("we went L then R"), never objects.

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


@dataclass(frozen=True)
class Triad:
    root: int          # pitch class 0-11
    major: bool

    def pcs(self) -> Tuple[int, int, int]:
        return (self.root % 12, (self.root + (4 if self.major else 3)) % 12, (self.root + 7) % 12)

    def P(self) -> "Triad":
        return Triad(self.root, not self.major)

    def L(self) -> "Triad":
        # major: root down a semitone -> relative minor's neighbour (C -> Em)
        return Triad((self.root + 4) % 12, False) if self.major else Triad((self.root + 8) % 12, True)

    def R(self) -> "Triad":
        return Triad((self.root + 9) % 12, False) if self.major else Triad((self.root + 3) % 12, True)

    def apply(self, op: str) -> "Triad":
        t = self
        for ch in op:
            t = {"P": t.P, "L": t.L, "R": t.R}[ch]()
        return t

    @property
    def name(self) -> str:
        return NOTE_NAMES[self.root % 12] + ("" if self.major else "m")

    def __repr__(self):
        return self.name


ALL_TRIADS = [Triad(r, m) for r in range(12) for m in (True, False)]
PRIMITIVE_OPS = ("P", "L", "R")
COMPOUND_OPS = ("PL", "LP", "PR", "RP", "LR", "RL")     # two-step moves


def _bfs(start: Triad) -> Dict[Triad, Tuple[int, str]]:
    seen = {start: (0, "")}
    q = deque([start])
    while q:
        t = q.popleft()
        d, path = seen[t]
        for op in PRIMITIVE_OPS:
            u = t.apply(op)
            if u not in seen:
                seen[u] = (d + 1, path + op)
                q.append(u)
    return seen


_PATHS: Dict[Triad, Dict[Triad, Tuple[int, str]]] = {t: _bfs(t) for t in ALL_TRIADS}


def tonnetz_distance(a: Triad, b: Triad) -> int:
    return _PATHS[a][b][0]


def tonnetz_path(a: Triad, b: Triad) -> str:
    """Shortest P/L/R word from a to b ('' if a == b)."""
    return _PATHS[a][b][1]


def voice_leading_cost(a: Triad, b: Triad) -> int:
    """Total semitone motion of the smoothest 3-voice mapping between triads."""
    pa, pb = sorted(a.pcs()), sorted(b.pcs())
    best = 99
    import itertools
    for perm in itertools.permutations(pb):
        c = sum(min((x - y) % 12, (y - x) % 12) for x, y in zip(pa, perm))
        best = min(best, c)
    return best


def roughness_proxy(pcs: Sequence[int], octave_spread: int = 1) -> float:
    """Sensory-dissonance proxy on pitch classes: sum over pairs of a
    Plomp-Levelt-shaped curve in semitone distance (minor 2nd / major 7th
    roughest, tritone moderately rough, perfect 5th/octave smooth).
    Returns [0,1] normalized by the pair count."""
    table = {0: 0.0, 1: 1.0, 2: 0.7, 3: 0.35, 4: 0.3, 5: 0.15, 6: 0.55,
             7: 0.1, 8: 0.32, 9: 0.28, 10: 0.5, 11: 0.85}
    pcs = list(pcs)
    if len(pcs) < 2:
        return 0.0
    tot, n = 0.0, 0
    for i in range(len(pcs)):
        for j in range(i + 1, len(pcs)):
            d = (pcs[i] - pcs[j]) % 12
            tot += table[min(d, 12 - d) if d != 6 else 6]; n += 1
    return tot / n


class TonnetzWalker:
    """Governed walk on the Tonnetz.

    step(when, how_far, direction) is called every frame with governor-derived
    quantities; the walker accumulates a step "budget" from `rate` (e.g.
    diagram-Wasserstein change rate) and moves when the budget crosses 1.

    move choice:
        distance_from_home gradient > 0 (moving away from equilibrium) ->
            prefer moves that INCREASE tonnetz distance to home
        gradient < 0 (returning) ->
            take the next step of the shortest path home
        size: tension/surprisal in [0,1] -> primitive (L smoothest, then P,
            then R) or compound (PL/RP...) when high
    cardinality: conditional entropy in [0,1] -> 3 (clean triad) up to 5
        pitch classes (added 7th/9th-type tones), raising roughness.
    """

    def __init__(self, home: Triad = Triad(0, True), step_threshold: float = 1.0,
                 seed: int = 3):
        import random
        self.home = home
        self.current = home
        self.budget = 0.0
        self.step_threshold = float(step_threshold)
        self.rng = random.Random(seed)
        self.history: deque = deque(maxlen=64)      # (triad, op) transformations taken
        self.last_op = ""
        self.predicted_next: Optional[Triad] = None

    def distance_home(self) -> int:
        return tonnetz_distance(self.current, self.home)

    def _candidates(self, size: float) -> List[str]:
        if size < 0.33:
            return ["L", "P"]
        if size < 0.66:
            return ["L", "P", "R"]
        return list(PRIMITIVE_OPS) + list(COMPOUND_OPS)

    def choose_op(self, size: float, direction: float) -> str:
        """direction: +1 away from home, -1 toward home, 0 free drift."""
        ops = self._candidates(size)
        if direction < -0.15 and self.current != self.home:
            path = tonnetz_path(self.current, self.home)
            step = path[:2] if size > 0.66 and len(path) >= 2 else path[:1]
            return step or "L"
        scored = []
        d0 = self.distance_home()
        for op in ops:
            t = self.current.apply(op)
            d1 = tonnetz_distance(t, self.home)
            if direction > 0.15:
                s = (d1 - d0)                 # prefer increasing distance
            else:
                s = -abs(d1 - d0) * 0.2       # free drift: mild preference to stay in orbit
            s += self.rng.random() * 0.3      # tie-break / exploration
            scored.append((s, op))
        scored.sort(reverse=True)
        return scored[0][1]

    def predict(self, size: float, direction: float) -> Triad:
        """The most likely next triad (for Imagination-phase pre-sounding)."""
        # deterministic preview: same logic without the random term
        ops = self._candidates(size)
        if direction < -0.15 and self.current != self.home:
            path = tonnetz_path(self.current, self.home)
            return self.current.apply(path[:1] or "L")
        d0 = self.distance_home()
        best = max(ops, key=lambda op: (tonnetz_distance(self.current.apply(op), self.home) - d0)
                   if direction > 0.15 else -abs(tonnetz_distance(self.current.apply(op), self.home) - d0))
        return self.current.apply(best)

    def update(self, rate: float, size: float, direction: float, dt: float) -> Optional[str]:
        """Accumulate step budget from `rate` (per second); move when it
        crosses the threshold. Returns the op taken (or None)."""
        self.budget += max(0.0, float(rate)) * float(dt)
        self.predicted_next = self.predict(size, direction)
        if self.budget < self.step_threshold:
            return None
        self.budget = 0.0
        op = self.choose_op(size, direction)
        self.current = self.current.apply(op)
        self.last_op = op
        self.history.append((self.current, op))
        return op

    def pitch_field(self, ambiguity: float) -> List[int]:
        """Allowed pitch classes: the triad, plus added tones as ambiguity
        (conditional entropy, [0,1]) rises."""
        pcs = list(self.current.pcs())
        extras = [(self.current.root + 10) % 12, (self.current.root + 2) % 12,
                  (self.current.root + 5) % 12]
        n_extra = 0 if ambiguity < 0.35 else (1 if ambiguity < 0.6 else (2 if ambiguity < 0.85 else 3))
        return pcs + extras[:n_extra]

    def nearest_field_freq(self, freq_hz: float, field: Sequence[int]) -> float:
        """Nearest frequency (Hz) whose pitch class is in `field`."""
        if freq_hz <= 0 or not field:
            return freq_hz
        midi = 69 + 12 * math.log2(freq_hz / 440.0)
        best, best_d = midi, 1e9
        for pc in field:
            for octv in range(-1, 2):
                cand = 12 * (round(midi / 12) + octv) + pc
                d = abs(cand - midi)
                if d < best_d:
                    best, best_d = cand, d
        return 440.0 * 2 ** ((best - 69) / 12)

    def state(self) -> Dict[str, object]:
        return {"current": self.current.name, "home": self.home.name,
                "distance_home": self.distance_home(), "last_op": self.last_op,
                "predicted_next": self.predicted_next.name if self.predicted_next else None,
                "budget": round(self.budget, 3)}
