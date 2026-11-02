# schema_lexicon.py -- Audible Lava / DSO v4.2
# Schema lexicon: the pattern layer lifted from rhythm to voice-leading frames.
#
#   Schema = (melodic degree contour, bass degree contour,
#             metric footprint in beats, tension profile)
#   Degrees are scale degrees relative to the current pitch field's root:
#   1 = root, 3, 5 = chord tones; 2, 4, 6, 7 = passing/neighbour tones;
#   negative = below the root. Each contour entry is one beat (None = rest).
#
# Lexicon (12 frames):
#   Borrowed (Gjerdingen 2007, cited skeletons):
#     prinner, romanesca, fonte, cadence (converging)
#   Original, idiomatic to convection dynamics:
#     plume, sink, merge, split, stall, overturn, plateau, swarm
#
# Succession: first-order Markov table over schema names; governors
# reweight rows with `bias()` (e.g. rising tension prefers plume/eruption
# families, resolution prefers cadence/plateau). Realization returns two
# Patterns (melody, bass) whose events carry `degree`, ready for
# pattern_language transforms (degrade, ply, off ...) and the scheduler.

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from pattern_language import Event, FnPattern, from_events


@dataclass(frozen=True)
class Schema:
    name: str
    melody: Tuple[Optional[int], ...]      # degree per beat
    bass: Tuple[Optional[int], ...]        # degree per beat
    footprint: int                         # beats
    tension: Tuple[float, ...]             # per-beat tension profile [0,1]
    origin: str                            # "gjerdingen" | "original"
    doc: str = ""
    tags: Tuple[str, ...] = ()

    def __post_init__(self):
        assert len(self.melody) == len(self.bass) == len(self.tension) == self.footprint


LEXICON: Dict[str, Schema] = {}


def _add(s: Schema) -> Schema:
    LEXICON[s.name] = s
    return s


# ── Borrowed skeletons ───────────────────────────────────────────────────────
_add(Schema("prinner", (6, 5, 4, 3), (4, 3, 2, 1), 4, (0.6, 0.5, 0.4, 0.2), "gjerdingen",
            "Descending 6-5-4-3 over 4-3-2-1; the galant 'riposte'. Relaxation frame.",
            ("descending", "resolving")))
_add(Schema("romanesca", (1, 5, 6, 3), (1, 5, 6, 3), 4, (0.3, 0.4, 0.5, 0.35), "gjerdingen",
            "Leaping bass by descending thirds (1-5-6-3); stately opening gambit.",
            ("opening", "stable")))
_add(Schema("fonte", (4, 3, 3, 2), (7, 1, 6, 7), 4, (0.7, 0.5, 0.6, 0.4), "gjerdingen",
            "Sequential down-a-step pair (minor then major); post-boundary descent.",
            ("sequence", "descending")))
_add(Schema("cadence", (3, 2, 7, 1), (1, 4, 5, 1), 4, (0.4, 0.7, 0.9, 0.0), "gjerdingen",
            "Converging cadence: outer voices close onto the root; the arrival gesture.",
            ("closing", "resolving")))

# ── Original frames (convection idiom) ───────────────────────────────────────
_add(Schema("plume", (1, 2, 3, 5, 6, 8), (1, 1, 1, 3, 3, 5), 6, (0.2, 0.35, 0.5, 0.65, 0.8, 0.9), "original",
            "Rising contour with accelerating rhythm: a blob ascending.", ("rising", "tension")))
_add(Schema("sink", (8, 6, 5, 3, 2, 1), (5, 3, 3, 1, 1, 1), 6, (0.8, 0.65, 0.5, 0.35, 0.2, 0.1), "original",
            "Falling contour, decelerating: a blob sinking to rest.", ("descending", "resolving")))
_add(Schema("merge", (5, 4, 3, 1), (1, 2, 3, 1), 4, (0.6, 0.7, 0.5, 0.1), "original",
            "Two lines converging to a unison: blob merge.", ("converging", "resolving")))
_add(Schema("split", (1, 2, 3, 5), (1, 7, 6, 4), 4, (0.2, 0.4, 0.6, 0.7), "original",
            "Unison diverging into two lines: blob split.", ("diverging", "tension")))
_add(Schema("stall", (3, 4, 3, 2, 3), (1, 1, 1, 1, 1), 5, (0.4, 0.5, 0.4, 0.5, 0.4), "original",
            "Oscillation around one degree over a pedal: a blob hovering at turnaround.",
            ("static", "neighbour")))
_add(Schema("overturn", (1, 3, 5, 6, 5, 3, 1), (1, 1, 3, 4, 3, 1, 1), 7,
            (0.2, 0.4, 0.6, 0.8, 0.6, 0.4, 0.15), "original",
            "Rise-then-fall arc: a full convection overturn.", ("arc", "cycle")))
_add(Schema("plateau", (5, None, 5, None), (1, None, None, 1), 4, (0.3, 0.3, 0.3, 0.3), "original",
            "Sparse sustained tones: the lamp at rest / low activity.", ("static", "sparse")))
_add(Schema("swarm", (3, 5, 4, 6, 5, 7, 6, 8), (1, 1, 2, 2, 3, 3, 4, 4), 8,
            (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85), "original",
            "Dense, climbing cluster: many blobs active at once (Xenakian mass).",
            ("dense", "tension")))

NAMES: List[str] = list(LEXICON.keys())


# ── Succession ───────────────────────────────────────────────────────────────

class SchemaMarkov:
    """First-order succession table over schema names. Rows sum to 1.
    `bias(tag, factor)` multiplies the probability of every schema carrying
    the tag (then renormalizes) -- the governor's lever: rising tension
    biases toward 'tension' frames, resolution toward 'resolving' ones,
    a merge event toward 'converging', etc."""

    def __init__(self, seed: int = 11):
        self.rng = random.Random(seed)
        n = len(NAMES)
        self.base: Dict[str, Dict[str, float]] = {}
        for a in NAMES:
            sa = LEXICON[a]
            row = {}
            for b in NAMES:
                sb = LEXICON[b]
                w = 1.0
                if a == b:
                    w = 0.35                                   # avoid immediate repeats
                if "resolving" in sa.tags and "tension" in sb.tags:
                    w *= 1.6                                   # after resolution, build again
                if "tension" in sa.tags and "resolving" in sb.tags:
                    w *= 1.6                                   # after tension, resolve
                if "closing" in sa.tags and "opening" in sb.tags:
                    w *= 2.0
                if "rising" in sa.tags and "descending" in sb.tags:
                    w *= 1.3
                row[b] = w
            z = sum(row.values())
            self.base[a] = {b: w / z for b, w in row.items()}
        self.table: Dict[str, Dict[str, float]] = {a: dict(r) for a, r in self.base.items()}
        self.current: str = "plateau"
        self.history: List[str] = []

    def reset_bias(self) -> None:
        self.table = {a: dict(r) for a, r in self.base.items()}

    def bias(self, tag: str, factor: float) -> None:
        for a, row in self.table.items():
            for b in row:
                if tag in LEXICON[b].tags:
                    row[b] *= factor
            z = sum(row.values())
            for b in row:
                row[b] /= z

    def bias_from_state(self, tension: float, ambiguity: float, events: Sequence[str] = ()) -> None:
        """Governor reweighting from ScoreState-like quantities."""
        self.reset_bias()
        self.bias("tension", 0.6 + 1.2 * tension)
        self.bias("resolving", 1.6 - 1.0 * tension)
        self.bias("dense", 0.5 + ambiguity)
        self.bias("sparse", 1.4 - ambiguity)
        for ev in events:
            if ev == "merge":
                self.bias("converging", 3.0)
            elif ev == "split":
                self.bias("diverging", 3.0)
            elif ev == "boundary":
                self.bias("opening", 2.0)

    def next(self) -> str:
        row = self.table[self.current]
        r = self.rng.random()
        acc = 0.0
        for b, p in row.items():
            acc += p
            if r <= acc:
                self.current = b
                break
        self.history.append(self.current)
        return self.current

    def distribution(self, name: Optional[str] = None) -> Dict[str, float]:
        return dict(self.table[name or self.current])


# ── Realization ──────────────────────────────────────────────────────────────

def realize(schema: Schema, beats_per_bar: int = 4, velocity: float = 0.8,
            ornament: float = 0.0) -> Tuple[FnPattern, FnPattern]:
    """Turn a schema into (melody, bass) Patterns. The frame's footprint is
    laid across `footprint / beats_per_bar` cycles (one cycle = one bar), so
    a 6-beat plume over a 4-beat bar spans 1.5 cycles -- schemas and meter
    are decoupled on purpose. `ornament` in [0,1] adds neighbour-tone
    diminution between successive melody degrees (Schenkerian surface)."""
    period = schema.footprint / float(beats_per_bar)
    beat = 1.0 / beats_per_bar
    mel, bass = [], []
    for i in range(schema.footprint):
        t = i * beat
        if schema.melody[i] is not None:
            deg = schema.melody[i]
            mel.append(Event(t, t + beat, 1, velocity * (0.7 + 0.3 * schema.tension[i]), deg, (schema.name, "melody")))
            if ornament > 0 and i + 1 < schema.footprint and schema.melody[i + 1] is not None:
                nxt = schema.melody[i + 1]
                if abs(nxt - deg) >= 2:
                    step = 1 if nxt > deg else -1
                    mel.append(Event(t + beat * (1 - 0.5 * ornament), t + beat, 1,
                                     velocity * 0.5, deg + step, (schema.name, "ornament")))
        if schema.bass[i] is not None:
            bass.append(Event(t, t + beat, 1, velocity * 0.9, schema.bass[i] - 7, (schema.name, "bass")))
    m = from_events(mel, period); m.label = f"{schema.name}:melody"
    b = from_events(bass, period); b.label = f"{schema.name}:bass"
    return m, b


def degree_to_pc(degree: int, root_pc: int, mode_pcs: Sequence[int] = (0, 2, 4, 5, 7, 9, 11)) -> int:
    """Diatonic degree (1-based, may be negative or > 7) -> pitch class."""
    d = degree - 1
    octv, idx = divmod(d, len(mode_pcs))
    return (root_pc + mode_pcs[idx]) % 12


def degree_to_freq(degree: Optional[int], root_pc: int = 0, base_midi: int = 60,
                   mode_pcs: Sequence[int] = (0, 2, 4, 5, 7, 9, 11)) -> float:
    if degree is None:
        return 440.0 * 2 ** ((base_midi - 69) / 12)
    d = degree - 1
    octv, idx = divmod(d, len(mode_pcs))
    midi = base_midi + root_pc + mode_pcs[idx] + 12 * octv
    return 440.0 * 2 ** ((midi - 69) / 12)
