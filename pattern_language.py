# pattern_language.py -- Audible Lava / DSO v4.2
# Pattern-language layer: patterns are words, transformations are syntax.
#
# Core abstraction (Tidal's model): a Pattern is a function from a time SPAN
# (in cycles) to the list of EVENTS active within it. Transformations are
# Pattern -> Pattern functions, composable; patterns may be infinite and
# aperiodic; any future pattern type is just another queryable function.
#
#   Span(start, end)                       cycle positions (floats)
#   Event(start, end, value, velocity,     one HIT: a discrete onset with
#         degree, tags)                    duration, accent, optional pitch
#                                          degree, free tags
#   Pattern.query(span) -> [Event]
#
# Primitives:  silence, pure, steps, euclid, from_events, sequence, stack
# Transforms:  registered via @transform("name"); see TRANSFORMS
# Spec:        apply_spec(pattern, [("rotate", 1), ("degrade", 0.2)])
# Scheduling:  PatternScheduler renders cycle time -> seconds via an
#              ElasticClock and emits timestamped OSC bundles for \mpeHit.
#
# A "hit" = a discrete note-onset event (velocity, duration, pitch degree)
# as opposed to the continuous sustained \mpeVoice tone.

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple


# ── Types ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Span:
    start: float
    end: float

    def cycles(self) -> List[int]:
        return list(range(int(math.floor(self.start)), int(math.ceil(self.end))))

    def intersect(self, other: "Span") -> Optional["Span"]:
        s, e = max(self.start, other.start), min(self.end, other.end)
        return Span(s, e) if e > s else None


@dataclass(frozen=True)
class Event:
    start: float                 # cycle position of onset
    end: float                   # cycle position of offset
    value: Any = 1               # payload (1 for a plain hit, or a sample/name)
    velocity: float = 0.8        # accent, [0,1]
    degree: Optional[int] = None # scale/chord degree (schemas fill this in)
    tags: Tuple[str, ...] = ()   # free labels ("bass", "fill", schema name)

    @property
    def dur(self) -> float:
        return self.end - self.start


class Pattern(Protocol):
    def query(self, span: Span) -> List[Event]: ...


class FnPattern:
    """Pattern from a query function. All primitives/transforms return this."""
    __slots__ = ("fn", "label")

    def __init__(self, fn: Callable[[Span], List[Event]], label: str = "pattern"):
        self.fn, self.label = fn, label

    def query(self, span: Span) -> List[Event]:
        return [e for e in self.fn(span) if e.end > span.start and e.start < span.end]

    def __repr__(self):
        return f"<Pattern {self.label}>"

    # fluent transform application
    def __rshift__(self, tf: Callable[["FnPattern"], "FnPattern"]) -> "FnPattern":
        return tf(self)


# ── Primitives ───────────────────────────────────────────────────────────────

def silence() -> FnPattern:
    return FnPattern(lambda span: [], "silence")


def pure(value: Any = 1, velocity: float = 0.8) -> FnPattern:
    """One event per cycle, whole-cycle duration."""
    def q(span: Span):
        return [Event(c, c + 1, value, velocity) for c in span.cycles()]
    return FnPattern(q, f"pure({value})")


def steps(seq: Sequence[Any], velocity: float = 0.8) -> FnPattern:
    """Subdivide each cycle into len(seq) equal slots; falsy = rest. A
    binary list [1,0,1,1] is the simplest rhythm; ints > 1 are values."""
    n = len(seq)
    if n == 0:
        return silence()
    def q(span: Span):
        out = []
        for c in span.cycles():
            for i, v in enumerate(seq):
                if v:
                    out.append(Event(c + i / n, c + (i + 1) / n, v, velocity))
        return out
    return FnPattern(q, f"steps({list(seq)})")


def bjorklund(k: int, n: int) -> List[int]:
    """Euclidean rhythm E(k, n): k hits distributed as evenly as possible
    over n slots (Toussaint 2005). E(3,8) = [1,0,0,1,0,0,1,0] (tresillo)."""
    if n <= 0:
        return []
    k = max(0, min(k, n))
    if k == 0:
        return [0] * n
    if k == n:
        return [1] * n
    # Bresenham-style formulation (equivalent to Bjorklund's algorithm)
    pat, prev = [], -1
    for i in range(n):
        cur = int(math.floor(i * k / n))
        pat.append(1 if cur != prev else 0)
        prev = cur
    return pat


def euclid(k: int, n: int, rotation: int = 0, velocity: float = 0.8) -> FnPattern:
    pat = bjorklund(k, n)
    if n:
        r = rotation % n
        pat = pat[r:] + pat[:r]
    p = steps(pat, velocity)
    p.label = f"euclid({k},{n},rot={rotation})"
    return p


def from_events(events: Sequence[Event], period: float = 1.0) -> FnPattern:
    """Explicit events within [0, period) repeated every period."""
    evs = list(events)
    def q(span: Span):
        out = []
        c0, c1 = int(math.floor(span.start / period)), int(math.ceil(span.end / period))
        for c in range(c0, c1):
            for e in evs:
                out.append(replace(e, start=e.start + c * period, end=e.end + c * period))
        return out
    return FnPattern(q, f"events[{len(evs)}]")


def sequence(*pats: FnPattern) -> FnPattern:
    """Play each pattern in turn, one per cycle (Tidal `cat`)."""
    n = len(pats)
    if n == 0:
        return silence()
    def q(span: Span):
        out = []
        for c in span.cycles():
            p = pats[c % n]
            for e in p.query(Span(c, c + 1)):
                out.append(e)
        return out
    return FnPattern(q, f"seq[{n}]")


def stack(*pats: FnPattern) -> FnPattern:
    """Superimpose (play simultaneously)."""
    def q(span: Span):
        out = []
        for p in pats:
            out.extend(p.query(span))
        return sorted(out, key=lambda e: e.start)
    return FnPattern(q, f"stack[{len(pats)}]")


# ── Transformation registry / algebra ────────────────────────────────────────

Transform = Callable[[FnPattern], FnPattern]
TRANSFORMS: Dict[str, Callable[..., Transform]] = {}
TRANSFORM_DOCS: Dict[str, str] = {}


def transform(name: str, doc: str = ""):
    """Register a transform FACTORY: f(*args) -> (Pattern -> Pattern)."""
    def deco(factory):
        TRANSFORMS[name] = factory
        TRANSFORM_DOCS[name] = doc or (factory.__doc__ or "").strip()
        return factory
    return deco


def _time_map(p: FnPattern, f_query: Callable[[float], float],
              f_event: Callable[[float], float], label: str) -> FnPattern:
    def q(span: Span):
        inner = Span(f_query(span.start), f_query(span.end))
        if inner.end < inner.start:
            inner = Span(inner.end, inner.start)
        return [replace(e, start=f_event(e.start), end=f_event(e.end)) if f_event(e.start) <= f_event(e.end)
                else replace(e, start=f_event(e.end), end=f_event(e.start))
                for e in p.query(inner)]
    return FnPattern(q, f"{label}({p.label})")


@transform("fast", "Speed up by factor (density up): fast(2) plays two cycles per cycle.")
def fast(factor: float) -> Transform:
    f = float(factor)
    return lambda p: _time_map(p, lambda t: t * f, lambda t: t / f, f"fast{f}")


@transform("slow", "Slow down by factor (density down).")
def slow(factor: float) -> Transform:
    return fast(1.0 / float(factor))


@transform("rotate", "Shift the pattern earlier by `steps/n` of a cycle (Tidal `<~`).")
def rotate(steps_: float, n: int = 1) -> Transform:
    sh = float(steps_) / float(n)
    return lambda p: _time_map(p, lambda t: t + sh, lambda t: t - sh, f"rot{sh:.3f}")


@transform("rev", "Reverse each cycle.")
def rev() -> Transform:
    def tf(p: FnPattern) -> FnPattern:
        def q(span: Span):
            out = []
            for c in span.cycles():
                for e in p.query(Span(c, c + 1)):
                    s, en = c + (c + 1 - e.end), c + (c + 1 - e.start)
                    out.append(replace(e, start=s, end=en))
            return out
        return FnPattern(q, f"rev({p.label})")
    return tf


@transform("degrade", "Randomly drop hits with probability p (deterministic per event position + seed).")
def degrade(prob: float, seed: int = 0) -> Transform:
    import random
    def tf(p: FnPattern) -> FnPattern:
        def q(span: Span):
            out = []
            for e in p.query(span):
                r = random.Random(hash((round(e.start, 6), seed))).random()
                if r >= prob:
                    out.append(e)
            return out
        return FnPattern(q, f"degrade{prob}({p.label})")
    return tf


@transform("mask", "Keep only events overlapping a hit of the mask pattern (boolean AND).")
def mask(mask_pat: FnPattern) -> Transform:
    def tf(p: FnPattern) -> FnPattern:
        def q(span: Span):
            m = mask_pat.query(span)
            return [e for e in p.query(span) if any(me.start <= e.start < me.end for me in m)]
        return FnPattern(q, f"mask({p.label})")
    return tf


@transform("superimpose", "Play the transformed copy on top of the original (stack(p, f(p))).")
def superimpose(f: Transform) -> Transform:
    return lambda p: stack(p, f(p))


@transform("every", "Apply f on every n-th cycle (offset selects which).")
def every(n: int, f: Transform, offset: int = 0) -> Transform:
    def tf(p: FnPattern) -> FnPattern:
        fp = f(p)
        def q(span: Span):
            out = []
            for c in span.cycles():
                src = fp if (c - offset) % n == 0 else p
                out.extend(src.query(Span(c, c + 1)))
            return out
        return FnPattern(q, f"every{n}({p.label})")
    return tf


@transform("iter", "Rotate progressively: cycle c is shifted by c/n (Tidal `iter`).")
def iter_(n: int) -> Transform:
    def tf(p: FnPattern) -> FnPattern:
        def q(span: Span):
            out = []
            for c in span.cycles():
                sh = (c % n) / n
                shifted = _time_map(p, lambda t: t + sh, lambda t: t - sh, "iter")
                out.extend(shifted.query(Span(c, c + 1)))
            return out
        return FnPattern(q, f"iter{n}({p.label})")
    return tf


@transform("palindrome", "Alternate forward and reversed cycles.")
def palindrome() -> Transform:
    return lambda p: every(2, rev(), offset=1)(p)


@transform("chunk", "Apply f to the k-th 1/n of the cycle, rotating k each cycle.")
def chunk(n: int, f: Transform) -> Transform:
    def tf(p: FnPattern) -> FnPattern:
        fp = f(p)
        def q(span: Span):
            out = []
            for c in span.cycles():
                k = c % n
                lo, hi = c + k / n, c + (k + 1) / n
                for e in p.query(Span(c, c + 1)):
                    if not (lo <= e.start < hi):
                        out.append(e)
                for e in fp.query(Span(lo, hi)):
                    if lo <= e.start < hi:
                        out.append(e)
            return out
        return FnPattern(q, f"chunk{n}({p.label})")
    return tf


@transform("swing", "Delay every second subdivision by `amount` of a slot (0..1).")
def swing(amount: float, subdiv: int = 8) -> Transform:
    a = float(amount) / subdiv
    def tf(p: FnPattern) -> FnPattern:
        def q(span: Span):
            out = []
            for e in p.query(Span(span.start - 1.0 / subdiv, span.end)):
                slot = int(round((e.start % 1.0) * subdiv))
                if slot % 2 == 1:
                    e = replace(e, start=e.start + a, end=e.end + a)
                out.append(e)
            return out
        return FnPattern(q, f"swing{amount}({p.label})")
    return tf


@transform("ply", "Repeat each hit n times within its own duration.")
def ply(n: int) -> Transform:
    def tf(p: FnPattern) -> FnPattern:
        def q(span: Span):
            out = []
            for e in p.query(span):
                d = e.dur / n
                for i in range(n):
                    out.append(replace(e, start=e.start + i * d, end=e.start + (i + 1) * d))
            return out
        return FnPattern(q, f"ply{n}({p.label})")
    return tf


@transform("off", "Superimpose a copy delayed by t cycles and transformed by f (instant canon).")
def off(t: float, f: Transform) -> Transform:
    return lambda p: stack(p, rotate(-t)(f(p)))


@transform("accent", "Multiply velocity by the accent pattern's value at each onset (e.g. steps([1.0,0.6,0.8,0.6])).")
def accent(acc: FnPattern, floor: float = 0.4) -> Transform:
    def tf(p: FnPattern) -> FnPattern:
        def q(span: Span):
            a = acc.query(span)
            out = []
            for e in p.query(span):
                v = next((float(ae.value) for ae in a if ae.start <= e.start < ae.end), floor)
                out.append(replace(e, velocity=max(0.0, min(1.0, e.velocity * v))))
            return out
        return FnPattern(q, f"accent({p.label})")
    return tf


@transform("velocity", "Scale all velocities by k.")
def velocity(k: float) -> Transform:
    return lambda p: FnPattern(lambda span: [replace(e, velocity=max(0.0, min(1.0, e.velocity * k)))
                                             for e in p.query(span)], f"vel{k}({p.label})")


@transform("transpose", "Add n to every event's degree (within the current field).")
def transpose(n: int) -> Transform:
    return lambda p: FnPattern(lambda span: [replace(e, degree=(e.degree + n) if e.degree is not None else None)
                                             for e in p.query(span)], f"tr{n}({p.label})")


@transform("invert", "Mirror degrees around a pivot degree.")
def invert(pivot: int = 0) -> Transform:
    return lambda p: FnPattern(lambda span: [replace(e, degree=(2 * pivot - e.degree) if e.degree is not None else None)
                                             for e in p.query(span)], f"inv{pivot}({p.label})")


@transform("arp", "Assign successive degrees from a list to successive hits (arpeggio).")
def arp(degrees: Sequence[int]) -> Transform:
    ds = list(degrees)
    def tf(p: FnPattern) -> FnPattern:
        def q(span: Span):
            out = []
            for i, e in enumerate(sorted(p.query(span), key=lambda e: e.start)):
                out.append(replace(e, degree=ds[i % len(ds)] if ds else e.degree))
            return out
        return FnPattern(q, f"arp({p.label})")
    return tf


@transform("tag", "Attach a tag to every event.")
def tag(label: str) -> Transform:
    return lambda p: FnPattern(lambda span: [replace(e, tags=e.tags + (label,)) for e in p.query(span)],
                               f"tag:{label}({p.label})")


@transform("density", "Continuous density control: keep a fraction `d` of hits (rank-ordered by velocity).")
def density(d: float) -> Transform:
    d = max(0.0, min(1.0, float(d)))
    def tf(p: FnPattern) -> FnPattern:
        def q(span: Span):
            out = []
            for c in span.cycles():
                evs = sorted(p.query(Span(c, c + 1)), key=lambda e: -e.velocity)
                keep = int(round(len(evs) * d))
                out.extend(sorted(evs[:keep], key=lambda e: e.start))
            return out
        return FnPattern(q, f"density{d}({p.label})")
    return tf


def compose(*tfs: Transform) -> Transform:
    """Left-to-right composition of transforms."""
    def tf(p: FnPattern) -> FnPattern:
        for t in tfs:
            p = t(p)
        return p
    return tf


def apply_spec(p: FnPattern, spec: Sequence[Tuple[str, Any]]) -> FnPattern:
    """Apply a list of (transform_name, *args) tuples -- serializable
    syntax that governors emit and the EDIT surface can display."""
    for item in spec:
        name, *args = item
        if name not in TRANSFORMS:
            raise KeyError(f"unknown transform {name!r}; registered: {sorted(TRANSFORMS)}")
        p = TRANSFORMS[name](*args)(p)
    return p


# ── Scheduler: cycles -> seconds -> OSC bundles ──────────────────────────────

class PatternScheduler:
    """Renders a pattern's events for the upcoming look-ahead window into
    timestamped OSC bundles for the gated \\mpeHit SynthDef.

    Cycle time is converted to seconds by the ElasticClock (one cycle = one
    bar); each onset is pulled toward the nearest clock grid line by the
    clock's elastic strength (Kuramoto-governed). Bundles are scheduled from
    Python so meter stays projective (Hasty), not a rigid SC TempoClock.
    """

    def __init__(self, osc_client=None, lookahead_s: float = 0.25, synthdef: str = "mpeHit"):
        self.osc = osc_client
        self.lookahead = float(lookahead_s)
        self.synthdef = synthdef
        self._sent_until: Dict[int, float] = {}       # voice slot -> SECONDS rendered up to

    def render(self, slot: int, pattern: FnPattern, clock, now_s: float,
               degree_to_freq: Callable[[Optional[int]], float], amp: float = 0.3,
               extra_params: Optional[Dict[str, float]] = None,
               dur_scale: float = 0.9) -> List[Tuple[float, list]]:
        """Return [(time_s, /s_new args)] for events in the look-ahead window
        (and send them if an OSC client is attached).

        Progress is tracked in SECONDS, not cycles. ElasticClock.set_period
        re-anchors t0 every frame to preserve beat phase, so cycle numbering
        is not monotonic across frames -- a cycle-based marker races ahead and
        then silences the layer permanently. Seconds are anchor-invariant.
        """
        t_from = max(self._sent_until.get(slot, now_s), now_s)
        t_end = now_s + self.lookahead
        if t_end <= t_from:
            return []
        c_from = clock.seconds_to_cycles(t_from)
        c_end = clock.seconds_to_cycles(t_end)
        if c_end < c_from:                       # period changed under us
            c_from, c_end = c_end, c_from
        out = []
        for e in pattern.query(Span(c_from, c_end)):
            if e.start < c_from:
                continue
            # Elastic pull can move an onset slightly earlier than the window
            # start. Clamping rather than dropping matters: with a strong pull
            # this silently discarded ~75% of all hits.
            t_on = max(clock.pull(clock.cycles_to_seconds(e.start)), t_from)
            # dur_scale sets how much of its slot a hit occupies. 0.9 is a
            # near-legato articulation; values around 0.4 leave an audible
            # SILENCE between hits, which is what makes an articulated voice
            # read as "tap, rest, tap" rather than as a pulsing drone.
            dur = max(0.03,
                      (clock.cycles_to_seconds(e.end) - clock.cycles_to_seconds(e.start))
                      * float(dur_scale))
            args = ["/s_new", self.synthdef, -1, 0, 0,
                    "freq", float(degree_to_freq(e.degree)), "amp", float(amp * e.velocity),
                    "dur", float(dur), "vel", float(e.velocity)]
            for k, v in (extra_params or {}).items():
                args += [k, float(v)]
            out.append((t_on, args))
        self._sent_until[slot] = t_end
        if self.osc is not None and out:
            try:
                from pythonosc import osc_bundle_builder, osc_message_builder
                for t_on, args in out:
                    b = osc_bundle_builder.OscBundleBuilder(t_on)
                    m = osc_message_builder.OscMessageBuilder(address=args[0])
                    for a in args[1:]:
                        m.add_arg(a)
                    b.add_content(m.build())
                    self.osc.send(b.build())
            except Exception:
                pass
        return sorted(out, key=lambda r: r[0])

    def reset(self, slot: Optional[int] = None) -> None:
        if slot is None:
            self._sent_until.clear()
        else:
            self._sent_until.pop(slot, None)
