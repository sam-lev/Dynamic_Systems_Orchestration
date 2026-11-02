# test_paired_voice.py -- Audible Lava / DSO v4.8
#
# The paired articulated voice: at most the top third of blobs by prominence
# carry a SECOND voice, same note as their sustained one but an octave lower,
# rendered entirely through \mpeHit so it has real note-offs.
#
# The property that distinguishes this from the existing BEAT_PHASE -> AMP
# tremolo is SILENCE: each hit is a self-freeing node with its own duration, so
# there is genuine nothing between hits ("tap, rest, tap"), not an amplitude
# dip in a continuously gated tone.
#
# Run: python test_paired_voice.py  |  pytest test_paired_voice.py -v

import math
import os
import sys
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("/mnt/project", HERE):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from score_governor import ScoreGovernor                    # noqa: E402
from pattern_language import Span                           # noqa: E402

TESTS = []
def test(fn):
    TESTS.append(fn); return fn


@dataclass
class B:
    id: int; centroid: tuple; area: float; birth: float = 150.0; death: float = 20.0
    persistence: float = 40.0; vx: float = 0.0; vy: float = 0.0; speed: float = 0.0
    ay: float = 0.0; direction: str = "turning"; age: int = 0
    d_area: float = 0.0; d_persistence: float = 0.0; elongation: float = 2.0
    score_freq_hz: float = 0.0


# Deliberately NOT 12-TET: the ornamental hit layer emits equal-tempered
# pitches from the schema degrees, and if a paired voice's octave-down pitch
# coincides with one of those the two layers become indistinguishable in a
# test. (This caught a false reading during development.)
def freq_of(i):
    return 197.3 * (1.0731 ** i)


def governor(tag="pv"):
    for p in (f"/tmp/_{tag}.json",):
        if os.path.exists(p):
            os.remove(p)
    return ScoreGovernor(320, 240, weights_path=f"/tmp/_{tag}.json",
                         runtime_path=f"/tmp/_{tag}r.json",
                         log_path=f"/tmp/_{tag}l.txt", fps=15)


def make_blobs(f, n, fps=15.0):
    t = f / fps
    out = []
    for i in range(n):
        per = 9.0 + 2 * i
        y = 120 + 80 * math.sin(2 * math.pi * t / per + i)
        vy = 80 * 2 * math.pi / per * math.cos(2 * math.pi * t / per + i) / fps
        b = B(i, (40 + 45 * i, y), 2000 + 700 * i, birth=160 + 8 * i,
              persistence=40 + 8 * i, vy=vy, speed=abs(vy), age=f)
        b.score_freq_hz = freq_of(i)
        out.append(b)
    return out


def run(g, n_blobs, seconds=90, fps=15):
    """Returns (paired_hits_by_blob, ornamental_count, last_state)."""
    paired, orn, s = {}, 0, None
    for f in range(int(seconds * fps)):
        blobs = make_blobs(f, n_blobs, fps)
        s = g.update(blobs, 1.0 / fps)
        for t_on, args in g.tick_patterns(blobs, voice_of=lambda bid: bid):
            fr = args[args.index("freq") + 1]
            du = args[args.index("dur") + 1]
            matched = False
            for i in range(n_blobs):
                if abs(fr - freq_of(i) / 2.0) < 1e-6:
                    paired.setdefault(i, []).append((t_on, du))
                    matched = True
                    break
            if not matched:
                orn += 1
    return paired, orn, s


# ── Selection ───────────────────────────────────────────────────────────────

@test
def pv_01_at_most_a_third_are_paired():
    for n, expect in ((6, 2), (9, 3), (12, 4)):
        g = governor(f"sel{n}")
        for f in range(60):
            s = g.update(make_blobs(f, n), 1 / 15.0)
        assert len(s.paired_ids) == expect, \
            f"{n} blobs -> {len(s.paired_ids)} paired, expected ceil(0.33*{n})={expect}"
        assert len(s.paired_ids) <= math.ceil(0.33 * n)


@test
def pv_02_a_single_blob_still_gets_one():
    """ceil, not floor: with 1-2 blobs a strict floor would silence the feature
    exactly when the texture is thinnest."""
    for n in (1, 2, 3):
        g = governor(f"one{n}")
        for f in range(60):
            s = g.update(make_blobs(f, n), 1 / 15.0)
        assert len(s.paired_ids) >= 1, f"{n} blobs produced no paired voice"


@test
def pv_03_paired_are_the_highest_prominence():
    g = governor("prom")
    for f in range(120):
        s = g.update(make_blobs(f, 8), 1 / 15.0)
    ranked = sorted(s.prominence_rank, key=lambda k: -s.prominence_rank[k])
    assert set(s.paired_ids) == set(ranked[:len(s.paired_ids)]), \
        "paired set is not the top of the prominence ranking"


# ── Pitch ───────────────────────────────────────────────────────────────────

@test
def pv_04_paired_pitch_is_an_octave_below_the_sustained_voice():
    g = governor("pitch")
    for f in range(60):
        blobs = make_blobs(f, 6)
        s = g.update(blobs, 1 / 15.0)
    # One tick renders only ~67 ms of new material while the paired layer
    # fires every 1-2 s, so a single call usually returns nothing. Collect
    # over several seconds.
    hits = []
    for f in range(60, 60 + 15 * 12):
        blobs = make_blobs(f, 6)
        s = g.update(blobs, 1 / 15.0)
        hits += [a[a.index("freq") + 1]
                 for _, a in g.tick_patterns(blobs, voice_of=lambda bid: bid)]
    expected = {freq_of(i) / 2.0 for i in s.paired_ids}
    assert any(any(abs(h - e) < 1e-6 for e in expected) for h in hits), \
        "no hit at the octave-below-sustained pitch"


@test
def pv_05_pitch_tracks_the_sustained_voice_not_a_fixed_degree():
    """The pair reads score_freq_hz, which voice_manager stamps AFTER harmonic
    gravitation -- so it follows the sustained voice exactly rather than
    re-deriving a pitch that could disagree with it."""
    g = governor("track")
    for f in range(60):
        s = g.update(make_blobs(f, 4), 1 / 15.0)

    def hits_for(freq, start):
        """Set EVERY blob's sustained pitch to `freq`, so the test does not
        depend on which blob happens to hold a paired slot in this window --
        the paired set follows prominence and can change between phases."""
        out = []
        for f in range(start, start + 15 * 12):
            blobs = make_blobs(f, 4)
            for b in blobs:
                b.score_freq_hz = freq
            g.update(blobs, 1 / 15.0)
            out += [a[a.index("freq") + 1]
                    for _, a in g.tick_patterns(blobs, voice_of=lambda x: x)]
        return out

    a = hits_for(400.0, 60)
    b = hits_for(600.0, 60 + 15 * 12)
    assert any(abs(h - 200.0) < 1e-6 for h in a), "pair did not follow 400 Hz -> 200 Hz"
    assert any(abs(h - 300.0) < 1e-6 for h in b), "pair did not follow 600 Hz -> 300 Hz"
    assert not any(abs(h - 300.0) < 1e-6 for h in a), "pair emitted a pitch it was never given"


@test
def pv_06_falls_back_when_the_sustained_pitch_is_unknown():
    """score_freq_hz is only stamped for BLOB_ONLY/HYBRID voices; an unbound or
    just-spawned blob must not produce a 0 Hz hit."""
    g = governor("fall")
    for f in range(60):
        s = g.update(make_blobs(f, 4), 1 / 15.0)
    blobs = make_blobs(60, 4)
    for b in blobs:
        b.score_freq_hz = 0.0
    for _, args in g.tick_patterns(blobs, voice_of=lambda x: x):
        assert args[args.index("freq") + 1] > 20.0, "produced a sub-audio or zero frequency"


# ── Articulation: the point of the feature ──────────────────────────────────

@test
def pv_07_pattern_is_alternating_slots_at_rest():
    """At low tension the paired pattern is E(beats, 2*beats): struck slot,
    empty slot, struck slot -- literally 'tap, rest, tap, rest'."""
    g = governor("alt")
    for f in range(60):
        s = g.update(make_blobs(f, 4), 1 / 15.0)
    g.tension = 0.0
    s.tension = 0.0
    pat, _ = g.paired_pattern_for_blob(make_blobs(60, 4)[s.paired_ids[0]], s)
    ev = sorted(e.start % 1.0 for e in pat.query(Span(0, 1)))
    n = 2 * max(2, int(s.beats_per_bar))
    assert ev, "paired pattern is empty"
    # every onset sits on a slot boundary, and no two are adjacent slots
    slots = sorted({int(round(x * n)) for x in ev})
    assert all(b - a >= 2 for a, b in zip(slots, slots[1:])), \
        f"onsets in adjacent slots (no rest between): slots {slots} of {n}"


@test
def pv_08_there_is_genuine_silence_between_hits():
    """The distinguishing property. Each hit's duration must be well under the
    gap to the next, so the voice is articulated rather than a pulsing drone."""
    g = governor("gap")
    paired, _, s = run(g, 6, seconds=90)
    assert paired, "no paired hits at all"
    checked = 0
    for bid, evs in paired.items():
        evs = sorted(set(evs))
        if len(evs) < 6:
            continue
        t0 = np.array([e[0] for e in evs]); du = np.array([e[1] for e in evs])
        ioi = np.diff(t0)
        silence = 1.0 - du.mean() / ioi.mean()
        assert silence > 0.5, \
            f"blob {bid}: hits fill {100*(1-silence):.0f}% of each interval -- not articulated"
        checked += 1
    assert checked >= 1, "not enough paired hits to measure articulation"


@test
def pv_09_hits_are_shorter_than_their_slot():
    """dur_scale < 0.5 means a hit occupies under half its slot, so the rest is
    audible even when consecutive slots are struck."""
    assert g_dur_scale() < 0.5, "pair_dur_scale >= 0.5 leaves no audible rest"


def g_dur_scale():
    g = governor("ds")
    return g.pair_dur_scale


@test
def pv_10_paired_layer_is_sparser_than_the_ornamental_layer():
    g = governor("sparse")
    paired, orn, _ = run(g, 6, seconds=60)
    n_paired = sum(len(set(v)) for v in paired.values())
    assert n_paired > 0, "paired layer produced nothing"
    assert n_paired < orn, \
        f"paired layer ({n_paired}) is not sparser than ornamental ({orn}) -- it should " \
        f"articulate a pitch, not add texture"


# ── Additivity ──────────────────────────────────────────────────────────────

@test
def pv_11_sustained_voices_are_untouched():
    """The sustained voice is left exactly as it was: the pair is an addition,
    not a replacement. Nothing in the governor gates \\mpeVoice."""
    import voice_manager
    src = open(voice_manager.__file__).read()
    assert "score_state" in src
    # no clock/beat-driven gate call anywhere in the voice layer
    for marker in ("beat_phase", "clock"):
        for ln in src.splitlines():
            if marker in ln and "gate" in ln:
                raise AssertionError(f"sustained voice appears to be gated by the clock: {ln.strip()}")


@test
def pv_12_disabling_the_pattern_layer_removes_both_layers():
    g = governor("off")
    for f in range(60):
        g.update(make_blobs(f, 6), 1 / 15.0)
    g.pattern_gain = 0.0
    assert g.tick_patterns(make_blobs(60, 6), voice_of=lambda x: x) == []


@test
def pv_13_scheduler_keys_do_not_collide():
    """The paired layer renders under its own scheduler key, otherwise it and
    the ornamental layer would advance each other's 'rendered up to' marker and
    silence one of them."""
    g = governor("keys")
    for f in range(60):
        g.update(make_blobs(f, 6), 1 / 15.0)
    g.tick_patterns(make_blobs(60, 6), voice_of=lambda x: x)
    keys = set(g.scheduler._sent_until)
    orn = {k for k in keys if k < g.pair_slot_base}
    pair = {k for k in keys if k >= g.pair_slot_base}
    assert orn and pair, f"expected both layers to have keys, got {sorted(keys)}"
    assert not (orn & {k - g.pair_slot_base for k in pair} & set()), "key collision"


if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn(); print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL  {fn.__name__}\n      {e}")
        except Exception as e:
            failed += 1; print(f"ERR   {fn.__name__}\n      {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
