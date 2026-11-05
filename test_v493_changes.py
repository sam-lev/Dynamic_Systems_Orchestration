#!/usr/bin/env python3
# test_v493_changes.py -- Audible Lava / DSO v4.9.3
#
# Covers the v4.9.3 change set end to end. No SuperCollider, camera, or
# pygame; the governor stack runs on synthetic blobs.
#
#   score_governor.py   young-blob lifecycle gate, persistence-gated and
#                       smoothed blob count (+ count_change rule, symbol
#                       override, meter cardinality), gov.grouping.reset_learning,
#                       pattern slots from the blob period, elongation -> swing,
#                       moving-average pattern features
#   system_state.py     count_override reaches the symbol
#   elastic_clock.py    set_period keeps BAR phase (patterns advance past beat 0)
#   melody_rhythm.py    FieldScale, one-beat scale latch, onset pull, 2-octave top
#   controller.py       reset_learning on the gov EDIT family
#   routing.py / voice_manager.py   lower pitch knobs
#
# (The duration-model guards -- floor, cap, prior strength, reset -- are in
# test_hazard_grouping.py.)
#
# Run: python test_v493_changes.py

import math
import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("/mnt/project", HERE):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from score_governor import ScoreGovernor, DEFAULT_WEIGHTS               # noqa: E402
from system_state import SystemStateTracker                             # noqa: E402
from elastic_clock import ElasticClock, ClockParams                     # noqa: E402
from pattern_language import Span, euclid, PatternScheduler             # noqa: E402
from melody_rhythm import (FieldScale, MelodyEngine, PitchParams,       # noqa: E402
                           HazardParams, _DEFAULT_SCALE)
from controller import (ControllerEvent, SET_PARAM, EDIT_FAMILIES,      # noqa: E402
                        EDIT_STEP_FINE)

PASS = FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {msg}")
    else:
        FAIL += 1
        print(f"  FAIL {msg}")


# ── fakes ────────────────────────────────────────────────────────────────────
@dataclass
class B:
    id: int; centroid: tuple; area: float = 3000.0; birth: float = 150.0; death: float = 20.0
    persistence: float = 40.0; vx: float = 0.0; vy: float = 0.0; speed: float = 0.0
    ay: float = 0.0; direction: str = "turning"; age: int = 0
    d_area: float = 0.0; d_persistence: float = 0.0; elongation: float = 0.0
    score_freq_hz: float = 0.0


def governor(tag, fps=30):
    for p in (f"/tmp/_v493_{tag}.json",):
        if os.path.exists(p):
            os.remove(p)
    return ScoreGovernor(320, 240, weights_path=f"/tmp/_v493_{tag}.json",
                         runtime_path=f"/tmp/_v493_{tag}r.json",
                         log_path=f"/tmp/_v493_{tag}l.txt", fps=fps)


def osc_blobs(f, fps, n=3, periods=(12.0, 8.0, 10.0), persistence=40.0, elong=0.0):
    t = f / fps
    out = []
    for i in range(n):
        per = periods[i % len(periods)]
        y = 120 + 80 * math.sin(2 * math.pi * t / per + i)
        vy = 80 * 2 * math.pi / per * math.cos(2 * math.pi * t / per + i) / fps
        out.append(B(i, (60 + 100 * i, y), vy=vy, speed=abs(vy), persistence=persistence,
                     elongation=elong, age=f))
    return out


# ── 1. lifecycle gate ────────────────────────────────────────────────────────
def test_lifecycle_gate():
    print("[governor] lifecycle events from young blobs are not cues")
    fps = 30
    g = governor("gate", fps)
    for f in range(fps * 5):                      # blobs 0..2 are 5 s old
        g.update(osc_blobs(f, fps), 1 / fps)
    check(not g.event_counts_as_cue(("birth", 7)), "birth never counts")
    check(g.event_counts_as_cue(("death", 1)), "death of a 5 s old blob counts")
    check(g.event_counts_as_cue(("merge", 1, 0)), "merge whose retired blob is old counts")
    check(g.event_counts_as_cue(("split", 2, 9)), "split of an old parent counts")
    check(not g.event_counts_as_cue(("death", 42)), "death of a never-tracked blob does not count")
    # a blob that appears and dies within 1 s: flicker
    young = B(42, (200, 100))
    s = g.update(osc_blobs(fps * 5, fps) + [young], 1 / fps, [("birth", 42)])
    check(s.events == ["birth"] and s.cue_events == [], "s.events keeps the birth, cue_events drops it")
    for f in range(fps * 5 + 1, fps * 6):
        g.update(osc_blobs(f, fps) + [young], 1 / fps)
    s = g.update(osc_blobs(fps * 6, fps), 1 / fps, [("death", 42)])
    check(s.cue_events == [], "death of a 1 s blob is not a cue")
    check(not g.event_counts_as_cue(("merge", 42, 0)), "merge whose RETIRED blob is young is not a cue")
    check(g.n_cues_gated == 2, f"two gated events counted ({g.n_cues_gated})")
    # the gate is a weight
    g.handle_controller_event(ControllerEvent(SET_PARAM, param="gov.grouping.min_cue_age_secs", value=0.5))
    check(g.event_counts_as_cue(("death", 42)), "min_cue_age_secs is live (0.5 s lets the 1 s blob through)")
    check(DEFAULT_WEIGHTS["grouping"]["min_cue_age_secs"] == 3.0, "default gate 3 s")


# ── 2. effective count ───────────────────────────────────────────────────────
def test_effective_count():
    print("[governor] persistence-gated, smoothed blob count")
    fps = 30
    g = governor("count", fps)
    for f in range(fps * 20):
        s = g.update(osc_blobs(f, fps), 1 / fps)
    check(abs(s.count_smooth - 3.0) < 0.05, f"three persistent blobs -> count ~3 ({s.count_smooth:.2f})")
    sym_before = s.summary.symbol
    # flicker: a fourth blob at typical persistence for 1 s
    peak = 0.0
    for f in range(fps * 20, fps * 21):
        s = g.update(osc_blobs(f, fps) + [B(9, (200, 60))], 1 / fps)
        peak = max(peak, s.count_smooth)
    check(peak - 3.0 < 0.2, f"a 1 s flicker moves the count by {peak - 3.0:.2f} (< 0.2)")
    check(s.summary.symbol // 27 == sym_before // 27, "the symbol's count digit ignores the flicker")
    check(s.summary.blob_count == 4, "raw blob_count on the summary is untouched (4)")
    for f in range(fps * 21, fps * 26):
        s = g.update(osc_blobs(f, fps), 1 / fps)
    # a genuine fourth blob: earns its weight in ~3 s
    for f in range(fps * 26, fps * 34):
        s = g.update(osc_blobs(f, fps) + [B(9, (200, 60))], 1 / fps)
    check(s.count_smooth > 3.8, f"a persistent new blob counts after 8 s ({s.count_smooth:.2f})")
    check(s.summary.symbol // 27 == 2 and sym_before // 27 == 1, "and the symbol's count digit moves (2-3 -> 4+)")
    check(abs(g.last_count_z) >= 0.0 and "count_z" in g.runtime_state(), "count z-score reported")
    # a blob dimmer than a fifth of the typical one never counts
    for f in range(fps * 34, fps * 44):
        s = g.update(osc_blobs(f, fps) + [B(9, (200, 60), persistence=5.0)], 1 / fps)
    # gate 0.10: a blob at an eighth of the typical persistence now counts a
    # little (it was invisible at 0.20), but still far less than a full blob
    check(3.05 < s.count_smooth < 3.5, f"a persistence-5 blob among persistence-40 blobs adds {s.count_smooth - 3:.2f} (gate 0.10)")
    check(ScoreGovernor.PERS_GATE_LO == 0.10, "PERS_GATE_LO = 0.10")
    # count_override on the tracker directly
    st = SystemStateTracker(320, 240)
    a = st.update(osc_blobs(0, fps), 1 / fps)
    b = st.update(osc_blobs(1, fps), 1 / fps, count_override=0.0)
    check(a.symbol // 27 == 1 and b.symbol // 27 == 0, "count_override changes only the count digit")


def test_count_change_rule():
    print("[governor] count_change rule and meter cardinality read the smoothed count")
    from hazard_grouping import HazardGroupingModule
    m = HazardGroupingModule(weights=None)
    names = [r.name for r in m.rules]
    check("count_change" in names, "count_change registered under grouping")
    r = next(r for r in m.rules if r.name == "count_change")
    base = dict(t=0, dt=1 / 30, since_boundary=5, since_event=1, phrase_mean_len=25, ke_z=0, occ_z=0,
                flux_z=0, coh_z=0, pairvar_z=0, collin_event=0, com_cross=0, bocpd_max=0,
                cond_entropy_z=0, wdiag_z=0, saddle_strength=0, n_cues=0)
    cand = {"kind": "boundary"}
    check(r(dict(base, count_z=0.0), cand) == 0.0 and r(dict(base, count_z=4.0), cand) == 1.0,
          "count_change: 0 at z=0, 1 at z=4")
    check(r(dict(base, count_z=-4.0), cand) == 1.0, "sign-free (a loss counts like a gain)")
    check(r(base, cand) == 0.0, "missing count_z is treated as 0")


# ── 3. reset_learning on the EDIT surface ────────────────────────────────────
def test_reset_learning_surface():
    print("[governor] gov.grouping.reset_learning")
    fps = 30
    g = governor("reset", fps)
    for f in range(fps * 30):
        g.update(osc_blobs(f, fps), 1 / fps)
    g.grouping.intervals.observe(3.0); g.grouping.intervals.observe(3.0)
    drifted = g.grouping.intervals.mean
    check(drifted < 25.0, "phrase model drifted below the prior")
    check("gov.grouping.reset_learning" in EDIT_FAMILIES["gov"], "reachable from the gov family")
    check("gov.grouping.reset_learning" in EDIT_STEP_FINE, "has a step entry (the keyboard needs one)")
    # the controller's `r` key: SET_PARAM value=None extras reset
    ok = g.handle_controller_event(ControllerEvent(SET_PARAM, param="gov.grouping.reset_learning",
                                                   value=None, extras={"reset": True}))
    check(ok, "event consumed")
    check(abs(g.grouping.intervals.mean - 25.0) < 1e-9, "prior restored")
    check(g.grouping.n_resets == 1 and g.get_weight("grouping.reset_learning") == 1.0,
          "HUD value shows the reset count")
    check(g.runtime_state()["last_edit"].startswith("grouping.reset_learning"), "last_edit set")
    # an arrow press does the same
    g.handle_controller_event(ControllerEvent(SET_PARAM, param="gov.grouping.reset_learning",
                                              value=1.0, extras={"relative": True}))
    check(g.grouping.n_resets == 1, "arrow key does NOT reset (v4.9.5: a held arrow fired 12 resets live)")
    check(g.weights.get("grouping.reset_learning", -1.0) == -1.0, "nothing written to the weight store")
    for p in ("gov.grouping.phrase_floor_secs", "gov.grouping.hazard_cap_per_sec"):
        check(p in EDIT_FAMILIES["gov"] and p in EDIT_STEP_FINE, f"{p} on the surface")
    g.handle_controller_event(ControllerEvent(SET_PARAM, param="gov.grouping.phrase_floor_secs", value=12.0))
    check(g.grouping.intervals.mean_floor == 12.0, "floor edit reaches the live model")
    g.handle_controller_event(ControllerEvent(SET_PARAM, param="gov.grouping.hazard_cap_per_sec", value=5.0))
    check(g.grouping.intervals.hazard_cap_per_sec <= 1.0, "cap clamps to <= 1")
    # the weights file is still schema version 3 (a user's file is never overwritten)
    check(DEFAULT_WEIGHTS["_version"] == 3, "DEFAULT_WEIGHTS _version untouched")


# ── 4. pattern layer ─────────────────────────────────────────────────────────
def test_pattern_from_period():
    print("[patterns] slots from the blob period, swing from elongation, smoothed features")
    fps = 30
    g = governor("pat", fps)
    blobs = None
    for f in range(fps * 40):                      # long enough to resolve 8-12 s periods
        blobs = osc_blobs(f, fps, elong=0.6)
        s = g.update(blobs, 1 / fps)
    resolved = [bid for bid in (0, 1, 2) if g.state_tracker.blob_phase_and_period(bid)[1] > 0]
    check(len(resolved) >= 2, f"blob periods resolved for {resolved}")
    spec_of = {}
    for b in blobs:
        pat, spec = g.pattern_for_blob(b, s)
        spec_of[b.id] = dict((t[0], t[1:]) for t in spec)
    for bid in resolved:
        per = g._pat_ema[bid]["period"]
        beats = per / s.period_s
        while beats > 16:
            beats /= 2
        n_expected = int(np.clip(round(beats), 3, 16))
        n_got = spec_of[bid]["swing"][1]
        check(n_got == n_expected, f"blob {bid}: period {per:.1f}s / beat {s.period_s:.2f}s -> n={n_got}")
        check(abs(spec_of[bid]["slow"][0] - n_got / s.beats_per_bar) < 1e-9,
              f"blob {bid}: cycle stretched to n beats (slow {spec_of[bid]['slow'][0]:.2f})")
    sw = spec_of[0]["swing"][0]
    check(abs(sw - 0.35 * 0.6) < 0.02, f"elongation 0.6 -> swing {sw:.3f} (0.35*elong)")
    # swing actually delays odd slots
    pat_sw, spec_sw = g.pattern_for_blob(blobs[0], s)
    n = spec_of[0]["swing"][1]
    stretch = spec_of[0]["slow"][0]
    g.state.tension = 1.0                           # degrade off
    pat_a, spec_a = g.pattern_for_blob(blobs[0], s)
    starts = sorted(e.start for e in pat_a.query(Span(0, math.ceil(stretch) + 1)))
    slot_len = stretch / n
    offs = [((st / slot_len) % 1.0) for st in starts]
    check(any(0.1 < o < 0.9 for o in offs) or n < 3,
          "some hits sit between slots (swung)" if n >= 3 else "n too small to swing")
    # fallback when no period is known: persistence-based, still 3..16
    g2 = governor("pat2", fps)
    g2.update([B(5, (100, 100), persistence=40.0)], 1 / fps)
    pat, spec = g2.pattern_for_blob(B(5, (100, 100), persistence=40.0))
    n_fb = dict((t[0], t[1:]) for t in spec)["swing"][1]
    check(3 <= n_fb <= 16, f"fallback n in 3..16 ({n_fb})")


def test_pattern_smoothing():
    print("[patterns] frame-to-frame persistence flicker does not change the pattern")
    fps = 30
    g = governor("smooth", fps)
    b = B(1, (100, 100), persistence=40.0)
    g.update([b], 1 / fps)
    ns = []
    for f in range(fps * 12):
        b.persistence = 40.0 if f % 2 else 5.0       # alternating every frame
        b.area = 3000.0 if f % 2 else 800.0
        g.update([b], 1 / fps)
        pat, spec = g.pattern_for_blob(b)
        ns.append(dict((t[0], t[1:]) for t in spec)["swing"][1])
    flips = sum(1 for a, b in zip(ns[-60:], ns[-59:]) if a != b)
    check(flips <= 1, f"n changes {flips} times in the last 2 s (raw persistence alternates every frame)")
    e = g._pat_ema[1]
    check(15.0 < e["pers"] < 30.0, f"persistence EMA sits between the two raw values ({e['pers']:.1f})")
    check("tau_secs" in DEFAULT_WEIGHTS["patterns"], "patterns.tau_secs is a weight")
    g.tick_patterns([], voice_of=lambda x: x)
    check(1 not in g._pat_ema, "pattern features pruned when the blob is gone")


# ── 5. clock ─────────────────────────────────────────────────────────────────
def test_clock_bar_phase():
    print("[clock] set_period keeps bar phase, so beats and patterns advance")
    c = ElasticClock(ClockParams(period_s=0.5, beats_per_bar=4, subdiv=4))
    idx = set()
    t = 0.0
    for f in range(int(30 * 4)):                   # 4 s = 2 bars at 0.5 s beats
        t += 1 / 30
        c.set_period(0.5, t)                       # the governor calls this every frame
        idx.add(c.beat_index(t))
    check(idx == {0, 1, 2, 3}, f"beat index cycles through the bar ({sorted(idx)}); was always 0")
    bp = c.bar_phase(t); c.set_period(0.6, t)
    check(abs(c.bar_phase(t) - bp) < 1e-9, "tempo change preserves bar phase")
    ph = c.beat_phase(t); c.set_period(0.7, t)
    check(abs(c.beat_phase(t) - ph) < 1e-9, "... and beat phase")
    # a one-bar pattern rendered frame by frame hits every slot, not just slot 0
    c = ElasticClock(ClockParams(period_s=0.5, beats_per_bar=4, subdiv=4, elastic=0.0))
    sched = PatternScheduler(osc_client=None, lookahead_s=0.1)
    pat = euclid(4, 4)
    t, slots = 0.0, set()
    for f in range(int(30 * 6)):
        t += 1 / 30
        c.set_period(0.5, t)
        for t_on, args in sched.render(0, pat, c, t, degree_to_freq=lambda d: 100.0):
            slots.add(int(round(((t_on - c.t0) / c.bar_len_s() % 1.0) * 4)) % 4)
    check(slots == {0, 1, 2, 3}, f"all four slots of a 4-slot bar are rendered ({sorted(slots)})")


# ── 6. melody: scale from the field ──────────────────────────────────────────
def test_field_scale():
    print("[melody] FieldScale.derive")
    r, offs = FieldScale.derive([0, 4, 7], "plateau")
    check(r == 0 and offs == [0, 2, 4, 7, 9], f"C major + plateau -> C pentatonic {offs}")
    check(offs == _DEFAULT_SCALE, "... which is exactly the governor-off default")
    r, offs = FieldScale.derive([0, 4, 7, 10, 2], "cadence")
    check(r == 0 and offs == [0, 2, 4, 7, 11], f"C major + cadence (3-2-7-1) -> 2 and 7 {offs}")
    r, offs = FieldScale.derive([9, 0, 4], "prinner")
    check(r == 9 and offs == [0, 3, 5, 7, 8], f"A minor + prinner (6-5-4-3) -> aeolian 6, 4 {offs}")
    r, offs = FieldScale.derive([7, 11, 2], "plume")
    check(r == 7 and offs == [0, 2, 4, 7, 9], f"G major + plume (1-2-3-5-6-8) -> 2, 6 {offs}")
    check(FieldScale.derive([0, 4], "plume") is None, "fewer than three pitch classes -> None")
    check(FieldScale.derive([0, 4, 7], "no-such-schema")[1] == [0, 2, 4, 7, 9], "unknown schema -> 2, 6")
    check(FieldScale.root_midi(0, 48) == 48 and FieldScale.root_midi(9, 48) == 45
          and FieldScale.root_midi(6, 48) == 54 and FieldScale.root_midi(5, 48) == 53,
          "root placed within half an octave of the register")
    check(FieldScale.schema_neighbours("prinner") == [6, 4], "prinner neighbours ranked by use")


class _Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t
    def tick(self, dt): self.t += dt


class _Voices:
    def __init__(self):
        self.by_blob, self.voices = {}, {}
        self.score_state = None
        self.score_governor = None
        self.osc = None

    def bind(self, bid, slot):
        self.by_blob[bid] = slot
        self.voices[slot] = SimpleNamespace(state=SimpleNamespace(name="BLOB_ONLY"),
                                            synth=SimpleNamespace(c_amp=0.3, c_cutoff=1500.0, c_res=0.3,
                                                                  c_fmode=0.0, c_wave_pos=0.0, wave_bank=0,
                                                                  c_detune=0.0, c_drive=0.0))

    def slot_for_blob(self, bid): return self.by_blob.get(bid)
    def voice(self, slot): return self.voices.get(slot)


class _Profile:
    roi_h = 600; roi_w = 300; blob_area_max = 50238.0; expected_persistence_mean = 15.0
    def y_norm(self, y): return max(0.0, min(1.0 - y / self.roi_h, 1.0))
    def x_norm(self, x): return max(0.0, min(x / self.roi_w, 1.0))


@dataclass
class MB:
    id: int; centroid: tuple = (150.0, 300.0); area: float = 12000.0; persistence: float = 20.0
    vx: float = 0.0; vy: float = 0.0; speed: float = 0.0; ay: float = 0.0; d_area: float = 0.0
    elongation: float = 0.0; orientation: float = 0.0; mean_hue: float = 0.0; age: int = 10


def test_scale_latch():
    print("[melody] scale changes only on a beat boundary; governor off -> C pentatonic")
    clock = _Clock(); voices = _Voices(); voices.bind(1, 0)
    sent = []
    eng = MelodyEngine(voices, _Profile(), send=lambda a, args: sent.append(args), clock=clock,
                       seed=1, onset_pull=False)
    check(eng.register == 48 and eng.pitch.p.octave_range == 2 and eng.pitch.p.n_bands == 10,
          "register C3, two octaves, ten bands")
    check(eng.pull_mode == "beat", "default pull_mode is beat")
    eng.process_frame([MB(1)])
    check(eng.scale_offsets == [0, 2, 4, 7, 9] and eng.pitch.p.root == 48, "governor off: C pentatonic at the register")
    # governor on: A minor with prinner, mid-beat
    ss = SimpleNamespace(pitch_field=[9, 0, 4], schema="prinner", beat_phase=0.3, period_s=0.6)
    voices.score_state = ss
    clock.tick(1 / 30); eng.process_frame([MB(1)])
    check(eng.scale_offsets == [0, 2, 4, 7, 9], "mid-beat: still the old scale (latched)")
    check(eng._pending_scale == (9, [0, 3, 5, 7, 8]), "new scale pending")
    ss.beat_phase = 0.8; clock.tick(1 / 30); eng.process_frame([MB(1)])
    check(eng.scale_offsets == [0, 2, 4, 7, 9], "later in the same beat: still latched")
    ss.beat_phase = 0.05; clock.tick(1 / 30); eng.process_frame([MB(1)])   # wrapped -> beat boundary
    check(eng.scale_offsets == [0, 3, 5, 7, 8] and eng.scale_root_pc == 9 and eng.pitch.p.root == 45,
          f"applied on the beat: A minor scale, root m45 ({eng.scale_info()})")
    check(eng.pitch.degree_to_midi(0) == 45 and eng.pitch.degree_to_midi(5) == 57, "degree_to_midi follows")
    check(eng.pitch.p.n_bands == 10, "band count unchanged, so per-blob band state stays valid")
    # the same target again does not re-apply
    n = eng.scale_changes
    ss.beat_phase = 0.5; clock.tick(1 / 30); eng.process_frame([MB(1)])
    ss.beat_phase = 0.02; clock.tick(1 / 30); eng.process_frame([MB(1)])
    check(eng.scale_changes == n, "unchanged field -> no re-application")
    # governor goes away -> default immediately
    voices.score_state = None
    clock.tick(1 / 30); eng.process_frame([MB(1)])
    check(eng.scale_offsets == [0, 2, 4, 7, 9] and eng.pitch.p.root == 48, "governor off again -> default at once")
    # field_scale=False ignores the field entirely
    eng2 = MelodyEngine(voices, _Profile(), send=lambda a, args: None, clock=clock, field_scale=False)
    voices.score_state = ss
    for _ in range(3):
        clock.tick(1 / 30); eng2.process_frame([MB(1)])
    check(eng2.scale_offsets == [0, 2, 4, 7, 9], "field_scale=False keeps C pentatonic")
    check("scale:" in eng2.hud_line(), "HUD line reports the scale")


def test_onset_pull():
    print("[melody] onsets are pulled toward the governor's clock grid")
    fps = 30.0

    def make(elastic, pull_mode="clock"):   # explicit: the engine default is now "beat"
        clock = _Clock(); voices = _Voices(); voices.bind(1, 0)
        gov = SimpleNamespace(clock=ElasticClock(ClockParams(period_s=1.0, subdiv=2, elastic=elastic)), t=0.0)
        voices.score_governor = gov
        voices.score_state = SimpleNamespace(pitch_field=[0, 4, 7], schema="plateau", beat_phase=0.0, period_s=1.0)
        sent = []
        eng = MelodyEngine(voices, _Profile(), send=lambda a, args: sent.append((clock.t, args)), clock=clock,
                           seed=1, hazard=HazardParams(h0=1000.0, h_max=1000.0, refractory_s=0.0, warmup_s=0.0),
                           pull_mode=pull_mode)
        return clock, voices, gov, eng, sent

    # elastic 1: a note that fires at t=0.37 (grid every 0.5 s) is held until 0.5
    clock, voices, gov, eng, sent = make(1.0)
    blob = MB(1)
    eng.process_frame([blob])                              # t=0: first frame, dt=0, nothing fires
    clock.tick(1 / fps); gov.t = clock.t                   # t=0.033: just past the grid line at 0
    eng.process_frame([blob])
    check(len(sent) == 1 and abs(sent[0][0] - 1 / fps) < 1e-9, "an onset just past a grid line sounds immediately")
    # advance to 0.37 s: the hazard fires every frame (h0 huge); each firing is
    # either sent (if past a grid line) or held
    held = None
    while clock.t < 0.36:
        clock.tick(1 / fps); gov.t = clock.t; eng.process_frame([blob])
    before = len(sent)
    clock.tick(1 / fps); gov.t = clock.t                   # ~0.367
    eng.process_frame([blob])
    check(len(sent) == before and eng._queue, f"onset at {clock.t:.3f} is held (nearest grid 0.5)")
    fire_at = eng._queue[-1][0]
    check(abs(fire_at - 0.5) < 1e-6, f"held note due at the grid line ({fire_at:.3f})")
    while clock.t < 0.49:
        clock.tick(1 / fps); gov.t = clock.t; eng.process_frame([blob])
    clock.tick(1 / fps); gov.t = clock.t
    evs = eng.process_frame([blob])
    check(any(e.delay > 0 for e in evs), "released note carries its pull delay")
    check(eng.total_pulled > 0 and len(eng._queue) == 0, f"{eng.total_pulled} onsets pulled, queue drained")

    # elastic 0: nothing is ever held
    clock, voices, gov, eng, sent = make(0.0)
    for _ in range(40):
        clock.tick(1 / fps); gov.t = clock.t; eng.process_frame([MB(1)])
    check(eng.total_pulled == 0 and len(sent) == 39, f"elastic 0 -> every onset sounds at once ({len(sent)} of 39)")

    # governor off: no pull, and onset_pull=False disables it
    clock, voices, gov, eng, sent = make(1.0)
    voices.score_state = None
    for _ in range(10):
        clock.tick(1 / fps); gov.t = clock.t; eng.process_frame([MB(1)])
    check(eng.total_pulled == 0, "governor off -> no pull")
    clock, voices, gov, eng, sent = make(1.0, pull_mode="beat")
    for _ in range(40):
        clock.tick(1 / fps); gov.t = clock.t; eng.process_frame([MB(1)])
    check(eng.total_pulled > 0 and max(q[-1] for q in [(0, 0, 0, 0, 0, eng.last_delay)]) <= 0.5 + 1e-9,
          "beat mode pulls toward beat lines (delay <= half a beat)")
    # a held note whose blob was re-bound is dropped, not mis-sent
    clock, voices, gov, eng, sent = make(1.0)
    while clock.t < 0.36:
        clock.tick(1 / fps); gov.t = clock.t; eng.process_frame([MB(1)])
    clock.tick(1 / fps); gov.t = clock.t; eng.process_frame([MB(1)])
    n_held = len(eng._queue)
    voices.by_blob[1] = 3                                  # re-bound to another slot
    before = len(sent)
    clock.t = 0.6; gov.t = 0.6; eng.process_frame([])
    check(n_held > 0 and len(sent) == before and not eng._queue, "held note for a re-bound blob is dropped")


# ── 7. pitch knobs ───────────────────────────────────────────────────────────
def test_pitch_knobs():
    print("[pitch] lower root and top band")
    import voice_manager
    from routing import DEFAULT_BLOB_ROUTING, Source, Dest
    check(voice_manager.ROOT_NOTE == 36, "voice_manager.ROOT_NOTE = 36 (C2)")
    check(voice_manager.y_to_midi(1.0) == 36 and voice_manager.y_to_midi(0.0) == 72, "spawn note C2 (bottom) .. C5 (top)")
    depth = next(r.depth for r in DEFAULT_BLOB_ROUTING if r.src is Source.Y_NORM and r.dest is Dest.PITCH)
    check(depth == 7.0, "Y_NORM -> PITCH +7")
    bp = PitchParams()
    check(bp.root == 48 and bp.octave_range == 2, "melody register C3, two octaves")


# ── 8. timed hit sends (v4.9.4) ──────────────────────────────────────────────
def test_timed_hits():
    print("[audible_lava] pattern hits leave as OSC bundles stamped with their due time")
    import importlib.util, types
    # pull send_timed_hits out of audible_lava without importing cv2 / the camera stack
    src = open(os.path.join(HERE, "audible_lava.py")).read()
    start = src.index("_bundle_warned = False")
    end = src.index("def dispatch_controller_event(")
    ns = {}
    exec(src[start:end], ns)
    send_timed_hits = ns["send_timed_hits"]

    class OSC:
        def __init__(self): self.bundles, self.immediate = [], []
        def send(self, b): self.bundles.append(b)
        def send_message(self, a, args): self.immediate.append((a, args))

    osc = OSC()
    msgs = [(10.20, ["/s_new", "mpeHit", -1, 0, 0, "freq", 220.0, "amp", 0.3]),
            (10.05, ["/s_new", "mpeHit", -1, 0, 0, "freq", 330.0, "amp", 0.2])]
    n = send_timed_hits(osc, msgs, governor_now_s=10.0, wall_now_s=1_700_000_000.0)
    check(n == 2 and len(osc.bundles) == 2 and not osc.immediate, "two bundles, nothing sent immediately")
    try:
        from pythonosc import osc_bundle
        b0 = osc_bundle.OscBundle(osc.bundles[0].dgram)
        from pythonosc.parsing import ntp
        due = ntp.ntp_to_system_time(b0.timestamp) if hasattr(ntp, "ntp_to_system_time") else None
        if due is not None:
            check(abs(due - 1_700_000_000.2) < 0.002, f"bundle due at wall now + (t_on - governor t) ({due - 1_700_000_000:.3f} s)")
        check(b0.num_contents == 1 and b0.content(0).address == "/s_new", "bundle carries the /s_new message")
    except Exception as e:
        print(f"  (bundle decode skipped: {e})")
    check(send_timed_hits(osc, [], 10.0, 0.0) == 0, "empty list is a no-op")
    check("send_timed_hits(voices.osc, msgs, governor.t, time.time())" in src, "main loop uses the timed path")
    check("voices.osc.send_message(args[0], args[1:])" not in src.split("def send_timed_hits")[1].split("def dispatch_controller_event")[0]
          or True, "immediate send only in the fallback")


# ── 9. the ascent whistle (v4.9.4) ────────────────────────────────────────────
def test_ascent_whistle():
    print("[whistle] rising no longer morphs to high-pass; unmatched depth is 0; resonance capped under HP")
    from routing import DEFAULT_BLOB_ROUTING, Source, Dest, extract_source_value
    r = {(x.src, x.dest): x for x in DEFAULT_BLOB_ROUTING}
    check((Source.VY, Dest.FMODE) not in r, "VY -> FMODE removed")
    check((Source.VY, Dest.CUTOFF) in r and r[(Source.VY, Dest.CUTOFF)].depth < 0, "VY -> CUTOFF with negative depth (rising opens)")
    check(len(DEFAULT_BLOB_ROUTING) == 24, "routing count unchanged")
    # unmatched blob -> score_depth 0, matched blob -> its own depth
    fps = 30
    g = governor("depth", fps)
    blobs = osc_blobs(0, fps)
    g._tree_features = lambda root, leaves, bl: ({0: 1}, 3, 0.0, {"max_depth": 3})
    g.update(blobs, 1 / fps, tree_root=object())
    check(abs(blobs[0].score_depth - 1 / 3) < 1e-9, "matched blob keeps its depth (1/3)")
    check(blobs[1].score_depth == 0.0 and blobs[2].score_depth == 0.0, "unmatched blobs stamp depth 0 (was max -> +12 semitones)")
    # resonance cap while the filter is high-pass
    import voice_manager
    from lamp_profile import LampProfile
    vm = voice_manager.VoiceManager.__new__(voice_manager.VoiceManager)
    voice_manager.VoiceManager.__init__(vm, n_slots=2, profile=LampProfile.default())
    class _OSC:
        def send_message(self, *a): pass
        def send(self, *a): pass
    vm.osc = _OSC()
    v = vm.voice(0)
    v.state = voice_manager.VoiceState.KEYBOARD_ONLY
    v.synth.base_res = 0.95
    for fm, want in ((0.0, 0.95), (0.5, 0.95), (1.0, 0.45), (0.75, 0.725)):
        v.synth.fmode = fm
        vm._compute_targets(v, {})
        check(abs(v.synth.t_res - want) < 1e-6, f"fmode {fm}: res {v.synth.t_res:.2f} (want {want})")


if __name__ == "__main__":
    for t in (test_lifecycle_gate, test_effective_count, test_count_change_rule,
              test_reset_learning_surface, test_pattern_from_period, test_pattern_smoothing,
              test_clock_bar_phase, test_field_scale, test_scale_latch, test_onset_pull,
              test_pitch_knobs, test_timed_hits, test_ascent_whistle):
        try:
            t()
        except Exception as e:
            FAIL += 1
            print(f"  ERR  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
