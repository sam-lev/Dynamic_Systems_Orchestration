#!/usr/bin/env python3
# test_melody_rhythm.py -- tests for melody_rhythm.py
#
# No SuperCollider, camera, or pygame. Uses fake blobs, a fake LampProfile,
# a fake VoiceManager, and a recording OSC sender. Run:
#     python test_melody_rhythm.py

import math
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace

from melody_rhythm import (
    HazardOnsets, HazardParams, BandPitch, PitchParams,
    MelodyEngine, midi_to_freq,
)

PASS = FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {msg}")
    else:
        FAIL += 1
        print(f"  FAIL {msg}")


# ── Fakes ────────────────────────────────────────────────────────────────────
class FakeProfile:
    """Mirrors lamp_profile.LampProfile's v4.1 contract: y_norm 1 = top."""
    roi_h = 600
    roi_w = 300
    blob_area_max = 50238.0
    expected_persistence_mean = 15.0

    def y_norm(self, roi_y):
        return max(0.0, min(1.0 - roi_y / self.roi_h, 1.0))

    def x_norm(self, roi_x):
        return max(0.0, min(roi_x / self.roi_w, 1.0))


@dataclass
class FakeBlob:
    id: int
    centroid: tuple = (150.0, 300.0)
    area: float = 12000.0
    persistence: float = 20.0
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0
    ay: float = 0.0
    d_area: float = 0.0
    elongation: float = 0.0
    orientation: float = 0.0
    mean_hue: float = 0.0
    age: int = 10


class FakeSynth(SimpleNamespace):
    pass


def make_synth():
    return FakeSynth(c_amp=0.3, c_cutoff=1500.0, c_res=0.3, c_fmode=0.0,
                     c_wave_pos=0.0, wave_bank=0, c_detune=0.0, c_drive=0.0)


class FakeVoices:
    """Enough of VoiceManager for MelodyEngine: slot_for_blob + voice()."""
    def __init__(self):
        self.by_blob = {}
        self.voices = {}

    def bind(self, blob_id, slot_id, state="BLOB_ONLY"):
        self.by_blob[blob_id] = slot_id
        self.voices[slot_id] = SimpleNamespace(
            state=SimpleNamespace(name=state), synth=make_synth())

    def slot_for_blob(self, blob_id):
        return self.by_blob.get(blob_id)

    def voice(self, slot_id):
        return self.voices.get(slot_id)


class Clock:
    def __init__(self, t=0.0): self.t = t
    def __call__(self): return self.t
    def tick(self, dt): self.t += dt


# ── 1. Hazard onsets ─────────────────────────────────────────────────────────
def test_hazard_rate_shape():
    print("[hazard] rate function")
    h = HazardOnsets(HazardParams(), seed=1)
    check(abs(h.rate(0, 0) - 0.25) < 1e-9, "still blob -> h0 (0.25/s)")
    check(h.rate(0.5, 0) > h.rate(0, 0), "speed raises rate")
    check(h.rate(0, 0.5) > h.rate(0, 0), "accel raises rate")
    check(h.rate(1, 1) == HazardParams().h_max, "rate clamps at h_max")
    check(h.expected_interval(0, 0) == 4.0, "still blob: one note per 4 s")


def run_hazard(speed_n, accel_n, seconds, dt=1 / 15, seed=3):
    h = HazardOnsets(HazardParams(), seed=seed)
    t, n = 0.0, 0
    while t < seconds:
        if h.step(7, speed_n, accel_n, dt, t):
            n += 1
        t += dt
    return n


def test_hazard_counts():
    print("[hazard] fire counts over 120 s at 15 fps")
    still = run_hazard(0.0, 0.0, 120)
    fast  = run_hazard(0.8, 0.0, 120)
    turn  = run_hazard(0.2, 0.9, 120)
    print(f"       still={still} fast={fast} turning={turn}")
    # Expected: still ≈ 120*0.25 ≈ 30 (accumulator, near-deterministic)
    check(26 <= still <= 36, "still blob fires ~0.25/s")
    check(fast > 3 * still, "fast blob fires far more often")
    check(turn > still, "accelerating blob fires more often")
    # The random residual after a fire (mean reset_jitter/2) shortens the next
    # interval, so the effective ceiling is h_max / (1 - reset_jitter/2).
    p = HazardParams()
    ceiling = 120 * p.h_max / (1.0 - p.reset_jitter / 2.0) + 2
    check(fast <= ceiling, f"never exceeds effective ceiling ({ceiling:.0f})")


def test_hazard_refractory_and_warmup():
    print("[hazard] refractory + warm-up")
    p = HazardParams(h0=10.0, h_max=10.0, refractory_s=0.25, warmup_s=0.5)
    h = HazardOnsets(p, seed=0)
    times = []
    t, dt = 0.0, 0.01
    while t < 3.0:
        if h.step(1, 0, 0, dt, t):
            times.append(t)
        t += dt
    check(times and times[0] >= 0.5 - 1e-9, "no fire during warm-up")
    gaps = [b - a for a, b in zip(times, times[1:])]
    check(gaps and min(gaps) >= 0.25 - 1e-9, "refractory spacing enforced")


def test_hazard_prune():
    print("[hazard] prune")
    h = HazardOnsets(seed=0)
    h.step(1, 0, 0, 0.1, 0.0); h.step(2, 0, 0, 0.1, 0.0)
    h.prune([2])
    check(h.state(1) is None and h.state(2) is not None, "dead blob state dropped")


# ── 2. Band pitch ────────────────────────────────────────────────────────────
def test_band_mapping():
    print("[pitch] band -> midi")
    bp = BandPitch(PitchParams(octave_range=3))
    check(bp.p.n_bands == 15, "15 bands for pentatonic x 3 octaves")
    check(BandPitch(PitchParams()).p.n_bands == 10, "v4.9.3 default: 10 bands (2 octaves)")
    check(BandPitch(PitchParams()).degree_to_midi(9) == 48 + 12 + 9, "v4.9.3 default top band -> A4 (m69)")
    check(bp.degree_to_midi(0) == 48, "band 0 -> C3")
    check(bp.degree_to_midi(5) == 60, "band 5 -> C4 (next octave)")
    check(bp.degree_to_midi(14) == 48 + 24 + 9, "band 14 -> A5")
    check(bp.raw_band(0.0) == 0 and bp.raw_band(0.999) == 14, "height extremes")


def test_hysteresis_no_flicker():
    print("[pitch] hysteresis at a band boundary")
    bp = BandPitch(PitchParams(octave_range=3, hysteresis=0.25))
    bands = []
    # Boundary between band 7 and 8 is at height 8/15. Jitter +/- 0.1 band.
    boundary = 8 / 15
    for i in range(200):
        jitter = (0.1 / 15) * (1 if i % 2 else -1)
        st = bp.update(1, boundary + jitter, 0.0, i / 15)
        bands.append(st.band)
    check(len(set(bands)) == 1, f"jitter of 0.1 band never re-bands (band {bands[0]})")

    # A real move of 1.5 bands does re-band.
    st = bp.update(1, boundary + 1.5 / 15, 0.0, 20.0)
    check(st.band == 9, "crossing beyond hysteresis re-bands")


def test_contour_dwell():
    print("[pitch] contour inertia (dwell)")
    bp = BandPitch(PitchParams(octave_range=3, dwell_s=0.8))
    t = 0.0
    bp.update(1, 0.5, 0.0, t)
    # Rising for 0.3 s: not long enough
    for _ in range(5):
        t += 1 / 15
        st = bp.update(1, 0.5, -1.0, t)
    check(st.direction == 0, "0.3 s of rising does not flip contour")
    for _ in range(12):
        t += 1 / 15
        st = bp.update(1, 0.5, -1.0, t)
    check(st.direction == +1, "1.1 s of rising flips contour up")
    # Brief downward blip
    t += 1 / 15
    st = bp.update(1, 0.5, +1.0, t)
    check(st.direction == +1 and not st.last_turn, "one falling frame is ignored")
    for _ in range(14):
        t += 1 / 15
        st = bp.update(1, 0.5, +1.0, t)
    check(st.direction == -1 and st.last_turn, "sustained reversal = turning point")


def test_stepwise_and_leaps():
    print("[pitch] interval size from speed, walking toward the band")
    bp = BandPitch(PitchParams(octave_range=3))
    bp.update(1, 0.0, 0.0, 0.0)          # band 0, deg 0
    # Move the attractor far away (band 10) without a direction; speed slow.
    bp.update(1, 10.5 / 15, 0.0, 1.0)
    st = bp.state(1)
    check(st.band == 10 and st.deg == 0, "attractor moved, note has not")
    d1 = bp.next_degree(1, speed_n=0.1)
    check(d1 == 1, "slow: step of 1 toward band")
    d2 = bp.next_degree(1, speed_n=0.4)
    check(d2 == 3, "medium: step of 2")
    d3 = bp.next_degree(1, speed_n=0.9)
    check(d3 == 6, "fast: step of 3")
    # Keep stepping; must land on band, never overshoot
    seq = [bp.next_degree(1, 0.9) for _ in range(5)]
    check(max(seq) <= 10 and seq[-1] == 10, f"converges to band without overshoot {seq}")
    midi = bp.next_midi(1, 0.1)
    check(midi == bp.degree_to_midi(10), "next_midi returns scale pitch")


def test_contour_within_slack():
    print("[pitch] contour wanders only within slack, then resolves")
    bp = BandPitch(PitchParams(octave_range=3, slack=2, dwell_s=0.0))
    bp.update(1, 7.5 / 15, 0.0, 0.0)         # band 7
    bp.update(1, 7.5 / 15, -1.0, 0.0)        # rising, dwell 0 -> immediate
    bp.update(1, 7.5 / 15, -1.0, 0.1)
    check(bp.state(1).direction == +1, "direction up")
    seq = [bp.next_degree(1, 0.1) for _ in range(6)]
    check(seq[0] == 8 and seq[1] == 9, f"ascends stepwise {seq}")
    check(seq[2] == 7, "beyond slack -> resolves to band")
    check(all(abs(d - 7) <= 2 for d in seq), "never leaves band +/- slack")


def test_turning_point_resolves():
    print("[pitch] turning point forces resolution")
    bp = BandPitch(PitchParams(octave_range=3, dwell_s=0.0, slack=3))
    bp.update(1, 7.5 / 15, 0.0, 0.0)
    bp.update(1, 7.5 / 15, -1.0, 0.0); bp.update(1, 7.5 / 15, -1.0, 0.1)
    bp.next_degree(1, 0.1); bp.next_degree(1, 0.1)
    check(bp.state(1).deg == 9, "walked up to 9")
    bp.update(1, 7.5 / 15, +1.0, 0.2); bp.update(1, 7.5 / 15, +1.0, 0.3)
    check(bp.state(1).last_turn, "reversal flagged")
    check(bp.next_degree(1, 0.1) == 7, "resolves onto band at turning point")


# ── 3. Engine integration (fake VoiceManager, recorded OSC) ─────────────────
def test_engine_end_to_end():
    print("[engine] end-to-end with fake voices")
    prof   = FakeProfile()
    voices = FakeVoices()
    voices.bind(1, 0)                 # blob 1 -> slot 0 (BLOB_ONLY)
    voices.bind(2, 1, state="ZOMBIE")  # zombie slot: must not play
    sent   = []
    clock  = Clock()
    eng    = MelodyEngine(voices, prof, send=lambda a, args: sent.append((a, args)),
                          seed=5, clock=clock)

    fast   = FakeBlob(1, centroid=(150.0, 150.0), speed=2.5, vy=-2.5, area=20000)
    zomb   = FakeBlob(2, centroid=(150.0, 450.0), speed=2.5, vy=-2.5)
    unbound = FakeBlob(3, centroid=(150.0, 450.0), speed=2.5, vy=-2.5)

    notes = []
    for _ in range(15 * 30):          # 30 s
        clock.tick(1 / 15)
        notes += eng.process_frame([fast, zomb, unbound])

    check(len(notes) > 5, f"fast bound blob produced notes ({len(notes)})")
    check(all(ev.blob_id == 1 for ev in notes), "zombie + unbound blobs silent")
    check(len(sent) == len(notes), "one /s_new per note")
    addr, args = sent[0]
    check(addr == "/s_new" and args[0] == "melodyHit" and args[1] == -1,
          "one-shot /s_new with server-assigned node id")
    kv = dict(zip(args[4::2], args[5::2]))
    check(set(["freq", "amp", "pan", "dur", "attack", "release", "cutoff",
               "res", "fmode", "wave", "bank", "detune", "drive"]) <= set(kv),
          "all \\melodyHit params present")
    check(abs(kv["freq"] - midi_to_freq(notes[0].midi)) < 1e-6, "freq matches midi")
    check(abs(kv["dur"] - (0.25 + 2.5 * math.sqrt(20000 / 50238.0))) < 1e-6,
          "governor off: duration from area (seconds)")
    check(notes[0].beats == 0.0, "governor off: beats = 0 (fallback)")
    check(kv["release"] == 1.2, "non-shrinking blob -> legato release")
    midis = [ev.midi for ev in notes]
    top_band_midi = eng.pitch.degree_to_midi(eng.pitch.raw_band(prof.y_norm(150.0)))
    check(all(48 <= m <= 48 + 33 for m in midis), "pitches inside 3-octave range")
    check(midis[-1] == top_band_midi or abs(midis[-1] - top_band_midi) <= 7,
          f"settles near the blob's band (last {midis[-1]}, band {top_band_midi})")

    # Disable -> silence; state pruned when blob leaves
    eng.enabled = False
    before = len(sent)
    for _ in range(15 * 5):
        clock.tick(1 / 15)
        eng.process_frame([fast])
    check(len(sent) == before, "disabled engine sends nothing")
    eng.process_frame([])
    check(eng.hazard.state(1) is None and eng.pitch.state(1) is None,
          "state pruned when blob disappears")


def test_engine_amp_floor_and_shrink():
    print("[engine] amp floor + staccato on shrinking blob")
    prof, voices, sent = FakeProfile(), FakeVoices(), []
    voices.bind(1, 0)
    voices.voice(0).synth.c_amp = 0.0           # gated-silent drone
    clock = Clock()
    eng = MelodyEngine(voices, prof, hazard=HazardParams(h0=5.0, h_max=5.0, warmup_s=0),
                       send=lambda a, args: sent.append(args), seed=1, clock=clock)
    b = FakeBlob(1, d_area=-0.3)
    for _ in range(30):
        clock.tick(1 / 15); eng.process_frame([b])
    kv = dict(zip(sent[0][4::2], sent[0][5::2]))
    check(kv["amp"] == eng.amp_floor, "silent drone still gets audible floor")
    check(kv["release"] == 0.25, "shrinking blob -> staccato release")


def test_engine_beat_quantised_duration():
    print("[engine] duration in whole beats when the governor clock is present")
    from melody_rhythm import beats_for_area, BEAT_VALUES
    prof, voices, sent = FakeProfile(), FakeVoices(), []
    voices.bind(1, 0); voices.bind(2, 1)
    voices.score_state = SimpleNamespace(period_s=0.8)       # 75 BPM
    clock = Clock()
    eng = MelodyEngine(voices, prof, hazard=HazardParams(h0=5.0, h_max=5.0, warmup_s=0),
                       send=lambda a, args: sent.append(args), seed=1, clock=clock)
    small = FakeBlob(1, area=2000)                            # area_n 0.04 -> 1/2 beat
    big   = FakeBlob(2, area=40000)                           # area_n 0.80 -> 4 beats
    notes = []
    for _ in range(30):
        clock.tick(1 / 15); notes += eng.process_frame([small, big])
    by = {}
    for n in notes: by.setdefault(n.blob_id, n)
    check(by[1].beats == 0.5 and abs(by[1].dur - 0.4) < 1e-9, "small blob: 1/2 beat = 0.4 s")
    check(by[2].beats == 4.0 and abs(by[2].dur - 3.2) < 1e-9, "big blob: 4 beats = 3.2 s")
    check(all(n.beats in BEAT_VALUES for n in notes), "every note is a listed beat value")
    voices.score_state.period_s = 0.4                         # tempo doubles
    clock.tick(1 / 15); n2 = eng.process_frame([small, big])
    for _ in range(30):
        clock.tick(1 / 15); n2 += eng.process_frame([small, big])
    b2 = next(n for n in n2 if n.blob_id == 2)
    check(abs(b2.dur - 1.6) < 1e-9, "duration follows the clock (4 beats = 1.6 s)")
    voices.score_state = None
    clock.tick(1 / 15); n3 = []
    for _ in range(30):
        clock.tick(1 / 15); n3 += eng.process_frame([small, big])
    check(all(n.beats == 0.0 for n in n3), "governor removed -> seconds fallback")
    check(beats_for_area(0.0) == 0.5 and beats_for_area(1.0) == 4.0, "beats_for_area bounds")


if __name__ == "__main__":
    test_engine_beat_quantised_duration()
    test_hazard_rate_shape()
    test_hazard_counts()
    test_hazard_refractory_and_warmup()
    test_hazard_prune()
    test_band_mapping()
    test_hysteresis_no_flicker()
    test_contour_dwell()
    test_stepwise_and_leaps()
    test_contour_within_slack()
    test_turning_point_resolves()
    test_engine_end_to_end()
    test_engine_amp_floor_and_shrink()
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
