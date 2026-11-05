#!/usr/bin/env python3
# test_synth_palette.py -- Audible Lava / DSO v4.10
# Tests for synth_palette.py. No SuperCollider, camera or pygame: every OSC
# send goes to a recorder. Runs standalone:  python test_synth_palette.py

from __future__ import annotations

import math
import sys
import types

import synth_palette as sp
from synth_palette import (CLASSIC, RICH, HIT_MODEL_PLUCKED, HIT_MODEL_PIANO,
                           MEL_MODEL_BOWED, MEL_MODEL_PLUCKED, RichPatternScheduler,
                           RichMelodyEngine, VoiceExtras, MasterSpace, PreSoundLayer,
                           master_params_from_state, triad_freqs, parse_triad,
                           voice_extras_for_blob, melody_model_for_blob, install)
from pattern_language import PatternScheduler, euclid
from elastic_clock import ElasticClock, ClockParams


# ── fixtures ─────────────────────────────────────────────────────────────────

class Recorder:
    def __init__(self):
        self.msgs = []

    def __call__(self, addr, args):
        self.msgs.append((addr, list(args)))

    def send_message(self, addr, args):          # osc-client shape
        self(addr, args)

    def of(self, addr):
        return [a for ad, a in self.msgs if ad == addr]

    def clear(self):
        self.msgs = []


class Profile:
    lamp_y1, lamp_y2 = 0.0, 400.0
    expected_persistence_mean = 20.0
    blob_area_max = 20000.0

    def y_norm(self, y):            # 1 = top of lamp (project convention)
        return 1.0 - max(0.0, min(1.0, y / 400.0))

    def x_norm(self, x):
        return max(0.0, min(1.0, x / 300.0))


class Blob:
    def __init__(self, id, cx=150.0, cy=200.0, **kw):
        self.id = id
        self.centroid = (cx, cy)
        self.area = kw.pop("area", 5000.0)
        self.persistence = kw.pop("persistence", 30.0)
        self.elongation = kw.pop("elongation", 0.3)
        self.orientation = kw.pop("orientation", 0.0)
        self.mean_hue = 20.0
        self.direction = "rising"
        for k, v in kw.items():
            setattr(self, k, v)


class State:
    def __init__(self, name):
        self.name = name


class Synth:
    c_amp = 0.3; c_cutoff = 1500.0; c_res = 0.3; c_fmode = 0.0; c_wave_pos = 0.2
    wave_bank = 1; c_detune = 0.1; c_drive = 0.2


class Voice:
    def __init__(self, slot_id, blob_id=None, sc_node=None, state="BLOB_ONLY", kind="mpe"):
        self.slot_id = slot_id; self.blob_id = blob_id; self.sc_node = sc_node
        self.state = State(state); self.synth_kind = kind; self.synth = Synth()


class Voices:
    """Minimal VoiceManager stand-in: the attributes synth_palette touches."""
    def __init__(self, osc):
        self.osc = osc
        self.mpe_synthdef = "mpeVoice"; self.angle_synthdef = "blobAngle"
        self.strike_synthdef = "blobStrike"; self.bass_synthdef = "bassStutter"
        self.voices_by_slot = []
        self.voices_by_blob = {}
        self.score_gain = 1.0
        self.score_state = None

    def slot_for_blob(self, bid):
        return self.voices_by_blob.get(bid)

    def voice(self, slot):
        return self.voices_by_slot[slot]


class Gov:
    def __init__(self):
        self.scheduler = PatternScheduler(osc_client=None, lookahead_s=0.3, synthdef="mpeHit")
        self.pair_slot_base = 10000


class SS:
    """ScoreState look-alike with the fields the palette reads."""
    def __init__(self, **kw):
        self.tension = 0.0; self.ambiguity = 0.0; self.surprisal = 0.0; self.phrase_position = 0.0
        self.kuramoto = 0.0; self.count_smooth = 0.0; self.register_center = 0.5
        self.boundary = False; self.boundary_strength = 0.0
        self.triad = "C"; self.predicted_triad = ""; self.harmony_posterior_other = 0.0
        for k, v in kw.items():
            setattr(self, k, v)


def argdict(args):
    """Flatten an OSC arg list into {name: value}, skipping the header
    (address, synthdef name, node id / addAction / target)."""
    a = list(args)
    if a and isinstance(a[0], str) and a[0].startswith("/"):
        a = a[1:]
    if a and isinstance(a[0], str) and len(a) > 1 and not isinstance(a[1], str):
        a = a[1:]                                   # synthdef name
    while a and not isinstance(a[0], str):
        a = a[1:]                                   # node id / addAction / target
    d = {}
    for i in range(0, len(a) - 1, 2):
        d[a[i]] = a[i + 1]
    return d


# ── tests ────────────────────────────────────────────────────────────────────

def test_apply_palette_rich_and_back():
    rec = Recorder(); v = Voices(rec); g = Gov()
    class M: synthdef = "melodyHit"
    m = M()
    sp.apply_palette(v, RICH, governor=g, melody=m)
    assert (v.mpe_synthdef, v.angle_synthdef, v.strike_synthdef, v.bass_synthdef) == \
           ("mpeVoiceRich", "blobAngleRich", "blobStrikeRich", "bassStutterRich")
    assert isinstance(g.scheduler, RichPatternScheduler) and g.scheduler.synthdef == "mpeHitRich"
    assert m.synthdef == "melodyHitRich"
    sent = g.scheduler._sent_until
    sp.apply_palette(v, CLASSIC, governor=g, melody=m)
    assert v.mpe_synthdef == "mpeVoice" and g.scheduler.synthdef == "mpeHit"
    assert not isinstance(g.scheduler, RichPatternScheduler)
    assert g.scheduler._sent_until is sent, "rendered-until marks must survive a swap"
    assert m.synthdef == "melodyHit"


def test_scheduler_injects_model_per_layer():
    clock = ElasticClock(ClockParams(period_s=0.5, beats_per_bar=4))
    s = RichPatternScheduler.from_scheduler(PatternScheduler(lookahead_s=2.5, synthdef="mpeHit"))
    s.synthdef = "mpeHitRich"
    pat = euclid(4, 4)
    orn = s.render(3, pat, clock, 0.0, lambda d: 220.0, amp=0.3,
                   extra_params={"cutoff": 1000.0, "drive": 0.2})
    pair = s.render(10003, pat, clock, 0.0, lambda d: 110.0, amp=0.3,
                    extra_params={"cutoff": 600.0, "fmode": 0.0})
    assert orn and pair
    for _t, args in orn:
        d = argdict(args)
        assert args[1] == "mpeHitRich"
        assert d["model"] == HIT_MODEL_PLUCKED and d["cutoff"] == 1000.0 and d["drive"] == 0.2
    for _t, args in pair:
        d = argdict(args)
        assert d["model"] == HIT_MODEL_PIANO and d["cutoff"] == 600.0 and d["fmode"] == 0.0


def test_scheduler_governor_params_win():
    clock = ElasticClock(ClockParams(period_s=0.5, beats_per_bar=4))
    s = RichPatternScheduler(lookahead_s=2.5, synthdef="mpeHitRich")
    out = s.render(0, euclid(2, 2), clock, 0.0, lambda d: 220.0,
                   extra_params={"model": 0})          # explicit override
    assert out and all(argdict(a)["model"] == 0 for _t, a in out)


def test_melody_model_from_blob():
    p = Profile()
    grow = Blob(1, d_area=0.2, speed=0.0)
    shrink = Blob(2, d_area=-0.2, speed=3.0)
    a = argdict([0] + melody_model_for_blob(grow, p))
    b = argdict([0] + melody_model_for_blob(shrink, p))
    assert a["model"] == MEL_MODEL_BOWED and abs(a["bow"] - 0.15) < 1e-6
    assert b["model"] == MEL_MODEL_PLUCKED and abs(b["bow"] - 0.95) < 1e-6


def test_rich_melody_engine_appends_extras():
    rec = Recorder(); p = Profile(); v = Voices(rec)
    v.voices_by_slot = [Voice(0, blob_id=7, sc_node=1000)]
    v.voices_by_blob = {7: 0}
    t = [100.0]
    eng = RichMelodyEngine(v, p, send=rec, clock=lambda: t[0], seed=1)
    assert eng.synthdef == "melodyHitRich"
    blob = Blob(7, cy=100.0, d_area=0.1, speed=1.5)
    # drive the hazard until a note fires
    for _ in range(400):
        t[0] += 0.1
        if eng.process_frame([blob]):
            break
    assert eng.total_notes >= 1, "hazard never fired"
    args = rec.of("/s_new")[-1]
    assert args[0] == "melodyHitRich"
    d = argdict(args)
    assert d["model"] == MEL_MODEL_BOWED and "bow" in d and "freq" in d and "release" in d
    assert eng._note_extra == [], "extras must not leak into later sends"
    # a direct non-note send is untouched
    rec.clear(); eng._send("/n_set", [1, "amp", 0.5])
    assert rec.msgs[-1] == ("/n_set", [1, "amp", 0.5])


def test_voice_extras_mapping_and_dedup():
    rec = Recorder(); p = Profile(); v = Voices(rec)
    v.voices_by_slot = [Voice(0, blob_id=1, sc_node=1000),
                        Voice(1, blob_id=2, sc_node=1002, kind="bass"),   # bass: skipped
                        Voice(2, blob_id=3, sc_node=1004, state="ZOMBIE"),  # zombie: skipped
                        Voice(3, None, None, state="IDLE")]
    top = Blob(1, cy=0.0, area=20000.0, vx=3.0)          # top, huge, drifting sideways
    ex = voice_extras_for_blob(top, p)
    assert ex["formant"] == 1.0 and abs(ex["sub"] - 0.85) < 1e-6 and abs(ex["chorus"] - 0.8) < 1e-6
    bottom = Blob(9, cy=400.0, area=0.0, vx=0.0)
    ex2 = voice_extras_for_blob(bottom, p)
    assert ex2 == {"formant": 0.0, "sub": 0.0, "chorus": 0.0}
    extras = VoiceExtras(v, p, send=rec)
    n = extras.update([top, Blob(2), Blob(3)])
    assert n == 1 and rec.of("/n_set")[0][0] == 1000
    d = argdict(rec.of("/n_set")[0]); assert d["formant"] == 1.0
    assert extras.update([top, Blob(2), Blob(3)]) == 0, "unchanged values are not resent"
    top.centroid = (150.0, 200.0)
    assert extras.update([top]) == 1, "a moved blob resends"


def test_master_params_mapping():
    neutral = master_params_from_state(None)
    assert neutral.t60 == 2.5 and neutral.wet == 0.25 and neutral.holeMix == 0.0
    calm = master_params_from_state(SS())
    tense = master_params_from_state(SS(tension=1.0, ambiguity=1.0, surprisal=6.0,
                                        phrase_position=1.0, kuramoto=1.0, count_smooth=6.0,
                                        register_center=1.0))
    assert tense.t60 > calm.t60 and tense.damp < calm.damp and tense.drive > calm.drive
    assert tense.holeFb > calm.holeFb and tense.wet > calm.wet and tense.holeMix > calm.holeMix
    assert tense.size < calm.size and tense.modDepth < calm.modDepth and tense.highcut > calm.highcut
    assert 0.0 <= tense.wet <= 0.9 and 0.3 <= tense.t60 <= 40 and 0.5 <= tense.size <= 5
    # the hall only opens above tension 0.35 (v4.10.6)
    assert master_params_from_state(SS(tension=0.3)).holeMix == 0.0
    assert master_params_from_state(SS(tension=1.0)).holeMix > 0.25
    half = master_params_from_state(SS(tension=1.0), depth=0.5)
    full = master_params_from_state(SS(tension=1.0), depth=1.0)
    assert abs(half.drive - 0.5 * full.drive) < 1e-9
    assert master_params_from_state(SS(tension=1.0), depth=0.0).drive == neutral.drive


def test_master_space_spawn_update_bloom_close():
    rec = Recorder()
    m = MasterSpace(send=rec)
    m.spawn()
    sn = rec.of("/s_new")[0]
    assert sn[:4] == ["lavaMaster", 900, 1, 0], "tail of group 0 (addAction 1)"
    assert "bypass" in argdict(sn)
    rec.clear()
    m.update(SS(tension=0.5))
    d = argdict(rec.of("/n_set")[0])
    assert "t_bloom" not in d and abs(d["drive"] - 0.175) < 1e-9
    rec.clear()
    m.update(SS(boundary=True, boundary_strength=0.1))      # below bloom_min_strength
    assert "t_bloom" not in argdict(rec.of("/n_set")[0])
    rec.clear()
    m.update(SS(boundary=True, boundary_strength=0.8))
    d = argdict(rec.of("/n_set")[0])
    assert d["t_bloom"] == 1 and abs(d["bloomAmt"] - 0.8) < 1e-9 and m.blooms == 1
    m.toggle_bypass()
    assert rec.of("/n_set")[-1] == [900, "bypass", 1]
    m.close()
    assert rec.of("/n_free")[0] == [900] and not m.spawned
    rec.clear(); assert m.update(SS()) is None and rec.msgs == []


def test_triads():
    assert parse_triad("Eb") == (3, True) and parse_triad("F#m") == (6, False)
    assert parse_triad("") is None and parse_triad("H") is None
    f = triad_freqs("C", 48)
    assert len(f) == 3 and abs(f[0] - 130.81) < 0.05 and abs(f[1] / f[0] - 2 ** (4 / 12)) < 1e-6
    fm = triad_freqs("Am", 48)
    assert abs(fm[1] / fm[0] - 2 ** (3 / 12)) < 1e-6
    # root lands within half an octave of the register
    for name in ("C", "F#", "B", "Gm", "Ebm"):
        root_midi = 69 + 12 * math.log2(triad_freqs(name, 48)[0] / 440.0)
        assert 43 - 1e-6 <= root_midi <= 54 + 1e-6, name


def test_presound_layer():
    rec = Recorder()
    pre = PreSoundLayer(send=rec, gain=0.1)
    pre.spawn()
    assert [a[1] for a in rec.of("/s_new")] == [901, 902, 903]
    assert all(argdict(a)["amp"] == 0.0 for a in rec.of("/s_new"))
    rec.clear()
    # no prediction -> silent
    assert pre.update(SS(predicted_triad="", harmony_posterior_other=0.9)) == 0.0
    # prediction == current triad -> silent
    assert pre.update(SS(triad="C", predicted_triad="C", harmony_posterior_other=0.9)) == 0.0
    # a real prediction with posterior mass -> pitched + audible
    rec.clear()
    amp = pre.update(SS(triad="C", predicted_triad="Em", harmony_posterior_other=0.4))
    assert abs(amp - 0.04) < 1e-9
    freqs = [argdict(a)["freq"] for a in rec.of("/n_set") if "freq" in argdict(a)]
    assert len(freqs) == 3 and abs(freqs[1] / freqs[0] - 2 ** (3 / 12)) < 1e-6
    amps = [argdict(a)["amp"] for a in rec.of("/n_set") if "amp" in argdict(a)]
    assert len(amps) == 3 and all(abs(a - 0.04) < 1e-9 for a in amps)
    rec.clear()
    pre.update(SS(triad="C", predicted_triad="Em", harmony_posterior_other=0.4))
    assert rec.msgs == [], "unchanged -> nothing sent"
    # score_gain scales it, and the walker stepping there silences it
    assert abs(pre.update(SS(triad="C", predicted_triad="Em", harmony_posterior_other=0.4), 0.5) - 0.02) < 1e-9
    assert pre.update(SS(triad="Em", predicted_triad="Em", harmony_posterior_other=0.4)) == 0.0
    pre.close()
    assert all(a[1:] == ["gate", 0] for a in rec.of("/n_set")[-3:])


def test_install_rig_end_to_end():
    rec = Recorder(); v = Voices(rec); g = Gov(); p = Profile()
    class M: synthdef = "melodyHit"
    rig = install(v, p, governor=g, melody=M(), palette="rich", send=rec)
    assert v.mpe_synthdef == "mpeVoiceRich" and isinstance(g.scheduler, RichPatternScheduler)
    assert rig.master.spawned and rig.presound.spawned
    assert [a[1] for a in rec.of("/s_new")] == [900, 901, 902, 903]
    v.voices_by_slot = [Voice(0, blob_id=1, sc_node=1000)]
    rec.clear()
    rig.update(SS(tension=0.3, predicted_triad="G", harmony_posterior_other=0.2),
               [Blob(1, cy=50.0)])
    nodes = [a[0] for a in rec.of("/n_set")]
    assert 1000 in nodes and 900 in nodes and 901 in nodes
    assert "SYNTH rich" in rig.hud_line() and "pre=G" in rig.hud_line()
    rig.close()
    assert rec.of("/n_free") == [[900]]
    # classic: no extras, names restored, master/presound still available
    rec.clear(); v2 = Voices(rec); g2 = Gov()
    rig2 = install(v2, p, governor=g2, palette="classic", send=rec, master=False)
    assert v2.mpe_synthdef == "mpeVoice" and rig2.extras is None and rig2.master is None
    assert g2.scheduler.synthdef == "mpeHit" and rig2.presound.synthdef == "preSound"
    rig2.update(None, []); rig2.close()


def test_live_palette_switch():
    rec = Recorder(); v = Voices(rec); g = Gov(); p = Profile()
    class M(sp.RichMelodyEngine):
        pass
    melody = sp.RichMelodyEngine(v, p, send=rec, clock=lambda: 0.0)
    rig = install(v, p, governor=g, melody=melody, palette="rich", send=rec, master=False, presound=False)
    assert rig.palette.rich and rig.extras is not None
    rig.toggle_palette()
    assert rig.palette.name == "classic" and rig.extras is None
    assert v.mpe_synthdef == "mpeVoice" and g.scheduler.synthdef == "mpeHit" and melody.synthdef == "melodyHit"
    assert not isinstance(g.scheduler, RichPatternScheduler)
    melody._note_extra = ["model", 2]; rec.clear()
    melody._send("/s_new", ["melodyHit", -1, 0, 0, "freq", 220.0])
    assert rec.msgs[-1][1] == ["melodyHit", -1, 0, 0, "freq", 220.0], "no model extras in classic"
    rig.toggle_palette()
    assert rig.palette.rich and rig.extras is not None and v.mpe_synthdef == "mpeVoiceRich"
    assert isinstance(g.scheduler, RichPatternScheduler) and melody.synthdef == "melodyHitRich"
    rec.clear(); melody._send("/s_new", ["melodyHitRich", -1, 0, 0])
    assert rec.msgs[-1][1][-2:] == ["model", 2]
    assert rig.set_palette("rich") is rig.palette, "same palette is a no-op"


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    passed = failed = 0
    for t in TESTS:
        try:
            t(); passed += 1; print(f"  ok   {t.__name__}")
        except Exception as exc:
            failed += 1; print(f"  FAIL {t.__name__}: {exc!r}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
