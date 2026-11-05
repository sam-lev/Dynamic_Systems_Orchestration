#!/usr/bin/env python3
# test_rave_bench.py -- Audible Lava / DSO v4.11
# Tests for rave_bench.py's analysis and plumbing. No SuperCollider: WAV
# analysis runs on synthetic numpy signals; the scsynth driver is exercised
# against a recorder. Run:  python test_rave_bench.py

from __future__ import annotations
import math, os, sys, tempfile
import numpy as np
import rave_bench as rb
import rave_layer as rl

SR = 48000


def tone(secs, freq=220.0, amp=0.3, sr=SR):
    t = np.arange(int(secs * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def bursts(secs, period=1.0, width=0.06, sr=SR, seed=0):
    """Same aperiodic schedule as \\benchBurst in rave_bench.scd."""
    rng = np.random.default_rng(seed)
    x = np.zeros(int(secs * sr), np.float32)
    n = int(width * sr)
    pat = [0.7, 1.3, 0.9, 1.1, 0.8, 1.2]
    t, k = 0.0, 0
    while t < secs:
        i = int(t * sr)
        x[i:i + n] = rng.uniform(-0.5, 0.5, n)
        t += pat[k % len(pat)] * period; k += 1
    return x


class Rec:
    def __init__(self): self.msgs = []
    def send_message(self, a, args): self.msgs.append((a, list(args)))
    def send(self, b): self.msgs.append(("<bundle>", b))
    def of(self, a): return [m for ad, m in self.msgs if ad == a]


def make_sc(master=True):
    sc = rb.SC.__new__(rb.SC)
    sc.osc = Rec(); sc.ip, sc.port = "x", 0
    sc.master_node = 900 if master else None
    sc._stim = []; sc._next_stim = rb.STIM_BASE; sc.sent = 0
    return sc


# ── analysis ─────────────────────────────────────────────────────────────────

def test_measure_tone_and_silence():
    m = rb.measure(tone(2.0, amp=0.3))
    assert abs(m.rms_db - rb.db(0.3 / math.sqrt(2))) < 0.2 and abs(m.peak_db - rb.db(0.3)) < 0.2
    assert m.dropouts == 0 and m.clicks == 0 and m.silent_frac == 0.0
    assert 200 < m.centroid_hz < 240
    s = rb.measure(np.zeros(SR, np.float32))
    assert s.rms_db <= -99 and s.silent_frac == 1.0 and s.onset_s is None


def test_blowup_and_nan_are_reported_not_hidden():
    x = tone(2.0)
    x[SR // 2] = np.nan; x[SR] = 40.0
    m = rb.measure(x)
    assert m.nonfinite == 1 and m.over_fs == 1
    assert np.isfinite(m.rms_db) and np.isfinite(m.centroid_hz), "one NaN must not poison the means"
    y = tone(1.0); y[100] = 1e28
    assert np.isfinite(rb.measure(y).rms_db), "huge finite values must not overflow"
    z = tone(3.0); z[int(2.0 * SR):] = 0
    assert rb.measure(z, end_s=2.0).silent_frac < 0.05, "end_s cuts the release tail"


def test_dropout_detection_counts_gaps_inside_sound_only():
    x = tone(4.0)
    x[int(1.0 * SR):int(1.02 * SR)] = 0            # 20 ms gap
    x[int(2.5 * SR):int(2.6 * SR)] = 0             # 100 ms gap
    d = rb.detect_dropouts(x)
    assert d.count == 2 and 15 <= d.total_ms <= 130 and 90 <= d.longest_ms <= 110
    assert abs(d.starts_s[0] - 1.0) < 0.02 and abs(d.starts_s[1] - 2.5) < 0.02
    # silence at the ends is not a dropout; skip_s ignores the head
    y = np.concatenate([np.zeros(SR, np.float32), tone(2.0), np.zeros(SR, np.float32)])
    assert rb.detect_dropouts(y).count == 0
    assert rb.detect_dropouts(x, skip_s=2.0).count == 1


def test_click_counting():
    x = tone(1.0)
    assert rb.count_clicks(x) == 0
    x[SR // 2] = 0.9; x[SR // 2 + 1] = -0.9
    assert rb.count_clicks(x) >= 2
    assert rb.count_clicks(x, skip_s=0.8) == 0


def test_latency_estimate():
    dry = bursts(8.0)
    delay = int(0.137 * SR)
    wet = np.concatenate([np.zeros(delay, np.float32), dry[:-delay]]) * 0.6
    lat, conf = rb.estimate_latency_ms(dry, wet)
    assert lat is not None and abs(lat - 137.0) < 3.0, lat
    assert conf > 0.1
    assert rb.estimate_latency_ms(dry, np.zeros_like(dry)) == (None, 0.0)


def test_wav_roundtrip_and_measure():
    try:
        import soundfile as sf
    except ImportError:
        return
    d = tempfile.mkdtemp(); p = os.path.join(d, "t.wav")
    x = np.stack([tone(1.0), tone(1.0, 330)], axis=1)
    sf.write(p, x, SR, subtype="FLOAT")
    y, sr = rb.read_wav(p)
    assert sr == SR and y.shape == x.shape and np.allclose(x, y, atol=1e-6)


# ── scenarios and OSC plumbing ───────────────────────────────────────────────

def test_build_scenarios_cover_every_model_and_buffer():
    S = rb.build_scenarios(["organ", "birds"], [2048, 8192], quick=False)
    names = {s.name for s in S}
    for k in ("organ", "birds"):
        for suffix in ("tone", "burst", "lamp", "high", "silence", "gain0.25", "gain4", "fadeoff"):
            assert f"{k}_b2048_{suffix}" in names, suffix
        assert f"{k}_b8192_tone" in names and f"{k}_b8192_burst" in names
    assert "stack2_b8192_lamp" in names and "stack3_b8192_lamp" not in names
    assert {"split_passthrough", "tone_reference", "dry_scaling", "burst_reference", "lamp_reference"} <= names
    q = rb.build_scenarios(["organ"], [4096], quick=True)
    assert all(s.seconds <= 8.0 for s in q) and not any("gain" in s.name for s in q)
    assert all(k in {m.key for m in rl.MODELS} for s in S for k in s.models)


def test_run_scenario_places_nodes_like_the_live_layer(monkeypatch=None):
    rb.time.sleep = lambda s: None                           # no waiting
    rb.SC.process_cpu = staticmethod(lambda sample_s=1.0: 42.0)
    rb.SC.status = lambda self: {"avg_cpu": 3.0, "peak_cpu": 5.0, "synths": 7}
    sc = make_sc(master=True)
    d = tempfile.mkdtemp()
    s = rb.Scenario("organ_b4096_tone", "tone", ["organ"], buf=4096, mix=1.0, in_gain=2.0, seconds=1.0)
    r = rb.run_scenario(sc, s, d, record=True)
    snew = sc.osc.of("/s_new")
    split = next(a for a in snew if a[0] == "raveSplit")
    assert split[:4] == ["raveSplit", rl.SPLIT_NODE, 2, 900], "addBefore the master, as RaveLayer"
    assert split[split.index("dry") + 1] == 0.0, "mix 1.0 -> dry 0"
    mdl = next(a for a in snew if a[0].startswith("raveBench_"))
    assert mdl[:4] == ["raveBench_organ_4096", rb.model_node(0), 2, 900]
    assert mdl[mdl.index("inGain") + 1] == 2.0 and mdl[mdl.index("inBus") + 1] == rl.RAVE_BUS
    assert mdl[mdl.index("pan") + 1] == rl.model_pan(7)                 # organ is index 7
    stim = next(a for a in snew if a[0] == "benchTone")
    assert stim[1:4] == [rb.STIM_BASE, 0, 0], "stimuli at the head of group 0"
    rec = next(a for a in snew if a[0] == "benchRec")
    assert rec[1:4] == [rb.REC_NODE, 1, 0], "recorder at the tail"
    assert any(a[0] == rb.REC_BUF and a[1].endswith("organ_b4096_tone.wav") for a in sc.osc.of("/b_write"))
    assert r["cpu_process_pct"] == 42.0 and r["cpu_dsp_avg"] == 3.0
    gates = [a for a in sc.osc.of("/n_set") if a[1:] == ["gate", 0]]
    assert {a[0] for a in gates} >= {rb.model_node(0), rb.STIM_BASE}, "model and stimulus gated off"
    assert [rl.SPLIT_NODE] in sc.osc.of("/n_free")
    # safety-net free goes out as one /error -1 bundle (no "Node not found" spam)
    assert any(ad == "<bundle>" for ad, _ in sc.osc.msgs), "stimulus safety free sent as a bundle"
    # no master -> tail of group 0; stacking -> amp = mix/sqrt(n)
    sc2 = make_sc(master=False)
    s2 = rb.Scenario("stack2", "tone", ["organ", "water"], buf=8192, mix=0.7, seconds=1.0)
    r2 = rb.run_scenario(sc2, s2, d, record=False)
    snew = sc2.osc.of("/s_new")
    assert next(a for a in snew if a[0] == "raveSplit")[2:4] == [1, 0]
    amps = [a[a.index("amp") + 1] for a in snew if a[0].startswith("raveBench_")]
    assert len(amps) == 2 and all(abs(x - 0.7 / math.sqrt(2)) < 1e-3 for x in amps)
    assert abs(r2["dry"] - 0.3) < 1e-9
    # fade-off scenario gates the model before the stimulus ends
    sc3 = make_sc()
    s3 = rb.Scenario("fade", "tone", ["organ"], seconds=2.0, fade_off_at=1.0)
    rb.run_scenario(sc3, s3, d, record=False)
    order = [(ad, m) for ad, m in sc3.osc.msgs if ad == "/n_set" and m[1:] == ["gate", 0]]
    assert order[0][1][0] == rb.model_node(0), "model gated first (fade-off), stimulus after"


def test_lamp_stimulus_uses_real_synthdefs_with_timed_bundles():
    sc = make_sc()
    rb.stim_start(sc, "lamp", 8.0)
    snew = sc.osc.of("/s_new")
    assert sum(1 for a in snew if a[0] == "mpeVoiceRich") == 4
    assert any(a[0] == "bassStutterRich" for a in snew)
    bundles = [m for ad, m in sc.osc.msgs if ad == "<bundle>"]
    assert len(bundles) > 80, "drift updates + hits + notes + strike go out as timed bundles"
    # each part alone spawns only its SynthDef
    for part, sdef in (("drones", "mpeVoiceRich"), ("bass", "bassStutterRich")):
        sc = make_sc(); rb.stim_start(sc, f"lamp:{part}", 8.0)
        assert {a[0] for a in sc.osc.of("/s_new")} == {sdef}, part
    sc = make_sc(); rb.stim_start(sc, "lamp:strike", 8.0)
    assert sc.osc.of("/s_new") == [] and len([m for ad, m in sc.osc.msgs if ad == "<bundle>"]) == 1
    try:
        rb.stim_start(make_sc(), "lamp:nope", 8.0); assert False
    except ValueError:
        pass
    assert {s.name for s in rb.build_scenarios(["organ"], [8192], True)} >= {f"iso_{p}" for p in rb.LAMP_PARTS}
    for kind in ("burst", "tone", "high", "noise", "silence"):
        sc = make_sc(); rb.stim_start(sc, kind, 2.0)
    try:
        rb.stim_start(make_sc(), "nope", 1.0); assert False
    except ValueError:
        pass


def test_report_verdicts():
    d = tempfile.mkdtemp()
    try:
        import soundfile as sf
    except ImportError:
        return
    ref = np.stack([tone(7.0, amp=0.3)] * 2, axis=1)
    sf.write(os.path.join(d, "tone_reference.wav"), ref, SR, subtype="FLOAT")
    sf.write(os.path.join(d, "split_passthrough.wav"), ref, SR, subtype="FLOAT")
    sf.write(os.path.join(d, "dry_scaling.wav"), ref * 0.3, SR, subtype="FLOAT")
    bad = ref.copy(); bad[int(4.0 * SR):int(4.05 * SR)] = 0
    sf.write(os.path.join(d, "organ_b2048_tone.wav"), bad, SR, subtype="FLOAT")
    sf.write(os.path.join(d, "organ_b2048_silence.wav"), ref * 0.05, SR, subtype="FLOAT")
    S = [rb.Scenario("tone_reference", "tone", [], layer_on=False, dry=1.0, group="reference"),
         rb.Scenario("split_passthrough", "tone", [], layer_on=False, group="functional"),
         rb.Scenario("dry_scaling", "tone", [], layer_on=False, dry=0.3, group="functional"),
         rb.Scenario("organ_b2048_tone", "tone", ["organ"], group="model_on"),
         rb.Scenario("organ_b2048_silence", "silence", ["organ"], group="silence_in"),
         rb.Scenario("water_b2048_tone", "tone", ["water"], group="model_on")]
    sf.write(os.path.join(d, "burst_reference.wav"), np.stack([bursts(7.0)] * 2, axis=1), SR, subtype="FLOAT")
    sf.write(os.path.join(d, "organ_b2048_burst.wav"), np.stack([bursts(7.0) * 0.5] * 2, axis=1), SR, subtype="FLOAT")
    blown = ref.copy(); blown[SR] = 50.0
    sf.write(os.path.join(d, "lamp_reference.wav"), blown, SR, subtype="FLOAT")
    S = [rb.Scenario("burst_reference", "burst", [], layer_on=False, group="reference")] + S + [
         rb.Scenario("organ_b2048_burst", "burst", ["organ"], group="latency"),
         rb.Scenario("lamp_reference", "lamp", [], layer_on=False, group="reference")]
    res = rb.analyse_all(S, {}, d)
    assert res["organ_b2048_burst"]["verdict"].startswith("wet ok"), res["organ_b2048_burst"]["verdict"]
    assert "BLOWUP" in res["lamp_reference"]["verdict"]
    assert res["split_passthrough"]["verdict"] == "PASS"
    assert res["dry_scaling"]["verdict"] == "PASS"
    assert "DROPOUTS 1" in res["organ_b2048_tone"]["verdict"]
    assert "GAPPY" not in res["organ_b2048_tone"]["verdict"], "a single 50 ms gap is a dropout, not gappy"
    assert "HALLUCINATES" in res["organ_b2048_silence"]["verdict"]
    assert res["water_b2048_tone"].get("error") == "wav missing"
    mp = rb.write_report(res, d, {"models": ["organ"]})
    txt = open(mp).read()
    assert "| organ |" in txt and "DROPOUTS" in txt and "Reading the numbers" in txt
    assert os.path.isfile(os.path.join(d, "rave_bench.json"))


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    passed = failed = 0
    for t in TESTS:
        try:
            t(); passed += 1; print(f"  ok   {t.__name__}")
        except Exception as exc:
            failed += 1; import traceback; traceback.print_exc(); print(f"  FAIL {t.__name__}: {exc!r}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
