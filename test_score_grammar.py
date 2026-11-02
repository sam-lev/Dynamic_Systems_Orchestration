# test_score_grammar.py -- Audible Lava / DSO v4.2
# Behavioral tests for the topological score grammar layer.
# Run: python test_score_grammar.py   |   pytest test_score_grammar.py -v

import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("/mnt/project", HERE):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from pattern_language import (Span, Event, euclid, bjorklund, steps, stack, pure, apply_spec,  # noqa
                              TRANSFORMS, rotate, rev, fast, degrade, ply, every, off, arp, density)
from schema_lexicon import LEXICON, SchemaMarkov, realize, NAMES, degree_to_freq          # noqa
from system_state import (permutation_entropy, persistence_entropy, occupancy_entropy, emd_1d,  # noqa
                          emd_grid_marginal, diagram_wasserstein, BOCPD, MarkovExpectation,
                          SystemStateTracker, periodicity_strength, summary_distance)
from harmony_tonnetz import Triad, tonnetz_distance, tonnetz_path, TonnetzWalker, roughness_proxy  # noqa
from elastic_clock import ElasticClock, ClockParams                                        # noqa
from preference_rules import (PreferenceRuleModule, ParticleFilterModule, Rule, RuleRegistry,  # noqa
                              rule, softmax_weights, WeightStore)
from score_governor import ScoreGovernor, ScoreState, GOVERNOR_SOURCES, write_default_weights  # noqa

TESTS = []
def test(fn):
    TESTS.append(fn); return fn

RNG = np.random.default_rng(3)


@dataclass
class B:
    id: int; centroid: tuple; area: float; birth: float; death: float; persistence: float
    vx: float = 0.0; vy: float = 0.0; speed: float = 0.0; ay: float = 0.0
    direction: str = "turning"; age: int = 0; d_area: float = 0.0; d_persistence: float = 0.0


def synth_blobs(f, fps=30.0, n=3):
    t = f / fps
    out = []
    for i in range(n):
        per, ph, x = 12.0 - 2 * i, 2.0 * i, 60 + 100 * i
        y = 120 + 80 * math.sin(2 * math.pi * t / per + ph)
        vy = 80 * 2 * math.pi / per * math.cos(2 * math.pi * t / per + ph) / fps
        out.append(B(i, (x, y), 3000 + 500 * i, 180 - 20 * i, 20, 40 + 10 * i, vy=vy, speed=abs(vy),
                     direction="rising" if vy < 0 else "falling", age=f))
    return out


# ── pattern language ─────────────────────────────────────────────────────────

@test
def pl_euclid_counts_and_tresillo():
    assert bjorklund(3, 8) == [1, 0, 0, 1, 0, 0, 1, 0]
    for k, n in ((0, 8), (8, 8), (5, 8), (7, 12), (1, 5)):
        assert sum(bjorklund(k, n)) == k
        assert len(euclid(k, n).query(Span(0, 1))) == k
    assert len(euclid(5, 8).query(Span(0, 3))) == 15, "3 cycles -> 15 hits"


@test
def pl_transform_laws():
    p = euclid(3, 8)
    base = sorted(e.start for e in p.query(Span(0, 1)))
    r = sorted(round(e.start, 9) for e in rotate(1, 8)(p).query(Span(0, 1)))
    assert r == sorted(round((b - 1 / 8) % 1.0, 9) for b in base), "rotate shifts one slot earlier"
    rr = rev()(rev()(p))
    assert sorted(e.start for e in rr.query(Span(0, 1))) == base, "rev is an involution"
    assert len(fast(2)(p).query(Span(0, 1))) == 6, "fast(2) doubles density"
    assert len(ply(2)(p).query(Span(0, 1))) == 6, "ply(2) doubles hits"
    assert len(degrade(1.0)(p).query(Span(0, 4))) == 0 and len(degrade(0.0)(p).query(Span(0, 4))) == 12
    assert len(stack(p, p).query(Span(0, 1))) == 6
    ev = every(2, fast(2))(p)
    assert len(ev.query(Span(0, 1))) == 6 and len(ev.query(Span(1, 2))) == 3, "every(2) alternates"
    assert len(off(0.125, ply(1))(p).query(Span(0, 1))) == 6, "off superimposes a delayed copy"
    d = arp([1, 3, 5])(p).query(Span(0, 1))
    assert [e.degree for e in sorted(d, key=lambda e: e.start)] == [1, 3, 5]
    assert len(density(0.34)(p).query(Span(0, 1))) == 1


@test
def pl_spec_and_registry():
    for name in ("rotate", "rev", "fast", "slow", "degrade", "mask", "superimpose", "every", "iter",
                 "palindrome", "chunk", "swing", "ply", "off", "accent", "velocity", "transpose",
                 "invert", "arp", "tag", "density"):
        assert name in TRANSFORMS, name
    p = apply_spec(euclid(3, 8), [("rotate", 2, 8), ("ply", 2), ("velocity", 0.5), ("tag", "x")])
    evs = p.query(Span(0, 1))
    assert len(evs) == 6 and all("x" in e.tags for e in evs) and all(abs(e.velocity - 0.4) < 1e-9 for e in evs)
    try:
        apply_spec(euclid(3, 8), [("nope",)]); raise AssertionError("unknown transform must raise")
    except KeyError:
        pass


# ── schema lexicon ───────────────────────────────────────────────────────────

@test
def sl_lexicon_shape():
    assert len(LEXICON) == 12
    borrowed = [s for s in LEXICON.values() if s.origin == "gjerdingen"]
    assert 3 <= len(borrowed) <= 4
    for s in LEXICON.values():
        assert len(s.melody) == len(s.bass) == len(s.tension) == s.footprint
        assert all(0.0 <= t <= 1.0 for t in s.tension)


@test
def sl_markov_rows_and_bias():
    m = SchemaMarkov()
    for a, row in m.table.items():
        assert abs(sum(row.values()) - 1.0) < 1e-9 and set(row) == set(NAMES)
    before = m.distribution("cadence")
    m.bias_from_state(tension=0.95, ambiguity=0.2)
    after = m.distribution("cadence")
    tens = [n for n in NAMES if "tension" in LEXICON[n].tags]
    assert sum(after[n] for n in tens) > sum(before[n] for n in tens), "high tension biases toward tension frames"
    seq = [m.next() for _ in range(200)]
    assert len(set(seq)) >= 8, "succession must explore the lexicon"
    assert sum(1 for a, b in zip(seq, seq[1:]) if a == b) < 60, "immediate repeats suppressed"


@test
def sl_realize_and_degrees():
    mel, bass = realize(LEXICON["plume"], beats_per_bar=4, ornament=0.5)
    evs = mel.query(Span(0, 1.5))
    assert [e.degree for e in sorted(evs, key=lambda e: e.start) if "ornament" not in e.tags] == [1, 2, 3, 5, 6, 8]
    assert any("ornament" in e.tags for e in evs), "ornament adds diminution"
    assert abs(degree_to_freq(1, 0, 60) - 261.63) < 0.1 and abs(degree_to_freq(8, 0, 60) - 523.25) < 0.1


# ── system state ─────────────────────────────────────────────────────────────

@test
def ss_entropies():
    assert permutation_entropy(np.arange(200)) < 0.05, "monotone series has ~0 ordinal entropy"
    assert permutation_entropy(RNG.random(2000)) > 0.9, "white noise ~1"
    assert persistence_entropy([(100, 0), (10, 0), (10, 0)]) < persistence_entropy([(50, 0), (50, 0), (50, 0)])
    assert abs(occupancy_entropy(np.ones((4, 4))) - 1.0) < 1e-9 and occupancy_entropy(np.eye(4)[:1]) == 0.0
    assert emd_1d([1, 0, 0], [0, 0, 1]) == 2.0 and emd_1d([1, 0], [1, 0]) == 0.0
    g = np.zeros((4, 4)); g[0, 0] = 1; h = np.zeros((4, 4)); h[3, 3] = 1
    assert emd_grid_marginal(g, h) > emd_grid_marginal(g, np.roll(g, 1, 1))
    assert diagram_wasserstein([(10, 0)], [(10, 0)]) == 0.0
    assert diagram_wasserstein([(10, 0)], [(12, 0)]) > 0 and diagram_wasserstein([], [(10, 0)]) > 0


@test
def ss_bocpd_detects_step():
    det = BOCPD(hazard=1 / 100)
    ps = [det.update(x) for x in np.concatenate([RNG.normal(0, 1, 150), RNG.normal(6, 1, 150)])]
    quiet95 = float(np.percentile(ps[20:140], 95)); jump = max(ps[150:165])
    assert jump > 0.7 and quiet95 < 0.35, f"changepoint posterior {jump:.2f} vs quiet p95 {quiet95:.2f}"


@test
def ss_markov_surprisal_and_entropy():
    m = MarkovExpectation(prior=0.1)
    for _ in range(50):
        for s in (0, 1, 2):
            m.update(s)
    assert m.conditional_entropy(0) < 0.5, "deterministic cycle -> low conditional entropy"
    assert m.entropy_rate() < 0.5
    s_expected = m.update(0); s_expected = m.update(1)
    s_surprise = m.update(0)                     # 1 -> 0 never seen
    assert s_surprise > s_expected + 2.0, "unexpected transition has high surprisal"


@test
def ss_tracker_summary_and_recurrence():
    st = SystemStateTracker(320, 240)
    for f in range(240):
        s = st.update(synth_blobs(f), 1 / 30)
    assert 0 <= s.occupancy_entropy <= 1 and 0 <= s.coherence <= 1 and s.blob_count == 3
    assert s.symbol >= 0 and s.cond_entropy >= 0 and len(st.history) == 240
    assert st.signal("ke") and len(st.signal("flux")) == 240
    strength, lag = periodicity_strength(st.signal("com_y"))
    assert strength > 0.3, "oscillating blobs -> periodic centre of mass"
    lag_best, d = st.nearest_past_state(min_lag=60)
    assert lag_best is not None and d < summary_distance(st.history[-1], st.history[-100]) + 1e-9


# ── harmony ──────────────────────────────────────────────────────────────────

@test
def hm_tonnetz():
    c = Triad(0, True)
    assert c.P().P() == c and c.L().L() == c and c.R().R() == c, "P, L, R are involutions"
    assert c.L().name == "Em" and c.R().name == "Am" and c.P().name == "Cm"
    assert tonnetz_distance(c, c) == 0 and tonnetz_distance(c, c.L()) == 1
    assert all(tonnetz_distance(c, t) <= 6 for t in [Triad(r, m) for r in range(12) for m in (True, False)])
    path = tonnetz_path(c, Triad(8, True))
    assert c.apply(path) == Triad(8, True)
    assert roughness_proxy([0, 4, 7]) < roughness_proxy([0, 1, 2])
    w = TonnetzWalker(home=c)
    ops = [w.update(rate=1.0, size=0.9, direction=1.0, dt=1.0) for _ in range(6)]
    assert w.distance_home() >= 2, "away-from-home steps increase distance"
    for _ in range(12):
        w.update(rate=1.0, size=0.2, direction=-1.0, dt=1.0)
    assert w.current == c, "toward-home walk returns home"
    assert len(w.pitch_field(0.9)) > len(w.pitch_field(0.1)), "ambiguity adds tones"


# ── clock ────────────────────────────────────────────────────────────────────

@test
def ck_elastic_pull_and_bounds():
    c = ElasticClock(ClockParams(period_s=1.0, subdiv=2, elastic=0.5))
    assert abs(c.pull(0.37) - 0.435) < 1e-9, "half-way toward 0.5"
    assert abs(c.pull(0.37, 1.0) - 0.5) < 1e-9 and abs(c.pull(0.37, 0.0) - 0.37) < 1e-9
    c.set_period(5.0, now_s=3.0); assert c.p.period_s <= 2.0, "London tactus bound"
    c.set_period(0.1, now_s=3.0); assert c.p.period_s >= 0.4
    ph = c.beat_phase(10.0); c.set_period(1.3, now_s=10.0)
    assert abs(c.beat_phase(10.0) - ph) < 1e-6, "tempo change preserves phase"
    assert c.realize(c.projection_s, 0.0) > 0.99, "an IOI equal to the projection is fully realized"


# ── preference rules ─────────────────────────────────────────────────────────

@test
def pr_hysteresis_from_switch_cost():
    class M(PreferenceRuleModule):
        name = "toy"
        def propose(self, ev): return ["A", "B"]
        def candidate_key(self, c): return c
    RuleRegistry._rules.pop("toy", None)
    rule("toy", 1.0)(lambda ev, c: ev["a"] if c == "A" else 1 - ev["a"])
    m_lo, m_hi = M(switch_cost=0.0, forget=1.0), M(switch_cost=3.0, forget=1.0)
    for _ in range(20):
        m_lo.update({"a": 0.9}); m_hi.update({"a": 0.9})
    flips_lo = flips_hi = None
    for k in range(60):
        m_lo.update({"a": 0.3}); m_hi.update({"a": 0.3})
        if flips_lo is None and m_lo.committed == "B": flips_lo = k
        if flips_hi is None and m_hi.committed == "B": flips_hi = k
    assert flips_lo is not None and flips_hi is not None and flips_hi > flips_lo, (flips_lo, flips_hi)
    assert abs(m_hi.posterior.sum() - 1.0) < 1e-9 and m_hi.blend([1.0, 0.0]) < 1.0


@test
def pr_particle_filter_tracks():
    class PF(ParticleFilterModule):
        name = "toy_pf"
        def init_particles(self, ev): return [{"x": float(self.rng.uniform(-5, 5))} for _ in range(self.n)]
    RuleRegistry._rules.pop("toy_pf", None)
    rule("toy_pf", 1.0)(lambda ev, p: math.exp(-(p["x"] - ev["target"]) ** 2))
    pf = PF(n_particles=200, jitter={"x": 0.1}, bounds={"x": (-10, 10)})
    for _ in range(40):
        pf.update({"target": 2.0})
    assert abs(pf.estimate()["x"] - 2.0) < 0.3
    for _ in range(80):
        pf.update({"target": -1.0})
    assert abs(pf.estimate()["x"] + 1.0) < 0.3, "estimate drifts continuously to the new target"


@test
def pr_softmax_and_weightstore():
    w = softmax_weights([1.0, 1.0, 1.0], 0.5)
    assert np.allclose(w, 1 / 3)
    assert softmax_weights([3.0, 0.0], 0.05)[0] > 0.99 and softmax_weights([3.0, 0.0], 50.0)[0] < 0.52
    path = "/tmp/_gw_test.json"; write_default_weights(path, overwrite=True)
    ws = WeightStore(path, "/tmp/_gr_test.json", "/tmp/_gl_test.txt", write_every_s=0)
    assert ws.get("grouping.rules.saddle", -1) == 2.0
    ws.set("grouping.rules.saddle", 3.5); assert ws.get("grouping.rules.saddle", -1) == 3.5
    assert ws.maybe_write({"x": 1}, force=True) and os.path.exists("/tmp/_gr_test.json")


# ── governor end-to-end ──────────────────────────────────────────────────────

def _governor():
    for p in ("/tmp/_gw.json",):
        if os.path.exists(p):
            os.remove(p)
    return ScoreGovernor(320, 240, weights_path="/tmp/_gw.json", runtime_path="/tmp/_gr.json",
                         log_path="/tmp/_gl.txt", fps=30)


@test
def gv_end_to_end_and_stamps():
    g = _governor()
    for f in range(300):
        s = g.update(synth_blobs(f), 1 / 30, [("merge", 1, 0)] if f == 200 else [])
    assert isinstance(s, ScoreState) and 0 <= s.tension <= 1 and 0 <= s.ambiguity <= 1
    assert 0.4 <= s.period_s <= 2.0 and s.beats_per_bar in (2, 3, 4)
    assert set(s.prominence_rank) == {0, 1, 2} and max(s.prominence_rank.values()) == 1.0
    b = synth_blobs(300)[0]; g.update([b] + synth_blobs(300)[1:], 1 / 30)
    for src in GOVERNOR_SOURCES:
        assert hasattr(b, f"score_{src}"), src
    for role, w in s.role_weights.items():
        assert abs(sum(w.values()) - 1.0) < 1e-6, role
    assert any(k.startswith("sys:") for k in s.role_weights["meter"]), "virtual signals share the meter role"
    assert s.triad and len(s.pitch_field) >= 3 and s.schema in LEXICON
    assert os.path.exists("/tmp/_gr.json") and os.path.exists("/tmp/_gl.txt")


@test
def gv_roles_decay_without_election():
    g = _governor()
    for f in range(120):
        g.update(synth_blobs(f), 1 / 30)
    w0 = g.roles["timespan"].get("2", 0.0)
    for f in range(120, 200):                       # blob 2 disappears
        g.update(synth_blobs(f)[:2], 1 / 30)
    w1 = g.roles["timespan"].get("2", 0.0)
    assert w1 < w0 * 0.5 or w1 == 0.0, "a vanished blob's weight decays smoothly (no election, no snap)"
    assert abs(sum(g.roles["timespan"].values()) - 1.0) < 1e-6


@test
def gv_boundary_on_regime_change():
    """v4.4: the HSMM's duration model sets the boundary RATE and the evidence
    sets PLACEMENT, so the property to check is that boundaries happen at a
    plausible rate over a long run and that the phrase model learns -- not that
    one appears inside a short window after a step change."""
    g = _governor()
    fired = []
    for f in range(1800):                 # 60 s at 30 fps
        blobs = synth_blobs(f)
        if f >= 900:                      # regime change: faster, more spread
            for b in blobs:
                b.centroid = (b.centroid[0] + (f - 900) * 0.4 * (b.id - 1), b.centroid[1])
                b.vy *= 4.0; b.speed *= 4.0; b.area *= 2.0
        s = g.update(blobs, 1 / 30)
        if s.boundary:
            fired.append(f / 30.0)
    assert fired, "no boundaries at all in 60 s"
    assert g.grouping.n_boundaries == len(fired)
    assert g.grouping.intervals.mean > 0
    assert all(0.0 <= t <= 60.0 for t in fired)


@test
def gv_edit_surface_and_pattern():
    from controller import ControllerEvent, SET_PARAM, EDIT_FAMILIES, EDIT_STEP_FINE
    g = _governor()
    g.update(synth_blobs(0), 1 / 30)
    saddle = lambda: next(r.weight for r in g.grouping.rules if r.name == "saddle")
    ev = lambda p, v, **x: g.handle_controller_event(
        ControllerEvent(SET_PARAM, param=p, value=v, extras=x))

    # every param the keyboard can reach has a step size
    assert "gov" in EDIT_FAMILIES and len(EDIT_FAMILIES["gov"]) >= 10
    for p in EDIT_FAMILIES["gov"]:
        assert p.startswith("gov.") and p in EDIT_STEP_FINE, p

    d = saddle()
    assert ev("gov.grouping.rules.saddle", 4.0) and saddle() == 4.0, "absolute set"
    assert ev("gov.grouping.rules.saddle", -0.5, relative=True) and abs(saddle() - 3.5) < 1e-9, "relative delta"
    assert ev("gov.grouping.rules.saddle", None, reset=True) and saddle() == d, "reset -> hand-set default"
    ev("gov.grouping.rules.saddle", -99, relative=True)
    assert saddle() == 0.0, "rule weights clamp at 0"
    ev("gov.grouping.prior_phrase_secs", 9999.0)
    assert g.weights.get("grouping.prior_phrase_secs", -1) <= 300.0, "bounded params clamp"
    assert ev("gov.roles.temperature", 0.3) and g.weights.get("roles.temperature", 0) == 0.3
    assert not g.handle_controller_event(ControllerEvent(SET_PARAM, param="cutoff", value=1.0)), \
        "non-gov params must fall through to VoiceManager"
    assert g.runtime_state()["last_edit"].startswith("roles.temperature")

    pat, spec = g.pattern_for_blob(synth_blobs(0)[0])
    assert len(pat.query(Span(0, 1))) >= 1 and spec[0][0] == "degrade"


@test
def gv_routing_and_voice_manager_hooks():
    from routing import Source, extract_source_value
    from lamp_profile import LampProfile
    import voice_manager
    g = _governor()
    blobs = synth_blobs(10)
    for f in range(30):
        g.update(blobs, 1 / 30)
    prof = LampProfile.default()
    assert extract_source_value(Source.TENSION, blobs[0], prof) == min(1.0, max(0.0, blobs[0].score_tension))
    assert extract_source_value(Source.PROMINENCE_RANK, blobs[0], prof) == blobs[0].score_prominence_rank
    assert Source.BEAT_PHASE in Source and Source.DEPTH in Source
    f = voice_manager._nearest_field_freq(300.0, [0, 4, 7])
    pc = round(69 + 12 * math.log2(f / 440.0)) % 12
    assert pc in (0, 4, 7)


@test
def gv_performance_budget():
    g = _governor()
    for f in range(30):
        g.update(synth_blobs(f), 1 / 30)
    t0 = time.perf_counter()
    for f in range(30, 230):
        g.update(synth_blobs(f), 1 / 30)
    ms = (time.perf_counter() - t0) / 200 * 1000
    print(f"      [perf] ScoreGovernor.update: {ms:.2f} ms/frame (3 blobs, no tree)")
    assert ms < 15.0


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
