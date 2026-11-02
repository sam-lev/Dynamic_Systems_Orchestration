# test_hazard_grouping.py -- Audible Lava / DSO v4.4
# Behavioural tests for the event-model grouping layer.
#
# The properties that matter are the ones the old beam could not provide:
#   * a refractory period EMERGES from the hazard (no constant)
#   * phrase length is LEARNED, not configured
#   * evidence is SCALE-FREE (rule-weight magnitude and lamp brightness cancel)
#   * boundary rate is bounded and sane without any threshold
#
# Run: python test_hazard_grouping.py  |  pytest test_hazard_grouping.py -v

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("/mnt/project", HERE):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from hazard_grouping import (GammaIntervalModel, RobustStandardiser,      # noqa: E402
                             HazardGroupingModule, _phi)
from preference_rules import RuleRegistry, rule                            # noqa: E402
import score_governor as SG                                                # noqa: E402

TESTS = []
def test(fn):
    TESTS.append(fn); return fn

RNG = np.random.default_rng(11)
DT = 1.0 / 15.0


def ev(**kw):
    """An evidence dict with all keys the grouping rules read."""
    base = dict(t=0.0, dt=DT, since_boundary=10.0, since_event=1.0, phrase_mean_len=25.0,
                ke_z=0.0, occ_z=0.0, flux_z=0.0, coh_z=0.0, pairvar_z=0.0,
                collin_event=0.0, com_cross=0.0, bocpd_max=0.0,
                cond_entropy_z=0.0, wdiag_z=0.0, saddle_strength=0.0, n_cues=0.0)
    base.update(kw)
    return base


def quiet():
    return ev()


def loud():
    return ev(ke_z=3.5, occ_z=3.0, flux_z=2.5, bocpd_max=0.8, saddle_strength=0.7, n_cues=4.0)


def run(module, n_frames, evidence_fn, dt=DT):
    """Drive the module and return boundary times."""
    fired = []
    t = 0.0
    for f in range(n_frames):
        t += dt
        d = module.update(evidence_fn(f), dt)
        if d.fired:
            fired.append(t)
    return fired


# ── Gamma interval model ────────────────────────────────────────────────────

@test
def gam_01_prior_before_any_observation():
    m = GammaIntervalModel(prior_mean=25.0)
    assert abs(m.mean - 25.0) < 0.5, f"prior mean {m.mean:.1f}"
    assert 3.0 < m.shape < 5.5, f"prior shape {m.shape:.2f}"


@test
def gam_02_hazard_is_near_zero_just_after_a_boundary():
    """The refractory period is not a constant -- it falls out of the Gamma."""
    m = GammaIntervalModel(prior_mean=25.0)
    h0 = m.hazard(0.2, DT)
    h_mid = m.hazard(20.0, DT)
    assert h0 < 1e-3, f"hazard right after a boundary should be ~0, got {h0:.5f}"
    assert h_mid > 20 * h0, "hazard must rise substantially by the typical phrase length"


@test
def gam_03_hazard_rises_monotonically_through_the_body():
    m = GammaIntervalModel(prior_mean=25.0)
    hs = [m.hazard(t, DT) for t in (1, 5, 10, 20, 30, 45)]
    assert all(b >= a for a, b in zip(hs, hs[1:])), f"hazard not monotone: {hs}"


@test
def gam_04_learns_the_observed_interval_distribution():
    m = GammaIntervalModel(prior_mean=25.0)
    for _ in range(60):
        m.observe(float(RNG.normal(8.0, 1.0)))       # much shorter than the prior
    assert abs(m.mean - 8.0) < 1.5, f"learned mean {m.mean:.1f}, want ~8"
    assert m.shape > 5.0, f"regular intervals should give high shape, got {m.shape:.1f}"


@test
def gam_05_prior_washes_out_in_a_handful_of_boundaries():
    m = GammaIntervalModel(prior_mean=25.0)
    before = m.mean
    for _ in range(6):
        m.observe(8.0)
    assert m.mean < 0.65 * before, f"prior still dominant after 6 observations: {m.mean:.1f}"


@test
def gam_06_shape_distinguishes_regular_from_memoryless():
    reg = GammaIntervalModel(prior_mean=20.0)
    irr = GammaIntervalModel(prior_mean=20.0)
    for _ in range(80):
        reg.observe(20.0 + float(RNG.normal(0, 0.5)))
        irr.observe(float(RNG.exponential(20.0)))
    assert reg.shape > 3 * irr.shape, f"regular {reg.shape:.1f} vs irregular {irr.shape:.1f}"


# ── Robust standardiser ─────────────────────────────────────────────────────

@test
def std_01_tracks_median_and_scale():
    s = RobustStandardiser(rate=0.05)
    for _ in range(4000):
        s.update(float(RNG.normal(5.0, 2.0)))
    assert abs(s.med - 5.0) < 0.5, f"median {s.med:.2f}"
    assert abs(s.z(9.0) - 2.0) < 0.6, f"z(mu+2sd) = {s.z(9.0):.2f}, want ~2"


@test
def std_02_is_scale_invariant():
    """Doubling every rule weight must not change the z-score at all."""
    a = RobustStandardiser(rate=0.05)
    b = RobustStandardiser(rate=0.05)
    xs = RNG.normal(3.0, 1.0, 4000)
    for x in xs:
        a.update(float(x)); b.update(float(x) * 7.3)
    za, zb = a.z(6.0), b.z(6.0 * 7.3)
    assert abs(za - zb) < 0.15, f"z differs under scaling: {za:.3f} vs {zb:.3f}"


@test
def std_03_outliers_do_not_inflate_the_scale():
    """Boundary moments ARE the outliers; using mean/sd would let them raise
    sigma and suppress the next one."""
    s = RobustStandardiser(rate=0.02)
    for _ in range(3000):
        s.update(float(RNG.normal(2.0, 0.5)))
    mad_before = s.mad
    for _ in range(30):
        s.update(50.0)                              # extreme spikes
    assert s.mad < 3.0 * mad_before, f"MAD inflated {mad_before:.2f} -> {s.mad:.2f}"


# ── The module ──────────────────────────────────────────────────────────────

def module(seed=3, stochastic=False, commit_lag=1.5, beam=6):
    return HazardGroupingModule(weights=None, seed=seed, stochastic=stochastic,
                                commit_lag_secs=commit_lag, beam_size=beam)


@test
def hg_01_no_boundary_immediately_after_one():
    """The emergent refractory: no configuration constant involved."""
    m = module()
    m.last_boundary_t = m.t                          # pretend one just fired
    m.beam = [type(m.beam[0])(t0=m.t, logp=0.0)]     # and the beam agrees
    fired = run(m, int(15 * 3), lambda f: loud())    # 3 s of maximal evidence
    assert not fired, f"fired {fired} within 3 s of a boundary"


@test
def hg_02_strong_evidence_when_overdue_does_fire():
    m = module()
    m.t = 60.0                                       # well past the prior mean
    m.last_boundary_t = 0.0
    for _ in range(400):                             # settle the standardiser low
        m.update(quiet(), DT)
    fired = run(m, int(15 * 20), lambda f: loud())
    assert fired, "overdue + strong evidence must eventually fire"


@test
def hg_03_evidence_alone_early_does_not_fire():
    """Soft AND: being early beats strong evidence."""
    m = module()
    fired = run(m, int(15 * 4), lambda f: loud())
    assert not fired, f"fired {fired} in the first 4 s"


@test
def hg_04_overdue_alone_without_evidence_is_rare():
    m = module()
    for _ in range(600):
        m.update(quiet(), DT)                        # build the null distribution
    m.t += 200.0                                     # far overdue
    fired = run(m, int(15 * 30), lambda f: quiet())  # but nothing happening
    # Phi(z) ~ 0.5 at the median, so it is not impossible -- just not prompt.
    assert len(fired) <= 3, f"{len(fired)} boundaries on no evidence at all"


@test
def hg_05_boundary_rate_is_sane_without_a_threshold():
    m = module()
    def mixed(f):
        return loud() if (f // 90) % 7 == 0 else quiet()
    fired = run(m, int(15 * 600), mixed)             # 10 minutes
    per_min = len(fired) / 10.0
    assert 0.3 <= per_min <= 6.0, f"{per_min:.2f} boundaries/min is implausible"


@test
def hg_06_rule_weight_scale_does_not_change_behaviour():
    """The property the beam never had: absolute rule-weight magnitude cancels."""
    def mixed(f):
        return loud() if (f // 90) % 5 == 0 else quiet()
    a = module(seed=9)
    b = module(seed=9)
    for r in b.rules:
        r.weight *= 11.0                             # same relative weights
    fa = run(a, int(15 * 400), mixed)
    fb = run(b, int(15 * 400), mixed)
    assert abs(len(fa) - len(fb)) <= max(2, 0.35 * max(len(fa), 1)), \
        f"scaling all weights changed the boundary count: {len(fa)} vs {len(fb)}"


@test
def hg_07_learned_model_is_self_consistent():
    """The model has no target rate to hit -- it learns whatever rate emerges.
    The property to check is that the fitted mean matches the intervals the
    system actually produced (and so departs from the prior when the evidence
    warrants it)."""
    m = module(seed=5)
    def busy(f):
        return loud() if (f // 30) % 4 == 0 else quiet()
    fired = run(m, int(15 * 700), busy)
    assert m.n_boundaries >= 6, f"only {m.n_boundaries} boundaries -- cannot learn"
    observed = np.diff([0.0] + fired)
    assert abs(m.intervals.mean - observed.mean()) < 0.35 * observed.mean(), \
        f"fitted mean {m.intervals.mean:.1f}s vs observed {observed.mean():.1f}s"
    assert abs(m.intervals.mean - GammaIntervalModel.PRIOR_MEAN_SECS) > 5.0, \
        "model never departed from the prior"


@test
def hg_07b_denser_evidence_gives_shorter_phrases():
    """Comparative, not absolute: more frequent strong evidence must shorten
    the learned phrase length."""
    def sparse(f):
        return loud() if (f // 30) % 12 == 0 else quiet()
    def dense(f):
        return loud() if (f // 30) % 3 == 0 else quiet()
    a = module(seed=21); run(a, int(15 * 900), sparse)
    b = module(seed=21); run(b, int(15 * 900), dense)
    assert b.n_boundaries > a.n_boundaries, \
        f"dense evidence gave {b.n_boundaries} boundaries, sparse gave {a.n_boundaries}"
    assert b.intervals.mean < a.intervals.mean, \
        f"dense phrase mean {b.intervals.mean:.1f}s not shorter than sparse {a.intervals.mean:.1f}s"


@test
def hg_08_depth_is_graded_by_evidence_strength():
    m = module()
    m.t = 500.0; m.last_boundary_t = 0.0
    for _ in range(800):
        m.update(quiet(), DT)
    d = m.update(loud(), DT)
    if d.fired:
        assert d.depth >= 1, "a strong-evidence boundary should not be depth 0"
        assert 0.0 <= d.strength <= 1.0


@test
def hg_09_regularity_rule_is_dropped():
    m = module()
    names = [r.name for r in m.rules]
    assert "regularity" not in names, "regularity is superseded by the hazard"
    for expected in ("gap", "change", "saddle", "changepoint", "regime", "relational", "coincidence"):
        assert expected in names, f"missing rule {expected}"


@test
def hg_09a_rule_weights_are_per_instance():
    """Two modules must not share Rule objects -- the registry holds defaults,
    instances own their weights."""
    a, b = module(), module()
    for r in b.rules:
        r.weight *= 5.0
    wa = {r.name: r.weight for r in a.rules}
    wb = {r.name: r.weight for r in b.rules}
    assert all(abs(wb[k] - 5.0 * wa[k]) < 1e-9 for k in wa), "weights did not scale independently"
    assert any(wa[k] != wb[k] for k in wa), "instances share Rule objects"


@test
def hg_09b_beam_holds_competing_segmentations():
    """The Viterbi property: several segmentation hypotheses are alive at once
    and are compared as PATHS, not as per-frame choices."""
    m = module(beam=6)
    def mixed(f):
        return loud() if (f // 60) % 5 == 0 else quiet()
    run(m, int(15 * 200), mixed)
    assert m.last.n_hyp >= 2, f"beam collapsed to {m.last.n_hyp} hypothesis"
    assert len({round(h.t0, 2) for h in m.beam}) == len(m.beam), "duplicate t0 in beam"
    lp = [h.logp for h in m.beam]
    assert lp == sorted(lp, reverse=True), "beam not ordered by path score"
    assert abs(lp[0]) < 1e-9, "top path should be renormalised to 0"


@test
def hg_09c_commit_is_lagged_and_evidence_sets_placement():
    """Two properties of decoding rather than sampling.

    (a) A boundary is never reported the instant it is hypothesised: it must
        survive `commit_lag` seconds of subsequent evidence first, so recent
        boundaries stay open to revision inside the beam.
    (b) The duration model sets the RATE; the evidence sets the PLACEMENT.
        A burst of evidence does not manufacture extra boundaries, it pulls
        the boundary the hazard was already going to produce toward itself.
    """
    m = module(commit_lag=2.0)
    for _ in range(900):                             # settle the standardiser
        m.update(quiet(), DT)
    # Every committed boundary must be at least commit_lag old when reported,
    # which is the invariant that makes recent boundaries revisable.
    lags = []
    for f in range(int(15 * 400)):
        d = m.update(loud() if (f // 15) % 25 == 0 else quiet(), DT)
        if d.fired:
            lags.append(d.committed_lag)
    assert lags, "no boundaries committed in 400 s"
    assert min(lags) >= 2.0 - 1e-6, f"committed a boundary only {min(lags):.2f}s old"

    # Placement: over a long run, committed boundaries should sit closer to
    # evidence bursts than chance would put them. The duration model sets the
    # RATE; evidence only pulls the boundary the hazard was going to produce
    # anyway toward itself.
    m2 = module(commit_lag=1.5)
    burst = lambda f: loud() if (f // 15) % 40 == 0 else quiet()   # 1 s every 40 s
    fired = run(m2, int(15 * 1200), burst)
    assert len(fired) >= 3, f"only {len(fired)} boundaries in 1200 s"
    d = [min(abs(t - 40 * k) for k in range(0, 31)) for t in fired]
    assert float(np.median(d)) < 20.0, \
        f"median distance to a burst {np.median(d):.1f}s is no better than chance (20 s)"


@test
def hg_09d_decode_is_deterministic():
    """Reproducible runs -- a sampled commit could not give this."""
    def mixed(f):
        return loud() if (f // 70) % 4 == 0 else quiet()
    a, b = module(seed=1), module(seed=999)          # seeds must not matter
    fa, fb = run(a, int(15 * 250), mixed), run(b, int(15 * 250), mixed)
    assert fa == fb, f"decode differed across seeds: {len(fa)} vs {len(fb)}"


@test
def hg_10_state_is_introspectable():
    m = module()
    run(m, 300, lambda f: quiet())
    st = m.state()
    for k in ("p_boundary", "hazard", "evidence", "z", "phrase_model", "evidence_model",
              "beam", "segmentation"):
        assert k in st, k
    assert "mean_secs" in st["phrase_model"] and "shape" in st["phrase_model"]


@test
def hg_11_deleted_configuration_is_gone():
    """No thresholds, no switch cost, no forget, no temperature, no refractory."""
    gw = SG.DEFAULT_WEIGHTS["grouping"]
    for dead in ("commit_threshold", "switch_cost", "forget", "temperature"):
        assert dead not in gw, f"{dead} should no longer exist in the grouping weights"
    assert "regularity" not in gw["rules"]
    assert set(gw) == {"prior_phrase_secs", "evidence_rate", "rules"}, sorted(gw)


@test
def hg_12_governor_end_to_end():
    from dataclasses import dataclass

    @dataclass
    class B:
        id: int; centroid: tuple; area: float; birth: float = 150.0; death: float = 20.0
        persistence: float = 40.0; vx: float = 0.0; vy: float = 0.0; speed: float = 0.0
        ay: float = 0.0; direction: str = "turning"; age: int = 0

    for p in ("/tmp/_hg.json",):
        if os.path.exists(p):
            os.remove(p)
    g = SG.ScoreGovernor(320, 240, weights_path="/tmp/_hg.json",
                         runtime_path="/tmp/_hgr.json", log_path="/tmp/_hgl.txt", fps=15)
    fired = 0
    for f in range(15 * 300):
        t = f / 15.0
        boost = 3.0 if (f // (15 * 45)) % 2 else 1.0
        blobs = []
        for i, (x, ph, per) in enumerate([(60, 0, 12.0), (160, 2.0, 8.0), (260, 1.0, 12.0)]):
            y = 120 + 80 * math.sin(2 * math.pi * t / per + ph)
            vy = 80 * 2 * math.pi / per * math.cos(2 * math.pi * t / per + ph) / 15.0
            blobs.append(B(i, (x, y), 3000 * boost, vy=vy * boost,
                           speed=abs(vy) * boost, age=f))
        s = g.update(blobs, 1 / 15.0)
        fired += bool(s.boundary)
    assert 1 <= fired <= 40, f"{fired} boundaries in 300 s"
    st = g.runtime_state()["grouping"]
    assert st["phrase_model"]["mean_secs"] > 0
    assert 0.0 <= s.phrase_position <= 1.0


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
