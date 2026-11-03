# test_hazard_grouping.py -- Audible Lava / DSO v4.9.3
# Behavioural tests for the event-model grouping layer.
#
# The properties that matter are the ones the old beam could not provide:
#   * a refractory period EMERGES from the hazard (no constant)
#   * phrase length is LEARNED, not configured
#   * evidence is SCALE-FREE (rule-weight magnitude and lamp brightness cancel)
#   * boundary rate is bounded and sane without any threshold
#
# v4.9.3 adds the guards that break the churn feedback loop:
#   * the learned mean is FLOORED (default 8 s) and the hazard is CAPPED
#     (default 25 %/s), so a burst of evidence cannot collapse the phrase
#     model and pin the hazard at 1
#   * the prior carries 20 pseudo-observations, so the first dozen
#     boundaries cannot drag the mean to the floor
#   * reset_learning() restores the prior live
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
                cond_entropy_z=0.0, wdiag_z=0.0, saddle_strength=0.0, n_cues=0.0,
                count_z=0.0)
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
    """With the v4.4 prior strength (2 pseudo-observations) the model still
    learns the observed distribution outright; with the v4.9.3 default (20)
    it moves toward it and keeps ~9 % of the prior at steady state."""
    m = GammaIntervalModel(prior_mean=25.0, prior_strength=2.0)
    for _ in range(60):
        m.observe(float(RNG.normal(8.0, 1.0)))       # much shorter than the prior
    assert abs(m.mean - 8.0) < 1.5, f"learned mean {m.mean:.1f}, want ~8"
    assert m.shape > 5.0, f"regular intervals should give high shape, got {m.shape:.1f}"

    d = GammaIntervalModel(prior_mean=25.0)          # default prior_strength 20
    for _ in range(60):
        d.observe(float(RNG.normal(8.0, 1.0)))
    assert 8.0 < d.mean < 25.0 and abs(d.mean - 8.0) < abs(25.0 - 8.0) * 0.6, \
        f"after 60 obs the default model should sit between prior and data, nearer the data: {d.mean:.1f}"
    for _ in range(600):
        d.observe(float(RNG.normal(8.0, 1.0)))
    # steady state: observation count saturates at 1/(1-0.995) = 200, so the
    # prior keeps 20/220 = 9 % of the weight -> mean ~ 0.09*25 + 0.91*8 = 9.5
    assert 8.5 < d.mean < 11.0, f"steady-state mean {d.mean:.1f}, expected ~9.5"
    assert d.shape > 3.0, f"shape {d.shape:.1f}"


@test
def gam_05_prior_resists_a_handful_of_boundaries():
    """v4.9.3 inverts the v4.4 property: 6 short intervals must NOT move the
    mean much, 20 of them should move it about halfway."""
    m = GammaIntervalModel(prior_mean=25.0)
    before = m.mean
    for _ in range(6):
        m.observe(8.0)
    assert m.mean > 0.8 * before, f"prior gave way after 6 observations: {m.mean:.1f}"
    for _ in range(14):
        m.observe(8.0)
    assert 15.0 < m.mean < 18.5, f"after 20 observations expected ~16.5, got {m.mean:.1f}"
    assert 0.45 < m._prior_weight < 0.55, f"prior weight {m._prior_weight:.2f} after 20 obs"


@test
def gam_06_shape_distinguishes_regular_from_memoryless():
    reg = GammaIntervalModel(prior_mean=20.0, prior_strength=2.0)
    irr = GammaIntervalModel(prior_mean=20.0, prior_strength=2.0)
    for _ in range(80):
        reg.observe(20.0 + float(RNG.normal(0, 0.5)))
        irr.observe(float(RNG.exponential(20.0)))
    assert reg.shape > 3 * irr.shape, f"regular {reg.shape:.1f} vs irregular {irr.shape:.1f}"


@test
def gam_07_mean_floor_breaks_the_collapse():
    """The failure seen live: a run of sub-second intervals dragged the mean
    to ~0.4 s and the hazard to 1. Now the mean stops at the floor, the
    floored model keeps a refractory tail (shape capped at the prior's), and
    the per-frame hazard stays small."""
    m = GammaIntervalModel(prior_mean=25.0, prior_strength=2.0)
    for _ in range(400):
        m.observe(0.4)
    assert m.raw_mean < 1.0, f"raw blend should have collapsed: {m.raw_mean:.2f}"
    assert abs(m.mean - m.mean_floor) < 1e-9 and m.floor_active, f"mean {m.mean:.2f} not at floor"
    assert m.shape <= m.prior_shape + 1e-9, f"floored model became near-periodic: shape {m.shape:.1f}"
    assert m.hazard(0.3, DT) < 1e-3, "no refractory period at the floor"
    assert m.hazard(2.0, DT) < 0.01
    assert m.hazard(0.4, 1.0) < 0.05, "a 0.4 s phrase must not be 'due'"
    m2 = GammaIntervalModel(prior_mean=25.0, mean_floor=10.0)
    assert m2.mean_floor == 10.0 and m2.mean >= 10.0


@test
def gam_08_hazard_is_capped():
    """However overdue, at most hazard_cap_per_sec per second (and the exact
    survival form 1 - exp(-h dt) per frame, never the old clip(h dt, 0, 1))."""
    m = GammaIntervalModel(prior_mean=25.0)
    cap = m.hazard_cap_per_sec
    for tau in (40.0, 100.0, 1000.0, 1e6):
        assert m.hazard(tau, 1.0) <= cap + 1e-9, f"tau={tau}: {m.hazard(tau, 1.0):.3f} > cap {cap}"
        per_frame = m.hazard(tau, DT)
        assert per_frame <= 1.0 - math.exp(-m.hazard_rate_cap * DT) + 1e-12
    assert m.hazard(1e6, 1.0) > 0.9 * cap, "deep in the tail the hazard should sit AT the cap"
    assert m.hazard(1e6, DT) < 0.03, "per-frame hazard must stay far below 1"
    # cap is a probability per second: 30 frames at the cap compound to ~cap
    p = 1.0 - (1.0 - m.hazard(1e6, 1.0 / 30.0)) ** 30
    assert abs(p - cap) < 0.01, f"30 capped frames compound to {p:.3f}, cap {cap}"
    m3 = GammaIntervalModel(prior_mean=25.0, hazard_cap_per_sec=0.5)
    assert m3.hazard(1e6, 1.0) > m.hazard(1e6, 1.0)


@test
def gam_09_reset_restores_the_prior():
    m = GammaIntervalModel(prior_mean=25.0)
    for _ in range(300):
        m.observe(6.0)
    assert m.mean < 20.0
    m.reset()
    assert abs(m.mean - 25.0) < 1e-9 and len(m.intervals) == 0 and m.state()["n_observed"] == 0
    assert abs(m.shape - m.prior_shape) < 1e-6


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
    v4.9.3: with 50 pseudo-observations the fitted mean sits BETWEEN the
    prior and the intervals the system actually produced, having departed
    from the prior toward the data."""
    m = module(seed=5)
    def busy(f):
        return loud() if (f // 30) % 4 == 0 else quiet()
    fired = run(m, int(15 * 700), busy)
    assert m.n_boundaries >= 6, f"only {m.n_boundaries} boundaries -- cannot learn"
    observed = np.diff([0.0] + fired)
    prior = GammaIntervalModel.PRIOR_MEAN_SECS
    lo, hi = sorted((float(observed.mean()), prior))
    assert lo - 1.0 <= m.intervals.mean <= hi + 1.0, \
        f"fitted mean {m.intervals.mean:.1f}s outside [observed {observed.mean():.1f}, prior {prior}]"
    assert abs(m.intervals.mean - prior) > 2.0, "model never departed from the prior"


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
def hg_07c_continuous_churn_cannot_pin_the_hazard():
    """The live failure, reproduced: evidence that is loud a third of the time
    for ten minutes. v4.4 collapsed the mean to well under a second and fired
    ~150 boundaries a minute; now the floor, cap and prior bound the rate."""
    m = module(seed=4)
    churn = lambda f: loud() if (f // 15) % 3 == 0 else quiet()     # 1 s loud every 3 s
    max_hz = 0.0
    fired, placed, t = [], [], 0.0
    for f in range(int(15 * 600)):
        t += DT
        d = m.update(churn(f), DT)
        max_hz = max(max_hz, d.hazard)
        if d.fired:
            fired.append(t)                      # when it was REPORTED (commit lag later)
            placed.append(m.last_boundary_t)     # where the boundary was PLACED
    per_min = len(fired) / 10.0
    assert per_min <= 60.0 / m.intervals.mean_floor + 0.5, \
        f"{per_min:.1f} boundaries/min exceeds the floor's ceiling"
    assert per_min <= 8.0, f"{per_min:.1f} boundaries/min under churn"
    assert m.intervals.mean >= m.intervals.mean_floor - 1e-9
    assert max_hz < 0.05, f"per-frame hazard reached {max_hz:.3f} -- was pinned at 1.0 in v4.4"
    ivals = np.diff([0.0] + placed)              # phrase lengths = placement intervals
    assert ivals.min() >= HazardGroupingModule.REVISION_WINDOW_S, \
        f"a {ivals.min():.2f}s phrase slipped through (a re-placed boundary fired twice)"
    # the floor is on the MEAN, not on each phrase: with the evidence bursting
    # every 3 s and a 20-observation prior the model legitimately learns a
    # ~13 s mean and places a fair share of boundaries on the 3 s bursts.
    # What must NOT happen is the sub-second collapse; short phrases should
    # stay a minority.
    assert float((ivals < 4.0).mean()) < 0.40, f"{100*(ivals < 4.0).mean():.0f}% of phrases under 4 s"


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
    for expected in ("gap", "change", "saddle", "changepoint", "regime", "relational", "coincidence",
                     "count_change"):
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
              "beam", "segmentation", "n_resets"):
        assert k in st, k
    pm = st["phrase_model"]
    for k in ("mean_secs", "shape", "raw_mean_secs", "floor_active", "mean_floor_secs",
              "hazard_cap_per_sec", "prior_strength", "prior_weight"):
        assert k in pm, k


@test
def hg_10a_reset_learning_restores_the_prior_live():
    """The EDIT-surface action: after a drifted run, reset_learning() puts
    the phrase prior back, forgets the evidence null, keeps the beam and
    the edited prior parameters, and the module keeps decoding."""
    m = module(seed=7)
    m.set_weight("prior_phrase_secs", 20.0)
    m.set_weight("phrase_floor_secs", 6.0)
    dense = lambda f: loud() if (f // 30) % 3 == 0 else quiet()
    run(m, int(15 * 600), dense)
    assert m.n_boundaries > 5 and m.intervals.mean < 20.0
    assert m.evidence.n > 100
    beam_len = len(m.beam)
    m.reset_learning()
    assert m.n_resets == 1
    assert abs(m.intervals.mean - 20.0) < 1e-9, f"prior not restored: {m.intervals.mean:.1f}"
    assert m.intervals.mean_floor == 6.0, "edited floor lost on reset"
    assert m.intervals.state()["n_observed"] == 0 and m.evidence.n == 0
    assert len(m.beam) == beam_len, "beam should survive a learning reset"
    before = m.n_boundaries
    run(m, int(15 * 200), dense)
    assert m.n_boundaries > before, "module stopped decoding after reset"


@test
def hg_10b_new_weights_are_live():
    m = module()
    m.set_weight("prior_strength", 10.0);        assert m.intervals.n0 == 10.0
    m.set_weight("phrase_floor_secs", 12.0);     assert m.intervals.mean_floor == 12.0
    m.set_weight("hazard_cap_per_sec", 0.1);     assert abs(m.intervals.hazard_cap_per_sec - 0.1) < 1e-12
    assert m.intervals.hazard(1e6, 1.0) <= 0.1 + 1e-9
    assert m.intervals.mean >= 12.0


@test
def hg_11_deleted_configuration_is_gone():
    """No thresholds, no switch cost, no forget, no temperature, no refractory."""
    gw = SG.DEFAULT_WEIGHTS["grouping"]
    for dead in ("commit_threshold", "switch_cost", "forget", "temperature"):
        assert dead not in gw, f"{dead} should no longer exist in the grouping weights"
    assert "regularity" not in gw["rules"]
    assert set(gw) == {"prior_phrase_secs", "prior_strength", "phrase_floor_secs", "hazard_cap_per_sec",
                       "evidence_rate", "min_cue_age_secs", "pers_tau_secs", "count_tau_secs",
                       "rules"}, sorted(gw)
    assert gw["prior_strength"] == GammaIntervalModel.PRIOR_STRENGTH
    assert gw["phrase_floor_secs"] == GammaIntervalModel.MEAN_FLOOR_SECS
    assert gw["hazard_cap_per_sec"] == GammaIntervalModel.HAZARD_CAP_PER_SEC
    assert "count_change" in gw["rules"]
    assert SG.DEFAULT_WEIGHTS["_version"] == 3, "do not bump _version: it would overwrite a user's weights file"


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
