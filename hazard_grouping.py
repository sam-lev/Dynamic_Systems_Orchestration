# hazard_grouping.py -- Audible Lava / DSO v4.9.3
# Grouping as an EVENT model, not a state-selection problem.
#
# Why this replaces the Viterbi beam
# ----------------------------------
# A boundary is an event; "continue" is a state. Modelling them as two
# competing states in a beam with a switching cost and a memory decay is a
# category error, and every symptom followed from it:
#
#   * commit required argmax(posterior) == boundary AND posterior > threshold.
#     With two candidates argmax is exactly posterior > 0.5, so the threshold
#     did nothing below 0.5 and the argmax did nothing above it: two knobs that
#     were one knob, with the useful half of its range unreachable. Walking
#     commit_threshold 0.25 -> 0.19 therefore produced no boundaries at all
#     while posteriors sat at 0.46-0.48.
#   * grouping.forget had to be detuned to 0.7 (vs 0.98 elsewhere) to stop a
#     "boundary state" from persisting across frames.
#   * a refractory constant had to be bolted on separately.
#   * a `regularity` rule had to hand-encode a phrase-length preference the
#     model could not represent.
#
# The formulation
# ---------------
#   P(boundary at t) = h(tau) * Phi(z)
#
#   tau  time since the last boundary.
#   h    hazard of an online Gamma(k, theta) fitted to observed inter-boundary
#        intervals -- the piece learns its own phrase-length distribution.
#        This subsumes the refractory period (hazard ~ 0 just after a
#        boundary), the `regularity` rule (hazard peaks near the typical
#        phrase length) and the threshold's role of setting boundary rate.
#   z    (s - mu) / sigma of the weighted rule sum s, with mu/sigma estimated
#        online from the median and MAD. Scale-free: the absolute range of the
#        rule weights, and the brightness of a particular lamp, both cancel.
#   Phi  standard normal CDF, so evidence enters as a probability.
#
# The product is a soft AND: a boundary needs both "we are due" and "something
# happened". Evidence alone early in a phrase will not fire; being overdue
# alone without evidence will not either. That is closer to how GTTM's
# preference rules actually interact than a sum-and-threshold.
#
# Commitment is STOCHASTIC (draw with probability P), consistent with the
# project's principle that structure emerges from the probabilistic model
# rather than from explicit machinery. It also yields graded boundaries for
# free: boundary_strength = P, so a high-P moment reads as a section break and
# a low-P one as motive-level articulation.
#
# Learned state (4 quantities, no free hyperparameters):
#   Gamma shape k, Gamma scale theta      -- phrase-length distribution
#   evidence median mu, evidence MAD sigma -- evidence null distribution
#
# The only number carrying a musical opinion is the prior phrase length
# (PRIOR_MEAN_SECS), which is a unit-bearing quantity you can reason about,
# unlike a posterior threshold of 0.55.
#
# v4.9.3 -- breaking the positive-feedback loop
# ---------------------------------------------
# In a live run the learned phrase mean collapsed to well under a second: a
# noisy evidence stream fired a few boundaries close together, each short
# interval dragged the fitted mean down, the hazard rose everywhere, more
# boundaries fired, and the hazard pinned at 1.0 (~2500 boundaries in 17
# minutes; the schema flipped every ~0.4 s). Four guards, all on the
# duration half of the model:
#
#   * MEAN_FLOOR_SECS   the fitted mean is clamped to >= 8 s. Below the floor
#                       the data are no longer believed about regularity
#                       either: shape is capped at the prior shape, so the
#                       floored distribution keeps a refractory tail rather
#                       than becoming near-periodic at exactly the floor.
#   * HAZARD_CAP_PER_SEC the hazard RATE is capped so that even an overdue
#                       phrase fires with at most this probability per
#                       second (0.25 -> 25 %/s at neutral evidence). The
#                       per-frame probability is 1 - exp(-min(h, h_max) dt),
#                       the exact survival form, instead of clip(h dt, 0, 1).
#   * PRIOR_STRENGTH    2 -> 20 pseudo-observations. The first dozen
#                       boundaries cannot drag the mean to the floor; the
#                       prior still has half the weight after 20 observed
#                       intervals and ~9 % of it at steady state (the
#                       observation count saturates at 1/(1-forget) = 200).
#   * reset()           restores the prior live; exposed on the EDIT surface
#                       as gov.grouping.reset_learning (controller `r` key).
#
# The evidence half is gated in score_governor.py (young-blob lifecycle
# events ignored, count changes smoothed and gated on persistence).

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from preference_rules import RuleRegistry, Rule, WeightStore


# ── Online interval model ────────────────────────────────────────────────────

class GammaIntervalModel:
    """Online Gamma(k, theta) over inter-boundary intervals, with a prior.

    Gamma is the right family here: it is the standard renewal-process
    interval distribution, its hazard rises from ~0 (giving a refractory
    period at no cost) and its shape parameter controls how regular the
    phrasing is (k -> 1 is memoryless/Poisson, large k is near-periodic).

    Fitted by moment matching on an EMA of the first two moments, so it tracks
    a piece whose phrase lengths drift rather than assuming stationarity.

    v4.9.3: floored mean, capped hazard, stronger prior, reset(). See the
    module header for why.
    """

    # Prior: musically plausible phrase length, shape 4 so it is not
    # exponential. Deliberately the ONLY musically-opinionated constant.
    PRIOR_MEAN_SECS = 25.0
    PRIOR_SHAPE     = 4.0
    PRIOR_STRENGTH  = 20.0       # equivalent observation count (v4.9.3: was 2)
    # Guards (v4.9.3)
    MEAN_FLOOR_SECS     = 8.0    # fitted mean never drops below this
    HAZARD_CAP_PER_SEC  = 0.25   # max P(boundary within 1 s) at neutral evidence

    def __init__(self, prior_mean: float = PRIOR_MEAN_SECS,
                 prior_shape: float = PRIOR_SHAPE,
                 prior_strength: float = PRIOR_STRENGTH,
                 mean_floor: float = MEAN_FLOOR_SECS,
                 hazard_cap_per_sec: float = HAZARD_CAP_PER_SEC):
        self.prior_mean  = float(prior_mean)
        self.prior_shape = float(prior_shape)
        self.n0          = float(prior_strength)
        self.mean_floor  = float(mean_floor)
        self.hazard_cap_per_sec = float(hazard_cap_per_sec)
        # Observations are accumulated SEPARATELY from the prior and blended by
        # a weight that decays with observation count. Folding the prior into
        # the running sums instead lets its variance (prior_mean^2/shape, which
        # is large) keep contaminating the shape estimate long after its mean
        # has washed out -- the shape would read ~3 after 60 tightly-clustered
        # observations that truly imply ~60.
        self._n    = 0.0
        self._sum  = 0.0
        self._sum2 = 0.0
        self.intervals: Deque[float] = deque(maxlen=64)

    def reset(self) -> None:
        """Forget every observed interval; the prior alone is in force."""
        self._n = self._sum = self._sum2 = 0.0
        self.intervals.clear()

    @property
    def prior_var(self) -> float:
        return self.prior_mean ** 2 / max(self.prior_shape, 1e-6)

    @property
    def _prior_weight(self) -> float:
        """Prior share of the estimate; -> n0/(n0 + n) as observations accumulate."""
        return self.n0 / (self.n0 + self._n)

    # -- fitted parameters -------------------------------------------------
    @property
    def raw_mean(self) -> float:
        """Prior/observation blend BEFORE the floor (for introspection)."""
        w = self._prior_weight
        obs = self._sum / self._n if self._n > 0 else self.prior_mean
        return w * self.prior_mean + (1.0 - w) * obs

    @property
    def floor_active(self) -> bool:
        return self.raw_mean < self.mean_floor

    @property
    def mean(self) -> float:
        return max(self.raw_mean, self.mean_floor)

    @property
    def var(self) -> float:
        w = self._prior_weight
        if self._n > 1:
            om = self._sum / self._n
            ov = max(self._sum2 / self._n - om * om, 1e-6)
        else:
            ov = self.prior_var
        return max(w * self.prior_var + (1.0 - w) * ov, 1e-6)

    @property
    def shape(self) -> float:
        """k = mean^2 / var. Large k = regular phrasing, k -> 1 = memoryless.

        When the floor is active the observed intervals are shorter than the
        model is allowed to believe, so their (tiny) variance says nothing
        about regularity at the floored mean: k is then capped at the prior
        shape. Without the cap a run of 0.4 s intervals would turn the floored
        model into a near-periodic Gamma(50) at exactly 8 s, i.e. a hazard
        that is ~0 for 7 s and then a wall.
        """
        k = float(np.clip(self.mean ** 2 / self.var, 0.5, 50.0))
        if self.floor_active:
            k = min(k, self.prior_shape)
        return k

    @property
    def scale(self) -> float:
        return self.mean / self.shape

    @property
    def hazard_rate_cap(self) -> float:
        """Rate (per second) equivalent to hazard_cap_per_sec as a probability:
        1 - exp(-rate) = cap  =>  rate = -ln(1 - cap)."""
        c = float(np.clip(self.hazard_cap_per_sec, 1e-4, 1.0 - 1e-6))
        return -math.log1p(-c)

    def observe(self, interval: float, forget: float = 0.995) -> None:
        """Record an inter-boundary interval. `forget` lets the estimate track
        drift in phrase length rather than averaging the whole session."""
        iv = float(max(interval, 1e-3))
        self._n    = self._n * forget + 1.0
        self._sum  = self._sum * forget + iv
        self._sum2 = self._sum2 * forget + iv * iv
        self.intervals.append(iv)

    # -- hazard ------------------------------------------------------------
    def hazard_rate(self, tau: float) -> float:
        """Instantaneous hazard f(tau)/S(tau) of the fitted Gamma, per second,
        capped at hazard_rate_cap."""
        tau = float(max(tau, 0.0))
        k, th = self.shape, self.scale
        cap = self.hazard_rate_cap
        if th <= 0.0:
            return 0.0
        try:
            from scipy.stats import gamma as _g
            sf = float(_g.sf(tau, a=k, scale=th))
            pdf = float(_g.pdf(tau, a=k, scale=th))
        except Exception:                      # pragma: no cover - scipy present
            x = tau / th
            pdf = (x ** (k - 1) * math.exp(-x)) / (th * math.gamma(k))
            sf = max(1.0 - _lower_gamma_reg(k, x), 1e-9)
        if sf <= 1e-9:
            # Far into the tail: as due as the model is allowed to be.
            return cap
        return float(min(pdf / sf, cap))

    def hazard(self, tau: float, dt: float) -> float:
        """Probability that a boundary is due in the next dt seconds, given
        that none has occurred for tau seconds.

        p = 1 - exp(-h(tau) dt), h capped at hazard_rate_cap, so the
        probability of a boundary within any one second never exceeds
        hazard_cap_per_sec however overdue the phrase is.
        """
        h = self.hazard_rate(tau)
        return float(np.clip(-math.expm1(-h * max(dt, 0.0)), 0.0, 1.0))

    def state(self) -> Dict[str, float]:
        return {"mean_secs": round(self.mean, 2), "shape": round(self.shape, 2),
                "scale": round(self.scale, 2), "n_observed": len(self.intervals),
                "raw_mean_secs": round(self.raw_mean, 2),
                "floor_active": bool(self.floor_active),
                "mean_floor_secs": round(self.mean_floor, 2),
                "hazard_cap_per_sec": round(self.hazard_cap_per_sec, 3),
                "prior_strength": round(self.n0, 1),
                "prior_weight": round(self._prior_weight, 3)}


def _lower_gamma_reg(k: float, x: float, terms: int = 200) -> float:
    """Regularized lower incomplete gamma P(k, x) by series -- fallback only."""
    if x <= 0:
        return 0.0
    s, term = 1.0 / k, 1.0 / k
    for n in range(1, terms):
        term *= x / (k + n)
        s += term
        if term < 1e-12 * s:
            break
    return float(min(1.0, s * math.exp(-x + k * math.log(max(x, 1e-12)) - math.lgamma(k))))


# ── Online robust standardiser ───────────────────────────────────────────────

class RobustStandardiser:
    """Online median / MAD, so evidence enters as a scale-free z-score.

    Median and MAD rather than mean and sd because the evidence stream is
    heavy-tailed by construction: boundary moments are exactly the outliers we
    care about, and letting them inflate sigma would suppress the next one.
    Updated by the standard stochastic-approximation rule (a sign step on the
    median, which converges to the true median for any step schedule that
    decays appropriately -- here a fixed small rate, so it tracks drift).
    """

    # Samples further than CLIP * mad from the median do NOT update the scale
    # at all (trimming, not winsorizing). Without this the spikes the model
    # exists to FIND inflate the scale that makes them findable, so a burst of
    # boundaries desensitises the model to the next one -- and winsorizing is
    # not enough, because a run of consecutive extremes still walks the scale
    # up. The median still tracks every sample: its sign step moves by at most
    # rate * mad, so it is robust by construction. 4 sigma is far enough out
    # that trimming loses nothing on normal data.
    CLIP = 4.0

    def __init__(self, rate: float = 0.01, init_scale: float = 1.0):
        self.rate = float(rate)
        self.med: Optional[float] = None
        self.mad: float = float(init_scale)
        self.n = 0

    def update(self, x: float) -> None:
        x = float(x)
        self.n += 1
        if self.med is None:
            self.med = x
            return
        step = self.rate * max(self.mad, 1e-6)
        self.med += step * (1.0 if x > self.med else -1.0 if x < self.med else 0.0)
        dev = abs(x - self.med)
        if dev <= self.CLIP * max(self.mad, 1e-6) or self.n < 30:
            # Early on (n < 30) accept everything, otherwise the initial scale
            # can never grow away from init_scale.
            self.mad += self.rate * (dev - self.mad)

    def z(self, x: float) -> float:
        if self.med is None or self.n < 10:
            return 0.0
        # 1.4826 * MAD is the consistent estimator of sigma for normal data.
        sigma = max(1.4826 * self.mad, 1e-6)
        return float((x - self.med) / sigma)

    def state(self) -> Dict[str, float]:
        return {"median": round(self.med, 3) if self.med is not None else None,
                "mad": round(self.mad, 3), "n": self.n}


def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ── The module ───────────────────────────────────────────────────────────────

@dataclass
class BoundaryDecision:
    fired: bool = False
    p: float = 0.0                 # P(boundary this frame) = hazard * evidence
    hazard: float = 0.0
    evidence: float = 0.0          # Phi(z)
    z: float = 0.0
    raw_score: float = 0.0
    strength: float = 0.0
    depth: int = 0
    n_hyp: int = 0                 # live segmentation hypotheses in the beam
    committed_lag: float = 0.0     # how far back the committed boundary was


@dataclass
class Hypothesis:
    """One segmentation hypothesis: 'the last boundary was at t0'.

    `logp` is the accumulated log-likelihood of the whole segmentation path
    that led here, so comparing hypotheses compares SEGMENTATIONS, not frames.
    `parent` chains back through previous boundaries, which is what makes this
    a Viterbi decode rather than a greedy per-frame choice.
    """
    t0: float
    logp: float
    parent: Optional["Hypothesis"] = field(default=None, repr=False)
    depth_at_birth: int = 0
    strength_at_birth: float = 0.0


class HazardGroupingModule:
    """Grouping as an online hidden semi-Markov model, decoded by beam Viterbi.

    The two requirements this reconciles:

      * A boundary is an EVENT. The per-frame boundary probability is
        P = h(tau) * Phi(z): the Gamma hazard says whether we are DUE, the
        z-scored evidence says whether something HAPPENED. Neither alone
        fires. This is the emission/duration model.

      * Structure is chosen by VITERBI over a BEAM. Rather than drawing a coin
        each frame, the module carries K competing segmentation hypotheses --
        each one an assertion about where the last boundary fell -- and scores
        whole segmentation paths. Each frame every hypothesis may either
        CONTINUE (accumulating log(1 - P)) or SPAWN a child that asserts a
        boundary now (accumulating log P). The beam keeps the K best paths.
        The committed boundary is read off the best path, so a boundary is
        confirmed by what follows it, not only by the frame it occurred on.

    Consequences of decoding rather than sampling:
      * Boundary placement is REVISABLE within the beam's horizon: a
        hypothesis that placed a boundary two seconds ago can be overtaken by
        one that placed it now, and the reported segmentation improves.
      * There is still no threshold and no refractory constant. The duration
        model supplies both (hazard ~ 0 immediately after a boundary), and the
        beam supplies hysteresis (a rival must out-score an incumbent PATH,
        not merely a frame).
      * Commitment is deterministic given the evidence -- reproducible runs,
        which matters for the paper.

    Deleted configuration is still deleted: commit_threshold, switch_cost,
    forget, temperature, the 1.5 s refractory, the regularity rule. Added:
    beam_size (structural, not a tuning knob) and commit_lag_secs, the delay
    before a boundary is considered settled.

    Weight keys (all under "grouping."): rules.<name>, prior_phrase_secs,
    prior_strength, phrase_floor_secs, hazard_cap_per_sec, evidence_rate,
    beam_size, commit_lag_secs. reset_learning() is the live restore.
    """

    name = "grouping"
    IGNORED_RULES = ("regularity",)
    # Bounds on the evidence likelihood ratio. Structural, not a tuning knob:
    # they cap how far a single frame of evidence can move the decode, which
    # is what keeps a sustained loud passage from firing a boundary per frame.
    LR_MIN, LR_MAX = 0.05, 20.0
    # v4.9.3. The beam keeps one hypothesis per 250 ms bin, so two rival
    # PLACEMENTS of the same boundary can sit 0.1-0.9 s apart; when the best
    # path later switches between them the old code committed both, i.e.
    # fired twice and taught the interval model a 0.2 s "phrase". A commit
    # this close to the previous one is a re-placement: the boundary time
    # moves, nothing fires. Structural (the duration model makes a genuine
    # second phrase inside one second all but impossible anyway).
    REVISION_WINDOW_S = 1.0

    def __init__(self, weights: Optional[WeightStore] = None,
                 beam_size: int = 6, commit_lag_secs: float = 1.5,
                 seed: int = 17, stochastic: bool = False):
        self.weights = weights
        self.rules: List[Rule] = [r for r in RuleRegistry.rules(self.name)
                                  if r.name not in self.IGNORED_RULES]
        for r in self.rules:
            r.weight = self._w(f"rules.{r.name}", r.weight)

        self.intervals = self._new_interval_model()
        self.evidence = RobustStandardiser(rate=self._w("evidence_rate", 0.01))
        self.beam_size = int(self._w("beam_size", beam_size))
        self.commit_lag = float(self._w("commit_lag_secs", commit_lag_secs))
        self.rng = np.random.default_rng(seed)
        self.stochastic = bool(stochastic)      # legacy path, off by default

        self.t = 0.0
        self.beam: List[Hypothesis] = [Hypothesis(t0=0.0, logp=0.0)]
        self.last_boundary_t = 0.0              # last COMMITTED boundary
        self.n_boundaries = 0
        self.n_revisions = 0                    # re-placements absorbed (v4.9.3)
        self.n_resets = 0
        self.last: BoundaryDecision = BoundaryDecision()

    # weight helper (mirrors PreferenceRuleModule so the EDIT surface works)
    def _w(self, key: str, default: float) -> float:
        return self.weights.get(f"{self.name}.{key}", default) if self.weights else default

    def _new_interval_model(self) -> GammaIntervalModel:
        return GammaIntervalModel(
            prior_mean=self._w("prior_phrase_secs", GammaIntervalModel.PRIOR_MEAN_SECS),
            prior_strength=self._w("prior_strength", GammaIntervalModel.PRIOR_STRENGTH),
            mean_floor=self._w("phrase_floor_secs", GammaIntervalModel.MEAN_FLOOR_SECS),
            hazard_cap_per_sec=self._w("hazard_cap_per_sec", GammaIntervalModel.HAZARD_CAP_PER_SEC))

    def set_weight(self, key: str, value: float) -> None:
        if key.startswith("rules."):
            rn = key.split(".", 1)[1]
            for r in self.rules:
                if r.name == rn:
                    r.weight = float(value)
        elif key == "prior_phrase_secs":
            self.intervals.prior_mean = float(value)
        elif key == "prior_strength":
            self.intervals.n0 = max(float(value), 1e-3)
        elif key == "phrase_floor_secs":
            self.intervals.mean_floor = max(float(value), 0.0)
        elif key == "hazard_cap_per_sec":
            self.intervals.hazard_cap_per_sec = float(np.clip(value, 1e-4, 1.0 - 1e-6))
        elif key == "evidence_rate":
            self.evidence.rate = float(value)
        elif key == "beam_size":
            self.beam_size = max(2, int(value))
        elif key == "commit_lag_secs":
            self.commit_lag = float(value)
        if self.weights:
            self.weights.set(f"{self.name}.{key}", value)

    def reset_learning(self) -> None:
        """Restore the prior live (v4.9.3). Drops every observed interval and
        the evidence null distribution, keeping the current prior parameters
        (including anything edited on the EDIT surface) and the beam, so a
        drifted model is put back without restarting Python."""
        cur = self.intervals
        fresh = self._new_interval_model()
        # keep live edits that bypassed the weight store
        fresh.prior_mean, fresh.prior_shape = cur.prior_mean, cur.prior_shape
        fresh.n0, fresh.mean_floor = cur.n0, cur.mean_floor
        fresh.hazard_cap_per_sec = cur.hazard_cap_per_sec
        self.intervals = fresh
        self.evidence = RobustStandardiser(rate=self.evidence.rate)
        self.n_resets += 1

    # -- scoring -----------------------------------------------------------
    def raw_score(self, evidence: Dict[str, Any]) -> float:
        """Weighted rule sum for the boundary hypothesis. The rules are
        unchanged; there is no `continue` rival to compare against in an event
        model, so the boundary candidate is passed explicitly."""
        cand = {"kind": "boundary"}
        return sum(r.weight * r(evidence, cand) for r in self.rules)

    def _p_boundary(self, tau: float, ev_p: float, dt: float) -> float:
        return float(np.clip(self.intervals.hazard(tau, dt) * ev_p, 1e-9, 1.0 - 1e-9))

    # -- decode ------------------------------------------------------------
    def update(self, evidence: Dict[str, Any], dt: float) -> BoundaryDecision:
        self.t += float(dt)

        s = self.raw_score(evidence)
        self.evidence.update(s)
        z = self.evidence.z(s)
        ev_p = _phi(z)

        # Extend every hypothesis: continue, or spawn a boundary-now child.
        #
        # The per-frame boundary probability is a Bayes update. The hazard is
        # the PRIOR odds that this segment ends now; the evidence contributes a
        # LIKELIHOOD RATIO, ev/(1-ev), which is exactly 1 at the median (z = 0)
        # and so is neutral by construction:
        #
        #     odds(boundary) = h(tau)/(1-h(tau)) * Phi(z)/(1-Phi(z))
        #
        # Two earlier formulations failed here, and both failures are
        # instructive. Using only the duration term (spawn pays log h, continue
        # pays log(1-h)) makes the never-boundary path always cheapest, so a
        # boundary needed ~35 s of elapsed time to pay for itself whatever the
        # evidence said. Adding a per-frame emission on both branches -- spawn
        # log(Phi), continue log(1-Phi) -- overcorrected: a boundary is
        # instantaneous but strong evidence PERSISTS for many frames, so a
        # single loud frame drove log(1-Phi) toward -inf and forced boundaries
        # at ~115/min. The odds form fixes both: evidence tilts the hazard
        # rather than being re-observed as an independent event each frame, and
        # clipping the ratio bounds how far one frame can move the decode.
        eps = 1e-9
        ev_c = float(np.clip(ev_p, 0.02, 0.98))
        lr = float(np.clip(ev_c / (1.0 - ev_c), self.LR_MIN, self.LR_MAX))
        nxt: List[Hypothesis] = []
        best_p = 0.0
        for h in self.beam:
            tau = self.t - h.t0
            hz = float(np.clip(self.intervals.hazard(tau, dt), eps, 1.0 - 1e-6))
            odds = (hz / (1.0 - hz)) * lr
            p = float(np.clip(odds / (1.0 + odds), eps, 1.0 - eps))
            best_p = max(best_p, p)
            nxt.append(Hypothesis(t0=h.t0, logp=h.logp + math.log1p(-p),
                                  parent=h.parent,
                                  depth_at_birth=h.depth_at_birth,
                                  strength_at_birth=h.strength_at_birth))
            strength = float(np.clip(0.5 * ev_p + 0.5 * min(1.0, max(z, 0.0) / 3.0), 0.0, 1.0))
            depth = 2 if (z > 2.0 and ev_p > 0.95) else (1 if z > 1.0 else 0)
            nxt.append(Hypothesis(t0=self.t, logp=h.logp + math.log(p), parent=h,
                                  depth_at_birth=depth, strength_at_birth=strength))

        # Prune: keep the K best paths, and only one hypothesis per boundary
        # time (identical t0 with a worse score can never win later).
        nxt.sort(key=lambda h: -h.logp)
        seen, beam = set(), []
        for h in nxt:
            key = round(h.t0 / 0.25)   # 250 ms resolution: near-identical t0 is the same hypothesis
            if key in seen:
                continue
            seen.add(key)
            beam.append(h)
            if len(beam) >= self.beam_size:
                break
        # Renormalise so logp does not drift to -inf over a long installation.
        top = beam[0].logp
        for h in beam:
            h.logp -= top
        self.beam = beam

        # Commit: walk the best path back to the most recent boundary that is
        # older than commit_lag. Boundaries younger than that are still open to
        # revision by the beam, which is the point of decoding rather than
        # sampling.
        best = beam[0]
        node, committed = best, None
        while node is not None:
            if node.t0 > 0.0 and (self.t - node.t0) >= self.commit_lag:
                committed = node
                break
            node = node.parent
        fired = False
        d = BoundaryDecision(p=best_p, hazard=self.intervals.hazard(self.t - best.t0, dt),
                             evidence=ev_p, z=z, raw_score=s, n_hyp=len(beam))
        if committed is not None and committed.t0 > self.last_boundary_t + 1e-6:
            if self.n_boundaries > 0 and (committed.t0 - self.last_boundary_t) < self.REVISION_WINDOW_S:
                # re-placement of the boundary already committed (see REVISION_WINDOW_S)
                self.last_boundary_t = committed.t0
                self.n_revisions += 1
            else:
                fired = True
                d.fired = True
                d.strength = committed.strength_at_birth
                d.depth = committed.depth_at_birth
                d.committed_lag = self.t - committed.t0
                self.intervals.observe(committed.t0 - self.last_boundary_t)
                self.last_boundary_t = committed.t0
                self.n_boundaries += 1

        self.last = d
        return d

    # -- introspection ------------------------------------------------------
    @property
    def phrase_mean(self) -> float:
        return self.intervals.mean

    def segmentation(self, max_back: int = 8) -> List[float]:
        """Boundary times on the current best path, most recent first."""
        out, node = [], self.beam[0]
        while node is not None and len(out) < max_back:
            if node.t0 > 0.0:
                out.append(round(node.t0, 2))
            node = node.parent
        return out

    def state(self) -> Dict[str, Any]:
        d = self.last
        return {
            "p_boundary": round(d.p, 5),
            "hazard": round(d.hazard, 5),
            "evidence": round(d.evidence, 4),
            "z": round(d.z, 3),
            "raw_score": round(d.raw_score, 3),
            "n_boundaries": self.n_boundaries,
            "n_revisions": self.n_revisions,
            "n_resets": self.n_resets,
            "beam": [{"t0": round(h.t0, 2), "logp": round(h.logp, 2)} for h in self.beam[:4]],
            "segmentation": self.segmentation(),
            "phrase_model": self.intervals.state(),
            "evidence_model": self.evidence.state(),
            "weights": {r.name: r.weight for r in self.rules},
        }
