# preference_rules.py -- Audible Lava / DSO v4.2
#
# v4.9.5: PreferenceRuleModule.update keeps the incumbent in the beam
# (_beam), matches re-proposed candidates to old ones with same_candidate
# (tolerance hook for continuous candidates), enters unseen candidates one
# switching cost below the best path, and renormalises V. See _beam.
# Shared preference-rule machinery (Temperley-style, incremental, real-time).
#
#   Rule                 -- f(evidence, candidate) -> [0,1], with a weight
#   RuleRegistry         -- pluggable rule sets, keyed by module name
#   PreferenceRuleModule -- discrete beam of candidate interpretations scored
#                           left-to-right; committed interpretation chosen by
#                           Viterbi over a small HMM whose hidden states are the
#                           candidates and whose transition penalty is the
#                           switching cost. The Viterbi path score is kept in
#                           the log domain with a forgetting factor so the
#                           analysis lives in a trailing window. The module also
#                           exposes a SOFTMAX POSTERIOR over candidates -- this
#                           posterior is what makes structural changes gradual:
#                           when evidence shifts, mass moves smoothly from the
#                           incumbent to the rival, and consumers that blend by
#                           posterior get a crossfade for free. No explicit
#                           crossfade machinery exists anywhere in the system.
#   ParticleFilterModule -- continuous candidates (meter period/phase, tension
#                           level): particles scored by the same rules, resampled
#                           with jitter. Its posterior is the particle cloud.
#   Weights              -- initial weights from a hand-set JSON, live-editable
#                           (EDIT surface), periodically written back.
#
# All modules are strictly additive: they only read evidence dicts.

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


RuleFn = Callable[[Dict[str, Any], Any], float]


@dataclass
class Rule:
    name: str
    fn: RuleFn
    weight: float = 1.0
    doc: str = ""

    def __call__(self, evidence: Dict[str, Any], candidate: Any) -> float:
        v = float(self.fn(evidence, candidate))
        return 0.0 if math.isnan(v) else max(0.0, min(1.0, v))


class RuleRegistry:
    """Global registry: module name -> list of Rule. New rules plug in with
    @rule("grouping", weight=2.0) without touching the modules."""
    _rules: Dict[str, List[Rule]] = {}

    @classmethod
    def register(cls, module: str, rule: Rule) -> None:
        cls._rules.setdefault(module, []).append(rule)

    @classmethod
    def rules(cls, module: str) -> List[Rule]:
        """Return COPIES of the registered rules.

        Returning the shared objects let every module instance mutate the same
        Rule.weight: two governors (or two tests) would silently share weights,
        and a live EDIT-surface change on one would move the other. The
        registry holds the defaults; instances own their weights.
        """
        return [replace(r) for r in cls._rules.get(module, [])]

    @classmethod
    def modules(cls) -> List[str]:
        return list(cls._rules.keys())


def rule(module: str, weight: float = 1.0, doc: str = ""):
    def deco(fn: RuleFn):
        RuleRegistry.register(module, Rule(fn.__name__, fn, weight, doc or (fn.__doc__ or "").strip()))
        return fn
    return deco


# ── Weight configuration ─────────────────────────────────────────────────────

class WeightStore:
    """Hand-set initial weights (JSON, edited before running) + a runtime
    JSON that mirrors the live state (edited on the EDIT surface, learned
    particle weights, committed candidates) and a plain-text log."""

    def __init__(self, initial_path: str = "governor_weights.json",
                 runtime_path: str = "governor_runtime.json",
                 log_path: str = "governor_log.txt", write_every_s: float = 5.0):
        self.initial_path, self.runtime_path, self.log_path = initial_path, runtime_path, log_path
        self.write_every_s = write_every_s
        self._last_write = 0.0
        self.data: Dict[str, Any] = {}
        self.load()

    def load(self) -> Dict[str, Any]:
        if os.path.exists(self.initial_path):
            with open(self.initial_path) as f:
                self.data = json.load(f)
        else:
            self.data = {}
        return self.data

    def get(self, path: str, default: float) -> float:
        node: Any = self.data
        for k in path.split("."):
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        try:
            return float(node)
        except (TypeError, ValueError):
            return default

    def set(self, path: str, value: float) -> None:
        node = self.data
        keys = path.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = float(value)

    def maybe_write(self, runtime_state: Dict[str, Any], force: bool = False) -> bool:
        now = time.time()
        if not force and now - self._last_write < self.write_every_s:
            return False
        self._last_write = now
        payload = {"written_at": now, "weights": self.data, "runtime": runtime_state}
        tmp = self.runtime_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=1, default=_json_default)
        os.replace(tmp, self.runtime_path)
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"t": now, **_flatten_log(runtime_state)}, default=_json_default) + "\n")
        return True


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _flatten_log(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, (int, float, str)):
                    out[f"{k}.{k2}"] = v2
        elif isinstance(v, (int, float, str)):
            out[k] = v
    return out


# ── Discrete beam module (Viterbi HMM) ───────────────────────────────────────

class PreferenceRuleModule:
    """Base class. Subclasses define `propose(evidence) -> list[candidate]`
    (the beam; usually a fixed small set, possibly re-proposed each frame)
    and register rules under `self.name`.

    Scoring:  s_i(t) = Σ_r w_r · f_r(evidence_t, c_i)            in [0, Σw]
    Viterbi:  V_i(t) = log s_i(t) + max_j [ V_j(t-1) - cost(j→i) ]
              with V decayed by `forget` each frame (trailing window) and
              cost(j→i) = switch_cost if i ≠ j else 0.
    Posterior: softmax(V(t) / temperature) -- smooth, used for blending.
    """

    name: str = "base"
    beam_size: int = 5

    def __init__(self, weights: Optional[WeightStore] = None, switch_cost: float = 2.0,
                 forget: float = 0.97, temperature: float = 1.0):
        self.weights = weights
        self.switch_cost = self._w("switch_cost", switch_cost)
        self.forget = self._w("forget", forget)
        self.temperature = self._w("temperature", temperature)
        self.rules: List[Rule] = RuleRegistry.rules(self.name)
        for r in self.rules:
            r.weight = self._w(f"rules.{r.name}", r.weight)
        self.candidates: List[Any] = []
        self.V: np.ndarray = np.zeros(0)
        self.posterior: np.ndarray = np.zeros(0)
        self.committed_idx: int = -1
        self.committed: Any = None
        self.last_scores: np.ndarray = np.zeros(0)
        self.switches: int = 0

    # weight helper: module.<key> in the JSON
    def _w(self, key: str, default: float) -> float:
        return self.weights.get(f"{self.name}.{key}", default) if self.weights else default

    def set_weight(self, key: str, value: float) -> None:
        """Live adjustment (EDIT surface). key: 'rules.<name>' | 'switch_cost' |
        'forget' | 'temperature'."""
        if key.startswith("rules."):
            rn = key.split(".", 1)[1]
            for r in self.rules:
                if r.name == rn:
                    r.weight = float(value)
        elif hasattr(self, key):
            setattr(self, key, float(value))
        if self.weights:
            self.weights.set(f"{self.name}.{key}", value)

    def propose(self, evidence: Dict[str, Any]) -> List[Any]:
        raise NotImplementedError

    def candidate_key(self, c: Any) -> Any:
        """Identity used to carry Viterbi state across re-proposals."""
        return repr(c)

    def score(self, evidence: Dict[str, Any], c: Any) -> float:
        return sum(r.weight * r(evidence, c) for r in self.rules)

    # v4.9.5: whether a re-proposed candidate that matches the incumbent
    # should keep the incumbent's own values (True: a committed meter period
    # stays put until a genuinely different period wins) or take the fresh
    # proposal's (False: harmony candidates carry fields such as
    # dist_from_current that are only valid relative to the current triad).
    keep_incumbent_value: bool = False

    def same_candidate(self, a: Any, b: Any) -> bool:
        """Whether two candidates are the same hypothesis (so Viterbi state
        carries over). Default: equal candidate_key. Subclasses with
        continuous fields override this with a tolerance."""
        return self.candidate_key(a) == self.candidate_key(b)

    def _beam(self, proposed: List[Any]) -> List[Any]:
        """The candidate set for this frame: proposals deduplicated by
        same_candidate, the incumbent ALWAYS included, truncated to
        beam_size with the incumbent kept.

        v4.9.5. Before this the beam was `propose(evidence)[:beam_size]`.
        Both subclasses rank their proposals by the current frame's score
        and truncate, so the incumbent routinely fell out of the set before
        Viterbi saw it; with no incumbent there is no path to stay on, the
        switching cost never applies, and the decode reduces to argmax of
        the current frame. That is why a run logged ~2.8 meter switches and
        ~1.1 harmony switches per second whatever switch_cost was set to.
        Continuous candidates (meter periods) also re-appeared each frame
        with a slightly different value and therefore a new key, so they
        never inherited Viterbi state either; same_candidate fixes that.
        """
        inc = self.committed
        out: List[Any] = []
        inc_idx = -1
        for c in proposed:
            if inc is not None and inc_idx < 0 and self.same_candidate(c, inc):
                out.append(inc if self.keep_incumbent_value else c)
                inc_idx = len(out) - 1
                continue
            if inc is not None and inc_idx >= 0 and self.same_candidate(c, inc):
                continue
            if any(self.same_candidate(c, d) for d in out):
                continue
            out.append(c)
        if inc is not None and inc_idx < 0:
            out.insert(0, inc)
            inc_idx = 0
        if len(out) > self.beam_size:
            if inc_idx >= 0:
                others = [c for k, c in enumerate(out) if k != inc_idx]
                out = [out[inc_idx]] + others[: self.beam_size - 1]
            else:
                out = out[: self.beam_size]
        return out

    def update(self, evidence: Dict[str, Any]) -> Any:
        new = self._beam(self.propose(evidence))
        if not new:
            return self.committed
        # carry V across re-proposals by candidate identity (with tolerance)
        prev_V = np.empty(len(new))
        prev_max = float(self.V.max()) if len(self.V) else 0.0
        for i, c in enumerate(new):
            v = None
            for oc, ov in zip(self.candidates, self.V):
                if self.same_candidate(c, oc):
                    v = float(ov)
                    break
            # an unseen hypothesis enters one switching cost below the best
            # existing path (it was not the state we were in). The old
            # absolute -switch_cost could exceed the incumbent's path when
            # scores were < 1 (negative logs) and let newcomers in for free.
            prev_V[i] = v if v is not None else prev_max - self.switch_cost
        prev_committed = self.committed

        scores = np.array([self.score(evidence, c) for c in new], dtype=np.float64)
        self.last_scores = scores
        logs = np.log(scores + 1e-6)
        n = len(new)
        V = np.empty(n)
        for i in range(n):
            best = -np.inf
            for j in range(n):
                cost = 0.0 if i == j else self.switch_cost
                best = max(best, prev_V[j] - cost)
            V[i] = logs[i] + best
        V = V * self.forget
        V = V - V.max()                                  # keep V bounded over long runs
        self.V, self.candidates = V, new
        z = V / max(self.temperature, 1e-6)
        z = z - z.max()
        self.posterior = np.exp(z) / np.exp(z).sum()
        idx = int(np.argmax(V))
        if prev_committed is not None and not self.same_candidate(new[idx], prev_committed):
            self.switches += 1
        self.committed_idx, self.committed = idx, new[idx]
        return self.committed

    def blend(self, values: Sequence[float]) -> float:
        """Posterior-weighted blend of a per-candidate scalar -- the organic
        crossfade: no explicit ramp, just probability mass moving."""
        if len(values) != len(self.posterior) or not len(values):
            return 0.0
        return float(np.dot(self.posterior, np.asarray(values, dtype=np.float64)))

    def state(self) -> Dict[str, Any]:
        return {
            "committed": repr(self.committed),
            "posterior": [round(float(p), 4) for p in self.posterior],
            "candidates": [repr(c) for c in self.candidates],
            "switches": self.switches,
            "weights": {r.name: r.weight for r in self.rules},
            "switch_cost": self.switch_cost, "forget": self.forget, "temperature": self.temperature,
        }


# ── Continuous particle-filter module ───────────────────────────────────────

class ParticleFilterModule(PreferenceRuleModule):
    """Continuous candidates. Each particle is a dict of real-valued state
    fields; `jitter` gives per-field perturbation scale; particles are scored
    by the same registered rules, weights normalized, resampled
    (systematic) and jittered. `estimate()` is the weighted mean state;
    `posterior` is the particle weight vector.

    The committed candidate is the weighted-mean particle, so the estimate
    drifts continuously -- there is nothing to snap and nothing to crossfade.
    """

    def __init__(self, n_particles: int = 200, jitter: Optional[Dict[str, float]] = None,
                 bounds: Optional[Dict[str, tuple]] = None, seed: int = 7, **kw):
        super().__init__(**kw)
        self.n = int(self._w("n_particles", n_particles))
        self.jitter = jitter or {}
        self.bounds = bounds or {}
        self.rng = np.random.default_rng(seed)
        self.particles: List[Dict[str, float]] = []
        self.w = np.ones(0)

    def init_particles(self, evidence: Dict[str, Any]) -> List[Dict[str, float]]:
        raise NotImplementedError

    def propose(self, evidence):          # unused; PF has its own update
        return self.particles

    def _clip(self, p: Dict[str, float]) -> Dict[str, float]:
        for k, (lo, hi) in self.bounds.items():
            if k in p:
                p[k] = float(min(hi, max(lo, p[k])))
        return p

    def update(self, evidence: Dict[str, Any]) -> Dict[str, float]:
        if not self.particles:
            self.particles = [self._clip(p) for p in self.init_particles(evidence)][: self.n]
            self.w = np.ones(len(self.particles)) / len(self.particles)
        scores = np.array([self.score(evidence, p) for p in self.particles]) + 1e-6
        w = self.w * scores
        w /= w.sum()
        self.last_scores = scores
        # systematic resampling when effective sample size is low
        ess = 1.0 / float((w ** 2).sum())
        if ess < 0.5 * len(w):
            positions = (self.rng.random() + np.arange(len(w))) / len(w)
            cdf = np.cumsum(w)
            idx = np.searchsorted(cdf, positions)
            self.particles = [dict(self.particles[i]) for i in idx]
            w = np.ones(len(w)) / len(w)
            for p in self.particles:
                for k, s in self.jitter.items():
                    if k in p:
                        p[k] += float(self.rng.normal(0.0, s))
                self._clip(p)
        self.w = w
        self.posterior = w
        self.candidates = self.particles
        self.committed = self.estimate()
        return self.committed

    def estimate(self) -> Dict[str, float]:
        if not self.particles:
            return {}
        keys = self.particles[0].keys()
        return {k: float(sum(wi * p[k] for wi, p in zip(self.w, self.particles))) for k in keys}

    def spread(self) -> Dict[str, float]:
        """Posterior std per field -- the model's own uncertainty."""
        est = self.estimate()
        return {k: float(math.sqrt(sum(wi * (p[k] - est[k]) ** 2 for wi, p in zip(self.w, self.particles))))
                for k in est}

    def state(self) -> Dict[str, Any]:
        return {"estimate": self.estimate(), "spread": self.spread(),
                "n_particles": len(self.particles),
                "weights": {r.name: r.weight for r in self.rules},
                "top_particles": [dict(self.particles[i]) for i in np.argsort(-self.w)[:5]] if len(self.w) else []}


# ── Softmax role weights (no election) ───────────────────────────────────────

def softmax_weights(scores: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    """w_i = exp(s_i / T) / Σ_j exp(s_j / T). T -> 0 approaches hard
    election (argmax); T large -> uniform sharing. All roles are carried
    partially by every contributor; when one contributor's score decays
    (a blob dying) its weight decays with it and the others' rise --
    the handover is the softmax itself."""
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return s
    z = s / max(float(temperature), 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()
