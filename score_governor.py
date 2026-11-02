# score_governor.py -- Audible Lava / DSO v4.2
# Topological score grammar: the governor layer.
#
# Reads (never writes): tracked blobs, the corrected merge tree, the H0
# diagram, tracker lifecycle events. Emits ScoreState per frame and stamps
# per-blob `score_*` attributes that routing.py reads as Sources.
#
# Architecture (one algorithm, several instantiations):
#   SystemStateTracker    (system_state.py)   macrostate, entropies, Markov
#   PreferenceRuleModule  (preference_rules)  Viterbi beam with switch cost
#   ParticleFilterModule  (preference_rules)  continuous candidates
#   ElasticClock          (elastic_clock.py)  projective hierarchical clock
#   TonnetzWalker         (harmony_tonnetz)   harmonic field
#   SchemaMarkov          (schema_lexicon)    schema succession
#
# Governance is a set of CONTINUOUS ROLE WEIGHTS (softmax) carried by every
# blob and by system-level "virtual" contributors -- there is no election,
# no governor class, no handover machinery. When a contributor's role score
# decays, its weight decays with it; the others' weights rise. Structural
# commitments (boundaries, meter, harmonic field) come from Viterbi/particle
# posteriors, so they too move gradually. Crossfades are a byproduct.
#
# All blobs generate AND all blobs govern (guideline 14).

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from system_state import (SystemStateTracker, SystemSummary, BOCPD, permutation_entropy,
                          periodicity_strength, spectral_flatness)
from preference_rules import (PreferenceRuleModule, ParticleFilterModule, WeightStore,
                              RuleRegistry, rule, softmax_weights)
from elastic_clock import ElasticClock, ClockParams
from hazard_grouping import HazardGroupingModule
from harmony_tonnetz import Triad, TonnetzWalker, tonnetz_distance, roughness_proxy, PRIMITIVE_OPS, COMPOUND_OPS
from schema_lexicon import SchemaMarkov, LEXICON, realize, degree_to_freq
from pattern_language import PatternScheduler, Span

try:
    from merge_tree import iter_nodes, assign_depths, tree_stats
except Exception:      # governor importable without the tree module (tests)
    iter_nodes = assign_depths = tree_stats = None


ROLES = ("grouping", "meter", "timespan", "prolongation", "harmony")

# routing.py reads these off blobs via getattr(blob, "score_<name>")
GOVERNOR_SOURCES = ("tension", "prominence_rank", "phrase_position", "beat_phase",
                    "surprisal", "ambiguity", "depth", "role_weight")


@dataclass
class ScoreState:
    """What the governor layer emits each frame (consumed by
    voice_manager._compute_targets, the pattern scheduler, controllers)."""
    t: float = 0.0
    # grouping
    boundary: bool = False
    boundary_strength: float = 0.0
    boundary_depth: int = 0                 # 0 motive, 1 phrase, 2 section
    phrase_position: float = 0.0            # 0 at boundary -> 1 as next is expected
    time_since_boundary: float = 0.0
    boundary_posterior: float = 0.0
    # meter / clock
    period_s: float = 1.0
    beats_per_bar: int = 4
    beat_phase: float = 0.0
    bar_phase: float = 0.0
    elastic: float = 0.5
    meter_source: str = ""
    kuramoto: float = 0.0
    # time-span
    prominence: Dict[int, float] = field(default_factory=dict)      # blob id -> [0,1]
    prominence_rank: Dict[int, float] = field(default_factory=dict) # 1.0 = head
    # Blob ids that carry a SECOND, rhythmically articulated voice an octave
    # below their sustained one (top PAIR_FRACTION by prominence).
    paired_ids:      List[int] = field(default_factory=list)
    density_ceiling: int = 8
    # prolongational
    tension: float = 0.0
    ambiguity: float = 0.0
    surprisal: float = 0.0
    hazard: float = 0.0
    roughness_target: float = 0.0
    # harmony
    triad: str = "C"
    home: str = "C"
    pitch_field: List[int] = field(default_factory=lambda: [0, 4, 7])
    distance_home: int = 0
    last_op: str = ""
    predicted_triad: str = ""
    harmony_posterior_other: float = 0.0    # mass on non-current candidates (pre-sound gain)
    gravitation: float = 0.5                # pull of voices toward chord tones
    register_center: float = 0.5
    # schema
    schema: str = "plateau"
    # roles: blob id (or 'sys:<signal>') -> weight per role
    role_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)
    events: List[str] = field(default_factory=list)
    summary: Optional[SystemSummary] = None


# ── Rules ────────────────────────────────────────────────────────────────────
# Evidence dict keys (all floats unless noted):
#   t, dt, since_boundary, ke_z, occ_z, flux_z, coh_z, pairvar_z, collin_z, com_cross,
#   bocpd_max, bocpd_ke, bocpd_occ, bocpd_flux, bocpd_pairvar, cond_entropy_z, wdiag_z,
#   saddle_strength, blob_event (0/1), n_cues, phrase_mean_len, ...

@rule("grouping", 1.5, "GPR2 analog: long quiet since the last blob/system event, then an event.")
def gap(ev, c):
    if c["kind"] != "boundary":
        return 0.5 if ev["since_event"] < ev["phrase_mean_len"] * 0.5 else 0.2
    return min(1.0, ev["since_boundary"] / max(ev["phrase_mean_len"], 1.0)) * min(1.0, ev["n_cues"] / 1.0)

@rule("grouping", 2.0, "GPR3 analog: abrupt change in an aggregate attribute (z-scored).")
def change(ev, c):
    z = max(ev["ke_z"], ev["occ_z"], ev["flux_z"], ev["coh_z"])
    v = min(1.0, max(0.0, (z - 1.0) / 3.0))
    return v if c["kind"] == "boundary" else 1.0 - v

@rule("grouping", 2.0, "Merge-tree saddle event: strength = absorbed branch persistence (normalized).")
def saddle(ev, c):
    v = min(1.0, ev["saddle_strength"])
    return v if c["kind"] == "boundary" else 1.0 - 0.5 * v

@rule("grouping", 2.5, "BOCPD posterior on any monitored signal (one mechanism, many signals).")
def changepoint(ev, c):
    v = min(1.0, ev["bocpd_max"] * 2.0)
    return v if c["kind"] == "boundary" else 1.0 - v

@rule("grouping", 1.0, "Entropic/regime: conditional-entropy spike; topological: diagram-Wasserstein spike.")
def regime(ev, c):
    z = max(ev["cond_entropy_z"], ev["wdiag_z"])
    v = min(1.0, max(0.0, (z - 1.0) / 3.0))
    return v if c["kind"] == "boundary" else 1.0 - v

@rule("grouping", 1.0, "Relational: pairwise-distance variance shift, collinearity event, COM midline crossing.")
def relational(ev, c):
    v = min(1.0, max(ev["pairvar_z"] - 1.0, 0.0) / 3.0 + 0.6 * ev["com_cross"] + 0.4 * ev["collin_event"])
    return v if c["kind"] == "boundary" else 1.0 - 0.5 * v

@rule("grouping", 1.5, "Coincidence bonus: independent cues aligned within the window.")
def coincidence(ev, c):
    v = min(1.0, max(0.0, (ev["n_cues"] - 1) / 3.0))
    return v if c["kind"] == "boundary" else 0.5

@rule("grouping", 1.5, "SUPERSEDED in v4.4 by the learned Gamma hazard, which represents phrase length "
                       "directly. Left registered for reference; HazardGroupingModule.IGNORED_RULES drops it.")
def regularity(ev, c):
    r = ev["since_boundary"] / max(ev["phrase_mean_len"], 1.0)
    v = min(1.0, r) if r < 1.0 else max(0.3, 1.0 - 0.3 * (r - 1.0))
    return v if c["kind"] == "boundary" else 1.0 - 0.5 * v

# meter (discrete beam of (period, beats_per_bar) hypotheses)
@rule("meter", 2.0, "Periodicity strength of the elected signal at the candidate's period.")
def periodicity(ev, c):
    per = c["period"]
    strengths = ev["period_strength"]           # dict period(s) -> strength
    best = 0.0
    for p, s in strengths.items():
        for m in (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0):
            if abs(math.log2(max(per, 1e-6) / max(p * m, 1e-6))) < 0.12:
                best = max(best, s)
    return best

@rule("meter", 1.0, "London prior: tactus near 0.6-0.8 s preferred.")
def london_prior(ev, c):
    return math.exp(-((math.log(c["period"] / 0.7)) ** 2) / (2 * 0.6 ** 2))

@rule("meter", 1.0, "Blob-count cardinality evidence: n bodies -> n-beat grouping.")
def cardinality(ev, c):
    n = ev["blob_count"]
    return 1.0 if c["beats"] == max(2, min(int(n), 6)) else 0.5

@rule("meter", 1.0, "Period-ratio evidence: a 3:2 ratio between two blobs' cycles favours 3- or 6-beat bars.")
def ratio_evidence(ev, c):
    r = ev["period_ratio"]
    if r <= 0:
        return 0.5
    near32 = abs(r - 1.5) < 0.15 or abs(r - 2 / 3) < 0.1
    return 1.0 if (near32 and c["beats"] in (3, 6)) or (not near32 and c["beats"] in (2, 4)) else 0.4

# meter particle filter (continuous period/phase)
@rule("meter_pf", 2.0, "Event-onset alignment: blob turnaround events near the particle's beat grid.")
def onset_alignment(ev, p):
    if not ev["onsets"]:
        return 0.5
    per, ph = p["period"], p["phase"]
    s = 0.0
    for t in ev["onsets"]:
        x = ((t - ph) / per) % 1.0
        s += 1.0 - min(x, 1.0 - x) * 2.0
    return s / len(ev["onsets"])

@rule("meter_pf", 1.0, "Continuity with the discrete beam's committed period.")
def pf_continuity(ev, p):
    return math.exp(-abs(math.log2(p["period"] / max(ev["beam_period"], 1e-6))) ** 2 / 0.5)

@rule("meter_pf", 0.5, "London prior on the particle period.")
def pf_london(ev, p):
    return math.exp(-((math.log(p["period"] / 0.7)) ** 2) / (2 * 0.6 ** 2))

# harmony (candidates: current triad + Tonnetz neighbours)
@rule("harmony", 2.0, "Parsimony: fewer/smaller voice-leading moves from the current field.")
def parsimony(ev, c):
    return 1.0 / (1.0 + c["dist_from_current"])

@rule("harmony", 2.0, "Step gate: moving is licensed only when the topological change budget is spent.")
def step_gate(ev, c):
    if c["dist_from_current"] == 0:
        return 1.0 - 0.6 * min(1.0, ev["step_budget"])
    return min(1.0, ev["step_budget"])

@rule("harmony", 1.5, "Direction: away from home when leaving equilibrium, toward home when returning.")
def direction(ev, c):
    g = ev["equilibrium_gradient"]      # +1 away, -1 toward
    d = c["dist_home_delta"]            # +1 farther, -1 closer, 0 same
    if abs(g) < 0.15:
        return 0.7 if d == 0 else 0.5
    return 1.0 if (g > 0) == (d > 0) or d == 0 else 0.2

@rule("harmony", 1.0, "Size licensing: compound moves only under high tension / surprisal.")
def size_license(ev, c):
    lic = max(ev["tension"], ev["surprisal_norm"])
    if c["dist_from_current"] >= 2:
        return lic
    return 1.0 - 0.3 * lic

@rule("harmony", 1.0, "Fit to recently sounded pitch content (chord tones present in the texture).")
def content_fit(ev, c):
    pcs = ev["sounding_pcs"]
    if not pcs:
        return 0.6
    hit = sum(1 for pc in pcs if pc in c["pcs"])
    return 0.3 + 0.7 * hit / len(pcs)


# ── Modules ──────────────────────────────────────────────────────────────────

# NOTE (v4.4): the Viterbi GroupingModule is gone. Grouping is an EVENT model
# now -- see hazard_grouping.HazardGroupingModule. A boundary is an event, not
# a state, so a two-state beam with a switching cost was a category error: the
# commit test (argmax == boundary AND posterior > threshold) collapsed to
# posterior > max(0.5, threshold), which made half the threshold's range
# unreachable and produced no boundaries however it was tuned.


class MeterModule(PreferenceRuleModule):
    name = "meter"

    def propose(self, ev):
        cands = []
        for p in ev["candidate_periods"]:
            for beats in (2, 3, 4):
                cands.append({"period": float(p), "beats": beats})
        # keep the current committed one in the beam
        if self.committed is not None and self.committed not in cands:
            cands.append(self.committed)
        # rank by periodicity so the beam holds the most supported first
        cands.sort(key=lambda c: -periodicity(ev, c))
        return cands[: self.beam_size]

    def candidate_key(self, c):
        return (round(c["period"], 2), c["beats"])


class MeterParticleFilter(ParticleFilterModule):
    name = "meter_pf"

    def init_particles(self, ev):
        per = ev.get("beam_period", 0.8)
        return [{"period": float(np.clip(per * 2 ** self.rng.normal(0, 0.3), 0.4, 2.0)),
                 "phase": float(self.rng.random() * 2.0)} for _ in range(self.n)]


class HarmonyModule(PreferenceRuleModule):
    name = "harmony"

    def __init__(self, walker: TonnetzWalker, **kw):
        super().__init__(**kw)
        self.walker = walker

    def propose(self, ev):
        cur = self.walker.current
        d0 = self.walker.distance_home()
        cands = [{"triad": cur, "op": "", "dist_from_current": 0, "dist_home_delta": 0, "pcs": cur.pcs()}]
        ops = list(PRIMITIVE_OPS) + (list(COMPOUND_OPS) if max(ev["tension"], ev["surprisal_norm"]) > 0.5 else [])
        seen = {cur}
        for op in ops:
            t = cur.apply(op)
            if t in seen:
                continue
            seen.add(t)
            cands.append({"triad": t, "op": op, "dist_from_current": len(op),
                          "dist_home_delta": int(np.sign(tonnetz_distance(t, self.walker.home) - d0)),
                          "pcs": t.pcs()})
        cands.sort(key=lambda c: -self.score(ev, c))
        return cands[: self.beam_size]

    def candidate_key(self, c):
        return c["triad"].name


# ── The governor ─────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS: Dict[str, Any] = {
    "_version": 3,
    "_doc": "Hand-set initial weights. Edit before running; live edits from the EDIT surface "
            "(SET_PARAM 'gov.<module>.<key>') are mirrored to governor_runtime.json.",
    "roles": {"temperature": 0.6,
              "virtual_signal_bias": 0.5,
              "init": {"grouping": 1.0, "meter": 1.0, "timespan": 1.0, "prolongation": 1.0, "harmony": 1.0}},
    # v4.4: grouping is an event model. commit_threshold / switch_cost /
    # forget / temperature / the 1.5 s refractory / the regularity weight are
    # all GONE -- the learned Gamma hazard and the robust standardiser replace
    # them. `prior_phrase_secs` is the only musically-opinionated number left
    # and it is a unit-bearing quantity that washes out after ~5 boundaries.
    "grouping": {"prior_phrase_secs": 25.0, "evidence_rate": 0.01,
                 "rules": {"gap": 1.5, "change": 2.0, "saddle": 2.0, "changepoint": 2.5, "regime": 1.0,
                           "relational": 1.0, "coincidence": 1.5}},
    "meter": {"switch_cost": 3.0, "forget": 0.98, "temperature": 1.0, "pf_mix": 0.5,
              "rules": {"periodicity": 2.0, "london_prior": 1.0, "cardinality": 1.0, "ratio_evidence": 1.0}},
    "meter_pf": {"n_particles": 150, "rules": {"onset_alignment": 2.0, "pf_continuity": 1.0, "pf_london": 0.5}},
    "timespan": {"smoothing": 0.9,
                 "rules": {"integrated_persistence": 1.0, "tree_depth": 1.0, "coupling": 0.8,
                           "recurrence": 0.6, "energy": 0.6, "surface": 1.2}},
    "prolongation": {"smoothing": 0.92,
                     "rules": {"surprisal": 1.0, "w_equilibrium": 1.0, "ke": 0.8, "cond_entropy": 0.8}},
    "harmony": {"switch_cost": 2.0, "forget": 0.95, "temperature": 0.8, "step_rate_gain": 4.0,
                "rules": {"parsimony": 2.0, "step_gate": 2.0, "direction": 1.5, "size_license": 1.0,
                          "content_fit": 1.0}},
    "bocpd": {"hazard": 0.0083, "signals": ["ke", "occupancy_entropy", "flux", "pair_dist_var", "coherence"]},
}


def write_default_weights(path: str = "governor_weights.json", overwrite: bool = False) -> str:
    """Write the hand-set defaults if the file is missing or from an older
    schema version (user edits to a current-version file are never touched)."""
    need = overwrite or not os.path.exists(path)
    if not need:
        try:
            with open(path) as f:
                need = int(json.load(f).get("_version", 0)) < int(DEFAULT_WEIGHTS["_version"])
        except Exception:
            need = True
    if need:
        with open(path, "w") as f:
            json.dump(DEFAULT_WEIGHTS, f, indent=1)
    return path


class ScoreGovernor:
    def __init__(self, roi_w: float, roi_h: float, weights_path: str = "governor_weights.json",
                 runtime_path: str = "governor_runtime.json", log_path: str = "governor_log.txt",
                 home: Triad = Triad(0, True), fps: float = 30.0, seed: int = 5):
        write_default_weights(weights_path)
        self.weights = WeightStore(weights_path, runtime_path, log_path)
        self.state_tracker = SystemStateTracker(roi_w, roi_h)
        self.clock = ElasticClock(ClockParams())
        self.walker = TonnetzWalker(home=home)
        self.schema = SchemaMarkov(seed=seed)
        self.grouping = HazardGroupingModule(self.weights)
        self.meter = MeterModule(self.weights)
        self.meter_pf = MeterParticleFilter(n_particles=150,
                                            jitter={"period": 0.02, "phase": 0.03},
                                            bounds={"period": (0.4, 2.0), "phase": (0.0, 4.0)},
                                            weights=self.weights)
        self.harmony = HarmonyModule(self.walker, weights=self.weights)
        self.bocpd: Dict[str, BOCPD] = {}
        for s in self.weights.data.get("bocpd", {}).get("signals", DEFAULT_WEIGHTS["bocpd"]["signals"]):
            self.bocpd[s] = BOCPD(hazard=self.weights.get("bocpd.hazard", 0.0083))
        self.fps = float(fps)
        self.t = 0.0
        self.last_boundary_t = 0.0
        self.boundary_intervals: deque = deque([12.0], maxlen=16)
        self.last_event_t = 0.0
        self.prominence: Dict[int, float] = {}
        self.integrated_persistence: Dict[int, float] = {}
        self.energy_integral: Dict[int, float] = {}
        self.seen_ids: set = set()
        self.tension = 0.0
        self.onsets: deque = deque(maxlen=32)      # blob turnaround times (meter evidence)
        self.prev_direction: Dict[int, str] = {}
        self.roles: Dict[str, Dict[str, float]] = {}
        self.role_scores_ema: Dict[str, Dict[str, float]] = {}
        self.state = ScoreState()
        self.hazard_stats = {"n_events": 0.0, "mean_gap": 40.0}
        self.last_tree_saddles: set = set()
        # ── v4.4: the pattern (hit) layer. Previously PatternScheduler existed
        # but was called from nowhere, so no \mpeHit node was ever spawned and
        # there was no rhythmic layer at all -- the elastic clock computed a
        # tempo that nothing consumed. `pattern_gain` scales hit amplitude;
        # 0 disables the layer entirely.
        self.scheduler = PatternScheduler(osc_client=None, lookahead_s=0.30)
        self.pattern_gain = 0.35
        # ── v4.8 paired articulated voice ─────────────────────────────────
        # At most the top third of blobs by prominence carry a SECOND voice:
        # same note as their sustained one but an octave lower, and rendered
        # entirely through \mpeHit so it is genuinely articulated -- each hit
        # is a self-freeing node with its own duration, so there are real
        # note-offs and real silence between them ("tap, _, tap"), not an
        # amplitude dip in a continuous tone. This moves the head voices
        # toward the pattern-language design in the companion, where a blob
        # OWNS a rhythmic generator rather than having one layered over a
        # drone -- while the sustained voice is left exactly as it was, so
        # the change is additive.
        self.PAIR_FRACTION  = 0.33   # at most this share of blobs are paired
        self.pair_octaves   = 1.0    # how far below the sustained voice
        self.pair_gain      = 0.55   # amplitude relative to pattern_gain
        self.pair_dur_scale = 0.42   # <0.5 leaves audible gaps between hits
        self.pair_slot_base = 10000  # scheduler key offset for the paired layer
        self.pattern_root_pc = 0
        self._pattern_cache: Dict[int, tuple] = {}   # blob id -> (spec, pattern)
        self.sounding_pcs: List[int] = []
        self._ema: Dict[str, Tuple[float, float]] = {}    # signal -> (mean, var) for z-scores
        self._frame = 0
        self.period_every = 6                              # periodicity analyses every N frames
        self._per_cache: Dict[str, Tuple[float, int]] = {}
        self.last_edit: Optional[Tuple[str, float]] = None      # for HUD / runtime JSON

    # -- utilities ----------------------------------------------------------
    def _z(self, name: str, v: float, alpha: float = 0.02, clip_sigma: float = 3.0) -> float:
        """Rolling z-score with winsorized updates: a regime shift keeps
        reading ~clip_sigma until the mean catches up (~1/alpha frames)
        instead of instantly inflating the variance and vanishing."""
        m, var = self._ema.get(name, (v, 1.0))
        sd = math.sqrt(var + 1e-9)
        z = (v - m) / sd
        r = max(-clip_sigma * sd, min(clip_sigma * sd, v - m))
        m2 = m + alpha * r
        var2 = var + alpha * (r * r - var)
        self._ema[name] = (m2, max(var2, 1e-6))
        return float(z)

    def set_sounding_pcs(self, pcs: Sequence[int]) -> None:
        """VoiceManager (or the scheduler) can report currently sounding
        pitch classes for the harmony content-fit rule."""
        self.sounding_pcs = list(pcs)

    # -- role weights (softmax; no election) ---------------------------------
    def _role_scores(self, blobs, summ: SystemSummary, depth_of: Dict[int, int], max_depth: int) -> Dict[str, Dict[str, float]]:
        """Per-contributor raw score for each role. Contributors are blobs
        (by id) plus system-level virtual signals ('sys:ke', ...)."""
        scores: Dict[str, Dict[str, float]] = {r: {} for r in ROLES}
        st = self.state_tracker
        for b in blobs:
            bid = str(b.id)
            d = depth_of.get(b.id, max_depth)
            root_prox = 1.0 - d / max(max_depth, 1)          # 1 near root
            per_s, erratic = self._blob_periodicity(b.id)
            scores["grouping"][bid] = root_prox
            scores["meter"][bid] = per_s
            scores["timespan"][bid] = self.prominence.get(b.id, 0.0)
            scores["prolongation"][bid] = erratic
            scores["harmony"][bid] = 1.0 - min(1.0, float(getattr(b, "speed", 0.0)) / 4.0)   # slowest
        # virtual contributors from the signal bank
        bias = self.weights.get("roles.virtual_signal_bias", 0.5)
        for sig in ("ke", "flux", "occupancy_entropy", "kuramoto"):
            s, flat = self._signal_periodicity(sig)
            if s is not None:
                scores["meter"][f"sys:{sig}"] = bias * s
                scores["prolongation"][f"sys:{sig}"] = bias * flat
        scores["grouping"]["sys:tree"] = bias * (1.0 if max_depth > 1 else 0.3)
        scores["harmony"]["sys:equilibrium"] = bias
        return scores

    def _update_roles(self, scores: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        T = self.weights.get("roles.temperature", 0.6)
        out: Dict[str, Dict[str, float]] = {}
        for role, sc in scores.items():
            ema = self.role_scores_ema.setdefault(role, {})
            for k in list(ema):
                if k not in sc:
                    ema[k] *= 0.9                  # absent contributors decay (softmax handles the rest)
                    if ema[k] < 1e-3:
                        del ema[k]
            for k, v in sc.items():
                ema[k] = 0.85 * ema.get(k, v) + 0.15 * v
            keys = list(ema)
            if keys:
                w = softmax_weights([ema[k] for k in keys], T)
                out[role] = {k: float(wi) for k, wi in zip(keys, w)}
            else:
                out[role] = {}
        self.roles = out
        return out

    def _blob_periodicity(self, bid: int) -> Tuple[float, float]:
        key = f"blob:{bid}"
        if self._frame % self.period_every == 0 or key not in self._per_cache:
            ys = [r[1] for r in self.state_tracker.blob_hist.get(bid, [])][-96:]
            per_s, _ = periodicity_strength(ys) if len(ys) > 12 else (0.0, 0)
            erratic = permutation_entropy(ys[-48:]) if len(ys) > 12 else 0.5
            self._per_cache[key] = (per_s, erratic)
        return self._per_cache[key]

    def _signal_periodicity(self, sig: str):
        key = f"sig:{sig}"
        if self._frame % self.period_every == 0 or key not in self._per_cache:
            x = self.state_tracker.signal(sig)[-128:]
            if len(x) <= 16:
                self._per_cache[key] = (None, None)
            else:
                s, _ = periodicity_strength(x)
                self._per_cache[key] = (s, spectral_flatness(x))
        return self._per_cache[key]

    def _bank_periodicity(self, sig: str):
        key = f"bank:{sig}"
        if self._frame % self.period_every == 0 or key not in self._per_cache:
            x = self.state_tracker.signal(sig)[-256:]
            self._per_cache[key] = periodicity_strength(x) if len(x) > 24 else None
        return self._per_cache[key]

    def blend_by_role(self, role: str, values: Dict[str, float], default: float = 0.0) -> float:
        w = self.roles.get(role, {})
        tot = sum(w.get(k, 0.0) for k in values)
        if tot <= 0:
            return default
        return sum(w.get(k, 0.0) * v for k, v in values.items()) / tot

    # -- time-span prominence (a)-(f) ----------------------------------------
    def _prominence(self, blobs, depth_of: Dict[int, int], max_depth: int, dt: float,
                    st: SystemStateTracker) -> Dict[int, float]:
        wr = lambda k, d: self.weights.get(f"timespan.rules.{k}", d)
        smooth = self.weights.get("timespan.smoothing", 0.9)
        pmax = max([float(getattr(b, "persistence", 0.0)) for b in blobs] + [1.0])
        out = {}
        for b in blobs:
            self.integrated_persistence[b.id] = self.integrated_persistence.get(b.id, 0.0) + float(b.persistence) * dt
            sp = float(getattr(b, "speed", 0.0))
            self.energy_integral[b.id] = self.energy_integral.get(b.id, 0.0) + 0.5 * float(b.area) * sp * sp * dt
        ip_max = max(list(self.integrated_persistence.values()) + [1.0])
        en_max = max(list(self.energy_integral.values()) + [1.0])
        # (c) coupling: correlation of vy with the aggregate flux
        flux = np.array(st.signal("flux")[-64:])
        for b in blobs:
            a = min(1.0, self.integrated_persistence[b.id] / ip_max)
            d = 1.0 - depth_of.get(b.id, max_depth) / max(max_depth, 1)          # (b) depth: near root = head
            hist = [r[2] for r in st.blob_hist.get(b.id, [])][-64:]
            c = 0.0
            if len(hist) >= 16 and len(flux) >= len(hist):
                x = np.array(hist); y = flux[-len(hist):]
                if x.std() > 1e-9 and y.std() > 1e-9:
                    c = abs(float(np.corrcoef(x, y)[0, 1]))
            r = 1.0 if b.id in self.seen_ids and getattr(b, "age", 99) < 5 else 0.0             # (d) recurrence: re-identified
            e = min(1.0, self.energy_integral[b.id] / en_max)                   # (e) energy
            on_beat = 1.0 if self.clock.is_strong_beat(self.t) else 0.0         # (f) musical surface
            in_field = 0.0
            f_hz = getattr(b, "score_freq_hz", None)
            if f_hz:
                pc = int(round(69 + 12 * math.log2(f_hz / 440.0))) % 12
                in_field = 1.0 if pc in self.state.pitch_field else 0.0
            raw = (wr("integrated_persistence", 1.0) * a + wr("tree_depth", 1.0) * d + wr("coupling", 0.8) * c
                   + wr("recurrence", 0.6) * r + wr("energy", 0.6) * e + wr("surface", 1.2) * 0.5 * (on_beat + in_field)
                   + 0.5 * float(b.persistence) / pmax)
            prev = self.prominence.get(b.id, raw)
            out[b.id] = smooth * prev + (1 - smooth) * raw
            self.seen_ids.add(b.id)
        self.prominence = out
        return out

    # -- merge tree consumption -----------------------------------------------
    def _tree_features(self, tree_root, leaves, blobs) -> Tuple[Dict[int, int], int, float, dict]:
        depth_of: Dict[int, int] = {}
        max_depth, saddle_strength, stats = 0, 0.0, {}
        if tree_root is None or tree_stats is None:
            return depth_of, max_depth, saddle_strength, stats
        stats = tree_stats(tree_root)
        max_depth = int(stats.get("max_depth", 0))
        # leaf -> nearest blob (within 30 px)
        for leaf in leaves or []:
            py, px = leaf.pixel
            best = min(blobs, key=lambda b: (b.centroid[0] - px) ** 2 + (b.centroid[1] - py) ** 2, default=None)
            if best is not None and math.hypot(best.centroid[0] - px, best.centroid[1] - py) < 30:
                depth_of[best.id] = int(leaf.depth)
        # saddle events: new internal nodes (by rounded pixel+threshold) since last frame
        cur = set()
        for n in iter_nodes(tree_root):
            if not n.is_leaf:
                cur.add((n.pixel[0] // 8, n.pixel[1] // 8, int(n.threshold)))
                if len(n.children) == 2:
                    absorbed = min(n.children, key=lambda c: c.threshold)
                    key = (n.pixel[0] // 8, n.pixel[1] // 8, int(n.threshold))
                    if key not in self.last_tree_saddles:
                        pers = float(absorbed.persistence) if absorbed.is_leaf else float(absorbed.threshold - n.threshold)
                        saddle_strength = max(saddle_strength, min(1.0, pers / 60.0))
        self.last_tree_saddles = cur
        return depth_of, max_depth, saddle_strength, stats

    # -- main per-frame update -------------------------------------------------
    def update(self, blobs, dt: float, tracker_events: Sequence[tuple] = (),
               tree_root=None, leaves=None, diagram=None) -> ScoreState:
        blobs = list(blobs)
        self.t += dt
        self._frame += 1
        s = ScoreState(t=self.t, home=self.walker.home.name)
        st = self.state_tracker

        depth_of, max_depth, saddle_strength, tstats = self._tree_features(tree_root, leaves, blobs)
        summ = st.update(blobs, dt, diagram=diagram, tree_stats=tstats)
        s.summary = summ
        s.kuramoto = summ.extra.get("rel_kuramoto", 0.0)

        # events
        ev_names = [e[0] for e in tracker_events]
        for b in blobs:                                 # turnaround onsets for meter evidence
            d = getattr(b, "direction", None)
            if self.prev_direction.get(b.id) not in (None, d):
                self.onsets.append(self.t)
            self.prev_direction[b.id] = d
        if ev_names:
            self.last_event_t = self.t
            self.hazard_stats["n_events"] += len(ev_names)
        s.events = list(ev_names)

        # z-scores and changepoints
        ke_log = math.log1p(summ.ke_total)               # scale-free activity level
        ke_z = self._z("ke", ke_log)
        occ_z = self._z("occ", summ.occupancy_entropy)
        flux_z = self._z("flux", summ.extra.get("agg_flux", 0.0))
        coh_z = self._z("coh", summ.coherence)
        pairvar_z = self._z("pairvar", summ.extra.get("rel_pair_dist_var", 0.0))
        ce_z = self._z("ce", summ.cond_entropy)
        wd_z = self._z("wdiag", summ.w_diagram)
        collin = summ.extra.get("rel_collinearity", 0.0)
        collin_event = 1.0 if collin > 0.9 and self._z("collin", collin) > 1.5 else 0.0
        com_y = summ.extra.get("rel_com_y", 0.0)
        prev_com = self._ema.get("com_prev", (com_y, 0.0))[0]
        com_cross = 1.0 if (prev_com - st.roi_h / 2) * (com_y - st.roi_h / 2) < 0 else 0.0
        self._ema["com_prev"] = (com_y, 0.0)
        bocpd_p = {}
        sig_vals = {"ke": ke_log, "occupancy_entropy": summ.occupancy_entropy,
                    "flux": summ.extra.get("agg_flux", 0.0), "pair_dist_var": summ.extra.get("rel_pair_dist_var", 0.0),
                    "coherence": summ.coherence}
        for name, det in self.bocpd.items():
            if name in sig_vals:
                bocpd_p[name] = det.update(sig_vals[name])
        bocpd_max = max(bocpd_p.values()) if bocpd_p else 0.0

        cues = [abs(ke_z) > 2, abs(occ_z) > 2, abs(flux_z) > 2, bocpd_max > 0.3, saddle_strength > 0.2,
                bool(ev_names), ce_z > 2, wd_z > 2, com_cross > 0, collin_event > 0]
        n_cues = float(sum(cues))

        # -- grouping (v4.4 hazard x evidence event model) ------------------------
        gev = {"t": self.t, "dt": dt, "since_boundary": self.t - self.last_boundary_t,
               "since_event": self.t - self.last_event_t, "phrase_mean_len": self.grouping.phrase_mean,
               "ke_z": abs(ke_z), "occ_z": abs(occ_z), "flux_z": abs(flux_z), "coh_z": abs(coh_z),
               "pairvar_z": abs(pairvar_z), "collin_event": collin_event, "com_cross": com_cross,
               "bocpd_max": bocpd_max, "cond_entropy_z": ce_z, "wdiag_z": wd_z,
               "saddle_strength": saddle_strength, "n_cues": n_cues}
        d = self.grouping.update(gev, dt)
        # `boundary_posterior` now reports the EVIDENCE probability Phi(z) --
        # the per-frame P is tiny by construction (a phrase is hundreds of
        # frames) and would be useless on a HUD. Evidence answers the question
        # a listener has: "does it feel like something is happening?"
        s.boundary_posterior = d.evidence
        if d.fired:
            s.boundary = True
            s.boundary_strength = d.strength
            s.boundary_depth = d.depth
            self.boundary_intervals.append(self.t - self.last_boundary_t)
            self.last_boundary_t = self.t
            self.schema.bias_from_state(self.tension, summ.cond_entropy, ev_names + ["boundary"])
            s.schema = self.schema.next()
        else:
            s.schema = self.schema.current
        s.time_since_boundary = self.t - self.last_boundary_t
        s.phrase_position = float(min(1.0, s.time_since_boundary / max(self.grouping.phrase_mean, 1e-6)))

        # -- meter (signal election bank + discrete beam + particle filter) ----------
        bank: Dict[str, Tuple[float, int]] = {}
        for sig in ("ke", "flux", "occupancy_entropy", "kuramoto", "com_y"):
            r = self._bank_periodicity(sig)
            if r is not None:
                bank[sig] = r
        periods = {}
        for b in blobs:
            ph, per = st.blob_phase_and_period(b.id)
            if per > 0:
                periods[b.id] = per
        per_ratio = 0.0
        if len(periods) >= 2:
            v = sorted(periods.values())
            per_ratio = v[-1] / max(v[0], 1e-6)
        period_strength = {}
        best_sig, best_s = "", 0.0
        for sig, (strength, lag) in bank.items():
            per_s = lag / self.fps
            # fold the lamp's slow cycle into the tactus range by octave division
            while per_s > self.clock.TACTUS_MAX:
                per_s /= 2.0
            if per_s >= self.clock.TACTUS_MIN:
                w_sig = self.roles.get("meter", {}).get(f"sys:{sig}", 0.2)
                period_strength[round(per_s, 3)] = max(period_strength.get(round(per_s, 3), 0.0), strength * (0.5 + w_sig))
                if strength > best_s:
                    best_sig, best_s = sig, strength
        for bid, per in periods.items():
            p = per
            while p > self.clock.TACTUS_MAX:
                p /= 2.0
            if p >= self.clock.TACTUS_MIN:
                w_b = self.roles.get("meter", {}).get(str(bid), 0.2)
                period_strength[round(p, 3)] = max(period_strength.get(round(p, 3), 0.0), 0.6 * (0.5 + w_b))
        cand_periods = sorted(period_strength, key=lambda p: -period_strength[p])[:3] or [self.clock.p.period_s]
        mev = {"candidate_periods": cand_periods, "period_strength": period_strength,
               "blob_count": summ.blob_count, "period_ratio": per_ratio}
        mc = self.meter.update(mev)
        beam_period = mc["period"] if mc else self.clock.p.period_s
        pev = {"onsets": [t for t in self.onsets if self.t - t < 8.0], "beam_period": beam_period}
        est = self.meter_pf.update(pev)
        mix = self.weights.get("meter.pf_mix", 0.5)
        period = (1 - mix) * beam_period + mix * est.get("period", beam_period)
        self.clock.set_period(period, self.t)
        self.clock.set_meter(mc["beats"] if mc else 4)
        self.clock.set_elastic(0.15 + 0.85 * s.kuramoto)
        s.period_s, s.beats_per_bar = self.clock.p.period_s, self.clock.p.beats_per_bar
        s.beat_phase, s.bar_phase, s.elastic = self.clock.beat_phase(self.t), self.clock.bar_phase(self.t), self.clock.p.elastic
        s.meter_source = best_sig or "blobs"

        # -- prolongational: tension composite --------------------------------------
        wp = lambda k, d: self.weights.get(f"prolongation.rules.{k}", d)
        surpr_norm = min(1.0, summ.surprisal / 6.0)
        raw_t = (wp("surprisal", 1.0) * surpr_norm + wp("w_equilibrium", 1.0) * min(1.0, summ.w_equilibrium * 2.0)
                 + wp("ke", 0.8) * min(1.0, summ.ke_total / st.ke_ref)
                 + wp("cond_entropy", 0.8)
                 * self.state_tracker.markov.normalised_conditional_entropy(summ.symbol))
        raw_t /= max(wp("surprisal", 1.0) + wp("w_equilibrium", 1.0) + wp("ke", 0.8) + wp("cond_entropy", 0.8), 1e-6)
        sm = self.weights.get("prolongation.smoothing", 0.92)
        self.tension = sm * self.tension + (1 - sm) * raw_t
        s.tension, s.surprisal = float(self.tension), float(summ.surprisal)
        # v4.6: normalise by the OBSERVED alphabet rather than a fixed 3.0.
        # The old divisor saturated at 1.0 on every frame, freezing chord
        # cardinality, density ceiling, gravitation and one tension term.
        s.ambiguity = self.state_tracker.markov.normalised_conditional_entropy(summ.symbol)
        # hazard: rising probability of a lifecycle event as the mean gap is approached
        gap = self.t - self.last_event_t
        self.hazard_stats["mean_gap"] = 0.995 * self.hazard_stats["mean_gap"] + 0.005 * gap
        s.hazard = float(min(1.0, gap / max(self.hazard_stats["mean_gap"], 1.0)))

        # -- harmony --------------------------------------------------------------
        w_eq_hist = st.signal("w_equilibrium")[-30:]
        grad = 0.0
        if len(w_eq_hist) >= 10:
            grad = float(np.clip((np.mean(w_eq_hist[-5:]) - np.mean(w_eq_hist[:5])) * 20.0, -1, 1))
        self.walker.budget += summ.w_diagram * self.weights.get("harmony.step_rate_gain", 4.0) * dt
        hev = {"step_budget": self.walker.budget, "equilibrium_gradient": grad, "tension": s.tension,
               "surprisal_norm": surpr_norm, "sounding_pcs": self.sounding_pcs}
        hc = self.harmony.update(hev)
        if hc and hc["triad"] != self.walker.current:
            self.walker.current, self.walker.last_op = hc["triad"], hc["op"]
            self.walker.history.append((hc["triad"], hc["op"]))
            self.walker.budget = 0.0
        self.walker.predicted_next = self.walker.predict(max(s.tension, surpr_norm), grad)
        s.triad, s.distance_home, s.last_op = self.walker.current.name, self.walker.distance_home(), self.walker.last_op
        s.pitch_field = self.walker.pitch_field(s.ambiguity)
        s.predicted_triad = self.walker.predicted_next.name if self.walker.predicted_next else ""
        s.harmony_posterior_other = float(1.0 - self.harmony.posterior[self.harmony.committed_idx]) if len(self.harmony.posterior) else 0.0
        s.roughness_target = float(min(1.0, 0.6 * roughness_proxy(s.pitch_field) + 0.4 * s.tension))
        s.gravitation = float(0.3 + 0.6 * (1.0 - s.ambiguity))
        s.register_center = float(1.0 - summ.extra.get("agg_com_h", st.roi_h / 2) / max(st.roi_h, 1.0))

        # -- time-span prominence & roles ----------------------------------------
        prom = self._prominence(blobs, depth_of, max_depth, dt, st)
        s.prominence = prom
        if prom:
            order = sorted(prom, key=lambda k: -prom[k])
            n = len(order)
            s.prominence_rank = {bid: 1.0 - i / max(n - 1, 1) for i, bid in enumerate(order)}
        # Top PAIR_FRACTION by prominence get the articulated second voice.
        # ceil() so a single blob still gets one -- with 2-3 blobs a strict
        # floor would leave the feature silent exactly when it matters most.
        if prom:
            order = sorted(prom, key=lambda k: -prom[k])
            n_pair = max(1, int(math.ceil(self.PAIR_FRACTION * len(order))))
            s.paired_ids = order[:n_pair]
        s.density_ceiling = int(max(2, min(16, 4 + 8 * (1 - s.ambiguity) + 4 * s.tension)))
        self._update_roles(self._role_scores(blobs, summ, depth_of, max_depth))
        s.role_weights = self.roles

        # -- stamp per-blob routing sources ---------------------------------------
        for b in blobs:
            b.score_tension = s.tension
            b.score_prominence_rank = s.prominence_rank.get(b.id, 0.0)
            b.score_phrase_position = s.phrase_position
            b.score_beat_phase = s.beat_phase
            b.score_surprisal = surpr_norm
            b.score_ambiguity = s.ambiguity
            b.score_depth = depth_of.get(b.id, max_depth) / max(max_depth, 1) if max_depth else 0.0
            b.score_role_weight = max(self.roles.get(r, {}).get(str(b.id), 0.0) for r in ROLES) if self.roles else 0.0

        self.state = s
        self.weights.maybe_write(self.runtime_state())
        return s

    # -- controller / EDIT surface -------------------------------------------------
    # module name -> live object (these hold their own copies of the weights;
    # everything else lives only in the WeightStore and is read via .get())
    def _module(self, name: str):
        return {"grouping": self.grouping, "meter": self.meter,
                "meter_pf": self.meter_pf, "harmony": self.harmony}.get(name)

    def default_weight(self, path: str) -> float:
        node: Any = DEFAULT_WEIGHTS
        for k in path.split("."):
            if not isinstance(node, dict) or k not in node:
                return 0.0
            node = node[k]
        try:
            return float(node)
        except (TypeError, ValueError):
            return 0.0

    def get_weight(self, path: str) -> float:
        """Current value of a dotted weight path (live module value wins)."""
        mod, _, key = path.partition(".")
        target = self._module(mod)
        if target is not None and key:
            if key.startswith("rules."):
                rn = key.split(".", 1)[1]
                for r in target.rules:
                    if r.name == rn:
                        return float(r.weight)
            elif hasattr(target, key):
                return float(getattr(target, key))
        return self.weights.get(path, self.default_weight(path))

    # bounds so a runaway EDIT press cannot wedge a module
    WEIGHT_BOUNDS = {"prior_phrase_secs": (3.0, 300.0), "evidence_rate": (0.001, 0.2),
                     "switch_cost": (0.0, 10.0),
                     "forget": (0.5, 1.0), "temperature": (0.05, 5.0), "pf_mix": (0.0, 1.0),
                     "smoothing": (0.0, 0.999), "virtual_signal_bias": (0.0, 3.0),
                     "step_rate_gain": (0.0, 40.0), "n_particles": (16.0, 1000.0)}
    RULE_BOUNDS = (0.0, 10.0)

    def handle_controller_event(self, ev) -> bool:
        """Consume EDIT-surface events addressed to the governor.

        ControllerEvent(SET_PARAM, param='gov.<module>.<key>', value=v):
          * absolute by default
          * a DELTA when extras['relative'] is set (the keyboard's arrow keys)
          * snap back to the hand-set default when extras['reset'] is set
        Returns True if consumed, so the event never reaches VoiceManager.
        """
        p = getattr(ev, "param", None)
        if not p or not str(p).startswith("gov."):
            return False
        path = str(p)[4:]
        mod, _, key = path.partition(".")
        extras = getattr(ev, "extras", None) or {}

        if extras.get("reset"):
            value = self.default_weight(path)
        else:
            v = getattr(ev, "value", None)
            if v is None:
                return True                       # addressed to us, no-op
            value = self.get_weight(path) + float(v) if extras.get("relative") else float(v)

        leaf = key.split(".")[-1] if key else ""
        lo, hi = (self.RULE_BOUNDS if key.startswith("rules.")
                  else self.WEIGHT_BOUNDS.get(leaf, (-1e6, 1e6)))
        value = max(lo, min(hi, value))

        target = self._module(mod)
        if target is not None and key:
            target.set_weight(key, value)         # live module (+ WeightStore)
        self.weights.set(path, value)             # always mirror into the store
        self.last_edit = (path, value)
        self.weights.maybe_write(self.runtime_state(), force=True)
        return True

    def runtime_state(self) -> Dict[str, Any]:
        s = self.state
        return {
            "t": s.t, "tension": s.tension, "ambiguity": s.ambiguity, "surprisal": s.surprisal,
            "triad": s.triad, "distance_home": s.distance_home, "schema": s.schema,
            "period_s": s.period_s, "beats_per_bar": s.beats_per_bar, "elastic": s.elastic,
            "meter_source": s.meter_source, "boundary_posterior": s.boundary_posterior,
            "grouping": self.grouping.state(), "meter": self.meter.state(),
            "meter_pf": self.meter_pf.state(), "harmony": self.harmony.state(),
            "roles": {r: {k: round(v, 3) for k, v in w.items()} for r, w in self.roles.items()},
            "last_edit": (f"{self.last_edit[0]} = {self.last_edit[1]:.3f}" if self.last_edit else None),
            "prominence": {str(k): round(v, 3) for k, v in s.prominence.items()},
        }

    # -- pattern realization for a blob --------------------------------------------
    def pattern_for_blob(self, blob, s: Optional[ScoreState] = None):
        """Blob-owned rhythmic pattern: E(k, n) parameterized by blob
        properties, transformed by governor state (a serializable spec)."""
        from pattern_language import euclid, apply_spec
        s = s or self.state
        # Normalised against RUNNING maxima, not fixed constants: a fixed
        # 20000 px divisor put typical 3000 px blobs at area_n = 0.15, i.e.
        # 2 hits per bar however dense the lamp was.
        self._area_ref = max(getattr(self, "_area_ref", 1.0) * 0.999,
                             float(getattr(blob, "area", 0.0)), 1500.0)
        self._pers_ref = max(getattr(self, "_pers_ref", 1.0) * 0.999,
                             float(getattr(blob, "persistence", 0.0)), 20.0)
        area_n = min(1.0, float(getattr(blob, "area", 0.0)) / self._area_ref)
        pers_n = min(1.0, float(getattr(blob, "persistence", 0.0)) / self._pers_ref)
        n = int(4 + round(pers_n * 8))                       # 4..12 slots
        k = int(max(2, round(0.25 * n + area_n * 0.6 * n)))   # 2..~0.85n hits
        rot = int(round(s.beat_phase * n))
        # Degrees are drawn from the current pitch field so hits are in key.
        degrees = [1, 3, 5, 8][: max(2, 1 + int(3 * s.tension))]
        spec = [("degrade", max(0.0, 0.55 - 0.55 * s.tension), blob.id),
                ("arp", degrees),
                ("velocity", 0.55 + 0.45 * s.prominence_rank.get(blob.id, 0.5))]
        if s.boundary:
            spec.append(("ply", 2))
        if s.tension > 0.7:
            spec.append(("fast", 2))
        return apply_spec(euclid(k, n, rot), spec), spec

    def tick_patterns(self, blobs, voice_of=None, now_s: Optional[float] = None) -> list:
        """Render the pattern layer for the look-ahead window and return the
        OSC messages to send. Called once per frame from the main loop.

        Each blob owns a Euclidean pattern parameterised by its own geometry
        and transformed by governor state; events are placed against the
        ElasticClock, so timing is pulled toward the grid by the Kuramoto
        strength rather than hard-quantised.

        `voice_of(blob_id) -> slot | None` lets the caller suppress hits for
        blobs with no active voice. Returns [(address, args)] for the caller
        to send; the governor does not own an OSC client.
        """
        if self.pattern_gain <= 0.0:
            return []
        now = self.t if now_s is None else float(now_s)
        s = self.state
        root_pc = s.pitch_field[0] if s.pitch_field else self.pattern_root_pc
        out = []
        for b in blobs:
            slot = voice_of(b.id) if voice_of else 0
            if slot is None:
                continue
            pat, spec = self.pattern_for_blob(b, s)
            rank = s.prominence_rank.get(b.id, 0.5)
            # Ornamental voices (low rank) get the busier hit layer; heads keep
            # the sustained voice, which is the Schenkerian reading made audible.
            amp = self.pattern_gain * (0.45 + 0.55 * (1.0 - rank))
            msgs = self.scheduler.render(
                slot, pat, self.clock, now,
                degree_to_freq=lambda d, _r=root_pc: degree_to_freq(d if d else 1, _r, 48),
                amp=amp,
                extra_params={"cutoff": 900.0 + 3500.0 * s.tension,
                              "drive": 0.1 + 0.5 * s.tension,
                              "noise": 0.03 + 0.25 * s.ambiguity,
                              "fmode": 0.15 + 0.5 * s.roughness_target},
            )
            out.extend(msgs)

            # ── paired articulated voice (top PAIR_FRACTION only) ─────────
            if b.id in s.paired_ids:
                # Same note as the sustained voice, an octave lower. The
                # sustained pitch is read back from `score_freq_hz`, which
                # voice_manager._compute_targets stamps after applying
                # harmonic gravitation -- so the pair tracks the sustained
                # voice exactly, including its pull toward the pitch field,
                # rather than re-deriving a pitch that could disagree with it.
                f_sus = float(getattr(b, "score_freq_hz", 0.0) or 0.0)
                if f_sus <= 0.0:
                    f_sus = degree_to_freq(1, root_pc, 60)
                f_pair = max(20.0, f_sus / (2.0 ** self.pair_octaves))
                ppat, _ = self.paired_pattern_for_blob(b, s)
                out.extend(self.scheduler.render(
                    self.pair_slot_base + int(slot), ppat, self.clock, now,
                    degree_to_freq=lambda _d, _f=f_pair: _f,
                    amp=self.pattern_gain * self.pair_gain,
                    dur_scale=self.pair_dur_scale,
                    extra_params={"cutoff": 600.0 + 2200.0 * s.tension,
                                  "drive": 0.15 + 0.45 * s.tension,
                                  "noise": 0.02 + 0.10 * s.ambiguity,
                                  "fmode": 0.0,
                                  "detune": 2.0 * s.roughness_target},
                ))
        return out

    def paired_pattern_for_blob(self, blob, s: Optional[ScoreState] = None):
        """Sparse, clearly articulated pattern for a blob's second voice.

        Deliberately much sparser than the ornamental hit layer: this voice
        carries the blob's OWN note, so it should read as an articulation of
        that pitch rather than as a texture. Density follows the metrical
        cardinality rather than blob area, so the paired voices land on the
        beat structure and the ensemble locks together.
        """
        from pattern_language import euclid, apply_spec
        s = s or self.state
        beats = max(2, int(s.beats_per_bar))
        # Two slots per beat, half of them struck: at rest this is exactly
        # E(beats, 2*beats) = alternating slots, i.e. "tap, rest, tap, rest".
        # Tension fills in the gaps rather than subdividing further, so the
        # articulation gets busier without the tactus changing.
        n = beats * 2
        k = max(1, int(round(n * (0.5 + 0.25 * s.tension))))
        rot = int(round(s.phrase_position * n)) % max(n, 1)
        spec = [("degrade", max(0.0, 0.30 - 0.30 * s.tension), 7919 + blob.id)]
        if s.boundary and s.boundary_strength > 0.6:
            spec.append(("ply", 2))
        return apply_spec(euclid(k, n, rot), spec), spec

    def schema_patterns(self, s: Optional[ScoreState] = None):
        s = s or self.state
        return realize(LEXICON[s.schema], beats_per_bar=s.beats_per_bar, ornament=s.tension)
