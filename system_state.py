# system_state.py -- Audible Lava / DSO v4.2
# Stochastic / entropic system state (read-only over tracker + topology).
#
# Provides:
#   * SystemSummary            -- the single-instant macrostate (dataclass with
#                                 room to grow via `extra`)
#   * entropy utilities        -- shannon, permutation entropy (ordinal), persistence
#                                 entropy, occupancy-grid entropy
#   * distances                -- 1-D EMD, marginal-EMD for grids, 2-Wasserstein
#                                 between persistence diagrams (Hungarian w/ diagonal)
#   * BOCPD                    -- Bayesian Online Changepoint Detection (Adams &
#                                 MacKay 2007, Gaussian mean-unknown / Student-t
#                                 predictive), one instance per monitored signal
#   * MarkovExpectation        -- symbolized-macrostate transition model giving
#                                 conditional entropy per state, entropy rate
#                                 (global unpredictability), surprisal, predicted
#                                 next-state distribution and its entropy
#   * SystemStateTracker       -- per-frame pipeline: per-blob kinematics ->
#                                 relational -> aggregate physical -> topological
#                                 -> distributional -> SystemSummary -> symbol ->
#                                 Markov update; history ring buffer; distance
#                                 between summaries (Euclidean on scalars + EMD on
#                                 distributional components) for boundary
#                                 evidence and form-level recurrence.
#
# Nothing here mutates tracker / detection / audio state.

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import gammaln

try:                                   # optional exact diagram Wasserstein (needs POT)
    from gudhi.wasserstein import wasserstein_distance as _gudhi_wasserstein
except Exception:
    _gudhi_wasserstein = None
try:
    from scipy.spatial import ConvexHull as _ConvexHull
except Exception:
    _ConvexHull = None


# ── Entropies ────────────────────────────────────────────────────────────────

def shannon_entropy(p: Sequence[float], base: float = 2.0) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log(p)).sum() / math.log(base))


def permutation_entropy(x: Sequence[float], order: int = 3, delay: int = 1,
                        normalize: bool = True) -> float:
    """Bandt & Pompe ordinal-pattern entropy of a time series.

    Model-free, robust to monotone distortion, meaningful on windows as short
    as ~30 samples. Returns bits (normalized to [0,1] by log2(order!) when
    `normalize`). Low = regular/predictable; high = disordered.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x) - (order - 1) * delay
    if n < 2:
        return 0.0
    idx = np.arange(order) * delay
    patterns: Dict[Tuple[int, ...], int] = {}
    for i in range(n):
        w = x[i + idx]
        key = tuple(np.argsort(w, kind="stable"))
        patterns[key] = patterns.get(key, 0) + 1
    h = shannon_entropy(list(patterns.values()))
    if normalize:
        h /= math.log2(math.factorial(order))
    return float(min(1.0, h))


def persistence_entropy(diagram: Sequence[Tuple[float, float]],
                        normalize: bool = True) -> float:
    """Shannon entropy of normalized lifetimes. Low when one feature
    dominates, high when significance is evenly spread."""
    life = np.array([b - d for b, d in diagram if np.isfinite(d)], dtype=np.float64)
    life = life[life > 0]
    if life.size <= 1:
        return 0.0
    h = shannon_entropy(life)
    return float(h / math.log2(life.size)) if normalize else float(h)


def occupancy_entropy(grid: np.ndarray, normalize: bool = True) -> float:
    g = np.asarray(grid, dtype=np.float64).ravel()
    if g.sum() <= 0:
        return 0.0
    h = shannon_entropy(g)
    return float(h / math.log2(g.size)) if normalize else float(h)


# ── Distances ────────────────────────────────────────────────────────────────

def emd_1d(p: Sequence[float], q: Sequence[float]) -> float:
    """Earth mover's distance between two 1-D histograms on the same bins
    (unit spacing): integral of |CDF_p - CDF_q|."""
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    p = p / p.sum(); q = q / q.sum()
    return float(np.abs(np.cumsum(p) - np.cumsum(q)).sum())


def emd_grid_marginal(a: np.ndarray, b: np.ndarray) -> float:
    """Cheap EMD approximation for 2-D occupancy grids: EMD of the row
    marginals plus EMD of the column marginals (normalized by grid size so the
    value is in cells / grid extent). Exact 2-D EMD (POT / linear program) can
    replace this without changing callers."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("grids must share shape")
    h, w = a.shape
    return float(emd_1d(a.sum(1), b.sum(1)) / max(h, 1) + emd_1d(a.sum(0), b.sum(0)) / max(w, 1))


def diagram_wasserstein(d0: Sequence[Tuple[float, float]],
                        d1: Sequence[Tuple[float, float]], p: int = 2) -> float:
    """p-Wasserstein distance between two finite persistence diagrams, with
    diagonal projections as the standard augmentation (Hungarian assignment).
    Uses gudhi.wasserstein when POT is available, else the built-in solver."""
    a = [(b, d) for b, d in d0 if np.isfinite(d)]
    b_ = [(b, d) for b, d in d1 if np.isfinite(d)]
    if not a and not b_:
        return 0.0
    if _gudhi_wasserstein is not None:
        try:
            return float(_gudhi_wasserstein(np.array(a).reshape(-1, 2), np.array(b_).reshape(-1, 2),
                                            order=p, internal_p=2))
        except Exception:
            pass
    A = np.array(a, dtype=np.float64).reshape(-1, 2)
    B = np.array(b_, dtype=np.float64).reshape(-1, 2)
    n, m = len(A), len(B)
    N = n + m
    C = np.zeros((N, N))
    if n and m:
        C[:n, :m] = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2) ** p
    # cost of matching a point to the diagonal = |b-d|/sqrt(2)
    diagA = (np.abs(A[:, 0] - A[:, 1]) / math.sqrt(2)) ** p if n else np.zeros(0)
    diagB = (np.abs(B[:, 0] - B[:, 1]) / math.sqrt(2)) ** p if m else np.zeros(0)
    C[:n, m:] = np.inf
    for i in range(n):
        C[i, m + i] = diagA[i]
    C[n:, :m] = np.inf
    for j in range(m):
        C[n + j, j] = diagB[j]
    C[n:, m:] = 0.0
    C[np.isinf(C)] = 1e12
    r, c = linear_sum_assignment(C)
    return float(C[r, c].sum() ** (1.0 / p))


# ── BOCPD ────────────────────────────────────────────────────────────────────

class BOCPD:
    """Bayesian Online Changepoint Detection (Adams & MacKay 2007) for a
    scalar stream with unknown mean and variance (Normal-Gamma prior,
    Student-t predictive). `update(x)` returns P(run length == 0), i.e. the
    posterior probability that a changepoint occurred at this sample --
    directly usable as soft boundary evidence.

    One instance per monitored signal; the same mechanism serves KE,
    coherence, occupancy entropy, flux, ... (one mechanism, many signals).
    """

    def __init__(self, hazard: float = 1.0 / 120.0, max_run: int = 400,
                 mu0: float = 0.0, kappa0: float = 1.0, alpha0: float = 1.0,
                 beta0: float = 1.0):
        self.h = float(hazard)
        self.max_run = int(max_run)
        self.mu0, self.k0, self.a0, self.b0 = mu0, kappa0, alpha0, beta0
        self.R = np.array([1.0])                   # run-length posterior
        self.mu = np.array([mu0]); self.kappa = np.array([kappa0])
        self.alpha = np.array([alpha0]); self.beta = np.array([beta0])
        self.p_change = 0.0
        self.n = 0

    def _pred(self, x: float) -> np.ndarray:
        # Student-t predictive density for each run length
        df = 2.0 * self.alpha
        scale2 = self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa)
        t = (x - self.mu) / np.sqrt(scale2)
        lg = (gammaln((df + 1) / 2) - gammaln(df / 2)
              - 0.5 * np.log(df * math.pi * scale2) - (df + 1) / 2 * np.log1p(t * t / df))
        return np.exp(lg)

    def update(self, x: float) -> float:
        x = float(x)
        pi = self._pred(x)
        growth = self.R * pi * (1.0 - self.h)
        cp = float((self.R * pi * self.h).sum())
        R = np.concatenate(([cp], growth))
        s = R.sum()
        R = R / s if s > 0 else np.array([1.0])
        # posterior parameter updates (prepend fresh prior for r = 0)
        mu_n = (self.kappa * self.mu + x) / (self.kappa + 1.0)
        beta_n = self.beta + self.kappa * (x - self.mu) ** 2 / (2.0 * (self.kappa + 1.0))
        self.mu = np.concatenate(([self.mu0], mu_n))
        self.kappa = np.concatenate(([self.k0], self.kappa + 1.0))
        self.alpha = np.concatenate(([self.a0], self.alpha + 0.5))
        self.beta = np.concatenate(([self.b0], beta_n))
        if len(R) > self.max_run:
            R = R[:self.max_run]; R /= R.sum()
            self.mu, self.kappa = self.mu[:self.max_run], self.kappa[:self.max_run]
            self.alpha, self.beta = self.alpha[:self.max_run], self.beta[:self.max_run]
        self.R = R
        self.n += 1
        # P(changepoint within the last `recent` samples): posterior mass on
        # short run lengths, with the hazard-driven baseline removed so a
        # quiet stream reads ~0 and a fresh regime reads ~1.
        recent = 4
        base = recent * self.h
        mass = float(R[:recent].sum())
        raw = max(0.0, (mass - base) / max(1.0 - base, 1e-9)) if self.n > 5 else 0.0
        # short EMA: a single-sample spike (noise outlier) is halved, a
        # genuine regime change (several consecutive short-run frames) passes
        self.p_change = 0.5 * self.p_change + 0.5 * raw
        return self.p_change

    @property
    def expected_run_length(self) -> float:
        return float((np.arange(len(self.R)) * self.R).sum())


# ── Markov expectation over symbolized macrostates ───────────────────────────

class MarkovExpectation:
    """First-order Markov model over a discrete symbol alphabet with Dirichlet
    smoothing. Exposes the IDyOM-style pair (surprisal, conditional entropy)
    plus entropy rate and predicted next-state entropy.

    Symbols are ints; `n_symbols` may grow (alphabet is open) -- counts are
    stored sparsely."""

    def __init__(self, prior: float = 0.05, forgetting: float = 0.999):
        """
        v4.6: `prior` reduced 0.5 -> 0.05. The prior mass is prior x |alphabet|,
        so with 243 possible symbols a 0.5 prior asserted ~120 units of
        imaginary evidence. A five-minute run yields ~4500 observations spread
        across many symbols, so no individual symbol ever accumulated enough to
        outweigh the assumption: the predictive distribution stayed near
        uniform and the conditional entropy stayed at its maximum.
        """
        self.prior = float(prior)
        self.forget = float(forgetting)
        self.counts: Dict[int, Dict[int, float]] = {}
        self.alphabet: set = set()
        self.prev: Optional[int] = None
        self.last_surprisal = 0.0
        self.last_transition: Optional[Tuple[int, int]] = None

    def _row(self, s: int) -> Dict[int, float]:
        return self.counts.setdefault(s, {})

    def next_distribution(self, s: int) -> Dict[int, float]:
        row = self._row(s)
        K = max(len(self.alphabet), 1)
        tot = sum(row.values()) + self.prior * K
        return {t: (row.get(t, 0.0) + self.prior) / tot for t in self.alphabet} if self.alphabet else {}

    def conditional_entropy(self, s: int) -> float:
        """Uncertainty about the next symbol given `s`, in bits."""
        d = self.next_distribution(s)
        return shannon_entropy(list(d.values())) if d else 0.0

    def max_entropy(self) -> float:
        """Entropy of a uniform distribution over the symbols ACTUALLY SEEN.

        Consumers normalise by this rather than by a fixed constant. The old
        code divided by a hardcoded 3.0 and clipped to 1.0; since the raw
        entropy over a 243-symbol alphabet sits well above 3 bits, the result
        was exactly 1.0 on every frame and the four things ambiguity controls
        (chord cardinality, note density, pitch gravitation, and one of the
        four tension terms) were all frozen at one end of their range.
        """
        return float(math.log2(max(len(self.alphabet), 2)))

    def normalised_conditional_entropy(self, s: int) -> float:
        """Conditional entropy scaled to [0, 1] by the observed alphabet."""
        return float(min(1.0, self.conditional_entropy(s) / max(self.max_entropy(), 1e-9)))

    def stationary(self) -> Dict[int, float]:
        tot = {s: sum(r.values()) for s, r in self.counts.items()}
        z = sum(tot.values())
        return {s: v / z for s, v in tot.items()} if z > 0 else {}

    def entropy_rate(self) -> float:
        """Global unpredictability: Σ π(s) H(next | s)."""
        st = self.stationary()
        return float(sum(p * self.conditional_entropy(s) for s, p in st.items()))

    def predicted_next_entropy(self, s: int) -> float:
        """Expected conditional entropy one step ahead: Σ_t P(t|s) H(next|t)."""
        d = self.next_distribution(s)
        return float(sum(p * self.conditional_entropy(t) for t, p in d.items()))

    def update(self, symbol: int) -> float:
        """Observe a symbol; returns surprisal (bits) of the transition."""
        self.alphabet.add(symbol)
        if self.prev is not None:
            d = self.next_distribution(self.prev)
            p = d.get(symbol, self.prior / (sum(self._row(self.prev).values()) + self.prior * max(len(self.alphabet), 1)))
            self.last_surprisal = float(-math.log2(max(p, 1e-12)))
            self.last_transition = (self.prev, symbol)
            row = self._row(self.prev)
            if self.forget < 1.0:
                for k in row:
                    row[k] *= self.forget
            row[symbol] = row.get(symbol, 0.0) + 1.0
        else:
            self.last_surprisal = 0.0
        self.prev = symbol
        return self.last_surprisal


# ── The macrostate ───────────────────────────────────────────────────────────

@dataclass
class SystemSummary:
    """Single-instant system summary (the 8-vector macrostate + extras).

    Scalars (Euclidean-comparable after normalization):
        occupancy_entropy     spatial disorder of wax over the ROI grid, [0,1]
        persistence_entropy   topological evenness of H0 lifetimes, [0,1]
        coherence             Vicsek velocity-coherence order parameter, [0,1]
        ke_total              Σ ½·area·speed² (px²·px²/frame², normalized later)
        w_equilibrium         EMD between current occupancy and equilibrium
        w_diagram             2-Wasserstein between this and previous diagram
        blob_count            number of tracked blobs
        tree_depth            max depth of the merge tree
    Extras (grow freely): per-blob, relational, aggregate and topological
        features that feed governors but are not part of the symbol.
    Distributional components (EMD-comparable):
        occupancy             coarse occupancy grid (normalized)
        velocity_hist         histogram of vertical velocities (normalized)
        diagram               list of (birth, death)
    """
    t: float = 0.0
    occupancy_entropy: float = 0.0
    persistence_entropy: float = 0.0
    coherence: float = 0.0
    ke_total: float = 0.0
    w_equilibrium: float = 0.0
    w_diagram: float = 0.0
    blob_count: int = 0
    tree_depth: int = 0
    # information-theoretic outputs of the Markov layer
    symbol: int = 0
    surprisal: float = 0.0
    cond_entropy: float = 0.0
    entropy_rate: float = 0.0
    pred_next_entropy: float = 0.0
    # distributional components
    occupancy: Optional[np.ndarray] = None
    velocity_hist: Optional[np.ndarray] = None
    diagram: List[Tuple[float, float]] = field(default_factory=list)
    extra: Dict[str, float] = field(default_factory=dict)

    SCALAR_KEYS = ("occupancy_entropy", "persistence_entropy", "coherence", "ke_total",
                   "w_equilibrium", "w_diagram", "blob_count", "tree_depth")

    def scalar_vector(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in self.SCALAR_KEYS], dtype=np.float64)


def summary_distance(a: SystemSummary, b: SystemSummary, scale: Optional[np.ndarray] = None,
                     w_scalar: float = 1.0, w_dist: float = 1.0) -> float:
    """'How different is now from then': Euclidean on normalized scalars +
    EMD on distributional components."""
    va, vb = a.scalar_vector(), b.scalar_vector()
    if scale is None:
        scale = np.maximum(np.abs(va) + np.abs(vb), 1e-9)
    d = float(np.linalg.norm((va - vb) / scale)) / math.sqrt(len(va))
    e = 0.0
    if a.occupancy is not None and b.occupancy is not None and a.occupancy.shape == b.occupancy.shape:
        e += emd_grid_marginal(a.occupancy, b.occupancy)
    if a.velocity_hist is not None and b.velocity_hist is not None:
        e += emd_1d(a.velocity_hist, b.velocity_hist) / max(len(a.velocity_hist), 1)
    return w_scalar * d + w_dist * e


# ── Per-frame tracker ────────────────────────────────────────────────────────

class SystemStateTracker:
    """Computes SystemSummary per frame from tracked blobs (+ optional
    persistence diagram and merge-tree stats), symbolizes it, updates the
    Markov expectation model, and keeps histories for periodicity / entropy
    analysis. Strictly read-only over its inputs."""

    def __init__(self, roi_w: float, roi_h: float, grid: Tuple[int, int] = (8, 6),
                 history: int = 600, vel_bins: int = 9, vel_scale: float = 4.0,
                 quant: Optional[Dict[str, Sequence[float]]] = None):
        self.roi_w, self.roi_h = float(roi_w), float(roi_h)
        self.grid = grid
        self.vel_bins, self.vel_scale = vel_bins, vel_scale
        self.history: Deque[SystemSummary] = deque(maxlen=history)
        self.markov = MarkovExpectation()
        self.equilibrium: Optional[np.ndarray] = None
        self.prev_diagram: List[Tuple[float, float]] = []
        # signal histories (name -> deque) for periodicity / permutation entropy
        self.signals: Dict[str, Deque[float]] = {}
        self.blob_hist: Dict[int, Deque[Tuple[float, float, float]]] = {}   # id -> (t, y, vy)
        # quantization edges for symbolization (default terciles-ish)
        self.quant = quant or {
            "blob_count": [1.5, 3.5],              # 1 | 2-3 | 4+
            "coherence": [0.35, 0.7],
            "ke_norm": [0.15, 0.5],
            "flux_sign": [-0.05, 0.05],
            "occupancy_entropy": [0.4, 0.7],
        }
        self.ke_ref = 1.0                        # running max for normalization
        self.t = 0.0

    # -- helpers ----------------------------------------------------------
    def _sig(self, name: str, v: float, maxlen: int = 600) -> None:
        self.signals.setdefault(name, deque(maxlen=maxlen)).append(float(v))

    def signal(self, name: str) -> List[float]:
        return list(self.signals.get(name, []))

    def set_equilibrium(self, grid: np.ndarray) -> None:
        g = np.asarray(grid, dtype=np.float64)
        self.equilibrium = g / g.sum() if g.sum() > 0 else g

    def occupancy_grid(self, blobs) -> np.ndarray:
        gw, gh = self.grid
        g = np.zeros((gh, gw))
        for b in blobs:
            x, y = b.centroid
            i = min(gh - 1, max(0, int(y / self.roi_h * gh)))
            j = min(gw - 1, max(0, int(x / self.roi_w * gw)))
            g[i, j] += max(float(getattr(b, "area", 1.0)), 1.0)
        return g / g.sum() if g.sum() > 0 else g

    # -- relational --------------------------------------------------------
    def relational(self, blobs) -> Dict[str, float]:
        """Configurational statistics over the blob constellation.

        v4.3: no longer a @staticmethod, because the Kuramoto order parameter
        now reads each blob's TRUE oscillation phase out of self.blob_hist via
        blob_phase_and_period(). The previous implementation used a pixel-space
        proxy, atan2(-vy, y - 0.5), whose second argument was a raw pixel
        ordinate minus a hardcoded 0.5 -- always large and positive, so every
        phase collapsed to ~0 and the order parameter read ~1.0 permanently.
        Downstream that pinned the elastic clock at full strength, i.e. the
        projective/elastic meter was silently hard-quantized at all times.
        """
        n = len(blobs)
        out = {"pair_dist_mean": 0.0, "pair_dist_var": 0.0, "collinearity": 0.0,
               "clustering": 0.0, "com_x": 0.0, "com_y": 0.0, "kuramoto": 0.0,
               "hull_area": 0.0, "kuramoto_n": 0.0, "kuramoto_phase": 0.0}
        if n == 0:
            return out
        P = np.array([b.centroid for b in blobs], dtype=np.float64)
        A = np.array([max(float(getattr(b, "area", 1.0)), 1.0) for b in blobs])
        com = (P * A[:, None]).sum(0) / A.sum()
        out["com_x"], out["com_y"] = float(com[0]), float(com[1])
        if n >= 2:
            D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
            iu = np.triu_indices(n, 1)
            out["pair_dist_mean"] = float(D[iu].mean())
            out["pair_dist_var"] = float(D[iu].var())
            nn = np.sort(D, axis=1)[:, 1]
            out["clustering"] = float(nn.mean() / (out["pair_dist_mean"] + 1e-9))
        if n >= 3:
            Q = P - P.mean(0)
            sv = np.linalg.svd(Q, compute_uv=False)
            out["collinearity"] = float(1.0 - sv[1] / (sv[0] + 1e-9))   # 1 = collinear
            try:
                out["hull_area"] = float(_ConvexHull(P).volume) if _ConvexHull else 0.0
            except Exception:
                out["hull_area"] = 0.0

        # -- Kuramoto order parameter over TRUE vertical-oscillation phases --
        # R = |mean(exp(i*phase))| in [0,1]: 1 = all blobs rising and falling
        # in lockstep, 0 = phases spread uniformly. Only blobs whose cycle has
        # actually been resolved (period > 0, i.e. at least one full up-down
        # swing in their y history) contribute -- an unresolved blob has no
        # meaningful phase and including it would bias R either way.
        phases = []
        for b in blobs:
            ph, per = self.blob_phase_and_period(b.id)
            if per > 0.0:
                phases.append(ph)
        out["kuramoto_n"] = float(len(phases))
        if len(phases) >= 2:
            z = np.mean(np.exp(1j * np.asarray(phases, dtype=np.float64)))
            out["kuramoto"] = float(abs(z))
            out["kuramoto_phase"] = float(np.angle(z))
        else:
            # Fewer than two resolved oscillators: synchronisation is undefined.
            # Return 0.0, NOT 1.0 -- a lone blob is trivially "in phase with
            # itself", and defaulting high is exactly what pinned the elastic
            # clock. 0.0 makes the clock maximally elastic, which is the right
            # behaviour when there is no collective pulse to lock to.
            out["kuramoto"] = 0.0
        return out

    # -- aggregate physical -------------------------------------------------
    @staticmethod
    def aggregate(blobs) -> Dict[str, float]:
        if not blobs:
            return {"ke": 0.0, "flux": 0.0, "coherence": 0.0, "com_h": 0.0, "mean_speed": 0.0}
        A = np.array([max(float(getattr(b, "area", 1.0)), 1.0) for b in blobs])
        V = np.array([[float(getattr(b, "vx", 0.0)), float(getattr(b, "vy", 0.0))] for b in blobs])
        sp = np.linalg.norm(V, axis=1)
        ke = float(0.5 * (A * sp ** 2).sum())
        flux = float((A * V[:, 1]).sum() / A.sum())       # + = downward in image coords
        nz = sp > 1e-6
        coh = float(np.linalg.norm(V[nz].sum(0) / (sp[nz].sum() + 1e-9))) if nz.any() else 0.0
        return {"ke": ke, "flux": -flux, "coherence": coh,
                "com_h": float((A * np.array([b.centroid[1] for b in blobs])).sum() / A.sum()),
                "mean_speed": float(sp.mean())}

    # -- main update --------------------------------------------------------
    def update(self, blobs, dt: float, diagram: Optional[Sequence[Tuple[float, float]]] = None,
               tree_stats: Optional[dict] = None) -> SystemSummary:
        self.t += float(dt)
        s = SystemSummary(t=self.t)
        blobs = list(blobs)

        # per-blob kinematic histories
        for b in blobs:
            self.blob_hist.setdefault(b.id, deque(maxlen=600)).append(
                (self.t, float(b.centroid[1]), float(getattr(b, "vy", 0.0))))

        rel = self.relational(blobs)
        agg = self.aggregate(blobs)
        occ = self.occupancy_grid(blobs)
        vy = np.array([float(getattr(b, "vy", 0.0)) for b in blobs]) if blobs else np.zeros(0)
        vh, _ = np.histogram(np.clip(vy, -self.vel_scale, self.vel_scale),
                             bins=self.vel_bins, range=(-self.vel_scale, self.vel_scale))
        vh = vh.astype(np.float64); vh = vh / vh.sum() if vh.sum() > 0 else vh

        diag = [(float(b), float(d)) for b, d in (diagram or []) if np.isfinite(d)] if diagram is not None \
            else [(float(b.birth), float(b.death)) for b in blobs]

        s.blob_count = len(blobs)
        s.coherence = agg["coherence"]
        s.ke_total = agg["ke"]
        self.ke_ref = max(self.ke_ref * 0.999, s.ke_total, 1e-6)
        s.occupancy_entropy = occupancy_entropy(occ)
        s.persistence_entropy = persistence_entropy(diag)
        if self.equilibrium is None or self.equilibrium.shape != occ.shape:
            self.equilibrium = occ.copy() if occ.sum() > 0 else None
        s.w_equilibrium = emd_grid_marginal(occ, self.equilibrium) if self.equilibrium is not None else 0.0
        s.w_diagram = diagram_wasserstein(self.prev_diagram, diag) if self.prev_diagram or diag else 0.0
        self.prev_diagram = diag
        s.tree_depth = int(tree_stats.get("max_depth", 0)) if tree_stats else 0
        s.occupancy, s.velocity_hist, s.diagram = occ, vh, diag
        s.extra.update({f"rel_{k}": v for k, v in rel.items()})
        s.extra.update({f"agg_{k}": v for k, v in agg.items()})
        if tree_stats:
            s.extra.update({f"tree_{k}": float(v) for k, v in tree_stats.items()})

        # signal histories (for periodicity bank, permutation entropy, BOCPD)
        self._sig("ke", s.ke_total / self.ke_ref)
        self._sig("flux", agg["flux"])
        self._sig("coherence", s.coherence)
        self._sig("occupancy_entropy", s.occupancy_entropy)
        self._sig("persistence_entropy", s.persistence_entropy)
        self._sig("w_equilibrium", s.w_equilibrium)
        self._sig("w_diagram", s.w_diagram)
        self._sig("pair_dist_var", rel["pair_dist_var"])
        self._sig("collinearity", rel["collinearity"])
        self._sig("clustering", rel["clustering"])
        self._sig("com_y", rel["com_y"])
        self._sig("kuramoto", rel["kuramoto"])
        s.extra["perm_entropy_ke"] = permutation_entropy(self.signal("ke")[-64:])
        s.extra["perm_entropy_flux"] = permutation_entropy(self.signal("flux")[-64:])

        # symbolize + Markov
        s.symbol = self.symbolize(s, agg)
        s.surprisal = self.markov.update(s.symbol)
        s.cond_entropy = self.markov.conditional_entropy(s.symbol)
        s.entropy_rate = self.markov.entropy_rate()
        s.pred_next_entropy = self.markov.predicted_next_entropy(s.symbol)
        self._sig("cond_entropy", s.cond_entropy)
        self._sig("surprisal", s.surprisal)
        self.history.append(s)
        return s

    def symbolize(self, s: SystemSummary, agg: Dict[str, float]) -> int:
        """Coarse-grain the macrostate into a small integer alphabet.
        Each quantized dimension contributes a digit (mixed radix)."""
        def q(v, edges):
            return int(np.searchsorted(np.asarray(edges), v))
        # v4.6: the occupancy-entropy digit is dropped. It was the least
        # informative of the five (it correlates strongly with blob count) and
        # it tripled the alphabet, 81 -> 243. A smaller alphabet means each
        # symbol accumulates observations sooner, which is what lets the
        # transition model actually learn within a session.
        digits = [
            q(s.blob_count, self.quant["blob_count"]),
            q(s.coherence, self.quant["coherence"]),
            q(s.ke_total / self.ke_ref, self.quant["ke_norm"]),
            q(agg["flux"], self.quant["flux_sign"]),
        ]
        sym = 0
        for d in digits:
            sym = sym * 3 + d
        return sym

    # -- recurrence / boundary helpers --------------------------------------
    def distance_to_past(self, lag: int) -> float:
        if len(self.history) <= lag:
            return 0.0
        return summary_distance(self.history[-1], self.history[-1 - lag])

    def nearest_past_state(self, min_lag: int = 60) -> Tuple[Optional[int], float]:
        """Form-level recurrence: index (lag) and distance of the most similar
        past macrostate at least `min_lag` frames ago."""
        if len(self.history) <= min_lag:
            return None, float("inf")
        cur = self.history[-1]
        best, best_d = None, float("inf")
        for lag in range(min_lag, len(self.history)):
            d = summary_distance(cur, self.history[-1 - lag])
            if d < best_d:
                best, best_d = lag, d
        return best, best_d

    def blob_phase_and_period(self, blob_id: int) -> Tuple[float, float]:
        """Vertical-oscillation phase (radians) and dominant period (seconds).

        The phase is the analytic-signal angle of the mean-and-drift-removed
        vertical position: atan2(-vy/omega, y_detrended). Scaling velocity by
        the estimated angular frequency puts both axes in comparable units, so
        the pair traces a circle and the angle advances linearly through 2*pi
        per cycle. Two blobs at opposite points of their arcs therefore differ
        by pi -- the property the Kuramoto sum needs, and the one the old
        pixel-space proxy could not express.

        Returns (0.0, 0.0) when the history is too short, the signal is flat,
        or no full cycle has been observed. Callers MUST treat period == 0.0
        as "unresolved" rather than as a valid phase.
        """
        h = self.blob_hist.get(blob_id)
        if not h or len(h) < 12:
            return 0.0, 0.0
        t = np.array([r[0] for r in h], dtype=np.float64)
        y = np.array([r[1] for r in h], dtype=np.float64)
        vy = np.array([r[2] for r in h], dtype=np.float64)

        # Detrend: remove mean AND linear drift, so a blob rising steadily
        # without oscillating does not register a spurious phase.
        A = np.vstack([t - t.mean(), np.ones_like(t)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        yd = y - A @ coef

        amp = float(np.std(yd))
        if amp < 1e-6:
            return 0.0, 0.0

        # Period from zero crossings: two per cycle, so period = 2 * mean
        # crossing interval. At least three crossings (a full cycle plus a
        # confirming edge) are required before the period counts as resolved.
        zc = np.where(np.diff(np.sign(yd)) != 0)[0]
        if len(zc) < 3:
            return 0.0, 0.0
        dt = np.diff(t[zc])
        dt = dt[dt > 1e-6]
        if len(dt) == 0:
            return 0.0, 0.0
        period = float(2.0 * np.mean(dt))
        if period <= 0.0:
            return 0.0, 0.0

        # Velocity is taken as a NUMERICAL derivative of the detrended
        # position (px/second), not from the stored per-blob vy: the tracker
        # reports vy in px/FRAME, and mixing that with omega in rad/SECOND
        # scales the velocity axis by ~1/fps, which collapses every phase onto
        # 0 or pi and makes R meaningless. Deriving dy/dt here is unit-safe
        # regardless of frame rate.
        if len(t) >= 3 and (t[-1] - t[-3]) > 1e-9:
            dydt = (yd[-1] - yd[-3]) / (t[-1] - t[-3])
        else:
            dydt = 0.0
        omega = 2.0 * math.pi / period
        phase = math.atan2(-dydt / omega, float(yd[-1]))
        return float(phase), period


def periodicity_strength(x: Sequence[float], min_lag: int = 4) -> Tuple[float, int]:
    """Autocorrelation-peak periodicity of a signal. Returns (strength in
    [0,1], lag of the peak). Strength 0 for flat / too-short signals."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2 * min_lag + 2 or np.std(x) < 1e-9:
        return 0.0, 0
    x = (x - x.mean()) / (x.std() + 1e-12)
    ac = np.correlate(x, x, mode="full")[len(x) - 1:] / len(x)
    if len(ac) <= min_lag + 1:
        return 0.0, 0
    lag = int(np.argmax(ac[min_lag:])) + min_lag
    return float(max(0.0, min(1.0, ac[lag]))), lag


def spectral_flatness(x: Sequence[float]) -> float:
    """Geometric/arithmetic mean ratio of the power spectrum, [0,1]; 1 = white."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 8 or np.std(x) < 1e-9:
        return 1.0
    p = np.abs(np.fft.rfft(x - x.mean())) ** 2 + 1e-12
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))
