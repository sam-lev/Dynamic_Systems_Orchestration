# test_merge_tree.py -- Audible Lava / DSO v4.2
# Validation suite for h0_persistence.h0_peaks, merge_tree.compute_merge_tree,
# and the H0 stage of blob_detection.BlobDetector.
#
# Purpose: validate ACCURACY (does the tree reflect the true superlevel-set
# topology of the luminance field?) and EXPRESSIVENESS (are persistence,
# localization, nesting, saddles, depth, and enrichment reliable enough to
# hang grouping / Schenkerian-level semantics on?) before score_governor.py
# builds on it.
#
# Groups
#   H0-xx  h0_persistence unit tests (stride, cofaces, essential class)
#   MT-xx  merge tree structure vs. an independent union-find oracle
#   FZ-xx  fuzzing over random peak configurations
#   IV-xx  invariances (noise, monotone transform, temporal coherence)
#   DT-xx  BlobDetector integration on synthetic wax frames
#   PF-xx  performance budget
# Tags: CORE (must pass before governor work) / STRUCT (licenses the tree
# as a grouping / level hierarchy) / INTEG (detection pipeline).
#
# Run:  python test_merge_tree.py      or      pytest test_merge_tree.py -v

import os
import sys
import time

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
# Own directory first (the fixed modules), project dir as fallback for
# lamp_profile / background_reference when run outside the project tree.
for p in ("/mnt/project", HERE):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from h0_persistence import h0_peaks, _fallback_localize          # noqa: E402
from merge_tree import (compute_merge_tree, MergeTreeNode,          # noqa: E402
                        iter_nodes, assign_depths, tree_stats)

DOWNSCALE = 4
PERS_TH = 10.0
RNG = np.random.default_rng(1917)
SHAPE = (240, 320)          # non-square on purpose: exposes stride bugs


# --------------------------------------------------------------------------
# Synthetic fields
# --------------------------------------------------------------------------

def gaussian_field(shape, peaks, base=20.0):
    """peaks=[(y, x, height_above_base, sigma)] -> uint8 image."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    f = np.full(shape, base, dtype=np.float64)
    for (py, px, height, sigma) in peaks:
        f += height * np.exp(-(((yy - py) ** 2) + ((xx - px) ** 2)) / (2 * sigma ** 2))
    return np.clip(f, 0, 255).astype(np.uint8)


def ridge_pair(shape, p1, p2, h1, h2, saddle_h, sigma, base=20.0):
    """Two peaks joined by a ridge whose saddle sits at base+saddle_h."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    def bump(py, px, height):
        return height * np.exp(-(((yy - py) ** 2) + ((xx - px) ** 2)) / (2 * sigma ** 2))

    p1 = np.array(p1, dtype=np.float64); p2 = np.array(p2, dtype=np.float64)
    d = p2 - p1
    t = np.clip(((yy - p1[0]) * d[0] + (xx - p1[1]) * d[1]) / float(d @ d), 0, 1)
    dist2 = (yy - (p1[0] + t * d[0])) ** 2 + (xx - (p1[1] + t * d[1])) ** 2
    ridge = saddle_h * np.exp(-dist2 / (2 * sigma ** 2))
    f = base + np.maximum.reduce([bump(*p1, h1), bump(*p2, h2), ridge])
    return np.clip(f, 0, 255).astype(np.uint8)


def downscaled(img):
    h, w = img.shape[:2]
    return cv2.resize(img, (w // DOWNSCALE, h // DOWNSCALE), interpolation=cv2.INTER_AREA)


def random_peaks(n, shape=SHAPE, hmin=40, hmax=200):
    h, w = shape
    return [(int(RNG.integers(25, h - 25)), int(RNG.integers(25, w - 25)),
             float(RNG.integers(hmin, hmax)), int(RNG.integers(8, 20)))
            for _ in range(n)]


# --------------------------------------------------------------------------
# Oracle: independent union-find over the SAME filtration (8-connectivity,
# matching the cubical complex where diagonal cells share a vertex).
# --------------------------------------------------------------------------

class OracleTree:
    """Sweep pixels from high to low intensity; elder rule on merges.

    records      : list of (birth, death, peak_yx, winner_peak_yx)
    essential    : [peak_yx] of the never-dying component(s)
    parent_of    : peak_yx -> winner_peak_yx (the peak of the elder component
                   it merged INTO at its death level)
    """

    def __init__(self, small):
        f = np.asarray(small, dtype=np.float64)
        h, w = f.shape
        flat = f.ravel()
        # stable descending sort so ties resolve deterministically
        order = np.argsort(-flat, kind="stable")
        parent = np.arange(h * w)
        comp_peak = {}
        comp_birth = {}
        active = np.zeros(h * w, dtype=bool)
        self.records = []
        self.parent_of = {}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for idx in order:
            idx = int(idx)
            level = flat[idx]
            comp_peak[idx] = idx
            comp_birth[idx] = level
            active[idx] = True
            y, x = divmod(idx, w)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < h and 0 <= nx < w):
                        continue
                    nidx = ny * w + nx
                    if not active[nidx]:
                        continue
                    ra, rb = find(idx), find(nidx)
                    if ra == rb:
                        continue
                    if comp_birth[ra] > comp_birth[rb] or (
                            comp_birth[ra] == comp_birth[rb] and comp_peak[ra] < comp_peak[rb]):
                        win, lose = ra, rb
                    else:
                        win, lose = rb, ra
                    lose_peak = divmod(comp_peak[lose], w)
                    win_peak = divmod(comp_peak[win], w)
                    self.records.append((comp_birth[lose], level, lose_peak, win_peak))
                    self.parent_of[lose_peak] = win_peak
                    parent[lose] = win
                    comp_peak.pop(lose); comp_birth.pop(lose)
        self.essential = [divmod(p, w) for p in comp_peak.values()]

    def finite_pairs(self, min_pers=0.0):
        return [r for r in self.records if (r[0] - r[1]) >= min_pers]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def run_tree(img, blobs=None, pers_th=PERS_TH):
    return compute_merge_tree(img, blobs=blobs, downscale=DOWNSCALE,
                              persistence_threshold=pers_th)


def small_yx(node):
    """Full-res node pixel -> downscaled grid coords (inverse of _full_res)."""
    return (node.pixel[0] // DOWNSCALE, node.pixel[1] // DOWNSCALE)


def oldest_leaf_under(node):
    return max((n for n in iter_nodes(node) if n.is_leaf), key=lambda n: n.threshold)


def parent_map(root):
    """For each finite leaf: the oldest leaf of the sibling subtree it merged
    into (== the oracle's 'winner peak' of the elder component)."""
    assign_depths(root)
    out = {}
    for n in iter_nodes(root):
        if n.is_leaf or len(n.children) != 2:
            continue
        a, b = n.children
        # the child whose oldest leaf is younger is the one that merged in
        oa, ob = oldest_leaf_under(a), oldest_leaf_under(b)
        if oa.threshold < ob.threshold:
            out[small_yx(oa)] = (small_yx(ob), n.threshold, small_yx(n))
        else:
            out[small_yx(ob)] = (small_yx(oa), n.threshold, small_yx(n))
    return out


class FakeBlob:
    def __init__(self, centroid, orientation=1.1, elongation=2.2, hu=None,
                 mean_hue=33.0, direction="rising"):
        self.centroid = centroid      # (x, y)
        self.orientation = orientation; self.elongation = elongation
        self.hu = hu if hu is not None else [float(i) for i in range(7)]
        self.mean_hue = mean_hue; self.direction = direction


TESTS = []

def test(tag):
    def deco(fn):
        TESTS.append((tag, fn)); return fn
    return deco


# ==========================================================================
# H0 -- h0_persistence unit tests
# ==========================================================================

@test("CORE")
def h0_01_nonsquare_pairs_match_oracle():
    """Exact multiset of (birth, death) finite pairs equals the oracle's on a
    non-square field. Any stride error fails this immediately."""
    img = gaussian_field(SHAPE, [(60, 60, 170.0, 14), (60, 250, 130.0, 14),
                                 (180, 80, 100.0, 14), (180, 240, 70.0, 14)])
    sm = downscaled(img)
    got = sorted((p.birth, p.death) for p in h0_peaks(sm, PERS_TH) if not p.essential)
    orc = OracleTree(sm)
    exp = sorted((r[0], r[1]) for r in orc.finite_pairs(PERS_TH))
    assert got == exp, f"finite pairs differ\n got={got}\n exp={exp}"


@test("CORE")
def h0_02_birth_cell_carries_birth_value():
    img = gaussian_field(SHAPE, random_peaks(5))
    sm = downscaled(img).astype(np.float64)
    for p in h0_peaks(sm, 5.0):
        assert sm[p.peak_yx] == p.birth, f"peak cell value {sm[p.peak_yx]} != birth {p.birth}"
        if p.saddle_yx is not None:
            assert sm[p.saddle_yx] == p.death, "saddle cell value != death level"


@test("CORE")
def h0_03_equal_height_peaks_distinct():
    img = gaussian_field(SHAPE, [(60, 80, 140.0, 14), (180, 240, 140.0, 14),
                                 (120, 160, 200.0, 12)])
    pk = [p for p in h0_peaks(downscaled(img), PERS_TH) if not p.essential]
    assert len(pk) == 2, f"expected 2 finite peaks, got {len(pk)}"
    d = np.hypot(pk[0].peak_yx[0] - pk[1].peak_yx[0], pk[0].peak_yx[1] - pk[1].peak_yx[1])
    assert d > 30, f"equal-height peaks collapsed ({d:.0f} cells apart)"


@test("CORE")
def h0_04_essential_class_present_and_is_global_max():
    img = gaussian_field(SHAPE, [(60, 60, 170.0, 14), (60, 250, 130.0, 14)])
    sm = downscaled(img)
    pk = h0_peaks(sm, PERS_TH)
    ess = [p for p in pk if p.essential]
    assert len(ess) == 1, f"expected exactly one essential class, got {len(ess)}"
    e = ess[0]
    assert sm[e.peak_yx] == sm.max(), "essential peak is not the global maximum"
    assert e.death == float(sm.min()), "essential death must be the field minimum"
    assert e.persistence == e.birth - e.death
    assert pk[0].essential, "essential class must rank first by persistence"


@test("CORE")
def h0_05_single_peak_yields_only_essential():
    img = gaussian_field(SHAPE, [(120, 160, 180.0, 18)])
    pk = h0_peaks(downscaled(img), PERS_TH)
    assert len(pk) == 1 and pk[0].essential, (
        f"single smooth peak must give exactly the essential class, got {len(pk)}")
    assert not h0_peaks(downscaled(img), PERS_TH, include_essential=False)


@test("CORE")
def h0_06_threshold_semantics():
    img = ridge_pair(SHAPE, (120, 90), (120, 230), 150.0, 100.0, 60.0, 16)
    sm = downscaled(img)
    all_fin = [p for p in h0_peaks(sm, 0.0) if not p.essential]
    p_true = max(p.persistence for p in all_fin)
    assert abs(p_true - 40.0) < 8.0, f"ridge persistence {p_true:.1f} vs analytic 40"
    assert len([p for p in h0_peaks(sm, p_true + 1.0) if not p.essential]) == 0
    assert len([p for p in h0_peaks(sm, p_true - 1.0) if not p.essential]) >= 1


@test("CORE")
def h0_07_fallback_localizer_is_a_local_min():
    img = gaussian_field(SHAPE, [(60, 80, 140.0, 14), (180, 240, 140.0, 14)])
    inv = 255.0 - downscaled(img).astype(np.float64)
    used = set()
    a = _fallback_localize(inv, inv.min(), used); used.add(a)
    b = _fallback_localize(inv, inv.min(), used)
    assert a is not None and b is not None and a != b, "fallback must find distinct equal-value minima"
    for yx in (a, b):
        y, x = yx
        nb = [inv[y + dy, x + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)
              if 0 <= y + dy < inv.shape[0] and 0 <= x + dx < inv.shape[1]]
        assert inv[yx] <= min(nb)


@test("CORE")
def h0_08_square_and_transposed_agree():
    """Persistence multiset must be invariant to transposition (h <-> w)."""
    img = gaussian_field((200, 320), random_peaks(4, (200, 320)))
    a = sorted(round(p.persistence, 6) for p in h0_peaks(downscaled(img), 5.0))
    b = sorted(round(p.persistence, 6) for p in h0_peaks(downscaled(img).T.copy(), 5.0))
    assert a == b, "persistence changed under transposition -> stride bug"


# ==========================================================================
# MT -- merge tree structure
# ==========================================================================

@test("CORE")
def mt_01_single_peak_tree():
    root, leaves = run_tree(gaussian_field(SHAPE, [(120, 160, 180.0, 18)]))
    assert len(leaves) == 1 and leaves[0].is_essential
    assert root is leaves[0] and root.depth == 0 and root.leaf_count == 1


@test("STRUCT")
def mt_02_two_peaks_true_saddle():
    img = ridge_pair(SHAPE, (120, 90), (120, 230), 150.0, 100.0, 60.0, 16)
    root, leaves = run_tree(img)
    assert len(leaves) == 2, f"expected essential + 1 finite leaf, got {len(leaves)}"
    assert not root.is_leaf and len(root.children) == 2
    saddle_true = 20.0 + 60.0
    assert abs(root.threshold - saddle_true) < 8.0, (
        f"internal threshold {root.threshold:.1f} != saddle {saddle_true:.1f}")
    sm = downscaled(img)
    assert sm[small_yx(root)] == root.threshold, "saddle pixel value must equal saddle level"
    # saddle lies between the two peaks (x between 90 and 230 at full res)
    assert 90 < root.pixel[1] < 230 and abs(root.pixel[0] - 120) < 3 * DOWNSCALE


@test("STRUCT")
def mt_03_three_peaks_nesting_and_levels():
    """A(tall)-B(mid) joined by a HIGH saddle; C far with a LOW saddle.
    True tree: (A,B) merge first at ~110, then C joins at ~base."""
    h, w = SHAPE
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ridge = ridge_pair(SHAPE, (80, 100), (80, 180), 170.0, 120.0, 90.0, 14).astype(np.float64)
    c = 100.0 * np.exp(-(((yy - 180) ** 2 + (xx - 270) ** 2)) / (2 * 12 ** 2))
    img = np.clip(np.maximum(ridge, 20.0 + c), 0, 255).astype(np.uint8)

    root, leaves = run_tree(img)
    assert len(leaves) == 3
    pm = parent_map(root)
    orc = OracleTree(downscaled(img))
    for leaf_yx, (partner_yx, level, saddle_yx) in pm.items():
        exp_partner = orc.parent_of.get(leaf_yx)
        assert exp_partner is not None, f"leaf {leaf_yx} not in oracle"
        assert partner_yx == exp_partner, (
            f"leaf {leaf_yx} nested under {partner_yx}, oracle says {exp_partner}")
    # levels: first merge high (~110), root low (~20)
    internal = sorted((n.threshold for n in iter_nodes(root) if not n.is_leaf), reverse=True)
    assert abs(internal[0] - 110.0) < 8.0 and internal[1] < 40.0, internal


@test("STRUCT")
def mt_04_depth_and_leaf_count_consistency():
    img = gaussian_field(SHAPE, random_peaks(6))
    root, leaves = run_tree(img, pers_th=5.0)
    assert root.depth == 0 and root.parent is None
    assert root.leaf_count == len(leaves)
    for n in iter_nodes(root):
        for ch in n.children:
            assert ch.depth == n.depth + 1 and ch.parent is n
        if not n.is_leaf:
            assert n.leaf_count == sum(c.leaf_count for c in n.children)
            # every child's threshold (birth or saddle) is >= parent's saddle
            assert all(c.threshold >= n.threshold for c in n.children), "non-monotone levels"
    st = tree_stats(root)
    assert st["leaf_count"] == len(leaves) and st["internal_count"] == len(leaves) - 1


@test("STRUCT")
def mt_05_essential_is_oldest_and_never_child_of_lower_birth():
    img = gaussian_field(SHAPE, random_peaks(5))
    root, leaves = run_tree(img, pers_th=5.0)
    ess = [l for l in leaves if l.is_essential]
    assert len(ess) == 1
    assert ess[0].threshold == max(l.threshold for l in leaves)


@test("CORE")
def mt_06_leaf_localization_full_res():
    peaks = [(60, 60, 170.0, 14), (60, 250, 130.0, 14), (180, 80, 100.0, 14)]
    root, leaves = run_tree(gaussian_field(SHAPE, peaks))
    for ty, tx, _, _ in peaks:
        d = min(np.hypot(l.pixel[0] - ty, l.pixel[1] - tx) for l in leaves)
        assert d <= 2 * DOWNSCALE, f"peak ({ty},{tx}) localized {d:.0f}px away"


@test("CORE")
def mt_07_blob_enrichment_and_gate():
    img = gaussian_field(SHAPE, [(60, 60, 170.0, 14), (60, 250, 130.0, 14)])
    near = FakeBlob(centroid=(250, 60), mean_hue=77.0, direction="rising")
    far = FakeBlob(centroid=(10, 230), mean_hue=11.0, direction="sinking")
    _, leaves = run_tree(img, blobs=[near, far])
    fin = [l for l in leaves if not l.is_essential][0]
    assert fin.mean_hue == 77.0 and fin.direction == "rising"
    _, leaves = run_tree(img, blobs=[far])
    fin = [l for l in leaves if not l.is_essential][0]
    assert fin.mean_hue == 0.0 and fin.direction == "turning", "far blob must not enrich"


@test("CORE")
def mt_08_degenerate_inputs():
    flat = np.full(SHAPE, 40, dtype=np.uint8)
    root, leaves = run_tree(flat)
    assert isinstance(root, MergeTreeNode)
    assert len(leaves) <= 1 and all(l.is_essential for l in leaves), "flat field: at most the essential class"
    faint = gaussian_field(SHAPE, [(60, 60, 200.0, 14), (180, 240, 6.0, 20)])
    _, leaves = run_tree(faint)
    assert len([l for l in leaves if not l.is_essential]) == 0, "sub-threshold peak must not produce a leaf"


@test("STRUCT")
def mt_09_draw_overlay_runs():
    from merge_tree import draw_tree_overlay
    img = gaussian_field(SHAPE, random_peaks(4))
    root, _ = run_tree(img, pers_th=5.0)
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    draw_tree_overlay(canvas, root, origin=(0, 0))
    assert canvas.shape == (SHAPE[0], SHAPE[1], 3)


# ==========================================================================
# FZ -- fuzzing against the oracle
# ==========================================================================

@test("STRUCT")
def fz_01_random_fields_nesting_matches_oracle():
    mismatches = []
    for trial in range(25):
        n = int(RNG.integers(2, 8))
        img = gaussian_field(SHAPE, random_peaks(n))
        img = np.clip(img.astype(int) + RNG.integers(-3, 4, img.shape), 0, 255).astype(np.uint8)
        sm = downscaled(img)
        root, leaves = run_tree(img, pers_th=8.0)
        orc = OracleTree(sm)
        exp_pairs = sorted((r[0], r[1]) for r in orc.finite_pairs(8.0))
        got_pairs = sorted((l.threshold, l.death) for l in leaves if not l.is_essential)
        if exp_pairs != got_pairs:
            mismatches.append((trial, "pairs", exp_pairs, got_pairs)); continue
        pm = parent_map(root)
        kept = {small_yx(l): l.threshold for l in leaves}

        def canon(yx):
            """Map an oracle peak cell to the kept leaf on the same plateau.
            GUDHI and the oracle may pick different cells of an equal-valued
            plateau maximum; treat cells within 2 px at equal value as one."""
            if yx is None:
                return None
            if yx in kept:
                return yx
            v = sm[yx]
            for k, b in kept.items():
                if b == v and max(abs(k[0] - yx[0]), abs(k[1] - yx[1])) <= 2:
                    return k
            return None

        for leaf_yx, (partner_yx, level, saddle_yx) in pm.items():
            exp = orc.parent_of.get(leaf_yx)
            if exp is None:
                # leaf_yx may itself be the plateau-shifted cell: find the
                # oracle record by value/proximity
                for k in orc.parent_of:
                    if sm[k] == sm[leaf_yx] and max(abs(k[0]-leaf_yx[0]), abs(k[1]-leaf_yx[1])) <= 2:
                        exp = orc.parent_of[k]; break
            # a sub-threshold (or plateau-shifted) elder: walk up to a kept
            # leaf, skipping zero-persistence hops within the leaf's own plateau
            while exp is not None and (canon(exp) is None or canon(exp) == leaf_yx):
                exp = orc.parent_of.get(exp)
            if canon(exp) != partner_yx:
                # Tie: several clusters merging at the SAME level form one
                # n-ary node; any binary nesting of them is the same
                # topology. Accept if the oracle partner and the tree
                # partner are joined at exactly this level and the tree
                # partner is elder (higher birth) than the leaf.
                _, lbl = cv2.connectedComponents((sm >= level).astype(np.uint8), connectivity=8)
                ce = canon(exp)
                tie_ok = (ce is not None and lbl[ce] == lbl[partner_yx] == lbl[leaf_yx]
                          and kept[partner_yx] > kept[leaf_yx])
                if not tie_ok:
                    mismatches.append((trial, "nesting", leaf_yx, partner_yx, exp))
            if sm[saddle_yx] != level:
                mismatches.append((trial, "saddle-value", leaf_yx, sm[saddle_yx], level))
    assert not mismatches, f"{len(mismatches)} mismatch(es): {mismatches[:3]}"


@test("STRUCT")
def fz_02_saddle_connects_both_children():
    """At the saddle level the two child components are one 8-connected
    region containing the saddle cell; one level up they are separate."""
    for trial in range(10):
        img = gaussian_field(SHAPE, random_peaks(int(RNG.integers(2, 6))))
        sm = downscaled(img)
        root, _ = run_tree(img, pers_th=8.0)
        for n in iter_nodes(root):
            if n.is_leaf or len(n.children) != 2:
                continue
            lvl = n.threshold
            _, lbl = cv2.connectedComponents((sm >= lvl).astype(np.uint8), connectivity=8)
            a = small_yx(oldest_leaf_under(n.children[0]))
            b = small_yx(oldest_leaf_under(n.children[1]))
            s = small_yx(n)
            assert lbl[a] == lbl[b] == lbl[s] != 0, f"trial {trial}: children not joined at saddle"
            _, lbl2 = cv2.connectedComponents((sm > lvl).astype(np.uint8), connectivity=8)
            assert lbl2[a] != lbl2[b], f"trial {trial}: children already joined above saddle"


# ==========================================================================
# IV -- invariances
# ==========================================================================

@test("CORE")
def iv_01_noise_stability():
    peaks = [(60, 60, 170.0, 14), (60, 250, 130.0, 14), (180, 80, 100.0, 14)]
    img = gaussian_field(SHAPE, peaks)
    _, l0 = run_tree(img)
    eps = 6
    noisy = np.clip(img.astype(np.int16) + RNG.integers(-eps, eps + 1, size=img.shape),
                    0, 255).astype(np.uint8)
    _, l1 = run_tree(noisy)
    assert len(l1) == len(l0), f"leaf count changed under +/-{eps} noise: {len(l0)} -> {len(l1)}"
    for a, b in zip(sorted(l.persistence for l in l0), sorted(l.persistence for l in l1)):
        assert abs(a - b) <= 2 * eps + 4, f"persistence moved {abs(a-b):.1f} (bound ~{2*eps})"


@test("STRUCT")
def iv_02_monotone_transform_preserves_structure():
    img = gaussian_field(SHAPE, random_peaks(5))
    lut = (255.0 * (np.arange(256) / 255.0) ** 0.6).astype(np.uint8)
    r0, l0 = run_tree(img, pers_th=3.0)
    r1, l1 = run_tree(lut[img], pers_th=3.0)
    assert len(l0) == len(l1), f"gamma changed leaf count {len(l0)} -> {len(l1)}"
    rank0 = [small_yx(l) for l in sorted(l0, key=lambda l: -l.persistence)]
    rank1 = [small_yx(l) for l in sorted(l1, key=lambda l: -l.persistence)]
    assert rank0 == rank1, "persistence ranking not invariant under monotone transform"
    pm0 = {k: v[0] for k, v in parent_map(r0).items()}
    pm1 = {k: v[0] for k, v in parent_map(r1).items()}
    assert pm0 == pm1, "nesting changed under monotone transform"


@test("CORE")
def iv_03_temporal_coherence_leaf_and_saddle():
    prev_leaf = prev_saddle = None
    for x in np.linspace(80, 240, 25):
        img = gaussian_field(SHAPE, [(60, 40, 200.0, 12), (170, float(x), 120.0, 14)])
        root, leaves = run_tree(img)
        fin = [l for l in leaves if not l.is_essential]
        assert len(fin) == 1 and not root.is_leaf
        if prev_leaf is not None:
            jl = np.hypot(fin[0].pixel[0] - prev_leaf[0], fin[0].pixel[1] - prev_leaf[1])
            assert jl < 6 * DOWNSCALE, f"leaf jumped {jl:.0f}px between adjacent frames"
            js = np.hypot(root.pixel[0] - prev_saddle[0], root.pixel[1] - prev_saddle[1])
            # the saddle between a fixed and a moving peak drifts smoothly
            # (a 2-D gap between Gaussians can shift faster than the peak,
            # so the bound is looser than the leaf bound)
            assert js < 20 * DOWNSCALE, f"saddle jumped {js:.0f}px between adjacent frames"
        prev_leaf = fin[0].pixel
        prev_saddle = root.pixel


@test("STRUCT")
def iv_04_downscale_robustness():
    img = gaussian_field(SHAPE, [(60, 60, 170.0, 16), (60, 250, 130.0, 16), (180, 80, 100.0, 16)])
    counts = {}
    for ds in (2, 4, 8):
        _, leaves = compute_merge_tree(img, downscale=ds, persistence_threshold=PERS_TH)
        counts[ds] = len(leaves)
    assert len(set(counts.values())) == 1, f"leaf count depends on downscale: {counts}"


# ==========================================================================
# DT -- BlobDetector integration (synthetic wax frames)
# ==========================================================================

def _wax_frame(shape, peaks, hue=160, sat=120):
    """Value channel = gaussian field; hue inside the default profile's wax
    band; background value below wax_val_min so the colour mask isolates it."""
    val = gaussian_field(shape, peaks, base=40.0)
    hsv = np.zeros((*shape, 3), dtype=np.uint8)
    hsv[..., 0] = hue; hsv[..., 1] = sat; hsv[..., 2] = val
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _detector():
    from blob_detection import BlobDetector
    from lamp_profile import LampProfile
    return BlobDetector(profile=LampProfile.default(), persistence_threshold=PERS_TH,
                        downscale=DOWNSCALE, background="none")


@test("INTEG")
def dt_01_brightest_blob_is_detected():
    peaks = [(120, 160, 200.0, 18)]
    blobs = _detector().detect(_wax_frame(SHAPE, peaks))
    assert len(blobs) == 1, f"single (essential) blob must be detected, got {len(blobs)}"
    cx, cy = blobs[0].centroid
    assert np.hypot(cx - 160, cy - 120) < 3 * DOWNSCALE


@test("INTEG")
def dt_02_all_blobs_detected_nonsquare():
    # sigma 24 keeps each blob's above-wax_val_min disc well over the
    # default profile's blob_area_min (200 px) -- smaller sigmas fail on
    # min_area, not on topology.
    peaks = [(60, 60, 200.0, 24), (60, 250, 180.0, 24), (180, 80, 170.0, 24), (180, 240, 165.0, 24)]
    blobs = _detector().detect(_wax_frame(SHAPE, peaks))
    assert len(blobs) == 4, f"expected 4 blobs, got {len(blobs)}"
    for ty, tx, _, _ in peaks:
        d = min(np.hypot(b.centroid[0] - tx, b.centroid[1] - ty) for b in blobs)
        assert d < 3 * DOWNSCALE, f"blob at ({ty},{tx}) missing/mislocalized ({d:.0f}px)"
    assert blobs[0].persistence >= blobs[-1].persistence, "must be sorted by persistence desc"


@test("INTEG")
def dt_03_persistence_values_are_true_not_scrambled():
    """On the masked field each blob is an island over a zero floor, so every
    finite blob dies at 0 and persistence == birth; the essential blob (the
    brightest) ranks first; persistence order follows true height order."""
    peaks = [(60, 60, 200.0, 24), (60, 250, 175.0, 24), (180, 80, 150.0, 24)]
    blobs = _detector().detect(_wax_frame(SHAPE, peaks))
    assert len(blobs) == 3, f"expected 3 blobs, got {len(blobs)}"
    for b in blobs:
        assert b.death == 0.0, f"island blob must die at the mask floor, death={b.death}"
        assert abs(b.persistence - b.birth) < 1e-6
    # brightest first; centroid of the first blob is the tallest peak
    cx, cy = blobs[0].centroid
    assert np.hypot(cx - 60, cy - 60) < 3 * DOWNSCALE, "essential (brightest) blob must rank first"
    order = [min(range(3), key=lambda i: np.hypot(b.centroid[0] - peaks[i][1], b.centroid[1] - peaks[i][0]))
             for b in blobs]
    assert order == [0, 1, 2], f"persistence order {order} does not follow height order"


@test("INTEG")
def dt_04_no_spurious_blobs_on_smooth_field():
    peaks = [(120, 160, 200.0, 30)]
    blobs = _detector().detect(_wax_frame(SHAPE, peaks))
    assert len(blobs) == 1, f"smooth single peak produced {len(blobs)} blobs (stride bug signature)"


# ==========================================================================
# PR -- peak pruning to detected regions
# ==========================================================================

class _PB:
    def __init__(self, x, y, area=3000.0):
        self.centroid = (x, y); self.area = area
        self.orientation = 0.0; self.elongation = 2.0
        self.hu = [0.0] * 7; self.mean_hue = 30.0; self.direction = "turning"


def _noisy_scene():
    """Three wax blobs, sensor grain, and one bright non-wax reflection."""
    rng = np.random.default_rng(3)
    peaks = [(60, 60, 170.0, 16), (60, 250, 150.0, 16), (180, 80, 140.0, 16)]
    img = gaussian_field(SHAPE, peaks).astype(float)
    img += rng.normal(0, 7, SHAPE)
    img[30:36, 290:298] = 235                      # specular highlight on the glass
    img = np.clip(img, 0, 255).astype(np.uint8)
    blobs = [_PB(60, 60), _PB(250, 60), _PB(80, 180)]
    mask = np.zeros(SHAPE, np.uint8)
    for b in blobs:
        cv2.circle(mask, (int(b.centroid[0]), int(b.centroid[1])), 34, 255, -1)
    return img, blobs, mask


@test("CORE")
def pr_01_reflection_is_rejected():
    img, blobs, mask = _noisy_scene()
    _, l_raw = compute_merge_tree(img, downscale=DOWNSCALE, persistence_threshold=10.0,
                                  prune_to_blobs=False)
    _, l_cut = compute_merge_tree(img, blobs=blobs, mask=mask, downscale=DOWNSCALE,
                                  persistence_threshold=10.0)
    _, l_disc = compute_merge_tree(img, blobs=blobs, downscale=DOWNSCALE,
                                   persistence_threshold=10.0, prune_using_blobs=True)
    def has_reflection(leaves):
        return any(abs(x.pixel[1] - 294) < 25 and abs(x.pixel[0] - 33) < 25 for x in leaves)
    assert has_reflection(l_raw), "test scene should contain a spurious reflection peak"
    assert not has_reflection(l_cut), "mask pruning failed to reject the reflection"
    assert not has_reflection(l_disc), "disc pruning failed to reject the reflection"


@test("CORE")
def pr_02_real_blobs_survive():
    img, blobs, mask = _noisy_scene()
    _, leaves = compute_merge_tree(img, blobs=blobs, mask=mask, downscale=DOWNSCALE,
                                   persistence_threshold=10.0)
    for b in blobs:
        d = min(np.hypot(x.pixel[1] - b.centroid[0], x.pixel[0] - b.centroid[1]) for x in leaves)
        assert d < 40, f"blob at {b.centroid} lost a leaf (nearest {d:.0f}px)"


@test("CORE")
def pr_03_tree_stays_consistent_after_pruning():
    """Pruning happens at the PEAK level and the tree is rebuilt, so no
    internal node may reference a removed child."""
    img, blobs, mask = _noisy_scene()
    root, leaves = compute_merge_tree(img, blobs=blobs, mask=mask, downscale=DOWNSCALE,
                                      persistence_threshold=10.0)
    internal = [n for n in iter_nodes(root) if not n.is_leaf]
    assert len(internal) == len(leaves) - 1, \
        f"{len(leaves)} leaves should give {len(leaves)-1} internal nodes, got {len(internal)}"
    seen = {id(n) for n in iter_nodes(root)}
    for n in internal:
        for ch in n.children:
            assert id(ch) in seen, "internal node references a node outside the tree"
    assert sum(1 for l in leaves if l.is_essential) == 1, "exactly one essential leaf"


@test("CORE")
def pr_04_essential_is_promoted_if_rejected():
    """If the brightest peak is itself noise outside the wax, the brightest
    SURVIVOR must be promoted so the tree still has a root."""
    img, blobs, mask = _noisy_scene()
    img = img.copy(); img[30:36, 290:298] = 255          # reflection now brightest
    root, leaves = compute_merge_tree(img, blobs=blobs, mask=mask, downscale=DOWNSCALE,
                                      persistence_threshold=10.0)
    ess = [l for l in leaves if l.is_essential]
    assert len(ess) == 1, f"expected one essential leaf, got {len(ess)}"
    assert abs(ess[0].pixel[1] - 294) > 25, "the reflection was kept as the essential leaf"


@test("CORE")
def pr_05_pruning_is_opt_out_and_degrades_safely():
    img, blobs, _ = _noisy_scene()
    a = compute_merge_tree(img, downscale=DOWNSCALE, persistence_threshold=10.0,
                           prune_to_blobs=False)[1]
    b = compute_merge_tree(img, downscale=DOWNSCALE, persistence_threshold=10.0)[1]
    assert len(a) == len(b), "with no blobs and no mask, pruning must be a no-op"
    # passing `blobs` alone must NOT prune (enrichment-only callers)
    c = compute_merge_tree(img, blobs=blobs, downscale=DOWNSCALE,
                           persistence_threshold=10.0)[1]
    assert len(c) == len(a), "passing blobs for enrichment must not prune the tree"
    _, _, st = compute_merge_tree(img, blobs=blobs, downscale=DOWNSCALE,
                                  persistence_threshold=10.0, return_stats=True,
                                  prune_using_blobs=True)
    assert st["peaks_kept"] + st["peaks_pruned"] == st["peaks_raw"]
    assert 0.0 <= st["prune_rate"] <= 1.0


@test("CORE")
def pr_06_mask_beats_discs_on_an_elongated_blob():
    """A disc around the centroid misses the ends of a long blob; the real
    mask does not."""
    h, w = SHAPE
    img = np.full(SHAPE, 25, np.uint8)
    cv2.ellipse(img, (160, 120), (70, 16), 0, 0, 360, 210, -1)
    img = cv2.GaussianBlur(img, (0, 0), 6)
    tip = (int(160 + 58), 120)
    img[tip[1] - 3:tip[1] + 3, tip[0] - 3:tip[0] + 3] = 245     # peak near one tip
    blobs = [_PB(160, 120, area=np.pi * 20 ** 2)]               # disc r~20: misses the tip
    mask = np.zeros(SHAPE, np.uint8)
    cv2.ellipse(mask, (160, 120), (70, 16), 0, 0, 360, 255, -1)
    _, l_disc = compute_merge_tree(img, blobs=blobs, downscale=DOWNSCALE,
                                   persistence_threshold=8.0, prune_using_blobs=True)
    _, l_mask = compute_merge_tree(img, blobs=blobs, mask=mask, downscale=DOWNSCALE,
                                   persistence_threshold=8.0)
    near = lambda ls: min(np.hypot(x.pixel[1] - tip[0], x.pixel[0] - tip[1]) for x in ls)
    assert near(l_mask) <= near(l_disc) + 1e-9, \
        "the real mask should retain the tip peak at least as well as a disc"


# ==========================================================================
# PF -- performance
# ==========================================================================

@test("CORE")
def pf_01_merge_tree_budget():
    img = gaussian_field(SHAPE, random_peaks(6))
    run_tree(img, pers_th=5.0)
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        run_tree(img, pers_th=5.0)
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"      [perf] compute_merge_tree: {ms:.1f} ms/frame at {SHAPE} ds={DOWNSCALE}")
    assert ms < 50.0, f"{ms:.1f} ms/frame exceeds 50 ms budget"


@test("INTEG")
def pf_02_h0_stage_budget():
    sm = downscaled(gaussian_field(SHAPE, random_peaks(6)))
    h0_peaks(sm, PERS_TH)
    n = 30
    t0 = time.perf_counter()
    for _ in range(n):
        h0_peaks(sm, PERS_TH)
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"      [perf] h0_peaks: {ms:.1f} ms at {sm.shape}")
    assert ms < 25.0


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

if __name__ == "__main__":
    passed, failed = [], []
    for tag, fn in TESTS:
        name = fn.__name__
        try:
            fn()
            passed.append((tag, name)); print(f"PASS  [{tag:6}] {name}")
        except AssertionError as e:
            failed.append((tag, name, str(e))); print(f"FAIL  [{tag:6}] {name}\n      {e}")
        except Exception as e:
            failed.append((tag, name, f"ERROR {type(e).__name__}: {e}"))
            print(f"ERR   [{tag:6}] {name}\n      {type(e).__name__}: {e}")
    print(f"\n{len(passed)} passed, {len(failed)} failed / errored")
    for tag, label in (("CORE", "CORE failures -- fix before any governor work:"),
                       ("STRUCT", "STRUCT failures -- grouping/level hierarchy NOT yet licensed:"),
                       ("INTEG", "INTEG failures -- detection pipeline regression:")):
        fl = [f for f in failed if f[0] == tag]
        if fl:
            print(label)
            for _, n, _ in fl:
                print(f"  - {n}")
    sys.exit(1 if failed else 0)
