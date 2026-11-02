# merge_tree.py -- Audible Lava v4.2
# Merge tree of the greyscale luminance field, built from validated H0
# persistence (h0_persistence.py) plus explicit saddle-level connectivity.
#
# v4.2 vs v3.0 (fixes from test_merge_tree.py, Aug 2026):
#   * GUDHI dimension order corrected (was scrambling non-square frames).
#   * Leaf pixels come from GUDHI's exact birth cells, not first-pixel-at-
#     intensity matching (equal-height peaks no longer collapse).
#   * The essential class (brightest blob) is now a leaf. Previously the
#     dominant blob was absent from the tree; a 2-blob lamp produced a
#     single leaf and no internal node at all.
#   * Internal nodes are TRUE saddles: threshold == the level at which the
#     two child components actually connect, pixel == the saddle cell,
#     partner chosen by connectivity + elder rule -- not by popping a
#     death-sorted list. The tree is now a faithful superlevel-set
#     hierarchy and may be read as a grouping / Schenkerian level tree.
#   * Additive: `death`, `is_essential`, `depth`, `leaf_count`, `parent`
#     fields; `assign_depths`, `iter_nodes`, `tree_stats` helpers for the
#     score-governor layer. Public API of compute_merge_tree and
#     draw_tree_overlay is unchanged.

import cv2
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

from h0_persistence import h0_peaks, H0Peak


@dataclass
class MergeTreeNode:
    """
    A node in the blob merge tree.

    Leaf nodes → local intensity maxima (blob peaks).
    Internal nodes → saddle points where two components merge.
    Root → single surviving component at the lowest threshold.

    threshold : leaf -> birth intensity; internal -> saddle intensity
    pixel     : (y, x) FULL-RES frame coords (ROI-local in v4.1);
                leaf -> peak cell, internal -> saddle cell
    depth     : 0 at root, +1 per level toward the leaves
    """
    node_id:      int
    threshold:    float
    is_leaf:      bool
    children:     list
    persistence:  float = 0.0
    pixel:        tuple = (0, 0)
    # Shape attributes from matched PersistentBlob
    orientation:  float = 0.0
    elongation:   float = 0.0
    hu:           list  = field(default_factory=lambda: [0.0] * 7)
    # Thermodynamic attributes (v3.0)
    mean_hue:     float = 0.0
    direction:    str   = "turning"
    # v4.2 additive structural attributes
    death:        float = 0.0        # leaf: merge level; internal: == threshold
    is_essential: bool  = False      # leaf carrying the global maximum
    depth:        int   = 0
    leaf_count:   int   = 1
    parent:       Optional["MergeTreeNode"] = field(default=None, repr=False)


# ── Tree utilities (additive, used by score_governor) ────────────────────────

def iter_nodes(root: MergeTreeNode) -> Iterator[MergeTreeNode]:
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)


def assign_depths(root: MergeTreeNode) -> int:
    """Set depth/parent/leaf_count on every node. Returns max depth."""
    max_depth = 0
    stack = [(root, 0, None)]
    order: List[MergeTreeNode] = []
    while stack:
        n, d, p = stack.pop()
        n.depth, n.parent = d, p
        max_depth = max(max_depth, d)
        order.append(n)
        for c in n.children:
            stack.append((c, d + 1, n))
    for n in reversed(order):                # children before parents
        n.leaf_count = 1 if n.is_leaf else sum(c.leaf_count for c in n.children)
    return max_depth


def tree_stats(root: MergeTreeNode) -> dict:
    """Shape statistics for texture-level governance.

    colless : sum over internal nodes of |leaves(left) - leaves(right)|,
              0 for perfectly balanced, large for caterpillar trees.
    """
    max_depth = assign_depths(root)
    leaves = [n for n in iter_nodes(root) if n.is_leaf]
    internal = [n for n in iter_nodes(root) if not n.is_leaf]
    colless = sum(abs(n.children[0].leaf_count - n.children[1].leaf_count)
                  for n in internal if len(n.children) == 2)
    return {
        "leaf_count":     len(leaves),
        "internal_count": len(internal),
        "max_depth":      max_depth,
        "mean_leaf_depth": float(np.mean([n.depth for n in leaves])) if leaves else 0.0,
        "colless":        colless,
        "root_threshold": root.threshold,
    }


# ── Construction ─────────────────────────────────────────────────────────────

def _full_res(yx_small: Tuple[int, int], downscale: int) -> Tuple[int, int]:
    return (int(yx_small[0] * downscale + downscale // 2),
            int(yx_small[1] * downscale + downscale // 2))


def build_acceptance_mask(
    small_shape: Tuple[int, int],
    downscale: int,
    blobs: list = None,
    mask: np.ndarray = None,
    dilate_px: int = 6,
    min_radius_px: int = 8,
) -> Optional[np.ndarray]:
    """Region, at DOWNSCALED resolution, in which a peak is allowed to live.

    Peaks outside it are noise: sensor grain, a bright reflection on the glass,
    a hot spot in the backlight. They are genuine local maxima of the luminance
    field, so persistence cannot tell them from wax -- only detection can, and
    detection already knows. This lets the merge tree inherit that knowledge.

    Two sources, in order of preference:
      mask   the `combined` wax mask detection already computed and cached
             (BlobDetector.get_debug_masks(frame)["combined"]). Exact.
      blobs  fallback: a disc per blob, radius from its area, when no mask is
             available. Coarser but needs nothing new plumbed through.

    `dilate_px` (full-res) grows the region so a peak sitting a few pixels
    outside an eroded mask edge is still accepted; morphological opening in
    detection tends to shave exactly those pixels.

    Returns None when neither source is available, meaning "accept everything"
    -- pruning is opt-in and degrades to the old behaviour.
    """
    sh, sw = small_shape
    if mask is not None:
        m = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
        m = (m > 0).astype(np.uint8)
    elif blobs:
        m = np.zeros((sh, sw), dtype=np.uint8)
        for b in blobs:
            cx, cy = b.centroid
            r = max(min_radius_px,
                    math.sqrt(max(float(getattr(b, "area", 0.0)), 1.0) / math.pi))
            cv2.circle(m, (int(cx / downscale), int(cy / downscale)),
                       max(1, int(r / downscale)), 1, -1)
    else:
        return None
    d = max(1, int(round(dilate_px / max(downscale, 1))))
    if d > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
        m = cv2.dilate(m, k)
    return m


def prune_peaks(peaks: List[H0Peak], accept: Optional[np.ndarray]) -> Tuple[List[H0Peak], int]:
    """Drop peaks whose birth cell lies outside the accepted region.

    Pruning happens at the PEAK level, before the tree is built. Deleting nodes
    from an assembled tree would leave internal nodes pointing at dead
    children; rebuilding from the surviving peaks keeps the structure
    consistent by construction.

    Death levels of survivors remain valid: a peak's death is the level at
    which it stopped being a separate maximum of the ORIGINAL field, which does
    not change because some other peak was removed from consideration. The tree
    builder re-finds each survivor's partner among survivors by connected
    components at that level.

    If the essential (never-dying) peak is itself rejected, the brightest
    surviving peak is promoted to essential, so the tree still has a root.
    """
    if accept is None or not peaks:
        return peaks, 0
    sh, sw = accept.shape
    kept, dropped = [], 0
    for p in peaks:
        y, x = p.peak_yx
        if 0 <= y < sh and 0 <= x < sw and accept[y, x]:
            kept.append(p)
        else:
            dropped += 1
    if not kept:
        return [], dropped
    if not any(p.essential for p in kept):
        brightest = max(kept, key=lambda p: p.birth)
        i = kept.index(brightest)
        kept[i] = H0Peak(brightest.birth, brightest.death, brightest.persistence,
                         brightest.peak_yx, brightest.saddle_yx, essential=True)
    return kept, dropped


def _enrich_from_blobs(leaf: MergeTreeNode, blobs: list, match_radius: float = 30.0):
    """Copy shape/thermodynamic attrs from the nearest PersistentBlob."""
    if not blobs:
        return
    py, px = leaf.pixel
    closest = min(blobs, key=lambda b: np.hypot(b.centroid[0] - px, b.centroid[1] - py))
    if np.hypot(closest.centroid[0] - px, closest.centroid[1] - py) < match_radius:
        leaf.orientation = closest.orientation
        leaf.elongation  = closest.elongation
        leaf.hu          = closest.hu
        leaf.mean_hue    = closest.mean_hue
        leaf.direction   = closest.direction


def build_tree_from_peaks(
    peaks: List[H0Peak],
    small: np.ndarray,
    downscale: int = 4,
) -> Tuple[MergeTreeNode, List[MergeTreeNode]]:
    """
    Build the merge hierarchy over an already-thresholded set of H0 peaks.

    For each finite peak, in order of decreasing death level (i.e. the order
    in which components actually merge as the threshold descends), the
    superlevel set {small >= death} is labelled; the component containing
    the peak also contains exactly the older branch it merged into. The
    partner cluster is the one holding the highest-birth peak in that
    component (elder rule). Cluster roots are joined under an internal node
    whose threshold is the death level and whose pixel is GUDHI's death
    (saddle) cell.

    8-connectivity matches the cubical complex (diagonal pixels share a
    vertex whose filtration value is the min of its cofaces).
    """
    if not peaks:
        return MergeTreeNode(0, 0.0, False, []), []

    small_f = np.asarray(small)
    node_counter = 0
    leaves: List[MergeTreeNode] = []
    for p in peaks:
        leaves.append(MergeTreeNode(
            node_id      = node_counter,
            threshold    = p.birth,
            is_leaf      = True,
            children     = [],
            persistence  = p.persistence,
            pixel        = _full_res(p.peak_yx, downscale),
            death        = p.death,
            is_essential = p.essential,
        ))
        node_counter += 1

    if len(leaves) == 1:
        return leaves[0], leaves

    # Union-find over LEAF indices; cluster_root maps a UF root -> tree node.
    uf_parent = list(range(len(leaves)))
    cluster_node = {i: leaves[i] for i in range(len(leaves))}

    def find(i):
        while uf_parent[i] != i:
            uf_parent[i] = uf_parent[uf_parent[i]]
            i = uf_parent[i]
        return i

    peak_small = [p.peak_yx for p in peaks]
    births = [p.birth for p in peaks]

    # Finite peaks merge in order of decreasing death; equal deaths: younger
    # (lower birth) first so the elder is present when its junior arrives.
    order = sorted(
        (i for i, p in enumerate(peaks) if not p.essential),
        key=lambda i: (-peaks[i].death, peaks[i].birth),
    )

    for i in order:
        death = peaks[i].death
        # superlevel set at the merge level (>= death; death is an actual
        # pixel value, so subtracting 0.5 is exact for integer fields).
        _, binmask = cv2.threshold(small_f.astype(np.float32), float(death) - 0.5,
                                   255.0, cv2.THRESH_BINARY)
        _, lbl = cv2.connectedComponents(binmask.astype(np.uint8), connectivity=8)

        my_root = find(i)
        my_lbl = int(lbl[peak_small[i]])

        partner_root = None
        best_birth = -np.inf
        for j in range(len(leaves)):
            if find(j) == my_root:
                continue
            if int(lbl[peak_small[j]]) == my_lbl and births[j] > best_birth:
                best_birth = births[j]
                partner_root = find(j)

        if partner_root is None:
            # Should not happen for a consistent pair set; attach to the
            # essential cluster so the tree still closes.
            ess = [k for k, p in enumerate(peaks) if p.essential]
            if not ess or find(ess[0]) == my_root:
                continue
            partner_root = find(ess[0])

        child_a = cluster_node[my_root]
        child_b = cluster_node[partner_root]
        saddle = peaks[i].saddle_yx
        internal = MergeTreeNode(
            node_id   = node_counter,
            threshold = float(death),
            is_leaf   = False,
            children  = [child_a, child_b],
            pixel     = _full_res(saddle, downscale) if saddle is not None
                        else child_a.pixel,
            death     = float(death),
        )
        node_counter += 1

        # union: partner root absorbs (arbitrary; node mapping is what matters)
        uf_parent[my_root] = partner_root
        cluster_node[partner_root] = internal
        cluster_node.pop(my_root, None)

    roots = {find(i) for i in range(len(leaves))}
    if len(roots) == 1:
        root = cluster_node[next(iter(roots))]
    else:
        # Disconnected superlevel sets that never join (cannot happen for a
        # connected image with one essential class, but close the tree anyway).
        children = [cluster_node[r] for r in roots]
        root = MergeTreeNode(node_counter, float(min(c.threshold for c in children)),
                             False, children, pixel=children[0].pixel)
    assign_depths(root)
    return root, leaves


def compute_merge_tree(
    grey_frame:            np.ndarray,
    blobs:                 list  = None,
    downscale:             int   = 4,
    persistence_threshold: float = 10.0,
    *,
    mask:                  np.ndarray = None,
    prune_to_blobs:        bool  = True,
    prune_using_blobs:     bool  = False,
    prune_dilate_px:       int   = 6,
    return_stats:          bool  = False,
):
    """
    Compute the merge tree of the greyscale intensity image.

    Leaf nodes are matched to PersistentBlob objects (from the same frame)
    by pixel proximity and enriched with their shape + thermodynamic attrs.

    Args:
        grey_frame:            single-channel uint8 greyscale image (lamp ROI)
        blobs:                 list of PersistentBlob from the same frame
        downscale:             downsample factor before GUDHI (must match BlobDetector)
        persistence_threshold: minimum persistence for a finite leaf. The
                               essential (brightest) component is always kept.
        mask:                  full-res binary wax mask (detection's `combined`).
                               Peaks outside it are rejected as noise. Preferred
                               over `blobs` for pruning: it follows the actual
                               blob outline rather than a disc.
        prune_to_blobs:        set False to keep every peak (v4.6 behaviour).
        prune_using_blobs:     opt in to the coarse disc fallback when no mask
                               is available. Off by default so that passing
                               `blobs` purely for leaf enrichment does not
                               silently prune the tree.
        prune_dilate_px:       full-res dilation of the accepted region, so a
                               peak just outside an eroded mask edge survives.
        return_stats:          also return a dict with the prune counts, useful
                               for checking how much noise is being rejected.

    Returns:
        (root MergeTreeNode, list of leaf MergeTreeNode objects)
        Leaves are sorted by persistence descending; the essential leaf is
        included and flagged `is_essential`.
    """
    h, w = grey_frame.shape[:2]
    sh, sw = h // downscale, w // downscale
    small = cv2.resize(grey_frame, (sw, sh), interpolation=cv2.INTER_AREA)

    peaks = h0_peaks(small, min_persistence=persistence_threshold,
                     include_essential=True)

    # v4.7: prune peaks that are not inside a detected wax region. Sensor
    # grain, glass reflections and backlight hot spots are real local maxima of
    # the luminance field, so persistence alone cannot reject them -- but
    # detection already segments the wax, and the tree can inherit that.
    n_raw = len(peaks)
    dropped = 0
    # Pruning is triggered by a MASK, not by the presence of `blobs`: callers
    # pass `blobs` to enrich leaves with shape/thermodynamic attributes, and
    # silently pruning their tree to coarse discs around those blobs would
    # change the result under them. The disc fallback is opt-in.
    use_blobs = blobs if (prune_using_blobs and mask is None) else None
    if prune_to_blobs and (mask is not None or use_blobs):
        accept = build_acceptance_mask((sh, sw), downscale, blobs=use_blobs,
                                       mask=mask, dilate_px=prune_dilate_px)
        peaks, dropped = prune_peaks(peaks, accept)

    root, leaves = build_tree_from_peaks(peaks, small, downscale=downscale)

    if blobs:
        for leaf in leaves:
            _enrich_from_blobs(leaf, blobs)
    if return_stats:
        return root, leaves, {"peaks_raw": n_raw, "peaks_pruned": dropped,
                              "peaks_kept": len(peaks),
                              "prune_rate": dropped / max(n_raw, 1)}
    return root, leaves


def draw_tree_overlay(frame: np.ndarray, root: MergeTreeNode,
                      origin: tuple = (0, 0)) -> None:
    """Draw merge tree as connecting lines on the frame (in-place).

    Node pixel coordinates are ROI-local (v4.1); `origin` is the lamp ROI's
    top-left (x, y) and shifts every drawn point onto the full-frame overlay.
    Internal nodes now sit at true saddle cells, so each leaf is joined to
    the actual ridge-gap at which it merges; the essential leaf is ringed.
    """
    ox, oy = origin
    for node in iter_nodes(root):
        ny, nx = node.pixel
        if node.is_leaf:
            if node.is_essential:
                cv2.circle(frame, (nx + ox, ny + oy), 6, (255, 120, 80), 1, cv2.LINE_AA)
            continue
        cv2.circle(frame, (nx + ox, ny + oy), 3, (255, 120, 80), -1, cv2.LINE_AA)
        for child in node.children:
            cy, cx = child.pixel
            cv2.line(frame, (cx + ox, cy + oy), (nx + ox, ny + oy),
                     (255, 120, 80), 1, cv2.LINE_AA)
