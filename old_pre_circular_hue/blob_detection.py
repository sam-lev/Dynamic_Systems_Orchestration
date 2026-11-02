# blob_detection.py -- Audible Lava v3.1
# Detection pipeline:
#   MOG2 foreground mask ∩ wax colour HSV mask
#   → GUDHI H0 persistence
#   → Sublevel-set branch deduplication                  (Option 2, NEW v3.1)
#   → Watershed segmentation seeded by surviving peaks   (Option 3, NEW v3.1)
#   → Contour moment enrichment (orientation, elongation, hu)
#   → Circular mean hue per blob
#   → Thermodynamic direction classification (rising / falling / turning)
#
# v3.1 vs v3.0:
#   The centroid-in-polygon `deduplicate_by_containment` filter is removed.
#   It is replaced by two topologically grounded steps applied in sequence:
#
#     1. Sublevel-set branch dedup (`_deduplicate_by_branch`)
#        Candidates are processed in descending persistence (elder rule).
#        For each candidate, the downsampled image is thresholded just
#        below the candidate's death level. At that threshold the
#        candidate has *just* merged with its parent branch on the merge
#        tree. Connected components are computed on the binary mask.
#        If the candidate's peak shares a component with any already-kept
#        seed's peak, the candidate is on a kept branch and is dropped.
#
#     2. Watershed segmentation (`_watershed_segment`)
#        Surviving peaks become markers. Non-wax pixels are pre-labeled
#        as background. cv2.watershed floods the inverted greyscale so
#        every wax pixel is assigned to exactly one seed's basin.
#        This eliminates the per-pair cv2.threshold + connectedComponents
#        calls of v3.0, gives non-overlapping regions, and naturally
#        handles cases where one connected wax mass contains many H0
#        peaks (only the highest-persistence peak survives dedup, and
#        watershed assigns the entire mass to it).
#
# BlobDetector is stateful: one instance must persist across all frames.

import cv2
import numpy as np
import gudhi
from dataclasses import dataclass, field
from typing import Optional
from lamp_profile import LampProfile


# ── Direction constants ────────────────────────────────────────────────────────
RISING  = "rising"
FALLING = "falling"
TURNING = "turning"


@dataclass
class PersistentBlob:
    """
    A blob extracted from the H0 persistence diagram, enriched with
    contour moment shape descriptors and thermodynamic state.

    Topological attributes (GUDHI):
        birth       -- intensity at which blob peak appears
        death       -- intensity at which blob merges
        persistence -- birth - death

    Shape attributes (OpenCV contour moments):
        orientation  -- principal axis angle in radians [-π/2, π/2]
        aspect_ratio -- long_axis / short_axis
        elongation   -- 1 - short/long  (0=circle, 1=line)
        hu           -- 7 Hu moment invariants (log-scaled, sign-preserved)

    Thermodynamic attributes:
        mean_hue  -- circular mean H-channel value (0–180 OpenCV)
        direction -- 'rising' | 'falling' | 'turning'
    """
    centroid:     tuple
    area:         float
    birth:        float
    death:        float
    persistence:  float
    contour:      np.ndarray
    orientation:  float = 0.0
    aspect_ratio: float = 1.0
    elongation:   float = 0.0
    hu:           list  = field(default_factory=lambda: [0.0] * 7)
    mean_hue:     float = 0.0
    direction:    str   = TURNING


def _circular_mean_hue(hue_vals: np.ndarray) -> float:
    """
    Compute circular mean of HSV H values (0–180 OpenCV scale).
    Avoids the wrap-around error of arithmetic mean near H=0/180.
    """
    if len(hue_vals) == 0:
        return 0.0
    # Map [0,180] to [0,360] for circular arithmetic
    angles   = np.radians(hue_vals.astype(np.float64) * 2.0)
    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))
    mean_deg = np.degrees(np.arctan2(sin_mean, cos_mean))
    return float(mean_deg / 2.0 % 180.0)


def _make_wax_mask(
    frame_hsv: np.ndarray,
    h_lo: int, h_hi: int,
    s_min: int, v_min: int,
) -> np.ndarray:
    """
    Binary mask selecting pixels within the configured wax HSV range.
    Handles hue wrap-around at 0/180 for red-adjacent wax colours.
    """
    lower = np.array([h_lo, s_min, v_min], dtype=np.uint8)
    upper = np.array([h_hi, 255,   255  ], dtype=np.uint8)
    mask  = cv2.inRange(frame_hsv, lower, upper)
    if h_lo <= 5:
        wl = np.array([175, s_min, v_min], dtype=np.uint8)
        wu = np.array([180, 255,   255  ], dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(frame_hsv, wl, wu))
    return mask


class BlobDetector:
    """
    Stateful detector: MOG2 ∩ wax-colour mask → GUDHI persistence
    → branch dedup → watershed segmentation.

    Accepts a LampProfile for all colour and geometry parameters.
    Enriches each detected blob with mean hue and thermodynamic direction.

    Pipeline per frame:
        1. Greyscale blur
        2. MOG2 foreground mask
        3. Wax colour HSV mask
        4. Intersection + morphological clean-up
        5. GUDHI CubicalComplex H0 persistence on downsampled image
        6. Filter raw candidates by self.pers_thresh
        7. Sublevel-set branch dedup (Option 2)
        8. Watershed segmentation seeded by surviving peaks (Option 3)
        9. Contour moments per watershed region
       10. Circular mean hue per region
       11. Thermodynamic direction classification

    Args:
        profile:               LampProfile (colour + geometry parameters)
        persistence_threshold: minimum H0 persistence to keep a feature.
                               Defaults to profile.expected_persistence_mean - 0.5σ.
        downscale:             GUDHI grid size = (h/ds) × (w/ds).
                               At 1280×720: ds=4 → 58K cells (~15–40ms OK)
                                            ds=6 → 26K cells (~8–15ms fast)
        seed_radius:           radius (full-res pixels) of the disc-shaped
                               watershed marker placed at each surviving
                               H0 peak. Only pixels of this disc that lie
                               inside the wax mask receive the seed label.
    """

    def __init__(
        self,
        profile:               Optional[LampProfile] = None,
        persistence_threshold: Optional[float]       = None,
        downscale:             int                   = 4,
        mog2_history:          int                   = 500,
        mog2_var_threshold:    float                 = 16.0,
        seed_radius:           int                   = 3,
    ):
        self.profile     = profile or LampProfile.default()
        self.downscale   = downscale
        self.min_area    = int(self.profile.blob_area_min)
        self.seed_radius = max(1, int(seed_radius))

        if persistence_threshold is not None:
            self.pers_thresh = persistence_threshold
        else:
            # Use profile mean - 0.5σ so we don't miss smaller blobs
            self.pers_thresh = max(
                8.,
                self.profile.expected_persistence_mean
                - 0.5 * self.profile.expected_persistence_std,
            )

        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history      = mog2_history,
            varThreshold = mog2_var_threshold,
            detectShadows= False,
        )
        self._k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self._k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def _classify_direction(self, mean_hue: float) -> str:
        p = self.profile
        if mean_hue > p.rising_hue_threshold:
            return RISING
        elif mean_hue < p.falling_hue_threshold:
            return FALLING
        return TURNING

    def get_debug_masks(self, frame: np.ndarray) -> dict:
        """Return intermediate masks without updating MOG2 model."""
        grey      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred   = cv2.GaussianBlur(grey, (0, 0), sigmaX=5)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        p         = self.profile

        mog2_mask   = self.bg_sub.apply(blurred, learningRate=0)
        colour_mask = _make_wax_mask(frame_hsv,
                                     p.wax_hue_lo, p.wax_hue_hi,
                                     p.wax_sat_min, p.wax_val_min)
        combined    = cv2.bitwise_and(mog2_mask, colour_mask)
        combined    = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self._k_close)
        combined    = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  self._k_open)
        return {"mog2": mog2_mask, "colour": colour_mask, "combined": combined}

    # ── Option 2: Sublevel-set branch deduplication ───────────────────────────
    def _deduplicate_by_branch(self, candidates: list, small: np.ndarray) -> list:
        """
        Filter same-branch sub-features from H0 candidates.

        Candidates MUST be sorted by persistence descending so the elder
        rule applies — older (higher-persistence) features are kept,
        younger same-branch features are discarded.

        For each candidate we threshold `small` at one less than the
        candidate's death level. cv2.threshold with cv2.THRESH_BINARY
        keeps pixels strictly greater than the threshold, so this
        captures all pixels with intensity ≥ death — i.e. the merged
        component including the saddle at which this branch joined its
        parent. cv2.connectedComponents then tells us which pixels
        belong to that component.

        If any already-kept seed's peak pixel lies in the same component
        as this candidate's peak, this candidate has merged with a kept
        branch by the time we reach this threshold and is therefore
        a sub-feature on that branch — drop it.

        Args:
            candidates: list of dicts with keys 'birth', 'death',
                        'persistence', 'peak_yx_small', 'peak_yx_full'.
                        Must be sorted by persistence descending.
            small:      downsampled (sh, sw) uint8 greyscale image.

        Returns:
            Filtered subset of `candidates`, in original (sorted) order.
        """
        sh, sw = small.shape[:2]
        kept: list           = []
        kept_peaks_yx: list  = []   # (py, px) in downsampled coords

        for cand in candidates:
            py, px = cand['peak_yx_small']
            if not (0 <= py < sh and 0 <= px < sw):
                continue

            # Threshold just below death so the candidate is connected with
            # its parent at the saddle (saddle pixel intensity == death).
            thresh_level = max(int(cand['death']) - 1, 0)
            _, binmask   = cv2.threshold(
                small, thresh_level, 255, cv2.THRESH_BINARY
            )
            _, lbl       = cv2.connectedComponents(binmask)

            my_label = int(lbl[py, px])
            if my_label == 0:
                # Peak landed in the zero region (sub-pixel boundary case);
                # cannot reason about its branch — drop conservatively.
                continue

            # Same-branch test: O(1) per kept peak, no extra mask ops.
            shares_branch = any(
                int(lbl[kpy, kpx]) == my_label
                for kpy, kpx in kept_peaks_yx
            )
            if shares_branch:
                continue

            kept.append(cand)
            kept_peaks_yx.append((py, px))

        return kept

    # ── Option 3: Watershed segmentation seeded by surviving peaks ────────────
    def _watershed_segment(
        self,
        masked_grey:    np.ndarray,
        combined_mask:  np.ndarray,
        seeds_full_yx:  list,
    ):
        """
        Run a single full-resolution watershed pass.

        Background (non-wax) pixels are pre-labeled with label 1 so the
        watershed cannot leak the seed labels into the surrounding frame.
        Each surviving H0 peak is given a disc-shaped marker with label
        i + 2 (i = seed index), restricted to wax-mask pixels so the
        marker never sits on top of the background label.

        cv2.watershed expects a 3-channel BGR image. We pass the inverted
        greyscale: blob peaks become wells and the dark gaps between
        adjacent blobs become bright ridges, so flooding from each
        seed terminates at the natural inter-blob ridge or the wax mask
        boundary, whichever comes first.

        Args:
            masked_grey:   full-resolution uint8 image, brighter inside blobs.
            combined_mask: full-resolution uint8 wax mask (0 outside, 255 inside).
            seeds_full_yx: list of (cy, cx) full-res seed coords, in the
                           same order as the dedup'd candidates.

        Returns:
            (markers, placed_labels)
              markers:        int32 label image of same shape as masked_grey.
                              -1 = watershed boundary, 1 = background,
                              i+2 = region for seed i.
              placed_labels:  list of length len(seeds_full_yx). Each entry
                              is the label assigned to that seed (i+2), or
                              None if the seed could not be placed inside
                              the wax mask.
        """
        h, w     = masked_grey.shape[:2]
        markers  = np.zeros((h, w), dtype=np.int32)

        # Background marker — prevents watershed leaking out of the wax.
        markers[combined_mask == 0] = 1

        radius                  = self.seed_radius
        placed_labels: list     = []

        for i, (cy, cx) in enumerate(seeds_full_yx):
            label = i + 2
            if not (0 <= cy < h and 0 <= cx < w):
                placed_labels.append(None)
                continue

            # Build disc within bounding box, restricted to wax pixels.
            y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
            x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
            ys, xs       = np.ogrid[y0:y1, x0:x1]
            disc_in_box  = (ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2
            sub_in_mask  = combined_mask[y0:y1, x0:x1] > 0
            valid        = disc_in_box & sub_in_mask

            if not np.any(valid):
                placed_labels.append(None)
                continue

            markers[y0:y1, x0:x1][valid] = label
            placed_labels.append(label)

        # If no seed markers were placed, watershed still runs harmlessly
        # — every wax pixel will end up labeled 1 (background) or -1
        # (boundary), and the caller will produce no PersistentBlobs.
        inverted_full = (255 - masked_grey).astype(np.uint8)
        inverted_bgr  = cv2.cvtColor(inverted_full, cv2.COLOR_GRAY2BGR)
        cv2.watershed(inverted_bgr, markers)

        return markers, placed_labels

    # ── Main detect entry point ───────────────────────────────────────────────
    def detect(self, frame: np.ndarray, learning_rate: float = -1) -> list:
        """
        Full detection on one BGR frame.
        Returns list of PersistentBlob sorted by persistence descending.

        Args:
            frame:         BGR camera frame
            learning_rate: MOG2 background-update rate passed directly to
                           cv2.BackgroundSubtractorMOG2.apply().
                           -1      = auto (1/min(frameCount, history)); default
                                     suitable for the warm-up period where we
                                     want MOG2 to converge to a stable model.
                           0.0     = freeze the model entirely (no updates).
                           ~1e-4   = near-zero update; use after warm-up so
                                     stationary wax pooling at lamp top/bottom
                                     is NOT absorbed into the background, which
                                     would otherwise cause blobs to disappear
                                     at turning points and fire spurious
                                     death/birth events.
        """
        h, w = frame.shape[:2]
        sh   = h // self.downscale
        sw   = w // self.downscale
        p    = self.profile

        # ── 1-4: Pre-process & build combined wax mask ────────────────────
        grey      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred   = cv2.GaussianBlur(grey, (0, 0), sigmaX=5)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mog2_mask   = self.bg_sub.apply(blurred, learningRate=learning_rate)
        colour_mask = _make_wax_mask(frame_hsv,
                                     p.wax_hue_lo, p.wax_hue_hi,
                                     p.wax_sat_min, p.wax_val_min)
        combined    = cv2.bitwise_and(mog2_mask, colour_mask)
        combined    = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self._k_close)
        combined    = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  self._k_open)

        masked_grey = np.where(combined > 0, grey, 0).astype(np.uint8)
        # Blur before GUDHI to suppress sensor noise peaks.
        masked_grey = cv2.GaussianBlur(masked_grey, (0, 0), sigmaX=5)
        # Re-apply mask after blur (blur spreads zero-boundary inward).
        masked_grey = np.where(combined > 0, masked_grey, 0).astype(np.uint8)

        # ── 5: GUDHI persistence on downsampled inverted image ────────────
        small    = cv2.resize(masked_grey, (sw, sh), interpolation=cv2.INTER_AREA)
        inverted = 255.0 - small.astype(np.float64)

        cc = gudhi.CubicalComplex(
            top_dimensional_cells=inverted.flatten(),
            dimensions=[sh, sw],
        )
        cc.persistence()
        pairs = cc.persistence_intervals_in_dimension(0)

        # ── 6: Collect raw candidates passing pers_thresh ─────────────────
        candidates: list = []
        for birth_inv, death_inv in pairs:
            if death_inv == float("inf"):
                continue
            birth_real = 255.0 - birth_inv
            death_real = 255.0 - death_inv
            pers       = birth_real - death_real
            if pers < self.pers_thresh:
                continue

            peak_mask   = np.abs(inverted - birth_inv) < 1.0
            peak_coords = np.argwhere(peak_mask)
            if len(peak_coords) == 0:
                continue
            py_s, px_s = peak_coords[0]

            candidates.append({
                'birth':         birth_real,
                'death':         death_real,
                'persistence':   pers,
                'peak_yx_small': (int(py_s), int(px_s)),
                'peak_yx_full':  (
                    int(py_s * self.downscale + self.downscale // 2),
                    int(px_s * self.downscale + self.downscale // 2),
                ),
            })

        if not candidates:
            return []

        # Sort high → low persistence (elder rule for branch dedup).
        candidates.sort(key=lambda c: c['persistence'], reverse=True)

        # ── 7: Sublevel-set branch deduplication (Option 2) ───────────────
        seeds = self._deduplicate_by_branch(candidates, small)
        if not seeds:
            return []

        # ── 8: Watershed segmentation (Option 3) ──────────────────────────
        seeds_full_yx = [c['peak_yx_full'] for c in seeds]
        markers, placed_labels = self._watershed_segment(
            masked_grey, combined, seeds_full_yx
        )

        # ── 9-11: Build PersistentBlob from each watershed region ─────────
        blobs: list = []
        for i, cand in enumerate(seeds):
            label = placed_labels[i]
            if label is None:
                continue

            region = (markers == label).astype(np.uint8)
            area   = int(region.sum())
            if area < self.min_area:
                continue

            contours, _ = cv2.findContours(
                region * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)

            # Centroid from region moments (more stable than peak pixel)
            Mr = cv2.moments(region * 255)
            if Mr["m00"] > 0:
                cx_b = Mr["m10"] / Mr["m00"]
                cy_b = Mr["m01"] / Mr["m00"]
            else:
                cy_b, cx_b = cand['peak_yx_full']

            # Shape moments from contour
            orientation  = 0.0
            aspect_ratio = 1.0
            elongation   = 0.0
            hu           = [0.0] * 7

            if len(contour) >= 5:
                Mc = cv2.moments(contour)
                if Mc["m00"] > 0:
                    num = 2.0 * Mc["mu11"]
                    den = Mc["mu20"] - Mc["mu02"]
                    orientation = 0.5 * np.arctan2(num, den) if den != 0 else 0.0
                (_, _), (ea, eb), _ = cv2.fitEllipse(contour)
                long_ax      = max(ea, eb)
                short_ax     = min(ea, eb) + 1e-6
                aspect_ratio = long_ax / short_ax
                elongation   = float(np.clip(1.0 - short_ax / long_ax, 0.0, 1.0))
                hu_raw = cv2.HuMoments(cv2.moments(contour)).flatten()
                hu = [
                    float(-np.copysign(1.0, v) * np.log10(abs(v) + 1e-10))
                    for v in hu_raw
                ]

            # Mean hue from this region's pixels
            hue_pixels = frame_hsv[region > 0]
            if len(hue_pixels) > 0:
                mean_hue = _circular_mean_hue(hue_pixels[:, 0])
            else:
                mean_hue = float(self.profile.observed_hue_mean)

            direction = self._classify_direction(mean_hue)

            blobs.append(PersistentBlob(
                centroid     = (float(cx_b), float(cy_b)),
                area         = float(area),
                birth        = cand['birth'],
                death        = cand['death'],
                persistence  = cand['persistence'],
                contour      = contour,
                orientation  = float(orientation),
                aspect_ratio = float(aspect_ratio),
                elongation   = elongation,
                hu           = hu,
                mean_hue     = mean_hue,
                direction    = direction,
            ))

        # Already in descending-persistence order from the seed list,
        # but re-sort defensively in case future filters reorder anything.
        blobs.sort(key=lambda b: b.persistence, reverse=True)
        return blobs
