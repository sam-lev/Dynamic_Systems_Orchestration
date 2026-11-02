# blob_detection.py -- Audible Lava v4.2
#
# v4.9 (Oct 2026) -- warm-lamp background capture is now REACHABLE:
#   background_reference.py has supported mode="warm" (per-pixel low
#   percentile over a long window, so a running lamp's wax is not baked into
#   the reference) since v4.3, and test_v43_fixes.py covers it, but this
#   constructor never exposed it: BackgroundReference was always built with
#   the cold-median defaults, so the warm path was dead in the live pipeline.
#   New kwargs bg_ref_mode / bg_ref_capture_secs / bg_ref_percentile /
#   bg_ref_warm_samples pass straight through, and reset_background(mode=)
#   lets a HUD key request a warm re-capture on a lamp that is already
#   running. Defaults are unchanged (cold median), so existing callers are
#   bit-identical.
#
# v4.2 (Aug 2026) -- H0 extraction moved to h0_persistence.h0_peaks():
#   * GUDHI dimension order fixed ([sw, sh], fastest-varying first). The
#     previous [sh, sw] read the field at the wrong stride for non-square
#     ROIs, scrambling every persistence value. RE-CALIBRATE pers_thresh
#     (lamp_profile expected_persistence_*) after adopting this version:
#     true persistence values differ from the scrambled ones the current
#     profile was tuned against.
#   * Peak cells come from GUDHI's exact birth cofaces (no first-pixel-at-
#     intensity ambiguity).
#   * The essential class (brightest blob) is now a candidate. It was
#     previously skipped (death == inf) and only got seeded by accident via
#     spurious pairs; with the stride fix it would otherwise vanish.
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
from h0_persistence import h0_peaks
from dataclasses import dataclass, field
from typing import Optional
from lamp_profile import LampProfile
from background_reference import BackgroundReference


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

    Convention: if `h_lo <= h_hi` the range is non-wrapping and the mask
    is simply [h_lo, h_hi]. If `h_lo > h_hi` the range wraps the 0/180
    boundary on the OpenCV hue circle and the mask is [h_lo, 180] ∪
    [0, h_hi]. This convention is produced by LampProfile.from_drawn_regions
    when the calibrated wax distribution straddles the wraparound (red wax).
    """
    if h_lo <= h_hi:
        # Non-wrapping range — single inRange call
        lower = np.array([h_lo, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_hi, 255,   255  ], dtype=np.uint8)
        return cv2.inRange(frame_hsv, lower, upper)

    # Wrapping range: [h_lo, 180] ∪ [0, h_hi]
    l1 = np.array([h_lo, s_min, v_min], dtype=np.uint8)
    u1 = np.array([180,  255,   255  ], dtype=np.uint8)
    l2 = np.array([0,    s_min, v_min], dtype=np.uint8)
    u2 = np.array([h_hi, 255,   255  ], dtype=np.uint8)
    m1 = cv2.inRange(frame_hsv, l1, u1)
    m2 = cv2.inRange(frame_hsv, l2, u2)
    return cv2.bitwise_or(m1, m2)


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
        bg_ref_mode:           "cold" (median of bg_ref_frames frames from a
                               settled lamp) or "warm" (per-pixel low
                               percentile over bg_ref_capture_secs, usable on
                               a lamp that is already running). See
                               background_reference.py for the rationale.
    """

    def __init__(
        self,
        profile:               Optional[LampProfile] = None,
        persistence_threshold: Optional[float]       = None,
        downscale:             int                   = 4,
        mog2_history:          int                   = 500,
        mog2_var_threshold:    float                 = 16.0,
        seed_radius:           int                   = 3,
        # ── Background-subtraction mode (v4.1) ────────────────────────────
        # "mog2"   : adaptive MOG2 (legacy; absorbs slow wax -> use a frozen
        #            or near-zero learning rate via detect(learning_rate=...)).
        # "frozen" : non-adaptive BackgroundReference captured once at startup
        #            from the cold lamp; never fades long-lived blobs.
        # "none"   : no background subtraction (colour mask only).
        background:        str   = "mog2",
        bg_ref_frames:     int   = 40,
        bg_ref_threshold:  int   = 18,
        bg_ref_settle:     int   = 10,
        bg_ref_method:     str   = "median",
        # ── Warm-lamp capture (v4.3 in background_reference; plumbed v4.9) ──
        bg_ref_mode:         str   = "cold",
        bg_ref_capture_secs: float = 90.0,
        bg_ref_percentile:   float = 25.0,
        bg_ref_warm_samples: int   = 270,
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

        self.background = background

        # MOG2 backend (kept for A/B comparison and back-compat). The history
        # and var threshold are stored so reset_background() can rebuild it.
        self._mog2_history       = mog2_history
        self._mog2_var_threshold = mog2_var_threshold
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history      = mog2_history,
            varThreshold = mog2_var_threshold,
            detectShadows= False,
        )

        # Frozen-reference backend (used when background == "frozen").
        self.bg_ref_mode = bg_ref_mode
        self.bg_ref = BackgroundReference(
            n_frames       = bg_ref_frames,
            diff_threshold = bg_ref_threshold,
            settle_frames  = bg_ref_settle,
            method         = bg_ref_method,
            mode           = bg_ref_mode,
            capture_secs   = bg_ref_capture_secs,
            percentile     = bg_ref_percentile,
            warm_samples   = bg_ref_warm_samples,
        )

        self._k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self._k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # ── Mask cache (v4.1 perf) ────────────────────────────────────────
        # detect() stashes the foreground / colour / combined masks it builds
        # so get_debug_masks() can return them without recomputing (no second
        # background pass, colour mask, or morphology on the display path).
        # Holds a reference to the source frame and compares with `is`, so a
        # stale cache is never served and id() can't be reused under us.
        self._mask_cache:       Optional[dict]       = None
        self._mask_cache_frame: Optional[np.ndarray] = None

    # ── Background-subtraction helpers ────────────────────────────────────────
    def _foreground_mask(
        self, grey: np.ndarray, blurred: np.ndarray,
        learning_rate: float, *, update: bool = True,
    ) -> np.ndarray:
        """
        Return the binary foreground mask for the active background mode.

        `update=False` is a non-consuming peek (does not advance MOG2's model
        or the frozen reference's capture) for debug-mask previews.
        """
        if self.background == "frozen":
            return self.bg_ref.apply(grey, update=update)
        if self.background == "none":
            return np.full(grey.shape[:2], 255, dtype=np.uint8)
        # default: MOG2
        lr = learning_rate if update else 0
        return self.bg_sub.apply(blurred, learningRate=lr)

    def background_ready(self) -> bool:
        """True once the background model is usable. For 'frozen' this means the
        reference has been captured; 'mog2'/'none' have no capture phase here
        (the caller manages any MOG2 warm-up timer itself)."""
        if self.background == "frozen":
            return self.bg_ref.ready
        return True

    def background_progress(self) -> tuple:
        """(collected, target) frames for the frozen capture; (1, 1) otherwise."""
        if self.background == "frozen":
            return self.bg_ref.progress
        return (1, 1)

    def background_status(self) -> str:
        """One-line HUD string for the frozen capture ('' for other modes)."""
        if self.background == "frozen":
            try:
                return self.bg_ref.status()
            except Exception:
                return ""
        return ""

    def reset_background(self, *, mode: Optional[str] = None) -> None:
        """Re-capture the frozen reference, or rebuild the MOG2 model.

        `mode` ("cold" | "warm") optionally switches the frozen capture mode
        for this re-capture, e.g. a HUD key requesting a warm re-capture on a
        lamp that is already running. None keeps the current mode.
        """
        if self.background == "frozen":
            if mode is not None:
                self.bg_ref_mode = mode
                self.bg_ref.reset(mode=mode)
            else:
                self.bg_ref.reset()
        else:
            self.bg_sub = cv2.createBackgroundSubtractorMOG2(
                history      = self._mog2_history,
                varThreshold = self._mog2_var_threshold,
                detectShadows= False,
            )

    def _classify_direction(self, mean_hue: float) -> str:
        """
        DEPRECATED in v3.2: direction classification has moved to BlobTracker
        and is now sourced from the Kalman filter's vy estimate (kinematic
        direction), not from hue. Hue proved camera-firmware-sensitive and
        weakly correlated with thermal phase in practice. This method is
        retained only so external callers don't break; it returns TURNING
        unconditionally so any caller still using it produces a neutral
        label rather than a confidently-wrong one.
        """
        return TURNING

    def get_debug_masks(self, frame: np.ndarray, *, allow_cache: bool = True,
                        grey: Optional[np.ndarray] = None) -> dict:
        """Return intermediate masks without advancing the background model.

        If detect() already ran on this exact frame and allow_cache is True,
        the cached masks are returned directly (skips a redundant background
        peek, colour mask, and morphology). Otherwise they are recomputed with
        a non-consuming background peek (update=False).

        `frame` must be the same image detect() receives (the lamp ROI). If a
        differently-sized frame is passed with a `grey` of another size the
        bitwise_and below raises; callers should pass the ROI.
        """
        if allow_cache and self._mask_cache is not None \
                and self._mask_cache_frame is frame:
            return self._mask_cache

        grey      = grey if grey is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred   = cv2.GaussianBlur(grey, (0, 0), sigmaX=5)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        p         = self.profile

        fg_mask     = self._foreground_mask(grey, blurred, 0, update=False)
        colour_mask = _make_wax_mask(frame_hsv,
                                     p.wax_hue_lo, p.wax_hue_hi,
                                     p.wax_sat_min, p.wax_val_min)
        combined    = cv2.bitwise_and(fg_mask, colour_mask)
        combined    = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self._k_close)
        combined    = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  self._k_open)
        # "mog2" key kept as a back-compat alias for the foreground mask.
        return {"foreground": fg_mask, "mog2": fg_mask,
                "colour": colour_mask, "combined": combined}

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
    def detect(self, frame: np.ndarray, learning_rate: float = -1,
               grey: Optional[np.ndarray] = None) -> list:
        """
        Full detection on one BGR frame (the lamp ROI in v4.1).
        Returns list of PersistentBlob sorted by persistence descending.

        Args:
            frame:         BGR frame, already cropped to the lamp ROI.
            learning_rate: MOG2 background-update rate (only used when
                           background == "mog2"; ignored for "frozen"/"none").
                           -1 = auto, 0.0 = frozen, ~1e-4 = near-zero update.
            grey:          optional precomputed greyscale of `frame`. The
                           runtime computes BGR→GRAY once and shares it across
                           the detector, tracker, and merge tree; pass it to
                           avoid a redundant cvtColor.
        """
        h, w = frame.shape[:2]
        sh   = h // self.downscale
        sw   = w // self.downscale
        p    = self.profile

        # ── 1-4: Pre-process & build combined wax mask ────────────────────
        grey      = grey if grey is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred   = cv2.GaussianBlur(grey, (0, 0), sigmaX=5)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        fg_mask     = self._foreground_mask(grey, blurred, learning_rate)
        colour_mask = _make_wax_mask(frame_hsv,
                                     p.wax_hue_lo, p.wax_hue_hi,
                                     p.wax_sat_min, p.wax_val_min)
        combined    = cv2.bitwise_and(fg_mask, colour_mask)
        combined    = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self._k_close)
        combined    = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  self._k_open)

        # Stash masks so get_debug_masks() can reuse them for this frame.
        self._mask_cache       = {"foreground": fg_mask, "mog2": fg_mask,
                                  "colour": colour_mask, "combined": combined}
        self._mask_cache_frame = frame

        # Greyscale restricted to wax pixels (C-side AND, no boolean broadcast).
        masked_grey = cv2.bitwise_and(grey, grey, mask=combined)

        # ── 5: GUDHI persistence on downsampled inverted image ────────────
        # INTER_AREA already low-passes as it downsamples; a small blur on the
        # *small* grid (not the full-res frame) removes residual sensor-noise
        # peaks far more cheaply than the previous full-res GaussianBlur.
        small    = cv2.resize(masked_grey, (sw, sh), interpolation=cv2.INTER_AREA)
        small    = cv2.GaussianBlur(small, (3, 3), sigmaX=1.0)

        # v4.2: validated H0 extraction (correct stride, exact peak cells,
        # essential class included with death = field minimum, i.e. 0 for
        # the masked image -> its persistence == its peak brightness).
        peaks = h0_peaks(small, min_persistence=self.pers_thresh,
                         include_essential=True)

        # ── 6: Collect raw candidates passing pers_thresh ─────────────────
        candidates: list = []
        for pk in peaks:
            py_s, px_s = pk.peak_yx
            candidates.append({
                'birth':         pk.birth,
                'death':         pk.death,
                'persistence':   pk.persistence,
                'essential':     pk.essential,
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

            # Direction is no longer set from hue at detection time. Each
            # PersistentBlob ships with a neutral TURNING label; BlobTracker
            # overwrites it from the Kalman vy estimate once the blob has
            # been matched and the filter has accumulated motion evidence.
            direction = TURNING

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
