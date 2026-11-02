#!/usr/bin/env python3
# diagnose_tracking.py -- Audible Lava v3.1
#
# Methodical diagnostic harness for the detection + tracking pipeline.
# Runs for a fixed number of frames after MOG2 warm-up, collects per-
# frame measurements for each known failure mode, and prints a
# structured summary report at the end. A JSON dump of the report is
# written to diagnostic_report.json.
#
# Failure modes tested:
#   1. MOG2 background absorption of stationary/slow blobs
#   2. Farneback flow returning ~0 inside textureless wax interiors
#   3. Watershed mask shape volatility for elongating/collapsing blobs
#   4. Mask refinement blend ghost (parameter inspection only)
#   5. cv2.watershed -1 boundaries fragmenting thin elongated regions
#   6. wax_sat_min cutoff excluding low-saturation wax pixels
#   7. Tracking outcomes broken down by blob motion class
#
# Usage:
#   python diagnose_tracking.py [--frames N] [--warmup N] [--no-display]
#   python diagnose_tracking.py --video path/to/recording.mp4
#
# Press q in the live window to stop early; the report still prints.

import argparse
import cv2
import json
import numpy as np
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from lamp_profile   import LampProfile, PROFILE_PATH
from blob_detection import BlobDetector
from blob_tracker   import BlobTracker


# ── Per-stage timing instrumentation ──────────────────────────────────────────
# Three layers of timing: (a) module-level monkey-patch of gudhi.CubicalComplex
# to time the dominant suspected bottleneck; (b) wrapper around the detector's
# bg_sub to time MOG2.apply; (c) DiagnosticDetector / DiagnosticTracker
# subclasses that expose dedup / watershed / Farneback timings. All timings
# are recorded in milliseconds via time.perf_counter (~50ns overhead per
# call, negligible compared to the per-frame work being measured).

import gudhi as _gudhi_mod
_LAST_GUDHI_MS:        list = [0.0]
_orig_CubicalComplex        = _gudhi_mod.CubicalComplex


class _TimedCubicalComplex:
    """
    Drop-in wrapper around gudhi.CubicalComplex that records the time
    spent in .persistence() to _LAST_GUDHI_MS[0] every call. Other attribute
    access falls through to the real object via __getattr__, so existing
    code that calls .persistence_intervals_in_dimension() etc. is unaffected.

    Activation is via module-level reassignment of gudhi.CubicalComplex
    below. blob_detection.py looks up gudhi.CubicalComplex at call time
    rather than import time, so the wrapper takes effect even though
    blob_detection has already been imported.
    """
    def __init__(self, *args, **kwargs):
        self._real = _orig_CubicalComplex(*args, **kwargs)

    def persistence(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = self._real.persistence(*args, **kwargs)
        _LAST_GUDHI_MS[0] = (time.perf_counter() - t0) * 1000.0
        return out

    def __getattr__(self, name):
        return getattr(self._real, name)


_gudhi_mod.CubicalComplex = _TimedCubicalComplex


class _TimingMOG2:
    """
    Wraps a cv2.BackgroundSubtractorMOG2 to time .apply() calls. Other
    attribute access (getHistory, setVarThreshold, etc.) falls through
    to the real MOG2 via __getattr__.

    The owner is a DiagnosticDetector instance; the wrapper writes the
    measured milliseconds onto owner.last_mog2_ms so the harness can pick
    it up after each detect() call.
    """
    def __init__(self, real_bg_sub, owner):
        self._real  = real_bg_sub
        self._owner = owner

    def apply(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = self._real.apply(*args, **kwargs)
        self._owner.last_mog2_ms = (time.perf_counter() - t0) * 1000.0
        return out

    def __getattr__(self, name):
        return getattr(self._real, name)


# ── Configuration ─────────────────────────────────────────────────────────────
CAMERA_INDEX               = 0
DEFAULT_TOTAL_FRAMES       = 600     # ~20s @ 30fps measurement window
DEFAULT_WARMUP_FRAMES      = 90      # MOG2 background-model warm-up
FAST_MOTION_PX_PER_FRAME   = 8.0     # threshold for "fast" motion class
SLOW_MOTION_PX_PER_FRAME   = 2.0     # threshold for "slow" motion class
INTERIOR_ERODE_KERNEL_PX   = 9       # erosion kernel for "interior" of mask
EDGE_BAND_KERNEL_PX        = 9       # dilation kernel for "edge band" of mask
VOLATILITY_RATIO_RED_FLAG  = 3.0     # frame-to-frame area ratio that breaks IoU


# ── Instrumented detector ─────────────────────────────────────────────────────
class DiagnosticDetector(BlobDetector):
    """
    Wraps BlobDetector to capture intermediate state per frame and to
    record per-stage timings:

      State:
        * raw H0 candidate count (post pers_thresh)
        * surviving seed count (post branch dedup)
        * watershed markers, seed positions, placed labels

      Timings (milliseconds, populated each detect() call):
        * last_mog2_ms      — bg_sub.apply
        * last_dedup_ms     — _deduplicate_by_branch
        * last_watershed_ms — _watershed_segment
        * last_gudhi_ms     — gudhi.CubicalComplex.persistence (read from
                              the module-level _LAST_GUDHI_MS recorder
                              after super().detect() returns)
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._reset()
        # Wrap bg_sub so its .apply() calls are timed. After this line,
        # self.bg_sub is a _TimingMOG2 proxy; the underlying MOG2 is at
        # self.bg_sub._real and .apply on the proxy writes to
        # self.last_mog2_ms.
        self.bg_sub = _TimingMOG2(self.bg_sub, self)

    def _reset(self):
        self.last_markers          = None
        self.last_seeds_full_yx    = []
        self.last_placed_labels    = []
        self.last_n_candidates     = 0
        self.last_n_after_dedup    = 0
        # Per-frame stage timings (ms)
        self.last_mog2_ms          = 0.0
        self.last_dedup_ms         = 0.0
        self.last_watershed_ms     = 0.0
        self.last_gudhi_ms         = 0.0

    def detect(self, frame, learning_rate=-1, grey=None):
        self._reset()
        # Reset the module-level GUDHI timer; super().detect() will set it
        # via the _TimedCubicalComplex wrapper.
        _LAST_GUDHI_MS[0] = 0.0
        out = super().detect(frame, learning_rate, grey=grey)
        self.last_gudhi_ms = _LAST_GUDHI_MS[0]
        return out

    def _deduplicate_by_branch(self, candidates, small):
        self.last_n_candidates = len(candidates)
        t0 = time.perf_counter()
        kept = super()._deduplicate_by_branch(candidates, small)
        self.last_dedup_ms = (time.perf_counter() - t0) * 1000.0
        self.last_n_after_dedup = len(kept)
        return kept

    def _watershed_segment(self, masked_grey, combined_mask, seeds_full_yx):
        t0 = time.perf_counter()
        markers, placed = super()._watershed_segment(
            masked_grey, combined_mask, seeds_full_yx
        )
        self.last_watershed_ms = (time.perf_counter() - t0) * 1000.0
        self.last_markers       = markers.copy()
        self.last_seeds_full_yx = list(seeds_full_yx)
        self.last_placed_labels = list(placed)
        return markers, placed


# ── Instrumented tracker ──────────────────────────────────────────────────────
class DiagnosticTracker(BlobTracker):
    """
    Wraps BlobTracker to record the time spent in Farneback optical-flow
    computation, the suspected heavy stage on the tracker side. All other
    tracker behaviour is unchanged.

      Timings (milliseconds, populated each update() call):
        * last_flow_ms — _compute_flow (cv2.calcOpticalFlowFarneback)
    """
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.last_flow_ms = 0.0

    def _compute_flow(self, curr_grey):
        t0 = time.perf_counter()
        flow = super()._compute_flow(curr_grey)
        self.last_flow_ms = (time.perf_counter() - t0) * 1000.0
        return flow


# ── Per-track history ─────────────────────────────────────────────────────────
@dataclass
class TrackHistory:
    track_id:     int
    born_frame:   int
    died_frame:   Optional[int]    = None
    centroids:    list             = field(default_factory=list)  # (frame, x, y)
    areas:        list             = field(default_factory=list)  # (frame, area)
    coast_frames: list             = field(default_factory=list)
    direction:    list             = field(default_factory=list)
    aspect_ratio: list             = field(default_factory=list)

    @property
    def lifespan_frames(self) -> int:
        if not self.centroids:
            return 0
        end = self.died_frame if self.died_frame is not None else self.centroids[-1][0]
        return max(end - self.born_frame, 1)

    def median_motion_px_per_frame(self) -> float:
        if len(self.centroids) < 2:
            return 0.0
        deltas = [
            float(np.hypot(
                self.centroids[i][1] - self.centroids[i-1][1],
                self.centroids[i][2] - self.centroids[i-1][2],
            ))
            for i in range(1, len(self.centroids))
        ]
        return float(np.median(deltas))

    def median_aspect_ratio(self) -> float:
        if not self.aspect_ratio:
            return 1.0
        return float(np.median(self.aspect_ratio))

    def median_y_norm(self, profile: LampProfile) -> float:
        if not self.centroids:
            return 0.5
        ys = [profile.y_norm(c[2]) for c in self.centroids]
        return float(np.median(ys))

    def shape_volatility_ratio(self) -> float:
        """Median frame-to-frame area ratio (always >= 1)."""
        if len(self.areas) < 2:
            return 1.0
        ratios = []
        for i in range(1, len(self.areas)):
            a0, a1 = self.areas[i-1][1], self.areas[i][1]
            if a0 > 0 and a1 > 0:
                ratios.append(max(a0, a1) / min(a0, a1))
        return float(np.median(ratios)) if ratios else 1.0

    def motion_class(self) -> str:
        m = self.median_motion_px_per_frame()
        if m >= FAST_MOTION_PX_PER_FRAME: return "fast"
        if m <= SLOW_MOTION_PX_PER_FRAME: return "slow"
        return "moderate"

    def shape_class(self) -> str:
        ar = self.median_aspect_ratio()
        if ar >= 2.5: return "elongated"
        if ar >= 1.5: return "oblong"
        return "round"

    def position_class(self, profile) -> str:
        y = self.median_y_norm(profile)
        if y < 0.33: return "lower"
        if y > 0.66: return "upper"
        return "middle"


# ── Helper ────────────────────────────────────────────────────────────────────
def split_thirds_mean(seq):
    """Mean of first/middle/last third of a list. Useful for time-trend detection."""
    if not seq:
        return 0.0, 0.0, 0.0
    n = len(seq)
    third = max(1, n // 3)
    a = float(np.mean(seq[:third]))
    b = float(np.mean(seq[third:2*third])) if n >= 2*third else a
    c = float(np.mean(seq[2*third:])) if n >= 2*third else b
    return a, b, c


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Audible Lava tracking diagnostic")
    parser.add_argument("--frames",  type=int, default=DEFAULT_TOTAL_FRAMES)
    parser.add_argument("--warmup",  type=int, default=DEFAULT_WARMUP_FRAMES)
    parser.add_argument("--video",   type=str, default=None,
                        help="path to video file (overrides camera)")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--report",  type=str, default="diagnostic_report.json")
    args = parser.parse_args()

    t_loop_start = time.time()

    profile = LampProfile.load(PROFILE_PATH) or LampProfile.default()

    # Open input source
    if args.video:
        cap = cv2.VideoCapture(args.video)
        src_label = f"video={args.video}"
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        src_label = f"camera={CAMERA_INDEX}"
    if not cap.isOpened():
        print(f"[diagnostic] ERROR: cannot open source ({src_label})")
        sys.exit(1)

    # Force the camera to the profile's resolution. Video files keep their
    # native resolution; we update the profile to match below.
    if not args.video:
        profile.apply_to_capture(cap)

    # Probe frame size and update profile (handles video files and any
    # residual mismatch the camera negotiated to).
    ret, frame = cap.read()
    if not ret:
        print("[diagnostic] ERROR: cannot read first frame")
        sys.exit(1)
    h, w = frame.shape[:2]
    profile.frame_h, profile.frame_w = h, w

    detector = DiagnosticDetector(profile=profile)
    tracker  = DiagnosticTracker(profile=profile)

    # Reset capture so frame 0 is processed normally
    if args.video:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── Per-frame measurements ────────────────────────────────────────────────
    mog2_coverage          = []   # fraction of frame
    colour_coverage        = []
    combined_coverage      = []
    abandoned_wax_frac     = []   # fraction of wax-coloured pixels MOG2 calls bg
    sat_excluded_count     = []
    flow_interior_mag      = []   # mean |flow| inside blob masks (eroded)
    flow_edge_mag          = []   # mean |flow| in edge band
    candidate_counts       = []
    seed_counts            = []
    blob_counts            = []
    fragmentation_count    = []   # blobs whose watershed region had >1 cc

    # ── Per-stage timings (milliseconds) ──────────────────────────────────────
    # Top-level stages cover the whole per-frame loop. Within `detect`, four
    # sub-stages are timed individually; the residual "detect_other" =
    # detect - (mog2 + gudhi + dedup + watershed) absorbs preprocessing
    # (BGR→grey/HSV, GaussianBlur, masking) and post-processing (per-blob
    # contour/HuMoments/fitEllipse/mean_hue extraction).
    timings = {
        'frame_read':       [],
        'probe_masks':      [],
        'detect':           [],
        'detect_mog2':      [],
        'detect_gudhi':     [],
        'detect_dedup':     [],
        'detect_watershed': [],
        'track':            [],
        'track_flow':       [],   # Farneback inside track
        'harness_probes':   [],
        'display':          [],
    }

    histories = {}                # id → TrackHistory
    event_counts = defaultdict(int)

    erode_kernel = np.ones((INTERIOR_ERODE_KERNEL_PX, INTERIOR_ERODE_KERNEL_PX),
                           dtype=np.uint8)
    dilate_kernel = np.ones((EDGE_BAND_KERNEL_PX, EDGE_BAND_KERNEL_PX),
                            dtype=np.uint8)

    print(f"[diagnostic] source={src_label}  frame={w}x{h}")
    print(f"[diagnostic] warmup={args.warmup}f  total={args.frames}f  "
          f"profile_sat_min={profile.wax_sat_min}")

    frame_idx        = 0
    measured_frames  = 0
    aborted          = False

    try:
        while frame_idx < args.frames:
            # ── 0. Frame read ──────────────────────────────────────────────
            t = time.perf_counter()
            ret, frame = cap.read()
            t_frame_read = (time.perf_counter() - t) * 1000.0
            if not ret:
                break

            # ── v4.1: crop to lamp ROI + shared greyscale ─────────────────
            # The whole pipeline runs on the ROI now; measuring it here makes
            # the per-stage timings reflect the deployed path. (Detection runs
            # every frame in the diagnostic, i.e. K=1 — the correct baseline.)
            roi  = profile.crop(frame)
            grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_h, roi_w = roi.shape[:2]

            # ── 1. Probe MOG2/colour/combined masks WITHOUT updating model ────
            #     (get_debug_masks uses learningRate=0; safe to call before detect)
            t = time.perf_counter()
            masks  = detector.get_debug_masks(roi, allow_cache=False, grey=grey)
            t_probe_masks = (time.perf_counter() - t) * 1000.0
            mog2   = masks["mog2"]
            colour = masks["colour"]
            combo  = masks["combined"]

            # ── 2. Run real pipeline (this updates MOG2 and tracker state) ────
            t = time.perf_counter()
            blobs   = detector.detect(roi, grey=grey)
            t_detect = (time.perf_counter() - t) * 1000.0

            t = time.perf_counter()
            tracked = tracker.update(blobs, roi, grey=grey)
            t_track = (time.perf_counter() - t) * 1000.0

            # Always record top-level timings (they're stable across warmup
            # vs measurement; recording during warmup gives more samples).
            timings['frame_read'      ].append(t_frame_read)
            timings['probe_masks'     ].append(t_probe_masks)
            timings['detect'          ].append(t_detect)
            timings['detect_mog2'     ].append(detector.last_mog2_ms)
            timings['detect_gudhi'    ].append(detector.last_gudhi_ms)
            timings['detect_dedup'    ].append(detector.last_dedup_ms)
            timings['detect_watershed'].append(detector.last_watershed_ms)
            timings['track'           ].append(t_track)
            timings['track_flow'      ].append(tracker.last_flow_ms)

            # ── 3. Skip measurement during MOG2 warm-up ────────────────────────
            if frame_idx < args.warmup:
                frame_idx += 1
                continue

            # ── 4. Harness probes (timed as one bucket) ───────────────────────
            t_probes_start = time.perf_counter()

            n_pixels = roi_h * roi_w

            # ── Mask coverage ─────────────────────────────────────────────────
            mog2_cov   = float((mog2 > 0).sum())   / n_pixels
            col_cov    = float((colour > 0).sum()) / n_pixels
            comb_cov   = float((combo > 0).sum())  / n_pixels
            mog2_coverage.append(mog2_cov)
            colour_coverage.append(col_cov)
            combined_coverage.append(comb_cov)

            # ── Failure mode 1: MOG2 absorption ───────────────────────────────
            #     Wax-coloured pixels that MOG2 has absorbed into background.
            colour_n   = int((colour > 0).sum())
            abandoned  = int(((colour > 0) & (mog2 == 0)).sum())
            abandoned_wax_frac.append(
                abandoned / colour_n if colour_n > 0 else 0.0
            )

            # ── Failure mode 6: saturation cutoff ─────────────────────────────
            #     How many pixels pass hue+value but fail saturation threshold?
            hsv      = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hue_pass = ((hsv[..., 0] >= profile.wax_hue_lo) &
                        (hsv[..., 0] <= profile.wax_hue_hi))
            val_pass = (hsv[..., 2] >= profile.wax_val_min)
            sat_fail = (hsv[..., 1] <  profile.wax_sat_min)
            sat_excluded_count.append(int((hue_pass & val_pass & sat_fail).sum()))

            # ── Pipeline cardinality ──────────────────────────────────────────
            candidate_counts.append(detector.last_n_candidates)
            seed_counts.append(detector.last_n_after_dedup)
            blob_counts.append(len(blobs))

            # ── Failure mode 5: watershed fragmentation ───────────────────────
            n_frag = 0
            if detector.last_markers is not None:
                for label in detector.last_placed_labels:
                    if label is None:
                        continue
                    region = (detector.last_markers == label).astype(np.uint8)
                    n_cc, _ = cv2.connectedComponents(region)
                    # n_cc includes label-0 background, so foreground count = n_cc - 1
                    if n_cc - 1 > 1:
                        n_frag += 1
            fragmentation_count.append(n_frag)

            # ── Failure mode 2: Farneback interior vs edge flow ──────────────
            flow = tracker.get_flow()
            if flow is not None and tracked:
                flow_mag = np.hypot(flow[..., 0], flow[..., 1])
                interior_means, edge_means = [], []
                for blob in tracked:
                    track = tracker._tracks.get(blob.id)
                    if track is None:
                        continue
                    mask_u8 = (track["mask"] > 0).astype(np.uint8)
                    if mask_u8.sum() < 100:
                        continue
                    eroded  = cv2.erode (mask_u8, erode_kernel) > 0
                    dilated = cv2.dilate(mask_u8, dilate_kernel) > 0
                    edge_band = dilated & ~eroded
                    if eroded.sum() > 0:
                        interior_means.append(float(flow_mag[eroded].mean()))
                    if edge_band.sum() > 0:
                        edge_means.append(float(flow_mag[edge_band].mean()))
                if interior_means:
                    flow_interior_mag.append(float(np.mean(interior_means)))
                if edge_means:
                    flow_edge_mag.append(float(np.mean(edge_means)))

            # ── Per-track history update ──────────────────────────────────────
            for blob in tracked:
                if blob.id not in histories:
                    histories[blob.id] = TrackHistory(
                        track_id=blob.id, born_frame=frame_idx,
                    )
                hist = histories[blob.id]
                hist.centroids.append((frame_idx, blob.centroid[0], blob.centroid[1]))
                hist.areas.append((frame_idx, blob.area))
                hist.coast_frames.append(blob.coast_frames)
                hist.direction.append(blob.direction)
                hist.aspect_ratio.append(blob.aspect_ratio)

            # ── Events ────────────────────────────────────────────────────────
            for event in tracker.events:
                ev_type = event[0]
                event_counts[ev_type] += 1
                if ev_type == "death":
                    tid = event[1]
                    if tid in histories:
                        histories[tid].died_frame = frame_idx

            # End of harness_probes bucket — every per-frame measurement
            # above this point (mask coverage, abandoned_wax, sat exclusion,
            # fragmentation check, Farneback interior/edge sampling, track
            # history bookkeeping, event accounting) accumulates here.
            timings['harness_probes'].append(
                (time.perf_counter() - t_probes_start) * 1000.0
            )

            # ── Live overlay ──────────────────────────────────────────────────
            t_display_start = time.perf_counter()
            if not args.no_display:
                overlay = roi.copy()
                # Tint MOG2-abandoned wax pixels red
                ab_mask = (colour > 0) & (mog2 == 0)
                ab_overlay = np.zeros_like(roi)
                ab_overlay[ab_mask] = (0, 0, 255)
                overlay = cv2.addWeighted(overlay, 0.85, ab_overlay, 0.4, 0)
                # Tint sat-excluded pixels yellow
                sat_ex_mask = hue_pass & val_pass & sat_fail
                sat_overlay = np.zeros_like(roi)
                sat_overlay[sat_ex_mask] = (0, 255, 255)
                overlay = cv2.addWeighted(overlay, 0.9, sat_overlay, 0.3, 0)
                # Tracked blobs (ROI-local coordinates → overlay is the ROI)
                for b in tracked:
                    cx, cy = int(b.centroid[0]), int(b.centroid[1])
                    cv2.circle(overlay, (cx, cy), 8, (255, 255, 255), 2)
                    cv2.putText(overlay, f"{b.id}", (cx + 10, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)
                status = (
                    f"f={frame_idx:4d}  blobs={len(tracked)}  "
                    f"mog2_cov={mog2_cov:.3f}  abandoned_wax={abandoned_wax_frac[-1]:.2f}  "
                    f"frag={n_frag}"
                )
                cv2.putText(overlay, status, (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)
                cv2.putText(overlay, "red=MOG2-abandoned-wax  yellow=sat-rejected",
                            (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (200, 200, 200), 1)
                cv2.imshow("diagnostic", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    aborted = True
                    break
            timings['display'].append(
                (time.perf_counter() - t_display_start) * 1000.0
            )

            frame_idx       += 1
            measured_frames += 1

    except KeyboardInterrupt:
        print("\n[diagnostic] keyboard interrupt — generating report")
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    # ── Build report ─────────────────────────────────────────────────────────
    report = {
        "meta": {
            "source":            src_label,
            "frame_size":        [w, h],
            "warmup_frames":     args.warmup,
            "measured_frames":   measured_frames,
            "aborted_early":     aborted,
            "tracks_observed":   len(histories),
            "lamp_profile_sat_min": profile.wax_sat_min,
            "lamp_profile_hue_range": [profile.wax_hue_lo, profile.wax_hue_hi],
            "tracker_max_coast":     tracker.max_coast,
            "tracker_iou_threshold": tracker.iou_threshold,
        }
    }
    elapsed = time.time() - t_loop_start
    report["meta"]["actual_fps"] = measured_frames / max(elapsed, 1e-6)
    report["meta"]["elapsed_seconds"] = elapsed

    # 1. MOG2 absorption
    if abandoned_wax_frac:
        a, b, c = split_thirds_mean(abandoned_wax_frac)
        report["1_mog2_absorption"] = {
            "mean_fraction_wax_called_background": float(np.mean(abandoned_wax_frac)),
            "p95_fraction":                         float(np.percentile(abandoned_wax_frac, 95)),
            "first_third_mean":                     a,
            "middle_third_mean":                    b,
            "last_third_mean":                      c,
            "absorption_drift":                     c - a,
            "interpretation": (
                "Fraction of wax-coloured pixels MOG2 has absorbed into "
                "background. >0.20 sustained = absorption is killing slow blobs. "
                "Positive absorption_drift = problem worsens over the window."
            ),
        }

    # 2. Farneback interior vs edge flow
    if flow_interior_mag and flow_edge_mag:
        m_int  = float(np.mean(flow_interior_mag))
        m_edge = float(np.mean(flow_edge_mag))
        report["2_farneback_textureless_interior"] = {
            "mean_interior_flow_px_per_frame": m_int,
            "mean_edge_flow_px_per_frame":     m_edge,
            "edge_to_interior_ratio":          m_edge / max(m_int, 1e-6),
            "interpretation": (
                "Interior << edge means Farneback finds no texture inside "
                "uniform wax. Mask advection then under-shifts and fast "
                "blobs slip out from under their tracks. Ratio >3 is a "
                "strong indicator."
            ),
        }

    # 3. Mask shape volatility
    volatilities = [
        h_.shape_volatility_ratio()
        for h_ in histories.values()
        if len(h_.areas) >= 5
    ]
    if volatilities:
        report["3_mask_shape_volatility"] = {
            "n_tracks_measured":             len(volatilities),
            "median_frame_to_frame_ratio":   float(np.median(volatilities)),
            "p95_frame_to_frame_ratio":      float(np.percentile(volatilities, 95)),
            "n_tracks_with_ratio_gt_3":      int(sum(v > VOLATILITY_RATIO_RED_FLAG
                                                    for v in volatilities)),
            "interpretation": (
                "Frame-to-frame area ratio >3 collapses IoU below 0.25 "
                "and breaks the match. Watershed-driven shape changes on "
                "growing or collapsing columns are the typical cause."
            ),
        }

    # 4. Mask refinement blend (parameter inspection only)
    report["4_mask_refinement_blend"] = {
        "refine_interval_frames": getattr(BlobTracker, "REFINE_INTERVAL", 5),
        "interpretation": (
            "Every REFINE_INTERVAL frames the advected mask is blended "
            "50/50 with the detection. For volatile shapes this drags a "
            "stale geometry into the new frame. No direct measure here; "
            "if section 3 reports high volatility, the blend is amplifying it."
        ),
    }

    # 5. Watershed fragmentation
    if fragmentation_count:
        n_meas        = len(fragmentation_count)
        n_frag_frames = sum(1 for v in fragmentation_count if v > 0)
        report["5_watershed_neck_fragmentation"] = {
            "frames_with_at_least_one_fragmented_blob": n_frag_frames,
            "frames_measured":                          n_meas,
            "fraction_frames_with_fragmentation":       n_frag_frames / max(n_meas, 1),
            "max_fragmented_blobs_per_frame":           int(max(fragmentation_count)),
            "interpretation": (
                "Each event is a cv2.watershed -1 ridge slicing through a "
                "thin neck. The smaller fragment falls below min_area and "
                "the blob is dropped. Tall columns are most affected."
            ),
        }

    # 6. Saturation cutoff
    if sat_excluded_count:
        report["6_saturation_cutoff"] = {
            "current_wax_sat_min":               int(profile.wax_sat_min),
            "mean_pixels_per_frame_excluded":    float(np.mean(sat_excluded_count)),
            "p95_pixels_per_frame_excluded":     float(np.percentile(sat_excluded_count, 95)),
            "max_pixels_per_frame_excluded":     int(max(sat_excluded_count)),
            "interpretation": (
                "Pixels per frame whose hue and value say wax but whose "
                "saturation falls below wax_sat_min. High counts = the "
                "saturation gate is over-restrictive; thin/fast wax has "
                "lower per-pixel saturation than fat blobs."
            ),
        }

    # 7. Tracking outcomes by motion / shape / position class
    by_motion   = defaultdict(list)
    by_shape    = defaultdict(list)
    by_position = defaultdict(list)

    for hist in histories.values():
        if len(hist.centroids) < 5:
            continue
        died    = hist.died_frame is not None
        record  = (hist.lifespan_frames, died, hist.shape_volatility_ratio())
        by_motion  [hist.motion_class()       ].append(record)
        by_shape   [hist.shape_class()        ].append(record)
        by_position[hist.position_class(profile)].append(record)

    def _summarise(buckets):
        out = {}
        for k, recs in buckets.items():
            if not recs:
                continue
            lifespans = [r[0] for r in recs]
            died      = [r[1] for r in recs]
            vols      = [r[2] for r in recs]
            out[k] = {
                "n":                       len(recs),
                "mean_lifespan_frames":    float(np.mean(lifespans)),
                "death_rate":              float(np.mean(died)),
                "mean_volatility_ratio":   float(np.mean(vols)),
            }
        return out

    report["7_tracking_outcomes"] = {
        "by_motion_class":   _summarise(by_motion),
        "by_shape_class":    _summarise(by_shape),
        "by_position_class": _summarise(by_position),
        "interpretation": (
            "Per-class mean lifespan and death rate. If 'fast' tracks "
            "have low lifespan and high death rate, mask advection is "
            "failing them (sections 2 and 3). If 'slow' tracks die early, "
            "MOG2 absorption is the cause (section 1). If 'elongated' "
            "tracks die or never appear, watershed fragmentation "
            "(section 5) and saturation cutoff (section 6) are involved."
        ),
    }

    # 8. Pipeline cardinality
    if candidate_counts:
        m_cand = float(np.mean(candidate_counts))
        m_seed = float(np.mean(seed_counts))
        m_blob = float(np.mean(blob_counts))
        report["8_pipeline_counts"] = {
            "mean_raw_candidates_post_pers_thresh": m_cand,
            "mean_seeds_post_branch_dedup":         m_seed,
            "mean_blobs_post_watershed_min_area":   m_blob,
            "branch_dedup_filter_rate":             1.0 - m_seed / max(m_cand, 1e-6),
            "watershed_min_area_filter_rate":       1.0 - m_blob / max(m_seed, 1e-6),
            "interpretation": (
                "Where blobs are getting filtered out. High "
                "watershed_min_area_filter_rate suggests fragmentation "
                "(section 5) is dropping legitimate blobs."
            ),
        }

    # 9. Mask coverage trend
    if mog2_coverage:
        a, b, c = split_thirds_mean(mog2_coverage)
        report["9_mask_coverage_over_time"] = {
            "mog2_first_third":     a,
            "mog2_middle_third":    b,
            "mog2_last_third":      c,
            "mog2_drift":           c - a,
            "colour_mean":          float(np.mean(colour_coverage)),
            "combined_mean":        float(np.mean(combined_coverage)),
            "interpretation": (
                "MOG2 coverage drift: negative drift = MOG2 absorbing more "
                "wax over time (slow blobs vanishing). Compare to colour_mean "
                "which should be steady — if MOG2 drops while colour holds, "
                "absorption is confirmed."
            ),
        }

    # 10. Events
    report["10_events"] = dict(event_counts)

    # 11. Per-stage timing
    # Compute mean / median / p95 for each timed stage. The most useful
    # field for spotting the bottleneck is `mean_ms` paired with
    # `fraction_of_frame`, which expresses each stage as a share of the
    # total per-frame compute. A derived `detect_other` row absorbs the
    # part of detect not covered by the four sub-stage timers (mog2,
    # gudhi, dedup, watershed) — this is preprocessing (cvtColor,
    # GaussianBlur, masking) plus per-blob feature extraction (HuMoments,
    # fitEllipse, mean_hue). On a frame that's CPU-bound at 1-2 fps,
    # whichever stage dominates `fraction_of_frame` is the place to act:
    #   • detect_gudhi dominant       → bump BlobDetector(downscale=4) → 6 or 8
    #   • track_flow dominant         → reduce Farneback pyramid levels or downscale
    #   • detect_other dominant       → likely per-blob feature extraction; profile finer
    #   • probe_masks comparable to
    #     detect                      → harness is computing MOG2/colour mask twice;
    #                                   consider caching the masks from detect()
    #   • display dominant            → run with --no-display
    def _stats(vals):
        if not vals:
            return None
        return {
            'mean_ms':   float(np.mean(vals)),
            'median_ms': float(np.median(vals)),
            'p95_ms':    float(np.percentile(vals, 95)),
            'n_samples': len(vals),
        }

    if any(timings.values()):
        # Compute detect_other as a residual: detect_total - sum(sub-stages).
        # Done element-wise per frame so distributional stats remain valid.
        detect_other_per_frame = []
        n = min(len(timings['detect']), len(timings['detect_mog2']),
                len(timings['detect_gudhi']), len(timings['detect_dedup']),
                len(timings['detect_watershed']))
        for i in range(n):
            other = (timings['detect'][i]
                     - timings['detect_mog2'][i]
                     - timings['detect_gudhi'][i]
                     - timings['detect_dedup'][i]
                     - timings['detect_watershed'][i])
            detect_other_per_frame.append(max(other, 0.0))

        # Use only top-level stages to compute the per-frame total (the
        # sub-stages of detect would double-count). Probe_masks, detect,
        # track, harness_probes, display = the wall-clock sequence.
        top_level_total_mean = (
            (np.mean(timings['frame_read'])     if timings['frame_read']     else 0.0) +
            (np.mean(timings['probe_masks'])    if timings['probe_masks']    else 0.0) +
            (np.mean(timings['detect'])         if timings['detect']         else 0.0) +
            (np.mean(timings['track'])          if timings['track']          else 0.0) +
            (np.mean(timings['harness_probes']) if timings['harness_probes'] else 0.0) +
            (np.mean(timings['display'])        if timings['display']        else 0.0)
        )

        timing_section = {
            'top_level_total_ms_per_frame': float(top_level_total_mean),
            'derived_fps_from_timing':      float(1000.0 / max(top_level_total_mean, 1e-6)),
            'top_level_stages':             {},
            'detect_sub_stages':            {},
        }

        # Top-level stages with fractional share of the per-frame total
        for stage in ('frame_read', 'probe_masks', 'detect', 'track',
                      'harness_probes', 'display'):
            s = _stats(timings[stage])
            if s is None:
                continue
            s['fraction_of_frame'] = (
                s['mean_ms'] / top_level_total_mean
                if top_level_total_mean > 0 else 0.0
            )
            timing_section['top_level_stages'][stage] = s

        # Sub-stages of detect, with fractional share of detect_total
        detect_total_mean = (
            np.mean(timings['detect']) if timings['detect'] else 0.0
        )
        for stage, vals in (
            ('detect_mog2',      timings['detect_mog2']),
            ('detect_gudhi',     timings['detect_gudhi']),
            ('detect_dedup',     timings['detect_dedup']),
            ('detect_watershed', timings['detect_watershed']),
            ('detect_other',     detect_other_per_frame),
        ):
            s = _stats(vals)
            if s is None:
                continue
            s['fraction_of_detect'] = (
                s['mean_ms'] / detect_total_mean
                if detect_total_mean > 0 else 0.0
            )
            timing_section['detect_sub_stages'][stage] = s

        # Track flow alone, with fractional share of track_total
        track_total_mean = (
            np.mean(timings['track']) if timings['track'] else 0.0
        )
        flow_stats = _stats(timings['track_flow'])
        if flow_stats is not None:
            flow_stats['fraction_of_track'] = (
                flow_stats['mean_ms'] / track_total_mean
                if track_total_mean > 0 else 0.0
            )
            timing_section['track_sub_stages'] = {'track_flow': flow_stats}

        timing_section['interpretation'] = (
            "Per-stage wall-clock timing per frame (mean, median, p95 in ms). "
            "top_level_stages partitions each frame into the sequential "
            "phases of the loop; their mean_ms values sum to "
            "top_level_total_ms_per_frame. detect_sub_stages decomposes "
            "the dominant detect call into MOG2.apply, GUDHI persistence, "
            "branch dedup, watershed, and a residual 'detect_other' bucket "
            "(preprocessing + per-blob feature extraction). On a pipeline "
            "running at 1-2 fps, look at top_level_stages: whichever "
            "fraction_of_frame is largest is the bottleneck. If 'detect' "
            "is largest, drill into detect_sub_stages — most commonly "
            "detect_gudhi will dominate at the current downscale=4 grid "
            "size, and bumping downscale to 6 or 8 typically yields a 4-8x "
            "speedup on the persistence step."
        )

        report["11_per_stage_timing"] = timing_section

    # ── Print report ─────────────────────────────────────────────────────────
    _print_report(report)

    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[diagnostic] full JSON report written to {args.report}")


def _print_report(report: dict) -> None:
    print("\n" + "=" * 72)
    print(" AUDIBLE LAVA — TRACKING DIAGNOSTIC REPORT")
    print("=" * 72)

    def _fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        if isinstance(v, list):
            return str(v)
        return str(v)

    for section, data in report.items():
        print(f"\n── {section} ──")
        if isinstance(data, dict):
            interp = data.pop("interpretation", None)
            for k, v in data.items():
                if isinstance(v, dict):
                    print(f"  {k}:")
                    for k2, v2 in v.items():
                        if isinstance(v2, dict):
                            print(f"    {k2}:")
                            for k3, v3 in v2.items():
                                print(f"      {k3}: {_fmt(v3)}")
                        else:
                            print(f"    {k2}: {_fmt(v2)}")
                else:
                    print(f"  {k}: {_fmt(v)}")
            if interp:
                print(f"  ↳ {interp}")
        else:
            print(f"  {data}")


if __name__ == "__main__":
    main()
