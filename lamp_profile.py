# lamp_profile.py -- Audible Lava v3.0
# Stores all per-lamp calibration data extracted by calibration_gui.py.
# Passed to BlobDetector and BlobTracker so every parameter is grounded
# in the actual physical lamp being filmed, not hardcoded defaults.

import json
import os
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Optional, List

PROFILE_PATH = "lamp_profile.json"


# ── Circular hue statistics ───────────────────────────────────────────────────
# OpenCV hue is a circular quantity in [0, 180): a pixel at H=178 and one at
# H=2 are 4° apart on the colour wheel, not 176° apart. Linear np.percentile()
# on hue values that straddle the 0/180 boundary produces nonsense — e.g.
# percentile-5 = 1, percentile-95 = 179 for a tight distribution centred at
# H=0. The helper below avoids this by:
#
#   1. Computing the circular mean via the standard sin/cos trick.
#   2. Rotating every hue into a frame where the mean lands at 90 (the linear
#      mid-point of [0, 180)). All hues now sit near 90, far from the
#      wraparound boundary.
#   3. Taking ordinary linear percentiles in this rotated frame.
#   4. Rotating the resulting bounds back into the original [0, 180) space.
#
# The returned (lo_abs, hi_abs) pair may have lo_abs > hi_abs when the wax
# colour band wraps across 0/180. Downstream code (blob_detection._make_wax_mask)
# treats lo > hi as the explicit signal to mask [lo, 180] ∪ [0, hi].

def _circular_hue_stats(
    hues:   np.ndarray,
    p_low:  float = 5.0,
    p_high: float = 95.0,
    margin: float = 0.0,
) -> dict:
    """
    Compute circular-aware statistics on a 1D array of OpenCV hue values.

    Args:
        hues:    1D numpy array of hue values in [0, 180).
        p_low:   low percentile to compute (default 5).
        p_high:  high percentile to compute (default 95).
        margin:  symmetric expansion of the rotated range, in OpenCV hue
                 degrees, applied BEFORE rotating back to absolute hue.

    Returns:
        dict with keys:
          'mean_abs'  : circular mean in OpenCV hue space [0, 180)
          'std_abs'   : circular standard deviation in OpenCV hue degrees
          'lo_abs'    : p_low percentile in OpenCV hue space (may wrap)
          'hi_abs'    : p_high percentile in OpenCV hue space (may wrap)
          'lo_rot'    : p_low percentile in the rotated frame (post-margin)
          'hi_rot'    : p_high percentile in the rotated frame (post-margin)
          'range_rot' : hi_rot - lo_rot, always positive, total angular span
          'wraps'     : True if lo_abs > hi_abs (the band crosses 0/180)
    """
    if len(hues) == 0:
        return {
            'mean_abs': 0.0, 'std_abs': 0.0,
            'lo_abs':   0.0, 'hi_abs':   0.0,
            'lo_rot':   90.0, 'hi_rot':  90.0,
            'range_rot': 0.0, 'wraps':   False,
        }

    h = hues.astype(np.float64)
    angles = np.radians(h * 2.0)        # OpenCV H is degrees/2 → ×2 for full wheel
    sin_m  = float(np.mean(np.sin(angles)))
    cos_m  = float(np.mean(np.cos(angles)))
    R      = float(np.hypot(sin_m, cos_m))
    mean_abs = float(np.degrees(np.arctan2(sin_m, cos_m)) / 2.0 % 180.0)

    # Circular standard deviation, halved back to OpenCV hue degrees
    if R > 1e-9:
        std_abs = float(np.degrees(np.sqrt(-2.0 * np.log(min(R, 1.0)))) / 2.0)
    else:
        std_abs = 90.0   # uniform distribution — std is maximal

    # Rotate hues so the circular mean lands at 90 — the linear midpoint
    rotated = (h - mean_abs + 90.0) % 180.0
    lo_rot  = float(np.percentile(rotated, p_low))  - margin
    hi_rot  = float(np.percentile(rotated, p_high)) + margin
    range_rot = hi_rot - lo_rot

    # Rotate the bounds back to absolute OpenCV hue space
    lo_abs = (lo_rot + mean_abs - 90.0) % 180.0
    hi_abs = (hi_rot + mean_abs - 90.0) % 180.0

    return {
        'mean_abs':  mean_abs,
        'std_abs':   std_abs,
        'lo_abs':    lo_abs,
        'hi_abs':    hi_abs,
        'lo_rot':    lo_rot,
        'hi_rot':    hi_rot,
        'range_rot': range_rot,
        'wraps':     lo_abs > hi_abs,
    }


@dataclass
class LampProfile:
    """
    Per-lamp calibration profile extracted by the freehand-drawing GUI.

    Colour parameters are derived from HSV statistics of the user-drawn
    blob regions. Physics parameters are derived from the lamp ROI bounds.
    Thermodynamic hue thresholds are derived from the spread of hue values
    across all drawn regions (colder = lower hue / more red; hotter = higher
    hue / more yellow-orange for typical orange-red wax, or darker vs lighter
    pink for pink wax).

    All pixel coordinates are in the full camera frame (before any crop).
    """

    # ── Wax colour HSV range (OpenCV: H 0–180, S/V 0–255) ───────────────
    wax_hue_lo:  int   = 148    # hue lower bound
    wax_hue_hi:  int   = 172    # hue upper bound
    wax_sat_min: int   = 15     # minimum saturation
    wax_val_min: int   = 160    # minimum brightness

    # ── Lamp region of interest (pixel coords in full camera frame) ──────
    lamp_x1: int = 0
    lamp_y1: int = 0
    lamp_x2: int = 1280
    lamp_y2: int = 720

    # ── Blob size statistics (pixels, from drawn regions) ────────────────
    blob_area_min:    float = 200.0
    blob_area_max:    float = 50000.0
    n_blobs_expected: int   = 4

    # ── Frame dimensions ─────────────────────────────────────────────────
    # Authoritative resolution for the entire pipeline. Set by calibration
    # to whatever the camera was actually delivering. Re-applied to every
    # downstream cv2.VideoCapture via apply_to_capture() so detection,
    # tracking, and diagnostic scripts all see the same pixel scale that
    # blob_area_min / lamp_x1.. were calibrated against.
    frame_h: int = 720
    frame_w: int = 1280

    # ── Pipeline cadence (v4.1) ──────────────────────────────────────────
    # detect_interval (K): run the heavy CV stack (BlobDetector.detect +
    # BlobTracker association/correction) only every K-th frame. On the
    # intervening frames the tracker runs a cheap predict-only pass (flow
    # advection + Kalman prediction) so voice parameters keep streaming
    # smoothly at the full frame rate. K=1 reproduces the v4.0 every-frame
    # behaviour; K=2..4 is safe given lamp blob timescales (90–180 s cycles,
    # 0.5–2 px/frame) and roughly divides detection compute by K.
    detect_interval: int = 1

    # ── Thermodynamic hue thresholds ─────────────────────────────────────
    # Blobs shift hue as they heat (rise) and cool (fall).
    # rising_hue_threshold:  mean_hue ABOVE this → classified as rising
    # falling_hue_threshold: mean_hue BELOW this → classified as falling
    # Values between the two thresholds = "turning" (near reversal point).
    # Set automatically from drawn-region hue spread during calibration;
    # default values assume pink wax (hue ~160 centre).
    rising_hue_threshold:  float = 165.0
    falling_hue_threshold: float = 155.0

    # ── Hue range observed across all drawn blob regions ─────────────────
    # Used by BlobDetector to decide how wide a hue band to accept.
    # Stored separately from wax_hue_lo/hi so the tracker can access the
    # full observed range for thermodynamic reasoning.
    observed_hue_min: float = 148.0
    observed_hue_max: float = 172.0
    observed_hue_mean: float = 160.0

    # ── Persistence statistics from drawn regions ─────────────────────────
    # Used to set adaptive persistence_threshold in BlobDetector.
    expected_persistence_mean: float = 15
    expected_persistence_std:  float = 3.0

    # ------------------------------------------------------------------
    def save(self, path: str = PROFILE_PATH) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        print(f"[LampProfile] Saved to {path}")

    @classmethod
    def load(cls, path: str = PROFILE_PATH) -> Optional["LampProfile"]:
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        # Filter to only known fields (forward-compatibility)
        known = {k for k in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def default(cls, frame_h: int = 720, frame_w: int = 1280) -> "LampProfile":
        """Return a default profile with sensible values for pink wax."""
        return cls(frame_h=frame_h, frame_w=frame_w,
                   lamp_x2=frame_w, lamp_y2=frame_h)

    # ── Camera resolution enforcement ─────────────────────────────────────
    def apply_to_capture(self, cap, *, verbose: bool = True) -> tuple:
        """
        Configure a cv2.VideoCapture to deliver this profile's frame size.

        Sets CAP_PROP_FRAME_WIDTH and CAP_PROP_FRAME_HEIGHT from
        self.frame_w / self.frame_h, then reads back what the camera
        actually negotiated. Must be called BEFORE the first cap.read():
        most macOS UVC backends silently ignore resolution changes once
        streaming has started.

        If the camera negotiates to a different mode (e.g. 1280x720 was
        requested but the camera delivers 1920x1080 because the requested
        mode isn't supported), self.frame_w / self.frame_h are updated to
        match what was actually delivered, and lamp ROI coordinates are
        rescaled proportionally so blob_area_min and lamp_x1.. remain
        consistent with the delivered pixel scale.

        Args:
            cap:     cv2.VideoCapture instance.
            verbose: if True, print a one-line summary of requested vs
                     delivered resolution.

        Returns:
            (actual_w, actual_h) tuple of what the camera is delivering.
        """
        import cv2
        req_w, req_h = self.frame_w, self.frame_h
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  req_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if (actual_w, actual_h) != (req_w, req_h):
            if verbose:
                print(f"[LampProfile] requested {req_w}x{req_h}, "
                      f"camera delivered {actual_w}x{actual_h} — "
                      f"rescaling profile to match")
            # Rescale lamp ROI to the delivered size
            sx = actual_w / max(req_w, 1)
            sy = actual_h / max(req_h, 1)
            self.lamp_x1 = int(self.lamp_x1 * sx)
            self.lamp_x2 = int(self.lamp_x2 * sx)
            self.lamp_y1 = int(self.lamp_y1 * sy)
            self.lamp_y2 = int(self.lamp_y2 * sy)
            self.frame_w = actual_w
            self.frame_h = actual_h
        elif verbose:
            print(f"[LampProfile] capture set to {actual_w}x{actual_h}")

        return actual_w, actual_h

    # ── Derived geometry ──────────────────────────────────────────────────
    @property
    def lamp_height(self) -> int:
        return self.lamp_y2 - self.lamp_y1

    @property
    def lamp_width(self) -> int:
        return self.lamp_x2 - self.lamp_x1

    @property
    def lamp_top_y(self) -> int:
        """Pixel y of the top of the lamp (smallest y = top of image)."""
        return self.lamp_y1

    @property
    def lamp_bottom_y(self) -> int:
        """Pixel y of the base of the lamp."""
        return self.lamp_y2

    # ── ROI cropping (v4.1) ───────────────────────────────────────────────
    # The whole detection/tracking/synthesis pipeline now runs on the lamp
    # ROI sub-image rather than the full camera frame. The ROI is the lamp
    # bounding box (lamp_x1..lamp_y2), typically ~25–30 % of the frame, so
    # every pixel-bound stage (MOG2, blur, colour mask, Farneback flow, mask
    # advection, IoU) shrinks by 3–4× for free. All blob/voice coordinates
    # are ROI-local; only the debug overlay maps back via roi_rect()'s origin.
    def roi_rect(self) -> tuple:
        """Return the clamped (x1, y1, x2, y2) crop rectangle for this lamp."""
        x1 = max(0, min(int(self.lamp_x1), self.frame_w - 1))
        y1 = max(0, min(int(self.lamp_y1), self.frame_h - 1))
        x2 = max(x1 + 1, min(int(self.lamp_x2), self.frame_w))
        y2 = max(y1 + 1, min(int(self.lamp_y2), self.frame_h))
        return (x1, y1, x2, y2)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Crop a full-frame image to the lamp ROI (returns a view, no copy)."""
        x1, y1, x2, y2 = self.roi_rect()
        return frame[y1:y2, x1:x2]

    @property
    def roi_w(self) -> int:
        x1, _, x2, _ = self.roi_rect()
        return x2 - x1

    @property
    def roi_h(self) -> int:
        _, y1, _, y2 = self.roi_rect()
        return y2 - y1

    def y_norm(self, roi_y: float) -> float:
        """Normalise an ROI-local y to [0=bottom, 1=top] within the lamp.

        Coordinates are ROI-local (0 = top edge of the crop) after the v4.1
        ROI refactor, so this is simply 1 - roi_y / roi_h.
        """
        H = self.roi_h
        if H <= 0:
            return 0.5
        return float(np.clip(1.0 - roi_y / H, 0.0, 1.0))

    def x_norm(self, roi_x: float) -> float:
        """Normalise an ROI-local x to [0=left, 1=right] within the lamp."""
        W = self.roi_w
        if W <= 0:
            return 0.5
        return float(np.clip(roi_x / W, 0.0, 1.0))

    # ── Factory from drawn calibration regions ────────────────────────────
    @classmethod
    def from_drawn_regions(
        cls,
        region_masks: list,   # list of (h×w uint8) masks, one per drawn blob
        frame_bgr: np.ndarray,
        lamp_roi: Optional[tuple] = None,  # (x1, y1, x2, y2) or None
    ) -> "LampProfile":
        """
        Derive a LampProfile from freehand-drawn blob regions on a camera frame.

        Extracts HSV statistics from all drawn pixels, estimates typical blob
        area, and sets thermodynamic hue thresholds from the hue spread.
        """
        h, w = frame_bgr.shape[:2]
        frame_hsv = None

        import cv2
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        all_hues = []
        all_sats = []
        all_vals = []
        areas    = []

        for mask in region_masks:
            pixels = frame_hsv[mask > 0]
            if len(pixels) == 0:
                continue
            all_hues.extend(pixels[:, 0].tolist())
            all_sats.extend(pixels[:, 1].tolist())
            all_vals.extend(pixels[:, 2].tolist())
            areas.append(float(np.sum(mask > 0)))

        if not all_hues:
            return cls.default(h, w)

        hues = np.array(all_hues, dtype=np.float32)
        sats = np.array(all_sats, dtype=np.float32)
        vals = np.array(all_vals, dtype=np.float32)

        # ── Circular-aware hue range ──────────────────────────────────────
        # Use rotation + linear-percentile trick to handle distributions
        # that straddle the 0/180 wraparound (red wax). Resulting
        # hue_lo_int / hue_hi_int may be hue_lo > hue_hi to signal that
        # the band wraps; blob_detection._make_wax_mask handles this.
        hue_stats = _circular_hue_stats(hues, p_low=5.0, p_high=95.0, margin=5.0)
        hue_mean   = hue_stats['mean_abs']
        hue_lo_int = int(round(hue_stats['lo_abs'])) % 180
        hue_hi_int = int(round(hue_stats['hi_abs'])) % 180

        sat_min = max(0, int(np.percentile(sats,  5) - 10))
        val_min = max(0, int(np.percentile(vals,  5) - 20))

        # ── Thermodynamic thresholds, computed in the rotated frame so the
        # 35% / 65% split lands within the actual wax arc even when the
        # arc wraps. The values are stored in absolute OpenCV hue space;
        # blob_detection._classify_direction does the matching rotation.
        lo_rot     = hue_stats['lo_rot']
        range_rot  = hue_stats['range_rot']
        falling_rot = lo_rot + range_rot * 0.35
        rising_rot  = lo_rot + range_rot * 0.65
        falling_th  = (falling_rot + hue_mean - 90.0) % 180.0
        rising_th   = (rising_rot  + hue_mean - 90.0) % 180.0

        area_min = float(np.min(areas)) * 0.5 if areas else 200.0
        area_max = float(np.max(areas)) * 2.0 if areas else 50000.0

        # Lamp ROI
        if lamp_roi:
            lx1, ly1, lx2, ly2 = lamp_roi
        else:
            lx1, ly1, lx2, ly2 = 0, 0, w, h

        # Diagnostic: warn if the calibration looks dubious
        if hue_stats['std_abs'] > 30.0:
            print(f"[LampProfile] WARNING: hue std={hue_stats['std_abs']:.1f}° "
                  f"is large — wax distribution may be bi-modal or noisy. "
                  f"Re-draw on cleaner blob interiors.")
        if hue_stats['wraps']:
            print(f"[LampProfile] hue band wraps 0/180: "
                  f"H ∈ [{hue_lo_int}, 180] ∪ [0, {hue_hi_int}]  "
                  f"(circular mean = {hue_mean:.1f}°)")
        else:
            print(f"[LampProfile] hue band: H ∈ [{hue_lo_int}, {hue_hi_int}]  "
                  f"(circular mean = {hue_mean:.1f}°)")

        return cls(
            wax_hue_lo           = hue_lo_int,
            wax_hue_hi           = hue_hi_int,
            wax_sat_min          = sat_min,
            wax_val_min          = val_min,
            lamp_x1              = lx1,
            lamp_y1              = ly1,
            lamp_x2              = lx2,
            lamp_y2              = ly2,
            blob_area_min        = max(50.0, area_min),
            blob_area_max        = area_max,
            n_blobs_expected     = max(1, len(region_masks)),
            frame_h              = h,
            frame_w              = w,
            rising_hue_threshold = rising_th,
            falling_hue_threshold= falling_th,
            observed_hue_min     = float(np.min(hues)),
            observed_hue_max     = float(np.max(hues)),
            observed_hue_mean    = hue_mean,
        )
