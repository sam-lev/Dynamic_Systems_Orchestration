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

    def y_norm(self, pixel_y: float) -> float:
        """Normalise pixel y to [0=bottom, 1=top] within lamp bounds."""
        if self.lamp_height == 0:
            return 0.5
        norm = 1.0 - (pixel_y - self.lamp_y1) / self.lamp_height
        return float(np.clip(norm, 0.0, 1.0))

    def x_norm(self, pixel_x: float) -> float:
        """Normalise pixel x to [0=left, 1=right] within lamp bounds."""
        if self.lamp_width == 0:
            return 0.5
        return float(np.clip((pixel_x - self.lamp_x1) / self.lamp_width, 0.0, 1.0))

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

        # Circular mean and std for hue
        angles   = np.radians(hues * 2.0)
        sin_mean = np.mean(np.sin(angles))
        cos_mean = np.mean(np.cos(angles))
        hue_mean = float(np.degrees(np.arctan2(sin_mean, cos_mean)) / 2.0 % 180.0)

        # Use percentiles for robust range (ignore extreme pixels)
        hue_lo = float(np.percentile(hues,  5))
        hue_hi = float(np.percentile(hues, 95))
        # Add small margin
        margin      = 5.0
        hue_lo_int  = max(0,   int(hue_lo  - margin))
        hue_hi_int  = min(180, int(hue_hi  + margin))
        sat_min     = max(0,   int(np.percentile(sats,  5) - 10))
        val_min     = max(0,   int(np.percentile(vals,  5) - 20))

        # Thermodynamic thresholds: split the hue range at 1/3 and 2/3
        hue_range  = hue_hi - hue_lo
        falling_th = hue_lo + hue_range * 0.35
        rising_th  = hue_lo + hue_range * 0.65

        area_min = float(np.min(areas)) * 0.5 if areas else 200.0
        area_max = float(np.max(areas)) * 2.0 if areas else 50000.0

        # Lamp ROI
        if lamp_roi:
            lx1, ly1, lx2, ly2 = lamp_roi
        else:
            lx1, ly1, lx2, ly2 = 0, 0, w, h

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
