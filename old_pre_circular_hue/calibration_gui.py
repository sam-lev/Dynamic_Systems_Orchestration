# calibration_gui.py -- Audible Lava v3.0
# Popup calibration GUI. Opens on first run (or when --calibrate passed).
# User draws freehand around visible wax blobs on a frozen camera frame,
# then draws a rectangle to set the lamp ROI. The GUI extracts HSV stats
# from drawn regions and saves a LampProfile to disk.
#
# Controls:
#   SPACE         -- freeze/unfreeze live feed
#   Left drag     -- draw freehand blob region
#   Right drag    -- set lamp ROI rectangle
#   C             -- clear last drawn region
#   Shift+C       -- clear all drawn regions
#   ENTER / S     -- save profile and exit
#   Q             -- exit without saving (use defaults)

import cv2
import numpy as np
from typing import List, Optional, Tuple
from lamp_profile import LampProfile


# ── Colours for overlay rendering ─────────────────────────────────────────────
REGION_COLOURS = [
    (0, 255, 128),
    (0, 200, 255),
    (255, 180, 0),
    (200, 0, 255),
    (0, 255, 0),
    (255, 80, 80),
]
ROI_COLOUR   = (80, 200, 255)
PANEL_BG     = (30, 30, 30)
TEXT_WHITE   = (255, 255, 255)
TEXT_YELLOW  = (80, 220, 255)
TEXT_GREEN   = (80, 255, 128)


class CalibrationGUI:
    """
    OpenCV-based freehand calibration GUI.

    Captures a still frame from the camera, then lets the user draw
    freehand polygon regions around each visible wax blob. From those
    regions, HSV statistics are computed and a LampProfile is built and
    saved to disk.

    Usage:
        profile = CalibrationGUI.run(cap)
        if profile is None:
            profile = LampProfile.default()
    """

    def __init__(self, frame: np.ndarray):
        self.frame         = frame.copy()
        self.h, self.w     = frame.shape[:2]

        # State
        self.regions:  List[np.ndarray] = []   # list of (h×w) uint8 masks
        self.strokes:  List[List]       = []   # raw polyline points per region
        self.current_stroke: List       = []   # points being drawn right now
        self.is_drawing: bool           = False

        self.lamp_roi: Optional[Tuple[int,int,int,int]] = None
        self.roi_start: Optional[Tuple[int,int]]        = None
        self.roi_drag:  Optional[Tuple[int,int]]        = None
        self.is_drawing_roi: bool                       = False

        self.saved   = False
        self.profile: Optional[LampProfile] = None

    # ── Mouse callback ─────────────────────────────────────────────────────
    def _mouse(self, event, x, y, flags, param):
        # ── Left button: freehand blob region ─────────────────────────────
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing    = True
            self.current_stroke = [(x, y)]

        elif event == cv2.EVENT_MOUSEMOVE and self.is_drawing:
            self.current_stroke.append((x, y))

        elif event == cv2.EVENT_LBUTTONUP and self.is_drawing:
            self.is_drawing = False
            self.current_stroke.append((x, y))
            if len(self.current_stroke) >= 3:
                # Build mask from polygon
                mask = np.zeros((self.h, self.w), dtype=np.uint8)
                pts  = np.array(self.current_stroke, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
                self.regions.append(mask)
                self.strokes.append(list(self.current_stroke))
            self.current_stroke = []

        # ── Right button: lamp ROI rectangle ──────────────────────────────
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.is_drawing_roi = True
            self.roi_start = (x, y)
            self.roi_drag  = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.is_drawing_roi:
            self.roi_drag = (x, y)

        elif event == cv2.EVENT_RBUTTONUP and self.is_drawing_roi:
            self.is_drawing_roi = False
            if self.roi_start and self.roi_drag:
                x1 = min(self.roi_start[0], self.roi_drag[0])
                y1 = min(self.roi_start[1], self.roi_drag[1])
                x2 = max(self.roi_start[0], self.roi_drag[0])
                y2 = max(self.roi_start[1], self.roi_drag[1])
                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    self.lamp_roi = (x1, y1, x2, y2)

    # ── Rendering ──────────────────────────────────────────────────────────
    def _render(self) -> np.ndarray:
        canvas = self.frame.copy()
        frame_hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

        # Draw completed regions
        for i, (mask, stroke) in enumerate(zip(self.regions, self.strokes)):
            col = REGION_COLOURS[i % len(REGION_COLOURS)]
            overlay = canvas.copy()
            overlay[mask > 0] = (
                overlay[mask > 0].astype(np.float32) * 0.4
                + np.array(col, dtype=np.float32) * 0.6
            ).astype(np.uint8)
            canvas = cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0)
            # Draw polygon outline
            pts = np.array(stroke, dtype=np.int32)
            cv2.polylines(canvas, [pts], True, col, 2)
            # Region label
            cy = int(np.mean([p[1] for p in stroke]))
            cx = int(np.mean([p[0] for p in stroke]))
            cv2.putText(canvas, f"Blob {i+1}", (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        # Draw active stroke
        if self.current_stroke and len(self.current_stroke) > 1:
            pts = np.array(self.current_stroke, dtype=np.int32)
            cv2.polylines(canvas, [pts], False, (255, 255, 80), 2)

        # Draw ROI
        if self.lamp_roi:
            x1, y1, x2, y2 = self.lamp_roi
            cv2.rectangle(canvas, (x1, y1), (x2, y2), ROI_COLOUR, 2)
            cv2.putText(canvas, "Lamp ROI", (x1 + 4, y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, ROI_COLOUR, 2)
        elif self.roi_drag and self.roi_start:
            cv2.rectangle(canvas, self.roi_start, self.roi_drag, ROI_COLOUR, 1)

        # ── Info panel (right side) ────────────────────────────────────────
        panel_x = self.w - 310
        cv2.rectangle(canvas, (panel_x, 0), (self.w, self.h), PANEL_BG, -1)

        lines = [
            ("AUDIBLE LAVA -- Calibration",  TEXT_YELLOW, 0.55),
            ("",                              TEXT_WHITE,  0.45),
            ("LEFT DRAG: draw blob region",   TEXT_WHITE,  0.42),
            ("RIGHT DRAG: set lamp ROI",      TEXT_WHITE,  0.42),
            ("C: clear last region",          TEXT_WHITE,  0.42),
            ("SHIFT+C: clear all",            TEXT_WHITE,  0.42),
            ("ENTER or S: save + run",        TEXT_WHITE,  0.42),
            ("Q: skip (use defaults)",        TEXT_WHITE,  0.42),
            ("",                              TEXT_WHITE,  0.42),
            (f"Regions drawn: {len(self.regions)}", TEXT_GREEN, 0.50),
            (f"ROI: {'set' if self.lamp_roi else 'not set (full frame)'}",
             TEXT_GREEN if self.lamp_roi else TEXT_WHITE, 0.50),
        ]
        y = 22
        for text, col, size in lines:
            cv2.putText(canvas, text, (panel_x + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, size, col, 1)
            y += int(size * 36 + 4)

        # ── Extracted HSV stats ────────────────────────────────────────────
        if self.regions:
            y += 8
            cv2.putText(canvas, "-- Extracted colour --",
                        (panel_x + 8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.44, TEXT_YELLOW, 1)
            y += 20

            all_hues, all_sats, all_vals = [], [], []
            for mask in self.regions:
                px = frame_hsv[mask > 0]
                if len(px):
                    all_hues.extend(px[:, 0].tolist())
                    all_sats.extend(px[:, 1].tolist())
                    all_vals.extend(px[:, 2].tolist())

            if all_hues:
                h_lo = int(np.percentile(all_hues,  5))
                h_hi = int(np.percentile(all_hues, 95))
                s_lo = int(np.percentile(all_sats,  5))
                v_lo = int(np.percentile(all_vals,  5))
                stats = [
                    f"Hue:  {h_lo} – {h_hi}",
                    f"Sat >= {s_lo}",
                    f"Val >= {v_lo}",
                    f"Blobs: {len(self.regions)}",
                ]
                for stat in stats:
                    cv2.putText(canvas, stat, (panel_x + 8, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, TEXT_GREEN, 1)
                    y += 18

                # Colour swatch
                y += 4
                swatch_h_lo = max(0,   h_lo - 5)
                swatch_h_hi = min(180, h_hi + 5)
                swatch = np.zeros((30, 280, 3), dtype=np.uint8)
                for xi in range(280):
                    hue = swatch_h_lo + (swatch_h_hi - swatch_h_lo) * xi / 280
                    swatch[:, xi] = cv2.cvtColor(
                        np.array([[[int(hue), 180, 220]]], dtype=np.uint8),
                        cv2.COLOR_HSV2BGR
                    )[0, 0]
                canvas[y:y+30, panel_x+8:panel_x+288] = swatch
                y += 36
                cv2.putText(canvas, "  Detected hue range",
                            (panel_x + 8, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_WHITE, 1)

        return canvas

    # ── Main run loop ──────────────────────────────────────────────────────
    def run(self) -> Optional[LampProfile]:
        win = "Audible Lava -- Calibration (draw blobs, then ENTER to save)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, min(self.w + 320, 1600), min(self.h, 900))
        cv2.setMouseCallback(win, self._mouse)

        while True:
            canvas = self._render()
            cv2.imshow(win, canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q") or key == 27:   # Q / ESC = skip
                break

            elif key == ord("c"):
                if self.regions:
                    self.regions.pop()
                    self.strokes.pop()

            elif key == ord("C"):              # Shift+C = clear all
                self.regions.clear()
                self.strokes.clear()

            elif key in (13, ord("s")):        # ENTER or S = save
                if not self.regions:
                    cv2.putText(canvas, "Draw at least one blob region first!",
                                (30, self.h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.imshow(win, canvas)
                    cv2.waitKey(1500)
                    continue
                self.profile = LampProfile.from_drawn_regions(
                    self.regions, self.frame, self.lamp_roi
                )
                self.profile.save()
                self.saved = True
                break

        cv2.destroyWindow(win)
        return self.profile


# ── Entry point ────────────────────────────────────────────────────────────────
def run_calibration(camera_index: int = 0,
                    profile_path: str = "lamp_profile.json",
                    *,
                    frame_w: int = 1280,
                    frame_h: int = 720) -> LampProfile:
    """
    Open the calibration GUI and return a LampProfile.
    If the user skips (presses Q), returns a default profile.

    Called automatically from audible_lava.py if no profile exists,
    or explicitly with:  python calibration_gui.py [camera_idx] [W H]

    The frame_w / frame_h kwargs control the capture resolution requested
    from the camera. The resulting profile records whatever resolution
    the camera actually delivers, and downstream scripts (test_detection,
    test_tracker, audible_lava, diagnose_tracking) re-apply that resolution
    via LampProfile.apply_to_capture so every stage of the pipeline sees
    the same pixel scale that blob_area_min / lamp ROI were calibrated at.
    """
    import time

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[Calibration] Cannot open camera -- using default profile")
        return LampProfile.default(frame_h=frame_h, frame_w=frame_w)

    # Set resolution BEFORE the first cap.read(). macOS UVC backends ignore
    # CAP_PROP_FRAME_* set after streaming has started.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_w, actual_h) != (frame_w, frame_h):
        print(f"[Calibration] WARNING: requested {frame_w}x{frame_h}, "
              f"camera delivered {actual_w}x{actual_h}")
    else:
        print(f"[Calibration] Capture resolution: {actual_w}x{actual_h}")

    print("[Calibration] Live feed open. Press SPACE to freeze frame for drawing.")
    print("              Press Q to skip calibration and use defaults.")

    frozen_frame = None
    win_live     = "Audible Lava -- Live feed (SPACE to freeze, Q to skip)"
    cv2.namedWindow(win_live, cv2.WINDOW_NORMAL)

    while frozen_frame is None:
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()
        cv2.putText(display,
                    "SPACE = freeze and draw    Q = skip calibration",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (80, 255, 128), 2)
        cv2.imshow(win_live, display)
        key = cv2.waitKey(30) & 0xFF
        if key == 32:   # SPACE
            frozen_frame = frame.copy()
        elif key in (ord("q"), 27):
            break

    cv2.destroyWindow(win_live)
    cap.release()

    if frozen_frame is None:
        print("[Calibration] Skipped -- using default profile")
        return LampProfile.default(frame_h=frame_h, frame_w=frame_w)

    gui     = CalibrationGUI(frozen_frame)
    profile = gui.run()

    if profile is None:
        print("[Calibration] Skipped -- using default profile")
        h, w = frozen_frame.shape[:2]
        return LampProfile.default(frame_h=h, frame_w=w)

    print(f"[Calibration] Profile saved.")
    print(f"  Hue range  : {profile.wax_hue_lo} – {profile.wax_hue_hi}")
    print(f"  Sat min    : {profile.wax_sat_min}")
    print(f"  Val min    : {profile.wax_val_min}")
    print(f"  Lamp ROI   : ({profile.lamp_x1},{profile.lamp_y1}) – "
          f"({profile.lamp_x2},{profile.lamp_y2})")
    print(f"  Blobs drawn: {profile.n_blobs_expected}")
    return profile


if __name__ == "__main__":
    import sys
    cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    fw  = int(sys.argv[2]) if len(sys.argv) > 2 else 1280
    fh  = int(sys.argv[3]) if len(sys.argv) > 3 else 720
    run_calibration(cam, frame_w=fw, frame_h=fh)
