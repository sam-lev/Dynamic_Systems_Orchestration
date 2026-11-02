#!/usr/bin/env python3
# test_detection.py -- Audible Lava v4.1
# Live visual test: detection pipeline + hue/direction overlay.
# q=quit  +/-=persistence  c=masks  h/H=hue  s/S=sat  v/V=val  b=rebaseline-bg
#
# v4.1 — background-subtraction A/B:
#   BACKGROUND_MODE selects how the foreground is isolated before the colour
#   mask is applied:
#     "frozen" : non-adaptive median reference captured once from the COLD lamp
#                (default). Never absorbs slow wax, so blobs don't fade. Press
#                'b' to re-capture (e.g. if you started on an active lamp). This
#                is the mode audible_lava.py now uses.
#     "mog2"   : adaptive MOG2 at a near-zero learning rate (MOG2_LEARNING_RATE).
#                Kept for comparison with the old behaviour.
#     "none"   : colour mask only.
#   Toggle the mask windows with 'c' to see the raw foreground mask, the colour
#   mask, and their intersection — use these to tune BG_REF_THRESHOLD / HSV.
#
#   Tuning keys edit the detector IN PLACE (no rebuild), so the background model
#   is preserved while you tune. (Rebuilding per-keypress was what caused the
#   old "flash then collapse" with MOG2.)

import cv2
import os
from lamp_profile  import LampProfile, PROFILE_PATH
from blob_detection import BlobDetector
from blob_detection import RISING, FALLING, TURNING

DIR_COLOURS = {RISING: (80,255,128), FALLING: (80,128,255), TURNING: (255,220,80)}
CAMERA_INDEX = 0

# ── Background mode (matches audible_lava.py options) ────────────────────────
BACKGROUND_MODE   = "frozen"   # "frozen" | "mog2" | "none"
BG_REF_FRAMES     = 40         # frozen: frames averaged into the reference
BG_REF_THRESHOLD  = 18         # frozen: |grey - reference| foreground cutoff
BG_REF_SETTLE     = 10         # frozen: frames discarded first (exposure settle)
BG_REF_METHOD     = "median"   # frozen: "median" (robust) or "mean" (low-memory)

# Only used when BACKGROUND_MODE == "mog2". Near-zero so slow wax isn't absorbed.
MOG2_LEARNING_RATE = 1e-4

def main():
    profile = LampProfile.load(PROFILE_PATH) or LampProfile.default()

    h_lo  = profile.wax_hue_lo
    h_hi  = profile.wax_hue_hi
    s_min = profile.wax_sat_min
    v_min = profile.wax_val_min
    thresh = max(8.0, profile.expected_persistence_mean
                      - 0.5 * profile.expected_persistence_std)
    show_masks = False

    def make_det():
        p2 = LampProfile.load(PROFILE_PATH) or LampProfile.default()
        p2.wax_hue_lo  = h_lo
        p2.wax_hue_hi  = h_hi
        p2.wax_sat_min = s_min
        p2.wax_val_min = v_min
        return BlobDetector(profile=p2, persistence_threshold=thresh,
                            background=BACKGROUND_MODE,
                            bg_ref_frames=BG_REF_FRAMES,
                            bg_ref_threshold=BG_REF_THRESHOLD,
                            bg_ref_settle=BG_REF_SETTLE,
                            bg_ref_method=BG_REF_METHOD)

    # Build the detector ONCE. Tuning keys edit it in place (see below).
    detector = make_det()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    profile.apply_to_capture(cap)
    print(f"Detection test  bg={BACKGROUND_MODE}  p={thresh:.0f}  "
          f"H=[{h_lo},{h_hi}]  S>={s_min}  V>={v_min}")
    if BACKGROUND_MODE == "frozen":
        print(f"Frozen reference: {BG_REF_FRAMES} frames ({BG_REF_METHOD}), "
              f"thr={BG_REF_THRESHOLD}, +{BG_REF_SETTLE} settle — keep the lamp "
              f"cold/settled until capture completes.")
    print("q=quit  +/-=thresh  c=masks  h/H=hue-hi  s/S=sat  v/V=val  "
          "b=rebaseline-bg")

    while True:
        ret, frame = cap.read()
        if not ret: break
        blobs   = detector.detect(frame, learning_rate=MOG2_LEARNING_RATE)
        overlay = frame.copy()

        for b in blobs:
            cx, cy = int(b.centroid[0]), int(b.centroid[1])
            col    = DIR_COLOURS.get(b.direction, (0,255,0))
            cv2.drawContours(overlay, [b.contour], -1, col, 2)
            cv2.circle(overlay, (cx, cy), 6, col, -1)
            cv2.putText(overlay,
                f"p={b.persistence:.0f} h={b.mean_hue:.0f} {b.direction[0].upper()}",
                (cx+10, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1)

        # Status line — show background-capture progress while it fills.
        if BACKGROUND_MODE == "frozen" and not detector.background_ready():
            done, tot = detector.background_progress()
            status = f"BG capture {done}/{tot}  (bg={BACKGROUND_MODE})"
        else:
            status = (f"Blobs:{len(blobs)}  p={thresh:.0f}  "
                      f"H=[{h_lo},{h_hi}]  bg={BACKGROUND_MODE}")
        cv2.putText(overlay, status,
            (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.imshow("Detection v4.1", overlay)

        if show_masks:
            m = detector.get_debug_masks(frame)
            cv2.imshow("Foreground (bg-sub)", m["foreground"])
            cv2.imshow("Wax colour",          m["colour"])
            cv2.imshow("Combined",            m["combined"])
        else:
            for w in ("Foreground (bg-sub)", "Wax colour", "Combined"):
                try: cv2.destroyWindow(w)
                except: pass

        # Tuning keys edit the live detector in place — no rebuild.
        key = cv2.waitKey(1) & 0xFF
        if   key == ord("q"):
            break
        elif key in (ord("+"), ord("=")):
            thresh += 5.0
            detector.pers_thresh = thresh
        elif key == ord("-"):
            thresh = max(1., thresh - 5.)
            detector.pers_thresh = thresh
        elif key == ord("c"):
            show_masks = not show_masks
        elif key == ord("b"):
            detector.reset_background()
            print("[Background] re-capturing reference"
                  if BACKGROUND_MODE == "frozen" else "[Background] MOG2 reset")
        elif key == ord("h"):
            h_hi = max(h_lo + 1, h_hi - 1)
            detector.profile.wax_hue_hi = h_hi
        elif key == ord("H"):
            h_hi = min(179, h_hi + 1)
            detector.profile.wax_hue_hi = h_hi
        elif key == ord("s"):
            s_min = max(0, s_min - 5)
            detector.profile.wax_sat_min = s_min
        elif key == ord("S"):
            s_min = min(255, s_min + 5)
            detector.profile.wax_sat_min = s_min
        elif key == ord("v"):
            v_min = max(0, v_min - 5)
            detector.profile.wax_val_min = v_min
        elif key == ord("V"):
            v_min = min(255, v_min + 5)
            detector.profile.wax_val_min = v_min

    cap.release(); cv2.destroyAllWindows()
    print(f"\nFinal values:\n  bg={BACKGROUND_MODE}  persistence={thresh:.0f}  "
          f"H=[{h_lo},{h_hi}]  S>={s_min}  V>={v_min}")

if __name__ == "__main__":
    main()
