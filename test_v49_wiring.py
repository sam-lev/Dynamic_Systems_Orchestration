#!/usr/bin/env python3
# test_v49_wiring.py -- Audible Lava / DSO v4.9
#
# Covers the four v4.9 changes. No SuperCollider, camera, or pygame.
#   1. routing.py   : v3.2 base routings restored (Y_NORM->PITCH, ELONGATION->RES)
#   2. blob_detection.py : warm-lamp capture reachable through BlobDetector
#   3. audible_lava.py   : merge-tree wax mask is requested with the ROI
#                          (regression test reproduces the old silent failure)
#   4. melody_rhythm.py  : wired into the main loop (import + call-site check)
#
# Run: python test_v49_wiring.py

import os
import re
import sys

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PASS = FAIL = 0
def check(cond, msg):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok   {msg}")
    else:    FAIL += 1; print(f"  FAIL {msg}")


def test_routing_defaults():
    print("[routing] v3.2 base set restored")
    from routing import DEFAULT_BLOB_ROUTING, Source, Dest
    r = {(x.src, x.dest): x.depth for x in DEFAULT_BLOB_ROUTING}
    check(r.get((Source.Y_NORM, Dest.PITCH)) == 12.0, "Y_NORM -> PITCH +12")
    check(r.get((Source.ELONGATION, Dest.RES)) == 0.6, "ELONGATION -> RES +0.6")
    check((Source.Y_NORM, Dest.RES) not in r, "Y_NORM -> RES removed")
    check((Source.ELONGATION, Dest.PITCH) not in r, "ELONGATION -> PITCH removed")
    pitch_srcs = {s for (s, d) in r if d == Dest.PITCH}
    check(Source.Y_NORM in pitch_srcs, "height is a continuous PITCH input again")
    check(len(DEFAULT_BLOB_ROUTING) == 24, f"routing count unchanged ({len(DEFAULT_BLOB_ROUTING)})")


def test_warm_capture_reachable():
    print("[detector] warm-lamp capture plumbed through BlobDetector")
    from blob_detection import BlobDetector
    from lamp_profile import LampProfile
    p = LampProfile.default(240, 320)
    d = BlobDetector(profile=p, background="frozen")
    check(d.bg_ref.mode == "cold" and d.bg_ref.method == "median",
          "defaults unchanged (cold median)")
    w = BlobDetector(profile=p, background="frozen", bg_ref_mode="warm",
                     bg_ref_capture_secs=1.0, bg_ref_percentile=30.0,
                     bg_ref_warm_samples=40)
    check(w.bg_ref.mode == "warm" and w.bg_ref.method == "percentile",
          "bg_ref_mode='warm' reaches BackgroundReference")
    check(w.bg_ref.percentile == 30.0 and w.bg_ref.capture_secs == 1.0,
          "percentile / capture_secs pass through")
    check("warm" in w.background_status(), "background_status() reports mode")
    d.reset_background(mode="warm")
    check(d.bg_ref.mode == "warm" and d.bg_ref_mode == "warm",
          "reset_background(mode='warm') switches a running detector")
    try:
        BlobDetector(profile=p, background="frozen", bg_ref_mode="tepid")
        check(False, "invalid mode rejected")
    except ValueError:
        check(True, "invalid mode rejected")


def test_wax_mask_regression():
    print("[detector] merge-tree wax mask: full frame vs ROI")
    from blob_detection import BlobDetector
    from lamp_profile import LampProfile
    p = LampProfile.default(240, 320)
    full = np.random.randint(0, 255, (480, 640, 3), np.uint8)
    roi  = full[100:340, 200:520]                   # view, as profile.crop() returns
    grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    det  = BlobDetector(profile=p, background="frozen", bg_ref_frames=2, bg_ref_settle=0)
    for _ in range(3):
        det.detect(roi, grey=grey)
    raised = False
    try:
        det.get_debug_masks(full, grey=grey)
    except cv2.error:
        raised = True
    check(raised, "full frame + ROI grey raises (the pre-v4.9 silent-None path)")
    m = det.get_debug_masks(roi, grey=grey)
    check(m is det._mask_cache, "ROI call is served from detect()'s cache")
    check(m["combined"].shape == grey.shape, "combined mask is ROI-sized")


def test_main_loop_wiring():
    print("[audible_lava] call sites present (source inspection)")
    src = open(os.path.join(HERE, "audible_lava.py"), encoding="utf-8").read()
    check("from melody_rhythm" in src and "MelodyEngine(" in src, "MelodyEngine imported + constructed")
    check("melody.process_frame(tracked)" in src, "melody.process_frame called per frame")
    i_voices = src.find("voices.process_frame(tracked)")
    i_melody = src.find("melody.process_frame(tracked)")
    check(0 < i_voices < i_melody, "melody runs after voices.process_frame")
    check(re.search(r'ord\("n"\)', src) is not None and "melody.toggle()" in src, "'n' toggles melody")
    check("get_debug_masks(roi, grey=grey)" in src, "merge-tree wax mask requested with roi")
    check("get_debug_masks(frame, grey=grey)" not in src, "no full-frame wax-mask call remains")
    check("bg_ref_mode=BG_REF_MODE" in src, "BG_REF_MODE passed to BlobDetector")
    check(re.search(r'ord\("w"\)', src) is not None and 'reset_background(mode="warm")' in src,
          "'w' requests a warm re-capture")


if __name__ == "__main__":
    test_routing_defaults()
    test_warm_capture_reachable()
    test_wax_mask_regression()
    test_main_loop_wiring()
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
