#!/usr/bin/env python3
# test_v4101_ynorm.py -- Audible Lava / DSO v4.10.1
#
# Regression test for the Source.Y_NORM inversion. No SuperCollider, camera
# or pygame; only routing.py and lamp_profile.py are imported.
#
#   before v4.10.1: extract_source_value(Y_NORM) = 1 - profile.y_norm -> 1 at
#                   the BOTTOM, so Y_NORM -> PITCH +7 raised bottom blobs and a
#                   rising blob fell a fifth.
#   after:          Y_NORM = profile.y_norm -> 1 at the top; the routed pitch
#                   contribution rises with the blob, as routing.py documents.
#
# Run: python test_v4101_ynorm.py

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from routing import (Source, Dest, DEFAULT_BLOB_ROUTING, extract_source_value,
                     apply_curve)

PASS = FAIL = 0
def check(cond, msg):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok   {msg}")
    else:    FAIL += 1; print(f"  FAIL {msg}")


class _Blob:
    def __init__(self, cy):
        self.centroid = (100.0, cy)


def _profile():
    """The real LampProfile when importable (needs numpy); else a stand-in
    with the same y_norm contract (1 = top)."""
    try:
        from lamp_profile import LampProfile
        p = LampProfile.default()
        p.frame_w, p.frame_h = 1280, 720
        p.lamp_x1, p.lamp_y1, p.lamp_x2, p.lamp_y2 = 361, 68, 729, 579
        return p, "LampProfile"
    except Exception:
        class P:
            lamp_y1, lamp_y2 = 68, 579
            @property
            def roi_h(self): return self.lamp_y2 - self.lamp_y1
            def y_norm(self, roi_y):
                return float(max(0.0, min(1.0, 1.0 - roi_y / self.roi_h)))
            def x_norm(self, roi_x): return 0.5
        return P(), "stand-in"


def test_profile_contract():
    p, kind = _profile()
    print(f"[profile] y_norm contract ({kind})")
    H = p.roi_h
    check(abs(p.y_norm(0.0) - 1.0) < 1e-9, "profile.y_norm(top edge) == 1")
    check(abs(p.y_norm(H) - 0.0) < 1e-9, "profile.y_norm(bottom edge) == 0")


def test_source_follows_profile():
    p, _ = _profile()
    print("[routing] Source.Y_NORM == profile.y_norm")
    H = p.roi_h
    top, mid, bottom = _Blob(0.0), _Blob(H / 2), _Blob(H)
    yt = extract_source_value(Source.Y_NORM, top, p)
    ym = extract_source_value(Source.Y_NORM, mid, p)
    yb = extract_source_value(Source.Y_NORM, bottom, p)
    check(abs(yt - 1.0) < 1e-9, f"top blob -> 1.0 (got {yt:.3f})")
    check(abs(ym - 0.5) < 1e-9, f"middle blob -> 0.5 (got {ym:.3f})")
    check(abs(yb - 0.0) < 1e-9, f"bottom blob -> 0.0 (got {yb:.3f})")
    check(yt > yb, "the source rises with the blob (was the reverse)")


def test_pitch_routing_rises():
    p, _ = _profile()
    print("[routing] Y_NORM -> PITCH contribution rises with height")
    H = p.roi_h
    r = next(x for x in DEFAULT_BLOB_ROUTING if x.src is Source.Y_NORM and x.dest is Dest.PITCH)
    semis = {name: apply_curve(r.curve, extract_source_value(Source.Y_NORM, _Blob(cy), p)) * r.depth
             for name, cy in (("bottom", H), ("middle", H / 2), ("top", 0.0))}
    check(abs(semis["bottom"]) < 1e-9, f"bottom contributes 0 st (got {semis['bottom']:+.2f})")
    check(abs(semis["top"] - r.depth) < 1e-9, f"top contributes +{r.depth:g} st (got {semis['top']:+.2f})")
    check(semis["bottom"] < semis["middle"] < semis["top"], "monotone: bottom < middle < top")


def test_melody_and_drone_agree():
    """melody_rhythm reads profile.y_norm directly; the routed drone source
    must now be the same number for the same blob."""
    p, _ = _profile()
    print("[consistency] routing height == melody height")
    for cy in (0.0, 0.25 * p.roi_h, 0.9 * p.roi_h):
        b = _Blob(cy)
        check(abs(extract_source_value(Source.Y_NORM, b, p) - p.y_norm(cy)) < 1e-9,
              f"cy={cy:.0f}: identical")


if __name__ == "__main__":
    test_profile_contract()
    test_source_follows_profile()
    test_pitch_routing_rises()
    test_melody_and_drone_agree()
    print(f"Results: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
