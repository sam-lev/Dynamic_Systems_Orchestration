# test_tracker_stability.py -- Audible Lava / DSO v4.13
#
# Does StableBlobTracker keep ONE id (one voice) for a blob whose detected
# region fluctuates, while still tracking real motion, real births and real
# splits? Every scenario renders synthetic grey frames (so the real Farneback
# flow and the real buoyancy Kalman run) and builds detections from masks.
# Where it matters, the plain BlobTracker is run on the same input to show the
# failure being fixed.
#
# No camera, SuperCollider or pygame.
# Run: python test_tracker_stability.py  |  pytest test_tracker_stability.py -v

import math
import os
import sys
import types

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# blob_detection imports GUDHI-backed helpers that these tests never call.
for _mod, _attrs in (("h0_persistence", {"h0_peaks": lambda *a, **k: []}),
                     ("background_reference", {"BackgroundReference": type("BR", (), {"__init__": lambda s, *a, **k: None})})):
    try:
        __import__(_mod)
    except Exception:
        m = types.ModuleType(_mod)
        m.__dict__.update(_attrs)
        sys.modules[_mod] = m

import blob_tracker as BT                                # noqa: E402
from blob_tracker      import BlobTracker               # noqa: E402
from blob_detection    import PersistentBlob, TURNING, RISING  # noqa: E402
from lamp_profile      import LampProfile               # noqa: E402
from tracker_stability import StableBlobTracker, _shape_from_contour  # noqa: E402

H, W = 400, 300
TESTS = []
def case(fn):
    TESTS.append(fn); return fn


def profile():
    return LampProfile.default(frame_h=H, frame_w=W)


def ellipse(cx, cy, ax, ay, angle=0.0):
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (int(round(cx)), int(round(cy))), (int(ax), int(ay)), angle, 0, 360, 255, -1)
    return m


def det_from_mask(m, birth=200.0, death=20.0):
    cs, _ = cv2.findContours((m > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    M = cv2.moments((m > 0).astype(np.uint8) * 255)
    o, a, e, hu = _shape_from_contour(c)
    return PersistentBlob(centroid=(M["m10"] / M["m00"], M["m01"] / M["m00"]),
                          area=float(np.count_nonzero(m)), birth=birth, death=death,
                          persistence=birth - death, contour=c, orientation=o,
                          aspect_ratio=a, elongation=e, hu=hu, mean_hue=160.0,
                          direction=TURNING)


def frame_of(masks):
    g = np.full((H, W), 20, np.uint8)
    for m in masks:
        g[m > 0] = 200
    g = cv2.GaussianBlur(g, (0, 0), 2)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), g


def run(tracker, cycles, K=1):
    """cycles: list of lists of masks (one list per detection cycle).
    Runs K-1 predict-only frames after each detection frame (same image).
    Returns (births, ids_per_cycle, centroid_log, events)."""
    births, ids, cents, events = 0, [], [], []
    for masks in cycles:
        fr, g = frame_of(masks)
        out = tracker.update([det_from_mask(m) for m in masks], fr, grey=g)
        births += sum(1 for e in tracker.events if e[0] == "birth")
        events.extend(tracker.events)
        ids.append(sorted(b.id for b in out))
        cents.append({b.id: b.centroid for b in out})
        for _ in range(K - 1):
            out = tracker.update(None, fr, grey=g)
            cents.append({b.id: b.centroid for b in out})
    return births, ids, cents, events


def fluctuating_still_blob(n=40, cx=150, cy=45):
    """A blob that does not move, near the top of the lamp (inside the 12 %
    margin where the base trusts its buoyancy model), whose detected region
    cycles through shapes that change in place: whole, one lobe, tall-narrow,
    shifted lobe."""
    # Each alternate shape overlaps the whole blob by IoU < 0.25 (the base's
    # association threshold), though all sit inside it.
    shapes = [ellipse(cx, cy, 34, 18),
              ellipse(cx - 18, cy + 2, 12, 9),
              ellipse(cx + 2, cy - 1, 7, 15),
              ellipse(cx + 19, cy + 1, 11, 8)]
    return [[shapes[0]] if i % 2 == 0 else [shapes[1 + (i // 2) % 3]] for i in range(n)]


# ── the reported failure ────────────────────────────────────────────────────

@case
def ts_01_still_fluctuating_blob_keeps_one_id():
    seq = fluctuating_still_blob()
    b_base, ids_base, _, _ = run(BlobTracker(profile=profile()), seq)
    b_st, ids_st, _, _ = run(StableBlobTracker(profile=profile()), seq)
    assert b_base > 1, f"sanity: the base tracker should split this blob (births={b_base})"
    assert b_st == 1, f"stable tracker gave {b_st} births (base {b_base})"
    seen = {i for ids in ids_st for i in ids}
    assert len(seen) == 1, f"ids seen {seen}"
    assert all(len(ids) == 1 for ids in ids_st[3:]), "more than one visible track at once"


@case
def ts_01b_still_blob_centroid_holds_still():
    """Its voice's pitch and pan follow the centroid: for a blob that does not
    move, that must not jump to whichever core or lobe was detected."""
    seq = fluctuating_still_blob()
    def spread(tk):
        _, _, cents, _ = run(tk, seq)
        pts = [c[k] for c in cents[4:] for k in c]
        return (max(p[0] for p in pts) - min(p[0] for p in pts),
                max(p[1] for p in pts) - min(p[1] for p in pts))
    bx, by = spread(BlobTracker(profile=profile()))
    sx, sy = spread(StableBlobTracker(profile=profile()))
    assert sx < 12 and sy < 6, f"stable centroid spread {sx:.1f} x {sy:.1f} px"
    assert sx < bx / 4, f"stable {sx:.1f}px vs base {bx:.1f}px"


@case
def ts_01c_a_real_shrink_is_accepted():
    """A blob that genuinely shrinks to a third and stays small is accepted
    after partial_max_run cycles (not held at its old size for ever)."""
    big, small = ellipse(150, 200, 30, 20), ellipse(150, 200, 12, 10)
    seq = [[big]] * 6 + [[small]] * 14
    tk = StableBlobTracker(profile=profile(), partial_max_run=8)
    b, _, _, _ = run(tk, seq)
    blob = tk.get_active()[0]
    assert b == 1, f"births {b}"
    assert blob.area < 0.5 * np.count_nonzero(big), f"area still {blob.area:.0f}"


@case
def ts_02_coasting_does_not_fly_away():
    """Jittered measurements near the top give the base filter a stale
    velocity; then the blob drops out for 6 cycles of K=3 frames. The stable
    track's centroid must stay within one radius of where it was last seen."""
    cx, cy = 150, 40
    seq = [[ellipse(cx, cy + (6 if i % 2 else -6) + i * 0.6, 26, 18)] for i in range(14)]
    seq += [[] for _ in range(6)]
    last_y = cy + 6 + 13 * 0.6
    r = math.sqrt(math.pi * 26 * 18 / math.pi)

    def drift(tk):
        _, _, cents, _ = run(tk, seq, K=3)
        ys = [c[k][1] for c in cents[-18:] for k in c]
        return max((abs(y - last_y) for y in ys), default=0.0)

    d_base = drift(BlobTracker(profile=profile()))
    d_st = drift(StableBlobTracker(profile=profile()))
    assert d_st <= 1.05 * r, f"stable coasting drift {d_st:.1f}px > radius {r:.1f}px"
    assert d_st < d_base, f"no improvement: stable {d_st:.1f}px, base {d_base:.1f}px"


@case
def ts_03_vanish_and_return_changed_shape_same_id():
    cx, cy = 150, 200
    seq = [[ellipse(cx, cy, 30, 18)] for _ in range(6)]
    seq += [[] for _ in range(3)]
    seq += [[ellipse(cx + 4, cy - 3, 12, 28)] for _ in range(6)]   # back, other shape
    b, ids, _, _ = run(StableBlobTracker(profile=profile()), seq, K=2)
    assert b == 1, f"births {b}"
    assert len({i for x in ids for i in x}) == 1, f"ids {ids}"
    assert ids[-1], "blob not tracked after it came back"


@case
def ts_04_watershed_fragments_are_fused():
    """The same blob cut in two by a 1 px watershed line for some cycles."""
    cx, cy = 150, 200
    whole = ellipse(cx, cy, 34, 20)
    left, right = whole.copy(), whole.copy()
    left[:, cx:] = 0
    right[:, :cx + 1] = 0
    seq = [[whole]] * 5 + [[left, right]] * 6 + [[whole]] * 3
    tk = StableBlobTracker(profile=profile())
    b, ids, cents, _ = run(tk, seq)
    assert b == 1, f"births {b}"
    assert tk.stats["fused"] >= 6, f"fused {tk.stats['fused']}"
    tid = ids[-1][0]
    for c in cents[5:11]:
        assert abs(c[tid][0] - cx) < 4.0, f"centroid pulled to a fragment: x={c[tid][0]:.1f}"
    b_base, _, _, _ = run(BlobTracker(profile=profile()), seq)
    assert b_base > 1, f"sanity: base should birth the fragment (births={b_base})"


# ── nothing real is lost ────────────────────────────────────────────────────

@case
def ts_05_real_new_blob_is_born_after_confirmation_flicker_is_not():
    a = ellipse(80, 200, 24, 16)
    b = ellipse(220, 300, 22, 14)
    flick = ellipse(220, 80, 16, 12)
    seq = [[a]] * 4 + [[a, flick]] + [[a]] * 2 + [[a, b]] * 5
    tk = StableBlobTracker(profile=profile(), confirm_cycles=3)
    births_at, visible = [], []
    for i, masks in enumerate(seq):
        fr, g = frame_of(masks)
        out = tk.update([det_from_mask(m) for m in masks], fr, grey=g)
        births_at += [(i, e[1]) for e in tk.events if e[0] == "birth"]
        visible.append(len(out))
    assert len(births_at) == 2, f"births {births_at}"
    assert births_at[1][0] == 7 + 3 - 1, f"second birth on cycle {births_at[1][0]}, expected 9"
    assert visible[4] == 1, "the one-cycle flicker was visible"
    assert tk.stats["tentative_dropped"] == 1
    assert not any(e[0] == "death" for e in tk.events)


@case
def ts_06_moving_blob_keeps_its_motion():
    """A blob rising 1.5 px/frame must keep one id, be classed RISING, and not
    be held as stationary."""
    seq = [[ellipse(150, 330 - 1.5 * i, 24, 16)] for i in range(60)]
    tk = StableBlobTracker(profile=profile())
    b, ids, cents, _ = run(tk, seq)
    assert b == 1, f"births {b}"
    blob = tk.get_active()[0]
    assert blob.direction == RISING, f"direction {blob.direction}, vy {blob.vy:.2f}"
    assert blob.vy < -0.8, f"motion damped away: vy {blob.vy:.2f}"
    assert abs(blob.centroid[1] - (330 - 1.5 * 59)) < 6


@case
def ts_06b_moving_blob_seen_in_part_keeps_moving():
    """A rising blob whose detection alternates with a core view still rises
    at about the right rate (completion does not freeze real motion)."""
    seq = []
    for i in range(50):
        y = 330 - 1.2 * i
        seq.append([ellipse(150, y, 26, 18)] if i % 2 == 0 else [ellipse(144, y, 10, 8)])
    tk = StableBlobTracker(profile=profile())
    b, _, _, _ = run(tk, seq)
    blob = tk.get_active()[0]
    assert b == 1, f"births {b}"
    assert abs(blob.centroid[1] - (330 - 1.2 * 49)) < 10, f"y {blob.centroid[1]:.1f}"
    assert blob.direction == RISING, f"direction {blob.direction}"


@case
def ts_07_neighbouring_blobs_stay_distinct():
    """Two still blobs 70 px apart, both fluctuating: two ids, never swapped,
    never fused."""
    def shapes(cx, cy):
        return [ellipse(cx, cy, 24, 14), ellipse(cx - 8, cy, 14, 12), ellipse(cx, cy, 10, 20)]
    sa, sb = shapes(115, 200), shapes(185, 200)
    seq = [[sa[i % 3], sb[(i + 1) % 3]] for i in range(30)]
    tk = StableBlobTracker(profile=profile())
    b, ids, cents, _ = run(tk, seq)
    assert b == 2, f"births {b}"
    final = {bid: c for bid, c in cents[-1].items()}
    xs = sorted(c[0] for c in final.values())
    assert len(final) == 2 and xs[0] < 140 < xs[1], f"tracks {final}"
    assert tk.stats["fused"] == 0


@case
def ts_08_persistent_disjoint_fragment_confirms_as_split():
    whole = ellipse(150, 200, 40, 20)
    big = ellipse(138, 200, 26, 18)
    small = ellipse(180, 200, 7, 7)
    seq = [[whole]] * 5 + [[big, small]] * 5
    tk = StableBlobTracker(profile=profile())
    _, _, _, events = run(tk, seq)
    splits = [e for e in events if e[0] == "split"]
    births = [e for e in events if e[0] == "birth"]
    assert len(births) == 1, f"births {births}"
    assert len(splits) == 1, f"events {events}"


# ── additivity ──────────────────────────────────────────────────────────────

@case
def ts_09_all_fixes_off_matches_the_base_exactly():
    seq = fluctuating_still_blob(24) + [[]] * 3 + fluctuating_still_blob(8)
    off = dict(fuse_fragments=False, reassociate=False, confirm_cycles=1,
               damp_coasting=False, stationary_mode=False, complete_partials=False)
    _, ids_a, cents_a, ev_a = run(BlobTracker(profile=profile()), seq, K=2)
    _, ids_b, cents_b, ev_b = run(StableBlobTracker(profile=profile(), **off), seq, K=2)
    assert ev_a == ev_b and ids_a == ids_b, "event streams differ"
    for ca, cb in zip(cents_a, cents_b):
        for k in ca:
            assert np.allclose(ca[k], cb[k]), "centroids differ"


@case
def ts_10_module_noise_function_is_restored():
    orig = BT._adapt_measurement_noise
    tk = StableBlobTracker(profile=profile())
    run(tk, fluctuating_still_blob(12))
    assert BT._adapt_measurement_noise is orig


@case
def ts_11_tentative_tracks_never_reach_the_binding_layer():
    a = ellipse(80, 200, 24, 16)
    flick = ellipse(220, 80, 16, 12)
    seq = [[a]] * 3 + [[a, flick], [a, flick], [a]] + [[a]] * 3
    tk = StableBlobTracker(profile=profile(), confirm_cycles=3)
    seen_ids = set()
    for masks in seq:
        fr, g = frame_of(masks)
        out = tk.update([det_from_mask(m) for m in masks], fr, grey=g)
        seen_ids |= {b.id for b in out}
        for e in tk.events:
            seen_ids |= {x for x in e[1:]}
    assert len(seen_ids) == 1, f"tentative id leaked: {seen_ids}"
    assert tk.reident.records == {}, "a dropped flicker was left in the re-ID buffer"


try:                                    # pytest: one test per case
    import pytest

    @pytest.mark.parametrize("fn", TESTS, ids=lambda f: f.__name__)
    def test_case(fn):
        fn()
except ImportError:
    pass


if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn(); print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL  {fn.__name__}\n      {e}")
        except Exception as e:
            import traceback; traceback.print_exc()
            failed += 1; print(f"ERR   {fn.__name__}\n      {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
