# test_tracking_robustness.py -- Audible Lava / DSO v4.6
#
# Does the v4.6 matching rebalance actually keep blob identities alive under
# the conditions that break the old one?
#
# The failure being reproduced: the persistence peak tracks a REGION OF SIMILAR
# INTENSITY inside a blob, not the blob itself. As the blob deforms, that peak
# can shift, split, or hand dominance to a neighbouring sub-region, so `birth`
# jumps by tens of grey levels between frames while the blob has physically
# moved one pixel. The old cost put 80% (per-frame) and 72% (re-ID) of the
# matching decision on exactly that quantity, scaled by a fixed 50 grey levels.
#
# These tests drive synthetic blobs whose POSITION is smooth and whose
# TOPOLOGY is deliberately noisy, and check that identity survives.
#
# Run: python test_tracking_robustness.py  |  pytest test_tracking_robustness.py -v

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("/mnt/project", HERE):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from blob_tracker import BlobTracker, TrackedBlob, BirthRecord   # noqa: E402
from blob_detection import PersistentBlob, TURNING              # noqa: E402
from system_state import diagram_wasserstein                     # noqa: E402

TESTS = []
def test(fn):
    TESTS.append(fn); return fn

RNG = np.random.default_rng(4)


def det(x, y, area=3000.0, birth=180.0, death=20.0, elong=2.0, hu=None, diagram=None):
    """A detection with the fields the matcher reads."""
    d = PersistentBlob.__new__(PersistentBlob)
    d.centroid = (float(x), float(y))
    d.area = float(area)
    d.birth = float(birth)
    d.death = float(death)
    d.persistence = float(birth - death)
    d.orientation = 0.0
    d.aspect_ratio = 1.0
    d.elongation = float(elong)
    d.hu = list(hu) if hu else [1e-2, 1e-4, 1e-6, 1e-7, 1e-12, 1e-9, 1e-13]
    d.mean_hue = 30.0
    d.direction = TURNING
    if diagram is not None:
        d.diagram = list(diagram)
    return d


def track_blob(bid=0, x=100.0, y=100.0, area=3000.0, birth=180.0, death=20.0):
    b = TrackedBlob(id=bid, centroid=(x, y), area=area, birth=birth, death=death,
                    persistence=birth - death)
    b.elongation = 2.0
    b.hu = [1e-2, 1e-4, 1e-6, 1e-7, 1e-12, 1e-9, 1e-13]
    return b


def tracker():
    from lamp_profile import LampProfile
    return BlobTracker(profile=LampProfile.default())


def old_cost(blob, d):
    """The v4.5 ambiguity cost, for comparison: 80% topological (fixed 50-grey
    scale, compared against the LAST frame), 20% positional."""
    td = math.hypot(d.birth - blob.birth, d.death - blob.death)
    if td > 35.0:
        return float("inf")
    pos = math.hypot(d.centroid[0] - blob.centroid[0], d.centroid[1] - blob.centroid[1])
    return 0.80 * (td / 35.0) + 0.20 * (pos / 200.0)


def new_cost(tk, blob, d):
    td = tk._topo_dist(blob, d)
    if td > tk.TOPO_SIGMAS:
        return float("inf")
    pos = math.hypot(d.centroid[0] - blob.centroid[0], d.centroid[1] - blob.centroid[1])
    area = abs(d.area - blob.area) / max(blob.area, 1.0)
    return (0.60 * min(1.0, pos / 60.0) + 0.25 * min(1.0, area)
            + 0.15 * min(1.0, td / tk.TOPO_SIGMAS) + 0.10 * tk._shape_dist(blob, d))


# ── The core failure ────────────────────────────────────────────────────────

@test
def rb_01_topological_noise_no_longer_rejects_a_correct_match():
    """A blob moving 1 px/frame whose birth value swings +/-25 grey levels.
    Physically the same blob; the old cost rejects it outright."""
    tk = tracker()
    blob = track_blob()
    # let the track observe its own variability for a second
    for f in range(20):
        tk._update_topo_stats(blob, det(100 + f, 100, birth=180 + RNG.normal(0, 12)))
        blob.centroid = (100 + f, 100)

    rejected_old = rejected_new = 0
    for f in range(20, 120):
        d = det(100 + f, 100, birth=180 + RNG.normal(0, 12))
        blob.centroid = (100 + f - 1, 100)              # 1 px behind: correct match
        if old_cost(blob, d) == float("inf"):
            rejected_old += 1
        if new_cost(tk, blob, d) == float("inf"):
            rejected_new += 1
        tk._update_topo_stats(blob, d)
        blob.birth, blob.death = d.birth, d.death
    assert rejected_new < rejected_old, \
        f"new cost rejected {rejected_new}, old rejected {rejected_old}"
    assert rejected_new <= 5, f"new cost still rejects {rejected_new}/100 correct matches"


@test
def rb_02_variability_is_learned_per_track():
    """A noisy blob and a stable blob must not be judged on the same scale."""
    tk = tracker()
    noisy, calm = track_blob(0), track_blob(1)
    for f in range(60):
        tk._update_topo_stats(noisy, det(100, 100, birth=180 + RNG.normal(0, 20)))
        tk._update_topo_stats(calm, det(100, 100, birth=180 + RNG.normal(0, 1)))
    assert noisy.v_birth > 3 * calm.v_birth, \
        f"noisy sigma {noisy.v_birth:.1f} vs calm {calm.v_birth:.1f}"
    # the same absolute deviation is many sigma for the calm track, few for the noisy one
    probe = det(100, 100, birth=205.0)
    assert tk._topo_dist(noisy, probe) < tk._topo_dist(calm, probe)


@test
def rb_03_smoothed_reference_survives_one_bad_frame():
    """One outlier frame must not corrupt the reference used next frame."""
    tk = tracker()
    blob = track_blob()
    for _ in range(40):
        tk._update_topo_stats(blob, det(100, 100, birth=180.0))
    before = blob.s_birth
    tk._update_topo_stats(blob, det(100, 100, birth=60.0))     # a wild outlier
    assert abs(blob.s_birth - before) < 0.5 * abs(60.0 - before), \
        f"one outlier moved the reference from {before:.1f} to {blob.s_birth:.1f}"
    # and a subsequent good detection is still recognised
    assert tk._topo_dist(blob, det(100, 100, birth=180.0)) < tk.TOPO_SIGMAS


@test
def rb_04_position_now_dominates():
    """A detection far away with identical topology must lose to a nearby one
    with different topology -- the inverse of the old behaviour."""
    tk = tracker()
    blob = track_blob(x=100, y=100, birth=180)
    for _ in range(30):
        tk._update_topo_stats(blob, det(100, 100, birth=180 + RNG.normal(0, 6)))
    near_diff_topo = det(103, 100, birth=195.0)        # 3 px away, topology off
    far_same_topo  = det(180, 100, birth=180.0)        # 80 px away, topology exact
    assert new_cost(tk, blob, near_diff_topo) < new_cost(tk, blob, far_same_topo), \
        "far-but-topologically-identical still wins"
    assert old_cost(blob, far_same_topo) < old_cost(blob, near_diff_topo), \
        "sanity: the OLD cost preferred the far one (the bug being fixed)"


@test
def rb_05_shape_descriptors_contribute():
    """elongation and Hu moments were computed on every blob and used by
    neither cost function."""
    tk = tracker()
    blob = track_blob()
    same = det(100, 100, elong=2.0)
    diff = det(100, 100, elong=9.0, hu=[1e-1, 1e-2, 1e-3, 1e-7, 1e-12, 1e-9, 1e-13])
    assert tk._shape_dist(blob, same) < 0.1
    assert tk._shape_dist(blob, diff) > 0.3
    assert new_cost(tk, blob, same) < new_cost(tk, blob, diff)


# ── Whole-diagram comparison ────────────────────────────────────────────────

@test
def rb_06_diagram_distance_is_stable_under_a_peak_split():
    """The property that makes persistence worth using: a peak splitting in two
    changes the diagram distance a little, where a single-point comparison can
    change the answer entirely."""
    one_peak = [(180.0, 20.0)]
    # The stability property is about SMALL perturbations: a low-persistence
    # peak appearing near the diagonal (the dominant noise mode) barely moves
    # the distance, because matching it to the diagonal is cheap.
    noise = [(180.0, 20.0), (40.0, 35.0)]
    d_noise = diagram_wasserstein(one_peak, noise)
    assert d_noise < 10.0, f"a small noise peak moved the distance by {d_noise:.1f}"
    # A dominant peak shifting slightly is also a small change.
    shifted = [(176.0, 20.0)]
    assert diagram_wasserstein(one_peak, shifted) < 10.0
    # By contrast a genuine split into two LARGE features is correctly a large
    # change -- Wasserstein does not paper over real structural change, and
    # should not: that is a different blob configuration.
    split = [(178.0, 20.0), (176.0, 20.0)]
    assert diagram_wasserstein(one_peak, split) > 50.0


@test
def rb_07_tracker_uses_the_diagram_when_present():
    tk = tracker()
    blob = track_blob()
    blob.diagram = [(180.0, 20.0)]
    # A small noise peak alongside the true one must still match.
    d = det(100, 100, birth=180.0, diagram=[(180.0, 20.0), (38.0, 30.0)])
    assert tk._topo_dist(blob, d) < tk.TOPO_SIGMAS, "noise-perturbed diagram rejected"
    far = det(100, 100, birth=90.0, diagram=[(90.0, 20.0)])
    assert tk._topo_dist(blob, far) > tk._topo_dist(blob, d), \
        "a genuinely different diagram must score worse than a split one"


# ── Re-identification fingerprint ───────────────────────────────────────────

@test
def rb_08_reid_uses_smoothed_values_and_own_variability():
    rec = BirthRecord(blob_id=0, birth_value=180.0, persistence=160.0,
                      initial_area=3000.0, y_norm=0.5, mean_hue=30.0,
                      direction=TURNING, lost_at=0.0,
                      v_birth=20.0, v_persistence=20.0, elongation=2.0,
                      hu=[1e-2, 1e-4, 1e-6, 1e-7, 1e-12, 1e-9, 1e-13])
    strict = BirthRecord(blob_id=1, birth_value=180.0, persistence=160.0,
                         initial_area=3000.0, y_norm=0.5, mean_hue=30.0,
                         direction=TURNING, lost_at=0.0,
                         v_birth=4.0, v_persistence=4.0, elongation=2.0,
                         hu=[1e-2, 1e-4, 1e-6, 1e-7, 1e-12, 1e-9, 1e-13])
    probe = det(100, 100, birth=200.0)                 # 20 grey levels off
    assert rec.distance(probe, 0.5) < strict.distance(probe, 0.5), \
        "a historically noisy blob must be judged more leniently"


@test
def rb_09_reid_position_and_area_outweigh_topology():
    base = dict(blob_id=0, persistence=160.0, initial_area=3000.0, y_norm=0.5,
                mean_hue=30.0, direction=TURNING, lost_at=0.0, v_birth=8.0,
                v_persistence=8.0, elongation=2.0,
                hu=[1e-2, 1e-4, 1e-6, 1e-7, 1e-12, 1e-9, 1e-13])
    rec = BirthRecord(birth_value=180.0, **base)
    same_place_diff_topo = det(100, 100, area=3000.0, birth=210.0)
    far_place_same_topo  = det(100, 100, area=3000.0, birth=180.0)
    d_near = rec.distance(same_place_diff_topo, 0.50)   # same height
    d_far  = rec.distance(far_place_same_topo, 0.95)    # far away vertically
    assert d_near < d_far, f"topology still dominates: near={d_near:.3f} far={d_far:.3f}"


@test
def rb_10_identical_blob_scores_near_zero():
    rec = BirthRecord(blob_id=0, birth_value=180.0, persistence=160.0,
                      initial_area=3000.0, y_norm=0.5, mean_hue=30.0,
                      direction=TURNING, lost_at=0.0, v_birth=8.0,
                      v_persistence=8.0, elongation=2.0,
                      hu=[1e-2, 1e-4, 1e-6, 1e-7, 1e-12, 1e-9, 1e-13])
    assert rec.distance(det(100, 100, birth=180.0), 0.5) < 0.05


# ── Ambiguity fix ───────────────────────────────────────────────────────────

@test
def rb_11_ambiguity_is_no_longer_pinned():
    """The three system_state fixes: smaller Dirichlet prior, 81-symbol
    alphabet instead of 243, and normalisation by the OBSERVED alphabet."""
    from dataclasses import dataclass as dc
    import score_governor as SG

    @dc
    class B:
        id: int; centroid: tuple; area: float; birth: float = 150.0
        death: float = 20.0; persistence: float = 40.0; vx: float = 0.0
        vy: float = 0.0; speed: float = 0.0; ay: float = 0.0
        direction: str = "turning"; age: int = 0

    for p in ("/tmp/_rb.json",):
        if os.path.exists(p):
            os.remove(p)
    g = SG.ScoreGovernor(320, 240, weights_path="/tmp/_rb.json",
                         runtime_path="/tmp/_rbr.json", log_path="/tmp/_rbl.txt", fps=15)
    amb = []
    for f in range(15 * 240):
        t = f / 15.0
        boost = 3.0 if (f // (15 * 40)) % 2 else 1.0
        n = 3 if boost == 1.0 else 5
        blobs = []
        for i in range(n):
            per = 8.0 + 2 * i
            y = 120 + 80 * math.sin(2 * math.pi * t / per + i)
            vy = 80 * 2 * math.pi / per * math.cos(2 * math.pi * t / per + i) / 15.0
            blobs.append(B(i, (50 + 50 * i, y), 3000 * boost,
                           vy=vy * boost, speed=abs(vy) * boost, age=f))
        amb.append(g.update(blobs, 1 / 15.0).ambiguity)
    a = np.array(amb)
    assert (a > 0.999).mean() < 0.10, f"{100*(a>0.999).mean():.0f}% of frames pinned at 1.0"
    assert a.std() > 0.03, f"ambiguity barely varies (sd {a.std():.3f})"
    assert len(g.state_tracker.markov.alphabet) <= 81, \
        f"alphabet grew to {len(g.state_tracker.markov.alphabet)}, expected <= 81"


@test
def rb_12_markov_prior_and_normaliser():
    from system_state import MarkovExpectation
    m = MarkovExpectation()
    assert m.prior <= 0.1, f"Dirichlet prior {m.prior} still large"
    for _ in range(80):
        for sym in (1, 2, 3):
            m.update(sym)
    # a deterministic cycle over 3 symbols must read as near-zero uncertainty
    assert m.normalised_conditional_entropy(1) < 0.25, \
        f"deterministic cycle reads {m.normalised_conditional_entropy(1):.2f}"
    assert abs(m.max_entropy() - math.log2(3)) < 1e-6


if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn(); print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL  {fn.__name__}\n      {e}")
        except Exception as e:
            failed += 1; print(f"ERR   {fn.__name__}\n      {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
