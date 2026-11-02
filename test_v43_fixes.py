# test_v43_fixes.py -- Audible Lava / DSO v4.3
# Behavioural tests for three fixes:
#   BASS-xx  bass-stutter no longer sticks on (hysteresis, time cap, debounce,
#            voice cap, retrigger cooldown)
#   BG-xx    warm-lamp background capture (percentile over a long window)
#   KUR-xx   Kuramoto order parameter from true per-blob oscillation phase
#
# Run: python test_v43_fixes.py   |   pytest test_v43_fixes.py -v

import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("/mnt/project", HERE):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from system_state import SystemStateTracker                       # noqa: E402
from background_reference import BackgroundReference              # noqa: E402
from voice_manager import VoiceManager, VoiceState                # noqa: E402
from lamp_profile import LampProfile                              # noqa: E402

TESTS = []
def test(fn):
    TESTS.append(fn); return fn

RNG = np.random.default_rng(5)


@dataclass
class B:
    id: int
    centroid: tuple
    area: float = 3000.0
    birth: float = 150.0
    death: float = 20.0
    persistence: float = 40.0
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0
    ay: float = 0.0
    direction: str = "turning"
    age: int = 100
    d_area: float = 0.0
    d_persistence: float = 0.0


class FakeOSC:
    def __init__(self): self.sent = []
    def send_message(self, addr, args): self.sent.append((addr, args))
    def send(self, *a, **k): pass


def make_vm(n_slots=6):
    vm = VoiceManager.__new__(VoiceManager)          # bypass OSC construction
    VoiceManager.__init__(vm, n_slots=n_slots, profile=LampProfile.default())
    vm.osc = FakeOSC()
    return vm


def bind(vm, slot, blob):
    v = vm.voice(slot)
    v.state = VoiceState.BLOB_ONLY
    v.blob_id = blob.id
    vm.voices_by_blob[blob.id] = slot
    return v


def low_falling_growing(bid, prof, *, d_area=0.10, vy=0.5):
    """A blob that satisfies the bass ENTRY condition."""
    y = prof.roi_h * 0.9          # y_norm ~0.1 -> low in the lamp
    return B(bid, (prof.roi_w * 0.5, y), vy=vy, speed=abs(vy), d_area=d_area, age=100)


# ── BASS ────────────────────────────────────────────────────────────────────

@test
def bass_01_steady_blob_exits():
    """The original bug: a blob with steady area (d_area ~ 0) satisfied the old
    exit test `d_area >= -0.02` forever and the voice droned."""
    vm = make_vm(); p = vm.profile
    b = low_falling_growing(0, p)
    v = bind(vm, 0, b)
    vm._update_bass_stutter_modes({0: b})
    assert v.synth_kind == "bass", "entry condition should fire"

    b.d_area = 0.0                                   # steady area, still low+falling
    for _ in range(vm.bass_exit_frames):
        vm._update_bass_stutter_modes({0: b})
    assert v.synth_kind == "mpe", "steady-area blob must leave bass mode"


@test
def bass_02_exit_is_debounced():
    vm = make_vm(); p = vm.profile
    b = low_falling_growing(0, p)
    v = bind(vm, 0, b)
    vm._update_bass_stutter_modes({0: b})
    assert v.synth_kind == "bass"

    b.d_area = 0.0
    for i in range(vm.bass_exit_frames - 1):         # one short of the streak
        vm._update_bass_stutter_modes({0: b})
        assert v.synth_kind == "bass", f"exited after {i+1} frames, want {vm.bass_exit_frames}"
    b.d_area = 0.10                                  # recovers before the streak completes
    vm._update_bass_stutter_modes({0: b})
    assert v.synth_kind == "bass" and v.bass_exit_streak == 0, "streak must reset on recovery"


@test
def bass_03_hard_time_cap():
    """Whatever the conditions do, no episode may exceed bass_max_secs."""
    vm = make_vm(); p = vm.profile
    vm.bass_max_secs = 0.25
    b = low_falling_growing(0, p)
    v = bind(vm, 0, b)
    vm._update_bass_stutter_modes({0: b})
    assert v.synth_kind == "bass"
    for _ in range(5):                               # conditions stay satisfied
        vm._update_bass_stutter_modes({0: b})
    assert v.synth_kind == "bass", "must not exit while under the cap"
    time.sleep(0.3)
    vm._update_bass_stutter_modes({0: b})
    assert v.synth_kind == "mpe", "time cap must force an exit"


@test
def bass_04_voice_cap():
    vm = make_vm(n_slots=6); p = vm.profile
    blobs = {}
    for i in range(5):
        b = low_falling_growing(i, p)
        blobs[i] = b
        bind(vm, i, b)
    vm._update_bass_stutter_modes(blobs)
    assert vm.bass_voice_count() == vm.max_bass_voices, \
        f"expected {vm.max_bass_voices} bass voices, got {vm.bass_voice_count()}"


@test
def bass_05_retrigger_cooldown():
    """A blob oscillating around the threshold must not respawn nodes every
    few frames."""
    vm = make_vm(); p = vm.profile
    vm.bass_retrigger_secs = 5.0
    b = low_falling_growing(0, p)
    v = bind(vm, 0, b)
    vm._update_bass_stutter_modes({0: b})
    assert v.synth_kind == "bass"
    b.d_area = 0.0
    for _ in range(vm.bass_exit_frames):
        vm._update_bass_stutter_modes({0: b})
    assert v.synth_kind == "mpe"
    b.d_area = 0.10                                  # entry condition again
    for _ in range(10):
        vm._update_bass_stutter_modes({0: b})
    assert v.synth_kind == "mpe", "cooldown must block immediate re-entry"


@test
def bass_06_min_age_blocks_churn_blobs():
    """With ~5 s blob lifetimes, a 4-frame age gate let every new blob enter
    bass. The gate is now ~1 s of tracking."""
    vm = make_vm(); p = vm.profile
    b = low_falling_growing(0, p); b.age = 5         # young
    v = bind(vm, 0, b)
    vm._update_bass_stutter_modes({0: b})
    assert v.synth_kind == "mpe", "young blob must not enter bass"
    b.age = vm.bass_min_age
    vm._update_bass_stutter_modes({0: b})
    assert v.synth_kind == "bass"


@test
def bass_07_lost_blob_exits_immediately():
    vm = make_vm(); p = vm.profile
    b = low_falling_growing(0, p)
    v = bind(vm, 0, b)
    vm._update_bass_stutter_modes({0: b})
    assert v.synth_kind == "bass"
    vm._update_bass_stutter_modes({})                # blob vanished
    assert v.synth_kind == "mpe"


@test
def bass_08_hold_region_is_narrower_than_entry():
    """Structural guarantee: every OFF threshold must be at most as permissive
    as its ON counterpart plus a small margin -- never unbounded."""
    vm = make_vm()
    assert vm.bass_y_off >= vm.bass_y_on, "y hold must be at least as low as entry"
    assert vm.bass_y_off - vm.bass_y_on <= 0.10, "y hold margin too wide"
    assert 0.0 < vm.bass_vy_off <= vm.bass_vy_on, "vy hold must stay positive (descending)"
    assert 0.0 < vm.bass_darea_off <= vm.bass_darea_on, \
        "d_area hold must stay positive (still growing) -- a non-positive floor is the drone bug"


# ── BACKGROUND ──────────────────────────────────────────────────────────────

def warm_frame(n, H=120, W=80, rng=RNG):
    bg = np.full((H, W), 45, np.float32)
    bg[100:, :] = 90                                  # bright cold reservoir
    f = bg.copy()
    for (x0, per, ph) in [(15, 130, 0), (40, 95, 30), (62, 160, 70)]:
        y = int(10 + (H - 40) * (0.5 + 0.5 * math.sin(2 * math.pi * (n + ph) / per)))
        f[y:y + 22, x0 - 8:x0 + 8] = 210
    return np.clip(f + rng.normal(0, 2, (H, W)), 0, 255).astype(np.uint8), bg.astype(np.uint8)


@test
def bg_01_warm_capture_recovers_background():
    """Wax is present in EVERY frame; the reference must still be background."""
    br = BackgroundReference(mode="warm", capture_secs=2.0, warm_samples=80, settle_frames=3)
    n = 0
    while not br.ready and n < 30000:
        br.apply(warm_frame(n)[0]); n += 1
        time.sleep(0.0005)
    assert br.ready, "warm capture never completed"
    _, truth = warm_frame(0)
    err = np.abs(br.reference.astype(float) - truth.astype(float))
    assert err.mean() < 6.0, f"warm reference mean error {err.mean():.1f}"
    assert (err > 25).sum() == 0, f"{(err>25).sum()} pixels badly wrong -- wax baked in"


@test
def bg_02_warm_uses_percentile_not_median():
    br = BackgroundReference(mode="warm", method="median")     # method overridden
    assert br.method == "percentile", "warm mode must force the percentile reducer"
    assert br.percentile < 50.0, "warm percentile must sit below the median"


@test
def bg_03_capture_finishes_in_about_capture_secs():
    """Wall-clock pacing: the window must bound the capture regardless of frame
    rate (the earlier fps-estimating pacer could over-decimate and stall)."""
    br = BackgroundReference(mode="warm", capture_secs=1.0, warm_samples=500, settle_frames=2)
    t0 = time.time(); n = 0
    while not br.ready and n < 100000:
        br.apply(warm_frame(n)[0]); n += 1
    dt = time.time() - t0
    assert 0.7 <= dt <= 2.5, f"capture took {dt:.2f}s, want ~1s"
    assert br.ready


@test
def bg_04_cold_path_unchanged():
    _, truth = warm_frame(0)
    br = BackgroundReference(mode="cold", n_frames=20, settle_frames=3)
    for _ in range(40):
        br.apply(truth)
    assert br.ready and br.method == "median"
    assert np.abs(br.reference.astype(float) - truth.astype(float)).max() < 25


@test
def bg_05_reset_can_switch_mode():
    br = BackgroundReference(mode="cold", n_frames=20, settle_frames=1)
    for _ in range(30):
        br.apply(warm_frame(0)[0])
    assert br.ready
    br.reset(mode="warm")
    assert not br.ready and br.mode == "warm" and br.method == "percentile"


@test
def bg_06_refine_excludes_foreground():
    br = BackgroundReference(mode="warm", capture_secs=1.0, warm_samples=60, settle_frames=2)
    n = 0
    while not br.ready and n < 50000:
        br.apply(warm_frame(n)[0]); n += 1
    before = br.reference.copy()
    br.refine()
    assert br._exclude_fg, "refine must re-enter an accumulation pass"
    n2 = 0
    while br._exclude_fg and n2 < 50000:
        br.apply(warm_frame(n + n2)[0]); n2 += 1
    assert br.ready and not br._exclude_fg, "refine must terminate"
    _, truth = warm_frame(0)
    err = np.abs(br.reference.astype(float) - truth.astype(float))
    assert (err > 25).sum() == 0, "refine made the reference worse"
    assert br.reference.shape == before.shape


@test
def bg_07_occupancy_diagnostic():
    br = BackgroundReference(mode="warm", capture_secs=1.0, warm_samples=60, settle_frames=2)
    n = 0
    while not br.ready and n < 50000:
        br.apply(warm_frame(n)[0]); n += 1
    occ = br.occupancy(warm_frame(500)[0])
    # three 16x22 blobs in a 120x80 frame ~= 0.11 of the area
    assert 0.03 < occ < 0.45, f"occupancy {occ:.3f} implausible for 3 blobs"
    assert "ready" in br.status()


# ── KURAMOTO ────────────────────────────────────────────────────────────────

def run_phases(phases, per=8.0, n=400, fps=30.0):
    st = SystemStateTracker(320, 240)
    s = None
    for f in range(n):
        t = f / fps
        blobs = []
        for i, ph in enumerate(phases):
            y = 120 + 80 * math.sin(2 * math.pi * t / per + ph)
            vy = 80 * (2 * math.pi / per) * math.cos(2 * math.pi * t / per + ph) / fps
            blobs.append(B(i, (60 + 80 * i, y), vy=vy, speed=abs(vy), age=f))
        s = st.update(blobs, 1 / fps)
    return s.extra["rel_kuramoto"], int(s.extra["rel_kuramoto_n"])


@test
def kur_01_in_phase_is_one():
    R, n = run_phases([0.0, 0.0, 0.0])
    assert R > 0.9 and n == 3, f"R={R:.3f} n={n}"


@test
def kur_02_anti_phase_is_zero():
    """The decisive test the old atan2 proxy could not pass: two blobs at
    opposite points of their arcs must differ by pi and cancel."""
    R, n = run_phases([0.0, math.pi])
    assert R < 0.2 and n == 2, f"anti-phase R={R:.3f} (proxy returned ~1.0)"


@test
def kur_03_even_spread_is_low():
    R, _ = run_phases([0.0, 2 * math.pi / 3, 4 * math.pi / 3])
    assert R < 0.3, f"evenly spread phases should cancel, got R={R:.3f}"


@test
def kur_04_partial_spread_is_intermediate():
    R_tight, _ = run_phases([0.0, 0.3, 0.6])
    R_wide, _ = run_phases([0.0, math.pi / 2])
    assert R_tight > 0.85, f"tight cluster R={R_tight:.3f}"
    assert 0.3 < R_wide < 0.9, f"quarter-apart R={R_wide:.3f} should be intermediate"
    assert R_tight > R_wide, "tighter phases must give a higher order parameter"


@test
def kur_05_single_blob_is_zero_not_one():
    """Defaulting a lone oscillator to 1.0 is what pinned elastic at ~1.0."""
    R, n = run_phases([0.0])
    assert R == 0.0 and n == 1, f"single blob R={R:.3f}"


@test
def kur_06_non_oscillating_blobs_do_not_count():
    st = SystemStateTracker(320, 240)
    s = None
    for f in range(300):
        y = 20 + 0.4 * f                       # steady rise, no cycle
        s = st.update([B(0, (60, y), vy=0.4, speed=0.4, age=f),
                       B(1, (140, y + 30), vy=0.4, speed=0.4, age=f)], 1 / 30)
    assert s.extra["rel_kuramoto_n"] == 0, "unresolved blobs must not contribute"
    assert s.extra["rel_kuramoto"] == 0.0


@test
def kur_07_elastic_strength_spans_its_range():
    """elastic = 0.15 + 0.85*R must actually move; the proxy pinned it at ~1.0
    for every frame of every logged run."""
    els = []
    for phases in ([0.0, 0.0, 0.0], [0.0, math.pi / 2], [0.0, math.pi],
                   [0.0, 2 * math.pi / 3, 4 * math.pi / 3]):
        R, _ = run_phases(phases)
        els.append(0.15 + 0.85 * R)
    assert max(els) > 0.85 and min(els) < 0.40, f"elastic range {min(els):.2f}-{max(els):.2f}"


@test
def kur_08_phase_is_frame_rate_invariant():
    """vy is px/frame, omega is rad/s: the phase must not depend on fps.
    Deriving dy/dt numerically is what makes this hold."""
    R30, _ = run_phases([0.0, math.pi / 2], fps=30.0, n=400)
    R15, _ = run_phases([0.0, math.pi / 2], fps=15.0, n=200)
    assert abs(R30 - R15) < 0.2, f"R differs with frame rate: {R30:.3f} vs {R15:.3f}"


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
