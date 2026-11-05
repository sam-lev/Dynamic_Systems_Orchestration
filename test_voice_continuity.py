#!/usr/bin/env python3
# test_voice_continuity.py -- Audible Lava / DSO v4.12
# Tests for voice_continuity.py against the real VoiceManager / BindingManager.
# No SuperCollider, camera or pygame: OSC goes to a recorder.
#   python test_voice_continuity.py

from __future__ import annotations
import sys

from lamp_profile import LampProfile
from voice_continuity import ContinuityVoiceManager, ContinuityBindingManager
from voice_manager import VoiceManager, VoiceState
from binding_manager import BindingManager


class Rec:
    def __init__(self): self.msgs = []
    def send_message(self, a, args): self.msgs.append((a, list(args)))
    def send(self, b): self.msgs.append(("<bundle>", b))
    def of(self, a): return [m for ad, m in self.msgs if ad == a]


class Blob:
    def __init__(self, id, cx=150.0, cy=200.0, **kw):
        self.id = id; self.centroid = (cx, cy)
        self.area = kw.pop("area", 5000.0); self.persistence = kw.pop("persistence", 20.0)
        self.elongation = 0.2; self.orientation = 0.0; self.mean_hue = 20.0
        self.direction = "turning"; self.birth = 120.0; self.age = kw.pop("age", 30)
        for k, v in kw.items():
            setattr(self, k, v)


class SS:
    """ScoreState look-alike: the fields _compute_targets reads."""
    def __init__(self, rank):
        self.prominence_rank = rank; self.gravitation = 0.0; self.pitch_field = []
        self.roughness_target = 0.0; self.tension = 0.0; self.surprisal = 0.0
        self.boundary = False; self.boundary_strength = 0.0


def managers(cls_v=ContinuityVoiceManager, cls_b=ContinuityBindingManager, **kw):
    vm = cls_v(profile=LampProfile(), n_slots=kw.pop("n_slots", 16), **kw)
    vm.osc = Rec()
    bm = cls_b(vm, verbose=False) if cls_b is ContinuityBindingManager else cls_b(vm)
    return vm, bm


def strikes(vm):
    return [a for a in vm.osc.of("/s_new") if a[0] == vm.strike_synthdef]


# ── 1. every tracked blob gets a voice ───────────────────────────────────────

def test_reidentified_blob_without_birth_gets_a_voice():
    """The bug path: birth -> voice; death -> zombie -> freed; the tracker
    restores the SAME id with no birth event. v4.11 leaves it silent."""
    for cls_v, cls_b, expect in ((VoiceManager, BindingManager, None),
                                 (ContinuityVoiceManager, ContinuityBindingManager, "bound")):
        vm, bm = managers(cls_v, cls_b)
        b = Blob(7)
        bm.handle_tracker_events([("birth", 7)], [b]); vm.process_frame([b])
        assert vm.slot_for_blob(7) is not None
        bm.handle_tracker_events([("death", 7)], []); vm.process_frame([])
        slot = [v for v in vm.voices_by_slot if v.state == VoiceState.ZOMBIE][0].slot_id
        vm.voices_by_slot[slot].zombie_started -= 10.0     # zombie window long past
        vm.process_frame([])
        assert vm.voices_by_slot[slot].state == VoiceState.IDLE
        b2 = Blob(7, cx=400.0)                              # re-identified, far from the old spot
        for _ in range(3):
            bm.handle_tracker_events([], [b2]); vm.process_frame([b2])
        if expect is None:
            assert vm.slot_for_blob(7) is None, "documents the v4.11 hole"
        else:
            s = vm.slot_for_blob(7)
            assert s is not None and vm.voice(s).state == VoiceState.BLOB_ONLY
            assert bm.rebinds == 1


def test_rebind_resurrects_a_matching_zombie_and_fires_no_strike():
    vm, bm = managers()
    b = Blob(3)
    bm.handle_tracker_events([("birth", 3)], [b]); vm.process_frame([b])
    first_slot = vm.slot_for_blob(3)
    n_strikes = len(strikes(vm))
    bm.handle_tracker_events([("death", 3)], []); vm.process_frame([])
    assert vm.voice(first_slot).state == VoiceState.ZOMBIE
    b2 = Blob(3, cx=155.0, cy=205.0)                        # back within the match radius
    bm.handle_tracker_events([], [b2])
    assert vm.slot_for_blob(3) == first_slot and bm.resurrections == 1, "same voice continues"
    assert len(strikes(vm)) == n_strikes, "a returning blob is not a birth"


def test_new_voice_for_unbound_blob_has_no_strike_and_respects_limits():
    vm, bm = managers(n_slots=3)
    blobs = [Blob(i, cx=50.0 + 60 * i, persistence=10.0 + i, age=30) for i in range(5)]
    young = Blob(99, age=2)
    bm.handle_tracker_events([], blobs + [young])
    bound = [b.id for b in blobs if vm.slot_for_blob(b.id) is not None]
    assert bound == [3, 4], "max 2 per frame, most persistent first"
    assert vm.slot_for_blob(99) is None, "too young: its own birth handles it"
    assert strikes(vm) == [], "no birth strike on a rebind"
    bm.handle_tracker_events([], blobs + [young])
    assert sum(1 for b in blobs if vm.slot_for_blob(b.id) is not None) == 3, "pool of 3 fills, then stops"
    bm.handle_tracker_events([], blobs + [young])           # pool full: no error, no change
    assert bm.rebinds == 3


def test_births_still_strike_and_bound_blobs_are_left_alone():
    vm, bm = managers()
    b = Blob(1)
    bm.handle_tracker_events([("birth", 1)], [b])
    assert len(strikes(vm)) == 1
    s = vm.slot_for_blob(1)
    for _ in range(5):
        bm.handle_tracker_events([], [b]); vm.process_frame([b])
    assert vm.slot_for_blob(1) == s and bm.rebinds == 0 and len(strikes(vm)) == 1


def test_merged_blob_is_not_rebound():
    vm, bm = managers()
    a, b = Blob(1, cx=100.0), Blob(2, cx=200.0)
    bm.handle_tracker_events([("birth", 1), ("birth", 2)], [a, b])
    bm.handle_tracker_events([("merge", 2, 1)], [a])         # blob 2 merged into 1, gone from the tracked set
    assert vm.slot_for_blob(2) is None and bm.rebinds == 0


# ── 2. level floor, 3. proportional beat dip ─────────────────────────────────

def _bound_voice(vm, bm, **blob_kw):
    b = Blob(5, **blob_kw)
    bm.handle_tracker_events([("birth", 5)], [b])
    return b, vm.voice(vm.slot_for_blob(5))


def test_dim_still_ornament_on_the_beat_is_not_silent():
    """Dim (persistence 4), stationary, lowest prominence, at the beat dip."""
    vm_old, bm_old = managers(VoiceManager, BindingManager)
    vm_new, bm_new = managers()
    for vm, bm in ((vm_old, bm_old), (vm_new, bm_new)):
        b, v = _bound_voice(vm, bm, persistence=4.0, score_beat_phase=1.0, score_prominence_rank=0.0,
                            d_area=-0.2)
        vm.score_state = SS({5: 0.0})
        vm._compute_targets(v, {5: b})
    old = vm_old.voice(vm_old.slot_for_blob(5)).synth
    new = vm_new.voice(vm_new.slot_for_blob(5)).synth
    assert old.t_amp < 0.02, f"v4.11 mutes it: {old.t_amp:.3f}"
    assert new.t_amp >= 0.4 * new.base_amp - 1e-9, "floor at 40 % of base"
    assert new.t_amp > 0.04, f"audible in the background: {new.t_amp:.3f}"


def test_beat_dip_is_proportional_at_every_level():
    vm, bm = managers()
    out = {}
    for pers in (4.0, 60.0):
        b, v = _bound_voice(vm, bm, persistence=pers, score_prominence_rank=1.0)
        vm.score_state = None
        b.score_beat_phase = 0.0; vm._compute_targets(v, {5: b}); hi = v.synth.t_amp
        b.score_beat_phase = 1.0; vm._compute_targets(v, {5: b}); lo = v.synth.t_amp
        out[pers] = lo / hi
        vm._release(v.slot_id)
    assert all(abs(r - 0.72) < 1e-6 for r in out.values()), out   # depth -0.28 at full phase = 28 % dip


def test_floor_not_applied_to_zombies_merges_or_keyboard_voices():
    vm, bm = managers()
    b, v = _bound_voice(vm, bm)
    vm.handle_merge(5, 999, [])                             # merge fade-out
    vm._compute_targets(v, {5: b})
    assert v.synth.t_amp == 0.0, "merge fade still reaches 0"
    vm2, bm2 = managers()
    b2, v2 = _bound_voice(vm2, bm2)
    vm2.zombify(v2.slot_id); vm2._compute_targets(v2, {})
    assert abs(v2.synth.t_amp - v2.synth.base_amp * vm2.zombie_sustain_amp_scale) < 1e-9


def test_loud_head_unchanged_except_beat_shape():
    """A prominent, bright, moving head is above the floor: same level as v4.11 off-beat."""
    vm_old, bm_old = managers(VoiceManager, BindingManager)
    vm_new, bm_new = managers()
    res = []
    for vm, bm in ((vm_old, bm_old), (vm_new, bm_new)):
        b, v = _bound_voice(vm, bm, persistence=60.0, score_beat_phase=0.0, score_prominence_rank=1.0,
                            speed=2.0)
        vm.score_state = SS({5: 1.0})
        vm._compute_targets(v, {5: b})
        res.append(v.synth.t_amp)
    assert abs(res[0] - res[1]) < 1e-9, res


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    passed = failed = 0
    for t in TESTS:
        try:
            t(); passed += 1; print(f"  ok   {t.__name__}")
        except Exception as exc:
            failed += 1; import traceback; traceback.print_exc(); print(f"  FAIL {t.__name__}: {exc!r}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
