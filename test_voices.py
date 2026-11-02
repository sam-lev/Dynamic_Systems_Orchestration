#!/usr/bin/env python3
# test_voices.py -- v4.0 voice-slot system smoke test
#
# Exercises the VoiceManager + BindingManager + KeyboardController API
# without requiring a working camera, MOG2, GUDHI, or running SC server.
#
# What it does:
#   1. Construct a VoiceManager with 16 slots and a default LampProfile
#   2. Spawn 5 fake "blobs" via the BindingManager.on_blob_birth path
#   3. Verify slot allocations and routings
#   4. Send a few synthetic ControllerEvents (SELECT_SLOT, NOTE_ON, SET_PARAM,
#      CREATE_EMPTY, TIE_BLOB, DETACH, CLAIM_HYBRID)
#   5. Step process_frame a few times to exercise routing + smoothing
#   6. Kill blobs, watch them transition to ZOMBIE, then resurrect or expire
#
# Designed to PASS WITHOUT SC running (OSC sends are best-effort UDP — the
# packets just go nowhere if the port isn't open). pygame is also not
# required; the controller is exercised by constructing ControllerEvent
# instances directly.

from __future__ import annotations
import sys
import time
import math
from dataclasses import dataclass, field

# Try to import — print which import failed loudly if so
try:
    from lamp_profile   import LampProfile
    from voice_manager  import VoiceManager, VoiceState
    from binding_manager import BindingManager
    from controller     import (
        ControllerEvent,
        SELECT_SLOT, NOTE_ON, NOTE_OFF, SET_PARAM,
        CREATE_EMPTY, TIE_BLOB, DETACH, CLAIM_HYBRID,
    )
    from routing import Source, Dest, Curve
except Exception as exc:
    print(f"IMPORT FAILED: {exc}")
    sys.exit(2)


# ── Synthetic blob ────────────────────────────────────────────────────────────
@dataclass
class FakeBlob:
    """Minimal stand-in for blob_tracker.TrackedBlob, sufficient for what
    the routing layer reads (centroid, persistence, area, mean_hue,
    elongation, orientation, direction, age, id)."""
    id:           int
    centroid:     tuple
    area:         float = 5000.0
    persistence:  float = 30.0
    mean_hue:     float = 12.0
    elongation:   float = 0.4
    orientation:  float = 0.0
    direction:    str   = "rising"
    age:          int   = 0
    birth:        float = 100.0
    death:        float = 80.0


# ── Default lamp profile (matches calibration_gui defaults) ───────────────────
def _default_profile() -> LampProfile:
    p = LampProfile.default()
    # Tighten ROI to a sensible test rectangle if defaults give 0
    if p.lamp_x2 - p.lamp_x1 < 50 or p.lamp_y2 - p.lamp_y1 < 50:
        p.lamp_x1, p.lamp_y1 = 200, 50
        p.lamp_x2, p.lamp_y2 = 800, 900
    p.frame_w = max(p.frame_w, 1280)
    p.frame_h = max(p.frame_h, 960)
    p.observed_hue_mean = 0.0
    p.expected_persistence_mean = 15.0
    p.blob_area_max = 80000.0
    return p


# ── Tiny test runner ──────────────────────────────────────────────────────────
PASS = 0
FAIL = 0


def assert_eq(label, expected, got):
    global PASS, FAIL
    ok = expected == got
    if ok:
        PASS += 1
        print(f"  ✓ {label}")
    else:
        FAIL += 1
        print(f"  ✗ {label}: expected {expected!r}, got {got!r}")


def assert_ne(label, not_expected, got):
    global PASS, FAIL
    ok = not_expected != got
    if ok:
        PASS += 1
        print(f"  ✓ {label}")
    else:
        FAIL += 1
        print(f"  ✗ {label}: expected NOT {not_expected!r}, got {got!r}")


def assert_true(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {label}")
    else:
        FAIL += 1
        print(f"  ✗ {label}")


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_basic_spawn():
    print("\n[1] Basic spawn from blob births")
    p     = _default_profile()
    vm    = VoiceManager(profile=p, n_slots=16)
    bm    = BindingManager(vm)

    blobs = [
        FakeBlob(id=1, centroid=(400, 400), persistence=30, direction="rising"),
        FakeBlob(id=2, centroid=(500, 200), persistence=22, direction="falling"),
        FakeBlob(id=3, centroid=(600, 500), persistence=25, direction="turning"),
    ]
    for b in blobs:
        bm.on_blob_birth(b)

    assert_eq("3 BLOB_ONLY voices",      3, vm.hud_summary()["BLOB_ONLY"])
    assert_eq("0 KEYBOARD_ONLY voices",  0, vm.hud_summary()["KEYBOARD_ONLY"])
    assert_eq("13 IDLE",                13, vm.hud_summary()["IDLE"])
    assert_eq("blob 1 in slot 0",        0, vm.slot_for_blob(1))
    assert_eq("blob 2 in slot 1",        1, vm.slot_for_blob(2))

    from routing import DEFAULT_BLOB_ROUTING
    v = vm.voice(0)
    assert_true("blob_only voice has DEFAULT_BLOB_ROUTING",
                len(v.routing) == len(DEFAULT_BLOB_ROUTING))


def test_keyboard_create_and_tie():
    print("\n[2] CREATE_EMPTY + TIE_BLOB workflow")
    p     = _default_profile()
    vm    = VoiceManager(profile=p, n_slots=16)
    bm    = BindingManager(vm)

    s = bm.request_create_empty()
    assert_eq("created KEYBOARD_ONLY in slot 0", 0, s)
    assert_true("slot 0 is KEYBOARD_ONLY", vm.is_keyboard_only(0))

    # Player presses 'T' on slot 0 with no live blobs -> queues for next birth
    tied_now = bm.request_tie(0, tracked_blobs=[])
    assert_eq("not tied immediately (no blobs)", False, tied_now)
    assert_true("slot 0 in pending list", 0 in bm.pending_keyboard_voices)

    # Now a blob is born — should bind to slot 0 first (skipping fresh spawn)
    b = FakeBlob(id=10, centroid=(400, 500), persistence=30, direction="rising")
    bm.on_blob_birth(b)
    assert_true("slot 0 became HYBRID", vm.is_hybrid(0))
    assert_eq("blob 10 in slot 0", 0, vm.slot_for_blob(10))
    assert_eq("pending list emptied", 0, len(bm.pending_keyboard_voices))


def test_immediate_tie_to_unbound():
    print("\n[3] CREATE_EMPTY + TIE_BLOB with live unbound blobs")
    p  = _default_profile()
    vm = VoiceManager(profile=p, n_slots=16)
    bm = BindingManager(vm)

    # Spawn 2 BLOB_ONLY voices
    a = FakeBlob(id=1, centroid=(400, 400), persistence=20, direction="rising")
    bm.on_blob_birth(a)

    # Now create empty + tie. There is one BOUND blob (#1) and we'll add one
    # UNBOUND blob (#2) to the tracked list.
    bm.request_create_empty()
    b = FakeBlob(id=2, centroid=(500, 500), persistence=30, direction="rising")
    tied_now = bm.request_tie(1, tracked_blobs=[a, b])
    assert_eq("tied immediately", True, tied_now)
    assert_true("slot 1 became HYBRID", vm.is_hybrid(1))


def test_zombie_pool():
    print("\n[4] BLOB_ONLY → ZOMBIE → resurrect / expire")
    p  = _default_profile()
    vm = VoiceManager(profile=p, n_slots=16,
                      zombie_window=0.5,             # short window for test
                      zombie_match_radius_frac=0.5,
                      zombie_min_match_radius=200.0)
    bm = BindingManager(vm)

    a  = FakeBlob(id=1, centroid=(400, 400), persistence=20, direction="rising")
    bm.on_blob_birth(a)
    assert_true("slot 0 BLOB_ONLY", vm.is_blob_only(0))

    # Blob dies
    bm.on_blob_death(1)
    assert_true("slot 0 ZOMBIE", vm.is_zombie(0))

    # New blob nearby: should resurrect slot 0
    a2 = FakeBlob(id=2, centroid=(420, 410), persistence=25, direction="rising")
    bm.on_blob_birth(a2)
    assert_true("slot 0 resurrected to BLOB_ONLY", vm.is_blob_only(0))
    assert_eq("blob 2 in slot 0", 0, vm.slot_for_blob(2))

    # Kill blob 2, then wait past zombie window -> slot back to IDLE
    bm.on_blob_death(2)
    assert_true("slot 0 ZOMBIE again", vm.is_zombie(0))
    time.sleep(0.6)
    vm.process_frame([])  # triggers _expire_zombies
    assert_true("slot 0 IDLE after window", vm.is_idle(0))


def test_claim_hybrid_and_set_param():
    print("\n[5] CLAIM_HYBRID + SET_PARAM relative edits")
    p  = _default_profile()
    vm = VoiceManager(profile=p, n_slots=16)
    bm = BindingManager(vm)

    b  = FakeBlob(id=1, centroid=(400, 400), persistence=20, direction="rising")
    bm.on_blob_birth(b)
    assert_true("slot 0 BLOB_ONLY before claim", vm.is_blob_only(0))

    bm.request_claim_hybrid(0)
    assert_true("slot 0 HYBRID after claim", vm.is_hybrid(0))

    # Capture pre-edit cutoff and apply a relative bump
    pre_cutoff = vm.voice(0).synth.base_cutoff
    vm.set_param(0, "cutoff", 500.0, relative=True)
    post_cutoff = vm.voice(0).synth.base_cutoff
    assert_eq("cutoff += 500", pre_cutoff + 500.0, post_cutoff)

    # Reset to default
    vm.set_param(0, "cutoff", None, reset=True)
    assert_true("cutoff reset to default",
                abs(vm.voice(0).synth.base_cutoff - 1500.0) < 1e-6)


def test_routing_application():
    print("\n[6] Routing matrix produces non-trivial target values")
    p  = _default_profile()
    vm = VoiceManager(profile=p, n_slots=16)
    bm = BindingManager(vm)

    # Blob at top of lamp -> y_norm should produce high pitch contribution
    b_top    = FakeBlob(id=1, centroid=(400, 100),  persistence=30, direction="rising")
    # Blob at bottom -> y_norm should produce low pitch contribution
    b_bottom = FakeBlob(id=2, centroid=(400, 850),  persistence=30, direction="rising")
    bm.on_blob_birth(b_top)
    bm.on_blob_birth(b_bottom)

    vm.process_frame([b_top, b_bottom])
    f_top    = vm.voice(0).synth.t_freq
    f_bottom = vm.voice(1).synth.t_freq
    print(f"     top blob  -> t_freq = {f_top:.1f} Hz")
    print(f"     bot blob  -> t_freq = {f_bottom:.1f} Hz")
    assert_true("top blob produces higher freq than bottom",
                f_top > f_bottom)


def test_dispatch_controller_events():
    print("\n[7] Controller-event dispatch through VoiceManager")
    p  = _default_profile()
    vm = VoiceManager(profile=p, n_slots=16)
    bm = BindingManager(vm)

    bm.request_create_empty()      # slot 0 KEYBOARD_ONLY

    # Select the slot
    vm.dispatch(ControllerEvent(SELECT_SLOT, slot=0, value=0))
    assert_eq("selected slot=0", 0, vm.selected_slot)

    # Play a note
    vm.dispatch(ControllerEvent(NOTE_ON, slot=0, note=72, velocity=110))
    assert_eq("base_note=72", 72.0, vm.voice(0).synth.base_note)
    assert_true("base_amp non-zero", vm.voice(0).synth.base_amp > 0.05)
    assert_eq("gate=1", 1, vm.voice(0).synth.gate)

    # Edit cutoff (relative)
    vm.dispatch(ControllerEvent(SET_PARAM, slot=0, param="cutoff",
                                value=200.0, extras={"relative": True}))
    assert_true("cutoff bumped by 200",
                vm.voice(0).synth.base_cutoff == 1700.0)

    # Note-off releases gate (KEYBOARD_ONLY)
    vm.dispatch(ControllerEvent(NOTE_OFF, slot=0, note=72))
    assert_eq("gate=0 after KEYBOARD_ONLY note_off",
              0, vm.voice(0).synth.gate)


def test_detach():
    print("\n[8] DETACH transitions HYBRID -> KEYBOARD_ONLY")
    p  = _default_profile()
    vm = VoiceManager(profile=p, n_slots=16)
    bm = BindingManager(vm)
    b  = FakeBlob(id=1, centroid=(400, 400), persistence=30, direction="rising")
    bm.on_blob_birth(b)
    bm.request_claim_hybrid(0)
    assert_true("slot 0 HYBRID", vm.is_hybrid(0))
    bm.request_detach(0)
    assert_true("slot 0 KEYBOARD_ONLY after detach", vm.is_keyboard_only(0))
    assert_eq("blob 1 unbound", None, vm.slot_for_blob(1))


# ── Run ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Audible Lava / DSO v4.0 — voice slot system tests")
    print("=" * 60)

    test_basic_spawn()
    test_keyboard_create_and_tie()
    test_immediate_tie_to_unbound()
    test_zombie_pool()
    test_claim_hybrid_and_set_param()
    test_routing_application()
    test_dispatch_controller_events()
    test_detach()

    print()
    print(f"Results: {PASS} passed, {FAIL} failed")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
