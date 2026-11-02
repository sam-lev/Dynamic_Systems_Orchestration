#!/usr/bin/env python3
# test_merge_release.py -- Audible Lava / DSO v4.9.1
#
# The merge slot leak and the orphan guard in voice_manager.py.
# No SuperCollider (OSC client is replaced by a recorder), no camera, no pygame.
# Run: python test_merge_release.py

import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

from voice_manager import VoiceManager, VoiceState
from binding_manager import BindingManager
from lamp_profile import LampProfile

PASS = FAIL = 0
def check(c, m):
    global PASS, FAIL
    if c: PASS += 1; print(f"  ok   {m}")
    else: FAIL += 1; print(f"  FAIL {m}")

class Rec:
    def __init__(self): self.sent = []
    def send_message(self, a, args): self.sent.append((a, list(args)))

class B:
    def __init__(self, id, y=300.0, x=150.0):
        self.id = id; self.centroid = (x, y); self.area = 12000.0; self.birth = 150.0
        self.death = 20.0; self.persistence = 30.0; self.orientation = 0.0
        self.elongation = 0.2; self.mean_hue = 0.0; self.direction = "rising"
        self.age = 20; self.vx = 0.0; self.vy = -0.5; self.speed = 0.5; self.ay = 0.0
        self.d_area = 0.0; self.d_persistence = 0.0

def vm(**kw):
    m = VoiceManager(profile=LampProfile.default(600, 300), n_slots=4, **kw)
    m.osc = Rec()
    return m

def gate_offs(rec):
    return [a for (addr, a) in rec.sent if addr == "/n_set" and "gate" in a and a[a.index("gate") + 1] == 0]


def test_merge_releases_slot():
    print("[merge] merged slot fades and is released (the v4.9 leak)")
    m = vm(); bm = BindingManager(m)
    a, b = B(1, y=300), B(2, y=320)
    bm.on_blob_birth(a); bm.on_blob_birth(b)
    for _ in range(5): m.process_frame([a, b])
    check(m.voice(0).synth.c_amp > 0.05, "merged-to-be voice is audible before the merge")
    bm.on_blob_merge(1, 2, [b])                     # blob 1 merges into 2
    check(m.slot_for_blob(1) is None and m.voice(0).blob_id is None, "binding dropped on merge")
    amps = []
    for _ in range(60):                             # 4 s at 15 fps
        m.process_frame([b]); amps.append(m.voice(0).synth.c_amp)
    check(amps[10] < amps[0], "amp decreases after merge (t_amp pinned at 0)")
    check(m.voice(0).state == VoiceState.IDLE, "merged slot returned to IDLE")
    check(0 not in m._pending_merge_release, "pending-merge entry cleared")
    check(len(gate_offs(m.osc)) >= 1, "gate 0 sent to the merged node")
    check(m.first_idle_slot() == 0, "slot 0 reusable")


def test_merge_timeout_path():
    print("[merge] timeout releases even if amp cannot fade")
    m = vm(merge_release_max_secs=0.05); bm = BindingManager(m)
    a, b = B(1), B(2, y=320)
    bm.on_blob_birth(a); bm.on_blob_birth(b)
    m.process_frame([a, b]); bm.on_blob_merge(1, 2, [b])
    m.smoothing = 0.0                               # amp frozen: fade path impossible
    time.sleep(0.06); m.process_frame([b])
    check(m.voice(0).state == VoiceState.IDLE, "released by timeout")


def test_pool_recovers_after_many_merges():
    print("[merge] pool does not fill up under repeated merges")
    m = vm(); bm = BindingManager(m)
    nid = 10
    for round_ in range(6):
        keep = B(nid); gone = B(nid + 1, y=320); nid += 2
        bm.on_blob_birth(keep); bm.on_blob_birth(gone)
        check(m.slot_for_blob(keep.id) is not None, f"round {round_}: blob got a slot")
        m.process_frame([keep, gone]); bm.on_blob_merge(gone.id, keep.id, [keep])
        for _ in range(60): m.process_frame([keep])
        bm.on_blob_death(keep.id)
        m.voice(m.voices_by_slot.index(next(v for v in m.voices_by_slot if v.state == VoiceState.ZOMBIE))).zombie_started -= 100
        m.process_frame([])
    check(all(v.state == VoiceState.IDLE for v in m.voices_by_slot), "all 4 slots idle after 6 merge rounds")


def test_orphan_guard():
    print("[orphan] bound voice whose blob vanished without a death event")
    m = vm(orphan_timeout_secs=0.05); bm = BindingManager(m)
    a = B(1); bm.on_blob_birth(a)
    m.process_frame([a])
    for _ in range(3): m.process_frame([])          # blob gone, no death event
    check(m.voice(0).state == VoiceState.BLOB_ONLY, "not reaped before the timeout")
    time.sleep(0.06); m.process_frame([])
    check(m.voice(0).state == VoiceState.ZOMBIE, "reaped to ZOMBIE after the timeout")
    check(m.slot_for_blob(1) is None, "binding dropped")
    m.voice(0).zombie_started -= 100; m.process_frame([])
    check(m.voice(0).state == VoiceState.IDLE, "zombie then expires normally")


def test_orphan_guard_ignores_coasting():
    print("[orphan] a blob still in the tracked list is never reaped")
    m = vm(orphan_timeout_secs=0.05); bm = BindingManager(m)
    a = B(1); bm.on_blob_birth(a)
    time.sleep(0.06)
    for _ in range(3): m.process_frame([a])         # still tracked (coasting counts)
    check(m.voice(0).state == VoiceState.BLOB_ONLY, "tracked blob keeps its voice")


def test_orphan_guard_hybrid_recent_play_detaches():
    print("[orphan] recently-played HYBRID detaches instead of zombifying")
    m = vm(orphan_timeout_secs=0.05); bm = BindingManager(m)
    a = B(1); bm.on_blob_birth(a); m.process_frame([a])
    m.claim_hybrid(0); m.note_on(0, 60, 100)
    time.sleep(0.06); m.process_frame([])
    check(m.voice(0).state == VoiceState.KEYBOARD_ONLY, "HYBRID with recent play -> KEYBOARD_ONLY")


if __name__ == "__main__":
    test_merge_releases_slot()
    test_merge_timeout_path()
    test_pool_recovers_after_many_merges()
    test_orphan_guard()
    test_orphan_guard_ignores_coasting()
    test_orphan_guard_hybrid_recent_play_detaches()
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
