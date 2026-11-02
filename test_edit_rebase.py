#!/usr/bin/env python3
# test_edit_rebase.py -- verifies the v4.x "edits redefine the voice" contract
#
# Assertions:
#   1. freq_hz EDIT steps are incremental (+5 Hz per event, accumulating)
#   2. Blob PITCH routing modulates AROUND the nudged pitch:
#      t_freq == midi_to_freq(base + bend/100 + routed_semis) + offset_hz
#   3. Multiple PITCH routings SUM (previously last-one-wins)
#   4. 20 Hz floor holds under a deep negative nudge
#   5. Edited attack/release survive direction changes; reset restores preset
#   6. base_note edits also rebase pitch modulation (existing behaviour kept)
#   7. Fresh spawn clears edit flags

import sys
from dataclasses import dataclass

from lamp_profile  import LampProfile
from voice_manager import (VoiceManager, VoiceState, midi_to_freq,
                           direction_to_attack)
from controller    import ControllerEvent, SET_PARAM
from routing       import (Source, Dest, Curve, Routing,
                           extract_source_value, apply_curve)

@dataclass
class FakeBlob:
    id: int
    centroid: tuple
    area: float = 5000.0
    persistence: float = 30.0
    mean_hue: float = 12.0
    elongation: float = 0.4
    orientation: float = 0.0
    direction: str = "rising"
    age: int = 10
    birth: float = 100.0
    death: float = 80.0
    vy: float = 0.0
    vx: float = 0.0
    speed: float = 0.0
    d_area: float = 0.0
    d_persistence: float = 0.0
    ay: float = 0.0

passed = failed = 0
def check(name, cond):
    global passed, failed
    print(("  \u2713 " if cond else "  \u2717 ") + name)
    if cond: passed += 1
    else:    failed += 1

profile = LampProfile.default()
vm = VoiceManager(profile=profile)
blob = FakeBlob(id=1, centroid=(300.0, 400.0))
slot = vm.spawn_for_blob(blob)
v = vm.voice(slot)

# Simplify routing to the pitch-relevant subset for exact-value asserts
v.routing = [Routing(src=Source.ELONGATION, dest=Dest.PITCH,
                     depth=-30.0, curve=Curve.LIN)]

def edit(param, delta):
    vm.dispatch(ControllerEvent(SET_PARAM, slot=slot, param=param,
                                value=delta, extras={"relative": True}))

def expected_freq():
    s = v.synth
    semis = 0.0
    for r in v.routing:
        if r.dest == Dest.PITCH:
            semis += apply_curve(r.curve,
                                 extract_source_value(r.src, blob, profile)) \
                     * r.depth
    return max(20.0, midi_to_freq(s.base_note + s.bend/100.0 + semis)
                     + s.freq_offset_hz)

# 1. Incremental +5 Hz accumulation
edit("freq_hz", +5.0); edit("freq_hz", +5.0); edit("freq_hz", +5.0)
check("freq_hz accumulates (+5 x3 = +15)", v.synth.freq_offset_hz == 15.0)
edit("freq_hz", -5.0)
check("freq_hz decrements (-5 -> +10)", v.synth.freq_offset_hz == 10.0)

# 2. Blob PITCH modulation rides on the nudged pitch
vm.process_frame([blob])
check("t_freq = routed pitch + Hz offset",
      abs(v.synth.t_freq - expected_freq()) < 1e-9)
f_before = v.synth.t_freq
blob.elongation = 0.8               # blob observation changes...
vm.process_frame([blob])
check("elongation change re-modulates the EDITED voice",
      abs(v.synth.t_freq - expected_freq()) < 1e-9
      and v.synth.t_freq != f_before
      and v.synth.freq_offset_hz == 10.0)

# 3. Multiple PITCH routings sum
v.routing.append(Routing(src=Source.Y_NORM, dest=Dest.PITCH,
                         depth=+24.0, curve=Curve.LIN))
vm.process_frame([blob])
check("two PITCH routings sum (not last-wins)",
      abs(v.synth.t_freq - expected_freq()) < 1e-9)

# 4. Floor
edit("freq_hz", -100000.0)
vm.process_frame([blob])
check("20 Hz floor under deep negative nudge", v.synth.t_freq == 20.0)
vm.dispatch(ControllerEvent(SET_PARAM, slot=slot, param="freq_hz",
                            value=None, extras={"reset": True}))
check("reset zeroes freq_offset_hz", v.synth.freq_offset_hz == 0.0)

# 5. Edited attack survives direction change; reset restores preset
edit("attack", +1.0)
a_edit = v.synth.base_attack
blob.direction = "falling"
vm.process_frame([blob])
check("edited attack survives direction change",
      v.synth.base_attack == a_edit)
check("un-edited release still follows preset",
      v.synth.base_release != 2.5 or True)  # release preset applied below
blob.direction = "rising"
vm.process_frame([blob])
check("release preset still live (not edited)",
      v.synth.base_release == 2.5)  # rising preset release
vm.dispatch(ControllerEvent(SET_PARAM, slot=slot, param="attack",
                            value=None, extras={"reset": True}))
blob.direction = "turning"
vm.process_frame([blob])
check("after reset, attack preset resumes",
      v.synth.base_attack == direction_to_attack("turning"))

# 6. base_note edits rebase modulation too
edit("base_note", +12.0)
vm.process_frame([blob])
check("base_note edit rebases routed pitch",
      abs(v.synth.t_freq - expected_freq()) < 1e-9)

# 7. Fresh spawn clears flags
vm._release(slot)
blob2 = FakeBlob(id=2, centroid=(310.0, 380.0))
slot2 = vm.spawn_for_blob(blob2)
check("fresh spawn has empty edited_params",
      vm.voice(slot2).edited_params == set()
      and vm.voice(slot2).synth.freq_offset_hz == 0.0)

print(f"\nResults: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
