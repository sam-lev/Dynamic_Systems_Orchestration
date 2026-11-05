# voice_continuity.py -- Audible Lava / DSO v4.12 "voice continuity"
#
# Additive. Subclasses VoiceManager and BindingManager; nothing in either file
# changes. audible_lava.py constructs these instead (--no-continuity restores
# the v4.11 classes). Fixes the symptom "blobs sound when first detected, then
# fall silent while they sit still, although they are still tracked":
#
# 1. EVERY TRACKED BLOB GETS A VOICE (ContinuityBindingManager.bind_unbound).
#    A voice was only ever bound on a tracker `birth` event. When a blob
#    flickers out of detection (stationary blobs near the top/bottom pool do:
#    their contrast against the frozen background hovers at the persistence
#    threshold), the tracker emits `death`, the voice is zombified and freed
#    after zombie_window (4 s); when the blob is found again the tracker
#    restores its OLD id through the re-identification buffer and, by design,
#    emits NO birth. Nothing re-bound it, so the blob stayed labelled "#id"
#    with no slot and no sound for the rest of its life. The same hole opened
#    when the pool was full at birth and a slot freed later. Now, after the
#    tracker events of each frame, any tracked blob older than rebind_min_age
#    frames without a slot is bound: resurrected from a matching zombie if one
#    exists (its sound continues), else given a fresh voice WITHOUT the birth
#    strike (it is not a new blob) that fades in on its normal attack.
#    At most rebind_max_per_frame per frame, most persistent first.
#
# 2. A LEVEL FLOOR FOR LIVE VOICES (ContinuityVoiceManager._compute_targets).
#    Governance (prominence mixer x0.5, the absent PROMINENCE_RANK +0.35, the
#    beat dip) could take a dim, still blob's level to 0. A voice bound to a
#    tracked blob now never goes below amp_floor_frac (0.4) x its base level:
#    governance can push it into the background, not mute it. Not applied to
#    zombies, merge fade-outs or keyboard-only voices.
#
# 3. THE BEAT DIP IS PROPORTIONAL. BEAT_PHASE -> AMP (depth -0.28) was a fixed
#    subtraction, so on a voice at amp 0.1 it removed everything for most of
#    each beat while a loud voice barely moved. It is now applied as a factor,
#    t_amp x (1 + depth x curve(phase)) = up to a 28 % dip on every voice: the
#    same accent at every level. The routing entry and its depth are unchanged;
#    only how it combines differs (MULTIPLICATIVE_AMP_SOURCES).
#
# Tests: test_voice_continuity.py (no SuperCollider, camera or pygame).

from __future__ import annotations

import time
from typing import Optional

from voice_manager   import VoiceManager, VoiceState
from binding_manager import BindingManager
from routing         import Source, Dest, apply_curve, extract_source_value

# AMP routings from these sources scale the level instead of adding to it.
MULTIPLICATIVE_AMP_SOURCES = {Source.BEAT_PHASE}


class ContinuityVoiceManager(VoiceManager):
    """VoiceManager with a level floor for live voices, a proportional beat
    dip, and a strike-less spawn for blobs that are not new (re-identified,
    or unbound because the pool was full at their birth)."""

    def __init__(self, *args, amp_floor_frac: float = 0.4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.amp_floor_frac = float(amp_floor_frac)

    def _is_live(self, v, by_id: dict) -> bool:
        return (v.state in (VoiceState.BLOB_ONLY, VoiceState.HYBRID)
                and v.blob_id is not None and v.blob_id in by_id
                and v.slot_id not in self._pending_merge_release)

    def _compute_targets(self, v, by_id: dict) -> None:
        live = self._is_live(v, by_id)
        mult = ([r for r in v.routing
                 if r.dest == Dest.AMP and r.src in MULTIPLICATIVE_AMP_SOURCES]
                if live else [])
        if mult:
            saved = v.routing
            v.routing = [r for r in saved if r not in mult]
            try:
                super()._compute_targets(v, by_id)
            finally:
                v.routing = saved
            blob = by_id[v.blob_id]
            factor = 1.0
            for r in mult:
                shaped = apply_curve(r.curve, extract_source_value(r.src, blob, self.profile)) * r.depth
                factor *= max(0.0, 1.0 + shaped)
            v.synth.t_amp *= factor
        else:
            super()._compute_targets(v, by_id)
        if live and self.amp_floor_frac > 0.0:
            v.synth.t_amp = max(v.synth.t_amp, self.amp_floor_frac * v.synth.base_amp)

    def spawn_continuing(self, blob) -> Optional[int]:
        """Like spawn_for_blob, but no \\blobStrike: for a blob that already
        existed (re-identified, or missed at birth). The voice fades in on its
        normal direction-dependent attack."""
        slot_id = self.first_idle_slot()
        if slot_id is None:
            return None
        v = self.voices_by_slot[slot_id]
        self._init_voice_from_blob(v, blob, spawn_origin="blob")
        v.state        = VoiceState.BLOB_ONLY
        v.blob_id      = blob.id
        v.synth_kind   = "mpe"
        v.last_seen_at = time.time()
        self.voices_by_blob[blob.id] = slot_id
        self._send_spawn(v, blob, fire_strike=False)
        return slot_id


class ContinuityBindingManager(BindingManager):
    """BindingManager that also binds any tracked blob left without a voice."""

    def __init__(self, voices, *, rebind_min_age: int = 8,
                 rebind_max_per_frame: int = 2, verbose: bool = True, **kwargs) -> None:
        super().__init__(voices, **kwargs)
        self.rebind_min_age       = int(rebind_min_age)
        self.rebind_max_per_frame = int(rebind_max_per_frame)
        self.verbose              = verbose
        self.rebinds              = 0     # total, for the HUD
        self.resurrections        = 0

    def handle_tracker_events(self, tracker_events: list, tracked_blobs: list) -> None:
        super().handle_tracker_events(tracker_events, tracked_blobs)
        self.bind_unbound(tracked_blobs)

    def bind_unbound(self, tracked_blobs: list) -> list:
        """Bind tracked blobs that have no slot. Returns [(blob_id, slot_id, how)]."""
        cands = [b for b in tracked_blobs
                 if self.voices.slot_for_blob(b.id) is None
                 and int(getattr(b, "age", self.rebind_min_age)) >= self.rebind_min_age]
        cands.sort(key=lambda b: float(getattr(b, "persistence", 0.0)), reverse=True)
        done = []
        for b in cands[: self.rebind_max_per_frame]:
            if self.voices.try_resurrect(b):
                how = "resurrected"
                self.resurrections += 1
            else:
                spawn = getattr(self.voices, "spawn_continuing", None) or self.voices.spawn_for_blob
                slot = spawn(b)
                if slot is None:
                    break                       # pool full: try again next frame
                how = "new voice"
            slot = self.voices.slot_for_blob(b.id)
            self.rebinds += 1
            done.append((b.id, slot, how))
            if self.verbose:
                print(f"  ^ rebind blob #{b.id} -> S{slot:02d} ({how}; it had no voice)")
        return done

    def hud_summary(self) -> str:
        return super().hud_summary() + f"  rebinds:{self.rebinds}"
