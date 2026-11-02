# binding_manager.py -- Audible Lava / DSO v4.0 (MPE Edition)
#
# Sits between BlobTracker.events and the VoiceManager. Implements the
# bidirectional blob<->voice binding policy:
#
#   on blob birth:
#     1. If the player has CREATE_EMPTY'd voice(s) and pressed 'T' on this
#        very slot to mark it as awaiting a blob, tie the new blob to it.
#     2. Otherwise, scan the zombie pool for a compatible voice to resurrect.
#     3. Otherwise, spawn a fresh BLOB_ONLY voice in the next IDLE slot.
#
#   on blob death:
#     - HYBRID  voice: detach (becomes KEYBOARD_ONLY) if player has played
#                       it within blob_recent_play_window, otherwise zombify
#                       like a BLOB_ONLY voice would.
#     - BLOB_ONLY voice: zombify (the v3.2 mechanism).
#     - KEYBOARD_ONLY: shouldn't reach here (the voice has no blob), but if
#                       it does, drop the bind silently.
#
#   on blob merge / split:
#     - delegate to the VoiceManager's existing v3.x merge/split handlers,
#       which now operate on slots-by-blob_id.
#
# The controller layer pipes its TIE / DETACH / CLAIM_HYBRID / CREATE_EMPTY
# events through this file too, so all "bind/unbind" semantics live in one place.

from __future__ import annotations
import time
from typing import Iterable, Optional


class BindingManager:
    """
    Lifecycle bridge between tracker events, controller events, and the
    voice manager. Single source of truth for slot allocation policy.

    Attributes:
        voices                The VoiceManager (16-slot pool).
        pending_keyboard_voices   list[int] of slot ids of KEYBOARD_ONLY
                                  voices that have signalled "tie me to the
                                  next blob to be born". Earliest first.
        blob_recent_play_window   seconds; if a HYBRID voice's player_played_at
                                  is within this window when its blob dies,
                                  it stays alive as KEYBOARD_ONLY rather than
                                  zombifying.
    """

    def __init__(self, voices, *, blob_recent_play_window: float = 5.0) -> None:
        self.voices                  = voices
        self.pending_keyboard_voices: list[int] = []
        self.blob_recent_play_window = blob_recent_play_window

    # ── Tracker event entry points ───────────────────────────────────────
    def handle_tracker_events(self, tracker_events: list, tracked_blobs: list) -> None:
        """
        Dispatch tracker.events through the right policy. tracked_blobs is
        passed through to merge/split handlers so they can look up blob
        state by id.
        """
        # Pre-index for cheap lookups
        by_id = {b.id: b for b in tracked_blobs}
        for ev in tracker_events:
            if not ev:
                continue
            kind = ev[0]
            if kind == "birth":
                blob = by_id.get(ev[1])
                if blob is not None:
                    self.on_blob_birth(blob)
            elif kind == "death":
                self.on_blob_death(ev[1])
            elif kind == "merge":
                _, merged_id, survivor_id = ev
                self.on_blob_merge(merged_id, survivor_id, tracked_blobs)
            elif kind == "split":
                _, parent_id, child_id = ev
                self.on_blob_split(parent_id, child_id, tracked_blobs)

    def on_blob_birth(self, blob) -> None:
        # 1. Pending keyboard-created voice waiting?
        while self.pending_keyboard_voices:
            slot_id = self.pending_keyboard_voices.pop(0)
            if self.voices.is_keyboard_only(slot_id):
                self.voices.tie_to_blob(slot_id, blob)
                return
            # else stale — keep popping

        # 2. Zombie pool resurrection?
        if self.voices.try_resurrect(blob):
            return

        # 3. Fresh BLOB_ONLY spawn into next IDLE slot.
        self.voices.spawn_for_blob(blob)

    def on_blob_death(self, blob_id: int) -> None:
        slot_id = self.voices.slot_for_blob(blob_id)
        if slot_id is None:
            return
        v       = self.voices.voice(slot_id)
        if v is None:
            return

        if self.voices.is_hybrid(slot_id):
            # HYBRID: did the player touch this voice recently?
            played_recently = (time.time() - v.last_played_at
                               < self.blob_recent_play_window)
            if played_recently:
                self.voices.detach(slot_id)
                return
            # else fall through to zombify

        if self.voices.is_blob_only(slot_id) or self.voices.is_hybrid(slot_id):
            self.voices.zombify(slot_id)
            return

        # KEYBOARD_ONLY shouldn't have a blob bind; if it somehow did,
        # break the bind silently.
        self.voices.detach(slot_id)

    def on_blob_merge(self, merged_id: int, survivor_id: int,
                      tracked_blobs: list) -> None:
        self.voices.handle_merge(merged_id, survivor_id, tracked_blobs)

    def on_blob_split(self, parent_id: int, child_id: int,
                      tracked_blobs: list) -> None:
        self.voices.handle_split(parent_id, child_id, tracked_blobs)

    # ── Controller event entry points ────────────────────────────────────
    # The main loop forwards controller events that affect bindings here
    # (rather than to VoiceManager directly), so the bind-state policy
    # stays in one place.

    def request_create_empty(self) -> Optional[int]:
        """User pressed 'N' (or its MIDI equivalent). Allocate the next
        IDLE slot as KEYBOARD_ONLY and return the slot id (or None if all
        slots are in use)."""
        slot_id = self.voices.create_empty()
        return slot_id

    def request_tie(self, slot_id: int,
                    tracked_blobs: Optional[list] = None) -> bool:
        """User pressed 'T' on slot_id. Two semantics:

          - If a tracked, unbound blob exists right now, tie immediately
            to the geometrically nearest unbound blob.
          - Otherwise, queue the slot in pending_keyboard_voices so the
            *next* blob birth ties to it.

        Returns True if an immediate tie happened.
        """
        if not self.voices.is_keyboard_only(slot_id):
            return False

        if tracked_blobs:
            # Find unbound blobs (no slot already bound to them)
            unbound = [b for b in tracked_blobs
                       if self.voices.slot_for_blob(b.id) is None]
            if unbound:
                # Pick the most-persistent unbound blob — likely the most
                # musically meaningful target.
                tgt = max(unbound, key=lambda b: b.persistence)
                self.voices.tie_to_blob(slot_id, tgt)
                return True

        if slot_id not in self.pending_keyboard_voices:
            self.pending_keyboard_voices.append(slot_id)
        return False

    def request_detach(self, slot_id: int) -> None:
        """User pressed 'D' (or DETACH event). Break the blob bind on
        slot_id; the voice continues as KEYBOARD_ONLY."""
        self.voices.detach(slot_id)
        # Also drop from pending list if it was queued there
        if slot_id in self.pending_keyboard_voices:
            self.pending_keyboard_voices.remove(slot_id)

    def request_claim_hybrid(self, slot_id: int) -> None:
        """User pressed 'B'. Promote a BLOB_ONLY voice to HYBRID so the
        player can play notes on it while the blob continues to modulate."""
        self.voices.claim_hybrid(slot_id)

    # ── Diagnostics ──────────────────────────────────────────────────────
    def hud_summary(self) -> str:
        n_pending = len(self.pending_keyboard_voices)
        return (f"pending_tie:{n_pending}  "
                f"window:{self.blob_recent_play_window:.1f}s")
