# voice_manager.py -- Audible Lava / DSO v4.0 (MPE Edition)
#
# v4.9.2 (Oct 2026) -- recreate the default group after /g_freeAll 0 at startup.
# v4.9.1 (Oct 2026) -- two slot-leak fixes in the release path:
#
#   * MERGE LEAK. handle_merge() set the merged voice's t_amp = 0 once and
#     queued the slot in _pending_merge_release, expecting c_amp to fade below
#     0.005 so _check_merge_releases() could free it. But the slot stayed
#     BLOB_ONLY with blob_id = None, and _compute_targets() runs every frame
#     and rebuilt t_amp from base_amp (no routings apply without a blob), so
#     c_amp converged to base_amp and the slot was NEVER released. Every merge
#     permanently consumed a slot, left droning at base_amp with nothing
#     modulating it. Lava lamps merge constantly, so the pool filled within
#     minutes: a static multi-voice chord, new blobs unable to spawn, and
#     nothing to release when the camera was covered (the orphans had no blob
#     to die). Fix: _compute_targets() pins t_amp = 0 while a slot is pending
#     merge release, and _check_merge_releases() also frees the slot after
#     merge_release_max_secs regardless of amplitude.
#
#   * ORPHAN GUARD. A BLOB_ONLY/HYBRID voice whose blob_id has not appeared in
#     the tracked set for orphan_timeout_secs is zombified (or detached, for a
#     recently-played HYBRID), exactly as a death event would do. Covers any
#     path where a track disappears without the tracker emitting "death"
#     (re-identification under a new id, a dropped event, a tracker restart).
#     Coasting blobs are still in the tracked list, so this never fires on a
#     blob that is merely between detections.
#
# Major refactor from v3.x. The implicit blob_id -> voice mapping is replaced
# by an explicit pool of 16 voice slots. Each slot carries:
#
#   - A lifecycle state: IDLE | BLOB_ONLY | KEYBOARD_ONLY | HYBRID | ZOMBIE
#   - A SynthState with both editable bases and smoothed current/target
#     values for every continuous output parameter.
#   - A routing list of (Source -> Dest, depth, curve) tuples.
#   - The SC node id for the running \mpeVoice synth (or None when IDLE).
#
# State machine:
#                create_empty                    tie_to_blob
#       IDLE  ───────────────────►  KEYBOARD_ONLY ───────────────────►  HYBRID
#         │                              ▲                                │
#         │ spawn_for_blob               │                                │
#         ▼                              │  detach (after blob died OR    │
#     BLOB_ONLY  ─── claim_hybrid ──────►│         user pressed D)        │
#         │                              ▲                                │
#         │ blob died,                   │ claim_hybrid                   │
#         │ no recent player input       │                                │
#         ▼                              │                                ▼
#       ZOMBIE ─── try_resurrect ───► BLOB_ONLY ────────────────────►  HYBRID
#         │
#         │ window expires
#         ▼
#       IDLE  (release SC node)
#
# The v3.x merge/split handlers are preserved as handle_merge / handle_split
# operating on slots-by-blob-id.
#
# The v3.2 zombie pool sits inside this file as the BLOB_ONLY -> ZOMBIE ->
# BLOB_ONLY transition. Resurrection criteria: distance + persistence ratio.

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from pythonosc import udp_client

from lamp_profile   import LampProfile
from blob_detection import RISING, FALLING, TURNING
from routing        import (
    Source, Dest, Curve, Routing, EventRoute,
    DEFAULT_BLOB_ROUTING, DEFAULT_BLOB_EVENTS,
    apply_curve, extract_source_value,
)


# ── Musical helpers (unchanged from v3.x) ────────────────────────────────────
PENTATONIC   = [0, 2, 4, 7, 9]
ROOT_NOTE    = 48     # C3
OCTAVE_RANGE = 3


def y_to_midi(y_norm: float, root: int = ROOT_NOTE,
              octave_range: int = OCTAVE_RANGE) -> int:
    continuous = (1.0 - y_norm) * octave_range * 12   # y_norm: 1 = top
    octave     = int(continuous // 12)
    chromatic  = continuous % 12
    degree     = min(PENTATONIC, key=lambda s: abs(s - chromatic))
    return root + octave * 12 + degree


def _nearest_field_freq(freq_hz: float, field: list) -> float:
    """Nearest frequency whose pitch class is in `field` (v4.2 harmony)."""
    if freq_hz <= 0 or not field:
        return freq_hz
    midi = 69.0 + 12.0 * math.log2(freq_hz / 440.0)
    best, best_d = midi, 1e9
    base = 12 * round(midi / 12)
    for pc in field:
        for octv in (-12, 0, 12):
            cand = base + octv + pc
            d = abs(cand - midi)
            if d < best_d:
                best, best_d = cand, d
    return 440.0 * 2 ** ((best - 69.0) / 12.0)


def midi_to_freq(n: float) -> float:
    return 440.0 * (2.0 ** ((n - 69) / 12.0))


def x_to_pan(x_norm: float) -> float:
    return x_norm * 2.0 - 1.0


def direction_to_attack(direction: str) -> float:
    return {RISING: 1.5, FALLING: 3.0, TURNING: 2.0}.get(direction, 2.0)


def direction_to_release(direction: str) -> float:
    return {RISING: 2.5, FALLING: 5.0, TURNING: 3.5}.get(direction, 3.5)


# ── Voice lifecycle states ───────────────────────────────────────────────────
class VoiceState(Enum):
    IDLE          = 0
    BLOB_ONLY     = 1
    KEYBOARD_ONLY = 2
    HYBRID        = 3
    ZOMBIE        = 4


# ── SynthState: editable bases + smoothed output ─────────────────────────────
@dataclass
class SynthState:
    """
    Full per-voice synth state for the \\mpeVoice SynthDef.

    Each continuous output parameter has both:
      - base_<x>     : the player-editable starting value (set in EDIT mode)
      - t_<x>        : the target after routing modulation has been added
      - c_<x>        : the smoothed current value sent to SC last frame

    Static fields (no smoothing) are written to SC only when they change.
    """
    # Pitch
    base_note:   float = 60.0      # MIDI note (the keyboard base; HYBRID adds
                                   # routing modulation on top)
    bend:        float = 0.0       # cents
    freq_offset_hz: float = 0.0    # EDIT-mode incremental pitch nudge, in
                                   # absolute Hz (linear, not musical units).
                                   # Applied AFTER routing modulation in
                                   # _compute_targets, so it re-centres the
                                   # voice: blob-driven PITCH modulation always
                                   # rides on top of the nudged pitch. Deliberate
                                   # contrast with base_note (semitones) / bend
                                   # (cents), which are log/musical increments.
    scale_id:    int   = 0         # 0 = chromatic, 1 = pentatonic, 2 = whole-tone
    chord_id:    int   = 0         # 0 = single, 1 = unison, 2 = +5th, 3 = triad

    # Wavetable
    base_wave_pos: float = 0.0     # 0..1
    wave_bank:     int   = 0       # 0..3

    # Envelope
    base_attack:  float = 1.5
    base_decay:   float = 0.4
    base_sustain: float = 0.7
    base_release: float = 2.5

    # Filter
    base_cutoff:  float = 1500.0
    base_res:     float = 0.30
    fmode:        float = 0.0      # 0 = LP, 0.5 = BP, 1 = HP  (routable base)

    # Amp/pan bases
    base_amp:     float = 0.30
    base_pan:     float = 0.0

    # Expressive timbre bases (v4.1) — all 0..1, default 0 = inert so an
    # un-routed voice is bit-identical to v4.0. Driven by blob motion /
    # change-of-state via the routing matrix (see routing.DEFAULT_BLOB_ROUTING).
    base_detune:  float = 0.0      # unison spread / beating
    base_shimmer: float = 0.0      # vibrato + tremolo depth
    base_noise:   float = 0.0      # breath / air blend
    base_drive:   float = 0.0      # wavefold saturation

    # Smoothed current values (sent to SC)
    c_freq:    float = 220.0
    c_amp:     float = 0.0
    c_pan:     float = 0.0
    c_cutoff:  float = 1500.0
    c_res:     float = 0.30
    c_wave_pos: float= 0.0
    c_attack:   float= 1.5
    c_release:  float= 2.5
    c_detune:   float= 0.0
    c_shimmer:  float= 0.0
    c_noise:    float= 0.0
    c_drive:    float= 0.0
    c_fmode:    float= 0.0

    # Targets (recomputed every frame from base + routing)
    t_freq:    float = 220.0
    t_amp:     float = 0.0
    t_pan:     float = 0.0
    t_cutoff:  float = 1500.0
    t_res:     float = 0.30
    t_wave_pos: float= 0.0
    t_attack:   float= 1.5
    t_release:  float= 2.5
    t_detune:   float= 0.0
    t_shimmer:  float= 0.0
    t_noise:    float= 0.0
    t_drive:    float= 0.0
    t_fmode:    float= 0.0

    # Gate (1 while voice is playing; 0 triggers release in SC)
    gate: int = 0

    @classmethod
    def default(cls) -> "SynthState":
        return cls()

    def freeze_current_to_target(self) -> None:
        """Snap c_* = t_* (used at spawn so the first OSC packet sets state
        without a smoothing transient)."""
        self.c_freq    = self.t_freq
        self.c_amp     = self.t_amp
        self.c_pan     = self.t_pan
        self.c_cutoff  = self.t_cutoff
        self.c_res     = self.t_res
        self.c_wave_pos= self.t_wave_pos
        self.c_attack  = self.t_attack
        self.c_release = self.t_release
        self.c_detune  = self.t_detune
        self.c_shimmer = self.t_shimmer
        self.c_noise   = self.t_noise
        self.c_drive   = self.t_drive
        self.c_fmode   = self.t_fmode


# ── Voice (slot in the pool) ─────────────────────────────────────────────────
@dataclass
class Voice:
    slot_id:    int
    state:      VoiceState              = VoiceState.IDLE
    blob_id:    Optional[int]           = None
    sc_node:    Optional[int]           = None
    angle_node: Optional[int]           = None
    synth:      SynthState              = field(default_factory=SynthState.default)
    routing:    List[Routing]           = field(default_factory=list)
    events:     List[EventRoute]        = field(default_factory=list)

    # Which SuperCollider SynthDef currently represents this slot's main node.
    # "mpe" is the normal wavetable voice; "bass" is the low stutter voice
    # used when the bound blob is low, falling, and expanding.
    synth_kind: str = "mpe"

    born_at:        float = 0.0
    zombie_started: float = 0.0
    # v4.3 bass-stutter episode bookkeeping. `bass_started` stamps entry so a
    # single episode can be time-capped; `bass_exit_streak` counts consecutive
    # frames failing the continue condition, so one noisy frame cannot cause a
    # node teardown/respawn (which is audible as a click).
    bass_started:     float = 0.0
    bass_exit_streak: int   = 0
    last_played_at: float = 0.0       # when the player last sent NOTE_ON here
    last_direction: str   = TURNING   # for direction-change event detection
    spawn_origin:   str   = "blob"    # 'blob' or 'keyboard'
    # v4.9.1: last wall-clock time the bound blob appeared in the tracked set.
    # 0.0 = never (fresh slot). Drives the orphan guard in process_frame().
    last_seen_at:   float = 0.0

    # UI param names the player has touched in EDIT mode (set_param adds,
    # reset removes). Discrete blob EVENT handlers (attack_release_preset)
    # consult this so an EDIT-mode change REDEFINES the voice: once the
    # player sets 'attack' or 'release', direction changes no longer stomp
    # those bases. Continuous routings are unaffected — they already
    # modulate additively relative to the (edited) base every frame.
    # Pressing R (reset) on a param clears its flag, restoring preset
    # behaviour for that param.
    edited_params:  set = field(default_factory=set)


# ── VoiceManager (16-slot pool) ──────────────────────────────────────────────
class VoiceManager:
    """
    Central per-frame engine. Owns the 16 slot pool and the OSC client.
    Receives:
      - tracked_blobs                each frame (for routing application)
      - ControllerEvents             via dispatch()  from controller layer
      - tracker.events / merges      via BindingManager

    Sends:
      - /s_new   on slot spawn
      - /n_set   each frame for active slots
      - /n_set   gate=0 on slot release
    """

    def __init__(
        self,
        sc_ip:   str = "127.0.0.1",
        sc_port: int = 57110,
        smoothing: float = 0.12,
        n_slots:  int = 16,
        profile:  Optional[LampProfile] = None,
        # zombie pool (v3.2)
        zombie_window:           float = 4.0,
        zombie_sustain_amp_scale: float = 0.35,
        zombie_match_radius_frac: float = 0.15,
        zombie_min_match_radius:  float = 60.0,
        zombie_persistence_ratio: float = 3.0,
        # routing
        default_routing_smoothing: float = 0.10,
        # synth def names
        mpe_synthdef:   str = "mpeVoice",
        angle_synthdef: str = "blobAngle",
        strike_synthdef: str = "blobStrike",
        bass_synthdef: str = "bassStutter",
        # v4.9.1 release-path safety nets
        merge_release_max_secs: float = 3.0,
        orphan_timeout_secs:    float = 3.0,
        blob_recent_play_window: float = 5.0,
    ) -> None:
        self.osc       = udp_client.SimpleUDPClient(sc_ip, sc_port)
        # ── v4.2 score-grammar layer (additive). audible_lava sets
        # `score_state` each frame; `score_gain` (0..1) dials governance in
        # and out live. None -> identical behaviour to v4.1.
        self.score_state = None
        self.score_gain  = 1.0
        self.profile   = profile or LampProfile.default()
        self.smoothing = smoothing
        self.n_slots   = n_slots
        self.next_node_id = 1000

        # The pool itself + the index by blob id.
        self.voices_by_slot: List[Voice] = [Voice(slot_id=i) for i in range(n_slots)]
        self.voices_by_blob: Dict[int, int] = {}     # blob_id -> slot_id

        self.selected_slot: int = 0

        # Zombie pool params
        self.zombie_window            = zombie_window
        self.zombie_sustain_amp_scale = zombie_sustain_amp_scale
        self.zombie_match_radius_frac = zombie_match_radius_frac
        self.zombie_min_match_radius  = zombie_min_match_radius
        self.zombie_persistence_ratio = zombie_persistence_ratio

        self.default_routing_smoothing = default_routing_smoothing

        # SynthDef names
        self.mpe_synthdef    = mpe_synthdef
        self.angle_synthdef  = angle_synthdef
        self.strike_synthdef = strike_synthdef
        self.bass_synthdef   = bass_synthdef

        # Existing blob voices temporarily turn into \bassStutter when the
        # blob is low in the lamp, moving downward, and expanding.  Hysteresis
        # keeps the mode from chattering at the threshold.  Tune these live.
        self.bass_y_on      = 0.30   # y_norm <= lower 30% of lamp
        self.bass_y_off     = 0.35   # was 0.40
        self.bass_vy_on     = 0.30   # px/frame downward; tracker uses vy > 0
        self.bass_vy_off    = 0.15   # was 0.10
        self.bass_darea_on  = 0.05   # +5% area/sec
        self.bass_darea_off = 0.01   # was -0.02 (a steady blob never exited)
        self.bass_min_age   = 15     # was 4; ~1s at 15fps, real blobs only

        # Episode limits. `bass_max_secs` is the safety net: whatever the
        # thresholds do on a given lamp, no single voice can drone past it.
        # `bass_exit_frames` debounces the exit so threshold noise does not
        # cause repeated respawns. `max_bass_voices` keeps the gesture a
        # gesture -- with 10+ simultaneous blobs, an uncapped bass layer is
        # a wall of sub-bass rather than an accent.
        self.bass_max_secs    = 6.0
        self.bass_exit_frames = 3
        self.max_bass_voices  = 2
        self.bass_retrigger_secs = 3.0   # cooldown before a slot may re-enter

        # Pending merge fade-outs (slot_id -> time queued) preserved from v3.x
        self._pending_merge_release: dict[int, float] = {}
        # v4.9.1: a merged slot is freed when its amp has faded OR after this
        # long, whichever comes first, so it can never be stranded.
        self.merge_release_max_secs = merge_release_max_secs
        # v4.9.1: a bound voice whose blob has been absent from the tracked
        # set for this long is treated as dead (see process_frame step 3b).
        self.orphan_timeout_secs    = orphan_timeout_secs
        self.blob_recent_play_window = blob_recent_play_window

        # MIDI Learn target (slot, param_name) — set by LEARN_ARM event
        self.learn_target: Optional[tuple[int, str]] = None

        # Free all stale SC nodes from a previous run. /g_freeAll 0 also
        # removes the server's default group (node 1), which nothing here
        # needs (we spawn into group 0) but language-side Synth(...) tests in
        # the SC IDE do -- so recreate it (v4.9.2).
        try:
            self.osc.send_message("/g_freeAll", [0])
            self.osc.send_message("/g_new", [1, 0, 0])
        except Exception:
            pass

    # ── Public read helpers (used by BindingManager and HUD) ─────────────
    def voice(self, slot_id: int) -> Optional[Voice]:
        if 0 <= slot_id < self.n_slots:
            return self.voices_by_slot[slot_id]
        return None

    def slot_for_blob(self, blob_id: int) -> Optional[int]:
        return self.voices_by_blob.get(blob_id)

    def is_idle(self, slot_id: int) -> bool:
        v = self.voice(slot_id);  return v is not None and v.state == VoiceState.IDLE
    def is_blob_only(self, slot_id: int) -> bool:
        v = self.voice(slot_id);  return v is not None and v.state == VoiceState.BLOB_ONLY
    def is_keyboard_only(self, slot_id: int) -> bool:
        v = self.voice(slot_id);  return v is not None and v.state == VoiceState.KEYBOARD_ONLY
    def is_hybrid(self, slot_id: int) -> bool:
        v = self.voice(slot_id);  return v is not None and v.state == VoiceState.HYBRID
    def is_zombie(self, slot_id: int) -> bool:
        v = self.voice(slot_id);  return v is not None and v.state == VoiceState.ZOMBIE

    def alive_slots(self) -> list[int]:
        return [i for i, v in enumerate(self.voices_by_slot)
                if v.state != VoiceState.IDLE]

    def first_idle_slot(self) -> Optional[int]:
        for i, v in enumerate(self.voices_by_slot):
            if v.state == VoiceState.IDLE:
                return i
        return None

    # ── Slot lifecycle ───────────────────────────────────────────────────
    def spawn_for_blob(self, blob) -> Optional[int]:
        """Allocate a fresh BLOB_ONLY voice for blob into the next IDLE slot.
        Returns slot_id, or None if the pool is full."""
        slot_id = self.first_idle_slot()
        if slot_id is None:
            return None
        v = self.voices_by_slot[slot_id]
        self._init_voice_from_blob(v, blob, spawn_origin="blob")
        v.state         = VoiceState.BLOB_ONLY
        v.blob_id       = blob.id
        v.synth_kind    = "mpe"
        v.last_seen_at  = time.time()
        self.voices_by_blob[blob.id] = slot_id
        self._send_spawn(v, blob, fire_strike=True)
        return slot_id

    def create_empty(self) -> Optional[int]:
        """Allocate a new KEYBOARD_ONLY voice in the next IDLE slot."""
        slot_id = self.first_idle_slot()
        if slot_id is None:
            return None
        v = self.voices_by_slot[slot_id]
        # Default neutral synth state; player edits from here
        v.synth          = SynthState.default()
        v.routing        = []        # KEYBOARD_ONLY voices have no routings
        v.events         = []
        v.edited_params  = set()
        v.state          = VoiceState.KEYBOARD_ONLY
        v.blob_id        = None
        v.synth_kind     = "mpe"
        v.born_at        = time.time()
        v.spawn_origin   = "keyboard"
        v.synth.gate     = 1
        v.synth.t_amp    = v.synth.base_amp
        v.synth.t_freq   = midi_to_freq(v.synth.base_note)
        v.synth.t_cutoff = v.synth.base_cutoff
        v.synth.t_pan    = v.synth.base_pan
        v.synth.t_res    = v.synth.base_res
        v.synth.t_wave_pos = v.synth.base_wave_pos
        v.synth.t_attack = v.synth.base_attack
        v.synth.t_release= v.synth.base_release
        v.synth.freeze_current_to_target()
        # NOTE: Spawn the SC node already with gate=0 (silent), so the
        # voice exists for editing but doesn't make sound until NOTE_ON.
        # We'll re-gate on NOTE_ON.
        self._send_spawn(v, blob=None, fire_strike=False, gate=0)
        return slot_id

    def tie_to_blob(self, slot_id: int, blob) -> None:
        """Bind a KEYBOARD_ONLY voice to a blob -> HYBRID."""
        v = self.voice(slot_id)
        if v is None or v.state != VoiceState.KEYBOARD_ONLY:
            return
        # Don't double-bind a blob already on another slot
        existing = self.voices_by_blob.get(blob.id)
        if existing is not None and existing != slot_id:
            return
        v.state      = VoiceState.HYBRID
        v.blob_id    = blob.id
        v.last_seen_at = time.time()
        self._clear_bass_state(v)
        v.routing    = list(DEFAULT_BLOB_ROUTING)
        v.events     = list(DEFAULT_BLOB_EVENTS)
        v.last_direction = blob.direction
        self.voices_by_blob[blob.id] = slot_id

    def claim_hybrid(self, slot_id: int) -> None:
        """Promote BLOB_ONLY -> HYBRID. The player can now play notes on
        this voice in addition to the blob's modulation."""
        v = self.voice(slot_id)
        if v is None or v.state != VoiceState.BLOB_ONLY:
            return
        v.state = VoiceState.HYBRID
        # Set the keyboard's base_note to the voice's current pitch so the
        # player doesn't get a sudden octave jump when they hit the first
        # note key.
        if v.synth.c_freq > 1.0:
            v.synth.base_note = 69.0 + 12.0 * math.log2(v.synth.c_freq / 440.0)

    def detach(self, slot_id: int) -> None:
        """Break a HYBRID/KEYBOARD_ONLY/BLOB_ONLY voice's blob bind. The
        voice continues as KEYBOARD_ONLY (or stays IDLE if it had no blob)."""
        v = self.voice(slot_id)
        if v is None:
            return
        if v.synth_kind == "bass":
            # Detaching removes blob modulation, so return the slot to the normal
            # keyboard/MPE synth before dropping the blob binding.
            live_blob = None
            self._switch_to_mpe(v, live_blob)
        if v.blob_id is not None:
            self.voices_by_blob.pop(v.blob_id, None)
        v.blob_id = None
        if v.state in (VoiceState.HYBRID, VoiceState.BLOB_ONLY):
            v.state = VoiceState.KEYBOARD_ONLY
            # Drop blob-driven routings on detach so the player isn't
            # surprised by silent CC-on-zero values; keep the synth state.
            v.routing = []
            v.events  = []

    def zombify(self, slot_id: int) -> None:
        """BLOB_ONLY (or HYBRID without recent play) -> ZOMBIE. SC node stays
        alive but amp is scaled down to a soft sustain. resurrect or
        timeout follow."""
        v = self.voice(slot_id)
        if v is None or v.state == VoiceState.ZOMBIE:
            return
        if v.blob_id is not None:
            self.voices_by_blob.pop(v.blob_id, None)
        v.state           = VoiceState.ZOMBIE
        v.zombie_started  = time.time()
        # Freeze a soft-sustain amp target; smoothing pulls c_amp down
        v.synth.t_amp = max(v.synth.t_amp * self.zombie_sustain_amp_scale,
                            v.synth.base_amp * self.zombie_sustain_amp_scale)

    def try_resurrect(self, blob) -> bool:
        """Attempt to bind a new blob to an existing zombie voice. The match
        criterion is centroid distance + persistence ratio."""
        # Match radius scales with lamp height, with a floor
        lamp_h     = max(self.profile.lamp_y2 - self.profile.lamp_y1, 100)
        match_r    = max(lamp_h * self.zombie_match_radius_frac,
                         self.zombie_min_match_radius)

        best_slot  = None
        best_dist  = float("inf")

        for v in self.voices_by_slot:
            if v.state != VoiceState.ZOMBIE:
                continue
            # Position check (we use the smoothed c_freq -> implied y)
            dx = (blob.centroid[0]) - (
                self.profile.x_norm(blob.centroid[0]) * 1.0   # placeholder
            )
            # Actually compare current zombie's last known position by
            # storing it on the voice; we keep a compact (zx, zy) pair.
            zx = getattr(v, "_zx", None)
            zy = getattr(v, "_zy", None)
            if zx is None or zy is None:
                continue
            d = math.hypot(blob.centroid[0] - zx, blob.centroid[1] - zy)
            if d > match_r:
                continue
            # Persistence ratio sanity gate
            zp = getattr(v, "_zp", 1.0)
            ratio = max(blob.persistence, 1e-6) / max(zp, 1e-6)
            if ratio > self.zombie_persistence_ratio or \
               ratio < (1.0 / self.zombie_persistence_ratio):
                continue
            if d < best_dist:
                best_dist  = d
                best_slot  = v.slot_id

        if best_slot is None:
            return False

        v = self.voices_by_slot[best_slot]
        v.state    = VoiceState.BLOB_ONLY
        v.blob_id  = blob.id
        v.last_seen_at = time.time()
        # Preserve the current synth_kind during resurrection. If this zombie
        # was already in bass mode, the next frame will either keep it there
        # or switch it back to "mpe" based on the new blob's gesture state.
        v.routing  = list(DEFAULT_BLOB_ROUTING)
        v.events   = list(DEFAULT_BLOB_EVENTS)
        v.last_direction = blob.direction
        self.voices_by_blob[blob.id] = best_slot
        # Don't re-fire blobStrike — the voice is recovered, not born.
        return True

    def _expire_zombies(self) -> None:
        now = time.time()
        for v in self.voices_by_slot:
            if v.state == VoiceState.ZOMBIE and \
               (now - v.zombie_started) > self.zombie_window:
                self._release(v.slot_id)

    def _clear_bass_state(self, v: Voice) -> None:
        v.synth_kind       = "mpe"
        v.bass_started     = 0.0
        v.bass_exit_streak = 0

    def _release(self, slot_id: int) -> None:
        """Return a slot to IDLE. Sends gate=0 to SC and clears state."""
        v = self.voice(slot_id)
        if v is None:
            return
        if v.sc_node is not None:
            try:
                self.osc.send_message("/n_set", [v.sc_node, "gate", 0])
            except Exception:
                pass
        if v.angle_node is not None:
            try:
                self.osc.send_message("/n_set", [v.angle_node, "gate", 0])
            except Exception:
                pass
        if v.blob_id is not None:
            self.voices_by_blob.pop(v.blob_id, None)
        self._pending_merge_release.pop(slot_id, None)
        # Reset the slot
        self.voices_by_slot[slot_id] = Voice(slot_id=slot_id)

    def release_all(self) -> None:
        for i in range(self.n_slots):
            self._release(i)

    # ── Player API (called via dispatch from controller events) ──────────
    def select_slot(self, slot_id: int) -> None:
        if 0 <= slot_id < self.n_slots:
            self.selected_slot = slot_id

    def note_on(self, slot_id: int, note: int, velocity: int) -> None:
        v = self.voice(slot_id)
        if v is None:
            return
        # v4.4: an IDLE slot used to swallow the note entirely, so the MPE
        # keyboard was silent unless a blob happened to be bound to the
        # selected slot -- the whole player path existed but was unreachable
        # in the common case. Pressing a key on an idle slot now SPAWNS a
        # KEYBOARD_ONLY voice there, which is what the state diagram at the
        # top of this file always described (IDLE -> KEYBOARD_ONLY).
        if v.state == VoiceState.IDLE:
            v.synth         = SynthState.default()
            v.routing       = []
            v.events        = []
            v.edited_params = set()
            v.state         = VoiceState.KEYBOARD_ONLY
            v.blob_id       = None
            v.synth_kind    = "mpe"
            v.born_at       = time.time()
            v.spawn_origin  = "keyboard"
            v.synth.base_note = float(note)
            v.synth.gate    = 1
            self._send_spawn(v, blob=None, fire_strike=False, gate=1)
        # Player note input should sound through the normal playable MPE voice.
        # A HYBRID/BLOB voice may later re-enter bass mode if the blob still
        # satisfies the gesture condition.
        if v.synth_kind == "bass":
            self._switch_to_mpe(v, None)
        v.synth.base_note  = float(note)
        v.synth.base_amp   = max(0.05, min(velocity / 127.0, 0.6))
        v.synth.gate       = 1
        v.last_played_at   = time.time()
        # Re-gate the SC node if it was previously released
        if v.sc_node is not None:
            self.osc.send_message("/n_set",
                [v.sc_node, "gate", 1, "amp", v.synth.base_amp])

    def note_off(self, slot_id: int, note: int) -> None:
        v = self.voice(slot_id)
        if v is None:
            return
        # Only KEYBOARD_ONLY voices fully release on note_off; HYBRID and
        # BLOB_ONLY voices keep their gate open since the blob is still
        # holding them.
        if v.state == VoiceState.KEYBOARD_ONLY:
            if v.sc_node is not None:
                self.osc.send_message("/n_set", [v.sc_node, "gate", 0])
            # Schedule a re-spawn on next note_on by retaining state but
            # leaving sc_node so the SC envelope completes
            v.synth.gate = 0

    def set_param(self, slot_id: int, param: str, value: Optional[float],
                  *, relative: bool = False, reset: bool = False) -> None:
        """
        Edit a synth parameter on the slot. The 'param' string is one of
        the keys listed in EDIT_FAMILIES (in controller.py), e.g.
        'base_note', 'cutoff', 'attack', 'wave_pos'.

        relative=True   -> value is a delta added to the current base
        reset=True      -> snap the parameter back to its default
        otherwise       -> value is the new absolute base
        """
        v = self.voice(slot_id)
        if v is None or v.state == VoiceState.IDLE:
            return
        s = v.synth

        # Map UI-facing param names to SynthState attributes
        attr_map = {
            "base_note":  ("base_note",   None),
            "bend":       ("bend",        None),
            "freq_hz":    ("freq_offset_hz", None),   # linear Hz nudge; the
                                                      # final freq is floored
                                                      # at 20 Hz in
                                                      # _compute_targets
            "scale_id":   ("scale_id",    None),
            "chord_id":   ("chord_id",    None),
            "wave_pos":   ("base_wave_pos", (0.0, 1.0)),
            "wave_bank":  ("wave_bank",   (0,    3)),
            "attack":     ("base_attack", (0.005, 8.0)),
            "decay":      ("base_decay",  (0.005, 4.0)),
            "sustain":    ("base_sustain", (0.0, 1.0)),
            "release":    ("base_release", (0.005, 12.0)),
            "cutoff":     ("base_cutoff",  (40.0, 16000.0)),
            "res":        ("base_res",     (0.05, 1.0)),
            "fmode":      ("fmode",        (0.0,  1.0)),
            "amp":        ("base_amp",     (0.0,  1.0)),
            "pan":        ("base_pan",    (-1.0,  1.0)),
        }
        if param not in attr_map:
            return
        attr, clip_range = attr_map[param]

        if reset:
            default = SynthState.default()
            setattr(s, attr, getattr(default, attr))
            # Reset relinquishes the edit: observation-driven presets
            # (e.g. direction_change -> attack/release) may act again.
            v.edited_params.discard(param)
        elif relative and value is not None:
            cur = getattr(s, attr)
            new = cur + value
            if clip_range:
                lo, hi = clip_range
                new = max(lo, min(hi, new))
            setattr(s, attr, new)
            v.edited_params.add(param)
        elif value is not None:
            new = value
            if clip_range:
                lo, hi = clip_range
                new = max(lo, min(hi, new))
            setattr(s, attr, new)
            v.edited_params.add(param)

    def add_routing(self, slot_id: int, src: Source, dest: Dest,
                    depth: float, curve: Curve = Curve.LIN) -> None:
        v = self.voice(slot_id)
        if v is None or v.state == VoiceState.IDLE:
            return
        # Overwrite any existing routing with same (src, dest)
        v.routing = [r for r in v.routing if not (r.src == src and r.dest == dest)]
        v.routing.append(Routing(src=src, dest=dest, depth=depth, curve=curve))

    def remove_routing(self, slot_id: int,
                       src: Optional[Source] = None,
                       dest: Optional[Dest]   = None,
                       *, all: bool = False) -> None:
        v = self.voice(slot_id)
        if v is None:
            return
        if all or (src is None and dest is None):
            v.routing = []
            return
        v.routing = [r for r in v.routing
                     if not (r.src == src and r.dest == dest)]

    def arm_learn(self, slot_id: int, param: str) -> None:
        """Mark (slot, param) as the next-incoming-MIDI-message target."""
        self.learn_target = (slot_id, param)

    def clear_learn(self) -> None:
        self.learn_target = None

    # ── Controller-event dispatch ────────────────────────────────────────
    def dispatch(self, ev) -> None:
        """Handle a single ControllerEvent. The main loop is expected to
        call this in order on every poll() result."""
        from controller import (
            NOTE_ON, NOTE_OFF, SELECT_SLOT, SET_PARAM,
            ADD_ROUTING, REMOVE_ROUTING, LEARN_ARM, LEARN_CLEAR,
        )
        t = ev.type
        if t == SELECT_SLOT and ev.slot is not None:
            self.select_slot(ev.slot)
        elif t == NOTE_ON and ev.slot is not None and ev.note is not None:
            self.note_on(ev.slot, ev.note, ev.velocity or 100)
        elif t == NOTE_OFF and ev.slot is not None and ev.note is not None:
            self.note_off(ev.slot, ev.note)
        elif t == SET_PARAM and ev.slot is not None and ev.param is not None:
            relative = bool(ev.extras.get("relative", False))
            reset    = bool(ev.extras.get("reset",    False))
            self.set_param(ev.slot, ev.param, ev.value,
                           relative=relative, reset=reset)
        elif t == ADD_ROUTING and ev.slot is not None:
            src   = Source(ev.extras["src"])
            dest  = Dest(ev.extras["dest"])
            depth = float(ev.extras.get("depth", 1.0))
            curve = Curve(ev.extras.get("curve", "LIN"))
            self.add_routing(ev.slot, src, dest, depth, curve)
        elif t == REMOVE_ROUTING and ev.slot is not None:
            if ev.extras.get("all"):
                self.remove_routing(ev.slot, all=True)
            else:
                src  = Source(ev.extras["src"])  if "src"  in ev.extras else None
                dest = Dest(ev.extras["dest"])   if "dest" in ev.extras else None
                self.remove_routing(ev.slot, src, dest)
        elif t == LEARN_ARM and ev.slot is not None and ev.param is not None:
            self.arm_learn(ev.slot, ev.param)
        elif t == LEARN_CLEAR:
            self.clear_learn()
        # Other events (CREATE_EMPTY, TIE_BLOB, DETACH, CLAIM_HYBRID,
        # MODE_TOGGLE, OCTAVE_*) are handled by the main loop directly,
        # since they need access to the BindingManager and tracker state.

    # ── Per-frame processing ─────────────────────────────────────────────
    def process_frame(self, tracked_blobs: list) -> None:
        """
        Per-frame work for every non-IDLE slot:
            1. Stamp last-known position on bound voices (for zombie matching)
            2. Detect direction-change events
            3. Switch eligible blob-bound voices into/out of bass-stutter mode
            3b. Orphan guard: bound voices whose blob has vanished (v4.9.1)
            4. Compute target = base + sum(routings)
            5. Smooth toward target, send /n_set
            6. Expire zombies past their window
            7. Release any merge fade-outs that have decayed below threshold

        Lifecycle events (birth / death / merge / split) are NOT handled
        here — they flow through the BindingManager. Call:
            binding.handle_tracker_events(tracker.events, tracked_blobs)
        BEFORE calling this method, on every frame.
        """
        by_id = {b.id: b for b in tracked_blobs}
        now   = time.time()

        # 1. Refresh zombie-match anchor (last-known blob position)
        for v in self.voices_by_slot:
            if v.state in (VoiceState.BLOB_ONLY, VoiceState.HYBRID) and \
               v.blob_id in by_id:
                b = by_id[v.blob_id]
                v._zx = float(b.centroid[0])
                v._zy = float(b.centroid[1])
                v._zp = float(b.persistence)
                v.last_seen_at = now

        # 2. Direction-change events
        for v in self.voices_by_slot:
            if v.state in (VoiceState.BLOB_ONLY, VoiceState.HYBRID) and \
               v.blob_id in by_id:
                b = by_id[v.blob_id]
                if b.direction != v.last_direction:
                    self._handle_direction_change(v, b)
                    v.last_direction = b.direction

        # 3. Switch existing blob-bound voices into/out of \bassStutter mode.
        self._update_bass_stutter_modes(by_id)

        # 3b. Orphan guard (v4.9.1). A bound voice whose blob has not been in
        # the tracked set for orphan_timeout_secs gets the same treatment a
        # death event would give it. Coasting blobs are still tracked, so a
        # blob between detections never trips this.
        self._reap_orphans(by_id, now)

        # 4 & 5. Per-slot target compute + smooth + send
        s_alpha = self.smoothing
        for v in self.voices_by_slot:
            if v.state == VoiceState.IDLE:
                continue
            self._compute_targets(v, by_id)
            self._smooth_and_send(v, s_alpha, by_id)

        # 6. Zombie expiry
        self._expire_zombies()

        # 7. Merge fade-out releases
        self._check_merge_releases()

    def _reap_orphans(self, by_id: dict, now: float) -> None:
        if self.orphan_timeout_secs <= 0.0:
            return
        for v in self.voices_by_slot:
            if v.state not in (VoiceState.BLOB_ONLY, VoiceState.HYBRID):
                continue
            if v.blob_id is None:
                # A BLOB_ONLY voice with no blob at all: only the merge path
                # produces these, and it is handled by _check_merge_releases.
                # Anything else here is a leak -> treat as dead.
                if v.slot_id not in self._pending_merge_release:
                    self.zombify(v.slot_id)
                continue
            if v.blob_id in by_id:
                continue
            if v.last_seen_at <= 0.0:
                v.last_seen_at = now        # never seen yet: start the clock
                continue
            if (now - v.last_seen_at) < self.orphan_timeout_secs:
                continue
            if v.state == VoiceState.HYBRID and \
               (now - v.last_played_at) < self.blob_recent_play_window:
                self.detach(v.slot_id)
            else:
                self.zombify(v.slot_id)

    # ── Bass-stutter mode switching ─────────────────────────────────────
    def _bass_start_condition(self, blob) -> bool:
        """
        Start bass-stutter mode when an existing blob-bound voice is:
          - low in the lamp (small y_norm),
          - moving downward (vy > 0 in image coordinates), and
          - expanding (d_area > 0).
        """
        y_norm = self.profile.y_norm(blob.centroid[1])  # 0=bottom, 1=top
        vy = float(getattr(blob, "vy", 0.0))
        d_area = float(getattr(blob, "d_area", 0.0))
        age = int(getattr(blob, "age", 0))

        return (
            age >= self.bass_min_age
            and y_norm <= self.bass_y_on
            and vy >= self.bass_vy_on
            and d_area >= self.bass_darea_on
        )

    def _bass_continue_condition(self, blob) -> bool:
        """Slightly looser than the entry test so bass mode does not flicker --
        but strictly narrower than 'anything at all', which is what the old
        thresholds amounted to."""
        y_norm = self.profile.y_norm(blob.centroid[1])
        vy = float(getattr(blob, "vy", 0.0))
        d_area = float(getattr(blob, "d_area", 0.0))

        return (
            y_norm <= self.bass_y_off
            and vy >= self.bass_vy_off
            and d_area >= self.bass_darea_off
        )

    def bass_voice_count(self) -> int:
        """Slots currently rendering as \\bassStutter."""
        return sum(1 for v in self.voices_by_slot if v.synth_kind == "bass")

    def _update_bass_stutter_modes(self, by_id: dict) -> None:
        """Turn eligible existing BLOB_ONLY/HYBRID voices into bass stutters.

        Exit takes priority over entry so a freed slot can be reused in the
        same frame, and so the voice cap is evaluated against a current count.
        """
        now = time.time()

        # ── Exits first (debounced + time-capped) ────────────────────────
        for v in self.voices_by_slot:
            if v.synth_kind != "bass":
                continue
            blob = by_id.get(v.blob_id) if v.blob_id is not None else None

            # Blob gone entirely: leave bass immediately.
            if blob is None or v.state not in (VoiceState.BLOB_ONLY, VoiceState.HYBRID):
                self._switch_to_mpe(v, blob)
                continue

            # Hard ceiling on episode length, independent of the conditions.
            if v.bass_started > 0.0 and (now - v.bass_started) >= self.bass_max_secs:
                self._switch_to_mpe(v, blob)
                continue

            if self._bass_continue_condition(blob):
                v.bass_exit_streak = 0
            else:
                v.bass_exit_streak += 1
                if v.bass_exit_streak >= self.bass_exit_frames:
                    self._switch_to_mpe(v, blob)

        # ── Entries, subject to the global cap and a per-slot cooldown ───
        n_bass = self.bass_voice_count()
        for v in self.voices_by_slot:
            if n_bass >= self.max_bass_voices:
                break
            if v.state not in (VoiceState.BLOB_ONLY, VoiceState.HYBRID):
                continue
            if v.synth_kind == "bass":
                continue
            if v.blob_id is None or v.blob_id not in by_id:
                continue
            # Cooldown: a slot that just left bass must wait before re-entering,
            # otherwise a blob hovering at the threshold respawns nodes forever.
            if v.bass_started > 0.0 and (now - v.bass_started) < self.bass_retrigger_secs:
                continue
            if self._bass_start_condition(by_id[v.blob_id]):
                self._switch_to_bass_stutter(v, by_id[v.blob_id])
                n_bass += 1

    def _gate_current_nodes(self, v: Voice) -> None:
        """Release the slot's current SC nodes without clearing slot identity."""
        if v.sc_node is not None:
            try:
                self.osc.send_message("/n_set", [v.sc_node, "gate", 0])
            except Exception:
                pass
        if v.angle_node is not None:
            try:
                self.osc.send_message("/n_set", [v.angle_node, "gate", 0])
            except Exception:
                pass
        v.sc_node = None
        v.angle_node = None

    def _switch_to_bass_stutter(self, v: Voice, blob) -> None:
        r"""Replace this slot's normal \mpeVoice node with \bassStutter."""
        self._gate_current_nodes(v)
        v.synth_kind       = "bass"
        v.bass_started     = time.time()
        v.bass_exit_streak = 0
        self._send_bass_spawn(v, blob)

    def _switch_to_mpe(self, v: Voice, blob=None) -> None:
        r"""Return this slot from \bassStutter to the normal \mpeVoice.

        `bass_started` is deliberately NOT cleared: it doubles as the start of
        the re-trigger cooldown, so a blob oscillating around the threshold
        cannot respawn nodes every few frames.
        """
        self._gate_current_nodes(v)
        v.synth_kind       = "mpe"
        v.bass_exit_streak = 0
        # Re-enter normal voice without firing a birth strike.
        self._send_spawn(v, blob=blob, fire_strike=False, gate=1)

    def _bass_params_for_blob(self, blob) -> dict:
        r"""Map blob motion/size change to \bassStutter parameters."""
        y_norm = self.profile.y_norm(blob.centroid[1])  # 0=bottom, 1=top
        x_norm = self.profile.x_norm(blob.centroid[0])
        vy = float(getattr(blob, "vy", 0.0))
        d_area = float(getattr(blob, "d_area", 0.0))

        down = max(0.0, min(vy / 3.0, 1.0))
        grow = max(0.0, min(d_area / 0.5, 1.0))

        return {
            # Lower blob -> lower bass; still track y slightly for motion.
            "freq": 36.0 + 24.0 * y_norm,
            # Growing blob -> more weight and faster chopping.
            "amp": 0.16 + 0.26 * grow,
            "stutterRate": 8.0 + 18.0 * grow,
            # Stronger downward motion -> deeper filter wobble.
            "lfoDepth": 180.0 + 420.0 * down,
            # Keep the same spatial image as the blob.
            "pan": x_norm * 2.0 - 1.0,
        }

    def _send_bass_spawn(self, v: Voice, blob) -> None:
        nid = self.next_node_id
        self.next_node_id += 1
        v.sc_node = nid
        v.angle_node = None

        p = self._bass_params_for_blob(blob)
        # Keep SynthState's smoothed values close to the bass node so returning
        # to \mpeVoice is less discontinuous.
        v.synth.c_freq = p["freq"]
        v.synth.c_amp = p["amp"]
        v.synth.c_pan = p["pan"]
        v.synth.t_freq = p["freq"]
        v.synth.t_amp = p["amp"]
        v.synth.t_pan = p["pan"]

        try:
            self.osc.send_message("/s_new", [
                self.bass_synthdef, nid, 0, 0,
                "freq", p["freq"],
                "amp", p["amp"],
                "pan", p["pan"],
                "gate", 1,
                "stutterRate", p["stutterRate"],
                "lfoDepth", p["lfoDepth"],
            ])
            print(f"  ~ slot {v.slot_id:02d} blob #{blob.id} -> bassStutter")
        except Exception:
            pass

    def _send_bass_update(self, v: Voice, blob) -> None:
        if v.sc_node is None:
            return

        p = self._bass_params_for_blob(blob)
        v.synth.c_freq = p["freq"]
        v.synth.c_amp = p["amp"]
        v.synth.c_pan = p["pan"]
        v.synth.t_freq = p["freq"]
        v.synth.t_amp = p["amp"]
        v.synth.t_pan = p["pan"]

        try:
            self.osc.send_message("/n_set", [
                v.sc_node,
                "freq", p["freq"],
                "amp", p["amp"],
                "pan", p["pan"],
                "stutterRate", p["stutterRate"],
                "lfoDepth", p["lfoDepth"],
            ])
        except Exception:
            pass

    def _compute_targets(self, v: Voice, by_id: dict) -> None:
        """Set t_* values from base + sum of (depth * curve(source))."""
        s = v.synth

        # Start from bases. Pitch is assembled in semitone space first
        # (base_note + bend + summed PITCH routings), converted to Hz, and
        # only THEN offset by the EDIT-mode linear-Hz nudge. Ordering is the
        # contract that makes edits redefine the voice: every frame the blob's
        # routed sources (elongation, speed, ...) re-modulate around whatever
        # the player last dialled in, rather than the edit being a one-shot
        # value the next frame overwrites.
        pitch_semis = 0.0
        t_amp     = s.base_amp     if v.state != VoiceState.ZOMBIE else \
                    s.base_amp * self.zombie_sustain_amp_scale
        t_pan     = s.base_pan
        t_cutoff  = s.base_cutoff
        t_res     = s.base_res
        t_wave_pos= s.base_wave_pos
        t_attack  = s.base_attack
        t_release = s.base_release
        t_detune  = s.base_detune
        t_shimmer = s.base_shimmer
        t_noise   = s.base_noise
        t_drive   = s.base_drive
        t_fmode   = s.fmode

        # Routing contributions only when bound to a live blob
        if v.state in (VoiceState.BLOB_ONLY, VoiceState.HYBRID) and \
           v.blob_id is not None and v.blob_id in by_id:
            blob = by_id[v.blob_id]
            for r in v.routing:
                src_val = extract_source_value(r.src, blob, self.profile)
                shaped  = apply_curve(r.curve, src_val) * r.depth
                if r.dest == Dest.PITCH:
                    # Accumulate: multiple PITCH routings now sum instead of
                    # last-one-wins (previously each overwrote t_freq).
                    pitch_semis += shaped
                elif r.dest == Dest.AMP:
                    t_amp     = max(0.0, t_amp + shaped)
                elif r.dest == Dest.PAN:
                    t_pan     = max(-1.0, min(1.0, t_pan + shaped))
                elif r.dest == Dest.CUTOFF:
                    t_cutoff  = max(40.0, min(16000.0, t_cutoff + shaped))
                elif r.dest == Dest.RES:
                    t_res     = max(0.05, min(1.0, t_res + shaped))
                elif r.dest == Dest.WAVE_POS:
                    t_wave_pos= max(0.0, min(1.0, t_wave_pos + shaped))
                elif r.dest == Dest.ATTACK:
                    t_attack  = max(0.005, t_attack + shaped)
                elif r.dest == Dest.RELEASE:
                    t_release = max(0.005, t_release + shaped)
                elif r.dest == Dest.DETUNE:
                    t_detune  = max(0.0, min(1.0, t_detune  + shaped))
                elif r.dest == Dest.SHIMMER:
                    t_shimmer = max(0.0, min(1.0, t_shimmer + shaped))
                elif r.dest == Dest.NOISE:
                    t_noise   = max(0.0, min(1.0, t_noise   + shaped))
                elif r.dest == Dest.DRIVE:
                    t_drive   = max(0.0, min(1.0, t_drive   + shaped))
                elif r.dest == Dest.FMODE:
                    t_fmode   = max(0.0, min(1.0, t_fmode   + shaped))

        # Final pitch: musical-space assembly -> Hz -> linear EDIT-mode
        # offset -> positive floor (20 Hz) so a deep negative nudge can
        # never fold the oscillator through 0 Hz.
        t_freq = midi_to_freq(s.base_note + s.bend / 100.0 + pitch_semis) \
                 + s.freq_offset_hz
        t_freq = max(20.0, t_freq)

        # ── v4.2 score-grammar layer: one more summed/multiplied layer before
        # smoothing. Everything is scaled by score_gain and, per voice, by the
        # blob's governance/prominence, so it fades with the softmax weights
        # rather than switching. Nothing here persists across frames.
        ss = self.score_state
        if ss is not None and self.score_gain > 0.0 and \
           v.state in (VoiceState.BLOB_ONLY, VoiceState.HYBRID) and v.blob_id is not None:
            g    = self.score_gain
            rank = ss.prominence_rank.get(v.blob_id, 0.5)
            # (i) harmonic gravitation toward the current pitch field; heads
            #     keep licensed dissonance (weaker pull), ornaments snap harder
            pull = ss.gravitation * (1.0 - 0.6 * rank) * g
            if ss.pitch_field and pull > 0.0:
                f_field = _nearest_field_freq(t_freq, ss.pitch_field)
                t_freq  = t_freq + pull * (f_field - t_freq)
            # (ii) time-span mixer: foreground/background by prominence rank
            t_amp   = t_amp * (1.0 - g * 0.5 * (1.0 - rank))
            # (iii) prolongational roughness: detune / noise track tension
            t_detune  = max(0.0, min(1.0, t_detune  + g * 0.35 * ss.roughness_target))
            t_noise   = max(0.0, min(1.0, t_noise   + g * 0.20 * ss.tension))
            t_drive   = max(0.0, min(1.0, t_drive   + g * 0.15 * ss.surprisal / 6.0))
            # (iv) boundary gesture: envelope reset (short attack) on strong boundaries
            if ss.boundary and ss.boundary_strength > 0.6:
                t_attack = min(t_attack, 0.05 + 0.4 * (1.0 - ss.boundary_strength))
            # report the sounding pitch back for the time-span "surface" rule
            blob = by_id.get(v.blob_id)
            if blob is not None:
                blob.score_freq_hz = t_freq

        # ── v4.9.1 merge fade-out. handle_merge() queues the slot and sets
        # t_amp = 0 once; this keeps it at 0 on every subsequent frame so the
        # smoothing in _smooth_and_send can actually reach the release
        # threshold. Without this line the base_amp assignment at the top of
        # this method silently re-opened the voice every frame.
        if v.slot_id in self._pending_merge_release:
            t_amp = 0.0

        # Write back to synth target fields
        s.t_freq     = t_freq
        s.t_amp      = t_amp
        s.t_pan      = t_pan
        s.t_cutoff   = t_cutoff
        s.t_res      = t_res
        s.t_wave_pos = t_wave_pos
        s.t_attack   = t_attack
        s.t_release  = t_release
        s.t_detune   = t_detune
        s.t_shimmer  = t_shimmer
        s.t_noise    = t_noise
        s.t_drive    = t_drive
        s.t_fmode    = t_fmode

    def _smooth_and_send(self, v: Voice, alpha: float, by_id: dict | None = None) -> None:
        if v.sc_node is None:
            return
        s = v.synth
        s.c_freq    += alpha * (s.t_freq    - s.c_freq)
        s.c_amp     += alpha * (s.t_amp     - s.c_amp)
        s.c_pan     += alpha * (s.t_pan     - s.c_pan)
        s.c_cutoff  += alpha * (s.t_cutoff  - s.c_cutoff)
        s.c_res     += alpha * (s.t_res     - s.c_res)
        s.c_wave_pos+= alpha * (s.t_wave_pos- s.c_wave_pos)
        s.c_attack  += alpha * (s.t_attack  - s.c_attack)
        s.c_release += alpha * (s.t_release - s.c_release)
        s.c_detune  += alpha * (s.t_detune  - s.c_detune)
        s.c_shimmer += alpha * (s.t_shimmer - s.c_shimmer)
        s.c_noise   += alpha * (s.t_noise   - s.c_noise)
        s.c_drive   += alpha * (s.t_drive   - s.c_drive)
        s.c_fmode   += alpha * (s.t_fmode   - s.c_fmode)

        # In bass mode, the same voice slot is represented by \bassStutter
        # instead of \mpeVoice.  It only accepts bass-specific params, so do
        # not send wavetable/filter-only args to it.
        if v.synth_kind == "bass":
            blob = None
            if by_id is not None and v.blob_id is not None:
                blob = by_id.get(v.blob_id)
            if blob is not None:
                self._send_bass_update(v, blob)
            else:
                try:
                    self.osc.send_message("/n_set", [
                        v.sc_node,
                        "freq", s.c_freq,
                        "amp",  s.c_amp,
                        "pan",  s.c_pan,
                    ])
                except Exception:
                    pass
            return

        try:
            self.osc.send_message("/n_set", [
                v.sc_node,
                "freq",     s.c_freq,
                "amp",      s.c_amp,
                "pan",      s.c_pan,
                "cutoff",   s.c_cutoff,
                "res",      s.c_res,
                "wave",     s.c_wave_pos,
                "bank",     s.wave_bank,
                "attack",   s.c_attack,
                "decay",    s.base_decay,
                "sustain",  s.base_sustain,
                "release",  s.c_release,
                "fmode",    s.c_fmode,
                "detune",   s.c_detune,
                "shimmer",  s.c_shimmer,
                "noise",    s.c_noise,
                "drive",    s.c_drive,
            ])
        except Exception:
            pass

    # ── Discrete event handler (direction change) ────────────────────────
    def _handle_direction_change(self, v: Voice, blob) -> None:
        for ev in v.events:
            if ev.src == "direction_change" and ev.handler == "attack_release_preset":
                # EDIT-mode changes redefine the voice: if the player has set
                # attack/release, the preset no longer overwrites those bases.
                # (Pressing R on the param clears the flag and re-enables it.)
                if "attack" not in v.edited_params:
                    v.synth.base_attack  = direction_to_attack(blob.direction)
                if "release" not in v.edited_params:
                    v.synth.base_release = direction_to_release(blob.direction)

    # ── Merge / split (preserved from v3.x, refactored for slot pool) ────
    def handle_merge(self, merged_id: int, survivor_id: int,
                     tracked_blobs: list) -> None:
        merged_slot   = self.voices_by_blob.get(merged_id)
        survivor_slot = self.voices_by_blob.get(survivor_id)
        if merged_slot is None:
            return
        v_m = self.voices_by_slot[merged_slot]
        # Glide merged's pitch to survivor's, fade amp -> 0, then release
        if survivor_slot is not None:
            v_s = self.voices_by_slot[survivor_slot]
            v_m.synth.t_freq = v_s.synth.c_freq
            v_m.synth.t_pan  = v_s.synth.c_pan
        v_m.synth.t_amp = 0.0
        # Faster glide on freq so it reaches the survivor in ~1s
        v_m.synth.c_freq += 0.25 * (v_m.synth.t_freq - v_m.synth.c_freq)
        # Mark for release once amp fades below threshold (or on timeout --
        # see _check_merge_releases / _compute_targets, v4.9.1)
        self._pending_merge_release[merged_slot] = time.time()
        # Drop the blob_id binding now so a future birth doesn't try to
        # re-use this dying voice
        self.voices_by_blob.pop(merged_id, None)
        v_m.blob_id = None

    def handle_split(self, parent_id: int, child_id: int,
                     tracked_blobs: list) -> None:
        parent_slot = self.voices_by_blob.get(parent_id)
        child_blob  = next((b for b in tracked_blobs if b.id == child_id), None)
        if child_blob is None:
            return
        if child_id in self.voices_by_blob:
            return
        # Allocate next IDLE slot for child
        child_slot = self.first_idle_slot()
        if child_slot is None:
            return
        v_c = self.voices_by_slot[child_slot]
        # Inherit parent's smoothed state if possible
        if parent_slot is not None:
            parent = self.voices_by_slot[parent_slot]
            v_c.synth = SynthState(**vars(parent.synth))
        else:
            self._init_voice_from_blob(v_c, child_blob, spawn_origin="blob")
        v_c.state    = VoiceState.BLOB_ONLY
        v_c.blob_id  = child_id
        v_c.synth_kind = "mpe"
        v_c.last_seen_at = time.time()
        v_c.routing  = list(DEFAULT_BLOB_ROUTING)
        v_c.events   = list(DEFAULT_BLOB_EVENTS)
        v_c.last_direction = child_blob.direction
        self.voices_by_blob[child_id] = child_slot
        # Spawn SC node WITHOUT firing blobStrike (silent emergence)
        self._send_spawn(v_c, child_blob, fire_strike=False)

    def _check_merge_releases(self) -> None:
        now = time.time()
        for slot_id, queued_at in list(self._pending_merge_release.items()):
            v = self.voices_by_slot[slot_id]
            if v.state == VoiceState.IDLE:
                self._pending_merge_release.pop(slot_id, None)
                continue
            faded    = v.synth.c_amp < 0.005
            timed_out = (now - queued_at) >= self.merge_release_max_secs
            if faded or timed_out:
                self._release(slot_id)
                self._pending_merge_release.pop(slot_id, None)

    # ── Internal: initialise a Voice's synth state from a blob ───────────
    def _init_voice_from_blob(self, v: Voice, blob, *, spawn_origin: str = "blob"
                              ) -> None:
        p      = self.profile
        y_norm = 1.0 - p.y_norm(blob.centroid[1])    # 1 = top of lamp
        x_norm = p.x_norm(blob.centroid[0])
        midi   = y_to_midi(y_norm)
        v.synth = SynthState.default()
        v.synth_kind = "mpe"
        v.synth.base_note   = float(midi)
        v.synth.base_pan    = x_to_pan(x_norm)
        v.synth.base_cutoff = max(200.0, min(6000.0,
                                  200.0 * ((6000.0 / 200.0) **
                                           (blob.birth / 255.0))))
        v.synth.base_amp    = 0.05 + 0.30 * math.sqrt(min(blob.persistence /
                                                          80.0, 1.0))
        v.synth.base_attack = direction_to_attack(blob.direction)
        v.synth.base_release= direction_to_release(blob.direction)
        v.routing       = list(DEFAULT_BLOB_ROUTING)
        v.events        = list(DEFAULT_BLOB_EVENTS)
        v.edited_params = set()          # fresh spawn = fresh edit slate
        v.born_at       = time.time()
        v.spawn_origin  = spawn_origin
        v.last_direction= blob.direction
        # Initialise targets and freeze current
        v.synth.gate    = 1
        v.synth.t_freq  = midi_to_freq(v.synth.base_note)
        v.synth.t_amp   = v.synth.base_amp
        v.synth.t_pan   = v.synth.base_pan
        v.synth.t_cutoff= v.synth.base_cutoff
        v.synth.t_res   = v.synth.base_res
        v.synth.t_wave_pos = v.synth.base_wave_pos
        v.synth.t_attack= v.synth.base_attack
        v.synth.t_release= v.synth.base_release
        v.synth.freeze_current_to_target()
        # Stamp last-known position for zombie matching
        v._zx = float(blob.centroid[0])
        v._zy = float(blob.centroid[1])
        v._zp = float(blob.persistence)

    # ── Internal: send /s_new for a voice ────────────────────────────────
    def _send_spawn(self, v: Voice, blob=None, *, fire_strike: bool = True,
                    gate: int = 1) -> None:
        nid    = self.next_node_id;  self.next_node_id += 1
        ang_id = self.next_node_id;  self.next_node_id += 1
        v.sc_node    = nid
        v.angle_node = ang_id

        s = v.synth
        try:
            self.osc.send_message("/s_new", [
                self.mpe_synthdef, nid, 0, 0,
                "freq",    s.c_freq,
                "amp",     s.c_amp if gate else 0.0,
                "pan",     s.c_pan,
                "gate",    int(gate),
                "cutoff",  s.c_cutoff,
                "res",     s.c_res,
                "wave",    s.c_wave_pos,
                "bank",    s.wave_bank,
                "attack",  s.c_attack,
                "decay",   s.base_decay,
                "sustain", s.base_sustain,
                "release", s.c_release,
                "fmode",   s.c_fmode,
                "detune",  s.c_detune,
                "shimmer", s.c_shimmer,
                "noise",   s.c_noise,
                "drive",   s.c_drive,
            ])
            # The blobAngle pad is auxiliary; spawn only for blob-driven voices
            if blob is not None:
                self.osc.send_message("/s_new", [
                    self.angle_synthdef, ang_id, 0, 0,
                    "freq",        s.c_freq * 1.5,
                    "amp",         s.c_amp * 0.3,
                    "orientation", float(getattr(blob, "orientation", 0.0)),
                    "gate",        1,
                ])
            if fire_strike and blob is not None:
                strike_nid = self.next_node_id;  self.next_node_id += 1
                self.osc.send_message("/s_new", [
                    self.strike_synthdef, strike_nid, 0, 0,
                    "freq",       s.c_freq,
                    "amp",        s.c_amp * 1.2,
                    "decay",      max(0.5, 4.0 - blob.persistence * 0.05),
                    "blobSize",   min(blob.area / 5000.0, 1.0),
                    "position",   self.profile.x_norm(blob.centroid[0]),
                    "elongation", float(getattr(blob, "elongation", 0.0)),
                ])
        except Exception:
            pass

    # ── Diagnostics ──────────────────────────────────────────────────────
    def hud_summary(self) -> dict:
        counts = {st.name: 0 for st in VoiceState}
        for v in self.voices_by_slot:
            counts[v.state.name] += 1
        return counts
