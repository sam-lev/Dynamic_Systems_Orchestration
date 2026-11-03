# controller.py -- Audible Lava / DSO v4.0 (MPE Edition)
#
# Player-input abstraction layer. The VoiceManager and BindingManager only
# ever see ControllerEvent objects produced by a Controller.poll() call;
# they never know whether the events came from a computer keyboard, an
# external MIDI controller, an OSC tablet, or anything else. This is what
# makes the Phase 1 -> Phase 2 transition cheap: Phase 2 adds a single new
# Controller adapter without touching anything downstream of it.
#
# Phase 1 ships KeyboardController (pygame). Phase 2 will activate
# MIDIController (mido) and the LearnMode workflow on the same protocol.
#
# v4.9.3: the `gov` EDIT family gains gov.grouping.reset_learning (an
# ACTION: select it and press `r` -- or any arrow -- to restore the phrase
# prior live; the HUD value shows how many resets have been done), plus the
# two new duration guards gov.grouping.phrase_floor_secs and
# gov.grouping.hazard_cap_per_sec.

from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol, Iterable

# ── ControllerEvent ──────────────────────────────────────────────────────────
# The lingua franca between Controller adapters and the VoiceManager.
# Each event has a .type and a subset of fields filled in.
#
# Event types (string constants, not an Enum, so adapters from later layers
# can introduce new ones without coordinating):
#
#   Slot focus and lifecycle:
#     SELECT_SLOT       slot
#     CREATE_EMPTY      (no fields; VoiceManager picks next IDLE slot)
#     CLAIM_HYBRID      slot          # promote BLOB_ONLY -> HYBRID
#     TIE_BLOB          slot, blob_id # bind KEYBOARD_ONLY voice to blob
#     DETACH            slot          # break blob bind, voice -> KEYBOARD_ONLY
#
#   Performance:
#     NOTE_ON           slot, note, velocity
#     NOTE_OFF          slot, note
#     OCTAVE_UP / DOWN  (no fields; controller-internal)
#
#   Synth parameter editing:
#     SET_PARAM         slot, param, value          # value in param's natural units
#
#   Routing matrix editing:
#     ADD_ROUTING       slot, source, dest, depth, curve
#     REMOVE_ROUTING    slot, source, dest
#
#   MIDI Learn (Phase 2):
#     LEARN_ARM         slot, param   # next incoming CC binds to this
#     LEARN_CLEAR       slot, param
#
#   System:
#     MODE_TOGGLE       (no fields; PERFORMANCE <-> EDIT)
#     QUIT              (no fields)
#
# The 'value' field is overloaded: for SET_PARAM it carries the new param
# value; for SELECT_SLOT it carries the slot index again (redundant but
# convenient); for routing events the source/dest/depth/curve fields are
# carried in the keyword extras dict.

NOTE_ON       = "NOTE_ON"
NOTE_OFF      = "NOTE_OFF"
SELECT_SLOT   = "SELECT_SLOT"
CREATE_EMPTY  = "CREATE_EMPTY"
CLAIM_HYBRID  = "CLAIM_HYBRID"
TIE_BLOB      = "TIE_BLOB"
DETACH        = "DETACH"
SET_PARAM     = "SET_PARAM"
ADD_ROUTING   = "ADD_ROUTING"
REMOVE_ROUTING= "REMOVE_ROUTING"
LEARN_ARM     = "LEARN_ARM"
LEARN_CLEAR   = "LEARN_CLEAR"
OCTAVE_UP     = "OCTAVE_UP"
OCTAVE_DOWN   = "OCTAVE_DOWN"
MODE_TOGGLE   = "MODE_TOGGLE"
QUIT          = "QUIT"


@dataclass
class ControllerEvent:
    type:       str
    slot:       Optional[int]   = None
    note:       Optional[int]   = None
    velocity:   Optional[int]   = None
    param:      Optional[str]   = None
    value:      Optional[float] = None
    blob_id:    Optional[int]   = None
    extras:     dict            = field(default_factory=dict)


# ── Protocol (anything implementing this is a Controller) ────────────────────
class Controller(Protocol):
    def poll(self) -> list[ControllerEvent]: ...
    def close(self) -> None: ...


# ── KeyboardController (Phase 1) ─────────────────────────────────────────────
# pygame-based two-mode keyboard surface.
#
# PERFORMANCE mode (default):
#   1..0           select slot 0..9
#   q..\           select slots 10..15  (q=10 w=11 e=12 r=13 t=14 y=15)   --> see SLOT_KEYS_HIGH
#       (we use only 0..9 for slot select; slots 10..15 are reachable via
#        the "shift+1" combos handled later — keep the layout simple here)
#   a s d f g h j k l ;     C major octave (white keys, "home row")
#   w e   t y u    o p      black keys above the home row (sharps)
#   z x   v b n    , .      C major octave one below home row (white)
#   [ / ]                   octave down / up
#   space                   panic-stop all notes on selected slot
#   tab                     toggle EDIT mode
#
# EDIT mode (TAB toggles):
#   1..0           still select slot
#   p / w / e / f  parameter family: pitch / wave / envelope / filter
#   m              parameter family: mod (routing edit)
#   left/right     scroll selected param within family
#   up/down        finer adjustment of currently focused param
#   - / =          coarse adjust  (- decrease, = increase by 10x step)
#   b              CLAIM_HYBRID on selected slot   (BLOB_ONLY -> HYBRID)
#   n              CREATE_EMPTY                    (next IDLE -> KEYBOARD_ONLY)
#   t              TIE_BLOB on selected slot to nearest unbound blob
#   d              DETACH selected slot's blob bind
#   r              reset selected param to default base value
#   c              clear all routings on selected slot
#   tab            return to PERFORMANCE
#
# This module imports pygame lazily so that the rest of the project can be
# imported on machines without pygame installed (e.g. for unit tests).

PERFORMANCE_MODE = "PERFORMANCE"
EDIT_MODE        = "EDIT"

# Map from QWERTY home/upper rows to MIDI offset within the current octave.
# Layout is "piano-style" wrapped onto two rows; ';' = high B, "'" = high C.
_NOTE_LAYOUT_LOWER = {  # zsxdcvgbhnjm,l. -> chromatic from C
    'z':  0, 's':  1, 'x':  2, 'd':  3, 'c':  4, 'v':  5,
    'g':  6, 'b':  7, 'h':  8, 'n':  9, 'j': 10, 'm': 11,
    ',': 12, 'l': 13, '.': 14,
}
_NOTE_LAYOUT_UPPER = {  # qwerty row + 2/3/5/6/7 sharps -> next octave up
    'q':  0, '2':  1, 'w':  2, '3':  3, 'e':  4, 'r':  5,
    '5':  6, 't':  7, '6':  8, 'y':  9, '7': 10, 'u': 11,
    'i': 12, '9': 13, 'o': 14, '0': 15, 'p': 16,
}

# Slot-select keys (Performance mode only, by default):
_SLOT_KEYS = {str(i): (i % 10) for i in range(10)}   # '1'..'9' -> 1..9, '0' -> 0
# We override above so that keypad-friendly: '1' -> 0, '2' -> 1, ... '0' -> 9
_SLOT_KEYS = {str((i + 1) % 10): i for i in range(10)}

# In EDIT mode the number keys still SELECT_SLOT (so the player can cross-
# reference the panel HUD without leaving Edit mode).

# Default editable parameter families and the param cycle within each.
EDIT_FAMILIES: dict[str, list[str]] = {
    # freq_hz is the linear absolute-Hz pitch nudge (+/-5 Hz per up/down
    # press, +/-50 coarse). It maps to SynthState.freq_offset_hz, applied
    # AFTER blob routing so modulation rides on the nudged pitch. Contrast
    # with base_note (semitones) and bend (cents), which are musical-unit
    # increments — a deliberate log-vs-linear design distinction.
    "pitch":   ["base_note", "bend", "freq_hz", "scale_id", "chord_id"],
    "wave":    ["wave_pos", "wave_bank"],
    "env":     ["attack", "decay", "sustain", "release"],
    "filter":  ["cutoff", "res", "fmode"],
    "mod":     ["routing_cursor"],   # 'mod' family uses arrow keys to walk
                                     # the slot's routing list and edit depth
    # v4.2: score-governor weights. These are GLOBAL (not per-slot); the
    # param name is a dotted path into governor_weights.json and is routed
    # to ScoreGovernor.handle_controller_event, never to VoiceManager.
    # v4.9.3: reset_learning is an action (press `r` on it); phrase_floor_secs
    # and hazard_cap_per_sec are the new duration-model guards.
    "gov":     ["gov.grouping.prior_phrase_secs",
                "gov.grouping.reset_learning",
                "gov.grouping.phrase_floor_secs",
                "gov.grouping.hazard_cap_per_sec",
                "gov.grouping.evidence_rate",
                "gov.grouping.rules.saddle",
                "gov.grouping.rules.changepoint",
                "gov.grouping.rules.change",
                "gov.meter.pf_mix",
                "gov.meter.switch_cost",
                "gov.meter.rules.periodicity",
                "gov.roles.temperature",
                "gov.roles.virtual_signal_bias",
                "gov.prolongation.smoothing",
                "gov.prolongation.rules.surprisal",
                "gov.prolongation.rules.w_equilibrium",
                "gov.harmony.step_rate_gain",
                "gov.harmony.rules.direction",
                "gov.timespan.rules.surface"],
}

# Per-parameter step size for arrow-key edits.
EDIT_STEP_FINE = {
    # v4.2 governor weights: rule weights and costs step by 0.1; bounded
    # [0,1] parameters (thresholds, mixes, temperature, smoothing) by 0.02
    # so a coarse press (x10) is still a sane 0.2 move.
    # v4.4: grouping has no threshold or switch cost. prior_phrase_secs is a
    # DURATION in seconds, so it steps by 1 s fine / 10 s coarse.
    # v4.9.3: reset_learning's step is nominal (any event on it resets);
    # phrase_floor_secs steps by 0.5 s, hazard_cap_per_sec by 0.01.
    "gov.grouping.prior_phrase_secs":       1.0,
    "gov.grouping.reset_learning":          1.0,
    "gov.grouping.phrase_floor_secs":       0.5,
    "gov.grouping.hazard_cap_per_sec":      0.01,
    "gov.grouping.evidence_rate":           0.002,
    "gov.grouping.rules.saddle":            0.1,
    "gov.grouping.rules.changepoint":       0.1,
    "gov.grouping.rules.change":            0.1,
    "gov.meter.pf_mix":                     0.02,
    "gov.meter.switch_cost":                0.1,
    "gov.meter.rules.periodicity":          0.1,
    "gov.roles.temperature":                0.02,
    "gov.roles.virtual_signal_bias":        0.02,
    "gov.prolongation.smoothing":           0.005,
    "gov.prolongation.rules.surprisal":     0.1,
    "gov.prolongation.rules.w_equilibrium": 0.1,
    "gov.harmony.step_rate_gain":           0.25,
    "gov.harmony.rules.direction":          0.1,
    "gov.timespan.rules.surface":           0.1,
    "base_note": 1.0,        # semitones
    "bend":      1.0,        # cents
    "freq_hz":   5.0,        # absolute Hz (linear) — matches the +/-5 Hz
                             # pitchShifter reference; coarse = 50 Hz
    "scale_id":  1.0,
    "chord_id":  1.0,
    "wave_pos":  0.05,
    "wave_bank": 1.0,
    "attack":    0.1,
    "decay":     0.05,
    "sustain":   0.05,
    "release":   0.1,
    "cutoff":    50.0,       # Hz
    "res":       0.05,
    "fmode":     0.5,        # 0=LP, 0.5=BP, 1=HP
    "routing_cursor": 0.05,  # depth nudge
}
EDIT_STEP_COARSE = {k: v * 10.0 for k, v in EDIT_STEP_FINE.items()}


class KeyboardController:
    """
    pygame-based keyboard adapter implementing the Controller protocol.

    Construction does NOT require a video display; pygame's event subsystem
    can be initialised on its own. If no display is available (headless CI),
    the controller still constructs and just returns no events.
    """

    def __init__(self, *, init_display: bool = True, window_size=(580, 360),
                 caption="DSO v4.0 — keyboard controller"):
        import pygame  # lazy import
        pygame.init()
        # Only key events are required, but a tiny window is created so the
        # OS gives us keyboard focus. Headless mode skips this.
        self._pygame = pygame
        self._screen = None
        self._font_big   = None
        self._font_med   = None
        self._font_small = None
        self._font_mono  = None
        if init_display:
            try:
                self._screen = pygame.display.set_mode(window_size,
                                                      pygame.RESIZABLE)
                pygame.display.set_caption(caption)
                # Pre-load fonts (best-effort SysFont; falls back to default)
                self._font_big   = pygame.font.SysFont(
                    "helvetica,arial,sans", 22, bold=True)
                self._font_med   = pygame.font.SysFont(
                    "helvetica,arial,sans", 15, bold=True)
                self._font_small = pygame.font.SysFont(
                    "helvetica,arial,sans", 13)
                self._font_mono  = pygame.font.SysFont(
                    "menlo,monaco,courier,monospace", 13)
                # Try to bring window to front on first show
                try:
                    pygame.display.flip()
                except Exception:
                    pass
            except pygame.error:
                # Likely headless; carry on without a window.
                self._screen = None

        self.mode:    str  = PERFORMANCE_MODE
        self.octave:  int  = 4         # MIDI octave for lower row's 'z' (= C4)
        self.selected_slot: int = 0
        self.current_family: str = "pitch"
        self.current_param_idx: int = 0
        self._pressed_notes: dict[str, int] = {}   # key_char -> MIDI note
        self._n_slots = 16

        # Public state read by the HUD overlay
        self.last_event_msg: str = ""

    # ── Helpers ──────────────────────────────────────────────────────────
    def _current_param_name(self) -> str:
        params = EDIT_FAMILIES[self.current_family]
        if not params:
            return ""
        return params[self.current_param_idx % len(params)]

    def _emit(self, evs: list[ControllerEvent], ev: ControllerEvent,
              msg: str = "") -> None:
        evs.append(ev)
        if msg:
            self.last_event_msg = msg

    def _note_for(self, ch: str) -> Optional[int]:
        """Map a typed character to a MIDI note number based on octave."""
        base = (self.octave + 1) * 12   # MIDI: octave 4 -> C4 = 60
        if ch in _NOTE_LAYOUT_LOWER:
            return base + _NOTE_LAYOUT_LOWER[ch]
        if ch in _NOTE_LAYOUT_UPPER:
            return base + 12 + _NOTE_LAYOUT_UPPER[ch]
        return None

    # ── Main poll ────────────────────────────────────────────────────────
    def poll(self) -> list[ControllerEvent]:
        pg = self._pygame
        events: list[ControllerEvent] = []

        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                events.append(ControllerEvent(QUIT))
                continue

            if ev.type not in (pg.KEYDOWN, pg.KEYUP):
                continue

            try:
                ch = chr(ev.key) if 0 <= ev.key < 0x110000 else ""
            except ValueError:
                ch = ""
            ch = ch.lower()

            # --- Mode toggle (works in both modes) -----------------------
            if ev.type == pg.KEYDOWN and ev.key == pg.K_TAB:
                self.mode = (EDIT_MODE if self.mode == PERFORMANCE_MODE
                             else PERFORMANCE_MODE)
                self._emit(events,
                           ControllerEvent(MODE_TOGGLE),
                           f"mode -> {self.mode}")
                continue

            # --- Slot select (works in both modes) -----------------------
            if ev.type == pg.KEYDOWN and ch in _SLOT_KEYS:
                slot = _SLOT_KEYS[ch]
                # PERFORMANCE: number keys are slot select, period.
                # EDIT: same — but slot-select doesn't conflict here because
                # there's no overlap with EDIT's param-step keys.
                self.selected_slot = slot
                self._emit(events,
                           ControllerEvent(SELECT_SLOT, slot=slot, value=slot),
                           f"slot -> {slot}")
                continue

            # --- PERFORMANCE-mode keys -----------------------------------
            if self.mode == PERFORMANCE_MODE:
                if ev.type == pg.KEYDOWN:
                    # Octave shift
                    if ev.key == pg.K_LEFTBRACKET:
                        self.octave = max(0, self.octave - 1)
                        self._emit(events, ControllerEvent(OCTAVE_DOWN),
                                   f"octave -> {self.octave}")
                        continue
                    if ev.key == pg.K_RIGHTBRACKET:
                        self.octave = min(8, self.octave + 1)
                        self._emit(events, ControllerEvent(OCTAVE_UP),
                                   f"octave -> {self.octave}")
                        continue
                    # All-notes-off
                    if ev.key == pg.K_SPACE:
                        for k, n in list(self._pressed_notes.items()):
                            self._emit(events,
                                       ControllerEvent(NOTE_OFF,
                                           slot=self.selected_slot, note=n),
                                       "")
                        self._pressed_notes.clear()
                        continue
                    # Note keys
                    note = self._note_for(ch)
                    if note is not None and ch not in self._pressed_notes:
                        self._pressed_notes[ch] = note
                        self._emit(events,
                                   ControllerEvent(NOTE_ON,
                                       slot=self.selected_slot,
                                       note=note, velocity=100),
                                   f"NOTE_ON slot {self.selected_slot} n={note}")
                        continue
                elif ev.type == pg.KEYUP:
                    note = self._pressed_notes.pop(ch, None)
                    if note is not None:
                        self._emit(events,
                                   ControllerEvent(NOTE_OFF,
                                       slot=self.selected_slot, note=note),
                                   "")
                continue

            # --- EDIT-mode keys ------------------------------------------
            if self.mode == EDIT_MODE and ev.type == pg.KEYDOWN:
                # Family switch (UI-only state — no downstream event)
                if ch == "p":
                    self.current_family, self.current_param_idx = "pitch", 0
                    self.last_event_msg = "edit family -> pitch"
                    continue
                if ch == "w":
                    self.current_family, self.current_param_idx = "wave", 0
                    self.last_event_msg = "edit family -> wave"
                    continue
                if ch == "e":
                    self.current_family, self.current_param_idx = "env", 0
                    self.last_event_msg = "edit family -> env"
                    continue
                if ch == "f":
                    self.current_family, self.current_param_idx = "filter", 0
                    self.last_event_msg = "edit family -> filter"
                    continue
                if ch == "m":
                    self.current_family, self.current_param_idx = "mod", 0
                    self.last_event_msg = "edit family -> mod"
                    continue
                if ch == "g":
                    self.current_family, self.current_param_idx = "gov", 0
                    self.last_event_msg = "edit family -> gov (score grammar)"
                    continue

                # Bidirectional binding commands
                if ch == "b":
                    self._emit(events,
                               ControllerEvent(CLAIM_HYBRID,
                                   slot=self.selected_slot),
                               f"CLAIM_HYBRID slot {self.selected_slot}")
                    continue
                if ch == "n":
                    self._emit(events, ControllerEvent(CREATE_EMPTY),
                               "CREATE_EMPTY")
                    continue
                if ch == "t":
                    self._emit(events,
                               ControllerEvent(TIE_BLOB,
                                   slot=self.selected_slot),
                               f"TIE_BLOB slot {self.selected_slot}")
                    continue
                if ch == "d":
                    self._emit(events,
                               ControllerEvent(DETACH,
                                   slot=self.selected_slot),
                               f"DETACH slot {self.selected_slot}")
                    continue

                # Parameter scrolling within current family (UI-only state)
                if ev.key == pg.K_LEFT:
                    self.current_param_idx = (self.current_param_idx - 1) \
                                             % max(1, len(EDIT_FAMILIES[self.current_family]))
                    self.last_event_msg = (
                        f"edit param -> {self._current_param_name()}")
                    continue
                if ev.key == pg.K_RIGHT:
                    self.current_param_idx = (self.current_param_idx + 1) \
                                             % max(1, len(EDIT_FAMILIES[self.current_family]))
                    self.last_event_msg = (
                        f"edit param -> {self._current_param_name()}")
                    continue

                # Step current parameter
                step_dir = 0
                step_amt = 1.0
                if ev.key == pg.K_UP:
                    step_dir = +1
                elif ev.key == pg.K_DOWN:
                    step_dir = -1
                elif ev.key == pg.K_EQUALS:
                    step_dir, step_amt = +1, 10.0    # coarse
                elif ev.key == pg.K_MINUS:
                    step_dir, step_amt = -1, 10.0
                if step_dir != 0 and self.current_family != "mod":
                    pname    = self._current_param_name()
                    base_step= EDIT_STEP_FINE.get(pname, 0.05)
                    delta    = step_dir * base_step * step_amt
                    self._emit(events,
                               ControllerEvent(SET_PARAM,
                                   slot=self.selected_slot,
                                   param=pname,
                                   value=delta,
                                   extras={"relative": True}),
                               f"set {pname} += {delta:+.3f}")
                    continue

                # Reset / clear
                if ch == "r":
                    pname = self._current_param_name()
                    self._emit(events,
                               ControllerEvent(SET_PARAM,
                                   slot=self.selected_slot,
                                   param=pname,
                                   value=None,
                                   extras={"reset": True}),
                               f"reset {pname}")
                    continue
                if ch == "c":
                    self._emit(events,
                               ControllerEvent(REMOVE_ROUTING,
                                   slot=self.selected_slot,
                                   extras={"all": True}),
                               f"clear routings on slot {self.selected_slot}")
                    continue

        # End of for-each pygame event
        # Paint feedback to the pygame surface so the window is visible AND
        # informative (mode, selected slot, current family/param, last event)
        self._render()
        return events

    # ── Rendering ────────────────────────────────────────────────────────
    def _render(self) -> None:
        """Paint mode/slot/octave/family/param/last-event into the pygame
        window. Called every poll() so the surface is never blank."""
        if self._screen is None or self._font_med is None:
            return
        pg = self._pygame

        # Theme — matches the rest of the project (dark navy + lava-orange)
        BG       = (10, 14, 26)
        PANEL    = (17, 24, 43)
        PANEL_HI = (26, 35, 64)
        TEXT     = (232, 236, 245)
        TEXT_DIM = (154, 163, 184)
        LAVA     = (255, 90, 44)
        LAVA_HI  = (255, 179, 71)
        TEAL     = (93, 210, 199)
        LINE     = (42, 53, 80)

        try:
            self._screen.fill(BG)
            w, h = self._screen.get_size()

            # Top accent
            pg.draw.rect(self._screen, LAVA, (16, 12, 32, 2))
            pg.draw.line(self._screen, LINE, (16, 24), (w - 16, 24), 1)

            # Mode indicator (big)
            mode_color = LAVA_HI if self.mode == EDIT_MODE else LAVA
            mode_surf = self._font_big.render(
                f"[{self.mode}]", True, mode_color)
            self._screen.blit(mode_surf, (16, 32))

            # Slot + octave (top right)
            info = f"slot S{self.selected_slot:02d}    oct {self.octave}"
            info_surf = self._font_med.render(info, True, TEXT)
            self._screen.blit(info_surf,
                              (w - 16 - info_surf.get_width(), 38))

            # Mode-specific body
            y = 76
            if self.mode == PERFORMANCE_MODE:
                lines = [
                    ("Notes:", TEXT_DIM, self._font_small),
                    ("  z s x d c v g b h n j m , l .   (lower octave)",
                     TEXT, self._font_mono),
                    ("  q 2 w 3 e r 5 t 6 y 7 u i o p   (upper octave)",
                     TEXT_DIM, self._font_mono),
                    ("", None, None),
                    ("[ / ]   octave shift",         TEXT_DIM, self._font_small),
                    ("space   panic (all notes off)", TEXT_DIM, self._font_small),
                    ("1 - 0   select slot 0..9",      TEXT_DIM, self._font_small),
                    ("TAB     enter EDIT mode",       TEAL,     self._font_small),
                ]
            else:  # EDIT
                pname = self._current_param_name()
                lines = [
                    (f"family:  {self.current_family}",
                     TEAL, self._font_med),
                    (f"param:   {pname}",
                     LAVA_HI, self._font_med),
                    ("", None, None),
                    ("<- / ->  cycle parameter within family",
                     TEXT_DIM, self._font_small),
                    ("up / dn  fine adjust  (+/- = coarse, x10)",
                     TEXT_DIM, self._font_small),
                    ("",  None, None),
                    ("B  claim hybrid     N  create empty",
                     TEXT, self._font_small),
                    ("T  tie blob         D  detach",
                     TEXT, self._font_small),
                    ("R  reset param      C  clear routings",
                     TEXT, self._font_small),
                    ("",  None, None),
                    ("TAB  return to PERFORMANCE",
                     TEAL, self._font_small),
                ]

            for text, color, font in lines:
                if text and font is not None and color is not None:
                    surf = font.render(text, True, color)
                    self._screen.blit(surf, (16, y))
                y += 18

            # Last event message (bottom strip)
            pg.draw.line(self._screen, LINE,
                         (16, h - 36), (w - 16, h - 36), 1)
            if self.last_event_msg:
                msg = self.last_event_msg
                if len(msg) > 70:
                    msg = msg[:67] + "..."
                surf = self._font_small.render(
                    f"last:  {msg}", True, TEAL)
                self._screen.blit(surf, (16, h - 26))
            else:
                surf = self._font_small.render(
                    "(click this window to give it keyboard focus)",
                    True, TEXT_DIM)
                self._screen.blit(surf, (16, h - 26))

            pg.display.flip()
        except Exception:
            # Never let a render failure break the main loop
            pass

    def close(self) -> None:
        try:
            self._pygame.quit()
        except Exception:
            pass


# ── MIDIController stub (Phase 2) ────────────────────────────────────────────
# The full implementation lands in Phase 2. The stub exists so type checks
# and imports work today and the rest of the system can be wired against
# it.

class MIDIController:
    """
    Phase 2 placeholder. When activated, reads from a mido input port,
    consults a JSON learn-map, and emits ControllerEvents identical in
    shape to the keyboard's. Phase 1 ships this class as a no-op so that
    audible_lava.py can pass a controller list including it without crashing.
    """

    def __init__(self, port_name: Optional[str] = None,
                 learn_map_path: str = "learn_map.json") -> None:
        self.port_name      = port_name
        self.learn_map_path = learn_map_path
        self.learn_map: dict = self._load(learn_map_path)
        self.learn_target: Optional[tuple[int, str]] = None
        self._port = None  # opened lazily in poll()

    def _load(self, path: str) -> dict:
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self) -> None:
        try:
            with open(self.learn_map_path, "w") as f:
                json.dump(self.learn_map, f, indent=2)
        except Exception:
            pass

    def poll(self) -> list[ControllerEvent]:
        # Phase 2 will:
        #   - lazy-open self._port via mido
        #   - iter pending messages
        #   - if learn_target is set, capture the next CC/note message
        #     into learn_map and clear the target
        #   - otherwise look up (channel, cc) -> (slot, param) and emit
        #     SET_PARAM events
        return []

    def close(self) -> None:
        if self._port is not None:
            try: self._port.close()
            except Exception: pass
