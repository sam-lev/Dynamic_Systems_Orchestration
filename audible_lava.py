#!/usr/bin/env python3
# audible_lava.py -- Audible Lava / DSO v4.13 (MPE Edition)
#
# v4.13 (Nov 2026): one blob, one voice (tracker_stability.py, additive).
#   A still blob whose detected persistence region fluctuates no longer
#   spawns a stream of new voices that glide away. StableBlobTracker wraps
#   BlobTracker: fragments of one tracked blob are fused; a region that is a
#   small part of a known blob is treated as a partial view (known geometry,
#   centroid barely moves); new regions are offered to coasting tracks before
#   any birth; births need 3 consecutive detections (one-cycle flickers never
#   get a voice); coasting tracks stop extrapolating and stay near where they
#   were last seen; still blobs drop the buoyancy prior.
#   --no-stable-tracker restores the plain BlobTracker.
#
# v4.12 (Nov 2026): voice continuity (voice_continuity.py, additive).
#   - Every tracked blob gets a voice: a blob re-identified under its old id
#     (the tracker emits no birth for it), or missed at birth because the pool
#     was full, used to stay silent for the rest of its life. It is now bound
#     each frame: its zombie is resurrected if one matches, else it gets a new
#     voice without a birth strike.
#   - Level floor: governance can push a live voice into the background but
#     not below 40 % of its base level (CONTINUITY_AMP_FLOOR).
#   - The BEAT_PHASE -> AMP dip is proportional (a 28 % accent at any level)
#     instead of a fixed -0.28 that muted dim voices on every beat.
#   --no-continuity restores the v4.11 VoiceManager / BindingManager.
#
# v4.11 (Nov 2026): RAVE timbre-transfer layer (rave_layer.py +
#   blob_synth_rave.scd, via the nn.ar SuperCollider extension). RAVE =
#   Realtime Audio Variational autoEncoder (IRCAM).
#   - A final stage after the fully formed lava mix: \raveSplit (node 910)
#     copies bus 0 to a private bus and scales the dry mix; one
#     \raveXfer_<key> node per ACTIVE pre-trained model re-synthesises that
#     copy through its own timbre and adds it back to bus 0. Both sit BEFORE
#     \lavaMaster, so the space hears the transferred timbre.
#   - OFF by default: the lamp plays the rich SynthDefs untouched. The
#     keyboard window gains a third tab, RAVE (TAB cycles PERFORMANCE ->
#     EDIT -> RAVE): the home row toggles each of the twelve models, SPACE
#     turns the whole layer on/off, up/down set the wet mix. 'a' on the video
#     window also toggles the layer. --no-rave leaves it out; --rave-on
#     starts with it on (models still chosen from the tab).
#   - Models live in rave_layer.RAVE_MODEL_DIR (env AUDIBLE_LAVA_RAVE_DIR
#     overrides); SuperCollider loads them when blob_synth_rave.scd is
#     evaluated and writes rave_status.json there, which the tab shows.
#
# v4.10 (Nov 2026): rich synth palette (synth_palette.py + blob_synth_rich.scd).
#   - Every SynthDef has a rich sibling (plugins from sc3-plugins with stock
#     fallbacks) selected by name through synth_palette.install(); the OSC
#     contract to SuperCollider is unchanged.
#   - A master space (\lavaMaster, tail of group 0) is driven every frame by
#     ScoreState: ambiguity -> reverb time, tension -> damping / drive, phrase
#     position -> wet, surprisal -> Greyhole feedback, Kuramoto order ->
#     modulation depth, blob count -> room size, grouping boundary -> bloom.
#   - \preSound is finally wired: the predicted triad is pre-sounded at the
#     harmony posterior's non-current mass (PreSoundLayer).
#   - per-voice extras of \mpeVoiceRich: height -> vowel formant, area -> sub,
#     sideways drift -> chorus (VoiceExtras).
#   - hit models: ornamental hits are plucked strings, the paired head voice is
#     a piano hammer; melody notes are bowed when a blob expands, plucked when
#     it shrinks.
#   Flags: --classic-synths (old SynthDef names, no extras), --no-space (no
#   master insert), --no-presound. Key 'x' toggles the master bypass.
#
# v4.9.6 (Nov 2026): 'r' in the tracking window toggles a view of the frozen
# background reference (reference | current frame | difference) and saves it
# to background_reference.png.
#
# v4.9.4 (Nov 2026): pattern-layer hits are sent as OSC bundles stamped with
# their due wall-clock time (send_timed_hits) instead of immediately; nothing
# else changed.
#
# Startup:
#   if no lamp_profile.json exists, the calibration GUI opens first.
#
# v4.9 changes (Oct 2026) -- all additive to the v4.4 wiring:
#   - Melody / rhythm layer (melody_rhythm.py): hazard-driven onsets per
#     blob + height-band pitch with hysteresis and contour inertia, rendered
#     as one-shot \melodyHit notes (melody_hit.scd, evaluate after
#     blob_synth.scd). MELODY_ENABLED flag; 'n' key toggles at runtime.
#   - Warm-lamp background capture is reachable: BG_REF_MODE = "warm" takes
#     a per-pixel low percentile over BG_REF_CAPTURE_SECS so a lamp that is
#     already running is not baked into the reference. 'w' re-captures warm,
#     'b' re-captures in the configured mode.
#   - Merge-tree wax-mask pruning (v4.7) now actually runs: the loop called
#     detector.get_debug_masks(frame, ...) with the FULL frame and an
#     ROI-sized grey, which missed the mask cache, raised on the size
#     mismatch, and was silently caught -> wax_mask was None every frame.
#     It now passes `roi`, the same object detect() received, so the cached
#     masks are served and the tree is pruned to detected wax.
#
# v4.0 changes vs v3.x:
#   - 16-slot voice pool (VoiceManager refactor)
#   - \mpeVoice SynthDef in SuperCollider (wavetable + SVF + ADSR + lag)
#   - Computer-keyboard MPE controller via pygame
#   - BindingManager bridging tracker events <-> voice lifecycle
#   - Bidirectional blob<->voice binding (CREATE_EMPTY / TIE / DETACH /
#     CLAIM_HYBRID)
#   - Voice-ID overlay on video, 16-slot HUD on right side
#   - MAX_COAST raised 8 -> 20  (v3.2)
#   - MOG2 learning rate frozen to 1e-4 post-warmup  (v3.2)
#   - Frozen median background reference as default bg-sub  (v4.1):
#     captured once from the cold lamp, never adapts, so long-lived blobs
#     never fade. Flip BACKGROUND_MODE to "mog2" to A/B the old behaviour.
#
# Usage:
#   python audible_lava.py               # auto-calibrate on first run
#   python audible_lava.py --calibrate   # force re-calibration
#   python audible_lava.py --no-gui      # headless (no OpenCV window)
#   python audible_lava.py --no-keyboard # tracker + voices only, no player
#   python audible_lava.py --no-governor # drones only (no score grammar)
#   python audible_lava.py --no-patterns # governor without the \mpeHit layer
#   python audible_lava.py --no-melody   # start with the melody layer off
#   python audible_lava.py --classic-synths  # v4.9 SynthDefs, no extras (v4.10)
#   python audible_lava.py --no-space    # no master space insert (v4.10)
#   python audible_lava.py --no-presound # no predicted-triad pre-sound (v4.10)
#   python audible_lava.py --no-rave     # no RAVE layer at all (v4.11)
#   python audible_lava.py --rave-on     # start with the RAVE layer on (v4.11)
#   python audible_lava.py --no-continuity  # v4.11 voice binding / levels (v4.12)
#   python audible_lava.py --no-stable-tracker  # plain v4.12 BlobTracker (v4.13)
#
# SuperCollider load order for the rich palette + RAVE:
#   blob_synth.scd -> blob_synth_hits.scd -> melody_hit.scd
#     -> blob_synth_rich.scd -> blob_synth_rave.scd
#
# v4.9.1: the Lava Climb game feed (lava_bridge.py / lava_climb.py) is removed
# from the main loop; no UDP publisher is created.
#
# OpenCV-window keys (focus on the video window):
#   q    quit
#   m    toggle merge tree
#   t    toggle tracking debug
#   f    toggle optical flow
#   c    toggle colour mask preview
#   o    toggle blob contours
#   v    toggle voice slot HUD
#   n    toggle melody / rhythm layer                      (v4.9)
#   x    toggle master space bypass (A/B the insert)       (v4.10)
#   a    toggle RAVE timbre-transfer layer                 (v4.11)
#   b    re-capture background reference (configured mode) / reset MOG2
#   w    re-capture background reference WARM (lamp already running) (v4.9)
#   [    persistence threshold -
#   ]    persistence threshold +
#
# pygame-window keys (focus on the keyboard window): see controller.py.

import cv2
import numpy as np
import time
import sys
import os

from lamp_profile    import LampProfile, PROFILE_PATH
from blob_detection  import BlobDetector
from blob_tracker    import BlobTracker
from merge_tree      import compute_merge_tree, draw_tree_overlay

# v4.0 modules
from voice_manager   import VoiceManager, VoiceState
from binding_manager import BindingManager
from score_governor  import ScoreGovernor           # v4.2 topological score grammar
from melody_rhythm   import MelodyEngine            # v4.9 hazard onsets + band pitch
import synth_palette                                # v4.10 rich palette + master space
import rave_layer                                   # v4.11 RAVE timbre transfer
import voice_continuity                             # v4.12 every tracked blob keeps a voice
import tracker_stability                            # v4.13 one blob, one voice
from controller      import (
    KeyboardController,
    NOTE_ON, NOTE_OFF, SELECT_SLOT, SET_PARAM,
    ADD_ROUTING, REMOVE_ROUTING,
    CREATE_EMPTY, CLAIM_HYBRID, TIE_BLOB, DETACH,
    LEARN_ARM, LEARN_CLEAR, OCTAVE_UP, OCTAVE_DOWN,
    MODE_TOGGLE, QUIT, PERFORMANCE_MODE, EDIT_MODE,
)


# ── Configuration ──────────────────────────────────────────────────────────────
CAMERA_INDEX         = 0
TARGET_FPS           = 15
MAX_COAST            = 20         # v3.2 (was 8 in v3.0)
SC_IP                = "127.0.0.1"
SC_PORT              = 57110
MOG2_WARMUP_SECS     = 35
MOG2_POST_WARMUP_LR  = 1e-4       # v3.2: freeze background after warm-up

# ── Background subtraction (v4.1) ────────────────────────────────────────────
# "frozen" : capture a non-adaptive reference at startup, then never adapt ->
#            long-lived blobs never fade, and the static scene + cold wax
#            reservoir are suppressed. Recommended.
#            (Lock camera exposure/white balance -- see the camera setup script
#            -- since a frozen reference does not track lighting drift.)
# "mog2"   : legacy adaptive MOG2 with the warm-up -> 1e-4 freeze (flip here to
#            A/B against the old behaviour).
BACKGROUND_MODE      = "frozen"
BG_REF_FRAMES        = 40         # cold mode: frames averaged into the reference
BG_REF_THRESHOLD     = 18         # |grey - reference| foreground cutoff (0-255)
BG_REF_SETTLE        = 10         # frames discarded first so exposure settles
BG_REF_METHOD        = "median"   # cold mode: "median" (robust) or "mean" (low-memory)

# ── Frozen-reference capture mode (v4.3 in background_reference; wired v4.9) ──
# "cold" : median of BG_REF_FRAMES consecutive frames. Only correct if the
#          lamp is settled (no wax moving) when the program starts; a running
#          lamp gets its blobs baked into the reference.
# "warm" : per-pixel BG_REF_PERCENTILE over BG_REF_CAPTURE_SECS of wall-clock
#          time. Over a window spanning several blob transits each pixel is
#          background most of the time and (brighter) wax only intermittently,
#          so the low percentile is the background. Use this when you start
#          on a lamp that is already running. 90 s is a floor; 150 s is
#          better. Press 'w' to re-capture warm at any time.
BG_REF_MODE          = "warm"
BG_REF_CAPTURE_SECS  = 90.0
BG_REF_PERCENTILE    = 25.0       # lower toward 10 if wax still shows in the
                                  # reference; raise toward 40 if blobs read
                                  # larger than they are

SHOW_WINDOW          = True
DOWNSCALE            = 4
DETECT_INTERVAL_DEFAULT = 1   # K fallback if not set in lamp_profile.json

N_VOICE_SLOTS        = 16
ZOMBIE_WINDOW_SECS   = 4.0
BLOB_RECENT_PLAY_WINDOW = 5.0

# ── Melody / rhythm layer (v4.9) ─────────────────────────────────────────────
# One-shot \melodyHit notes on top of the sustained drones. Onsets come from a
# per-blob hazard accumulator (still blob ~1 note / 12 s, fast or turning blob
# up to 0.6 notes/s); pitch from the blob's height band with hysteresis and
# contour inertia. Tunables live in melody_rhythm.HazardParams / PitchParams.
MELODY_ENABLED       = True

# ── Synth palette (v4.10) ────────────────────────────────────────────────────
# "rich"    : blob_synth_rich.scd SynthDefs (needs that file evaluated; the
#             plugins are optional, stock fallbacks are built in)
# "classic" : the v4.9 SynthDefs exactly as before. --classic-synths overrides.
SYNTH_PALETTE        = "rich"
MASTER_SPACE         = True       # \lavaMaster insert on bus 0 (--no-space)
MASTER_DEPTH         = 1.0        # 0 = static default room, 1 = full ScoreState drive
PRESOUND             = True       # predicted-triad pre-sound (--no-presound)
PRESOUND_GAIN        = 0.10       # amplitude at posterior_other = 1

# ── RAVE timbre transfer (v4.11) ─────────────────────────────────────────────
# RAVE_LAYER installs the split node and the keyboard tab (--no-rave removes
# both). The layer starts OFF (RAVE_ON_AT_START / --rave-on turn it on at
# startup; models are still chosen from the tab), so by default the lamp
# plays the rich SynthDefs untouched. RAVE_MIX is the wet fraction while on
# (dry * (1 - mix) + sum of models at mix / sqrt(n)); RAVE_IN_GAIN drives the
# dry copy into the models (RAVE likes a healthy input level).
# ── Voice continuity (v4.12) ─────────────────────────────────────────────────
# CONTINUITY: bind tracked blobs that have no voice (re-identified / missed at
# birth), floor live voices at CONTINUITY_AMP_FLOOR x base level, beat dip
# proportional. --no-continuity = v4.11 behaviour.
CONTINUITY           = True
CONTINUITY_AMP_FLOOR = 0.4        # 0 disables the floor only
CONTINUITY_MIN_AGE   = 8          # frames a blob must be tracked before a rebind

# ── Tracker stability (v4.13) ────────────────────────────────────────────────
# STABLE_TRACKER: fuse fragments, complete partial views, re-associate new
# regions with coasting tracks, confirm births, damp coasting, stationary
# mode. STABLE_CONFIRM_CYCLES detection cycles before a birth (1 = immediate;
# 3 at DETECT_K=2 and 15 fps is ~0.4 s). --no-stable-tracker = v4.12 tracker.
STABLE_TRACKER        = True
STABLE_CONFIRM_CYCLES = 3
STABLE_VERBOSE        = False      # print fused / re-associated / dropped events

RAVE_LAYER           = True
RAVE_ON_AT_START     = False
RAVE_MIX             = 0.7
RAVE_IN_GAIN         = 1.0

PERS_STEP            = 2.0
PERS_MIN             = 2.0
PERS_MAX             = 80.0


# ── Direction colour coding ────────────────────────────────────────────────────
DIR_COLOURS = {
    "rising":  (80,  255, 128),
    "falling": (80,  128, 255),
    "turning": (255, 220, 80 ),
}

# Voice state colour coding (BGR) — used for the voice ring around each blob.
STATE_COLOURS = {
    VoiceState.IDLE:          (90, 90, 90),
    VoiceState.BLOB_ONLY:     (240, 160, 60),     # cyan-ish for blob-driven
    VoiceState.KEYBOARD_ONLY: (80, 230, 120),     # green for keyboard-only
    VoiceState.HYBRID:        (60, 150, 255),     # warm orange for hybrid
    VoiceState.ZOMBIE:        (150, 130, 130),    # dim grey for zombie
}


# ── Overlay helpers ────────────────────────────────────────────────────────────
def draw_blob_contours(overlay, blobs, origin=(0, 0)):
    if not blobs:
        return
    ox, oy = origin
    max_pers = max(b.persistence for b in blobs) or 1.0
    for b in blobs:
        if len(b.contour) < 3:
            continue
        t = b.persistence / max_pers
        colour = (int(40 + 40 * t), int(80 + 120 * t), int(180 + 75 * t))
        cnt = (b.contour + np.array([ox, oy])).astype(np.int32)
        cv2.drawContours(overlay, [cnt], -1, colour, 2)
        cx, cy = int(b.centroid[0]) + ox, int(b.centroid[1]) + oy
        cv2.putText(overlay, f"p{b.persistence:.0f}",
                    (cx - 14, cy - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                    (220, 220, 220), 1)


def draw_voice_overlay(overlay, tracked, voices, selected_slot, origin=(0, 0),
                       melody=None):
    """
    For every tracked blob: draw a centroid dot in its direction colour, a
    voice ring colour-coded by state (with a thicker ring on the currently
    selected slot), and a label showing slot ID + state + key params.
    Coordinates are ROI-local and shifted by `origin` (the lamp ROI top-left).

    v4.9: if `melody` (a MelodyEngine) is given, a third line shows the
    blob's hazard rate h (notes/s), accumulator A, band, sounding degree,
    contour arrow and MIDI note -- so you can see whether the rhythm is
    actually driven by motion and whether pitch flickers at a band edge.
    """
    ox, oy = origin
    for b in tracked:
        cx, cy = int(b.centroid[0]) + ox, int(b.centroid[1]) + oy
        dir_col = DIR_COLOURS.get(b.direction, (0, 255, 0))
        cv2.circle(overlay, (cx, cy), 5, dir_col, -1)

        slot_id = voices.slot_for_blob(b.id)
        if slot_id is None:
            cv2.putText(overlay, f"#{b.id}",
                        (cx + 10, cy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, dir_col, 1)
            continue

        v        = voices.voice(slot_id)
        st_col   = STATE_COLOURS[v.state]
        ring_w   = 3 if slot_id == selected_slot else 2
        radius   = 22 if slot_id == selected_slot else 18
        cv2.circle(overlay, (cx, cy), radius, st_col, ring_w)

        st_label = v.state.name.split("_")[0][:4]
        head     = f"S{slot_id:02d} {st_label} #{b.id}"
        cv2.putText(overlay, head,
                    (cx + 10, cy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, st_col, 1)

        info = (f"{v.synth.c_freq:.0f}Hz  "
                f"{b.direction[:3]}  "
                f"hue={b.mean_hue:.0f}")
        cv2.putText(overlay, info,
                    (cx + 10, cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, st_col, 1)

        if melody is not None:
            mi = melody.blob_info(b.id)
            if mi:
                cv2.putText(overlay, mi, (cx + 10, cy + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, st_col, 1)


def draw_voice_hud(overlay, voices, controller, binding,
                   show_hud=True, panel_w=320):
    """
    Right-side panel listing all 16 slots: index, state, blob_id, current
    freq, gate. The selected slot is highlighted. A mode/family/octave
    line tops the panel.
    """
    if not show_hud:
        return
    h_img, w_img = overlay.shape[:2]
    px = w_img - panel_w
    py = 0
    # Semi-transparent backdrop
    rect = overlay[py:h_img, px:w_img].copy()
    rect[:] = (rect[:] * 0.35).astype(np.uint8)
    overlay[py:h_img, px:w_img] = cv2.addWeighted(
        overlay[py:h_img, px:w_img], 0.35, rect, 0.65, 0)
    cv2.rectangle(overlay, (px, py), (w_img - 1, h_img - 1),
                  (50, 70, 110), 1)

    # Header
    mode_txt = controller.mode if controller is not None else "—"
    family   = (f"  {controller.current_family}" if controller is not None and
                controller.mode == EDIT_MODE else "")
    oct_txt  = f"  oct {controller.octave}" if controller is not None else ""
    sel      = controller.selected_slot if controller is not None else 0
    cv2.putText(overlay, f"VOICE SLOTS  [{mode_txt}{family}{oct_txt}]",
                (px + 10, py + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 240), 1)
    cv2.putText(overlay, f"selected: S{sel:02d}",
                (px + 10, py + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 200, 220), 1)
    if binding is not None:
        cv2.putText(overlay, binding.hud_summary(),
                    (px + 10, py + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 180, 200), 1)

    # Per-slot rows
    row_h = 14
    base_y = py + 78
    for i in range(N_VOICE_SLOTS):
        v = voices.voice(i)
        y = base_y + i * row_h
        col = STATE_COLOURS.get(v.state, (90, 90, 90))
        if i == sel:
            cv2.rectangle(overlay,
                          (px + 4, y - 10),
                          (w_img - 5, y + 3),
                          (40, 60, 90), -1)
        st_label = v.state.name[:8]
        blob_lbl = "—" if v.blob_id is None else f"#{v.blob_id}"
        ngate    = (v.synth.gate if v.state != VoiceState.IDLE else 0)
        msg = (f"S{i:02d} {st_label:<8} blob:{blob_lbl:<5}"
               f" {v.synth.c_freq:6.1f}Hz g{ngate}")
        cv2.putText(overlay, msg,
                    (px + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1)

    # Last controller event message
    if controller is not None and controller.last_event_msg:
        cv2.putText(overlay, controller.last_event_msg[:38],
                    (px + 10, h_img - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 230, 200), 1)


def draw_flow_overlay(overlay, flow, step=16, origin=(0, 0)):
    if flow is None:
        return
    ox, oy = origin
    h, w = flow.shape[:2]
    for y in range(0, h, step):
        for x in range(0, w, step):
            dx, dy = flow[y, x]
            if abs(dx) + abs(dy) < 0.5:
                continue
            start = (x + ox, y + oy)
            end   = (int(x + dx * 3) + ox, int(y + dy * 3) + oy)
            cv2.arrowedLine(overlay, start, end, (200, 200, 80), 1,
                            tipLength=0.3)


def draw_status(overlay, voices, leaves, blobs, fps,
                warm_note, flags, pers_thresh, panel_w=320, extra="", extra2=""):
    """Top status bar + persistence threshold bar at the bottom-left."""
    h_img, w_img = overlay.shape[:2]
    counts = voices.hud_summary()

    warm_str = warm_note
    flag_str = "  ".join(f"[{k}]" for k, v in flags.items() if v)
    top_line = (
        f"Blobs:{len(blobs)}  Slots: "
        f"BO{counts['BLOB_ONLY']} KO{counts['KEYBOARD_ONLY']} "
        f"H{counts['HYBRID']} Z{counts['ZOMBIE']} I{counts['IDLE']}  "
        f"Leaves:{len(leaves)}  FPS:{fps:.0f}"
        + (f"  {flag_str}" if flag_str else "")
        + warm_str)
    cv2.putText(overlay, top_line, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    if extra:
        cv2.putText(overlay, extra, (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 200), 1)
    if extra2:                                    # v4.10: synth palette / space line
        cv2.putText(overlay, extra2, (10, 63),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 230), 1)

    bar_x  = 10
    bar_y  = h_img - 18
    label  = f"Persistence threshold: {pers_thresh:.1f}   ( [ / ] to adjust )"
    cv2.putText(overlay, label,
                (bar_x, bar_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1)
    bar_w  = 200
    filled = int(bar_w * (pers_thresh - PERS_MIN) / (PERS_MAX - PERS_MIN))
    filled = max(0, min(filled, bar_w))
    cv2.rectangle(overlay, (bar_x, bar_y + 4),
                  (bar_x + bar_w, bar_y + 10), (50, 50, 50), -1)
    cv2.rectangle(overlay, (bar_x, bar_y + 4),
                  (bar_x + filled, bar_y + 10), (100, 200, 255), -1)


# ── Controller-event dispatch (the bits VoiceManager doesn't handle alone) ────

# ── v4.9.4 timed hit sends ────────────────────────────────────────────────────
# The pattern scheduler returns (t_on, args) with t_on in GOVERNOR time (the
# governor's own clock, which advances by frame_interval per frame and starts
# at 0). Until v4.9.3 the main loop ignored t_on and sent every hit the
# moment it was rendered, so hits sounded up to one look-ahead window
# (0.3 s) early and the elastic pull on the hit layer was lost on the way
# out. Now each hit goes out as an OSC bundle stamped with the wall-clock time
# it is due: wall_due = now + (t_on - governor.t). scsynth schedules bundles
# itself (sample-accurate, independent of Python's frame jitter). A bundle
# that is already late is still played, with a "late" line in the post window.
_bundle_warned = False


def send_timed_hits(osc, msgs, governor_now_s, wall_now_s):
    """Send [(t_on, args)] as timestamped OSC bundles; fall back to immediate
    sends if the bundle classes are unavailable."""
    global _bundle_warned
    if not msgs:
        return 0
    try:
        from pythonosc import osc_bundle_builder, osc_message_builder
    except Exception as exc:
        if not _bundle_warned:
            print(f"  ! timed hits unavailable ({exc}); sending immediately")
            _bundle_warned = True
        for _t_on, args in msgs:
            osc.send_message(args[0], args[1:])
        return len(msgs)
    n = 0
    for t_on, args in msgs:
        due = wall_now_s + (float(t_on) - float(governor_now_s))
        b = osc_bundle_builder.OscBundleBuilder(due)
        m = osc_message_builder.OscMessageBuilder(address=args[0])
        for a in args[1:]:
            m.add_arg(a)
        b.add_content(m.build())
        osc.send(b.build())
        n += 1
    return n


def dispatch_controller_event(ev, voices, binding, tracked, rave=None):
    """
    Some events need the BindingManager (lifecycle changes); the rest go
    straight to VoiceManager.dispatch. v4.11: RAVE_* events go to the RAVE
    layer and never reach the voice pool.
    """
    if ev.type == CREATE_EMPTY:
        binding.request_create_empty()
        return
    if ev.type == TIE_BLOB and ev.slot is not None:
        binding.request_tie(ev.slot, tracked_blobs=tracked)
        return
    if ev.type == DETACH and ev.slot is not None:
        binding.request_detach(ev.slot)
        return
    if ev.type == CLAIM_HYBRID and ev.slot is not None:
        binding.request_claim_hybrid(ev.slot)
        return
    # MODE_TOGGLE / OCTAVE_*: controller-internal, no system action
    if ev.type in (MODE_TOGGLE, OCTAVE_UP, OCTAVE_DOWN):
        return
    if ev.type == QUIT:
        return
    # v4.11: RAVE tab events
    if str(ev.type).startswith("RAVE_"):
        if rave is not None:
            if rave.handle_controller_event(ev):
                print(f"[RAVE] {rave.last_msg}")
        else:
            print("[RAVE] layer not installed (--no-rave)")
        return
    # v4.2: EDIT-surface governor weights ("gov.<module>.<key>") -- live tuning
    if ev.type == SET_PARAM and getattr(ev, "param", None) and str(ev.param).startswith("gov."):
        gov = getattr(voices, "score_governor", None)
        if gov is not None and gov.handle_controller_event(ev):
            return
    # Everything else: voice-level event
    voices.dispatch(ev)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    force_calibrate = "--calibrate"   in sys.argv
    headless        = "--no-gui"      in sys.argv
    no_keyboard     = "--no-keyboard" in sys.argv
    # v4.10 synth palette flags
    palette_name    = "classic" if "--classic-synths" in sys.argv else SYNTH_PALETTE
    palette         = synth_palette.PALETTES[palette_name]
    use_space       = MASTER_SPACE and "--no-space" not in sys.argv
    use_presound    = PRESOUND and "--no-presound" not in sys.argv
    # v4.11 RAVE flags
    use_rave        = RAVE_LAYER and "--no-rave" not in sys.argv
    # v4.12
    use_continuity  = CONTINUITY and "--no-continuity" not in sys.argv
    # v4.13
    use_stable      = STABLE_TRACKER and "--no-stable-tracker" not in sys.argv
    rave_on         = RAVE_ON_AT_START or "--rave-on" in sys.argv

    # ── Load or create LampProfile ────────────────────────────────────────
    profile = LampProfile.load(PROFILE_PATH) if not force_calibrate else None

    if profile is None:
        print("[Startup] No lamp profile found. Opening calibration GUI...")
        from calibration_gui import run_calibration
        profile = run_calibration(CAMERA_INDEX, PROFILE_PATH)
        print("[Startup] Calibration complete.")
    else:
        print(f"[Startup] Loaded lamp profile from {PROFILE_PATH}")

    # ── Camera ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open camera"); return

    # Force capture to the resolution the profile was calibrated at.
    # If the camera negotiates to a different mode, apply_to_capture
    # rescales the lamp ROI to keep blob_area_min / lamp_x1.. consistent.
    profile.apply_to_capture(cap)

    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read first frame"); cap.release(); return

    h, w = frame.shape[:2]
    if BACKGROUND_MODE == "frozen":
        if BG_REF_MODE == "warm":
            bg_note = (f"background: frozen ref, WARM capture "
                       f"({BG_REF_CAPTURE_SECS:.0f}s, p{BG_REF_PERCENTILE:.0f})")
        else:
            bg_note = (f"background: frozen ref, cold ({BG_REF_FRAMES} frames, "
                       f"{BG_REF_METHOD}, +{BG_REF_SETTLE} settle)")
    else:
        bg_note = f"MOG2 warm-up: {MOG2_WARMUP_SECS}s"
    print(f"[Startup] Camera: {w}x{h}  {bg_note}")
    print(f"          OSC -> {SC_IP}:{SC_PORT}")

    # ── Pipeline components ───────────────────────────────────────────────
    # Detection cadence K (v4.1): run the heavy CV stack every K-th frame and
    # predict-only in between. coast_frames counts missed DETECTION cycles, so
    # scale MAX_COAST (a frame count) by K to preserve wall-clock coast time.
    DETECT_K     = max(1, int(getattr(profile, "detect_interval",
                                      DETECT_INTERVAL_DEFAULT)))
    coast_cycles = max(3, round(MAX_COAST / DETECT_K))

    x1, y1, x2, y2 = profile.roi_rect()
    print(f"[Startup] Lamp ROI: ({x1},{y1})-({x2},{y2}) "
          f"= {x2 - x1}x{y2 - y1} px "
          f"({100.0 * (x2 - x1) * (y2 - y1) / float(w * h):.0f}% of frame)")

    detector = BlobDetector(profile=profile, downscale=DOWNSCALE,
                            background=BACKGROUND_MODE,
                            bg_ref_frames=BG_REF_FRAMES,
                            bg_ref_threshold=BG_REF_THRESHOLD,
                            bg_ref_settle=BG_REF_SETTLE,
                            bg_ref_method=BG_REF_METHOD,
                            # v4.9: warm-lamp capture plumbed through
                            bg_ref_mode=BG_REF_MODE,
                            bg_ref_capture_secs=BG_REF_CAPTURE_SECS,
                            bg_ref_percentile=BG_REF_PERCENTILE)
    if use_stable:                       # v4.13: same arguments, plus the fixes
        tracker = tracker_stability.StableBlobTracker(
            profile=profile, max_coast=coast_cycles,
            confirm_cycles=STABLE_CONFIRM_CYCLES, verbose=STABLE_VERBOSE)
    else:
        tracker = BlobTracker(profile=profile,  max_coast=coast_cycles)
    print(f"[Startup] Tracker: {'stable (v4.13)' if use_stable else 'plain BlobTracker (--no-stable-tracker)'}"
          + (f"  (births confirmed after {STABLE_CONFIRM_CYCLES} detections)" if use_stable else ""))
    # v4.12: the continuity subclasses (additive; same constructor arguments)
    if use_continuity:
        voices  = voice_continuity.ContinuityVoiceManager(
            sc_ip=SC_IP, sc_port=SC_PORT,
            n_slots=N_VOICE_SLOTS,
            profile=profile,
            zombie_window=ZOMBIE_WINDOW_SECS,
            amp_floor_frac=CONTINUITY_AMP_FLOOR,
        )
        binding = voice_continuity.ContinuityBindingManager(
            voices,
            blob_recent_play_window=BLOB_RECENT_PLAY_WINDOW,
            rebind_min_age=CONTINUITY_MIN_AGE)
    else:
        voices   = VoiceManager(
            sc_ip=SC_IP, sc_port=SC_PORT,
            n_slots=N_VOICE_SLOTS,
            profile=profile,
            zombie_window=ZOMBIE_WINDOW_SECS,
        )
        binding  = BindingManager(voices,
                                  blob_recent_play_window=BLOB_RECENT_PLAY_WINDOW)
    print(f"[Startup] Voice continuity: {'on' if use_continuity else 'off (--no-continuity)'}"
          + (f"  (floor {CONTINUITY_AMP_FLOOR:.0%} of base, rebind after {CONTINUITY_MIN_AGE} frames)"
             if use_continuity else ""))
    # ── v4.9 melody / rhythm layer (additive; reads VoiceManager read-only) ──
    # Requires melody_hit.scd to have been evaluated in SuperCollider after
    # blob_synth.scd. Only blobs bound to a live slot play notes, so the layer
    # inherits the binding policy unchanged.
    # v4.10: constructed through the palette so the rich engine can add the
    # bowed/plucked model per note (plain MelodyEngine for the classic palette).
    melody   = synth_palette.make_melody_engine(
        voices, profile, palette,
        enabled=MELODY_ENABLED and "--no-melody" not in sys.argv)

    keyboard = None
    if not no_keyboard:
        try:
            keyboard = KeyboardController(init_display=not headless)
            print(f"[Startup] Keyboard controller ready ({N_VOICE_SLOTS} slots)")
            print( "   PERFORMANCE: 1-0 select slot · zsxdcvgbhnjm,l. lower octave")
            print( "                 q2w3er5t6y7uiopl. upper octave · [ ] = octave")
            print( "                 space = panic · TAB to enter EDIT")
            print( "   EDIT:        P/W/E/F/M family · arrows step · = / -")
            print( "                 B claim · N create-empty · T tie · D detach")
            if use_rave:
                print( "   RAVE:        home row = model on/off · space = layer on/off")
                print( "                 up/dn = mix · Z all off · R re-read status · TAB back")
        except Exception as exc:
            print(f"[Startup] Keyboard controller failed: {exc}")
            keyboard = None

    print(f"[Startup] MAX_COAST={MAX_COAST} frames -> {coast_cycles} detection "
          f"cycles (K={DETECT_K})  background={BACKGROUND_MODE}"
          + (f" (MOG2_POST_WARMUP_LR={MOG2_POST_WARMUP_LR})"
             if BACKGROUND_MODE == "mog2" else ""))
    print(f"[Startup] Melody layer: {'on' if melody.enabled else 'off'}  ('n' toggles)")

    # ── Loop state ────────────────────────────────────────────────────────
    frame_interval = 1.0 / TARGET_FPS
    # ── v4.2 score governor (additive; reads tracker state read-only) ─────
    governor = ScoreGovernor(roi_w=float(x2 - x1), roi_h=float(y2 - y1), fps=TARGET_FPS,
                             weights_path="governor_weights.json",
                             runtime_path="governor_runtime.json", log_path="governor_log.txt")
    voices.score_governor = governor        # for EDIT-surface param routing
    use_governor  = "--no-governor" not in sys.argv
    # The gated \mpeHit pattern layer. Requires blob_synth_hits.scd to have
    # been evaluated in SuperCollider after blob_synth.scd.
    pattern_layer = use_governor and "--no-patterns" not in sys.argv

    # ── v4.10 synth palette, master space, pre-sound ─────────────────────
    # Installed AFTER VoiceManager (its /g_freeAll 0 would free the master
    # node) and after the governor (whose scheduler it wraps). Requires
    # blob_synth_rich.scd to have been evaluated for the "rich" palette; the
    # classic palette uses the v4.9 SynthDefs and sends no extras.
    rig = synth_palette.install(
        voices, profile,
        governor=governor if use_governor else None,
        melody=melody,
        palette=palette,
        master=use_space,
        presound=use_presound and use_governor,
        master_depth=MASTER_DEPTH,
        presound_gain=PRESOUND_GAIN,
    )
    print(f"[Startup] Synth palette: {palette.name}  "
          f"space={'on' if rig.master is not None else 'off'}  "
          f"presound={'on' if rig.presound is not None else 'off'}  ('x' = space bypass)")

    # ── v4.11 RAVE timbre transfer ────────────────────────────────────────
    # Installed AFTER the rig so the split and the model nodes can be placed
    # before the master node (addBefore 900); without a master they go to the
    # tail of group 0. Requires blob_synth_rave.scd to have been evaluated for
    # anything to sound; with it missing the split's /s_new is a no-op in SC
    # and the tab lists every model as unverified.
    rave = None
    if use_rave:
        rave = rave_layer.install(
            voices,
            master_node=rig.master.node_id if rig.master is not None else None,
            enabled=rave_on, mix=RAVE_MIX, in_gain=RAVE_IN_GAIN)
        if keyboard is not None:
            keyboard.enable_rave_tab(rave.panel_lines, rave_layer.RAVE_MODEL_KEYS)
        n_loaded = sum(1 for s in rave.status.models.values() if s.status == "loaded")
        print(f"[Startup] RAVE layer: {'on' if rave.enabled else 'off'}  "
              f"models dir {rave_layer.RAVE_MODEL_DIR}  "
              f"status: {n_loaded} loaded in SC"
              + ("" if rave.status.models else " (no rave_status.json yet)")
              + "  ('a' = layer on/off, RAVE tab = models)")
    else:
        print("[Startup] RAVE layer: not installed (--no-rave)")

    start_time     = time.time()
    frame_count    = 0
    last_blobs     = []   # most recent detection result (overlay continuity)

    show_tree     = False
    show_track    = True
    show_flow     = False
    show_masks    = False
    show_ref      = False   # v4.9.6: 'r' shows the frozen background reference
    show_contours = True
    show_hud      = True

    print("[OpenCV window keys] q=quit  m=tree  t=track  f=flow"
          "  c=masks  o=contours  v=HUD  n=melody  x=space-bypass  a=rave"
          "  b=rebaseline-bg  w=rebaseline-warm  r=show-reference  [/] persistence")
    quit_requested = False

    try:
        while not quit_requested:
            t0         = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            elapsed   = time.time() - start_time
            # "warmed up" gates the whole downstream pipeline. For the frozen
            # reference this means the capture has completed; for MOG2 it is the
            # fixed warm-up timer as before.
            if BACKGROUND_MODE == "frozen":
                warmed_up = detector.background_ready()
            else:
                warmed_up = elapsed > MOG2_WARMUP_SECS

            # ── ROI crop + shared greyscale (v4.1) ────────────────────────
            # Everything downstream runs on the lamp ROI sub-image, computed
            # once. grey is shared by the detector, tracker, and merge tree.
            # NOTE: the frozen background reference therefore captures and
            # thresholds at ROI resolution — self-consistent, because detect()
            # always receives the ROI (during capture and after).
            roi  = profile.crop(frame)
            grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # ── 1. Detection (every K-th frame; every frame until ready) ───
            # During background capture / warm-up we detect every frame so the
            # reference fills; afterwards only every K-th frame runs the heavy
            # CV stack. The intervening frames get a cheap predict-only tracker
            # pass so voice parameters keep streaming at the full frame rate.
            do_detect = (not warmed_up) or (frame_count % DETECT_K == 0)
            blobs = None
            if do_detect:
                lr = (MOG2_POST_WARMUP_LR if warmed_up else -1)
                try:
                    blobs = detector.detect(roi, learning_rate=lr, grey=grey)
                except TypeError:
                    blobs = detector.detect(roi)
                last_blobs = blobs

            tracked = []
            leaves  = []
            root    = None

            if warmed_up:
                # ── 2. Tracking ───────────────────────────────────────────
                # Detection frames pass `blobs` (full association + Kalman
                # correction); intervening frames pass None (predict-only).
                tracked = tracker.update(blobs, roi, grey=grey)

                # ── 3. Merge tree (visualisation + governor; detection frames) ──
                if (show_tree or use_governor) and blobs is not None:
                    # v4.7: pass detection's own wax mask so the tree prunes
                    # peaks that are not inside a detected blob. Sensor grain,
                    # glass reflections and backlight hot spots are genuine
                    # local maxima of the luminance field, so persistence
                    # cannot reject them -- but detection already segments the
                    # wax, and the tree inherits that here.
                    # v4.9: pass `roi` (the object detect() received), NOT
                    # `frame`. With the full frame the identity-keyed mask
                    # cache missed, the colour mask was built at full-frame
                    # size against an ROI-sized foreground mask, the
                    # bitwise_and raised, and the except below left
                    # wax_mask = None -- so pruning never ran.
                    try:
                        wax_mask = detector.get_debug_masks(roi, grey=grey)["combined"]
                    except Exception as exc:
                        wax_mask = None
                        if not getattr(main, "_mask_warned", False):
                            print(f"  ! wax-mask for merge tree unavailable: {exc}")
                            main._mask_warned = True
                    root, leaves = compute_merge_tree(
                        grey, blobs=blobs,
                        downscale=DOWNSCALE,
                        persistence_threshold=detector.pers_thresh * 0.8,
                        mask=wax_mask,
                    )

                # ── 3b. Score governor -> ScoreState (v4.2) ──────────────
                if use_governor:
                    score_state = governor.update(
                        tracked, frame_interval, tracker.events,
                        tree_root=root if blobs is not None else None,
                        leaves=leaves if blobs is not None else None,
                        diagram=[(b.birth, b.death) for b in (blobs or last_blobs or [])],
                    )
                    voices.score_state = score_state

                    # ── 3c. Pattern (hit) layer -> timed OSC bundles ────────
                    # This is what produces an audible BEAT. Without it the
                    # elastic clock computes a tempo that nothing consumes and
                    # only the sustained \mpeVoice sounds, which reads as a
                    # drone however the governor weights are set.
                    if pattern_layer:
                        try:
                            msgs = governor.tick_patterns(
                                tracked,
                                voice_of=lambda bid: voices.voices_by_blob.get(bid),
                            )
                            # v4.9.4: honour the scheduler's timestamps (see
                            # send_timed_hits) instead of sending at once.
                            send_timed_hits(voices.osc, msgs, governor.t, time.time())
                        except Exception as exc:
                            if not getattr(dispatch_controller_event, "_pat_warned", False):
                                print(f"  ! pattern layer error: {exc}")
                                dispatch_controller_event._pat_warned = True

                # ── 4. Lifecycle: tracker events → BindingManager ────────
                binding.handle_tracker_events(tracker.events, tracked)

                # ── 5. Player input → controller events ───────────────────
                if keyboard is not None:
                    cevs = keyboard.poll()
                    for ev in cevs:
                        if ev.type == QUIT:
                            quit_requested = True
                            continue
                        dispatch_controller_event(ev, voices, binding, tracked, rave=rave)

                # ── 6. Per-frame voice work (routing, smooth, send) ──────
                voices.process_frame(tracked)

                # ── 6b. Melody / rhythm layer (v4.9): hazard onsets + band pitch ──
                # Runs AFTER process_frame so each note copies the bound
                # voice's smoothed timbre (cutoff/res/wave/detune/drive) and
                # the blob's melody sounds like its drone.
                for nev in melody.process_frame(tracked):
                    print(f"  . note blob #{nev.blob_id} S{nev.slot_id:02d} "
                          f"m{nev.midi} {nev.freq:.0f}Hz dur={nev.dur:.2f}")

                # ── 6c. v4.10: voice extras, master space, pre-sound ─────
                # After process_frame so the extras address this frame's
                # nodes; reads ScoreState (None with the governor off, which
                # leaves the master at its neutral room and the pre-sound silent).
                try:
                    rig.update(voices.score_state if use_governor else None, tracked)
                except Exception as exc:
                    if not getattr(main, "_rig_warned", False):
                        print(f"  ! synth palette error: {exc}")
                        main._rig_warned = True

                # ── 6d. v4.11: RAVE layer housekeeping (status-file poll) ─
                # The layer only sends OSC on events; this re-reads
                # rave_status.json when SC rewrites it so the tab stays true.
                if rave is not None:
                    try:
                        rave.update()
                    except Exception as exc:
                        if not getattr(main, "_rave_warned", False):
                            print(f"  ! rave layer error: {exc}")
                            main._rave_warned = True

                # ── 7. Log lifecycle events ──────────────────────────────
                for event in tracker.events:
                    ev_type, ev_id = event[0], event[1]
                    blob = next((b for b in tracked if b.id == ev_id), None)
                    dir_str = f" [{blob.direction}]" if blob else ""
                    sym = "+" if ev_type == "birth" else "-"
                    if ev_type in ("birth", "death"):
                        slot = voices.slot_for_blob(ev_id) if ev_type == "birth" else None
                        slot_str = f" -> S{slot:02d}" if slot is not None else ""
                        print(f"  {sym} Voice {ev_type}: ID {ev_id}{dir_str}{slot_str}")
                    elif ev_type == "merge":
                        print(f"  ~ merge {event[1]} -> {event[2]}")
                    elif ev_type == "split":
                        print(f"  ~ split {event[1]} -> {event[2]}")

            # ── 8. Debug overlay ──────────────────────────────────────────
            if SHOW_WINDOW and not headless:
                overlay = frame.copy()
                origin  = (x1, y1)   # lamp ROI top-left (ROI-local -> full frame)

                if show_contours:
                    draw_blob_contours(overlay, last_blobs, origin=origin)
                if show_tree and root is not None:
                    draw_tree_overlay(overlay, root, origin=origin)
                if show_track:
                    draw_voice_overlay(overlay, tracked, voices,
                                       keyboard.selected_slot if keyboard else 0,
                                       origin=origin,
                                       melody=melody if melody.enabled else None)
                if show_flow:
                    draw_flow_overlay(overlay, tracker.get_flow(), origin=origin)

                # Lamp ROI border (the region everything now runs on)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (60, 60, 120), 1)

                fps_actual = 1.0 / max(time.time() - t0, 1e-6)
                if not warmed_up:
                    if BACKGROUND_MODE == "frozen":
                        # v4.9: background_status() reports mode, progress and
                        # ETA (the warm capture is wall-clock paced, so a frame
                        # count alone would be misleading).
                        st = detector.background_status()
                        if not st:
                            done, tot = detector.background_progress()
                            st = f"BG capture {done}/{tot}"
                        warm_note = f"  {st}"
                    else:
                        warm_note = (f"  MOG2 warm-up "
                                     f"{int(elapsed)}s/{MOG2_WARMUP_SECS}s")
                else:
                    warm_note = ""
                draw_status(overlay, voices, leaves, last_blobs,
                            fps_actual, warm_note,
                            {"T": show_track, "M": show_tree,
                             "F": show_flow,  "C": show_masks,
                             "O": show_contours, "V": show_hud,
                             "N": melody.enabled,
                             "A": rave is not None and rave.enabled},   # v4.11
                            detector.pers_thresh,
                            extra=melody.hud_line()
                                  + (f"  {tracker.hud_line()}" if use_stable else ""),
                            extra2=rig.hud_line()
                                   + (f"  {rave.hud_line()}" if rave is not None else ""))

                draw_voice_hud(overlay, voices, keyboard, binding,
                               show_hud=show_hud)

                cv2.imshow("Audible Lava v4.13 (MPE)", overlay)

                # v4.9.6: the frozen background reference -- the "empty lamp"
                # image every frame is differenced against. Shown next to the
                # current frame and their difference so wax baked into the
                # reference (ghosts) or missing from it stands out. Saved once
                # per toggle to background_reference.png for closer inspection.
                if show_ref:
                    ref = getattr(getattr(detector, "bg_ref", None), "reference", None)
                    if ref is None:
                        blank = np.zeros_like(grey)
                        cv2.putText(blank, "reference not captured yet", (8, 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
                        cv2.imshow("Background reference", blank)
                    else:
                        if ref.shape != grey.shape:
                            ref = cv2.resize(ref, (grey.shape[1], grey.shape[0]))
                        diff = cv2.absdiff(grey, ref)
                        diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
                        cv2.imshow("Background reference", np.hstack([ref, grey, diff]))
                        if not getattr(main, "_ref_saved", False):
                            cv2.imwrite("background_reference.png", ref)
                            print("[Background] reference shown (left: reference, "
                                  "middle: current frame, right: difference); "
                                  "saved to background_reference.png")
                            main._ref_saved = True
                else:
                    try: cv2.destroyWindow("Background reference")
                    except: pass
                    main._ref_saved = False

                if show_masks:
                    masks = detector.get_debug_masks(roi, grey=grey)
                    cv2.imshow("Foreground (bg-sub)", masks["foreground"])
                    cv2.imshow("Wax colour mask",     masks["colour"])
                    cv2.imshow("Combined mask",       masks["combined"])
                else:
                    for w_name in ("Foreground (bg-sub)",
                                   "Wax colour mask", "Combined mask"):
                        try: cv2.destroyWindow(w_name)
                        except: pass

                key = cv2.waitKey(1) & 0xFF
                if   key == ord("q"):                           break
                elif key == ord("m"): show_tree     = not show_tree
                elif key == ord("t"): show_track    = not show_track
                elif key == ord("f"): show_flow     = not show_flow
                elif key == ord("c"): show_masks    = not show_masks
                elif key == ord("r"): show_ref      = not show_ref
                elif key == ord("o"): show_contours = not show_contours
                elif key == ord("v"): show_hud      = not show_hud
                elif key == ord("n"):
                    print(f"[Melody] {'on' if melody.toggle() else 'off'}")
                elif key == ord("x"):
                    if rig.master is not None:
                        print(f"[Space] master insert "
                              f"{'BYPASSED' if rig.toggle_master_bypass() else 'active'}")
                    else:
                        print("[Space] master insert not installed (--no-space)")
                elif key == ord("a"):                           # v4.11
                    if rave is not None:
                        rave.toggle_enabled()
                        print(f"[RAVE] {rave.last_msg}  active models: "
                              f"{', '.join(m.key for m in rave.active_models()) or 'none'}")
                    else:
                        print("[RAVE] layer not installed (--no-rave)")
                elif key == ord("]"):
                    detector.pers_thresh = min(
                        detector.pers_thresh + PERS_STEP, PERS_MAX)
                    print(f"[Thresh] persistence_threshold "
                          f"-> {detector.pers_thresh:.1f}")
                elif key == ord("["):
                    detector.pers_thresh = max(
                        detector.pers_thresh - PERS_STEP, PERS_MIN)
                    print(f"[Thresh] persistence_threshold "
                          f"-> {detector.pers_thresh:.1f}")
                elif key == ord("b"):
                    detector.reset_background()
                    if BACKGROUND_MODE == "frozen":
                        if detector.bg_ref_mode == "warm":
                            print(f"[Background] re-capturing WARM reference "
                                  f"({BG_REF_CAPTURE_SECS:.0f}s) -- lamp may keep running")
                        else:
                            print("[Background] re-capturing cold reference -- "
                                  "keep the lamp's wax settled while it fills")
                    else:
                        print("[Background] MOG2 model reset")
                elif key == ord("w"):
                    if BACKGROUND_MODE == "frozen":
                        detector.reset_background(mode="warm")
                        print(f"[Background] re-capturing WARM reference "
                              f"({BG_REF_CAPTURE_SECS:.0f}s, p{BG_REF_PERCENTILE:.0f}) "
                              f"-- the lamp may keep running")
                    else:
                        print("[Background] 'w' only applies to BACKGROUND_MODE='frozen'")

            # ── Frame rate ─────────────────────────────────────────────────
            slp = frame_interval - (time.time() - t0)
            if slp > 0:
                time.sleep(slp)
            frame_count += 1

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        print("Releasing voices...")
        voices.release_all()
        # v4.10: let the voices' releases pass through the master before it is
        # freed (the pre-sound pads are gated first so they fade with them).
        if rig.presound is not None:
            rig.presound.close()
        # v4.11: gate the RAVE model nodes now (they fade and free themselves);
        # the split stays up until the fades are through so the dry mix and
        # the fades keep passing to the master.
        if rave is not None:
            rave.close()
        time.sleep(3.5)
        rig.close()
        if rave is not None:
            rave.free_split()
        cap.release()
        if SHOW_WINDOW and not headless:
            cv2.destroyAllWindows()
        if keyboard is not None:
            keyboard.close()
        print(f"Done. {frame_count} frames processed.")


if __name__ == "__main__":
    main()
