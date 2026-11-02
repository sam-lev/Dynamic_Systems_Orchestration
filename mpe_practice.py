#!/usr/bin/env python3
# mpe_practice.py -- Audible Lava / DSO v4.1
#
# An interactive console harness for practising the MPE voice system WITHOUT
# the camera / blob pipeline. It drives the real VoiceManager and the
# \mpeVoice SynthDef over OSC, so what you hear is exactly the synthesis path
# the lamp drives -- you're just the modulation source instead of the wax.
#
# It mirrors the keyboard controller's two activities:
#   PERFORMANCE  -- allocate a voice slot and play a note on it
#   EDIT         -- shape that voice's sound (filter, envelope, wavetable,
#                   and the v4.1 expressive params: detune / shimmer / noise /
#                   drive / filter-morph)
#
# ─ Requirements ───────────────────────────────────────────────────────────────
#   SuperCollider must be running with blob_synth.scd evaluated, so scsynth is
#   listening on 127.0.0.1:57110 and the \mpeVoice SynthDef + wavetable buffers
#   are loaded. (In SC: open blob_synth.scd, evaluate the whole file.)
#   OSC is fire-and-forget UDP, so this script can't detect whether SC is up --
#   if you hear nothing, that's the first thing to check.
#
# ─ Usage ──────────────────────────────────────────────────────────────────────
#   python mpe_practice.py                 # interactive REPL
#   python mpe_practice.py --lesson        # run the guided tour, then REPL
#   python mpe_practice.py --port 57110    # override scsynth port
#   python mpe_practice.py --bufbase 0     # set VOsc wavetable base buffer id
#
#   Then, at the prompt:
#       new                 allocate + select a voice slot   (PERFORMANCE)
#       note C3             play C3 on the selected slot      (PERFORMANCE)
#       edit cutoff 4000    open the filter                   (EDIT)
#       edit detune 0.6     spread the unison                 (EDIT)
#       status              show the selected voice's state
#       off                 release the note
#       lesson              guided walkthrough of every sound dimension
#       help                full command list
#       quit
#
# Type 'help' once running for the full command + parameter reference.

from __future__ import annotations
import sys
import re
import time
import threading

from lamp_profile  import LampProfile
from voice_manager import VoiceManager, VoiceState, SynthState, midi_to_freq

try:
    from routing import Source, Dest, Curve   # noqa: F401  (used by 'route')
except Exception:
    Source = Dest = Curve = None


# ── Note-name parsing ─────────────────────────────────────────────────────────
_NOTE_BASE = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_NOTE_RE   = re.compile(r"^([A-Ga-g])([#sb]?)(-?\d+)$")
_MIDI_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def parse_note(tok: str):
    """'C4' / 'A#3' / 'Eb5' / '60' -> MIDI note int (A4 = 69 = 440 Hz). None on error."""
    tok = tok.strip()
    if re.fullmatch(r"-?\d+", tok):
        n = int(tok)
        return n if 0 <= n <= 127 else None
    m = _NOTE_RE.match(tok)
    if not m:
        return None
    letter, acc, octave = m.group(1).lower(), m.group(2), int(m.group(3))
    semitone = _NOTE_BASE[letter] + ({"#": 1, "s": 1, "b": -1}.get(acc, 0))
    midi = (octave + 1) * 12 + semitone
    return midi if 0 <= midi <= 127 else None


def midi_name(n: float) -> str:
    n = int(round(n))
    return f"{_MIDI_NAMES[n % 12]}{n // 12 - 1}"


# ── Editable-parameter table (harness-side) ───────────────────────────────────
# Each entry: name -> (mechanism, clip_range)
#   "param"  : routed through VoiceManager.set_param (the real EDIT path)
#   "base"   : set the SynthState base_<x> field directly (params set_param
#              doesn't expose yet: the v4.1 expressive timbre controls)
EDIT_PARAMS = {
    # filter
    "cutoff":  ("param", (40.0, 16000.0)),
    "res":     ("param", (0.05, 1.0)),
    "fmode":   ("param", (0.0, 1.0)),      # 0 LP · 0.5 BP · 1 HP
    # envelope
    "attack":  ("param", (0.005, 8.0)),
    "decay":   ("param", (0.005, 4.0)),
    "sustain": ("param", (0.0, 1.0)),
    "release": ("param", (0.005, 12.0)),
    # level / space
    "amp":     ("param", (0.0, 1.0)),
    "pan":     ("param", (-1.0, 1.0)),
    # wavetable
    "wave_bank": ("param", (0, 3)),
    "wave_pos":  ("param", (0.0, 1.0)),    # NB: set_param clips to [0,1]; use
                                           # the 'wave' command for full 0..3
    "bend":      ("param", (-200.0, 200.0)),
    # v4.1 expressive timbre (driven by blob motion in the full system;
    # set_param doesn't expose these, so we write the base_ fields directly)
    "detune":  ("base", (0.0, 1.0)),       # 3-osc unison spread / beating
    "shimmer": ("base", (0.0, 1.0)),       # vibrato + tremolo depth
    "noise":   ("base", (0.0, 1.0)),       # breath / air blend
    "drive":   ("base", (0.0, 1.0)),       # wavefold saturation
}
_BASE_ATTR = {"detune": "base_detune", "shimmer": "base_shimmer",
              "noise": "base_noise", "drive": "base_drive"}


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


# ── The harness ───────────────────────────────────────────────────────────────
class MPEPractice:
    def __init__(self, sc_ip="127.0.0.1", sc_port=57110, bufbase=None):
        # A default profile is fine: keyboard voices don't use lamp geometry.
        profile = LampProfile.default()
        self.voices = VoiceManager(sc_ip=sc_ip, sc_port=sc_port,
                                   n_slots=16, profile=profile)
        self.bufbase = bufbase
        self._lock = threading.Lock()
        self._running = True

        # Background tick: pushes smoothed parameter values to SC every ~20 ms
        # (this is what makes edits glide in on a held note). Mirrors the role
        # of audible_lava's per-frame voices.process_frame() call.
        self._ticker = threading.Thread(target=self._tick_loop, daemon=True)
        self._ticker.start()

    # ── background sender ─────────────────────────────────────────────────
    def _tick_loop(self):
        while self._running:
            try:
                with self._lock:
                    self.voices.process_frame([])   # no blobs; just smooth+send
            except Exception:
                pass
            time.sleep(0.02)

    # ── voice lifecycle ───────────────────────────────────────────────────
    def _ensure_voice(self, slot: int) -> bool:
        v = self.voices.voice(slot)
        if v is None:
            return False
        if v.state == VoiceState.IDLE:
            v.synth        = SynthState.default()
            v.routing      = []
            v.events       = []
            v.state        = VoiceState.KEYBOARD_ONLY
            v.blob_id      = None
            v.born_at      = time.time()
            v.spawn_origin = "keyboard"
        return True

    @property
    def sel(self) -> int:
        return self.voices.selected_slot

    def cmd_new(self, args):
        slot = self.voices.first_idle_slot()
        if slot is None:
            print("  ! voice pool full (16/16). Use 'panic' or pick a slot with 'sel'.")
            return
        self._ensure_voice(slot)
        self.voices.select_slot(slot)
        print(f"  + slot {slot:02d} created (KEYBOARD_ONLY) and selected. "
              f"Play it with e.g. 'note C3'.")

    def cmd_sel(self, args):
        if not args:
            print(f"  selected slot = {self.sel:02d}");  return
        try:
            n = int(args[0])
        except ValueError:
            print("  usage: sel <0-15>");  return
        if not (0 <= n < self.voices.n_slots):
            print("  ! slot out of range 0-15");  return
        self.voices.select_slot(n)
        print(f"  selected slot = {n:02d}")

    def _note_on(self, slot: int, midi: int, vel: int = 100):
        """Performance: (re)spawn a fresh SC node for this slot and gate it on.

        We spawn per note-on rather than re-gating one persistent node so a
        retrigger always sounds even after a previous note has released and
        freed itself. Current edits are frozen into the new node so they carry
        across notes with no transient.
        """
        if not self._ensure_voice(slot):
            print("  ! bad slot");  return
        v = self.voices.voice(slot)
        s = v.synth
        # release any node currently sounding on this slot
        if v.sc_node is not None and s.gate == 1:
            try:
                self.voices.osc.send_message("/n_set", [v.sc_node, "gate", 0])
            except Exception:
                pass
        s.base_note      = float(midi)
        s.base_amp       = _clip(vel / 127.0, 0.05, 0.6)
        s.gate           = 1
        v.last_played_at = time.time()
        # bring targets up to date from the (possibly edited) bases, then snap
        # current = target so the spawn packet starts at the right values
        self.voices._compute_targets(v, {})
        s.freeze_current_to_target()
        self.voices._send_spawn(v, blob=None, fire_strike=False, gate=1)
        if self.bufbase is not None and v.sc_node is not None:
            try:
                self.voices.osc.send_message(
                    "/n_set", [v.sc_node, "bufBase", int(self.bufbase)])
            except Exception:
                pass

    def cmd_note(self, args):
        if not args:
            print("  usage: note <pitch> [velocity]   e.g. 'note C3'  'note 60 110'")
            return
        midi = parse_note(args[0])
        if midi is None:
            print(f"  ! couldn't parse pitch {args[0]!r} (try C3, A#4, Eb2, or 0-127)")
            return
        vel = 100
        if len(args) > 1:
            try: vel = int(args[1])
            except ValueError: pass
        with self._lock:
            self._note_on(self.sel, midi, vel)
        print(f"  ♪ slot {self.sel:02d}: {midi_name(midi)} "
              f"({midi_to_freq(midi):.1f} Hz, vel {vel}) — sustaining; 'off' to release")

    def cmd_play(self, args):
        """note-on, hold for a duration, note-off — quick audition."""
        if not args:
            print("  usage: play <pitch> [seconds]   e.g. 'play C3 1.5'");  return
        midi = parse_note(args[0])
        if midi is None:
            print(f"  ! bad pitch {args[0]!r}");  return
        dur = 1.2
        if len(args) > 1:
            try: dur = float(args[1])
            except ValueError: pass
        with self._lock:
            self._note_on(self.sel, midi, 100)
        print(f"  ♪ {midi_name(midi)} for {dur:.1f}s …")
        time.sleep(dur)
        self.cmd_off([])

    def cmd_off(self, args):
        v = self.voices.voice(self.sel)
        if v is None or v.sc_node is None:
            print("  (nothing to release)");  return
        with self._lock:
            try:
                self.voices.osc.send_message("/n_set", [v.sc_node, "gate", 0])
            except Exception:
                pass
            v.synth.gate = 0
        print(f"  □ slot {self.sel:02d} released")

    # ── editing ───────────────────────────────────────────────────────────
    def _apply_edit(self, param, value, relative=False, reset=False):
        slot = self.sel
        v = self.voices.voice(slot)
        if v is None or v.state == VoiceState.IDLE:
            print("  ! no voice on the selected slot — 'new' first");  return None
        mech, (lo, hi) = EDIT_PARAMS[param]
        with self._lock:
            if mech == "param":
                self.voices.set_param(slot, param, value,
                                      relative=relative, reset=reset)
                cur = self._read_param(v, param)
            else:  # direct base field (timbre)
                attr = _BASE_ATTR[param]
                if reset:
                    new = getattr(SynthState.default(), attr)
                else:
                    base = getattr(v.synth, attr)
                    new  = _clip((base + value) if relative else value, lo, hi)
                setattr(v.synth, attr, new)
                cur = new
        return cur

    def _read_param(self, v, param):
        s = v.synth
        return {
            "cutoff": s.base_cutoff, "res": s.base_res, "fmode": s.fmode,
            "attack": s.base_attack, "decay": s.base_decay,
            "sustain": s.base_sustain, "release": s.base_release,
            "amp": s.base_amp, "pan": s.base_pan,
            "wave_bank": s.wave_bank, "wave_pos": s.base_wave_pos,
            "bend": s.bend,
        }.get(param)

    def cmd_edit(self, args):
        if len(args) < 2:
            print("  usage: edit <param> <value>   (absolute)   e.g. 'edit cutoff 4000'")
            print("         params: " + ", ".join(EDIT_PARAMS))
            return
        param = args[0].lower()
        if param not in EDIT_PARAMS:
            print(f"  ! unknown param {param!r}. valid: {', '.join(EDIT_PARAMS)}");  return
        try:
            value = float(args[1])
        except ValueError:
            print(f"  ! {args[1]!r} is not a number");  return
        cur = self._apply_edit(param, value, relative=False)
        if cur is not None:
            print(f"  ~ {param} = {cur:g}   (slot {self.sel:02d})")

    def cmd_bump(self, args):
        if len(args) < 2:
            print("  usage: bump <param> <delta>   (relative)   e.g. 'bump cutoff +500'")
            return
        param = args[0].lower()
        if param not in EDIT_PARAMS:
            print(f"  ! unknown param {param!r}");  return
        try:
            delta = float(args[1])
        except ValueError:
            print(f"  ! {args[1]!r} is not a number");  return
        cur = self._apply_edit(param, delta, relative=True)
        if cur is not None:
            print(f"  ~ {param} {'+' if delta >= 0 else ''}{delta:g} -> {cur:g}")

    def cmd_reset(self, args):
        if not args:
            print("  usage: reset <param>   (or 'reset all')");  return
        if args[0].lower() == "all":
            for p in EDIT_PARAMS:
                self._apply_edit(p, 0.0, reset=True)
            print("  ~ all params reset to defaults");  return
        param = args[0].lower()
        if param not in EDIT_PARAMS:
            print(f"  ! unknown param {param!r}");  return
        cur = self._apply_edit(param, 0.0, reset=True)
        if cur is not None:
            print(f"  ~ {param} reset -> {cur:g}")

    def cmd_wave(self, args):
        """Select wavetable: wave <bank 0-3> <index 0-3>.

        16 tables = 4 banks x 4 waves:
          bank0 analogue (saw/square/pulse/tri) · bank1 harmonic sweeps
          bank2 formants (ah/eh/ih/oh)          · bank3 metallic/bell
        (Sets base_wave_pos directly because set_param clips wave_pos to [0,1],
        which would only reach the first two waves of a bank.)
        """
        if len(args) < 2:
            print("  usage: wave <bank 0-3> <index 0-3>   e.g. 'wave 2 1' (formant 'eh')")
            return
        v = self.voices.voice(self.sel)
        if v is None or v.state == VoiceState.IDLE:
            print("  ! no voice — 'new' first");  return
        try:
            bank, idx = int(args[0]), float(args[1])
        except ValueError:
            print("  ! bank and index must be numbers");  return
        with self._lock:
            self.voices.set_param(self.sel, "wave_bank", _clip(bank, 0, 3))
            v.synth.base_wave_pos = _clip(idx, 0.0, 3.999)
        print(f"  ~ wave bank {int(_clip(bank,0,3))}, index {idx:g}")

    # ── inspection ────────────────────────────────────────────────────────
    def cmd_status(self, args):
        v = self.voices.voice(self.sel)
        if v is None:
            print("  ! bad slot");  return
        s = v.synth
        print(f"  ── slot {self.sel:02d}  [{v.state.name}]  gate={s.gate} ──")
        print(f"     pitch  base_note={midi_name(s.base_note)} ({s.base_note:.0f})  "
              f"sent {s.c_freq:7.1f} Hz   bend={s.bend:g}")
        print(f"     amp    base={s.base_amp:.2f}  sent={s.c_amp:.2f}      "
              f"pan base={s.base_pan:+.2f} sent={s.c_pan:+.2f}")
        print(f"     filter cutoff base={s.base_cutoff:7.0f} sent={s.c_cutoff:7.0f}  "
              f"res={s.base_res:.2f}  fmode={s.fmode:.2f} "
              f"({'LP' if s.fmode < .25 else 'HP' if s.fmode > .75 else 'BP'})")
        print(f"     wave   bank={s.wave_bank} index(base_wave_pos)={s.base_wave_pos:.2f} "
              f"sent={s.c_wave_pos:.2f}")
        print(f"     env    A={s.base_attack:.2f} D={s.base_decay:.2f} "
              f"S={s.base_sustain:.2f} R={s.base_release:.2f}")
        print(f"     timbre detune={s.base_detune:.2f} shimmer={s.base_shimmer:.2f} "
              f"noise={s.base_noise:.2f} drive={s.base_drive:.2f}")

    def cmd_list(self, args):
        print("  slot  state          gate   freq")
        for i in range(self.voices.n_slots):
            v = self.voices.voice(i)
            star = "*" if i == self.sel else " "
            if v.state == VoiceState.IDLE:
                print(f" {star}{i:02d}   IDLE")
            else:
                print(f" {star}{i:02d}   {v.state.name:<13} "
                      f"{v.synth.gate}     {v.synth.c_freq:7.1f} Hz")

    def cmd_panic(self, args):
        with self._lock:
            try: self.voices.release_all()
            except Exception: pass
            try: self.voices.osc.send_message("/g_freeAll", [0])
            except Exception: pass
        print("  ! panic — all voices released")

    def cmd_bufbase(self, args):
        if not args:
            print(f"  bufbase = {self.bufbase}  (None = SynthDef default 0)");  return
        try:
            self.bufbase = int(args[0])
        except ValueError:
            print("  usage: bufbase <int>");  return
        v = self.voices.voice(self.sel)
        if v is not None and v.sc_node is not None:
            with self._lock:
                try:
                    self.voices.osc.send_message(
                        "/n_set", [v.sc_node, "bufBase", self.bufbase])
                except Exception:
                    pass
        print(f"  bufbase = {self.bufbase} (applied to new notes; "
              f"find the right value in SC with ~wtBuffers[0].bufnum)")

    # ── guided lesson ─────────────────────────────────────────────────────
    def cmd_lesson(self, args):
        def step(msg, cmd=None, hold=1.6):
            print(f"\n  ▶ {msg}")
            if cmd:
                print(f"      ({cmd})")
                self.do(cmd)
            time.sleep(hold)

        print("\n" + "=" * 64)
        print("  GUIDED TOUR — listen to each sound dimension in isolation")
        print("  (make sure SuperCollider + blob_synth.scd are running)")
        print("=" * 64)

        step("PERFORMANCE: allocate a voice and play a sustained C3.",
             "new", hold=0.4)
        self.do("note C3"); time.sleep(1.6)

        step("EDIT — open the low-pass filter (cutoff 400 -> 6000 Hz).",
             "edit cutoff 400", hold=0.8)
        for c in (900, 1800, 3500, 6000):
            self.do(f"edit cutoff {c}"); time.sleep(0.7)

        step("EDIT — raise resonance for a vocal peak at the cutoff.",
             "edit res 0.85")
        step("EDIT — morph the filter LP -> BP -> HP (fmode 0 -> 0.5 -> 1).",
             "edit fmode 0.0", hold=0.6)
        for f in (0.5, 1.0, 0.0):
            self.do(f"edit fmode {f}"); time.sleep(0.9)
        self.do("edit res 0.3")

        step("EDIT — sweep the wavetable banks (analogue/harmonic/formant/bell).",
             "edit cutoff 5000", hold=0.5)
        for bk, nm in [(0, "saw"), (1, "harmonic"), (2, "formant ah"), (3, "bell")]:
            self.do(f"wave {bk} 0"); print(f"      bank {bk}: {nm}"); time.sleep(1.1)
        self.do("wave 0 0")

        step("EDIT — DETUNE: spread 3 detuned oscillators (0 -> 0.8). Beating/chorus.",
             "edit detune 0.0", hold=0.5)
        for d in (0.25, 0.5, 0.8):
            self.do(f"edit detune {d}"); time.sleep(1.0)
        self.do("edit detune 0.0")

        step("EDIT — SHIMMER: vibrato + tremolo depth (0 -> 0.8).",
             "edit shimmer 0.0", hold=0.5)
        for sh in (0.3, 0.6, 0.85):
            self.do(f"edit shimmer {sh}"); time.sleep(1.1)
        self.do("edit shimmer 0.0")

        step("EDIT — DRIVE: tanh wavefold saturation (0 -> 0.8). Harmonic grit.",
             "edit drive 0.0", hold=0.5)
        for dr in (0.3, 0.6, 0.85):
            self.do(f"edit drive {dr}"); time.sleep(1.0)
        self.do("edit drive 0.0")

        step("EDIT — NOISE: pitch-tracked breath/air blend (0 -> 0.6).",
             "edit noise 0.0", hold=0.5)
        for ns in (0.25, 0.45, 0.6):
            self.do(f"edit noise {ns}"); time.sleep(1.0)
        self.do("edit noise 0.0")

        step("EDIT — slow envelope (attack 1.5s, release 3s), then retrigger.",
             "edit attack 1.5", hold=0.3)
        self.do("edit release 3.0")
        self.do("off"); time.sleep(0.4)
        self.do("note C3"); print("      …note swells in over 1.5s"); time.sleep(2.5)

        step("Done — releasing. Now try it yourself: 'new', 'note', 'edit', 'status'.",
             "off", hold=0.3)
        print("=" * 64)

    # ── help / dispatch ───────────────────────────────────────────────────
    def cmd_help(self, args):
        print(HELP_TEXT)

    _DISPATCH = {
        "new": "cmd_new", "sel": "cmd_sel", "select": "cmd_sel",
        "note": "cmd_note", "n": "cmd_note", "play": "cmd_play", "off": "cmd_off",
        "edit": "cmd_edit", "e": "cmd_edit", "bump": "cmd_bump", "b": "cmd_bump",
        "reset": "cmd_reset", "wave": "cmd_wave",
        "status": "cmd_status", "s": "cmd_status", "list": "cmd_list", "ls": "cmd_list",
        "panic": "cmd_panic", "bufbase": "cmd_bufbase",
        "lesson": "cmd_lesson", "help": "cmd_help", "?": "cmd_help",
    }

    def do(self, line: str):
        line = line.strip()
        if not line or line.startswith("#"):
            return
        parts = line.split()
        verb, args = parts[0].lower(), parts[1:]
        if verb in ("quit", "q", "exit"):
            raise EOFError
        meth = self._DISPATCH.get(verb)
        if meth is None:
            print(f"  ? unknown command {verb!r} — type 'help'")
            return
        getattr(self, meth)(args)

    def shutdown(self):
        self._running = False
        try:
            with self._lock:
                self.voices.release_all()
                self.voices.osc.send_message("/g_freeAll", [0])
        except Exception:
            pass


HELP_TEXT = """
  ── PERFORMANCE (play notes) ────────────────────────────────────────────────
    new                     allocate the next free voice slot and select it
    sel <0-15>              select which slot subsequent commands act on
    note <pitch> [vel]      play & SUSTAIN a note (C3, A#4, Eb2, or 0-127)
    play <pitch> [secs]     play a note for N seconds then release
    off                     release the selected slot's note
    panic                   release everything

  ── EDIT (shape the selected voice's sound) ─────────────────────────────────
    edit <param> <value>    set a parameter (absolute)
    bump <param> <delta>    nudge a parameter (relative, e.g. 'bump cutoff +500')
    reset <param> | all     restore default(s)
    wave <bank> <index>     pick wavetable: banks 0-3, index 0-3 (16 total)

    parameters:
      cutoff  40-16000 Hz      res     0.05-1.0       fmode  0=LP 0.5=BP 1=HP
      attack  0.005-8 s        decay   0.005-4 s      sustain 0-1
      release 0.005-12 s       amp     0-1            pan    -1..+1
      bend    cents            wave_bank 0-3          wave_pos 0-1 (see 'wave')
      detune  0-1 (beating)    shimmer 0-1 (vibrato)  noise  0-1 (breath)
      drive   0-1 (wavefold)

  ── INSPECT / UTILITY ───────────────────────────────────────────────────────
    status                  full parameter readout for the selected voice
    list                    all 16 slots (state / gate / pitch)
    bufbase <int>           VOsc wavetable base buffer id (if waves sound wrong)
    lesson                  guided tour of every sound dimension
    help                    this text          quit
"""


def main():
    argv = sys.argv[1:]
    port = 57110
    bufbase = None
    run_lesson = "--lesson" in argv
    if "--port" in argv:
        try: port = int(argv[argv.index("--port") + 1])
        except (ValueError, IndexError): pass
    if "--bufbase" in argv:
        try: bufbase = int(argv[argv.index("--bufbase") + 1])
        except (ValueError, IndexError): pass

    app = MPEPractice(sc_port=port, bufbase=bufbase)

    print("=" * 64)
    print("  Audible Lava / DSO v4.1 — MPE practice console")
    print("=" * 64)
    print(f"  OSC -> 127.0.0.1:{port}   (scsynth must be running with")
    print("         blob_synth.scd evaluated — \\mpeVoice + wavetables loaded)")
    print()
    print("  Quick start:")
    print("     new            -> note C3      -> edit cutoff 4000")
    print("     edit detune 0.6  -> edit drive 0.5  -> status  -> off")
    print()
    print("  Type 'help' for all commands, 'lesson' for a guided tour, 'quit' to exit.")
    print("=" * 64)

    if run_lesson:
        app.cmd_lesson([])

    try:
        while True:
            try:
                line = input(f"[slot {app.sel:02d}] > ")
            except EOFError:
                break
            try:
                app.do(line)
            except EOFError:
                break
            except Exception as exc:
                print(f"  ! error: {exc}")
    finally:
        app.shutdown()
        print("\n  bye — voices released.")


if __name__ == "__main__":
    main()
