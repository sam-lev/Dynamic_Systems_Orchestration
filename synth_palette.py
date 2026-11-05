# synth_palette.py -- Audible Lava / DSO v4.10 "Rich Synth Palette"
#
# Additive wiring module. Nothing in voice_manager / score_governor /
# pattern_language / melody_rhythm is modified: this file reaches the SynthDef
# names they already expose (VoiceManager.mpe_synthdef & co.,
# governor.scheduler.synthdef, MelodyEngine.synthdef), swaps the governor's
# PatternScheduler for a subclass that adds per-layer parameters, subclasses
# MelodyEngine to add per-note physical-model parameters, and runs three new
# per-frame layers that read ScoreState and the tracked blobs:
#
#   VoiceExtras   per-voice extras of \mpeVoiceRich that no routing covers
#                   height        -> formant   (vowel register: oh at the
#                                               bottom of the lamp, ee at the top)
#                   area          -> sub       (big blob = sub-octave weight)
#                   |vx|          -> chorus    (sideways drift widens the image)
#   MasterSpace   \lavaMaster at the tail of group 0, driven by ScoreState
#                   ambiguity          -> t60       (harmonic ambiguity = haze)
#                   1 - tension        -> damp      (tension = bright tail)
#                   phrase_position    -> wet       (the phrase dissolves toward
#                                                    its boundary)
#                   surprisal          -> Greyhole feedback
#                   tension            -> Greyhole send, pre-reverb drive
#                   1 - kuramoto       -> modDepth  (incoherent motion = drift)
#                   count_smooth       -> size      (fewer blobs = bigger room)
#                   register_center    -> highcut
#                   boundary           -> t_bloom / bloomAmt (one swell per
#                                         grouping boundary, by its strength)
#   PreSoundLayer \preSoundRich x3 holding the PREDICTED triad at a gain equal
#                 to the harmony posterior mass on non-current candidates.
#                 This finally wires \preSound, which blob_synth_hits.scd has
#                 defined since v4.2 but nothing ever spawned.
#
# Hit models (\mpeHitRich `model`): the ornamental hit layer (low prominence
# rank, busy Euclidean patterns) is rendered as a PLUCKED STRING and the paired
# articulated head voice (octave below the sustained note) as a PIANO HAMMER,
# so the time-span hierarchy is audible as an instrumentation, not only as a
# density difference. Melody notes (\melodyHitRich) are BOWED when the blob is
# expanding (legato release) and PLUCKED when it is shrinking (staccato), bow
# pressure from blob speed.
#
# Everything here degrades gracefully: with `--classic-synths` (or
# palette=CLASSIC) the original SynthDef names are used and no extras are sent;
# with sc3-plugins missing, blob_synth_rich.scd substitutes stock UGens.
#
# Tests: test_synth_palette.py (no SuperCollider, camera or pygame).

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from routing import Source, extract_source_value
from pattern_language import PatternScheduler
from melody_rhythm import MelodyEngine


# ── Palettes ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Palette:
    name:     str
    mpe:      str
    angle:    str
    strike:   str
    bass:     str
    hit:      str
    melody:   str
    presound: str
    master:   str
    rich:     bool = False          # send extras / models / physical params


CLASSIC = Palette("classic", "mpeVoice", "blobAngle", "blobStrike", "bassStutter",
                  "mpeHit", "melodyHit", "preSound", "lavaMaster", rich=False)

RICH = Palette("rich", "mpeVoiceRich", "blobAngleRich", "blobStrikeRich",
               "bassStutterRich", "mpeHitRich", "melodyHitRich", "preSoundRich",
               "lavaMaster", rich=True)

PALETTES: Dict[str, Palette] = {"classic": CLASSIC, "rich": RICH}

# \mpeHitRich model ids
HIT_MODEL_WAVETABLE = 0
HIT_MODEL_PLUCKED   = 1
HIT_MODEL_PIANO     = 2
# \melodyHitRich model ids
MEL_MODEL_AUTO      = -1
MEL_MODEL_WAVETABLE = 0
MEL_MODEL_PLUCKED   = 1
MEL_MODEL_BOWED     = 2


def apply_palette(voices, palette: Palette, governor=None, melody=None) -> Palette:
    """Point the existing engines at the palette's SynthDef names.

    VoiceManager keeps its four names as plain attributes; the governor owns a
    PatternScheduler with a `synthdef` attribute; MelodyEngine has `synthdef`.
    For the rich palette the governor's scheduler is replaced by a
    RichPatternScheduler that injects the per-layer hit model.
    """
    voices.mpe_synthdef    = palette.mpe
    voices.angle_synthdef  = palette.angle
    voices.strike_synthdef = palette.strike
    voices.bass_synthdef   = palette.bass
    if governor is not None and getattr(governor, "scheduler", None) is not None:
        sched = governor.scheduler
        if palette.rich and not isinstance(sched, RichPatternScheduler):
            sched = RichPatternScheduler.from_scheduler(
                sched, pair_slot_base=int(getattr(governor, "pair_slot_base", 10000)))
            governor.scheduler = sched
        elif not palette.rich and isinstance(sched, RichPatternScheduler):
            governor.scheduler = sched.to_plain()
            sched = governor.scheduler
        sched.synthdef = palette.hit
    if melody is not None:
        melody.synthdef = palette.melody
    return palette


# ── Pattern layer: per-layer hit models ──────────────────────────────────────

class RichPatternScheduler(PatternScheduler):
    """PatternScheduler that merges a per-layer parameter set into every hit.

    The governor renders two layers through one scheduler, distinguished only
    by the slot key: ornamental hits use the voice slot, the paired head voice
    uses pair_slot_base + slot. Parameters the governor passes explicitly win;
    the layer defaults only add what it does not send (the `model`).
    """

    def __init__(self, osc_client=None, lookahead_s: float = 0.25, synthdef: str = "mpeHitRich",
                 pair_slot_base: int = 10000,
                 ornament_params: Optional[Dict[str, float]] = None,
                 pair_params: Optional[Dict[str, float]] = None):
        super().__init__(osc_client=osc_client, lookahead_s=lookahead_s, synthdef=synthdef)
        self.pair_slot_base  = int(pair_slot_base)
        self.ornament_params = dict(ornament_params if ornament_params is not None
                                    else {"model": HIT_MODEL_PLUCKED})
        self.pair_params     = dict(pair_params if pair_params is not None
                                    else {"model": HIT_MODEL_PIANO})

    @classmethod
    def from_scheduler(cls, base: PatternScheduler, pair_slot_base: int = 10000,
                       **kw) -> "RichPatternScheduler":
        s = cls(osc_client=base.osc, lookahead_s=base.lookahead, synthdef=base.synthdef,
                pair_slot_base=pair_slot_base, **kw)
        s._sent_until = base._sent_until          # share the rendered-until marks
        return s

    def to_plain(self) -> PatternScheduler:
        p = PatternScheduler(osc_client=self.osc, lookahead_s=self.lookahead, synthdef=self.synthdef)
        p._sent_until = self._sent_until
        return p

    def layer_params(self, slot: int) -> Dict[str, float]:
        return self.pair_params if int(slot) >= self.pair_slot_base else self.ornament_params

    def render(self, slot, pattern, clock, now_s, degree_to_freq, amp=0.3,
               extra_params=None, dur_scale=0.9):
        merged = dict(self.layer_params(slot))
        merged.update(extra_params or {})
        return super().render(slot, pattern, clock, now_s, degree_to_freq, amp=amp,
                              extra_params=merged, dur_scale=dur_scale)


# ── Melody layer: bowed / plucked per note ───────────────────────────────────

def melody_model_for_blob(blob, profile) -> List:
    """Extra /s_new args for \\melodyHitRich from the blob's state at fire time.

    expanding (d_area >= 0) -> bowed, shrinking -> plucked (the same split
    melody_rhythm uses for legato/staccato release); bow pressure from speed.
    """
    d_area  = float(getattr(blob, "d_area", 0.0))
    speed_n = extract_source_value(Source.SPEED, blob, profile)
    model   = MEL_MODEL_BOWED if d_area >= 0.0 else MEL_MODEL_PLUCKED
    bow     = round(0.15 + 0.80 * max(0.0, min(1.0, speed_n)), 4)
    return ["model", int(model), "bow", float(bow)]


class RichMelodyEngine(MelodyEngine):
    """MelodyEngine whose notes carry `model` and `bow` for \\melodyHitRich.

    _fire is the single construction point for a note (immediate and
    queued-pull paths both end there), so the extras are computed once per
    note from the blob at the moment it sounds and appended to the /s_new.
    """

    def __init__(self, voices, profile, *, synthdef: str = "melodyHitRich", **kw):
        super().__init__(voices, profile, synthdef=synthdef, **kw)
        self._base_send = self._send
        self._note_extra: List = []
        self._send = self._send_with_extra

    def _send_with_extra(self, addr: str, args: list) -> None:
        extra = self._note_extra if getattr(self, "_extras_enabled", True) else []
        self._base_send(addr, list(args) + list(extra))

    def _fire(self, blob, voice, slot_id, midi, now, delay=0.0):
        self._note_extra = melody_model_for_blob(blob, self.profile)
        try:
            return super()._fire(blob, voice, slot_id, midi, now, delay=delay)
        finally:
            self._note_extra = []


def make_melody_engine(voices, profile, palette: Palette, **kw) -> MelodyEngine:
    """Construct the melody engine for a palette (drop-in for MelodyEngine(...))."""
    if palette.rich:
        return RichMelodyEngine(voices, profile, synthdef=palette.melody, **kw)
    return MelodyEngine(voices, profile, synthdef=palette.melody, **kw)


# ── Per-voice extras for \mpeVoiceRich ───────────────────────────────────────

def voice_extras_for_blob(blob, profile) -> Dict[str, float]:
    """formant / sub / chorus from the blob: the three \\mpeVoiceRich controls
    that no routing destination covers."""
    height = extract_source_value(Source.Y_NORM, blob, profile)       # 1 = top (v4.10.1 routing)
    area_n = extract_source_value(Source.AREA,   blob, profile)
    vx_n   = abs(extract_source_value(Source.VX, blob, profile))
    return {
        "formant": round(max(0.0, min(1.0, height)), 4),
        "sub":     round(max(0.0, min(1.0, 0.85 * math.sqrt(area_n))), 4),
        "chorus":  round(max(0.0, min(1.0, 0.8 * vx_n)), 4),
    }


class VoiceExtras:
    """Each frame, send formant/sub/chorus to every live \\mpeVoiceRich node.
    Values are only sent when they move by more than `eps`, so an idle lamp
    costs nothing."""

    def __init__(self, voices, profile, send: Optional[Callable[[str, list], None]] = None,
                 eps: float = 0.01):
        self.voices  = voices
        self.profile = profile
        self._send   = send or (lambda addr, args: voices.osc.send_message(addr, args))
        self.eps     = eps
        self._last: Dict[int, Dict[str, float]] = {}
        self.sent = 0

    def update(self, tracked: list) -> int:
        by_id = {b.id: b for b in tracked}
        n = 0
        for v in self.voices.voices_by_slot:
            if v.sc_node is None or v.blob_id is None or v.blob_id not in by_id:
                continue
            if getattr(v, "synth_kind", "mpe") != "mpe" or v.state.name in ("IDLE", "ZOMBIE"):
                continue
            ex   = voice_extras_for_blob(by_id[v.blob_id], self.profile)
            prev = self._last.get(v.sc_node)
            if prev is not None and all(abs(ex[k] - prev[k]) < self.eps for k in ex):
                continue
            args = [v.sc_node]
            for k, val in ex.items():
                args += [k, float(val)]
            try:
                self._send("/n_set", args)
                n += 1
            except Exception:
                pass
            self._last[v.sc_node] = ex
        # forget nodes that are gone
        live = {v.sc_node for v in self.voices.voices_by_slot if v.sc_node is not None}
        for nid in [k for k in self._last if k not in live]:
            self._last.pop(nid, None)
        self.sent += n
        return n


# ── Master space ─────────────────────────────────────────────────────────────

@dataclass
class MasterParams:
    wet:      float = 0.25
    t60:      float = 2.5
    size:     float = 2.0
    damp:     float = 0.5
    modDepth: float = 0.3
    holeMix:  float = 0.0
    holeFb:   float = 0.40
    highcut:  float = 5000.0
    drive:    float = 0.0

    def as_args(self) -> list:
        out: list = []
        for k, v in vars(self).items():
            out += [k, float(v)]
        return out


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def master_params_from_state(ss, depth: float = 1.0) -> MasterParams:
    """Pure mapping ScoreState -> \\lavaMaster controls. `depth` (0..1) scales
    every departure from the neutral setting (0 = the static default room)."""
    d = _clip(depth)
    neutral = MasterParams()
    if ss is None:
        return neutral
    tension   = _clip(getattr(ss, "tension", 0.0))
    ambiguity = _clip(getattr(ss, "ambiguity", 0.0))
    surpr     = _clip(float(getattr(ss, "surprisal", 0.0)) / 6.0)   # raw nats -> 0..1 as the governor does
    phrase    = _clip(getattr(ss, "phrase_position", 0.0))
    kur       = _clip(getattr(ss, "kuramoto", 0.0))
    count     = _clip(float(getattr(ss, "count_smooth", 0.0)) / 6.0)
    reg       = _clip(getattr(ss, "register_center", 0.5))

    # v4.10.6: shorter tails and a hall that only opens under tension, so
    # onsets, births and direction changes are not smeared; the governed
    # quantities still move the room, just over a smaller range.
    hall_open = _clip((tension - 0.35) / 0.65)         # 0 below tension 0.35
    target = MasterParams(
        wet      = 0.15 + 0.30 * phrase,
        t60      = 1.0 + 4.0 * ambiguity,
        size     = 1.2 + 1.8 * (1.0 - count),
        damp     = 0.20 + 0.60 * (1.0 - tension),
        modDepth = 0.10 + 0.50 * (1.0 - kur),
        holeMix  = 0.30 * hall_open,
        holeFb   = 0.20 + 0.60 * surpr,
        highcut  = 2200.0 + 5000.0 * reg,
        drive    = 0.35 * tension,
    )
    if d >= 1.0:
        return target
    blended = MasterParams()
    for k in vars(neutral):
        setattr(blended, k, getattr(neutral, k) + d * (getattr(target, k) - getattr(neutral, k)))
    return blended


class MasterSpace:
    """Owns the \\lavaMaster node (id 900, tail of group 0) and refreshes its
    controls every frame. Spawn AFTER VoiceManager has been constructed:
    VoiceManager sends /g_freeAll 0 at startup, which would free it."""

    NODE_ID = 900

    def __init__(self, osc=None, synthdef: str = "lavaMaster", node_id: int = NODE_ID,
                 depth: float = 1.0, send: Optional[Callable[[str, list], None]] = None,
                 bloom_min_strength: float = 0.3):
        self.osc      = osc
        self.synthdef = synthdef
        self.node_id  = int(node_id)
        self.depth    = float(depth)
        self._send    = send or (lambda addr, args: osc.send_message(addr, args))
        self.bloom_min_strength = bloom_min_strength
        self.params   = MasterParams()
        self.spawned  = False
        self.bypass   = False
        self.blooms   = 0
        self.frames   = 0

    def spawn(self) -> None:
        # addAction 1 = add to the TAIL of target group 0: after every node
        # Python spawns at the head, so In.ar(0) sees the complete mix.
        args = [self.synthdef, self.node_id, 1, 0] + self.params.as_args() + ["bypass", int(self.bypass)]
        try:
            self._send("/s_new", args)
            self.spawned = True
        except Exception:
            self.spawned = False

    def respawn(self) -> None:
        """Free and rebuild the master node (clears the reverb / Greyhole state,
        e.g. after a bad value has silenced the tails). v4.10.3."""
        self.close()
        self.spawn()

    def set_bypass(self, on: bool) -> None:
        self.bypass = bool(on)
        if self.spawned:
            self._send("/n_set", [self.node_id, "bypass", int(self.bypass)])

    def toggle_bypass(self) -> bool:
        self.set_bypass(not self.bypass)
        return self.bypass

    def update(self, ss) -> Optional[MasterParams]:
        if not self.spawned:
            return None
        self.frames += 1
        p = master_params_from_state(ss, self.depth)
        self.params = p
        args = [self.node_id] + p.as_args()
        if ss is not None and getattr(ss, "boundary", False):
            strength = _clip(getattr(ss, "boundary_strength", 0.0))
            if strength >= self.bloom_min_strength:
                args += ["t_bloom", 1, "bloomAmt", float(strength * self.depth)]
                self.blooms += 1
        try:
            self._send("/n_set", args)
        except Exception:
            pass
        return p

    def close(self) -> None:
        if self.spawned:
            try:
                self._send("/n_free", [self.node_id])
            except Exception:
                pass
        self.spawned = False


# ── Pre-sound of the predicted triad ─────────────────────────────────────────

_PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6,
       "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}


def parse_triad(name: str):
    """'Eb' -> (3, True); 'F#m' -> (6, False); '' / unknown -> None."""
    if not name:
        return None
    n = str(name).strip()
    minor = n.endswith("m")
    if minor:
        n = n[:-1]
    if n not in _PC:
        return None
    return _PC[n], (not minor)


def triad_freqs(name: str, register: int = 48) -> List[float]:
    """Three frequencies (root, third, fifth) with the root placed within half
    an octave of `register` (MIDI), close voicing above it."""
    t = parse_triad(name)
    if t is None:
        return []
    root_pc, major = t
    root = register + ((root_pc - register + 5) % 12) - 5
    midis = [root, root + (4 if major else 3), root + 7]
    return [440.0 * 2.0 ** ((m - 69) / 12.0) for m in midis]


class PreSoundLayer:
    """Holds three \\preSoundRich nodes on the governor's PREDICTED triad.
    Gain = gain * harmony_posterior_other * score_gain: as posterior mass
    leaks onto a neighbouring triad the coming chord becomes audible as
    coming, and when the walker steps there the pre-sound falls to zero
    (the prediction now equals the current triad)."""

    NODE_BASE = 901

    def __init__(self, osc=None, synthdef: str = "preSoundRich", node_base: int = NODE_BASE,
                 register: int = 48, gain: float = 0.10,
                 send: Optional[Callable[[str, list], None]] = None):
        self.osc      = osc
        self.synthdef = synthdef
        self.node_ids = [int(node_base) + i for i in range(3)]
        self.register = int(register)
        self.gain     = float(gain)
        self._send    = send or (lambda addr, args: osc.send_message(addr, args))
        self.spawned  = False
        self.triad    = ""
        self.freqs: List[float] = []
        self.amp      = 0.0
        self.pans     = [-0.4, 0.0, 0.4]

    def spawn(self) -> None:
        try:
            for nid, pan in zip(self.node_ids, self.pans):
                self._send("/s_new", [self.synthdef, nid, 0, 0, "freq", 220.0, "amp", 0.0,
                                      "gate", 1, "pan", pan])
            self.spawned = True
        except Exception:
            self.spawned = False

    def update(self, ss, score_gain: float = 1.0) -> float:
        if not self.spawned:
            return 0.0
        if ss is None:
            amp = 0.0
            triad = ""
        else:
            triad = str(getattr(ss, "predicted_triad", "") or "")
            other = _clip(getattr(ss, "harmony_posterior_other", 0.0))
            amp   = self.gain * other * _clip(score_gain)
            if not triad or triad == str(getattr(ss, "triad", "")):
                amp = 0.0
        if triad and triad != self.triad:
            freqs = triad_freqs(triad, self.register)
            if freqs:
                self.triad, self.freqs = triad, freqs
                for nid, f in zip(self.node_ids, freqs):
                    try:
                        self._send("/n_set", [nid, "freq", float(f)])
                    except Exception:
                        pass
        if abs(amp - self.amp) > 1e-4:
            self.amp = amp
            for nid in self.node_ids:
                try:
                    self._send("/n_set", [nid, "amp", float(amp)])
                except Exception:
                    pass
        return amp

    def close(self) -> None:
        if self.spawned:
            for nid in self.node_ids:
                try:
                    self._send("/n_set", [nid, "gate", 0])
                except Exception:
                    pass
        self.spawned = False


# ── The rig: one object the main loop talks to ───────────────────────────────

class SynthPaletteRig:
    """Everything v4.10 adds at runtime, behind update(score_state, tracked)
    and close(). Construct AFTER VoiceManager (and the governor / melody
    engine, if any) exist."""

    def __init__(self, voices, profile, *, governor=None, melody=None,
                 palette: Palette = RICH, master: bool = True, presound: bool = True,
                 master_depth: float = 1.0, presound_gain: float = 0.10,
                 send: Optional[Callable[[str, list], None]] = None):
        self.voices   = voices
        self.profile  = profile
        self.governor = governor
        self.melody   = melody
        self.palette  = apply_palette(voices, palette, governor=governor, melody=melody)
        send = send or (lambda addr, args: voices.osc.send_message(addr, args))
        self._send    = send
        self.extras   = VoiceExtras(voices, profile, send=send) if palette.rich else None
        self.master   = MasterSpace(synthdef=palette.master, depth=master_depth, send=send) if master else None
        self.presound = PreSoundLayer(synthdef=palette.presound, gain=presound_gain, send=send) if presound else None
        if self.master is not None:
            self.master.spawn()
        if self.presound is not None:
            self.presound.spawn()

    def update(self, score_state, tracked: list) -> None:
        if self.extras is not None:
            self.extras.update(tracked)
        if self.master is not None:
            self.master.update(score_state)
        if self.presound is not None:
            self.presound.update(score_state, getattr(self.voices, "score_gain", 1.0))

    def set_palette(self, palette) -> Palette:
        """Switch palettes while running (v4.10.4). Takes effect per family
        the way the system already spawns: hits, pairs and melody notes change
        on their next onset; each sustained voice changes when its slot next
        spawns a node (a birth, a bass-stutter entry/exit, a resurrection), so
        the ensemble migrates voice by voice rather than switching at once.
        The master space and the pre-sound are unaffected (use x / --no-space
        for the space). Extras are only sent for the rich palette."""
        if isinstance(palette, str):
            palette = PALETTES[palette]
        if palette.name == self.palette.name:
            return self.palette
        self.palette = apply_palette(self.voices, palette, governor=self.governor, melody=self.melody)
        if self.melody is not None and hasattr(self.melody, "_note_extra"):
            # a RichMelodyEngine appends model/bow; the classic \melodyHit
            # ignores unknown controls, so leaving it is harmless, but keep the
            # OSC clean: silence the extras while classic is selected
            self.melody._extras_enabled = palette.rich
        self.extras = VoiceExtras(self.voices, self.profile, send=self._send) if palette.rich else None
        return self.palette

    def toggle_palette(self) -> Palette:
        return self.set_palette(CLASSIC if self.palette.rich else RICH)

    def toggle_master_bypass(self) -> bool:
        return self.master.toggle_bypass() if self.master is not None else False

    def hud_line(self) -> str:
        parts = [f"SYNTH {self.palette.name}"]
        if self.master is not None:
            p = self.master.params
            parts.append(f"space wet={p.wet:.2f} t60={p.t60:.1f}s size={p.size:.1f} "
                         f"fb={p.holeFb:.2f}{' BYPASS' if self.master.bypass else ''} "
                         f"blooms={self.master.blooms}")
        if self.presound is not None:
            parts.append(f"pre={self.presound.triad or '-'}@{self.presound.amp:.3f}")
        return "  ".join(parts)

    def close(self) -> None:
        if self.presound is not None:
            self.presound.close()
        if self.master is not None:
            self.master.close()


def install(voices, profile, *, governor=None, melody=None, palette=RICH,
            master: bool = True, presound: bool = True, **kw) -> SynthPaletteRig:
    """One call from the main loop. Returns the rig; call rig.update(score_state,
    tracked) once per frame after voices.process_frame(), rig.close() at exit."""
    if isinstance(palette, str):
        palette = PALETTES[palette]
    return SynthPaletteRig(voices, profile, governor=governor, melody=melody,
                           palette=palette, master=master, presound=presound, **kw)
