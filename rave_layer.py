# rave_layer.py -- Audible Lava / DSO v4.11 "RAVE timbre transfer"
#
# v4.14 (Nov 2026): nineteen models (seven added from IRCAM's ACIDS catalogue:
# sol_ordinario, sol_full, sol_ordinario_fast, musicnet, vintage, percussion,
# VCTK) and the list is ORDERED BY FIT WITH THE LAMP, best first (see the
# ranking note above MODELS). The tab maps the home row and then the top row
# to that order (controller.RAVE_MODEL_KEYS). Rates the file name does not
# give are taken from rave_status.json, so mismatches are flagged for the
# ACIDS models too.
#
# Additive module. A final stage that translates the fully formed lava mix
# (every \mpeVoiceRich, hit, melody note, strike and pad on bus 0) through one
# or more pre-trained RAVE models -- RAVE being the Realtime Audio Variational
# autoEncoder (Caillon & Esling, IRCAM) -- running inside SuperCollider via the
# nn.ar extension (https://github.com/elgiano/nn.ar). Off by default: the lamp
# plays the rich SynthDefs untouched until the layer is switched on.
#
# SIGNAL PATH (order of nodes in group 0 = execution order)
#
#   [voices, hits, pads ... at the HEAD]            -> bus 0 (dry mix)
#   \raveSplit        node 910                      In.ar(0) -> Out.ar(RAVE_BUS)
#                                                   ReplaceOut.ar(0, dry * (1-mix))
#   \raveXfer_<key>   nodes 920.. (one per ACTIVE model)
#                                                   In.ar(RAVE_BUS) -> RAVE -> Out.ar(0, wet)
#   \lavaMaster       node 900 (synth_palette)      the space hears dry + wet
#
# The split and the model nodes are added BEFORE node 900 (addAction 2 =
# addBefore) when the master space exists, else at the tail of group 0, so the
# reverb/Greyhole treat the transferred timbre like any other voice. The split
# is spawned once at install and stays; with the layer off its dry gain is 1
# and it is a passthrough. Model nodes are spawned when a model is toggled on
# and gated (fade-out, then self-free) when toggled off -- each node owns an
# independent instance of its model (that is how nn.ar works), so an inactive
# model costs nothing.
#
# WHY A PARALLEL BANK, NOT A SERIES CHAIN
#   Several models can be on at once. Each reads the SAME dry copy on RAVE_BUS
#   and adds its own wet signal to bus 0, so "birds + organ" is the lamp heard
#   through two ears at once, not birds-of-an-organ. Per-model amplitude is
#   mix / sqrt(n_active) so stacking models does not stack level. Each active
#   model gets a fixed pan position from its index (-0.6 .. 0.6) so the bank
#   spreads across the image.
#
# WHAT PYTHON KNOWS ABOUT THE MODELS
#   blob_synth_rave.scd (sclang) loads the model descriptions with NN.load,
#   builds one \raveXfer_<key> SynthDef per model that loaded, and writes
#   rave_status.json next to the model files (which keys loaded, their trained
#   sample rate, the server's rate). Python reads that file to annotate the GUI
#   (loaded / missing / failed, rate mismatch) and never blocks on it: sending
#   /s_new for a SynthDef that does not exist just prints a line in the SC
#   post window. The canonical model list (key <-> file) lives here and is
#   mirrored in the .scd; test_rave_layer.py checks the two agree.
#
# LATENCY (nn.ar README): buffering 2048/48000 = 43 ms, inference 15-30 ms on
# a 2016 i5, plus RAVE's own cached-convolution delay (~650 ms for non-causal
# v2 models). The wet layer therefore trails the dry layer by roughly 0.7 s.
# On the lamp's 90-180 s cycles this reads as a halo rather than an echo; at
# mix = 1.0 the dry is gone and the delay is invisible. No alignment delay is
# applied to the dry path (it would have to be a guess per model).
#
# RATES: scsynth runs at 48 kHz (blob_synth_rich.scd). Models trained at
# 44.1 kHz (mrp_strengjavera, voice_vctk, and possibly some ACIDS models,
# whose names carry no rate) run ~8.8 % off-rate because nn.ar does not
# resample. They still work, with a slight spectral tilt; the GUI marks them
# once the rate is known (from the file name, else from rave_status.json).
#
# Tests: test_rave_layer.py (no SuperCollider, camera or pygame).

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# ── Controller event types (defined with the others in controller.py) ───────
#   RAVE_TOGGLE_MODEL  extras: index          RAVE_TOGGLE   layer on/off
#   RAVE_SET           param "mix" | "in_gain", value = relative delta
#   RAVE_ALL_OFF                               RAVE_RELOAD   re-read rave_status.json
from controller import (RAVE_TOGGLE_MODEL, RAVE_TOGGLE, RAVE_SET, RAVE_ALL_OFF,
                        RAVE_RELOAD, RAVE_MODEL_KEYS)

# ── Configuration ────────────────────────────────────────────────────────────
RAVE_MODEL_DIR  = os.environ.get("AUDIBLE_LAVA_RAVE_DIR",
                                 "/Users/multivax/Documents/RAVE_PreTrained_Models")
RAVE_STATUS_FILE = "rave_status.json"      # written by blob_synth_rave.scd
RAVE_BUS         = 64                       # private audio bus (stereo: 64, 65)
SPLIT_NODE       = 910
MODEL_NODE_BASE  = 920                      # node for model i = MODEL_NODE_BASE + i
SPLIT_SYNTHDEF   = "raveSplit"
XFER_PREFIX      = "raveXfer_"


@dataclass(frozen=True)
class RaveModel:
    key:     str              # NN.load key / SynthDef suffix (shared with the .scd)
    file:    str              # file name inside RAVE_MODEL_DIR
    label:   str              # what the GUI shows
    buffer:  Optional[int]    # from the file name (b2048), None if absent
    rate:    Optional[int]    # trained sample rate from the file name (r48000)
    latent:  Optional[int]    # latent dims (z16); informational

    @property
    def synthdef(self) -> str:
        return XFER_PREFIX + self.key

    @property
    def path(self) -> str:
        return os.path.join(RAVE_MODEL_DIR, self.file)


def parse_model_filename(name: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """'x_b2048_r48000_z8.ts' -> (2048, 48000, 8); fields absent -> None.
    Both '_' and '-' separators occur in the shipped names."""
    def grab(letter: str) -> Optional[int]:
        m = re.search(r"[_-]" + letter + r"(\d+)(?=[_\-.]|$)", name)
        return int(m.group(1)) if m else None
    return grab("b"), grab("r"), grab("z")


def _model(key: str, file: str, label: str) -> RaveModel:
    b, r, z = parse_model_filename(file)
    return RaveModel(key, file, label, b, r, z)


# The nineteen shipped models, ORDERED BY FIT WITH THE LAMP (v4.14). The lamp
# sends slow, sustained harmonic drones with plucks, piano-like melody notes
# and soft strikes on top. Models trained on pitched, sustained instrumental
# sound follow those drones' pitch and dynamics, so the lamp's harmony
# survives the transfer; texture models mostly map loudness to texture and
# lose the pitch. Ranked by (1) how well the training set matches that input,
# (2) instrumental range, (3) how clean and pleasant the result tends to be:
#
#   instrumental, wide range   sol_ordinario, sol_full, musicnet, crozzoli
#   sustained harmonic         organ, mrp, solfast (cheaper SOL), vocalset
#   plucked / slow contours    guitar, humpbacks
#   broad but heavy            vintage (RAVE v1 "large", 482 MB: most CPU)
#   voice blend                voicemulti
#   texture                    birds, water, percussion
#   speech (babble on drones)  vctkacids, vctk
#   least pleasant on the lamp marine (decoder can chirp on start), magnets
#
# Keys are the handles both sides use; keep this in step with ~raveModels in
# blob_synth_rave.scd (test_rave_layer.py checks). The GUI lists and keys the
# models in this order.
MODELS: List[RaveModel] = [
    _model("solord",      "sol_ordinario.ts",                          "SOL ordinario"),
    _model("solfull",     "sol_full.ts",                               "SOL full"),
    _model("musicnet",    "musicnet.ts",                               "MusicNet chamber"),
    _model("crozzoli",    "crozzoli_bigensemblesmusic_18d.ts",         "crozzoli ensemble"),
    _model("organ",       "organ_archive_b2048_r48000_z16.ts",         "organ archive"),
    _model("mrp",         "mrp_strengjavera_b2048_r44100_z16.ts",      "MRP piano"),
    _model("solfast",     "sol_ordinario_fast.ts",                     "SOL fast (light)"),
    _model("vocalset",    "voice_vocalset_b2048_r48000_z16.ts",        "voice VocalSet"),
    _model("guitar",      "guitar_iil_b2048_r48000_z16.ts",            "guitar IIL"),
    _model("humpbacks",   "humpbacks_pondbrain_b2048_r48000_z20.ts",   "humpback whales"),
    _model("vintage",     "vintage.ts",                                "vintage (heavy)"),
    _model("voicemulti",  "voice-multi-b2048-r48000-z11.ts",           "voice multi"),
    _model("birds",       "birds_dawnchorus_b2048_r48000_z8.ts",       "birds dawn chorus"),
    _model("water",       "water_pondbrain_b2048_r48000_z16.ts",       "water"),
    _model("percussion",  "percussion.ts",                             "percussion"),
    _model("vctkacids",   "VCTK.ts",                                   "voice VCTK ACIDS"),
    _model("vctk",        "voice_vctk_b2048_r44100_z22.ts",            "voice VCTK IIL"),
    _model("marine",      "marinemammals_pondbrain_b2048_r48000_z20.ts", "marine mammals"),
    _model("magnets",     "magnets_b2048_r48000_z8.ts",                "magnets"),
]

MODEL_BY_KEY: Dict[str, RaveModel] = {m.key: m for m in MODELS}
LABEL_W = max(len(m.label) for m in MODELS) + 1      # listing column width


def model_pan(index: int, n_models: int = len(MODELS), width: float = 0.6) -> float:
    """Fixed stereo position for model `index`: evenly spread over [-width, width]."""
    if n_models <= 1:
        return 0.0
    return round(-width + 2.0 * width * index / float(n_models - 1), 4)


def per_model_amp(mix: float, n_active: int) -> float:
    """Wet amplitude for each active model: mix / sqrt(n) keeps the stacked
    bank at roughly the level of a single model."""
    if n_active <= 0:
        return 0.0
    return round(max(0.0, min(1.0, mix)) / math.sqrt(n_active), 4)


def dry_gain(enabled: bool, n_active: int, mix: float) -> float:
    """What \\raveSplit multiplies the dry mix by. Passthrough unless the
    layer is on AND at least one model is active."""
    if not enabled or n_active <= 0:
        return 1.0
    return round(1.0 - max(0.0, min(1.0, mix)), 4)


# ── Status written by sclang ─────────────────────────────────────────────────

@dataclass
class ModelStatus:
    status:   str = "unknown"      # loaded | missing | failed | unknown
    rate:     Optional[int] = None
    min_buffer: Optional[int] = None
    error:    str = ""


@dataclass
class RaveStatus:
    available:   bool = False               # nn.ar present in sclang
    server_rate: Optional[int] = None
    models:      Dict[str, ModelStatus] = field(default_factory=dict)
    written:     str = ""
    path:        str = ""
    mtime:       float = 0.0

    @classmethod
    def read(cls, path: str) -> "RaveStatus":
        st = cls(path=path)
        try:
            with open(path, "r") as f:
                raw = json.load(f)
            st.mtime = os.path.getmtime(path)
        except Exception:
            return st
        st.available   = bool(raw.get("nn_available", False))
        st.server_rate = raw.get("sampleRate")
        st.written     = str(raw.get("written", ""))
        for key, info in (raw.get("models") or {}).items():
            if not isinstance(info, dict):
                continue
            st.models[str(key)] = ModelStatus(
                status=str(info.get("status", "unknown")),
                rate=info.get("rate"),
                min_buffer=info.get("minBufferSize"),
                error=str(info.get("error", "")),
            )
        return st

    def of(self, key: str) -> ModelStatus:
        return self.models.get(key, ModelStatus())


# ── The layer ────────────────────────────────────────────────────────────────

class RaveLayer:
    """Owns the \\raveSplit node and one \\raveXfer_<key> node per active model.

    Construct AFTER VoiceManager (its /g_freeAll 0 would free the split) and
    after synth_palette.install (so master_node can be given). All state
    changes go through handle_controller_event / the public methods, each of
    which pushes the new state to scsynth immediately.
    """

    def __init__(self, osc=None, *, master_node: Optional[int] = None,
                 send: Optional[Callable[[str, list], None]] = None,
                 enabled: bool = False, mix: float = 0.7, in_gain: float = 1.0,
                 fade_in: float = 3.0, fade_out: float = 1.0,
                 # v4.11.6: 3.0 s (was 1.5). A model needs 1.3-2.9 s to load
                 # its thread on the Intel Mac (rave_bench); a 1.5 s fade was
                 # over before any sound existed, so models popped in.
                 status_path: Optional[str] = None,
                 status_poll_s: float = 2.0,
                 clock: Callable[[], float] = time.time):
        self.osc         = osc
        self._send       = send or (lambda addr, args: osc.send_message(addr, args))
        self.master_node = master_node
        self.enabled     = bool(enabled)
        self.mix         = float(max(0.0, min(1.0, mix)))
        self.in_gain     = float(max(0.1, min(4.0, in_gain)))
        self.fade_in     = float(fade_in)
        self.fade_out    = float(fade_out)
        self.active: List[bool] = [False] * len(MODELS)
        self.spawned_nodes: Dict[int, int] = {}      # model index -> node id
        self.split_spawned = False
        self.status_path = status_path or os.path.join(RAVE_MODEL_DIR, RAVE_STATUS_FILE)
        self.status      = RaveStatus.read(self.status_path)
        self._status_poll_s = float(status_poll_s)
        self._clock      = clock
        self._next_poll  = clock() + self._status_poll_s
        self.last_msg    = ""
        self.toggles     = 0

    # ── placement: before the master space if there is one, else tail of 0 ──
    def _place(self) -> List:
        if self.master_node is not None:
            return [2, int(self.master_node)]       # addBefore master
        return [1, 0]                               # addToTail group 0

    # ── split ──
    def spawn_split(self) -> None:
        args = [SPLIT_SYNTHDEF, SPLIT_NODE] + self._place() + [
            "raveBus", RAVE_BUS, "dry", 1.0]
        try:
            self._send("/s_new", args)
            self.split_spawned = True
        except Exception:
            self.split_spawned = False

    def _push_dry(self) -> float:
        g = dry_gain(self.enabled, self.n_active, self.mix)
        if self.split_spawned:
            try:
                self._send("/n_set", [SPLIT_NODE, "dry", float(g)])
            except Exception:
                pass
        return g

    # ── models ──
    @property
    def n_active(self) -> int:
        return sum(1 for a in self.active if a)

    def active_models(self) -> List[RaveModel]:
        return [m for m, a in zip(MODELS, self.active) if a]

    def _node_for(self, index: int) -> int:
        return MODEL_NODE_BASE + int(index)

    def _spawn_model(self, index: int) -> None:
        m = MODELS[index]
        nid = self._node_for(index)
        args = [m.synthdef, nid] + self._place() + [
            "inBus", RAVE_BUS, "out", 0,
            "amp", per_model_amp(self.mix, self.n_active),
            "inGain", float(self.in_gain),
            "pan", model_pan(index),
            "fadeIn", float(self.fade_in), "fadeOut", float(self.fade_out),
            "gate", 1]
        try:
            self._send("/s_new", args)
            self.spawned_nodes[index] = nid
        except Exception:
            pass

    def _release_model(self, index: int) -> None:
        nid = self.spawned_nodes.pop(index, None)
        if nid is None:
            return
        try:
            self._send("/n_set", [nid, "gate", 0])      # fade, then self-free
        except Exception:
            pass

    def _push_amps(self) -> None:
        amp = per_model_amp(self.mix, self.n_active)
        for index, nid in self.spawned_nodes.items():
            try:
                self._send("/n_set", [nid, "amp", float(amp)])
            except Exception:
                pass

    def _push_in_gain(self) -> None:
        for nid in self.spawned_nodes.values():
            try:
                self._send("/n_set", [nid, "inGain", float(self.in_gain)])
            except Exception:
                pass

    def _sync_nodes(self) -> None:
        """Make the set of spawned nodes equal to {active models} when the
        layer is on, and empty when it is off; then level and dry gain."""
        want = {i for i, a in enumerate(self.active) if a} if self.enabled else set()
        for i in list(self.spawned_nodes):
            if i not in want:
                self._release_model(i)
        for i in sorted(want):
            if i not in self.spawned_nodes:
                self._spawn_model(i)
        self._push_amps()
        self._push_dry()

    # ── public controls ──
    def set_enabled(self, on: bool) -> bool:
        self.enabled = bool(on)
        self._sync_nodes()
        self.last_msg = f"RAVE layer {'ON' if self.enabled else 'OFF'}"
        return self.enabled

    def toggle_enabled(self) -> bool:
        return self.set_enabled(not self.enabled)

    def set_model(self, index: int, on: bool) -> bool:
        if not (0 <= index < len(MODELS)):
            return False
        self.active[index] = bool(on)
        self.toggles += 1
        self._sync_nodes()
        st = self.status.of(MODELS[index].key).status
        note = "" if st in ("loaded", "unknown") else f"  ({st} in SC)"
        self.last_msg = f"{MODELS[index].label}: {'on' if on else 'off'}{note}"
        return self.active[index]

    def toggle_model(self, index: int) -> bool:
        if not (0 <= index < len(MODELS)):
            return False
        return self.set_model(index, not self.active[index])

    def all_off(self) -> None:
        self.active = [False] * len(MODELS)
        self._sync_nodes()
        self.last_msg = "RAVE: all models off"

    def set_mix(self, mix: float) -> float:
        self.mix = float(max(0.0, min(1.0, mix)))
        self._push_amps()
        self._push_dry()
        self.last_msg = f"RAVE mix -> {self.mix:.2f}"
        return self.mix

    def set_in_gain(self, g: float) -> float:
        self.in_gain = float(max(0.1, min(4.0, g)))
        self._push_in_gain()
        self.last_msg = f"RAVE input gain -> {self.in_gain:.2f}"
        return self.in_gain

    def reload_status(self) -> RaveStatus:
        """Re-read rave_status.json (after re-evaluating the .scd). Nodes are
        not touched: if SuperCollider was (re)started after Python, restart
        Python too -- its split and model nodes live on the old server."""
        self.status = RaveStatus.read(self.status_path)
        n = sum(1 for s in self.status.models.values() if s.status == "loaded")
        self.last_msg = (f"RAVE status: {n} model(s) loaded in SC"
                         if self.status.models else "RAVE status file not found (evaluate blob_synth_rave.scd)")
        return self.status

    # ── controller events ──
    def handle_controller_event(self, ev) -> bool:
        t = getattr(ev, "type", None)
        if t == RAVE_TOGGLE_MODEL:
            idx = int((getattr(ev, "extras", {}) or {}).get("index", -1))
            self.toggle_model(idx)
            return True
        if t == RAVE_TOGGLE:
            self.toggle_enabled()
            return True
        if t == RAVE_ALL_OFF:
            self.all_off()
            return True
        if t == RAVE_RELOAD:
            self.reload_status()
            return True
        if t == RAVE_SET:
            p = getattr(ev, "param", None)
            v = float(getattr(ev, "value", 0.0) or 0.0)
            rel = bool((getattr(ev, "extras", {}) or {}).get("relative", True))
            if p == "mix":
                self.set_mix((self.mix + v) if rel else v)
                return True
            if p == "in_gain":
                self.set_in_gain((self.in_gain * (2.0 ** v)) if rel else v)
                return True
        return False

    # ── per frame ──
    def update(self) -> None:
        """Cheap: re-read rave_status.json every status_poll_s if it changed
        (so a model loaded in SC after startup shows up in the GUI)."""
        now = self._clock()
        if now < self._next_poll:
            return
        self._next_poll = now + self._status_poll_s
        try:
            mt = os.path.getmtime(self.status_path)
        except OSError:
            return
        if mt != self.status.mtime:
            self.status = RaveStatus.read(self.status_path)

    # ── GUI ──
    def model_line(self, index: int, key_char: str) -> Tuple[str, str]:
        """One listing row: (text, kind). kind in on | off | missing | failed | unverified."""
        m  = MODELS[index]
        st = self.status.of(m.key)
        on = self.active[index]
        r = m.rate or (st.rate if isinstance(st.rate, int) and st.rate > 0 else None)
        rate = f"{r // 1000}k" if r else " ?k"
        if r and self.status.server_rate and r != self.status.server_rate:
            rate += "!"
        if st.status == "loaded":
            tag, kind = "", ("on" if on else "off")
        elif st.status == "missing":
            tag, kind = "  file missing", "missing"
        elif st.status == "failed":
            tag, kind = "  failed to load", "failed"
        else:
            tag, kind = "  (not verified)", ("on" if on else "unverified")
        box = "[x]" if on else "[ ]"
        return (f" {key_char}  {box}  {m.label:<{LABEL_W}}{rate:>5}{tag}", kind)

    def panel_lines(self, keys: str = RAVE_MODEL_KEYS) -> List[Tuple[str, str]]:
        head = (f"RAVE timbre transfer   {'ON ' if self.enabled else 'OFF'}   "
                f"mix {self.mix:.2f}   in {self.in_gain:.2f}   "
                f"active {self.n_active}")
        lines: List[Tuple[str, str]] = [(head, "title")]
        if not self.status.models:
            lines.append(("rave_status.json not found: evaluate blob_synth_rave.scd in SuperCollider "
                          "(models are listed unverified)", "warn"))
        elif not self.status.available:
            lines.append(("nn.ar not installed in SuperCollider: nothing will sound through RAVE", "warn"))
        lines.append(("models listed best-first for the lamp: instrumental > voice > texture", "hint"))
        lines.append(("", ""))
        for i in range(len(MODELS)):
            k = keys[i] if i < len(keys) else "?"
            lines.append(self.model_line(i, k))
        lines.append(("", ""))
        lines.append(("home row, then q w e t y u i  toggle model    SPACE  layer on/off    "
                      "Z  all off    R  re-read status", "hint"))
        lines.append(("up/dn  mix +/-0.05     +/-  mix +/-0.25     left/right  input gain /1.4 x1.4", "hint"))
        return lines

    def hud_line(self) -> str:
        if not self.enabled:
            return "RAVE off"
        names = ",".join(m.key for m in self.active_models()) or "-"
        return f"RAVE on mix={self.mix:.2f} [{names}]"

    # ── shutdown ──
    def close(self) -> None:
        """Gate every model node (fade-out, then self-free); the split stays so
        the fades and the dry mix keep passing. Call free_split() after."""
        for i in list(self.spawned_nodes):
            self._release_model(i)
        if self.split_spawned:
            try:
                self._send("/n_set", [SPLIT_NODE, "dry", 1.0])
            except Exception:
                pass

    def free_split(self) -> None:
        if self.split_spawned:
            try:
                self._send("/n_free", [SPLIT_NODE])
            except Exception:
                pass
        self.split_spawned = False


def install(voices, *, master_node: Optional[int] = None, enabled: bool = False,
            mix: float = 0.7, in_gain: float = 1.0, send=None, **kw) -> RaveLayer:
    """One call from the main loop. Spawns the split (passthrough) and, if
    `enabled` and any model is pre-selected, the model nodes."""
    layer = RaveLayer(getattr(voices, "osc", None), master_node=master_node,
                      send=send, enabled=enabled, mix=mix, in_gain=in_gain, **kw)
    layer.spawn_split()
    layer._sync_nodes()
    return layer
