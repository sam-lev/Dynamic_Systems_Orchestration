#!/usr/bin/env python3
# rave_bench.py -- Audible Lava / DSO v4.11 RAVE test bench (Python half)
#
# Drives a running scsynth over the SAME OSC the live system uses (node ids,
# add actions, control names from rave_layer.py), plays stimuli that emulate
# what the lamp produces, records every scenario to WAV, analyses the WAVs and
# writes a report. Also a listen mode so you can hear each scenario.
#
#   SuperCollider first:  blob_synth.scd -> blob_synth_hits.scd -> melody_hit.scd
#                         -> blob_synth_rich.scd -> blob_synth_rave.scd -> rave_bench.scd
#   then:
#     python rave_bench.py                        # full sweep (~ 1 min per model)
#     python rave_bench.py --models organ birds   # a subset
#     python rave_bench.py --buffers 8192         # one buffer size
#     python rave_bench.py --quick                # fewer scenarios, 6 s each
#     python rave_bench.py --listen               # play, don't record (Enter = next)
#     python rave_bench.py --listen --models organ --stim lamp
#     python rave_bench.py --no-master            # skip the \lavaMaster insert
#     python rave_bench.py --analyze-only         # re-analyse existing WAVs
#     python rave_bench.py --out rave_bench_out   # output folder
#
# WHAT IS TESTED (each is one WAV + one row in the report)
#
#   Functional (does the layer do what the live system asks of it)
#     split_passthrough   \raveSplit at dry=1, no model: output == stimulus level
#     dry_scaling         split at dry=0.3: output ~ -10.5 dB of passthrough
#     model_on            each model, mix 1.0: wet present, dry gone
#     model_fade_off      gate 0 -> output falls to silence within fadeOut
#     silence_in          model fed silence: how much does it hallucinate?
#     stack_N             2, 3, 4 models at once: level stays bounded (mix/sqrt n)
#
#   Audio quality / limits
#     dropouts            gaps (underruns) in the wet signal while the input
#                         is steady -- THE symptom of inference not keeping up
#     clicks              sample-to-sample discontinuities
#     latency             dry->wet delay from burst-train cross-correlation
#     in_gain sweep       wet level vs input gain 0.25..4 (where the model
#                         "wakes up", where it saturates)
#     buffer sweep        the above at each buffer size rave_bench.scd built
#     cpu                 scsynth process CPU (psutil, includes the inference
#                         threads) and s.avgCPU from /status (DSP only)
#
#   Stimuli ("--stim" picks one for listen mode; the sweep uses what each
#   test needs)
#     burst   latency probe: 60 ms pink bursts, aperiodic (~1 s)
#     tone    steady 110 Hz lamp-like tone (VarSaw + octave + sub)
#     high    same at 880 Hz
#     noise   pink noise
#     lamp    emulated run: 4 \mpeVoiceRich drones with slow freq/cutoff
#             drift, \mpeHitRich ornaments, \melodyHitRich notes, one
#             \blobStrikeRich birth, one \bassStutterRich gesture --
#             the real SynthDefs with the control values VoiceManager sends
#     silence nothing
#
# OUTPUT  <out>/<scenario>.wav, <out>/rave_bench.json, <out>/rave_bench.md
#
# Requires numpy; soundfile OR scipy for WAV reading; psutil optional (CPU).

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from pythonosc import udp_client
except Exception:                       # pragma: no cover
    udp_client = None

import rave_layer as rl

SC_IP, SC_PORT = "127.0.0.1", 57110
SR = 48000
REC_BUF = 400                           # buffer id for DiskOut
REC_NODE = 990
METER_NODE = 991
STIM_BASE = 2000                        # stimulus node ids
BENCH_MODEL_BASE = 950                  # bench model nodes 950..


# ═══════════════════════════════════════════════════════════════════════════
# analysis (pure numpy; tested without SC in test_rave_bench.py)
# ═══════════════════════════════════════════════════════════════════════════

def db(x: float) -> float:
    return 20.0 * math.log10(max(float(x), 1e-9))


def frame_rms(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """RMS per frame of a mono signal."""
    if len(x) < win:
        return np.array([math.sqrt(float(np.mean(x ** 2)))]) if len(x) else np.zeros(1)
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return np.sqrt(np.mean(x[idx] ** 2, axis=1))


def to_mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else x.mean(axis=1)


@dataclass
class Dropouts:
    count: int = 0
    total_ms: float = 0.0
    longest_ms: float = 0.0
    per_second: float = 0.0
    starts_s: List[float] = field(default_factory=list)


def detect_dropouts(x: np.ndarray, sr: int = SR, win_ms: float = 5.0,
                    floor_db: float = -60.0, context_db: float = -40.0,
                    context_ms: float = 250.0, skip_s: float = 0.0) -> Dropouts:
    """A dropout is a run of short frames below floor_db whose surrounding
    context (median RMS over +/- context_ms) is above context_db -- i.e.
    silence inside sound, not silence in silence. skip_s ignores the head
    (model load / fade-in)."""
    m = to_mono(x)
    win = max(1, int(sr * win_ms / 1000)); hop = win
    r = frame_rms(m, win, hop)
    rdb = 20 * np.log10(np.maximum(r, 1e-9))
    k = max(1, int(context_ms / win_ms))
    ctx = np.array([np.median(rdb[max(0, i - k):i + k + 1]) for i in range(len(rdb))])
    bad = (rdb < floor_db) & (ctx > context_db)
    start_f = int(skip_s * 1000 / win_ms)
    bad[:start_f] = False
    out = Dropouts()
    i = 0
    while i < len(bad):
        if bad[i]:
            j = i
            while j < len(bad) and bad[j]:
                j += 1
            ms = (j - i) * win_ms
            out.count += 1; out.total_ms += ms; out.longest_ms = max(out.longest_ms, ms)
            out.starts_s.append(round(i * win_ms / 1000, 3))
            i = j
        else:
            i += 1
    dur = max(1e-9, len(m) / sr - skip_s)
    out.per_second = out.count / dur
    out.starts_s = out.starts_s[:20]
    return out


def count_clicks(x: np.ndarray, thresh: float = 0.5, skip_s: float = 0.0, sr: int = SR) -> int:
    """Sample-to-sample jumps larger than `thresh` (full scale = 1)."""
    m = to_mono(x)[int(skip_s * sr):]
    if len(m) < 2:
        return 0
    d = np.abs(np.diff(m))
    return int(np.sum(d > thresh))


def envelope(x: np.ndarray, sr: int = SR, hop_ms: float = 1.0) -> Tuple[np.ndarray, float]:
    hop = max(1, int(sr * hop_ms / 1000))
    return frame_rms(to_mono(x), hop * 2, hop), hop


def estimate_latency_ms(dry: np.ndarray, wet: np.ndarray, sr: int = SR,
                        max_ms: float = 2000.0, skip_s: float = 1.0) -> Tuple[Optional[float], float]:
    """Delay of wet relative to dry from envelope cross-correlation.
    Returns (latency_ms or None, confidence 0..1 = peak / second peak)."""
    ed, hop = envelope(dry, sr); ew, _ = envelope(wet, sr)
    n = min(len(ed), len(ew))
    s = int(skip_s * sr / hop)
    ed, ew = ed[s:n], ew[s:n]
    if len(ed) < 10:
        return None, 0.0
    ed = ed - ed.mean(); ew = ew - ew.mean()
    if np.std(ed) < 1e-6 or np.std(ew) < 1e-6:
        return None, 0.0
    ed /= np.std(ed); ew /= np.std(ew)
    maxlag = int(max_ms * sr / 1000 / hop)
    lags = np.arange(0, min(maxlag, len(ed) - 1))
    cc = np.array([np.dot(ed[:len(ed) - L], ew[L:]) / (len(ed) - L) for L in lags])
    if len(cc) == 0 or cc.max() <= 0:
        return None, 0.0
    best = int(np.argmax(cc))
    rest = np.concatenate([cc[:max(0, best - 5)], cc[best + 6:]])
    second = rest.max() if len(rest) and rest.max() > 0 else 1e-9
    conf = float(min(1.0, (cc[best] / second - 1.0))) if second > 0 else 1.0
    return round(lags[best] * hop * 1000.0 / sr, 1), round(max(0.0, conf), 2)


def spectral_centroid_hz(x: np.ndarray, sr: int = SR, skip_s: float = 0.0) -> float:
    m = to_mono(x)[int(skip_s * sr):]
    if len(m) < 2048:
        return 0.0
    n = 1 << 15
    seg = m[: min(len(m), n * 8)]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    if spec.sum() <= 0:
        return 0.0
    return float(np.sum(freqs * spec) / np.sum(spec))


@dataclass
class Measures:
    rms_db: float = -99.0
    peak_db: float = -99.0
    crest_db: float = 0.0
    silent_frac: float = 1.0
    centroid_hz: float = 0.0
    dropouts: int = 0
    dropout_ms: float = 0.0
    dropout_longest_ms: float = 0.0
    dropouts_per_s: float = 0.0
    dropout_starts_s: List[float] = field(default_factory=list)
    clicks: int = 0
    onset_s: Optional[float] = None         # first frame above -40 dB after skip
    tail_s: Optional[float] = None          # last frame above -60 dB
    nonfinite: int = 0                      # NaN / inf samples (any channel)
    over_fs: int = 0                        # samples with |x| > 1.0 (over full scale)


def measure(x: np.ndarray, sr: int = SR, skip_s: float = 0.0,
            end_s: Optional[float] = None) -> Measures:
    """Measures on a CLEANED copy (non-finite -> 0); the counts of non-finite
    and over-full-scale samples are reported separately, because one NaN
    turns every mean into NaN and hides everything else."""
    bad = ~np.isfinite(x)
    nonfinite = int(np.sum(bad))
    x = np.where(bad, 0.0, x).astype(np.float64)        # float64: 1e28**2 must not overflow
    over = int(np.sum(np.abs(x) > 1.0))
    if end_s is not None:
        x = x[: int(end_s * sr)]
    m = to_mono(x)
    body = m[int(skip_s * sr):]
    out = Measures()
    if len(body) == 0:
        return out
    out.rms_db = round(db(math.sqrt(float(np.mean(body ** 2)))), 1)
    out.peak_db = round(db(float(np.max(np.abs(body)))), 1)
    out.crest_db = round(out.peak_db - out.rms_db, 1)
    r = frame_rms(body, int(sr * 0.02), int(sr * 0.01))
    out.silent_frac = round(float(np.mean(r < 10 ** (-60 / 20))), 3)
    out.centroid_hz = round(spectral_centroid_hz(x, sr, skip_s), 0)
    d = detect_dropouts(x, sr, skip_s=skip_s)
    out.dropouts, out.dropout_ms = d.count, round(d.total_ms, 1)
    out.dropout_longest_ms, out.dropouts_per_s = round(d.longest_ms, 1), round(d.per_second, 3)
    out.dropout_starts_s = d.starts_s
    out.clicks = count_clicks(x, skip_s=skip_s, sr=sr)
    rall = frame_rms(m, int(sr * 0.02), int(sr * 0.01))
    above = np.where(rall > 10 ** (-40 / 20))[0]
    out.onset_s = round(float(above[0] * 0.01), 2) if len(above) else None
    tail = np.where(rall > 10 ** (-60 / 20))[0]
    out.tail_s = round(float(tail[-1] * 0.01), 2) if len(tail) else None
    out.nonfinite, out.over_fs = nonfinite, over
    return out


# ═══════════════════════════════════════════════════════════════════════════
# WAV i/o
# ═══════════════════════════════════════════════════════════════════════════

def read_wav(path: str) -> Tuple[np.ndarray, int]:
    try:
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        return x, sr
    except ImportError:
        from scipy.io import wavfile
        sr, x = wavfile.read(path)
        if x.dtype.kind in "iu":
            x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
        if x.ndim == 1:
            x = x[:, None]
        return x.astype(np.float32), sr


# ═══════════════════════════════════════════════════════════════════════════
# scsynth driver
# ═══════════════════════════════════════════════════════════════════════════

class SC:
    """Thin OSC sender + /status reader + recorder, in the live system's idiom."""

    def __init__(self, ip=SC_IP, port=SC_PORT, master: bool = True):
        if udp_client is None:
            raise SystemExit("python-osc is required (pip install python-osc)")
        self.osc = udp_client.SimpleUDPClient(ip, port)
        self.ip, self.port = ip, port
        self.master_node = 900 if master else None
        self._stim: List[int] = []
        self._next_stim = STIM_BASE
        self.sent = 0

    def send(self, addr, args):
        self.osc.send_message(addr, args); self.sent += 1

    # -- group hygiene, as VoiceManager + synth_palette do at startup --
    def reset(self):
        self.send("/g_freeAll", [0]); self.send("/g_new", [1, 0, 0])
        time.sleep(0.1)
        if self.master_node is not None:
            self.send("/s_new", ["lavaMaster", self.master_node, 1, 0, "wet", 0.0, "bypass", 1])
        time.sleep(0.05)

    def status(self) -> Dict[str, float]:
        """s.avgCPU / s.peakCPU / ugens / synths from /status.reply (DSP thread only)."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(0.5)
            sock.sendto(b"/status\x00\x00,\x00\x00\x00", (self.ip, self.port))
            data, _ = sock.recvfrom(1024)
            sock.close()
            from pythonosc.osc_message import OscMessage
            m = OscMessage(data).params
            return {"ugens": m[1], "synths": m[2], "groups": m[3], "synthdefs": m[4],
                    "avg_cpu": round(float(m[5]), 1), "peak_cpu": round(float(m[6]), 1),
                    "sample_rate": m[7]}
        except Exception:
            return {}

    @staticmethod
    def process_cpu(sample_s: float = 1.0) -> Optional[float]:
        """scsynth process CPU % over sample_s (all threads, inference included)."""
        try:
            import psutil
        except ImportError:
            return None
        for p in psutil.process_iter(["name"]):
            if (p.info["name"] or "").lower() in ("scsynth", "supernova"):
                p.cpu_percent(None); time.sleep(sample_s)
                return round(p.cpu_percent(None), 1)
        return None

    # -- recorder --
    def rec_start(self, path: str):
        self.send("/b_alloc", [REC_BUF, 65536, 2]); time.sleep(0.05)
        self.send("/b_write", [REC_BUF, path, "wav", "float", 0, 0, 1]); time.sleep(0.05)
        self.send("/s_new", ["benchRec", REC_NODE, 1, 0, "buf", REC_BUF])     # tail of group 0

    def rec_stop(self):
        self.send("/n_free", [REC_NODE]); time.sleep(0.05)
        self.send("/b_close", [REC_BUF]); self.send("/b_free", [REC_BUF]); time.sleep(0.1)

    # -- stimuli (all at the HEAD of group 0, like the live voices) --
    def stim(self, synthdef: str, *args) -> int:
        nid = self._next_stim; self._next_stim += 1
        self.send("/s_new", [synthdef, nid, 0, 0] + list(args))
        self._stim.append(nid)
        return nid

    def stim_free(self, gated: bool = True):
        """Gate every stimulus node (release tails play). The ids are KEPT so
        stim_kill can free anything that ignores its gate -- clearing them
        here left \\benchBurst running under every later scenario."""
        for nid in self._stim:
            self.send("/n_set", [nid, "gate", 0]) if gated else self.send("/n_free", [nid])

    def stim_kill(self):
        """Safety net: free any stimulus that ignored its gate. Most have
        already freed themselves (doneAction 2), so the frees go in one bundle
        led by /error -1, which silences scsynth's "Node not found" lines for
        that bundle only."""
        if not self._stim:
            return
        try:
            from pythonosc import osc_bundle_builder, osc_message_builder
            b = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
            for addr, args in [("/error", [-1])] + [("/n_free", [nid]) for nid in self._stim]:
                m = osc_message_builder.OscMessageBuilder(addr)
                for x in args:
                    m.add_arg(x)
                b.add_content(m.build())
            self.osc.send(b.build()); self.sent += 1
        except Exception:
            for nid in self._stim:
                self.send("/n_free", [nid])
        self._stim = []


# ═══════════════════════════════════════════════════════════════════════════
# stimulus programs (emulating the lamp)
# ═══════════════════════════════════════════════════════════════════════════

# parts of the emulated lamp run, each runnable alone as stim "lamp:<part>"
#   drones     4 x \\mpeVoiceRich with per-frame drift
#   pluck      \\mpeHitRich model 1 (DWGPlucked)     ornamental hits
#   piano      \\mpeHitRich model 2 (OteyPiano)      paired head voice
#   bowed      \\melodyHitRich legato (DWGBowed)
#   plucknote  \\melodyHitRich staccato (DWGPlucked)
#   strike     \\blobStrikeRich                      birth swell
#   bass       \\bassStutterRich (MoogLadder, SmoothDecimator)
LAMP_PARTS = ("drones", "pluck", "piano", "bowed", "plucknote", "strike", "bass")


def stim_start(sc: SC, kind: str, seconds: float, seed: int = 1) -> None:
    """Spawn the stimulus; the caller sleeps `seconds` then calls stim_free.
    `lamp` schedules its own hits/notes inside that window via timed bundles."""
    rng = np.random.default_rng(seed)
    if kind == "silence":
        return
    if kind == "burst":
        sc.stim("benchBurst", "amp", 0.5, "period", 1.0, "width", 0.06); return
    if kind == "tone":
        sc.stim("benchTone", "amp", 0.3, "freq", 110.0); return
    if kind == "high":
        sc.stim("benchTone", "amp", 0.25, "freq", 880.0); return
    if kind == "noise":
        sc.stim("benchTone", "amp", 0.2, "freq", 0.0); return
    if kind == "lamp" or kind.startswith("lamp:"):
        # "lamp" = everything; "lamp:<part>" = one part alone, to find which
        # SynthDef misbehaves (LAMP_PARTS). Same values VoiceManager sends.
        parts = set(LAMP_PARTS) if kind == "lamp" else {kind.split(":", 1)[1]}
        unknown = parts - set(LAMP_PARTS)
        if unknown:
            raise ValueError(f"unknown lamp part {unknown}")
        from pythonosc import osc_bundle_builder, osc_message_builder

        def timed(t, addr, args):
            b = osc_bundle_builder.OscBundleBuilder(t)
            m = osc_message_builder.OscMessageBuilder(addr)
            for x in args:
                m.add_arg(x)
            b.add_content(m.build()); sc.osc.send(b.build())

        t0 = time.time() + 0.3
        if "drones" in parts:
            # 4 drones, slowly drifting, 10 updates/s (like process_frame)
            drones = []
            for i, (f, pan, bank) in enumerate([(82.4, -0.5, 1), (130.8, 0.2, 2), (196.0, -0.1, 0), (261.6, 0.6, 3)]):
                nid = sc.stim("mpeVoiceRich", "freq", f, "amp", 0.16, "pan", pan, "gate", 1,
                              "attack", 1.5, "release", 2.5, "cutoff", 900 + 300 * i, "res", 0.3,
                              "wave", 0.3 + 0.2 * i, "bank", bank, "detune", 0.2, "shimmer", 0.15,
                              "noise", 0.1, "drive", 0.15, "sub", 0.3, "formant", 0.25 * i, "chorus", 0.2)
                drones.append((nid, f))
            for k in range(int(seconds * 10)):
                t = t0 + k / 10.0
                b = osc_bundle_builder.OscBundleBuilder(t)
                ph = k / 10.0
                for nid, f in drones:
                    m = osc_message_builder.OscMessageBuilder("/n_set")
                    for x in [nid, "freq", f * (1 + 0.03 * math.sin(0.2 * ph + nid)),
                              "cutoff", 800 + 600 * (0.5 + 0.5 * math.sin(0.15 * ph + nid * 0.7)),
                              "wave", 0.2 + 0.3 * (0.5 + 0.5 * math.sin(0.1 * ph)),
                              "drive", 0.1 + 0.3 * (0.5 + 0.5 * math.sin(0.07 * ph + 1))]:
                        m.add_arg(x)
                    b.add_content(m.build())
                sc.osc.send(b.build())
        # ornamental hits (plucked, model 1) and paired head (piano, model 2)
        for k in range(int(seconds * 2)):
            model = 2 if k % 4 == 0 else 1
            if (model == 1 and "pluck" not in parts) or (model == 2 and "piano" not in parts):
                continue
            t = t0 + 0.5 + k * 0.5 + float(rng.uniform(-0.03, 0.03))
            timed(t, "/s_new", ["mpeHitRich", -1, 0, 0,
                  "freq", float(rng.choice([196, 261.6, 329.6, 392])) * (0.5 if model == 2 else 1),
                  "amp", 0.25, "pan", float(rng.uniform(-0.5, 0.5)), "dur", 0.25 if model == 1 else 0.6,
                  "vel", float(rng.uniform(0.5, 0.9)), "model", model, "cutoff", 2000.0, "drive", 0.1])
        # melody notes ~1 per 2.5 s: release 1.0 -> bowed, 0.3 -> plucked (auto model)
        for k in range(int(seconds / 2.5)):
            bowed = (k % 2 == 0)
            if (bowed and "bowed" not in parts) or (not bowed and "plucknote" not in parts):
                continue
            timed(t0 + 1.2 + k * 2.5, "/s_new", ["melodyHitRich", -1, 0, 0,
                  "freq", float(rng.choice([293.7, 349.2, 440, 523.3])),
                  "amp", 0.22, "dur", 1.0, "release", 1.0 if bowed else 0.3,
                  "model", -1, "bow", 0.4, "cutoff", 1800.0])
        if seconds >= 6 and "strike" in parts:
            timed(t0 + 2.0, "/s_new", ["blobStrikeRich", -1, 0, 0, "freq", 330.0, "amp", 0.5,
                  "decay", 2.5, "blobSize", 0.6, "position", 0.4, "elongation", 0.3])
        if seconds >= 6 and "bass" in parts:
            bass = sc.stim("bassStutterRich", "freq", 45.0, "amp", 0.0, "stutterRate", 12.0, "lfoDepth", 300.0)
            for t, amp in [(t0 + 3.5, 0.2), (t0 + 6.5, 0.0)]:
                timed(t, "/n_set", [bass, "amp", amp])
        return
    raise ValueError(f"unknown stimulus {kind}")


# ═══════════════════════════════════════════════════════════════════════════
# scenarios
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    name: str
    stim: str
    models: List[str]                # keys; [] = none
    buf: int = 2048
    mix: float = 1.0
    in_gain: float = 1.0
    dry: Optional[float] = None      # override split dry (None = from mix/models)
    seconds: float = 10.0
    layer_on: bool = True
    fade_off_at: Optional[float] = None   # gate models off at t (model_fade_off)
    group: str = ""
    expect: str = ""


def model_node(i: int) -> int:
    return BENCH_MODEL_BASE + i


def run_scenario(sc: SC, sc_: Scenario, out_dir: str, record: bool = True,
                 listen: bool = False) -> Dict:
    """Spawn split + bench model nodes exactly as RaveLayer does (addBefore the
    master, or tail), play the stimulus, record, tear down. Returns run info."""
    path = os.path.abspath(os.path.join(out_dir, sc_.name + ".wav"))
    place = [2, sc.master_node] if sc.master_node is not None else [1, 0]
    n = len(sc_.models)
    dry = sc_.dry if sc_.dry is not None else rl.dry_gain(sc_.layer_on, n, sc_.mix)
    amp = rl.per_model_amp(sc_.mix, n)

    sc.send("/s_new", ["raveSplit", rl.SPLIT_NODE] + place + ["raveBus", rl.RAVE_BUS, "dry", float(dry)])
    nodes = []
    if sc_.layer_on:
        for i, key in enumerate(sc_.models):
            idx = [m.key for m in rl.MODELS].index(key)
            nid = model_node(i)
            sc.send("/s_new", [f"raveBench_{key}_{sc_.buf}", nid] + place + [
                "inBus", rl.RAVE_BUS, "out", 0, "amp", float(amp), "inGain", float(sc_.in_gain),
                "pan", rl.model_pan(idx), "gate", 1, "fadeIn", 0.05, "fadeOut", 0.3])
            nodes.append(nid)
    if listen:
        sc.send("/s_new", ["benchMeter", METER_NODE, 1, 0])
    time.sleep(0.2)

    cpu_dsp_before = sc.status()
    if record:
        sc.rec_start(path)
    t_start = time.time()
    stim_start(sc, sc_.stim, sc_.seconds)
    # settle, then sample CPU mid-scenario
    mid = max(1.0, sc_.seconds * 0.5)
    time.sleep(min(mid, 3.0))
    cpu_proc = SC.process_cpu(1.0)
    cpu_dsp = sc.status()
    elapsed = time.time() - t_start
    if sc_.fade_off_at is not None:
        rem = sc_.fade_off_at - elapsed
        if rem > 0:
            time.sleep(rem)
        for nid in nodes:
            sc.send("/n_set", [nid, "gate", 0])
        nodes = []
        elapsed = time.time() - t_start
    time.sleep(max(0.0, sc_.seconds - elapsed))
    sc.stim_free(gated=True)
    time.sleep(3.0 if sc_.stim.startswith("lamp") else 0.6)   # let releases through
    if record:
        sc.rec_stop()
    for nid in nodes:
        sc.send("/n_set", [nid, "gate", 0])
    time.sleep(0.4)
    sc.stim_kill()
    sc.send("/n_free", [rl.SPLIT_NODE])
    if listen:
        sc.send("/n_free", [METER_NODE])
    time.sleep(0.3)
    return {"wav": path if record else None, "dry": dry, "amp": amp,
            "cpu_process_pct": cpu_proc,
            "cpu_dsp_avg": cpu_dsp.get("avg_cpu"), "cpu_dsp_peak": cpu_dsp.get("peak_cpu"),
            "synths": cpu_dsp.get("synths"), "cpu_dsp_idle_avg": cpu_dsp_before.get("avg_cpu")}


def build_scenarios(models: List[str], buffers: List[int], quick: bool) -> List[Scenario]:
    secs = 6.0 if quick else 10.0
    S: List[Scenario] = []
    # functional, model-independent
    S.append(Scenario("split_passthrough", "tone", [], layer_on=False, seconds=secs, group="functional",
                      expect="rms within 0.5 dB of tone_reference"))
    S.append(Scenario("tone_reference", "tone", [], layer_on=False, dry=1.0, seconds=secs, group="reference"))
    S.append(Scenario("dry_scaling", "tone", [], layer_on=False, dry=0.3, seconds=secs, group="functional",
                      expect="rms ~ reference - 10.5 dB"))
    S.append(Scenario("burst_reference", "burst", [], layer_on=False, seconds=secs, group="reference"))
    S.append(Scenario("lamp_reference", "lamp", [], layer_on=False, seconds=secs + 2, group="reference"))
    # each part of the lamp run alone, no RAVE: finds a SynthDef that blows up
    for part in LAMP_PARTS:
        S.append(Scenario(f"iso_{part}", f"lamp:{part}", [], layer_on=False, seconds=8.0, group="isolation",
                          expect="no BLOWUP"))
    b0 = buffers[0]
    for key in models:
        S.append(Scenario(f"{key}_b{b0}_tone", "tone", [key], buf=b0, seconds=secs, group="model_on",
                          expect="wet present (rms > -40 dB), no dropouts"))
        S.append(Scenario(f"{key}_b{b0}_burst", "burst", [key], buf=b0, seconds=secs, group="latency"))
        S.append(Scenario(f"{key}_b{b0}_lamp", "lamp", [key], buf=b0, seconds=secs + 2, group="lamp"))
        if not quick:
            S.append(Scenario(f"{key}_b{b0}_high", "high", [key], buf=b0, seconds=secs, group="model_on"))
            S.append(Scenario(f"{key}_b{b0}_silence", "silence", [key], buf=b0, seconds=secs, group="silence_in",
                              expect="rms < -50 dB (no hallucination)"))
            for g in (0.25, 0.5, 2.0, 4.0):
                S.append(Scenario(f"{key}_b{b0}_gain{g:g}", "tone", [key], buf=b0, in_gain=g, seconds=secs,
                                  group="in_gain"))
            S.append(Scenario(f"{key}_b{b0}_fadeoff", "tone", [key], buf=b0, seconds=secs,
                              fade_off_at=secs * 0.5, group="model_fade_off",
                              expect="tail ends within ~0.4 s of fade point"))
        for buf in buffers[1:]:
            S.append(Scenario(f"{key}_b{buf}_tone", "tone", [key], buf=buf, seconds=secs, group="buffer_sweep"))
            S.append(Scenario(f"{key}_b{buf}_burst", "burst", [key], buf=buf, seconds=secs, group="buffer_sweep"))
    # stacking at the largest buffer
    bl = buffers[-1]
    for n in (2, 3, 4):
        if len(models) >= n:
            S.append(Scenario(f"stack{n}_b{bl}_lamp", "lamp", models[:n], buf=bl, mix=0.7, seconds=secs + 2,
                              group="stack", expect="rms bounded; dropouts = 0"))
    return S


# ═══════════════════════════════════════════════════════════════════════════
# report
# ═══════════════════════════════════════════════════════════════════════════

def analyse_all(scenarios: List[Scenario], runs: Dict[str, Dict], out_dir: str) -> Dict[str, Dict]:
    res: Dict[str, Dict] = {}
    ref_cache: Dict[str, np.ndarray] = {}

    def load(name):
        p = os.path.join(out_dir, name + ".wav")
        if name not in ref_cache and os.path.isfile(p):
            ref_cache[name] = read_wav(p)[0]
        return ref_cache.get(name)

    for sc_ in scenarios:
        x = load(sc_.name)
        row = {"scenario": asdict(sc_), **(runs.get(sc_.name) or {})}
        if x is None:
            row["error"] = "wav missing"; res[sc_.name] = row; continue
        # window: skip the model's load time (wet onset was 1.3-2.9 s in your
        # runs) and stop when the stimulus stops (+0.2 s), so warm-up and the
        # release tail are not counted as dropouts / silence
        skip = 3.0 if sc_.models else 0.3
        end = None if sc_.stim.startswith("lamp") else sc_.seconds + 0.2
        m = measure(x, SR, skip_s=skip, end_s=end)
        row["measures"] = asdict(m)
        # latency against the burst reference
        if sc_.stim == "burst" and sc_.models:
            ref = load("burst_reference")
            if ref is not None:
                lat, conf = estimate_latency_ms(ref, x)
                row["latency_conf"] = conf
                # below 0.3 the cross-correlation peak is not distinguishable
                # from the others (RAVE does not reproduce burst envelopes well)
                row["latency_ms"] = lat if conf >= 0.3 else None
        # verdicts
        v: List[str] = []
        if sc_.name == "split_passthrough":
            r = res.get("tone_reference", {}).get("measures")
            if r:
                v.append("PASS" if abs(r["rms_db"] - m.rms_db) < 0.5 else f"FAIL level diff {m.rms_db - r['rms_db']:.1f} dB")
        if sc_.name == "dry_scaling":
            r = res.get("tone_reference", {}).get("measures")
            if r:
                diff = m.rms_db - r["rms_db"]
                v.append("PASS" if abs(diff + 10.46) < 1.0 else f"FAIL expected -10.5 dB, got {diff:.1f}")
        if m.nonfinite > 0 or m.over_fs > 0:
            v.append(f"BLOWUP ({m.nonfinite} NaN/inf, {m.over_fs} samples over full scale)")
        if sc_.group in ("model_on", "lamp", "buffer_sweep", "latency", "in_gain", "stack"):
            if sc_.stim == "burst":
                # bursts are mostly silence even dry: judge against the burst reference
                r = res.get("burst_reference", {}).get("measures")
                ref_db = r["rms_db"] if r else -37.0
                v.append("silent" if m.rms_db < ref_db - 20 else "wet ok")
            else:
                v.append("silent" if m.rms_db < -40 else "wet ok")
                v.append("no dropouts" if m.dropouts == 0 else f"DROPOUTS {m.dropouts} ({m.dropout_ms:.0f} ms)")
                if m.silent_frac > 0.05:
                    v.append(f"GAPPY {100 * m.silent_frac:.0f}% silent")
            if m.clicks > 0 and m.over_fs == 0:
                v.append(f"clicks {m.clicks}")
        if sc_.group == "silence_in":
            v.append("quiet" if m.rms_db < -50 else f"HALLUCINATES {m.rms_db:.0f} dB")
        if sc_.group == "model_fade_off":
            if m.tail_s is not None and sc_.fade_off_at is not None:
                v.append("PASS" if m.tail_s <= sc_.fade_off_at + 0.8 else f"FAIL tail at {m.tail_s}s")
        row["verdict"] = "; ".join(v)
        res[sc_.name] = row
    return res


def write_report(res: Dict[str, Dict], out_dir: str, meta: Dict) -> str:
    jp = os.path.join(out_dir, "rave_bench.json")
    with open(jp, "w") as f:
        json.dump({"meta": meta, "results": res}, f, indent=1)
    lines = ["# RAVE bench report", "", f"models dir: {rl.RAVE_MODEL_DIR}", f"out: {out_dir}", ""]
    for k, v in meta.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "| scenario | group | stim | models | buf | gain | rms dB | peak | silent % | dropouts | clicks | latency ms | cpu proc % | dsp % | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        s = r["scenario"]; m = r.get("measures", {})
        lat = r.get("latency_ms")
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            name, s["group"], s["stim"], ",".join(s["models"]) or "-", s["buf"], s["in_gain"],
            m.get("rms_db", "-"), m.get("peak_db", "-"),
            f"{100 * m['silent_frac']:.0f}" if m else "-",
            f"{m.get('dropouts', '-')} ({m.get('dropout_ms', 0):.0f}ms)" if m else "-",
            m.get("clicks", "-"), lat if lat is not None else "-",
            r.get("cpu_process_pct", "-"), r.get("cpu_dsp_avg", "-"), r.get("verdict", r.get("error", ""))))
    # per-model summary
    lines += ["", "## Per model (first buffer)", "",
              "| model | wet rms dB | latency ms | dropouts/s | cpu proc % | silence-in dB | gain where rms > -30 dB |", "|---|---|---|---|---|---|---|"]
    for mdl in rl.MODELS:
        k = mdl.key
        tone = next((r for n, r in res.items() if n.startswith(f"{k}_b") and n.endswith("_tone")
                     and "measures" in r), None)
        bur = next((r for n, r in res.items() if n.startswith(f"{k}_b") and n.endswith("_burst")), None)
        sil = res.get(next((n for n in res if n.startswith(f"{k}_b") and n.endswith("_silence")), ""), {})
        if tone is None:
            continue
        gains = sorted([(float(n.split("gain")[1]), r["measures"]["rms_db"]) for n, r in res.items()
                        if n.startswith(f"{k}_b") and "gain" in n and "measures" in r] +
                       [(1.0, tone["measures"]["rms_db"])])
        wake = next((g for g, v in gains if v > -30), None)
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            k, tone["measures"]["rms_db"], (bur.get("latency_ms") or "-") if bur else "-",
            tone["measures"]["dropouts_per_s"], tone.get("cpu_process_pct", "-"),
            sil.get("measures", {}).get("rms_db", "-"), wake if wake is not None else ">4"))
    lines += ["", "## Reading the numbers", "",
              "- **iso_<part>** = one part of the emulated lamp run alone, no RAVE. A BLOWUP there names the SynthDef that goes unstable.",
              "- **BLOWUP** = NaN/inf or samples over full scale in the recording. In a *_reference row it is the stimulus SynthDefs themselves; fix that first, because a RAVE model fed it is corrupted for the rest of its life (the v4.11.2 \\raveSplit now sanitises and limits its feed to stop that).",
              "- **silent %** = share of 20 ms frames below -60 dB after the model's load time, on a steady input. Above ~5 % the wet signal is audibly choppy.",
              "- **dropouts** in a tone/lamp scenario = inference not keeping up (underruns). Compare the same model across buffers; if 8192 is clean and 2048 is not, raise ~raveBufferSize.",
              "- **cpu proc %** is the whole scsynth process (inference threads included); **dsp %** is s.avgCPU, DSP thread only. A model can peg the machine while dsp % stays low.",
              "- **latency** = burst-train delay dry->wet, shown only when the cross-correlation is unambiguous (confidence >= 0.3; raw value in the JSON).",
              "- **silence-in dB** = what the model emits when fed nothing. Above -50 dB it hallucinates; keep such models off when the lamp is quiet.",
              "- **gain where rms > -30 dB** = the input gain at which the model starts producing a useful level from a -10 dBFS tone; that is where to set RAVE_IN_GAIN.",
              "- **stackN**: level should not grow with N (mix/sqrt n); dropouts appearing only at N>1 means one model at a time is the machine's limit."]
    mp = os.path.join(out_dir, "rave_bench.md")
    with open(mp, "w") as f:
        f.write("\n".join(lines) + "\n")
    return mp


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=None, help="model keys (default: all loaded per rave_status.json)")
    ap.add_argument("--buffers", nargs="*", type=int, default=[2048, 4096, 8192])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--listen", action="store_true", help="play scenarios, no recording; Enter = next, q = quit")
    ap.add_argument("--stim", default=None, help="listen mode: restrict to one stimulus (burst tone high noise lamp silence)")
    ap.add_argument("--iso", action="store_true", help="only the isolation scenarios (each lamp part alone, no RAVE)")
    ap.add_argument("--no-master", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--out", default="rave_bench_out")
    ap.add_argument("--ip", default=SC_IP); ap.add_argument("--port", type=int, default=SC_PORT)
    a = ap.parse_args(argv)

    status = rl.RaveStatus.read(os.path.join(rl.RAVE_MODEL_DIR, rl.RAVE_STATUS_FILE))
    loaded = [m.key for m in rl.MODELS if status.of(m.key).status == "loaded"]
    models = a.models if a.models else loaded
    if not models:
        print("no models: rave_status.json lists none as loaded -- evaluate blob_synth_rave.scd + rave_bench.scd")
        return 1
    os.makedirs(a.out, exist_ok=True)
    scenarios = build_scenarios(models, a.buffers, a.quick)
    if a.iso:
        scenarios = [s for s in scenarios if s.group == "isolation" or s.name == "lamp_reference"]
    if a.stim:
        scenarios = [s for s in scenarios if s.stim == a.stim]
    meta = {"models": models, "buffers": a.buffers, "quick": a.quick, "master": not a.no_master,
            "sc_rate": status.server_rate, "nn_available": status.available, "scenarios": len(scenarios)}
    print(f"[bench] {len(scenarios)} scenarios  models={models}  buffers={a.buffers}  out={a.out}")

    runs: Dict[str, Dict] = {}
    if not a.analyze_only:
        sc = SC(a.ip, a.port, master=not a.no_master)
        sc.reset()
        for i, s in enumerate(scenarios):
            print(f"[{i + 1}/{len(scenarios)}] {s.name:34s} stim={s.stim:7s} models={','.join(s.models) or '-':20s} "
                  f"buf={s.buf} gain={s.in_gain} mix={s.mix} {s.expect}")
            if a.listen:
                r = run_scenario(sc, s, a.out, record=False, listen=True)
                print(f"    cpu proc {r['cpu_process_pct']}%  dsp {r['cpu_dsp_avg']}%")
                try:
                    if input("    Enter = next, q = quit: ").strip().lower() == "q":
                        break
                except EOFError:
                    break
                continue
            runs[s.name] = run_scenario(sc, s, a.out, record=True)
            print(f"    cpu proc {runs[s.name]['cpu_process_pct']}%  dsp {runs[s.name]['cpu_dsp_avg']}%")
        sc.send("/g_freeAll", [0]); sc.send("/g_new", [1, 0, 0])
        if a.listen:
            return 0
        with open(os.path.join(a.out, "runs.json"), "w") as f:
            json.dump(runs, f, indent=1)
    else:
        try:
            runs = json.load(open(os.path.join(a.out, "runs.json")))
        except Exception:
            runs = {}
    res = analyse_all(scenarios, runs, a.out)
    mp = write_report(res, a.out, meta)
    print(f"[bench] report: {mp}")
    bad = [n for n, r in res.items() if any(w in r.get("verdict", "") for w in
           ("FAIL", "DROPOUTS", "HALLUCINATES", "silent", "BLOWUP", "GAPPY"))]
    for n in bad:
        print(f"  ! {n}: {res[n]['verdict']}")
    print(f"[bench] {len(res) - len(bad)} clean, {len(bad)} flagged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
