#!/usr/bin/env python3
"""
diagnose_score_grammar.py -- Audible Lava / DSO v4.6

Answers one question: for each layer between the lamp and the loudspeaker, is
it actually doing anything?

The governor can compute a perfectly good ScoreState every frame and still be
inaudible, because there are five separate places the signal can stop. This
walks all five and prints VERDICTs. Run it standalone for a static check, or
import `probe(...)` and call it from the main loop for a live one.

    python diagnose_score_grammar.py            # static + synthetic check
    python diagnose_score_grammar.py --live     # instructions for in-loop use
"""

import math
import os
import sys
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OK, BAD, WARN = "  OK  ", " FAIL ", " WARN "


def line(tag, name, detail=""):
    print(f"[{tag}] {name}" + (f"\n         {detail}" if detail else ""))


# ── 1. Is the code that is running the code you think is running? ───────────

def check_modules():
    print("\n=== 1. MODULE / VERSION CHECK ==========================================")
    import routing, voice_manager, score_governor, blob_tracker, system_state
    for m in (routing, voice_manager, score_governor, blob_tracker, system_state):
        p = os.path.dirname(os.path.abspath(m.__file__))
        tag = OK if p == HERE else WARN
        line(tag, f"{m.__name__:16s} loaded from {p}",
             "" if p == HERE else "NOT this directory -- you may be running an older copy")

    # Feature fingerprints: each of these exists only in the current version.
    feats = [
        ("governor Sources in routing", hasattr(routing.Source, "TENSION")),
        ("governor routings in DEFAULT_BLOB_ROUTING",
         any(r.src.value in ("tension", "beat_phase") for r in routing.DEFAULT_BLOB_ROUTING)),
        ("VoiceManager.score_state hook", "score_state" in voice_manager.VoiceManager.__init__.__code__.co_names
         or hasattr(voice_manager.VoiceManager, "score_gain")),
        ("pattern layer on governor", hasattr(score_governor.ScoreGovernor, "tick_patterns")),
        ("hazard grouping", "hazard_grouping" in str(score_governor.__dict__.get("HazardGroupingModule", ""))
         or hasattr(score_governor, "HazardGroupingModule")),
        ("tracker smoothed topology", hasattr(blob_tracker.BlobTracker, "_update_topo_stats")),
        ("normalised entropy", hasattr(system_state.MarkovExpectation, "normalised_conditional_entropy")),
    ]
    for name, ok in feats:
        line(OK if ok else BAD, name,
             "" if ok else "this file is from an older version -- re-copy it")


# ── 2. Does the governor produce VARYING output? ────────────────────────────

@dataclass
class _B:
    id: int; centroid: tuple; area: float; birth: float = 150.0; death: float = 20.0
    persistence: float = 40.0; vx: float = 0.0; vy: float = 0.0; speed: float = 0.0
    ay: float = 0.0; direction: str = "turning"; age: int = 0
    d_area: float = 0.0; d_persistence: float = 0.0; elongation: float = 2.0


def check_governor(seconds=120, fps=15):
    print("\n=== 2. GOVERNOR OUTPUT =================================================")
    from score_governor import ScoreGovernor
    for p in ("/tmp/_dsg.json", "/tmp/_dsgr.json"):
        if os.path.exists(p):
            os.remove(p)
    g = ScoreGovernor(320, 240, weights_path="/tmp/_dsg.json",
                      runtime_path="/tmp/_dsgr.json", log_path="/tmp/_dsgl.txt", fps=fps)
    rec = {k: [] for k in ("tension", "ambiguity", "beat_phase", "period_s", "roughness")}
    triads, hits, bounds = [], 0, 0
    for f in range(int(seconds * fps)):
        t = f / fps
        blobs = []
        for i in range(3):
            per = 9.0 + 3 * i
            y = 120 + 80 * math.sin(2 * math.pi * t / per + i)
            vy = 80 * 2 * math.pi / per * math.cos(2 * math.pi * t / per + i) / fps
            pers = 40 + 20 * math.sin(2 * math.pi * t / (11 + i))
            blobs.append(_B(i, (60 + 80 * i, y), 3000 + 900 * math.sin(t / 5 + i),
                            birth=120 + pers, persistence=pers, vy=vy, speed=abs(vy), age=f))
        s = g.update(blobs, 1.0 / fps)
        hits += len(g.tick_patterns(blobs, voice_of=lambda bid: bid))
        bounds += bool(s.boundary)
        for k in rec:
            rec[k].append(getattr(s, k if k != "roughness" else "roughness_target"))
        triads.append(s.triad)
    for k, v in rec.items():
        a = np.array(v, dtype=float)
        moving = a.std() > 1e-3
        line(OK if moving else BAD, f"{k:12s} min {a.min():.3f} max {a.max():.3f} sd {a.std():.4f}",
             "" if moving else "CONSTANT -- this parameter modulates nothing")
    nch = sum(1 for a, b in zip(triads, triads[1:]) if a != b)
    line(OK if nch else BAD, f"chord changes: {nch} in {seconds}s")
    line(OK if hits else BAD, f"pattern hits generated: {hits} in {seconds}s",
         "" if hits else "no hits -- the beat layer is producing nothing")
    line(OK if bounds else WARN, f"boundaries: {bounds} in {seconds}s",
         "" if bounds else "none -- expected at this phrase length only for short runs")
    return g, blobs


# ── 3. Do the stamped values reach the routing matrix? ──────────────────────

def check_routing(blobs):
    print("\n=== 3. ROUTING =========================================================")
    from routing import DEFAULT_BLOB_ROUTING, Source, extract_source_value
    from lamp_profile import LampProfile
    prof = LampProfile.default()
    gov = [r for r in DEFAULT_BLOB_ROUTING
           if r.src.value in ("tension", "beat_phase", "prominence_rank", "phrase_position",
                              "surprisal", "ambiguity", "depth", "role_weight")]
    line(OK if gov else BAD, f"{len(gov)} governor routings in DEFAULT_BLOB_ROUTING",
         "" if gov else "the governor modulates nothing -- routing.py is out of date")

    print("         stamped values on each blob (all-identical = no differentiation):")
    for src in (Source.TENSION, Source.BEAT_PHASE, Source.PROMINENCE_RANK,
                Source.DEPTH, Source.AMBIGUITY, Source.ROLE_WEIGHT):
        vals = [extract_source_value(src, b, prof) for b in blobs]
        spread = max(vals) - min(vals)
        if src in (Source.TENSION, Source.BEAT_PHASE, Source.AMBIGUITY):
            tag = OK          # these are global by design
            note = "global by design"
        else:
            tag = OK if spread > 1e-6 else WARN
            note = "" if spread > 1e-6 else "identical across blobs -- no per-voice differentiation"
        print(f"  [{tag}] {src.value:16s} {[round(v,3) for v in vals]}  {note}")


# ── 4. Does the voice layer consume it? ─────────────────────────────────────

def check_voices(gov_state):
    print("\n=== 4. VOICE LAYER =====================================================")
    from voice_manager import VoiceManager, VoiceState
    from lamp_profile import LampProfile

    class _OSC:
        def __init__(self): self.sent = []
        def send_message(self, a, b): self.sent.append((a, b))

    vm = VoiceManager.__new__(VoiceManager)
    VoiceManager.__init__(vm, n_slots=6, profile=LampProfile.default())
    vm.osc = _OSC()
    line(OK if hasattr(vm, "score_state") else BAD, "VoiceManager has score_state attribute")
    line(OK if getattr(vm, "score_gain", 0) > 0 else BAD,
         f"score_gain = {getattr(vm, 'score_gain', None)}",
         "" if getattr(vm, "score_gain", 0) > 0 else "zero -- the score layer is muted")

    print("         NOTE: routing contributions apply ONLY when a voice is in")
    print("         BLOB_ONLY or HYBRID with a live blob bound. IDLE and")
    print("         KEYBOARD_ONLY voices get no routing at all -- if the")
    print("         BindingManager is not binding blobs to slots, every")
    print("         routing in the system is inert, not just the governor ones.")


# ── 5. What can still fail silently ─────────────────────────────────────────

def check_silent_failures():
    print("\n=== 5. SILENT FAILURE MODES (cannot be checked from Python) ============")
    items = [
        ("\\mpeHit not defined on the SuperCollider server",
         "The pattern layer sends /s_new mpeHit. If blob_synth_hits.scd was not "
         "evaluated AFTER booting the server, scsynth logs 'SynthDef not found' "
         "to the SC post window and drops every message. Python sees no error. "
         "This is the single most likely cause of 'no beat'. Test in SC:\n"
         "             Synth(\\mpeHit, [\\freq, 330, \\amp, 0.4, \\dur, 0.15]);"),
        ("Hits are quiet by design for high-prominence blobs",
         "pattern_gain 0.35 x (0.45 + 0.55*(1-rank)). A head (rank 1.0) gets "
         "0.16 x velocity. With only 2-3 blobs everyone ranks near 1.0, so every "
         "hit is at its quietest. Raise governor.pattern_gain to 0.6-0.8 to hear "
         "whether the layer is alive at all."),
        ("Stale bytecode / running from another directory",
         "Delete __pycache__ and confirm section 1 shows every module loading "
         "from the directory you edited."),
        ("--no-governor or --no-patterns left on the command line",
         "Both silently disable a whole layer."),
        ("No blobs bound to voice slots",
         "tick_patterns skips any blob whose id is not in voices.voices_by_blob, "
         "and _compute_targets applies no routing to unbound voices. Print "
         "len(voices.voices_by_blob) in the main loop -- if it is 0, nothing "
         "downstream of detection can sound."),
        ("Merge tree not reaching the governor",
         "DEPTH is stamped from leaf depth; with no tree every blob gets the "
         "same value, so DEPTH->PITCH transposes everything uniformly instead "
         "of separating background from surface. Check that root/leaves are "
         "non-None on detection frames."),
    ]
    for name, detail in items:
        line(WARN, name, detail)


def probe(governor, voices, tracked, every=45, _n=[0]):
    """Call once per frame from the main loop for a live version of this check.

        from diagnose_score_grammar import probe
        probe(governor, voices, tracked)
    """
    _n[0] += 1
    if _n[0] % every:
        return
    s = governor.state
    bound = len(getattr(voices, "voices_by_blob", {}))
    active = sum(1 for v in voices.voices_by_slot
                 if v.state.name in ("BLOB_ONLY", "HYBRID"))
    print(f"[probe] t={s.t:6.1f} blobs={len(tracked):2d} bound={bound:2d} routed_voices={active:2d} "
          f"tension={s.tension:.2f} amb={s.ambiguity:.2f} beat={s.beat_phase:.2f} "
          f"period={s.period_s:.2f} triad={s.triad} bounds={governor.grouping.n_boundaries}")
    if bound == 0:
        print("        !! no blobs bound to voice slots -- ALL routing is inert")
    if active == 0:
        print("        !! no voice in BLOB_ONLY/HYBRID -- routing never applied")


if __name__ == "__main__":
    if "--live" in sys.argv:
        print(__doc__)
        print("Add to the main loop after governor.update(...):\n"
              "    from diagnose_score_grammar import probe\n"
              "    probe(governor, voices, tracked)")
        sys.exit(0)
    check_modules()
    g, blobs = check_governor()
    check_routing(blobs)
    check_voices(g.state)
    check_silent_failures()
    print("\nIf sections 1-4 are green, the break is in section 5 -- most often "
          "\\mpeHit missing on the server, or no blobs bound to voice slots.\n")
