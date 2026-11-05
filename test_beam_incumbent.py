#!/usr/bin/env python3
# test_beam_incumbent.py -- Audible Lava / DSO v4.9.5
#
# The beam defect in PreferenceRuleModule: the incumbent fell out of the
# candidate set before Viterbi scored it, so switch_cost never applied, and
# continuous candidates (meter periods) never inherited Viterbi state because
# their keys changed every frame. Logged runs: ~2.8 meter switches and ~1.1
# harmony switches per second. No SuperCollider, camera, or pygame.
#
# Run: python test_beam_incumbent.py

import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("/mnt/project", HERE):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from preference_rules import PreferenceRuleModule, RuleRegistry, rule     # noqa: E402
import score_governor as SG                                               # noqa: E402
from harmony_tonnetz import Triad, TonnetzWalker                          # noqa: E402

PASS = FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok   {msg}")
    else:
        FAIL += 1; print(f"  FAIL {msg}")


# A toy module whose proposals are ranked by this frame's score and truncated
# -- exactly the pattern MeterModule and HarmonyModule used.
RuleRegistry._rules.pop("toy_beam", None)
rule("toy_beam", 1.0)(lambda ev, c: ev["s"].get(c, 0.0))


class Toy(PreferenceRuleModule):
    name = "toy_beam"
    beam_size = 2

    def propose(self, ev):
        return sorted(ev["s"], key=lambda c: -ev["s"][c])[: self.beam_size]

    def candidate_key(self, c):
        return c


def test_incumbent_survives_truncation():
    print("[beam] the incumbent stays in the beam when it ranks out of the proposals")
    m = Toy(switch_cost=3.0, forget=0.98)
    for _ in range(40):
        m.update({"s": {"A": 0.9, "B": 0.5, "C": 0.4}})
    check(m.committed == "A", "A committed")
    # one noisy frame where A ranks third: before the fix A was truncated away
    # and the committed choice jumped on that single frame
    m.update({"s": {"A": 0.55, "B": 0.6, "C": 0.6}})
    check("A" in m.candidates, "A still in the beam")
    check(m.committed == "A", "one noisy frame does not switch (switch_cost applies)")
    # a sustained, decisive change still switches
    for _ in range(60):
        m.update({"s": {"A": 0.1, "B": 0.95, "C": 0.2}})
    check(m.committed == "B", "a sustained change does switch")
    check(len(m.candidates) <= m.beam_size, f"beam size respected ({len(m.candidates)})")


def test_switch_rate_under_noise():
    print("[beam] switching rate under per-frame noise")
    rng = random.Random(3)
    m = Toy(switch_cost=1.0, forget=0.94)
    sw0 = m.switches
    for _ in range(3000):
        m.update({"s": {c: 0.5 + rng.uniform(-0.15, 0.15) for c in "ABCD"}})
    rate = (m.switches - sw0) / 3000
    check(rate < 0.02, f"near-equal noisy candidates switch on {100 * rate:.2f}% of frames (argmax would be ~75%)")


def test_unseen_candidates_do_not_enter_free():
    print("[beam] an unseen candidate enters one switching cost below the best path")
    m = Toy(switch_cost=2.0, forget=0.98)
    for _ in range(30):
        m.update({"s": {"A": 0.5, "B": 0.4}})     # scores < 1 -> negative logs
    m.update({"s": {"Z": 0.55, "A": 0.5}})
    check(m.committed == "A", "a brand-new, slightly better candidate does not win on its first frame")


def test_meter_tolerance():
    print("[meter] continuous periods: same hypothesis within 6 %, incumbent keeps its value")
    mm = SG.MeterModule(None)
    a, b, c = {"period": 0.80, "beats": 2}, {"period": 0.83, "beats": 2}, {"period": 0.90, "beats": 2}
    check(mm.same_candidate(a, b), "0.80 and 0.83 s are the same hypothesis")
    check(not mm.same_candidate(a, c), "0.80 and 0.90 s are different")
    check(not mm.same_candidate(a, {"period": 0.80, "beats": 3}), "different beat counts are different")
    rng = random.Random(5)
    for _ in range(300):
        base = 0.8 * (1 + rng.uniform(-0.03, 0.03))
        ev = {"candidate_periods": [base, base * 1.5], "period_strength": {round(base, 3): 0.8, round(base * 1.5, 3): 0.3},
              "blob_count": 2.0, "period_ratio": 0.0}
        mm.update(ev)
    check(mm.switches <= 3, f"jittering period estimates: {mm.switches} switches in 300 frames (was one per key change)")
    check(abs(mm.committed["period"] - 0.8) < 0.05, f"committed period stays put ({mm.committed['period']:.3f})")


def test_harmony_keeps_stay_candidate():
    print("[harmony] the stay-put triad is always proposed")
    w = TonnetzWalker(home=Triad(0, True))
    hm = SG.HarmonyModule(w)
    hm.beam_size = 3
    ev = {"step_budget": 1.0, "equilibrium_gradient": 1.0, "tension": 0.9, "surprisal_norm": 0.9,
          "sounding_pcs": []}
    props = hm.propose(ev)
    check(props[0]["triad"] == w.current and props[0]["dist_from_current"] == 0, "stay-put candidate first")
    check(len(props) == 3, "beam size respected")


def test_governor_rates():
    print("[governor] meter switches stay rare on oscillating blobs")
    for p in ("/tmp/_beam.json",):
        if os.path.exists(p):
            os.remove(p)
    g = SG.ScoreGovernor(320, 240, weights_path="/tmp/_beam.json", runtime_path="/tmp/_beamr.json",
                         log_path="/tmp/_beaml.txt", fps=15)
    from dataclasses import dataclass

    @dataclass
    class B:
        id: int; centroid: tuple; area: float = 3000.0; birth: float = 150.0; death: float = 20.0
        persistence: float = 40.0; vx: float = 0.0; vy: float = 0.0; speed: float = 0.0
        ay: float = 0.0; direction: str = "turning"; age: int = 0

    for f in range(15 * 120):
        t = f / 15
        bl = []
        for i, (P, ph) in enumerate([(12, 0), (9, 2), (10, 4)]):
            y = 120 + 80 * math.sin(2 * math.pi * t / P + ph)
            vy = 80 * 2 * math.pi / P * math.cos(2 * math.pi * t / P + ph) / 15
            bl.append(B(i, (60 + 90 * i, y), vy=vy, speed=abs(vy), age=f))
        g.update(bl, 1 / 15)
    check(g.meter.switches / 120 < 0.3, f"meter switches {g.meter.switches / 120:.2f}/s (logged 2.8/s)")


if __name__ == "__main__":
    for t in (test_incumbent_survives_truncation, test_switch_rate_under_noise,
              test_unseen_candidates_do_not_enter_free, test_meter_tolerance,
              test_harmony_keeps_stay_candidate, test_governor_rates):
        try:
            t()
        except Exception as e:
            FAIL += 1
            print(f"  ERR  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
