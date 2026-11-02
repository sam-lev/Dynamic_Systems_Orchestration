# Audible Lava / DSO — v4.5 complete tree

Every script in the project at its most recent version. Files not listed as
changed are byte-identical to the originals in the project folder.

Verified in this directory: all 18 modules import cleanly, no syntax errors,
and every test suite passes —

| Suite | Result |
|---|---|
| `test_merge_tree.py` | 29 passed |
| `test_hazard_grouping.py` | 26 passed |
| `test_score_grammar.py` | 21 passed |
| `test_v43_fixes.py` | 23 passed |
| `test_edit_rebase.py` (legacy) | 13 passed |
| `test_voices.py` (legacy) | 35 passed |

---

## Changed files

### Detection / topology — v4.2

**`h0_persistence.py`** *(new)* — the single validated H0 extractor, shared by
detection and the merge tree so the two can no longer drift. Fixes three
defects found by the merge-tree suite: the GUDHI dimension order was
`[sh, sw]` when it must be `[sw, sh]` (a row-major flatten was read at the
wrong stride, scrambling every non-square frame and producing 29 spurious
"persistent" leaves from a single smooth Gaussian); peak localization used
first-pixel-at-intensity matching, which collapsed equal-height peaks; and the
essential class was dropped entirely, so the *brightest* blob never appeared as
a leaf.

**`merge_tree.py`** — rewritten around `h0_persistence`. Internal nodes are now
true saddles found by connected-component analysis at the merge level rather
than by death-sorted popping, so the tree is a faithful superlevel-set
hierarchy and can legitimately be read as a grouping / Schenkerian level tree.
Adds `depth`, `parent`, `leaf_count`, `is_essential`, `death`, plus
`iter_nodes`, `assign_depths`, `tree_stats` (Colless imbalance). Public API
unchanged.

**`blob_detection.py`** — consumes `h0_peaks` for steps 5–6. **Re-calibrate
`pers_thresh` after adopting this**: the profile's `expected_persistence_*`
were tuned against scrambled values, and on the masked field isolated blobs now
report persistence ≈ their peak brightness.

### Score grammar — v4.2

**`system_state.py`** *(new, updated v4.3)* — `SystemSummary` macrostate,
entropies (Shannon, permutation, persistence, occupancy), EMD and diagram
Wasserstein, BOCPD, `MarkovExpectation`, `SystemStateTracker`.

**`preference_rules.py`** *(new, updated v4.5)* — `Rule`/`RuleRegistry`,
`PreferenceRuleModule` (Viterbi beam with switching cost), `ParticleFilterModule`,
`WeightStore`, `softmax_weights`.

**`pattern_language.py`** *(new, updated v4.4)* — Tidal-style `Pattern` protocol
(`query(span) -> [Event]`), primitives including Euclidean rhythms, 21
registered transformations, `apply_spec`, `PatternScheduler`.

**`elastic_clock.py`**, **`harmony_tonnetz.py`**, **`schema_lexicon.py`** *(new)* —
projective clock with Kuramoto-governed elastic pull and London bounds;
Tonnetz with P/L/R and parsimonious paths; 12-frame schema lexicon with
first-order succession.

**`score_governor.py`** *(new, updated v4.5)* — `ScoreState`, all rules, the
governor modules, softmax role weights, boundary evidence, merge-tree
consumption, JSON weight I/O, EDIT-surface hook, per-blob source stamping,
and the pattern-layer tick.

**`routing.py`** — 8 governor `Source`s plus (v4.4) ten default routings that
actually use them.

**`controller.py`** — `gov` EDIT family (16 parameters), `g` key, step sizes.

### Fixes — v4.3

**`background_reference.py`** — warm-lamp capture. A cold-lamp median bakes wax
into the reference if you start while running; over a long window each pixel
shows background most of the time and wax only intermittently, and since wax is
*brighter*, the low percentile is the background. Simulation showed the 25th
percentile holds to ~75% per-pixel occupancy where the median fails past 50%.
Wall-clock paced, with `refine()` and an `occupancy()` diagnostic.

```python
BlobDetector(..., background="frozen", bg_ref_mode="warm",
             bg_ref_capture_secs=90.0, bg_ref_percentile=25.0)
```

**`voice_manager.py`** — bass-stutter no longer sticks on: the exit thresholds
were looser than entry in every dimension, so a blob with steady area satisfied
`d_area >= -0.02` forever. Narrowed hold region, 6 s hard cap, 3-frame exit
debounce, 2-voice cap, 3 s retrigger cooldown. Also carries the v4.2 governor
consumption layer and the v4.5 keyboard fix.

**`system_state.py`** — Kuramoto reads true per-blob oscillation phase. The old
pixel-space `atan2` proxy returned ~1.0 permanently, silently hard-quantizing
the elastic clock. Two bugs fixed: the proxy itself, and a px/frame vs rad/sec
unit mismatch that collapsed phases onto 0 and π.

### Grouping — v4.4 → v4.5

**`hazard_grouping.py`** *(new)* — grouping as a hidden semi-Markov model
decoded by beam Viterbi. Boundary is an event: per-frame probability is a Bayes
update where the online Gamma hazard is the prior odds and the z-scored evidence
contributes the likelihood ratio `Φ(z)/(1−Φ(z))`, exactly 1 at the median.
A six-hypothesis beam compares whole segmentation *paths*, so a boundary is
confirmed by what follows it; commitment is deterministic and lagged, leaving
recent boundaries revisable.

This replaced a two-state Viterbi beam whose commit test was
`argmax == boundary AND posterior > threshold` — with two candidates argmax *is*
posterior > 0.5, so the threshold did nothing below 0.5 and the argmax nothing
above it. Deleted with it: `commit_threshold`, `switch_cost`, `forget`,
`temperature`, the 1.5 s refractory constant, and the `regularity` rule weight.
Added: `prior_phrase_secs` and `evidence_rate`.

**`preference_rules.py`** — `RuleRegistry.rules()` returns copies. It previously
returned the shared `Rule` objects, so every module instance mutated the same
weights: two governors would silently share them.

**`voice_manager.py`** — `note_on` on an `IDLE` slot spawns a `KEYBOARD_ONLY`
voice. It previously returned immediately, so the MPE keyboard was silent unless
a blob happened to be bound to the selected slot — the whole player path existed
but was unreachable in the common case.

### Wiring — v4.4

**`audible_lava.py`** — constructs the governor, computes the merge tree on
detection frames, calls `governor.update()` and `governor.tick_patterns()` every
frame, and routes `gov.*` events. Flags: `--no-governor`, `--no-patterns`.

**`pattern_language.py`** — the scheduler tracked progress in *cycle numbers*,
but `set_period` re-anchors `t0` every frame to preserve phase, so cycle
numbering is not monotonic and the marker raced ahead and silenced the layer
permanently (3 hits, then nothing). Now tracks seconds. Separately, elastic pull
moved onsets slightly before the window start where they were dropped rather
than clamped — that was discarding ~75% of all hits.

**`blob_synth_hits.scd`** *(new)* — `\mpeHit` (gated/percussive sibling of
`\mpeVoice`) and `\preSound`. Stock UGens only: `\mpeVoice`'s header comment
says "SVF" but its code uses `RLPF`/`BPF`/`RHPF` + `SelectX`, and this file
follows the code.

---

## Running

1. Boot scsynth, evaluate `blob_synth.scd`, then `blob_synth_hits.scd`.
2. `python audible_lava.py` — the governor and pattern layer are on by default.
3. EDIT mode → `g` for governor weights; arrows step, `=`/`-` coarse, `r` reset.
4. Watch `governor_runtime.json`; hand-set values live in `governor_weights.json`
   (`governor_weights_live.json` is a measured, more reactive alternative).

## Outstanding

* **Detection churn is the limiting factor.** Logged runs showed ~418 blob IDs
  in 198 s (mean lifetime ~5 s) for wax moving 0.5–2 px/frame. Three of six
  time-span prominence terms and most per-blob periodicity estimates are starved
  by this. Verify the GUDHI fix is deployed, re-calibrate `pers_thresh`,
  re-capture the background warm, then tune tracker association.
* **Two mechanism issues remain in the beam modules** (meter, harmony):
  the meter proposal step evicts the incumbent before Viterbi sees it, so
  `switch_cost` cannot behave as documented; and the Markov alphabet is large
  enough relative to run length that `ambiguity` saturates at 1.0, making one of
  four tension terms a constant.
* SuperCollider files are written to the existing conventions but have not been
  run against a live server here.
