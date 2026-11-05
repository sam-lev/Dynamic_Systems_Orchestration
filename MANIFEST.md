# Audible Lava / DSO — v4.9.5 complete tree

Every script in the project at its most recent version. Files not listed as
changed are byte-identical to the originals in the project folder.

Verified with the real governor stack (stubs only for `h0_persistence`,
`lamp_profile`, SuperCollider, camera and pygame): every suite passes —

| Suite | Result |
|---|---|
| `test_hazard_grouping.py` | 32 passed |
| `test_score_grammar.py` | 21 passed |
| `test_paired_voice.py` | 13 passed |
| `claude/test_v493_changes.py` | 105 passed |
| `test_beam_incumbent.py` | 15 passed (7 of them fail on the pre-fix code) |
| `claude/test_melody_rhythm.py` | 59 passed |
| `claude/test_merge_release.py` | 21 passed |
| `test_v49_wiring.py` | 23 passed |
| `test_merge_tree.py`, `test_v43_fixes.py` | not re-run here (need the real `h0_persistence`); their modules are unchanged since v4.5 |

---

## Changed files

### v4.9.5 (November 2026)

**`preference_rules.py`** — the beam defect behind the meter and harmony
churn. `PreferenceRuleModule.update` used `propose(evidence)[:beam_size]`,
and both subclasses rank proposals by the current frame's score before
truncating, so the incumbent routinely fell out of the set before Viterbi
scored it; with no incumbent there is no path to stay on, `switch_cost` never
applies, and the decode is argmax of one frame. New `_beam()` deduplicates
proposals, always includes the incumbent, and truncates around it;
`same_candidate()` lets continuous candidates match with a tolerance so they
inherit Viterbi state; unseen candidates enter one switching cost below the
best path (the old absolute `-switch_cost` let newcomers in free whenever
scores were below 1); V is renormalised each frame. On a simulated churning
lamp: meter switches 1.9/s → 0.08/s, beat-period jitter 0.11 → 0.006 s per
second, triad changes 29 → 20 per minute from the code alone.

**`score_governor.py`** — `MeterModule.same_candidate` treats periods within
6 % with the same beat count as one hypothesis and keeps the committed
period's own value (`keep_incumbent_value`); `HarmonyModule.propose` always
keeps the stay-put triad and ranks only the moves.

**`governor_weights.json`** — re-centred after the v4.9.3/4 fixes:
`grouping.prior_phrase_secs` 9 → 22, `grouping.phrase_floor_secs` 8 → 10,
`harmony.step_rate_gain` 6 → 1, `harmony.forget` 0.9 → 0.98. With the beam
fix, same simulation: ~5 boundaries/min (was 14), a triad change every ~8 s
(was every 2 s).

**`controller.py`** — the EDIT window now shows the keys that switch family
(`P W E F M G`), a one-line description of the family, the family's options
as a list with the selected one highlighted (a window of seven around the
selection, with step sizes), each option's description inline when the window
is at least 840 px wide, and the selected option's description underneath in
any case. Default window 880x440. `FAMILY_DOCS` / `PARAM_DOCS` hold the text.

**`score_governor.py`** — `gov.grouping.reset_learning` fires only on the
`r` key (a reset event). Arrows on it used to reset too; the run logged on
2 November shows twelve resets inside four seconds from a held arrow.

### v4.9.4 (November 2026)

**`audible_lava.py`** — pattern-layer hits leave as OSC bundles stamped with
their due wall-clock time (`send_timed_hits`: `due = now + (t_on − governor.t)`,
scsynth schedules them), instead of being sent the moment they were rendered
up to one look-ahead window early. The hit layer's elastic pull is now
audible. Falls back to immediate sends if the bundle classes are missing.

**`hazard_grouping.py`, `score_governor.py`** — `prior_strength` 50 → 20
(the prior keeps ~9 % of the weight at steady state instead of a fifth: a lamp
that phrases at 12 s settles at ~13.2 s); `PERS_GATE_LO` 0.20 → 0.10 (a blob
at a fifth of typical persistence now counts a third, one under a tenth still
nothing).

**The ascent whistle** (`routing.py`, `score_governor.py`, `voice_manager.py`)
— some fast rising blobs produced a shrill tone climbing with speed. Three
causes, all fixed: `VY → FMODE −0.70` morphed rising blobs toward high-pass
(which removes the fundamental), while `SPEED → CUTOFF` and
`ELONGATION → RES` on the same stretched, fast blob left a resonant peak
sliding upward — "rising brightens" is now `VY → CUTOFF −900` (+900 Hz at
full rise) and the mode is untouched; blobs the merge tree could not place
were stamped at MAX depth so `DEPTH → PITCH +12` lifted every unmatched
(small, fast, new) blob an octave — unknown depth is now 0; and
`VoiceManager._compute_targets` caps resonance while the filter is in
high-pass territory (`res_cap_hp` 0.45 at full high-pass, inactive at or
below band-pass).

**`melody_rhythm.py`** — `pull_mode` defaults to `"beat"` (onsets pulled
toward beat lines, up to half a beat); `"clock"` remains available.

**`governor_weights.json`** — all v4.9.3 keys added explicitly
(`grouping.prior_strength`, `phrase_floor_secs`, `hazard_cap_per_sec`,
`min_cue_age_secs`, `pers_tau_secs`, `count_tau_secs`, `rules.count_change`,
`patterns.tau_secs`) alongside the existing hand-tuned values. Note the file's
`prior_phrase_secs` is 9 s with an 8 s floor: the floor will bind almost
immediately, so the learned mean can only move upward from 8 s.

### Grouping feedback loop, pattern physics, melody couplings — v4.9.3

The live symptom: the learned phrase mean collapsed to under a second, the
hazard pinned at 1.0, ~2500 boundaries in 17 minutes, the schema flipping
every ~0.4 s. Both halves of `P(boundary) = h(τ)·Φ(z)` were at their bad
end — the duration model had learned nonsense and the evidence was loud all
the time. The fix guards each half separately.

**`hazard_grouping.py`** — duration half.
`GammaIntervalModel` gains `mean_floor` (`phrase_floor_secs`, 8 s: the
fitted mean never drops below it; while the floor binds the shape is capped at
the prior's so the floored model keeps a refractory tail instead of becoming a
wall at exactly 8 s), `hazard_cap_per_sec` (0.25: the per-frame probability is
now the exact survival form `1 − exp(−min(h, h_max)·dt)`, so an overdue phrase
fires with at most 25 % probability per second at neutral evidence; the old
`clip(h·dt, 0, 1)` and the "far in the tail → 1.0" branch are gone),
`PRIOR_STRENGTH` 2 → 50 pseudo-observations (20 since v4.9.4; the first
boundaries cannot drag the mean down; with the 0.995 forgetting the
observation count saturates at 200, so the prior keeps a share of the weight
at steady state), and `reset()`.
`HazardGroupingModule.reset_learning()` restores the prior live (fresh
interval model and evidence null, beam and edited prior parameters kept).
New weight keys `prior_strength`, `phrase_floor_secs`, `hazard_cap_per_sec`
are live via `set_weight`. Also `REVISION_WINDOW_S` (1 s): the beam keeps one
hypothesis per 250 ms bin, so two rival placements of the *same* boundary could
both be committed when the best path switched between them — firing twice and
teaching the interval model a 0.2 s "phrase"; a commit within a second of the
previous one now moves the boundary instead of firing.

**`score_governor.py`** — evidence half, plus the pattern layer.
Lifecycle gate: a birth is never a cue; a death, merge (its retired blob) or
split (its parent) counts only when that blob has been tracked for
`min_cue_age_secs` (3 s). `ScoreState.events` still lists every event for the
HUD; `ScoreState.cue_events` lists the ones that counted. Count evidence: each
blob is weighted by an exponential moving average of its persistence (time
constant `pers_tau_secs` 3 s, started at zero so a blob earns its weight)
relative to the typical blob, the weighted sum is smoothed over
`count_tau_secs` (2 s) into `ScoreState.count_smooth`, and a new
`count_change` grouping rule reads its z-score (sigma floored at 0.15 blobs).
That smoothed count also replaces the raw count in the macrostate symbol
(`SystemStateTracker.update(count_override=…)`) and in the meter cardinality
rule. `gov.grouping.reset_learning` is handled as an action. Pattern layer:
`pattern_for_blob` takes its slot count `n` from the blob's own rise/fall
period (beats per cycle, folded by octave division into 3..16 as the meter
folds the tactus), one slot per beat, the cycle stretched to `n` beats
(`("slow", n/beats_per_bar)`); hits `k = 1 + 3·area_n + 0.25·n` clamped to
`2..n`, so density per beat falls as the cycle lengthens — a slow blob gets a
long sparse cycle, a fast one a short busy one; elongation becomes swing
(`("swing", 0.35·elongation, n)`); area, persistence, elongation and period
are moving averages (`patterns.tau_secs` 3 s) so the rhythm cannot change
frame to frame; rotation is fixed per blob instead of following beat phase.
Ornamental hits occupy half their (now one-beat) slot (`orn_dur_scale`).

**`elastic_clock.py`** — `set_period()` preserves BAR phase. It used to
re-anchor bar 0 to the current beat, and the governor calls it every frame, so
the bar never left its first beat: `beat_index` was always 0 (every beat read
as the downbeat), and the hit layer could only ever render a pattern's opening
slots — which, combined with a rotation that swept with beat phase, compressed
a whole bar's pattern into every beat (the "busy hit layer"). Beat phase is
preserved exactly as before.

**`system_state.py`** — `count_override` keyword on `update()`/`symbolize()`;
default behaviour unchanged.

**`controller.py`** — `gov` family gains `gov.grouping.reset_learning`
(select it, press `r` or an arrow; the HUD value is the reset count),
`gov.grouping.phrase_floor_secs`, `gov.grouping.hazard_cap_per_sec`.

**`melody_rhythm.py`** — `FieldScale`: a five-note scale from the governor's
pitch field (the triad) plus the two diatonic neighbour degrees the current
schema's melody contour uses most (2, 4, 6 or 7 of the triad's own major or
minor mode; C major + the default neighbours 2, 6 is exactly the old C
pentatonic, which is also what plays with the governor off). The root is the
triad root placed within half an octave of `PitchParams.root`, which is now a
register. A one-beat latch applies a new scale only when `beat_phase` wraps.
Onset pull: a note that fires at governor time `t` is held until
`clock.pull(t)` with the elastic strength the governor sets from the Kuramoto
order parameter (`pull_mode="clock"` = subdivision grid as `clock.pull` does,
`"beat"` = beat lines); only delays are possible, so the pull is one-sided, and
the note is released on the first frame at or after its time, frame-quantised
like everything else. Top band lowered: `octave_range` 3 → 2 (C3–C5).

**`routing.py`** — `Y_NORM → PITCH` +12 → +7. **`voice_manager.py`** —
`ROOT_NOTE` 48 → 36 (C2), on top of the v4.9.1/v4.9.2 release-path fixes and
default-group recreation.

**`governor_weights.json`** — `_version` is deliberately NOT bumped, so a
file without the new keys keeps working (they fall back to the module
defaults); the Project copy carries them explicitly since v4.9.4.

### v4.9 → v4.9.2 (October 2026, delivered as files)

`routing.py` defaults restored (`Y_NORM → RES +24`, `ELONGATION → PITCH −30`
reverted); warm-lamp background capture plumbed through `BlobDetector`;
merge-tree wax mask computed on the ROI; `melody_rhythm.py` + `melody_hit.scd`
(hazard-driven onsets, height-band pitch with hysteresis, beat-length
durations); game bridge removed from `audible_lava.py`; `voice_manager.py`
merge-slot leak and orphan guard, `/g_new 1` after `/g_freeAll 0`;
`blob_synth.scd` wavetables actually filled (`sine1`, not `sine1Msg`; buffers
allocated from 0).

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

1. Boot scsynth, evaluate `blob_synth.scd`, then `blob_synth_hits.scd`, then
   `melody_hit.scd`.
2. `python audible_lava.py` — the governor, pattern layer and melody layer are
   on by default (`--no-governor`, `--no-patterns`, `--no-melody`).
3. EDIT mode → `g` for governor weights; arrows step, `=`/`-` coarse, `r` reset.
   On `gov.grouping.reset_learning`, `r` restores the phrase prior live.
4. Watch `governor_runtime.json` (`grouping.phrase_model` shows the fitted
   mean, whether the floor is active, and the prior weight; `count_smooth`,
   `count_z` and `n_cues_gated` show what the evidence gate is doing);
   hand-set values live in `governor_weights.json`.

## Outstanding

* **Detection churn is the limiting factor.** Logged runs showed ~418 blob IDs
  in 198 s (mean lifetime ~5 s) for wax moving 0.5–2 px/frame. v4.9.3 stops
  that churn from driving the grouping model, but three of six time-span
  prominence terms and most per-blob periodicity estimates are still starved
  by it. Verify the GUDHI fix is deployed, re-calibrate `pers_thresh`,
  re-capture the background warm, then tune tracker association.
* The Markov alphabet is large enough relative to run length that `ambiguity`
  can saturate at 1.0, making one of four tension terms a constant. (The beam
  incumbent defect listed here before is fixed in v4.9.5.)
* SuperCollider files are written to the existing conventions but have not been
  run against a live server here.
