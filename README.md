# Dynamical Systems Orchestration (DSO)

**A lava lamp, filmed, is read as a score.** Each wax blob gets a voice. Where the
blob is, how fast it moves, how it is shaped, and how it is born, merges, splits
and dies all become sound. Above the voices, a score grammar listens to the whole
lamp and decides where phrases end, what the pulse is, which chord the ensemble
is leaning toward, and which blob is leading. The lamp is a closed thermodynamic
system that never repeats itself, so every run is a new piece.

*Samuel D. Leventhal · v4.x, November 2026 · macOS (Intel and M-series) · Python 3.11 + SuperCollider 3.13*

- Project page: https://samleventhal.com/dynamical-systems-orchestration
- Companion PDFs (methodology, score grammar, synth palette, tracker stability) are
  linked from the project page.

---

## How it works, in one pass through the loop

Every frame (~15 fps) the main loop in `audible_lava.py` does this:

1. **See the lamp.** A USB camera frames the lamp. A frozen per-pixel median
   background is subtracted and a wax-colour mask applied, so only moving wax
   remains.
2. **Find the blobs topologically.** Instead of thresholding by size, the mask is
   handed to GUDHI's `CubicalComplex` for 0-dimensional persistent homology (H0).
   A blob is "real" if it is prominent relative to its surroundings across many
   intensity thresholds, which is a better notion of significance than area.
3. **Keep each blob's identity.** Dense optical flow carries every blob's mask
   forward frame to frame; a physics-informed Kalman filter that knows blobs
   decelerate at the top and bottom of the lamp predicts where it is going; a
   topological fingerprint lets a blob that vanished for a few seconds be
   recognised when it returns. (`blob_tracker.py`, `tracker_stability.py`)
4. **Give each blob a voice.** A 16-slot voice pool binds blobs to sustained
   synth voices. A routing matrix maps blob features (height, speed, area,
   elongation, orientation, persistence…) onto synth parameters. A keyboard
   player can also own slots, so the lamp and the player share one instrument.
   (`voice_manager.py`, `binding_manager.py`, `routing.py`, `controller.py`)
5. **Read the lamp as a score.** The governor layer watches all blobs at once and
   emits a `ScoreState`: phrase boundaries, a tempo and meter, a harmonic field,
   a tension value, and a prominence ranking of the blobs. Nothing is elected;
   every blob both plays and governs, weighted by a softmax, so leadership drifts
   rather than switching. (`score_governor.py` and friends)
6. **Articulate.** On top of the sustained drones, three generators fire short
   notes: a per-blob **melody** line driven by motion, a per-blob **rhythmic
   pattern** layer on the governor's elastic clock, and a **paired octave-down
   voice** for the most prominent blobs. (`melody_rhythm.py`, `pattern_language.py`)
7. **Make it sound.** SuperCollider renders everything: wavetable drones through
   a morphing filter, physically modelled plucks, hammers and bows for the hits,
   an orchestral swell on blob birth, and a master reverb "space" whose size and
   haze follow the score state. Optionally the whole mix is re-synthesised
   through pre-trained RAVE neural audio models. (`*.scd`, `synth_palette.py`,
   `rave_layer.py`)

Everything from step 5 onward is **additive**: each layer reads tracker and voice
state and never modifies the layer below it. Every layer can be switched off from
the command line, and every Python module has a test suite that runs without a
camera, SuperCollider or pygame.

---

## The layers in more detail

### Detection and tracking (v2–v4.13)

| What | Why |
|---|---|
| **H0 persistence** on the downscaled mask (GUDHI `CubicalComplex`) | A tiny bright blob and a large dim one are both significant; area thresholds get one of them wrong. Persistence measures prominence, not size. |
| **Frozen median background** (`background_reference.py`, `b`/`w` keys to re-capture) | The adaptive MOG2 subtractor absorbed slow wax (0.5–2 px/frame) into the background within a second and the blob vanished. A frozen reference does not adapt. |
| **Flow-advected mask identity** | Identity is the mask, solved once at birth and carried by optical flow. A blob that goes briefly transparent is still "there" because its mask is. Per-frame nearest-centroid matching caused birth/death churn. |
| **Wasserstein-inspired matching** in the (birth, death) persistence plane | When two advected masks overlap one detection, topology disambiguates better than pixel distance. |
| **5-state buoyancy Kalman** `[x, y, vx, vy, ay]` | Blobs do not move at constant velocity. They slow to zero at the turning points, which is exactly where identity is most ambiguous. |
| **Topological fingerprint re-identification** | A blob lost for up to 15 s comes back with its old id, so its voice continues. |
| **Tracker stability** (`tracker_stability.py`, v4.13) | A stationary blob whose detected region flickered — growing, shrinking to a core, splitting into lobes — used to spawn a stream of new voices, each with a birth strike. Six fixes (fuse fragments, complete partial views, re-associate new regions with coasting tracks, confirm births over 3 cycles, damp coasting, stationary mode) make it one blob, one voice. |

### The voice pool and the player (v4.0)

The voice, not the blob, is the first-class object. Sixteen slots, each with a
lifecycle state:

- `BLOB_ONLY` — driven by the routing matrix from blob features
- `KEYBOARD_ONLY` — the player owns it
- `HYBRID` — the player sets pitch and timbre; the blob keeps modulating cutoff, resonance, pan and level
- `ZOMBIE` — the blob vanished; the voice holds at soft sustain for 4 s waiting to be resurrected by a nearby blob
- `IDLE`

Each slot carries a list of `Routing(source → destination, depth, curve)` tuples.
Default routings put height on pitch, persistence on level, elongation on
resonance, orientation on pan, speed and area on wavetable position, and so on.
Governor-derived sources (`TENSION`, `PROMINENCE_RANK`, `PHRASE_POSITION`,
`BEAT_PHASE`, `DEPTH`…) use the same matrix, so the player can rewire score
governance live.

**Voice continuity** (`voice_continuity.py`, v4.12) closes a hole: a blob that was
re-identified, or born while the pool was full, had no voice bound to it and sat
silent for the rest of its life. Now every tracked blob older than 8 frames gets
a voice (resurrected from a zombie if one matches, else a new one without the
birth strike), live voices never drop below 40 % of their base level under
governance, and the beat dip is proportional instead of a fixed subtraction.

### The score grammar (v4.2–v4.9)

This is the part that turns six independent voices into an ensemble. The design
borrows the *categories* of Lerdahl and Jackendoff's Generative Theory of Tonal
Music (GTTM) — grouping, meter, time-span reduction, prolongation — and treats
them as a control decomposition for generation, with Temperley's preference-rule
machinery and Hasty's projective meter underneath. The lamp's topology supplies
the evidence:

- **Grouping — where phrases end.** A boundary is an *event*, modelled as
  `P(boundary) = h(τ) · Φ(z)`: a hazard `h` from an online Gamma fit to observed
  phrase lengths (the piece learns its own phrase-length distribution) times the
  probability `Φ(z)` that the evidence — merge-tree saddle events, Bayesian
  change-points on kinetic energy and occupancy entropy, a persistence-gated
  blob-count change, lifecycle events of blobs older than 3 s — is out of the
  ordinary. The product is a soft AND: a boundary needs both "we are due" and
  "something happened". Commitment is a draw with probability `P`, so boundary
  strength is graded for free. (`hazard_grouping.py`)
- **Meter — the pulse.** The lamp's 90–180 s cycles are folded by octave division
  into a tactus range of 0.4–2 s. A discrete beam over (period, beats-per-bar)
  hypotheses and a particle filter over continuous period/phase feed an
  `ElasticClock`. The Kuramoto order parameter of the blobs' phases sets how
  strongly events are pulled toward the grid, so coherent lamps play in time and
  incoherent ones drift. (`elastic_clock.py`, `system_state.py`)
- **Time-span — who leads.** Prominence is integrated persistence, merge-tree
  depth, coupling to the aggregate flux, recurrence, kinetic energy and whether
  the voice sits on a strong beat and a chord tone. The top third by prominence
  keep the sustained voice and gain an articulated octave-down partner; the rest
  carry the busy ornamental layer. The Schenkerian head/ornament reading is made
  audible as instrumentation.
- **Prolongation — tension.** Surprisal of the system's macrostate under a learned
  Markov model, Wasserstein distance from the equilibrium persistence diagram,
  kinetic energy and conditional entropy, smoothed into one `tension` value that
  the synth palette, the pattern transforms and the harmony walker all read.
- **Harmony — the field.** A `TonnetzWalker` moves between triads by neo-Riemannian
  P/L/R operations. A step is licensed only when a topological change budget
  (accumulated diagram distance) is spent; direction away from or back toward the
  home triad follows the equilibrium gradient; compound moves need high tension.
  The predicted next triad is quietly pre-sounded at a gain equal to the
  posterior mass on non-current candidates. (`harmony_tonnetz.py`)
- **Schemata.** A small Gjerdingen-style lexicon (plateau, prinner, cadence…) gives
  the ornamental hits a melodic contour, chosen by a Markov chain biased by tension
  and entropy at each boundary. (`schema_lexicon.py`)

Governance is a set of continuous **softmax role weights**: for each role, every
blob and a few system-level virtual signals get a score (periodicity of vertical
motion for meter, slowness for harmony, erraticness for prolongation, nearness to
the merge-tree root for grouping), and the softmax with temperature 0.6 turns
them into weights that sum to one. There is no election, no handover code and no
crossfade timer. When blob A stops oscillating cleanly and blob B starts, their
weights cross over by themselves.

### Melody and rhythm (v4.9, `melody_rhythm.py`)

The drones are continuous; this layer gives each blob a *line*.

- **Hazard-driven onsets.** Each blob owns an accumulator `A += h·dt`, with
  `h = h0 (1 + a·speed + b·accel)` capped at 1 note/s. A still blob plays a note
  every ~4 s; a blob mid-rise or at a turning point plays a cluster. Because the
  accumulator integrates, a burst of motion yields a run of notes rather than a
  single trigger, and a slow blob still eventually speaks.
- **Height-band pitch with hysteresis and contour inertia.** The lamp is divided
  into ten horizontal bands (five scale degrees × two octaves). A blob's band only
  changes when its centroid leaves the band by a quarter of a band height, so
  hovering at a boundary never flickers the pitch. The band is an attractor, not
  the note: the sounding note walks toward it in steps whose size depends on
  speed (slow → stepwise, fast → thirds and leaps), in the direction of travel
  once that direction has been held for 0.8 s. A reversal is a turning point and
  forces resolution onto the band.
- **Scale from the field.** The five-note scale is derived from the governor's
  current triad plus the two diatonic neighbours the current schema uses most,
  latched on a beat boundary. With the governor off it is C pentatonic.
- **Onset pull.** A note that fires is delayed toward the nearest beat line by the
  same Kuramoto elastic strength the pattern layer uses, so a locked lamp plays on
  the grid and a free one does not.
- Durations are whole beats of the elastic clock chosen by blob area; expanding
  blobs play legato, shrinking ones staccato. Each note copies its drone's timbre.

Pattern layer (`pattern_language.py`) is the cyclic complement: each blob owns a
Euclidean rhythm `E(k, n)` whose slot count comes from its own rise/fall period in
beats and whose hit count from its area, transformed Tidal-style (degrade, swing
from elongation, arpeggiate the pitch field, ply at boundaries).

### The synth palette (v4.10, `synth_palette.py`, `blob_synth_rich.scd`)

Every classic SynthDef has a richer drop-in sibling that takes the same control
names plus a few extras, so no OSC call had to change:

- `\mpeVoiceRich` — wavetable drone through a morphing LP/BP/HP filter, with a
  vowel formant set by height, sub-octave weight by area, and chorus width by
  sideways drift.
- `\mpeHitRich` — ornamental hits are a **plucked string**, the paired head voice
  a **piano hammer**.
- `\melodyHitRich` — **bowed** when the blob is expanding, **plucked** when it is
  shrinking; bow pressure from speed.
- `\blobStrikeRich` — an orchestral swell on every birth.
- `\lavaMaster` — a master reverb/space at the tail of the graph driven by the
  score state: ambiguity → reverb time, tension → bright tail and drive, phrase
  position → wet mix (the phrase dissolves toward its boundary), blob count →
  room size, Kuramoto order → modulation depth, each boundary → a bloom.
- `\preSoundRich` — the predicted next triad, quietly.

Toggle rich/classic live with `p`; the ensemble migrates voice by voice.

### RAVE timbre transfer (v4.11–v4.14, `rave_layer.py`, `blob_synth_rave.scd`)

A final, optional stage runs the finished lava mix through one or more
pre-trained **RAVE** models (Realtime Audio Variational autoEncoder, Caillon and
Esling, IRCAM) inside SuperCollider via the `nn.ar` extension. Off by default.

- A `\raveSplit` node copies bus 0 to a private bus and scales the dry signal by
  `1 − mix`. One `\raveXfer_<key>` node per *active* model reads that copy and adds
  its wet output back, panned to a fixed position in the bank, at `mix / √n`.
  The master space hears dry and wet together.
- **Parallel, not serial.** "Birds + organ" is the lamp heard through two ears at
  once, not birds-of-an-organ.
- Nineteen models, listed best-first for the lamp: pitched instrumental models
  (SOL, MusicNet, organ, MRP piano) keep the harmony; voice models follow it
  loosely; texture models (birds, water, percussion) mostly map loudness.
- Latency is ~0.7 s for non-causal models. On 90–180 s blob cycles it reads as a
  halo, so the dry path is deliberately not delayed to align.
- The RAVE tab in the keyboard window lists each model with its loaded/missing
  status and sample rate; `rave_status.json` is written by the `.scd` and polled
  by Python.

---

## Repository layout

| Area | Files |
|---|---|
| Main loop | `audible_lava.py`, `lamp_profile.py` / `lamp_profile.json`, `calibration_gui.py` |
| Detection | `blob_detection.py`, `h0_persistence.py`, `merge_tree.py`, `background_reference.py` |
| Tracking | `blob_tracker.py`, `tracker_stability.py` |
| Voices | `voice_manager.py`, `binding_manager.py`, `voice_continuity.py`, `routing.py`, `controller.py` |
| Score grammar | `score_governor.py`, `hazard_grouping.py`, `system_state.py`, `preference_rules.py`, `elastic_clock.py`, `harmony_tonnetz.py`, `schema_lexicon.py`, `pattern_language.py`, `governor_weights_stable.json` |
| Articulation | `melody_rhythm.py` |
| Synthesis | `synth_palette.py`, `rave_layer.py` |
| SuperCollider | `dso_main.scd` (one-shot loader), `blob_synth.scd`, `blob_synth_hits.scd`, `melody_hit.scd`, `blob_synth_rich.scd`, `blob_synth_rave.scd`, `rave_bench.scd` |
| Tests | `test_*.py` (no camera, SuperCollider or pygame needed) |
| Diagnostics | `diagnose_tracking.py`, `diagnose_score_grammar.py`, `rave_bench.py`, `mpe_practice.py` |
| Docs | `build_companion_*.py` (ReportLab generators for the companion PDFs) |

---

## Setup

### Python

```bash
conda create -n dso python=3.11
conda activate dso
pip install -r requirements.txt      # opencv-python, numpy, scipy, gudhi, python-osc, pygame
```

Python 3.11 is pinned for GUDHI and current OpenCV wheels on macOS.

### SuperCollider

SuperCollider 3.13+. `sc3-plugins` is optional (stock fallbacks are built in).
For RAVE, install [nn.ar](https://github.com/elgiano/nn.ar) into your Extensions
folder and put the `.ts` model files in the folder named by `AUDIBLE_LAVA_RAVE_DIR`
(default `~/Documents/RAVE_PreTrained_Models`).

Open `dso_main.scd`, put the cursor inside the outer parentheses and evaluate it.
It boots the server at 48 kHz, loads the five `.scd` files in order, waits for each
to be truly ready, and posts `==== dso_main.scd: ready ====`. Set `~dsoRave = false`
first to skip RAVE.

Manual load order if you prefer: `blob_synth.scd` → `blob_synth_hits.scd` →
`melody_hit.scd` → `blob_synth_rich.scd` → `blob_synth_rave.scd`.

### Run

```bash
python audible_lava.py               # first run opens the calibration GUI
```

Flags: `--calibrate`, `--no-gui`, `--no-keyboard`, `--no-governor`, `--no-patterns`,
`--no-melody`, `--classic-synths`, `--no-space`, `--no-presound`, `--no-rave`,
`--rave-on`, `--no-continuity`, `--no-stable-tracker`.

If SuperCollider is rebooted or re-evaluated for any reason, restart Python: its
nodes died with the old server.

### Verify without hardware

```bash
for t in test_*.py; do python "$t"; done
```

Each suite prints `Results: N passed, 0 failed`.

---

## Controls

Two windows run at once. The **OpenCV video window** takes toggle keys; the
**pygame keyboard window** must have focus for music keys. `TAB` cycles
PERFORMANCE → EDIT → RAVE in the keyboard window.

**Video window**

| Key | Action |
|---|---|
| `q` | quit |
| `v` `t` `o` `m` `f` `c` | HUD · tracking · contours · merge tree · optical flow · masks |
| `n` | melody layer on/off |
| `p` | rich / classic synth palette |
| `x` / `k` | bypass / respawn the master space |
| `a` | RAVE layer on/off |
| `b` / `w` | re-capture background (cold / warm, lamp already running) |
| `[` / `]` | persistence threshold −/+ |

**Keyboard window — PERFORMANCE**

| Key | Action |
|---|---|
| `1`–`0` | select slot 0–9 |
| `z s x d c v g b h n j m , l .` | lower octave, chromatic |
| `q 2 w 3 e r 5 t 6 y 7 u i o p` | upper octave, chromatic |
| `[` / `]` | octave down / up |
| `space` | all notes off |

**Keyboard window — EDIT**

| Key | Action |
|---|---|
| `P W E F M G` | parameter family: pitch · wave · envelope · filter · mod · governor |
| `← →` / `↑ ↓` / `+ −` | cycle · fine · coarse |
| `B` `N` `T` `D` | claim hybrid · create empty · tie to next blob · detach |
| `R` / `C` | reset parameter (in `G`, resets phrase learning) · clear routings |

The governor family exposes rule weights, phrase prior, temperature and smoothing
constants live; edits are mirrored to `governor_runtime.json`.

**Keyboard window — RAVE**

| Key | Action |
|---|---|
| home row, then `q w e t y u i` | toggle model 0–18 |
| `space` | layer on/off |
| `↑ ↓` / `+ −` | wet mix ±0.05 / ±0.25 |
| `← →` | input gain ÷ / × 1.4 |
| `z` / `r` | all models off / re-read status |

---

## Things that bit, so you don't have to

- **Wire it, then listen.** A routing or layer can be fully implemented and
  produce no sound because nothing calls it. Every new layer ships with a test
  that proves a message reaches OSC.
- `freq_offset_hz` must be summed inside `_compute_targets`; sending `/n_set freq`
  directly does not persist because the smoother overwrites `freq` every frame.
- Multiple `PITCH` routings accumulate semitones. Last-one-wins was a latent bug.
- `y_norm` is 1 at the top of the lamp. One extractor inverted it again (v4.10.1).
- Farneback flow is ~49 % of frame time and GUDHI ~21 %. Halve the flow
  resolution or raise `BlobDetector(downscale=)` before anything else.
- Disable SuperCollider's input (`numInputBusChannels = 0`); Python owns the
  sensors, and a sample-rate mismatch on an unused input will crash the server.
- A frozen background captured on a running lamp bakes the blobs in. Use `w`.
- `nn.ar` on Intel macOS needs `libiomp5.dylib` in its `ignore/` folder; models
  trained at 44.1 kHz run 8.8 % off-rate on a 48 kHz server (flagged with `!`);
  choppy RAVE output is inference underrun — raise `~raveBufferSize`.

---

## Theoretical lineage

Di Scipio's *Audible Ecosystemics* (the name is a nod), Xenakis's stochastic
macrostates, Lerdahl and Jackendoff's GTTM, Temperley's preference rules, Hasty's
*Meter as Rhythm* and London's tactus prior, Cohn and Lewin's Tonnetz, Gjerdingen's
schemata, Toussaint's Euclidean rhythms, Huron and Meyer on expectation. GTTM is a
theory of listening, not generation; here its structural categories are
repurposed as a generative control decomposition, and the lamp's persistent
homology supplies the evidence those categories need.

## Novelty

To the author's knowledge this is the first system to combine a physical lava
lamp as live input; H0 persistent homology as a blob-significance filter;
flow-advected mask identity with topological matching; a physics-informed
buoyancy Kalman; topological re-identification; a zombie voice pool; a
bidirectional blob ↔ player voice-slot architecture; a GTTM-shaped score grammar
driven by merge-tree and persistence-diagram evidence with softmax role
governance; and RAVE timbre transfer as a final stage.

## License and citation

Code and documentation © Samuel D. Leventhal. Contact via samleventhal.com.
