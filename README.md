# Dynamical Systems Orchestration — v4.0 (MPE Edition)

This bundle is **Phase 1 of v4.0**: voice-slot pool + computer-keyboard MPE
controller + bidirectional blob ↔ voice binding. The pivot from v3.x is
described in detail in `DSO_Complete_Guide_v4.0_MPE.pdf`; this file is the
short upgrade path.

## What's in the bundle

| File | Status | Purpose |
|---|---|---|
| `routing.py` | **NEW** | `Routing` tuple, `Source`/`Dest`/`Curve` enums, default blob-driven routings, source-value extraction |
| `controller.py` | **NEW** | `Controller` protocol, `ControllerEvent`, `KeyboardController` (pygame), `MIDIController` stub for Phase 2 |
| `binding_manager.py` | **NEW** | Lifecycle bridge: tracker events ↔ slot allocation policy |
| `voice_manager.py` | **REWRITTEN** | 16-slot pool, `VoiceState` enum, `SynthState` dataclass, routing application, zombie pool, merge/split (preserved from v3.x) |
| `blob_synth.scd` | **EXTENDED** | `\mpeVoice` SynthDef added. v3.x `\blobVoice` / `\blobAngle` / `\blobStrike` preserved |
| `audible_lava.py` | **REWRITTEN** | New main loop — controller poll, binding-manager dispatch, voice overlay, slot HUD. `MAX_COAST=20`, MOG2 frozen post-warmup (v3.2) |
| `test_voices.py` | **NEW** | Standalone test suite for the slot system (no SC, no camera, no pygame required) |
| `requirements.txt` | **UPDATED** | `pygame` added; `mido` + `python-rtmidi` reserved for Phase 2 |

Files **not in this bundle** but expected to exist alongside it (unchanged
from your v3.x install):

- `lamp_profile.py`
- `calibration_gui.py`
- `blob_detection.py`
- `blob_tracker.py`  *(Phase 1 also expects `max_coast=20`; constructor
  call in `audible_lava.py` already passes this)*
- `merge_tree.py`
- `test_setup.py`, `test_detection.py`, `test_tracker.py`

## Installing into an existing v3.x project

```bash
cd ~/lava                       # your existing project root
cp /path/to/v4_bundle/*.py .    # routing/controller/binding/voice_manager/audible_lava
cp /path/to/v4_bundle/blob_synth.scd .
cp /path/to/v4_bundle/test_voices.py .
cp /path/to/v4_bundle/requirements.txt .

# pip install pygame (or full requirements):
pip install -r requirements.txt

# Boot SuperCollider, run blob_synth.scd to load \mpeVoice + the wavetable banks.
# Then:
python audible_lava.py
```

The previous `voice_manager.py` is fully replaced. The previous `audible_lava.py`
is fully replaced. SuperCollider continues to work — `\blobVoice`,
`\blobAngle`, and `\blobStrike` are still defined for back-compat, but the
new `VoiceManager` only spawns `\mpeVoice` (plus the auxiliary `\blobAngle`
pad on blob-driven voices).

## Verifying the bundle (no SC, no camera)

```bash
python test_voices.py
```

Expected output: `Results: 35 passed, 0 failed`.

The test suite exercises:

1. Basic spawn from blob births (3 → BLOB_ONLY)
2. CREATE_EMPTY + TIE_BLOB workflow (queues a slot, ties on next birth)
3. Immediate-tie when an unbound blob is already present
4. Zombie pool (BLOB_ONLY → ZOMBIE → resurrect; expire after window → IDLE)
5. CLAIM_HYBRID + SET_PARAM relative edits
6. Routing matrix produces non-trivial target values (top blob → higher freq)
7. ControllerEvent dispatch through VoiceManager (SELECT_SLOT, NOTE_ON, SET_PARAM, NOTE_OFF)
8. DETACH transitions HYBRID → KEYBOARD_ONLY

## Live keyboard cheat-sheet

The pygame window must have keyboard focus. The OpenCV window only handles
`q m t f c o v [ ]`.

**PERFORMANCE mode** (default; `TAB` toggles to EDIT):

| Key | Action |
|---|---|
| `1`–`9`, `0` | Select slot 0–9 |
| `z s x d c v g b h n j m , l .` | Lower-octave notes (chromatic) |
| `q 2 w 3 e r 5 t 6 y 7 u i o p` | Upper-octave notes (chromatic) |
| `[` / `]` | Octave down / up |
| `space` | All-notes-off on selected slot |
| `TAB` | Switch to EDIT mode |

**EDIT mode**:

| Key | Action |
|---|---|
| `1`–`0` | Select slot |
| `P W E F M` | Parameter family: pitch / wave / env / filter / mod |
| `← / →` | Cycle parameter within family |
| `↑ / ↓` | Fine adjust focused parameter |
| `+ / −` | Coarse adjust (10× step) |
| `B` | CLAIM_HYBRID — promote BLOB_ONLY → HYBRID |
| `N` | CREATE_EMPTY — new KEYBOARD_ONLY voice in next IDLE slot |
| `T` | TIE_BLOB — bind selected KEYBOARD_ONLY to nearest unbound blob |
| `D` | DETACH — break blob bind on selected slot |
| `R` | Reset focused parameter to default |
| `C` | Clear all routings on selected slot |
| `TAB` | Back to PERFORMANCE |

## Where Phase 2 plugs in

The `MIDIController` class in `controller.py` is a no-op stub today. Phase 2
fills in:

1. `mido.open_input(port)` in `__init__` (lazy)
2. `iter_pending()` in `poll()` to read the port
3. Learn-map (channel, cc) → (slot, param) consulted on every CC message
4. `LEARN_ARM` on the keyboard side calls `voices.arm_learn(slot, param)`;
   the next CC the controller adapter sees binds itself to that target

No other module changes. All downstream code already speaks `ControllerEvent`.

## Known caveats / Phase 1.5 follow-ups

- Edit-mode `M` family ("mod") parks the cursor for routing-list walking but
  doesn't yet emit `ADD_ROUTING` from arrow keys. The
  `VoiceManager.add_routing()` API works programmatically; the keyboard-side
  affordance is a Phase 1.5 polish task.
- `BlobDetector.detect()` is called with `learning_rate=` if the installed
  detector accepts it (v3.2 patch); a `TypeError` fallback covers older
  signatures so the v4.0 main loop runs against both v3.0 and v3.2
  detection layers.
- The `\mpeVoice` SynthDef expects 16 contiguous wavetable buffers. When
  `blob_synth.scd` has finished loading them, `~wtBuffers[0].bufnum` is the
  base index; `audible_lava.py` does not currently push `bufBase` to SC,
  so the SynthDef hardcodes a base of 0. If `Buffer.allocConsecutive`
  returns a different base id on your machine, set `bufBase` via
  `/n_set` after spawning each voice — or hardcode the lowest id you
  observe in the post window.
