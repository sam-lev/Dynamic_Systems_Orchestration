# Wiring melody_rhythm.py into audible_lava.py

Five insertions, nothing removed. Line anchors refer to the v4.1 `audible_lava.py`
in the Project; in v4.5 find the same anchors by their text.

## 1. Import (after the `from controller import (...)` block)

```python
# melody / rhythm layer (additive; see melody_rhythm.py)
from melody_rhythm import MelodyEngine
```

## 2. Config (next to `N_VOICE_SLOTS`)

```python
MELODY_ENABLED       = True       # 'n' key toggles at runtime
```

## 3. Construct (after `binding = BindingManager(...)`)

```python
    melody   = MelodyEngine(voices, profile, enabled=MELODY_ENABLED)
```

## 4. Per-frame call (immediately after `voices.process_frame(tracked)` in step 6)

```python
                # ── 6b. Melody / rhythm layer: hazard onsets + band pitch ──
                for ev in melody.process_frame(tracked):
                    print(f"  . note blob #{ev.blob_id} S{ev.slot_id:02d} "
                          f"m{ev.midi} {ev.freq:.0f}Hz dur={ev.dur:.2f}")
```

Drop the `print` once you trust it; `melody.last_events` is available for the
overlay either way.

## 5. Key toggle (in the `cv2.waitKey` chain, next to `elif key == ord("v")`)

```python
                elif key == ord("n"):
                    print(f"[Melody] {'on' if melody.toggle() else 'off'}")
```

`n` is unused in the OpenCV window key list (q m t f c o v b [ ]). The pygame
keyboard window has its own key map and is unaffected.

## Optional: overlay line per blob

In `draw_voice_overlay`, after the `info = (...)` / `cv2.putText(overlay, info, ...)`
pair, add a third line. This needs `melody` passed in, so either add a parameter
or read it from a module global; the least invasive is a module-level
`MELODY_ENGINE = None` set after construction:

```python
        mi = MELODY_ENGINE.blob_info(b.id) if MELODY_ENGINE is not None else ""
        if mi:
            cv2.putText(overlay, mi, (cx + 10, cy + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, st_col, 1)
```

This shows per blob: hazard rate h (notes/s), accumulator A, current band,
sounding degree, contour arrow, and MIDI note — enough to see whether the
rhythm is actually driven by motion (h should rise visibly when a blob moves
or turns) and whether pitch is flickering at band boundaries (it should not).

## SuperCollider

Evaluate `melody_hit.scd` after `blob_synth.scd` (it uses `~wtBuffers`).
`blob_synth.scd` is unchanged. Confirm with the manual test block in
`melody_hit.scd` before running Python; if notes sound like plain saws
regardless of bank, pass `bufBase` the same way `mpe_practice.py` does.

## Verify

```bash
python test_melody_rhythm.py        # 50 checks, no SC / camera / pygame
python audible_lava.py              # watch for ". note blob #…" lines
```

Audible checks, in order:
1. A still, bound blob plays roughly one note every 10-15 s.
2. A blob mid-rise plays noticeably more often (several notes per 10 s).
3. Notes cluster around a reversal (ay spikes) and the pitch snaps to the
   blob's band at the turn.
4. A blob hovering at a band boundary does not alternate between two pitches.
5. Press `n`: notes stop, drones continue; press again: notes resume.
