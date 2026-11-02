# h0_persistence.py -- Audible Lava v4.2
# Shared, validated H0 (superlevel-set) persistence extraction.
#
# Both blob_detection.py and merge_tree.py previously called GUDHI directly
# and shared three defects (found by test_merge_tree.py, Aug 2026):
#
#   1. DIMENSION ORDER. `CubicalComplex(top_dimensional_cells=arr.flatten(),
#      dimensions=[sh, sw])` is wrong for a row-major (h, w) array: GUDHI
#      takes the FASTEST-varying dimension first, i.e. [sw, sh]. With the
#      old order the field was read at the wrong stride whenever h != w,
#      producing dozens of spurious "persistent" pairs from a single smooth
#      peak and scrambling every persistence value.
#
#   2. PEAK LOCALIZATION. `np.argwhere(abs(inverted - birth) < 1)[0]` picks
#      the first pixel anywhere in the image at that intensity. Two peaks of
#      equal height collapse onto one pixel; a peak's leaf can be placed on
#      another blob's flank. GUDHI reports the exact birth cell via
#      `cofaces_of_persistence_pairs()`; we use it.
#
#   3. ESSENTIAL CLASS. The component born at the global maximum never dies
#      (death = inf) and was skipped -- so the BRIGHTEST blob was never a
#      candidate. In the scrambled regime it was seeded by accident through
#      spurious pairs; once (1) is fixed it would vanish. It is now returned
#      with death = field minimum (the standard extended-persistence
#      convention: it lives across the full dynamic range of the frame).
#
# Everything downstream (branch dedup, watershed seeding, merge-tree leaves)
# consumes H0Peak objects from here, so the two pipelines can no longer drift.

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import gudhi


@dataclass
class H0Peak:
    """One superlevel-set H0 feature of a 2-D intensity field.

    birth       -- intensity at which the peak appears (its maximum value)
    death       -- intensity at which it merges into an older component
                   (for the essential class: the field minimum)
    persistence -- birth - death
    peak_yx     -- (row, col) of the birth cell IN THE FIELD'S OWN GRID
    saddle_yx   -- (row, col) of the death cell: the saddle at which this
                   component merged into an older one (None for essential)
    essential   -- True for the never-dying class (global maximum)
    """
    birth:       float
    death:       float
    persistence: float
    peak_yx:     tuple
    saddle_yx:   Optional[tuple] = None
    essential:   bool = False


def _decode_index(idx: int, width: int) -> tuple:
    """Row-major flat index -> (y, x) for the [sw, sh] cubical layout."""
    y, x = divmod(int(idx), width)
    return (int(y), int(x))


def _fallback_localize(inverted: np.ndarray, birth_inv: float,
                       used: set) -> Optional[tuple]:
    """Localize a birth cell without cofaces: the cell must have value
    == birth_inv AND be a local minimum of `inverted` (4-neighbourhood),
    and must not have been assigned to another pair already."""
    cand = np.argwhere(np.abs(inverted - birth_inv) < 0.5)
    h, w = inverted.shape
    for y, x in cand:
        y, x = int(y), int(x)
        if (y, x) in used:
            continue
        v = inverted[y, x]
        ok = True
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and inverted[ny, nx] < v:
                ok = False
                break
        if ok:
            return (y, x)
    return None


def h0_peaks(
    field:             np.ndarray,
    min_persistence:   float = 0.0,
    include_essential: bool  = True,
    essential_death:   Optional[float] = None,
) -> List[H0Peak]:
    """
    Superlevel-set H0 persistence of a 2-D intensity field.

    Bright regions are features (peaks); the field is inverted internally so
    GUDHI's sublevel filtration sees peaks as minima. All returned values are
    in the ORIGINAL (non-inverted) intensity scale.

    Args:
        field:             2-D array (uint8 or float), brighter = feature.
        min_persistence:   drop finite pairs below this persistence. The
                           essential class is never dropped when included.
        include_essential: return the never-dying global-maximum component.
        essential_death:   death value to assign to the essential class.
                           Defaults to field.min() (extended persistence).

    Returns:
        list of H0Peak sorted by persistence descending.
    """
    f = np.asarray(field, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError(f"h0_peaks expects a 2-D field, got shape {f.shape}")
    sh, sw = f.shape
    inverted = 255.0 - f

    # GUDHI: fastest-varying dimension FIRST. For a row-major (sh, sw) array
    # that is width, then height.
    cc = gudhi.CubicalComplex(
        top_dimensional_cells=inverted.flatten(),
        dimensions=[sw, sh],
    )
    # Filtering inside GUDHI keeps the coface list aligned with `pairs` and
    # avoids iterating every pixel-level pair in Python. GUDHI drops pairs
    # with persistence <= min_persistence, so subtract a hair to keep the
    # "equal to threshold" case consistent with a `pers < thr` test below.
    gudhi_min = max(0.0, float(min_persistence) - 1e-6)
    cc.persistence(homology_coeff_field=2, min_persistence=gudhi_min)
    pairs = cc.persistence_intervals_in_dimension(0)

    # Exact birth cells. regular[0] rows are (birth_cell, death_cell) for
    # dim-0 pairs in the same order as `pairs`; essential[0] holds the
    # birth cell of the essential class.
    reg_cells = ess_cells = None
    try:
        regular, essential = cc.cofaces_of_persistence_pairs()
        reg_cells = regular[0] if len(regular) > 0 else np.zeros((0, 2), dtype=np.int64)
        ess_cells = essential[0] if len(essential) > 0 else np.zeros((0,), dtype=np.int64)
    except Exception:            # very old GUDHI: fall back to value matching
        reg_cells = ess_cells = None

    peaks: List[H0Peak] = []
    used: set = set()

    finite_idx = 0
    for birth_inv, death_inv in pairs:
        if death_inv == float("inf"):
            continue
        birth_real = 255.0 - float(birth_inv)
        death_real = 255.0 - float(death_inv)
        pers = birth_real - death_real

        saddle = None
        yx = None
        if reg_cells is not None and finite_idx < len(reg_cells):
            b_cell, d_cell = reg_cells[finite_idx]
            finite_idx += 1
            cand = _decode_index(b_cell, sw)
            # Defensive: the decoded birth cell must carry the birth value.
            if abs(inverted[cand] - float(birth_inv)) < 0.5:
                yx = cand
                saddle = _decode_index(d_cell, sw)
        if yx is None:
            yx = _fallback_localize(inverted, float(birth_inv), used)
            if yx is None:
                continue
        used.add(yx)

        if pers < min_persistence:
            continue
        peaks.append(H0Peak(birth_real, death_real, pers, yx,
                            saddle_yx=saddle, essential=False))

    if include_essential:
        ess_death = float(f.min()) if essential_death is None else float(essential_death)
        if ess_cells is not None and len(ess_cells) > 0:
            for cell in ess_cells:
                yx = _decode_index(cell, sw)
                birth_real = float(f[yx])
                peaks.append(H0Peak(birth_real, ess_death,
                                    birth_real - ess_death, yx, essential=True))
        else:
            # Fallback: essential class is born at the global maximum.
            yx = tuple(int(v) for v in np.unravel_index(int(np.argmax(f)), f.shape))
            birth_real = float(f[yx])
            peaks.append(H0Peak(birth_real, ess_death,
                                birth_real - ess_death, yx, essential=True))

    peaks.sort(key=lambda p: p.persistence, reverse=True)
    return peaks
