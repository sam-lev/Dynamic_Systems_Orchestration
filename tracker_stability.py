# tracker_stability.py -- Audible Lava / DSO v4.13 "one blob, one voice"
#
# Additive. StableBlobTracker subclasses BlobTracker and wraps its update();
# blob_tracker.py is not modified. audible_lava.py constructs it instead
# (--no-stable-tracker restores the plain v4.12 BlobTracker).
#
# THE SYMPTOM: a stationary blob whose detected persistence region fluctuates
# (grows, shrinks, splits into lobes, drops out for a cycle) produces a stream
# of new voices that glide up or down and away, instead of one voice that
# follows the blob.
#
# THE CAUSES (all in BlobTracker.update):
#   1. Shape change = new blob. Primary association is IoU (intersection over
#      union) >= 0.25 against the flow-advected mask. A region that changes
#      shape in place fails it; the detection is unmatched and, unless the
#      re-identification buffer recognises it, _spawn() emits `birth`.
#   2. Re-identification only covers DEAD tracks, but the original track is
#      not dead: it coasts for max_coast detection cycles, so nothing compares
#      the "new" detection with the track it belongs to.
#   3. Coasting extrapolates stale motion ("flying away"). kf.predict() runs
#      every frame (y += vy + 0.5 ay, vy += ay). Centroid jumps from shape
#      changes read as velocity, and near the top/bottom the measurement
#      noise is raised to 2.0 so the buoyancy model is trusted. When the
#      detection is lost the predicted centroid keeps going for max_coast x K
#      frames; the voice's pitch follows y_norm, so it glides away.
#   4. Fragments of a matched blob become births. Branch dedup can keep a
#      second peak inside one wax mass for a few cycles; watershed then cuts
#      the blob in two. The track matches one piece, the other is unmatched,
#      and split detection only fires when the parent is coasting.
#
# THE FIXES (each can be switched off for A/B comparison):
#   F  fuse_fragments    Before association: detections that are adjacent to
#                        each other and lie mostly inside the SAME existing
#                        track's mask are fused into one detection (one blob
#                        cut by the watershed), so the track also measures the
#                        whole blob and its centroid does not jump.
#   R  reassociate       After association: every newly born track is offered
#                        to the tracks that missed this cycle (coasting),
#                        compared with their LAST MEASURED mask and centroid
#                        (not the extrapolated ones), using the overlap
#                        coefficient |A n B| / min(|A|, |B|), which tolerates
#                        shape change. Hungarian (one-to-one) within a gate. A
#                        match folds the detection back into the old id and
#                        drops the birth: same voice, no strike.
#   T  confirm_cycles    A remaining new track is TENTATIVE: hidden from the
#                        returned list (no voice, no rebind) and its `birth`
#                        withheld until it has been detected on confirm_cycles
#                        consecutive cycles. A tentative track that misses once
#                        is dropped silently. One that was born inside a live
#                        track's mask confirms as ('split', parent, child).
#   D  damp_coasting     While a track coasts, its velocity and buoyancy terms
#                        decay every frame and its centroid stays within
#                        coast_radius_frac x its radius of the last measured
#                        position. Stationary tracks simply hold still.
#   S  stationary_mode   A track whose smoothed speed is below
#                        stationary_speed px/frame keeps the mid-column
#                        measurement noise (0.05) even near the top/bottom,
#                        has ay zeroed and its velocity damped on each
#                        detection, so shape jitter is not read as motion.
#
# Tests: test_tracker_stability.py (no camera, SuperCollider or pygame).

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

import blob_tracker as _bt
from blob_tracker   import BlobTracker
from blob_detection import PersistentBlob, TURNING

_COPY_FIELDS = ("centroid", "area", "birth", "death", "persistence", "orientation",
                "aspect_ratio", "elongation", "hu", "mean_hue")


@contextmanager
def _trusted_measurements(kf_ids: set):
    """For the duration of one BlobTracker.update(), give the Kalman filters in
    kf_ids the mid-column measurement noise everywhere (stationary mode). The
    base looks _adapt_measurement_noise up as a module global at call time, so
    patching it here affects only this call, and it is always restored."""
    orig = _bt._adapt_measurement_noise
    if not kf_ids:
        yield
        return

    def patched(kf, y, top, bottom, margin_frac=0.12):
        if id(kf) in kf_ids:
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.05
            return
        orig(kf, y, top, bottom, margin_frac)

    _bt._adapt_measurement_noise = patched
    try:
        yield
    finally:
        _bt._adapt_measurement_noise = orig


def _shape_from_contour(contour) -> tuple:
    """orientation, aspect_ratio, elongation, hu -- same formulas as
    BlobDetector.detect, so a fused detection looks like a detected one."""
    orientation, aspect_ratio, elongation, hu = 0.0, 1.0, 0.0, [0.0] * 7
    if contour is not None and len(contour) >= 5:
        Mc = cv2.moments(contour)
        if Mc["m00"] > 0:
            num = 2.0 * Mc["mu11"]
            den = Mc["mu20"] - Mc["mu02"]
            orientation = 0.5 * np.arctan2(num, den) if den != 0 else 0.0
        (_, _), (ea, eb), _ = cv2.fitEllipse(contour)
        long_ax, short_ax = max(ea, eb), min(ea, eb) + 1e-6
        aspect_ratio = long_ax / short_ax
        elongation = float(np.clip(1.0 - short_ax / long_ax, 0.0, 1.0))
        hu = [float(-np.copysign(1.0, v) * np.log10(abs(v) + 1e-10))
              for v in cv2.HuMoments(Mc).flatten()]
    return float(orientation), float(aspect_ratio), elongation, hu


class StableBlobTracker(BlobTracker):
    """BlobTracker that keeps one id per physical blob through region
    fluctuation. Same constructor arguments as BlobTracker, plus:

        fuse_fragments      fuse adjacent fragments of one tracked blob   (F)
        reassociate         offer new tracks to coasting tracks first     (R)
        confirm_cycles      detections before `birth` (1 = immediate)     (T)
        damp_coasting       stop coasting tracks from flying away         (D)
        stationary_mode     no buoyancy prior for still blobs             (S)
        absorb_frac         fraction of a detection inside a track mask
                            for it to count as part of that blob
        reassoc_overlap     minimum overlap coefficient for re-association
        reassoc_gate        centroid gate, in radii of the larger blob
        coast_damping       per-frame factor on vx, vy, ay while coasting
        coast_radius_frac   max coasting drift, in radii, from last measured
        stationary_speed    px/frame below which a track is stationary
    """

    def __init__(self, *args,
                 fuse_fragments: bool = True,
                 reassociate: bool = True,
                 confirm_cycles: int = 3,
                 damp_coasting: bool = True,
                 stationary_mode: bool = True,
                 absorb_frac: float = 0.5,
                 reassoc_overlap: float = 0.3,
                 reassoc_gate: float = 1.5,
                 coast_damping: float = 0.7,
                 coast_radius_frac: float = 1.0,
                 stationary_speed: float = 0.3,
                 complete_partials: bool = True,
                 partial_frac: float = 0.5,
                 partial_pull: float = 0.15,
                 partial_max_run: int = 8,
                 verbose: bool = False,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fuse_fragments    = bool(fuse_fragments)
        self.reassociate       = bool(reassociate)
        self.confirm_cycles    = max(1, int(confirm_cycles))
        self.damp_coasting     = bool(damp_coasting)
        self.stationary_mode   = bool(stationary_mode)
        self.absorb_frac       = float(absorb_frac)
        self.reassoc_overlap   = float(reassoc_overlap)
        self.reassoc_gate      = float(reassoc_gate)
        self.coast_damping     = float(coast_damping)
        self.coast_radius_frac = float(coast_radius_frac)
        self.stationary_speed  = float(stationary_speed)
        self.complete_partials = bool(complete_partials)
        self.partial_frac      = float(partial_frac)
        self.partial_pull      = float(partial_pull)
        self.partial_max_run   = int(partial_max_run)
        self.verbose           = verbose
        self._k_adj = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._k_seal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._mask_cache_cycle: dict = {}
        self.stats = {"births": 0, "fused": 0, "reassociated": 0, "partial_views": 0,
                      "tentative_dropped": 0, "confirmed": 0, "split_confirmed": 0}

    # ── public ─────────────────────────────────────────────────────────────
    def update(self, detections, frame: np.ndarray,
               grey: Optional[np.ndarray] = None) -> list:
        if detections is None:
            super().update(None, frame, grey=grey)
            self._constrain_coasting()
            return self.get_active()

        dets = list(detections)
        self._mask_cache_cycle = {}
        if self.fuse_fragments:
            dets = self._fuse(dets)
        if self.complete_partials:
            dets = self._complete(dets)

        before   = set(self._tracks.keys())
        tent_before = {t for t, tr in self._tracks.items() if tr.get("st_tentative", False)}
        # Masks as they were before this frame's flow advection. Advection can
        # shrink a mask in the very frame a lobe separates, so parent tests
        # use the better of the two. (The base replaces track["mask"] with a
        # new array, so these references stay valid without a copy.)
        pre_masks   = {t: tr["mask"] for t, tr in self._tracks.items()}
        slow_ids = ({id(t["kf"]) for t in self._tracks.values() if self._is_slow(t)}
                    if self.stationary_mode else set())
        with _trusted_measurements(slow_ids):
            super().update(dets, frame, grey=grey)

        new_ids = [t for t in self._tracks if t not in before]
        born    = {e[1] for e in self.events if e[0] == "birth"}
        if self.reassociate:
            self._reassociate([t for t in new_ids if t in born], before)
        self._gate_births([t for t in new_ids if t in self._tracks and t in born], pre_masks)
        self._settle_tentatives(tent_before)
        self._bookkeep_measured(dets)
        self._constrain_coasting()
        self.stats["births"] += sum(1 for e in self.events if e[0] == "birth")
        return self.get_active()

    def get_active(self) -> list:
        return [v["blob"] for v in self._tracks.values()
                if v["blob"].alive and not v.get("st_tentative", False)]

    def is_tentative(self, tid: int) -> bool:
        t = self._tracks.get(tid)
        return bool(t and t.get("st_tentative", False))

    def hud_line(self) -> str:
        s = self.stats
        n_t = sum(1 for t in self._tracks.values() if t.get("st_tentative"))
        return (f"trk births:{s['births']} reassoc:{s['reassociated']} fused:{s['fused']} "
                f"partial:{s['partial_views']} tentative:{n_t} dropped:{s['tentative_dropped']}")

    # ── helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _radius(area: float) -> float:
        return math.sqrt(max(float(area), 1.0) / math.pi)

    def _is_slow(self, track: dict) -> bool:
        return track.get("st_speed", 0.0) < self.stationary_speed

    @staticmethod
    def _bbox(mask) -> tuple:
        m = mask.view(np.uint8) if mask.dtype == np.bool_ else mask
        x, y, w, h = cv2.boundingRect(m)
        return (x, y, x + w, y + h)

    def _dmask(self, d) -> np.ndarray:
        """Boolean detection mask, cached for this detection cycle."""
        key = id(d)
        m = self._mask_cache_cycle.get(key)
        if m is None:
            m = self._det_mask(d) > 0
            self._mask_cache_cycle[key] = m
        return m

    @staticmethod
    def _boxes_meet(a: tuple, b: tuple, pad: int = 0) -> bool:
        return not (a[2] + pad < b[0] or b[2] + pad < a[0] or
                    a[3] + pad < b[1] or b[3] + pad < a[1])

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  ~ {msg}")

    # ── F: fuse fragments of one tracked blob ──────────────────────────────
    def _fuse(self, dets: list) -> list:
        if len(dets) < 2 or not self._tracks:
            return dets
        masks = [self._dmask(d) for d in dets]
        areas = [max(int(np.count_nonzero(m)), 1) for m in masks]
        dbox  = [self._bbox(m) for m in masks]
        tids  = list(self._tracks.keys())
        tmask = [self._tracks[t]["mask"] > 0 for t in tids]
        tbox  = [self._bbox(m) for m in tmask]

        owner = {}
        for j, m in enumerate(masks):
            best, best_cov = None, self.absorb_frac
            for i, tm in enumerate(tmask):
                if not self._boxes_meet(dbox[j], tbox[i]):
                    continue
                cov = np.count_nonzero(m & tm) / areas[j]
                if cov >= best_cov:
                    best, best_cov = tids[i], cov
            if best is not None:
                owner.setdefault(best, []).append(j)

        fused_away, out_extra = set(), []
        for tid, js in owner.items():
            if len(js) < 2:
                continue
            dil = {j: cv2.dilate(masks[j].astype(np.uint8), self._k_adj) > 0 for j in js}
            # connected components of the adjacency graph
            seen = set()
            for j0 in js:
                if j0 in seen:
                    continue
                comp, stack = [], [j0]
                seen.add(j0)
                while stack:
                    a = stack.pop()
                    comp.append(a)
                    for b in js:
                        if (b not in seen and self._boxes_meet(dbox[a], dbox[b], pad=3)
                                and np.any(dil[a] & masks[b])):
                            seen.add(b)
                            stack.append(b)
                if len(comp) >= 2:
                    out_extra.append(self._fused_detection([dets[j] for j in comp],
                                                           [masks[j] for j in comp]))
                    fused_away.update(comp)
                    self.stats["fused"] += len(comp) - 1
                    self._log(f"fused {len(comp)} fragments of #{tid}")
        if not fused_away:
            return dets
        return [d for j, d in enumerate(dets) if j not in fused_away] + out_extra

    def _fused_detection(self, parts: list, masks: list) -> PersistentBlob:
        union = np.zeros_like(masks[0], dtype=np.uint8)
        for m in masks:
            union[m] = 255
        union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, self._k_seal)  # seal watershed lines
        contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea) if contours else parts[0].contour
        M = cv2.moments(union)
        area = float(sum(p.area for p in parts))
        if M["m00"] > 0:
            centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"])
        else:
            centroid = (sum(p.centroid[0] * p.area for p in parts) / area,
                        sum(p.centroid[1] * p.area for p in parts) / area)
        lead = max(parts, key=lambda p: p.persistence)
        ang  = [math.radians(p.mean_hue * 2.0) for p in parts]
        w    = [p.area for p in parts]
        hue  = (math.degrees(math.atan2(sum(a * math.sin(x) for a, x in zip(w, ang)),
                                        sum(a * math.cos(x) for a, x in zip(w, ang)))) / 2.0) % 180.0
        orientation, aspect, elong, hu = _shape_from_contour(contour)
        det = PersistentBlob(centroid=(float(centroid[0]), float(centroid[1])), area=area,
                             birth=lead.birth, death=lead.death, persistence=lead.persistence,
                             contour=contour, orientation=orientation, aspect_ratio=aspect,
                             elongation=elong, hu=hu, mean_hue=float(hue), direction=TURNING)
        diag = []
        for p in parts:
            diag.extend(list(getattr(p, "diagram", []) or []))
        if diag:
            det.diagram = diag
        return det

    # ── P: partial views ───────────────────────────────────────────────────
    def _complete(self, dets: list) -> list:
        """A lone detection lying inside a track's KNOWN extent (its last full
        detection mask) at under partial_frac of its known area is a partial
        view: the persistence region shrank to a core or one lobe, the blob did
        not. It is replaced by a detection with the blob's known geometry and
        the detection's topology, whose centroid moves only partial_pull of the
        way toward the core, so neither association nor the centroid (pitch)
        jumps. A shrink that lasts partial_max_run cycles is accepted as real."""
        if not dets:
            return dets
        masks = [self._dmask(d) for d in dets]
        areas = [max(int(np.count_nonzero(m)), 1) for m in masks]
        dbox  = [self._bbox(m) for m in masks]
        inside = {}                                 # det j -> [tid, ...]
        per_track = {}                              # tid -> [det j, ...]
        for tid, tr in self._tracks.items():
            known = tr.get("st_known_mask")
            if known is None:
                continue
            kbox = tr.get("st_known_bbox") or self._bbox(known)
            km = None
            for j, m in enumerate(masks):
                if not self._boxes_meet(dbox[j], kbox):
                    continue
                km = known > 0 if km is None else km
                if np.count_nonzero(m & km) / areas[j] >= self.absorb_frac:
                    inside.setdefault(j, []).append(tid)
                    per_track.setdefault(tid, []).append(j)
        out = list(dets)
        for tid, js in per_track.items():
            if len(js) != 1 or len(inside[js[0]]) != 1:
                continue                            # ambiguous: leave it to the base
            j, tr = js[0], self._tracks[tid]
            d = dets[j]
            known_area = tr.get("st_known_area", tr["blob"].area)
            if d.area >= self.partial_frac * known_area:
                continue
            if tr.get("st_partial_run", 0) >= self.partial_max_run:
                continue                            # it has really shrunk: accept it
            out[j] = self._completed(d, tr)
        return out

    def _completed(self, d: PersistentBlob, tr: dict) -> PersistentBlob:
        b = tr["blob"]
        lx, ly = tr.get("st_last_meas", b.centroid)
        a = self.partial_pull
        c = (lx + a * (d.centroid[0] - lx), ly + a * (d.centroid[1] - ly))
        km = (tr["st_known_mask"] > 0).astype(np.uint8) * 255
        cs, _ = cv2.findContours(km, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(cs, key=cv2.contourArea) if cs else d.contour
        # shift the known contour to the completed centroid
        kc = tr.get("st_known_centroid", (lx, ly))
        contour = (contour + np.array([[c[0] - kc[0], c[1] - kc[1]]])).round().astype(np.int32)
        out = PersistentBlob(centroid=(float(c[0]), float(c[1])),
                             area=float(tr.get("st_known_area", b.area)),
                             birth=d.birth, death=d.death, persistence=d.persistence,
                             contour=contour, orientation=b.orientation,
                             aspect_ratio=b.aspect_ratio, elongation=b.elongation,
                             hu=list(b.hu), mean_hue=d.mean_hue, direction=TURNING)
        if getattr(d, "diagram", None):
            out.diagram = list(d.diagram)
        out._st_partial = True
        self.stats["partial_views"] += 1
        return out

    # ── R: re-associate new tracks with coasting ones ──────────────────────
    def _reassociate(self, new_ids: list, before: set) -> None:
        coasting = [t for t in before
                    if t in self._tracks and self._tracks[t]["blob"].coast_frames > 0
                    and not self._tracks[t].get("st_tentative", False)]
        if not new_ids or not coasting:
            return
        cost = np.full((len(new_ids), len(coasting)), 1e6)
        for i, nid in enumerate(new_ids):
            nt = self._tracks[nid]
            nm = nt["mask"] > 0
            na = max(int(np.count_nonzero(nm)), 1)
            nc = nt["blob"].centroid
            for k, cid in enumerate(coasting):
                ct = self._tracks[cid]
                ref_c = ct.get("st_last_meas", ct["blob"].centroid)
                ref_a = ct.get("st_known_area", ct["blob"].area)
                ratio = max(nt["blob"].area, 1.0) / max(ref_a, 1.0)
                ov = 0.0
                for m in (ct.get("st_known_mask"), ct["mask"]):
                    if m is None:
                        continue
                    mm = m > 0
                    denom = min(na, max(int(np.count_nonzero(mm)), 1))
                    ov = max(ov, np.count_nonzero(nm & mm) / denom)
                gate = self.reassoc_gate * self._radius(max(nt["blob"].area, ref_a))
                d = math.hypot(nc[0] - ref_c[0], nc[1] - ref_c[1])
                # Overlap decides on its own (a region shrinking to a core
                # inside the blob is the commonest fluctuation, and its area
                # ratio can be far below 1/4). Without overlap, the region
                # must be near the last measured centroid AND of similar size.
                if ov < self.reassoc_overlap and (d > gate or not (0.1 <= ratio <= 4.0)):
                    continue
                cost[i, k] = (1.0 - ov) + 0.5 * min(1.0, d / max(gate, 1e-6)) \
                             + 0.1 * min(2.0, abs(math.log(ratio)))
        rows, cols = linear_sum_assignment(cost)
        for i, k in zip(rows, cols):
            if cost[i, k] < 1e5:
                self._fold_into(new_ids[i], coasting[k])

    def _fold_into(self, new_id: int, old_id: int) -> None:
        """Move new track new_id's measurement onto the coasting track old_id
        and delete new_id with its birth event."""
        nt = self._tracks.pop(new_id)
        ot = self._tracks[old_id]
        nb, ob = nt["blob"], ot["blob"]

        now   = nt.get("rate_t", ot.get("rate_t", 0.0))
        dt    = now - ot.get("rate_t", now)
        if dt > 1e-3:
            prev_a = ot.get("rate_area", nb.area)
            prev_p = ot.get("rate_pers", nb.persistence)
            ob.d_area        = ((nb.area - prev_a) / max(prev_a, 1.0)) / dt
            ob.d_persistence = ((nb.persistence - prev_p) / max(prev_p, 1.0)) / dt
        ot["rate_t"], ot["rate_area"], ot["rate_pers"] = now, nb.area, nb.persistence

        # PARTIAL VIEW: the new region is a small part of the blob as last
        # measured (its persistence region shrank to a core, or one lobe was
        # seen). The blob has not shrunk or moved, so keep its known mask,
        # area and shape, take only the topological values, and correct the
        # position with high measurement noise so the centroid (and pitch)
        # does not jump to the core.
        known = ot.get("st_known_mask")
        nm = nt["mask"] > 0
        partial = False
        if known is not None:
            na = max(int(np.count_nonzero(nm)), 1)
            inside = np.count_nonzero(nm & (known > 0)) / na
            partial = (inside >= self.absorb_frac
                       and nb.area < self.partial_frac * ot.get("st_known_area", ob.area)
                       and ot.get("st_partial_run", 0) < self.partial_max_run)

        self._update_topo_stats(ob, nb)
        fields = (("birth", "death", "persistence", "mean_hue") if partial else _COPY_FIELDS)
        for f in fields:
            setattr(ob, f, getattr(nb, f))
        ob.coast_frames = 0
        ob.age += 1
        if partial:
            ot["mask"] = np.maximum(known, nt["mask"])
        else:
            ot["mask"] = nt["mask"].copy()

        kf = ot["kf"]
        if partial:
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4.0
        elif self.stationary_mode and self._is_slow(ot):
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.05
        else:
            _bt._adapt_measurement_noise(kf, float(nb.centroid[1]), 0.0, float(self.profile.roi_h))
        post = kf.correct(np.array([[nb.centroid[0]], [nb.centroid[1]]], dtype=np.float32))
        if partial:
            ob.centroid = (float(post[0, 0]), float(post[1, 0]))
        if partial:
            ot["st_partial_now"] = True
            self.stats["partial_views"] += 1

        self.events = [e for e in self.events if not (e[0] == "birth" and e[1] == new_id)]
        if new_id == self.next_id - 1:
            self.next_id -= 1                       # keep ids dense
        self.stats["reassociated"] += 1
        self._log(f"re-associated new region with #{old_id} (no birth)")

    # ── T: tentative births ────────────────────────────────────────────────
    def _gate_births(self, new_ids: list, pre_masks: dict) -> None:
        if self.confirm_cycles <= 1:
            return
        live = [t for t, tr in self._tracks.items()
                if t not in new_ids and tr["blob"].coast_frames == 0
                and not tr.get("st_tentative", False)]
        for nid in new_ids:
            nt = self._tracks[nid]
            nm = nt["mask"] > 0
            na = max(int(np.count_nonzero(nm)), 1)
            parent, best = None, self.absorb_frac
            for lid in live:
                cov = 0.0
                for m in (self._tracks[lid]["mask"], pre_masks.get(lid)):
                    if m is not None:
                        cov = max(cov, np.count_nonzero(nm & (m > 0)) / na)
                if cov >= best:
                    parent, best = lid, cov
            nt["st_tentative"] = True
            nt["st_confirm"]   = 0                  # incremented in _settle_tentatives
            nt["st_parent"]    = parent
            self.events = [e for e in self.events if not (e[0] == "birth" and e[1] == nid)]

    def _settle_tentatives(self, tent_before: set) -> None:
        tent = tent_before | {t for t, tr in self._tracks.items()
                              if tr.get("st_tentative", False)}
        # Rewrite events that mention a tentative id: the binding layer never
        # saw these tracks, so their deaths and merges must stay silent.
        out = []
        for e in self.events:
            kind = e[0]
            if kind == "death" and e[1] in tent:
                self.reident.records.pop(e[1], None)
                continue
            if kind == "merge":
                _, merged, survivor = e
                if merged in tent:
                    continue                    # a flicker merged away: silent
                if survivor in tent:
                    self._confirm(survivor, out, as_birth=True)   # now carries a voice
            if kind == "split":
                _, parent, child = e
                if parent in tent:
                    if child in self._tracks:
                        ct = self._tracks[child]
                        ct["st_tentative"], ct["st_confirm"], ct["st_parent"] = True, 0, None
                    continue
            out.append(e)
        self.events = out

        for tid in list(self._tracks.keys()):
            tr = self._tracks[tid]
            if not tr.get("st_tentative", False):
                continue
            if tr["blob"].coast_frames > 0:
                del self._tracks[tid]           # missed once: it never was
                self.reident.records.pop(tid, None)
                self.stats["tentative_dropped"] += 1
                self._log(f"dropped flicker #{tid}")
                continue
            tr["st_confirm"] = tr.get("st_confirm", 0) + 1
            if tr["st_confirm"] >= self.confirm_cycles:
                self._confirm(tid, self.events)

    def _confirm(self, tid: int, events: list, as_birth: bool = False) -> None:
        tr = self._tracks.get(tid)
        if tr is None or not tr.get("st_tentative", False):
            return
        tr["st_tentative"] = False
        parent = tr.get("st_parent")
        if (not as_birth and parent is not None and parent in self._tracks
                and not self._tracks[parent].get("st_tentative", False)):
            events.append(("split", parent, tid))
            self.stats["split_confirmed"] += 1
        else:
            events.append(("birth", tid))
        self.stats["confirmed"] += 1

    # ── D/S: bookkeeping and coasting constraints ──────────────────────────
    def _bookkeep_measured(self, dets: list) -> None:
        """For every track measured this cycle: last measured centroid, a
        smoothed speed (from an EMA of position, so shape jitter does not read
        as motion), and the KNOWN extent -- the mask and area of its last FULL
        detection. The base's own mask is flow-advected and erodes when the
        region fluctuates, so it is not used as the reference."""
        f = self.frame_count
        by_c = {(round(float(d.centroid[0]), 3), round(float(d.centroid[1]), 3)): d for d in dets}
        for tr in self._tracks.values():
            b = tr["blob"]
            if not b.alive or b.coast_frames != 0:
                continue
            c = (float(b.centroid[0]), float(b.centroid[1]))
            if "st_pos_ema" in tr:
                px, py = tr["st_pos_ema"]
                nx, ny = px + 0.3 * (c[0] - px), py + 0.3 * (c[1] - py)
                frames = max(1, f - tr.get("st_last_frame", f - 1))
                inst = math.hypot(nx - px, ny - py) / frames
                tr["st_speed"] = tr.get("st_speed", 0.0) + 0.3 * (inst - tr.get("st_speed", 0.0))
                tr["st_pos_ema"] = (nx, ny)
            else:
                tr["st_pos_ema"] = c
                tr["st_speed"] = 0.0
            tr["st_last_meas"]  = c
            tr["st_last_frame"] = f

            d = by_c.get((round(c[0], 3), round(c[1], 3)))
            partial = tr.pop("st_partial_now", False) or bool(d is not None and getattr(d, "_st_partial", False))
            if partial:
                tr["st_partial_run"] = tr.get("st_partial_run", 0) + 1
            else:
                tr["st_partial_run"] = 0
                if d is not None:
                    km = self._dmask(d).astype(np.uint8) * 255
                    tr["st_known_area"] = float(d.area)
                else:                               # merge survivor / split parent / fold
                    km = tr["mask"].copy()
                    tr["st_known_area"] = float(b.area)
                tr["st_known_mask"] = km
                tr["st_known_bbox"] = self._bbox(km)
                tr["st_known_centroid"] = c
            if "st_known_mask" not in tr:
                tr["st_known_mask"] = tr["mask"].copy()
                tr["st_known_bbox"] = self._bbox(tr["st_known_mask"])
                tr["st_known_area"] = float(b.area)
                tr["st_known_centroid"] = c

            if self.stationary_mode and self._is_slow(tr):
                sp = tr["kf"].statePost.copy()
                sp[2, 0] *= 0.5
                sp[3, 0] *= 0.5
                if sp.shape[0] > 4:
                    sp[4, 0] = 0.0
                tr["kf"].statePost = sp
        self._update_physical_fields()

    def _constrain_coasting(self) -> None:
        if not self.damp_coasting:
            return
        changed = False
        for tr in self._tracks.values():
            b = tr["blob"]
            if not b.alive or b.coast_frames == 0 or "st_last_meas" not in tr:
                continue
            sp = tr["kf"].statePost.copy()
            lx, ly = tr["st_last_meas"]
            if self.stationary_mode and self._is_slow(tr):
                sp[0, 0], sp[1, 0] = lx, ly
                sp[2:, 0] = 0.0
            else:
                sp[2:, 0] *= self.coast_damping
                r = self.coast_radius_frac * self._radius(tr.get("st_known_area", b.area))
                dx, dy = float(sp[0, 0]) - lx, float(sp[1, 0]) - ly
                d = math.hypot(dx, dy)
                if d > r:
                    sp[0, 0], sp[1, 0] = lx + dx * r / d, ly + dy * r / d
            tr["kf"].statePost = sp
            b.centroid = (float(sp[0, 0]), float(sp[1, 0]))
            changed = True
        if changed:
            self._update_physical_fields()
