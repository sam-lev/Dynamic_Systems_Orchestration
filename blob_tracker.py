# blob_tracker.py -- Audible Lava v3.0
#
# Complete rewrite implementing six improvements over v2.0:
#
#  1. OPTICAL FLOW MASK ADVECTION as primary identity carrier.
#     Each track owns a binary mask advected forward every frame with dense
#     Farneback flow. Identity is carried by the mask, not re-solved by
#     position matching. Detection refines boundaries and handles births/deaths.
#
#  2. WASSERSTEIN-INSPIRED MATCHING as secondary resolver.
#     When multiple advected masks overlap with one detection (ambiguous case),
#     topological distance in (birth, death) space resolves the ambiguity.
#     Position is a secondary tiebreaker only.
#
#  3. PHYSICS-INFORMED KALMAN with buoyancy dynamics.
#     5-state filter [x, y, vx, vy, ay] where ay is a slowly-decaying
#     vertical acceleration term encoding buoyancy. Near lamp top/bottom,
#     measurement noise increases so the filter trusts its physics model more.
#
#  4. BIRTH TOPOLOGY FINGERPRINTING for re-identification.
#     Lost blobs are stored in a ReidentificationBuffer with their topological
#     and thermodynamic fingerprint. New detections are checked against the
#     buffer before a birth event fires.
#
#  5. KINEMATIC DIRECTION FROM KALMAN VY (v3.2 — replaces hue-based).
#     `direction` (rising/falling/turning) is now read every frame from
#     the Kalman filter's vy state, not from blob hue. Hue proved
#     camera-firmware-sensitive and only weakly correlated with thermal
#     phase in practice; vy is a direct, physically grounded measure of
#     which way the blob is moving. The Kalman ay buoyancy prior is
#     seeded as zero on spawn (was direction-keyed) and the filter
#     accumulates evidence from successive measurements.
#
#  6. DIRECTION-CONSISTENT COST in ambiguous matching.
#     Small additional penalty for matches where direction is inconsistent.

import cv2
import math
import numpy as np
import time
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

from blob_detection import PersistentBlob, RISING, FALLING, TURNING
from lamp_profile import LampProfile


# ── TrackedBlob ────────────────────────────────────────────────────────────────
@dataclass
class TrackedBlob:
    """
    A blob with a stable integer ID. Carries all topological, shape, and
    thermodynamic attributes. `direction` is updated every frame from the
    Kalman vy estimate (kinematic direction) — not from hue, which proved
    camera-sensitive and weakly correlated with thermal phase.
    """
    id:           int
    centroid:     tuple
    area:         float
    birth:        float
    death:        float
    persistence:  float
    orientation:  float = 0.0
    aspect_ratio: float = 1.0
    elongation:   float = 0.0
    hu:           list  = field(default_factory=lambda: [0.0] * 7)
    mean_hue:     float = 0.0
    direction:    str   = TURNING
    age:          int   = 0
    coast_frames: int   = 0
    alive:        bool  = True

    # ── Physical / geometric state (v4.1) ─────────────────────────────────
    # Populated every frame from the Kalman filter (vx, vy, ay, speed) and on
    # each detection cycle from frame-to-frame deltas (d_area, d_persistence).
    # These feed the routing layer so a blob's *motion* and *change of shape /
    # topology* — not just its instantaneous position — shape its voice.
    #   vx, vy        : Kalman velocity, px/frame (image y: vy<0 = rising)
    #   speed         : |velocity|, px/frame
    #   ay            : Kalman buoyancy acceleration term
    #   d_area        : fractional area change per second (+ = expanding)
    #   d_persistence : fractional persistence change per second
    #                   (+ = consolidating, - = dissolving)
    vx:            float = 0.0
    vy:            float = 0.0
    speed:         float = 0.0
    ay:            float = 0.0
    d_area:        float = 0.0
    d_persistence: float = 0.0

    # ── v4.6 matching state ───────────────────────────────────────────────
    # `birth`/`persistence` above are the LAST measurement. Matching must not
    # compare against those: the topological peak tracks a region of similar
    # intensity inside the blob, not the blob itself, so as the blob deforms
    # the peak can shift, split, or hand dominance to another sub-region, and
    # the reported birth jumps even though the blob moved a pixel. One noisy
    # frame would then corrupt the reference used for the next comparison.
    # These smoothed values (~1 s time constant) are the matching reference.
    s_birth:       float = 0.0
    s_persistence: float = 0.0
    # Per-track observed variability of those two quantities (rolling stdev).
    # Matching costs are divided by these instead of a fixed 50-grey-level
    # constant, so a blob whose birth naturally swings +/-25 is compared
    # leniently and a stable one strictly. This also removes topo_threshold
    # as a hand-tuned number: the gate becomes "more than N sigma from this
    # track's own history".
    v_birth:       float = 8.0
    v_persistence: float = 8.0
    n_obs:         int   = 0
    # Full persistence diagram for this blob: every (birth, death) point
    # inside its mask, not just the dominant peak. Compared with Wasserstein,
    # which changes only slightly when a peak splits or a noise peak appears.
    diagram:       list  = field(default_factory=list)
    s_diagram:     list  = field(default_factory=list)


# ── BirthRecord for re-identification ─────────────────────────────────────────
@dataclass
class BirthRecord:
    """Fingerprint of a blob at the moment it was lost.

    v4.6: records the SMOOTHED topological values and the track's own observed
    variability, plus the shape descriptors, so a returning blob is compared
    against a stable reference on a scale that reflects how noisy that
    particular blob actually was.
    """
    blob_id:       int
    birth_value:   float    # smoothed topological birth (peak intensity)
    persistence:   float    # smoothed persistence
    initial_area:  float
    y_norm:        float    # normalised vertical position at loss
    mean_hue:      float
    direction:     str
    lost_at:       float    # timestamp (time.time())
    # v4.6 additions
    v_birth:       float = 8.0
    v_persistence: float = 8.0
    elongation:    float = 0.0
    hu:            list  = field(default_factory=lambda: [0.0] * 7)
    diagram:       list  = field(default_factory=list)

    def distance(self, det: PersistentBlob, y_norm: float) -> float:
        """Fingerprint distance. Lower = more likely the same blob.

        v4.6 weights:
            topological  0.28  (was 0.72 across birth+persistence)
            area         0.30  (was 0.18)
            position     0.32  (was 0.10)
            shape        0.10  (new: elongation + Hu moments)
            direction    +0.30 flat penalty on a confident mismatch

        The old split put 72% of the decision on birth and persistence, which
        are the least stable things measured: the persistence peak follows a
        region of similar intensity inside the blob, so deformation alone can
        move it discontinuously. Area and position are far better behaved for
        wax moving 0.5-2 px/frame. Topology is still the largest single term
        after position+area, but it no longer decides on its own.

        Topological terms are divided by the track's OWN measured variability
        rather than a fixed 50 grey levels, so a blob that naturally fluctuated
        widely is compared leniently and a stable one strictly.
        """
        vb = 1.4826 * max(self.v_birth, 3.0)
        vp = 1.4826 * max(self.v_persistence, 3.0)
        # Full-diagram comparison when both sides have one: stable when a peak
        # splits or a noise peak appears, where a single-point comparison is not.
        topo = None
        det_diag = list(getattr(det, "diagram", []) or [])
        if det_diag and self.diagram:
            try:
                from system_state import diagram_wasserstein
                topo = min(1.0, diagram_wasserstein(self.diagram, det_diag) / 30.0)
            except Exception:
                topo = None
        if topo is None:
            zb = abs(det.birth - self.birth_value) / vb
            zp = abs(det.persistence - self.persistence) / vp
            topo = min(1.0, math.hypot(zb, zp) / 4.0)

        area = min(1.0, abs(det.area - self.initial_area) / max(self.initial_area, 1.0))
        # Scaled so that half a lamp-height of vertical displacement saturates:
        # a blob that reappears at a very different height is almost certainly
        # not the same blob, whatever its topology says.
        pos  = min(1.0, abs(y_norm - self.y_norm) * 2.5)

        e1, e2 = float(self.elongation), float(getattr(det, "elongation", 0.0))
        d_elong = abs(e1 - e2) / max(1.0, max(e1, e2))
        h1, h2 = list(self.hu or []), list(getattr(det, "hu", []) or [])
        d_hu = 0.0
        if len(h1) >= 3 and len(h2) >= 3:
            acc = 0.0
            for a, b in list(zip(h1, h2))[:3]:
                la = math.copysign(math.log10(abs(a) + 1e-30), a)
                lb = math.copysign(math.log10(abs(b) + 1e-30), b)
                acc += abs(la - lb)
            d_hu = (acc / 3.0) / 3.0
        shape = min(1.0, 0.5 * d_elong + 0.5 * d_hu)

        dir_penalty = 0.0
        if det.direction != TURNING and self.direction != TURNING:
            if det.direction != self.direction:
                dir_penalty = 0.3

        return (0.28 * topo + 0.30 * area + 0.32 * pos
                + 0.10 * shape + dir_penalty)


class ReidentificationBuffer:
    """
    Stores BirthRecords of recently-lost blobs and tries to match new
    detections against them before firing a birth event.
    """

    def __init__(self, max_age_seconds: float = 15.0, match_threshold: float = 0.35):
        self.records:         Dict[int, BirthRecord] = {}
        self.max_age          = max_age_seconds
        self.match_threshold  = match_threshold

    def record_loss(self, blob: TrackedBlob, y_norm: float) -> None:
        self.records[blob.id] = BirthRecord(
            blob_id      = blob.id,
            # Smoothed values, not the last (possibly noisy) frame.
            birth_value  = blob.s_birth if blob.n_obs else blob.birth,
            persistence  = blob.s_persistence if blob.n_obs else blob.persistence,
            initial_area = blob.area,
            y_norm       = y_norm,
            mean_hue     = blob.mean_hue,
            direction    = blob.direction,
            lost_at      = time.time(),
            v_birth      = blob.v_birth,
            v_persistence= blob.v_persistence,
            elongation   = blob.elongation,
            hu           = list(blob.hu),
            diagram      = list(blob.diagram),
        )

    def try_reidentify(
        self, det: PersistentBlob, y_norm: float
    ) -> Optional[int]:
        """
        Returns original blob_id if the detection matches a lost record,
        None otherwise. Expires stale records.
        """
        now = time.time()
        self.records = {
            k: v for k, v in self.records.items()
            if now - v.lost_at < self.max_age
        }

        best_id   = None
        best_dist = self.match_threshold
        for bid, rec in self.records.items():
            d = rec.distance(det, y_norm)
            if d < best_dist:
                best_dist = d
                best_id   = bid

        if best_id is not None:
            del self.records[best_id]
        return best_id


# ── Physics-informed Kalman ────────────────────────────────────────────────────
def _make_buoyancy_kalman(
    x: float, y: float, direction: str, lamp_height: float
) -> cv2.KalmanFilter:
    """
    5-state Kalman filter: [x, y, vx, vy, ay]

    ay is a buoyancy acceleration term that decays toward zero (mean-
    reversion coefficient 0.9) but is initialized with a directional prior:
      rising  → small positive ay (upward buoyancy)
      falling → small negative ay (downward buoyancy)
      turning → ay = 0

    The transition integrates ay into y via Euler: y += vy + 0.5*ay.
    The lamp_height is used to scale process noise on ay relative to
    the physical scale of the lamp.
    """
    kf = cv2.KalmanFilter(5, 2)

    kf.transitionMatrix = np.array([
        [1, 0, 1, 0,   0  ],  # x  += vx
        [0, 1, 0, 1,   0.5],  # y  += vy + 0.5*ay  (Euler)
        [0, 0, 1, 0,   0  ],  # vx  = vx
        [0, 0, 0, 1,   1  ],  # vy += ay
        [0, 0, 0, 0,   0.9],  # ay decays (mean-reversion; buoyancy weakens at mid-column)
    ], dtype=np.float32)

    kf.measurementMatrix = np.array(
        [[1,0,0,0,0],[0,1,0,0,0]], dtype=np.float32)

    # Scale process noise on ay to lamp physical scale
    ay_noise = max(0.02, 0.5 / max(lamp_height, 1.0))
    kf.processNoiseCov = np.diag([
        1e-2, 1e-2,  # position noise
        1e-2, 1e-2,  # velocity noise
        ay_noise,    # acceleration noise: allows ay to evolve
    ]).astype(np.float32)

    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost        = np.eye(5, dtype=np.float32)

    # Directional prior on ay
    ay_prior = {RISING: 0.15, FALLING: -0.15, TURNING: 0.0}
    initial_ay = ay_prior.get(direction, 0.0)

    kf.statePost = np.array([[x],[y],[0],[0],[initial_ay]], dtype=np.float32)
    return kf


def _adapt_measurement_noise(
    kf: cv2.KalmanFilter,
    y: float, lamp_top_y: float, lamp_bottom_y: float,
    margin_frac: float = 0.12,
) -> None:
    """
    Near the top or bottom turning points, trust the physics model more
    (increase measurement noise). In mid-column, trust detections more.
    """
    lamp_h  = max(lamp_bottom_y - lamp_top_y, 1.0)
    margin  = lamp_h * margin_frac
    near_top    = y < lamp_top_y    + margin
    near_bottom = y > lamp_bottom_y - margin

    if near_top or near_bottom:
        # Turning point: high measurement noise → trust Kalman model
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
    else:
        # Mid-column: low measurement noise → trust detection
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.05


# ── BlobTracker ────────────────────────────────────────────────────────────────
class BlobTracker:
    """
    Flow-advected tracker with Wasserstein matching, physics Kalman,
    birth topology fingerprinting, and kinematic direction labelling.

    Per-frame pipeline:
        A. Compute dense Farneback optical flow (prev_grey → curr_grey)
        B. Advect all track masks forward with the flow field
        C. Match detections to advected masks by IoU overlap
        D. Resolve ambiguous overlaps with Wasserstein topological distance
        E. Every REFINE_INTERVAL frames: refine matched mask from detection contour
        F. Update physics Kalman filter for matched tracks
        G. Coast / retire unmatched tracks; record loss in fingerprint buffer
        H. For unmatched detections: check fingerprint buffer → reidentify or spawn

    Args:
        profile:           LampProfile providing lamp geometry
        max_coast:         frames a track coasts without detection before death
        iou_threshold:     minimum IoU for mask-based match
        topo_threshold:    max topological distance for Wasserstein match
        refine_interval:   frames between full mask refinements from detection
        reident_max_age:   seconds to keep fingerprint records
        reident_threshold: fingerprint distance threshold for re-identification
        direction_vy_threshold:
                           Kalman vy magnitude (px/frame) above which a
                           track is classified as RISING (vy<-thr) or
                           FALLING (vy>+thr); below ±thr it is TURNING.
                           Direction is now sourced kinematically from the
                           Kalman state rather than from blob hue.
    """

    REFINE_INTERVAL = 5   # refine mask from detection every N detection cycles

    def __init__(
        self,
        profile:           Optional[LampProfile] = None,
        max_coast:         int   = 20,
        iou_threshold:     float = 0.25,
        topo_threshold:    float = 35.0,
        reident_max_age:   float = 8.0,
        reident_threshold: float = 0.35,
        direction_vy_threshold: float = 0.3,
        flow_downscale:    float = 0.5,
        iou_downscale:     int   = 2,
    ):
        self.profile          = profile or LampProfile.default()
        self.max_coast        = max_coast
        self.iou_threshold    = iou_threshold
        self.topo_threshold   = topo_threshold
        self.direction_vy_threshold = direction_vy_threshold

        # v4.1: the pipeline runs on the lamp ROI, so all masks are ROI-sized.
        self.frame_h = self.profile.roi_h
        self.frame_w = self.profile.roi_w

        # v4.1 (#3a): Farneback runs at flow_downscale resolution and the
        # field is upsampled + rescaled back to full ROI resolution.
        self.flow_downscale = float(flow_downscale)
        # v4.1 (#5): the IoU matrix is computed on masks downsampled by this
        # integer factor (binary masks → INTER_NEAREST), which is visually
        # identical for blobs tens of px across at a fraction of the cost.
        self.iou_downscale  = max(1, int(iou_downscale))

        # track_id → {'blob': TrackedBlob, 'mask': np.ndarray, 'kf': KalmanFilter,
        #             'rate_t': float, 'rate_area': float, 'rate_pers': float}
        self._tracks:  Dict[int, dict] = {}
        self.next_id   = 0
        self.events:   List[tuple]     = []   # ('birth', id)
                                               # ('death', id)
                                               # ('merge', merged_id, survivor_id)
                                               # ('split', parent_id, child_id)
        self.frame_count  = 0    # every frame (predict + correct)
        self.detect_count = 0    # detection cycles only (gates REFINE_INTERVAL)

        self.prev_grey: Optional[np.ndarray] = None
        self._flow: Optional[np.ndarray]     = None   # cached flow field (ROI res)

        # v4.1 (#5): precomputed identity remap grids, reused by _advect_mask
        # every frame for every track (was reallocated per track per frame).
        self._base_map_x: Optional[np.ndarray] = None
        self._base_map_y: Optional[np.ndarray] = None
        self._ensure_remap_grids(self.frame_h, self.frame_w)

        self.reident = ReidentificationBuffer(reident_max_age, reident_threshold)

    # ── Remap-grid cache ────────────────────────────────────────────────────
    def _ensure_remap_grids(self, h: int, w: int) -> None:
        """(Re)build the identity coordinate grids used for flow advection."""
        if (h <= 0 or w <= 0):
            return
        if (self._base_map_x is not None
                and self._base_map_x.shape == (h, w)):
            return
        self._base_map_x = np.tile(
            np.arange(w, dtype=np.float32)[np.newaxis, :], (h, 1))
        self._base_map_y = np.tile(
            np.arange(h, dtype=np.float32)[:, np.newaxis], (1, w))

    # ── Flow advection helpers ─────────────────────────────────────────────
    def _compute_flow(self, curr_grey: np.ndarray) -> Optional[np.ndarray]:
        """Dense optical flow at reduced resolution (v4.1 #3a).

        Farneback is the single most expensive pipeline stage. The wax moves
        slowly (≤ ~2 px/frame) and the interior is texture-poor, so a half-res
        field is more than adequate. We compute at flow_downscale, upsample the
        field back to ROI resolution, and rescale the vectors by 1/scale so
        callers (mask advection) see full-resolution displacements.
        """
        if self.prev_grey is None:
            return None
        s = self.flow_downscale
        if s < 0.999:
            prev_s = cv2.resize(self.prev_grey, None, fx=s, fy=s,
                                interpolation=cv2.INTER_AREA)
            curr_s = cv2.resize(curr_grey, None, fx=s, fy=s,
                                interpolation=cv2.INTER_AREA)
            flow_s = cv2.calcOpticalFlowFarneback(
                prev_s, curr_s, None,
                pyr_scale=0.5, levels=2, winsize=11,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            h, w = curr_grey.shape[:2]
            flow = cv2.resize(flow_s, (w, h), interpolation=cv2.INTER_LINEAR)
            flow *= (1.0 / s)   # rescale displacement vectors to full res
            return flow
        return cv2.calcOpticalFlowFarneback(
            self.prev_grey, curr_grey, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

    def _advect_mask(
        self, mask: np.ndarray, flow: np.ndarray
    ) -> np.ndarray:
        """
        Advect a binary mask forward by one frame using inverse-warp remap.
        flow[y,x] = (dx, dy): pixel at (x,y) in prev moved to (x+dx, y+dy) now.
        Inverse remap: new[y,x] = old[y - dy, x - dx].

        v4.1 (#5): the identity coordinate grids are precomputed once
        (_ensure_remap_grids) and reused, so each advection is two array
        subtractions + a remap rather than rebuilding two H×W float32 grids
        per track per frame.
        """
        h, w = mask.shape[:2]
        if (self._base_map_x is None or self._base_map_x.shape != (h, w)):
            self._ensure_remap_grids(h, w)
        map_x = self._base_map_x - flow[..., 0]
        map_y = self._base_map_y - flow[..., 1]
        advected = cv2.remap(
            mask.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR
        )
        return (advected > 0.4).astype(np.uint8)

    # ── Mask from detection ────────────────────────────────────────────────
    def _det_mask(self, det: PersistentBlob) -> np.ndarray:
        m  = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        cx = int(det.centroid[0])
        cy = int(det.centroid[1])
        if len(det.contour) >= 3:
            cv2.fillPoly(m, [det.contour.astype(np.int32)], 255)
        else:
            r = max(10, int(np.sqrt(det.area / np.pi)))
            cv2.circle(m, (cx, cy), r, 255, -1)
        return m

    # ── IoU ────────────────────────────────────────────────────────────────
    @staticmethod
    def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        a  = mask_a > 0
        b  = mask_b > 0
        i  = np.sum(a & b)
        u  = np.sum(a | b)
        return float(i) / (float(u) + 1e-6)

    def _small_mask(self, mask: np.ndarray) -> np.ndarray:
        """Downsample a binary mask for cheap IoU (v4.1 #5).

        INTER_NEAREST keeps it binary; for blobs tens of px across the IoU of
        the downsampled pair tracks the full-res IoU to within a couple of
        percent at 1/iou_downscale² the pixel count.
        """
        f = self.iou_downscale
        if f <= 1:
            return (mask > 0).astype(np.uint8)
        h, w = mask.shape[:2]
        sm = cv2.resize(mask, (max(1, w // f), max(1, h // f)),
                        interpolation=cv2.INTER_NEAREST)
        return (sm > 0).astype(np.uint8)

    @staticmethod
    def _iou_small(a_small: np.ndarray, b_small: np.ndarray) -> float:
        """IoU on two precomputed boolean/uint8 small masks of equal shape."""
        a = a_small.astype(bool)
        b = b_small.astype(bool)
        i = np.count_nonzero(a & b)
        if i == 0:
            return 0.0
        u = np.count_nonzero(a | b)
        return float(i) / (float(u) + 1e-6)

    # ── Topological distance ───────────────────────────────────────────────
    # ── v4.6 topological matching ──────────────────────────────────────────
    TOPO_ALPHA   = 0.15    # EMA rate for smoothed birth/persistence (~1.5 s @ 15fps)
    VAR_ALPHA    = 0.08    # EMA rate for the per-track variability estimate
    VAR_FLOOR    = 3.0     # grey levels; stops the scale collapsing to zero
    # v_birth / v_persistence track MEAN ABSOLUTE DEVIATION, not standard
    # deviation. For normally distributed noise sigma = 1.4826 * MAD, so
    # dividing by MAD directly overstates the deviation by ~25% and the gate
    # rejects correct matches. Convert explicitly.
    MAD_TO_SIGMA = 1.4826
    TOPO_SIGMAS  = 4.0     # gate: reject beyond this many sigma
    DIAG_SCALE   = 60.0    # grey levels; normaliser for diagram-Wasserstein

    def _update_topo_stats(self, blob: TrackedBlob, det) -> None:
        """Fold a new detection into the track's smoothed topological
        reference and its own variability estimate."""
        b, p = float(det.birth), float(det.persistence)
        if blob.n_obs == 0:
            blob.s_birth, blob.s_persistence = b, p
        else:
            db = abs(b - blob.s_birth)
            dp = abs(p - blob.s_persistence)
            blob.v_birth = max(self.VAR_FLOOR,
                               blob.v_birth + self.VAR_ALPHA * (db - blob.v_birth))
            blob.v_persistence = max(self.VAR_FLOOR,
                                     blob.v_persistence + self.VAR_ALPHA * (dp - blob.v_persistence))
            a = self.TOPO_ALPHA
            blob.s_birth       += a * (b - blob.s_birth)
            blob.s_persistence += a * (p - blob.s_persistence)
        blob.n_obs += 1
        d = list(getattr(det, "diagram", []) or [])
        if d:
            blob.diagram = d
            blob.s_diagram = d if not blob.s_diagram else blob.s_diagram

    def _topo_dist(self, blob: TrackedBlob, det) -> float:
        """Topological distance in units of THIS TRACK'S OWN variability.

        Returns roughly "how many standard deviations away is this detection",
        so the caller can gate at a fixed number of sigma rather than a fixed
        number of grey levels. Compared against the SMOOTHED reference, not the
        last frame.

        When both sides carry a full persistence diagram, the diagram
        Wasserstein distance is used instead of the single dominant peak. That
        is the stability property worth having: a peak splitting in two, or a
        noise peak appearing, moves Wasserstein by a small amount, whereas the
        single-point comparison can change discontinuously.
        """
        det_diag = list(getattr(det, "diagram", []) or [])
        if det_diag and blob.diagram:
            try:
                from system_state import diagram_wasserstein
                w = diagram_wasserstein(blob.diagram, det_diag)
                return float(w / self.DIAG_SCALE)
            except Exception:
                pass
        ref_b = blob.s_birth if blob.n_obs else blob.birth
        ref_p = blob.s_persistence if blob.n_obs else blob.persistence
        sb = self.MAD_TO_SIGMA * max(blob.v_birth, self.VAR_FLOOR)
        sp = self.MAD_TO_SIGMA * max(blob.v_persistence, self.VAR_FLOOR)
        zb = (float(det.birth) - ref_b) / sb
        zp = (float(det.persistence) - ref_p) / sp
        return float(np.hypot(zb, zp))

    @staticmethod
    def _shape_dist(blob: TrackedBlob, det) -> float:
        """Distance over shape descriptors that are ALREADY computed on every
        blob and were previously unused in matching.

        Hu moments are invariant to translation, rotation and scale, so they
        are markedly more stable under deformation than the birth value is.
        Log-scaled because raw Hu moments span many orders of magnitude.
        Returns a value normalised to roughly [0, 1].
        """
        e1 = float(getattr(blob, "elongation", 0.0))
        e2 = float(getattr(det,  "elongation", 0.0))
        d_elong = abs(e1 - e2) / max(1.0, max(e1, e2))

        h1 = list(getattr(blob, "hu", []) or [])
        h2 = list(getattr(det,  "hu", []) or [])
        d_hu = 0.0
        if len(h1) >= 3 and len(h2) >= 3:
            acc, n = 0.0, 0
            for a, b in list(zip(h1, h2))[:3]:      # first 3 carry most shape info
                la = math.copysign(math.log10(abs(a) + 1e-30), a)
                lb = math.copysign(math.log10(abs(b) + 1e-30), b)
                acc += abs(la - lb); n += 1
            d_hu = (acc / max(n, 1)) / 3.0           # ~3 decades -> 1.0
        return float(min(1.0, 0.5 * d_elong + 0.5 * d_hu))

    # ── Direction consistency cost ─────────────────────────────────────────
    @staticmethod
    def _direction_consistency_cost(blob: TrackedBlob,
                                    det:  PersistentBlob) -> float:
        """
        Flat penalty for direction-inconsistent ambiguous matches.

        With kinematic direction, a candidate detection's direction label
        is always TURNING (the detector no longer assigns rising/falling),
        so this cost is asymmetric: only the existing track has meaningful
        direction. We compare the track's direction against TURNING — i.e.
        we never penalise the freshly-detected side. In practice this means
        the function returns 0 in current usage and is kept as a hook for
        future per-detection direction estimates (e.g. mask-flow direction).
        """
        if blob.direction != TURNING and det.direction != TURNING \
                and blob.direction != det.direction:
            return 0.2
        return 0.0

    # ── Kinematic direction classifier (Kalman vy → string label) ─────────
    @staticmethod
    def _classify_direction_from_vy(vy: float, threshold: float) -> str:
        """
        Map a Kalman vy estimate (pixels/frame, image-y convention) to a
        rising/falling/turning string label.

        OpenCV image y increases downward, so a blob physically rising in
        the lamp has vy < 0. The threshold is symmetric: if |vy| < threshold
        the blob is classified as TURNING (essentially stationary), which
        absorbs both turning-point states and Kalman noise on stationary
        wax. Typical lamp blob speeds are 0.5–2 px/frame at 30 fps; a
        threshold of 0.3 keeps noisy near-zero estimates out of the
        rising/falling categories.
        """
        if vy < -threshold:
            return RISING
        if vy >  threshold:
            return FALLING
        return TURNING

    # ── Kinematic state refresh ───────────────────────────────────────────
    def _update_physical_fields(self) -> None:
        """Refresh per-blob kinematic state from each track's Kalman filter.

        Reads vx, vy (px/frame) and the buoyancy acceleration term ay from the
        filter's post-state, derives speed = |v|, and assigns the
        rising/falling/turning label. These feed the routing layer so a voice
        reflects its blob's motion, not just its position. d_area /
        d_persistence are set separately on detection cycles and are left
        untouched here so they persist across predict-only frames.
        """
        for track in self._tracks.values():
            blob = track["blob"]
            if not blob.alive:
                continue
            sp = track["kf"].statePost
            vx = float(sp[2, 0])
            vy = float(sp[3, 0])
            ay = float(sp[4, 0]) if sp.shape[0] > 4 else 0.0
            blob.vx        = vx
            blob.vy        = vy
            blob.ay        = ay
            blob.speed     = float(np.hypot(vx, vy))
            blob._vy_hint  = vy      # back-compat for routing.Source.VY
            blob.direction = self._classify_direction_from_vy(
                vy, self.direction_vy_threshold,
            )

    # ── Main update ────────────────────────────────────────────────────────
    def update(self, detections, frame: np.ndarray,
               grey: Optional[np.ndarray] = None) -> list:
        """
        Process one frame. Returns the list of alive TrackedBlob objects.

        Two modes (v4.1 detection/frame-rate decoupling, #2):

          • PREDICT-ONLY  (detections is None) — runs on the K-1 frames
            between detection cycles. Computes optical flow, advects each
            track's mask, runs the Kalman *predict* step for every track, and
            refreshes kinematic state (vx/vy/speed/ay/direction). No
            association, births, deaths, merges, splits, or coast accounting.

          • DETECTION CYCLE (detections is a list) — does all of the above,
            then associates detections to tracks and runs correction, merge /
            split detection, coasting / retirement, re-identification, and
            spawning, and computes per-blob d_area / d_persistence rates.

        `grey` is the precomputed greyscale of `frame`, shared with the
        detector and merge tree; it is recomputed only if not supplied.

        NOTE: coast_frames now counts missed DETECTION CYCLES (not raw
        frames). The caller should scale max_coast by K to preserve
        wall-clock coast time.
        """
        self.events = []
        curr_grey   = grey if grey is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Adapt internal dimensions to the frame we were actually handed. When
        # the runner crops to the lamp ROI (the intended path) this matches
        # profile.roi_h/roi_w and nothing changes; if a full frame is passed
        # instead, masks, flow, and remap grids all stay mutually consistent
        # rather than raising a broadcast error. (Coordinates are assumed
        # ROI-local for y_norm-based re-identification, so feeding a full frame
        # works but is geometrically off — crop for correct behaviour.)
        fh, fw = curr_grey.shape[:2]
        if (fh, fw) != (self.frame_h, self.frame_w):
            self.frame_h, self.frame_w = fh, fw
            self._ensure_remap_grids(fh, fw)

        # ── A: flow + advection + Kalman PREDICT for every track ──────────
        # Runs on every frame so each track's centroid and velocity estimate
        # stay current between detection cycles. Calling predict() for ALL
        # tracks here (then correct() only for matched ones below) also gives
        # a proper predict-correct cycle — in v4.0 matched tracks were never
        # predicted, only corrected.
        flow = self._compute_flow(curr_grey)
        if self._tracks:
            for tid, track in self._tracks.items():
                if flow is not None:
                    track["mask"] = self._advect_mask(track["mask"], flow)
                pred = track["kf"].predict()
                track["blob"].centroid = (float(pred[0][0]), float(pred[1][0]))
        self.prev_grey = curr_grey
        self._flow     = flow

        # Kinematic fields from the freshly-predicted filter state.
        self._update_physical_fields()
        self.frame_count += 1

        # PREDICT-ONLY frame: stop here.
        if detections is None:
            return [v["blob"] for v in self._tracks.values() if v["blob"].alive]

        # ── DETECTION CYCLE from here on ──────────────────────────────────
        self.detect_count += 1

        # ── B: Build detection masks ───────────────────────────────────
        det_masks = [self._det_mask(d) for d in detections]
        # Precompute downsampled masks once per side for the cheap IoU matrix.
        det_small = [self._small_mask(m) for m in det_masks]

        track_ids = list(self._tracks.keys())
        track_small = {tid: self._small_mask(self._tracks[tid]["mask"])
                       for tid in track_ids}

        # ── C: IoU-based primary matching ─────────────────────────────
        # For each detection find which advected masks overlap it.
        matches:         List[Tuple[int,int]]  = []   # (tid, det_idx)
        claimed_tracks:  set                   = set()
        claimed_dets:    set                   = set()

        # Build full IoU matrix for resolving ambiguity
        if track_ids and detections:
            iou_mat = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
            for i, tid in enumerate(track_ids):
                ts = track_small[tid]
                for j in range(len(detections)):
                    iou_mat[i, j] = self._iou_small(ts, det_small[j])

            # Greedy best-match per detection
            for j in range(len(detections)):
                col = iou_mat[:, j]
                # All tracks that overlap this detection above threshold
                candidates = [
                    i for i in range(len(track_ids))
                    if col[i] >= self.iou_threshold
                    and track_ids[i] not in claimed_tracks
                ]
                if not candidates:
                    continue

                if len(candidates) == 1:
                    # Unambiguous
                    best_i = candidates[0]
                else:
                    # ── D: Resolve ambiguity with topological + positional
                    # cost. Hue is no longer used here (was 0.10 weight);
                    # the freed weight is redistributed: topo 0.75 → 0.80
                    # and position 0.15 → 0.20. Direction-inconsistency
                    # contributes a small flat penalty via
                    # _direction_consistency_cost (zero in current usage
                    # because det.direction is always TURNING — kept as a
                    # hook for future per-detection direction estimates).
                    best_i    = None
                    best_cost = float("inf")
                    det        = detections[j]
                    for i in candidates:
                        tid  = track_ids[i]
                        blob = self._tracks[tid]["blob"]
                        # v4.6 REBALANCE. This was 80% topological / 20%
                        # positional, which inverted the reliability of the two
                        # cues: at 0.5-2 px/frame a correct match sits within a
                        # few pixels of the Kalman prediction while a wrong one
                        # is tens of pixels away, whereas the topological birth
                        # value can jump 20+ grey levels frame to frame purely
                        # from the peak shifting inside a deforming blob.
                        # Position now leads, area and the previously-unused
                        # shape descriptors are independent cues, and topology
                        # contributes rather than decides.
                        td = self._topo_dist(blob, det)      # in track sigmas
                        if td > self.TOPO_SIGMAS:
                            continue
                        pos_d = np.hypot(
                            det.centroid[0] - blob.centroid[0],
                            det.centroid[1] - blob.centroid[1],
                        )
                        area_d  = abs(det.area - blob.area) / max(blob.area, 1.0)
                        shape_d = self._shape_dist(blob, det)
                        dc      = self._direction_consistency_cost(blob, det)
                        cost = (0.60 * min(1.0, pos_d / 60.0) +
                                0.25 * min(1.0, area_d) +
                                0.15 * min(1.0, td / self.TOPO_SIGMAS) +
                                0.10 * shape_d +
                                dc)
                        if cost < best_cost:
                            best_cost = cost
                            best_i    = i

                if best_i is not None:
                    matches.append((track_ids[best_i], j))
                    claimed_tracks.add(track_ids[best_i])
                    claimed_dets.add(j)

        # ── E/F: Update matched tracks ─────────────────────────────────
        for tid, j in matches:
            det    = detections[j]
            track  = self._tracks[tid]
            blob   = track["blob"]

            # Refine mask from detection contour periodically
            if self.detect_count % self.REFINE_INTERVAL == 0:
                refined = det_masks[j]
                # Blend: advected mask (anchor) + detection (boundary)
                blended = cv2.addWeighted(
                    track["mask"].astype(np.float32), 0.5,
                    (refined > 0).astype(np.float32),   0.5, 0
                )
                track["mask"] = (blended > 0.4).astype(np.uint8)

            # Physics Kalman correction
            kf   = track["kf"]
            y_px = float(det.centroid[1])
            # y is ROI-local now, so the lamp spans [0, roi_h].
            _adapt_measurement_noise(
                kf, y_px,
                0.0, float(self.profile.roi_h),
            )
            meas = np.array([[det.centroid[0]], [y_px]], dtype=np.float32)
            kf.correct(meas)

            # ── Rate of change since this track's previous detection ──────
            # Fractional change per second; held constant on predict-only
            # frames (area/persistence only refresh on detection cycles).
            now    = time.time()
            dt     = now - track.get("rate_t", now)
            if dt > 1e-3:
                prev_a = track.get("rate_area", det.area)
                prev_p = track.get("rate_pers", det.persistence)
                blob.d_area        = ((det.area - prev_a) / max(prev_a, 1.0)) / dt
                blob.d_persistence = ((det.persistence - prev_p) / max(prev_p, 1.0)) / dt
            track["rate_t"]    = now
            track["rate_area"] = det.area
            track["rate_pers"] = det.persistence

            # Update TrackedBlob attributes
            blob.centroid    = det.centroid
            blob.area        = det.area
            # v4.6: fold into the smoothed reference + variability estimate
            # BEFORE overwriting the raw values, so the comparison next frame
            # uses a stable reference rather than this frame's measurement.
            self._update_topo_stats(blob, det)
            blob.birth       = det.birth
            blob.death       = det.death
            blob.persistence = det.persistence
            blob.orientation = det.orientation
            blob.aspect_ratio= det.aspect_ratio
            blob.elongation  = det.elongation
            blob.hu          = det.hu
            blob.mean_hue    = det.mean_hue
            # blob.direction is updated kinematically from Kalman vy at the
            # end of update() — not copied from det.direction (which is
            # always TURNING after the v3.2 hue→direction refactor).
            blob.age        += 1
            blob.coast_frames = 0

        # ── Merge detection ────────────────────────────────────────────
        # Two unmatched tracks + one unmatched detection covering their
        # combined area → one blob has absorbed another.
        # The higher-persistence track is the survivor; the other is retired
        # with a ('merge', merged_id, survivor_id) event so the voice manager
        # can glide the merged voice to the survivor's pitch before releasing it.
        current_unmatched_tracks = [t for t in track_ids if t not in claimed_tracks]
        current_unmatched_dets   = [j for j in range(len(detections))
                                    if j not in claimed_dets]

        for j in list(current_unmatched_dets):
            if j in claimed_dets:
                continue
            det = detections[j]
            best_pair  = None
            best_score = float("inf")

            ut = [t for t in current_unmatched_tracks if t not in claimed_tracks]
            for ti in range(len(ut)):
                for tk in range(ti + 1, len(ut)):
                    bid_a  = ut[ti]
                    bid_b  = ut[tk]
                    blob_a = self._tracks[bid_a]["blob"]
                    blob_b = self._tracks[bid_b]["blob"]

                    # Area: detection ≈ sum of both blobs
                    expected = blob_a.area + blob_b.area
                    area_err = abs(det.area - expected) / max(expected, 1.0)
                    if area_err > 0.5:
                        continue

                    # Position: detection centroid near midpoint
                    mid_x = (blob_a.centroid[0] + blob_b.centroid[0]) / 2
                    mid_y = (blob_a.centroid[1] + blob_b.centroid[1]) / 2
                    pos_err = np.hypot(det.centroid[0] - mid_x,
                                       det.centroid[1] - mid_y)
                    if pos_err > 150:
                        continue

                    # Mask: detection overlaps union of both advected masks
                    union_mask = np.maximum(self._tracks[bid_a]["mask"],
                                            self._tracks[bid_b]["mask"])
                    if self._iou(union_mask, det_masks[j]) < 0.15:
                        continue

                    score = area_err + pos_err / 150.0
                    if score < best_score:
                        best_score = score
                        best_pair  = (bid_a, bid_b)

            if best_pair is not None:
                bid_a, bid_b = best_pair
                blob_a = self._tracks[bid_a]["blob"]
                blob_b = self._tracks[bid_b]["blob"]

                # Survivor = higher persistence; merged = lower
                if blob_a.persistence >= blob_b.persistence:
                    survivor_id, merged_id = bid_a, bid_b
                else:
                    survivor_id, merged_id = bid_b, bid_a

                # Update survivor track with the merged detection
                b = self._tracks[survivor_id]["blob"]
                b.centroid     = det.centroid
                b.area         = det.area
                self._update_topo_stats(b, det)
                b.birth        = det.birth
                b.death        = det.death
                b.persistence  = det.persistence
                b.orientation  = det.orientation
                b.aspect_ratio = det.aspect_ratio
                b.elongation   = det.elongation
                b.hu           = det.hu
                b.mean_hue     = det.mean_hue
                # b.direction set kinematically at end of update()
                b.coast_frames = 0
                b.age         += 1
                self._tracks[survivor_id]["mask"] = det_masks[j].copy()
                kf   = self._tracks[survivor_id]["kf"]
                meas = np.array([[det.centroid[0]], [det.centroid[1]]],
                                dtype=np.float32)
                kf.correct(meas)

                # Retire the merged track silently (voice manager handles fade)
                self._tracks[merged_id]["blob"].alive = False

                # Fire merge event
                self.events.append(("merge", merged_id, survivor_id))

                claimed_tracks.add(survivor_id)
                claimed_tracks.add(merged_id)
                claimed_dets.add(j)
                current_unmatched_tracks = [t for t in current_unmatched_tracks
                                            if t not in (bid_a, bid_b)]

        # ── G: Handle unmatched tracks (coast or retire) ───────────────
        unmatched_tracks = [
            tid for tid in track_ids if tid not in claimed_tracks
        ]
        for tid in unmatched_tracks:
            track = self._tracks[tid]
            blob  = track["blob"]
            blob.coast_frames += 1
            # (Kalman predict already ran for every track in phase A; the
            # coasting centroid is therefore already the predicted position.)

            if blob.coast_frames > self.max_coast:
                blob.alive = False
                y_norm = self.profile.y_norm(blob.centroid[1])
                self.reident.record_loss(blob, y_norm)
                self.events.append(("death", tid))

        # Remove dead tracks (normal deaths + merged)
        dead = [k for k, v in self._tracks.items() if not v["blob"].alive]
        for k in dead:
            del self._tracks[k]

        # ── H: Handle unmatched detections ────────────────────────────
        unmatched_dets = [j for j in range(len(detections))
                          if j not in claimed_dets]

        # ── Split detection ────────────────────────────────────────────
        # One track just went to coast_frames=1 (first frame it was undetected)
        # + two unmatched detections whose combined area matches the parent's area
        # and both sit near the parent's last centroid.
        # Parent track continues under its ID with the larger fragment.
        # Child gets a new ID with a ('split', parent_id, child_id) event so
        # the voice manager can initialise the child voice from the parent's state.
        recently_coasting = [
            tid for tid in self._tracks
            if self._tracks[tid]["blob"].coast_frames == 1
        ]

        for parent_id in recently_coasting:
            if parent_id not in self._tracks:
                continue
            parent_blob = self._tracks[parent_id]["blob"]
            parent_mask = self._tracks[parent_id]["mask"]

            best_pair  = None
            best_score = float("inf")
            avail = [j for j in unmatched_dets if j not in claimed_dets]

            for di in range(len(avail)):
                for dk in range(di + 1, len(avail)):
                    j1   = avail[di]
                    j2   = avail[dk]
                    det1 = detections[j1]
                    det2 = detections[j2]

                    # Area: det1 + det2 ≈ parent area
                    combined = det1.area + det2.area
                    area_err = abs(combined - parent_blob.area) / max(parent_blob.area, 1.0)
                    if area_err > 0.5:
                        continue

                    # Both near parent centroid
                    d1 = np.hypot(det1.centroid[0] - parent_blob.centroid[0],
                                  det1.centroid[1] - parent_blob.centroid[1])
                    d2 = np.hypot(det2.centroid[0] - parent_blob.centroid[0],
                                  det2.centroid[1] - parent_blob.centroid[1])
                    if d1 > 150 or d2 > 150:
                        continue

                    # Both overlap parent mask
                    if (self._iou(parent_mask, det_masks[j1]) < 0.1 or
                            self._iou(parent_mask, det_masks[j2]) < 0.1):
                        continue

                    score = area_err + (d1 + d2) / 300.0
                    if score < best_score:
                        best_score = score
                        best_pair  = (j1, j2)

            if best_pair is not None:
                j1, j2   = best_pair
                det1     = detections[j1]
                det2     = detections[j2]

                # Larger fragment → parent track; smaller → child
                if det1.area >= det2.area:
                    parent_det, child_det     = det1, det2
                    parent_det_j, child_det_j = j1, j2
                else:
                    parent_det, child_det     = det2, det1
                    parent_det_j, child_det_j = j2, j1

                # Re-attach parent track to its larger fragment
                b = self._tracks[parent_id]["blob"]
                b.centroid     = parent_det.centroid
                b.area         = parent_det.area
                b.birth        = parent_det.birth
                b.death        = parent_det.death
                b.persistence  = parent_det.persistence
                b.orientation  = parent_det.orientation
                b.aspect_ratio = parent_det.aspect_ratio
                b.elongation   = parent_det.elongation
                b.hu           = parent_det.hu
                b.mean_hue     = parent_det.mean_hue
                # b.direction set kinematically at end of update()
                b.coast_frames = 0
                b.age         += 1
                self._tracks[parent_id]["mask"] = det_masks[parent_det_j].copy()
                kf   = self._tracks[parent_id]["kf"]
                meas = np.array([[parent_det.centroid[0]],
                                 [parent_det.centroid[1]]], dtype=np.float32)
                kf.correct(meas)

                # Spawn child under a new ID; suppress the plain birth event
                # _spawn() appends ('birth', child_id) — replace with split event
                child_id = self.next_id   # peek at what _spawn will use
                self._spawn(child_det, det_masks[child_det_j])
                # Replace the birth event with the split event
                self.events = [e for e in self.events
                               if not (e[0] == "birth" and e[1] == child_id)]
                self.events.append(("split", parent_id, child_id))

                claimed_dets.add(j1)
                claimed_dets.add(j2)
                unmatched_dets = [j for j in unmatched_dets
                                  if j not in (j1, j2)]

        # Process remaining unmatched detections: reidentify or spawn normally
        for j in unmatched_dets:
            det    = detections[j]
            y_norm = self.profile.y_norm(det.centroid[1])

            # Check fingerprint buffer for re-identification
            recovered_id = self.reident.try_reidentify(det, y_norm)

            if recovered_id is not None:
                # Restore original ID with a new Kalman filter
                self._spawn(det, det_masks[j], blob_id=recovered_id)
                # Don't fire a birth event for re-identified blobs
            else:
                self._spawn(det, det_masks[j])

        # ── I: Refresh kinematic state after corrections ──────────────────
        # Matched tracks were just corrected, so re-read vx / vy / speed / ay
        # and the rising/falling/turning label from the updated filter state.
        # (frame_count was already incremented in phase A.)
        self._update_physical_fields()
        return [v["blob"] for v in self._tracks.values() if v["blob"].alive]

    # ── Spawn ──────────────────────────────────────────────────────────────
    def _spawn(
        self,
        det:     PersistentBlob,
        mask:    np.ndarray,
        blob_id: Optional[int] = None,
    ) -> None:
        new_id   = blob_id if blob_id is not None else self.next_id
        if blob_id is None:
            self.events.append(("birth", new_id))
            self.next_id += 1
        elif new_id >= self.next_id:
            self.next_id = new_id + 1

        # The buoyancy Kalman prior was previously seeded from the
        # detection's hue-derived direction. With kinematic direction the
        # detection always reports TURNING, so the spawn-time prior is
        # zero; the filter accumulates vy from successive measurements
        # and the direction label catches up after a few frames.
        kf = _make_buoyancy_kalman(
            det.centroid[0], det.centroid[1],
            TURNING,
            self.profile.lamp_height,
        )
        blob = TrackedBlob(
            id           = new_id,
            centroid     = det.centroid,
            area         = det.area,
            birth        = det.birth,
            death        = det.death,
            persistence  = det.persistence,
            orientation  = det.orientation,
            aspect_ratio = det.aspect_ratio,
            elongation   = det.elongation,
            hu           = det.hu,
            mean_hue     = det.mean_hue,
            direction    = TURNING,
        )
        self._tracks[new_id] = {
            "blob": blob, "mask": mask.copy(), "kf": kf,
            # Baseline for d_area / d_persistence rate computation.
            "rate_t":    time.time(),
            "rate_area": det.area,
            "rate_pers": det.persistence,
        }

    # ── Convenience ────────────────────────────────────────────────────────
    def get_active(self) -> list:
        return [v["blob"] for v in self._tracks.values() if v["blob"].alive]

    def get_flow(self) -> Optional[np.ndarray]:
        """Return the most recent optical flow field (for debug overlay)."""
        return self._flow
