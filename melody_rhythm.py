# melody_rhythm.py -- Audible Lava / DSO: per-voice rhythm + melody layer
#
# Additive module. Nothing in blob_detection / blob_tracker / voice_manager /
# binding_manager / routing is modified or monkey-patched. The main loop
# calls MelodyEngine.process_frame(tracked) once per frame, after
# voices.process_frame(tracked), and the engine spawns short one-shot
# \melodyHit synths in SuperCollider on top of the sustained \mpeVoice drones.
#
# Two mechanisms (the "minimal core set", items 1 and 2):
#
#   v4.9.1: baseline rate raised (h0 0.08 -> 0.25, one note per ~4 s for a
#   still blob) and note durations quantised to whole beats of the governor's
#   elastic clock when it is running (see beats_for_area).
#
#   1. HAZARD-DRIVEN ONSETS  (HazardOnsets)
#      Each blob owns an accumulator A. Every frame
#          A += h * dt,   h = clip(h0 * (1 + a*speed_n + b*accel_n), 0, h_max)
#      and when A >= 1 a note fires and A resets to a small random residual.
#      h is the instantaneous event rate (1/s): a still blob plays about one
#      note every 1/h0 seconds (Poisson-like, quiet); a fast or turning blob
#      plays up to h_max notes/s. Because A integrates, a burst of motion
#      produces a cluster of notes rather than a single trigger, and a slow
#      blob still eventually speaks. A refractory floor prevents machine-gun
#      retriggering when h_max is reached.
#
#   2. HEIGHT-BAND PITCH WITH HYSTERESIS + CONTOUR INERTIA  (BandPitch)
#      The lamp is divided into n_bands = len(scale) * octave_range horizontal
#      bands (15 for the pentatonic / 3-octave default). Each band maps to one
#      scale degree. A blob's band only changes when its centroid leaves the
#      current band by more than `hysteresis` band-heights, so jitter at a
#      boundary cannot flicker the pitch. The band is an ATTRACTOR, not the
#      note itself: the note that actually plays walks toward it in steps
#      whose size depends on speed (slow -> stepwise, fast -> thirds/leaps)
#      and whose direction follows sign(vy) only after that sign has been
#      held for `dwell_s` seconds (contour inertia). A direction reversal is a
#      turning point and forces resolution onto the band.
#
# Conventions used here (verified against the v4.1 files in the Project):
#   - profile.y_norm(roi_y) returns 1 = top of lamp, 0 = bottom.
#   - blob.vy is Kalman vertical velocity in px/frame; vy < 0 means rising.
#   - speed_n / accel_n / area_n / x_norm are read through
#     routing.extract_source_value so normalisation matches the routing layer.
#
# Tests: test_melody_rhythm.py (no SuperCollider, camera, or pygame).

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from routing import Source, extract_source_value

# Mirror of voice_manager's musical constants. Imported lazily below so this
# module (and its tests) can load without pythonosc / OpenCV present.
_DEFAULT_SCALE        = [0, 2, 4, 7, 9]   # major pentatonic
_DEFAULT_ROOT         = 48                # C3
_DEFAULT_OCTAVE_RANGE = 3


def midi_to_freq(n: float) -> float:
    return 440.0 * (2.0 ** ((n - 69) / 12.0))


# ── 1. Hazard-driven onsets ──────────────────────────────────────────────────
@dataclass
class HazardParams:
    h0:           float = 0.25   # baseline rate, notes/s, for a still blob (~4 s apart)
                                 # (v4.9.1: was 0.08 = one note per 12.5 s, too sparse)
    a_speed:      float = 4.0    # h multiplier at speed_n = 1  (speed_n = speed / VEL_SCALE)
    b_accel:      float = 3.0    # h multiplier at accel_n = 1  (accel_n = |ay| / ACCEL_SCALE)
    h_max:        float = 1.0    # ceiling, notes/s
    refractory_s: float = 0.25   # hard minimum spacing between a blob's notes
    reset_jitter: float = 0.30   # residual after a fire ~ U(0, reset_jitter)
    warmup_s:     float = 0.5    # no firing in a blob's first half second (Kalman settles)


@dataclass
class _HazardState:
    acc:        float = 0.0
    last_fire:  float = -1e9
    born:       float = 0.0
    n_fired:    int   = 0
    h:          float = 0.0      # last computed rate (for HUD / diagnostics)


class HazardOnsets:
    """Per-blob accumulator-based event generator. Pure Python, deterministic
    given `seed`."""

    def __init__(self, params: Optional[HazardParams] = None, seed: Optional[int] = None):
        self.p = params or HazardParams()
        self._rng = random.Random(seed)
        self._state: Dict[int, _HazardState] = {}

    # Rate function — exposed so tests and the HUD can inspect it.
    def rate(self, speed_n: float, accel_n: float) -> float:
        h = self.p.h0 * (1.0 + self.p.a_speed * speed_n + self.p.b_accel * accel_n)
        return max(0.0, min(h, self.p.h_max))

    def expected_interval(self, speed_n: float, accel_n: float) -> float:
        h = self.rate(speed_n, accel_n)
        return float("inf") if h <= 0 else 1.0 / h

    def step(self, blob_id: int, speed_n: float, accel_n: float,
             dt: float, now: float) -> bool:
        """Advance one blob by dt seconds. Returns True when an onset fires."""
        st = self._state.get(blob_id)
        if st is None:
            st = _HazardState(acc=self._rng.uniform(0.0, 0.5), born=now)
            self._state[blob_id] = st
        st.h = self.rate(speed_n, accel_n)
        st.acc += st.h * max(dt, 0.0)
        if st.acc < 1.0:
            return False
        if (now - st.born) < self.p.warmup_s:
            st.acc = 1.0            # hold at threshold; fire as soon as warm-up ends
            return False
        if (now - st.last_fire) < self.p.refractory_s:
            st.acc = 1.0
            return False
        st.acc = self._rng.uniform(0.0, self.p.reset_jitter)
        st.last_fire = now
        st.n_fired += 1
        return True

    def state(self, blob_id: int) -> Optional[_HazardState]:
        return self._state.get(blob_id)

    def forget(self, blob_id: int) -> None:
        self._state.pop(blob_id, None)

    def prune(self, alive_ids) -> None:
        alive = set(alive_ids)
        for bid in list(self._state):
            if bid not in alive:
                self._state.pop(bid, None)


# ── 2. Height-band pitch with hysteresis + contour inertia ───────────────────
@dataclass
class PitchParams:
    scale:         List[int] = field(default_factory=lambda: list(_DEFAULT_SCALE))
    root:          int   = _DEFAULT_ROOT
    octave_range:  int   = _DEFAULT_OCTAVE_RANGE
    hysteresis:    float = 0.25   # band-heights the centroid must overshoot before re-banding
    vy_dead:       float = 0.15   # px/frame; |vy| below this is "no direction"
    dwell_s:       float = 0.8    # a new sign(vy) must persist this long to flip contour
    speed_step1:   float = 0.25   # speed_n below -> step of 1 degree
    speed_step2:   float = 0.60   # speed_n below -> step of 2 degrees, else 3
    slack:         int   = 2      # degrees the walking note may wander from its band
    track_band_on_fire: bool = True  # on each fire, if far from band, pull toward it

    @property
    def n_bands(self) -> int:
        return len(self.scale) * self.octave_range


@dataclass
class _PitchState:
    band:        int   = 0       # current attractor band (hysteretic)
    deg:         int   = 0       # current scale-degree index actually sounding
    direction:   int   = 0       # contour: -1 down, 0 none, +1 up
    cand_dir:    int   = 0       # candidate direction waiting out the dwell
    cand_since:  float = 0.0
    initialised: bool  = False
    last_turn:   bool  = False   # latched on a contour reversal; cleared by next_degree


class BandPitch:
    """Maps blob height to a hysteretic scale-degree attractor and walks a
    melodic line toward it with speed-dependent intervals and contour
    inertia."""

    def __init__(self, params: Optional[PitchParams] = None):
        self.p = params or PitchParams()
        self._state: Dict[int, _PitchState] = {}

    # ── helpers ──────────────────────────────────────────────────────────
    def degree_to_midi(self, deg: int) -> int:
        n = len(self.p.scale)
        deg = max(0, min(deg, self.p.n_bands - 1))
        return self.p.root + 12 * (deg // n) + self.p.scale[deg % n]

    def raw_band(self, height: float) -> int:
        """Band index without hysteresis (height: 0 = bottom, 1 = top)."""
        h = max(0.0, min(height, 1.0 - 1e-9))
        return int(h * self.p.n_bands)

    def _hysteretic_band(self, st: _PitchState, height: float) -> int:
        nb  = self.p.n_bands
        pos = max(0.0, min(height, 1.0)) * nb       # position in band units
        lo  = st.band - self.p.hysteresis
        hi  = st.band + 1.0 + self.p.hysteresis
        if lo <= pos < hi:
            return st.band
        return max(0, min(int(pos), nb - 1))

    def step_size(self, speed_n: float) -> int:
        if speed_n < self.p.speed_step1:
            return 1
        if speed_n < self.p.speed_step2:
            return 2
        return 3

    # ── per-frame update (cheap) ─────────────────────────────────────────
    def update(self, blob_id: int, height: float, vy: float, now: float) -> _PitchState:
        """Refresh band (hysteretic) and contour direction (dwell-filtered).
        vy: px/frame, negative = rising. height: 0 = bottom, 1 = top."""
        st = self._state.get(blob_id)
        if st is None:
            st = _PitchState()
            st.band = self.raw_band(height)
            st.deg  = st.band
            st.initialised = True
            st.cand_since  = now
            self._state[blob_id] = st
        else:
            st.band = self._hysteretic_band(st, height)

        # Contour direction: image y grows downward, so rising = vy < 0 = +1.
        if vy < -self.p.vy_dead:
            new_dir = +1
        elif vy > self.p.vy_dead:
            new_dir = -1
        else:
            new_dir = 0

        # last_turn is latched here and consumed by next_degree(), so a
        # turning point is honoured by the next note however many frames
        # elapse before the hazard fires.
        if new_dir != st.cand_dir:
            st.cand_dir   = new_dir
            st.cand_since = now
        elif new_dir != st.direction and (now - st.cand_since) >= self.p.dwell_s:
            # Reversal (not merely stopping) is a turning point.
            if st.direction != 0 and new_dir != 0 and new_dir == -st.direction:
                st.last_turn = True
            st.direction = new_dir
        return st

    # ── on fire: choose the next note ────────────────────────────────────
    def next_degree(self, blob_id: int, speed_n: float) -> int:
        st = self._state[blob_id]
        nb = self.p.n_bands
        k  = self.step_size(speed_n)
        band, deg = st.band, st.deg

        if st.last_turn:
            deg = band                               # turning point -> resolve
            st.last_turn = False
        elif abs(deg - band) > self.p.slack:
            deg += k if band > deg else -k           # far from attractor -> pull in
            if (band - deg) * (band - st.deg) < 0:   # don't overshoot
                deg = band
        elif st.direction != 0:
            cand = deg + st.direction * k
            if abs(cand - band) <= self.p.slack:
                deg = cand                           # follow contour within slack
            else:
                deg = band                           # ran out of slack -> resolve
        elif deg != band and self.p.track_band_on_fire:
            deg += 1 if band > deg else -1           # hovering: drift stepwise home

        st.deg = max(0, min(deg, nb - 1))
        return st.deg

    def next_midi(self, blob_id: int, speed_n: float) -> int:
        return self.degree_to_midi(self.next_degree(blob_id, speed_n))

    def state(self, blob_id: int) -> Optional[_PitchState]:
        return self._state.get(blob_id)

    def forget(self, blob_id: int) -> None:
        self._state.pop(blob_id, None)

    def prune(self, alive_ids) -> None:
        alive = set(alive_ids)
        for bid in list(self._state):
            if bid not in alive:
                self._state.pop(bid, None)


# ── 3. Engine: ties the two together and talks to SuperCollider ──────────────
@dataclass
class NoteEvent:
    blob_id:  int
    slot_id:  Optional[int]
    midi:     int
    freq:     float
    amp:      float
    pan:      float
    dur:      float
    rel:      float
    t:        float
    beats:    float = 0.0        # metrical length in beats (0 = seconds-based fallback)


# ── Duration in beats (v4.9.1) ───────────────────────────────────────────────
# A note's hold time is a whole number of beats of the governor's elastic
# clock (score_state.period_s), chosen from blob area: small blobs play
# short values, large blobs long ones. With the governor off (no
# score_state) the engine falls back to the old area-in-seconds rule.
BEAT_VALUES = (0.5, 1.0, 2.0, 4.0)


def beats_for_area(area_n: float) -> float:
    if area_n < 0.12:
        return BEAT_VALUES[0]
    if area_n < 0.35:
        return BEAT_VALUES[1]
    if area_n < 0.65:
        return BEAT_VALUES[2]
    return BEAT_VALUES[3]


class MelodyEngine:
    """
    Call process_frame(tracked) once per frame AFTER voices.process_frame().

    Reads (never writes) VoiceManager state: slot_for_blob, voice(slot).synth
    for timbre, and voices.osc for sending. Only blobs currently bound to a
    non-IDLE, non-ZOMBIE slot play notes, so the melody layer inherits the
    binding policy (CREATE_EMPTY / TIE / DETACH) unchanged.

    `send` can be overridden (tests pass a recorder) — default is
    voices.osc.send_message.
    """

    def __init__(self, voices, profile, *,
                 synthdef: str = "melodyHit",
                 hazard: Optional[HazardParams] = None,
                 pitch:  Optional[PitchParams]  = None,
                 enabled: bool = True,
                 amp_scale: float = 0.9,
                 amp_floor: float = 0.06,
                 send: Optional[Callable[[str, list], None]] = None,
                 seed: Optional[int] = None,
                 clock: Callable[[], float] = time.time,
                 beat_period: Optional[Callable[[], Optional[float]]] = None,
                 min_dur: float = 0.15, max_dur: float = 8.0):
        self.voices   = voices
        self.profile  = profile
        self.synthdef = synthdef
        self.enabled  = enabled
        self.amp_scale = amp_scale
        self.amp_floor = amp_floor
        self.hazard   = HazardOnsets(hazard, seed=seed)
        self.pitch    = BandPitch(pitch)
        self._clock   = clock
        self._last_t: Optional[float] = None
        self._send    = send or (lambda addr, args: voices.osc.send_message(addr, args))
        # Beat period source. Default reads voices.score_state.period_s, which
        # audible_lava refreshes every frame from the governor; returns None
        # when the governor is off, which selects the seconds-based fallback.
        self._beat_period = beat_period or self._default_beat_period
        self.min_dur, self.max_dur = min_dur, max_dur
        self.last_events: List[NoteEvent] = []   # notes fired on the most recent frame
        self.total_notes = 0

    def _default_beat_period(self) -> Optional[float]:
        ss = getattr(self.voices, "score_state", None)
        p  = getattr(ss, "period_s", None) if ss is not None else None
        try:
            p = float(p)
        except (TypeError, ValueError):
            return None
        return p if 0.05 < p < 10.0 else None

    # ── toggles ──────────────────────────────────────────────────────────
    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled

    # ── per-frame ────────────────────────────────────────────────────────
    def process_frame(self, tracked: list) -> List[NoteEvent]:
        now = self._clock()
        dt  = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
        self._last_t = now
        self.last_events = []

        alive = [b.id for b in tracked]
        self.hazard.prune(alive)
        self.pitch.prune(alive)

        if not self.enabled:
            return self.last_events

        for b in tracked:
            slot_id = self.voices.slot_for_blob(b.id)
            if slot_id is None:
                continue
            v = self.voices.voice(slot_id)
            if v is None or v.state.name in ("IDLE", "ZOMBIE"):
                continue

            height  = self.profile.y_norm(b.centroid[1])          # 1 = top
            vy      = float(getattr(b, "vy", 0.0))
            speed_n = extract_source_value(Source.SPEED, b, self.profile)
            accel_n = extract_source_value(Source.ACCEL, b, self.profile)

            self.pitch.update(b.id, height, vy, now)
            if not self.hazard.step(b.id, speed_n, accel_n, dt, now):
                continue

            midi = self.pitch.next_midi(b.id, speed_n)
            ev   = self._fire(b, v, slot_id, midi, now)
            self.last_events.append(ev)
            self.total_notes += 1
        return self.last_events

    # ── note construction ────────────────────────────────────────────────
    def _fire(self, blob, voice, slot_id: int, midi: int, now: float) -> NoteEvent:
        s       = voice.synth
        area_n  = extract_source_value(Source.AREA,   blob, self.profile)
        x_norm  = extract_source_value(Source.X_NORM, blob, self.profile)
        d_area  = float(getattr(blob, "d_area", 0.0))

        freq = midi_to_freq(midi)
        amp  = max(self.amp_floor, float(s.c_amp) * self.amp_scale)
        pan  = x_norm * 2.0 - 1.0
        period = self._beat_period()
        if period is not None:
            beats = beats_for_area(area_n)
            dur   = beats * period                   # hold = whole beats of the elastic clock
        else:
            beats = 0.0
            dur   = 0.25 + 2.5 * math.sqrt(area_n)   # governor off: big blob -> long note
        dur = max(self.min_dur, min(self.max_dur, dur))
        rel  = 1.2 if d_area >= 0.0 else 0.25         # expanding -> legato, shrinking -> staccato
        attack = 0.01 + 0.08 * (1.0 - min(1.0, float(getattr(blob, "speed", 0.0)) / 3.0))

        args = [
            self.synthdef, -1, 0, 0,                 # node id -1: server assigns; one-shot
            "freq",    freq,
            "amp",     amp,
            "pan",     pan,
            "dur",     dur,
            "attack",  attack,
            "release", rel,
            "cutoff",  float(s.c_cutoff),
            "res",     float(s.c_res),
            "fmode",   float(s.c_fmode),
            "wave",    float(s.c_wave_pos),
            "bank",    int(s.wave_bank),
            "detune",  float(s.c_detune),
            "drive",   float(s.c_drive),
        ]
        try:
            self._send("/s_new", args)
        except Exception:
            pass
        return NoteEvent(blob_id=blob.id, slot_id=slot_id, midi=midi, freq=freq,
                         amp=amp, pan=pan, dur=dur, rel=rel, t=now, beats=beats)

    # ── diagnostics for the overlay / HUD ────────────────────────────────
    def hud_line(self) -> str:
        n_tracked = len(self.hazard._state)
        return (f"MELODY {'on ' if self.enabled else 'off'} "
                f"blobs:{n_tracked} notes:{self.total_notes}")

    def blob_info(self, blob_id: int) -> str:
        hs = self.hazard.state(blob_id)
        ps = self.pitch.state(blob_id)
        if hs is None or ps is None:
            return ""
        arrow = {1: "^", -1: "v", 0: "-"}[ps.direction]
        return (f"h={hs.h:.2f}/s A={hs.acc:.2f} band={ps.band} "
                f"deg={ps.deg} {arrow} m{self.pitch.degree_to_midi(ps.deg)}")
