"""Original 1978 arcade Space Invaders, image-only Gymnasium environment.

The machine's BCD score supplies the native reward. Observations stay pixels
only. When Jonny's shaping pack is enabled (shaping_weights), validated
read-only RAM events add shaped reward components, each logged separately.
No RAM is ever written.
Validated map: https://computerarcheology.com/Arcade/SpaceInvaders/RAMUse.html
Driver: https://github.com/finalburnneo/FBNeo/blob/master/src/burn/drv/pre90s/d_invaders.cpp
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
from PIL import Image

from libretro_host import LibretroHost


ROOT = Path(__file__).resolve().parent
ACTION_NAMES = ("noop", "left", "right", "fire", "left_fire", "right_fire")
ACTION_BUTTONS = ((), (6,), (7,), (0,), (6, 0), (7, 0))

# Validated own-barrier telemetry (2026-10-05, seeds 7/42/99, see
# evidence/barrier_telemetry_verified.json and
# evidence/barrier_contact_validated.json): the four barrier rects and
# band, in rotated frame pixels. A shot that hits a barrier registers
# contact (shot_status 2 -> 3) and carves its crater in two deterministic
# phases: a small bite at contact (+0 frames) and the full crater when the
# explosion resolves (+16 frames, register back to 0). Alien shots erode
# from the top, so deep-rows (>=198) + own-shot-column (cannon at fire
# time, +-4 px) + contact-window attribution isolates own damage.
BARRIER_RECTS = ((32, 53), (77, 98), (122, 143), (167, 188))
BARRIER_ROWS = (192, 207)
BARRIER_OWN_CRATER_MIN_Y = 198
BARRIER_CANNON_ROWS = (216, 223)
BARRIER_CONTACT_BEFORE = 2
BARRIER_CONTACT_AFTER = 20
BARRIER_HIT_REFRACTORY = 22  # merges the two crater phases (0 and +16); distinct hits are >=33 frames apart
BARRIER_LOST_MIN_PIXELS = 2
BARRIER_OWN_SHOT_COL_TOLERANCE = 4

# Jonny's shaping pack (2026-10-05 spec, verbatim in WORK_LOG). RAM addresses
# and thresholds are ROM-code constants (disassembly snapshot
# tools/_code_disassembly_cache.html of computerarcheology.com/Arcade/SpaceInvaders)
# calibrated live by tools/calibrate_shaping_telemetry.py, evidence in
# evidence/shaping_telemetry_calibration.json:
# - cannon travel stops 0x30/0xD9 observed (ROM 038E/0381), playerYr 0x20
# - player band 0x1E..0x27 (ROM 0603/0608; 18/18 bomb-player deaths correlated)
# - saucer/top band floor 0xCE (ROM 14F0); miss-explosion threshold 0xD8 (14E4)
# - bomb slots: status bits 0x80 active / 0x01 blowing only; fall 4 px/frame,
#   X constant per flight (zero drift), straight down
# - barrier RAM rects: validated pixel rects translated by the cannon X map
# - barrier top RAM Y 66: max of 85 bomb-contact blow-ups (pixel cross-check 63)
# - wasted shot: the blow-up handler mutates shot Y (-2) in the transition
#   frame, so the discriminator is the shot's Y on its last status-2 frame:
#   >= 0xCE split 88 top-misses from 68 ordinary contacts, zero ambiguous
PLAYER_X_MIN = 0x30
PLAYER_X_MAX = 0xD9
PLAYER_BAND = (0x1E, 0x27)
SAUCER_BAND_FLOOR = 0xCE
BOMB_SLOTS = ((0x35, 0x3D, 0x3E), (0x45, 0x4D, 0x4E), (0x55, 0x5D, 0x5E))
BARRIER_RAM_RECTS = ((56.0, 77.0), (101.0, 122.0), (146.0, 167.0), (191.0, 212.0))
BARRIER_TOP_RAM_Y = 66
# Cannon X map from the same calibration: pixel_col = ram_x + this offset.
RAM_TO_PIXEL_COL = -24
CORNER_MARGIN = 8  # RAM px from a hard travel stop counts as sitting in the corner
CORNER_GRACE_SECONDS = 2.0  # dwell under this pays the base rate (tactical visits stay cheap)
CORNER_ESCALATION = 3.0  # parked longer: rate multiplies (Jonny 2026-10-07: camping crept back)
DODGE_MARGIN_X = 3  # Jonny's 3-pixel safety margin
SHAPING_WEIGHT_KEYS = ("alien_kill", "saucer_kill", "wasted_shot", "cover_per_frame",
                       "dodge", "invasion", "death", "corner_per_second")
# Jonny's shaping pack, v1 (2026-10-05 spec). Preserved for provenance; the
# v3/v4 contracts trained on these weights.
JONNY_SHAPING_PACK_V1 = {"alien_kill": 1.0, "saucer_kill": 15.0, "wasted_shot": 0.2,
                         "cover_per_frame": 0.05, "dodge": 1.0, "invasion": 100.0,
                         "death": 50.0, "corner_per_second": 0.1}
# Jonny's v2 rebalance ruling (2026-10-07, after the review measured the scale):
# cover, saucer bonus and the wasted-shot tax drop to ZERO (native points carry
# them; detection and counting continue so the telemetry stays informative);
# death 50 -> 5, invasion 100 -> 10, alien 1.0 -> 0.1. Preserved for provenance.
JONNY_SHAPING_PACK_V2 = {"alien_kill": 0.1, "saucer_kill": 0.0, "wasted_shot": 0.0,
                         "cover_per_frame": 0.0, "dodge": 1.0, "invasion": 10.0,
                         "death": 5.0, "corner_per_second": 0.1}
# Jonny's v3 ruling (2026-10-07, same morning): dodge payment 1.0 -> 0 too —
# a close dodge is worth nothing extra. Detection and counting still run.
# Everything else holds at the v2 values; native score stays divided by 100.
JONNY_SHAPING_PACK = {"alien_kill": 0.1, "saucer_kill": 0.0, "wasted_shot": 0.0,
                      "cover_per_frame": 0.0, "dodge": 0.0, "invasion": 10.0,
                      "death": 5.0, "corner_per_second": 0.1}
# ALFRED 23 Oct 2026 (the kit's accuracy meter + 'A missed shot costs'): shots, hits and misses are now counted with or
# without the shaping pack. Measured on two ~5,000-point record games (1,200 shots): ~36% flew off the top, ~48% hit an
# invader, the rest ended low - blowing a hole in its own shields (below y 64, over a shield, no bomb near) or shooting
# down a bomb (a bomb within reach, any height). A miss = off the top (not the saucer) or into its own shields.
_NO_WEIGHTS = {key: 0.0 for key in SHAPING_WEIGHT_KEYS}
SHIELD_SHOT_Y_MAX = 64      # RAM y: a shot that blows up below this, over a shield, with no bomb near, hit its own shield
BOMB_NEAR = 14              # |dx| + |dy| in RAM px: a shot ending this close to a bomb shot the bomb down - never a miss


def bcd(byte: int) -> int:
    byte = int(byte)
    if byte & 15 > 9 or byte >> 4 > 9:
        raise RuntimeError(f"Invalid BCD score byte: {byte:02x}")
    return (byte >> 4) * 10 + (byte & 15)


class TornScoreRead(RuntimeError):
    """Read-only artifact: torn BCD score bytes on a carry frame."""

    def __init__(self, previous, current):
        super().__init__(f"Torn score read {previous} -> {current}; discarded, baseline kept")
        self.previous = previous
        self.current = current


def score_delta(previous: int, current: int) -> int:
    """Delta vs the last accepted score. Only a genuine four-digit score
    rollover may create a negative jump; that path is unchanged. Any other
    mid-game decrease is a torn BCD read on a carry frame, never a real score
    change: 10/10 observed blips were decreases of 70-90 at hundred carries
    (280 -> torn read 200 -> true 300), consistent with the low BCD byte being
    written before the high byte; zero up-blips or >600 violations have ever
    been observed. The decrease now raises TornScoreRead so the caller can
    discard the read and keep the baseline; the next clean frame reconciles
    exactly. The >600 implausibility guard still throws, as does any invalid
    BCD byte. Jonny's ruling 2026-10-06: stop the trainer dying on blips;
    fabricated reward is still refused, it is just no longer fatal."""
    delta = current - previous
    if delta < 0:
        if previous >= 9000 and current <= 1000:
            delta += 10000
        else:
            raise TornScoreRead(previous, current)
    if delta > 600:
        raise RuntimeError(f"Implausible per-frame score delta: {delta}")
    return delta


def saucer_component(delta: int) -> int | None:
    """Separate one native saucer award and at most one coincident alien hit.

    A new projectile can be fired during the saucer explosion's score delay.
    This decomposition is used only with a pending native saucer hit.
    """
    candidates = [points for points in (50, 100, 150, 300) if delta - points in (0, 10, 20, 30)]
    return candidates[0] if len(candidates) == 1 else None


class ArcadeInvadersEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, decision_repeat=2, root=None, core_path=None, rom_path=None,
                 render_mode=None, max_episode_frames=0, reset_noop_max=30, audio=False,
                 barrier_penalty_weight=0.0, shaping_weights=None):
        super().__init__()
        self.root = Path(root or ROOT)
        self.decision_repeat = int(decision_repeat)
        if not 1 <= self.decision_repeat <= 8:
            raise ValueError("decision_repeat must be 1..8")
        if not 0.0 <= barrier_penalty_weight <= 1.0:
            raise ValueError("barrier_penalty_weight must be in [0, 1]")
        self.barrier_penalty_weight = float(barrier_penalty_weight)
        if shaping_weights is not None:
            unknown = sorted(set(shaping_weights) - set(SHAPING_WEIGHT_KEYS))
            missing = sorted(set(SHAPING_WEIGHT_KEYS) - set(shaping_weights))
            if unknown or missing:
                raise ValueError(f"shaping_weights must contain exactly {SHAPING_WEIGHT_KEYS}; "
                                 f"unknown={unknown}, missing={missing}")
            if any(not np.isfinite(float(value)) or float(value) < 0 for value in shaping_weights.values()):
                raise ValueError("shaping_weights values must be finite and >= 0; rules apply the signs")
            shaping_weights = {key: float(shaping_weights[key]) for key in SHAPING_WEIGHT_KEYS}
        self.shaping_weights = shaping_weights
        self.max_episode_frames = int(max_episode_frames)
        self.reset_noop_max = int(reset_noop_max)
        self.render_mode = render_mode
        self.action_space = gym.spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = gym.spaces.Box(0, 255, (1, 84, 84), dtype=np.uint8)
        self.host = LibretroHost(core_path or self.root / "vendor" / "fbneo_libretro.dll",
                                 rom_path or self.root / "roms" / "invaders.zip", self.root, audio=audio)
        self.sample_rate = self.host.sample_rate
        self._done = True
        self.raw_frames = 0
        self.score = 0
        self.shots = 0
        self.ufo_hits = 0
        self.ufo_300_hits = 0
        self._events = []
        self._ufo_awards = []
        self._pending_ufo = None
        self._previous = {}
        self.barrier_penalty_reward = 0.0
        self.barrier_penalty_return = 0.0
        self.barrier_damage_events = 0
        self._barrier_band_prev = None
        self._barrier_last_hit_frame = -10 ** 9
        self._own_shot_col = None
        self._own_shot_frame = -10 ** 9
        self._contact_frame = -10 ** 9
        self._shaping_step = {key: 0.0 for key in SHAPING_WEIGHT_KEYS}
        self._shaping_return = {key: 0.0 for key in SHAPING_WEIGHT_KEYS}
        self._shaping_counts = {"alien_kills": 0, "saucer_kills": 0, "wasted_shots": 0,
                                "shield_hits": 0, "bombs_shot": 0,
                                "dodges": 0, "cover_frames": 0, "corner_frames": 0,
                                "corner_escalated_frames": 0,
                                "deaths": 0, "invasions": 0}
        self._score_blips = 0
        self._wave_count = 0
        self._native_reward = 0.0
        self._shot_last_y_status2 = None
        self._bomb_track = [None, None, None]
        self._corner_dwell_frames = 0
        self._shot_last_x_status2 = None          # ALFRED 23 Oct 2026: where its shot was (shield hit or bomb?)
        self._last_ufo_hit_frame = -10 ** 9       # a saucer hit is not a miss
        self._moment_marks = {}                   # practice_moment(): which moments this wave has had

    def _read(self):
        r = self.host.ram
        return {
            "display_score": bcd(r[0xf8]) + 100 * bcd(r[0xf9]),
            "game_mode": int(r[0xef]), "player_game_alive": int(r[0xe7]),
            "player_sprite_alive": int(r[0x15]) == 255,
            "reserve_ships": int(r[0x1ff]), "game_moving": int(r[0xe9]),
            "shot_status": int(r[0x25]), "native_completed_shot_counter": int(r[0x8f]),
            "ufo_active": int(r[0x84]), "ufo_hit": int(r[0x85]),
            "ufo_hit_timer": int(r[0x86]), "rack": int(r[0x1fe]),
            "native_saucer_score_pointer": int(r[0x8d]) + 256 * int(r[0x8e]),
            "player_x": int(r[0x1b]), "player_y": int(r[0x1a]),
            "player_ok": int(r[0x68]), "invaded": int(r[0x6d]),
            "shot_y": int(r[0x29]), "shot_x": int(r[0x2a]),
            "bombs": tuple((int(r[status]), int(r[y]), int(r[x])) for status, y, x in BOMB_SLOTS),
        }

    def _obs(self, frame=None):
        image = Image.fromarray(self.host.frame if frame is None else frame)
        # Full cabinet display, including the visible score. No hidden state.
        return np.asarray(image.convert("L").resize((84, 84), Image.Resampling.BILINEAR), dtype=np.uint8)[None].copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        self.host.reset()
        # Real boot -> real coin -> player-one start. No memory patches or saves.
        for _ in range(180):
            self.host.run(capture=False)
        for _ in range(6):
            self.host.run((2,), capture=False)
        for _ in range(30):
            self.host.run(capture=False)
        for _ in range(6):
            self.host.run((3,), capture=False)
        for _ in range(900):
            self.host.run(capture=False)
            r = self.host.ram
            if r[0xef] == 1 and r[0xe7] == 1 and r[0xe9] == 1 and r[0x15] == 255 and r[0x10] == 0 and r[0x11] == 0:
                break
        else:
            raise RuntimeError("Native coin/start sequence did not reach active gameplay")
        self.reset_noops = int(self.np_random.integers(self.reset_noop_max + 1))
        for _ in range(self.reset_noops + 1):
            self.host.run(capture=False)
        # Capture an actual native frame at the observation boundary.
        self.host.run()
        self.raw_frames = 0
        self.shots = self.ufo_hits = self.ufo_300_hits = 0
        self.barrier_penalty_reward = self.barrier_penalty_return = 0.0
        self.barrier_damage_events = 0
        self._barrier_band_prev = self._barrier_band()
        self._barrier_last_hit_frame = -10 ** 9
        self._own_shot_col = None
        self._own_shot_frame = -10 ** 9
        self._contact_frame = -10 ** 9
        self._shaping_step = {key: 0.0 for key in SHAPING_WEIGHT_KEYS}
        self._shaping_return = {key: 0.0 for key in SHAPING_WEIGHT_KEYS}
        self._shaping_counts = {key: 0 for key in self._shaping_counts}
        self._score_blips = 0
        self._wave_count = 0
        self._native_reward = 0.0
        self._shot_last_y_status2 = None
        self._bomb_track = [None, None, None]
        self._corner_dwell_frames = 0
        self._shot_last_x_status2 = None
        self._last_ufo_hit_frame = -10 ** 9
        self._moment_marks = {}
        self._ufo_awards = []
        self._pending_ufo = None
        self._events = []
        self._previous = self._read()
        if self._previous["display_score"] != 0 or self._previous["reserve_ships"] != 2:
            raise RuntimeError("New-game invariants failed: expected zero score and three lives")
        self.score = 0
        self._done = False
        self._wall_start = time.perf_counter()
        return self._obs(), self._info()

    def _barrier_band(self):
        """Bright-pixel boolean band across the four barrier rows."""
        frame = self.host.frame
        if frame is None or frame.shape[0] < BARRIER_ROWS[1] + 1:
            return None
        return (frame.mean(axis=2) > 100)[BARRIER_ROWS[0]:BARRIER_ROWS[1] + 1, :]

    def _barrier_material_near(self, ram_x):
        """Whether any shield pixels remain in the barrier band around this
        RAM X column (review item 18: a destroyed bunker earns no cover)."""
        frame = self.host.frame
        if frame is None or frame.shape[0] < BARRIER_ROWS[1] + 1:
            return False
        col = int(round(ram_x + RAM_TO_PIXEL_COL))
        band = (frame.mean(axis=2) > 100)[BARRIER_ROWS[0]:BARRIER_ROWS[1] + 1, max(0, col - 2):col + 3]
        return bool(band.any())

    def _cannon_center(self):
        """Player cannon column in rotated frame pixels, None if not visible."""
        frame = self.host.frame
        if frame is None or frame.shape[0] < BARRIER_CANNON_ROWS[1] + 1:
            return None
        cols = (frame.mean(axis=2) > 100)[BARRIER_CANNON_ROWS[0]:BARRIER_CANNON_ROWS[1] + 1, :].sum(axis=0)
        xs = np.nonzero(cols >= 4)[0]
        return None if len(xs) == 0 else float(xs.min() + xs.max()) / 2.0

    def _detect_own_barrier_damage(self):
        """Validated own-shot barrier damage: pixel erosion in deep rows of a
        barrier rect, in the column the cannon occupied when the current
        shot was fired, within the validated contact window around the
        shot_status 2->3 transition ([-2, +20] frames; craters arrive at
        +0 and +16). One event per barrier hit: the two crater phases merge
        inside a 22-frame refractory. Counts are always tracked; only the
        reward component is scaled by barrier_penalty_weight, so weight-0
        probes still validate detection.
        """
        band = self._barrier_band()
        if band is None:
            return 0
        events = 0
        prev = self._barrier_band_prev
        if prev is not None:
            lost = prev & ~band
            if lost.any():
                attributed = False
                for x0, x1 in BARRIER_RECTS:
                    sub = lost[:, x0:x1 + 1]
                    if int(sub.sum()) < BARRIER_LOST_MIN_PIXELS:
                        continue
                    rows, cols = np.nonzero(sub)
                    if not bool((rows + BARRIER_ROWS[0] >= BARRIER_OWN_CRATER_MIN_Y).any()):
                        continue
                    if self._own_shot_col is None:
                        continue
                    contact_offset = self.raw_frames - self._contact_frame
                    if not (-BARRIER_CONTACT_BEFORE <= contact_offset <= BARRIER_CONTACT_AFTER):
                        continue
                    lost_cols = {int(c + x0) for c in cols}
                    if any(abs(c - self._own_shot_col) <= BARRIER_OWN_SHOT_COL_TOLERANCE
                           for c in lost_cols):
                        attributed = True
                        break
                if attributed:
                    # One event per barrier hit: the validated crater phases
                    # (+0 and +16 frames) merge inside the refractory window;
                    # distinct hits are further apart than the refractory.
                    if self.raw_frames - self._barrier_last_hit_frame > BARRIER_HIT_REFRACTORY:
                        events = 1
                        self.barrier_damage_events += 1
                        self._barrier_last_hit_frame = self.raw_frames
        self._barrier_band_prev = band
        return events

    def _observe_frame(self):
        now = self._read()
        old = self._previous
        # Stop at native game-over, not on the death of one cannon.
        terminated = now["player_game_alive"] == 0 or now["game_mode"] == 0
        if old.get("rack") is not None and now["rack"] != old["rack"] and now["game_mode"] == 1:
            self._wave_count += 1
        try:
            delta = score_delta(old["display_score"], now["display_score"])
        except TornScoreRead as torn:
            # Refuse the fabricated read, keep the accepted baseline, log the
            # blip value; the next clean frame reconciles exactly. Never fatal.
            self._score_blips += 1
            self._events.append({"type": "score_blip", "frame": self.raw_frames,
                                 "accepted": torn.previous, "read": torn.current})
            delta = 0
            now["display_score"] = old["display_score"]
        self.score += delta
        if old["shot_status"] == 0 and now["shot_status"] != 0 and now["game_mode"] == 1:
            self.shots += 1
            # Own-barrier attribution anchor: the shot flies straight up from
            # here, so its column is the cannon column at fire time.
            self._own_shot_col = self._cannon_center()
            self._own_shot_frame = self.raw_frames
        if old["shot_status"] == 2 and now["shot_status"] == 3:
            # Shot contact (validated: barrier craters follow in two phases,
            # at +0 and +16 frames from this transition).
            self._contact_frame = self.raw_frames
        if now["ufo_active"] and now["ufo_hit"] and not old["ufo_hit"]:
            self.ufo_hits += 1
            self._last_ufo_hit_frame = self.raw_frames
            self._pending_ufo = {"hit_frame": self.raw_frames, "shot": self.shots,
                                 "native_completed_shots_at_hit": now["native_completed_shot_counter"],
                                 "native_saucer_score_pointer_at_hit": now["native_saucer_score_pointer"]}
            self._events.append({"type": "ufo_hit", "frame": self.raw_frames, **self._pending_ufo})
        # Award follows collision by an explosion delay. Preserve the hitting
        # shot's identity even if a subsequent shot has already been launched.
        saucer_points = saucer_component(delta)
        confirmed_saucer = 0
        if self._pending_ufo and saucer_points and now["ufo_hit_timer"] <= 24 and (now["ufo_hit"] or old["ufo_hit"]):
            award = {"points": saucer_points, "frame": self.raw_frames, **self._pending_ufo,
                     "frame_score_delta": delta}
            self._ufo_awards.append(award)
            self._events.append({"type": "ufo_award", **award})
            self._pending_ufo = None
            confirmed_saucer = saucer_points
            if saucer_points == 300:
                self.ufo_300_hits += 1
        if not now["ufo_active"] or terminated:
            self._pending_ufo = None
        if now["reserve_ships"] < old["reserve_ships"] or terminated:
            self._events.append({"type": "game_over" if terminated else "life_lost", "frame": self.raw_frames, "score": self.score})
        self._apply_shaping(old, now, delta, confirmed_saucer)
        self._previous = now
        return delta, terminated

    def _apply_shaping(self, old, now, delta, saucer_points):
        """Jonny's shaping pack, evaluated per native frame. The native reward
        is untouched; every component is signed by its rule and accounted
        separately in _shaping_step / _shaping_return / _shaping_counts.
        All triggers are read-only native RAM events validated against the
        disassembly and calibrated in evidence/shaping_telemetry_calibration.json.
        """
        w = self.shaping_weights
        if now["game_mode"] != 1:
            return
        if w is None:               # ALFRED 23 Oct 2026: 1a-1c count shots/hits/misses with or without the pack
            w = _NO_WEIGHTS

        def bump(key, value):
            self._shaping_step[key] += value

        # 1a. alien destroyed: score remainder after any confirmed saucer award
        if delta - saucer_points in (10, 20, 30):
            bump("alien_kill", w["alien_kill"])
            self._shaping_counts["alien_kills"] += 1
        # 1b. mystery ship destroyed: the validated hit -> award chain fired
        if saucer_points:
            bump("saucer_kill", w["saucer_kill"])
            self._shaping_counts["saucer_kills"] += 1
        # 1c. wasted shot: 2->3 whose last in-flight Y reached the top miss zone
        #     ALFRED 23 Oct 2026: not when it was the saucer being hit (it flies in the same band); and a shot that
        #     ended LOW either shot down a bomb (fine - that's defending) or blew a hole in its own shield (a miss)
        status = now["shot_status"]
        if old["shot_status"] == 2 and status == 3 and self._shot_last_y_status2 is not None:
            ly, lx = self._shot_last_y_status2, self._shot_last_x_status2
            if ly >= SAUCER_BAND_FLOOR:
                if self.raw_frames - self._last_ufo_hit_frame > 8 and not (now["ufo_hit"] or old["ufo_hit"]):
                    bump("wasted_shot", -w["wasted_shot"])
                    self._shaping_counts["wasted_shots"] += 1
                    self._events.append({"type": "wasted_shot", "frame": self.raw_frames,
                                         "shot": self.shots, "last_flight_y": ly})
            elif lx is not None:
                near = min([abs(by - ly) + abs(bx - lx) for frame in (old, now)
                            for bstatus, by, bx in frame["bombs"] if bstatus & 0x80] or [999])
                if near <= BOMB_NEAR:
                    self._shaping_counts["bombs_shot"] += 1
                elif ly < SHIELD_SHOT_Y_MAX and any(bx0 - 3 <= lx <= bx1 + 3 for bx0, bx1 in BARRIER_RAM_RECTS):
                    bump("wasted_shot", -w["wasted_shot"])
                    self._shaping_counts["shield_hits"] += 1
                    self._events.append({"type": "shield_hit", "frame": self.raw_frames,
                                         "shot": self.shots, "last_flight_y": ly})
        if status == 2:
            self._shot_last_y_status2 = now["shot_y"]
            self._shot_last_x_status2 = now["shot_x"]
        elif status == 0:
            self._shot_last_y_status2 = None
        if self.shaping_weights is None:      # no shaping pack: counting only, the rest is the pack's
            return
        # 3a. cannon lost: playerAlive (0x2015) drops from 255 at the collision
        # frame (ROM 060F) and stays 0/1 through the blow-up animation, so its
        # falling edge bills immediately, once per death. playerOK (0x2068)
        # lags ~60 frames in the explosion handler (0297-029B); the review
        # caught survival checks using it (item 17). An invasion death bills
        # to the invasion rule only, so invasion costs exactly Jonny's -100.
        if old["player_sprite_alive"] and not now["player_sprite_alive"]:
            if not (old["invaded"] or now["invaded"]):
                bump("death", -w["death"])
            self._shaping_counts["deaths"] += 1
        # 2c. grid invaded: the ROM's own flag; end of round follows natively
        if old["invaded"] == 0 and now["invaded"] != 0:
            bump("invasion", -w["invasion"])
            self._shaping_counts["invasions"] += 1
            self._events.append({"type": "invasion", "frame": self.raw_frames, "score": self.score})
        # Survival and billing gate: alive requires BOTH flags — playerOK is
        # 1 until the delayed explosion handler runs, playerAlive drops at the
        # collision frame. Together they mean "alive and not mid-explosion".
        alive = now["player_ok"] == 1 and now["player_sprite_alive"]
        cannon_cx = now["player_x"] + 8  # ROM 061B: sprite left edge + 8 = centre
        # 2a. bunker cover: a falling bomb is lined up on the barrier above
        # the cannon, still above the calibrated barrier top
        if alive:
            covered = False
            for bx0, bx1 in BARRIER_RAM_RECTS:
                if not bx0 <= cannon_cx <= bx1:
                    continue
                for bstatus, by, bx in now["bombs"]:
                    if (bstatus & 0x80) and not (bstatus & 0x01) and by > BARRIER_TOP_RAM_Y \
                            and bx0 <= bx <= bx1 and self._barrier_material_near(bx):
                        covered = True
                        break
                if covered:
                    break
            if covered:
                bump("cover_per_frame", w["cover_per_frame"])
                self._shaping_counts["cover_frames"] += 1
        # 3b. corner camping: pinned within the margin of a travel stop, not
        # moving. Jonny 2026-10-07: the tax escalates with dwell time —
        # base rate under 2 s parked (a tactical corner visit stays cheap),
        # 3x beyond that (camping crept back at 30% of frames under v6).
        parked = alive and now["player_x"] == old["player_x"] and (
            now["player_x"] - PLAYER_X_MIN <= CORNER_MARGIN
            or PLAYER_X_MAX - now["player_x"] <= CORNER_MARGIN)
        if parked:
            self._corner_dwell_frames += 1
            rate = w["corner_per_second"]
            if self._corner_dwell_frames > CORNER_GRACE_SECONDS * self.host.fps:
                rate *= CORNER_ESCALATION
                self._shaping_counts["corner_escalated_frames"] += 1
            bump("corner_per_second", -rate / self.host.fps)
            self._shaping_counts["corner_frames"] += 1
        else:
            self._corner_dwell_frames = 0
        # 2b. close dodge: a bomb reaches the player band within the 3-pixel
        # margin while the cannon moves away from it, and misses
        for i, (bstatus, by, bx) in enumerate(now["bombs"]):
            track = self._bomb_track[i]
            if (bstatus & 0x80) and not (bstatus & 0x01):
                if track is None:
                    track = self._bomb_track[i] = {"min_dx": None, "last_y": by}
                if PLAYER_BAND[0] - 4 <= by <= PLAYER_BAND[1] + 2:
                    dx = abs(bx - cannon_cx)
                    if track["min_dx"] is None or dx < track["min_dx"]:
                        track.update(min_dx=dx, bomb_x=bx, cannon_x=cannon_cx,
                                     cannon_dx=now["player_x"] - old["player_x"])
                track["last_y"] = by
            elif track is not None:
                arrived = track["last_y"] <= PLAYER_BAND[1] + 2
                close = track["min_dx"] is not None and track["min_dx"] <= DODGE_MARGIN_X
                if arrived and close and alive:
                    moving, toward = track["cannon_dx"], track["bomb_x"] - track["cannon_x"]
                    if moving != 0 and (moving > 0) == (toward < 0):
                        bump("dodge", w["dodge"])
                        self._shaping_counts["dodges"] += 1
                        self._events.append({"type": "dodge", "frame": self.raw_frames,
                                             "bomb_x": track["bomb_x"], "min_dx": track["min_dx"]})
                self._bomb_track[i] = None

    def step(self, action):
        if self._done:
            raise RuntimeError("reset() is required before step() and after episode end")
        if not self.action_space.contains(action):
            raise ValueError(f"Unknown action {action}")
        self._events = []
        self._shaping_step = {key: 0.0 for key in SHAPING_WEIGHT_KEYS}
        points = 0
        barrier_events = 0
        terminated = False
        frames = []
        for frame_index in range(self.decision_repeat):
            # Capture both most recent frames to keep alternating-frame sprites.
            capture = frame_index >= self.decision_repeat - 2
            self.host.run(ACTION_BUTTONS[int(action)], capture=capture)
            self.raw_frames += 1
            hook = getattr(self, 'frame_hook', None)
            if hook is not None:
                hook(self)  # measurement channel only; never part of observations
            delta, terminated = self._observe_frame()
            points += delta
            if capture:
                frames.append(self.host.frame)
                if self.barrier_penalty_weight > 0:
                    barrier_events += self._detect_own_barrier_damage()
                # weight 0: the pixel-band analysis is skipped (37% of episode
                # wall on this build); the field reports None, never a fake 0.
            if terminated:
                break
        truncated = self.max_episode_frames > 0 and self.raw_frames >= self.max_episode_frames and not terminated
        self._done = terminated or truncated
        merged = np.maximum(frames[-2], frames[-1]) if len(frames) >= 2 else self.host.frame
        # Shaped components, each logged separately from the native reward.
        self.barrier_penalty_reward = -self.barrier_penalty_weight * barrier_events
        self.barrier_penalty_return += self.barrier_penalty_reward
        shaping_reward = 0.0
        if self.shaping_weights is not None:
            for key, value in self._shaping_step.items():
                self._shaping_return[key] += value
                shaping_reward += value
        reward = points / 100.0 + self.barrier_penalty_reward + shaping_reward
        self._native_reward = points / 100.0  # pure native component for replay-reserve selection
        return self._obs(merged), reward, terminated, truncated, self._info()

    def _info(self):
        native = self._previous
        alive = bool(native.get("player_game_alive", 0)) and bool(native.get("game_mode", 0))
        return {
            "score": self.score, "display_score": native.get("display_score", 0),
            "lives": native.get("reserve_ships", 0) + 1 if alive else 0,
            "raw_frames": self.raw_frames, "shots": self.shots,
            # ALFRED 23 Oct 2026: for the accuracy meter and 'A missed shot costs' (arcade_vec.Cabinet reads these)
            "hits": self._shaping_counts["alien_kills"] + self._shaping_counts["saucer_kills"],
            "misses": self._shaping_counts["wasted_shots"] + self._shaping_counts["shield_hits"],
            "ufo_hits": self.ufo_hits, "ufo_300_hits": self.ufo_300_hits,
            "ufo_awards": list(self._ufo_awards), "events": list(self._events),
            "barrier_damage_events": self.barrier_damage_events if self.barrier_penalty_weight > 0 else None,
            "barrier_penalty_weight": self.barrier_penalty_weight,
            "barrier_penalty_reward": self.barrier_penalty_reward,
            "barrier_penalty_return": self.barrier_penalty_return,
            "score_blips": self._score_blips,
            "waves": self._wave_count,
            "native_reward": self._native_reward,
            "shaping": {"enabled": self.shaping_weights is not None,
                        "weights": dict(self.shaping_weights) if self.shaping_weights is not None else None,
                        "step": dict(self._shaping_step),
                        "return": dict(self._shaping_return),
                        "counts": dict(self._shaping_counts)},
            "episode_seconds_game": self.raw_frames / self.host.fps,
            "game_fps": self.host.fps, "rack": native.get("rack"),
            "reset_noops": getattr(self, "reset_noops", 0),
            "telemetry": {"source": "read-only original ROM RAM, per native frame",
                          "shots": "player-shot status rises from available to active",
                          "saucer_awards": "native score increments while saucer-hit flag is active",
                          "own_barrier_damage": "validated pixel crater under the four barrier rects, "
                                                "own-shot attributed; counts always logged, penalty "
                                                "scaled by barrier_penalty_weight"},
            "shaping_pack": "read-only native RAM events only: score-remainder alien kills, validated "
                            "saucer awards, top-miss wasted shots, barrier cover vs calibrated RAM "
                            "rects/top with shield material present, close dodges in the ROM player band "
                            "with survival on the immediate playerAlive flag, native invaded flag, deaths "
                            "billed at the collision frame, stationary corner idling; nothing enters "
                            "observations",
        }

    def practice_moment(self):
        """ALFRED 23 Oct 2026 ('Practise the hard parts', arcade_vec.Cabinet): is right now a moment worth starting
        practice games from? 'new_wave' (a fresh wave after the first), 'few_left' (8 or fewer invaders left) and
        'last_3' (3 or fewer) - each at most once a wave, only while the cannon is alive and the invaders moving.
        RAM 0x2082 = invaders left (checked on a record game: 55 at each wave's start, counting down)."""
        r = self.host.ram
        if not (r[0xef] == 1 and r[0x15] == 255 and r[0x68] == 1 and r[0x6d] == 0 and r[0xe9] == 1):
            return None
        n, wave, marks = int(r[0x82]), self._wave_count, self._moment_marks
        if marks.get("wave") != wave:
            marks.clear()
            marks["wave"] = wave
        for tag, ok in (("new_wave", wave >= 1 and n >= 50), ("few_left", 0 < n <= 8), ("last_3", 0 < n <= 3)):
            if ok and tag not in marks:
                marks[tag] = 1
                return tag
        return None

    def render(self):
        return self.host.frame.copy()

    def drain_audio(self):
        """Buffered stereo int16 samples since the last drain; empty unless audio=True."""
        return self.host.drain_audio()

    def close(self):
        self.host.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "logs" / "environment_smoke.json")
    parser.add_argument("--screenshot", type=Path, default=ROOT / "logs" / "environment_smoke.png")
    args = parser.parse_args()
    env = ArcadeInvadersEnv(decision_repeat=args.repeat)
    try:
        observation, info = env.reset(seed=args.seed)
        assert observation.shape == (1, 84, 84) and observation.dtype == np.uint8
        assert info["score"] == 0 and info["lives"] == 3
        started = time.perf_counter()
        native_frames, episodes, reward_sum, episode_points = 0, [], 0.0, 0.0
        for i in range(args.steps):
            previous_frames = info["raw_frames"]
            observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
            native_frames += info["raw_frames"] - previous_frames
            reward_sum += reward
            episode_points += reward * 100
            assert abs(episode_points - info["score"]) < 1e-6
            if i == min(args.steps - 1, 1500):
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(env.render()).save(args.screenshot)
            if terminated or truncated:
                episodes.append({**info, "terminated": terminated, "truncated": truncated})
                observation, info = env.reset()
                episode_points = 0.0
        elapsed = time.perf_counter() - started
        report = {"core": env.host.system_info.library_name.decode(),
                  "core_version": env.host.system_info.library_version.decode(),
                  "core_sha256": hashlib.sha256(env.host.core_path.read_bytes()).hexdigest(),
                  "steps": args.steps, "raw_frames": native_frames, "wall_seconds": elapsed,
                  "decisions_per_second": args.steps / elapsed,
                  "raw_frames_per_second": native_frames / elapsed,
                  "native_fps": env.host.fps, "reward_sum": reward_sum,
                  "episodes": episodes, "last_episode": info,
                  "checks": {"observation_shape": True, "reward_matches_native_score": True,
                             "native_new_game_three_lives": True}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
