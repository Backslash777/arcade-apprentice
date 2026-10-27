"""Original Donkey Kong (US set 2), a pixel-only Gymnasium environment.

Native score and terminal signals are read-only environment/evaluator plumbing.
Only cabinet pixels reach the policy. RAM is never written.
Driver: finalburnneo/FBNeo src/burn/drv/pre90s/d_dkong.cpp
RAM roles: furrykef/dkdasm dkong.asm, verified against this local dkongo set.
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
ACTION_NAMES = ("noop", "left", "right", "up", "down", "jump", "left_jump", "right_jump", "up_jump", "down_jump")
ACTION_BUTTONS = ((), (6,), (7,), (4,), (5,), (0,), (6, 0), (7, 0), (4, 0), (5, 0))

# Jonny's v6 reward stack (2026-10-05 evening). Telemetry validated on
# hardware the same day (evidence/hunt-v6-*, tools/probe_ram_dump.py):
#   marioOnLadder 0x6215, marioAirborne 0x6216, marioHammerActive 0x6217,
#   hammer timer 0x6394/95, hammer in-play flags 0x6681/0x6691.
#   Barrels: 0x6700 + i*0x20, active == 1 strictly (02 = spawn-pending),
#   x = +3, y = +5 (sprite mirror 0x6980 matches x/y exactly; dodge detector
#   coincident with native +100 leap awards 2/2, award lags crossing ~6-9f).
BARREL_BASE, BARREL_STRIDE, BARREL_COUNT = 0x700, 0x20, 10
# Floor band: 26 px of height per floor index. Measured anchors: bottom
# girder 240->233 (sloped), girder 2 tops at 211 (ladder x=200 climb);
# hammer record positions cross-check (girder-2 hammer sprite y=192,
# +16px sprite height = surface ~208-211). 25m's six girders at ~29px
# spacing land on band indices 0..5; jump apex (~13px) never crosses a band.
FLOOR_BAND = 26


def _weight(name, value, hi):
    value = float(value)
    if not 0.0 <= value <= hi:
        raise ValueError(f"{name} must be in [0, {hi}]")
    return value


def bcd(value: int) -> int:
    value = int(value)
    if value & 15 > 9 or value >> 4 > 9:
        raise RuntimeError(f"Invalid native BCD score byte {value:02x}")
    return 10 * (value >> 4) + (value & 15)


def score_delta(previous: int, current: int) -> int:
    change = current - previous
    if change < 0:
        if previous >= 990000 and current <= 10000:
            change += 1000000
        else:
            raise RuntimeError(f"Unexpected native score decrease {previous} -> {current}; no reward fabricated")
    if change > 20000:
        raise RuntimeError(f"Implausible native per-frame score increase {change}")
    return change


class ArcadeDonkeyKongEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, decision_repeat=2, root=None, core_path=None, rom_path=None,
                 render_mode=None, max_episode_frames=0, reset_noop_max=30, audio=False,
                 jump_penalty_weight=0.0, climb_weight=0.0,
                 altitude_gain_weight=0.0, floor_milestone_weight=0.0,
                 level_completion_weight=0.0, barrel_dodge_weight=0.0,
                 hammer_grab_weight=0.0, hammer_smash_weight=0.0,
                 death_penalty_weight=0.0, backslide_weight=0.0, stall_weight=0.0):
        super().__init__()
        if not 0.0 <= jump_penalty_weight <= 1.0:
            raise ValueError("jump_penalty_weight must be in [0, 1]")
        self.jump_penalty_weight = float(jump_penalty_weight)
        self.jump_penalty_reward = self.jump_penalty_return = 0.0
        if not 0.0 <= climb_weight <= 1.0:
            raise ValueError("climb_weight must be in [0, 1]")
        self.climb_weight = float(climb_weight)
        self.climb_reward = self.climb_return = 0.0
        self._prev_climb_potential = 0.0
        # Jonny's v6 reward stack (2026-10-05 evening), his numbers as raw
        # reward units; every component separately weighted and accounted.
        self.altitude_gain_weight = _weight("altitude_gain_weight", altitude_gain_weight, 10.0)
        self.floor_milestone_weight = _weight("floor_milestone_weight", floor_milestone_weight, 100.0)
        self.level_completion_weight = _weight("level_completion_weight", level_completion_weight, 1000.0)
        self.barrel_dodge_weight = _weight("barrel_dodge_weight", barrel_dodge_weight, 50.0)
        self.hammer_grab_weight = _weight("hammer_grab_weight", hammer_grab_weight, 100.0)
        self.hammer_smash_weight = _weight("hammer_smash_weight", hammer_smash_weight, 50.0)
        self.death_penalty_weight = _weight("death_penalty_weight", death_penalty_weight, 500.0)
        self.backslide_weight = _weight("backslide_weight", backslide_weight, 10.0)
        self.stall_weight = _weight("stall_weight", stall_weight, 1.0)
        self._v6_weights = {
            "altitude": self.altitude_gain_weight, "floor": self.floor_milestone_weight,
            "completion": self.level_completion_weight, "dodge": self.barrel_dodge_weight,
            "hammer_grab": self.hammer_grab_weight, "hammer_smash": self.hammer_smash_weight,
            "death": self.death_penalty_weight, "backslide": self.backslide_weight,
            "stall": self.stall_weight}
        self._v6_active = any(w != 0.0 for w in self._v6_weights.values())
        self._v6_step = {k: 0.0 for k in self._v6_weights}
        self._v6_returns = {k: 0.0 for k in self._v6_weights}
        self._barrel_serial = [0] * BARREL_COUNT
        self._barrel_prev = [None] * BARREL_COUNT
        self._dodged = set()
        self._max_height = 0.0
        self._max_floor = 0
        self._ground_floor = 0
        self.root = Path(root or ROOT)
        self.decision_repeat = int(decision_repeat)
        if not 1 <= self.decision_repeat <= 8:
            raise ValueError("decision_repeat must be 1..8")
        self.max_episode_frames = int(max_episode_frames)
        self.reset_noop_max = int(reset_noop_max)
        self.render_mode = render_mode
        self.action_space = gym.spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = gym.spaces.Box(0, 255, (1, 84, 84), dtype=np.uint8)
        self.host = LibretroHost(core_path or self.root / "vendor" / "fbneo_libretro.dll",
                                 rom_path or self.root / "roms" / "dkongo.zip", self.root, require_ram=False, audio=audio)
        self.sample_rate = self.host.sample_rate
        # d_dkong.cpp MemIndex: AllRam starts with DrvZ80RAM (0x1000).
        # The driver maps that region to Z80 addresses 0x6000..0x6fff.
        if self.host.ram is None or len(self.host.ram) != 0x2314:
            self.host.close()
            raise RuntimeError("Unexpected FBNeo Donkey Kong RAM layout; refusing unverified telemetry")
        self._done = True
        self.raw_frames = self.score = self.boards_completed = self.jump_presses = 0
        self.jump_penalty_reward = self.jump_penalty_return = 0.0
        self.native_reward = 0.0
        self.max_level = 1
        self._previous = {}
        self._events = []

    def _read(self):
        r = self.host.ram
        return {"display_score": bcd(r[0xb2]) + 100 * bcd(r[0xb3]) + 10000 * bcd(r[0xb4]),
                "game_mode": int(r[5]), "substate": int(r[0xa]), "mario_alive": int(r[0x200]),
                "lives": int(r[0x228]), "level": int(r[0x229]), "board": int(r[0x227]),
                "jump_edge": bool(r[0x10] & 128), "mario_x": int(r[0x203]), "mario_y": int(r[0x205]),
                "mario_airborne": int(r[0x216]), "mario_on_ladder": int(r[0x215]),
                "hammer_active": int(r[0x217])}

    def _barrels(self):
        """Live barrel records: (active, x, y) per slot. Validated 2026-10-05:
        active == 1 strictly (02 is spawn-pending at DK, not a live hazard)."""
        r = self.host.ram
        out = []
        for i in range(BARREL_COUNT):
            b = BARREL_BASE + i * BARREL_STRIDE
            out.append((int(r[b]) == 1, int(r[b + 3]), int(r[b + 5])))
        return out

    def _v6_frame(self, cur, old, terminated):
        """Jonny's v6 reward stack (2026-10-05 evening), per frame.

        Guardrails as he specified: altitude pays only on new max height for
        the life (anti ladder-loop); a barrel dodge pays once per barrel
        instance per life (anti jump-loop). Death charges -death_weight and
        re-anchors every record; board completion pays +completion_weight and
        re-anchors. Cutscene/death-anim frames charge nothing either way.
        The completion payment carries the same terminal guard as the native
        board counter (2026-10-06 review item 22). NOTE on naming (item 20):
        "floor" milestones are 26px HEIGHT BAND crossings, a proxy for floor
        arrival — an on-ladder band crossing can precede the actual platform
        by up to one band, and upper-floor edges are extrapolated from two
        measured anchors (240/233 bottom girder, 211 girder 2).
        """
        c = self._v6_step
        bars = self._barrels()
        mx, my = cur["mario_x"], cur["mario_y"]
        life_lost = cur["lives"] < old["lives"]
        board_changed = (cur["level"], cur["board"]) != (old["level"], old["board"])
        prev_bars = []
        for i, (act, bx, by) in enumerate(bars):
            prev = self._barrel_prev[i]
            prev_bars.append(prev)
            if act and (prev is None or not prev[0]):
                self._barrel_serial[i] += 1  # new barrel instance in this slot
            self._barrel_prev[i] = (act, bx, by)
        if life_lost:
            c["death"] -= self.death_penalty_weight
            self._max_height = 0.0
            self._max_floor = 0
            self._ground_floor = 0
            self._dodged.clear()
            return  # death frame: no other charges (respawn teleports)
        if board_changed:
            if not terminated:  # same guard as the native board counter
                c["completion"] += self.level_completion_weight
            yy = min(max(my, 32), 240)
            self._max_height = float(240 - yy)
            self._max_floor = int(self._max_height // FLOOR_BAND)
            self._ground_floor = self._max_floor
            self._dodged.clear()
            return
        if not (cur["mario_alive"] == 1 and cur["game_mode"] == 3 and cur["substate"] == 0xC):
            return
        yy = min(max(my, 32), 240)
        height = 240 - yy
        grounded = cur["mario_airborne"] == 0 or cur["mario_on_ladder"] == 1
        if grounded:
            # 1. altitude gain (GATED 2026-10-06, Jonny's ruling): the payout
            # AND the per-life record update only on grounded or on-ladder
            # frames. Airborne height earns nothing and does not raise the
            # record; landing on a genuinely higher platform is a grounded
            # frame at a new height, so it counts. Walking uphill (sloped
            # girders) still earns — this is not "only ladders pay".
            if height > self._max_height:
                c["altitude"] += self.altitude_gain_weight * (height - self._max_height)
                self._max_height = float(height)
            # 2. height-band milestones (floor-arrival proxy, item 20 label):
            # new highest 26px band this life
            floor_idx = int(height // FLOOR_BAND)
            if floor_idx > self._max_floor:
                c["floor"] += self.floor_milestone_weight * (floor_idx - self._max_floor)
                self._max_floor = floor_idx
            # 7. backsliding: downward floor-band transition
            if floor_idx < self._ground_floor:
                c["backslide"] -= self.backslide_weight * (self._ground_floor - floor_idx)
            self._ground_floor = floor_idx
        # 4. barrel dodge: airborne while a barrel's x crosses Mario's x in
        # the beneath-window; once per (slot, spawn serial) per life
        if cur["mario_airborne"] == 1:
            for i, (act, bx, by) in enumerate(bars):
                if not act:
                    continue
                prev = prev_bars[i]
                pdx = prev[1] - old["mario_x"] if prev is not None and prev[0] else None
                dx = bx - mx
                crossed = pdx is not None and (pdx == 0 or dx == 0 or (pdx < 0) != (dx < 0))
                if (crossed and my - 8 <= by <= my + 24
                        and (i, self._barrel_serial[i]) not in self._dodged):
                    c["dodge"] += self.barrel_dodge_weight
                    self._dodged.add((i, self._barrel_serial[i]))
        # 5. hammer grab edge
        if cur["hammer_active"] == 1 and old["hammer_active"] == 0:
            c["hammer_grab"] += self.hammer_grab_weight
        # 5b. smash: a barrel slot deactivates near Mario while hammered
        if cur["hammer_active"] == 1:
            for i, (act, bx, by) in enumerate(bars):
                prev = prev_bars[i]
                if prev is not None and prev[0] and not act:
                    pbx, pby = prev[1], prev[2]
                    if abs(pbx - mx) <= 32 and my - 8 <= pby <= my + 24:
                        c["hammer_smash"] += self.hammer_smash_weight
        # 8. stall: completely still (position unchanged, not airborne)
        if (cur["mario_x"] == old["mario_x"] and cur["mario_y"] == old["mario_y"]
                and cur["mario_airborne"] == 0):
            c["stall"] -= self.stall_weight

    def _obs(self, frame=None):
        pixels = Image.fromarray(self.host.frame if frame is None else frame)
        return np.asarray(pixels.convert("L").resize((84, 84), Image.Resampling.BILINEAR), dtype=np.uint8)[None].copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        self.host.reset()
        for _ in range(180):
            self.host.run(capture=False)
        for _ in range(6):
            self.host.run((2,), capture=False)
        for _ in range(30):
            self.host.run(capture=False)
        for _ in range(6):
            self.host.run((3,), capture=False)
        for _ in range(2400):
            self.host.run(capture=False)
            r = self.host.ram
            if r[5] == 3 and r[0xa] == 0xc and r[0x200] == 1:
                break
        else:
            raise RuntimeError("Native coin/start never reached active Donkey Kong gameplay")
        self.reset_noops = int(self.np_random.integers(self.reset_noop_max + 1))
        for _ in range(self.reset_noops):
            self.host.run(capture=False)
        self.host.run()
        self._previous = self._read()
        if self._previous["display_score"] != 0 or self._previous["lives"] != 3:
            raise RuntimeError(f"New-game score/lives invariants failed: {self._previous}")
        self.raw_frames = self.score = self.boards_completed = self.jump_presses = 0
        self.jump_penalty_reward = self.jump_penalty_return = 0.0
        self.climb_reward = self.climb_return = 0.0
        y0 = min(max(int(self._previous["mario_y"]), 32), 240)
        self._prev_climb_potential = min(max((240 - y0) / 200.0, 0.0), 1.0)
        self.max_level = self._previous["level"]
        # v6 per-life state (Jonny's guardrails: max-height record and
        # once-per-barrel-instance dodge awards, both reset on life loss).
        self._barrel_serial = [0] * BARREL_COUNT
        self._barrel_prev = [None] * BARREL_COUNT
        self._dodged = set()
        self._max_height = 0.0
        self._max_floor = 0
        self._ground_floor = 0
        self._v6_step = {k: 0.0 for k in self._v6_weights}
        self._v6_returns = {k: 0.0 for k in self._v6_weights}
        self._events = []
        self._done = False
        return self._obs(), self._info()

    def step(self, action):
        if self._done:
            raise RuntimeError("reset() is required before stepping a finished game")
        if not self.action_space.contains(action):
            raise ValueError(f"Unknown action {action}")
        points, terminated, frames = 0, False, []
        self._events = []
        if self._v6_active:
            self._v6_step = {k: 0.0 for k in self._v6_weights}
        presses_start = self.jump_presses
        climb_start = self.climb_return
        for frame_index in range(self.decision_repeat):
            capture = frame_index >= self.decision_repeat - 2
            self.host.run(ACTION_BUTTONS[int(action)], capture=capture)
            self.raw_frames += 1
            current, old = self._read(), self._previous
            # A lost life is not an episode boundary. Native remaining-lives zero
            # is terminal; the game-mode fallback catches any native exit.
            terminated = current["lives"] == 0 or current["game_mode"] != 3
            try:
                change = score_delta(old["display_score"], current["display_score"])
            except RuntimeError:
                if current["display_score"] >= old["display_score"]:
                    raise  # the implausible-increase guard stays fatal
                # Torn multi-byte BCD read at a carry frame (Space Invaders
                # handoff lesson, 2026-10-06): the core writes score bytes
                # non-atomically, so a carry frame can read as a spurious
                # decrease. Clamp to 0, freeze the accepted baseline (never
                # store the torn value), log a score_blip; the next clean
                # frame reconciles exactly. Invalid BCD stays fatal (bcd()).
                self._events.append({"type": "score_blip", "frame": self.raw_frames,
                                     "previous": old["display_score"], "read": current["display_score"]})
                current["display_score"] = old["display_score"]
                change = 0
            points += change
            self.score += change
            if current["jump_edge"]:
                self.jump_presses += 1
            # Jonny's climb-progress shaping (2026-10-05), DORMANT at weight 0
            # since v6. Heuristic note (2026-10-06 review item 28): the code
            # applies weight * (phi_next - phi_now) WITHOUT the gamma factor,
            # so this is NOT the Ng, Harada & Russell (1999) potential form
            # gamma*Phi(s') - Phi(s) and the policy-invariance guarantee does
            # not apply. Retained as available heuristic shaping only.
            # phi = clamp((240 - y)/200, 0, 1): 0 at spawn, ~1 at board top.
            # Life loss or board/level change re-anchors without charging
            # the teleport.
            y_now = min(max(int(current["mario_y"]), 32), 240)
            phi_now = min(max((240 - y_now) / 200.0, 0.0), 1.0)
            teleported = current["lives"] < old["lives"] \
                or (current["level"], current["board"]) != (old["level"], old["board"])
            if teleported:
                self._prev_climb_potential = phi_now
            else:
                self.climb_return += self.climb_weight * (phi_now - self._prev_climb_potential)
                self._prev_climb_potential = phi_now
            if current["lives"] < old["lives"]:
                self._events.append({"type": "life_lost", "frame": self.raw_frames, "lives": current["lives"], "score": self.score})
            if self._v6_active:
                self._v6_frame(current, old, terminated)
            if not terminated and (current["level"], current["board"]) != (old["level"], old["board"]):
                self.boards_completed += 1
                self._events.append({"type": "board_changed", "frame": self.raw_frames, "board": current["board"], "level": current["level"]})
            self.max_level = max(self.max_level, current["level"])
            self._previous = current
            if capture:
                frames.append(self.host.frame)
            if terminated:
                self._events.append({"type": "game_over", "frame": self.raw_frames, "score": self.score})
                break
        truncated = bool(self.max_episode_frames and self.raw_frames >= self.max_episode_frames and not terminated)
        self._done = terminated or truncated
        # Jonny's jump tax (2026-10-05): a small cost per native jump press,
        # sized below the barrel-leap reward so obstacle jumps stay profitable
        # but empty jumping is taxed. Uses the validated jump_edge telemetry;
        # native points are unchanged and logged separately.
        self.jump_penalty_reward = -self.jump_penalty_weight * (self.jump_presses - presses_start)
        self.jump_penalty_return += self.jump_penalty_reward
        self.climb_reward = self.climb_return - climb_start
        v6_step_total = 0.0
        if self._v6_active:
            for key, value in self._v6_step.items():
                self._v6_returns[key] += value
                v6_step_total += value
        merged = np.maximum(frames[-2], frames[-1]) if len(frames) > 1 else self.host.frame
        # native_reward is the cabinet's own score delta this decision,
        # exposed separately (2026-10-06 review item 16/17) so replay reserve
        # membership and logging never have to infer it from the combined sum.
        self.native_reward = points / 100.0
        return self._obs(merged), self.native_reward + self.jump_penalty_reward + self.climb_reward + v6_step_total, terminated, truncated, self._info()

    def _info(self):
        state = self._previous
        return {"score": self.score, "display_score": state.get("display_score", 0),
                "native_reward": getattr(self, "native_reward", 0.0),
                "lives": state.get("lives"), "level": state.get("level"), "board": state.get("board"),
                "max_level": self.max_level, "boards_completed": self.boards_completed,
                "jump_presses": self.jump_presses, "raw_frames": self.raw_frames,
                "jump_penalty_weight": self.jump_penalty_weight,
                "jump_penalty_reward": self.jump_penalty_reward,
                "jump_penalty_return": self.jump_penalty_return,
                "climb_weight": self.climb_weight,
                "climb_reward": self.climb_reward,
                "climb_return": self.climb_return,
                "v6_weights": dict(self._v6_weights),
                "v6_step": dict(self._v6_step),
                "v6_returns": dict(self._v6_returns),
                "game_fps": self.host.fps, "episode_seconds_game": self.raw_frames / self.host.fps,
                "reset_noops": getattr(self, "reset_noops", 0), "events": list(self._events),
                "telemetry": {"source": "read-only Z80 work RAM; evaluator only", "score": "60b2..60b4 little-endian BCD", "lives": "6228", "game_mode": "6005", "substate": "600a",
                              "airborne": "6216", "on_ladder": "6215", "hammer_active": "6217",
                              "barrels": "6700 + i*20 (active==1, +3 x, +5 y)"}}

    def render(self):
        return self.host.frame.copy()

    def drain_audio(self):
        """Buffered stereo int16 samples since the last drain; empty unless audio=True."""
        return self.host.drain_audio()

    def close(self):
        self.host.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "evidence" / "environment-check.json")
    args = parser.parse_args()
    env = ArcadeDonkeyKongEnv(decision_repeat=args.repeat)
    try:
        observation, info = env.reset(seed=args.seed)
        Image.fromarray(env.render()).save(ROOT / "evidence" / "new-game.png")
        assert observation.shape == (1, 84, 84) and info["score"] == 0 and info["lives"] == 3
        episodes, points, frames = [], 0.0, 0
        life_losses = 0
        started = time.perf_counter()
        for decision in range(args.steps):
            previous_frames = info["raw_frames"]
            observation, reward, done, truncated, info = env.step(env.action_space.sample())
            frames += info["raw_frames"] - previous_frames
            points += reward * 100
            assert abs(points - info["score"]) < 1e-6
            life_losses += sum(event["type"] == "life_lost" for event in info["events"])
            if info["score"] > 0 and not (ROOT / "evidence" / "scored-game.png").exists():
                Image.fromarray(env.render()).save(ROOT / "evidence" / "scored-game.png")
            if done or truncated:
                episodes.append({**info, "terminated": done, "truncated": truncated})
                Image.fromarray(env.render()).save(ROOT / "evidence" / "game-over.png")
                observation, info = env.reset()
                points = 0.0
        elapsed = time.perf_counter() - started
        result = {"game": "Donkey Kong (US set 2)", "set": "dkongo", "core": env.host.system_info.library_name.decode(),
                  "core_version": env.host.system_info.library_version.decode(), "ram_bytes": len(env.host.ram),
                  "core_sha256": hashlib.sha256(env.host.core_path.read_bytes()).hexdigest(),
                  "decisions": args.steps, "raw_frames": frames, "wall_seconds": elapsed,
                  "decisions_per_second": args.steps / elapsed, "life_losses": life_losses,
                  "episodes": episodes, "last_episode": info,
                  "checks": {"observation_shape": True, "native_start_3_lives_zero_score": True, "reward_matches_native_score": True}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
