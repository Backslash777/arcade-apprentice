"""Original arcade Ms. Pac-Man. Pixels/actions are the only policy observations.

FBNeo d_pacman.cpp maps CPU $4c00-$4fff to exposed RAM offset $400-$7ff.
Original score routine $2a5a uses three BCD bytes $4e80-$4e82.
Mode $4e00=3 means gameplay; substate $4e04=7 is native game over.
Sources retained in evidence/original_d_pacman.cpp and evidence/original_pacman-bart.asm.

Reward scheme: Jonny's 2026-10-05 specification (WORK_LOG, same day). Shaped
event rewards REPLACE native-score/100 as the learning signal, and the
powered-pill penalty era is scrapped with it. Native score accounting is never
touched by shaping and remains the metric of record (info["score"]).

Telemetry (validated before any use; shaping/logging ONLY, never policy
observations - the input contract is unchanged):
  powered flags 0x5A6/0x5A7/0x5A9/0x5AA: pillhunt pass 2, 16/16 pill events,
    0 false reads at 23 life losses (tools/pillhunt_pass2.json).
  pacman tile 0x539/0x53A and ghost tiles 0x50A-0x511 (red/pink/blue/orange
    Y,X): tools/reward_telemetry_live.json - spawn-consistent across resets and
    life starts, and 6/6 ghost-eat frames had a ghost within Manhattan 1 tile.
    The X axis is mirrored in RAM; Manhattan distances are unaffected.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import gymnasium as gym
import numpy as np
from PIL import Image
from libretro_host import LibretroHost

ROOT = Path(__file__).resolve().parent
GAME_NAME = "Ms Pac-Man"
ACTION_NAMES = ("noop", "up", "down", "left", "right")
ACTION_BUTTONS = ((), (4,), (5,), (6,), (7,))
REVERSE_ACTION = {1: 2, 2: 1, 3: 4, 4: 3}

POWERED_FLAG_OFFSETS = (0x5A6, 0x5A7, 0x5A9, 0x5AA)
GHOST_EDIBLE_OFFSETS = (0x5A7, 0x5A8, 0x5A9, 0x5AA)  # red/pink/blue/orange; mixed states occur (944 frames, live dump)
PAC_POS_OFFSETS = (0x539, 0x53A)
GHOST_POS_OFFSETS = ((0x50A, 0x50B), (0x50C, 0x50D), (0x50E, 0x50F), (0x510, 0x511))

# Jonny's reward weights, 2026-10-05. Positives are per event detected in a
# native frame; penalties are per decision (the agent's actual step unit).
REWARD_PELLET = 1.0
REWARD_POWER_PILL = 5.0
REWARD_BOARD_CLEAR = 100.0
REWARD_FRUIT = 15.0
GHOST_REWARDS = {200: 20.0, 400: 40.0, 800: 80.0, 1600: 160.0}
# Ms Pac-Man fruit table (cherry/strawberry/orange/pretzel/pear/apple/banana).
# 300/3000 are original Pac-Man values, NOT this ROM. Empirical to date:
# 100 x69, 200 x1 in training logs; never 300/3000 (WORK_LOG 2026-10-06).
FRUIT_PAYOUTS = frozenset((100, 200, 500, 700, 1000, 2000, 5000))
PENALTY_DEATH = 100.0
PENALTY_PROXIMITY = 0.1
PENALTY_BACKTRACK = 0.5
PENALTY_STALL = 0.02
PROXIMITY_RADIUS_TILES = 5
GHOST_EAT_ADJACENCY_TILES = 2
BACKTRACK_DRY_DECISIONS = 8


def bcd(value):
    value = int(value)
    if (value & 15) > 9 or (value >> 4) > 9:
        raise ValueError(f"Invalid native score BCD byte {value:02x}")
    return (value >> 4) * 10 + (value & 15)


def native_score(ram):
    return bcd(ram[0x680]) + 100 * bcd(ram[0x681]) + 10000 * bcd(ram[0x682])


def score_delta(old, new):
    delta = new - old
    if delta < 0 and old >= 990000 and new < 10000:
        delta += 1_000_000
    if delta < 0 or delta > 10000:
        raise RuntimeError(f"Unexpected native score change {old} -> {new}")
    return delta


def guarded_score_delta(old, new):
    """score_delta with the torn-BCD-read guard (production path).

    The core writes the three BCD score bytes low-first within a frame; a read
    landing mid-write shows a stale high byte as a one-frame DECREASE at carry
    boundaries (Space Invaders folder measured this 2026-10-05/06: ten false
    trainer deaths, all 70-90 decreases at carries, zero real anomalies). The
    native arcade score never decreases legitimately (the million rollover is
    handled separately), so a mid-game decrease clamps to delta 0 and reports a
    blip; the caller freezes the accepted baseline and the next clean frame
    reconciles exactly. Implausible jumps and invalid BCD stay fatal.
    """
    delta = new - old
    if delta < 0 and old >= 990000 and new < 10000:
        delta += 1_000_000
    if delta < 0:
        return 0, {"accepted": old, "torn_read": new}
    if delta > 10000:
        raise RuntimeError(f"Unexpected native score change {old} -> {new}")
    return delta, None


def decompose_frame(delta, powered, min_ghost_dist):
    """Split one native frame's score delta into reward events.

    Per frame the game posts at most one 50-point pill, one ghost payout
    (200/400/800/1600 while powered), one fruit, and a few 10-point pellets.
    400/800/1600 are never fruit values, so they are unambiguous ghosts.
    The only genuinely confounded value is 200 (first ghost vs strawberry):
    validated position adjacency decides it (6/6 real eats had a ghost within
    one tile; tools/reward_telemetry_live.json). An exact fruit-table value is
    NEVER re-read as a ghost combination - that was the item-16 defect
    (decompose(500, powered, adjacent) wrongly paid ghost400+fruit100).
    Combined frames (250 = pill+ghost, 300 = ghost+cherry) decompose greedily.
    Residual ambiguity, documented honestly: a strawberry eaten while a ghost
    happens to be adjacent during fright reads as a ghost (rare; unmeasured).
    Unclassifiable remainders are counted as anomalies and earn nothing.
    """
    result = {"pellets": 0, "pills": 0, "ghost": 0, "fruit": 0, "anomaly": 0}
    if delta <= 0:
        return result
    remainder = delta
    if remainder % 100 >= 50:
        result["pills"] = 1
        remainder -= 50
    result["pellets"] = (remainder % 100) // 10
    remainder -= result["pellets"] * 10
    if remainder <= 0:
        return result
    if remainder == 200:
        # The confounded case: adjacency is the only evidence that decides it.
        if powered and min_ghost_dist <= GHOST_EAT_ADJACENCY_TILES:
            result["ghost"] = 200
        else:
            result["fruit"] = 200
        return result
    if remainder in FRUIT_PAYOUTS:
        result["fruit"] = remainder
        return result
    if powered and min_ghost_dist <= GHOST_EAT_ADJACENCY_TILES:
        for payout in (1600, 800, 400, 200):
            rest = remainder - payout
            if rest == 0:
                result["ghost"] = payout
                return result
            if rest in FRUIT_PAYOUTS:
                result["ghost"] = payout
                result["fruit"] = rest
                return result
    result["anomaly"] = remainder
    return result


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class ScoreGuard:
    """Torn-BCD guard with a bounded sustained-disagreement policy (item 28).

    A one-frame tear clamps to delta 0 and freezes the accepted baseline; the
    next clean frame reconciles exactly. A SUSTAINED disagreement cannot hide
    inside that clamp: more than MAX_CONSECUTIVE_BLIPS blips running, or more
    than MAX_EPISODE_BLIPS in one episode, raises instead of clamping forever.
    Rollover (+1,000,000) passes through; implausible jumps and invalid BCD
    stay fatal. Zero score_blip events have ever been recorded on this ROM
    (checked 2026-10-06), so this policy exists so a real fault stays visible.
    """
    MAX_CONSECUTIVE_BLIPS = 5
    MAX_EPISODE_BLIPS = 100

    def __init__(self):
        self.consecutive = 0
        self.episode_total = 0

    def update(self, old, new):
        delta = new - old
        if delta < 0 and old >= 990000 and new < 10000:
            delta += 1_000_000
        if delta < 0:
            self.consecutive += 1
            self.episode_total += 1
            if self.consecutive > self.MAX_CONSECUTIVE_BLIPS or self.episode_total > self.MAX_EPISODE_BLIPS:
                raise RuntimeError(
                    f"Sustained native score-read disagreement {old} -> {new} "
                    f"({self.consecutive} consecutive, {self.episode_total} this episode)")
            return 0, {"accepted": old, "torn_read": new}
        if delta > 10000:
            raise RuntimeError(f"Unexpected native score change {old} -> {new}")
        self.consecutive = 0
        return delta, None


def pac_closes_on_ghost(prev_pac, pac, ghosts, dangerous_mask, radius):
    """Proximity (item 17): pac's OWN step closed on a dangerous ghost.

    Compares distances to each dangerous ghost at THIS frame using pac's old
    and new tile: a ghost walking up to a retreating pac does NOT count, and
    neither does pac standing still. Manhattan ignores walls/tunnel routes
    (documented limitation; tile units, mirrored X cancels out).
    """
    for gi, ghost in enumerate(ghosts):
        if not dangerous_mask[gi]:
            continue
        if manhattan(pac, ghost) <= radius and manhattan(pac, ghost) < manhattan(prev_pac, ghost):
            return True
    return False


def executed_backtrack(prev_prev_pac, prev_pac, pac, prev_action, action):
    """Backtrack (item 17): a reversal that actually happened.

    Requested button reversed AND pac moved last decision AND is now back on
    the tile it came from. A blocked reversal (positions frozen) is not an
    executed backtrack. Whether the destination has pellets is approximated by
    the caller's dry window (no tile pellet map is validated).
    """
    if prev_action not in REVERSE_ACTION or action != REVERSE_ACTION[prev_action]:
        return False
    return prev_pac != prev_prev_pac and pac == prev_prev_pac


def stall_kind(prev_pac, pac, ate_anything, powered):
    """Stall (item 17): stationary vs cleared-ground travel, logged separately.

    Jonny's rule charges both standing still and traversing ground with
    nothing eaten while unpowered; the two kinds are counted separately so the
    intended travel cost and genuine parking are distinguishable in reports.
    Tile-granular: sub-tile motion within one tile counts as stationary.
    """
    if powered or ate_anything:
        return None
    return "stationary" if pac == prev_pac else "travel"


class ArcadeMsPacmanEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, decision_repeat=4, root=None, core_path=None, rom_path=None,
                 render_mode=None, max_episode_frames=0, reset_noop_max=60, audio=False):
        super().__init__()
        self.root = Path(root or ROOT)
        self.decision_repeat = int(decision_repeat)
        if not 1 <= self.decision_repeat <= 8:
            raise ValueError("decision_repeat must be1..8")
        self.render_mode = render_mode
        self.max_episode_frames = int(max_episode_frames)
        self.reset_noop_max = int(reset_noop_max)
        self.action_space = gym.spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = gym.spaces.Box(0, 255, (1, 84, 84), np.uint8)
        self.host = LibretroHost(core_path or self.root / "vendor" / "fbneo_libretro.dll",
                                 rom_path or self.root / "roms" / "mspacman.zip", self.root, audio=audio)
        self.sample_rate = self.host.sample_rate
        self._done = True
        self.raw_frames = self.score = self.dots_eaten = self.boards_cleared = 0
        self._reset_episode_state()

    def _reset_episode_state(self):
        self.shaped_reward_total = 0.0
        self.ghost_reward_total = 0.0
        self.reward_counts = {"pellets": 0, "power_pills": 0, "ghosts": 0, "fruits": 0, "board_clears": 0}
        self.penalty_counts = {"deaths": 0, "proximity": 0, "backtrack": 0, "stall": 0,
                               "stall_stationary": 0, "stall_travel": 0}
        self.anomaly_total = 0
        self.score_blips = 0
        self.score_guard = ScoreGuard()
        self._events = []
        self._previous = {}
        self._eyes = [False] * 4        # env-local eaten-ghost tracking (validated positions)
        self._ghost_spawn = None        # captured at the first running frame of an episode
        self._reset_histories()

    def _reset_histories(self):
        """No stale first-step penalties after a death, board change or reset (item 18)."""
        self._prev_pac = None
        self._prev_prev_pac = None
        self._prev_action = 0
        self._dry_decisions = 0         # decisions since the last pellet/pill was eaten

    def _read(self):
        r = self.host.ram
        return {"display_score": native_score(r), "game_mode": int(r[0x600]),
                "game_substate": int(r[0x604]), "lives": int(r[0x614]),
                "reserve_lives": int(r[0x615]), "board": int(r[0x613]) + 1,
                "board_dots": int(r[0x60e]),
                "pac": (int(r[PAC_POS_OFFSETS[0]]), int(r[PAC_POS_OFFSETS[1]])),
                "ghosts": tuple((int(r[y]), int(r[x])) for y, x in GHOST_POS_OFFSETS),
                "ghost_edible": tuple(bool(int(r[o])) for o in GHOST_EDIBLE_OFFSETS),
                "powered": any(int(r[offset]) for offset in POWERED_FLAG_OFFSETS)}

    def _obs(self, frame=None):
        image = Image.fromarray(self.host.frame if frame is None else frame)
        return np.asarray(image.convert("L").resize((84, 84), Image.Resampling.BILINEAR), dtype=np.uint8)[None].copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        self.host.reset()
        self.reset_noops = int(self.np_random.integers(self.reset_noop_max + 1))
        for _ in range(500 + self.reset_noops):
            self.host.run(capture=False)
        for _ in range(6):
            self.host.run((2,), capture=False)
        for _ in range(30):
            self.host.run(capture=False)
        for _ in range(6):
            self.host.run((3,), capture=False)
        for _ in range(900):
            self.host.run()
            current = self._read()
            if current["game_mode"] == 3 and current["game_substate"] == 3:
                break
        else:
            raise RuntimeError(f"Native coin/start did not reach gameplay: {current}")
        if current["display_score"] != 0 or current["lives"] != 3:
            raise RuntimeError(f"Unexpected initial native score/lives: {current}")
        self.raw_frames = self.score = self.dots_eaten = self.boards_cleared = 0
        self._reset_episode_state()
        self._previous = current
        self._ghost_spawn = current["ghosts"]  # validated: identical at every life start
        self._done = False
        return self._obs(), self._info()

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset before stepping a new game")
        if not self.action_space.contains(action):
            raise ValueError(f"Unknown action {action}")
        action = int(action)
        points = 0
        reward = 0.0
        ate = {"pellets": 0, "pills": 0, "ghost": 0, "fruit": 0}
        deaths = 0
        frames = []
        self._events = []
        terminated = False
        running = False
        for frame_index in range(self.decision_repeat):
            capture = frame_index >= self.decision_repeat - 2
            self.host.run(ACTION_BUTTONS[action], capture=capture)
            self.raw_frames += 1
            now = self._read()
            old = self._previous
            delta, blip = self.score_guard.update(old["display_score"], now["display_score"])
            if blip is not None:
                self.score_blips += 1
                self._events.append({"type": "score_blip", "frame": self.raw_frames,
                                     "accepted": blip["accepted"], "torn_read": blip["torn_read"]})
                now["display_score"] = old["display_score"]  # freeze baseline; next clean frame reconciles
            points += delta
            self.score += delta
            running = now["game_mode"] == 3 and now["game_substate"] == 3
            if running:
                min_ghost_dist = min(manhattan(now["pac"], ghost) for ghost in now["ghosts"])
                comp = decompose_frame(delta, old["powered"], min_ghost_dist)
                if comp["anomaly"]:
                    self.anomaly_total += comp["anomaly"]
                ate["pellets"] += comp["pellets"]
                ate["pills"] += comp["pills"]
                self.reward_counts["pellets"] += comp["pellets"]
                self.reward_counts["power_pills"] += comp["pills"]
                reward += comp["pellets"] * REWARD_PELLET + comp["pills"] * REWARD_POWER_PILL
                if comp["ghost"]:
                    ate["ghost"] += 1
                    self.reward_counts["ghosts"] += 1
                    ghost_reward = GHOST_REWARDS[comp["ghost"]]
                    reward += ghost_reward
                    self.ghost_reward_total += ghost_reward
                    # eyes: the adjacent ghost that paid out is harmless until home
                    nearest = min(range(4), key=lambda gi: manhattan(now["pac"], now["ghosts"][gi]))
                    self._eyes[nearest] = True
                    self._events.append({"type": "ghost_eaten", "frame": self.raw_frames,
                                         "payout": comp["ghost"], "reward": ghost_reward,
                                         "lives": now["lives"], "score": self.score})
                if comp["fruit"]:
                    ate["fruit"] += 1
                    self.reward_counts["fruits"] += 1
                    reward += REWARD_FRUIT
                    self._events.append({"type": "fruit_eaten", "frame": self.raw_frames,
                                         "payout": comp["fruit"], "score": self.score})
                # eyes reaching the ghost house revive (validated spawn slots)
                if self._ghost_spawn is not None:
                    for gi in range(4):
                        if self._eyes[gi] and manhattan(now["ghosts"][gi], self._ghost_spawn[gi]) <= 2:
                            self._eyes[gi] = False
            if now["board"] != old["board"]:
                self.boards_cleared += 1
                reward += REWARD_BOARD_CLEAR
                self._reset_histories()  # new maze: no stale first-step penalties (item 18)
                self._events.append({"type": "board_clear", "frame": self.raw_frames, "board": old["board"], "score": self.score})
            if now["board_dots"] >= old["board_dots"]:
                self.dots_eaten += now["board_dots"] - old["board_dots"]
            if now["lives"] < old["lives"]:
                deaths += old["lives"] - now["lives"]
                self._eyes = [False] * 4  # positions reset with the new life
                self._reset_histories()   # no stale danger/backtrack/dry history (item 18)
                self._events.append({"type": "life_lost", "frame": self.raw_frames, "lives": now["lives"], "score": self.score})
            terminated = (now["game_mode"] != 3 or (now["game_substate"] == 7 and now["lives"] == 0))
            self._previous = now
            if capture:
                frames.append(self.host.frame)
            if terminated:
                self._events.append({"type": "game_over", "frame": self.raw_frames, "score": self.score})
                break
        # --- per-decision flow penalties (Jonny's spec; decision = the agent's step) ---
        now = self._previous
        if running and not terminated:
            if ate["pellets"] or ate["pills"]:
                self._dry_decisions = 0
            else:
                self._dry_decisions += 1
            prev_pac = self._prev_pac
            if prev_pac is not None:
                # proximity: pac's own step closed on a dangerous ghost (item 17).
                # danger is PER-GHOST (item 18): not edible and not tracked eyes;
                # mixed frightened/dangerous states occur and are handled individually.
                dangerous = [not now["ghost_edible"][gi] and not self._eyes[gi] for gi in range(4)]
                if any(dangerous) and pac_closes_on_ghost(prev_pac, now["pac"], now["ghosts"],
                                                          dangerous, PROXIMITY_RADIUS_TILES):
                    reward -= PENALTY_PROXIMITY
                    self.penalty_counts["proximity"] += 1
                # stall: stationary vs cleared-ground travel, counted separately (item 17)
                kind = stall_kind(prev_pac, now["pac"],
                                  (ate["pellets"] + ate["pills"] + ate["ghost"] + ate["fruit"]) > 0,
                                  now["powered"])
                if kind:
                    reward -= PENALTY_STALL
                    self.penalty_counts["stall"] += 1
                    self.penalty_counts["stall_stationary" if kind == "stationary" else "stall_travel"] += 1
                # backtrack: an executed reversal into a dry trail (item 17)
                if (self._dry_decisions >= BACKTRACK_DRY_DECISIONS
                        and executed_backtrack(self._prev_prev_pac, prev_pac, now["pac"],
                                               self._prev_action, action)):
                    reward -= PENALTY_BACKTRACK
                    self.penalty_counts["backtrack"] += 1
            self._prev_prev_pac = prev_pac
            self._prev_pac = now["pac"]
            self._prev_action = action
        if deaths:
            reward -= deaths * PENALTY_DEATH
            self.penalty_counts["deaths"] += deaths
        self.reward_counts["board_clears"] = self.boards_cleared
        truncated = self.max_episode_frames > 0 and self.raw_frames >= self.max_episode_frames and not terminated
        self._done = terminated or truncated
        self.shaped_reward_total += reward
        merged = np.maximum(frames[-2], frames[-1]) if len(frames) >= 2 else self.host.frame
        return self._obs(merged), reward, terminated, truncated, self._info()

    def _info(self):
        return {"score": self.score, "display_score": self._previous.get("display_score", 0),
                "lives": self._previous.get("lives", 0), "board": self._previous.get("board", 1),
                "dots_eaten": self.dots_eaten, "boards_cleared": self.boards_cleared,
                "raw_frames": self.raw_frames, "events": list(self._events), "game_fps": self.host.fps,
                "game_mode": self._previous.get("game_mode"), "game_substate": self._previous.get("game_substate"),
                "powered": self._previous.get("powered", False),
                "shaped_reward_total": round(self.shaped_reward_total, 6),
                "ghost_reward_total": round(self.ghost_reward_total, 6),
                "reward_counts": dict(self.reward_counts),
                "penalty_counts": dict(self.penalty_counts),
                "anomaly_total": self.anomaly_total,
                "score_blips": self.score_blips,
                "reset_noops": self.reset_noops,
                "telemetry": "read-only native RAM; shaped rewards from validated score-delta decomposition "
                             "and validated position telemetry; RAM never enters policy observations; "
                             "native score accounting untouched by shaping"}

    def render(self):
        return self.host.frame.copy()

    def drain_audio(self):
        """Buffered stereo int16 samples since the last drain; empty unless audio=True."""
        return self.host.drain_audio()

    def close(self):
        self.host.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=ROOT / "evidence" / "environment_smoke.json")
    args = parser.parse_args()
    env = ArcadeMsPacmanEnv()
    try:
        obs, info = env.reset(seed=args.seed)
        assert obs.shape == (1,84,84) and obs.dtype == np.uint8
        Image.fromarray(env.render()).save(ROOT / "evidence" / "native_start.png")
        started = time.perf_counter()
        frames = 0
        episode_reward = 0.0
        episodes = []
        for step in range(args.steps):
            old_frames = info["raw_frames"]
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            frames += info["raw_frames"] - old_frames
            episode_reward += reward
            # Shaped reward reconciles exactly with its separately-logged components;
            # native score accounting is untouched by any of it.
            counts, penalties = info["reward_counts"], info["penalty_counts"]
            components = (counts["pellets"] * REWARD_PELLET + counts["power_pills"] * REWARD_POWER_PILL
                          + info["ghost_reward_total"] + counts["fruits"] * REWARD_FRUIT
                          + counts["board_clears"] * REWARD_BOARD_CLEAR
                          - penalties["deaths"] * PENALTY_DEATH - penalties["proximity"] * PENALTY_PROXIMITY
                          - penalties["backtrack"] * PENALTY_BACKTRACK - penalties["stall"] * PENALTY_STALL)
            assert abs(info["shaped_reward_total"] - components) < 1e-4
            if step == 300:
                Image.fromarray(env.render()).save(ROOT / "evidence" / "native_play.png")
            if terminated or truncated:
                assert terminated and not truncated
                assert info["lives"] == 0
                assert abs(episode_reward - info["shaped_reward_total"]) < 1e-4
                episodes.append(info)
                Image.fromarray(env.render()).save(ROOT / "evidence" / "native_game_over.png")
                obs, info = env.reset()
                episode_reward = 0.0
        elapsed = time.perf_counter() - started
        result = {"game": GAME_NAME, "steps": args.steps, "raw_frames": frames, "wall_seconds": elapsed,
                  "decisions_per_second": args.steps / elapsed, "native_frames_per_second": frames / elapsed,
                  "episodes": episodes, "last_episode": info,
                  "checks": {"pixels_only_shape": True, "shaped_reward_reconciles": True,
                             "complete_native_games": len(episodes), "native_three_life_start": True}}
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
