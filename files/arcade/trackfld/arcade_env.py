"""Original Konami Track & Field with native inputs and pixel observations.

Driver layout is verified at runtime against libretro's serialized NVRAM.
Score/mode/event reads are diagnostics, never agent observations. No RAM writes.
"""
from __future__ import annotations

import ctypes as C
from pathlib import Path
import gymnasium as gym
import numpy as np
from PIL import Image
from libretro_host import LibretroHost
from progress_shaping import RaceProgressShaper
from held_button_penalty import HeldButtonPenalty
from event_rewards import EventRewardMachine

ROOT = Path(__file__).resolve().parent
EVENT_NAMES = ("100 metre dash", "long jump", "javelin throw", "110 metre hurdles", "hammer throw", "high jump")
ACTION_NAMES = ("noop", "run_left", "run_right", "both_run", "jump_throw", "run_left_jump", "run_right_jump", "all_buttons", "run_spam")
# FBNeo maps original buttons 1/3/2 to libretro B/Y/A. Raw actions have no automatic pulses; run_spam is this experiment's explicit macro.
ACTION_BUTTONS = tuple(tuple(button for bit, button in enumerate((0, 1, 8)) if action & (1 << bit)) for action in range(8)) + (None,)
RUN_SPAM_ACTION = len(ACTION_NAMES) - 1
# Macro cadence: left 2 frames, release 2, right 2, release 2. Native probes
# showed direct left/right without a gap stalls and two-frame releases are
# reliable, so every macro press is followed by a full release.
RUN_SPAM_SCHEDULE = ((0,), (), (1,), ())
RUN_SPAM_AS_RAW = {(): 0, (0,): 1, (1,): 2}
REWARD_SCALE = 1000.0


def bcd(byte):
    byte = int(byte)
    if byte >> 4 > 9 or (byte & 15) > 9:
        raise RuntimeError(f"Invalid native score BCD byte {byte:02x}")
    return 10 * (byte >> 4) + (byte & 15)


def score_delta(previous, current):
    """Returns the score delta, or None for a torn mid-write BCD read.
    The core writes the low score byte before the high byte; a read at a
    hundred-carry frame can show a phantom decrease for one frame (SI desk
    evidence: 10 runs killed by this ghost, all 70-90 blips at carries).
    Never die on a blip: caller freezes the baseline and logs score_blip.
    Fatal guards stay for implausible jumps and invalid BCD.
    """
    change = current - previous
    if change < 0:
        if previous >= 9_000_000 and current <= 1_000_000:
            change += 10_000_000
        else:
            return None  # torn read; NOT a real decrease (score never falls)
    if change > 1_000_000:
        raise RuntimeError(f"Unexpected per-frame score increase: {change}")
    return change


class ArcadeTrackFieldEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}
    # class-level v1 defaults: test fixtures that bypass __init__ stay on the v1 path
    event_machine = None
    event_reward = 0.0

    def __init__(self, decision_repeat=2, root=None, render_mode=None, max_episode_frames=0,
                 reset_noop_max=8, shaping_weight=0.25, shaping_gamma=0.999, audio=False,
                 first_race_only=False, hold_penalty_weight=0.0, hold_penalty_include_jump=False,
                 reward_scheme="v1", wr_benchmark_path=None, core_path=None, rom_path=None):
        super().__init__()
        self.root = Path(root or ROOT)
        self.decision_repeat = int(decision_repeat)
        if not 1 <= self.decision_repeat <= 8:
            raise ValueError("decision_repeat must be 1..8")
        self.max_episode_frames = int(max_episode_frames)
        self.reset_noop_max = int(reset_noop_max)
        self.render_mode = render_mode
        self.first_race_only = bool(first_race_only)
        if reward_scheme not in ("v1", "v3"):
            raise ValueError("reward_scheme must be 'v1' or 'v3'")
        self.reward_scheme = reward_scheme
        self.curriculum_end = self.native_game_over = False
        self.episode_end_reason = None
        self.shaper = RaceProgressShaper(shaping_weight, shaping_gamma)
        self.hold_penalty = HeldButtonPenalty(hold_penalty_weight, hold_penalty_include_jump)
        self._macro_phase = 0
        self.native_reward = self.shaping_reward = 0.0
        self.hold_penalty_reward = 0.0
        self.action_space = gym.spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = gym.spaces.Box(0, 255, (1, 84, 84), dtype=np.uint8)
        self.host = LibretroHost(core_path or self.root / "vendor" / "fbneo_libretro.dll",
                                 rom_path or self.root / "roms" / "trackfld.zip", self.root,
                                 variables={"fbneo-dipswitch-trackfld-World_Records": "Erase on Reset"}, audio=audio)
        self.sample_rate = self.host.sample_rate
        # d_trackfld.cpp MemIndex: DrvNVRAM[0x800] immediately precedes
        # AllRam[0x2900]. CPU 0x2800..0x2fff maps that NVRAM region. Reading a
        # source-defined allocation here avoids serializing the core each frame.
        if len(self.host.ram) != 0x2900:
            self.close()
            raise RuntimeError("Unexpected Track & Field RAM layout; refusing guessed telemetry")
        self.nv = np.ctypeslib.as_array((C.c_ubyte * 0x800).from_address(self.host.dll.retro_get_memory_data(2) - 0x800))
        self.nv.flags.writeable = False
        # item 22: evaluation passes its own benchmark path so eval games can
        # never mutate the training reward state (frozen reward state).
        benchmark = Path(wr_benchmark_path) if wr_benchmark_path is not None else self.root / "runtime" / "wr_benchmark.json"
        self.event_machine = (EventRewardMachine(fps=float(self.host.fps), benchmark_path=benchmark)
                              if reward_scheme == "v3" else None)
        self.event_reward = 0.0
        self._done = True
        self.score = self.raw_frames = self.events_completed = 0
        self.events_reached = []
        self._events = []

    def _native_score(self):
        # P1 cumulative score is three BCD bytes in tens, CPU 2a80..2a82.
        return (10000 * bcd(self.nv[0x280]) + 100 * bcd(self.nv[0x281]) + bcd(self.nv[0x282])) * 10

    def _read(self):
        return {"mode": int(self.nv[0]), "phase": int(self.nv[3]),
                "stage": int(self.nv[6]), "player_active": int(self.nv[0x81]),
                "event_index": int(self.nv[0x84]), "display_score": self._native_score(),
                # Empirically validated against legal-input movement, camera scroll,
                # stopping and the finish. See evidence/progress-probe and guide.
                "race_distance_units": int(self.nv[0xab]) * 256 + int(self.nv[0xac])}

    def _obs(self, rgb=None):
        return np.asarray(Image.fromarray(self.host.frame if rgb is None else rgb).convert("L").resize(
            (84, 84), Image.Resampling.BILINEAR), dtype=np.uint8)[None].copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        self.host.reset()
        for _ in range(300):
            self.host.run(capture=False)
        if bytes(self.nv) != self.host.save_state()[-0x800:]:
            raise RuntimeError("NVRAM view disagrees with native serialization; cannot trust game state")
        for _ in range(6):
            self.host.run((2,), capture=False)
        for _ in range(30):
            self.host.run(capture=False)
        for _ in range(6):
            self.host.run((3,), capture=False)
        for _ in range(120):
            self.host.run(capture=False)
        # Accept the cabinet's default name via real controls, then wait
        # neutrally through READY / SET. Learning begins at the first gun.
        self.host.run((0,), capture=False)
        self.host.run(capture=False)
        self.host.run((8,), capture=False)
        for _ in range(1800):
            self.host.run(capture=False)
            if self.nv[0] == 3 and self.nv[3] == 2 and self.nv[6] == 1:
                break
        else:
            raise RuntimeError("Coin/name/start sequence failed to reach the first race")
        self.reset_noops = int(self.np_random.integers(self.reset_noop_max + 1))
        for _ in range(self.reset_noops):
            self.host.run(capture=False)
        self.host.run()
        self._previous = self._read()
        self.shaper.reset(self._previous)
        self.hold_penalty.reset()
        self.event_reward = 0.0
        if self.event_machine is not None:
            self.event_machine.reset(self._previous)
        self._macro_phase = 0
        self.native_reward = self.shaping_reward = 0.0
        self.hold_penalty_reward = 0.0
        if self._previous["display_score"] != 0 or self._previous["event_index"] != 0:
            raise RuntimeError("New-game invariants failed: expected first event and zero score")
        self.raw_frames = self.score = self.events_completed = 0
        self.events_reached = [0]
        self.current_event = 0
        self._pending_event = None
        self._pending_event_frames = 0
        self._events = []
        self._done = False
        self.curriculum_end = self.native_game_over = False
        self.episode_end_reason = None
        return self._obs(), self._info()

    def step(self, action):
        if self._done:
            raise RuntimeError("reset() required before step() or after the game ends")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid raw-button action {action}")
        if self.first_race_only and int(action) not in (0, 1, 2, 3):
            raise ValueError("Running Focus permits only raw actions 0..3; jump is disabled")
        self._events = []
        points = 0
        self.event_reward = 0.0
        frames = []
        terminated = False
        action = int(action)
        if action == RUN_SPAM_ACTION:
            buttons = RUN_SPAM_SCHEDULE[self._macro_phase % len(RUN_SPAM_SCHEDULE)]
            self._macro_phase += 1
        else:
            buttons = ACTION_BUTTONS[action]
        index = 0
        while index < self.decision_repeat or (self.first_race_only and self._pending_event is not None):
            capture = index >= self.decision_repeat - 2
            # Once the raw event counter leaves the race, confirm it inside this
            # terminal transition with neutral inputs. No later-event observation
            # is handed back for another learner action during the ten-frame guard.
            confirming = self.first_race_only and self._pending_event is not None
            self.host.run(() if confirming else buttons, capture=capture)
            index += 1
            self.raw_frames += 1
            now = self._read()
            delta = score_delta(self._previous["display_score"], now["display_score"])
            if delta is None:
                # Freeze the accepted baseline (never store the torn value),
                # log loudly, carry on; the next clean frame reconciles exactly.
                self._events.append({"type": "score_blip", "frame": self.raw_frames,
                                     "torn": now["display_score"], "kept": self._previous["display_score"]})
                now["display_score"] = self._previous["display_score"]
                delta = 0
            points += delta
            self.score += delta
            if delta:
                self._events.append({"type": "score_award", "frame": self.raw_frames, "points": delta,
                                     "score": self.score, "event_index": self._previous["event_index"]})
            # Native mode 3 is active gameplay, 4 is end-of-game rankings, 1
            # is attract. Event transitions and individual fouls remain live.
            terminated = now["mode"] != 3
            event_index = now["event_index"]
            if not terminated and now["player_active"] > 0 and event_index != self.current_event and 0 <= event_index < len(EVENT_NAMES):
                # A failed race briefly advances the native event counter one
                # frame before mode becomes game-over. Confirm ten live frames
                # before reporting a reached event. This affects logs only.
                self._pending_event_frames = self._pending_event_frames + 1 if self._pending_event == event_index else 1
                self._pending_event = event_index
                if self._pending_event_frames >= 10:
                    self.events_completed += 1
                    self.events_reached.append(event_index)
                    self._events.append({"type": "event_transition", "frame": self.raw_frames,
                                         "from_event": self.current_event, "event_index": event_index,
                                         "event_name": EVENT_NAMES[event_index], "score": self.score})
                    if self.event_machine is not None:
                        self.event_reward += self.event_machine.on_confirmed_transition()
                    self.current_event = event_index
                    if self.first_race_only:
                        self.curriculum_end = True
                        self.episode_end_reason = "100m_qualified"
            else:
                self._pending_event = None
                self._pending_event_frames = 0
            self._previous = now
            if self.event_machine is not None:
                self.event_reward += self.event_machine.step_frame(now, action)
            if capture:
                frames.append(self.host.frame)
            if terminated:
                self.native_game_over = True
                self.episode_end_reason = "native_game_over"
                self._events.append({"type": "game_over", "frame": self.raw_frames, "score": self.score,
                                     "events_completed": self.events_completed})
                break
            if self.curriculum_end:
                self._events.append({"type": "curriculum_end", "frame": self.raw_frames,
                                     "score": self.score, "reason": self.episode_end_reason,
                                     "native_game_over": False, "events_completed": self.events_completed})
                terminated = True  # Terminal success in the defined 100m task.
                break
        truncated = self.max_episode_frames > 0 and self.raw_frames >= self.max_episode_frames and not terminated
        if truncated:
            self.episode_end_reason = "frame_limit"
        self._done = terminated or truncated
        rgb = np.maximum(frames[-2], frames[-1]) if len(frames) > 1 else self.host.frame
        self.native_reward = points / REWARD_SCALE
        self.shaping_reward = self.shaper.step(self._previous, done=self._done)
        # Exactly once per actor decision, independently of native frame repeat.
        # The soft cost never changes buttons or the cabinet's native score.
        # Macro decisions are accounted as the raw buttons actually emitted;
        # the release between presses makes the legal cadence free.
        self.hold_penalty_reward = self.hold_penalty.step(RUN_SPAM_AS_RAW[buttons] if action == RUN_SPAM_ACTION else action)
        if self.event_machine is not None:
            # v3: event components ARE the learning reward; native, shaping
            # and hold-penalty above stay computed as telemetry only.
            learning_reward = self.event_reward
        else:
            learning_reward = self.native_reward + self.shaping_reward + self.hold_penalty_reward
        return self._obs(rgb), learning_reward, terminated, truncated, self._info()

    def _info(self):
        native = self._previous
        event_index = self.current_event
        info = {"score": self.score, "display_score": native["display_score"],
                "native_reward": self.native_reward, "shaping_reward": self.shaping_reward,
                "hold_penalty_reward": self.hold_penalty_reward,
                "hold_penalty_return": self.hold_penalty.total,
                "hold_penalty_weight": self.hold_penalty.weight,
                "hold_penalty_include_jump": self.hold_penalty.include_jump,
                "last_held_mask": self.hold_penalty.last_held_mask,
                "held_button_count": self.hold_penalty.held_button_count,
                "unreleased_run_count": self.hold_penalty.extra_unreleased_run_count,
                "hold_penalty_units": self.hold_penalty.units,
                "hold_penalty_total_units": self.hold_penalty.total_count,
                "total_held_button_count": self.hold_penalty.total_held_button_count,
                "total_unreleased_run_count": self.hold_penalty.total_unreleased_run_count,
                "shaping_return": self.shaper.total,
                "shaping_positive_return": self.shaper.positive_total,
                "shaping_negative_return": self.shaper.negative_total,
                "shaping_weight": self.shaper.weight, "shaping_potential": self.shaper.previous,
                "race_distance_units": native["race_distance_units"] if native["event_index"] == 0 else None,
                "raw_frames": self.raw_frames, "episode_seconds_game": self.raw_frames / self.host.fps,
                "lives": int(native["player_active"] > 0) if native["mode"] == 3 else 0,
                "event_index": event_index, "event_name": EVENT_NAMES[event_index],
                "native_game_mode": native["mode"],
                "first_race_only": self.first_race_only, "curriculum_end": self.curriculum_end,
                "native_game_over": self.native_game_over, "episode_end_reason": self.episode_end_reason,
                "events_completed": self.events_completed, "events_reached": list(self.events_reached),
                "events": list(self._events), "reset_noops": self.reset_noops,
                "game_fps": self.host.fps,
                "telemetry_source": "read-only CPU RAM; score 2a80..2a82, game mode2800, event2884",
                "event_validation": "100m to long jump and long-jump elimination exercised; later events not yet reached"}
        if self.event_machine is not None:
            info.update({"reward_scheme": "v3",
                         "event_reward": self.event_reward,
                         "event_reward_return": self.event_machine.total(),
                         "event_reward_components": self.event_machine.totals(),
                         "wr_benchmark": self.event_machine.benchmark()})
        return info

    def render(self):
        return self.host.frame.copy()

    def drain_audio(self):
        """Buffered stereo int16 samples since the last drain; empty unless audio=True."""
        return self.host.drain_audio()

    def close(self):
        self.host.close()
