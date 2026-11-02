"""Any arcade game, as a pixel-only Gymnasium environment, driven by a "found" pack (pack.json + start.state).

The pack is made by the "Find the score" window (score_finder.py): it says where the game keeps its score (and,
if found, its lives and a "game is on" byte), which joystick directions and buttons the player used, and holds a
save-state taken just after a game started. Only screen pixels reach the AI; memory is only read, never written.

Signed: Claude, 10 Oct 2026.
"""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

import ramscan
from libretro_host import LibretroHost

# RetroPad button ids (libretro.h): FBNeo puts arcade Fire 1..6 on B, A, Y, X, L, R; coin = SELECT, start = START
UP, DOWN, LEFT, RIGHT = 4, 5, 6, 7
DIRS = {"none": (), "up": (UP,), "down": (DOWN,), "left": (LEFT,), "right": (RIGHT,),
        "up-left": (UP, LEFT), "up-right": (UP, RIGHT), "down-left": (DOWN, LEFT), "down-right": (DOWN, RIGHT)}
BUTTON_NAMES = {0: "B1", 8: "B2", 1: "B3", 9: "B4", 10: "B5", 11: "B6"}
GAME_OVER_FRAMES = 90          # "game on" byte must be off this long (1.5 s) before we call it game over
STUCK_FRAMES = 60 * 60 * 5     # no points for 5 minutes of game time: give up on this game


def action_table(pack):
    """Joystick directions the player used x (no button + each button they used, one at a time)."""
    axes = set(pack.get("axes", ["lr", "ud"]))
    dirs = ["none"]
    if "lr" in axes:
        dirs += ["left", "right"]
    if "ud" in axes:
        dirs += ["up", "down"]
    if {"lr", "ud"} <= axes:
        dirs += ["up-left", "up-right", "down-left", "down-right"]
    buttons = [None] + [int(b) for b in pack.get("buttons", [0])]
    table, names = [], []
    for b in buttons:
        for d in dirs:
            table.append(tuple(DIRS[d]) + ((b,) if b is not None else ()))
            names.append(d if b is None else (BUTTON_NAMES.get(b, f"btn{b}") if d == "none" else f"{d}+{BUTTON_NAMES.get(b, b)}"))
    return table, names


# ALFRED 2 Nov 2026: score readers found to be wrong after they shipped. A pack whose score reader is EXACTLY the old one
# reads with the fixed one - even if its owner changed something else in it (buttons, Show it how), so the updater
# kept their copy, or "Find the score again" wrote the old reader back.
SCORE_FIXES = {
    "shinobi": ({"kind": "bcd", "n": 3, "big": False, "offset": 112687, "swapped": False, "scale": 1},      # missed the
                {"kind": "bcd", "n": 4, "big": True, "offset": 112685, "swapped": True, "pair_base": 1}),   # 10,000s
}


def fix_score(pack):
    fix, s = SCORE_FIXES.get(pack.get("set")), pack.get("score")
    if fix and isinstance(s, dict) and not s.get("pair_base") and all(s.get(k) == v for k, v in fix[0].items()):
        pack = dict(pack, score=dict(s, **fix[1]))
    return pack


def load_pack(pack_dir):
    pack_dir = Path(pack_dir)
    return fix_score(json.loads((pack_dir / "pack.json").read_text(encoding="utf-8")))


class GenericArcadeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, pack_dir, decision_repeat=None, root=None, core_path=None, rom_path=None,
                 render_mode=None, reset_noop_max=30, audio=False):
        super().__init__()
        self.pack_dir = Path(pack_dir)
        self.pack = load_pack(self.pack_dir)
        self.decision_repeat = int(decision_repeat or self.pack.get("decision_repeat", 4))
        self.reset_noop_max = int(reset_noop_max)
        self.render_mode = render_mode
        self.actions, self.action_names = action_table(self.pack)
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.observation_space = gym.spaces.Box(0, 255, (1, 84, 84), dtype=np.uint8)
        self.root = Path(root or self.pack_dir)
        self.host = LibretroHost(core_path, rom_path, self.root, require_ram=False, audio=audio)
        self.from_state = self.host.ram is None       # MAME 2003-Plus: read the memory out of a save-state instead
        if self.from_state and not self.host.dll.retro_serialize_size():
            self.host.close()
            raise RuntimeError("This game doesn't show its memory to the emulator, so its score can't be read.")
        self.mem = None
        self.start_state = (self.pack_dir / "start.state").read_bytes()
        self.reboot = self.pack.get("reset") == "reboot"     # its save-states can't be restored: switch on, coin, start
        self.score_c = self.pack.get("score")
        self.screen = None
        if self.score_c and self.score_c.get("kind") == "screen":     # read the score off the screen, not memory
            from screenscore import ScreenScore
            self.screen = ScreenScore(self.score_c)
            self._seen = 0
        self.health = self.pack.get("health")      # fighting games: points = damage dealt - damage taken
        self.lives_c = self.pack.get("lives")
        self.progress_c = self.pack.get("progress")   # ALFRED 31 Oct 2026: where it keeps how far along it is
        self.screen_progress = False    # ALFRED 1 Nov 2026: no address -> read the scroll off the screen (set by Cabinet)
        self.flag = self.pack.get("playing")
        self.max_jump = int(self.pack.get("max_jump", 100000))
        self.rng = np.random.default_rng()
        self.score = self.lives = 0
        self._prev = None

    # -------------------------------------------------------------------------------------------- reading the game
    def snap(self):
        """The game's memory right now (a live view, or a fresh save-state for cores that don't share memory)."""
        self.mem = (np.frombuffer(self.host.save_state(), np.uint8) if self.from_state else self.host.ram)
        return self.mem

    def read_score(self):
        if self.screen:
            v = self.screen.read(self.host.frame)
            if v is not None:
                self._seen = v
            return self._seen                 # hidden or covered up right now: keep the last reading
        return ramscan.decode(self.mem, self.score_c) if self.score_c else 0

    def read_health(self):
        h = self.health
        return ramscan.decode_health(self.mem, h["mine"]), ramscan.decode_health(self.mem, h["theirs"])

    def read_lives(self):
        if not self.lives_c:
            return 0
        return max(0, ramscan.read_lives(self.mem, self.lives_c))

    def read_progress(self):
        """ALFRED 31 Oct 2026: how far along the level it is (e.g. the screen-scroll counter)."""
        c, m = self.progress_c, self.mem
        o, fmt = int(c["offset"]), c.get("fmt", "u16be")
        if fmt == "u8":
            v, top = int(m[o]), 255
        else:
            a, b = int(m[o]), int(m[o + 1])
            v, top = ((a << 8 | b) if fmt == "u16be" else (b << 8 | a)), 65535
        return top - v if c.get("invert") else v      # ALFRED 1 Nov 2026: counters that run down as it goes on

    def _track_progress(self, lives_now):
        """ALFRED 31 Oct 2026: count NEW ground reached this game. After a death the counter goes back (start or
        checkpoint) and walking it again pays nothing until it passes its best; a big drop with no death = a new
        stage, so the best starts again from there."""
        if self.lives_c and lives_now < self.lives:
            self.died = True
        p = self.read_progress()
        if p < self.pos - int(self.progress_c.get("new_stage_drop", 200)):
            if self.died:
                self.died = False
                if self.progress_c.get("per_life"):
                    self.far = p          # ALFRED 1 Nov 2026: each life is paid for walking on from where it starts
            else:
                self.far = p
        if p > self.far:
            if p - self.far <= int(self.progress_c.get("max_jump", 64)):
                self.progress += (p - self.far) / float(self.progress_c.get("units_per_px", 1) or 1)   # (a bigger jump is a
                # cut-scene or a warp: not paid for). ALFRED 1 Nov 2026: counters in sub-pixels are paid per pixel
            self.far = p
        self.pos = p

    def _track_screen_progress(self):
        """ALFRED 1 Nov 2026: new ground from the pixels - the view's furthest reach left, right, up and down this game
        (going back over old ground pays nothing; after a death the view usually jumps back, which isn't counted)."""
        import scrollscan
        g = np.asarray(Image.fromarray(self.host.frame).convert("L"), np.uint8)
        dx, dy = scrollscan.shift(self._sp_prev, g)
        self._sp_prev = g
        self.cam = (self.cam[0] + dx, self.cam[1] + dy)
        x0, x1, y0, y1 = self.ext
        cx, cy = self.cam
        grow = max(0, cx - x1) + max(0, x0 - cx) + max(0, cy - y1) + max(0, y0 - cy)
        if grow:
            self.ext = (min(x0, cx), max(x1, cx), min(y0, cy), max(y1, cy))
            self.progress += grow

    def playing(self):
        return ramscan.is_playing(self.mem, self.flag)

    def _obs(self):
        f = self.host.frame
        g = np.asarray(Image.fromarray(f).convert("L"), np.uint8)
        if self._prev is not None and self._prev.shape == g.shape:
            g = np.maximum(g, self._prev)            # sprites that flicker every other frame still show up
        return np.asarray(Image.fromarray(g).resize((84, 84), Image.Resampling.BILINEAR), np.uint8)[None].copy()

    def _info(self):
        info = {"score": self.score, "lives": self.lives, "raw_score": self.raw, "playing": self.on}
        if self.progress_c or self.screen_progress:
            info["progress"] = self.progress          # ALFRED 31 Oct 2026: new ground reached this game
        return info

    # -------------------------------------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.reboot:
            # this game's save-states can't be restored (some MAME 2003-Plus drivers): switch it off and on, put a coin
            # in and press start - about a third of a second
            self.host.reset()
            for _ in range(int(self.pack.get("boot_frames", 600))):
                self.host.run((), capture=False)
            for b, n in (((2,), 8), ((), 90), ((3,), 8), ((), 186)):
                for _ in range(n):
                    self.host.run(b, capture=False)
            self._booted = True
        else:
            if not getattr(self, "_booted", False):        # MAME 2003-Plus can't restore a save-state before its first frame
                for _ in range(2):
                    self.host.run((), capture=False)
                self._booted = True
            self.host.load_state(self.start_state)
        for _ in range(int(self.rng.integers(0, self.reset_noop_max + 1))):
            self.host.run((), capture=False)
        self._prev = None
        self.host.run((), capture=True)
        self._prev = np.asarray(Image.fromarray(self.host.frame).convert("L"), np.uint8)
        self.host.run((), capture=True)
        self.snap()
        if self.screen:
            self._seen = 0
        r = self.read_score()
        self.base = r if r >= 0 else 0
        self.best = self.base                       # points = only rises above the best reading this game
        self.score, self.raw = 0, self.base
        self.lives = self.read_lives()
        if self.progress_c:                          # ALFRED 31 Oct 2026
            self.pos = self.far = self.read_progress()
            self.progress, self.died = 0, False
        elif self.screen_progress:                   # ALFRED 1 Nov 2026
            self.progress, self.cam, self.ext = 0, (0, 0), (0, 0, 0, 0)
            self._sp_prev = np.asarray(Image.fromarray(self.host.frame).convert("L"), np.uint8)
        self.on, self.off_frames, self.quiet_frames, self.frames = True, 0, 0, 0
        if self.health:
            self.hp = self.read_health()
        return self._obs(), self._info()

    def step(self, action):
        buttons = self.actions[int(action)]
        terminated = False
        for i in range(self.decision_repeat):
            last2 = i >= self.decision_repeat - 2
            self.host.run(buttons, capture=last2)
            if i == self.decision_repeat - 2:
                self._prev = np.asarray(Image.fromarray(self.host.frame).convert("L"), np.uint8)
            self.frames += 1
            if self.flag is not None and (not self.from_state or i == self.decision_repeat - 1):
                self.snap()
                if self.playing():
                    self.off_frames = 0
                else:
                    self.off_frames += 1 if not self.from_state else self.decision_repeat
                    if self.off_frames >= GAME_OVER_FRAMES:
                        terminated = True
        if self.decision_repeat == 1:
            self._prev = None
        if self.flag is None and (not self.screen or self.lives_c or self.health or self.progress_c):
            self.snap()                                 # (a score read off the screen doesn't need the memory)
        r = self.read_score()
        self.raw = r
        if self.health:
            # 100 points for knocking out a full bar, -100 for losing one. Refills (new round) count for nothing.
            full = max(1, int(self.health["full"]))
            mine, theirs = self.read_health()
            dealt = min(full, max(0, self.hp[1] - theirs))
            taken = min(full, max(0, self.hp[0] - mine))
            if min(self.hp) <= 0 or (dealt and taken and mine <= 0 and theirs <= 0):
                dealt = taken = 0          # round already over (a KO), or both bars cleared at once between rounds
            self.hp = (mine, theirs)
            if dealt or taken:
                self.score += 100.0 * (dealt - taken) / full
                self.quiet_frames = 0
            else:
                self.quiet_frames += self.decision_repeat
        elif r > self.best and r - self.best <= self.max_jump:
            self.score += r - self.best
            self.best = r
            self.quiet_frames = 0
        else:
            self.quiet_frames += self.decision_repeat
        lives_now = self.read_lives()
        if self.progress_c:
            self._track_progress(lives_now)
        elif self.screen_progress:
            self._track_screen_progress()
        self.lives = lives_now
        self.on = not terminated
        truncated = (not terminated) and self.quiet_frames >= STUCK_FRAMES
        return self._obs(), float(0), terminated, truncated, self._info()

    def render(self):
        return self.host.frame

    def drain_audio(self):
        return self.host.drain_audio()

    def close(self):
        self.host.close()
