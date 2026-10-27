"""Portable AI Arcade - FIND THE SCORE: teach the kit a new arcade game by playing it.

You play; now and then you pause and type the score you can see. The kit searches the game's memory for the
place that holds that number, then lets you check its reading live before saving a "pack" for the game
(files/arcade/found/<game>/). After that the game can be picked on the page like any other.

Run: python score_finder.py "<path to the game's .zip>"
Keys: arrows move, Z X C A S D = buttons 1-6, 5 = insert coin, 1 = start, P = pause and type the score,
G = "it says GAME OVER", F = find it now, R = start again, - / + = slower / faster, Esc = close. A game pad works too.
Signed: Claude, 10 Oct 2026.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # embeddable Python ignores the script folder

import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES / "arcade" / "generic"))
os.environ.setdefault("SDL_VIDEO_CENTERED", "1")

import pygame                                   # noqa: E402
import arcade_vec                               # noqa: E402
import ramscan                                  # noqa: E402
import screenscore                              # noqa: E402
from generic_env import action_table, UP, DOWN, LEFT, RIGHT   # noqa: E402
from libretro_host import LibretroHost         # noqa: E402

COIN, START = 2, 3
FIRE = [0, 8, 1, 9, 10, 11]                     # arcade buttons 1-6 on the RetroPad (FBNeo's layout)
KEYS = {pygame.K_UP: UP, pygame.K_DOWN: DOWN, pygame.K_LEFT: LEFT, pygame.K_RIGHT: RIGHT,
        pygame.K_z: 0, pygame.K_x: 8, pygame.K_c: 1, pygame.K_a: 9, pygame.K_s: 10, pygame.K_d: 11,
        pygame.K_5: COIN, pygame.K_1: START}
DEMO_FRAMES = 60 * 40                           # after GAME OVER, watch the demo this long (40 s)
LOOKBACK = 90                                   # frames: J / K / U mean "in the last 1.5 seconds"
TRACE_BUDGET = 400e6                            # bytes of memory snapshots kept while you play
SHOT_BUDGET = 250e6                             # bytes of screen pictures kept (for reading the score off the screen)
STATE_BUDGET = 250e6                            # bytes of save-states kept just after you press start
HELP = """Right then - here's how this works.

This window is where you teach the kit a new game. The AI can't learn anything until it knows how
it's doing - its score, or its health. Every game hides that number somewhere different in its
memory, so you're going to help me find it. Takes about five minutes.

1.  Let the game's demo run for half a minute first. Then press 5 to put a coin in and 1 to start.
2.  Play it like you normally would. Arrows move you, Z X C A S D are the buttons.
    Use every button the game needs - whatever you use, the AI gets to use too.

    Got a score on screen?  Every so often press P and type in the score you can see.
    Two or three times is plenty, as long as it's a different score each time.
    If it asks how many lives you've got, pop the number in, or just press Enter.

    Fighting game with health bars?  It's an unknown value search, like a cheat finder. Press SPACE -
    the game pauses and you pick: H new search, J / U mine / theirs decreased,
    K / I mine / theirs increased, L unchanged. Then Enter to confirm, or Backspace to pick again.
    Don't do anything between a knockout and the next round.

3.  When it says GAME OVER, press G, put the controls down and let the demo run for 40 seconds.
    Then press F and I'll go and look.
4.  I'll show you what I've found while you have another go. If my numbers match the game,
    press Y and you're done. If they don't, press N and I'll try the next one.

Got in a muddle?  Press R and start again - nothing's broken.

Press F1, Enter or Esc to close this and carry on.   (F1 brings it back any time.)"""
MENU = [(pygame.K_h, "new", "New search - health"), (pygame.K_n, "new_score", "New search - score"),
        (pygame.K_j, "mine_down", "Mine decreased"), (pygame.K_u, "theirs_down", "Theirs decreased"),
        (pygame.K_k, "mine_up", "Mine increased"), (pygame.K_i, "theirs_up", "Theirs increased"),
        (pygame.K_l, "same", "Unchanged - nothing happened to either")]
MENU_NAME = {c: t for _, c, t in MENU}
ORANGE, CYAN, GREEN, RED, GREY, WHITE, BG = ((255, 150, 60), (91, 218, 216), (112, 215, 162), (255, 147, 137),
                                             (160, 174, 194), (244, 246, 250), (9, 17, 30))


class Speaker:
    """Game sound (same idea as the Watch window's). Short queue; drops anything that falls behind so the picture
    never lags. No sound device = stays quiet."""

    def __init__(self, rate):
        self.ok, self.on, self.pending, self.speed = False, True, [], 1.0
        self.rate = int(rate)
        try:
            pygame.mixer.quit()
            pygame.mixer.init(frequency=self.rate, size=-16, channels=2, buffer=1024)
            self.chan = pygame.mixer.Channel(0)
            self.chan.set_volume(0.6)
            self.ok = True
        except Exception:
            pass

    def toggle(self):
        self.on = not self.on
        if self.ok and not self.on:
            self.chan.stop()
            self.pending.clear()

    def feed(self, stereo):
        if not (self.ok and self.on) or stereo is None or not len(stereo):
            return
        if self.speed < 1 and len(stereo) > 1:         # slow motion: stretch it, like a tape played slower
            n = len(stereo)
            x = np.linspace(0, n - 1, int(n / self.speed))
            stereo = np.stack([np.interp(x, np.arange(n), stereo[:, c]) for c in range(2)], axis=1).astype(np.int16)
        self.pending.append(stereo)
        if self.chan.get_queue() is None:
            data = np.concatenate(self.pending)
            self.pending.clear()
            if len(data) > self.rate // 4:
                data = data[-self.rate // 8:]
            snd = pygame.sndarray.make_sound(np.ascontiguousarray(data))
            if self.chan.get_busy():
                self.chan.queue(snd)
            else:
                self.chan.play(snd)

    def label(self):
        return "no sound" if not self.ok else ("M = sound on" if self.on else "M = sound off")


def bring_to_front():
    if sys.platform != "win32":
        return
    try:
        import ctypes
        u = ctypes.windll.user32
        hwnd = pygame.display.get_wm_info()["window"]
        u.ShowWindow(hwnd, 9)
        u.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x43)
        u.SetWindowPos(hwnd, -2, 0, 0, 0, 0, 0x43)
        u.keybd_event(0x12, 0, 0, 0)
        u.keybd_event(0x12, 0, 2, 0)
        u.SetForegroundWindow(hwnd)
    except Exception:
        pass


class Finder:
    def __init__(self, rom, show_help=False, core="fbneo"):
        self.rom = Path(rom)
        self.help = show_help                    # the help card (F1) - opens by itself the first time
        self.set = re.sub(r"[^\w\-]", "", self.rom.stem.lower()) or "game"
        self.rt = arcade_vec.scratch_dir("finder")
        (self.rt / "roms").mkdir(parents=True, exist_ok=True)
        self.core = core
        core_path = arcade_vec.local_core(self.rt, core)
        local = self.rt / "roms" / self.rom.name
        shutil.copy2(self.rom, local)
        LibretroHost._video = arcade_vec._fast_video
        self.host = LibretroHost(core_path, local, self.rt, require_ram=False, audio=True)
        # FBNeo shares the game's memory directly; MAME 2003-Plus doesn't, so read it out of a save-state
        self.from_state = self.host.ram is None
        if self.from_state and not self.host.dll.retro_serialize_size():
            raise RuntimeError("This game doesn't show its memory to the emulator, so the kit can't find its score.")
        for _ in range(2):
            self.host.run((), capture=False)     # MAME can't save/restore before its first frames
        self.fps = self.host.fps or 60.0
        self.ram_size = len(self.mem())
        self.boot_state = self.host.save_state()
        self.pad = None
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count():
            self.pad = pygame.joystick.Joystick(0)
            self.pad.init()
        f = self.host.run(())
        self.h, self.w = f.shape[:2]
        info = pygame.display.Info()
        self.S = max(1, min(3, (info.current_w - 80) // self.w, (info.current_h - 300) // self.h))
        self.screen = pygame.display.set_mode((max(self.w * self.S, 1100), self.h * self.S + 200))
        pygame.display.set_caption(f"Arcade Apprentice - Find the score - {self.rom.name}")
        self.font = pygame.font.SysFont("segoeui,arial", 22, bold=True)
        self.mid = pygame.font.SysFont("segoeui,arial", 17)
        self.small = pygame.font.SysFont("segoeui,arial", 15)
        self.big = pygame.font.SysFont("consolas", 30, bold=True)
        self.clock = pygame.time.Clock()
        self.host.drain_audio()                  # throw away the boot frames' sound
        self.speaker = Speaker(self.host.sample_rate)
        self.speed = 1.0                         # - / + slow the game down (it still records every frame)
        bring_to_front()
        self.new_recording()

    def mem(self):
        if self.from_state:
            return np.frombuffer(self.host.save_state(), np.uint8).copy()
        return self.host.ram.copy()

    # ------------------------------------------------------------------------------------------------ recording
    def new_recording(self):
        self.host.load_state(self.boot_state)
        self.phase = "record"
        self.frame = 0
        self.trace, self.every = [], max(2, math.ceil(self.ram_size / 200_000))
        self.shots, self.shot_every, self.label_shots = [], 4, {}     # pictures of the screen, and at each P
        self.states, self.state_every = {}, 10
        self.labels = []            # (frame, ram copy, score, lives or None)
        self.coin_frames, self.start_frames, self.over_frame = [], [], None
        self.used = set()           # (frame, button) of every press while playing
        self.typing = None          # None, "score" or "lives"
        self.text, self.pending = "", None
        self.msg = ""
        self.cands, self.ci = [], 0
        self.held = set()
        self.hs_me = self.hs_them = None   # health-bar searches (fighting games), started by H
        self.marks = []                     # (frame, key) for H / J / K / U
        self.health = None                  # (yours, theirs) once found
        self.menu = None                    # None, "pick" or ("confirm", choice) - the unknown value search menu
        self.health_up_seen = False         # once you say a bar went UP, stop assuming bars only ever go down
        self.target = "health"              # what J / K / L talk about: "health" or "score" (N starts a score search)
        self.ss, self.ss_marks = None, []   # the score unknown-value search
        self.left = None                    # how many places in memory still fit the scores you've typed
        self.loose_left = None              # ... if none fit exactly: how many went up / stayed put in step with them
        self.lives_left = None              # ... and the lives numbers you've typed

    def pressed(self):
        b = set(self.held)
        if self.pad:
            try:
                hx, hy = self.pad.get_hat(0) if self.pad.get_numhats() else (0, 0)
                ax = self.pad.get_axis(0) if self.pad.get_numaxes() > 1 else 0
                ay = self.pad.get_axis(1) if self.pad.get_numaxes() > 1 else 0
                if hx < 0 or ax < -0.5: b.add(LEFT)
                if hx > 0 or ax > 0.5: b.add(RIGHT)
                if hy > 0 or ay < -0.5: b.add(UP)
                if hy < 0 or ay > 0.5: b.add(DOWN)
                for i, rb in enumerate(FIRE):
                    if i < self.pad.get_numbuttons() and self.pad.get_button(i):
                        b.add(rb)
                if self.pad.get_numbuttons() > 6 and self.pad.get_button(6): b.add(COIN)
                if self.pad.get_numbuttons() > 7 and self.pad.get_button(7): b.add(START)
            except Exception:
                pass
        return b

    def tick_record(self, buttons):
        for b in buttons:
            if b == COIN and (not self.coin_frames or self.frame - self.coin_frames[-1] > 20):
                self.coin_frames.append(self.frame)
            elif b == START and (not self.start_frames or self.frame - self.start_frames[-1] > 20):
                self.start_frames.append(self.frame)
            elif b not in (COIN, START) and self.start_frames:
                self.used.add((self.frame, b))
        # save-states for 25 s after each start press (the AI's games will begin from one of these)
        if self.start_frames and self.frame - self.start_frames[-1] <= 25 * 60 and self.frame % self.state_every == 0:
            st = self.host.save_state()
            self.states[self.frame] = st
            while len(self.states) * len(st) > STATE_BUDGET:
                del self.states[min(self.states)]
        if self.frame % self.every == 0:
            self.trace.append((self.frame, self.mem()))
            if len(self.trace) * self.ram_size > TRACE_BUDGET:      # keep the whole session, at half the detail
                self.trace = self.trace[::2]
                self.every *= 2
        if self.start_frames and self.frame % self.shot_every == 0:
            shot = screenscore.cids(self.host.frame)
            self.shots.append((self.frame, shot))
            if len(self.shots) * shot.nbytes > SHOT_BUDGET:
                self.shots = self.shots[::2]
                self.shot_every *= 2

    # ------------------------------------------------------------------------------------------------ health bars
    def health_mark(self, what):
        """what: new | mine_down | theirs_down | mine_up | theirs_up | same  (picked from the SPACE menu)."""
        ram = self.mem()
        if what == "new_score":
            self.ss, self.ss_marks, self.target, self.msg = ramscan.HealthSearch(ram, allow_zero=True), [], "score", ""
            self.ss_marks.append((self.frame, "N"))
            return
        if what == "new":
            self.target = "health"
        if self.target == "score" and what in ("mine_down", "mine_up", "same"):
            res = self.ss.mark(ram, {"mine_down": "down", "mine_up": "up", "same": "same"}[what])
            self.ss_marks.append((self.frame, what))
            self.msg = ("That would rule out everything, so I ignored it (no harm done). It compares with your "
                        "last pick.") if res == -1 else ""
            return
        if what == "new":
            if self.hs_me is None:
                self.hs_me, self.hs_them = ramscan.HealthSearch(ram), ramscan.HealthSearch(ram)
            else:
                self.hs_me.mark(ram, "full")
                self.hs_them.mark(ram, "full")
            self.marks.append((self.frame, "H"))
            self.msg = ""
            return
        if self.hs_me is None:
            self.msg = "Start with 'New search' (SPACE, then H) at the start of a round when both bars are full."
            return
        last_h = max(fr for fr, kind in self.marks if kind == "H")
        if not self.health_up_seen:
            # health never goes up during a round: check the recording since this round started (or the last check)
            since = [r for f, r in self.trace if f > max(last_h, self.marks[-1][0])]
            if len(since) > 30:
                since = [since[i] for i in np.linspace(0, len(since) - 1, 30).astype(int)]
            for r in since:
                self.hs_me.never_up(r)
                self.hs_them.never_up(r)
        # "what just happened" = now compared with ~1.5 s ago (never from before this round started)
        ref = None          # like a cheat finder: compare with the last time you told me, not "a moment ago"
        me, them = self.hs_me, self.hs_them
        if what == "same":
            res = [me.mark(ram, "same", ref), them.mark(ram, "same", ref)]
        elif what.endswith("down"):
            side, other = (me, them) if what == "mine_down" else (them, me)
            res = [side.mark(ram, "down", ref)] + ([] if self.health_up_seen else [other.mark(ram, "not_up", ref)])
        else:
            self.health_up_seen = True
            side = me if what == "mine_up" else them
            res = [side.mark(ram, "up", ref)]
        self.marks.append((self.frame, what))
        self.msg = ("That would rule out everything, so I ignored it (no harm done). It compares with your "
                    "last pick, not with a moment ago.") if -1 in res else ""

    # ------------------------------------------------------------------------------------------------ resume helpers
    def countdown(self, caption=""):
        """3-2-1 over the paused game, so you can find your bearings before it moves again."""
        if not self.start_frames:
            return
        self.held = set()
        size = max(60, min(160, self.h * self.S // 2))
        f = pygame.font.SysFont("consolas,arial", size, bold=True)
        for n in ("3", "2", "1"):
            end = time.time() + 1.0
            while time.time() < end:
                for e in pygame.event.get():
                    if e.type == pygame.KEYDOWN and e.key in KEYS:
                        self.held.add(KEYS[e.key])          # you can already be holding a direction
                    elif e.type == pygame.KEYUP and e.key in KEYS:
                        self.held.discard(KEYS[e.key])
                self.draw()
                img, sh = f.render(n, True, ORANGE), f.render(n, True, (0, 0, 0))
                x = (self.screen.get_width() - img.get_width()) // 2
                y = (self.h * self.S - img.get_height()) // 2
                self.screen.blit(sh, (x + 5, y + 5))
                self.screen.blit(img, (x, y))
                if caption:
                    c = self.font.render(caption, True, WHITE)
                    bx = pygame.Rect((self.screen.get_width() - c.get_width()) // 2 - 16, y + img.get_height() + 6,
                                     c.get_width() + 32, c.get_height() + 12)
                    pygame.draw.rect(self.screen, (17, 29, 45), bx, border_radius=10)
                    self.screen.blit(c, (bx.x + 16, bx.y + 6))
                pygame.display.flip()
                self.clock.tick(30)

    def count_left(self):
        """Run the score search on what you've typed so far, just to count the places that still fit."""
        labels = [(r, v) for _, r, v, _ in self.labels if v > 0]
        if not labels:
            self.left = None
            return
        try:
            self.left = len(ramscan.find_score(labels, max_candidates=5000))
        except Exception:
            self.left = None
        self.loose_left = None
        if self.left == 0:
            try:
                self.loose_left = len(ramscan.find_score_loose([(r, v) for _, r, v, _ in self.labels], self.gaps(),
                                                               max_candidates=5000))
            except Exception:
                self.loose_left = None
        # lives: a byte equal to lives, lives-1 or lives+1 every time (same rule as ramscan.find_lives, uncapped)
        lab = [(r, l) for _, r, _, l in self.labels if l is not None]
        if not lab:
            self.lives_left = None
            return
        n = min(len(r) for r, _ in lab)
        total = 0
        for adj in (0, -1, 1):
            ok = np.ones(n, bool)
            for r, l in lab:
                ok &= r[:n].astype(np.int16) == l + adj
            total += int(ok.sum())
        self.lives_left = total

    def gaps(self, per_gap=12):
        """A few memory snapshots from between each pair of typed scores (for the loose search)."""
        out = []
        for (f0, *_), (f1, *_) in zip(self.labels, self.labels[1:]):
            g = [m for f, m in self.trace if f0 < f < f1]
            if len(g) > per_gap:
                g = [g[i] for i in np.linspace(0, len(g) - 1, per_gap).astype(int)]
            out.append(g)
        return out

    def choice_name(self, c):
        if self.target == "score" and c in ("mine_down", "mine_up", "same"):
            return {"mine_down": "Score decreased", "mine_up": "Score increased", "same": "Score unchanged"}[c]
        return MENU_NAME[c]

    def peek(self, x, y, most=8):
        """Cheat-finder style: once 8 or fewer places are left, show where they are and what each reads right now."""
        search = self.ss if self.target == "score" else self.hs_me
        if search is None or not 0 < search.count() <= most:
            return
        ram = self.mem()
        names = {"u8": "1 byte", "s16le": "2 bytes LE", "s16be": "2 bytes BE"}
        self.text_at(self.small, ("Score" if self.target == "score" else "Your health") + " - what each place reads now:",
                     x, y, GREY)
        for i, c in enumerate(search.candidates(limit=most)[:most]):
            v = ramscan.decode_health(ram, c)
            raw = v & 0xFF if c["fmt"] == "u8" else v & 0xFFFF
            hx = f"{raw:02X}" if c["fmt"] == "u8" else f"{raw:04X}"
            t = f"{names[c['fmt']]}{' swapped' if c['swapped'] else ''} @{c['offset']:05X} = {v:,}  (hex {hx})"
            self.text_at(self.small, t, x + (i % 2) * 320, y + 20 + (i // 2) * 18, CYAN)

    def menu_places(self, x, y):
        """'Places left' line for the SPACE menu: health (yours / theirs), plus score and lives if you've typed any."""
        parts = []
        if self.ss is not None:
            n = self.ss.count()
            parts += [("Score search ", GREY), (f"{n:,}", GREEN if n == 1 else RED if n == 0 else ORANGE), ("     ", GREY)]
        if self.hs_me is not None:
            parts += [("Places left -  yours ", GREY), (f"{self.hs_me.count():,}", GREEN if self.hs_me.count() == 1 else ORANGE),
                      ("   theirs ", GREY), (f"{self.hs_them.count():,}", GREEN if self.hs_them.count() == 1 else ORANGE)]
        elif self.ss is None:
            parts += [("Places left -  none yet: start with H or N, new search", GREY)]
        if self.left is not None:
            n = self.left if self.left or self.loose_left is None else self.loose_left
            parts += [("     score ", GREY), (f"{n:,}", GREEN if n == 1 else RED if n == 0 else ORANGE)]
        if self.lives_left is not None:
            parts += [("   lives ", GREY), (f"{self.lives_left:,}", GREEN if self.lives_left == 1 else ORANGE)]
        for t, col in parts:
            self.text_at(self.mid, t, x, y, col)
            x += self.mid.size(t)[0]

    def menu_key(self, k):
        """SPACE menu: pick, then confirm (Enter / SPACE), pick another (Backspace) or cancel (Esc)."""
        if k == pygame.K_ESCAPE:
            self.menu = None
            self.countdown("Cancelled - nothing noted")
            return True
        if self.menu == "pick":
            for key, choice, _ in MENU:
                if k == key:
                    if self.target == "score" and choice in ("theirs_down", "theirs_up"):
                        return True
                    if not choice.startswith("new") and (self.ss if self.target == "score" else self.hs_me) is None:
                        self.msg = "Start with a new search first: H for a health bar, N for the score."
                        return True
                    self.menu, self.msg = ("confirm", choice), ""
            return True
        if k == pygame.K_BACKSPACE:
            self.menu = "pick"
        elif k in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            choice = self.menu[1]
            self.menu = None
            self.health_mark(choice)
            if self.msg:
                self.countdown()
            elif self.target == "score":
                self.countdown(f"{self.choice_name(choice)} - noted.   Places left for the score: {self.ss.count():,}")
            else:
                self.countdown(f"{self.choice_name(choice)} - noted.   Places left: yours "
                               f"{self.hs_me.count():,}, theirs {self.hs_them.count():,}")
        return True

    def change_speed(self, faster):
        speeds = (1.0, 0.75, 0.5, 0.25)
        i = min(range(len(speeds)), key=lambda j: abs(speeds[j] - self.speed))
        i = max(0, i - 1) if faster else min(len(speeds) - 1, i + 1)
        self.speed = speeds[i]
        self.speaker.speed = self.speed
        self.msg = ""

    # ------------------------------------------------------------------------------------------------ the search
    def play_window(self):
        """Frames when a game was on: from the start press before the first pause, to GAME OVER (or now)."""
        first = min([fr for fr, *_ in self.labels] + [fr for fr, _ in self.marks] + [fr for fr, _ in self.ss_marks])
        starts = [f for f in self.start_frames if f < first] or [max(0, first - 600)]
        s = starts[-1]
        e = self.over_frame if self.over_frame is not None else self.frame
        return s, e

    def ss_cands(self, play):
        """Turn what's left of the score search into score readings. BCD if the bytes always look like BCD;
        lined up with any scores you typed (x10, x100, half ...) so the reading matches the screen."""
        fmt = {"u8": (1, False), "s16le": (2, False), "s16be": (2, True)}
        thin = play[:: max(1, len(play) // 200)]
        typed = [(r, v) for _, r, v, _ in self.labels if v > 0]
        out = []
        for c in self.ss.candidates(limit=200):
            n, big = fmt[c["fmt"]]
            base = {"n": n, "big": big, "offset": c["offset"], "swapped": c["swapped"], "scale": 1, "loose": True,
                    "fits": False}
            kinds = ["bin"]
            if thin and all(ramscan.decode(r, dict(base, kind="bcd")) >= 0 for r in thin):
                kinds = ["bcd", "bin"]
            best = dict(base, kind=kinds[0])
            for kind in kinds:
                cand = dict(base, kind=kind)
                vals = [ramscan.decode(r, cand) for r, _ in typed]
                if not typed or not all(v > 0 for v in vals):
                    continue
                ratios = [t / v for (_, t), v in zip(typed, vals)]
                sc = float(np.median(ratios))
                if not all(abs(x - sc) <= 0.02 * sc for x in ratios):
                    continue
                if sc >= 0.98 and abs(sc - round(sc)) < 0.02 * sc:
                    sc = int(round(sc))
                elif sc < 1 and round(1 / sc) <= 10 and abs(1 / sc - round(1 / sc)) < 0.02 / sc:
                    sc = 1 / round(1 / sc)
                else:
                    continue
                best = dict(cand, scale=sc, fits=True, nice=sc in (1, 10, 100))
                break
            out.append(best)
        return out

    def screen_fit(self, s, e):
        """The score read off the screen, learned from the scores you typed (and where numbers come and go)."""
        lab = [(self.label_shots[fr], v) for fr, _, v, _ in self.labels if fr in self.label_shots]
        around = [[c for f2, c in self.shots if 0 < abs(f2 - fr) <= 40] for fr, _, v, _ in self.labels
                  if fr in self.label_shots]
        frames = [c for fr, c in self.shots if s <= fr <= e]
        try:
            spec = screenscore.fit_labelled(lab, frames, around)
        except Exception:
            traceback.print_exc()
            spec = None
        if spec:
            spec.update(fits=True, nice=True, penalty=-1.0,
                        why=f"read back {spec['report']['labels_read']} of the scores you typed")
        return spec

    def reading(self, c, ram):
        """What a score reading says right now: from memory, or off the screen. -1 = can't tell."""
        if c.get("kind") == "screen":
            self._readers = getattr(self, "_readers", {})
            rd = self._readers.get(id(c))
            if rd is None:
                rd = self._readers[id(c)] = screenscore.ScreenScore(c)
            v = rd.read(self.host.frame)
            return -1 if v is None else v
        return ramscan.decode(ram, c)

    def describe(self, c):
        if c.get("kind") == "screen":
            y0, y1, x0, x1 = c["box"]
            return (f"read off the screen (x {x0}-{x1}, y {y0}-{y1}) - digits learned from you: "
                    + " ".join(str(d) for d in c["report"]["digits_learned"]))
        return ramscan.describe(c)

    def analyse(self):
        self.phase = "busy"
        self.draw()
        pygame.display.flip()
        labels = [(r, v) for _, r, v, _ in self.labels]
        s, e = self.play_window()
        play = [r for f, r in self.trace if s + 180 <= f <= e]
        idle = [r for f, r in self.trace if 600 <= f < (self.coin_frames[0] if self.coin_frames else 0)]
        if self.over_frame is not None:
            idle += [r for f, r in self.trace if f > self.over_frame + 120]
        cands = ramscan.find_score(labels) if len({v for _, v in labels}) >= 2 else []
        self.typo = None
        if not cands and len(labels) >= 3:          # one mistyped number? try leaving each out
            for i in range(len(labels)):
                c = ramscan.find_score(labels[:i] + labels[i + 1:])
                if c:
                    cands, self.typo = c, i
                    break
        self.screen_spec = None
        if not cands and len({v for _, v in labels}) >= 2:
            self.screen_spec = self.screen_fit(s, e)
        self.loose = False
        if not cands and len({v for _, v in labels}) >= 2:
            cands = ramscan.find_score_loose([(r, v) for _, r, v, _ in self.labels], self.gaps())
            cands.sort(key=lambda c: (not c.get("fits"), not c.get("nice")))
            self.loose = bool(cands)
        if not cands and self.ss is not None and self.ss.count():
            cands = self.ss_cands(play)
            self.loose = bool(cands)
        self.cands = ramscan.rank(cands, play)[:12]
        if self.loose:                    # a reading that lines up with your numbers beats one that only goes up
            self.cands.sort(key=lambda c: (not c.get("fits"), not c.get("nice"), c["penalty"]))
        if self.screen_spec:              # read off the screen exactly as you typed it: best of all
            self.cands = [self.screen_spec] + self.cands[:11]
        lives_labels = [(r, l) for _, r, _, l in self.labels if l is not None]
        lc = ramscan.rank_lives(ramscan.find_lives(lives_labels), play)
        self.lives_c = lc[0] if lc else None
        # leave out the last 2 s before GAME OVER was pressed: the GAME OVER screen itself isn't "playing"
        flag_play = [r for f, r in self.trace if s + 180 <= f <= e - 120]
        self.flag = ramscan.find_playing_flag(flag_play, idle)
        self.ci = 0
        self.health = None
        if self.hs_me is not None and max(self.hs_me.count(), self.hs_them.count()) <= 200:
            self.health = ramscan.pick_health_pair(self.hs_me.candidates(), self.hs_them.candidates())
        if not self.cands and not self.health:
            self.phase = "record"
            if self.hs_me is not None and not labels:
                self.msg = ("Couldn't pin down the health bars yet. Keep going: SPACE after each hit, and a new search "
                            "(SPACE, H) at the start of each round.")
            else:
                self.msg = ("Couldn't find that number in the game's memory. Check the numbers you typed, then pause a "
                            "couple more times at different scores (and not while the score is counting up).")
            return
        self.phase = "check"
        self.msg = ""
        self.check_best = None

    # ------------------------------------------------------------------------------------------------ saving
    def save(self):
        c = self.cands[self.ci] if self.cands else None
        s, e = self.play_window()
        want = s + 180
        keys = sorted(self.states)
        if not keys:
            self.msg = "No start point was saved - press R and record again, pressing 1 (start) to begin the game."
            return
        k = min(keys, key=lambda f: (f < want, abs(f - want)))
        used = [b for f, b in self.used if s <= f <= e]
        axes = []
        if LEFT in used or RIGHT in used:
            axes.append("lr")
        if UP in used or DOWN in used:
            axes.append("ud")
        buttons = [b for b in FIRE if b in used]          # none used (e.g. Pac-Man) = joystick only
        with zipfile.ZipFile(self.rom) as z:
            crcs = sorted({"%08x" % i.CRC for i in z.infolist() if not i.is_dir()})
        pack = {"id": "found-" + self.set, "name": f"{self.set.upper()} (arcade, found by you)", "set": self.set,
                "rom_file": self.rom.name, "crcs": crcs, "score": c, "alternatives": self.cands[:8],
                "lives": self.lives_c, "playing": self.flag, "axes": axes or ["lr", "ud"], "buttons": buttons,
                "decision_repeat": 4, "core": self.core, "max_jump": int(max([50000] + [20 * v for _, _, v, _ in self.labels])),
                "health": ({"mine": self.health[0], "theirs": self.health[1], "full": self.health[0]["full"]}
                           if self.health else None), "reward": "health" if self.health else "score",
                "labels": [v for _, _, v, _ in self.labels], "made": time.strftime("%Y-%m-%d %H:%M")}
        dst = FILES / "arcade" / "found" / self.set
        old_ctrl = None
        try:                # found before? start from its controls plus anything you used this time
            old = json.loads((dst / "pack.json").read_text(encoding="utf-8"))
            if old.get("axes") is not None and "buttons" in old:
                old_ctrl = {"axes": list(old["axes"]), "buttons": [int(b) for b in old["buttons"]]}
                pack["controls_history"] = old.get("controls_history", [])
                pack["axes"] = [a for a in ("lr", "ud") if a in set(old["axes"]) | set(pack["axes"])]
                pack["buttons"] = list(dict.fromkeys([int(b) for b in old["buttons"]] + list(pack["buttons"])))
        except Exception:
            pass
        # THE BUTTON TEST (controls.py, 17 Oct 2026): from the moments saved while you played, press each direction and
        # button and compare the screen with doing nothing. Anything that never changed anything is only SUGGESTED for
        # dropping (Jonny's rule: the user decides - a button that only matters later, like a potion you have to pick up
        # first, looks exactly the same). The suggestion shows in the AI's Settings > Buttons.
        self.saved_note = ""
        try:
            import controls
            self.msg = "Checking which buttons the game really uses..."
            self.phase = "busy"
            self.draw()
            pygame.display.flip()
            cand = {"axes": list(pack["axes"]), "buttons": [int(b) for b in pack["buttons"]]}
            tested = controls.test_finder(self, s, e, cand, tick=pygame.event.pump)[0]
            pack.pop("finder_suggests", None)
            if tested and controls.moves(tested) != controls.moves(cand):
                drop = [c for c in controls.inputs_of(cand) if c not in controls.inputs_of(tested)]
                pack["finder_suggests"] = {"drop": drop, "when": time.strftime("%Y-%m-%d %H:%M")}
                self.saved_note = (f"While you played, {controls._names(drop)} never changed anything - the AI's "
                                   "Settings > Buttons can drop them (your call).")
            if old_ctrl is not None:      # anything new you used is added; AIs already trained on it get carried over
                pack["axes"], pack["buttons"] = old_ctrl["axes"], old_ctrl["buttons"]
                controls.record_change(pack, cand, "found again - you used more buttons")
        except Exception as ex:
            print("Button test skipped:", ex)
            if old_ctrl is not None:      # couldn't test: keep what its AIs were trained with
                pack["axes"], pack["buttons"] = old_ctrl["axes"], old_ctrl["buttons"]
        pack["n_actions"] = len(action_table(pack)[0])
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "start.state").write_bytes(self.states[k])
        if self.core != "fbneo":
            # some MAME 2003-Plus games' save-states can't be restored in a fresh emulator: then the AI starts each game
            # by switching the machine on, putting a coin in and pressing start instead
            self.phase = "busy"
            self.draw()
            pygame.display.flip()
            try:
                import restore_check
                if not restore_check.can_restore(dst / "start.state", self.rom, self.core):
                    pack["reset"] = "reboot"
            except Exception:
                pass
        (dst / "pack.json").write_text(json.dumps(pack, indent=1), encoding="utf-8")
        try:
            pygame.image.save(self.screen, str(dst / "found.png"))
        except Exception:
            pass
        self.phase = "saved"
        self.saved_as = pack["name"]

    # ------------------------------------------------------------------------------------------------ drawing
    def text_at(self, font, s, x, y, col=WHITE):
        self.screen.blit(font.render(s, True, col), (x, y))

    def tally(self, y):
        """One line: SCORE last 15,710 - 1 possibility    LIVES last 2 - 3 possibilities    HEALTH ..."""
        parts = []

        def left(n):
            if n is None:
                return [("-", GREY)]
            col = RED if n == 0 else GREEN if n == 1 else ORANGE
            return [(f"{n:,}", col), (" possibilit" + ("y" if n == 1 else "ies") + (" in memory - F reads it off the screen instead"
                                                                                   if n == 0 else ""), GREY)]
        if not self.labels and self.ss is not None:
            parts += [("SCORE ", WHITE), ("unknown value search  ", GREY)] + left(self.ss.count())
        elif self.labels and self.left == 0 and self.loose_left is not None:
            parts += [("SCORE ", WHITE), ("last ", GREY), (f"{self.labels[-1][2]:,}", CYAN), ("  0 exact, ", GREY)] + \
                     left(self.loose_left)[:1] + [(" that went up with you", GREY)]
        elif self.labels:
            parts += [("SCORE ", WHITE), ("last ", GREY), (f"{self.labels[-1][2]:,}", CYAN), ("  ", GREY)] + left(self.left)
        else:
            parts += [("SCORE ", WHITE), ("press P and type the score you see", GREY)]
        lives = [l for *_, l in self.labels if l is not None]
        if lives:
            parts += [("        LIVES ", WHITE), ("last ", GREY), (str(lives[-1]), CYAN), ("  ", GREY)] + left(self.lives_left)
            if len(set(lives)) < 2:
                parts += [(" (lose one, then tell me again)", GREY)]
        if self.hs_me is not None:
            parts += [("        HEALTH ", WHITE), (f"{len(self.marks)} presses  ", GREY), ("yours ", GREY)] + \
                     left(self.hs_me.count())[:1] + [("  theirs ", GREY)] + left(self.hs_them.count())[:1]
        x = 14
        room = self.screen.get_width() - 14
        for t, col in parts:
            w = self.mid.size(t)[0]
            if x + w > room:
                break
            self.text_at(self.mid, t, x, y, col)
            x += w

    def draw(self):
        sc = self.screen
        sc.fill(BG)
        f = self.host.frame
        surf = pygame.transform.scale(pygame.surfarray.make_surface(f.swapaxes(0, 1)), (self.w * self.S, self.h * self.S))
        gx = (sc.get_width() - self.w * self.S) // 2
        sc.blit(surf, (gx, 0))
        y = self.h * self.S + 8
        W = sc.get_width()
        if self.phase == "record":
            self.text_at(self.font, "Step 1 of 2:  play the game, and tell me what's happening", 14, y, ORANGE)
            demo = self.coin_frames == [] and self.frame < 1500
            self.text_at(self.mid, "5 = coin    1 = start    arrows = move    Z X C A S D = buttons 1-6    (a game pad works too)"
                         + (f"      (let the demo run {max(0, (1500 - self.frame) // 60)} s first)" if demo else ""),
                         14, y + 30, WHITE)
            self.text_at(self.mid, "SCORE:  press P and type the score you see - 2 or 3 times, at different scores.",
                         14, y + 56, WHITE)
            self.text_at(self.mid, "HEALTH BARS:  SPACE = pause for an unknown value search  (new / decreased / "
                                   "increased / unchanged)", 14, y + 80, WHITE)
            spd = f"    - / + = speed ({int(self.speed * 100)}%)    {self.speaker.label()}"
            if self.over_frame is None:
                self.text_at(self.mid, "At GAME OVER press G, then F to find it.    F1 = help    R = restart    Esc = close"
                             + spd, 14, y + 104, ORANGE if self.speed < 1 else WHITE)
            else:
                wait = (self.over_frame + DEMO_FRAMES - self.frame) // 60
                if wait > 0:
                    self.text_at(self.font, f"GAME OVER noted - hands off, let the demo play {wait} s more, then press F",
                                 14, y + 100, ORANGE)
                else:
                    self.text_at(self.font, "Ready!  Press F to find the score", 14, y + 100, GREEN)
            self.tally(y + 134)
            if self.msg:
                self.text_at(self.small, self.msg, 14, y + 164, RED)
        elif self.phase == "busy":
            self.text_at(self.font, "Searching the game's memory...", 14, y + 30, ORANGE)
        elif self.phase == "check":
            ram = self.mem()
            lives = ramscan.read_lives(ram, self.lives_c) if self.lives_c else None
            on = ramscan.is_playing(ram, self.flag)
            self.text_at(self.font, "Step 2 of 2:  keep playing - does my reading match the game?", 14, y, ORANGE)
            box = pygame.Rect(14, y + 34, W - 28, 44)
            pygame.draw.rect(sc, (17, 29, 45), box, border_radius=10)
            if self.cands:
                v = self.reading(self.cands[self.ci], ram)
                reading = f"SCORE {v:,}" if v >= 0 else "SCORE ?"
            else:
                reading = "SCORE (not used)"
            if not self.health:
                reading += f"     LIVES {lives}" if lives is not None else "     LIVES (not found - that's OK)"
            reading += ("     GAME ON" if on else "     GAME OVER") if self.flag else ""
            self.text_at(self.big, reading, 26, y + 40, CYAN)
            if self.health:
                x = 26 + self.big.size(reading)[0] + 30
                for i, (lbl, c) in enumerate((("YOU", self.health[0]), ("THEM", self.health[1]))):
                    hv = ramscan.decode_health(ram, c)
                    frac = max(0.0, min(1.0, hv / max(1, c["full"])))
                    self.text_at(self.small, f"{lbl} {hv}/{c['full']}", x, y + 38 + i * 20, WHITE)
                    pygame.draw.rect(sc, (60, 70, 90), (x + 110, y + 41 + i * 20, 160, 12))
                    pygame.draw.rect(sc, GREEN if i == 0 else RED, (x + 110, y + 41 + i * 20, int(160 * frac), 12))
            if self.cands:
                c = self.cands[self.ci]
                self.text_at(self.small, f"Score reading {self.ci + 1} of {len(self.cands)}: {self.describe(c)}  -  "
                                         f"{c.get('why', '')}", 14, y + 86, GREY)
            self.text_at(self.mid, "Y = yes, it matches - save it      " + ("N = no, try the next score reading      " if len(self.cands) > 1 else "")
                         + f"R = record again      - / + = speed ({int(self.speed * 100)}%)      F1 = help", 14, y + 112, WHITE)
            if len(self.cands) > 1:
                x = 14
                for t, col in [("All readings now:  ", GREY)] + [
                        (f"{i + 1}) " + (f"{v:,}" if (v := self.reading(c, ram)) >= 0 else "?") + "    ", CYAN if i == self.ci else WHITE)
                        for i, c in enumerate(self.cands)]:
                    if x + self.small.size(t)[0] > W - 14:
                        break
                    self.text_at(self.small, t, x, y + 140, col)
                    x += self.small.size(t)[0]
            else:
                tip = ("Bars drop when you / they get hit?  " if self.health else "Score goes up when you score?  ")
                tip += ("GAME ON / GAME OVER right?  " if self.flag else "") + "(Coin + 1 again if the game ended.)"
                self.text_at(self.small, tip, 14, y + 140, GREY)
            if self.cands and self.cands[self.ci].get("kind") == "screen":
                unseen = sorted(set(range(10)) - set(self.cands[self.ci]["report"]["digits_learned"]))
                self.text_at(self.small, "Read off the screen, the way you do. " + (
                    f"Digits you haven't typed yet ({' '.join(map(str, unseen))}) are its best guess - watch them."
                    if unseen else "It has seen every digit."), 14, y + 160, CYAN)
            elif getattr(self, "loose", False) and self.cands and not self.cands[self.ci].get("fits"):
                self.text_at(self.small, "Found with the 'went up' search: this number won't look like the game's score, "
                                         "but it should go UP every time you score, and only then.", 14, y + 160, ORANGE)
            elif self.typo is not None:
                self.text_at(self.small, f"Note: the score number {self.typo + 1} you typed didn't fit - maybe a typo; I left it out.",
                             14, y + 160, RED)
        elif self.phase == "saved":
            self.text_at(self.font, f"Saved!  '{self.saved_as}' is ready to train.", 14, y + 10, GREEN)
            self.text_at(self.mid, "Close this window, then press 'Scan the ROMs folder' on the kit's page and pick the game.",
                         14, y + 46, WHITE)
            self.text_at(self.mid, "Press Esc to close.", 14, y + 74, GREY)
            if getattr(self, "saved_note", ""):
                self.text_at(self.small, self.saved_note, 14, y + 102, ORANGE)
        if self.typing:
            ov = pygame.Surface(sc.get_size(), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 150))
            sc.blit(ov, (0, 0))
            q = ("What score does the game show right now?" if self.typing == "score"
                 else "How many lives are left? (type a number, or just press Enter to skip)")
            bx = pygame.Rect(W // 2 - 330, 120, 660, 150)
            pygame.draw.rect(sc, (17, 29, 45), bx, border_radius=14)
            pygame.draw.rect(sc, ORANGE, bx, 2, border_radius=14)
            self.text_at(self.mid, q, bx.x + 20, bx.y + 18)
            self.text_at(self.big, (self.text or "") + "_", bx.x + 20, bx.y + 58, ORANGE)
            self.text_at(self.small, "Numbers only. Enter = OK, Esc = cancel. (Lives: count the spare ones shown, or all of them - either works.)",
                         bx.x + 20, bx.y + 112, GREY)
        if self.menu:
            ov = pygame.Surface(sc.get_size(), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 150))
            sc.blit(ov, (0, 0))
            bx = pygame.Rect(W // 2 - 340, 70, 680, 392)
            pygame.draw.rect(sc, (17, 29, 45), bx, border_radius=14)
            pygame.draw.rect(sc, ORANGE, bx, 2, border_radius=14)
            if self.menu == "pick":
                self.text_at(self.font, "Unknown value search - what just happened?", bx.x + 20, bx.y + 16, ORANGE)
                if self.target == "score":
                    rows = [("H", "New search - health", ("N", "New search - score")), ("J", "Score decreased", None),
                            ("K", "Score increased", None), ("L", "Score unchanged", None)]
                else:
                    rows = [("H", "New search - health", ("N", "New search - score")),
                            ("J", "Mine decreased", ("U", "Theirs decreased")),
                            ("K", "Mine increased", ("I", "Theirs increased")), ("L", "Unchanged - nothing happened to either", None)]
                for i, (a, ta, b) in enumerate(rows):
                    yy = bx.y + 64 + i * 40
                    self.text_at(self.big, a, bx.x + 24, yy - 4, CYAN)
                    self.text_at(self.mid, ta, bx.x + 60, yy + 4, WHITE)
                    if b:
                        self.text_at(self.big, b[0], bx.x + 360, yy - 4, CYAN)
                        self.text_at(self.mid, b[1], bx.x + 396, yy + 4, WHITE)
                self.menu_places(bx.x + 20, bx.y + 222)
                self.peek(bx.x + 20, bx.y + 252)
                self.text_at(self.small, "Esc = cancel.  The game stays paused until you're done.", bx.x + 20, bx.y + 366, GREY)
                if self.msg:
                    self.text_at(self.small, self.msg, bx.x + 20, bx.y + 346, RED)
            else:
                self.text_at(self.font, "You picked:", bx.x + 20, bx.y + 16, ORANGE)
                self.text_at(self.big, self.choice_name(self.menu[1]), bx.x + 20, bx.y + 70, CYAN)
                self.text_at(self.mid, "Right?   Enter = yes, note it      Backspace = pick another      Esc = cancel",
                             bx.x + 20, bx.y + 150, WHITE)
                self.menu_places(bx.x + 20, bx.y + 222)
                self.peek(bx.x + 20, bx.y + 252)
        if self.help:
            ov = pygame.Surface(sc.get_size(), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 185))
            sc.blit(ov, (0, 0))
            lines = HELP.split("\n")
            lh = max(14, min(22, (sc.get_height() - 70) // len(lines)))
            font = pygame.font.SysFont("segoeui,arial", lh - 4)
            bw = min(W - 30, max(font.size(ln)[0] for ln in lines) + 50)
            bx = pygame.Rect((W - bw) // 2, 15, bw, lh * len(lines) + 40)
            pygame.draw.rect(sc, (17, 29, 45), bx, border_radius=14)
            pygame.draw.rect(sc, ORANGE, bx, 2, border_radius=14)
            for i, ln in enumerate(lines):
                col = ORANGE if i == 0 else (GREY if ln.startswith(("Press F1", "Got in")) else WHITE)
                self.screen.blit(font.render(ln, True, col), (bx.x + 25, bx.y + 20 + i * lh))

    # ------------------------------------------------------------------------------------------------ main loop
    def key(self, e):
        k = e.key
        if k == pygame.K_F1 and not self.typing:
            self.help = not self.help
            self.held = set()
            return True
        if self.help:
            if k in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self.help = False
                if self.phase in ("record", "check"):
                    self.countdown()
            return True
        if self.menu:
            return self.menu_key(k)
        if self.typing:
            if k == pygame.K_ESCAPE:
                self.typing, self.text = None, ""
                self.countdown()
            elif k == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                if self.typing == "score":
                    if self.text.strip():
                        self.pending = (self.frame, self.mem(), int(self.text))
                        self.label_shots[self.frame] = screenscore.cids(self.host.frame)
                        self.typing, self.text = "lives", ""
                else:
                    lives = int(self.text) if self.text.strip() else None
                    fr, ram, v = self.pending
                    self.labels.append((fr, ram, v, lives))
                    self.typing, self.text, self.pending, self.msg = None, "", None, ""
                    self.draw()
                    self.screen.fill(BG, (0, self.h * self.S, self.screen.get_width(), 30))
                    self.text_at(self.font, "Narrowing it down...", 14, self.h * self.S + 8, ORANGE)
                    pygame.display.flip()
                    self.count_left()
                    self.countdown()
            elif e.unicode and e.unicode.isdigit() and len(self.text) < 10:
                self.text += e.unicode
            return True
        if k == pygame.K_ESCAPE:
            return False
        if k in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.change_speed(False)
            return True
        if k == pygame.K_m:
            self.speaker.toggle()
            return True
        if k in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.change_speed(True)
            return True
        if self.phase == "record":
            if k == pygame.K_p:
                if not self.start_frames:
                    self.msg = "Start a game first: press 5 to put a coin in, then 1 to start."
                else:
                    self.typing, self.text, self.held = "score", "", set()
            elif k == pygame.K_g:
                self.over_frame = self.frame
                self.msg = ""
            elif k == pygame.K_SPACE:
                if not self.start_frames:
                    self.msg = "Start a game first: press 5 to put a coin in, then 1 to start."
                else:
                    self.menu, self.held, self.msg = "pick", set(), ""
            elif k == pygame.K_f:
                health_ready = (self.hs_me is not None and len(self.marks) >= 3) or len(self.ss_marks) >= 3
                demo_left = (self.over_frame + DEMO_FRAMES - self.frame) if self.over_frame is not None else 0
                if demo_left > 0 and not getattr(self, "_warned", False):
                    self._warned = True
                    self.msg = (f"Let the game's demo run another {demo_left // 60 + 1} s first - it's how I learn what "
                                "'no game on' looks like. (Press F again to search anyway.)")
                elif len({v for _, _, v, _ in self.labels}) < 2 and not health_ready:
                    self.msg = ("Tell me at least two different scores first (press P while playing) - or, for a "
                                "health-bar game, SPACE then H at a round start, then SPACE after a few hits.")
                else:
                    self.analyse()
            elif k == pygame.K_r:
                self.new_recording()
        elif self.phase == "check":
            if k == pygame.K_y:
                self.save()
            elif k in (pygame.K_n, pygame.K_TAB) and self.cands:
                self.ci = (self.ci + 1) % len(self.cands)
            elif k == pygame.K_r:
                self.new_recording()
        if k in KEYS:
            self.held.add(KEYS[k])
        return True

    def run(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return
                if e.type == pygame.KEYDOWN and not self.key(e):
                    return
                if e.type == pygame.KEYUP and e.key in KEYS:
                    self.held.discard(KEYS[e.key])
            if not self.typing and not self.help and not self.menu and self.phase in ("record", "check"):
                b = self.pressed()
                self.host.run(tuple(b))
                self.speaker.feed(self.host.drain_audio().reshape(-1, 2))
                if self.phase == "record":
                    self.tick_record(b)
                self.frame += 1
            self.draw()
            pygame.display.flip()
            self.clock.tick(self.fps * self.speed if not (self.typing or self.help or self.menu) else 30)


def already_open():
    """True if another Find-the-score window is running (Windows: a named mutex; it's freed when that window shuts)."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.CreateMutexW.restype = ctypes.c_void_p
        global _MUTEX
        _MUTEX = k.CreateMutexW(None, False, "ArcadeApprentice-FindTheScore")
        return k.GetLastError() == 183          # ERROR_ALREADY_EXISTS
    except Exception:
        return False


def main():
    rom = sys.argv[1] if len(sys.argv) > 1 else None
    if already_open():
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, "Find the score is already open - look for its window (it may be "
                                             "behind this page). Close it first if you want to start a new one.",
                                             "Arcade Apprentice", 0x40 | 0x40000)
        except Exception:
            pass
        return 0
    if not rom or not Path(rom).exists():
        print("Usage: score_finder.py <game .zip>")
        return 2
    core = sys.argv[sys.argv.index("--core") + 1] if "--core" in sys.argv else None
    if core is None:
        # does it run in FBNeo? A romset made for MAME 2003-Plus only shows FBNeo's missing-files screen
        core = "fbneo"
        try:
            import subprocess
            r = subprocess.run([sys.executable, "-B", str(FILES / "autopack" / "compat.py"), "--one", str(rom)],
                               capture_output=True, text=True, timeout=120)
            if json.loads(r.stdout.strip().splitlines()[-1])["runs"] == "no" and arcade_vec.CORES["mame2003_plus"].exists():
                core = "mame2003_plus"
        except Exception:
            pass
    try:
        Finder(rom, show_help=True, core=core).run()
    except Exception:
        err = traceback.format_exc()
        print(err)
        try:
            import diagnose
            diagnose.log_error("find the score", err)
        except Exception:
            pass
        try:
            pygame.init()
            sc = pygame.display.set_mode((900, 260))
            font = pygame.font.SysFont("segoeui,arial", 17)
            sc.fill(BG)
            lines = ["Sorry - the Find-the-score window hit a problem:"] + err.strip().splitlines()[-4:] + [
                "", "Press 'Make a problem report' on the page and send it to Jonny.  (Close this window to exit.)"]
            for i, ln in enumerate(lines):
                sc.blit(font.render(ln[:110], True, WHITE if i else RED), (14, 14 + i * 26))
            pygame.display.flip()
            while not any(e.type == pygame.QUIT for e in pygame.event.get()):
                time.sleep(0.1)
        except Exception:
            pass
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
