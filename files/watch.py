"""Portable AI Arcade - WATCH: the newest brain plays one game after another in a window.
Keys: F = fast/normal, - and + = slow motion (100/75/50/25%), P = pause, G = 'greedy' (always its top choice),
M = sound, L = you play and it learns from you, B = replay its best games (the Hall of Fame), Esc = close.
Signed: Claude."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # embeddable Python ignores the script folder

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pygame
import torch
from PIL import Image
from torch.distributions.categorical import Categorical

from trainer import Agent
import best_games


os.environ.setdefault("SDL_VIDEO_CENTERED", "1")      # open in the middle of the screen


def bring_to_front(on_top=False):
    """Pop the window in front of the browser. Windows stops background programs stealing focus, so: briefly make it
    'topmost' (always works), then ask for focus (with the usual Alt-key nudge). on_top=True keeps it above everything."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        u = ctypes.windll.user32
        hwnd = pygame.display.get_wm_info()["window"]
        SWP = 0x0001 | 0x0002 | 0x0040                    # NOSIZE | NOMOVE | SHOWWINDOW
        u.ShowWindow(hwnd, 9)                              # SW_RESTORE
        u.SetWindowPos(hwnd, -1, 0, 0, 0, 0, SWP)          # HWND_TOPMOST
        if not on_top:
            u.SetWindowPos(hwnd, -2, 0, 0, 0, 0, SWP)      # HWND_NOTOPMOST: in front now, but not stuck there
        u.keybd_event(0x12, 0, 0, 0)                       # Alt down/up lets SetForegroundWindow work
        u.keybd_event(0x12, 0, 2, 0)
        u.SetForegroundWindow(hwnd)
    except Exception:
        pass


class Speaker:
    """Game sound for the Watch window. M turns it on/off. Plays through a short queue and drops anything that
    falls behind, so sound never makes the picture lag. If the PC has no sound device it just stays quiet."""

    def __init__(self, rate):
        self.ok, self.on, self.pending = False, True, []
        self.rate = int(rate)
        self.speed = 1.0                               # slow motion: the sound is stretched to match (plays lower)
        try:
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
        """stereo: int16 array (n, 2). Ignored when muted / no sound device / empty."""
        if not (self.ok and self.on) or stereo is None or not len(stereo):
            return
        if self.speed < 1 and len(stereo) > 1:         # slow motion: stretch it, like a tape played slower
            n = len(stereo)
            x = np.linspace(0, n - 1, int(n / self.speed))
            stereo = np.stack([np.interp(x, np.arange(n), stereo[:, c]) for c in range(stereo.shape[1])],
                              axis=1).astype(np.int16)
        self.pending.append(stereo)
        if self.chan.get_queue() is None:
            data = np.concatenate(self.pending)
            self.pending.clear()
            if len(data) > self.rate // 4:            # more than 1/4 s waiting = we're behind: keep the newest bit
                data = data[-self.rate // 8:]
            snd = pygame.sndarray.make_sound(np.ascontiguousarray(data))
            if self.chan.get_busy():
                self.chan.queue(snd)
            else:
                self.chan.play(snd)

    def label(self):
        return "no sound device" if not self.ok else ("sound ON" if self.on else "sound off")


TOP = 92                    # height of the info panel above the game picture (was 72)
ORANGE = (255, 150, 60)
LABEL = (175, 195, 222)
NUMBER = (255, 228, 92)     # the scores people look for are the brightest thing on the panel
KEYCOL = (212, 220, 234)
GREEN = (125, 232, 172)


class Hud:
    """The info panel above the game. Line 1: game, score, lives, game number. Line 2: the AI's average and best
    score in this window (bright yellow) and how far its brain has trained. Line 3: settings and keys. Line 4: the
    Learn-from-me line. A line too long for the window shrinks to fit instead of being cut off."""

    def __init__(self):
        self.fonts = {}

    def font(self, size):
        if size not in self.fonts:
            self.fonts[size] = pygame.font.SysFont("consolas", size, bold=True)
        return self.fonts[size]

    def line(self, screen, y, size, parts):
        """parts = [(text, colour), ...], drawn left to right on one line."""
        room = screen.get_width() - 20
        text = "".join(t for t, _ in parts)
        while size > 10 and self.font(size).size(text)[0] > room:
            size -= 1
        x = 10
        for t, colour in parts:
            img = self.font(size).render(t, True, colour)
            screen.blit(img, (x, y))
            x += img.get_width()

    def draw(self, screen, top_line, history, info, keys, teacher, hint=None):
        avg = f"{sum(history) / len(history):,.0f}" if history else "-"
        best = f"{max(history):,.0f}" if history else "-"
        games = f"{len(history)} game" + ("" if len(history) == 1 else "s")
        self.line(screen, 3, 20, [(top_line, ORANGE)])
        self.line(screen, 28, 17, [("AVERAGE ", LABEL), (avg, NUMBER), (f" ({games})    BEST ", LABEL),
                                   (best, NUMBER), (f"    {info}", LABEL)])
        self.line(screen, 50, 15, [(keys, KEYCOL)])
        if hint:
            self.line(screen, 70, 15, [(hint, NUMBER)])
        else:
            self.line(screen, 70, 15, [(teacher.hint(), ORANGE if teacher.on else GREEN)])


SPEEDS = (1.0, 0.75, 0.5, 0.25)                        # - and + step through these
SLOWER = (pygame.K_MINUS, pygame.K_KP_MINUS)
FASTER = (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS)


def change_speed(speed, way):
    """way -1 = one step slower, +1 = one step faster."""
    i = min(range(len(SPEEDS)), key=lambda k: abs(SPEEDS[k] - speed))
    return SPEEDS[max(0, min(len(SPEEDS) - 1, i - way))]


def key_line(fast, speed, greedy, speaker, replaying=False):
    """The keys, each with how it's set right now. In a replay the moves are fixed, so G and L don't apply."""
    pace = "FAST" if fast else f"{round(speed * 100)}%"
    sound = "none" if not speaker.ok else ("on" if speaker.on else "off")
    if replaying:
        return f"F fast  -/+ speed {pace}  P pause  G -  L -  M sound {sound}  B next replay  T on top  Esc close"
    return (f"F fast  -/+ speed {pace}  P pause  G greedy {'on' if greedy else 'off'}  M sound {sound}  "
            f"B best games  T on top  Esc close")


def play_screens(screen, hud, clock, replay, top_line, moves_per_second):
    """Fall-back replay: the 84x84 grey screens the brain saw, when a game can't be re-run in the emulator (Atari
    games, or a game recorded before replays existed). Returns "next" (B), "quit" (Esc) or "done"."""
    frames = replay["frames"]
    if not len(frames):
        return "done"
    W, H = screen.get_width(), screen.get_height() - TOP
    side = min(W, H)
    fast = paused = False
    speed, i = 1.0, 0
    while i < len(frames):
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                return "quit"
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_b:
                    return "next"
                if e.key == pygame.K_p:
                    paused = not paused
                elif e.key == pygame.K_f:
                    fast = not fast
                elif e.key in SLOWER or e.key in FASTER:
                    speed, fast = change_speed(speed, -1 if e.key in SLOWER else 1), False
        if paused:
            clock.tick(20)
            continue
        rgb = np.repeat(frames[i][:, :, None], 3, axis=2)
        surf = pygame.transform.scale(pygame.surfarray.make_surface(rgb.swapaxes(0, 1)), (side, side))
        screen.fill((9, 17, 30))
        screen.blit(surf, ((W - side) // 2, TOP))
        pace = "FAST" if fast else f"{round(speed * 100)}%"
        hud.line(screen, 3, 20, [(top_line, ORANGE)])
        hud.line(screen, 28, 17, [("REPLAY - what the AI saw   ", LABEL), (f"move {i + 1:,} of {len(frames):,}", NUMBER)])
        hud.line(screen, 50, 15, [(f"F fast  -/+ speed {pace}  P pause  B next replay  Esc close", KEYCOL)])
        hud.line(screen, 70, 15, [(replay["label"], NUMBER)])
        pygame.display.flip()
        clock.tick(240 if fast else max(1, round(moves_per_second * speed)))
        i += 1
    time.sleep(1.0)
    return "done"


def show_every_frame(cab, draw):
    """Watch window only: draw every emulator frame, not just the last one of each move, so the picture is smooth
    (and so is slow motion). It wraps this one cabinet's run(); training never sees it. Returns an on/off switch."""
    host, run = cab.env.host, cab.env.host.run
    switch = {"on": False}

    def run_and_show(buttons=(), *, capture=True):
        out = run(buttons, capture=capture or switch["on"])
        if switch["on"]:
            draw(host.frame)
        return out

    host.run = run_and_show
    return switch


RETRO_TOKENS = {4: "U", 5: "D", 6: "L", 7: "R", 0: "B1", 8: "B2", 1: "B3", 9: "B4", 10: "B5", 11: "B6"}


class Teacher:
    """L = you take over and the AI learns from you ("Learn from me"). Arrows + Z X C A S D (or a pad-free keyboard).
    Your moves are saved in the AI's files\\demos folder; the next Train copies you first, then fades it out."""
    KEYS = {pygame.K_UP: "U", pygame.K_DOWN: "D", pygame.K_LEFT: "L", pygame.K_RIGHT: "R", pygame.K_z: "B1",
            pygame.K_SPACE: "B1", pygame.K_x: "B2", pygame.K_c: "B3", pygame.K_a: "B4", pygame.K_s: "B5", pygame.K_d: "B6"}

    def __init__(self, proj, table, game_id):
        import demos
        self.demos, self.proj, self.table, self.game_id = demos, proj, table, game_id
        self.on, self.held, self.msg = False, set(), ""
        self.frames, self.actions, self.starts, self.fresh = [], [], [], True
        self.total = demos.count_moves(proj)

    def toggle(self, stack):
        self.on = not self.on
        self.held = set()
        if self.on:      # the 3 screens before you took over, as context (not lessons)
            for k in range(3):
                self.frames.append(stack[k].copy()); self.actions.append(-1); self.starts.append(k == 0)
            self.fresh = False
            self.msg = "You're playing - it's watching and learning. Press L to hand back."
        else:
            self.save()

    def key(self, e):
        if self.on and e.key in self.KEYS:
            (self.held.add if e.type == pygame.KEYDOWN else self.held.discard)(self.KEYS[e.key])

    def action(self):
        return self.demos.keys_to_action(self.table, self.held)

    def record(self, newest, a):
        self.frames.append(newest.copy()); self.actions.append(int(a)); self.starts.append(self.fresh)
        self.fresh = False

    def new_game(self):
        self.fresh = True

    def save(self):
        r = self.demos.save_demo(self.proj, self.frames, self.actions, self.starts, len(self.table), self.game_id)
        self.frames, self.actions, self.starts = [], [], []
        if r:
            self.total += r[1]
            self.msg = f"Saved {r[1]:,} of your moves ({self.total:,} in all) - it learns them while it trains"
        elif not self.on:
            self.msg = "Not enough of your play to keep - stay on a bit longer next time."

    def hint(self):
        if self.on:
            n = sum(1 for a in self.actions if a >= 0)
            return f"YOU'RE PLAYING ({n:,} moves) - arrows + Z X C A S D - L hands back"
        return self.msg or (f"L = you play and it learns from you ({self.total:,} moves saved so far)")


def load(run):
    for name in ("latest.pt", "best.pt"):
        p = run / name
        if p.exists():
            st = torch.load(p, map_location="cpu", weights_only=False)
            a = Agent(st["n_actions"], st.get("width", 1))
            a.load_state_dict(st["agent"], strict=False)
            a.eval()
            return a, f"brain after {st['global_step']/1e6:.1f}M moves"
    return None, "no brain yet"


def watch_arcade(proj, cfg, run):
    """Arcade version: one real cabinet, full-colour screen, same brain inputs as training (4 x 84x84 grey)."""
    from arcade_vec import Cabinet, PACKS, local_core, scratch_dir
    try:                    # the game's buttons changed since this brain last played? carry it over first
        import controls
        controls.ensure_fits(proj, print)
    except Exception as e:
        print("Couldn't carry the brain over to the game's new buttons:", e)
    rom = proj.parent.parent / "ROMs - put your games here" / cfg["rom_file"]
    rt = scratch_dir("watch")
    local_core(rt)
    cab = Cabinet(cfg["game_id"], rom, rt, random.randrange(1 << 20), 0.25 if cfg.get("sticky", True) else 0.0,
                  audio=True)
    obs = cab.new_game()
    h, w = cab.env.render().shape[:2]
    S = 3 if h < 300 else 2
    pygame.mixer.pre_init(int(cab.env.host.sample_rate), -16, 2, 1024)
    pygame.init()
    speaker = Speaker(cab.env.host.sample_rate)
    screen = pygame.display.set_mode((max(w * S, 640), h * S + TOP))
    pygame.display.set_caption(f"Arcade Apprentice - {cfg['game_name']}")
    bring_to_front()
    on_top = False
    hud = Hud()
    clock = pygame.time.Clock()
    fast = paused = greedy = False
    history, game = [], 0
    mod = sys.modules.get(type(cab.env).__module__)
    raw = getattr(cab.env, "actions", None) or getattr(mod, "ACTION_BUTTONS", None) or [()]
    raw = list(raw)[:PACKS[cfg["game_id"]]["n_actions"]]
    teacher = Teacher(proj, [None if t is None else tuple(RETRO_TOKENS.get(b, b) for b in t) for t in raw],
                      cfg["game_id"])
    sticky0 = cab.sticky
    speed = 1.0
    # B = the Hall of Fame: its record games (best first), then the games it's studying right now; B after the
    # last one = back to watching it live. Arcade games are re-run in the emulator: same start + same buttons.
    rp = {"list": None, "pos": -1, "on": None, "want": None, "note": ""}
    flag_last = {}

    def next_replay():
        if rp["list"] is None:
            rp["list"] = [e for e in best_games.list_replays(proj)
                          if e["meta"].get("recipe") is not None or e["meta"].get("has_screens")]
            rp["pos"] = -1
            if not rp["list"]:
                rp["list"] = None
                rp["note"] = ("No best games saved yet - tick 'Learn from its own best games' in Settings and train it; "
                              "its record games show up here.")
                return "live"
        rp["pos"] += 1
        if rp["pos"] >= len(rp["list"]):
            rp["list"], rp["pos"] = None, -1
            rp["note"] = "Back to watching it live. B = the best games again."
            return "live"
        e = rp["list"][rp["pos"]]
        try:
            r = best_games.load_replay(e["path"])
        except Exception:
            return next_replay()
        r["label"] = f"REPLAY {rp['pos'] + 1} of {len(rp['list'])} - {e['label']}   B = next"
        r["pos"] = 0
        rp["note"] = ""
        rp["dirty"] = True
        if r["meta"].get("recipe") is None:
            r["label"] = (f"REPLAY {rp['pos'] + 1} of {len(rp['list'])} - {e['label']} - this game can't be re-run in "
                          f"the emulator, so this is what the AI saw"
                          f"{' (its last part - it was a long one)' if r['meta'].get('screens_from') else ''}.  B = next")
        return r

    def show(frame):
        surf = pygame.transform.scale(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (w * S, h * S))
        screen.fill((9, 17, 30))
        screen.blit(surf, ((screen.get_width() - w * S) // 2, TOP))
        replaying = rp["on"] is not None
        top = (f"{cfg['game_name']}   REPLAY   SCORE {cab.score:6.0f}   LIVES {cab.lives}" if replaying else
               f"{cfg['game_name']}   SCORE {cab.score:6.0f}   LIVES {cab.lives}   GAME {game}")
        hud.draw(screen, top, history, info, key_line(fast, speed, greedy, speaker, replaying), teacher,
                 rp["on"]["label"] if replaying else (rp["note"] or None))
        pygame.display.flip()

    def show_and_wait(frame):
        show(frame)
        clock.tick(max(1, round(60 * speed)))         # the game's own 60 frames a second, times the speed

    smooth = show_every_frame(cab, show_and_wait)
    if len(sys.argv) > 2 and sys.argv[2] == "best":      # opened from the page's "Best game" button
        rp["want"] = next_replay()
    while True:
        try:                                    # its buttons were changed (Settings > Buttons)? switch over between games
            newer = controls.refresh_buttons(cab.env, cfg["game_id"])
            if newer:
                controls.ensure_fits(proj, print)
                teacher.table = [None if t is None else tuple(RETRO_TOKENS.get(b, b) for b in t) for t in cab.env.actions]
                rp["note"], rp["list"] = f"The game's buttons changed - it now uses {newer}.", None
        except Exception as e:
            print("Couldn't switch to the game's new buttons:", e)
        agent, info = load(run)
        want, rp["want"] = rp["want"], None
        rp["on"] = want if isinstance(want, dict) else None
        if rp["on"] is not None and rp["on"]["meta"].get("recipe") is None:
            # can't be re-run in the emulator: show the screens the brain saw instead
            res = play_screens(screen, hud, clock, rp["on"], f"{cfg['game_name']}   REPLAY",
                               60 / max(1, int(PACKS[cfg["game_id"]].get("decision_repeat", 4))))
            if res == "quit":
                return
            rp["want"] = next_replay() if res == "next" else None
            if res == "done":
                rp["list"], rp["pos"] = None, -1
            continue
        if rp["on"] is not None:
            cab.sticky = 0.0                           # the recorded buttons already include the sticky joystick
            try:
                obs = cab.start_from(rp["on"]["meta"]["recipe"])
            except Exception as ex:
                rp["note"] = f"Couldn't replay that game ({ex}) - back to watching it live."
                rp["on"] = None
        if rp["on"] is None:
            game += 1
            cab.sticky = sticky0
            if game > 1 or rp.get("dirty"):
                obs = cab.new_game()
                teacher.new_game()
        cab.env.drain_audio()                          # throw away the sound of the reset
        stack = np.zeros((4, 84, 84), np.uint8)
        stack[-1] = obs
        over = False
        while not over:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    if teacher.on:
                        teacher.save()
                    return
                if e.type in (pygame.KEYDOWN, pygame.KEYUP):
                    teacher.key(e)
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_l and rp["on"] is None:
                        teacher.toggle(stack)
                        cab.sticky = 0.0 if teacher.on else sticky0
                        fast = False
                    elif e.key == pygame.K_b and not teacher.on:
                        rp["want"] = next_replay()
                    elif e.key == pygame.K_f and not teacher.on:
                        fast = not fast
                    elif e.key == pygame.K_p:
                        paused = not paused
                    elif e.key == pygame.K_g and rp["on"] is None:
                        greedy = not greedy
                    elif e.key == pygame.K_t:
                        on_top = not on_top
                        bring_to_front(on_top)
                    elif e.key == pygame.K_m:
                        speaker.toggle()
                    elif e.key in SLOWER or e.key in FASTER:
                        speed, fast = change_speed(speed, -1 if e.key in SLOWER else 1), False
                        speaker.speed = speed
            if rp["want"] is not None:                 # B: stop this game and go to the next replay (or live)
                break
            if paused:
                clock.tick(20)
                continue
            if rp["on"] is not None:                   # replay: the buttons that actually happened, in order
                a = int(rp["on"]["acted"][rp["on"]["pos"]])
                rp["on"]["pos"] += 1
            elif teacher.on:
                a = teacher.action()
                teacher.record(stack[-1], a)       # one lesson per move, whatever the speed: slow motion = same lessons
            elif agent is None:
                a = 0
            else:
                with torch.no_grad():
                    logits, _ = agent(torch.from_numpy(stack[None]))
                    a = int(logits.argmax(-1)) if greedy else int(Categorical(logits=logits).sample())
            smooth["on"] = not fast                    # normal: every frame is drawn (and timed) as the move plays
            obs, pts, term, trunc = cab.step(a)
            smooth["on"] = False
            flag = getattr(cab.env, "flag", None)          # the "game on" memory byte: note every change (watch.log)
            if flag is not None and getattr(cab.env, "mem", None) is not None:
                fv = int(cab.env.mem[flag["offset"]])
                if fv != flag_last.get("v"):
                    print(f"{time.strftime('%H:%M:%S')} game {game} move {cab.decisions}: game-on byte = {fv} "
                          f"(counts as playing: {fv in flag['values']}) score {cab.score:.0f}", flush=True)
                    flag_last["v"] = fv
            if term or trunc:
                print(f"{time.strftime('%H:%M:%S')} game {game} ENDED at move {cab.decisions}: "
                      f"{'game over' if term else 'gave up (no points for 5 minutes)'} score {cab.score:.0f}", flush=True)
            over = term or trunc or (rp["on"] is not None and rp["on"]["pos"] >= len(rp["on"]["acted"]))
            snd = cab.env.drain_audio()
            if not fast:
                speaker.feed(snd.reshape(-1, 2) if len(snd) else None)
            stack = np.roll(stack, -1, axis=0)
            stack[-1] = obs
            if fast:
                show(cab.env.render())                 # FAST: just the last frame of each move, no waiting
        if teacher.on:
            teacher.save()                             # keep each game you play as you go
        if rp["on"] is not None:
            if rp["want"] is None:                     # the replay played to the end by itself
                rec = rp["on"]["meta"]["score"]
                rp["note"] = (f"Replay finished: {cab.score:,.0f} points - exactly as it played it. B = the best games again."
                              if abs(cab.score - rec) < 0.5 else
                              f"Replay finished on {cab.score:,.0f} points, but it scored {rec:,.0f} at the time - this "
                              f"game doesn't replay exactly. B = the best games again.")
                rp["list"], rp["pos"] = None, -1
            cab.sticky = sticky0
        elif rp["want"] is None:
            history.append(cab.score)
            rp["note"] = ""
        if rp["want"] is None:
            time.sleep(0.3 if fast else 1.5)


def main():
    proj = Path(sys.argv[1])
    cfg = json.loads((proj / "project.json").read_text())
    run = proj / "files" / "run"
    torch.set_num_threads(2)
    if cfg.get("platform") == "arcade":
        return watch_arcade(proj, cfg, run)
    from ale_py import ALEInterface, roms
    ale = ALEInterface()
    ale.setInt("random_seed", random.randrange(1 << 30))
    ale.setFloat("repeat_action_probability", 0.25 if cfg.get("sticky", True) else 0.0)
    ale.setInt("frame_skip", 1)
    ale.setBool("sound_obs", True)                 # 512 sound samples per frame (8-bit, 30,720 a second)
    ale.loadROM(roms.get_rom_path(cfg["game_id"]))
    actions = ale.getMinimalActionSet()
    h, w = ale.getScreenDims()
    S = 3
    pygame.mixer.pre_init(30720, -16, 2, 1024)
    pygame.init()
    speaker = Speaker(30720)
    screen = pygame.display.set_mode((w * S * 2, h * S + TOP))
    pygame.display.set_caption(f"Arcade Apprentice - {cfg['game_name']}")
    bring_to_front()
    on_top = False
    hud = Hud()
    clock = pygame.time.Clock()
    fast = paused = greedy = False
    speed = 1.0
    history, game = [], 0
    agent, info = load(run)

    def tokens(act):
        n = str(act).split(".")[-1]
        return tuple(t for t, k in (("U", "UP"), ("D", "DOWN"), ("L", "LEFT"), ("R", "RIGHT"), ("B1", "FIRE")) if k in n)
    teacher = Teacher(proj, [tokens(x) for x in actions], cfg["game_id"])
    while True:
        game += 1
        agent, info = load(run)
        teacher.new_game()
        ale.reset_game()
        for _ in range(random.randint(0, 30)):
            ale.act(actions[0])
        if 1 < len(actions) and "FIRE" in str(actions[1]):
            ale.act(actions[1])
        frames = [ale.getScreenGrayscale().copy()] * 2
        stack = np.zeros((4, 84, 84), np.uint8)
        score = 0
        while not ale.game_over():
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    if teacher.on:
                        teacher.save()
                    return
                if e.type in (pygame.KEYDOWN, pygame.KEYUP):
                    teacher.key(e)
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_l:
                        teacher.toggle(stack)
                        # (Atari: the emulator's own sticky joystick stays as it is - there's no 'cab' here)
                        fast = False
                    elif e.key == pygame.K_b and not teacher.on:
                        shows = [x for x in best_games.list_replays(proj) if x["meta"].get("has_screens")]
                        for k, x in enumerate(shows):
                            try:
                                r = best_games.load_replay(x["path"])
                            except Exception:
                                continue
                            r["label"] = f"REPLAY {k + 1} of {len(shows)} - {x['label']}   B = next"
                            res = play_screens(screen, hud, clock, r, f"{cfg['game_name']}   REPLAY", 15)
                            if res == "quit":
                                return
                            if res == "done":
                                break
                    elif e.key == pygame.K_f and not teacher.on:
                        fast = not fast
                    elif e.key == pygame.K_p:
                        paused = not paused
                    elif e.key == pygame.K_g:
                        greedy = not greedy
                    elif e.key == pygame.K_t:
                        on_top = not on_top
                        bring_to_front(on_top)
                    elif e.key == pygame.K_m:
                        speaker.toggle()
                    elif e.key in SLOWER or e.key in FASTER:
                        speed, fast = change_speed(speed, -1 if e.key in SLOWER else 1), False
                        speaker.speed = speed
            if paused:
                clock.tick(20)
                continue
            img = np.maximum(frames[-1], frames[-2])
            small_img = np.asarray(Image.fromarray(img).resize((84, 84), Image.BOX))
            stack = np.roll(stack, -1, axis=0)
            stack[-1] = small_img
            if teacher.on:
                a = teacher.action()
                teacher.record(stack[-1], a)
            elif agent is None:
                a = 0
            else:
                with torch.no_grad():
                    logits, _ = agent(torch.from_numpy(stack[None]))
                    a = int(logits.argmax(-1)) if greedy else int(Categorical(logits=logits).sample())
            frames = []
            for _ in range(4):
                score += ale.act(actions[a])
                frames.append(ale.getScreenGrayscale().copy())
                if not fast and speaker.on:
                    mono = (np.asarray(ale.getAudio(), np.int16) - 128) * 160
                    speaker.feed(np.repeat(mono[:, None], 2, axis=1))
                rgb = ale.getScreenRGB()
                surf = pygame.transform.scale(pygame.surfarray.make_surface(rgb.swapaxes(0, 1)), (w * S * 2, h * S))
                screen.fill((9, 17, 30))
                screen.blit(surf, (0, TOP))
                hud.draw(screen, f"{cfg['game_name']}   SCORE {score:6.0f}   GAME {game}", history, info,
                         key_line(fast, speed, greedy, speaker), teacher)
                pygame.display.flip()
                if not fast:
                    clock.tick(max(1, round(60 * speed)))   # 60 frames a second, times the speed
                if ale.game_over():
                    break
            if len(frames) < 2:
                frames = frames * 2
        if teacher.on:
            teacher.save()                             # keep each game you play as you go
        history.append(score)
        time.sleep(0.3 if fast else 1.5)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        try:
            import diagnose
            diagnose.log_error(f"watch {Path(sys.argv[1]).name}", tb)
        except Exception:
            pass
