"""Portable AI Arcade - trainer. PPO + IMPALA-CNN learning Atari 2600 games from pixels.
Works on any PC: ALE's multi-threaded C++ engine runs the games on the CPU; the brain uses an NVIDIA GPU
if there is one, otherwise the CPU (slower, still learns). Settings come from <project>/project.json,
written by the setup wizard. Signed: Claude, 9 Oct 2026.

  python trainer.py <project folder>        (start or resume; <project>/run/STOP = save and stop)
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # embeddable Python ignores the script folder

import csv
import json
import math
import os
import sys
import threading
import subprocess
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
# ALFRED 21 Oct 2026: the move-odds maths no longer double-checks its inputs on every bite. Each check made the
# processor wait for the graphics card (~1,000 waits per batch of learning - _dev/gpu_lab). Same maths, same results.
torch.distributions.Distribution.set_default_validate_args(False)

LESSON, PASSES = 2048, 4      # the learning recipe: the same on every PC (see THE LEARNING RECIPE in main)
POPART_BETA = 0.02            # how quickly the score predictor's sense of scale follows the scores (per batch)



# ------------------------------------------------------------------ the brain
def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c0, self.c1 = nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c1(torch.relu(self.c0(torch.relu(x))))


class ConvSeq(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv, self.pool = nn.Conv2d(cin, cout, 3, padding=1), nn.MaxPool2d(3, stride=2, padding=1)
        self.r0, self.r1 = ResBlock(cout), ResBlock(cout)

    def forward(self, x):
        return self.r1(self.r0(self.pool(self.conv(x))))


def fast_layout(model, dev):
    """On NVIDIA cards, keep the brain 'channels last' in memory: same brain, same results, but its lessons run ~29%
    faster and its moves ~20% faster on an RTX 4090 (measured 19 Oct 2026, _dev/test_channels_last.py)."""
    # ALFRED 20 Oct 2026: PAA_FAST_LAYOUT=0 switches it off - testing whether it's behind the nvrtc64_120_0.dll crashes
    # that started on 19 Oct evening (24 overnight)
    # the switch: a file called FAST_LAYOUT_OFF in the files folder (or PAA_FAST_LAYOUT=0) = the standard layout
    if os.environ.get("PAA_FAST_LAYOUT", "1") == "0" or (Path(__file__).resolve().parent / "FAST_LAYOUT_OFF").exists():
        return model
    if getattr(dev, "type", "") == "cuda" and not getattr(torch.version, "hip", None):
        model = model.to(memory_format=torch.channels_last)
        model.fast_layout = True
    return model


class Agent(nn.Module):
    """IMPALA-CNN actor-critic. 'eyes' = 3 conv blocks, then two heads: actor (button odds), critic (expected points)."""

    def __init__(self, n_actions, width=1):
        super().__init__()
        chans, layers, cin = [16 * width, 32 * width, 32 * width], [], 4
        for c in chans:
            layers.append(ConvSeq(cin, c))
            cin = c
        self.body = nn.Sequential(*layers, nn.Flatten(), nn.ReLU(), layer_init(nn.Linear(cin * 11 * 11, 256)), nn.ReLU())
        self.actor = layer_init(nn.Linear(256, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1.0)
        self.critic_int = layer_init(nn.Linear(256, 1), std=1.0)   # expected explorer bonus (used when it's on)
        # PopArt (van Hasselt et al. 2016 - Jonny's idea, 17 Oct 2026): the critic works in small numbers and its answer
        # is multiplied back up to points: points-to-come = critic x v_sigma + v_mu. The rewards themselves are never
        # touched. Brains saved before this have no v_* and start at x1 + 0 - exactly as before.
        self.register_buffer("v_mu", torch.zeros(()))
        self.register_buffer("v_nu", torch.ones(()))
        self.register_buffer("v_sigma", torch.ones(()))
        self.register_buffer("v_ready", torch.zeros(()))

    def forward(self, x, both=False):
        if getattr(self, "fast_layout", False) and x.dim() == 4:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            h = self.body(x.float() / 255.0)
            logits, v = self.actor(h), self.critic(h).squeeze(-1)
            vi = self.critic_int(h).squeeze(-1) if both else None
        v = v.float() * self.v_sigma + self.v_mu
        if both:
            return logits.float(), v, vi.float()
        return logits.float(), v

    @torch.no_grad()
    def rescale(self, returns, beta=POPART_BETA):
        """PopArt: keep track of how big the points-to-come really are (a running average and spread), and re-tune
        the critic's last layer at the same moment so its answers in points stay exactly the same as the scale moves."""
        r = returns.detach().float()
        if float(self.v_ready) < 0.5:
            beta = 1.0                                   # first time: take this batch's scale straight away
        mu = (1 - beta) * self.v_mu + beta * r.mean()
        nu = (1 - beta) * self.v_nu + beta * (r * r).mean()
        sigma = (nu - mu * mu).clamp(min=1e-4).sqrt().clamp(1e-2, 1e6)
        self.critic.weight.mul_(self.v_sigma / sigma)
        self.critic.bias.mul_(self.v_sigma).add_(self.v_mu - mu).div_(sigma)
        self.v_mu.copy_(mu)
        self.v_nu.copy_(nu)
        self.v_sigma.copy_(sigma)
        self.v_ready.fill_(1.0)


# ------------------------------------------------------------------ explorer bonus (curiosity about NEW screens)
class RunningMeanStd:
    def __init__(self, shape, dev):
        self.mean, self.var, self.count = torch.zeros(shape, device=dev), torch.ones(shape, device=dev), 1e-4

    def update(self, x):
        bm, bv, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        d, tot = bm - self.mean, self.count + n
        self.mean = self.mean + d * n / tot
        self.var = (self.var * self.count + bv * n + d.pow(2) * self.count * n / tot) / tot
        self.count = tot

    def state(self):
        return {"mean": self.mean.cpu(), "var": self.var.cpu(), "count": self.count}

    def load(self, st, dev):
        self.mean, self.var, self.count = st["mean"].to(dev), st["var"].to(dev), st["count"]


class RNDNet(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        layers = [nn.Conv2d(1, 32, 8, 4), nn.LeakyReLU(), nn.Conv2d(32, 64, 4, 2), nn.LeakyReLU(),
                  nn.Conv2d(64, 64, 3, 1), nn.LeakyReLU(), nn.Flatten(), nn.Linear(64 * 7 * 7, 512)]
        if predictor:
            layers += [nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Explorer:
    """Random Network Distillation (Burda et al. 2018). A fixed random network scrambles each screen; a second
    network learns to predict the scramble. Screens it has seen a lot are easy to predict; NEW screens aren't ->
    the prediction error is a bonus for going somewhere new. Exactly what Pitfall/Montezuma-type games need."""

    def __init__(self, dev, strength, n_envs):
        self.dev, self.beta = dev, float(strength)
        self.target, self.pred = RNDNet(False).to(dev), RNDNet(True).to(dev)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.opt = optim.Adam(self.pred.parameters(), lr=1e-4)
        self.obs_rms = RunningMeanStd((1, 84, 84), dev)
        self.rew_rms = RunningMeanStd((), dev)
        self.keep = min(1.0, 32.0 / max(1, n_envs))     # train the predictor on a slice so the bonus fades slowly
        self.gamma_i, self.rf = 0.99, None

    def _norm(self, x_u8):
        x = x_u8[:, -1:].float()
        return ((x - self.obs_rms.mean) / (self.obs_rms.var.sqrt() + 1e-8)).clamp(-5, 5)

    def bonus(self, obs_u8):
        """Novelty of each screen (normalised); obs_u8 (M,C,84,84) uint8 - the newest frame is used."""
        out = []
        with torch.no_grad():
            for k in range(0, obs_u8.shape[0], 4096):
                n = self._norm(obs_u8[k:k + 4096])
                out.append((self.pred(n) - self.target(n)).pow(2).mean(1))
        return torch.cat(out)

    def normalise(self, raw):
        """raw (S,N) -> bonus divided by the spread of its running discounted sum. The sum never resets at death:
        novelty is about the whole journey, not one life (Burda et al.'s non-episodic intrinsic return)."""
        S_, N_ = raw.shape
        if self.rf is None or self.rf.shape[0] != N_:
            self.rf = torch.zeros(N_, device=raw.device)
        rets = []
        for t in range(S_):
            self.rf = self.rf * self.gamma_i + raw[t]
            rets.append(self.rf)
        self.rew_rms.update(torch.stack(rets).reshape(-1))
        return raw / (self.rew_rms.var.sqrt() + 1e-8)

    def train_on(self, obs_u8):
        n = self._norm(obs_u8)
        err = (self.pred(n) - self.target(n).detach()).pow(2).mean(1)
        mask = (torch.rand(err.shape[0], device=self.dev) < self.keep).float()
        loss = (err * mask).sum() / mask.sum().clamp(min=1.0)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

    def state(self):
        return {"target": self.target.state_dict(), "pred": self.pred.state_dict(), "opt": self.opt.state_dict(),
                "obs_rms": self.obs_rms.state(), "rew_rms": self.rew_rms.state()}

    def load(self, st):
        self.target.load_state_dict(st["target"])
        self.pred.load_state_dict(st["pred"])
        self.opt.load_state_dict(st["opt"])
        self.obs_rms.load(st["obs_rms"], self.dev)
        self.rew_rms.load(st["rew_rms"], self.dev)


# ------------------------------------------------------------------ rewards (wizard choices)
def make_reward_fn(cfg):
    style = cfg.get("reward_style", "log")          # equal | log | raw
    life_penalty = float(cfg.get("life_penalty", 0.0))
    alive_bonus = float(cfg.get("alive_bonus", 0.0))

    ignore_losses = bool(cfg.get("ignore_losses", False))

    def fn(points, life_lost):
        if ignore_losses:
            points = np.maximum(points, 0.0)      # e.g. Pitfall's log bumps: don't make it scared to run
        if style == "equal":
            r = np.sign(points)
        elif style == "raw":
            r = points / max(1.0, float(cfg.get("raw_scale", 100.0)))
        else:
            r = np.sign(points) * np.log10(1.0 + np.abs(points))   # losses (e.g. Pong) count as minus
        return r - life_penalty * life_lost + alive_bonus
    return fn


def atomic_save(obj, path: Path):
    """Save to a temporary file, make sure it's really on the disk, then swap it in - so a crash or power cut can't
    leave a half-written brain behind (17 Oct 2026: a crash 2 seconds after a save left latest.pt unreadable)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    for i in range(10):
        try:
            os.replace(tmp, path)
            return
        except OSError:
            time.sleep(1 + i)


def main():
    proj = Path(sys.argv[1]).resolve()
    cfg = json.loads((proj / "project.json").read_text())
    run = proj / "files" / "run"
    run.mkdir(parents=True, exist_ok=True)
    (run / "milestones").mkdir(exist_ok=True)
    log_f = open(run / "console.log", "a", buffering=1, encoding="utf-8")

    def say(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")

    # WATCHDOG: if training stops moving for 10 minutes (it froze once, 11 Oct 2026, 03:45, holding the graphics
    # card), leave a FROZE note and end this process and its game workers. keepgoing.py / evolution.py see the note
    # and start again from the last save.
    beat = {"t": time.time()}
    _start_watchdog(run, beat, say)

    hw = dict(cfg["hardware"])
    go_explore = float(cfg.get("go_explore", 0) or 0) if cfg.get("platform", "atari") != "arcade" else 0.0
    live_width = int(hw.get("width", 1))
    try:  # re-size for THIS machine (the AI may have been made on another PC); the brain's width must stay the same
        from app import hardware
        # Go-Explore runs one game per process (like arcade), so it's sized like arcade
        live = hardware("arcade" if go_explore else cfg.get("platform", "atari"))
        live_width = int(live.get("width", 1))
        for k in ("use_gpu", "gpu_name", "cpu_name", "num_envs", "emu_threads", "epochs", "minibatch_size", "minibatches",
                  "micro_batch", "trainer_cpus"):
            if k in live:
                hw[k] = live[k]
    except Exception as e:
        print("Using the saved settings (couldn't re-check this PC):", e)
    if os.environ.get("PAA_ENVS"):              # testing only: fewer games at once
        hw["num_envs"] = int(os.environ["PAA_ENVS"])
    import device as device_mod
    dev, dev_name, backend, problem = device_mod.pick(prefer_gpu=bool(hw.get("use_gpu", True)))
    if problem:
        say(f"Note: {problem} - training on the CPU instead.")
    hw["gpu_name"] = dev_name if backend != "cpu" else None
    sample_on_cpu = backend == "directml"      # DirectML lacks some sampling ops: pick buttons on the CPU (tiny job)
    if backend == "cpu":
        torch.set_num_threads(max(1, int(hw.get("cpu_threads", 4)) // 2))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    if backend != "cpu":
        torch.set_num_threads(2)   # leave the CPU threads to the game emulators
    N, S = int(hw["num_envs"]), int(hw["num_steps"])
    batch = N * S
    # THE LEARNING RECIPE (17 Oct 2026) is the same on every PC: lessons of LESSON moves, PASSES passes over each
    # batch. The PC only changes how FAST that goes: how many games run at once, and the size of the bites a lesson
    # is worked through in when the graphics card can't hold a whole lesson at once (same lesson, smaller spoon).
    # Before, a lesson was "the biggest chunk the card could hold", so a bigger card meant fewer lessons per move:
    # an RTX 4090 got 6 lessons per 8,192 moves, against 16 now.
    n_lessons = max(1, round(batch / LESSON))
    lesson = math.ceil(batch / n_lessons)
    bite = int(os.environ.get("PAA_BITE") or hw.get("micro_batch") or hw.get("minibatch_size") or lesson)
    bite = max(64, min(lesson, bite))
    checker = None                          # the button check during training (controls.LiveCheck)
    if cfg.get("platform") == "arcade":
        from arcade_vec import ArcadeVec
        rom = (Path(cfg["rom_path"]) if cfg.get("rom_path")          # evolution's brains live deeper in the folder
               else proj.parent.parent / "ROMs - put your games here" / cfg["rom_file"])
        if not rom.exists():
            raise FileNotFoundError(f"The game file '{cfg['rom_file']}' isn't in the ROMs folder any more - put it back.")
        try:
            import controls
            checker = controls.LiveCheck(proj, cfg, say)
        except Exception as e:
            say(f"(Button check off this time: {e!r})")
        envs = ArcadeVec(cfg["game_id"], rom, N, seed=int(time.time()) % 100000,
                         sticky=0.25 if cfg.get("sticky", True) else 0.0,
                         probe=checker.inputs() if checker is not None else None,
                         practice=float(cfg.get("practice", 0) or 0))      # ALFRED 23 Oct 2026
        try:     # ALFRED 21 Oct 2026: learning gets fast cores of its own, the cabinets the rest (cpu_split.py)
            import cpu_split
            _split = cpu_split.plan(int(hw["trainer_cpus"]) if hw.get("trainer_cpus") else None)
            if _split:
                say(cpu_split.apply(os.getpid(), [p.pid for p in getattr(envs, "procs", [])], _split))
        except Exception as e:
            say(f"(Core split off this time: {e!r})")
    elif go_explore:
        from goexplore import GoExploreVec
        envs = GoExploreVec(cfg["game_id"], N, seed=int(time.time()) % 100000,
                            sticky=0.25 if cfg.get("sticky", True) else 0.0, p_return=go_explore,
                            archive_path=run / "archive.pkl")
    else:
        from ale_py.vector_env import AtariVectorEnv
        envs = AtariVectorEnv(cfg["game_id"], num_envs=N, num_threads=int(hw.get("emu_threads", 0)),
                              repeat_action_probability=0.25 if cfg.get("sticky", True) else 0.0,
                              reward_clipping=False, episodic_life=False, use_fire_reset=True, noop_max=30)
    n_act = envs.single_action_space.n
    if cfg.get("platform") == "arcade":         # the game's buttons changed since this brain last played? carry it over
        try:
            import controls
            controls.ensure_fits(proj, say, dev)
        except Exception as e:
            say(f"Couldn't carry the brain over to the game's new buttons ({e!r}).")
    st = torch.load(run / "latest.pt", map_location="cpu", weights_only=False) if (run / "latest.pt").exists() else None
    if st and int(st.get("n_actions", n_act)) != int(n_act):
        raise RuntimeError(f"This AI's brain knows {st.get('n_actions')} button moves but the game now has {n_act}, and "
                           "the kit can't tell what the old ones were - press Reset training in Settings to start afresh")
    want_w = {"normal": 1, "large": 2}.get(cfg.get("brain_size"), int(hw.get("width", 1)))
    width = int(st.get("width", 1)) if st else want_w
    if st and width != want_w:
        say(f"Note: brain size change ({'large' if want_w == 2 else 'normal'}) needs a fresh start - press Reset training "
            "in Settings to use it. Carrying on with the current brain.")
    if width == 2 and live_width == 1 and backend != "cpu":
        bite = max(64, bite // 2)                    # large brain on a mid-size card: smaller bites, same lessons
    hw["width"] = width
    agent = fast_layout(Agent(n_act, width).to(dev), dev)
    lr = float(cfg.get("learning_rate", 2.5e-4))
    opt = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    total = float(cfg.get("total_steps", 1e10))
    gamma, lam, clip = float(cfg.get("gamma", 0.99)), 0.95, float(cfg.get("clip", 0.1))   # clip: an evolution gene
    ent_coef = float(cfg.get("curiosity", 0.01))
    popart = bool(cfg.get("popart", True))     # the score predictor works in small numbers (Agent.rescale)
    reward_fn = make_reward_fn(cfg)
    lives_at_go = bool(cfg.get("lives_at_game_over", False))   # ALFRED 21 Oct 2026: see 'r = reward_fn' below
    miss_price = float(cfg.get("miss_price", 0) or 0)          # ALFRED 23 Oct 2026: 'A missed shot costs'
    practising = float(cfg.get("practice", 0) or 0) > 0 and cfg.get("platform") == "arcade"

    explorer = Explorer(dev, cfg["explorer"], N) if float(cfg.get("explorer", 0) or 0) > 0 else None
    views = set()                           # rough count of different screens seen (exploration progress)
    if (run / "views.npy").exists():
        try:
            views = set(np.load(run / "views.npy").tolist())
        except Exception:
            pass
    hash_w = np.random.default_rng(12345).integers(1, 2**61, size=14 * 14, dtype=np.int64)

    step, it0, best = 0, 1, -1e9
    if st:
        agent.load_state_dict(st["agent"], strict=False)
        if st.get("opt"):
            opt.load_state_dict(st["opt"])
        for gp in opt.param_groups:
            gp["lr"] = lr                    # the learning-speed dial may have been changed since
        step, it0, best = st["global_step"], st["iteration"] + 1, st.get("best", -1e9)
        if explorer and st.get("explorer"):
            try:
                explorer.load(st["explorer"])
            except Exception:
                say("Explorer bonus restarted (saved one didn't fit).")
        say(f"RESUMED at {step:,} decisions")
    if go_explore:
        stx = envs.stats()
        say(f"Go-Explore ON ({int(go_explore * 100)}% of games restart from places on its map; map has "
            f"{stx['places']:,} places).")
    if explorer:
        say(f"Explorer bonus ON (strength {explorer.beta}): rewarded for reaching screens it hasn't seen much.")
    if miss_price:
        say(f"A MISSED SHOT COSTS {miss_price:g} (off the top of the screen, or into its own shields; shooting down a "
            "bomb is free). Only games that can tell a miss - for the others this does nothing.")
    if practising:
        say(f"PRACTISING THE HARD PARTS: about {100 * float(cfg['practice']):.0f}% of games start from moments kept from "
            "its own games (the last few invaders, the later waves) - they don't count on the scoreboard.")
    say(f"{cfg['game_name']} | PPO IMPALA-CNN on {backend.upper()} ({dev_name}) | "
        f"{N} games at once x {S} steps | {n_lessons * PASSES} lessons of {lesson:,} moves per batch"
        f"{f' (in bites of {bite:,})' if bite < lesson else ''} | {n_act} buttons | rewards: {cfg.get('reward_style', 'log')}")

    # ---- LEARN FROM ME: copy the player's own moves (recorded with L in the Watch window) before / while training.
    # Read at the start of a Train AND whenever new recordings appear mid-run (checked once per batch), so you can
    # play in the Watch window while it trains and it learns your games straight away.
    import demos
    lfm = float(cfg.get("learn_from_me", 1.0) or 0)
    bank, demo_frames, demo_step = None, (st or {}).get("demo_frames", 0), (st or {}).get("demo_step", 0)
    demo_sig = None

    def demo_files():
        """Names + sizes of the saved recordings: a cheap way to notice new ones."""
        try:
            d = demos.demo_dir(Path(cfg.get("demos_from") or proj))
            return tuple(sorted((p.name, p.stat().st_size) for p in d.glob("demo-*.npz"))) if d.exists() else ()
        except Exception:
            return None

    def absorb(now, starting):
        nonlocal bank, demo_frames, demo_step, demo_sig
        demo_sig = demo_files()
        if lfm <= 0:
            return
        try:
            data = demos.load_demos(Path(cfg.get("demos_from") or proj), n_act)
            if data is not None:
                bank = demos.DemoBank(data, dev)
        except Exception as e:
            say(f"Couldn't read your recorded games ({e!r}) - training without them.")
        if bank is None or bank.n <= demo_frames:
            if bank is not None and starting:
                left = max(0.0, 1 - (now - demo_step) / demos.FADE)
                say(f"Learning from {bank.n:,} of your moves ({100 * left:.0f}% strength left)." if left > 0 else
                    f"Your {bank.n:,} recorded moves have done their job - it's on its own now.")
            return
        # new lessons: copy them first (behaviour cloning), then let the pull fade over the next few million moves
        ups = int(min(3000, max(300, 20 * bank.n / 256)))
        if starting:
            say(f"LEARNING FROM YOU: {bank.n:,} of your moves. Copying you first ({ups} quick lessons)...")
        else:
            say(f"NEW GAMES FROM YOU: {bank.n - demo_frames:,} more of your moves ({bank.n:,} in all). "
                f"Copying you ({ups} quick lessons) without stopping...")
        hits = tot = 0
        k = 0
        while k < ups:
            for xb, ab in bank.all_batches(256):
                logits, _ = agent(xb)
                loss = nn.functional.cross_entropy(logits, ab)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                opt.step()
                k += 1
                if k >= ups:
                    break
        with torch.no_grad():
            for xb, ab in bank.all_batches(1024):
                hits += int((agent(xb)[0].argmax(-1) == ab).sum())
                tot += len(ab)
        say(f"It now picks the same button as you {100 * hits / max(1, tot):.0f}% of the time on your recordings. "
            f"Your lessons fade out over its next {demos.FADE / 1e6:.0f}M moves as it learns to beat you.")
        demo_frames, demo_step = bank.n, now

    absorb(step, True)

    # ---- LEARN FROM ITS OWN BEST GAMES (best_games.py): games that beat its average are studied, then thrown away;
    # new records go to the Hall of Fame for watching (Watch window, B). Off unless switched on in the settings.
    import best_games
    studier = recorder = None
    if cfg.get("learn_from_best", False):
        studier = best_games.Studier(proj, dev, int(cfg.get("best_study_times", 1) or 1), n_act, cfg["game_id"],
                                     cfg.get("game_name", ""), say)
        recorder = best_games.Recorder(N, best_games.memory_budget())
        times = int(cfg.get("best_study_times", 1) or 1)
        say(f"LEARNING FROM ITS OWN BEST GAMES: any game that beats its average joins a queue of {best_games.QUEUE_MAX}, "
            f"is studied {'once' if times == 1 else f'{times} times'}, then thrown away. Records go to the Hall of Fame "
            f"(Watch window, press B). Games longer than {recorder.cap:,} moves are studied from their last {recorder.cap:,}.")

    (run / "status.json").write_text(json.dumps({"state": "training", "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                 "step": step, "best_mean100": best, "pid": os.getpid(), "device": backend}))
    obs, info = envs.reset(seed=int(time.time()) % 100000)
    obs_t = torch.from_numpy(np.ascontiguousarray(obs)).to(dev)
    obs_np = np.asarray(obs)
    lives = np.asarray(info.get("lives", np.zeros(N)))
    ep_score = np.zeros(N)
    ep_len = np.zeros(N, dtype=np.int64)
    ep_restart = np.zeros(N, dtype=bool)
    recent = deque(maxlen=100)
    aim = deque(maxlen=100)          # ALFRED 23 Oct 2026: (shots, hits) of its last 100 real games - the accuracy meter
    drills = {"n": 0}                # practice games started ('Practise the hard parts')
    # ALFRED 20 Oct 2026: carry its last 100 scores over a restart, so "better than its average" (best games)
    # means its real average straight away. A fresh trainer used to start with an empty list and study nearly every
    # game - evolution restarts the trainer every turn, so it spent its time studying (9 s a batch, not 5 s).
    try:
        _gj = run / "games.jsonl"
        if not _gj.exists() and cfg.get("demos_from"):     # an evolution brain's first turn: the family's own games
            _gj = Path(cfg["demos_from"]) / "files" / "run" / "games.jsonl"
        if _gj.exists():
            with open(_gj, encoding="utf-8") as _fh:
                for _l in deque(_fh, maxlen=400):
                    try:
                        _x = json.loads(_l)
                    except ValueError:
                        continue
                    if not _x.get("restart") and "score" in _x:
                        recent.append(float(_x["score"]))
                        if _x.get("shots"):
                            aim.append((int(_x["shots"]), int(_x.get("hits", 0))))
    except OSError:
        pass
    # OVERLAP (NVIDIA/AMD GPUs): while the brain learns from the last batch on a background thread, the games keep
    # playing the next batch with a copy of the brain from just before (PPO's ratio already corrects for that lag).
    # Keeps the CPU (games) and GPU (learning) busy at the same time instead of taking turns.
    overlap = backend in ("cuda", "rocm") and cfg.get("overlap", True)
    if overlap:
        actor = fast_layout(Agent(n_act, width).to(dev), dev)
        actor.load_state_dict(agent.state_dict())
    else:
        actor = agent

    def new_bufs():
        return {"o": torch.zeros((S, N, 4, 84, 84), dtype=torch.uint8, device=dev),
                "a": torch.zeros((S, N), dtype=torch.long, device=dev),
                **{k: torch.zeros((S, N), device=dev) for k in ("lp", "r", "d", "v", "vi")}}
    bufs = [new_bufs(), new_bufs()] if overlap else [new_bufs()]
    best_game = 0.0
    try:
        with open(run / "games.jsonl") as f:
            # ALFRED 26 Oct 2026: games from the very start only - practice/restart games never count on the scoreboard
            best_game = max((r["score"] for r in (json.loads(l) for l in f if l.strip()) if not r.get("restart")),
                            default=0.0)
    except Exception:
        pass
    g = {"obs_t": obs_t, "next_done": torch.zeros(N, device=dev), "lives": lives, "step": step, "games": 0,
         "best_game": best_game, "obs_np": obs_np}
    g["record_bar"] = best_games.hof_best(proj)    # ALFRED 23 Oct 2026: see hof_best in best_games.py

    def collect(B, policy):
        """The games' moves for one batch - on the high-priority card stream when lessons are recorded (_graphs)."""
        if _graphs["hi"] is None:
            return _collect_eager(B, policy)
        _graphs["hi"].wait_stream(torch.cuda.current_stream())   # the brain was just copied into the actor
        with torch.cuda.stream(_graphs["hi"]):
            out = _collect_eager(B, policy)
        _graphs["hi"].synchronize()              # the batch is complete before the learning reads it
        return out

    def _collect_eager(B, policy):
        """Play S moves in all N games, filling buffer B. Returns (seconds in games, seconds in the brain)."""
        nonlocal ep_score, ep_len, ep_restart
        t_env = t_brain = 0.0
        obs_t, next_done, lives = g["obs_t"], g["next_done"], g["lives"]
        obs_np = g["obs_np"]
        for s in range(S):
            tb = time.time()
            B["o"][s], B["d"][s] = obs_t, next_done
            with torch.no_grad():
                if explorer:
                    logits, v, vi = policy(obs_t, both=True)
                    B["vi"][s] = vi
                else:
                    logits, v = policy(obs_t)
                if sample_on_cpu:
                    logits = logits.cpu()
                dist = Categorical(logits=logits)
                a = dist.sample()
            B["a"][s], B["lp"][s], B["v"][s] = a, dist.log_prob(a), v
            acts = a.cpu().numpy()
            te = time.time()
            t_brain += te - tb
            obs, rew, term, trunc, info = envs.step(acts)
            t_env += time.time() - te
            if checker is not None and info.get("probes"):
                checker.add(info["probes"])
            if recorder is not None:
                recorder.add(obs_np, acts, info.get("acted", acts))
            new_lives = np.asarray(info.get("lives", lives))
            life_lost = (new_lives < lives) & ~(term | trunc)
            rs = info.get("restart_score")
            if rs is not None:                       # Go-Explore restarted these games from a place on the map
                rmask = np.asarray(rs) >= 0
                if rmask.any():
                    drills["n"] += int(rmask.sum())
                    life_lost &= ~rmask
                    ep_score[rmask] = np.asarray(rs)[rmask]
                    ep_restart[rmask] = True
            ended = term | trunc
            ep_score += rew
            ep_len += 1
            lost = life_lost.astype(np.float32)
            if lives_at_go:                  # ALFRED 21 Oct 2026: a game over costs every life it had left (the
                go = np.asarray(term, bool) & ~np.asarray(trunc, bool)   # invaders landing with lives to spare)
                if go.any():
                    lost[go] = np.maximum(np.asarray(lives, np.float32)[go], 1.0)
            r = reward_fn(rew, lost)
            if miss_price and info.get("misses") is not None:      # ALFRED 23 Oct 2026: 'A missed shot costs'
                r = r - miss_price * np.asarray(info["misses"], np.float32)
            for i in np.flatnonzero(ended):
                g["games"] += 1
                if recorder is not None:
                    played = recorder.finish(i)
                    if not ep_restart[i]:
                        sc = float(ep_score[i])
                        avg = float(np.mean(recent)) if len(recent) >= best_games.MIN_GAMES else float("nan")
                        recipe = (info.get("ended_recipes") or [None] * N)[i]
                        if sc > g["record_bar"] and sc > 0:      # ALFRED 23 Oct 2026: beats the Hall of Fame's own
                            studier.record(played, sc, recipe, g["step"])   # best (not the scoreboard's) - see hof_best
                            g["record_bar"] = sc
                        studier.offer(played, sc, avg, recipe, g["step"])
                gs = (info.get("game_stats") or [None] * N)[i]
                if not ep_restart[i]:                # only games played from the very start count for the score
                    recent.append(float(ep_score[i]))
                    g["best_game"] = max(g["best_game"], float(ep_score[i]))
                    if gs and gs.get("shots"):
                        aim.append((int(gs["shots"]), int(gs["hits"])))
                row_g = {"step": g["step"], "score": float(ep_score[i]), "frames": int(ep_len[i] * 4),
                         "capped": bool(trunc[i] and not term[i]), "restart": bool(ep_restart[i])}
                if gs:
                    row_g.update(gs)                 # shots, hits, misses (games whose adapter counts them)
                with open(run / "games.jsonl", "a") as f:
                    f.write(json.dumps(row_g) + "\n")
                ep_score[i], ep_len[i], ep_restart[i] = 0, 0, False
            lives = new_lives
            B["r"][s] = torch.from_numpy(np.asarray(r, dtype=np.float32)).to(dev)
            next_done = torch.from_numpy((ended | life_lost).astype(np.float32)).to(dev)
            obs_t = torch.from_numpy(np.ascontiguousarray(obs)).to(dev)
            obs_np = np.asarray(obs)
            if len(views) < 3_000_000:
                small = (np.asarray(obs)[:, -1, ::6, ::6] >> 5).reshape(N, -1).astype(np.int64)
                views.update((small @ hash_w).tolist())
            g["step"] += N
        with torch.no_grad():                        # value of where the batch left off (for the last moves' credit)
            if explorer:
                _, B["nv"], B["nvi"] = policy(obs_t, both=True)
            else:
                B["nv"] = policy(obs_t)[1]
        B["next_done"] = next_done
        B["last_o"] = obs_t.clone()
        g["obs_t"], g["next_done"], g["lives"] = obs_t, next_done, lives
        g["obs_np"] = obs_np
        return t_env, t_brain

    # ---- ALFRED 21 Oct 2026 (review): THE WHOLE LESSON AS ONE RECORDED CARD JOB (CUDA graph). The maths is the very
    # same bite loop below; the card just replays it without the processor queuing ~7,000 tiny jobs a lesson (the
    # 21 Oct profile: the processor's queuing took as long as the card's work, and each lesson began with a wait).
    # PAA_GRAPH=0 switches it off. It switches itself off on a CPU/DirectML backend, with the explorer on, when the
    # lessons aren't all the same size, or if recording fails (then it carries on the slow way, with a note).
    _graphs = {"on": (dev.type == "cuda" and os.environ.get("PAA_GRAPH", "1") != "0" and not explorer
                      and batch % n_lessons == 0 and lesson % bite == 0),
               "pool": None, "by_buf": {}, "i": None, "ma": None, "a": None, "lp": None, "nv": None, "nret": None,
               "ent": None, "stream": None, "hi": None, "pending": []}
    if _graphs["on"]:        # recorded lessons run on their own card stream, and the games pick their moves on a
        _graphs["stream"] = torch.cuda.Stream()                 # HIGH-priority stream: the card serves the games'
        _graphs["hi"] = torch.cuda.Stream(priority=-1)          # small jobs first, so a whole lesson in progress
        torch.cuda.synchronize()                                # doesn't make every move wait (lab 3: G2/G6 vs G10)
        # Two streams = two queues: anything one thread copies to the card that the other thread reads must have
        # landed first (12:04 crash: a best game's screens, read as indices before the copy finished - demos.DemoBank
        # now waits for its copies and records the stream it's read on).

    def _graph_bites(b_o, vmu, vsig):
        """The bite loop over the static tensors: what gets recorded (the warm-ups run it the ordinary way)."""
        for p in agent.parameters():
            if p.grad is not None:
                p.grad.zero_()
        ent = None
        for k in range(0, lesson, bite):
            j = slice(k, k + bite)
            ii, ma = _graphs["i"][j], _graphs["ma"][j]
            logits, nvv = agent(b_o[ii])
            dist = Categorical(logits=logits)
            ratio = (dist.log_prob(_graphs["a"][ii]) - _graphs["lp"][ii]).exp()
            pg = torch.max(-ma * ratio, -ma * ratio.clamp(1 - clip, 1 + clip)).mean()
            nvn = nvv if vmu is None else (nvv - vmu) / vsig
            vc = _graphs["nv"][ii] + (nvn - _graphs["nv"][ii]).clamp(-clip, clip)
            vl = 0.5 * torch.max((nvn - _graphs["nret"][ii]) ** 2, (vc - _graphs["nret"][ii]) ** 2).mean()
            ent = dist.entropy().mean()
            ((pg - ent_coef * ent + 0.5 * vl) * (bite / lesson)).backward()
        _graphs["ent"].copy_(ent.detach())

    def _graph_throttle():
        """Keep at most two lessons queued on the card: the processor waits for the one before last (letting the
        games' thread run meanwhile) instead of blocking inside a full launch queue (lab 3: G10 vs G13)."""
        ev = torch.cuda.Event()
        ev.record()
        _graphs["pending"].append(ev)
        if len(_graphs["pending"]) > 2:
            _graphs["pending"].pop(0).synchronize()

    def _graph_for(b_o, b_a, b_lp, n_v, n_ret, vmu, vsig):
        """The recorded lesson for this batch's screen buffer (one recording per buffer), or None = the slow way."""
        if not _graphs["on"]:
            return None
        try:
            if _graphs["i"] is None:                      # first time: the static tensors every recording reads
                _graphs["i"] = torch.zeros(lesson, dtype=torch.long, device=dev)
                _graphs["ma"] = torch.zeros(lesson, device=dev)
                _graphs["a"] = torch.zeros(batch, dtype=torch.long, device=dev)
                for k in ("lp", "nv", "nret"):
                    _graphs[k] = torch.zeros(batch, device=dev)
                _graphs["ent"] = torch.zeros((), device=dev)
            _graphs["a"].copy_(b_a)
            _graphs["lp"].copy_(b_lp)
            _graphs["nv"].copy_(n_v)
            _graphs["nret"].copy_(n_ret)
            key = b_o.data_ptr()
            g = _graphs["by_buf"].get(key)
            if g is None:
                _graphs["i"].copy_(torch.arange(lesson, device=dev))
                _graphs["ma"].zero_()
                if not _graphs["by_buf"]:                 # before the first recording: give every weight a gradient
                    opt.zero_grad(set_to_none=True)       # tensor to accumulate into (the recordings write into
                    _graph_bites(b_o, vmu, vsig)          # these same tensors from now on - never set to None again)
                ptrs = [p.grad.data_ptr() if p.grad is not None else None for p in agent.parameters()]
                s = torch.cuda.Stream()                   # warm-ups on a side stream first (PyTorch's advice)
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        _graph_bites(b_o, vmu, vsig)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=_graphs["pool"], capture_error_mode="thread_local"):
                    _graph_bites(b_o, vmu, vsig)
                if [p.grad.data_ptr() if p.grad is not None else None for p in agent.parameters()] != ptrs:
                    raise RuntimeError("the gradients moved while recording")
                _graphs["pool"] = _graphs["pool"] or g.pool()
                _graphs["by_buf"][key] = g
                for p in agent.parameters():              # the warm-ups' gradients are not a lesson
                    if p.grad is not None:
                        p.grad.zero_()
                say(f"Lesson recorded for the graphics card (recording {len(_graphs['by_buf'])} of 2; "
                    f"PAA_GRAPH=0 turns this off).")
            return g
        except Exception as e:
            _graphs["on"] = False
            _graphs["by_buf"].clear()
            opt.zero_grad(set_to_none=True)
            say(f"(Lesson recording off this time: {type(e).__name__}: {str(e)[:160]})")
            return None

    def learn(B):
        """The learning for one batch - on its own card stream when lessons are recorded (see _graphs above)."""
        if _graphs["stream"] is None:
            return _learn_eager(B)
        _graphs["stream"].wait_stream(torch.cuda.current_stream())   # the batch was filled on the games' stream
        with torch.cuda.stream(_graphs["stream"]):
            return _learn_eager(B)

    def _learn_eager(B):
        """PPO: work out which moves beat expectations, then nudge the brain towards them (a few passes)."""
        t_l = time.time()
        r_buf, d_buf, v_buf, nv, next_done = B["r"], B["d"], B["v"], B["nv"], B["next_done"]
        novelty = 0.0
        if explorer:
            # the screen each move led to (newest frame only)
            nxt_o = torch.cat([B["o"][1:, :, -1:], B["last_o"][None, :, -1:]]).reshape(-1, 1, 84, 84)
            explorer.obs_rms.update(nxt_o[:, -1:].float())
            r_int = explorer.normalise(explorer.bonus(nxt_o).reshape(S, N))
            novelty = float(r_int.mean())
            vi_buf, nvi = B["vi"], B["nvi"]
            with torch.no_grad():                    # explorer's own credit-assignment: never cut at death
                adv_i = torch.zeros_like(r_int)
                last_i = torch.zeros(N, device=dev)
                for t in reversed(range(S)):
                    nxt_i = nvi if t == S - 1 else vi_buf[t + 1]
                    last_i = r_int[t] + explorer.gamma_i * nxt_i - vi_buf[t] + explorer.gamma_i * lam * last_i
                    adv_i[t] = last_i
                ret_i = adv_i + vi_buf
        with torch.no_grad():
            adv = torch.zeros_like(r_buf)
            last = torch.zeros(N, device=dev)
            for t in reversed(range(S)):
                nnt = 1.0 - (next_done if t == S - 1 else d_buf[t + 1])
                nxt = nv if t == S - 1 else v_buf[t + 1]
                delta = r_buf[t] + gamma * nxt * nnt - v_buf[t]
                last = delta + gamma * lam * nnt * last
                adv[t] = last
            ret = adv + v_buf
        b_o, b_a, b_lp = B["o"].reshape(-1, 4, 84, 84), B["a"].reshape(-1), B["lp"].reshape(-1)
        b_adv, b_ret, b_v = adv.reshape(-1), ret.reshape(-1), v_buf.reshape(-1)
        if popart:                                   # the critic's lessons are in its own small numbers (PopArt)
            agent.rescale(b_ret, POPART_BETA)
            vmu, vsig = agent.v_mu, agent.v_sigma
        else:
            vmu = vsig = None
        n_ret = b_ret if vmu is None else (b_ret - vmu) / vsig
        n_v = b_v if vmu is None else (b_v - vmu) / vsig
        if explorer:                                 # moves that beat expectations on points OR on exploring
            b_adv = b_adv + explorer.beta * adv_i.reshape(-1)
            b_reti, b_vi = ret_i.reshape(-1), vi_buf.reshape(-1)
        ent = torch.zeros(())
        gr = _graph_for(b_o, b_a, b_lp, n_v, n_ret, vmu, vsig)     # the recorded lesson (None = the slow way)
        for _ in range(PASSES):
            # ALFRED 21 Oct 2026 (review): the shuffle is made on the card (PAA_GPUPERM=0 = the old way). Made on the
            # processor, each lesson's copy to the card made the processor WAIT for the card to finish the lesson
            # before it (the profile: 64 waits x 12 ms a batch) - the card then sat idle while the next was queued.
            perm = (torch.randperm(batch, device=dev) if os.environ.get("PAA_GPUPERM", "1") != "0"
                    else torch.randperm(batch).to(dev))
            for i in torch.tensor_split(perm, n_lessons):
                n_i = len(i)
                ma_all = (b_adv[i] - b_adv[i].mean()) / (b_adv[i].std() + 1e-8)
                if gr is not None:                       # the whole lesson replays as one recorded card job
                    _graph_throttle()
                    _graphs["i"].copy_(i)
                    _graphs["ma"].copy_(ma_all)
                    gr.replay()
                    ent = _graphs["ent"]
                else:
                    opt.zero_grad(set_to_none=True)
                for k in (range(0, n_i, bite) if gr is None else ()):   # one lesson, in bites the card can hold
                    j = slice(k, min(n_i, k + bite))
                    ii, ma = i[j], ma_all[j]
                    if explorer:
                        logits, nvv, nvi_ = agent(b_o[ii], both=True)
                    else:
                        logits, nvv = agent(b_o[ii])
                    dist = Categorical(logits=logits)
                    ratio = (dist.log_prob(b_a[ii]) - b_lp[ii]).exp()
                    pg = torch.max(-ma * ratio, -ma * ratio.clamp(1 - clip, 1 + clip)).mean()
                    nvn = nvv if vmu is None else (nvv - vmu) / vsig
                    vc = n_v[ii] + (nvn - n_v[ii]).clamp(-clip, clip)
                    vl = 0.5 * torch.max((nvn - n_ret[ii]) ** 2, (vc - n_ret[ii]) ** 2).mean()
                    ent = dist.entropy().mean()
                    if explorer:
                        vci = b_vi[ii] + (nvi_ - b_vi[ii]).clamp(-clip, clip)
                        vl = vl + 0.5 * torch.max((nvi_ - b_reti[ii]) ** 2, (vci - b_reti[ii]) ** 2).mean()
                    ((pg - ent_coef * ent + 0.5 * vl) * (len(ii) / n_i)).backward()
                extra = None
                if bank is not None:
                    w = lfm * max(0.0, 1 - (g["step"] - demo_step) / demos.FADE)
                    if w > 0:                        # keep a (fading) pull towards the player's own moves
                        xb, ab = bank.sample(min(256, bite))
                        extra = w * nn.functional.cross_entropy(agent(xb)[0], ab)
                if studier is not None:              # a little of its own best games, every lesson
                    got = studier.sample(min(256, bite))
                    if got is not None:
                        e2 = best_games.WEIGHT * nn.functional.cross_entropy(agent(got[0])[0], got[1])
                        extra = e2 if extra is None else extra + e2
                if extra is not None:
                    extra.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                opt.step()
                if explorer:
                    explorer.train_on(nxt_o[i])
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return {"ent": float(ent.detach()), "t_learn": time.time() - t_l, "novelty": novelty,
                "expl_var": float(1 - (b_ret - b_v).var() / (b_ret.var() + 1e-8))}

    t0, s0 = time.time(), step
    it, idx = it0, 0
    t_it = time.time()
    t_env, t_brain = collect(bufs[0], actor)
    while True:
        if cfg.get("anneal", True) and total < 1e9 and cfg.get("length") != "more5":   # experiments: same learning speed throughout
            opt.param_groups[0]["lr"] = lr * max(0.05, 1 - g["step"] / total)
        stop = (run / "STOP").exists()
        more = g["step"] < total and not stop
        B = bufs[idx]
        if overlap:
            res = {}

            def bg():
                try:
                    res.update(learn(B))
                except BaseException as e:   # surface learner crashes on the main thread
                    res["error"] = e
            th = threading.Thread(target=bg, daemon=True)
            th.start()
            nxt = (0.0, 0.0)
            if more:
                nxt = collect(bufs[1 - idx], actor)
            th.join()
            if "error" in res:
                raise res["error"]
            actor.load_state_dict(agent.state_dict())
            idx = 1 - idx
        else:
            res = learn(B)
            nxt = None
        t_tot = max(1e-6, time.time() - t_it)
        if overlap:
            split = (f"games {100 * nxt[0] / t_tot:.0f}% brain {100 * nxt[1] / t_tot:.0f}% | learning "
                     f"{res['t_learn']:.2f}s in the background")
        else:
            split = (f"games {100 * t_env / t_tot:.0f}% brain {100 * t_brain / t_tot:.0f}% "
                     f"learning {100 * res['t_learn'] / t_tot:.0f}%")
        step = g["step"]
        games = g["games"]
        sps = int((step - s0) / max(1e-6, time.time() - t0))
        m100 = float(np.mean(recent)) if recent else float("nan")
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "iter": it, "step": step, "sps": sps, "games": games,
               "score_mean100": round(m100, 1), "entropy": round(res["ent"], 4), "expl_var": round(res["expl_var"], 3),
               "best_game": g["best_game"], "views": len(views),
               "novelty": round(res["novelty"], 4)}
        # ALFRED 23 Oct 2026: the accuracy meter - shots that hit (invaders + saucers), its last 100 real games
        _sh = sum(s for s, _h in aim)
        row["accuracy"] = round(100.0 * sum(h for _s, h in aim) / _sh, 1) if _sh else None
        row["shots_per_game"] = round(_sh / len(aim), 1) if aim else None
        if go_explore:
            stx = envs.stats()
            row.update({"places": stx["places"], "map_best": stx["map_best"]})
        new = not (run / "train_log.csv").exists()
        if not new and it == it0:      # columns added since this log was started: rewrite it with the new header once
            try:
                with open(run / "train_log.csv", newline="") as f:
                    old_rows = list(csv.DictReader(f))
                if old_rows and list(old_rows[0].keys()) != list(row):
                    with open(run / "train_log.csv", "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(row), extrasaction="ignore")
                        w.writeheader()
                        w.writerows(old_rows)
            except Exception:
                pass
        with open(run / "train_log.csv", "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        extra = f" | explorer {res['novelty']:.2f}" if explorer else ""
        if row["accuracy"] is not None:
            extra += f" | aim {row['accuracy']:.1f}% of {row['shots_per_game']:.0f} shots"
        if practising:
            extra += f" | practice games {drills['n']:,}"
        if studier is not None:
            extra += f" | {studier.status()}"
        if checker is not None and checker.status():
            extra += f" | {checker.status()}"
        if go_explore:
            extra += f" | map {row['places']:,} places, best {row['map_best']:.0f}"
        say(f"step {step/1e6:7.2f}M | {sps:6d}/s | games {games:6d} | avg last 100: {m100:8.1f} | best game {g['best_game']:7.0f} | "
            f"views seen {len(views):,} | randomness {row['entropy']:.2f}{extra} | {split}")
        if len(recent) >= 20 and m100 > best:
            best = m100
            atomic_save({"agent": agent.state_dict(), "global_step": step, "score": m100, "n_actions": n_act,
                         "width": width, "game_id": cfg["game_id"]}, run / "best.pt")
        rebutton = checker.tick() if checker is not None and not stop else False
        done = stop or not more or rebutton
        if it % 25 == 0 or done:
            if checker is not None:
                checker.save()
            atomic_save({"agent": agent.state_dict(), "opt": opt.state_dict(), "global_step": step, "iteration": it,
                         "best": best, "n_actions": n_act, "width": width, "game_id": cfg["game_id"],
                         "explorer": explorer.state() if explorer else None,
                         "demo_frames": demo_frames, "demo_step": demo_step}, run / "latest.pt")
            try:
                np.save(run / "views.npy", np.fromiter(views, dtype=np.int64, count=len(views)))
                if go_explore:
                    envs.save()
            except Exception:
                pass
        beat["t"] = time.time()
        (run / "status.json").write_text(json.dumps({
            "state": "stopping" if stop else "training", "updated": row["time"], "step": step,
            "target_steps": int(total) if total < 1e9 else 0, "sps": sps, "games": games,
            "score_mean100": row["score_mean100"], "best_mean100": best, "pid": os.getpid(), "device": backend}))
        if done and rebutton:                      # its buttons were changed (Settings > Buttons): start again with them
            if hasattr(envs, "close"):
                envs.close()
            if studier is not None:
                studier.close()
            say("NEW BUTTONS for this game (Settings > Buttons) - starting again with them; what it has learned is "
                "carried over.")
            (run / "RESTART").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            (run / "status.json").write_text(json.dumps({"state": "restarting", "updated": row["time"], "step": step}))
            return
        if done:
            (run / "STOP").unlink(missing_ok=True)
            (run / "status.json").write_text(json.dumps({"state": "stopped", "updated": row["time"], "step": step,
                                                         "best_mean100": best, "games": games}))
            if hasattr(envs, "close"):
                envs.close()
            if studier is not None:
                studier.close()
            say("Stopped and saved. Press TRAIN again any time to carry on where it left off." if stop
                else "Reached the training target. Saved.")
            return
        if lfm > 0 and demo_files() not in (demo_sig, None):   # new Learn-from-me games: learn them now
            absorb(g["step"], False)
            if overlap:
                actor.load_state_dict(agent.state_dict())
        it += 1
        t_it = time.time()
        if not overlap:
            t_env, t_brain = collect(bufs[0], actor)


FROZEN_AFTER = 600


def _start_watchdog(run, beat, say):
    import threading

    def watch():
        while True:
            time.sleep(30)
            if time.time() - beat["t"] > FROZEN_AFTER and not (run / "STOP").exists():
                try:
                    say(f"FROZE: no progress for {FROZEN_AFTER // 60} minutes - restarting from the last save.")
                    (run / "FROZE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
                    (run / "status.json").write_text(json.dumps({"state": "restarting",
                                                                 "updated": time.strftime("%Y-%m-%d %H:%M:%S")}))
                except Exception:
                    pass
                # end this process AND its game workers (they'd otherwise hang on)
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(os.getpid())], capture_output=True) \
                    if os.name == "nt" else None
                os._exit(75)
    threading.Thread(target=watch, daemon=True).start()


def crashed(exc_text):
    """Never die silently: put the error where the panel and the report card can see it, and keep the window open."""
    print("\n" + exc_text, flush=True)
    try:
        import diagnose
        diagnose.log_error(f"trainer {Path(sys.argv[1]).name}", exc_text)
    except Exception:
        pass
    try:
        run = Path(sys.argv[1]).resolve() / "files" / "run"
        last = exc_text.strip().splitlines()[-1]
        hint = ""
        if "no kernel image" in exc_text or "not compatible with the current PyTorch" in exc_text:
            hint = " Your graphics card needs a newer PyTorch: close everything and double-click START HERE.bat - it repairs itself."
        elif "out of memory" in exc_text.lower():
            hint = " Not enough memory: close other programs (web browsers especially) and press Train again."
        elif "WinError 1450" in exc_text or "MemoryError" in exc_text or "paging file" in exc_text:
            hint = " Windows ran out of resources: close other programs and press Train again."
        if not hint:
            hint = " If it keeps happening, press 'Make a problem report' on the page and send the file to Jonny."
        with open(run / "console.log", "a", encoding="utf-8") as f:
            f.write(f"CRASHED: {last}.{hint}\n")
        (run / "status.json").write_text(json.dumps({"state": "crashed", "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                     "error": last + "." + hint}))
    except Exception:
        pass
    try:
        input("\nTraining stopped because of the error above. Press Enter to close this window.")
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except BaseException:
        import traceback
        crashed(traceback.format_exc())
        sys.exit(1)
