"""Arcade Apprentice - which buttons does a game really use? And carrying a brain over when they change.

THE BUTTON TEST
A game found with "Find the score" starts out with every joystick direction and button you touched while showing
the kit the game. One stray tap of Up in a game where Up does nothing adds a whole direction - three times the
moves - and the AI then has to learn by trial and error that 'left', 'left+up' and 'left+down' are the same move,
with every lesson about 'left' split between them. So the kit tests each direction and button itself: from lots of
moments of real play it presses it briefly and compares the screen with doing nothing from the same moment.
Anything that never changes anything is dropped. (The Atari games get this for free: each one only gets the buttons
it really uses.)

CARRYING A BRAIN OVER
When a game's buttons change, each AI trained on it is converted the first time it trains, is watched or teaches:
its "which button" layer is rebuilt for the new buttons - each new move starts from what the brain knew about the
old moves that do the same thing, then is fine-tuned on real screens so it makes the same choices as before - and
its recorded games (yours, its best games, the Hall of Fame) are renumbered to match. A copy of the old files is
kept in the AI's files folder ("backup ... before the button change").

    python controls.py <AI folder name, or game pack id>        test that game's buttons now, and slim them
                                                                (a by-hand tool; the kit itself only SUGGESTS -
                                                                the user decides, in Settings > Buttons)
    python controls.py <...> --dry                               test only, change nothing

Signed: Claude, 17 Oct 2026.
"""
from __future__ import annotations

import bisect
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

FILES = Path(__file__).resolve().parent
if str(FILES) not in sys.path:
    sys.path.insert(0, str(FILES))
KIT = FILES.parent
PROJECTS = KIT / "My AIs"
ROMS = KIT / "ROMs - put your games here"
FOUND = FILES / "arcade" / "found"
UP, DOWN, LEFT, RIGHT = 4, 5, 6, 7                   # RetroPad ids, as in generic_env.py
NAMES = {LEFT: "left", RIGHT: "right", UP: "up", DOWN: "down",          # the keys you play with in Find the score / Watch
         0: "button 1 (Z)", 8: "button 2 (X)", 1: "button 3 (C)", 9: "button 4 (A)", 10: "button 5 (S)", 11: "button 6 (D)"}
WINDOW, HOLD = 90, 24      # watch 1.5 s of game; the input is held for the first 0.4 s of it
MIN_TRIES = 8              # an input is only dropped after this many moments where pressing it changed nothing


# ------------------------------------------------------------------------------------------------ controls
def _generic():
    g = str(FILES / "arcade" / "generic")
    if g not in sys.path:
        sys.path.insert(0, g)
    import generic_env
    return generic_env


def controls_of(pack):
    axes = pack.get("axes", ["lr", "ud"])
    return {"axes": [a for a in ("lr", "ud") if a in axes], "buttons": [int(b) for b in pack.get("buttons", [0])]}


def moves(ctrl):
    """The game's move list for these controls, as sets of RetroPad ids (in the order the game numbers them)."""
    return [frozenset(t) for t in _generic().action_table(ctrl)[0]]


def describe(ctrl):
    parts = (["left/right"] if "lr" in ctrl["axes"] else []) + (["up/down"] if "ud" in ctrl["axes"] else [])
    parts += [NAMES.get(b, f"button {b}") for b in ctrl["buttons"]]
    return ", ".join(parts) if parts else "no buttons at all"


def inputs_of(ctrl):
    return (([LEFT, RIGHT] if "lr" in ctrl["axes"] else []) + ([UP, DOWN] if "ud" in ctrl["axes"] else [])
            + [int(b) for b in ctrl["buttons"]])


def pack_dir_for(game_id):
    if str(game_id).startswith("found-"):          # quick: found packs live in found\<set name>
        d = FOUND / str(game_id)[len("found-"):]
        try:
            if json.loads((d / "pack.json").read_text(encoding="utf-8")).get("id") == game_id:
                return d
        except Exception:
            pass
    for p in sorted(FOUND.glob("*/pack.json")) if FOUND.exists() else []:
        try:
            if json.loads(p.read_text(encoding="utf-8")).get("id") == game_id:
                return p.parent
        except Exception:
            pass
    return None


def plan(old, new):
    """For each new move, the old moves that do the same thing (members); and old move number -> new (old2new)."""
    to, tn = moves(old), moves(new)
    keep = set().union(*tn)
    where = {s: k for k, s in enumerate(tn)}
    members, old2new = [[] for _ in tn], []
    for a, s in enumerate(to):
        g = where.get(frozenset(s & keep), where.get(frozenset(), 0))
        old2new.append(g)
        members[g].append(a)
    return members, old2new


def record_change(pack, new, why):
    """Put new controls into a pack (a dict), keeping the old ones on record so trained AIs can be carried over."""
    old = controls_of(pack)
    if moves(old) == moves(new):
        return False
    pack.setdefault("controls_history", []).append(
        {**old, "n_actions": len(moves(old)), "until": time.strftime("%Y-%m-%d %H:%M"), "why": why})
    pack["axes"], pack["buttons"] = list(new["axes"]), [int(b) for b in new["buttons"]]
    pack["n_actions"] = len(moves(new))
    return True


def write_pack(pack_dir, pack):
    p = Path(pack_dir) / "pack.json"
    tmp = p.with_name("pack.json.tmp")
    tmp.write_text(json.dumps(pack, indent=1), encoding="utf-8")
    os.replace(tmp, p)


# ------------------------------------------------------------------------------------------------ the button test
def _run(host, state, inputs):
    host.load_state(state)
    host.frame = None                       # a repeated ('duplicate') frame at the start mustn't carry the last run over
    seen = []
    for k in range(WINDOW):
        f = host.run(inputs if k < HOLD else (), capture=True)
        seen.append(None if f is None else hash(f.tobytes()))
    return seen


def test_inputs(host, moments, inputs, tick=None):
    """Press each input (a RetroPad id) from each moment (a save-state) and compare the screen with doing nothing.
    Returns (matters {input: bool}, tried {input: moments tried}, number of usable moments)."""
    matters = {c: False for c in inputs}
    tried = {c: 0 for c in inputs}
    reliable = 0
    frame = getattr(host, "frame", None)
    try:
        back = host.save_state()
    except Exception:
        back = None
    try:
        for st in moments:
            if tick:
                tick()
            try:
                base = _run(host, st, ())
                if _run(host, st, ()) != base:
                    continue                # this moment doesn't play out the same way twice: no use for a test
            except Exception:
                continue
            reliable += 1
            for c in inputs:
                if matters[c]:
                    continue
                tried[c] += 1
                try:
                    if _run(host, st, (c,)) != base:
                        matters[c] = True
                except Exception:
                    matters[c] = True       # couldn't tell: keep it
            if all(matters.values()):
                break
    finally:
        if back is not None:
            try:
                host.load_state(back)
            except Exception:
                pass
        host.frame = frame
        try:
            host.drain_audio()
        except Exception:
            pass
    return matters, tried, reliable


def decide(ctrl, matters, tried, reliable, protect=(), min_tries=MIN_TRIES):
    """The slimmed controls, or None if the test couldn't tell."""
    if reliable < 3:
        return None

    def dead(c):
        return c not in protect and not matters.get(c, True) and tried.get(c, 0) >= min_tries

    axes = [ax for ax, pair in (("lr", (LEFT, RIGHT)), ("ud", (UP, DOWN)))
            if ax in ctrl["axes"] and not all(dead(c) for c in pair)]
    buttons = [int(b) for b in ctrl["buttons"] if not dead(int(b))]
    if not axes and not buttons:
        return None                         # nothing does anything? more likely the test went wrong
    return {"axes": axes, "buttons": buttons}


def play_moments(env, want=32, every=150, limit=8000, seed=12345):
    """Save-states from random play: each game's start, moments just after each lost life (a serve or a restart -
    where buttons like 'launch the ball' matter), and every few seconds in between."""
    rng = np.random.default_rng(seed)
    n_act = env.action_space.n
    out = []
    obs, info = env.reset(seed=int(rng.integers(1 << 30)))
    out.append(env.host.save_state())
    lives, since = int(info.get("lives", 0) or 0), None
    for t in range(limit):
        if len(out) >= want:
            break
        obs, _r, term, trunc, info = env.step(int(rng.integers(n_act)))
        if term or trunc:
            obs, info = env.reset(seed=int(rng.integers(1 << 30)))
            out.append(env.host.save_state())
            lives, since = int(info.get("lives", 0) or 0), None
            continue
        now = int(info.get("lives", lives) or 0)
        if now < lives:
            since = 0
        lives = now
        if since is not None:
            since += 1
            if since in (15, 45, 90):
                out.append(env.host.save_state())
            if since > 90:
                since = None
        if t % every == every - 1:
            out.append(env.host.save_state())
    return out


def test_finder(finder, s, e, ctrl, tick=None):
    """The button test inside the Find-the-score window, from the moments it saved while you played: a spread over
    the game, plus the moments just before each time you pressed something. An input you used, but never where it
    could be tested, is kept. Returns (controls or None, matters, tried, usable moments)."""
    keys = sorted(finder.states)
    if not keys:
        return None, {}, {}, 0
    every = int(getattr(finder, "state_every", 10))
    inputs = inputs_of(ctrl)
    used = [(f, b) for f, b in finder.used if s <= f <= e]
    picks = set(keys[:: max(1, len(keys) // 20)]) | {keys[0]}
    protect = set()
    for c in inputs:
        presses = sorted({f for f, b in used if b == c})
        near = set()
        for f in presses:
            i = bisect.bisect_right(keys, f) - 1
            if i >= 0 and f - keys[i] <= every:
                near.add(keys[i])
        if presses and not near:
            protect.add(c)
        near = sorted(near)
        picks.update(near[:: max(1, len(near) // 6)][:6])
    moments = [finder.states[k] for k in sorted(picks)]
    matters, tried, reliable = test_inputs(finder.host, moments, inputs, tick=tick)
    return decide(ctrl, matters, tried, reliable, protect=protect, min_tries=4), matters, tried, reliable


def find_rom(game_id):
    for pj in sorted(PROJECTS.glob("*/project.json")) if PROJECTS.exists() else []:
        try:
            c = json.loads(pj.read_text())
        except Exception:
            continue
        if c.get("game_id") == game_id and c.get("rom_file") and (ROMS / c["rom_file"]).exists():
            return ROMS / c["rom_file"]
    pdir = pack_dir_for(game_id)
    if pdir:
        name = json.loads((pdir / "pack.json").read_text(encoding="utf-8")).get("rom_file")
        if name:
            for p in ROMS.rglob(name):
                return p
    return None


def check(target, dry=False, say=print):
    """Test a found game's buttons now (from random play) and slim them. target: an AI folder name or a game pack id."""
    ai = PROJECTS / target
    game_id = json.loads((ai / "project.json").read_text()).get("game_id") if (ai / "project.json").exists() else target
    pdir = pack_dir_for(game_id)
    if pdir is None:
        say(f"No found game pack '{game_id}' - only games found with 'Find the score' can be tested.")
        return None
    pack = json.loads((pdir / "pack.json").read_text(encoding="utf-8"))
    if pack.get("reset") == "reboot":
        say("This game's save-states don't restore, so its buttons can't be tested.")
        return None
    rom = find_rom(game_id)
    if rom is None:
        say("Couldn't find the game's ROM in the ROMs folder.")
        return None
    from arcade_vec import make_env, scratch_dir, local_core
    rt = scratch_dir("controls")
    local_core(rt, pack.get("core", "fbneo"))
    env = make_env(game_id, rom, rt)
    old = controls_of(pack)
    ins = inputs_of(old)
    say(f"{pack.get('name', game_id)}: testing {describe(old)} ({len(moves(old))} moves)...")
    t0 = time.time()
    try:
        moments = play_moments(env)
        matters, tried, reliable = test_inputs(env.host, moments, ins)
    finally:
        try:
            env.close()
        except Exception:
            pass
    for c in ins:
        say(f"  {NAMES.get(c, c):>5}: {'changes the game' if matters[c] else 'did nothing'} (tried from {tried[c]} moments)")
    say(f"  {reliable} of {len(moments)} moments usable, {time.time() - t0:.0f} s.")
    new = decide(old, matters, tried, reliable)
    if new is None:
        say("Couldn't tell for sure - keeping the buttons as they are.")
        return None
    if moves(new) == moves(old):
        say("Every button it has does something - nothing to trim.")
        return old
    say(f"Slimmed: {describe(old)} ({len(moves(old))} moves) -> {describe(new)} ({len(moves(new))} moves).")
    if not dry:
        record_change(pack, new, "button test")
        write_pack(pdir, pack)
        say("Saved. Each AI on this game is carried over to the new buttons the next time it trains, is watched "
            "or teaches.")
    return new


# ------------------------------------------------------------------------------------------------ carrying a brain over
def _brain_files(proj):
    run = proj / "files" / "run"
    files = [run / n for n in ("latest.pt", "best.pt") if (run / n).exists()]
    for d in (run / "milestones", proj / "files" / "checkpoints"):
        if d.exists():
            files += sorted(d.rglob("*.pt"))
    return files


def _game_files(proj):
    out = []
    dd = proj / "files" / "demos"
    if dd.exists():
        out += [("demo", p) for p in sorted(dd.glob("demo-*.npz"))]
    for sub in ("hall_of_fame", "studying"):
        d = proj / "files" / "best_games" / sub
        if d.exists():
            out += [("game", p) for p in sorted(d.glob("*.npz")) if not p.name.endswith(".part.npz")]
    return out


def _n_of(kind, p):
    """How many moves this file's brain or recording was made with (-1 = can't tell)."""
    try:
        if kind == "brain":
            import torch
            try:
                st = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
            except Exception:
                st = torch.load(p, map_location="cpu", weights_only=False)
            return int(st.get("n_actions", -1))
        with np.load(p) as z:
            if kind == "demo":
                return int(z["n_actions"])
            return int(json.loads(str(z["meta"])).get("n_actions", -1))
    except Exception:
        return -1


def ensure_fits(proj, say=print, dev=None):
    """Before an AI trains, is watched or teaches: if its game's buttons have changed since its brain last played,
    carry the brain and its recorded games over to the new buttons. Quick when nothing has changed. Returns True if
    anything was converted."""
    proj = Path(proj)
    try:
        cfg = json.loads((proj / "project.json").read_text())
    except Exception:
        return False
    if cfg.get("platform") != "arcade":
        return False
    pdir = pack_dir_for(cfg.get("game_id", ""))
    if pdir is None:
        return False
    pack = json.loads((pdir / "pack.json").read_text(encoding="utf-8"))
    new = controls_of(pack)
    n_new = len(moves(new))

    def stale():
        return [(k, p) for k, p in [("brain", p) for p in _brain_files(proj)] + _game_files(proj)
                if _n_of(k, p) not in (n_new, -1)]

    if not stale():
        return False
    run = proj / "files" / "run"
    run.mkdir(parents=True, exist_ok=True)
    lock, t0 = run / "MIGRATING", time.time()
    while True:                             # one window at a time (training and Watch can start together)
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > 900:
                    lock.unlink(missing_ok=True)
                    continue
            except OSError:
                continue
            if time.time() - t0 > 900:
                return False
            time.sleep(1)
    try:
        todo = stale()                      # another window may have done it while we waited
        if not todo:
            return False
        if say is print:                    # Watch / teacher windows: put it in the training diary too
            def say(msg, _p=print):
                _p(msg)
                try:
                    with open(run / "console.log", "a", encoding="utf-8") as f:
                        f.write(msg + "\n")
                except Exception:
                    pass
        _migrate(proj, pack, new, todo, say, dev)
        return True
    finally:
        lock.unlink(missing_ok=True)


def _migrate(proj, pack, new, todo, say, dev):
    hist = {}
    for h in pack.get("controls_history", []):
        c = controls_of(h)
        hist[int(h.get("n_actions", len(moves(c))))] = c          # the newest record of each size wins
    n_new = len(moves(new))
    backup = proj / "files" / f"backup {time.strftime('%Y-%m-%d %H%M')} - before the button change"
    big = dev is not None and str(dev) != "cpu"
    screens = _screens(proj, 6000 if big else 2000) if any(k == "brain" for k, _ in todo) else None
    done, skipped, agree = [], [], []
    for kind, p in todo:
        n_old = _n_of(kind, p)
        old = hist.get(n_old)
        if old is None:
            skipped.append(p)
            continue
        members, old2new = plan(old, new)
        dst = backup / p.relative_to(proj)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        if kind == "brain":
            a = _convert_brain(p, members, n_new, new, screens, dev)
            if a is not None and p.name == "latest.pt" and p.parent.name == "run":
                agree.append(a)
        elif kind == "demo":
            _remap_demo(p, old2new, n_new)
        else:
            _remap_game(p, old2new, n_new)
        done.append(p)
    if done:
        msg = (f"NEW BUTTONS: this game now uses {describe(new)} ({n_new} moves). Carried {len(done)} file"
               f"{'' if len(done) == 1 else 's'} over (brain, your recordings, best games)")
        if agree:
            msg += f"; on {len(screens):,} real screens it makes the same choice as before {100 * agree[0]:.0f}% of the time"
        say(msg + f". The old files are kept in '{backup.name}'.")
    if skipped:
        say(f"{len(skipped)} file(s) were made with buttons the kit has no record of, so they were left as they are.")


def _screens(proj, max_n=6000):
    """Up to max_n real screens this AI has seen (your recordings, its best games), as 4-screen stacks - or None."""
    srcs = []
    dd = proj / "files" / "demos"
    if dd.exists():
        srcs += sorted(dd.glob("demo-*.npz"))
    for sub in ("studying", "hall_of_fame"):
        d = proj / "files" / "best_games" / sub
        if d.exists():
            srcs += sorted((p for p in d.glob("*.npz") if not p.name.endswith(".part.npz")),
                           key=lambda p: p.stat().st_mtime, reverse=True)
    rng = np.random.default_rng(0)
    out, used, most = [], 0, 8
    for p in srcs:
        if len(out) >= max_n or used >= most:
            break
        try:
            with np.load(p) as z:
                f = z["frames"]
                s = z["starts"].copy() if "starts" in z.files else np.zeros(len(f), bool)
        except Exception:
            continue
        if not len(f):
            continue
        used += 1
        s[0] = True
        quota = max(1, min(len(f), (max_n - len(out)) // max(1, most - used + 1)))
        for i in rng.choice(len(f), size=quota, replace=False):
            first = i
            while first > 0 and not s[first] and i - first < 3:
                first -= 1
            stack = np.zeros((4, 84, 84), np.uint8)
            for k in range(4):
                if i - 3 + k >= first:
                    stack[k] = f[i - 3 + k]
            out.append(stack)
    return np.stack(out) if out else None


def _features(agent, screens, dev):
    """The brain's 256-number summary of each screen (what its 'which button' layer works from)."""
    import torch
    dev = dev if dev is not None else "cpu"
    agent = agent.to(dev).eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(screens), 512):
            x = torch.from_numpy(screens[k:k + 512]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
                h = agent.body(x.float() / 255.0)
            out.append(h.float().cpu())
    return torch.cat(out)


def new_head(W, b, members, feats=None, steps=500):
    """Rebuild the 'which button' layer for the new moves. Each new move starts as the average of the old moves that
    do the same thing, plus log(how many) so the group keeps its share; then, given real screens (as the brain's
    256-number summaries), it is fine-tuned so each new move gets the same chance its old group had on every screen.
    Returns (weights, biases, share of screens where the favourite move is unchanged - or None)."""
    import torch
    W, b = W.float(), b.float()
    n_new = len(members)
    W2 = torch.zeros(n_new, W.shape[1])
    b2 = torch.full((n_new,), float(b.mean()) - 2.0)          # a brand-new move: rarely tried at first
    for g, m in enumerate(members):
        if m:
            W2[g] = W[m].mean(0)
            b2[g] = b[m].mean() + math.log(len(m))
    if feats is None or not len(feats):
        return W2, b2, None
    with torch.no_grad():
        p_old = torch.softmax(feats @ W.T + b, -1)
        target = torch.zeros(len(feats), n_new)
        for g, m in enumerate(members):
            if m:
                target[:, g] = p_old[:, m].sum(1)
        target = 0.995 * target / target.sum(1, keepdim=True).clamp(min=1e-8) + 0.005 / n_new
    W2.requires_grad_(True)
    b2.requires_grad_(True)
    opt = torch.optim.Adam([W2, b2], lr=0.02)
    for _ in range(steps):
        idx = torch.randint(0, len(feats), (min(4096, len(feats)),))
        loss = -(target[idx] * torch.log_softmax(feats[idx] @ W2.T + b2, -1)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    W2, b2 = W2.detach(), b2.detach()
    with torch.no_grad():
        same = float(((feats @ W2.T + b2).argmax(1) == target.argmax(1)).float().mean())
    return W2, b2, same


def _group_mean(t, members):
    import torch
    return torch.stack([t[m].mean(0) if m else torch.zeros_like(t[0]) for m in members])


def _replace(tmp, p):
    for i in range(10):
        try:
            os.replace(tmp, p)
            return
        except OSError:
            time.sleep(1 + i)
    os.replace(tmp, p)


def _convert_brain(p, members, n_new, new, screens, dev):
    import torch
    from trainer import Agent
    st = torch.load(p, map_location="cpu", weights_only=False)
    sd = st["agent"]
    W, b = sd["actor.weight"], sd["actor.bias"]
    n_old, width = int(W.shape[0]), int(st.get("width", 1))
    feats = None
    if screens is not None:
        try:
            ag = Agent(n_old, width)
            ag.load_state_dict(sd, strict=False)
            feats = _features(ag, screens, dev)
        except Exception:
            feats = None
    W2, b2, same = new_head(W, b, members, feats)
    sd["actor.weight"], sd["actor.bias"] = W2.to(W.dtype).contiguous(), b2.to(b.dtype).contiguous()
    if st.get("opt"):
        try:
            names = [n for n, _ in Agent(n_old, width).named_parameters()]
            ids = [i for g in st["opt"]["param_groups"] for i in g["params"]]
            if len(ids) != len(names):
                raise ValueError("the learning memory doesn't line up with the brain")
            for pid, name in zip(ids, names):
                if name in ("actor.weight", "actor.bias") and pid in st["opt"]["state"]:
                    s = st["opt"]["state"][pid]
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in s:
                            s[key] = _group_mean(s[key], members)
        except Exception:
            st["opt"] = None                # the trainer starts its learning memory afresh instead
    st["n_actions"] = n_new
    st["controls"] = {"axes": list(new["axes"]), "buttons": list(new["buttons"])}
    tmp = p.with_name(p.name + ".tmp")
    torch.save(st, tmp)
    _replace(tmp, p)
    return same


def _save_npz(p, data):
    tmp = p.with_name("~" + p.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **data)
    _replace(tmp, p)


def _remap_demo(p, old2new, n_new):
    with np.load(p) as z:
        data = {k: z[k] for k in z.files}
    m = np.asarray(old2new, np.int64)
    a = data["actions"].astype(np.int64)
    data["actions"] = np.where(a >= 0, m[np.clip(a, 0, len(m) - 1)], a).astype(np.int16)
    data["n_actions"] = np.int32(n_new)
    _save_npz(p, data)


def _remap_game(p, old2new, n_new):
    with np.load(p) as z:
        data = {k: z[k] for k in z.files}
    m = np.asarray(old2new, np.int64)
    for k in ("chosen", "acted"):
        if k in data:
            data[k] = m[np.clip(data[k].astype(np.int64), 0, len(m) - 1)].astype(np.int16)
    meta = json.loads(str(data["meta"]))
    meta["n_actions"] = n_new
    data["meta"] = json.dumps(meta)
    _save_npz(p, data)


# ------------------------------------------------------------------------------------------------ the check during training
LIVE_MIN = int(os.environ.get("PAA_PROBE_MIN") or 500)   # clean tries in its own play before a do-nothing button is suggested


def _names(cs):
    n = [NAMES.get(c, f"button {c}") for c in cs]
    return n[0] if len(n) == 1 else ", ".join(n[:-1]) + " and " + n[-1]


def _without(ctrl, drop):
    """These controls without some inputs (a direction pair only goes when both of its directions do)."""
    drop = set(drop)
    axes = [ax for ax, pair in (("lr", (LEFT, RIGHT)), ("ud", (UP, DOWN))) if ax in ctrl["axes"] and not set(pair) <= drop]
    buttons = [int(b) for b in ctrl["buttons"] if int(b) not in drop]
    return {"axes": axes, "buttons": buttons} if (axes or buttons) else None


def _suggest(ctrl, stats, limit=LIVE_MIN):
    """From the evidence: the inputs that never changed anything in at least `limit` clean tries - once every unproven
    input has had its tries - and the controls without them. None if there's nothing to suggest (yet)."""
    sus = [c for c in inputs_of(ctrl) if stats.get(c, {}).get("changed", 0) == 0]
    if not sus or min(stats.get(c, {}).get("tried", 0) for c in sus) < limit:
        return None
    matters = {c: stats.get(c, {}).get("changed", 0) > 0 for c in inputs_of(ctrl)}
    tried = {c: stats.get(c, {}).get("tried", 0) for c in inputs_of(ctrl)}
    new = decide(ctrl, matters, tried, 999, min_tries=limit)
    if new is None or moves(new) == moves(ctrl):
        return None
    drop = [c for c in inputs_of(ctrl) if c not in inputs_of(new)]
    return {"drop": drop, "new": new, "tries": min(tried[c] for c in drop),
            "moves_before": len(moves(ctrl)), "moves_after": len(moves(new))}


class LiveCheck:
    """THE BUTTON CHECK DURING TRAINING. About once in 1,000 moves a cabinet tries one input on top of the move the AI
    is making, from wherever it has got to in the game, and compares the next three quarters of a second of screen
    with the same move without it (arcade_vec.Cabinet._probe; tries that don't replay cleanly don't count). Its own
    play reaches places random play never does - the ladder where Up matters, the serve where Fire matters. An input
    that changes anything even once is proven. One that never has after LIVE_MIN clean tries is SUGGESTED for dropping
    and never dropped automatically - Jonny's rule, 17 Oct 2026: a button that only matters later (a potion you have to
    pick up first) looks exactly the same, and dropping it would stop the AI ever discovering it. The user decides, in
    the AI's Settings > Buttons. It keeps checking meanwhile, so a suggestion firms up - or a button proves itself - the
    longer it plays. Evidence: <AI>\\files\\run\\button_checks.json. The trainer also asks tick() once a batch whether
    the game's buttons have been changed; if so it starts again with them (the brain is carried over)."""

    def __init__(self, proj, cfg, say=print):
        self.proj, self.say = Path(proj), say
        self.file = self.proj / "files" / "run" / "button_checks.json"
        self.limit = LIVE_MIN
        self.stats, self.ctrl, self.on, self.announced, self.mtime = {}, None, False, None, None
        self.pdir = pack_dir_for(cfg.get("game_id", "")) if cfg.get("platform") == "arcade" else None
        if self.pdir is None:
            return
        pack = json.loads((self.pdir / "pack.json").read_text(encoding="utf-8"))
        self.ctrl, self.mtime = controls_of(pack), (self.pdir / "pack.json").stat().st_mtime
        try:
            saved = json.loads(self.file.read_text())
            if saved.get("controls") == self.ctrl:
                self.stats = {int(k): dict(v) for k, v in saved.get("stats", {}).items()}
                self.announced = saved.get("announced")
        except Exception:
            pass
        self.stats = {c: self.stats.get(c, {"tried": 0, "changed": 0}) for c in inputs_of(self.ctrl)}
        self.on = bool(cfg.get("check_buttons", True)) and pack.get("reset") != "reboot" and not pack.get("keep_buttons")
        sus = self.inputs()
        if sus:
            say(f"BUTTON CHECK: now and then it tries {_names(sus)} in the middle of play to see what "
                f"{'it does' if len(sus) == 1 else 'they do'}. Any that never change anything are suggested for "
                "dropping - you decide, in Settings > Buttons.")

    def inputs(self):
        """The inputs still unproven (anything that has changed the game once is proven for good)."""
        return [c for c, s in self.stats.items() if s["changed"] == 0] if self.on else []

    def add(self, probes):
        for c, changed in probes:
            s = self.stats.get(int(c))
            if s is None:
                continue
            if changed is None:                     # didn't replay cleanly: doesn't count
                s["unclean"] = s.get("unclean", 0) + 1
            else:
                s["tried"] += 1
                s["changed"] += int(bool(changed))

    def save(self):
        if self.pdir is None or self.ctrl is None:
            return
        try:
            tmp = self.file.with_name(self.file.name + ".tmp")
            tmp.write_text(json.dumps({"controls": self.ctrl, "announced": self.announced,
                                       "stats": {str(c): s for c, s in self.stats.items()}}, indent=1))
            os.replace(tmp, self.file)
        except Exception:
            pass

    def status(self):
        sus = self.inputs()
        if not sus:
            return ""
        least = min(self.stats[c]["tried"] for c in sus)
        if least < self.limit:
            return f"button check: {len(sus)} unproven, {least}/{self.limit} tries"
        return f"button check: {_names(sus)} did nothing in {least:,}+ tries - see Settings > Buttons"

    def tick(self):
        """Once a batch: mention a new suggestion once (in the training diary), and say whether the game's buttons
        have been changed since training started (then the trainer starts again with them)."""
        if self.on:
            sug = _suggest(self.ctrl, self.stats, self.limit)
            key = sorted(sug["drop"]) if sug else None
            if key and key != self.announced:
                self.announced = key
                one = len(key) == 1
                self.say(f"BUTTON CHECK: {_names(sug['drop'])} never changed anything in {sug['tries']:,} clean tries "
                         f"{'' if one else 'each '}while it played. Dropping {'it' if one else 'them'} would leave "
                         f"{sug['moves_after']} moves instead of {sug['moves_before']}, so it learns faster - but only if "
                         f"{'it does' if one else 'they do'} nothing anywhere in the game. Your call: Settings > Buttons.")
                self.save()
        return self.pack_changed()

    def pack_changed(self):
        if self.pdir is None:
            return False
        try:
            m = (self.pdir / "pack.json").stat().st_mtime
            if m == self.mtime:
                return False
            self.mtime = m
            pack = json.loads((self.pdir / "pack.json").read_text(encoding="utf-8"))
            new = controls_of(pack)
            if moves(new) == moves(self.ctrl):
                return False
            self.ctrl = new                          # what's known about the inputs still there stays known
            self.stats = {c: s for c, s in self.stats.items() if c in inputs_of(new)}
            self.announced = None
            return True
        except Exception:
            return False


# ------------------------------------------------------------------------------------------------ Settings > Buttons
def _ai(proj):
    proj = Path(proj)
    cfg = json.loads((proj / "project.json").read_text())
    pdir = pack_dir_for(cfg.get("game_id", "")) if cfg.get("platform") == "arcade" else None
    return proj, cfg, pdir


def _evidence(proj, ctrl):
    try:
        saved = json.loads((Path(proj) / "files" / "run" / "button_checks.json").read_text())
        if saved.get("controls") == ctrl:
            return {int(k): v for k, v in saved.get("stats", {}).items()}
    except Exception:
        pass
    return {}


def _keep_evidence(proj, ctrl, stats):
    """After a change, what's known about the inputs that are still there stays known."""
    f = Path(proj) / "files" / "run" / "button_checks.json"
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        keep = {str(c): stats[c] for c in inputs_of(ctrl) if c in stats}
        f.write_text(json.dumps({"controls": ctrl, "announced": None, "stats": keep}, indent=1))
    except Exception:
        pass


def _give_back_to(pack):
    """The controls from before the most recent drop (more moves than now), or None."""
    now = len(moves(controls_of(pack)))
    for h in reversed(pack.get("controls_history", [])):
        c = controls_of(h)
        if len(moves(c)) > now:
            return c
    return None


def _game_of(pj):
    try:
        return json.loads(Path(pj).read_text()).get("game_id")
    except Exception:
        return None


def button_report(proj, light=False):
    """What the page's Settings > Buttons shows: each input and what it did, any suggestion, and what can be undone."""
    proj, cfg, pdir = _ai(proj)
    if pdir is None:
        return {"found": False, "msg": "This game's buttons were set up by hand (or it's an Atari game), so it only "
                                       "has the buttons the game really uses - nothing to check."}
    pack = json.loads((pdir / "pack.json").read_text(encoding="utf-8"))
    ctrl = controls_of(pack)
    stats = _evidence(proj, ctrl)
    rows = []
    for c in inputs_of(ctrl):
        s = stats.get(c, {})
        t, ch, un = s.get("tried", 0), s.get("changed", 0), s.get("unclean", 0)
        if ch:
            v, kind = f"does something ({ch:,} of {t:,} tries changed the game)", "useful"
        elif t >= LIVE_MIN:
            v, kind = f"never changed anything in {t:,} clean tries", "dead"
        elif t or un:
            v, kind = (f"still checking - {t:,} of {LIVE_MIN} clean tries so far" +
                       (f" ({un:,} more didn't replay cleanly, so they don't count)" if un else "")), "testing"
        else:
            v, kind = "not checked yet - it checks while it trains", "untested"
        rows.append({"name": NAMES.get(c, f"button {c}"), "verdict": v, "kind": kind})
    out = {"found": True, "controls": describe(ctrl), "moves": len(moves(ctrl)), "inputs": rows,
           "keep": bool(pack.get("keep_buttons")), "reboot": pack.get("reset") == "reboot", "suggest": None,
           "restore": None, "others": [] if light else sorted(p.parent.name for p in PROJECTS.glob("*/project.json")
                                                               if p.parent != proj and _game_of(p) == cfg.get("game_id"))}
    sug = _suggest(ctrl, stats) if not pack.get("keep_buttons") else None
    if sug:
        out["suggest"] = {"names": _names(sug["drop"]), "count": len(sug["drop"]), "moves_after": sug["moves_after"],
                          "where": f"in {sug['tries']:,}+ clean tries during its own play", "source": "training"}
        out["suggest"].update(tries=sug["tries"], drop=[NAMES.get(c, str(c)) for c in sug["drop"]])
    elif not pack.get("keep_buttons"):
        drop = [c for c in (pack.get("finder_suggests") or {}).get("drop", []) if c in inputs_of(ctrl)]
        new = _without(ctrl, drop) if drop else None
        if new and moves(new) != moves(ctrl):
            out["suggest"] = {"names": _names(drop), "count": len(drop), "moves_after": len(moves(new)),
                              "where": "while you played it in Find the score - only a few minutes of play, so training "
                                       "checks it more thoroughly", "source": "finder"}
            out["suggest"].update(tries=0, drop=[NAMES.get(c, str(c)) for c in drop])
    back = _give_back_to(pack)
    if back:
        out["restore"] = {"to": describe(back), "moves": len(moves(back))}
    return out


def drop_buttons(proj):
    """Settings > Buttons > Drop: the user agrees with the suggestion."""
    rep = button_report(proj)
    proj, cfg, pdir = _ai(proj)
    if not rep.get("found") or not rep.get("suggest"):
        return {"ok": False, "msg": "There's nothing to drop - no button has been shown to do nothing yet."}
    pack = json.loads((pdir / "pack.json").read_text(encoding="utf-8"))
    ctrl = controls_of(pack)
    stats = _evidence(proj, ctrl)
    if rep["suggest"]["source"] == "training":
        sug = _suggest(ctrl, stats)
        drop, new, how = sug["drop"], sug["new"], "in its own play"
    else:
        drop = [c for c in pack["finder_suggests"]["drop"] if c in inputs_of(ctrl)]
        new, how = _without(ctrl, drop), "while you played"
    record_change(pack, new, f"dropped by you (Settings > Buttons): {_names(drop)} never changed anything {how}")
    pack.pop("finder_suggests", None)
    write_pack(pdir, pack)
    _keep_evidence(proj, new, stats)
    return {"ok": True, "msg": f"Dropped {_names(drop)} - it now has {len(moves(new))} moves. If it's training it starts "
                               "again with them in a moment (what it has learned is carried over); otherwise next time "
                               "you press Train. Changed your mind? 'Give it its buttons back' is right there."}


def give_back(proj):
    """Settings > Buttons > Give it its buttons back: undo the most recent drop (and stop suggesting)."""
    proj, cfg, pdir = _ai(proj)
    if pdir is None:
        return {"ok": False, "msg": "This game's buttons can't be changed here."}
    pack = json.loads((pdir / "pack.json").read_text(encoding="utf-8"))
    back = _give_back_to(pack)
    if not back:
        return {"ok": False, "msg": "It already has every button it ever had."}
    stats = _evidence(proj, controls_of(pack))
    record_change(pack, back, "given back by you (Settings > Buttons)")
    pack["keep_buttons"] = True
    pack.pop("finder_suggests", None)
    write_pack(pdir, pack)
    _keep_evidence(proj, back, stats)
    return {"ok": True, "msg": f"Its buttons are back - {describe(back)} ({len(moves(back))} moves) - and it won't suggest "
                               "dropping any again unless you ask it to check. If it's training it starts again with "
                               "them in a moment; what it has learned is carried over."}


def keep_buttons(proj, keep=True):
    """Settings > Buttons > Keep all its buttons (stop checking and suggesting) - or check them again."""
    proj, cfg, pdir = _ai(proj)
    if pdir is None:
        return {"ok": False, "msg": "This game's buttons can't be changed here."}
    pack = json.loads((pdir / "pack.json").read_text(encoding="utf-8"))
    if keep:
        pack["keep_buttons"] = True
        pack.pop("finder_suggests", None)
    else:
        pack.pop("keep_buttons", None)
    write_pack(pdir, pack)
    return {"ok": True, "msg": "OK - it keeps all its buttons and stops checking them." if keep else
            "It'll check its buttons again the next time it trains."}


def _played(proj, status=None):
    """How many moves it has practised (its status, else the last good line of its training log)."""
    try:
        if status and status.get("step"):
            return int(status["step"])
    except Exception:
        pass
    try:
        with open(Path(proj) / "files" / "run" / "train_log.csv", "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 8192))
            tail = f.read().replace(b"\x00", b"").decode("utf-8", "replace").splitlines()
        for ln in reversed(tail):
            parts = ln.split(",")
            if len(parts) > 2 and parts[2].isdigit():
                return int(parts[2])
    except Exception:
        pass
    return 0


def chip_suggestion(proj, status=None):
    """For the AI's card on the page: what Chip asks when some buttons never did anything (else None)."""
    rep = button_report(proj, light=True)
    sug = rep.get("suggest")
    if not sug:
        return None
    return {"played": _played(proj, status), "useful": [r["name"] for r in rep["inputs"] if r["kind"] == "useful"],
            "dead": sug["names"], "drop": sug.get("drop", []), "tries": sug.get("tries", 0), "source": sug["source"]}


def refresh_buttons(env, game_id):
    """Watch window: if the game's buttons have changed since the window opened, switch this game over to them (the
    emulator carries on; only the list of moves changes). Returns a description of the new buttons, or None."""
    pdir = pack_dir_for(game_id)
    if pdir is None or getattr(env, "actions", None) is None or not hasattr(env, "pack"):
        return None
    pack = json.loads((pdir / "pack.json").read_text(encoding="utf-8"))
    new = controls_of(pack)
    if moves(new) == [frozenset(t) for t in env.actions]:
        return None
    import gymnasium as gym
    env.pack.update({"axes": new["axes"], "buttons": new["buttons"], "n_actions": len(moves(new))})
    env.actions, env.action_names = _generic().action_table(env.pack)
    env.action_space = gym.spaces.Discrete(len(env.actions))
    try:
        from arcade_vec import PACKS
        if game_id in PACKS:
            PACKS[game_id]["n_actions"] = len(env.actions)
    except Exception:
        pass
    return describe(new)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit(__doc__)
    check(args[0], dry="--dry" in sys.argv)
