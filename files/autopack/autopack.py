"""Portable AI Arcade - AUTO-PACK: find an arcade game's score with nobody typing anything.

For each game: boot it, watch the attract mode (that's what "no game on" looks like), put a coin in, start, and let
a button-masher play. Every way an old game stores a number (BCD / binary, 1-3 bytes, either byte order, 68000
word-swapped) is followed at every memory address through the game. The score is the one that starts at 0, only
ever goes up, goes up in steps (not every frame like a timer), and - best evidence of all - usually lives in two
places that agree (the game's own copy and the one it draws on screen).
Then the pack is built and played through GenericArcadeEnv as a check.

Run: python autopack.py <zip> [<zip> ...]   (or --dir <folder> [--recursive] [--limit N] [--retry] [--redo])
Progress: files/autopack/robot_status.json; create files/autopack/STOP to stop after the current game.
This is the optional LIBRARY ROBOT add-on: the kit shows Start/Stop for it when this file is present.
Results: files/autopack/results.csv; packs in files/arcade/found/<set>/ with "auto": true.
Signed: Claude, 10 Oct 2026.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import csv
import os
import json
import math
import random
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

FILES = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FILES / "arcade" / "generic"))
import arcade_vec                      # noqa: E402
import ramscan                         # noqa: E402
import screenscore                     # noqa: E402
from libretro_host import LibretroHost  # noqa: E402

OUT = FILES / "autopack"
FOUND = FILES / "arcade" / "found"
ATTRACT, PLAY = 60 * 45, 60 * 150          # frames: watch the demo 45 s, then play 2.5 min
SAMPLE = 4
ROUND = {1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000}
FIRE = [0, 8, 1]                            # buttons 1-3 (most classic games use no more)
ENCODINGS = [(k, n, big, sw) for sw in (False, True) for k in ("bcd", "bin") for n in (1, 2, 3)
             for big in ((False, True) if n > 1 else (False,))]


def scan(trace):
    """Follow every encoding at every address through the play recording. Returns score-like candidates.
    Only addresses within a few bytes of something that changed are followed: a number that never changes can't be
    a score that went up, so this loses nothing and is many times quicker on big, mostly-still memories."""
    out = []
    r0 = trace[0]
    size = len(r0)
    changed = np.zeros(size, bool)
    for r in trace[1:]:
        changed |= r != r0
    if not changed.any():
        return out
    near = changed.copy()
    for k in range(1, 5):
        near[:-k] |= changed[k:]
        near[k:] |= changed[:-k]
    near_all = np.flatnonzero(near)
    for kind, n, big, sw in ENCODINGS:
        offs = near_all[near_all <= size - n]
        idx = offs[:, None] + (np.arange(n)[::-1] if big else np.arange(n))[None, :]
        if sw:
            even = size & ~1
            idx = np.where(idx < even, idx ^ 1, idx)        # 68000 games: each pair of bytes flipped
        prev = first = None
        for r in trace:
            b = r[idx].astype(np.int64)
            v = ramscan._decode_bcd(b) if kind == "bcd" else ramscan._decode_bin(b)
            if prev is None:
                first, prev = v.copy(), v
                up = np.zeros(len(v), np.int32)
                down = np.zeros(len(v), np.int32)
                bad = (v < 0).astype(np.int32)
                g = np.zeros(len(v), np.int64)
                continue
            d = v - prev
            up += d > 0
            down += (d < 0) & (v != 0)
            bad += v < 0
            g = np.gcd(g, np.where(d > 0, d, 0))
            prev = v
        T = len(trace)
        # torn reads (a multi-byte score caught half-updated) happen now and then: allow a few
        ok = ((first == 0) & (bad <= max(2, T // 100)) & (down <= 3) & (up >= 2) & (up <= 0.3 * T) & (prev > 0)
              & (prev < 2_000_000))
        for jj in np.flatnonzero(ok)[:2000]:
            out.append({"kind": kind, "n": n, "big": big, "offset": int(offs[jj]), "swapped": sw, "scale": 1,
                        "gcd": int(g[jj]), "ups": int(up[jj]), "final": int(prev[jj]), "downs": int(down[jj]),
                        "bad": int(bad[jj])})
    return out


def busy_map(trace):
    """Fraction of bytes in each 256-byte block that changed during play. Screen bitmaps light up almost entirely."""
    r0 = trace[0]
    changed = np.zeros(len(r0), bool)
    for r in trace[1::2]:
        changed |= r != r0
    n = len(r0) // 256 * 256
    frac = changed[:n].reshape(-1, 256).mean(1)
    return np.append(frac, changed[n:].mean() if len(r0) > n else 0.0)


def features(c, trace, busy):
    s = np.array([ramscan.decode(r, c) for r in trace[::2]], np.int64)
    inc = np.diff(s)
    inc = inc[inc > 0]
    lo = min(ramscan.footprint(c))
    gs = max(1, int(np.gcd.reduce(inc))) if len(inc) else 1
    c.update({"series": s, "distinct_steps": int(len(set(inc.tolist()))), "step_gcd": gs,
              "max_step": int(inc.max()) if len(inc) else 0, "busy": round(float(busy[min(lo // 256, len(busy) - 1)]), 2)})
    return c


def rank_key(c):
    g, fin = c["step_gcd"], c["final"]
    sane = fin <= 300_000 and fin // g <= 5000 and c["max_step"] <= 50_000       # 2.5 min of button-mashing
    stepped = g in ROUND and c["ups"] >= 3 and fin >= 10
    tidy = c["distinct_steps"] <= 20                    # points come in a limited set of sizes (10, 20, 50, 100...)
    quiet = c["busy"] < 0.5                             # not in a block that's mostly screen picture
    clean = c["downs"] + c["bad"] <= 2                  # allow the odd torn read (score caught half-updated)
    # then: the number that went up most often (a score ticks up all game; level/lives/ship counters only now and then)
    return (sane and stepped, quiet, tidy, c["kind"] == "bcd", clean, c["ups"], -g, c["agree"] >= 2)


def pick(cands, trace):
    """Score every candidate on its own: round steps, a handful of step sizes, not in a screen-bitmap block, BCD,
    clean. Copies that tell the same story (same series up to x10/x100) are counted as agreement."""
    if not cands:
        return []
    busy = busy_map(trace)
    for c in cands:
        features(c, trace, busy)
    norm = {}
    for c in cands:
        s = c["series"]
        key = tuple((s // max(1, int(np.gcd.reduce(s[s > 0])) if (s > 0).any() else 1)).tolist())
        norm.setdefault(key, []).append(c)
    for group in norm.values():
        places = len({min(ramscan.footprint(c)) // 4 for c in group})
        for c in group:
            c["agree"] = places
            c["_group"] = id(group)
    ranked = sorted(cands, key=rank_key, reverse=True)
    out, seen = [], set()
    for c in ranked:
        if c["_group"] in seen:
            continue
        seen.add(c["_group"])
        group = [x for x in cands if x["_group"] == c["_group"] and rank_key(x)[:3] == rank_key(c)[:3]]
        # finest steps first (the reading whose lowest byte moves - not one with a stuck byte below or a swapped
        # order), then 3 bytes of headroom, then plain byte order
        best = sorted(group, key=lambda x: (x["step_gcd"], x["big"], x["swapped"], ramscan._width_pref(x), x["offset"]))[0]
        # a wider reading at the same address that agrees almost always (torn reads aside) = more digits of headroom
        for w in cands:
            if (w is not best and w["offset"] == best["offset"] and w["kind"] == best["kind"] and w["big"] == best["big"]
                    and w["swapped"] == best["swapped"] and w["n"] > best["n"] and w["n"] <= 3):
                agree = np.mean(w["series"] % (100 ** best["n"] if w["kind"] == "bcd" else 256 ** best["n"]) == best["series"])
                if agree >= 0.98 and (w["n"] > best.get("_upgraded", best["n"])):
                    best = dict(w, confidence=None, _upgraded=w["n"])
        k = rank_key(best)
        best["confidence"] = "high" if all(k[:5]) else "medium" if k[0] and k[1] else "low"
        out.append(best)
        if len(out) == 5:
            break
    pick.debug = [{"rank": [bool(x) if isinstance(x, (bool, np.bool_)) else int(x) for x in rank_key(c)],
                   "c": (c["kind"], c["n"], c["big"], c["swapped"], hex(c["offset"]), c["step_gcd"], c["final"],
                         c["distinct_steps"], c["busy"])} for c in ranked[:30]]
    for c in cands + out:
        for k in ("series", "_group", "_upgraded"):
            c.pop(k, None)
    return out


def do_game(zp, rt, log, core="fbneo"):
    zp = Path(zp)
    setname = zp.stem.lower()
    t0 = time.time()
    local = rt / "roms" / zp.name
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zp, local)
    try:
        h = LibretroHost(arcade_vec.local_core(rt, core), local, rt, require_ram=False)
    except Exception as e:
        return {"set": setname, "status": "won't run", "reason": f"{core}: " + str(e)[:110]}
    try:
        # does the game actually RUN on this emulator? (a romset made for another one shows a frozen error screen)
        f0 = h.run(()).copy()
        for _ in range(300):
            f = h.run(())
        moved = f.shape != f0.shape or (f != f0).any()     # many games change screen size after boot
        if not h.dll.retro_serialize_size() or not moved:
            return {"set": setname, "status": "won't run", "reason": f"{core}: frozen screen / no save-state"}

        def mem():          # the game's memory: shared directly (FBNeo) or read out of a save-state (MAME 2003-Plus)
            return h.ram.copy() if h.ram is not None else np.frombuffer(h.save_state(), np.uint8).copy()
        size = len(mem())
        # no memory to search (none exposed, or too big to follow): read the score off the screen instead; any memory
        # is still sampled now and then, just to tell 'a game is on' from the demo
        no_mem = size < 256
        screen_only = no_mem or size > 600_000
        every = 32 if screen_only else SAMPLE
        rng = random.Random(1)
        idle, play = [], []
        for f in range(ATTRACT):
            h.run((), capture=False)
            if f > 600 and f % (8 if not screen_only else 32) == 0 and not no_mem:
                idle.append(mem())
        for b, nfr in (((2,), 8), ((), 90), ((3,), 8), ((), 6)):
            for _ in range(nfr):
                h.run(b, capture=False)
        start_state, buttons = None, ()
        shots, shot_f = [], []          # what's on the screen every 8 frames: for reading the score off the screen
        hold = ()
        for f in range(PLAY):
            mode = (f // 1200) % 4          # 20 s each: random / tap fire while wandering / right + fire / up + fire
            if f % 8 == 0:
                d = rng.choice([(), (4,), (5,), (6,), (7,), (4, 6), (4, 7), (5, 6), (5, 7)])
                buttons = d + tuple(b for b in FIRE if rng.random() < 0.3)
                if f % 32 == 0:
                    hold = rng.choice([(), (4,), (5,), (6,), (7,)])
            if mode == 0:
                b = buttons
            else:
                b = {1: hold, 2: (7,), 3: (4,)}[mode] + ((FIRE[(f // 16) % 2],) if f % 8 < 4 else ())
            if f >= 1800 and f % 1800 < 6:
                b = (2,)                    # keep it going after a game over: coin...
            elif f >= 1800 and 60 <= f % 1800 < 66:
                b = (3,)                    # ... and start
            fr = h.run(b, capture=(f % 8 == 0))
            if f % 8 == 0 and fr is not None:
                shots.append(screenscore.cids(fr))
                shot_f.append(f)
            if f == 180:
                start_state = h.save_state()
            if f >= 180 and f % every == 0 and not no_mem:
                play.append(mem())
        # ---- the SCREEN first: read the digits a person would read
        try:
            spec = screenscore.fit(shots)
        except Exception:
            spec = None
        screen_vals = None
        if spec:
            rd = screenscore.ScreenScore(spec)
            screen_vals = [(f, rd.read(c)) for f, c in zip(shot_f, shots)]
            screen_vals = [(f, v) for f, v in screen_vals if v is not None and f >= 180]
        moving = bool(screen_vals) and len({v for _, v in screen_vals}) >= 3
        if moving and spec["report"].get("resets", 0) * 3 > spec["report"].get("ups", 0):
            moving = False                   # it keeps going back to 0: a counter or a timer, not a score

        def agrees(c):
            if not screen_vals:
                return 0.0
            hits = n = 0
            for f, v in screen_vals:
                i = (f - 180) // every
                if i < len(play):
                    n += 1
                    m = ramscan.decode(play[i], c)
                    hits += m == v or m * 10 == v or m * 100 == v
            return hits / max(1, n)

        cands, raw = [], []
        if moving and play:
            # what the screen shows, in the middle of each steady stretch, labels an exact memory search
            stretches, cur = [], []
            for f, v in screen_vals:
                if cur and v != cur[-1][1]:
                    stretches.append(cur)
                    cur = []
                cur.append((f, v))
            if cur:
                stretches.append(cur)
            labels = []
            for st in stretches:
                if len(st) >= 2:
                    f, v = st[len(st) // 2]
                    i = (f - 180) // every
                    if 0 <= i < len(play):
                        labels.append((play[i], v))
            if len({v for _, v in labels}) >= 3:
                try:
                    exact = ramscan.rank(ramscan.find_score(labels[-60:]), play)
                except Exception:
                    exact = []
                if exact:
                    best_m = max(exact[:12], key=agrees)
                    a = agrees(best_m)
                    if a >= 0.9:
                        vals = [v for _, v in screen_vals]
                        best_m.update(confidence="high", screen_agrees=round(a, 3), found_by="screen labels",
                                      ups=sum(1 for x, y in zip(vals, vals[1:]) if y > x), final=int(max(vals)),
                                      agree=2, step_gcd=0, distinct_steps=0, busy=0)
                        cands = [best_m]
        if not cands and moving and spec.get("confident"):
            cands = [{"kind": "screen", "screen": spec, "confidence": "high", "ups": spec["report"]["ups"],
                      "final": max(v for _, v in screen_vals), "agree": 1, "step_gcd": 0, "distinct_steps": 0,
                      "busy": 0, "n": 0, "big": False, "offset": -1, "swapped": False, "scale": 1}]
        if not cands and not screen_only:
            # the screen couldn't be read: follow every number in memory instead (slow), checked by the screen if it can
            raw = scan(play)
            cands = pick(raw, play)
            if moving:
                checked = sorted(cands, key=lambda c: -agrees(c))
                if checked and agrees(checked[0]) >= 0.9:
                    cands = checked
                    cands[0]["confidence"] = "high"
                    cands[0]["screen_agrees"] = round(agrees(checked[0]), 3)
        dbg = OUT / "debug"
        dbg.mkdir(parents=True, exist_ok=True)
        (dbg / f"{setname}.json").write_text(json.dumps({"n_raw": len(raw), "top": cands, "groups": getattr(pick, "debug", None)},
                                                        indent=1, default=int))
        if not cands:
            why = ("no score-like number found (maybe never scored)" if not screen_only else
                   "no memory exposed, and no score found on the screen" if no_mem else
                   f"memory too big to search ({size // 1024} KB), and no score found on the screen")
            return {"set": setname, "status": "needs a human", "reason": why}
        early = play[: (60 * 60) // every]
        flag = ramscan.find_playing_flag(early, idle) if early and idle else None
        best = cands[0]
        pack = {"id": "found-" + setname, "name": f"{setname.upper()} (arcade, auto)", "set": setname,
                "rom_file": zp.name, "crcs": sorted({"%08x" % i.CRC for i in zipfile.ZipFile(zp).infolist() if not i.is_dir()}),
                "score": (best["screen"] if best["kind"] == "screen" else
                          {k: best[k] for k in ("kind", "n", "big", "offset", "swapped", "scale", "stride", "base",
                                                "max_digits") if k in best}),
                "read_from": "screen" if best["kind"] == "screen" else "memory",
                "alternatives": [{k: c[k] for k in ("kind", "n", "big", "offset", "swapped", "scale", "stride", "base",
                                                    "max_digits") if k in c} for c in cands[1:] if c["kind"] != "screen"],
                "lives": None, "playing": flag, "axes": ["lr", "ud"], "buttons": FIRE, "decision_repeat": 4,
                "max_jump": int(max(50000, 20 * best["final"])), "health": None, "reward": "score", "auto": True, "core": core,
                "confidence": best["confidence"], "evidence": {k: best[k] for k in ("agree", "step_gcd", "ups", "final", "distinct_steps", "busy")},
                "made": time.strftime("%Y-%m-%d %H:%M")}
        from generic_env import action_table
        pack["n_actions"] = len(action_table(pack)[0])
    finally:
        h.close()
    # ---- self-check through the real environment (a fresh process would be cleaner; one core per process, so
    # the check runs in a child)
    dst = FOUND / setname
    tmp = OUT / "staging" / setname
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "start.state").write_bytes(start_state)
    (tmp / "pack.json").write_text(json.dumps(pack, indent=1))
    import subprocess
    def run_check():
        r = subprocess.run([sys.executable, "-B", __file__, "--check", str(tmp), str(zp)], capture_output=True,
                           text=True, timeout=900)
        try:
            return json.loads(r.stdout.strip().splitlines()[-1]), r
        except Exception:
            return None, r
    chk, r = run_check()
    if chk is None and "access violation" in (r.stderr or "") and core == "mame2003_plus":
        # its save-state can't be restored in a fresh emulator: the AI will start each game by switching on instead
        pack["reset"] = "reboot"
        (tmp / "pack.json").write_text(json.dumps(pack, indent=1))
        chk, r = run_check()
    if chk is None:
        return {"set": setname, "status": "check crashed", "reason": (r.stderr or r.stdout)[-150:].replace("\n", " ")}
    ok = chk["games"] >= 2 and chk["scored_games"] >= 1 and not chk["weird"]
    sure = best["confidence"] == "high" and best["ups"] >= 5           # it actually watched the score move a few times
    status = "ok" if ok and sure else ("maybe" if ok else "check failed")
    res = {"set": setname, "status": status, "reason": chk["note"], "confidence": best["confidence"],
           "offset": hex(best["offset"]) if best["kind"] != "screen" else "screen",
           "how": ramscan.describe(best) if best["kind"] != "screen" else "read off the screen " + str(best["screen"]["box"]), "flag": bool(flag),
           "minutes": round((time.time() - t0) / 60, 1)}
    if status in ("ok", "maybe") and "--dry" not in sys.argv:
        if dst.exists() and (dst / "pack.json").exists():
            res["status"] += " (kept the pack that was already there - an AI may be using it)"
        else:
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(tmp, dst, dirs_exist_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return res


def check(pack_dir, rom):
    """Play 3 games with random buttons through GenericArcadeEnv. Prints one JSON line."""
    from generic_env import GenericArcadeEnv
    rt = Path(tempfile.gettempdir()) / f"PAA_chk_{os.getpid()}"
    LibretroHost._video = arcade_vec._fast_video
    core = json.loads((Path(pack_dir) / "pack.json").read_text()).get("core", "fbneo")
    env = GenericArcadeEnv(pack_dir, root=rt, core_path=arcade_vec.local_core(rt, core), rom_path=Path(rom))
    games = scored = 0
    lens, scores, weird = [], [], False
    for g in range(3):
        env.reset(seed=g)
        n = 0
        while True:
            _, _, term, trunc, info = env.step(random.randrange(env.action_space.n))
            n += 1
            if term or trunc or n >= 9000:
                break
        games += 1
        scored += info["score"] > 0
        lens.append(n)
        scores.append(info["score"])
        weird |= info["score"] > 300_000
    idle = None
    if len(set(scores)) == 1 and scores[0] > 0:
        # three random games, the very same score: a counter that ticks whatever you do, or just the same easy points
        # at the start? Play one more with hands off: if that scores the same too, it isn't down to the player.
        env.reset(seed=3)
        n = 0
        while True:
            _, _, term, trunc, info = env.step(0)
            n += 1
            if term or trunc or n >= 9000:
                break
        idle = info["score"]
        weird |= idle == scores[0]
    note = (f"3 random games: scores {scores}, lengths {lens} moves" + ("" if any(l < 9000 for l in lens) else " (never hit game over)")
            + ("" if idle is None else f"; hands-off game: {idle}"))
    print(json.dumps({"games": games, "scored_games": scored, "weird": weird, "note": note}))


def main():
    if sys.argv[1] == "--check":
        return check(sys.argv[2], sys.argv[3])
    if sys.argv[1] == "--one":
        rt = Path(tempfile.gettempdir()) / "PortableAIArcade" / f"autopack-{os.getpid()}"
        shutil.rmtree(rt, ignore_errors=True)
        try:
            core = sys.argv[sys.argv.index("--core") + 1] if "--core" in sys.argv else "fbneo"
            res = do_game(Path(sys.argv[2]), rt, None, core)
        except Exception:
            res = {"set": Path(sys.argv[2]).stem.lower(), "status": "crashed",
                   "reason": traceback.format_exc()[-150:].replace("\n", " ")}
        shutil.rmtree(rt, ignore_errors=True)
        print(json.dumps(res, default=str), flush=True)
        return
    if sys.argv[1] == "--dir":
        zips = sorted(Path(sys.argv[2]).rglob("*.zip") if "--recursive" in sys.argv else Path(sys.argv[2]).glob("*.zip"))
        if "--limit" in sys.argv:
            zips = zips[: int(sys.argv[sys.argv.index("--limit") + 1])]
    else:
        zips = [Path(a) for a in sys.argv[1:] if not a.startswith("--")]
    OUT.mkdir(exist_ok=True)
    resf = OUT / ("dry-run.csv" if "--dry" in sys.argv else "results.csv")
    done = set()
    if resf.exists():
        with open(resf, newline="") as f:
            rows = {r["set"]: r for r in csv.DictReader(f)}        # the last try of each game counts
        retry = ("check failed", "needs a human", "maybe", "check crashed", "crashed") if "--retry" in sys.argv else ()
        done = {k for k, r in rows.items() if r["status"] not in retry}
    fields = ["set", "core", "status", "reason", "confidence", "offset", "how", "flag", "minutes", "when"]
    # skip games that already have a pack (built in, or found earlier) - nothing to find there
    arcade_vec.refresh_found()
    if "--redo" not in sys.argv:
        zips = [z for z in zips if z.stem.lower() not in done]
        zips = [z for z in zips if not arcade_vec.identify_zip(z)]
    STOP, STATUS = OUT / "STOP", OUT / "robot_status.json"
    STOP.unlink(missing_ok=True)
    stat = {"state": "working", "total": len(zips), "checked": 0, "found": 0, "current": "", "pid": os.getpid(),
            "started": time.strftime("%Y-%m-%d %H:%M"), "last": ""}

    def put(**k):
        stat.update(k, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        try:
            STATUS.write_text(json.dumps(stat))
        except Exception:
            pass
    put()
    t_start = time.time()
    for zp in zips:
        if STOP.exists():
            STOP.unlink(missing_ok=True)
            put(state="stopped", current="")
            return
        left = stat["total"] - stat["checked"]
        per = (time.time() - t_start) / max(1, stat["checked"])
        put(current=zp.stem, eta_min=round(left * per / 60) if stat["checked"] else None)
        # FBNeo can only start once per process, so every game gets a fresh one
        import subprocess
        tried = []
        for core in [c for c in ("fbneo", "mame2003_plus") if arcade_vec.CORES[c].exists()]:
            try:
                r = subprocess.run([sys.executable, "-B", __file__, "--one", str(zp), "--core", core]
                                   + (["--dry"] if "--dry" in sys.argv else []), capture_output=True, text=True, timeout=1800)
                res = json.loads(r.stdout.strip().splitlines()[-1])
            except Exception:
                res = {"set": zp.stem.lower(), "status": "crashed",
                       "reason": (locals().get("r") and (r.stderr or "")[-150:] or traceback.format_exc()[-150:]).replace("\n", " ")}
            tried.append(core)
            if res.get("status") != "won't run":
                break
        res["core"] = tried[-1]
        res["when"] = time.strftime("%Y-%m-%d %H:%M")
        new = not resf.exists()
        with open(resf, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(res)
        print(json.dumps(res), flush=True)
        stat["checked"] += 1
        if str(res.get("status", "")).startswith(("ok", "maybe")):
            stat["found"] += 1
        put(last=f"{res.get('set')}: {res.get('status')}")
    put(state="finished", current="")


if __name__ == "__main__":
    main()
