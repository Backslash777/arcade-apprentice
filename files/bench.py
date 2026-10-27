"""Arcade Apprentice - TUNE FOR THIS PC. Times the real thing on this machine and keeps the fastest settings.

The learning recipe is the same on every PC (trainer.py); a PC only changes how FAST it runs. So this measures:
  1. the brain on the graphics card - how big a bite of a lesson the card can chew at once, and how fast;
  2. the games - the real trainer runs for about a minute at a time with more and more games at once, and the
     fastest (moves a second, once it has warmed up) wins. Atari (Breakout, which comes with the kit) and, if you
     have one, an arcade game you've trained before.
The answers go in files\\benchmark.json. The kit uses them on this PC from then on and ignores them on any other PC
(different processor, memory or graphics card). Nothing else should be training while it runs. About 15 minutes.

    python bench.py [--quick] [--arcade "<AI folder name>"] [--skip-atari] [--skip-arcade]

Signed: Claude, 17 Oct 2026 (Jonny's idea: let the PC show what's fastest instead of guessing).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

FILES = Path(__file__).resolve().parent
if str(FILES) not in sys.path:
    sys.path.insert(0, str(FILES))
KIT = FILES.parent
PROJECTS = KIT / "My AIs"
ROMS = KIT / "ROMs - put your games here"
OUT = FILES / "benchmark.json"
TMP = Path(tempfile.gettempdir()) / "ArcadeApprentice-bench"
MACHINE_KEYS = ("cpu_name", "cpu_threads", "ram_gb", "gpu_name", "backend")
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def say(msg=""):
    print(msg, flush=True)
    try:
        (FILES / "logs").mkdir(exist_ok=True)
        with open(FILES / "logs" / "benchmark.log", "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def training_now():
    """The name of an AI that's training right now, or None."""
    for st in sorted(PROJECTS.glob("*/files/run/status.json")) if PROJECTS.exists() else []:
        try:
            s = json.loads(st.read_text())
            if s.get("state") in ("training", "starting", "restarting") and s.get("pid"):
                import psutil
                if psutil.pid_exists(int(s["pid"])):
                    return st.parents[2].name
        except Exception:
            continue
    return None


# ------------------------------------------------------------------------------------------------ 1. the brain
def brain_bites(hw):
    """Forward + backward + step of the real brain on bites of 256 .. 2,048 screens. The card is warmed up first (its
    clock settles as it heats), then each size is timed twice - smallest to largest and back - so a card that slows
    as it warms doesn't flatter whichever size went first. Returns (the bite to use, {bite: {"per_s", "peak_gb"}})."""
    import torch
    import device
    from trainer import Agent
    dev, name, backend, problem = device.pick(prefer_gpu=bool(hw.get("use_gpu", True)))
    if backend == "cpu":
        say("  No graphics card in use - the processor does the learning, in bites of 1,024.")
        return None, {}
    cuda = backend in ("cuda", "rocm")
    vram = float(hw.get("vram_gb") or 0)
    agent = Agent(18, int(hw.get("width", 1))).to(dev)
    opt = torch.optim.Adam(agent.parameters(), lr=1e-6, eps=1e-5)

    def timed(bite, secs):
        x = torch.randint(0, 255, (bite, 4, 84, 84), dtype=torch.uint8).to(dev)
        a = torch.randint(0, 18, (bite,)).to(dev)
        try:
            def lesson():
                logits, v = agent(x)
                loss = torch.nn.functional.cross_entropy(logits, a) + 1e-3 * v.pow(2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                return loss
            float(lesson().detach())
            t0, n = time.time(), 0
            while time.time() - t0 < secs:
                last = lesson()
                n += 1
            float(last.detach())
            return n * bite / (time.time() - t0)
        finally:
            del x, a

    sizes, fits, peak, rates = (256, 512, 1024, 2048), [], {}, {}
    for bite in sizes:                                  # which sizes fit at all, and how much memory each needs
        try:
            if cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            timed(bite, 0.5)
            peak[bite] = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else None
            fits.append(bite)
        except RuntimeError as e:
            say(f"  bites of {bite:>5,}: too big for this card ({str(e).splitlines()[0][:70]})")
            break
    if not fits:
        return 256, {}
    timed(fits[-1], 8)                                   # warm the card up
    for order in (fits, fits[::-1]):
        for bite in order:
            rates.setdefault(bite, []).append(timed(bite, 3))
    res = {}
    for bite in fits:
        r = int(sum(rates[bite]) / len(rates[bite]))
        res[bite] = {"per_s": r, "peak_gb": round(peak[bite], 2) if peak.get(bite) else None}
        say(f"  bites of {bite:>5,}: {r:>7,} screens a second" +
            (f", {peak[bite]:.1f} GB of {vram:.0f} GB" if peak.get(bite) else ""))
    del agent, opt
    if cuda:
        torch.cuda.empty_cache()
    ok = [b for b in fits if not (res[b]["peak_gb"] and vram) or res[b]["peak_gb"] <= 0.8 * vram] or fits[:1]
    big = max(ok)                                        # the biggest that fits with room to spare...
    quick = max(ok, key=lambda b: res[b]["per_s"])       # ...unless a smaller one is clearly faster
    return (quick if res[quick]["per_s"] > 1.10 * res[big]["per_s"] else big), res


# ------------------------------------------------------------------------------------------------ 2. the games
def _when(row):
    return time.mktime(time.strptime(row["time"], "%Y-%m-%d %H:%M:%S"))


def trial(cfg, n, seconds, bite=None, split=None):
    """Run the real trainer on a scratch AI with n games at once; moves a second once warmed up (+ details).
    ALFRED 21 Oct 2026: bite = bite size to use (PAA_BITE), split = fast cores for the learning (PAA_SPLIT_CPUS)."""
    d = TMP / f"{cfg['platform']}-{n}-{bite or 0}-{split or 0}"
    shutil.rmtree(d, ignore_errors=True)
    (d / "files" / "run").mkdir(parents=True)
    (d / "project.json").write_text(json.dumps(cfg, indent=1))
    env = dict(os.environ, PAA_ENVS=str(n))
    for k in ("PAA_BITE", "PAA_PROBE_EVERY", "PAA_PROBE_MIN", "PAA_SPLIT_CPUS"):
        env.pop(k, None)
    if bite:
        env["PAA_BITE"] = str(bite)
    if split:
        env["PAA_SPLIT_CPUS"] = str(split)
    out = open(d / "out.txt", "w", encoding="utf-8", errors="replace")
    p = subprocess.Popen([sys.executable, "-u", str(FILES / "trainer.py"), str(d)], cwd=str(FILES), env=env,
                         stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=NOWIN)
    t0, low, mem_hit = time.time(), 0, False

    def starved():   # ALFRED 21 Oct 2026 review: a PC short of memory crawls into its paging file long before any
        try:         # 'WinError 1455' is printed - stop the trial when free memory is nearly gone (two checks running)
            import psutil
            vm = psutil.virtual_memory()
            return vm.available < max(1.0e9, 0.06 * vm.total)
        except Exception:
            return False

    while time.time() - t0 < seconds and p.poll() is None:
        time.sleep(1)
        low = low + 1 if starved() else 0
        if low >= 2:
            mem_hit = True
            break
    # ALFRED 21 Oct 2026 (part 3): big sizes warm up slowly - wait (up to 2.5x as long) for enough lessons to time
    while not mem_hit and p.poll() is None and time.time() - t0 < 2.5 * seconds:
        try:
            with open(d / "files" / "run" / "train_log.csv", newline="") as f:
                if sum(1 for _ in f) >= 8:           # header + 7 lessons
                    break
        except OSError:
            pass
        time.sleep(2)
        low = low + 1 if starved() else 0
        if low >= 2:
            mem_hit = True
            break
    (d / "files" / "run" / "STOP").write_text("stop")
    try:
        p.wait(timeout=150)
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
    out.close()
    run = d / "files" / "run"
    try:
        with open(run / "train_log.csv", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        rows = []
    if len(rows) >= 2 and rows[-1]["step"] == rows[-2]["step"]:
        rows = rows[:-1]                            # the row written while stopping
    info = {"moves_per_s": 0, "iterations": len(rows)}
    if len(rows) >= 5:
        a, b = rows[2], rows[-1]                    # skip the warm-up
        secs = max(1.0, _when(b) - _when(a))
        info["moves_per_s"] = int((int(b["step"]) - int(a["step"])) / secs)
        per_it = secs / max(1, len(rows) - 3)
        try:
            learn = [float(x) for x in re.findall(r"learning ([\d.]+)s in the background",
                                                   (run / "console.log").read_text(encoding="utf-8"))[2:]]
            if learn:
                info["brain_busy"] = min(100, int(100 * (sum(learn) / len(learn)) / per_it))
        except Exception:
            pass
    else:
        try:
            _txt = (d / "out.txt").read_text(encoding="utf-8", errors="replace")
            info["error"] = _txt.strip().splitlines()[-1][:160]
            if any(w in _txt for w in ("WinError 1455", "paging file", "MemoryError", "out of memory")):
                info["memory"] = True                   # (part 4) ran out of memory somewhere in there
                info["error"] = "not enough memory for this many"
        except Exception:
            info["error"] = "it didn't get going"
    if mem_hit:                                     # we stopped it: this many machines don't fit this PC
        info.update(moves_per_s=0, memory=True, error="not enough memory for this many")
    shutil.rmtree(d, ignore_errors=True)
    return info


def _short_of_memory(r):
    return bool(r.get("memory")) or any(w in str(r.get("error", "")) for w in ("1455", "paging file", "MemoryError",
                                                                             "out of memory", "not enough memory"))


def ladder(cfg, candidates, seconds, guess, what, ceiling=128, bite=None):
    """Time each size; while the biggest so far is still clearly the fastest (5%+ better than the size before), keep
    climbing by a third, up to the ceiling."""
    tried, todo, mem_full = {}, list(candidates), False
    while todo:
        n = todo.pop(0)
        r = trial(cfg, n, seconds, bite=bite)
        if not r["moves_per_s"] and not _short_of_memory(r):      # a start-up crash: give it one more go
            r = trial(cfg, n, seconds, bite=bite)
        tried[str(n)] = r
        if r["moves_per_s"]:
            busy = f" (the brain was busy {r['brain_busy']}% of the time)" if "brain_busy" in r else ""
            say(f"  {n:>3} {what} at once: {r['moves_per_s']:>6,} moves a second{busy}")
        else:
            say(f"  {n:>3} {what} at once: no result - {r.get('error', '?')}")
            if _short_of_memory(r):                  # this PC's memory ceiling: nothing bigger will fit either
                mem_full = True
                todo = [m for m in todo if m < n]
        if not todo:
            sizes = sorted(int(k) for k, v in tried.items() if v["moves_per_s"])
            if len(sizes) >= 2 and sizes[-1] == n:
                gain = tried[str(n)]["moves_per_s"] / max(1, tried[str(sizes[-2])]["moves_per_s"])
                nxt = (n * 4 // 3 + 7) // 8 * 8
                if gain >= 1.05 and n < nxt <= ceiling and not mem_full:
                    say(f"  ...still getting faster - trying {nxt}")
                    todo.append(nxt)
    good = {int(k): v["moves_per_s"] for k, v in tried.items() if v["moves_per_s"]}
    if not good:
        return None, tried
    best = max(good.values())
    pick = min(n for n, v in good.items() if v >= 0.97 * best)    # as fast as the fastest, with the fewest games
    say(f"  -> {pick} {what} at once ({good[pick]:,} moves a second). The kit's own guess was {guess}.")
    return {"num_envs": pick, "moves_per_s": good[pick], "guess": guess, "tried": tried}, tried


def base_cfg(hw, platform, game_id, name):
    return {"game_id": game_id, "game_name": f"Benchmark ({name})", "platform": platform, "reward_style": "log",
            "life_penalty": 0.0, "curiosity": 0.01, "gamma": 0.99, "learning_rate": 0.00025, "sticky": True,
            "total_steps": 1e12, "length": "forever", "explorer": 0.0, "go_explore": 0.0, "learn_from_me": 0,
            "learn_from_best": False, "check_buttons": False, "brain_size": "auto", "hardware": hw}


def arcade_game(choice=None):
    """An arcade AI to time with: the one asked for, else the most recently trained one whose ROM is here."""
    best = None
    for pj in PROJECTS.glob("*/project.json") if PROJECTS.exists() else []:
        try:
            c = json.loads(pj.read_text())
        except Exception:
            continue
        if c.get("platform") != "arcade" or not c.get("rom_file") or not (ROMS / c["rom_file"]).exists():
            continue
        if choice and pj.parent.name.lower() != choice.lower():
            continue
        t = (pj.parent / "files" / "run" / "train_log.csv")
        when = t.stat().st_mtime if t.exists() else 0
        if best is None or when > best[0]:
            best = (when, pj.parent.name, c)
    return (best[1], best[2]) if best else (None, None)


def main():
    lock = FILES / "TUNING"                       # the page won't start any training while this is here
    lock.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + f" pid={os.getpid()}")   # ALFRED 21 Oct: the page checks it
    try:
        return _tune()
    finally:
        lock.unlink(missing_ok=True)


def _tune():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--arcade")
    ap.add_argument("--skip-atari", action="store_true")
    ap.add_argument("--skip-arcade", action="store_true")
    args = ap.parse_args()
    import app
    busy = training_now()
    if busy:
        say(f"'{busy}' is training right now - stop it first, then tune (the test needs the whole PC).")
        return 1
    say(f"\nTUNE FOR THIS PC - {time.strftime('%Y-%m-%d %H:%M')}")
    hw_atari = app.hardware("atari", use_tuning=False)
    hw_arcade = app.hardware("arcade", use_tuning=False)
    say(f"  {hw_atari.get('cpu_name')} ({hw_atari['cpu_threads']} threads), {hw_atari['ram_gb']} GB memory, "
        f"{hw_atari.get('gpu_name') or 'no graphics card in use'}"
        + (f" ({hw_atari.get('vram_gb')} GB)" if hw_atari.get("gpu_name") else ""))
    old = {}
    try:
        old = json.loads(OUT.read_text(encoding="utf-8"))
        if any(str(old.get("machine", {}).get(k)) != str(hw_atari.get(k)) for k in MACHINE_KEYS):
            old = {}
    except Exception:
        old = {}
    result = {"made": time.strftime("%Y-%m-%d %H:%M"), "machine": {k: hw_atari.get(k) for k in MACHINE_KEYS},
              "width": hw_atari.get("width")}
    for k in ("atari", "arcade"):
        if k in old:
            result[k] = old[k]

    say("\n1. THE BRAIN - how big a bite of a lesson the graphics card can chew at once")
    bite, bites = brain_bites(hw_atari)
    if bite:
        result["micro_batch"], result["bites"] = bite, {str(b): r for b, r in bites.items()}
        say(f"  -> bites of {bite:,} (the kit's own guess was {hw_atari['micro_batch']:,}).")

    if "atari" in result and not result["atari"].get("micro_batch") and old.get("micro_batch"):
        result["atari"]["micro_batch"] = old["micro_batch"]     # (part 2) keep the bite Atari was really using
    t = int(hw_atari["cpu_threads"])
    if not args.skip_atari:
        say("\n2. ATARI GAMES (Breakout) - the real trainer, about a minute for each size")
        cands = [64, 128, 256] if args.quick else [32, 64, 128, 256]
        if hw_atari.get("use_gpu") and (hw_atari.get("vram_gb") or 0) < 6:
            cands = [c for c in cands if c <= 64]
        # ALFRED 21 Oct 2026 (part 2): the bite size is checked in the real trainer too, and kept per platform
        acfg = base_cfg(hw_atari, "atari", "breakout", "Breakout")
        b1 = int(result.get("micro_batch") or hw_atari.get("micro_batch") or 1024)
        got, _ = ladder(acfg, cands, 60, hw_atari["num_envs"], "games",
                        ceiling=512 if (hw_atari.get("vram_gb") or 0) >= 12 else 256, bite=b1)
        if got:
            got["game"] = "Breakout"
            got["micro_batch"] = b1
            b2 = 1024 if b1 < 1024 else 256
            r = trial(acfg, got["num_envs"], 60, bite=b2)
            if not r["moves_per_s"] and not _short_of_memory(r):
                r = trial(acfg, got["num_envs"], 60, bite=b2)   # (part 3) one more go after a crash
            say(f"  bites of {b2:,} instead of {b1:,}: {r['moves_per_s']:,} moves a second" if r["moves_per_s"]
                else f"  bites of {b2:,} instead of {b1:,}: no result - {r.get('error', '?')}")
            if r["moves_per_s"] > 1.03 * got["moves_per_s"]:
                got["micro_batch"], got["moves_per_s"] = b2, r["moves_per_s"]
            result["atari"] = got
    if not args.skip_arcade:
        name, c = arcade_game(args.arcade)
        if not name:
            say("\n3. ARCADE GAMES - skipped: no arcade AI with its game file here yet (train one, then tune again).")
        else:
            say(f"\n3. ARCADE GAMES ({c.get('game_name', name)}) - the real trainer, about a minute and a half for each size")
            cfg = base_cfg(hw_arcade, "arcade", c["game_id"], c.get("game_name", name))
            cfg["rom_file"], cfg["rom_path"] = c["rom_file"], str(ROMS / c["rom_file"])
            spare = float(hw_arcade.get("ram_free_gb", hw_arcade["ram_gb"]) or 8)
            # ALFRED 21 Oct 2026: more machines kept getting faster past 128 (Alfred's PC: 64 2,283 / 96 2,843 / 192 2,997
            # moves a second; 256 trains well; 384 ran out of memory), so climb as far as memory allows
            cands = sorted({2 * t, 4 * t, 6 * t, 8 * t} if not args.quick else {4 * t, 8 * t})
            # (21 Oct review: a cabinet really costs ~0.16 GB - 192 machines used 29.6 GB on Alfred's PC - not 0.12)
            cands = [n for n in cands if 8 <= n <= 512 and n * 0.16 <= spare] or [max(8, min(t, int(spare / 0.16)))]
            got, _ = ladder(cfg, cands, 80, hw_arcade["num_envs"], "machines", ceiling=min(512, int(spare / 0.16)),
                            bite=256)
            if got:
                got["game"] = f"{c.get('game_name', name)} ({c['game_id']})"
                n, best_rate = got["num_envs"], got["moves_per_s"]
                got["micro_batch"] = 256
                r = trial(cfg, n, 80, bite=1024)
                if not r["moves_per_s"] and not _short_of_memory(r):
                    r = trial(cfg, n, 80, bite=1024)      # (part 3) one more go after a crash
                say(f"  bites of 1,024 instead of 256: {r['moves_per_s']:,} moves a second" if r["moves_per_s"]
                    else f"  bites of 1,024 instead of 256: no result - {r.get('error', '?')}")
                if r["moves_per_s"] > 1.03 * best_rate:
                    got["micro_batch"], best_rate = 1024, r["moves_per_s"]
                try:
                    import cpu_split
                    sp = cpu_split.plan(8)
                except Exception:
                    sp = None
                if sp:                             # two kinds of core: how many fast ones should the learning get?
                    fast = len(sp[0]) * 2
                    best_k = len(sp[0])
                    for k in sorted({max(2, fast // 4), max(2, (3 * fast) // 4)} - {best_k}):
                        r = trial(cfg, n, 80, bite=got["micro_batch"], split=k)
                        say(f"  learning on {k} fast cores: {r['moves_per_s']:,} moves a second" if r["moves_per_s"]
                            else f"  learning on {k} fast cores: no result - {r.get('error', '?')}")
                        if r["moves_per_s"] > 1.03 * best_rate:
                            best_k, best_rate = k, r["moves_per_s"]
                    got["trainer_cpus"] = best_k
                got["moves_per_s"] = best_rate
                result["arcade"] = got
    shutil.rmtree(TMP, ignore_errors=True)
    if not any(k in result for k in ("micro_batch", "atari", "arcade")):
        say("\nNothing could be measured, so nothing was changed.")
        return 1
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    say("\nTUNED FOR THIS PC - the kit uses these here from now on (and ignores them on any other PC):")
    if "atari" in result:
        say(f"  Atari:  {result['atari']['num_envs']} games at once ({result['atari']['moves_per_s']:,} moves a second)" +
            (f", bites of {result['atari']['micro_batch']:,}" if result["atari"].get("micro_batch") else ""))
    if "arcade" in result:
        say(f"  Arcade: {result['arcade']['num_envs']} machines at once ({result['arcade']['moves_per_s']:,} moves a second "
            f"with {result['arcade']['game']})")
        if result["arcade"].get("micro_batch"):
            say(f"          bites of {result['arcade']['micro_batch']:,}" + (
                f", learning on {result['arcade']['trainer_cpus']} fast cores" if result["arcade"].get("trainer_cpus") else ""))
    if "micro_batch" in result:
        say(f"  Brain:  bites of {result['micro_batch']:,}")
    say("  Run it again any time - after new hardware, or if things feel slow.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
