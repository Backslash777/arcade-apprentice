"""Portable AI Arcade - the setup wizard and control panel (local web page). Stdlib + psutil/torch if present.
Signed: Claude, 9 Oct 2026."""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # embeddable Python ignores the script folder

import hashlib
import json
import recipe_cards
import os
import platform as platform_mod
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

FILES = Path(__file__).resolve().parent
ROOT = FILES.parent
ROMS = ROOT / "ROMs - put your games here"
PROJECTS = ROOT / "My AIs"
PORT, REPORT_PORT = 8795, 8796
PY = sys.executable
# windows that should be SEEN (Teach) need the console Python - the page itself runs under pythonw.exe, which can't
# show a console window at all (19 Oct 2026: 'Teach' said 'in a new window' and none appeared)
PYCON = str(Path(PY).with_name('python.exe')) if Path(PY).name.lower() == 'pythonw.exe' and \
    Path(PY).with_name('python.exe').exists() else PY
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
BELOW = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)   # training gives way to whatever else you're doing
NEWCON = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)


def pretty(game_id):
    return " ".join(w.capitalize() for w in game_id.replace("_", " ").split())


def known_roms():
    """ALE recognises legitimate ROMs by fingerprint (md5)."""
    try:
        import ale_py.roms as r
        d = Path(r.__file__).parent
        return {hashlib.md5(p.read_bytes()).hexdigest(): p.stem for p in d.glob("*.bin")}
    except Exception:
        return {}


_KNOWN = None
_LIB_CACHE = {}


def atari_in_zip(path):
    """A zipped Atari 2600 game (one .bin/.a26 inside) - recognise the game inside."""
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            for i in z.infolist():
                if i.filename.lower().endswith((".bin", ".a26", ".rom")) and i.file_size <= 65536:
                    gid = _KNOWN.get(hashlib.md5(z.read(i)).hexdigest())
                    if gid:
                        return gid
    except Exception:
        pass
    return None


def atari_library(rescan=False):
    """Atari games in sub-folders (e.g. a big ROM collection): {relative path: game id} for the ones this kit can
    play. Fingerprinting thousands of files is slow on a cloud drive, so the answer is kept until the next Scan."""
    cache = FILES / "logs" / "atari-library.json"
    if not rescan:
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            return {}
    out = {}
    for p in ROMS.rglob("*"):
        if p.parent == ROMS or p.suffix.lower() not in (".bin", ".a26") or p.stat().st_size > 65536:
            continue
        gid = _KNOWN.get(hashlib.md5(p.read_bytes()).hexdigest())
        if gid:
            out[str(p.relative_to(ROMS))] = gid
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(out, indent=0), encoding="utf-8")
    return out


def titled(entry, path):
    """Add the proper game name (from the emulators' official lists) to a game-list entry."""
    try:
        import arcade_names
        r = arcade_names.lookup(path)
    except Exception:
        r = None
    if r:
        entry["title"] = r[0]
        entry["maker"] = ", ".join(x for x in (r[1], r[2]) if x)
    return entry


def games(rescan=False):
    global _KNOWN
    if _KNOWN is None:
        _KNOWN = known_roms()
    ROMS.mkdir(exist_ok=True)
    try:
        import arcade_vec
        arcade_vec.refresh_found()           # pick up games taught with Find the score since the page opened
    except Exception:
        arcade_vec = None
    found = []
    for p in sorted(ROMS.iterdir()):
        ext = p.suffix.lower()
        if ext in (".bin", ".a26", ".rom"):
            gid = _KNOWN.get(hashlib.md5(p.read_bytes()).hexdigest())
            found.append({"file": p.name, "game_id": gid, "name": pretty(gid) + " (Atari 2600)" if gid else None,
                          "platform": "atari", "why": None if gid else "not a supported Atari 2600 game"})
        elif ext == ".zip":
            inner = atari_in_zip(p)
            if inner:
                found.append({"file": p.name, "game_id": inner, "name": pretty(inner) + " (Atari 2600)",
                              "platform": "atari", "why": None})
                continue
            gid = arcade_vec.identify_zip(p) if arcade_vec else None
            why, can_find = None, False
            if arcade_vec and not arcade_vec.arcade_available():
                gid, why = None, "arcade games need Windows and files\\arcade\\fbneo_libretro.dll"
            elif not gid:
                why = "not set up yet - press Find the score to teach the kit this game (takes a few minutes of play)"
                can_find = True
            found.append(titled({"file": p.name, "game_id": gid, "platform": "arcade", "why": why, "can_find": can_find,
                                 "found": bool(gid and arcade_vec.PACKS[gid].get("found")),
                                 "name": arcade_vec.PACKS[gid]["name"] if gid else None}, p))
    for rel, gid in sorted(atari_library(rescan).items()):
        if (ROMS / rel).exists():
            found.append({"file": rel, "game_id": gid, "name": pretty(gid) + " (Atari 2600)", "platform": "atari",
                          "why": None, "library": True})
    # arcade games in sub-folders (a big ROM library)
    waiting = 0
    robot = {}                      # what the auto-pack robot made of each library game (files\autopack\results.csv)
    try:
        import csv
        with open(FILES / "autopack" / "results.csv", newline="") as f:
            robot = {r["set"]: r for r in csv.DictReader(f)}
    except Exception:
        pass
    if arcade_vec and arcade_vec.arcade_available():
        for p in sorted(ROMS.rglob("*.zip")):
            if p.parent == ROMS:
                continue
            st = p.stat()
            key = (str(p), st.st_size, st.st_mtime, len(arcade_vec.PACKS))
            if key not in _LIB_CACHE:                  # reading 375 zip indexes on every refresh is slow on a cloud drive
                _LIB_CACHE[key] = arcade_vec.identify_zip(p)
            gid = _LIB_CACHE[key]
            if gid:
                found.append(titled({"file": str(p.relative_to(ROMS)), "game_id": gid, "platform": "arcade", "why": None,
                                     "can_find": False, "found": bool(arcade_vec.PACKS[gid].get("found")),
                                     "name": arcade_vec.PACKS[gid]["name"], "library": True}, p))
            else:
                waiting += 1
                r = robot.get(p.stem.lower(), {})
                st = r.get("status", "")
                if st == "won't run":
                    why, can = "won't run in this kit (this copy of the game was made for a different emulator)", False
                elif st:
                    why, can = "the robot couldn't find the score by itself - press Find the score and show it", True
                else:
                    why, can = "not set up yet - press Find the score to teach the kit this game", True
                found.append(titled({"file": str(p.relative_to(ROMS)), "game_id": None, "platform": "arcade", "why": why,
                                     "can_find": can, "found": False, "library": True, "name": p.stem.upper()}, p))
    arcade_list = sorted(v["name"] for v in arcade_vec.PACKS.values()) if arcade_vec else []
    return {"yours": found, "supported": sorted({pretty(v): v for v in _KNOWN.values()}.items()),
            "arcade_supported": arcade_list, "library_waiting": waiting}


def tuned_for_this_pc(hw):
    """TUNE FOR THIS PC.bat's measurements (bench.py -> files/benchmark.json), if they were made on THIS machine: same
    processor, threads, memory and graphics card. On any other PC they're ignored, so the file can travel harmlessly."""
    try:
        b = json.loads((FILES / "benchmark.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    m = b.get("machine") or {}
    keys = ("cpu_name", "cpu_threads", "ram_gb", "gpu_name", "backend")
    return b if all(str(m.get(k)) == str(hw.get(k)) for k in keys) else None


def hardware(platform="atari", use_tuning=True):
    """Sniff the machine and size the job to it: GPU (proved working), CPU threads, RAM, graphics memory."""
    hw = {"os": f"{platform_mod.system()} {platform_mod.release()}", "cpu_threads": os.cpu_count() or 4,
          "cpu_name": platform_mod.processor() or platform_mod.machine(), "ram_gb": 8}
    try:
        import psutil
        vm = psutil.virtual_memory()
        hw["ram_gb"] = round(vm.total / 2**30)
        hw["ram_free_gb"] = round(vm.available / 2**30, 1)
        hw["cpu_cores"] = psutil.cpu_count(logical=False) or hw["cpu_threads"]
    except Exception:
        pass
    try:
        import torch
        import device
        hw["torch"] = torch.__version__
        dev, name, backend, problem = device.pick()
        hw["backend"] = backend
        if backend != "cpu":
            vram = 4
            if backend in ("cuda", "rocm"):
                vram = round(torch.cuda.get_device_properties(0).total_memory / 2**30)
            hw.update(gpu_name=name, vram_gb=vram, use_gpu=True)
        elif problem:
            hw["gpu_warning"] = problem + " - restart START HERE.bat to repair. Using the CPU for now."
    except Exception as e:
        hw["torch_error"] = repr(e)
    t, ram = hw["cpu_threads"], hw["ram_gb"]
    gpu, vram = hw.get("use_gpu", False), hw.get("vram_gb", 0)
    # memory we can use: what's free right now minus a safety margin (and never more than total minus 4 GB)
    spare_gb = max(1.0, min(ram - 4, hw.get("ram_free_gb", ram - 4) - 1.5))
    width = 2 if gpu and vram >= 16 and t >= 16 else 1
    if platform == "arcade":
        # one emulator process per game (~0.1 GB each). Two per CPU thread keeps every thread busy while the
        # brain thinks, and bigger batches use the GPU better. Memory and a ceiling of 64 cap it.
        n = int(max(4, min(t * 2, spare_gb / 0.1, 64)))
        emu = n
    elif gpu:
        # ALE runs several games per thread; lots of games at once = big batches that keep the GPU busy
        n = int(min(256, max(32, t * 4)))
        if vram and vram < 6:
            n = min(n, 64)
        n = n // 8 * 8
        emu = t
    else:
        n, emu = int(min(32, max(8, t))) // 4 * 4, t
    # The learning recipe is the same on every PC (trainer.py: lessons of 2,048 moves, 4 passes over each batch).
    # The PC only decides how FAST it goes: how many games at once, and the biggest bite of a lesson the graphics card
    # can hold at once (bf16 IMPALA ~ 4 MB per sample at width 2, 2 MB at width 1) - a lesson bigger than that is
    # worked through in bites, same lesson. (Before 17 Oct 2026 the bite WAS the lesson, so a bigger card meant fewer
    # lessons per move.)
    batch = n * 128
    lesson, passes = 2048, 4
    if gpu:
        bite = int(max(512, (vram - 1.5) * 1024 / (4 if width == 2 else 2)))
        bite = max(256, 1 << (bite.bit_length() - 1))       # round down to a power of two
    else:
        bite = 1024
    tuned = tuned_for_this_pc(hw) if use_tuning else None     # measured on this PC by TUNE FOR THIS PC.bat
    if tuned:
        got = tuned.get(platform) or {}
        if got.get("num_envs"):
            n = int(got["num_envs"])
            if platform == "arcade":
                n = int(max(4, min(n, spare_gb / 0.1)))          # never more than today's free memory allows
                emu = n
            batch = n * 128
        mb = got.get("micro_batch") or tuned.get("micro_batch")   # ALFRED 20 Oct: a platform can have its own bite size
        if mb:
            bite = int(mb)
        if got.get("trainer_cpus"):                     # ALFRED 21 Oct: the core split TUNE FOR THIS PC found best
            hw["trainer_cpus"] = int(got["trainer_cpus"])
    hw.update(use_gpu=gpu, num_envs=n, num_steps=128, minibatch_size=min(batch, lesson),
              minibatches=max(1, round(batch / lesson)), micro_batch=min(lesson, bite), epochs=passes,
              width=width, emu_threads=emu, platform=platform)
    what = "arcade machines" if platform == "arcade" else "games"
    if gpu:
        hw["plan"] = f"Your {hw['gpu_name']} will run the brain; your {t} CPU threads will run {n} {what} at once."
    else:
        lead = (hw["gpu_warning"] + " The CPU") if hw.get("gpu_warning") else (
            "No usable graphics card found, so the CPU")
        hw["plan"] = f"{lead} does everything: {n} {what} at once. It still learns - just slower. Leave it overnight."
    if tuned:
        hw["tuned"] = tuned.get("made", "")
        hw["plan"] += f" (Tuned for this PC on {hw['tuned'][:10]}.)"
    if ram < 8 or spare_gb < 3:
        hw["plan"] += (f" Memory is tight ({hw.get('ram_free_gb', '?')} GB free of {ram} GB)"
                       + (", so it runs fewer machines" if platform == "arcade" else "")
                       + " - close other programs (web browsers especially) for a faster AI.")
    return hw


def game_title(cfg):
    """The proper name of the game an AI plays."""
    gid = cfg.get("game_id") or ""
    if cfg.get("platform") == "arcade":
        try:
            import arcade_names
            t = cfg.get("rom_file") and arcade_names.name_of(ROMS / cfg["rom_file"])
            if t:
                return t
        except Exception:
            pass
        try:
            import arcade_vec
            arcade_vec.refresh_found()
            return re.sub(r" \(arcade.*\)$", "", arcade_vec.PACKS[gid]["name"])
        except Exception:
            return gid
    return pretty(gid) + " (Atari 2600)"


def projects():
    PROJECTS.mkdir(exist_ok=True)
    out = []
    for d in sorted(PROJECTS.iterdir()):
        pj = d / "project.json"
        if not pj.exists():
            continue
        try:
            cfg = json.loads(pj.read_text(encoding="utf-8-sig"))
        except Exception:
            continue                         # half-written or broken: skip it rather than break the page
        st = {}
        try:
            st = json.loads((d / "files" / "run" / "status.json").read_text())
            age = time.time() - (d / "files" / "run" / "status.json").stat().st_mtime
            st["live"] = st.get("state") in ("starting", "training", "stopping") and age < 180
            if st.get("state") == "crashed":
                st["msg"] = "Crashed: " + st.get("error", "see the training window")
        except Exception:
            pass
        out.append({"id": d.name, "name": cfg["game_name"], "game_title": game_title(cfg),
                    "reward_style": cfg.get("reward_style"),
                    "game_id": cfg.get("game_id"), "learn_style": cfg.get("learn_style", "practice"),
                    "status": st, "has_brain": (d / "files" / "run" / "latest.pt").exists()})
        if cfg.get("platform") == "arcade":      # Chip's question about buttons that never did anything (controls.py)
            try:
                import controls
                c = controls.chip_suggestion(d, st)
                if c:
                    out[-1]["chip"] = c
            except Exception:
                pass
        if not st.get("live"):                   # crash damage? the card offers Check & repair (repair.py)
            try:
                import repair
                dmg = repair.quick_check(d)
                if dmg:
                    out[-1]["damaged"] = dmg
            except Exception:
                pass
    return out


def create(p):
    gid = p["game_id"]
    name = (p.get("title") or pretty(gid)).strip()
    folder = PROJECTS / re.sub(r"[^\w \-]", "", name)
    (folder / "files" / "run").mkdir(parents=True, exist_ok=True)
    plat = p.get("platform", "atari")
    hw = hardware(plat)
    length = p.get("length", "overnight")
    total = {"taster": 5e6, "overnight": 100e6, "forever": 1e12, "more5": 5e6}.get(length, 100e6)
    cfg = {"game_id": gid, "game_name": name, "reward_style": p.get("reward_style", "log"),
           "life_penalty": float(p.get("life_penalty", 0)), "alive_bonus": float(p.get("alive_bonus", 0)),
           "curiosity": float(p.get("curiosity", 0.01)), "gamma": float(p.get("gamma", 0.99)),
           "sticky": bool(p.get("sticky", True)), "total_steps": total, "length": length,
           "learning_rate": float(p.get("learning_rate", 2.5e-4)), "explorer": float(p.get("explorer", 0)),
           "ignore_losses": bool(p.get("ignore_losses", False)), "go_explore": float(p.get("go_explore", 0)),
           "brain_size": p.get("brain_size", "auto"), "learn_from_me": float(p.get("learn_from_me", 1.0)), "learn_from_best": bool(p.get("learn_from_best", False)), "best_study_times": int(p.get("best_study_times", 1) or 1), "learn_style": p.get("learn_style", "practice"), "hardware": hw, "platform": plat, "rom_file": p.get("rom_file"), "created": time.strftime("%Y-%m-%d %H:%M")}
    cfg["lives_at_game_over"] = bool(p.get("lives_at_game_over", False))     # ALFRED 21 Oct 2026
    cfg["miss_price"] = float(p.get("miss_price", 0) or 0)                    # ALFRED 23 Oct 2026
    cfg["practice"] = float(p.get("practice", 0) or 0)                        # ALFRED 23 Oct 2026
    (folder / "project.json").write_text(json.dumps(cfg, indent=1))
    return {"ok": True, "id": folder.name}


# ------------------------------------------------------------------ dials, checkpoints, reset, graphs
DIALS = {"reward_style": str, "life_penalty": float, "curiosity": float, "gamma": float, "sticky": bool,
         "explorer": float, "ignore_losses": bool, "learning_rate": float, "length": str,
         "go_explore": float, "brain_size": str, "learn_from_me": float, "learn_style": str,
         "learn_from_best": bool, "best_study_times": int, "lives_at_game_over": bool,
         "miss_price": float, "practice": float}
LENGTHS = {"taster": 5e6, "overnight": 100e6, "forever": 1e12, "more5": 5e6}   # more5: set properly at each Train


def is_live(pid):
    return any(p["id"] == pid and p.get("status", {}).get("live") for p in projects())


def get_settings(pid):
    cfg = json.loads((PROJECTS / pid / "project.json").read_text())
    return {k: cfg.get(k, {"learn_from_me": 1.0, "learn_style": "practice", "learn_from_best": False, "best_study_times": 1, "miss_price": 0.0, "practice": 0.0}.get(k)) for k in DIALS} | {"game_name": cfg.get("game_name"), "platform": cfg.get("platform", "atari")}


def set_settings(body):
    pid = body["id"]
    pj = PROJECTS / pid / "project.json"
    cfg = json.loads(pj.read_text())
    for k, typ in DIALS.items():
        if k in body and body[k] is not None:
            cfg[k] = typ(body[k])
    cfg["total_steps"] = LENGTHS.get(cfg.get("length"), cfg.get("total_steps", 1e12))
    cfg["changed"] = time.strftime("%Y-%m-%d %H:%M")
    pj.write_text(json.dumps(cfg, indent=1))
    with open(PROJECTS / pid / "files" / "run" / "console.log", "a", encoding="utf-8") as f:
        f.write(f"SETTINGS CHANGED {cfg['changed']}: " + ", ".join(f"{k}={cfg.get(k)}" for k in DIALS) + "\n")
    if is_live(pid):
        return {"ok": True, "msg": "Saved. It's training right now - press Stop, then Train, for the new settings to kick in."}
    return {"ok": True, "msg": "Saved. They'll be used next time you press Train."}


def run_dir(pid):
    return PROJECTS / pid / "files" / "run"


def checkpoint(pid, name):
    run = run_dir(pid)
    if not (run / "latest.pt").exists():
        return {"ok": False, "msg": "Nothing to save yet - it saves its brain every few minutes of training."}
    label = re.sub(r"[^\w \-]", "", (name or "").strip())[:40] or "checkpoint"
    dst = PROJECTS / pid / "files" / "checkpoints" / f"{time.strftime('%Y-%m-%d %H%M')} - {label}"
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("latest.pt", "best.pt", "views.npy", "train_log.csv", "games.jsonl", "status.json"):
        if (run / f).exists():
            shutil.copy2(run / f, dst / f)
    shutil.copy2(PROJECTS / pid / "project.json", dst / "project.json")
    st = json.loads((run / "status.json").read_text()) if (run / "status.json").exists() else {}
    (dst / "info.json").write_text(json.dumps({"label": label, "made": time.strftime("%Y-%m-%d %H:%M"),
                                               "step": st.get("step"), "best_mean100": st.get("best_mean100")}))
    return {"ok": True, "msg": f"Checkpoint saved: {dst.name}"}


def checkpoints(pid):
    d = PROJECTS / pid / "files" / "checkpoints"
    out = []
    for c in sorted(d.iterdir(), reverse=True) if d.exists() else []:
        if not c.is_dir():
            continue           # Google Drive drops desktop.ini files in every folder
        try:
            info = json.loads((c / "info.json").read_text())
        except Exception:
            info = {}
        out.append({"name": c.name, **info})
    return out


def restore(pid, name):
    if is_live(pid):
        return {"ok": False, "msg": "Stop the training first."}
    src = PROJECTS / pid / "files" / "checkpoints" / name
    if not (src / "latest.pt").exists():
        return {"ok": False, "msg": "That checkpoint is missing its brain file."}
    archive(pid, "before restoring " + name[:30])
    run = run_dir(pid)
    run.mkdir(parents=True, exist_ok=True)
    for f in ("latest.pt", "best.pt", "views.npy", "train_log.csv", "games.jsonl"):
        if (src / f).exists():
            shutil.copy2(src / f, run / f)
    old = json.loads((PROJECTS / pid / "project.json").read_text())
    cp = json.loads((src / "project.json").read_text())
    cp["hardware"] = old.get("hardware", cp.get("hardware"))
    (PROJECTS / pid / "project.json").write_text(json.dumps(cp, indent=1))
    with open(run / "console.log", "a", encoding="utf-8") as f:
        f.write(f"RESTORED checkpoint '{name}' (brain + settings) {time.strftime('%Y-%m-%d %H:%M')}\n")
    return {"ok": True, "msg": f"Restored '{name}' - brain and settings. Press Train to carry on from there."}


def archive(pid, why):
    run = run_dir(pid)
    if run.exists() and any(run.iterdir()):
        dst = PROJECTS / pid / "files" / "old runs" / f"{time.strftime('%Y-%m-%d %H%M%S')} - {why}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run), str(dst))
        for extra in ("pop", "evolution_state.json", "evolution_log.jsonl"):     # an evolution family goes too
            src = PROJECTS / pid / "files" / extra
            if src.exists():
                shutil.move(str(src), str(dst / extra))
    run.mkdir(parents=True, exist_ok=True)


def _size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _pretty(n):
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024.0


def _ai_folder(pid):
    """The AI's folder - only ever a direct child of My AIs (never anything else on the disk)."""
    d = (PROJECTS / pid).resolve()
    if d.parent != PROJECTS.resolve() or not (d / "project.json").exists() or "/" in pid or "\\" in pid:
        raise ValueError("That isn't an AI in this kit.")
    return d


def delete_info(pid):
    d = _ai_folder(pid)
    files = d / "files"
    parts = {"brain + training history": _size(files / "run"), "checkpoints": _size(files / "checkpoints"),
             "old runs (from resets)": _size(files / "old runs")}
    total = _size(d)
    parts["everything else"] = max(0, total - sum(parts.values()))
    cfg = json.loads((d / "project.json").read_text())
    st = {}
    try:
        st = json.loads((files / "run" / "status.json").read_text())
    except Exception:
        pass
    return {"ok": True, "id": pid, "name": cfg.get("game_name", pid), "total": total, "pretty": _pretty(total),
            "parts": [[k, _pretty(v)] for k, v in parts.items() if v > 0], "live": is_live(pid),
            "moves": st.get("step"), "best": st.get("best_mean100")}


def delete_ai(pid):
    d = _ai_folder(pid)
    if is_live(pid):
        return {"ok": False, "msg": "It's training right now - press Stop first, then delete it."}
    freed = _size(d)

    def force(func, path, _exc):             # read-only files (Google Drive / Windows) - make writable and retry
        try:
            os.chmod(path, 0o666)
            func(path)
        except Exception:
            pass
    shutil.rmtree(d, onerror=force)
    if d.exists():
        left = _size(d)
        if left > 64 * 1024:
            return {"ok": False, "msg": "Some of its files are still in use (is its Watch window open?). Close it and "
                                        f"try again. Freed {_pretty(freed - left)} so far."}
        shutil.rmtree(d, ignore_errors=True)  # only tiny leftovers (e.g. Google Drive's desktop.ini) - tidy up
    return {"ok": True, "msg": f"Deleted '{pid}' - freed {_pretty(freed)}."}


# ------------------------------------------------------------------ LIBRARY ROBOT (optional add-on)
ROBOT = FILES / "autopack" / "autopack.py"
_ROBOT_SEEN = {"found": -1}


def _alive(pid):
    try:
        import psutil
        return bool(pid) and psutil.pid_exists(int(pid))
    except Exception:
        return False


def _robot_installed():
    """The add-on is files\\autopack\\autopack.py. Windows' "Extract All" often unzips it into a folder of its own
    inside the kit instead - if so, copy it into place, so it just works."""
    if ROBOT.exists():
        return True
    skip = {"files", "My AIs", "ROMs - put your games here", "runtime"}
    try:
        for d in ROOT.iterdir():
            if not d.is_dir() or d.name in skip:
                continue
            for p in [d / "files" / "autopack" / "autopack.py", *d.glob("*/files/autopack/autopack.py")]:
                if p.is_file():
                    ROBOT.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, ROBOT)
                    return True
    except OSError:
        pass
    return False


def robot_status():
    if not _robot_installed():
        return {"installed": False}
    st = {}
    try:
        st = json.loads((FILES / "autopack" / "robot_status.json").read_text())
    except Exception:
        pass
    running = st.get("state") == "working" and _alive(st.get("pid"))
    if st.get("state") == "working" and not running:
        st["state"] = "stopped"
    if st.get("found", 0) != _ROBOT_SEEN["found"]:          # new packs: let the game lists pick them up
        _ROBOT_SEEN["found"] = st.get("found", 0)
        try:
            import arcade_vec
            arcade_vec.refresh_found()
        except Exception:
            pass
        _LIB_CACHE.clear()
    return {"installed": True, "running": running, **st}


def robot_start():
    if not _robot_installed():
        return {"ok": False, "msg": "The Library Robot add-on isn't installed."}
    if robot_status().get("running"):
        return {"ok": False, "msg": "The robot is already working."}
    log = open(FILES / "autopack" / "robot.log", "a", encoding="utf-8")
    subprocess.Popen([PY, "-B", "-u", str(ROBOT), "--dir", str(ROMS), "--recursive"], cwd=str(FILES),
                     creationflags=NOWIN, stdout=log, stderr=subprocess.STDOUT)
    return {"ok": True, "msg": "The robot has started. It works through your arcade games one at a time - about "
                               "5 minutes each. You can close this page; it carries on."}


def robot_stop():
    (FILES / "autopack" / "STOP").write_text("stop")
    return {"ok": True, "msg": "The robot will stop after the game it's on now."}


def reset(pid):
    if is_live(pid):
        return {"ok": False, "msg": "Stop the training first."}
    archive(pid, "reset")
    return {"ok": True, "msg": "Fresh start: it will learn from scratch next time you press Train. "
                               "(The old run is kept in the AI's files\\old runs folder.)"}


def history(pid):
    """Rows for the little graphs on the AI's card."""
    import csv as _csv
    try:
        with open(run_dir(pid) / "train_log.csv", newline="") as f:
            rows = list(_csv.DictReader(f))
    except Exception:
        return {"rows": []}
    stride = max(1, len(rows) // 300)
    keep = rows[::stride] + ([rows[-1]] if rows and (len(rows) - 1) % stride else [])

    def num(r, k):
        try:
            v = float(r.get(k) or "nan")
            return v if v == v else None
        except ValueError:
            return None
    return {"rows": [{k: num(r, k) for k in ("step", "score_mean100", "best_game", "views", "novelty", "entropy", "sps", "places", "map_best", "accuracy", "shots_per_game")}
                     for r in keep]}


def _tail(path, n):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 200 * n))
            return f.read().decode("utf-8", "replace").splitlines()[-n:]
    except Exception:
        return []


def live(pid):
    """What a training AI is doing right now - for the live box on its card (the page asks every 5 seconds)."""
    if not pid or "/" in pid or "\\" in pid or ".." in pid:
        return {}
    run = run_dir(pid)
    try:
        st = json.loads((run / "status.json").read_text())
    except Exception:
        st = {}
    last = {}
    try:
        with open(run / "train_log.csv", newline="") as f:
            head = f.readline().strip().split(",")
            first = dict(zip(head, f.readline().strip().split(",")))
        row = _tail(run / "train_log.csv", 1)
        if row and not row[0].startswith("time"):
            last = dict(zip(head, row[0].split(",")))
    except Exception:
        pass
    n_act = None
    try:
        cfg = json.loads((PROJECTS / pid / "project.json").read_text(encoding="utf-8-sig"))
        n_act = cfg.get("n_actions")
    except Exception:
        cfg = {}
    con = _tail(run / "console.log", 40)
    win = _tail(run / "window.log", 40)
    if not any("Error" in l or "Traceback" in l for l in win):
        win = []
    try:
        start_ent = float(first.get("entropy"))
    except Exception:
        start_ent = None
    return {"status": st, "last": last, "console": con, "window": win, "n_actions": n_act, "start_entropy": start_ent,
            "target": st.get("target_steps") or cfg.get("total_steps")}


TRAIN_LOCK = threading.Lock()


def train(pid):
    with TRAIN_LOCK:                 # two clicks at once can't both get through
        return _train(pid)


def _tuning_now(lock):
    """ALFRED 21 Oct 2026: is TUNE FOR THIS PC really running? A lock left by a tune that died (a crash, a
    bluescreen) names a process that has gone, so it doesn't count."""
    if not lock.exists() or time.time() - lock.stat().st_mtime >= 3 * 3600:
        return False
    try:
        m = re.search(r"pid=(\d+)", lock.read_text(encoding="utf-8", errors="replace"))
        if m:
            import psutil                 # ALFRED 21 Oct 2026 review: after a reboot that number can belong to any
            try:                          # program (Windows re-uses them) - it only counts if it really is the tune
                p = psutil.Process(int(m.group(1)))
                try:
                    return "bench" in " ".join(p.cmdline()).lower()
                except psutil.AccessDenied:
                    return p.create_time() <= lock.stat().st_mtime + 5    # started before the lock was written
            except Exception:
                return False
    except Exception:
        pass
    return True


def _train(pid):
    d = PROJECTS / pid
    if any(p.get("status", {}).get("live") for p in projects()):
        return {"ok": False, "msg": "Another AI is already training. One at a time - stop it first."}
    tl = FILES / "TUNING"
    if _tuning_now(tl):
        return {"ok": False, "msg": "Tune for this PC is running - wait until its window says it's done (about 15 minutes)."}
    (d / "files" / "run" / "STOP").unlink(missing_ok=True)
    # Say "starting" straight away: the trainer takes a minute to load before it reports in, and until now that
    # gap let a second click start a second trainer on the same AI.
    sf = d / "files" / "run" / "status.json"
    try:
        st = json.loads(sf.read_text())
    except Exception:
        st = {}
    st.update(state="starting", updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    sf.write_text(json.dumps(_no_nan(st)))
    cfg = json.loads((d / "project.json").read_text())
    if cfg.get("length") == "more5":         # "5M more moves, then stop": counted from wherever it is now, every Train
        try:
            now = int(st.get("step") or 0)
        except Exception:
            now = 0
        cfg["total_steps"] = float(now + 5_000_000)
        (d / "project.json").write_text(json.dumps(cfg, indent=1))
    script = "evolution.py" if cfg.get("learn_style") == "evolution" else "trainer.py"
    (d / "files" / "run" / "FROZE").unlink(missing_ok=True)
    # no black window: it runs quietly and the page shows what it's doing (the old window's text is in window.log)
    out = open(d / "files" / "run" / "window.log", "w", encoding="utf-8", errors="replace")
    # normal priority (19 Oct 2026): below-normal made Windows 11 put all of it on the efficiency cores - the performance
    # cores sat idle and training ran about a third slower
    subprocess.Popen([PY, "-u", str(FILES / "keepgoing.py"), script, str(d)], cwd=str(FILES), creationflags=NOWIN,
                     stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
    if script == "evolution.py":
        return {"ok": True, "msg": "Evolution window opened: a family of brains takes turns practising. You can close "
                                   "this page - it carries on."}
    return {"ok": True, "msg": "Training started. Tip: close as many other programs as you can while it trains - it runs "
                               "faster and crashes are much less likely. You can close this page - training carries on."}


def tune():
    """TUNE FOR THIS PC (bench.py, via its .bat) in its own window, so you can watch it work. Not while training."""
    if any(p.get("status", {}).get("live") for p in projects()):
        return {"ok": False, "msg": "Stop the training first - tuning needs the whole PC (about 15 minutes)."}
    lock = FILES / "TUNING"
    if _tuning_now(lock):
        return {"ok": False, "msg": "It's already tuning - see its window."}
    bat = ROOT / "TUNE FOR THIS PC.bat"
    if not bat.exists():
        return {"ok": False, "msg": "TUNE FOR THIS PC.bat is missing from the kit folder."}
    subprocess.Popen(["cmd", "/c", str(bat)], cwd=str(ROOT), creationflags=NEWCON)
    return {"ok": True, "msg": "Tuning window opened - about 15 minutes. Leave training alone until it says it's done."}


def stop(pid):
    (PROJECTS / pid / "files" / "run" / "STOP").write_text("stop")
    return {"ok": True, "msg": "Stopping after this round - it saves first."}


def watch(pid, best=False):
    log = open(PROJECTS / pid / "files" / "run" / "watch.log", "w")   # if the window never appears, the reason is here
    subprocess.Popen([PY, str(FILES / "watch.py"), str(PROJECTS / pid)] + (["best"] if best else []), cwd=str(FILES),
                     creationflags=NOWIN, stdout=log, stderr=subprocess.STDOUT)
    if best:
        msg = "Opening its best game... (B in the window = the next best, then back to live)"
        try:        # ALFRED 23 Oct 2026: say so when its record game was never kept (see best_games.hof_best)
            import best_games
            kept = best_games.hof_best(PROJECTS / pid)
            with open(run_dir(pid) / "train_log.csv", newline="") as f:
                head = f.readline().strip().split(",")
            top = float(dict(zip(head, _tail(run_dir(pid) / "train_log.csv", 1)[0].split(","))).get("best_game") or 0)
            if kept and top > kept + 0.5:
                msg = (f"Opening the best game it kept: {kept:,.0f} points. Its record {top:,.0f} was too long to keep "
                       f"(fixed 23 Oct) - its next records will be kept. (B = the next best, then back to live)")
        except Exception:
            pass
        return {"ok": True, "msg": msg}
    return {"ok": True, "msg": "Opening the game window..."}


def find_score(name):
    """Open the Find-the-score window for one arcade game in the ROMs folder."""
    rom = (ROMS / name).resolve()
    if ROMS.resolve() not in rom.parents or not rom.exists():
        return {"ok": False, "msg": "That game file isn't in the ROMs folder any more."}
    (FILES / "logs").mkdir(exist_ok=True)
    log = open(FILES / "logs" / "find-the-score.log", "w")
    subprocess.Popen([PY, str(FILES / "score_finder.py"), str(rom)], cwd=str(FILES), creationflags=NOWIN,
                     stdout=log, stderr=subprocess.STDOUT)
    return {"ok": True, "msg": "Opening the Find-the-score window - follow the instructions under the game."}


def teach(src, dst):
    """One AI plays, another saves its moves as lessons (teacher.py, on the CPU)."""
    a, b = PROJECTS / Path(src).name, PROJECTS / Path(dst).name
    if not ((a / "files" / "run" / "latest.pt").exists() or (a / "files" / "run" / "best.pt").exists()):
        return {"ok": False, "msg": "The teacher hasn't learned anything yet - train it first."}
    subprocess.Popen([PYCON, "-u", str(FILES / "teacher.py"), str(a), str(b)], cwd=str(FILES), creationflags=NEWCON)
    return {"ok": True, "msg": f"'{a.name}' is playing in a new window and '{b.name}' is taking notes. When it says "
                               f"Done, press Train on '{b.name}' (Stop then Train if it's already training)."}


def report():
    with socket.socket() as s:
        busy = s.connect_ex(("127.0.0.1", REPORT_PORT)) == 0
    if not busy:
        env = dict(os.environ, PORTABLE_PROJECTS=str(PROJECTS))
        subprocess.Popen([PY, str(FILES / "dashboard_rl.py"), "--port", str(REPORT_PORT)], cwd=str(FILES), env=env,
                         creationflags=NOWIN)
        time.sleep(1.5)
    return {"ok": True, "url": f"http://127.0.0.1:{REPORT_PORT}/"}


# ------------------------------------------------------------------ CLOSE THE KIT (stops everything, politely)
SHUTDOWN = {"state": "", "waiting": []}


def _learner_procs():
    """Every training-type process the kit has running (trainers, evolution, teachers), whatever the page thinks."""
    try:
        import psutil
    except Exception:
        return []
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
        except Exception:
            continue
        if any(f"{s}.py" in cmd for s in ("trainer", "evolution", "goexplore", "teacher", "keepgoing")) and \
                str(PROJECTS).lower() in cmd.lower():
            out.append((p, cmd))
    return out


def stop_all():
    """STOP EVERYTHING: every AI that's training is told to stop and save. Anything still going after 3 minutes
    (a frozen trainer can't hear the stop note) is closed by force - it keeps what it last saved."""
    told = []
    for d in PROJECTS.iterdir() if PROJECTS.exists() else []:
        run = d / "files" / "run"
        if not run.exists():
            continue
        busy = is_live(d.name) or any(str(d).lower() in cmd.lower() for _p, cmd in _learner_procs())
        if busy:
            (run / "STOP").write_text("stop")
            told.append(d.name)

    def finish():
        t0 = time.time()
        while time.time() - t0 < 180 and _learner_procs():
            time.sleep(3)
        for p, cmd in _learner_procs():
            try:
                for c in p.children(recursive=True):
                    c.kill()
                p.kill()
            except Exception:
                pass
        for name in told:
            sf = PROJECTS / name / "files" / "run" / "status.json"
            try:
                st = json.loads(sf.read_text())
                if st.get("state") in ("starting", "training", "stopping", "restarting"):
                    st.update(state="stopped", updated=time.strftime("%Y-%m-%d %H:%M:%S"))
                    sf.write_text(json.dumps(_no_nan(st)))
            except Exception:
                pass
            (PROJECTS / name / "files" / "run" / "STOP").unlink(missing_ok=True)
    threading.Thread(target=finish, daemon=True).start()
    if not told:
        return {"ok": True, "msg": "Nothing is training."}
    return {"ok": True, "msg": f"Stopping {', '.join(told)} - each saves first (a minute or so). Anything stuck is "
                               "closed after 3 minutes."}


def shutdown():
    """Every AI that's training is told to stop (each one saves first), the Library Robot stops after its current
    game, the report card closes, and once nothing is training any more this little server closes too."""
    if SHUTDOWN["state"]:
        return {"ok": True, **SHUTDOWN}
    live = [x["id"] for x in projects() if x.get("status", {}).get("live")]
    for pid in live:
        stop(pid)
    if ROBOT.exists() and robot_status().get("running"):
        robot_stop()
    SHUTDOWN.update(state="stopping" if live else "closing", waiting=live)

    def finish():
        t0 = time.time()
        while SHUTDOWN["waiting"] and time.time() - t0 < 20 * 60:      # a save never takes this long
            time.sleep(3)
            try:
                SHUTDOWN["waiting"] = [x["id"] for x in projects() if x.get("status", {}).get("live")]
            except Exception:
                pass
        SHUTDOWN["state"] = "closing"
        try:                                                          # the report card, if it's open
            import urllib.request
            urllib.request.urlopen(f"http://127.0.0.1:{REPORT_PORT}/api/quit", timeout=3).read()
        except Exception:
            pass
        time.sleep(4)                                                 # let the page see "closing"
        os._exit(0)
    threading.Thread(target=finish, daemon=True).start()
    return {"ok": True, **SHUTDOWN}


def _no_nan(x):
    """JSON can't carry NaN or infinity: browsers refuse the whole reply. Send null instead."""
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return None
    if isinstance(x, dict):
        return {k: _no_nan(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_no_nan(v) for v in x]
    return x


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, obj, ctype="application/json"):
        if not isinstance(obj, str):
            obj = json.dumps(_no_nan(obj))     # NaN (e.g. an average of no games yet) breaks the page's JSON
        b = obj.encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                return self.send((FILES / "wizard.html").read_text(encoding="utf-8"), "text/html")
            if u.path == "/api/hardware":
                return self.send(hardware(q.get("platform", "atari")))
            if u.path == "/api/games":
                return self.send(games())
            if u.path == "/api/projects":
                return self.send(projects())
            if u.path == "/api/train":
                return self.send(train(q["id"]))
            if u.path == "/api/stopall":
                return self.send(stop_all())
            if u.path == "/api/stop":
                return self.send(stop(q["id"]))
            if u.path == "/api/watch":
                return self.send(watch(q["id"], q.get("best") == "1"))
            if u.path == "/api/teach":
                return self.send(teach(q["from"], q["to"]))
            if u.path == "/api/findscore":
                return self.send(find_score(q["file"]))
            if u.path == "/api/report":
                return self.send(report())
            if u.path == "/api/settings":
                return self.send(get_settings(q["id"]))
            if u.path == "/api/checkpoint":
                return self.send(checkpoint(q["id"], q.get("name", "")))
            if u.path == "/api/checkpoints":
                return self.send(checkpoints(q["id"]))
            if u.path == "/api/restore":
                return self.send(restore(q["id"], q["cp"]))
            if u.path == "/api/robot":
                return self.send(robot_status())
            if u.path == "/api/robot_start":
                return self.send(robot_start())
            if u.path == "/api/robot_stop":
                return self.send(robot_stop())
            if u.path == "/api/delete_info":
                return self.send(delete_info(q["id"]))
            if u.path == "/api/delete":
                return self.send(delete_ai(q["id"]))
            if u.path == "/api/reset":
                return self.send(reset(q["id"]))
            if u.path == "/api/quit":          # "Close the kit": stop training politely, then the robot, report card, server
                return self.send(shutdown())
            if u.path == "/api/quit_status":
                return self.send({"ok": True, **SHUTDOWN})
            if u.path == "/api/openkit":       # the kit's own folder (My AIs, ROMs, START HERE...)
                os.startfile(str(ROOT)) if hasattr(os, "startfile") else None
                return self.send({"ok": True, "msg": "Opening the kit's folder..."})
            if u.path == "/api/live":
                return self.send(live(q.get("id", "")))
            if u.path == "/api/history":
                return self.send(history(q["id"]))
            if u.path == "/api/repair":            # Check & repair after a crash (repair.py)
                import repair
                return self.send(repair.repair(PROJECTS / q["id"], fix=q.get("do", "fix") != "check"))
            if u.path == "/api/buttons":           # Settings > Buttons: what each button did, and the user's decision
                import controls
                a, d = q.get("do", ""), PROJECTS / q["id"]
                if a == "drop":
                    return self.send(controls.drop_buttons(d))
                if a == "restore":
                    return self.send(controls.give_back(d))
                if a in ("keep", "check"):
                    return self.send(controls.keep_buttons(d, a == "keep"))
                return self.send(controls.button_report(d))
            if u.path == "/api/tune":              # TUNE FOR THIS PC: times the real trainer here, keeps the fastest
                return self.send(tune())
            if u.path == "/api/diagnose":
                r = subprocess.run([PY, "-B", str(FILES / "diagnose.py")], cwd=str(FILES), capture_output=True,
                                   text=True, timeout=300, creationflags=NOWIN)
                names = re.findall(r"Made: (.+\.txt)", r.stdout)
                if names:
                    os.startfile(str(ROOT)) if hasattr(os, "startfile") else None
                    return self.send({"ok": True, "msg": f"Made '{names[0]}' next to START HERE.bat - please send it to Jonny."})
                return self.send({"ok": False, "msg": "Couldn't make the report: " + (r.stderr or r.stdout)[-300:]})
            if u.path.startswith("/help/"):        # the page's pictures (Chip, the coffee) and fonts, from files\help
                kinds = {".png": "image/png", ".webp": "image/webp", ".woff2": "font/woff2"}
                base = (FILES / "help").resolve()
                f = (base / unquote(u.path[len("/help/"):])).resolve()
                if f.suffix.lower() in kinds and f.is_file() and base in f.parents:
                    b = f.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", kinds[f.suffix.lower()])
                    self.send_header("Cache-Control", "max-age=3600")
                    self.end_headers()
                    self.wfile.write(b)
                    return
            if u.path == "/api/openvideo":         # the help video, in the computer's own video player
                vids = sorted((FILES / "help").glob("*.mp4"))
                if vids and hasattr(os, "startfile"):
                    os.startfile(str(vids[0]))
                    return self.send({"ok": True, "msg": "Opening the help video..."})
                link = FILES / "help" / "video-link.txt"     # the online copy (keeps the download small)
                url = link.read_text(encoding="utf-8").strip() if link.exists() else ""
                if url.startswith("http"):
                    webbrowser.open(url)
                    return self.send({"ok": True, "msg": "Opening the help video in your browser..."})
                return self.send({"ok": False, "msg": "The help video isn't included in this download."})
            if u.path == "/api/clutter":           # Free up this PC (clutter.py): which background apps on the list are running
                import clutter
                return self.send(clutter.scan())
            if u.path == "/api/clutter_list":      # ...and open that list in Notepad so people can add their own
                import clutter
                return self.send(clutter.open_list())
            if u.path == "/api/scan":              # Scan the ROMs folder: forget what we knew and look again
                _LIB_CACHE.clear()
                return self.send(games(rescan=True))
            if u.path == "/api/openroms":
                ROMS.mkdir(exist_ok=True)
                os.startfile(str(ROMS)) if hasattr(os, "startfile") else None
                return self.send({"ok": True})
        except Exception as e:
            import traceback
            try:
                import diagnose
                diagnose.log_error(f"control panel {u.path}", traceback.format_exc())
            except Exception:
                pass
            return self.send({"ok": False, "msg": repr(e)})
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        if n > 2_100_000:
            return self.send({"ok": False, "msg": "That request is too large (2 MB maximum)."})
        body = json.loads(self.rfile.read(n) or b"{}")
        try:
            path = urlparse(self.path).path
            if path == "/api/recipe/preview":
                return self.send(recipe_cards.preview(PROJECTS, ROMS, body["id"], body.get("notes", ""),
                                                       bool(body.get("include_specs"))))
            if path == "/api/recipe/export":
                return self.send(recipe_cards.export(PROJECTS, ROMS, body["id"], body["json"],
                                                      bool(body.get("share"))))
            if path == "/api/recipe/import_preview":
                return self.send(recipe_cards.import_preview(PROJECTS, ROMS, body["id"], body["recipe"],
                                                              bool(body.get("override_fingerprint"))))
            if path == "/api/recipe/import_apply":
                return self.send(recipe_cards.import_apply(PROJECTS, ROMS, body["id"], body["recipe"],
                                                            body.get("keys", []),
                                                            bool(body.get("override_fingerprint"))))
            if path == "/api/recipe/undo":
                return self.send(recipe_cards.undo(PROJECTS, body["id"], body["backup"]))
            if path == "/api/create":
                return self.send(create(body))
            if path == "/api/settings":
                return self.send(set_settings(body))
            if path == "/api/clutter":   # close the ticked apps; remember the unticked ones as "keep"
                import clutter
                running = set(body.get("running", []))
                clutter.set_keep((clutter.load_keep() - running) | set(body.get("keep", [])))
                return self.send(clutter.close(body.get("names", [])))
        except Exception as e:
            return self.send({"ok": False, "msg": repr(e)})
        self.send_response(404)
        self.end_headers()


def main():
    url = f"http://127.0.0.1:{PORT}/"
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            webbrowser.open(url)
            return
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print(f"Arcade Apprentice is open at {url}\nKeep this window open while you use the page. Training keeps going if you close it.")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
