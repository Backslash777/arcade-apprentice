"""Arcade Apprentice - CHECK & REPAIR an AI after a crash or power cut (Settings > Check & repair, the card's Repair
button, or:  python repair.py "<AI folder name>" [--check]).

When Windows crashes while an AI is training, the files it was writing at that moment can be left half-written: a
brain save that won't open, or blank padding (zero bytes) in its logs. The repair is always the same:
  1. Brain: if its latest save (latest.pt) won't open, it's rebuilt from the newest copy that does - its best brain
     (best.pt), else a checkpoint. A few minutes of progress are lost, and its learning momentum starts afresh (it
     builds that back up within minutes). Games it played after that copy are set aside so the graphs don't jump back.
  2. Logs (training log, game list, diary): blank padding and broken lines are dropped; every good line is kept.
  3. Small files that won't open are reset: its status, its button-check evidence, its exploring map.
  4. Recordings (yours, its best games) that won't open are moved aside - never deleted.
Everything is copied to files\\backup <date> - repair before it's changed. It won't run while the AI is training.

Signed: Claude, 17 Oct 2026 (after the 11:18 crash; Jonny: "a repair button, if the routine is always the same").
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import sys
import time
import zipfile
from pathlib import Path

FILES = Path(__file__).resolve().parent
if str(FILES) not in sys.path:
    sys.path.insert(0, str(FILES))
PROJECTS = FILES.parent / "My AIs"


# ------------------------------------------------------------------------------------------------ quick look (the page)
def quick_check(proj):
    """Cheap signs of crash damage for the page (no brain loading): a status file that won't read, a brain save that
    isn't a whole file, or logs ending in blank padding. Returns a list of short names (empty = looks fine)."""
    run = Path(proj) / "files" / "run"
    bad = []
    st = run / "status.json"
    try:
        if st.exists():
            json.loads(st.read_text())
    except Exception:
        bad.append("status")
    lp = run / "latest.pt"
    try:
        if (lp.exists() and not zipfile.is_zipfile(lp)) or (not lp.exists() and (run / "latest.pt.tmp").exists()):
            bad.append("brain")
    except Exception:
        bad.append("brain")
    for name in ("train_log.csv", "games.jsonl"):
        p = run / name
        try:
            if p.exists() and p.stat().st_size:
                with open(p, "rb") as f:
                    f.seek(-1, 2)
                    if f.read(1) == b"\x00":
                        bad.append(name)
        except Exception:
            pass
    return bad


# ------------------------------------------------------------------------------------------------ the full check
def _opens(p):
    """A brain file torch can actually open."""
    try:
        if not zipfile.is_zipfile(p):
            return False
        import torch
        st = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        return isinstance(st, dict) and "agent" in st
    except Exception:
        return False


def _training(proj):
    try:
        s = json.loads((Path(proj) / "files" / "run" / "status.json").read_text())
        if s.get("state") in ("starting", "training", "stopping", "restarting") and s.get("pid"):
            import psutil
            return psutil.pid_exists(int(s["pid"]))
    except Exception:
        pass
    return False


def _clean_log(p, ok, step_of=None, after=None):
    """Drop NUL bytes and broken lines (ok(i, line) says which are whole); optionally set aside lines past a step.
    Returns (new bytes, broken, set aside)."""
    raw = p.read_bytes()
    nl = "\r\n" if b"\r\n" in raw[:65536] else "\n"
    good, broken, later = [], 0, 0
    for i, ln in enumerate(raw.replace(b"\x00", b"").decode("utf-8", "replace").splitlines()):
        if not ln.strip():
            continue
        if not ok(i, ln):
            broken += 1
            continue
        if after is not None and step_of is not None:
            s = step_of(ln)
            if s is not None and s > after:
                later += 1
                continue
        good.append(ln)
    return ((nl.join(good) + nl).encode("utf-8") if good else b""), broken, later


def repair(proj, fix=True, say=print):
    """Check an AI's files and (with fix=True) repair what a crash broke. Returns {"ok", "found", "fixed", "msg"}."""
    proj = Path(proj)
    run = proj / "files" / "run"
    name = proj.name
    if not (proj / "project.json").exists():
        return {"ok": False, "msg": f"There's no AI called '{name}'."}
    if fix and _training(proj):
        return {"ok": False, "msg": "Stop it first - it can't be repaired while it's training."}
    bk = proj / "files" / f"backup {time.strftime('%Y-%m-%d %H%M')} - repair"
    found, fixed, unfixable = [], [], []

    def keep_copy(p):
        dst = bk / p.relative_to(proj)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(p, dst)

    def set_aside(p):
        dst = bk / p.relative_to(proj)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dst))

    # 1. the brain
    lp, bp, tmp = run / "latest.pt", run / "best.pt", run / "latest.pt.tmp"
    rebuilt = None
    if not lp.exists() and tmp.exists() and _opens(tmp):
        found.append("its last brain save was never finished being put in place")
        if fix:
            os.replace(tmp, lp)
            fixed.append("finished putting its last brain save in place")
    if lp.exists() and not _opens(lp):
        found.append("its latest brain save won't open")
        src = bp if bp.exists() and _opens(bp) else None
        if src is None:
            cps = sorted((proj / "files" / "checkpoints").glob("*/latest.pt"), key=lambda p: p.stat().st_mtime,
                         reverse=True) if (proj / "files" / "checkpoints").exists() else []
            src = next((c for c in cps if _opens(c)), None)
        if src is None:
            unfixable.append("there's no good copy of its brain to rebuild from - Reset training starts it afresh")
        elif fix:
            import torch
            import demos
            keep_copy(lp)
            b = torch.load(src, map_location="cpu", weights_only=False)
            rebuilt = int(b["global_step"])
            it = 0
            try:
                for ln in reversed((run / "train_log.csv").read_bytes().replace(b"\x00", b"").decode("utf-8", "replace")
                                   .splitlines()):
                    parts = ln.split(",")
                    if len(parts) > 2 and parts[2].isdigit() and int(parts[2]) <= rebuilt and parts[1].isdigit():
                        it = int(parts[1])
                        break
            except Exception:
                pass
            st = {"agent": b["agent"], "opt": b.get("opt"), "global_step": rebuilt, "iteration": it,
                  "best": float(b.get("best", b.get("score", -1e9))), "n_actions": b["n_actions"],
                  "width": b.get("width", 1), "game_id": b.get("game_id"), "explorer": b.get("explorer"),
                  "demo_frames": max(int(b.get("demo_frames", 0) or 0), demos.count_moves(proj)),
                  "demo_step": int(b.get("demo_step", 0) or 0)}
            for k in ("controls",):
                if k in b:
                    st[k] = b[k]
            t = run / "latest.pt.tmp"
            with open(t, "wb") as f:
                torch.save(st, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(t, lp)
            what = "its best brain" if src == bp else f"checkpoint '{src.parent.name}'"
            fixed.append(f"rebuilt its brain from {what} ({rebuilt:,} moves)")
    if bp.exists() and not _opens(bp):
        found.append("its best-brain save won't open")
        if fix:
            set_aside(bp)
            fixed.append("set its damaged best-brain save aside (a new one is saved when it next beats its best)")
    cp = proj / "files" / "checkpoints"
    for c in sorted(cp.glob("*/latest.pt")) if cp.exists() else []:
        if not _opens(c):
            unfixable.append(f"checkpoint '{c.parent.name}' won't open - don't restore that one")

    # 2. the logs
    def csv_ok(head):
        cols = head.count(",")
        return lambda i, ln: i == 0 or (ln.count(",") == cols and len(ln.split(",")[0]) == 19
                                        and ln.split(",")[2].isdigit())

    def json_ok(i, ln):
        try:
            return "score" in json.loads(ln)
        except Exception:
            return False

    def csv_step(ln):
        p = ln.split(",")
        return int(p[2]) if len(p) > 2 and p[2].isdigit() else None

    def json_step(ln):
        try:
            return int(json.loads(ln).get("step", 0))
        except Exception:
            return None

    logs = []
    if (run / "train_log.csv").exists():
        head = (run / "train_log.csv").read_bytes().replace(b"\x00", b"").decode("utf-8", "replace").split("\n", 1)[0]
        logs.append(("its training log", run / "train_log.csv", csv_ok(head), csv_step))
    logs.append(("its list of games", run / "games.jsonl", json_ok, json_step))
    logs.append(("its training diary", run / "console.log", lambda i, ln: True, None))
    for label, p, ok, step_of in logs:
        if not p.exists():
            continue
        raw = p.read_bytes()
        new, broken, later = _clean_log(p, ok, step_of, rebuilt)
        if new != raw and (b"\x00" in raw or broken or later):
            bits = ([f"{broken:,} broken line{'s' if broken != 1 else ''}"] if broken else []) + \
                   (["crash padding"] if b"\x00" in raw else [])
            found.append(f"{label} has {' and '.join(bits) or 'lines from after the rebuilt brain'}")
            if fix:
                keep_copy(p)
                p.write_bytes(new)
                fixed.append(f"cleaned {label}" + (f" ({later:,} lines from after the rebuilt brain set aside)" if later else ""))

    # 3. small files
    stp = run / "status.json"
    bad_status = False
    if stp.exists():
        try:
            json.loads(stp.read_text())
        except Exception:
            bad_status = True
            found.append("its status file won't read")
    if fix and (bad_status or rebuilt is not None):
        if stp.exists():
            keep_copy(stp)
        step = rebuilt
        if step is None:
            try:
                last = (run / "train_log.csv").read_text(encoding="utf-8").strip().splitlines()[-1].split(",")
                step = int(last[2])
            except Exception:
                step = 0
        stp.write_text(json.dumps({"state": "stopped", "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "step": step,
                                   "note": "repaired after a crash"}))
        fixed.append("reset its status")
    bc = run / "button_checks.json"
    if bc.exists():
        try:
            json.loads(bc.read_text())
        except Exception:
            found.append("its button-check evidence won't read")
            if fix:
                set_aside(bc)
                fixed.append("reset its button-check evidence (it checks again as it trains)")
    vw = run / "views.npy"
    if vw.exists():
        try:
            import numpy as np
            np.load(vw)
        except Exception:
            found.append("its count of screens seen won't read")
            if fix:
                set_aside(vw)
                fixed.append("reset its count of screens seen")
    ar = run / "archive.pkl"
    if ar.exists():
        try:
            with open(ar, "rb") as f:
                pickle.load(f)
        except Exception:
            found.append("its exploring map won't read")
            if fix:
                set_aside(ar)
                fixed.append("set its damaged exploring map aside (it starts a new one)")

    # 4. recordings - moved aside, never deleted
    import numpy as np
    recs = []
    for d in (proj / "files" / "demos", proj / "files" / "best_games" / "hall_of_fame",
              proj / "files" / "best_games" / "studying"):
        recs += sorted(d.glob("*.npz")) if d.exists() else []
    gone = 0
    for p in recs:
        try:
            with np.load(p) as z:
                for k in z.files:
                    z[k]
        except Exception:
            gone += 1
            if fix:
                set_aside(p)
    if gone:
        found.append(f"{gone} recorded game{'s' if gone != 1 else ''} won't open")
        if fix:
            fixed.append(f"moved {gone} damaged recording{'s' if gone != 1 else ''} aside")

    # the answer
    if not found and not unfixable:
        msg = f"Checked {name}: everything opens fine - nothing to repair."
    elif not fix:
        msg = f"Checked {name}: " + "; ".join(found + unfixable) + "."
    else:
        msg = f"Repaired {name}: " + "; ".join(fixed) + "."
        if unfixable:
            msg += " Couldn't fix: " + "; ".join(unfixable) + "."
        if bk.exists():
            msg += f" Copies of everything changed are in '{bk.name}'."
        if rebuilt is not None:
            msg += " Press Train to carry on."
        try:
            with open(run / "console.log", "a", encoding="utf-8") as f:
                f.write(f"REPAIRED {time.strftime('%Y-%m-%d %H:%M')}: " + "; ".join(fixed) + "\n")
        except Exception:
            pass
    return {"ok": True, "found": found, "fixed": fixed if fix else [], "problems": unfixable, "msg": msg}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit(__doc__)
    target = Path(args[0]) if Path(args[0]).is_dir() else PROJECTS / args[0]
    r = repair(target, fix="--check" not in sys.argv)
    print(r["msg"])
