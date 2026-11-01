"""UPDATE CHECKER + UPDATER for the kit. ALFRED 1 Nov 2026.

check()  - asks GitHub for the latest release (once every few hours at most; quiet if offline) and says if it's newer.
apply()  - downloads it, checks its SHA-256 against GitHub's, backs up every file it will replace into
           files\\_old\\<from>-to-<to>-<time>\\, then puts the new code in. NEVER touched: My AIs, ROMs, Recipe Cards,
           the private Python (files\\runtime), logs, _dev, this PC's tuning and settings. Game packs: new ones are
           added; one already here is only replaced if nobody changed it since it was installed (release_manifest.json).
undo()   - puts the backed-up files back.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

FILES = Path(__file__).resolve().parent
ROOT = FILES.parent
REPO = "Backslash777/arcade-apprentice"
API = __import__("os").environ.get("AA_UPDATE_API") or f"https://api.github.com/repos/{REPO}/releases/latest"   # (tests)
ASSET = "Arcade-Apprentice.zip"
CACHE = FILES / "logs" / "update_check.json"
LAST = FILES / "logs" / "update_last.json"
CHECK_EVERY = 6 * 3600
# never replaced or backed up (theirs, this PC's, or not part of the kit's code)
KEEP_TOP = {"My AIs", "ROMs - put your games here", "Recipe Cards"}
KEEP_FILES_DIRS = {"runtime", "logs", "_dev", "_old", "__pycache__"}
KEEP_NAMES = {"benchmark.json", "gpu-setup.json", "clutter_keep.json", "python-path.txt", "robot.log",
              "robot_status.json", "results.csv", "STOP", "desktop.ini", "HANDOVER.md", "FAST_LAYOUT_OFF.paused"}
MANIFEST = "release_manifest.json"


def current_version(files=None):
    try:
        t = (Path(files or FILES) / "recipe_cards.py").read_text(encoding="utf-8")
        return re.search(r'KIT_VERSION = "([^"]+)"', t).group(1)
    except Exception:
        return "0"


def vtuple(v):
    return tuple(int(x) for x in re.findall(r"\d+", str(v))[:4]) or (0,)


def _get_json(url, timeout=6):
    req = urllib.request.Request(url, headers={"User-Agent": "ArcadeApprentice-kit", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def check(force=False, fetch=_get_json):
    """{current, latest, newer, notes, page, size, sha256, url} - or {newer: False, error} if GitHub can't be reached."""
    cur = current_version()
    if not force and CACHE.exists():
        try:
            c = json.loads(CACHE.read_text(encoding="utf-8"))
            if time.time() - c.get("checked", 0) < CHECK_EVERY and c.get("current") == cur:
                return c
        except Exception:
            pass
    try:
        rel = fetch(API)
        asset = next((a for a in rel.get("assets", []) if a.get("name") == ASSET), None)
        latest = str(rel.get("tag_name", "")).lstrip("vV")
        sha = (asset or {}).get("digest", "") or ""
        sha = sha.split(":", 1)[1] if sha.startswith("sha256:") else ""
        if not sha:                                   # older releases: the SHA-256 written in the notes
            m = re.search(r"SHA-256\W*([0-9a-f]{64})", rel.get("body", "") or "", re.I)
            sha = m.group(1).lower() if m else ""
        res = {"current": cur, "latest": latest, "newer": bool(asset) and vtuple(latest) > vtuple(cur),
               "notes": (rel.get("body") or "")[:4000], "page": rel.get("html_url", ""),
               "size": (asset or {}).get("size", 0), "sha256": sha,
               "url": (asset or {}).get("browser_download_url", ""), "checked": time.time()}
    except Exception as e:
        res = {"current": cur, "newer": False, "error": f"couldn't reach GitHub ({type(e).__name__})", "checked": time.time()}
    try:
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps(res), encoding="utf-8")
    except Exception:
        pass
    return res


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _kept(rel_parts):
    """True for paths the updater must never write over or back up."""
    if not rel_parts:
        return True
    if rel_parts[0] in KEEP_TOP:
        return True
    if rel_parts[0] == "files" and len(rel_parts) > 1 and rel_parts[1] in KEEP_FILES_DIRS:
        return True
    return rel_parts[-1] in KEEP_NAMES or rel_parts[-1].endswith(".pyc")


def _pack_dir(rel_parts):
    """files/arcade/found/<set>/... -> '<set>', else None."""
    if len(rel_parts) >= 5 and rel_parts[:3] == ("files", "arcade", "found"):
        return rel_parts[3]
    return None


def make_manifest(kit_root):
    """sha256 of every file in a kit folder (used when building a release zip)."""
    kit_root = Path(kit_root)
    out = {}
    for p in sorted(kit_root.rglob("*")):
        if p.is_file():
            parts = p.relative_to(kit_root).parts
            if not _kept(parts) and p.name != MANIFEST:
                out["/".join(parts)] = _sha(p)
    return out


def apply(info=None, say=print, download=None, root=None):
    """Update this kit to the latest release. Returns {ok, msg, ...}."""
    root = Path(root or ROOT)
    files = root / "files"
    info = info or check(force=True)
    if not info.get("newer"):
        return {"ok": False, "msg": info.get("error") or f"You're already on the newest version (v{info.get('current')})."}
    if not info.get("sha256"):
        return {"ok": False, "msg": "That release has no SHA-256 fingerprint to check the download against - not risking it."}
    work = Path(tempfile.mkdtemp(prefix="aa-update-"))
    try:
        z = work / ASSET
        say(f"Downloading v{info['latest']} ({info.get('size', 0) / 1e6:.0f} MB)...")
        if download:
            download(info["url"], z)
        else:
            req = urllib.request.Request(info["url"], headers={"User-Agent": "ArcadeApprentice-kit"})
            with urllib.request.urlopen(req, timeout=60) as r, open(z, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
        got = _sha(z)
        if got != info["sha256"]:
            return {"ok": False, "msg": "The download didn't match its fingerprint (a broken or tampered download). Nothing was changed - try again later."}
        with zipfile.ZipFile(z) as zz:
            if zz.testzip() is not None:
                return {"ok": False, "msg": "The download is damaged. Nothing was changed - try again later."}
            zz.extractall(work / "new")
        tops = [p for p in (work / "new").iterdir() if p.is_dir()]
        new = tops[0] if len(tops) == 1 and (tops[0] / "files").exists() else work / "new"
        if not (new / "files" / "app.py").exists():
            return {"ok": False, "msg": "The download doesn't look like an Arcade Apprentice kit. Nothing was changed."}
        old_manifest = {}
        try:
            old_manifest = json.loads((files / MANIFEST).read_text(encoding="utf-8"))
        except Exception:
            pass
        # which existing game packs did they change (or find themselves)? those stay as they are
        touched = set()
        for p in (files / "arcade" / "found").glob("*/*") if (files / "arcade" / "found").exists() else []:
            if p.is_file() and not _kept(p.relative_to(root).parts):
                rel = "/".join(p.relative_to(root).parts)
                if old_manifest.get(rel) != _sha(p):
                    touched.add(p.parent.name)
        cur = current_version(files)
        bak = files / "_old" / f"v{cur}-to-v{info['latest']}-{time.strftime('%Y%m%d-%H%M%S')}"
        plan, kept_packs = [], set()
        for p in sorted(new.rglob("*")):
            if not p.is_file():
                continue
            parts = p.relative_to(new).parts
            if _kept(parts):
                continue
            pk = _pack_dir(parts)
            dst = root.joinpath(*parts)
            if pk and pk in touched and dst.exists():
                kept_packs.add(pk)
                continue
            if dst.exists() and _sha(dst) == _sha(p):
                continue
            plan.append((p, dst, parts))
        say(f"Backing up {sum(1 for _, d, _ in plan if d.exists())} files, then putting in {len(plan)} new or changed ones...")
        backed = []
        for src, dst, parts in plan:
            if dst.exists():
                b = bak.joinpath(*parts)
                b.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dst, b)
                backed.append("/".join(parts))
        added = []
        for src, dst, parts in plan:
            if not dst.exists():
                added.append("/".join(parts))
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        needs_setup = "files/setup.ps1" in backed     # new libraries: START HERE fetches them
        rec = {"from": cur, "to": info["latest"], "when": time.strftime("%Y-%m-%d %H:%M"), "backup": str(bak),
               "replaced": backed, "added": added, "kept_packs": sorted(kept_packs), "needs_setup": needs_setup,
               "seen": False}
        LAST_ = files / "logs" / "update_last.json"
        LAST_.parent.mkdir(exist_ok=True)
        LAST_.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        return {"ok": True, "msg": f"Updated to v{info['latest']} - your AIs, games and recipe cards are all still here.",
                **rec}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def undo(root=None):
    """Put back the files the last update replaced, and take out the ones it added."""
    root = Path(root or ROOT)
    files = root / "files"
    try:
        rec = json.loads((files / "logs" / "update_last.json").read_text(encoding="utf-8"))
    except Exception:
        return {"ok": False, "msg": "There's no update to undo."}
    if rec.get("undone"):
        return {"ok": False, "msg": "That update has already been undone."}
    bak = Path(rec["backup"])
    for rel in rec.get("replaced", []):
        src = bak.joinpath(*rel.split("/"))
        if src.exists():
            shutil.copy2(src, root.joinpath(*rel.split("/")))
    aside = bak / "_added_by_update"
    for rel in rec.get("added", []):                  # moved aside, never deleted
        p = root.joinpath(*rel.split("/"))
        if p.exists():
            d = aside.joinpath(*rel.split("/"))
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(d))
    rec["undone"] = time.strftime("%Y-%m-%d %H:%M")
    (files / "logs" / "update_last.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return {"ok": True, "msg": f"Back on v{rec['from']}. (The update's files are kept in files\\_old.)", "to": rec["from"]}
