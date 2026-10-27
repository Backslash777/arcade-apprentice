"""Free up this PC - closes background apps that slow training down (Alfred, 21 Oct 2026).

The list lives in clutter_list.txt next to this file - plain text, anyone can edit it in Notepad.
Apps a person ticks "keep" on the page are saved in clutter_keep.json and left alone.
Never deletes anything, never touches Windows services, and a restart brings every app back.

  python clutter.py        close everything on the list that's running (except "keep" ones)
  python clutter.py dry    just show what it would close
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

FILES = Path(__file__).resolve().parent
LIST = FILES / "clutter_list.txt"
KEEP = FILES / "clutter_keep.json"
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Never closed, whatever the list says: Windows itself, browsers (the kit's page lives in one),
# the kit's own Python, security software and graphics drivers.
NEVER = {n.lower() for n in (
    "explorer.exe", "csrss.exe", "winlogon.exe", "wininit.exe", "services.exe", "lsass.exe", "svchost.exe",
    "dwm.exe", "smss.exe", "system", "registry", "fontdrvhost.exe", "sihost.exe", "ctfmon.exe", "taskmgr.exe",
    "chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe", "vivaldi.exe",
    "python.exe", "pythonw.exe", "py.exe", "conhost.exe", "cmd.exe", "powershell.exe", "pwsh.exe",
    "windowsterminal.exe", "openconsole.exe", "claude.exe",
    "msmpeng.exe", "nissrv.exe", "securityhealthservice.exe", "securityhealthsystray.exe", "mpdefendercoreservice.exe",
    "nvcontainer.exe", "nvdisplay.container.exe", "atiesrxx.exe", "atieclxx.exe", "igfxem.exe",
)}

DEFAULT_LIST = """\
# FREE UP THIS PC - apps the kit can close while an AI trains.
# One line per program:   program.exe | Name people see | warning (optional)
# Lines with the same Name are closed together (one tick box on the page).
# "program.exe @Word" only closes it when its folder path contains Word (for shared helper names).
# Lines starting with # are ignored. Add your own at the bottom. A restart brings everything back.

# --- Game launchers ---
steam.exe | Steam
steamwebhelper.exe | Steam
EpicGamesLauncher.exe | Epic Games Launcher
EpicWebHelper.exe | Epic Games Launcher
EADesktop.exe | EA app
Battle.net.exe | Battle.net
upc.exe | Ubisoft Connect
UplayWebCore.exe | Ubisoft Connect
GalaxyClient.exe | GOG Galaxy
RiotClientServices.exe | Riot Client
Overwolf.exe | Overwolf

# --- Chat and calls ---
Discord.exe | Discord | You'll drop out of any voice chat.
WhatsApp.exe | WhatsApp
WhatsApp.Root.exe | WhatsApp
ms-teams.exe | Microsoft Teams | You'll drop out of any call.
Teams.exe | Microsoft Teams | You'll drop out of any call.
Telegram.exe | Telegram
Slack.exe | Slack
Zoom.exe | Zoom | You'll drop out of any call.

# --- Cloud sync ---
OneDrive.exe | OneDrive | Files stop syncing until you restart it.
GoogleDriveFS.exe | Google Drive | Files stop syncing until you restart it.
Dropbox.exe | Dropbox | Files stop syncing until you restart it.

# --- Lighting, mice and keyboards ---
iCUE.exe | Corsair iCUE | Lighting and mouse/keyboard macros may stop until you restart it.
QmlRenderer.exe @Corsair | Corsair iCUE
crashpad_handler.exe @Corsair | Corsair iCUE
RazerAppEngine.exe | Razer Synapse | Lighting and mouse/keyboard macros may stop until you restart it.
Razer Central.exe | Razer Synapse
RzEngineMon.exe | Razer Synapse
razerwdl.exe | Razer Synapse
lghub.exe | Logitech G HUB | Lighting and mouse/keyboard macros may stop until you restart it.
lghub_agent.exe | Logitech G HUB
ArmouryCrate.exe | ASUS Armoury Crate | Lighting may stop until you restart it.
SteelSeriesGG.exe | SteelSeries GG | Lighting and mouse/keyboard macros may stop until you restart it.
SteelSeriesEngine.exe | SteelSeries GG
SignalRgb.exe | SignalRGB | Lighting may stop until you restart it.

# --- Android emulators ---
dnplayer.exe | LDPlayer
dnmultiplayerex.exe | LDPlayer
dnmultiplayer.exe | LDPlayer
HD-Player.exe | BlueStacks
HD-MultiInstanceManager.exe | BlueStacks

# --- VPNs ---
Surfshark.exe | Surfshark | Your VPN may disconnect.
NordVPN.exe | NordVPN | Your VPN may disconnect.
ExpressVPN.exe | ExpressVPN | Your VPN may disconnect.

# --- Music ---
Spotify.exe | Spotify | Your music stops.

# --- Windows extras ---
PhoneExperienceHost.exe | Phone Link
CrossDeviceService.exe | Phone Link
Widgets.exe | Windows Widgets
WidgetBoard.exe | Windows Widgets
WidgetService.exe | Windows Widgets
MicrosoftStartFeedProvider.exe | Windows Widgets
SDXHelper.exe | Office background helper

# --- AI apps ---
Kimi.exe | Kimi
kimi-cu.exe | Kimi
kimi-webbridge.exe | Kimi
"""


def load_list():
    if not LIST.exists():
        LIST.write_text(DEFAULT_LIST, encoding="utf-8")
    rows = []
    for line in LIST.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        proc, path_has = parts[0], ""
        if "@" in proc:
            proc, path_has = (s.strip() for s in proc.split("@", 1))
        if not proc.lower().endswith(".exe"):
            proc += ".exe"
        if proc.lower() in NEVER:
            continue
        name = parts[1] if len(parts) > 1 and parts[1] else proc[:-4]
        note = parts[2] if len(parts) > 2 else ""
        rows.append({"proc": proc.lower(), "path_has": path_has.lower(), "name": name, "note": note})
    return rows


def load_keep():
    try:
        return set(json.loads(KEEP.read_text(encoding="utf-8")))
    except Exception:
        return set()


def set_keep(names):
    KEEP.write_text(json.dumps(sorted(set(names)), indent=1), encoding="utf-8")
    return {"ok": True}


def _running(rows):
    """{name: {"note", "procs": [psutil.Process]}} for list apps that are running now."""
    me = {os.getpid(), os.getppid()}
    by_proc, notes = {}, {}
    for r in rows:
        notes[r["name"]] = notes.get(r["name"]) or r["note"]
        by_proc.setdefault(r["proc"], []).append(r)
    found = {}
    for p in psutil.process_iter(["name", "exe"]):
        try:
            n = (p.info["name"] or "").lower()
            if n not in by_proc or p.pid in me:
                continue
            path = (p.info["exe"] or "").lower()
            for r in by_proc[n]:
                if r["path_has"] and r["path_has"] not in path:
                    continue
                g = found.setdefault(r["name"], {"note": notes[r["name"]], "procs": []})
                g["procs"].append(p)
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return found


def scan():
    keep = load_keep()
    apps = []
    for name, g in sorted(_running(load_list()).items(), key=lambda kv: kv[0].lower()):
        mem = 0
        for p in g["procs"]:
            try:
                mem += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        apps.append({"name": name, "note": g["note"], "count": len(g["procs"]),
                     "mem_mb": round(mem / 1048576), "keep": name in keep})
    return {"ok": True, "apps": apps, "list_file": str(LIST)}


def close(names=None):
    """Close the named apps (None = everything running on the list that isn't ticked "keep").
    Asks politely first, then insists after a few seconds."""
    found = _running(load_list())
    keep = load_keep()
    todo = {n: g for n, g in found.items() if (n in names if names is not None else n not in keep)}
    closed, failed = [], []
    for name, g in todo.items():
        if any(p.info["name"].lower() == "steam.exe" for p in g["procs"]):
            exe = next((p.info["exe"] for p in g["procs"] if p.info["name"].lower() == "steam.exe" and p.info["exe"]), "")
            if exe:
                try:
                    subprocess.Popen([exe, "-shutdown"], creationflags=NOWIN)
                except OSError:
                    pass
        for p in g["procs"]:                       # polite: like clicking the window's X
            subprocess.run(["taskkill", "/PID", str(p.pid)], capture_output=True, creationflags=NOWIN)
    if todo:
        deadline = time.time() + 6
        while time.time() < deadline and any(p.is_running() for g in todo.values() for p in g["procs"]):
            time.sleep(0.5)
    for name, g in todo.items():
        stuck = False
        for p in g["procs"]:
            try:
                if p.is_running():
                    p.kill()
                    p.wait(3)
            except psutil.NoSuchProcess:
                pass
            except (psutil.AccessDenied, psutil.TimeoutExpired):
                stuck = True
        (failed if stuck else closed).append(name)
    if not todo:
        msg = "Nothing to close - this PC is already tidy."
    else:
        msg = f"Closed {len(closed)} app{'s' if len(closed) != 1 else ''}" + (": " + ", ".join(closed) if closed else "") + "."
        if failed:
            msg += f" Couldn't close {', '.join(failed)} (Windows wants admin rights for those) - close them by hand if you like."
        msg += " A restart brings them all back."
    _log(msg)
    return {"ok": True, "closed": closed, "failed": failed, "msg": msg}


def open_list():
    load_list()
    if hasattr(os, "startfile"):
        os.startfile(str(LIST))
    return {"ok": True, "msg": "Opening the list in Notepad - save it, then press Look again."}


def _log(msg):
    try:
        logs = FILES / "logs"
        logs.mkdir(exist_ok=True)
        with open(logs / "alternate_log.txt", "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Free up this PC: {msg}\n")
    except OSError:
        pass


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "dry":
        s = scan()
        print("Free up this PC (dry run - closing nothing)")
        for a in s["apps"]:
            print(f"  {'KEEP ' if a['keep'] else 'would close'}  {a['name']}  ({a['count']} process{'es' if a['count'] != 1 else ''}, {a['mem_mb']} MB)")
        if not s["apps"]:
            print("  Nothing on the list is running.")
        print(f"\nThe list: {s['list_file']}")
    else:
        print(close()["msg"])