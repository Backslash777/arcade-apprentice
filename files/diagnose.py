"""Portable AI Arcade - PROBLEM REPORT. Collects everything needed to diagnose a tester's PC into ONE text file
next to START HERE.bat: "Problem report - <PC> - <date>.txt". The tester sends that file to Jonny, Jonny gives it
to Claude. Standard library only, so it works even when the AI libraries failed to install. Signed: Claude, 9 Oct 2026.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

FILES = Path(__file__).resolve().parent
ROOT = FILES.parent
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
out = []


def log_error(where, text):
    """Append a crash (full traceback) to files/logs/errors.log - the problem report includes it."""
    try:
        (FILES / "logs").mkdir(exist_ok=True)
        with open(FILES / "logs" / "errors.log", "a", encoding="utf-8") as f:
            f.write(f"\n----- {time.strftime('%Y-%m-%d %H:%M:%S')} | {where} -----\n{text.rstrip()}\n")
    except Exception:
        pass


def sec(title):
    out.append("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def line(k, v=""):
    out.append(f"{k}: {v}" if v != "" else str(k))


def run(cmd, timeout=40):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=NOWIN)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"(could not run: {e})"


def tail(path, n=80):
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:]) if lines else "(empty)"
    except FileNotFoundError:
        return "(missing)"
    except Exception as e:
        return f"(unreadable: {e})"


def main():
    stamp = time.strftime("%Y-%m-%d %H%M")
    pc = platform.node() or "PC"
    sec("PORTABLE AI ARCADE - PROBLEM REPORT")
    line("Made", time.strftime("%Y-%m-%d %H:%M:%S"))
    line("Kit folder", str(ROOT))
    line("Folder path length", len(str(ROOT)))
    try:
        line("Kit version files", ", ".join(f"{p.name} {time.strftime('%Y-%m-%d %H:%M', time.localtime(p.stat().st_mtime))}"
                                           for p in sorted(FILES.glob("*.py"))))
    except Exception:
        pass

    sec("COMPUTER")
    line("Windows", f"{platform.system()} {platform.release()} {platform.version()} {platform.machine()}")
    line("CPU", platform.processor())
    line("CPU threads", os.cpu_count())
    if sys.platform == "win32":
        line("Graphics cards", run(["powershell", "-NoProfile", "-Command",
                                    "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + ' | driver ' + $_.DriverVersion + ' | ' + [math]::Round($_.AdapterRAM/1GB,1) + ' GB (reported)' }"]))
        line("Memory", run(["powershell", "-NoProfile", "-Command",
                            "$o=Get-CimInstance Win32_OperatingSystem; '{0:N1} GB total, {1:N1} GB free' -f ($o.TotalVisibleMemorySize/1MB), ($o.FreePhysicalMemory/1MB)"]))
        line("Long paths enabled", run(["powershell", "-NoProfile", "-Command",
                                        "(Get-ItemProperty HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem).LongPathsEnabled"]))
    try:
        du = shutil.disk_usage(ROOT)
        line("Disk free", f"{du.free / 2**30:.1f} GB of {du.total / 2**30:.0f} GB")
    except Exception:
        pass
    if shutil.which("nvidia-smi"):
        line("nvidia-smi", run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,memory.total,memory.used,utilization.gpu", "--format=csv"]))

    sec("PYTHON + AI LIBRARIES")
    line("Python", f"{sys.version} at {sys.executable}")
    line("python-path.txt", tail(FILES / "python-path.txt", 2))
    for mod in ("torch", "torch_directml", "ale_py", "gymnasium", "numpy", "PIL", "pygame", "psutil"):
        try:
            m = __import__(mod)
            line(mod, getattr(m, "__version__", "installed"))
        except Exception as e:
            line(mod, f"NOT AVAILABLE ({type(e).__name__}: {e})")
    try:
        import torch
        line("torch CUDA build", f"cuda={torch.version.cuda} hip={getattr(torch.version, 'hip', None)}")
        line("torch.cuda.is_available", torch.cuda.is_available())
        if torch.cuda.is_available():
            line("torch arch list", torch.cuda.get_arch_list())
            line("device capability", torch.cuda.get_device_capability())
    except Exception:
        pass

    sec("GRAPHICS CARD TEST (a tiny real training step)")
    try:
        import device
        line("gpu-setup.json", tail(device.STATE, 20))
        try:
            dev, name, backend, problem = device.pick()
            line("Result", f"brain would run on {name} via {backend}")
            if problem:
                line("Problem", problem)
            if backend != "cpu":
                t = time.time()
                for _ in range(20):
                    device.smoke(dev)
                line("Speed check", f"20 mini training steps in {time.time() - t:.2f}s")
        except Exception:
            line("Test crashed", traceback.format_exc())
    except Exception:
        line("Could not run the test", traceback.format_exc())

    sec("ARCADE EMULATOR")
    try:
        import arcade_vec
        line("Emulator file present", arcade_vec.CORE.exists())
        line("Packs", ", ".join(arcade_vec.PACKS))
    except Exception:
        line("arcade_vec failed", traceback.format_exc())

    sec("GAMES IN THE ROMs FOLDER")
    try:
        import app
        for g in app.games()["yours"]:
            line(f"  {g['file']}", f"{g['name'] or '-'}  {g.get('why') or ''}")
        sec("HARDWARE PLAN THE WIZARD WOULD USE")
        for plat in ("atari", "arcade"):
            hw = app.hardware(plat)
            line(plat, json.dumps({k: hw.get(k) for k in ("backend", "gpu_name", "vram_gb", "ram_gb", "ram_free_gb",
                                                           "num_envs", "minibatch_size", "emu_threads", "width", "plan",
                                                           "gpu_warning", "torch_error")}))
    except Exception:
        line("Could not check games / hardware", traceback.format_exc())

    sec("SETUP LOG (last part)")
    out.append(tail(FILES / "logs" / "setup.log", 120))
    sec("ERRORS LOG (every crash, full detail)")
    out.append(tail(FILES / "logs" / "errors.log", 200))

    proj_root = ROOT / "My AIs"
    for d in sorted(proj_root.iterdir()) if proj_root.exists() else []:
        if not (d / "project.json").exists():
            continue
        run_dir = d / "files" / "run"
        sec(f"AI: {d.name}")
        line("project.json", tail(d / "project.json", 60))
        line("status.json", tail(run_dir / "status.json", 5))
        line("console.log (last 40 lines)", "\n" + tail(run_dir / "console.log", 40))
        line("watch.log", "\n" + tail(run_dir / "watch.log", 40))

    name = f"Problem report - {pc} - {stamp}.txt"
    (ROOT / name).write_text("\n".join(str(x) for x in out), encoding="utf-8")
    print(f"\n  Made: {name}\n  It's in the same folder as START HERE.bat. Please send that file to Jonny.\n"
          "  (It contains your PC's specs and folder names - nothing else personal.)")
    return name


if __name__ == "__main__":
    main()
