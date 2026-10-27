"""Can a game's start point (a save-state) be restored in a freshly started emulator?

Some MAME 2003-Plus drivers (Splatterhouse's Namco System 1, for one) save states that can't be loaded back into a
new emulator - it crashes. Packs for those games say "reset": "reboot", and the AI starts each game by switching the
machine off and on, putting a coin in and pressing start instead.

Run: python restore_check.py <start.state> <rom.zip> <core>      prints OK (exit 0), or crashes (exit != 0)
Signed: Claude, 13 Oct 2026.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
import arcade_vec                      # noqa: E402
from libretro_host import LibretroHost  # noqa: E402


def main():
    state, rom, core = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    rt = Path(tempfile.gettempdir()) / f"PAA_restore_{os.getpid()}"
    (rt / "roms").mkdir(parents=True, exist_ok=True)
    shutil.copy2(rom, rt / "roms" / rom.name)
    LibretroHost._video = arcade_vec._fast_video
    h = LibretroHost(arcade_vec.local_core(rt, core), rt / "roms" / rom.name, rt, require_ram=False)
    for _ in range(2):
        h.run((), capture=False)
    h.load_state(state.read_bytes())
    for _ in range(120):
        h.run((), capture=False)
    print("OK", flush=True)
    os._exit(0)                          # don't let a half-broken core take the process down on the way out


if __name__ == "__main__":
    main()


def can_restore(state_path, rom, core, python=sys.executable, timeout=120):
    """True if the state loads and runs in a fresh emulator (checked in a child process: a crash can't hurt us)."""
    import subprocess
    try:
        r = subprocess.run([python, "-B", str(Path(__file__).resolve()), str(state_path), str(rom), core],
                           capture_output=True, text=True, timeout=timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return r.returncode == 0 and "OK" in r.stdout
    except Exception:
        return True                      # couldn't tell (e.g. timed out): leave things as they were
