"""Portable AI Arcade - runs training and starts it again if it freezes.

python keepgoing.py <trainer.py | evolution.py> <AI folder>
The trainer's watchdog leaves files/run/FROZE when training stops moving for 10 minutes; this sees it and carries
on from the last save (up to 12 times, so a game that keeps freezing doesn't loop forever).
Signed: Claude, 11 Oct 2026."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import subprocess
import sys
import time
from pathlib import Path

FILES = Path(__file__).resolve().parent


def main():
    script, proj = sys.argv[1], Path(sys.argv[2]).resolve()
    run = proj / "files" / "run"
    rc = 0
    for attempt in range(13):
        rc = subprocess.call([sys.executable, "-u", str(FILES / script), str(proj)], cwd=str(FILES))
        froze, again = run / "FROZE", run / "RESTART"
        if rc not in (0, None) and not (run / "STOP").exists() and not (froze.exists() or again.exists()) and attempt < 3:
            # ALFRED 19 Oct 2026: a crash (e.g. inside NVIDIA's nvrtc at start-up) - carry on from the last save
            print(f"\n*** Training crashed (exit code {rc}) - starting it again from the last save in 30 seconds ***\n",
                  flush=True)
            time.sleep(30)
            continue
        if (run / "STOP").exists() or not (froze.exists() or again.exists()):
            break
        if again.exists():                   # the button check dropped buttons that do nothing (controls.py)
            again.unlink(missing_ok=True)
            print("\n*** The game's buttons changed - starting again with the new buttons in 5 seconds ***\n", flush=True)
            time.sleep(5)
            continue
        froze.unlink(missing_ok=True)
        print(f"\n*** Training froze - starting it again from the last save in 10 seconds (restart {attempt + 1}) ***\n",
              flush=True)
        time.sleep(10)
    return rc


if __name__ == "__main__":
    sys.exit(main())
