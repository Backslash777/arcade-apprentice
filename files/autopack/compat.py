"""Which games in a folder actually RUN in the kit's arcade emulator (FBNeo)? One child process per game.
python compat.py <folder>  -> files/autopack/compat.csv  (set, runs, ram_kb, state_kb, note)
A romset made for another emulator (e.g. MAME 2003-Plus) often 'loads' but only shows FBNeo's missing-files
screen: no save-state, no memory, a frozen picture. Those are marked runs = no.
Signed: Claude, 10 Oct 2026."""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FILES = Path(__file__).resolve().parent.parent


def one(zp):
    sys.path.insert(0, str(FILES))
    sys.path.insert(0, str(FILES / "arcade" / "generic"))
    import arcade_vec
    from libretro_host import LibretroHost
    rt = Path(tempfile.gettempdir()) / f"PAA_compat_{os.getpid()}"
    (rt / "roms").mkdir(parents=True, exist_ok=True)
    shutil.copy2(zp, rt / "roms" / Path(zp).name)
    try:
        h = LibretroHost(arcade_vec.local_core(rt), rt / "roms" / Path(zp).name, rt, require_ram=False)
    except Exception as e:
        print(json.dumps({"runs": "no", "note": "won't load: " + str(e)[:80]}))
        return
    f0 = h.run(()).copy()
    for _ in range(900):
        f = h.run(())
    st = len(h.save_state()) if h.dll.retro_serialize_size() else 0
    moved = bool(f.shape != f0.shape or (f != f0).any())
    ram = 0 if h.ram is None else len(h.ram)
    runs = "yes" if (st and moved) else "no"
    note = "" if runs == "yes" else ("frozen picture, no save-state - romset doesn't match FBNeo's" if not st else
                                     "picture never changes")
    print(json.dumps({"runs": runs, "ram_kb": ram // 1024, "state_kb": st // 1024, "note": note}))
    shutil.rmtree(rt, ignore_errors=True)


def main():
    if sys.argv[1] == "--one":
        return one(sys.argv[2])
    out = FILES / "autopack" / "compat.csv"
    rows = []
    for zp in sorted(Path(sys.argv[1]).glob("*.zip")):
        try:
            r = subprocess.run([sys.executable, "-B", __file__, "--one", str(zp)], capture_output=True, text=True, timeout=120)
            d = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception as e:
            d = {"runs": "no", "note": "crashed"}
        d["set"] = zp.stem.lower()
        rows.append(d)
        print(d["set"], d["runs"], d.get("note", ""), flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set", "runs", "ram_kb", "state_kb", "note"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{sum(r['runs'] == 'yes' for r in rows)} of {len(rows)} run in FBNeo", flush=True)


if __name__ == "__main__":
    main()
