"""Proper names for arcade game files: arcade_names.name_of(zip_path) -> "Aero Fighters (Video System, 1992)".
Uses arcade\\names.json (built from the emulators' official game lists by autopack\\build_names.py)."""
import json
import zipfile
from pathlib import Path

_DB = None


def _db():
    global _DB
    if _DB is None:
        try:
            _DB = json.loads((Path(__file__).resolve().parent / "arcade" / "names.json").read_text(encoding="utf-8"))
        except Exception:
            _DB = {"sets": {}, "crcs": {}}
    return _DB


def lookup(path):
    """(title, maker, year, setname) for a zip, or None. By file name first, then by the chips inside."""
    db = _db()
    p = Path(path)
    s = db["sets"].get(p.stem.lower())
    if s:
        return s[0], s[1], s[2], p.stem.lower()
    try:
        with zipfile.ZipFile(p) as z:
            votes = {}
            for i in z.infolist():
                g = db["crcs"].get("%08x" % i.CRC)
                if g:
                    votes[g] = votes.get(g, 0) + 1
    except Exception:
        return None
    if not votes:
        return None
    g = max(votes, key=votes.get)
    s = db["sets"][g]
    return s[0], s[1], s[2], g


def name_of(path, with_maker=False):
    r = lookup(path)
    if not r:
        return None
    t, m, y, _ = r
    if with_maker and (m or y):
        return f"{t} ({', '.join(x for x in (m, y) if x)})"
    return t
