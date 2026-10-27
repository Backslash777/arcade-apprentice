"""Portable AI Arcade - arcade games (FBNeo core) alongside the Atari 2600 ones.

Each supported arcade game has a "pack" in files/arcade/<id>/: Jonny's validated adapter (arcade_env.py +
libretro_host.py) that knows where that game keeps its score and lives. Arcade ROMs (.zip, MAME/FBNeo sets) are
recognised by the fingerprints (CRCs) of the chips inside, so the zip's name doesn't matter.

FBNeo allows one emulator per process, so ArcadeVec runs one cabinet per worker process and makes them look like
ALE's AtariVectorEnv to trainer.py: 4-frame stacks of 84x84 grey, real points as reward, lives in info.
Signed: Claude, 9 Oct 2026.
"""
from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import re
import shutil
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path

import numpy as np

FILES = Path(__file__).resolve().parent
ARCADE = FILES / "arcade"
CORE = ARCADE / "fbneo_libretro.dll"
# Second emulator: MAME 2003-Plus, for romsets made for it (it doesn't share its memory directly - the generic env
# reads the game's memory out of its save-state instead)
CORES = {"fbneo": CORE, "mame2003_plus": ARCADE / "mame2003_plus_libretro.dll"}
MAX_DECISIONS = 27_000          # ~30 minutes of cabinet time, then the game is cut off ("capped")
# THE BUTTON CHECK (controls.LiveCheck, 17 Oct 2026): about 1 move in PROBE_EVERY a cabinet tries one input on top of
# the move it is making, and compares the next second of screen with the same move without it.
PROBE_EVERY = int(os.environ.get("PAA_PROBE_EVERY") or 1000)
PROBE_FRAMES, PROBE_HOLD = 45, 12
_DIRS = {4, 5, 6, 7}                    # RetroPad up, down, left, right
_OPPOSITE = {4: 5, 5: 4, 6: 7, 7: 6}

PACKS = {
    "invaders": {"name": "Space Invaders (arcade)", "class": "ArcadeInvadersEnv", "n_actions": 6,
                 "decision_repeat": 4, "kwargs": {"reset_noop_max": 30},
                 "crcs": ["0ccead96", "14e538b0", "6bfaca4a", "734f5ad8"]},
    "mspacman": {"name": "Ms. Pac-Man (arcade)", "class": "ArcadeMsPacmanEnv", "n_actions": 5,
                 "decision_repeat": 4, "kwargs": {},
                 "crcs": ["1a6fb2d4", "2fc650bd", "3eb3a8e4", "5c281d01", "615af909", "77245b66", "817d94e3",
                          "a90e7000", "a9cc86bf", "bcdd1beb", "c1e6ab10", "c82cd714", "f45fbbcd"]},
    "dkong":    {"name": "Donkey Kong (arcade)", "class": "ArcadeDonkeyKongEnv", "n_actions": 10,
                 "decision_repeat": 2, "kwargs": {},
                 "crcs": ["12c8c95d", "15e9c5e9", "1d28895d", "20f2ef7e", "424f2b11", "45a4ed06", "4743fe92",
                          "553b89bb", "59f8054d", "5ec461ec", "672e4714", "b869b8f5", "d6412358", "e273ede5",
                          "feaa59ee"]},
    "trackfld": {"name": "Track & Field (arcade)", "class": "ArcadeTrackFieldEnv", "n_actions": 8,
                 "decision_repeat": 2, "kwargs": {"shaping_weight": 0.0},
                 "crcs": ["053e5861", "1743b5ee", "21d6c448", "2882f6d4", "50075768", "5886c802", "6c0d1ee9",
                          "800ff1f1", "82e2185a", "95bf79b6", "c2166a5c", "d2ba4d32", "d55f30b5", "d9faf183",
                          "dda9e29f", "f08c7b7e", "f546a56b"]},
}


FOUND = ARCADE / "found"     # packs made by the "Find the score" window (score_finder.py), one folder per game


def refresh_found():
    """Add every pack the Find-the-score window has made to PACKS (call again to pick up new ones)."""
    import json
    for d in sorted(FOUND.glob("*/pack.json")) if FOUND.exists() else []:
        try:
            p = json.loads(d.read_text(encoding="utf-8"))
            PACKS[p["id"]] = {"name": p["name"], "generic": str(d.parent), "n_actions": int(p["n_actions"]),
                              "decision_repeat": int(p.get("decision_repeat", 4)), "kwargs": {},
                              "crcs": p["crcs"], "fast_reset": False, "found": True, "core": p.get("core", "fbneo")}
        except Exception:
            pass


def identify_zip(path: Path):
    """Return the pack id whose chips this zip contains (>=75% of them), else None. Reads the zip's index only."""
    try:
        with zipfile.ZipFile(path) as z:
            have = {"%08x" % i.CRC for i in z.infolist()}
    except Exception:
        return None
    best, score = None, 0.0
    for pid, p in PACKS.items():
        s = len(have & set(p["crcs"])) / len(p["crcs"])
        if s > score:
            best, score = pid, s
    return best if score >= 0.75 else None


def arcade_available():
    return sys.platform == "win32" and CORE.exists()


def local_core(runtime_root, core="fbneo"):
    """Copy the emulator to local disk once: many processes loading a 62 MB DLL at once from a cloud-synced folder
    (Google Drive / OneDrive) can fail with WinError 1450 'insufficient system resources'."""
    src = CORES.get(core, CORE)
    dst = Path(runtime_root) / src.name
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return dst


# ALFRED 26 Oct 2026: some arcade games make their sounds from recordings ("samples") - arcade Space Invaders is
# silent without them. Sample sets (MAME/FBNeo .zip, named like the game) live in files\arcade\samples; a window
# that plays sound copies them into its own emulator's system folder. Training windows have no sound, so skip it.
SAMPLES = ARCADE / "samples"


def add_samples(runtime_root):
    try:
        zips = sorted(SAMPLES.glob("*.zip"))
    except OSError:
        return
    for sub in ("fbneo", "mame2003-plus"):
        dst = Path(runtime_root) / "runtime" / "system" / sub / "samples"
        for z in zips:
            try:
                target = dst / z.name
                if not target.exists() or target.stat().st_size != z.stat().st_size:
                    dst.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(z, target)
            except OSError:
                pass

# Each arcade window gets its own scratch folder in Windows' Temp folder (a ~60 MB copy of the emulator, plus the
# game), named after its process id. A window can't delete the emulator while it's still using it, so these used to
# pile up, one per Watch / Find-the-score / training window. Now each new window sweeps up the ones whose window has
# closed, and UNINSTALL.bat clears whatever's left.
SCRATCH = Path(tempfile.gettempdir()) / "PortableAIArcade"
_OURS = re.compile(r"^[a-z]+-(\d+)$")                       # run-1234, watch-1234, finder-1234, teach-1234...
_OURS_IN_TEMP = re.compile(r"^PAA_(?:chk|compat)_(\d+)$")   # the Library Robot's checks, straight in Temp


def tidy_scratch():
    """Delete scratch folders left behind by kit windows that have closed (or crashed)."""
    try:
        import psutil
    except Exception:
        return
    for folder, pattern in ((SCRATCH, _OURS), (SCRATCH.parent, _OURS_IN_TEMP)):
        try:
            entries = list(folder.iterdir())
        except OSError:
            continue
        for d in entries:
            m = pattern.match(d.name)
            try:
                if m and d.is_dir() and int(m.group(1)) != os.getpid() and not psutil.pid_exists(int(m.group(1))):
                    shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass


def scratch_dir(kind):
    """This window's own empty scratch folder, TEMP/PortableAIArcade/<kind>-<process id>. Sweeps up old ones first,
    and empties this one again when the window closes (anything still in use is swept up by the next window)."""
    tidy_scratch()
    d = SCRATCH / f"{kind}-{os.getpid()}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    atexit.register(shutil.rmtree, d, True)
    return d


_LUT565 = _LUT555 = None


def _fast_video(self, data, width, height, pitch):
    """Same pixels as the adapters' own _video, ~10x faster: byte views / lookup tables instead of per-channel maths."""
    global _LUT565, _LUT555
    if not data or not self.capture:
        return
    try:
        import ctypes as C
        if self.pixel_format == 1:   # XRGB8888, little-endian bytes are B,G,R,X
            raw = np.frombuffer(C.string_at(data, pitch * height), np.uint8).reshape(height, pitch // 4, 4)[:, :width]
            rgb = raw[..., 2::-1]
        else:
            raw = np.frombuffer(C.string_at(data, pitch * height), np.uint16).reshape(height, pitch // 2)[:, :width]
            if self.pixel_format == 2:
                if _LUT565 is None:
                    v = np.arange(65536, dtype=np.uint32)
                    _LUT565 = np.stack((((v >> 11) & 31) * 255 // 31, ((v >> 5) & 63) * 255 // 63,
                                        (v & 31) * 255 // 31), -1).astype(np.uint8)
                rgb = _LUT565[raw]
            else:
                if _LUT555 is None:
                    v = np.arange(65536, dtype=np.uint32)
                    _LUT555 = np.stack((((v >> 10) & 31) * 255 // 31, ((v >> 5) & 31) * 255 // 31,
                                        (v & 31) * 255 // 31), -1).astype(np.uint8)
                rgb = _LUT555[raw]
        self.frame = np.ascontiguousarray(np.rot90(rgb, self.rotation))
    except Exception as exc:
        self.callback_error = exc


def make_env(pack_id, rom_path, runtime_root, audio=False):
    if audio:
        add_samples(runtime_root)     # ALFRED 26 Oct 2026: games that need sound samples (arcade Space Invaders)
    if pack_id not in PACKS:
        refresh_found()
    p = PACKS[pack_id]
    core = Path(runtime_root) / CORES.get(p.get("core", "fbneo"), CORE).name
    if p.get("core", "fbneo") != "fbneo" and not core.exists():
        core = local_core(runtime_root, p["core"])
    if not core.exists():
        core = CORE
    if p.get("generic"):
        sys.path.insert(0, str(ARCADE / "generic"))
        import importlib
        host_mod = importlib.import_module("libretro_host")
        host_mod.LibretroHost._video = _fast_video
        from generic_env import GenericArcadeEnv
        return GenericArcadeEnv(p["generic"], decision_repeat=p["decision_repeat"], root=Path(runtime_root),
                                core_path=core, rom_path=Path(rom_path), audio=audio)
    sys.path.insert(0, str(ARCADE / pack_id))
    import importlib
    mod = importlib.import_module("arcade_env")
    host_mod = importlib.import_module("libretro_host")
    host_mod.LibretroHost._video = _fast_video
    cls = getattr(mod, p["class"])
    return cls(decision_repeat=p["decision_repeat"], root=Path(runtime_root), core_path=core,
               rom_path=Path(rom_path), audio=audio, **p["kwargs"])


refresh_found()



def _saucer_tally(info):
    """ALFRED 26 Oct 2026: saucers hit, 300-point saucers, saucer points and waves cleared for games.jsonl
    (arcade Space Invaders reports these; other games give nothing). Measures only - never touches rewards."""
    if info.get("ufo_hits") is None:
        return {}
    try:
        return {"ufo": int(info.get("ufo_hits") or 0), "ufo300": int(info.get("ufo_300_hits") or 0),
                "ufo_pts": int(sum(int(a.get("points", 0)) for a in (info.get("ufo_awards") or []))),
                "waves": int(info.get("waves") or 0)}
    except Exception:
        return {}

class Cabinet:
    """One arcade machine: points = change in the real on-screen score, lives from the game's own RAM."""

    def __init__(self, pack_id, rom_path, runtime_root, seed, sticky=0.25, audio=False, probe=None, practice=0.0):
        if pack_id not in PACKS:
            refresh_found()
        self.env = make_env(pack_id, rom_path, runtime_root, audio=audio)
        self.seed, self.games = seed, 0
        self.sticky, self.rng, self.prev = float(sticky), np.random.default_rng(seed), 0
        # the button check: found games only, and only ones whose save-states restore
        ok = PACKS[pack_id].get("found") and getattr(self.env, "actions", None) is not None \
            and not getattr(self.env, "reboot", False)
        self.probe = [int(c) for c in (probe or [])] if ok else []
        self.probe_rng, self.last_probe = np.random.default_rng(seed + 99991), None
        self.fast_reset = PACKS[pack_id].get("fast_reset", True)
        # How the current game was started, so it can be played again move for move (the Hall of Fame replays):
        # a real reset with this seed, then this many 'do nothing' moves (the snapshot starts' random delay).
        self.recipe, self.snap_seed, self.last_action = None, None, 0
        # ALFRED 23 Oct 2026: 'Practise the hard parts' - moments kept from its own games (practice_pool) that some
        # games start from instead of the beginning (not for games whose save-states don't restore); plus the shots
        # and misses the game's adapter counts, for the accuracy meter and 'A missed shot costs'.
        self.practice = float(practice or 0) if self.fast_reset and not getattr(self.env, "reboot", False) else 0.0
        self.drill_rng = np.random.default_rng(seed + 4242)
        self.practice_pool = {}             # kind of moment -> [(emulator state, adapter bookkeeping), ...]
        self.drill, self.restart_pending = False, -1.0
        self.step_misses, self.game_stats, self._miss_seen = 0, None, 0

    REAL_RESET_EVERY = 25      # a full boot -> coin -> start every 25 games refreshes the snapshot

    def _save_snapshot(self):
        """Remember the moment a game starts: the emulator's state + the adapter's own bookkeeping."""
        self.snap = self._capture()

    def _capture(self):
        """This moment: the emulator's state + the adapter's own bookkeeping."""
        import copy
        attrs = {}
        for k, v in self.env.__dict__.items():
            if k in ("host",) or callable(v):
                continue
            if isinstance(v, np.ndarray) and not v.flags.owndata:
                continue            # a live view into emulator memory (restored by load_state itself)
            try:
                attrs[k] = copy.deepcopy(v)
            except Exception:
                pass
        return self.env.host.save_state(), attrs

    POOL_PER_KIND = 6       # moments kept per kind, per cabinet (each one an emulator save-state)

    def _keep_moment(self):
        """ALFRED 23 Oct 2026: keep this moment for practice games if the game says it's a hard part (Space Invaders:
        a new wave, the last few invaders); games without their own idea get a moment every 400 moves, so practice
        games started from those go further each time."""
        hook = getattr(self.env, "practice_moment", None)
        if hook is not None:
            kind = hook()
        else:
            kind = "later" if self.decisions >= 400 and self.decisions % 400 == 0 else None
        if not kind:
            return
        try:
            moment = self._capture()
        except Exception:
            return
        pool = self.practice_pool.setdefault(kind, [])
        if len(pool) < self.POOL_PER_KIND:
            pool.append(moment)
        else:
            pool[int(self.drill_rng.integers(len(pool)))] = moment

    def _start_drill(self):
        """ALFRED 23 Oct 2026: start this game from a kept moment, after the same random 1-8 move wait as every start
        (so it can't learn a moment off by heart). Returns the first screen, or None if the moment didn't take."""
        import copy
        kinds = [k for k, v in self.practice_pool.items() if v]
        if not kinds:
            return None
        pool = self.practice_pool[kinds[int(self.drill_rng.integers(len(kinds)))]]
        state, attrs = pool[int(self.drill_rng.integers(len(pool)))]
        try:
            self.env.host.load_state(state)
            self.env.__dict__.update(copy.deepcopy(attrs))
            for _ in range(1 + int(self.drill_rng.integers(0, 8))):
                obs, _r, term, trunc, info = self.env.step(0)
                if term or trunc:
                    return None
        except Exception:
            return None
        self.drill, self.recipe = True, None       # started mid-game: it can't be re-run from a power-on
        self.score, self.lives, self.decisions = float(info.get("score", 0)), int(info.get("lives", 0) or 0), 0
        self.restart_pending = self.score          # reported with the next move (the trainer keeps it off the scoreboard)
        self._miss_seen = int(info.get("misses", 0) or 0)
        return obs[0]

    def pop_restart(self):
        """The score a practice game started on, once (-1 = none started)."""
        rs, self.restart_pending = self.restart_pending, -1.0
        return rs

    def new_game(self):
        """Start a game. Usually from the snapshot (fast) + a random 1-8 move delay so no two starts are identical."""
        import copy
        self.games += 1
        self.drill = False
        if self.practice > 0 and self.practice_pool and self.drill_rng.random() < self.practice:
            first = self._start_drill()
            if first is not None:
                return first
        snap = getattr(self, "snap", None)
        if snap is None or self.games % self.REAL_RESET_EVERY == 1 or not self.fast_reset:
            obs, info = self.env.reset(seed=self.seed + self.games)
            self.recipe = {"seed": int(self.seed + self.games), "noops": 0}
            if self.fast_reset:
                self._save_snapshot()
                self.snap_seed = int(self.seed + self.games)
        else:
            state, attrs = snap
            self.env.host.load_state(state)
            self.env.__dict__.update(copy.deepcopy(attrs))
            n = 1 + int(self.rng.integers(0, 8))
            self.recipe = {"seed": self.snap_seed, "noops": n} if self.snap_seed is not None else None
            for _ in range(n):
                obs, _r, term, trunc, info = self.env.step(0)
                if term or trunc:           # shouldn't happen; fall back to a real reset
                    obs, info = self.env.reset(seed=self.seed + self.games)
                    self.recipe = {"seed": int(self.seed + self.games), "noops": 0}
                    break
        if getattr(self.env, "reboot", False):
            # games that are switched off and on between games (their save-states don't restore) carry a little of
            # the last game into the next one, so they can't be re-run move for move: no recipe = replay the screens
            self.recipe = None
        self.score, self.lives, self.decisions = float(info.get("score", 0)), int(info.get("lives", 0) or 0), 0
        self._miss_seen = int(info.get("misses", 0) or 0)
        return obs[0]

    def start_from(self, recipe):
        """Start a game exactly the way a recorded one started (Watch window replays). Same seed + same delay +
        the same buttons afterwards = the same game, because the emulator is deterministic."""
        obs, info = self.env.reset(seed=int(recipe["seed"]))
        for _ in range(int(recipe.get("noops", 0))):
            obs, _r, term, trunc, info = self.env.step(0)
        self.recipe, self.prev = dict(recipe), 0
        self.score, self.lives, self.decisions = float(info.get("score", 0)), int(info.get("lives", 0) or 0), 0
        return obs[0]

    def step(self, a):
        if self.sticky and self.rng.random() < self.sticky:
            a = self.prev                 # joystick "sticks": the previous move happens again (like ALE)
        self.prev = int(a)
        self.last_action = int(a)         # what actually happened (recorded for replays)
        self.last_probe = None
        if self.probe and self.probe_rng.random() < 1.0 / PROBE_EVERY:
            try:
                self.last_probe = self._probe(int(a))
            except Exception:
                self.last_probe = None
        obs, _r, term, trunc, info = self.env.step(int(a))
        new_score = float(info.get("score", self.score))
        pts, self.score = new_score - self.score, new_score
        self.lives = int(info.get("lives", self.lives) or 0)
        self.decisions += 1
        capped = self.decisions >= MAX_DECISIONS and not (term or trunc)
        # ALFRED 23 Oct 2026: misses this move ('A missed shot costs'), the game's tally when it ends (the accuracy
        # meter), and moments worth practising from ('Practise the hard parts')
        m = info.get("misses")
        if m is None:
            self.step_misses = 0
        else:
            self.step_misses, self._miss_seen = max(0, int(m) - self._miss_seen), int(m)
        if term or trunc or capped:
            self.game_stats = ({"shots": int(info["shots"]), "hits": int(info["hits"]), "misses": int(m or 0),
                                **_saucer_tally(info)}   # ALFRED 26 Oct 2026: saucer tally (measures only)
                               if info.get("hits") is not None and info.get("shots") is not None else None)
        elif self.practice > 0:
            self._keep_moment()
        return obs[0], pts, bool(term), bool(trunc or capped)

    def _probe(self, a):
        """The button check: does one input change anything from here? The move it's making (its direction) with the
        input added, against the same move without it - three quarters of a second of screen each - then back to
        exactly where it was. The move without it is played twice: if those two don't match, this moment doesn't replay
        the same way and the try doesn't count. Returns (input, changed) - changed is None for a try that doesn't count."""
        host = self.env.host
        c = self.probe[int(self.probe_rng.integers(len(self.probe)))]
        base = tuple(sorted(d for d in self.env.actions[a] if d in _DIRS and (c not in _DIRS or d not in (c, _OPPOSITE[c]))))
        plus = tuple(sorted(set(base) | {c}))
        st, frame = host.save_state(), host.frame
        seen = []
        try:
            for inputs in (base, plus, base):
                host.load_state(st)
                host.frame = None
                h = []
                for k in range(PROBE_FRAMES):
                    f = host.run(inputs if k < PROBE_HOLD else (), capture=True)
                    h.append(None if f is None else hash(f.tobytes()))
                seen.append(h)
        finally:
            host.load_state(st)
            host.frame = frame
        if seen[0] != seen[2]:
            return c, None
        return c, seen[0] != seen[1]


def _worker(conn, pack_id, rom_path, runtime_root, seed, sticky, probe=None, practice=0.0):
    try:
        cab = Cabinet(pack_id, rom_path, runtime_root, seed, sticky, probe=probe, practice=practice)
        obs = cab.new_game()
        conn.send(("ok", obs.tobytes(), cab.lives, cab.recipe))
        while True:
            a = conn.recv()
            if a is None:
                break
            obs, pts, term, trunc = cab.step(a)
            lives = cab.lives
            acted = cab.last_action
            rs = cab.pop_restart()        # a practice game that began at the end of the last move (ALFRED 23 Oct)
            stats = cab.game_stats if (term or trunc) else None
            if term or trunc:
                obs = cab.new_game()      # same-step reset: obs is the first frame of the next game
            conn.send(("ok", obs.tobytes(), pts, lives, term, trunc, acted, cab.recipe if (term or trunc) else None,
                       cab.last_probe, cab.step_misses, stats, rs))
    except Exception:
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass


class _Space:
    def __init__(self, n):
        self.n = n


class ArcadeVec:
    """N cabinets in N processes, shaped like ale_py's AtariVectorEnv for trainer.py."""

    def __init__(self, pack_id, rom_path, num_envs, seed=1, sticky=0.25, probe=None, practice=0.0):
        self.practice = float(practice or 0)       # ALFRED 23 Oct 2026: 'Practise the hard parts'
        if pack_id not in PACKS:
            refresh_found()
        if pack_id not in PACKS:
            raise FileNotFoundError(f"The game pack '{pack_id}' is missing from files\\arcade\\found (its pack.json is gone). "
                                    "Press 'Find the score' for this game again to rebuild it.")
        self.sticky = sticky
        self.probe = [int(c) for c in (probe or [])]
        self.N, self.pack = int(num_envs), PACKS[pack_id]
        self.single_action_space = _Space(self.pack["n_actions"])
        self.runtime = scratch_dir("run")
        local_core(self.runtime, self.pack.get("core", "fbneo"))
        local_rom = self.runtime / "roms" / Path(rom_path).name     # same reason: read the game from local disk
        local_rom.parent.mkdir(exist_ok=True)
        shutil.copy2(rom_path, local_rom)
        rom_path = local_rom
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        # Keep workers light: hide the main script from spawn so children don't re-import it (and torch with it).
        main = sys.modules.get("__main__")
        saved = main.__dict__.pop("__file__", None) if main else None
        try:
            self._start(ctx, pack_id, rom_path, seed)
        finally:
            if saved is not None:
                main.__file__ = saved
        self.stack = np.zeros((self.N, 4, 84, 84), np.uint8)
        self.lives = np.zeros(self.N, np.int64)
        self.recipes = [None] * self.N     # how each cabinet's current game was started (for replays)

    def _start(self, ctx, pack_id, rom_path, seed):
        for i in range(self.N):
            a, b = ctx.Pipe()
            pr = ctx.Process(target=_worker, args=(b, pack_id, str(rom_path), str(self.runtime), seed + i * 1_000_003, self.sticky, self.probe,
                                                         self.practice),
                             daemon=True)
            pr.start()
            self.conns.append(a)
            self.procs.append(pr)

    def _recv(self, c):
        msg = c.recv()
        if msg[0] == "error":
            self.close()
            raise RuntimeError("An arcade cabinet crashed:\n" + msg[1])
        return msg

    def reset(self, seed=None):
        for i, c in enumerate(self.conns):
            msg = self._recv(c)
            ob, lives = msg[1], msg[2]
            self.recipes[i] = msg[3] if len(msg) > 3 else None
            self.stack[i] = 0
            self.stack[i, -1] = np.frombuffer(ob, np.uint8).reshape(84, 84)
            self.lives[i] = lives
        return self.stack.copy(), {"lives": self.lives.copy()}

    def step(self, actions):
        for c, a in zip(self.conns, actions):
            c.send(int(a))
        rew = np.zeros(self.N, np.float64)
        term = np.zeros(self.N, bool)
        trunc = np.zeros(self.N, bool)
        acted = np.asarray(actions, np.int64).copy()
        ended_recipes = [None] * self.N
        probes = []
        misses = np.zeros(self.N, np.float32)      # ALFRED 23 Oct 2026: see Cabinet.step / _worker
        stats = [None] * self.N
        restart = np.full(self.N, -1.0)
        self.stack = np.roll(self.stack, -1, axis=1)
        for i, c in enumerate(self.conns):
            msg = self._recv(c)
            _, ob, pts, lives, te, tr = msg[:6]
            rew[i], term[i], trunc[i], self.lives[i] = pts, te, tr, lives
            if len(msg) > 6:
                acted[i] = msg[6]
            if len(msg) > 8 and msg[8] is not None:
                probes.append(msg[8])
            if len(msg) > 11:
                misses[i], stats[i], restart[i] = msg[9], msg[10], msg[11]
            if te or tr:
                self.stack[i] = 0
                ended_recipes[i] = self.recipes[i]           # the game that just ended
                self.recipes[i] = msg[7] if len(msg) > 7 else None
            self.stack[i, -1] = np.frombuffer(ob, np.uint8).reshape(84, 84)
        return self.stack.copy(), rew, term, trunc, {"lives": self.lives.copy(), "acted": acted,
                                                       "ended_recipes": ended_recipes, "probes": probes,
                                                       "misses": misses, "game_stats": stats, "restart_score": restart}

    def close(self):
        for c in self.conns:
            try:
                c.send(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()
        shutil.rmtree(self.runtime, ignore_errors=True)
