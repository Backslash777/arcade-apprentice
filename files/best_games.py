"""Portable AI Arcade - LEARN FROM ITS OWN BEST GAMES (and the Hall of Fame you can watch).

While it trains, every game it plays is kept in memory until the game ends. A game that beats its current average
(over its last 100 games) joins a short queue - 5 games at most; a better game bumps the weakest one off. Each queued
game is studied a set number of times (the "times each best game is studied" dial, normally once): a little in
every batch, the brain is nudged towards the buttons it pressed in that game. Then the game is thrown away, so it
never gets stuck copying one lucky run. Because only games better than its own average get in, the bar rises as
it improves. This is "self-imitation learning" (Oh et al., 2018).

Every game that sets a new record is also saved to the Hall of Fame (files\\best_games\\hall_of_fame) - for you
to watch in the Watch window (press B), never used for learning again. Arcade games replay in full colour with
sound: the kit saves how the game was started and the buttons that actually happened, and the emulator plays it
out again move for move. The queued games are saved too (files\\best_games\\studying), so you can watch what it's
learning from right now; each one is deleted once it has been studied.

Signed: Claude, 13 Oct 2026.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

QUEUE_MAX = 5            # games waiting to be studied
WEIGHT = 0.1             # strength of the pull towards its best games (light: it learns from them, it doesn't parrot)
HOF_SCREENS_KEEP = 10    # the newest record games keep their screens too (a fall-back replay); buttons are kept for all
MIN_GAMES = 20           # it needs this many finished games before "better than its average" means anything
MOVE_BYTES = 84 * 84 + 4


def root(proj):
    return Path(proj) / "files" / "best_games"


def hof_dir(proj):
    return root(proj) / "hall_of_fame"


def studying_dir(proj):
    return root(proj) / "studying"


def memory_budget():
    """How much memory the games in progress may use: 40% of what's free, between 0.5 and 16 GB.
    ALFRED 21 Oct 2026: was a quarter, up to 8 GB - at 256 arcade machines that allowed 4,426 moves a game and
    S.Invaders Arcade's record games (~4,700 moves) no longer fitted. Now ~6,500 moves at 256 machines on a 64 GB PC."""
    try:
        import psutil
        free = psutil.virtual_memory().available
    except Exception:
        free = 4e9
    return int(min(16e9, max(5e8, free * 0.4)))


class Recorder:
    """Every game being played right now, one per cabinet, kept in memory until it ends.
    ALFRED 23 Oct 2026: a game longer than `cap` moves used to be thrown away whole - so once S.Invaders Arcade's games
    grew past ~6,500 moves (22 Oct, ~170M moves) its record games stopped reaching the Hall of Fame and its best games
    stopped being studied. Now every game keeps ALL its buttons (tiny - enough to re-run it in the emulator) and its
    last `cap` screens (the memory budget is the same), so long games are studied from their last stretch."""

    def __init__(self, n_envs, budget_bytes):
        self.n = int(n_envs)
        self.cap = max(1000, int(budget_bytes // (self.n * MOVE_BYTES)))     # screens per game it can hold
        self.f = [deque(maxlen=self.cap) for _ in range(self.n)]   # the newest screen the brain saw, each move
        self.a = [[] for _ in range(self.n)]     # the button the brain chose
        self.x = [[] for _ in range(self.n)]     # the button that actually happened (sticky joystick)
        self.too_long = np.zeros(self.n, bool)   # (kept for anything that looked at it; no longer used)

    def add(self, obs, chosen, acted):
        newest = np.asarray(obs)[:, -1]
        for i in range(self.n):
            self.f[i].append(newest[i].copy())  # past `cap` moves the oldest screen drops off the front
            self.a[i].append(int(chosen[i]))
            self.x[i].append(int(acted[i]))

    def finish(self, i):
        """The game in slot i has ended: hand it over (screens = its last `cap` moves) and start afresh."""
        game = None
        if self.a[i]:
            game = (np.stack(self.f[i]), np.asarray(self.a[i], np.int16), np.asarray(self.x[i], np.int16))
        self.f[i], self.a[i], self.x[i] = deque(maxlen=self.cap), [], []
        return game


class Studier:
    """The queue of best games, the lessons drawn from them, and the Hall of Fame."""

    def __init__(self, proj, dev, times, n_actions, game_id, game_name, say):
        self.proj, self.dev, self.times = Path(proj), dev, max(1, int(times))
        self.n_actions, self.game_id, self.game_name, self.say = int(n_actions), game_id, game_name, say
        self.lock = threading.Lock()
        self.queue = []                         # {"id", "score", "bank", "left", "moves", "file"}
        self.studied = self.offered = self.records = 0
        self.disk = ThreadPoolExecutor(max_workers=1)    # saving to disk never holds up the games
        self.rng = np.random.default_rng()
        sd = studying_dir(proj)
        sd.mkdir(parents=True, exist_ok=True)
        hof_dir(proj).mkdir(parents=True, exist_ok=True)
        for p in sd.glob("*.npz"):              # left over from last time: those lessons were never finished
            p.unlink(missing_ok=True)

    # ---- the queue
    def offer(self, game, score, average, recipe, step):
        """A game has just ended. Keep it if it beats its average and makes the top 5. Returns True if kept."""
        if game is None or not np.isfinite(average) or score <= average:
            return False
        frames, chosen, acted = game
        if len(frames) == 0:
            return False
        chosen = chosen[-len(frames):]          # a long game: study the stretch whose screens were kept (its end)
        import demos
        with self.lock:
            if len(self.queue) >= QUEUE_MAX:
                worst = min(self.queue, key=lambda q: q["score"])
                if score <= worst["score"]:
                    return False
                self.queue.remove(worst)
                self._drop_file(worst)
            starts = np.zeros(len(chosen), bool)
            starts[0] = True
            bank = demos.DemoBank((frames, chosen.astype(np.int64), starts), self.dev)
            gid = f"{time.strftime('%Y%m%d-%H%M%S')}-{int(score)}-{self.rng.integers(1000)}"
            item = {"id": gid, "score": float(score), "bank": bank, "left": self.times * bank.n, "moves": bank.n,
                    "file": studying_dir(self.proj) / f"study-{gid}.npz"}
            self.queue.append(item)
            self.offered += 1
        self.disk.submit(self._save, item["file"], frames, chosen, acted, recipe, score, step, "studying")
        return True

    def sample(self, n):
        """Moves to learn from in this batch: drawn from the queued games in proportion to what each has left."""
        import torch
        with self.lock:
            live = [q for q in self.queue if q["left"] > 0]
            if not live:
                return None
            left = np.array([q["left"] for q in live], float)
            counts = self.rng.multinomial(n, left / left.sum())
            xs, ys = [], []
            for q, c in zip(live, counts):
                if c <= 0:
                    continue
                xb, ab = q["bank"].sample(int(c))
                xs.append(xb)
                ys.append(ab)
                q["left"] -= int(c)
                if q["left"] <= 0:              # studied enough: throw it away
                    self.queue.remove(q)
                    self.studied += 1
                    self._drop_file(q)
        if not xs:
            return None
        return torch.cat(xs), torch.cat(ys)

    def status(self):
        with self.lock:
            return f"best games: {len(self.queue)} queued, {self.studied} studied"

    def _drop_file(self, item):
        self.disk.submit(lambda p=item["file"]: p.unlink(missing_ok=True))

    # ---- the Hall of Fame
    def record(self, game, score, recipe, step):
        """A new record: keep it for watching (never for learning again)."""
        if game is None:
            return
        frames, chosen, acted = game
        self.records += 1
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = hof_dir(self.proj) / f"record-{stamp}-{int(score)}.npz"
        self.disk.submit(self._save, path, frames, chosen, acted, recipe, score, step, "record")
        self.disk.submit(self._prune_screens)

    def _save(self, path, frames, chosen, acted, recipe, score, step, kind):
        try:
            meta = {"score": float(score), "step": int(step), "kind": kind, "when": time.strftime("%Y-%m-%d %H:%M"),
                    "game_id": self.game_id, "game_name": self.game_name, "n_actions": self.n_actions,
                    "recipe": recipe, "moves": int(len(acted)), "screens_from": int(len(acted) - len(frames))}
            tmp = path.with_name(path.stem + ".part.npz")
            np.savez_compressed(tmp, frames=frames, chosen=chosen, acted=acted, meta=json.dumps(meta))
            tmp.replace(path)
        except Exception as e:
            try:
                self.say(f"(Couldn't save a best game to disk: {e!r} - training carries on.)")
            except Exception:
                pass

    def _prune_screens(self):
        """Screens take room; buttons don't. Only the newest record games keep their screens."""
        files = sorted(hof_dir(self.proj).glob("record-*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[HOF_SCREENS_KEEP:]:
            try:
                z = np.load(p)
                if z["frames"].size == 0 or json.loads(str(z["meta"])).get("recipe") is None:
                    continue                    # no recipe = the screens ARE the replay: always keep them
                meta = str(z["meta"])
                chosen, acted = z["chosen"], z["acted"]
                z.close()
                tmp = p.with_name(p.stem + ".part.npz")
                np.savez_compressed(tmp, frames=np.zeros((0, 84, 84), np.uint8), chosen=chosen, acted=acted, meta=meta)
                tmp.replace(p)
            except Exception:
                pass

    def close(self):
        self.disk.shutdown(wait=True)


def hof_best(proj):
    """ALFRED 23 Oct 2026: the best score kept in the Hall of Fame (0 if none), from the file names
    (record-<date>-<time>-<score>.npz). New records are judged against THIS, not the scoreboard's best game: a record
    too long to keep used to raise the bar anyway, so S.Invaders Arcade's records from 5,420 up to 9,650 were never
    kept and the 'Best game' button kept showing the 5,420."""
    best = 0.0
    d = hof_dir(proj)
    for p in d.glob("record-*.npz") if d.exists() else []:
        try:
            best = max(best, float(p.stem.rsplit("-", 1)[1]))
        except (ValueError, IndexError):
            pass
    return best


def list_replays(proj):
    """Everything there is to watch, best first: Hall of Fame records, then the games being studied now.
    Each entry: {"path", "label", "meta"}."""
    out = []
    for kind, d in (("record", hof_dir(proj)), ("studying", studying_dir(proj))):
        items = []
        for p in d.glob("*.npz") if d.exists() else []:
            if p.name.endswith(".part.npz"):
                continue
            try:
                with np.load(p) as z:
                    meta = json.loads(str(z["meta"]))
                    meta["has_screens"] = bool(z["frames"].size)
            except Exception:
                continue
            items.append({"path": p, "meta": meta})
        items.sort(key=lambda e: -e["meta"]["score"])
        for e in items:
            m = e["meta"]
            what = "RECORD" if kind == "record" else "BEING STUDIED"
            e["label"] = f"{what}: {m['score']:,.0f} points ({m['when']}, brain at {m['step'] / 1e6:.1f}M moves)"
            out.append(e)
    return out


def load_replay(path):
    with np.load(path) as z:
        return {"frames": z["frames"].copy(), "acted": z["acted"].astype(np.int64), "meta": json.loads(str(z["meta"]))}