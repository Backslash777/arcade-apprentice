"""Portable AI Arcade - GO-EXPLORE for Atari 2600 games (Ecoffet et al., "First return, then explore", Nature 2021).

The hard part of games like Pitfall isn't learning to play - it's ever getting far enough to find anything to learn
from. Go-Explore keeps a MAP (the "archive") of every distinct place the AI has reached, together with an emulator
save-state for each. When a game ends, instead of always starting again from the beginning, a game can be restarted
from a place on the map - preferring places that have rarely been explored from - and the AI explores onwards from
there. The brain is the same PPO brain: it learns from these frontier starts, so it learns how to get further.

  "place" (cell) = the screen shrunk to 11x8 pixels with 8 shades of grey (the paper's domain-agnostic cells)
  a place's save-state is replaced when it's reached with a higher score, or the same score in fewer moves

GoExploreVec runs one Atari cabinet per process (so restores/snapshots are possible) and looks like ale_py's
AtariVectorEnv to trainer.py. Signed: Claude, 9 Oct 2026.
"""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import pickle
import random
import sys
import time
import traceback
import zlib
from pathlib import Path

import numpy as np

MAX_DECISIONS = 27_000           # 108k frames = the standard 30-minute cap
MAX_CELLS = 30_000


class AtariCabinet:
    """One ALE game with AtariVectorEnv-style preprocessing: 4-frame skip, max of last 2 frames, 84x84 grey,
    sticky joystick, up to 30 no-op starts, FIRE to start where the game needs it."""

    def __init__(self, game_id, seed, sticky):
        from ale_py import ALEInterface, roms, LoggerMode
        ALEInterface.setLoggerMode(LoggerMode.Error)
        self.ale = ale = ALEInterface()
        ale.setInt("random_seed", int(seed) % (2**31))
        ale.setFloat("repeat_action_probability", float(sticky))
        ale.setInt("frame_skip", 1)
        ale.setInt("max_num_frames_per_episode", 108_000)
        ale.loadROM(roms.get_rom_path(game_id))
        self.actions = ale.getMinimalActionSet()
        self.fire = 1 if len(self.actions) > 1 and "FIRE" in str(self.actions[1]) else None
        self.rng = random.Random(seed)
        self.decisions = 0

    def _obs(self, frames):
        from PIL import Image
        img = np.maximum(frames[-1], frames[-2]) if len(frames) > 1 else frames[-1]
        return np.asarray(Image.fromarray(img).resize((84, 84), Image.BOX), dtype=np.uint8)

    def new_game(self):
        ale = self.ale
        ale.reset_game()
        for _ in range(self.rng.randint(0, 30)):
            ale.act(self.actions[0])
        if self.fire is not None:
            ale.act(self.actions[self.fire])
        self.decisions = 0
        g = ale.getScreenGrayscale()
        return self._obs([g, g])

    def step(self, a):
        ale, pts, frames = self.ale, 0.0, []
        for k in range(4):
            pts += ale.act(self.actions[int(a)])
            if k >= 2:
                frames.append(ale.getScreenGrayscale())
            if ale.game_over():
                break
        if not frames:
            frames = [ale.getScreenGrayscale()]
        self.decisions += 1
        term = ale.game_over()
        trunc = (not term) and self.decisions >= MAX_DECISIONS
        return self._obs(frames), pts, bool(term), bool(trunc)

    def cell_key(self):
        """The 'place': screen shrunk to 11x8, 8 grey levels -> 64-bit fingerprint."""
        from PIL import Image
        small = np.asarray(Image.fromarray(self.ale.getScreenGrayscale()).resize((11, 8), Image.BOX)) >> 5
        return int.from_bytes(hashlib.blake2b(small.astype(np.uint8).tobytes(), digest_size=8).digest(), "little")

    def save(self):
        return self.ale.cloneState()

    def restore(self, blob, decisions):
        self.ale.restoreState(pickle.loads(zlib.decompress(blob)))
        self.decisions = decisions
        g = self.ale.getScreenGrayscale()
        return self._obs([g, g])


def _worker(conn, game_id, seed, sticky):
    try:
        cab = AtariCabinet(game_id, seed, sticky)
        obs = cab.new_game()
        last = cab.save()
        conn.send(("ok", obs.tobytes(), cab.ale.lives(), cab.cell_key(), len(cab.actions)))
        while True:
            msg = conn.recv()
            if msg is None:
                break
            a, restore, want = msg
            blob = zlib.compress(pickle.dumps(last), 1) if want else None   # the snapshot of the place we reported
            restored = False
            if restore is not None:
                cab.restore(*restore)
                restored = True
            obs, pts, term, trunc = cab.step(a)
            lives = cab.ale.lives()
            if term or trunc:
                obs = cab.new_game()
                restored = False
            key = cab.cell_key()
            last = cab.save()
            conn.send(("ok", obs.tobytes(), pts, lives, term, trunc, key, blob, restored))
    except Exception:
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass


class Archive:
    """The map of places reached. Each place: best score/shortest route there + a save-state to return to."""

    def __init__(self):
        self.cells = {}            # key -> [score, length, seen, chosen, blob]
        self.best_score = 0.0
        self.returns = 0

    def consider(self, key, score, length):
        """True if we want this place's save-state (new place, or reached it better than before)."""
        c = self.cells.get(key)
        if c is None:
            return True
        c[2] += 1
        return score > c[0] or (score == c[0] and length < c[1])

    def store(self, key, score, length, blob):
        c = self.cells.get(key)
        if c is None:
            self.cells[key] = [score, length, 1, 0, blob]
        elif score > c[0] or (score == c[0] and length < c[1]):
            c[0], c[1], c[4] = score, length, blob
            c[3] = 0                           # improved: worth exploring from again
        self.best_score = max(self.best_score, score)
        if len(self.cells) > MAX_CELLS:
            self._trim()

    def _trim(self):
        items = sorted(self.cells.items(), key=lambda kv: (kv[1][0], -kv[1][2]))   # low score, most visited first
        for k, _ in items[: len(items) // 10]:
            del self.cells[k]

    def choose(self, rng):
        """Pick a place to restart from: rarely-chosen, rarely-seen places first (Go-Explore's count weights),
        with a nudge towards higher-scoring places."""
        if not self.cells:
            return None
        keys = list(self.cells)
        v = np.array([self.cells[k][:4] for k in keys], dtype=np.float64)
        w = 1.0 / np.sqrt(v[:, 3] + 1.0) + 0.5 / np.sqrt(v[:, 2] + 1.0)
        if v[:, 0].max() > v[:, 0].min():
            w *= 1.0 + 0.5 * (v[:, 0] - v[:, 0].min()) / (v[:, 0].max() - v[:, 0].min())
        k = keys[rng.choices(range(len(keys)), weights=w.tolist())[0]]
        c = self.cells[k]
        c[3] += 1
        self.returns += 1
        return k, c

    def save(self, path):
        tmp = Path(str(path) + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"cells": self.cells, "best": self.best_score, "returns": self.returns}, f, protocol=4)
        tmp.replace(path)

    def load(self, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.cells, self.best_score, self.returns = d["cells"], d.get("best", 0.0), d.get("returns", 0)


class _Space:
    def __init__(self, n):
        self.n = n


class GoExploreVec:
    """N Atari cabinets in N processes + the shared map. Same interface as AtariVectorEnv for trainer.py, plus
    info["restart_score"] (>=0 where a game restarted from the map this step, the score it restarted on)."""

    def __init__(self, game_id, num_envs, seed=1, sticky=0.25, p_return=0.5, archive_path=None):
        self.N, self.p_return = int(num_envs), float(p_return)
        self.archive = Archive()
        self.archive_path = archive_path
        if archive_path and Path(archive_path).exists():
            try:
                self.archive.load(archive_path)
            except Exception:
                pass
        self.rng = random.Random(seed)
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        main = sys.modules.get("__main__")
        saved = main.__dict__.pop("__file__", None) if main else None
        try:
            for i in range(self.N):
                a, b = ctx.Pipe()
                pr = ctx.Process(target=_worker, args=(b, game_id, seed + i * 7919, sticky), daemon=True)
                pr.start()
                self.conns.append(a)
                self.procs.append(pr)
        finally:
            if saved is not None:
                main.__file__ = saved
        self.first = [self._recv(c) for c in self.conns]          # each cabinet's opening screen
        self.single_action_space = _Space(self.first[0][4])
        self.stack = np.zeros((self.N, 4, 84, 84), np.uint8)
        self.lives = np.zeros(self.N, np.int64)
        self.score = np.zeros(self.N)          # score since the true start of the game (restarts carry their score)
        self.length = np.zeros(self.N, np.int64)
        self.key = [0] * self.N
        self.want = [False] * self.N
        self.pending = [None] * self.N
        self.restore = [None] * self.N

    def _recv(self, c):
        msg = c.recv()
        if msg[0] == "error":
            self.close()
            raise RuntimeError("A game cabinet crashed:\n" + msg[1])
        return msg

    def reset(self, seed=None):
        for i, msg in enumerate(self.first):
            _, ob, lives, key, _n = msg
            self.stack[i] = 0
            self.stack[i, -1] = np.frombuffer(ob, np.uint8).reshape(84, 84)
            self.lives[i], self.key[i] = lives, key
            self.want[i] = self.archive.consider(key, 0.0, 0)
            self.pending[i] = (key, 0.0, 0)
        return self.stack.copy(), {"lives": self.lives.copy()}

    def step(self, actions):
        for i, (c, a) in enumerate(zip(self.conns, actions)):
            c.send((int(a), self.restore[i], self.want[i]))
        restarted = np.full(self.N, -1.0)
        rew = np.zeros(self.N)
        term = np.zeros(self.N, bool)
        trunc = np.zeros(self.N, bool)
        self.stack = np.roll(self.stack, -1, axis=1)
        for i, c in enumerate(self.conns):
            _, ob, pts, lives, te, tr, key, blob, restored = self._recv(c)
            if blob is not None and self.pending[i] is not None:
                self.archive.store(*self.pending[i], blob)
            if self.restore[i] is not None and restored:
                restarted[i] = self.score[i]           # score of the place it restarted from (set below last step)
            self.restore[i] = None
            rew[i], term[i], trunc[i], self.lives[i] = pts, te, tr, lives
            if te or tr or restarted[i] >= 0:
                self.stack[i] = 0
            self.stack[i, -1] = np.frombuffer(ob, np.uint8).reshape(84, 84)
            if te or tr:
                self.score[i], self.length[i] = 0.0, 0
                if self.p_return > 0 and self.rng.random() < self.p_return:
                    pick = self.archive.choose(self.rng)
                    if pick is not None:
                        k, cell = pick
                        self.restore[i] = (cell[4], int(cell[1]))
                        self.score[i], self.length[i] = cell[0], cell[1]   # carried into the restarted game
            else:
                self.score[i] += pts
                self.length[i] += 1
            self.want[i] = self.archive.consider(key, self.score[i], int(self.length[i]))
            self.pending[i] = (key, float(self.score[i]), int(self.length[i])) if self.want[i] else None
        return self.stack.copy(), rew, term, trunc, {"lives": self.lives.copy(), "restart_score": restarted}

    def stats(self):
        return {"places": len(self.archive.cells), "map_best": self.archive.best_score, "returns": self.archive.returns}

    def save(self):
        if self.archive_path:
            try:
                self.archive.save(self.archive_path)
            except Exception:
                pass

    def close(self):
        self.save()
        for c in self.conns:
            try:
                c.send(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()
