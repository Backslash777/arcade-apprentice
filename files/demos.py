"""Portable AI Arcade - LEARN FROM ME: your own play, recorded in the Watch window, as lessons for the AI.

Press L in the Watch window and you take over the controls. Every move you make is saved (the same 84x84 grey
screen the AI sees, plus the button you chose) in the AI's files\\demos folder. Next time it trains, it first
copies you ("behaviour cloning"), then keeps a gentle pull towards your moves that fades away over the first few
million moves while it learns to beat you on its own. The same trick OpenAI used for Minecraft (VPT, 2022) and
DeepMind for Montezuma's Revenge (learning from YouTube, 2018) - just with you instead of the internet.

Signed: Claude, 10 Oct 2026.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

FADE = 5_000_000          # moves of its own training over which your lessons fade out


def demo_dir(proj: Path) -> Path:
    return Path(proj) / "files" / "demos"


def save_demo(proj, frames, actions, starts, n_actions, game_id):
    """frames: list of 84x84 uint8 (the newest screen each move); actions: int per frame, -1 = context only
    (screens from just before you took over); starts: True where a fresh game begins."""
    labelled = sum(1 for a in actions if a >= 0)
    if labelled < 10:
        return None
    d = demo_dir(proj)
    d.mkdir(parents=True, exist_ok=True)
    stamp, k = time.strftime('%Y%m%d-%H%M%S'), 0
    path = d / f"demo-{stamp}.npz"
    while path.exists():                       # two saves in the same second (a quick game over, then L)
        k += 1
        path = d / f"demo-{stamp}-{k}.npz"
    np.savez_compressed(path, frames=np.asarray(frames, np.uint8), actions=np.asarray(actions, np.int16),
                        starts=np.asarray(starts, bool), n_actions=np.int32(n_actions), game_id=str(game_id))
    return path, labelled


def load_demos(proj, n_actions):
    """All saved demos that fit this brain. Returns (frames (M,84,84), actions (M,), starts (M,)) or None."""
    fs, acts, sts = [], [], []
    for p in sorted(demo_dir(proj).glob("demo-*.npz")) if demo_dir(proj).exists() else []:
        try:
            z = np.load(p)
            if int(z["n_actions"]) != int(n_actions):
                continue
            f, a, s = z["frames"], z["actions"].astype(np.int64), z["starts"].copy()
            s[0] = True                            # never stitch two recordings together
            fs.append(f), acts.append(a), sts.append(s)
        except Exception:
            continue
    if not fs:
        return None
    return np.concatenate(fs), np.concatenate(acts), np.concatenate(sts)


def count_moves(proj):
    n = 0
    for p in sorted(demo_dir(proj).glob("demo-*.npz")) if demo_dir(proj).exists() else []:
        try:
            n += int((np.load(p)["actions"] >= 0).sum())
        except Exception:
            pass
    return n


class DemoBank:
    """Your moves on the training device, served as 4-screen stacks exactly like the AI's own inputs."""

    def __init__(self, data, dev):
        import torch
        frames, actions, starts = data
        M = len(frames)
        idx = np.zeros((M, 4), np.int64)
        for i in range(M):
            # screens i-3..i, but nothing from before the start of that game (blank instead, like a fresh game)
            first = i
            while first > 0 and not starts[first] and i - first < 3:
                first -= 1
            for k in range(4):
                j = i - 3 + k
                idx[i, k] = j if j >= first else M        # M = the blank screen appended below
        keep = np.flatnonzero(actions >= 0)
        self.frames = torch.from_numpy(np.concatenate([frames, np.zeros((1, 84, 84), np.uint8)])).to(dev)
        self.idx = torch.from_numpy(idx[keep]).to(dev)
        self.actions = torch.from_numpy(actions[keep]).to(dev)
        self.n = len(keep)
        self.dev = dev
        if self.frames.is_cuda:   # ALFRED 21 Oct 2026: the learning thread may read this bank on ANOTHER card stream
            torch.cuda.current_stream(self.frames.device).synchronize()   # - the copies must have landed first

    def _used_on(self, stream):
        """Tell the memory manager these live on this stream too (freed only once its work is done)."""
        for t in (self.frames, self.idx, self.actions):
            t.record_stream(stream)

    def sample(self, n):
        import torch
        if self.frames.is_cuda:
            self._used_on(torch.cuda.current_stream(self.frames.device))
        i = torch.randint(0, self.n, (min(n, self.n),), device=self.dev)
        return self.frames[self.idx[i]], self.actions[i]

    def all_batches(self, bs):
        import torch
        if self.frames.is_cuda:
            self._used_on(torch.cuda.current_stream(self.frames.device))
        perm = torch.randperm(self.n, device=self.dev)
        for k in range(0, self.n, bs):
            i = perm[k:k + bs]
            yield self.frames[self.idx[i]], self.actions[i]


def keys_to_action(table, wanted):
    """Pick the action whose buttons best match what you're holding. table: list of button-tuples (None = skip)."""
    wanted = set(wanted)
    best, key = 0, None
    for i, t in enumerate(table):
        if t is None:
            continue
        t = set(t)
        k = (t != wanted, -len(t & wanted) + 2 * len(t - wanted), len(t))
        if key is None or k < key:
            best, key = i, k
    return best
