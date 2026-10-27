"""Portable AI Arcade - read a game's score off the SCREEN, the way a person does. No peeking in its memory.

How it works (no human, no memory):
  1. Watch a button-masher play. Pixels that never change are the background and labels; pixels that change all
     the time are the game. The score sits in between: it changes now and then, all at the same moments.
  2. Cut those places into characters (old games draw text on a fixed grid, so every "3" is pixel-identical).
  3. Work out which character is which digit. The display starts at 0, and a score only ever goes up - so every
     time it changes, the digit that changed must have got bigger. That gives an order between the characters.
     A little digit-recognising brain (digits.npz, trained on hundreds of computer fonts) breaks any ties.
  4. Keep the display whose numbers really behave like a score, and save how to read it in the game's pack.
Then GenericArcadeEnv reads the score from each frame with ScreenScore(spec).read(frame).

Signed: Claude, 11 Oct 2026.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIDE = 16                     # the digit brain looks at characters scaled to 16x16
_NET = None


# ------------------------------------------------------------------------------------------------ small helpers
def cids(frame):
    """A frame (H, W, 3) as one colour number per pixel (16-bit)."""
    f = np.asarray(frame)[..., :3].astype(np.uint16)
    return ((f[..., 0] >> 3) << 11) | ((f[..., 1] >> 2) << 5) | (f[..., 2] >> 3)


def key_of(mask):
    return hashlib.blake2s(np.packbits(mask).tobytes() + bytes(mask.shape), digest_size=6).hexdigest()


def runs(b):
    """[start, end) of each run of True in a 1-D bool array."""
    d = np.diff(np.concatenate(([0], b.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def norm_glyph(mask):
    """Crop a character to its ink and scale it into a SIDE x SIDE square (keeping its shape)."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return np.zeros((SIDE, SIDE), np.float32)
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    h, w = m.shape
    s = (SIDE - 2) / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    yi = np.minimum((np.arange(nh) + 0.5) / s, h - 1e-3).astype(int)
    xi = np.minimum((np.arange(nw) + 0.5) / s, w - 1e-3).astype(int)
    out = np.zeros((SIDE, SIDE), np.float32)
    oy, ox = (SIDE - nh) // 2, (SIDE - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = m[yi][:, xi]
    return out


def digit_logp(mask):
    """log-probabilities of 0-9 for one character, from the little digit brain (uniform if it's missing)."""
    global _NET
    if _NET is None:
        p = HERE / "digits.npz"
        _NET = dict(np.load(p)) if p.exists() else {}
    if not _NET:
        return np.full(10, -np.log(10.0))
    x = norm_glyph(mask).reshape(-1)
    for i in range(10):
        if f"w{i}" not in _NET:
            break
        x = x @ _NET[f"w{i}"] + _NET[f"b{i}"]
        if f"w{i + 1}" in _NET:
            x = np.maximum(x, 0)
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


# ------------------------------------------------------------------------------------------------ reading a box
def tokens(fg, gap):
    """Characters in one frame of a box: [(x0, x1, mask)], split on empty columns."""
    return [(a, b, fg[:, a:b]) for a, b in runs(fg.any(0))]


def words(toks, gap):
    out, cur = [], []
    for t in toks:
        if cur and t[0] - cur[-1][1] > gap:
            out.append(cur)
            cur = []
        cur.append(t)
    if cur:
        out.append(cur)
    return out


class ScreenScore:
    """Reads the number at one place on the screen. spec comes from fit() and lives in the pack."""

    def __init__(self, spec):
        self.s = spec
        self.y0, self.y1, self.x0, self.x1 = spec["box"]
        self.fg = spec["fg"]
        self.keys = {k: int(v) for k, v in spec["keys"].items()}
        self.edge, self.gap = spec["edge"], spec["gap"]
        self.cache = {}

    def digit(self, mask):
        k = key_of(mask)
        if k in self.keys:
            return self.keys[k]
        if k not in self.cache:          # a character it never saw while learning (e.g. the first 9): ask the brain
            lp = digit_logp(mask)
            self.cache[k] = int(lp.argmax()) if lp.max() > np.log(0.6) and mask.sum() >= 3 else None
        return self.cache[k]

    def read(self, frame):
        """The number shown, or None if it can't be read right now (hidden, blinking, something on top of it)."""
        frame = np.asarray(frame)
        if frame.shape[0] < self.y1 or frame.shape[1] < self.x1:
            return None
        crop = frame[self.y0:self.y1, self.x0:self.x1]
        fg = (crop if crop.ndim == 2 else cids(crop)) == self.fg
        for w in words(tokens(fg, self.gap), self.gap):
            if abs(w[-1][1] - self.edge) <= 2:
                v = 0
                for _, _, m in w:
                    d = self.digit(m)
                    if d is None:
                        return None
                    v = v * 10 + d
                return v * int(self.s.get("scale", 1))
        return 0 if self.s.get("blank_is_zero") else None


# ------------------------------------------------------------------------------------------------ learning
def _dilate(m, dy, dx):
    out = m.copy()
    for s in range(1, dx + 1):
        out[:, s:] |= m[:, :-s]
        out[:, :-s] |= m[:, s:]
    m2 = out.copy()
    for s in range(1, dy + 1):
        out[s:] |= m2[:-s]
        out[:-s] |= m2[s:]
    return out


def _components(m):
    """Bounding boxes (y0, y1, x0, x1) of the joined-up blobs in a bool image."""
    seen = np.zeros_like(m)
    H, W = m.shape
    out = []
    for y, x in zip(*np.nonzero(m)):
        if seen[y, x]:
            continue
        stack = [(y, x)]
        seen[y, x] = True
        y0 = y1 = y
        x0 = x1 = x
        while stack:
            cy, cx = stack.pop()
            y0, y1, x0, x1 = min(y0, cy), max(y1, cy), min(x0, cx), max(x1, cx)
            for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                if 0 <= ny < H and 0 <= nx < W and m[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        out.append((y0, y1 + 1, x0, x1 + 1))
    return out


def candidate_boxes(F, max_rate=0.06):
    """Places on the screen where one colour comes and goes now and then, but not all the time: maybe the digits of
    a number. Works colour by colour, so a number drawn over a scrolling background is still found."""
    sub = F[::2]
    T = len(sub)
    calm = (sub[1:] != sub[:-1]).mean((1, 2)) < 0.2       # ignore whole-screen changes (new level, screen going black)
    vals, cnts = np.unique(sub[:: max(1, T // 40)], return_counts=True)
    H, W = F.shape[1:]
    out = []
    for c in [int(v) for v in vals[np.argsort(-cnts)][:48]]:
        I = sub == c
        pres = I.mean(0)
        tc = (I[1:] != I[:-1])[calm].sum(0)
        cand = (pres > 0.08) & (tc >= 1) & (tc <= max(4, max_rate * T))
        if cand.sum() < 4:
            continue
        for y0, y1, x0, x1 in _components(_dilate(cand, 0, 4)):
            if not 4 <= y1 - y0 <= 26 or x1 - x0 > W * 0.6:
                continue
            # widen: painted-on zeros next to the number never change, so they aren't in cand
            out.append(((max(0, y0 - 1), min(H, y1 + 1), max(0, x0 - 26), min(W, x1 + 26)), c))
    return out


def _mode(a):
    v, c = np.unique(a, return_counts=True)
    return int(v[c.argmax()])


def displays_in_box(F, box, col):
    """Every number-like display in a box, drawn in one colour, one line of text at a time.
    Yields (box, colour, gap, right edge, [(t, (keys...), masks)])."""
    y0, y1, x0, x1 = box
    fga = F[:, y0:y1, x0:x1] == col
    for ya, yb in runs(fga.mean(0).max(1) > 0.02):
        if not 4 <= yb - ya <= 20:
            continue
        fg = fga[:, ya:yb]
        per, widths = [], []
        for t in range(len(fg)):
            toks = tokens(fg[t], 0)
            per.append(toks)
            widths += [b - a for a, b, _ in toks]
        if not widths:
            continue
        gap = max(3, int(np.median(widths) * 0.9))
        disp = {}
        for t, toks in enumerate(per):
            for w in words(toks, gap):
                if len(w) > 9:
                    continue
                e = int(w[-1][1])
                k = next((x for x in disp if abs(x - e) <= 1), e)
                disp.setdefault(k, []).append((t, tuple(key_of(mm) for _, _, mm in w), [mm for _, _, mm in w]))
        for e, seq in disp.items():
            yield (y0 + ya, y0 + yb, x0, x1), col, gap, e, seq


def _solve(nonzero, cons, logp):
    """Best digit (1-9) for each non-zero character: obeys every "a < b" in cons, uses each digit once."""
    order = sorted(nonzero, key=lambda k: -sum(k in c for c in cons))
    fav = [max(logp[k][1:]) for k in order]
    rest = [sum(fav[i:]) for i in range(len(order) + 1)]      # best that the remaining characters could add
    best = [None, -1e18]
    assign = {}

    def ok(k, d):
        for a, b in cons:
            if a == k and b in assign and not d < assign[b]:
                return False
            if b == k and a in assign and not assign[a] < d:
                return False
        return True

    def dfs(i, used, score):
        if score + rest[i] <= best[1]:
            return
        if i == len(order):
            best[0], best[1] = dict(assign), score
            return
        k = order[i]
        for d in sorted(range(1, 10), key=lambda d: -logp[k][d]):
            if d in used or not ok(k, d):
                continue
            assign[k] = d
            dfs(i + 1, used | {d}, score + logp[k][d])
            del assign[k]
    dfs(0, frozenset(), 0.0)
    return best[0]


def fit_display(seq, logp_of):
    """Work out the digits of one display from what it showed over time. Returns (keys->digit, report) or None."""
    if len(seq) < 10:
        return None
    count = {}
    for _, ks, _ in seq:
        for k in ks:
            count[k] = count.get(k, 0) + 1
    good = [(t, ks, ms) for t, ks, ms in seq if all(count[k] >= 3 for k in ks)]
    if len(good) < 10:
        return None
    # runs of the same reading
    rl = []
    for t, ks, ms in good:
        if not rl or rl[-1][1] != ks:
            rl.append([t, ks, ms, 1])
        else:
            rl[-1][3] += 1
    if len(rl) < 3:
        return None
    first = rl[0][1]
    if len(set(first)) != 1:
        return None                      # a score starts at 0 (or 00, 000...): one character, repeated
    zero = first[0]
    allkeys = {k for _, ks, _, _ in rl for k in ks}
    if len(allkeys) > 10:
        return None
    cons, ups, resets = set(), 0, 0
    for (_, a, _, _), (_, b, _, _) in zip(rl, rl[1:]):
        if all(k == zero for k in b):
            resets += 1                   # back to 0: a new game
            continue
        if len(b) < len(a):
            return None                   # a score never loses digits (except back to 0)
        if len(b) > len(a):
            if b[0] == zero:
                return None
            ups += 1
            continue
        p = next(i for i in range(len(a)) if a[i] != b[i])
        if b[p] == zero:
            return None                   # the changed digit went down to 0 - not a score
        if b[p] == a[p]:
            continue
        if a[p] != zero:
            cons.add((a[p], b[p]))        # that digit got bigger
        ups += 1
    if ups < 2 or resets > ups:
        return None
    nonzero = sorted(allkeys - {zero})
    if len(nonzero) > 9:
        return None
    masks = {}
    for _, ks, ms, _ in rl:
        for k, m in zip(ks, ms):
            masks.setdefault(k, m)
    logp = {k: logp_of(masks[k]) for k in allkeys}
    sol = _solve(nonzero, cons, logp)
    if sol is None:
        return None                       # the order is impossible (e.g. it counts down: a timer)
    sol[zero] = 0
    vals = []
    for _, ks, _, n in rl:
        v = 0
        for k in ks:
            v = v * 10 + sol[k]
        vals.append(v)
    # sanity: the readings must really go up (apart from new games)
    for a, b in zip(vals, vals[1:]):
        if b < a and b != 0:
            return None
    brain = sum(logp[k][sol[k]] for k in allkeys) / len(allkeys)
    return sol, {"ups": ups, "resets": resets, "digits_seen": len(allkeys), "rules": len(cons),
                 "brain_sure": round(float(np.exp(brain)), 3), "values": vals[:60], "zero": zero}


def fit(frames, logp_of=digit_logp):
    """Find the score on the screen from frames of play. Returns a spec for ScreenScore, or None.
    frames: list of RGB frames (all the same size) from one or more games, a few frames apart, starting at a
    game's start."""
    from collections import Counter
    shape = Counter(f.shape for f in frames).most_common(1)[0][0]
    F = np.stack([f if f.ndim == 2 else cids(f) for f in frames if f.shape == shape])
    best = None
    L = F[:: max(1, len(F) // 40)]
    boxes, seen_b = [], set()
    for b in _event_boxes(F, L) + candidate_boxes(F):
        if b not in seen_b:
            seen_b.add(b)
            boxes.append(b)
    for box, c in boxes:
        for bx, col, gap, edge, seq in displays_in_box(F, box, c):
            r = fit_display(seq, logp_of)
            if not r:
                continue
            sol, rep = r
            # a display the digit brain believes in, that rose the most (new games count against it)
            rank = (rep["brain_sure"] >= 0.5, rep["ups"] - 2 * rep["resets"], rep["brain_sure"])
            if best is None or rank > best[0]:
                best = (rank, {"kind": "screen", "box": [int(v) for v in bx], "fg": int(col), "edge": int(edge),
                               "gap": int(gap), "keys": {k: int(v) for k, v in sol.items()},
                               "frame_shape": list(shape[:2]), "report": rep,
                               "confident": bool(rep["brain_sure"] >= 0.6 and rep["ups"] >= 3
                                                  and rep["resets"] * 3 <= rep["ups"])})   # a counter that keeps wrapping round isn't a score
    return best[1] if best else None

# ------------------------------------------------------------------------------------------------ taught by the player
def _teach(lab, bx, col, gap, edge):
    """Do the typed numbers read back from this display? Learns each character's digit from them.
    Returns ((labels read, different values, -missed), keys, scale, read, missed) or None."""
    from collections import Counter
    y0, y1, x0, x1 = bx
    votes, per, scale = {}, [], None
    for f, v in lab:
        fg = f[y0:y1, x0:x1] == col
        w = next((w for w in words(tokens(fg, gap), gap) if abs(int(w[-1][1]) - edge) <= 2), None)
        s = str(v)
        if w is None or len(w) > 9:
            per.append(None)
            continue
        n, sc = len(w), 1
        if n > len(s):
            s = s.zfill(n)                       # leading zeros drawn, e.g. 000150
        elif n < len(s):
            tail = len(s) - n                    # zeros painted on in another colour, e.g. 18 + "00"
            if not s.endswith("0" * tail):
                per.append(None)
                continue
            s, sc = s[:n], 10 ** tail
        if scale is None:
            scale = sc
        if sc != scale:
            per.append(None)
            continue
        pairs = [(key_of(m), int(ch)) for (_, _, m), ch in zip(w, s)]
        per.append(pairs)
        for k, d in pairs:
            votes.setdefault(k, Counter())[d] += 1
    keys = {k: c.most_common(1)[0][0] for k, c in votes.items()}
    looks = Counter(keys.values())
    if looks and max(looks.values()) > 2:        # one digit drawn three different ways: not a font, just noise
        return None
    good = [v for (_, v), p in zip(lab, per) if p and all(keys[k] == d for k, d in p)]
    missed = len(lab) - len(good)
    if len(set(good)) < 2 or missed > (1 if len(lab) >= 3 else 0):
        return None
    return (len(good), len(set(good)), -missed), keys, scale or 1, len(good), missed


def _bands(cand, comp, W):
    """A blob too big to be one line of text (a whole status bar, or one touching the playfield): cut it into
    overlapping windows a line of text fits in, so the number inside still gets looked at."""
    y0, y1, x0, x1 = comp
    ys = [(y0, y1)] if y1 - y0 <= 26 else [(a, min(y1, a + 24)) for a in range(y0, max(y0 + 1, y1 - 12), 12)]
    ww = int(W * 0.4)
    xs = [(x0, x1)] if x1 - x0 <= W * 0.6 else [(a, min(x1, a + ww)) for a in range(x0, max(x0 + 1, x1 - ww // 2), ww // 2)]
    return [(a, b, c, d) for a, b in ys for c, d in xs]


def _boxes_from(cand, L, H, W):
    out = []
    if cand.sum() < 4:
        return out
    for comp in _components(_dilate(cand, 0, 4)):
        for y0, y1, x0, x1 in _bands(cand, comp, W):
            if not 3 <= y1 - y0 <= 26 or x1 - x0 > W * 0.6:
                continue
            box = (max(0, y0 - 1), min(H, y1 + 1), max(0, x0 - 26), min(W, x1 + 26))
            cols = L[:, y0:y1, x0:x1][:, cand[y0:y1, x0:x1]]
            if not cols.size:
                continue
            v, c = np.unique(cols, return_counts=True)
            keep = c >= max(2, 0.01 * c.sum())          # every colour drawn there now and then (digits are few pixels)
            out += [(box, int(col)) for col in v[keep][np.argsort(-c[keep])][:24]]
    return out


def _label_boxes(lab, around):
    """Where the typed numbers must be: pixels that sit still around each pause (a panel, not the game) but differ
    between two pauses with different scores."""
    L = np.stack([f for f, _ in lab])
    H, W = L.shape[1:]
    still = []
    for (f, _), near in zip(lab, around):
        st = np.ones((H, W), bool)
        for g in near:
            if g.shape == f.shape:
                st &= g == f
        still.append(st)
    cand = np.zeros((H, W), bool)
    for i in range(len(lab)):
        for j in range(i + 1, len(lab)):
            if lab[i][1] != lab[j][1]:
                cand |= still[i] & still[j] & (lab[i][0] != lab[j][0])
    return _boxes_from(cand, L, H, W)


def _change_boxes(F, L, max_rate=0.06):
    """Places where the picture changes now and then but not all the time, whatever the colour."""
    sub = F[::2]
    if len(sub) < 3:
        return []
    ch = sub[1:] != sub[:-1]
    calm = ch.mean((1, 2)) < 0.2
    tc = ch[calm].sum(0)
    cand = (tc >= 1) & (tc <= max(4, max_rate * len(sub)))
    H, W = F.shape[1:]
    return _boxes_from(cand, L, H, W)


def _event_boxes(F, L, max_rate=0.06, win=31, dense=0.35):
    """Places where something small changes now and then - a digit flipping on a still panel - while the area around
    it stays put. Counts a pixel's change only when most of its neighbourhood didn't change at the same moment, so a
    scrolling playfield, a flash or a fade doesn't count but a score ticking over on a status bar does."""
    if len(F) < 3:
        return []
    H, W = F.shape[1:]
    r = win // 2
    cnt = np.zeros((H, W), np.int32)
    act = np.zeros((H, W), np.float32)          # how busy each pixel's neighbourhood is, on average
    pairs = 0
    for t in range(1, len(F)):
        d = F[t] != F[t - 1]
        n = int(d.sum())
        if n == 0 or n > 0.9 * H * W:
            continue
        ii = np.pad(d.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        y0 = np.clip(np.arange(H) - r, 0, H)
        y1 = np.clip(np.arange(H) + r + 1, 0, H)
        x0 = np.clip(np.arange(W) - r, 0, W)
        x1 = np.clip(np.arange(W) + r + 1, 0, W)
        s = ii[y1][:, x1] - ii[y0][:, x1] - ii[y1][:, x0] + ii[y0][:, x0]
        area = (y1 - y0)[:, None] * (x1 - x0)[None, :]
        cnt += d & (s < dense * area)
        act += s / area
        pairs += 1
    # a status bar sits in a quiet neighbourhood; the playfield around a sprite is busy most of the time
    cand = (cnt >= 1) & (cnt <= max(4, max_rate * len(F))) & (act / max(1, pairs) < 0.08)
    return _boxes_from(cand, L, H, W)


def _plausible(spec, frames, limit=400):
    """How score-like a display reads across play: readable most of the time, and it doesn't go down
    (except back to 0 for a new game). 1.0 = perfect."""
    if not frames:
        return 0.0
    rd = ScreenScore(spec)
    idx = np.linspace(0, len(frames) - 1, min(limit, len(frames))).astype(int)
    vals = [rd.read(frames[i]) for i in idx]
    got = [v for v in vals if v is not None]
    if not got:
        return 0.0
    drops = sum(1 for a, b in zip(got, got[1:]) if b < a and b != 0)
    ups = sum(1 for a, b in zip(got, got[1:]) if b > a)
    return len(got) / len(vals) - drops / max(1, ups + drops)


def fit_labelled(labelled, frames=(), around=None):
    """Find the score on the screen with the player's help. labelled: [(frame, number on screen)] - the scores typed
    while playing. frames: more frames of play, a few apart. around: for each labelled frame, a few frames from just
    before/after it (they show what sits still - a score panel - and what's the game). Every typed number must read
    back exactly (one may be a typo); each digit is learned from what was typed (a digit never typed is left to the
    digit brain). Returns a spec for ScreenScore, or None."""
    from collections import Counter

    def as_ids(f):
        f = np.asarray(f)
        return f if f.ndim == 2 else cids(f)
    lab = [(as_ids(f), int(v)) for f, v in labelled]
    keep = [i for i, (_, v) in enumerate(lab) if v >= 0]
    lab = [lab[i] for i in keep]
    around = [[as_ids(g) for g in around[i]] for i in keep] if around else [[] for _ in lab]
    if len({v for _, v in lab}) < 2:
        return None
    shape = Counter(f.shape for f, _ in lab).most_common(1)[0][0]
    ok = [i for i, (f, _) in enumerate(lab) if f.shape == shape]
    lab, around = [lab[i] for i in ok], [around[i] for i in ok]
    play = [g for g in (as_ids(f) for f in frames) if g.shape == shape]
    L = np.stack([f for f, _ in lab])
    Fs = np.stack(play + [f for f, _ in lab]) if play else L
    best, tried = None, set()
    for box, c in _label_boxes(lab, around) + _event_boxes(Fs, L) + candidate_boxes(Fs):
        for bx, col, gap, edge, _ in displays_in_box(L, box, c):
            k = (tuple(bx), col, gap, edge)
            if k in tried:
                continue
            tried.add(k)
            r = _teach(lab, bx, col, gap, edge)
            if not r:
                continue
            spec = {"kind": "screen", "box": [int(v) for v in bx], "fg": int(col), "edge": int(edge), "gap": int(gap),
                    "keys": {kk: int(d) for kk, d in r[1].items()}, "scale": int(r[2])}
            p = _plausible(spec, play)
            rank = (r[0][0], r[0][1], round(p, 2), r[0][2])
            if best is None or rank > best[0]:
                best = (rank, spec, r[3], r[4], p)
    if not best:
        return None
    rank, spec, good, missed, p = best
    spec.update({"frame_shape": [int(shape[0]), int(shape[1])], "taught": True, "confident": bool(p >= 0.5),
                 "report": {"labels_read": int(good), "labels_missed": int(missed), "plausible": round(float(p), 3),
                            "digits_learned": sorted({int(d) for d in spec["keys"].values()})}})
    return spec
