"""SHOW IT HOW - find where an arcade game keeps how far along the level it is, and its lives, from a short recording
of someone playing it (show_how.py). ALFRED 1 Nov 2026.

Recording: the memory at the start, then for every move only the bytes that changed (sparse), how far the screen
scrolled that move (scrollscan.shift: dx right / dy up), and the moves where the player said "I lost a life".

SCROLL: among bytes that change often, every u8 / u16 (both byte orders, either byte as the low one) whose change each
move follows the screen's scrolling (correlation, resets after a death ignored). The best one wins; between near-ties the
one whose running total tracks the screen's running total best (that settles u8 vs u16 and which byte is which).
LIVES: a byte starting 1-9 that drops by exactly 1 once near each 'lost a life' press (6 s either side) and at no other
time, with no other changes except at most one +1 (an extra life). Needs 2+ presses."""
import numpy as np

CAP = 4 << 20            # memory looked at (bytes)
CLIP = {"u8": 48, "u16": 4096}
DEBUG = False
NEAR = 90                # moves either side of a press (4 frames a move = 6 s)


class Recording:
    def __init__(self, mem):
        self.m = min(len(mem), CAP)
        self.start = np.array(mem[:self.m], np.uint8)
        self.cur = self.start.copy()
        self.idx, self.val, self.dx, self.dy, self.marks = [], [], [], [], []

    def add(self, mem, dx, dy):
        new = np.asarray(mem[:self.m], np.uint8)
        ch = np.flatnonzero(new != self.cur)
        self.idx.append(ch.astype(np.int32)); self.val.append(new[ch].copy())
        self.cur[ch] = new[ch]
        self.dx.append(dx); self.dy.append(dy)

    def mark(self):
        self.marks.append(len(self.dx))

    @property
    def steps(self):
        return len(self.dx)

    def moved(self):
        return int(np.abs(self.dx).sum()), int(np.abs(self.dy).sum())

    def history(self, offsets):
        """steps+1 x len(offsets) values of these bytes."""
        offsets = np.asarray(offsets, np.int64)
        pos = {int(o): j for j, o in enumerate(offsets)}
        h = np.empty((self.steps + 1, len(offsets)), np.int32)
        row = self.start[offsets].astype(np.int32)
        h[0] = row
        for t, (ix, v) in enumerate(zip(self.idx, self.val)):
            if len(ix):
                hit = np.isin(ix, offsets)
                for i, x in zip(ix[hit], v[hit]):
                    row[pos[int(i)]] = x
            h[t + 1] = row
        return h

    def change_counts(self):
        c = np.zeros(self.m, np.int32)
        for ix in self.idx:
            c[ix] += 1
        return c


def _corr(d, s):
    d = d - d.mean(0); s = s - s.mean()
    den = np.sqrt((d * d).sum(0) * (s * s).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (d * s[:, None]).sum(0) / den
    return np.nan_to_num(r)


def find_scroll(rec):
    mx, my = rec.moved()
    axis = "x" if mx >= my else "y"
    s = np.asarray(rec.dx if axis == "x" else rec.dy, np.float64)
    if max(mx, my) < 300:
        return None, f"the screen only moved {max(mx, my)} pixels - walk further the way the level goes"
    fwd = 1 if s.sum() >= 0 else -1          # the way the player mostly went
    cnt = rec.change_counts()
    cand = np.flatnonzero(cnt >= max(10, rec.steps // 50))
    if len(cand) == 0:
        return None, "nothing in memory changed often enough"
    if len(cand) > 6000:
        cand = cand[np.argsort(-cnt[cand])[:6000]]
    near = np.unique(np.concatenate([cand - 1, cand, cand + 1]).clip(0, rec.m - 1))
    H = rec.history(near)
    col = {int(o): j for j, o in enumerate(near)}
    options = []                              # (name, offset, fmt, values over time)
    for o in cand:
        o = int(o)
        a = H[:, col[o]]
        options.append(("u8", o, "u8", a, False))
        if o + 1 in col:
            b = H[:, col[o + 1]]
            options.append(("u16be", o, "u16be", (a << 8) | b, False))        # o high, o+1 low
            options.append(("u16le", o, "u16le", a | (b << 8), False))        # o low, o+1 high
        if o - 1 in col:
            p = H[:, col[o - 1]]
            options.append(("u16be", o - 1, "u16be", (p << 8) | a, False))    # o-1 high, o low
            options.append(("u16le", o - 1, "u16le", p | (a << 8), False))    # o-1 low, o high
    vals = np.stack([v for *_, v, _ in options], 1)
    wrap = np.array([256 if f == "u8" else 65536 for _, _, f, _, _ in options])
    d = np.diff(vals, axis=0)
    d = (d + wrap // 2) % wrap - wrap // 2
    clip = np.array([CLIP["u8"] if f == "u8" else CLIP["u16"] for _, _, f, _, _ in options])
    d = np.where(np.abs(d) > clip, 0, d).astype(np.float64)
    keep = np.ones(len(s), bool)              # leave out the dying (death animations fool the screen reading)
    for mk in rec.marks:
        keep[max(0, mk - 60):mk + 120] = False
    if keep.sum() < 100:
        keep[:] = True
    r = _corr(d[keep], s[keep])
    order = np.argsort(-np.abs(r))
    best = abs(r[order[0]])
    if best < 0.85:
        return None, f"no place in memory follows the scrolling closely (best match {best:.2f})"
    # near-ties: the one whose running total tracks the screen's running total (no wrapping, right byte order)
    cam = np.concatenate([[0], np.cumsum(s)])
    pick, pick_err = None, None
    for k in order[:24]:
        if abs(r[k]) < best - 0.03:
            break
        name, o, fmt, v, _ = options[k]
        slope = (d[keep, k] * s[keep]).sum() / max((s[keep] * s[keep]).sum(), 1e-9)
        if not 0.2 <= abs(slope) <= 20:        # a real scroll counts pixels, half-pixels or a few sub-pixels
            continue
        pred = v[0] + slope * cam
        # compare only within stretches with no reset (a reset = a big jump even allowing for wrap-round, so a
        # counter that wraps round - the low byte on its own - is NOT forgiven)
        w = wrap[k]
        raw = (np.diff(v) + w // 2) % w - w // 2
        cut = np.flatnonzero(np.abs(raw) > 16 * abs(slope) + 8) + 1      # more than one move's scrolling = a reset
        cut = np.union1d(cut, np.flatnonzero(~keep) + 1)                   # and the dying is left out
        err, n = 0.0, 0
        for seg in np.split(np.arange(len(v)), cut):
            seg = seg[np.r_[keep, True][seg]] if len(seg) else seg
            if len(seg) < 20:
                continue
            e = (v[seg] - v[seg[0]]) - (pred[seg] - pred[seg[0]])
            err += float(np.abs(e).sum()); n += len(seg)
        err = err / max(n, 1) / abs(slope)
        if DEBUG:
            print('  tie', fmt, hex(o), round(float(r[k]), 3), 'slope', round(float(slope), 3), 'err px', round(err, 1), 'n', n)
        if pick_err is None or err < pick_err - 0.5:
            pick, pick_err = (o, fmt, float(slope), float(r[k])), err
    if DEBUG:
        for k in order[:10]:
            print('  cand', options[k][2], hex(options[k][1]), round(float(r[k]),3))
    if pick is None:
        return None, "couldn't settle on one place in memory"
    o, fmt, slope, rr = pick
    invert = (slope * fwd) < 0
    upp = abs(slope)
    neat = 2.0 ** round(np.log2(upp))         # counters count whole pixels, halves, or 2/4/8/16ths: snap when close
    upp = round(neat if abs(upp / neat - 1) < 0.15 else upp, 3)
    spec = {"offset": o, "fmt": fmt, "axis": axis, "units_per_px": upp,
            "max_jump": int(np.ceil(64 * max(upp, 1))), "new_stage_drop": int(np.ceil(200 * max(upp, 1))),
            "match": round(abs(rr), 3), "track_err_px": round(pick_err, 1)}
    if invert:
        spec["invert"] = True
    return spec, "found"


def find_lives(rec):
    marks = sorted(set(rec.marks))
    if len(marks) < 2:
        return None, "press ENTER each time you lose a life - it needs at least two"
    st = rec.start.astype(np.int32)
    ok = (st >= 1) & (st <= 9)
    cand = np.flatnonzero(ok & (rec.change_counts() >= len(marks)) & (rec.change_counts() <= len(marks) + 3))
    if len(cand) == 0:
        return None, "no place in memory changed once per life lost"
    H = rec.history(cand)
    good = []
    for j, o in enumerate(cand):
        v = H[:, j]
        dv = np.diff(v)
        ch = np.flatnonzero(dv)
        downs = [t for t in ch if dv[t] == -1]
        ups = [t for t in ch if dv[t] == 1]
        if len(ch) != len(downs) + len(ups) or len(ups) > 1 or len(downs) != len(marks):
            continue
        if all(any(abs(t - mk) <= NEAR for t in downs) for mk in marks):
            good.append((int(o), int(v[0]), int(v[-1])))
    if not good:
        return None, "no place in memory dropped by one each time you pressed ENTER"
    o, a, b = good[0]
    return {"offset": o, "adjust": 0, "start": a, "candidates": len(good)}, "found"
