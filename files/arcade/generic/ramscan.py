"""Find where an arcade game keeps its score (and lives, and "a game is on") by watching its memory.

Pure numpy, no emulator here, so it can be tested anywhere. Used by score_finder.py (the "Find the score" window)
and by generic_env.py (which decodes the score every move while the AI trains).

How it works, in one breath: the player pauses a few times and types the score shown on screen. Every way an old
arcade game might store that number (packed BCD, plain binary, one digit per byte in the screen's tile memory -
either byte order, 68000 games word-swapped) is tried at every memory address. Only places that hold the right
number at EVERY pause survive. Survivors are then ranked by how a score behaves over the whole recording: it only
goes up, it doesn't change every frame, and it always decodes cleanly.

Signed: Claude, 10 Oct 2026.
"""
from __future__ import annotations

import numpy as np

STRIDES = (1, -1, 2, -2, 32, -32, 64, -64)     # one digit per byte: neighbours, every other byte, screen rows
MAX_DIGITS = 8


# ------------------------------------------------------------------------------------------------ views + decoders
def view(ram: np.ndarray, swapped: bool) -> np.ndarray:
    """68000 games keep RAM as 16-bit words in the PC's byte order, so each pair of bytes is flipped."""
    ram = np.asarray(ram, np.uint8)
    if not swapped:
        return ram
    n = len(ram) & ~1
    out = ram.copy()
    out[:n] = ram[:n].reshape(-1, 2)[:, ::-1].reshape(-1)
    return out


def _bytes_at(r, offs, n, big):
    """Rows of n bytes starting at each offset, least significant first."""
    idx = offs[:, None] + (np.arange(n)[::-1] if big else np.arange(n))[None, :]
    return r[idx].astype(np.int64)


def _decode_bin(b):
    return (b << (8 * np.arange(b.shape[1]))[None, :]).sum(1)


def _decode_bcd(b):
    lo, hi = b & 15, b >> 4
    ok = (lo <= 9).all(1) & (hi <= 9).all(1)
    val = ((lo + 10 * hi) * (100 ** np.arange(b.shape[1]))[None, :]).sum(1)
    return np.where(ok, val, -1)


def decode(ram, c):
    """Decode one candidate ("c" is a dict made by find_score) from one RAM snapshot. -1 = doesn't decode.
    Reads only the bytes it needs, so it's cheap enough to run every move."""
    o, sw = c["offset"], 1 if c["swapped"] else 0
    size = len(ram)
    if c["kind"] == "digits":
        s, base, v, p = c["stride"], c["base"], 0, 1
        for j in range(c.get("max_digits", MAX_DIGITS)):
            a = o - j * s
            if not 0 <= a < size:
                break
            d = int(ram[a]) - base
            if not 0 <= d <= 9:
                if j == 0:
                    return -1
                break
            v += d * p
            p *= 10
        return int(round(v * c["scale"]))
    n, big = c["n"], c["big"]
    if o + n > size or ((o + n - 1) ^ sw) >= size:
        return -1
    v, mul = 0, 1
    for i in (range(n - 1, -1, -1) if big else range(n)):      # least significant byte first
        b = int(ram[(o + i) ^ sw])
        if c["kind"] == "bcd":
            if (b & 15) > 9 or (b >> 4) > 9:
                return -1
            v += ((b >> 4) * 10 + (b & 15)) * mul
            mul *= 100
        else:
            v += b * mul
            mul *= 256
    return int(round(v * c["scale"]))


def _decode_all(r, kind, n, big):
    """Decode every offset at once (vectorised). Returns values aligned to offsets 0..len-n."""
    offs = np.arange(len(r) - n + 1)
    b = _bytes_at(r, offs, n, big)
    return _decode_bcd(b) if kind == "bcd" else _decode_bin(b)


# ------------------------------------------------------------------------------------------------ the score search
def _scales(values):
    """Many games never store the last 0 (or two) of the score - it's painted on. Try score, score/10, score/100."""
    out = []
    for k in (1, 10, 100):
        if all(v % k == 0 for v in values):
            out.append(k)
    return out


def find_score(labels, max_candidates=400):
    """labels: list of (ram uint8 array, score shown on screen). Returns candidate dicts that match every label."""
    labels = [(np.asarray(r, np.uint8), int(v)) for r, v in labels]
    if not labels:
        return []
    # search from the most informative label (biggest number = fewest accidental matches), verify on the rest
    order = sorted(range(len(labels)), key=lambda i: -labels[i][1])
    ram0, v0 = labels[order[0]]
    rest = [labels[i] for i in order[1:]]
    cands = []
    for swapped in (False, True):
        r0 = view(ram0, swapped)
        rest_v = [(view(r, swapped), v) for r, v in rest]
        for scale in _scales([v for _, v in labels]):
            t0 = v0 // scale
            # --- packed BCD and plain binary, 1-4 bytes, either byte order
            for kind in ("bcd", "bin"):
                for n in (1, 2, 3, 4):
                    if t0 >= (100 ** n if kind == "bcd" else 256 ** n):
                        continue
                    for big in ((False, True) if n > 1 else (False,)):
                        vals = _decode_all(r0, kind, n, big)
                        hit = np.flatnonzero(vals == t0)
                        for r, v in rest_v:
                            if not len(hit):
                                break
                            hit = hit[_decode_all(r, kind, n, big)[hit] == v // scale]
                        for o in hit[:max_candidates]:
                            cands.append({"kind": kind, "n": n, "big": big, "offset": int(o), "swapped": swapped,
                                          "scale": scale})
            # --- one digit per byte (screen tile memory, or a digit array)
            if swapped:
                continue        # tile memory is byte-wide; no need to look at it word-swapped
            digits = [int(ch) for ch in str(t0)][::-1]            # units first
            for s in STRIDES:
                L = len(digits)
                lo, hi = max(0, (L - 1) * s), len(r0) + min(0, (L - 1) * s)
                if hi <= lo:
                    continue
                u = np.arange(lo, hi)
                base = r0[u].astype(np.int16) - digits[0]
                ok = np.ones(len(u), bool)
                for j in range(1, L):
                    ok &= (r0[u - j * s].astype(np.int16) - digits[j]) == base
                u, base = u[ok], base[ok]
                # the byte just past the top digit must NOT be another digit (else we're matching inside a longer number)
                nxt = u - L * s
                inside = (nxt >= 0) & (nxt < len(r0))
                nd = np.full(len(u), True)
                nd[inside] = ~((r0[nxt[inside]].astype(np.int16) - base[inside] >= 0) &
                               (r0[nxt[inside]].astype(np.int16) - base[inside] <= 9)) | (r0[nxt[inside]] - base[inside] == 0)
                u, base = u[nd], base[nd]
                for uu, bb in zip(u[:max_candidates], base[:max_candidates]):
                    c = {"kind": "digits", "stride": s, "base": int(bb), "offset": int(uu), "swapped": False,
                         "scale": scale, "max_digits": MAX_DIGITS}
                    if all(decode(r, c) == v for r, v in rest) and decode(ram0, c) == v0:
                        cands.append(c)
    return _dedupe(cands)


def find_score_loose(labels, between=(), max_candidates=400):
    """Backup search when no place holds the exact number typed (an odd format, or a typo in every pause).
    Only uses the ORDER of the scores: when the typed score went up, the stored number must have gone up too; when
    it stayed the same, so must the stored number (a new game - typed score went down - must go down). The bytes are
    compared as plain whole numbers, which keeps that order for binary AND BCD, so the format doesn't matter.
    labels: [(ram, typed score)] in the order they were typed. between: for each gap between label i-1 and i, a
    few RAM snapshots from that gap (the stored number must never drop inside a gap where the score went up).
    Returns candidate dicts like find_score's; each also gets "loose": True and a best-guess kind/scale."""
    labels = [(np.asarray(m, np.uint8), int(v)) for m, v in labels]
    if len(labels) < 2 or len({v for _, v in labels}) < 2:
        return []
    between = list(between) + [[]] * (len(labels) - 1 - len(between))
    size = min(len(m) for m, _ in labels)
    out = []
    for swapped in (False, True):
        views = [view(m[:size], swapped) for m, _ in labels]
        gaps = [[view(np.asarray(g, np.uint8)[:size], swapped) for g in gap] for gap in between]
        for n in (2, 3, 4, 1):
            for big in ((False, True) if n > 1 else (False,)):
                raw = [_decode_all(v, "bin", n, big) for v in views]
                keep = np.ones(len(raw[0]), bool)
                for (_, t0), (_, t1), a, b in zip(labels, labels[1:], raw, raw[1:]):
                    keep &= (b > a) if t1 > t0 else (b == a) if t1 == t0 else (b < a)
                for (_, t), a in zip(labels, raw):
                    if t > 0:
                        keep &= a > 0
                offs = np.flatnonzero(keep)
                for i, gap in enumerate(gaps):                     # never drops while the score is going up
                    if not len(offs) or labels[i + 1][1] <= labels[i][1] or not gap:
                        continue
                    prev = raw[i][offs]
                    for g in gap:
                        now = _decode_all(g, "bin", n, big)[offs]
                        ok = now >= prev
                        offs, prev = offs[ok], now[ok]
                    ok = raw[i + 1][offs] >= prev
                    offs = offs[ok]
                for o in offs[:max_candidates]:
                    out.append(_loose_guess(views, labels, int(o), n, big, swapped))
    return _dedupe(out)


def _loose_guess(views, labels, o, n, big, swapped):
    """Best reading of a loose find: BCD or binary, times a steady scale if the typed scores line up with one."""
    typed = np.array([t for _, t in labels], float)
    for kind in ("bcd", "bin"):
        vals = np.array([_decode_all(v[o:o + n], kind, n, big)[0] for v in views], float)
        if (vals < 0).any():
            continue
        nz = (vals > 0) & (typed > 0)
        if nz.sum() >= 2:
            ratio = typed[nz] / vals[nz]
            s = float(np.median(ratio))
            if s > 0 and np.all(np.abs(ratio - s) <= 0.02 * s):
                if s >= 0.98 and abs(s - round(s)) < 0.02 * s:
                    scale = int(round(s))                      # shown x2, x10 ...
                elif s < 1 and abs(1 / s - round(1 / s)) < 0.02 / s and round(1 / s) <= 10:
                    scale = 1 / round(1 / s)                   # stored doubled, x10 ...
                else:
                    continue
                return {"kind": kind, "n": n, "big": big, "offset": o, "swapped": swapped, "scale": scale,
                        "loose": True, "fits": True, "nice": scale in (1, 10, 100)}
    return {"kind": "bin", "n": n, "big": big, "offset": o, "swapped": swapped, "scale": 1, "loose": True,
            "fits": False}


def _dedupe(cands):
    """The same number found as 2-, 3- and 4-byte at one place is one find: keep 3 bytes if it fits (room for a
    million points without reaching into a neighbouring byte that just happened to be 0 while you played)."""
    best = {}
    for c in cands:
        key = (c["kind"], c["offset"], c["swapped"], c["scale"], c.get("big"), c.get("stride"))
        if key not in best or _width_pref(c) < _width_pref(best[key]):
            best[key] = c
    return list(best.values())


def _width_pref(c):
    n = c.get("n", 0)
    return (-min(n, 3), n)


def rank(cands, trace):
    """Score each candidate by how 'score-like' it behaves over the recording (list of RAM snapshots during play).
    Lower penalty = better. Adds c["penalty"], c["why"]; returns the list sorted best first."""
    trace = _thin(trace, 800)
    for c in cands:
        vals = np.array([decode(r, c) for r in trace], np.int64) if trace else np.zeros(0, np.int64)
        bad = int((vals < 0).sum())
        good = vals[vals >= 0]
        d = np.diff(good) if len(good) > 1 else np.zeros(0, np.int64)
        drops = int(((d < 0) & (good[1:] != 0)).sum())          # going down (other than a reset to 0) is un-score-like
        changes = int((d != 0).sum())
        busy = changes / max(1, len(d))                            # a score doesn't change on most frames
        pen = 3.0 * bad / max(1, len(vals)) + 2.0 * drops / max(1, len(d)) + max(0.0, busy - 0.25) * 4
        pen += {"bcd": 0.0, "bin": 0.05, "digits": 0.1}[c["kind"]]    # memory over screen tiles (tiles can flicker)
        if changes == 0 and len(d):
            pen += 0.5                                             # never moved during play: suspicious
        c["penalty"] = round(float(pen), 4)
        c["why"] = f"{changes} changes, {drops} drops, {bad} unreadable of {len(vals)}"
    cands = sorted(cands, key=lambda c: (c["penalty"], _width_pref(c), c["swapped"], c["scale"], c["offset"]))
    # one place in memory = one find: drop lower-ranked readings that share bytes with a better one
    kept, used = [], set()
    for c in cands:
        fp = footprint(c)
        if fp & used:
            continue
        used |= fp
        kept.append(c)
    return kept


def footprint(c):
    """The physical bytes a candidate reads."""
    if c["kind"] == "digits":
        return {c["offset"] - j * c["stride"] for j in range(c.get("max_digits", MAX_DIGITS))}
    idx = range(c["offset"], c["offset"] + c["n"])
    return {i ^ 1 for i in idx} if c["swapped"] else set(idx)


def describe(c):
    if c["kind"] == "digits":
        how = f"one digit per byte (step {c['stride']}, digit 0 = {c['base']:#04x})"
    else:
        how = (f"{c['n']}-byte {'BCD' if c['kind'] == 'bcd' else 'binary'}, "
               f"{'big' if c['big'] else 'little'}-endian")
    tail = f", shown x{c['scale']}" if c["scale"] > 1 else ""
    return f"{how} at {c['offset']:#x}{' (68000 word-swapped)' if c['swapped'] else ''}{tail}"


# ------------------------------------------------------------------------------------------------ lives + "game on"
def find_lives(labels):
    """labels: list of (ram, lives shown). A byte that equals lives, lives-1 or lives+1 at every label.
    Needs at least two different lives values to mean anything."""
    labels = [(np.asarray(r, np.uint8), int(v)) for r, v in labels if v is not None]
    if len({v for _, v in labels}) < 2:
        return []
    out = []
    for adj in (0, -1, 1):
        ok = np.ones(min(len(r) for r, _ in labels), bool)
        for r, v in labels:
            ok &= r[:len(ok)].astype(np.int16) == v + adj
        for o in np.flatnonzero(ok)[:50]:
            out.append({"offset": int(o), "adjust": adj})
    return out


def rank_lives(cands, trace):
    """Lives change rarely and stay small. Best first."""
    trace = _thin(trace, 800)
    for c in cands:
        v = np.array([int(r[c["offset"]]) for r in trace]) if trace else np.zeros(0)
        changes = int((np.diff(v) != 0).sum()) if len(v) > 1 else 0
        c["penalty"] = changes / max(1, len(v) - 1) * 10 + float((v > 10).mean() if len(v) else 0) + abs(c["adjust"]) * 0.01
    return sorted(cands, key=lambda c: (c["penalty"], c["offset"]))


def read_lives(ram, c):
    return int(ram[c["offset"]]) - c["adjust"] if c else None


def find_playing_flag(play, idle, max_values=8):
    """A byte whose values while a game is on never appear while it isn't (attract mode / after GAME OVER).
    play / idle: lists of RAM snapshots. Returns {"offset", "values"} for the best byte, or None."""
    if len(play) < 5 or len(idle) < 5:
        return None
    play, idle = _thin(play, 300), _thin(idle, 300)
    m = min(min(len(r) for r in play), min(len(r) for r in idle), 1 << 20)   # first 1 MB is plenty for a mode byte
    best = None                                   # (distinct values while playing, offset, value mask)
    for c0 in range(0, m, 1 << 15):               # in chunks, so big-memory games don't need gigabytes
        c1 = min(m, c0 + (1 << 15))
        mp_ = _value_masks(np.stack([np.asarray(r[c0:c1], np.uint8) for r in play]))
        mi_ = _value_masks(np.stack([np.asarray(r[c0:c1], np.uint8) for r in idle]))
        disjoint = ~(mp_ & mi_).any(0)
        nP = _popcount(mp_).sum(0)
        ok = np.flatnonzero(disjoint & (nP <= max_values))
        if len(ok):
            j = ok[np.lexsort((ok, nP[ok]))][0]
            if best is None or nP[j] < best[0]:
                best = (int(nP[j]), c0 + int(j), mp_[:, j].copy())
    if best is None:
        return None
    vals = [q * 64 + b for q in range(4) for b in range(64) if (int(best[2][q]) >> b) & 1]
    return {"offset": best[1], "values": vals}


def _thin(snaps, n):
    if len(snaps) <= n:
        return list(snaps)
    idx = np.linspace(0, len(snaps) - 1, n).astype(int)
    return [snaps[i] for i in idx]


def _value_masks(A):
    """A: (T, m) bytes -> (4, m) uint64 bitmasks of which of the 256 values each column ever took."""
    out = np.zeros((4, A.shape[1]), np.uint64)
    bit = np.left_shift(np.uint64(1), (A & 63).astype(np.uint64))
    q = A >> 6
    for k in range(4):
        out[k] = np.bitwise_or.reduce(np.where(q == k, bit, np.uint64(0)), axis=0)
    return out


def _popcount(x):
    x = x.copy()
    c = np.zeros(x.shape, np.int64)
    while x.any():
        c += (x & np.uint64(1)).astype(np.int64)
        x >>= np.uint64(1)
    return c


def is_playing(ram, flag):
    return flag is None or int(ram[flag["offset"]]) in flag["values"]


# ------------------------------------------------------------------------------------------------ health bars
# Fighting games show health as a bar, not a number, so there's nothing to type. Instead the player tells us what
# just happened ("I got hit", "I hit them", "nothing happened", "new round - both full") and we keep only the memory
# places that behaved like that - the "unknown value" search from Cheat Engine (and from Jonny's own Metric Finder
# scanner.py: start-all, then increased / decreased / unchanged filters). Two searches run side by side: yours, theirs.
HEALTH_FORMATS = (("u8", False), ("s16le", False), ("s16be", False), ("s16le", True), ("s16be", True))


def health_values(ram, fmt, swapped, offs=None):
    """Decode every offset (or just `offs`) as one health format. Returns int64 array."""
    r = view(ram, swapped)
    if fmt == "u8":
        v = r.astype(np.int64)
        return v if offs is None else v[offs]
    n = len(r) - 1
    if offs is None:
        offs = np.arange(n)
    lo, hi = r[offs].astype(np.int64), r[offs + 1].astype(np.int64)
    if fmt == "s16be":
        lo, hi = hi, lo
    v = lo | (hi << 8)
    return np.where(v >= 32768, v - 65536, v)


def decode_health(ram, c):
    o, sw = c["offset"], 1 if c["swapped"] else 0
    if c["fmt"] == "u8":
        return int(ram[o])
    a, b = int(ram[o ^ sw]), int(ram[(o + 1) ^ sw])
    if c["fmt"] == "s16be":
        a, b = b, a
    v = a | (b << 8)
    return v - 65536 if v >= 32768 else v


class HealthSearch:
    """One unknown-value search over every health format. Each format keeps arrays of: offset, value at the last
    mark, value when the bars were full, and the running in-round value (for the never-goes-up check)."""

    def __init__(self, ram, allow_zero=False):
        self.sets = []
        for fmt, sw in HEALTH_FORMATS:
            v = health_values(ram, fmt, sw)
            offs = np.flatnonzero(v >= 0 if allow_zero else v > 0)   # a full health bar isn't zero (a score can be)
            self.sets.append({"offs": offs, "last": v[offs], "full": v[offs].copy(), "run": v[offs].copy()})
        self.history = []

    def count(self):
        return int(sum(len(s["offs"]) for s in self.sets))

    def _keep(self, k, keep, now=None, update_last=True):
        s = self.sets[k]
        for key in ("offs", "last", "full", "run"):
            s[key] = s[key][keep]
        if now is not None:
            now = now[keep]
            s["run"] = now
            if update_last:
                s["last"] = now.copy()

    def mark(self, ram, what, ref=None):
        """what: 'down' (went down), 'not_up' (didn't go up), 'same', 'full' (back to the start value).
        Compared with `ref` (RAM from a moment ago - "what just happened") or, without it, the last mark."""
        before = self.count()
        saved = [{key: v.copy() for key, v in st.items()} for st in self.sets]
        for k, (fmt, sw) in enumerate(HEALTH_FORMATS):
            s = self.sets[k]
            if not len(s["offs"]):
                continue
            now = health_values(ram, fmt, sw, s["offs"])
            l = s["last"] if ref is None else health_values(ref, fmt, sw, s["offs"])
            f = s["full"]
            keep = {"down": now < l, "up": now > l, "not_up": now <= l, "same": now == l, "full": now == f}[what]
            if what in ("not_up", "same"):
                keep |= l <= 0                 # an empty bar (round over / GAME OVER) may reset to anything
            self._keep(k, keep, now)
        if before and not self.count():
            self.sets = saved               # that press can't be right (it would rule out everything): ignore it
            self.history.append((what, before, -1))
            return -1
        self.history.append((what, before, self.count()))
        return self.count()

    def never_up(self, ram):
        """Called often during a round: health never goes UP mid-round."""
        for k, (fmt, sw) in enumerate(HEALTH_FORMATS):
            s = self.sets[k]
            if not len(s["offs"]):
                continue
            now = health_values(ram, fmt, sw, s["offs"])
            self._keep(k, (now <= s["run"]) | (s["run"] <= 0), now, update_last=False)

    def candidates(self, limit=50):
        out = []
        for (fmt, sw), s in zip(HEALTH_FORMATS, self.sets):
            for o, f in zip(s["offs"][:limit], s["full"][:limit]):
                out.append({"fmt": fmt, "swapped": sw, "offset": int(o), "full": int(f)})
        return out


def pick_health_pair(mine, theirs):
    """Best (yours, theirs) pair: same format, same full value (both bars start equal), plain before word-swapped,
    u8 last (a u8 match is often just half of a 16-bit health value)."""
    best, key = None, None
    for a in mine:
        for b in theirs:
            if a["offset"] == b["offset"] and a["fmt"] == b["fmt"]:
                continue
            k = (a["fmt"] != b["fmt"] or a["swapped"] != b["swapped"], a["full"] != b["full"],
                 a["fmt"] == "u8", a["swapped"], abs(a["offset"] - b["offset"]))
            if key is None or k < key:
                best, key = (a, b), k
    return best
