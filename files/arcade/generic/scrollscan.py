"""How far the picture scrolled between two screens, from the pixels alone (no game memory). ALFRED 1 Nov 2026.

Used by 'Pay it for getting further' on games whose pack doesn't know where the game keeps its scroll, and by the
pack robot (autopack/find_progress.py) to find that place in memory. Works on 1-D brightness profiles (column means for
sideways scrolling, row means for up/down), which is cheap and shrugs off most sprites; the top and bottom eighth
(score bars) and the side eighths are left out."""
import numpy as np

MAX_SHIFT = 12          # pixels per move it will believe (4 frames at up to 3 px a frame)


def _best(a, b, m=MAX_SHIFT):
    """s such that b[i] ~ a[i + s] (content moved towards lower indices by s), or 0 if nothing clearly moved."""
    n = len(a)
    if n < 4 * m:
        return 0
    base = float(np.abs(a - b).mean())
    if base < 0.5:                      # nothing changed
        return 0
    best_s, best_v = 0, base
    for s in range(-m, m + 1):
        if s == 0:
            continue
        v = float(np.abs(a[m + s:n - m + s] - b[m:n - m]).mean())
        if v < best_v:
            best_s, best_v = s, v
    return best_s if best_v < 0.6 * base else 0


def shift(prev, cur):
    """(dx, dy) camera movement from grey screen prev to grey screen cur: dx > 0 = the view moved right (scenery slid
    left), dy > 0 = the view moved up (scenery slid down). (0, 0) when unsure or the screen changed size."""
    if prev is None or cur is None or prev.shape != cur.shape:
        return 0, 0
    h, w = cur.shape
    p, c = prev.astype(np.float32), cur.astype(np.float32)
    rows, cols = slice(h // 8, h - h // 8), slice(w // 8, w - w // 8)
    dx = _best(p[rows].mean(0), c[rows].mean(0))
    dy = -_best(p[:, cols].mean(1), c[:, cols].mean(1))
    return dx, dy
