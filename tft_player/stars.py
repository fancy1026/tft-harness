"""Star level of our units from the badge at the left end of their health bars.

A 1-star unit has a bronze single chevron, a 2-star unit a silver-blue diamond, a 3-star unit a gold double
chevron (from the user's screenshots; no 3-star bar is in our own recordings yet, so the player accepts a
3 only for a unit tracked as 2-star). The badge sits just left
of the first green pixel of the bar, so it is found from the bar itself, in screen pixels.
"""
from __future__ import annotations

import numpy as np


def read_star(frame: np.ndarray, bar) -> int | None:
    """bar = (x, y) in 1080p coordinates (board.green_bars) -> 1, 2, 3 or None (not readable)."""
    h, w = frame.shape[:2]
    s = h / 1080
    X, Y = round(w / 2 + (bar[0] - 960) * s), round(bar[1] * s)
    r = round(4 * s)
    band = frame[max(0, Y - r):Y + r + 1, max(0, X - round(70 * s)):X + round(30 * s)].astype(np.int16)
    if band.size == 0:
        return None
    green = (band[..., 1] > 200) & (band[..., 0] < 70) & (band[..., 2] < 70)
    cols = np.where(green.any(axis=0))[0]
    if len(cols) == 0:
        return None
    x_green = X - round(70 * s) + int(cols[0])            # where the bar's green starts
    x1, x0 = x_green - round(3 * s), x_green - round(32 * s)
    if x0 < 0:
        return None
    reg = frame[max(0, Y - round(14 * s)):Y + round(14 * s), x0:x1].reshape(-1, 3).astype(np.float32)
    if len(reg) < 20:
        return None
    ok = ~((reg[:, 1] > reg[:, 0] + 25) & (reg[:, 1] > reg[:, 2] + 10))          # not the bar's green
    reg = reg[ok]
    if len(reg) < 20:
        return None
    r_, g_, b_ = reg[:, 0], reg[:, 1], reg[:, 2]
    silver = int(((r_ >= 120) & (b_ >= r_ + 20) & (g_ >= r_ + 10) & (reg.max(1) >= 170)).sum())     # silver-blue
    bronze = int(((r_ >= 150) & (r_ >= g_ + 25) & (g_ >= b_ + 20)).sum())                             # bronze
    # silver diamond: ~120 silver pixels (the item icon under the bar can add bronze ones); bronze chevron: ~35
    # bronze pixels and no silver. Vivid effects over the badge match neither: abstain.
    k = (s / 1.481) ** 2                                   # pixel counts were measured on 2560x1600 frames
    gold = int(((r_ >= 220) & (g_ >= 190) & (b_ <= 150) & (g_ - b_ >= 60)).sum())      # bright gold (3-star border)
    if silver >= 60 * k:
        return 2
    if 20 * k <= gold <= 250 * k and silver <= 15 * k:    # (a coin or loot orb next to the bar is bigger and rarer)
        return 3
    if bronze >= 20 * k and silver <= 15 * k:
        return 1
    return None


def read_stars(frame: np.ndarray, places) -> dict:
    """places = board.bar_places(...)[2] -> {loc: star or None} for every own bar."""
    return {loc: read_star(frame, bar) for bar, loc in places}
