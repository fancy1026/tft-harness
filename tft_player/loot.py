"""Loot orbs and our tactician: detection + human-speed pickup routing.

Detection: circular-masked colour template matching at half resolution over
the arena (orbs are spheres with a '?', grey ones translucent - the mask
keeps background out of the score). ~10-20 ms per frame.

Tactician: its HP bar has a dark frame ~108px wide (unit bars ~75px) with a
level box on the left; the little legend stands ~60px below the bar centre.

Routing: like a human - click the nearest orb, and queue the next click the
moment we predict arrival (walking speed is measured live), instead of
waiting to confirm each pickup.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np

DATA = Path(__file__).resolve().parent.parent / 'data' / 'loot'
ARENA = (330, 50, 1560, 760)          # 1080p region orbs can be in (own half + middle). Top at 50, not 90:
                                      # run 20261005-022645 left a grey orb at y~76 (just under the stage
                                      # bar) uncollected from 1-3 to the end
DOWN = 0.5                            # match at half resolution, grayscale
SCALES = (0.9, 1.15)
THRESH = {'gold': 0.78, 'blue': 0.78, 'gray': 0.80}
HALF_THRESH = 0.84         # the upper half alone (a health bar across the lower half): stricter


def _load():
    """[(kind, RGB template)]: '<kind>.png' plus variants '<kind>__<what>.png' (a translucent grey orb
    over the dark arena edge looks unlike one over grass: 0.69 against the grass template)."""
    temps = []
    for name in ('gold', 'blue', 'gray'):
        for f in [DATA / f'{name}.png'] + sorted(DATA.glob(f'{name}__*.png')):
            temps.append((name, cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)))
    return temps


_TEMPLATES = _load()
_cache: dict = {}


def _templates(scale):
    """(name, gray inner-square template, inset, full size) at a frame scale."""
    key = round(scale, 3)
    if key not in _cache:
        out = []
        for name, t in _TEMPLATES:
            g = cv2.cvtColor(t, cv2.COLOR_RGB2GRAY)
            for s in SCALES:
                size = max(8, round(t.shape[0] * scale * DOWN * s))
                tt = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
                k = round(size * 0.18)    # inner square of the sphere: no background
                out.append((name, np.ascontiguousarray(tt[k:size - k, k:size - k]), size))
        _cache[key] = out
    return _cache[key]


def orb_colour(patch):
    """Classify an orb patch by colour; None if it doesn't look like an orb."""
    p = patch.reshape(-1, 3).astype(np.float32)
    r, g, b = p.mean(axis=0)
    sat = (p.max(axis=1) - p.min(axis=1)).mean()
    if b > r + 25 and b > 110:
        return 'blue'
    if r > b + 40 and r > 140 and g > 90:
        return 'gold'
    if sat < 60 and (r + g + b) / 3 > 95:
        return 'gray'
    return None


def find_orbs(frame: np.ndarray):
    """-> [(x, y, kind, score)] in 1080p reference coordinates."""
    h, w = frame.shape[:2]
    s = h / 1080
    x0 = round(w / 2 + (ARENA[0] - 960) * s)
    x1 = round(w / 2 + (ARENA[2] - 960) * s)
    y0, y1 = round(ARENA[1] * s), round(ARENA[3] * s)
    roi_c = cv2.resize(frame[y0:y1, x0:x1], None, fx=DOWN, fy=DOWN, interpolation=cv2.INTER_AREA)
    roi = cv2.cvtColor(roi_c, cv2.COLOR_RGB2GRAY)
    hits = []
    for name, t, size in _templates(s):
        th = t.shape[0]
        # the whole orb, and its upper half alone: a unit's health bar across the lower half breaks the
        # whole-orb match (run 185805 4-1: a grey orb under Kog'Maw's bar was never picked up)
        for part, thresh, top in ((t, THRESH[name], False), (t[:int(th * 0.55)], HALF_THRESH, True)):
            r = cv2.matchTemplate(roi, part, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(r >= thresh)
            for yy, xx in zip(ys, xs):
                hits.append((float(r[yy, xx]), xx + th / 2, yy + th / 2, name, size, top))
    hits.sort(reverse=True)
    orbs, bars = [], None
    ref = lambda cx, cy: ((x0 + cx / DOWN - w / 2) / s + 960, (y0 + cy / DOWN) / s)
    for score, cx, cy, name, size, top in hits:       # non-max suppression
        if top:
            # an upper half alone counts only where one of our health bars crosses its lower half (the case it is
            # for); elsewhere it matched a grey end screen (run 20261010-002419). Checked before the suppression,
            # so a whole-orb match at the same place is kept
            if bars is None:
                from tft_player.board import green_bars
                bars = green_bars(frame)
            X, Y = ref(cx, cy)
            if not any(abs(bx - X) < 70 and -5 <= by - Y <= 40 for bx, by in bars):
                continue
        if all((cx - o[0]) ** 2 + (cy - o[1]) ** 2 > (size * 0.7) ** 2 for o in orbs):
            q = max(2, int(size * 0.3))
            py = int(cy - size * 0.25) if top else int(cy)          # the colour from the uncovered upper part
            patch = roi_c[max(0, py - q):py + q, max(0, int(cx) - q):int(cx) + q]
            kind = orb_colour(patch) if patch.size else None
            if kind:
                orbs.append((cx, cy, kind, score, size, top))
    out = []
    for cx, cy, kind, score, size, top in orbs:
        X, Y = ref(cx, cy)
        out.append((X, Y, kind, round(score, 3)))
    return out


# A gold orb that lands among our units is hidden by them; only its light beam shows - a tall thin
# bright-yellow streak standing on the orb (run 20261005-022645: 2-3, 3-1 and 4-1, 30-70 s each, never
# clicked). Standing beams are 60-120 px tall; a falling orb's beam is 150-270 px and moves.
BEAM_MIN, BEAM_MAX = 60, 140


def find_beams(frame: np.ndarray):
    """-> [(x, y_top, y_bottom)] of standing gold light beams, 1080p reference coordinates."""
    h, w = frame.shape[:2]
    s = h / 1080
    x0 = round(w / 2 + (ARENA[0] - 960) * s)
    x1 = round(w / 2 + (ARENA[2] - 960) * s)
    y0, y1 = round(ARENA[1] * s), round(ARENA[3] * s)
    reg = frame[y0:y1, x0:x1:2].astype(np.int16)       # beams are >= 10 px wide: every 2nd column
    r, g, b = reg[..., 0], reg[..., 1], reg[..., 2]
    yellow = (r > 200) & (g > 160) & (b < 120) & (r - b > 110)
    out = []
    for c in np.where(yellow.sum(axis=0) >= BEAM_MIN * s)[0]:
        ys = np.where(yellow[:, c])[0]
        run = max(np.split(ys, np.where(np.diff(ys) > 4 * s)[0] + 1), key=len)
        if not (len(run) >= BEAM_MIN * s and run[-1] - run[0] <= BEAM_MAX * s):
            continue
        x = (x0 + 2 * c - w / 2) / s + 960
        if out and abs(out[-1][0] - x) <= 12:
            if run[-1] - run[0] > out[-1][2] - out[-1][1]:
                out[-1] = (x, (y0 + run[0]) / s, (y0 + run[-1]) / s)
            continue
        out.append((x, (y0 + run[0]) / s, (y0 + run[-1]) / s))
    return out


class BeamOrbs:
    """Standing beams -> gold orbs at their foot. A beam counts once it is seen twice at the same place
    within REPEAT s (a falling orb's beam moves; some beams pulse and show in 1 read of 5 - run
    20261005-022645 2-3); the orb is then kept until its beam has not been seen for KEEP s."""
    REPEAT, KEEP = 10.0, 6.0

    def __init__(self):
        self.seen = []          # (x, y_bottom, t) of recent sightings
        self.known = {}         # (x, y_bottom) -> last seen

    def update(self, beams, now=None):
        now = time.time() if now is None else now
        near = lambda a, b: abs(a[0] - b[0]) <= 15 and abs(a[1] - b[1]) <= 20
        self.seen = [o for o in self.seen if now - o[2] < self.REPEAT]
        for x, _, yb in beams:
            k = next((k for k in self.known if near(k, (x, yb))), None)
            if k is not None or any(near(o, (x, yb)) and now - o[2] > 0.2 for o in self.seen):
                self.known.pop(k, None)
                self.known[(x, yb)] = now
            self.seen.append((x, yb, now))
        self.known = {k: t for k, t in self.known.items() if now - t < self.KEEP}
        return [(x, yb + 10, 'gold', 0.99) for x, yb in self.known]


def _panels():
    """The overlay's panel boxes (1080p, x centred on 960): screen pixels our own drawing covers."""
    from tft_player.overlay import PANELS
    return list(PANELS.values())


def find_tactician(frame: np.ndarray):
    """Our little legend's feet position (1080p) from its HP bar, or None.
    Also returns the bar box so unit-bar detection can ignore it.
    The bar (patch of 2026-10): a level box, then a black frame whose green fill is the HP - the fill is ~13 px
    tall where a unit's bar is ~4 px, so a tall green fill is ours at any HP (the old dark-frame match found
    nothing any more, and our own overlay panels instead: run 20261010-064944)."""
    h, w = frame.shape[:2]
    s = h / 1080
    # up to x 1550: the unit info panel (x >= 1560) has a green HP bar of its own
    x0, x1 = round(w / 2 - 700 * s), round(w / 2 + 590 * s)
    y0, y1 = round(40 * s), round(800 * s)
    reg = frame[y0:y1, x0:x1].astype(np.int16)
    r_, g_, b_ = reg[..., 0], reg[..., 1], reg[..., 2]
    green = (g_ > 140) & (r_ < 130) & (b_ < 110) & (g_ > r_ + 60)
    dark = (r_ < 70) & (g_ < 70) & (b_ < 70)
    for px0, py0, px1, py1 in _panels():                 # our overlay's own drawing is never a bar
        cx0 = max(0, round(w / 2 + (px0 - 960) * s) - x0)
        cx1 = max(0, round(w / 2 + (px1 - 960) * s) - x0 + 1)
        ry0, ry1 = max(0, round(py0 * s) - y0), max(0, round(py1 * s) - y0)
        green[ry0:ry1, cx0:cx1] = False
    tall = max(3, round(9 * s))
    core = cv2.erode(green.astype(np.uint8), np.ones((tall, 1), np.uint8))     # columns green >= `tall` rows
    n, _, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    best, gap = None, max(2, round(2 * s))
    for i in range(1, n):
        cx, cy, cw, ch, area = stats[i]
        if cw < round(6 * s) or cw > round(125 * s):
            continue
        # a bar's fill is a solid rectangle with the bar's dark frame right above and below it (foliage, card art
        # and text are ragged)
        cols = green[:, cx:cx + cw]
        full = np.where(cols.mean(axis=1) > 0.9)[0]
        if len(full) < tall:
            continue
        t0, t1 = full[0], full[-1]
        if t1 - t0 + 1 > round(20 * s) or cols[t0:t1 + 1].mean() < 0.9 or t0 <= gap or t1 + gap + 1 >= len(dark):
            continue
        frame_dark = lambda rows: max(dark[r, cx:cx + cw].mean() for r in rows)     # the frame is 1-3 px thick
        if frame_dark(range(t0 - gap - 1, t0)) < 0.7 or frame_dark(range(t1 + 1, t1 + gap + 2)) < 0.7:
            continue
        if best is None or cw * (t1 - t0) > best[0]:
            best = (cw * (t1 - t0), cx, t0)
    if best is None:
        return None, None
    _, cx, top = best
    left, fill_top = x0 + cx, y0 + top                   # the fill starts at the bar's left edge
    by = fill_top - 2 * s                                # the bar frame's top edge
    mid = left + 40 * s                                  # the level box + bar's centre
    feet = ((mid - w / 2) / s + 960 - 10, by / s + 62)
    box = ((left - w / 2) / s + 960 - 35, by / s - 30, (left - w / 2) / s + 960 + 112, by / s + 22)
    return feet, box



class LootRunner:
    """Collect every visible orb as fast as a good human would."""

    PICKUP = 45          # px (1080p): tactician this close to an orb collects it

    def __init__(self):
        self.speed = 420.0          # px/s, refined from observation
        self.target = None          # (x, y) we're walking to
        self.eta = 0.0
        self.sent_at = 0.0
        self.last_pos = None        # (x, y, t) for speed estimation
        self.gone: list = []        # recently collected spots, ignore re-detections
        self.retries = 0
        self.sent_pos = None        # where the tactician stood when the last click went out
        self.rng = __import__('random').Random()

    def observe_speed(self, pos, now):
        if pos and self.last_pos and self.target:
            dt = now - self.last_pos[2]
            if 0.05 < dt < 0.6:
                d = math.dist(pos, self.last_pos[:2])
                if 40 < d / dt < 1500:
                    self.speed = 0.8 * self.speed + 0.2 * d / dt
        if pos:
            self.last_pos = (pos[0], pos[1], now)

    def next_click(self, orbs, pos, now):
        """Return an (x, y) to right-click now, or None."""
        self.observe_speed(pos, now)
        self.gone = [(x, y, t) for x, y, t in self.gone if now - t < 3]
        live = [o for o in orbs if all(math.dist(o[:2], g[:2]) > 50 for g in self.gone)]
        if not live:
            self.target = None
            return None
        here = pos or (self.target if self.target else (700, 600))
        alive = self.target and any(math.dist(o[:2], self.target) < 40 for o in live)
        # not moved at all since the click: the click did not take (the orb lies over a champion, a bar, ...)
        stuck = (alive and pos and self.sent_pos and now - self.sent_at > 0.7
                 and math.dist(pos, self.sent_pos) < 15 and math.dist(pos, self.target) > 90)
        # still walking to a live target and not yet due to arrive: keep going
        if alive and now < self.eta and not stuck:
            return None
        if alive and (now >= self.eta or stuck) and pos and math.dist(pos, self.target) > 90 and self.retries < 3:
            # due but not there (the click on the orb did not move us, or something blocks the way): click
            # somewhere else that still takes us over it (the user, 2026-10-15) - first a point past the orb, so
            # the walk runs through it, then random points around it
            self.retries += 1
            click = self.retry_point(pos, self.target, self.retries)
            self.sent_at, self.sent_pos = now, pos
            self.eta = now + math.dist(pos, click) / self.speed + 0.15
            return click
        if self.target and (now >= self.eta or (pos and math.dist(pos, self.target) < self.PICKUP)):
            self.retries = 0
            self.gone.append((*self.target, now))
            live = [o for o in live if math.dist(o[:2], self.target) > 50]
            here = self.target
            if not live:
                self.target = None
                return None
        # nearest-neighbour route; click the first stop
        route, cur, left = [], here, list(live)
        while left:
            o = min(left, key=lambda q: math.dist(cur, q[:2]))
            route.append(o)
            left.remove(o)
            cur = o[:2]
        tgt = route[0][:2]
        self.target = tgt
        self.sent_at, self.sent_pos = now, pos
        self.eta = now + math.dist(here, tgt) / self.speed + 0.15
        return tgt

    def retry_point(self, pos, orb, k):
        """Where to click on retry k: 1 = 70 px past the orb on the line from the tactician (the path crosses it),
        2-3 = a random point 40-70 px around it. Kept inside the arena."""
        dx, dy = orb[0] - pos[0], orb[1] - pos[1]
        d = math.hypot(dx, dy) or 1.0
        if k == 1:
            x, y = orb[0] + dx / d * 70, orb[1] + dy / d * 70
        else:
            a, r = self.rng.uniform(0, 2 * math.pi), self.rng.uniform(40, 70)
            x, y = orb[0] + r * math.cos(a), orb[1] + r * math.sin(a)
        return (min(max(x, ARENA[0]), ARENA[2]), min(max(y, ARENA[1]), ARENA[3]))
