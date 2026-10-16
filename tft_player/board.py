"""Board geometry (1080p reference) and unit occupancy from health bars.

Our units carry a pure-green health bar; enemies' bars are red. So occupancy of
our bench/board is a pixel scan, no sprite recognition needed.

Board rows: row 0 is the back row (nearest the camera / bench), row 3 the
front row facing the enemy. Columns 0..6 left to right.
"""
from __future__ import annotations

import numpy as np

BOARD_ROWS_Y = [651, 571, 494, 423]
BOARD_X = [
    [581, 707, 839, 966, 1091, 1222, 1349],
    [532, 660, 776, 902, 1022, 1147, 1275],
    [609, 723, 841, 959, 1076, 1193, 1310],
    [557, 687, 794, 911, 1017, 1129, 1238],
]
BENCH = [(425, 777), (542, 777), (658, 777), (778, 777), (892, 777),
         (1008, 777), (1127, 777), (1244, 777), (1357, 777)]
# Home or away: in an away fight our units fight on the opponent's arena, and the camera shows its bench elsewhere
# (run 20261014-063145 2-5: ~+90 px, all six sells pressed E on empty tiles; 4-1 on another arena: elsewhere again).
# Bench units show no health bars in a fight, so they can't be found there; a bench action waits until we are home.
# Home = the arena scenery beside the bench (no unit stands there, no panel covers it) looks like our own, taken in a
# stage-1 planning phase (always home): that run's home frames scored 0.61-0.99, its away frames 0.14-0.41.
HOME_REGIONS = [(1480, 560, 1820, 790), (255, 560, 395, 760)]      # x0, y0, x1, y1 (1080p, centred)
HOME_MIN = 0.5
_HOME_REF = [None]


def arena_sig(frame):
    """The scenery beside the bench, small and normalised. frame: RGB."""
    import cv2
    h, w = frame.shape[:2]
    s = h / 1080
    X = lambda x: round(w / 2 + (x - 960) * s)
    out = []
    for x0, y0, x1, y1 in HOME_REGIONS:
        c = cv2.cvtColor(np.ascontiguousarray(frame[round(y0 * s):round(y1 * s), X(x0):X(x1)]),
                         cv2.COLOR_RGB2GRAY).astype(np.float32)
        out.append(cv2.resize(c, ((x1 - x0) // 4, (y1 - y0) // 4), interpolation=cv2.INTER_AREA).ravel())
    v = np.concatenate(out)
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def home_similarity(frame):
    """How much the arena looks like our home one (1.0 = the same), or None before the reference is taken."""
    return None if _HOME_REF[0] is None else float(np.dot(_HOME_REF[0], arena_sig(frame)))


SHOP_CLICK = [(655 + 201.5 * i, 990) for i in range(5)]
BUY_XP = (445, 960)
REROLL = (445, 1040)
# where to drop a unit to sell it: the shop panel
SELL_DROP = (960, 1000)


def hex_xy(row, col):
    return BOARD_X[row][col], BOARD_ROWS_Y[row]


ALL_HEXES = [(r, c) for r in range(4) for c in range(7)]


def scale_point(p, w, h):
    s = h / 1080
    return round(w / 2 + (p[0] - 960) * s), round(p[1] * s)


def green_bars(frame: np.ndarray, y_range=(260, 760)):
    # front-row (row 3) bars sit at y~320
    """Return (center_x, y) of every own-unit health bar, in 1080p coordinates."""
    h, w = frame.shape[:2]
    s = h / 1080
    y0, y1 = round(y_range[0] * s), round(y_range[1] * s)
    x0, x1 = round(w / 2 - 620 * s), round(w / 2 + 520 * s)
    reg = frame[y0:y1, x0:x1].astype(np.int16)
    r, g, b = reg[..., 0], reg[..., 1], reg[..., 2]
    green = (g > 200) & (r < 70) & (b < 70)
    bars = []
    min_run = max(6, round(8 * s))
    rows = np.where(green.sum(axis=1) >= min_run)[0]
    used = np.zeros_like(green)
    for yy in rows:
        line = green[yy] & ~used[yy]
        if line.sum() < min_run:
            continue
        # a high-HP unit's tick marks cut its green into 6-7 px segments, all shorter than min_run: the bar
        # was found late (its point 24-32 px right: clicks, badge and item row off - run 221257 6-1 read
        # Protector's Vow as the first item and missed the Tear) or not at all (033507). Close 1-2 px gaps.
        p = np.pad(line, 2)
        line = line | (p[1:-3] & p[3:-1]) | (p[:-4] & p[3:-1]) | (p[1:-3] & p[4:])
        # runs of green along this line
        d = np.diff(np.concatenate(([0], line.view(np.int8), [0])))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        line_bars = []
        for a, e in zip(starts, ends):
            if e - a < min_run:
                continue
            if any(abs(a - b) < round(70 * s) for b in line_bars):
                continue            # one bar broken into segments by its tick marks
            line_bars.append(a)
            # a bar is ~75px wide incl. the dark part; the green part starts
            # right after the notch, so the bar centre is ~a + 32px
            cx = x0 + a + round(30 * s)
            bars.append(((cx - w / 2) / s + 960, (y0 + yy) / s))
            used[max(0, yy - 8):yy + 8, max(0, a - 10):a + round(80 * s)] = True
    return bars


def occupancy(frame: np.ndarray, exclude=None):
    """-> (bench: list[bool] x9, board: set[(row, col)]). `exclude` is our
    tactician's bar box, which can be green at full HP."""
    bench, board, _ = bar_places(frame, exclude)
    return bench, board


# A tall model carries its bar up to ~145 px over its hex: nearest-hex matching gives it to the hex behind (or to
# none). Its height is learned per champion and star level when a board check finds the unit under such a bar
# (benchmark game 20261018-161404: Kayle 2-star at (0,0), bar 142 px up, read as (1,0) in four board checks).
BAR_DY = {}            # (champion, stars) -> bar height over its hex centre (px, 1080p)
TALL_HINT = None       # () -> {(row, col): bar height} for our tall units' hexes now (set by the player)


def bar_places(frame: np.ndarray, exclude=None, bars=None):
    """-> (bench: list[bool] x9, board: set[(row, col)], [((bx, by), loc)]): every own health bar
    with the place it belongs to - ('bench', i) or ('board', r, c)."""
    bench = [False] * 9
    board = set()
    places = []
    todo = [b for b in (green_bars(frame) if bars is None else bars)
            if not (exclude and exclude[0] <= b[0] <= exclude[2] + 40 and exclude[1] <= b[1] <= exclude[3])]
    for (r, c), h in (TALL_HINT() if TALL_HINT else {}).items():
        # a tall unit's own bar first: right over its hex at its known height
        hx, hy = hex_xy(r, c)
        near = [b for b in todo if abs(b[0] - hx) < 45 and abs(hy - b[1] - h) <= 15]
        if near:
            b = min(near, key=lambda b: abs(hy - b[1] - h) + abs(b[0] - hx))
            todo.remove(b)
            board.add((r, c))
            places.append((b, ('board', r, c)))
    for bx, by in todo:
        if 650 <= by <= 750:        # bench bars; tall models push the bar up to ~y650
            i = min(range(9), key=lambda k: abs(BENCH[k][0] - bx))
            if abs(BENCH[i][0] - bx) < 60:
                bench[i] = True
                places.append(((bx, by), ('bench', i)))
        elif by < 650:
            # bar sits 30..150px above the unit's hex centre
            best, bd = None, 1e9
            for r, c in ALL_HEXES:
                hx, hy = hex_xy(r, c)
                dy = hy - by
                if not 25 <= dy <= 150:
                    continue
                # the bar floats over its unit: the camera's perspective pushes it away from the screen's middle, the
                # more the higher it is (1151 recorded bars: 0.042 px per px off-centre per 100 px up). Without it a
                # tall unit's bar went to the hex behind: Kayle** and Pebbles** at (0,0) read as (1,0) (161404,
                # 171815), Brambleback** at (2,5) as (3,6) (123359) - 18 of 565 recorded planning frames change
                px = hx + 0.042 * (hx - 960) * dy / 100 - 1.5
                d = abs(px - bx) + 0.35 * abs(dy - 70)
                if d < bd and (r, c) not in board:
                    best, bd = (r, c), d
            if best and bd < 75:
                board.add(best)
                places.append(((bx, by), ('board', *best)))
    return bench, board, places
