"""Carousel: see every unit's item, pick the one we want, chase it.

Each carousel unit shows an item icon under a dark 'eye' marker. One template
finds the markers; the icon below is read with the item icon bank. The unit
stands ~115px below its marker. Every tick: refit the ring from the markers, follow
the target on its circle, and replan the earliest head-on meeting with it: round the
outside lane to just ahead of it, then a cut back against the ring's turn
(CarouselChaser). Where we stand comes from where we clicked; our HP bar
(OwnTracker) only measures how far we moved.
tools/sim_carousel.py runs the same chaser in a model of the real carousel.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from tft_player import prints
from tft_player.items import BY_PAIR, COMPONENTS, ICONS, VALUE, _game_bank, _icon_bank, _icon_fp, _vec, holder_kind

DATA = Path(__file__).resolve().parent.parent / 'data'
LOOT_PRINTS = DATA / 'loot-fingerprints.npz'


def draw_marker(n=30):
    """The marker drawn in code: a dark shield badge (square top, pointed bottom) with a light eye, grey."""
    img = np.full((n, n), 70, np.uint8)                       # the ground around the badge
    badge = np.array([[1, 1], [n - 2, 1], [n - 2, int(n * 0.68)], [n // 2, n - 2], [1, int(n * 0.68)]], np.int32)
    cv2.fillPoly(img, [badge], 18)
    cv2.polylines(img, [badge], True, 40, 1, cv2.LINE_AA)
    c = (n // 2, int(n * 0.42))
    cv2.ellipse(img, c, (int(n * 0.30), int(n * 0.17)), 0, 0, 360, 100, -1, cv2.LINE_AA)
    cv2.circle(img, c, max(2, int(n * 0.12)), 28, -1, cv2.LINE_AA)
    cv2.circle(img, (c[0] + 1, c[1] - 1), max(1, int(n * 0.04)), 90, -1, cv2.LINE_AA)
    return img


# No picture of the marker is used: it is kept as a fingerprint (prints.py). Markers are searched with the template
# rebuilt from it (a blurred marker), and each place found must also match the marker drawn in code - the two make
# different false matches. 300 recorded carousel frames: every marker the picture found at 0.72, ~150 more real ones
# it missed (next to level badges and mana bars), ~14 non-markers (experiments/README.md section 6)
MARKER_PRINT, MARKER_SHAPE = prints.references(DATA / 'loot', LOOT_PRINTS)['carousel-marker']
MARKER = prints.template(MARKER_PRINT, MARKER_SHAPE)
DRAWN_MARKER = draw_marker()
REGION = (380, 60, 1540, 760)
MARKER_SEARCH, MARKER_DRAWN = 0.64, 0.50
ICON_DY, ICON_HALF = 38, 15
UNIT_DY = 115
# the in-game bank's similarities are fingerprints (items.fingerprint), which near the 0.75 cut run above the
# pictures' (0.72 -> 0.77): counted this much lower (300 recorded carousel frames: the pictures' names on 2072 of 2078
# markers, none added or changed; experiments/README.md section 6)
GAME_FP_LEAD = 0.03


_gbank = None


def _game_icons():
    """The in-game icon crops (labelled + learned) as (names, matrix): the game draws some items
    (Needlessly Large Rod, Sparring Gloves, Negatron Cloak...) differently from the official art."""
    global _gbank
    if _gbank is None or _gbank[2] != len(_game_bank()):
        g = _game_bank()
        _gbank = ([n for n, _ in g], np.stack([v for _, v in g]) if g else np.zeros((0, 256), np.float32), len(g))
    return _gbank[:2]


def marker_points(frame: np.ndarray, marker=None, thresh=None):
    """-> (points, x0, y0): marker centres in frame pixels relative to the search region's corner (x0, y0). With
    another template (`marker`), its own matches at `thresh`, unconfirmed."""
    h, w = frame.shape[:2]
    s = h / 1080
    x0, x1 = round(w / 2 + (REGION[0] - 960) * s), round(w / 2 + (REGION[2] - 960) * s)
    y0, y1 = round(REGION[1] * s), round(REGION[3] * s)
    # matched at half resolution (4x faster: the carousel loop must tick several times a second)
    g = cv2.resize(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY), None, fx=0.5, fy=0.5,
                   interpolation=cv2.INTER_AREA)
    t = cv2.resize(MARKER if marker is None else marker, None, fx=s / 2, fy=s / 2, interpolation=cv2.INTER_AREA)
    r = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(r >= (MARKER_SEARCH if marker is None else thresh))
    order = np.argsort(-r[ys, xs])
    pts, tops = [], []
    for k in order:
        cx, cy = 2 * (xs[k] + t.shape[1] / 2), 2 * (ys[k] + t.shape[0] / 2)
        if all((cx - p[0]) ** 2 + (cy - p[1]) ** 2 > (25 * s) ** 2 for p in pts):
            pts.append((cx, cy))
            tops.append((xs[k], ys[k]))
    if marker is None and pts:
        d = cv2.matchTemplate(g, cv2.resize(DRAWN_MARKER, t.shape[::-1], interpolation=cv2.INTER_AREA),
                              cv2.TM_CCOEFF_NORMED)
        pts = [p for p, (x, y) in zip(pts, tops)
               if d[max(0, y - 2):y + 3, max(0, x - 2):x + 3].size
               and float(d[max(0, y - 2):y + 3, max(0, x - 2):x + 3].max()) >= MARKER_DRAWN]
    return pts, x0, y0


def find_carousel_items(frame: np.ndarray):
    """-> [(x, y, item_name|None)] marker positions (1080p) and the item below."""
    h, w = frame.shape[:2]
    s = h / 1080
    pts, x0, y0 = marker_points(frame)
    half = round(ICON_HALF * s)
    names, bank = _icon_bank(2 * half)
    gnames, gbank = _game_icons()
    out = []
    for cx, cy in pts:
        fx, fy = x0 + cx, y0 + cy + ICON_DY * s
        best, bj = -1.0, 0
        for dy in range(-4, 5, 2):
            for dx in range(-4, 5, 2):
                yy, xx = round(fy + dy), round(fx + dx)
                c = frame[yy - half:yy + half, xx - half:xx + half]
                if c.shape[:2] != (2 * half, 2 * half):
                    continue
                sims = bank @ _vec(c)
                j = int(sims.argmax())
                if sims[j] > best:
                    best, bj = float(sims[j]), j
                if len(gnames):
                    gs = gbank @ _icon_fp(c) - GAME_FP_LEAD
                    j = int(gs.argmax())
                    if gs[j] > best:
                        best, bj = float(gs[j]), len(names) + j
        item = (names + gnames)[bj] if best > 0.75 else None
        out.append(((x0 + cx - w / 2) / s + 960, (y0 + cy) / s, item))
    return out


# release schedule (measured on run29-31 carousels): players leave their gates in pairs, lowest
# HP first; the last pair at round timer ~10, each earlier pair 6s before (5 alive: 22/16/10). The gate
# opens ~1.5 s after the timer shows that number (13 carousels where we stood at the gate edge: +0.7..+2.3 s,
# median +1.7 incl. ~0.2 s to step past it; the timer is read as a whole number)
LAST_RELEASE, RELEASE_GAP = 8.5, 6.0
LIST_ROW0, LIST_ROW_DY = 198, 80         # player list rows (1080p, right-anchored)
# our gate's countdown: a bright blue circle around it, shown ~5 s before it opens and gone the moment it does
# (user, 2026-10-14; run 20261013-220557 2-4/3-4: shown 12.7/12.8 s, gone 18.0/18.7 s, schedule ~17.5/18 s;
# run 20261014-032220 2-4: shown from 0 s, gone 6.7 s while the schedule - misread - said 33 s). Counted in an
# annulus around it, in 1080p pixels: ~500-800 without it (a blue Little Legend included), 2,800+ with it.
GATE_RING = (945, 890, 95, 180)          # centre x, y (1080p, centred) and radii
GATE_RING_ON, GATE_RING_OFF = 2500, 1500


def gate_ring(frame: np.ndarray):
    """Bright blue pixels (1080p units) of the countdown circle around our carousel gate."""
    h, w = frame.shape[:2]
    s = h / 1080
    cx, cy, r0, r1 = GATE_RING
    x0, x1 = round(w / 2 + (cx - r1 - 960) * s), round(w / 2 + (cx + r1 - 960) * s)
    y0, y1 = round((cy - r1) * s), min(h, round((cy + r1) * s))
    crop = frame[y0:y1, max(0, x0):min(w, x1)].astype(np.int16)
    ys, xs = np.mgrid[y0:y1, max(0, x0):min(w, x1)]
    d = np.hypot(960 + (xs - w / 2) / s - cx, ys / s - cy)
    r, b = crop[..., 0], crop[..., 2]                      # RGB frames
    blue = (d > r0) & (d < r1) & (b > 200) & (r < 120) & (b - r > 110)
    return int(blue.sum() / (s * s))


def release_timer(frame: np.ndarray):
    """-> (round timer value at which our gate opens, our row, players alive) or None.
    The player list is sorted by HP and the bottom pair goes first. Our row is the one whose
    portrait has a gold ring; the others have red ones (a gold-border rule misread run31's
    list: the digits in our HP box pair up with its top border)."""
    h, w = frame.shape[:2]
    s = h / 1080
    rows = []
    for k in range(8):
        y = round((LIST_ROW0 + LIST_ROW_DY * k) * s)
        reg = frame[y - round(38 * s):y + round(38 * s), round(w - (1920 - 1835) * s):round(w - (1920 - 1918) * s)]
        reg = reg.astype(np.int16)
        r, g, b = reg[..., 0], reg[..., 1], reg[..., 2]
        rows.append((int(((r > 150) & (g < 90) & (b < 90)).sum()),
                     int(((r > 170) & (g > 130) & (b < 90) & (r - b > 90)).sum())))
    mine = [k for k, (red, gold) in enumerate(rows) if red < 60 * s * s and gold > 100 * s * s]
    if len(mine) != 1:
        return None
    ours = mine[0]
    alive = max(k + 1 for k, (red, _) in enumerate(rows) if k == ours or red > 150 * s * s)
    pairs = (alive + 1) // 2
    p = (alive - 1 - ours) // 2                                # 0 = the first pair out
    return LAST_RELEASE + RELEASE_GAP * (pairs - 1 - p), ours, alive


def carried(frame: np.ndarray, markers):
    """Which markers belong to a unit a player already carries: its owner's HP bar sits just
    below the marker (~45-53px, measured on run29), where a free unit has none. -> [bool]."""
    h, w = frame.shape[:2]
    s = h / 1080
    out = []
    for x, y, _ in markers:
        # the owner stands beside its unit: its bar is up to ~150px to the side (run31 4-4)
        X0, X1 = round(w / 2 + (x - 150 - 960) * s), round(w / 2 + (x + 150 - 960) * s)
        Y0, Y1 = round((y + 20) * s), round((y + 100) * s)
        reg = frame[max(0, Y0):Y1, max(0, X0):X1].astype(np.int16)
        hit = False
        if reg.shape[0] > 10 and reg.shape[1] > 10:
            mx, mn = reg.max(axis=2), reg.min(axis=2)
            mask = (mx - mn > 90) & (mx > 150)
            th = max(3, round(5 * s))
            thick = mask[:-th] & mask[th:]
            for yy in np.where(thick.sum(axis=1) >= round(12 * s))[0]:
                if yy < 3:
                    continue
                d = np.diff(np.concatenate(([0], thick[yy].view(np.int8), [0])))
                for a_, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
                    if e - a_ >= round(12 * s) and reg[yy - 3, a_:e].max(axis=1).mean() < 90                             and reg[yy + 1, a_:e].std(axis=0).max() < 16:
                        hit = True
                        break
                if hit:
                    break
        out.append(hit)
    return out


def offer_text(markers):
    """What the carousel offers, for the model: 'B.F. Sword x2, Chain Vest, ...' (unread icons last)."""
    from collections import Counter
    c = Counter(it for _, _, it in markers if it)
    unread = sum(1 for _, _, it in markers if not it)
    parts = [f'{k} x{n}' if n > 1 else k for k, n in c.most_common()]
    return ', '.join(parts) + (f' (+{unread} unread)' if unread else '')


def _key(text):
    return ''.join(ch for ch in str(text).lower() if ch.isalnum())


# what a model may write for an item: the exact names (any case/punctuation) and the usual short forms
_BY_KEY = {_key(n): n for n in ICONS}
for _short, _full in {'ie': 'Infinity Edge', 'gs': 'Giant Slayer', 'lw': 'Last Whisper', 'bt': 'Bloodthirster',
                      'jg': 'Jeweled Gauntlet', 'hoj': 'Hand Of Justice', 'grb': "Guinsoo's Rageblade",
                      'rageblade': "Guinsoo's Rageblade", 'shojin': 'Spear of Shojin', 'sos': 'Spear of Shojin',
                      'rabadon': "Rabadon's Deathcap", 'deathcap': "Rabadon's Deathcap", 'sterak': "Sterak's Gage",
                      'steraks': "Sterak's Gage", 'gargoyle': 'Gargoyle Stoneplate', 'warmog': "Warmog's Armor",
                      'nashor': "Nashor's Tooth", 'nashors': "Nashor's Tooth", 'kraken': "Kraken's Fury",
                      'archangel': "Archangel's Staff", 'archangels': "Archangel's Staff", 'bf': 'B.F. Sword',
                      'bfsword': 'B.F. Sword', 'rod': 'Needlessly Large Rod', 'nlr': 'Needlessly Large Rod',
                      'tear': 'Tear of the Goddess', 'belt': "Giant's Belt", 'bow': 'Recurve Bow',
                      'gloves': 'Sparring Gloves', 'vest': 'Chain Vest', 'cloak': 'Negatron Cloak',
                      'eon': 'Edge of Night', 'dclaw': "Dragon's Claw", 'qss': 'Quicksilver',
                      'gunblade': 'Hextech Gunblade', 'titans': "Titan's Resolve", 'tr': "Titan's Resolve",
                      'morello': 'Morellonomicon',
                      'protectors': "Protector's Vow", 'strikers': "Striker's Flail", 'thieves': "Thief's Gloves",
                      'tg': "Thief's Gloves", 'visage': 'Spirit Visage', 'sunfire': 'Sunfire Cape',
                      'bramble': 'Bramble Vest', 'helm': 'Adaptive Helm', 'adaptive': 'Adaptive Helm'}.items():
    _BY_KEY.setdefault(_short, _full)


def resolve_item(text):
    """What the model wrote -> the exact item name we recognise ('IE' -> 'Infinity Edge'), or None."""
    if not isinstance(text, str):
        return None
    k = _key(text)
    if k in _BY_KEY:
        return _BY_KEY[k]
    return next((n for n in _game_icons()[0] if _key(n) == k), None)      # emblems, duplicators...


def _ranked(obj):
    """The ranked item names of an answer's JSON object, resolved, no duplicates -> [item] or None."""
    if not isinstance(obj, dict):
        return None
    want = obj.get('carousel_priority') or obj.get('item_targets')
    if not isinstance(want, list):
        return None
    return list(dict.fromkeys(r for r in map(resolve_item, want) if r)) or None


def priority_prompt(markers, state, targets):
    """The question asked as soon as the open carousel's offer is read (v2). The chase runs on the
    prepared list (v1) meanwhile and switches to this answer if it comes before we commit to a unit."""
    return (f"CAROUSEL OPEN - the units on the carousel hold: {offer_text(markers)}.\n"
            "The screenshot shows the carousel; each unit's item is written beside its icon, so you can see "
            "which champion carries which item (we get both).\n"
            f"Our state: {state}\n"
            f"Your prepared priority (running now): {', '.join(targets) or 'none'}.\n"
            "Rank EVERY item on offer, best first. The controller "
            "goes for the first one still there and moves down the list when another player takes it; this "
            "list replaces the prepared one if it arrives before we commit to a unit. Answer IMMEDIATELY with "
            'this JSON object first, one line, reasons after it: {"carousel_priority": ["<item on offer>", ...]}')


def prep_prompt(state):
    """The carousel preparation question, asked while the fight before the carousel is running. It does
    not know what the carousel will offer, so it is one list over every item: the controller keeps what is
    on offer, in this order, and asks again only when none of it is there."""
    return ("CAROUSEL PREPARATION - the next round is the item carousel, and there is no planning phase "
            "before it. You cannot see what it will offer, so rank EVERY item you would want from it, best "
            "first, completed items and components mixed in your own order (10-15 entries). When the "
            "carousel opens the controller keeps the items that are on it, in your order, and goes for the "
            "first one still there. Items: "
            + ', '.join(sorted(ICONS)) + ".\n"
            f"Our state: {state}\n"

            "Answer IMMEDIATELY with this JSON object first, one line, reasons after it: "
            '{"carousel_priority": ["<item>", ...]}')


def parse_prep(obj):
    """The prepared ranking -> [item] (exact names, in order) or None."""
    return _ranked(obj)


def prepared_for_offer(prepared, markers):
    """The prepared ranking kept to what is on offer, in its order -> [item] or None (no overlap)."""
    offer = {it for _, _, it in markers if it}
    return [w for w in prepared or [] if w in offer] or None


def parse_priority(obj, markers):
    """The model's ranked list, kept to items on offer -> [item] or None. While some units' items are
    unread, the model's other names stay: it sees them on the screenshot (run 233635 2-4: Rod, Chain Vest,
    Tear and Bow were dropped as not yet read, and the chaser went for its 4th choice)."""
    offer = {it for _, _, it in markers if it}
    unread = any(not it for _, _, it in markers)
    return [w for w in _ranked(obj) or [] if w in offer or unread] or None


BASE = {  # what a component is worth before we know our comp
    'AD': {'B.F. Sword': 8, 'Recurve Bow': 7, 'Sparring Gloves': 7, 'Tear of the Goddess': 5,
           'Needlessly Large Rod': 4, "Giant's Belt": 5, 'Chain Vest': 5, 'Negatron Cloak': 5},
    'AP': {'Needlessly Large Rod': 8, 'Tear of the Goddess': 7, 'Sparring Gloves': 7, 'Recurve Bow': 6,
           'B.F. Sword': 4, "Giant's Belt": 5, 'Chain Vest': 5, 'Negatron Cloak': 5},
}


def item_want(item, bench_items, roster, plan):
    """How much we want this carousel item."""
    if item is None:
        return 0.5                              # at least grab the unit
    targets = getattr(plan, 'item_targets', None) or []
    if plan is None:
        targets = []
    if item in targets:
        return 20.0 + (len(targets) - targets.index(item))      # named in the plan; earlier = wanted more
    if item in ('Spatula', 'Frying Pan'):
        return 7.5
    if item not in COMPONENTS:
        return 9.0                              # completed items are rare and strong
    board = [u for u in roster.units if u.info]
    kinds = [holder_kind(u) for u in board]
    carry_kind = 'AP' if kinds.count('AP') > kinds.count('AD') else 'AD'
    if plan is not None and plan.carry:
        from tft_player.state import Owned
        k = holder_kind(Owned(plan.carry))
        # (a tank carry has no component table: the board's AD/AP majority stands - replay of run 20261010-235723
        # 3-4 raised KeyError 'tank' every carousel tick)
        carry_kind = k if k in BASE else carry_kind
    v = BASE[carry_kind].get(item, 4)
    # completes something good with a component we already hold
    for c in [b for b in bench_items or [] if b in COMPONENTS]:
        done = BY_PAIR.get(tuple(sorted((c, item))))
        if done:
            v = max(v, 0.9 * max(VALUE['AD'].get(done, 0) if carry_kind == 'AD' else VALUE['AP'].get(done, 0),
                                 VALUE['tank'].get(done, 0)))
    return v


# ---- geometry (1080p, measured on 30 recorded carousels) ----------------------------
# The units stand on a circle seen in perspective: on screen an ellipse with a fixed
# centre and height/width K. Its radius grows with the number of units (9 units ~210px,
# 11 ~237px), so it is fitted every tick from the markers. Units turn clockwise on
# screen at ~0.30 rad/s (a lap in ~21s); the speed is also measured live.
# Planning runs in "world" coordinates: screen x, and screen y stretched by 1/K about
# the centre. There the ring is a circle and a pickup range or a walking speed is the
# same in every direction.
MARKER_C = (967.0, 312.0)
K = 0.65                # the ring's height/width on screen: 0.62-0.74 per carousel (camera framing), median 0.65 over
                        # the first looks of 25 carousels of 2026-10-10 (0.70 put top units at 0.8 of the radius)
PICKUP = 55.0           # world px: touching a unit within this grabs it
RIM_SPEED = 72.0        # world px/s the units move along the ring, until the turn is measured: the radius and the
                        # turn vary with the number of units (37 recorded carousels: R ~207 -> ~0.35 rad/s,
                        # R ~240 -> ~0.30, R 271 -> 0.20; run 010345 2-4 measured 0.42 at R 210), the rim speed less
TAKEN_AFTER = 1.2       # s a target's marker must stay missing before it counts as taken; until then it
                        # turns on with the ring's measured speed (0.6 s was one slow tick: run 213127's
                        # chase marked B.F. Swords 'taken' four times in one carousel). Seen carried: at once
FOLLOW_WALK = 40.0      # world px we walk with an icon keeping its place beside us before it counts as the unit we hold
SEEN_FRESH = 0.45       # s: a cut onto the ring only for a target seen this recently ...
NEAR_HIDE = 1.0         # s: ... for at most this long (longer hidden is hidden, however near)
NEAR_FRESH = 80.0       # world px: ... or this close to it (our own bar covers its icon as we reach it)
HIDDEN_GIVEUP = 4.0     # s hidden: given up as taken (a player took it unseen - run 20261012-194821 3-4, navE). 2.5 s gave up targets still free (235723 6-4's Jeweled Gauntlet, hidden 2.8 s); waiting is safe now (no touch on a hidden target, out of the band)
KEEP_HIDDEN = 4.0       # s a unit seen >= 3 times on the ring stays (turned on with the ring) while its icon is hidden
GATE = (931.0, 845.0)
GATE_OUT = 300.0        # world px from GATE: out of the gate (a closed gate lets us stand ~270 out: 37 carousels)
LOOK = 3.0              # s before our release: the second read (what is left, where, how fast it turns) starts
COMMIT = 0.4            # s before our release at the latest: the target and the route are fixed from that read   # our feet in our gate (bottom centre) before release


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Ring:
    """The unit ring, refitted from the markers every tick."""

    def __init__(self):
        self.cx, self.cy, self.R = MARKER_C[0], MARKER_C[1] + UNIT_DY, 210.0
        self.fitted = False
        self.last_r = []             # radius / R of each free unit fit() returned
        self.off = []                # markers of units already taken: (world pos, item)
        self.off_free = []
        self.held = []               # markers seen carried by a player (carried()): (world pos, item)
        self.all = []                # every marker: (world pos, item)

    def world(self, p):
        return p[0] - self.cx, (p[1] - self.cy) / K

    def screen(self, q):
        return self.cx + q[0], self.cy + K * q[1]

    def at(self, a, r=None):
        """World point at angle a, radius r (default: on the ring)."""
        r = self.R if r is None else r
        return r * math.cos(a), r * math.sin(a)

    def _on(self, markers, ref):
        pts = [(self.world((x, y + UNIT_DY)), it) for x, y, it in markers]
        return [(q, it) for q, it in pts if abs(math.hypot(*q) / ref - 1) < 0.25]

    def fit(self, markers, held=None):
        """markers [(x, y, item)] -> free units [(angle, item)]. A unit a player took follows
        that player off the ring with its marker still up: those are left out (kept in `off`)."""
        self.off, self.all, self.held = [], [], []
        if not markers:
            return []
        if held:                                              # carried units: never free
            held_off = [(self.world((x, y + UNIT_DY)), it) for (x, y, it), c in zip(markers, held) if c]
            free = [m for m, c in zip(markers, held) if not c]
            self.off_free = []
            out = self.fit(free) if free else []
            self.off = held_off + self.off_free
            self.held = held_off
            self.all = [(self.world((x, y + UNIT_DY)), it) for x, y, it in markers]
            return out
        if self.fitted:
            ref = self.R
        else:
            # the units still on the ring: radius 200-260 for 12-9 units (37 recorded carousels). Late in a carousel
            # most markers are carried off by players: their median was ~500 (sim, last release wave)
            rs = [r for r in (math.hypot(*self.world((x, y + UNIT_DY))) for x, y, _ in markers) if 170 < r < 300]
            if len(rs) < 2:
                return []
            ref = float(np.median(rs))
        on = self._on(markers, ref)
        if len(on) >= 6:
            # circle fit in world space (x^2 + y^2 + Dx + Ey + F = 0) for the centre
            A = np.array([[q[0], q[1], 1.0] for q, _ in on])
            b = -np.array([q[0] ** 2 + q[1] ** 2 for q, _ in on])
            D, E, _ = np.linalg.lstsq(A, b, rcond=None)[0]
            dx, dy = -D / 2, -E / 2 * K                       # centre shift, screen px
            if abs(dx) < 30 and abs(dy) < 30:
                k = 1.0 if not self.fitted else 0.2
                self.cx += k * dx
                self.cy += k * dy
                on = self._on(markers, ref)
        if len(on) >= (3 if not self.fitted else 5):
            # refitted only from a full-ish ring: 3-4 late units pulled R from 225 to 202 (run 100905 2-4)
            r = float(np.median([math.hypot(*q) for q, _ in on]))
            self.R = r if not self.fitted else 0.7 * self.R + 0.3 * r
            self.fitted = True
        pts = self.all = [(self.world((x, y + UNIT_DY)), it) for x, y, it in markers]
        self.off = self.off_free = [(q, it) for q, it in pts if abs(math.hypot(*q) / self.R - 1) >= 0.25]
        self.last_r = [math.hypot(*q) / self.R for q, _ in on]
        return [(math.atan2(q[1], q[0]), it) for q, it in on]


# overlay panels perception must not mistake for a tactician bar (overlay.PANELS, 1080p, centred)
from tft_player.overlay import PANELS as _OVERLAY_PANELS  # noqa: E402
_PANELS = list(_OVERLAY_PANELS.values())


AMBER = (255, 176, 46)       # our overlay's target ring and label: not a player's bar


def find_bars(frame: np.ndarray):
    """Every player's HP bar on the carousel (each player has its own colour): [(x_left, y_top, rgb, width)]
    in 1080p reference coordinates. The same test as OwnTracker without its colour: a flat-coloured fill
    >= 8 px tall with a dark frame just above it; never inside our overlay's panels or its amber label."""
    h, w = frame.shape[:2]
    s = h / 1080
    X = lambda x: round(w / 2 + (x - 960) * s)
    bx0, by0 = max(0, X(380)), round(40 * s)
    bx1, by1 = min(w, X(1540)), min(h, round(900 * s))
    reg = frame[by0:by1, bx0:bx1].astype(np.int16)
    if reg.shape[0] < 20 or reg.shape[1] < 20:
        return []
    mx, mn = reg.max(axis=2), reg.min(axis=2)
    mask = (mx - mn > 90) & (mx > 150)
    for x0, y0, x1, y1 in _PANELS:
        px0, px1 = X(x0) - bx0, X(x1) - bx0
        py0, py1 = round(y0 * s) - by0, round(y1 * s) - by0
        mask[max(0, py0):max(0, py1), max(0, px0):max(0, px1)] = False
    th = max(3, round(6 * s))
    thick = mask[:-th] & mask[th:]
    need = max(6, round(9 * s))
    out = []
    for yy in np.where(thick.sum(axis=1) >= need)[0]:
        if yy < 3:
            continue
        d = np.diff(np.concatenate(([0], thick[yy].view(np.int8), [0])))
        runs = []
        for a, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if runs and a - runs[-1][1] <= max(2, round(4 * s)):
                runs[-1][1] = e
            else:
                runs.append([a, e])
        for a, e in runs:
            if not need <= e - a <= round(115 * s):
                continue
            if reg[max(0, yy - round(3 * s)), a:e].max(axis=1).mean() > 90:
                continue                                      # a bar has a dark frame above its fill
            if reg[yy + 1, a:e].std(axis=0).max() > 14:
                continue                                      # ... and one flat colour along it
            col = np.median(reg[yy:yy + th, a:e].reshape(-1, 3), axis=0).astype(int).tolist()
            if col[0] > 225 and 150 < col[1] < 200 and col[2] < 70:
                continue                                      # our overlay's amber ring and label
            x, y = (bx0 + a - w / 2) / s + 960, (by0 + yy) / s
            if any(abs(x - o[0]) < 30 and abs(y - o[1]) < 14 for o in out):
                continue
            out.append((x, y, col, (e - a) / s))
    return out


class Legends:
    """The other players on the carousel, one per bar colour (the user, 2026-10-09: watch the little legends,
    not the items). Each player takes ONE unit: a player is 'free' until a unit walks with it (an item icon
    beside it that moves as it moves), then 'carrying' for good. So a unit is taken only when a FREE player
    starts carrying it at its place; a carrying player walking through the ring changes nothing (run 185805:
    B.F. Sword 'taken' 7x at 2-4, Negatron Cloak 7x at 4-4, by players passing with their own units)."""

    SAME = 35          # max colour difference within one player's bar
    SIDE, ABOVE = (150, (20, 100))     # a carried unit's icon: up to 150 px beside its owner's bar, 20-100 px above it

    def __init__(self):
        self.p = []
        self.ticks = 0

    REAL = 3           # a player: its colour and bar width seen on >= 3 ticks (icons and glows flicker)

    def real(self):
        """Players: a bar seen on >= REAL ticks and on at least half the ticks since it first showed (a player's
        bar is there nearly every tick; an icon's glow or a passing effect comes and goes)."""
        return [L for L in self.p if L['hits'] >= self.REAL and L['hits'] >= 0.5 * (self.ticks - L['first'] + 1)]

    @property
    def ready(self):
        return len(self.real()) >= 2

    def _beside(self, bar, markers):
        x, y = bar
        return [m for m in markers if abs(m[0] - (x + 30)) <= self.SIDE and self.ABOVE[0] <= y - m[1] <= self.ABOVE[1]]

    def update(self, bars, markers, now, own=None):
        """bars: find_bars(); markers: [(x, y, item)]; own: our feet (our bar is not another player's)."""
        self.ticks += 1
        owner = {}                                         # each marker's nearest bar: the only one that can carry it
        for m in markers:
            cands = [(abs(b[0] + 30 - m[0]), k) for k, b in enumerate(bars)
                     if abs(b[0] + 30 - m[0]) <= self.SIDE and self.ABOVE[0] <= b[1] - m[1] <= self.ABOVE[1]]
            if cands:
                owner[m] = min(cands)[1]
        for k, (x, y, col, wd) in enumerate(bars):
            feet = (x + 45, y + 62)
            if own is not None and math.dist(feet, own) < 45:
                continue
            # one player = one colour and one bar width (HP stays put in a carousel; a unit in front can only
            # shorten the fill)
            same = [L for L in self.p if max(abs(a - b) for a, b in zip(L['col'], col)) < self.SAME
                    and wd <= L['w'] * 1.15 + 5]
            L = min(same, key=lambda L: math.dist(L['feet'], feet), default=None)
            if L is None:
                L = {'col': col, 'w': wd, 'hits': 0, 'first': self.ticks, 'feet': feet, 'bar': (x, y), 'seen': now,
                     'carrying': False, 'carry': None, 'near': [], 'near_t': None, 'co': 0, 'co_item': None}
                self.p.append(L)
            L['hits'] += 1
            L['w'] = max(L['w'], wd)
            step = (feet[0] - L['feet'][0], feet[1] - L['feet'][1])
            L['feet'], L['bar'], L['seen'] = feet, (x, y), now
            if L['carrying']:
                continue
            near = [m for m in self._beside((x, y), markers) if owner.get(m) == k]
            moved_with = None
            if math.hypot(*step) > 40 and L['near']:
                # a unit walking with it: the same item beside it before and now, moved by the player's own step
                for m in near:
                    prev = [q for q in L['near'] if q[2] == m[2]]
                    if prev:
                        q = min(prev, key=lambda q: math.dist(q[:2], m[:2]))
                        if math.hypot(m[0] - q[0] - step[0], m[1] - q[1] - step[1]) < 45:
                            moved_with = (m, q)
                            break
            if moved_with and moved_with[0][2] == L['co_item']:
                L['co'] += 1                               # two steps in a row with the same unit: it is carrying
                if L['co'] >= 2:
                    L['carrying'], L['carry'] = True, (moved_with[0][2], L['co_at'][0], L['co_at'][1])
            elif moved_with:
                L['co'], L['co_item'], L['co_at'] = 1, moved_with[0][2], ((moved_with[1][0], moved_with[1][1]), L['near_t'] or now)
            elif math.hypot(*step) > 40:
                L['co'], L['co_item'] = 0, None            # walked without it: not carrying (yet)
            L['near'], L['near_t'] = near, now

    def held(self, markers):
        """Per marker: is it the unit a carrying player walks with?"""
        return [any(L['carrying'] and L['carry'][0] == m[2] and m in self._beside(L['bar'], [m]) for L in self.real())
                for m in markers]

    def took_near(self, at, item, now, r=150, within=3.0):
        """Did a player start carrying `item` at screen point `at` within the last `within` s?"""
        return any(L['carrying'] and L['carry'][0] == item and math.dist(L['carry'][1], at) < r
                   and now - L['carry'][2] < within for L in self.real())

    def free_near(self, at, now, r=150):
        """A free player (seen in the last second) right at screen point `at` - one that could take that unit."""
        return any(not L['carrying'] and now - L['seen'] < 1.0 and abs(L['bar'][0] + 30 - at[0]) <= r
                   and -20 <= L['bar'][1] - at[1] <= 140 for L in self.real())


class OwnTracker:
    """Our tactician during a carousel. Every player has a coloured HP bar and ours is not
    always green, but we always start in the bottom-centre gate: the bar there gives our
    colour, and from then on we follow that colour near where we last were."""

    def __init__(self):
        self.color = None
        self.width = None            # our fill's width (1080p px), learned in the gate
        self.feet = None
        self.seen_at = 0.0

    def find(self, frame, now=None, hp=None):
        """hp: our HP, when known - our fill's width at first sight must match it (~0.95 px per HP at 1080p: HP 78 ->
        78 px, 91 -> 87, 96 -> 90-94 in the recorded carousels); run 20261010-121458 3-4 took the gate's blue glow
        (24 px) for our bar at HP 96 and followed it all carousel."""
        import time as _t
        now = _t.time() if now is None else now
        h, w = frame.shape[:2]
        s = h / 1080
        X = lambda x: round(w / 2 + (x - 960) * s)
        if self.color is None:
            box = (780, 560, 1100, 880)                       # first: the gate
        elif self.feet is not None and now - self.seen_at < 1.0:
            fx, fy = self.feet                                # near where we were (fast)
            r = 160 + 500 * (now - self.seen_at)
            box = (fx - 45 - r, fy - 62 - r, fx + 70 + r, fy - 62 + r)
        else:
            box = (380, 40, 1540, 900)
        bx0, by0 = max(0, X(box[0])), max(0, round(box[1] * s))
        bx1, by1 = min(w, X(box[2])), min(h, round(box[3] * s))
        reg = frame[by0:by1, bx0:bx1]
        if reg.shape[0] < 20 or reg.shape[1] < 20:
            return None
        # (OpenCV on the 8-bit pixels, the same masks as numpy int16 arithmetic did: ~4x faster, and it lets the
        # other threads run meanwhile - the int16 max/abs were 35% of the chase loop's time in run 20261010-235723)
        if self.color is None:
            ch = cv2.split(np.ascontiguousarray(reg))
            mx, mn = cv2.max(cv2.max(ch[0], ch[1]), ch[2]), cv2.min(cv2.min(ch[0], ch[1]), ch[2])
            mask = (cv2.subtract(mx, mn) > 90) & (mx > 150)
        else:
            c = np.array(self.color, np.int16)
            mask = cv2.inRange(np.ascontiguousarray(reg), np.clip(c - 34, 0, 255).astype(np.uint8),
                               np.clip(c + 34, 0, 255).astype(np.uint8)) > 0
        for x0, y0, x1, y1 in _PANELS:                        # never our own overlay
            px0, px1 = X(x0) - bx0, X(x1) - bx0
            py0, py1 = round(y0 * s) - by0, round(y1 * s) - by0
            mask[max(0, py0):max(0, py1), max(0, px0):max(0, px1)] = False
        th = max(3, round(6 * s))
        thick = mask[:-th] & mask[th:]                        # a bar fill is >= 8px tall
        need = max(6, round(9 * s))
        best = None
        for yy in np.where(thick.sum(axis=1) >= need)[0]:
            d = np.diff(np.concatenate(([0], thick[yy].view(np.int8), [0])))
            runs = []
            for a, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
                if runs and a - runs[-1][1] <= max(2, round((4 if self.color is None else 14) * s)):
                    runs[-1][1] = e                           # tick marks (or a unit in front) split a fill
                else:
                    runs.append([a, e])
            for a, e in runs:
                if not need <= e - a <= round(115 * s):
                    continue
                # HP does not change during a carousel: our fill keeps the width it had in the gate
                # (another player's bar of a similar colour, or a blue item icon, does not). A unit
                # or marker in front can hide part of it: down to 45% (icons are under 25px)
                if self.width and not 0.45 * self.width <= (e - a) / s <= 1.15 * self.width + 5:
                    continue
                if self.width is None and hp and abs((e - a) / s - 0.95 * hp) > 12:
                    continue
                # a bar has a dark frame just above its fill (a glowing gate arc does not)
                top = reg[max(0, yy - round(3 * s)), a:e]
                if yy < 3 or top.max(axis=1).mean() > 90:
                    continue
                # ... and one flat colour along it (a glow fades along its length)
                if reg[yy + 1, a:e].std(axis=0).max() > 14:
                    continue
                feet = ((bx0 + a - w / 2) / s + 960 + 45, (by0 + yy) / s + 62)
                # first sight: the longest fill; then the one nearest where we were
                key = -(e - a) if self.feet is None else math.dist(feet, self.feet)
                if best is None or key < best[0]:
                    best = (key, feet, (yy, a, e))
        if best is None:
            return None
        _, feet, (yy, a, e) = best
        if self.feet is not None and math.dist(feet, self.feet) > 150 + 700 * (now - self.seen_at):
            return None                                       # farther than we can have walked
        if self.color is None:
            fill = reg[yy:yy + th, a:e].reshape(-1, 3)
            self.color = np.median(fill, axis=0).astype(int).tolist()
            self.width = (e - a) / s
        self.feet, self.seen_at = feet, now
        return feet


FEET = (-9.0, 14.0)     # 1080p px from OwnTracker's estimate to the bottom centre of our legend - where it touches units
                        # (2.5D: the feet, not the bar - the user, 2026-10-12). Measured on the frames of run
                        # 20261012-194821: 2-4 (level 4) -10,+12; 3-4 (level 6) -8,+16. Fixed: the old calibration in
                        # the gate and at every stop read -8 and +31 there (the legend stopped short in the closed gate;
                        # two equal reads were taken for a stop on the clicked point), 20-40 world px off our real feet
LANE_OUT = 60.0         # world px beyond the units' reach (R + PICKUP): the outside lane, where no unit touches us
CLEAR = 30.0            # world px kept between our walk and any other unit's reach (timing: 0.1 s is 19 px of our walk)
LANE_CLEAR = 50.0       # world px between the outside lane route and the units' reach: units stand up to ~15 px off the
                        # fitted ring (replay of run 105105 5-4: a lane at +30 touched the Dragon's Claw at the bottom)
AHEAD = (0.30, 0.22, 0.40, 0.14)   # rad ahead of the target's meeting place (the way the ring turns) the cut starts from (straight in, 0 and 0.07, tested 2026-10-12 on 2-4 and 3-4 of run 20261012-194821: 46/48 against 47/48 - not better)
LOCK_READS = 3          # reads of a unit's icon before its item counts as identified ...
LOCK_SHARE = 0.8        # ... with this share agreeing; from then on the item is fixed for the carousel (a unit read 6x Rod
                        # and 4x Giant's Belt is not identified: run 20261012-194821 2-4 chased it and got the Belt)
TRUST = 70.0            # world px: a detection this far from where we walked to is another bar or a glow - ignored
LOST_TAKEN = 8.0        # s unseen on its circle before a target counts as taken without being seen carried


class CarouselChaser:
    """Meet the wanted unit head-on (the user, 2026-10-11, after watching runs 20261010-100905/121458/130538).

    Where we are comes from where we clicked: we start in our gate, and the legend walks straight to the clicked
    point at its speed. Our name bar is read only for how far it moved and to recalibrate at every stop - its height
    over the feet changes with our level (38-88 px), and the old absolute reading was ~70 px off, more than the
    pickup range.

    The target is followed on its circle: every unit turns with the ring at one measured angular speed, so where it
    is at any moment is known. An icon hidden for a while (a player or a bar in front) is not taken: it turns on
    (run 121458 3-4 dropped the Recurve Bow after 4 s unseen behind two standing players; it turned on, untouched).
    Taken = its item seen OFF the circle near its place, away from us - a carried unit follows its player.

    The approach: along the outside lane (beyond every unit's reach) to a point just AHEAD of the target in the
    direction the ring turns, then a straight cut back against the turn onto it - never from behind (a cut from
    behind chases a unit that walks away: runs 100905 3-4 and 130538 5-4 circled their target for 7-10 s). The unit
    ahead of the target has already passed the cut, the one behind is a whole gap away. Every tick replans the
    earliest such meeting from where we are; once the cut starts it is walked to the end.
    """

    def __init__(self, speed=190.0):
        self.legends = Legends()     # the other players (logged; no decision uses them any more)
        self.speed = speed           # world px/s: 189 median over 14 recorded carousels
        self.ring = Ring()
        self.omega = None            # ring angular speed, rad/s (+ = clockwise on screen)
        self.units = []              # free units: [(angle, item)], seen now or turned on for < TAKEN_AFTER s
        self.targets = []            # of those, the ones established on the ring (choosable)
        self.solid = []              # of those, the ones to keep clear of
        self.tracks = []             # [angle, item, last seen, error, radius/R, hits, id, {item: reads}, radius change]
        self.n_tracks = 0
        self.track_of, self.targets_ids = {}, []
        self.target_id = None        # the target's track: the same unit tick to tick, whatever its icon reads
        self.seen = []
        self.target_angle = None     # the target's angle at the last tick
        self.item = None
        self.meet = None             # world point: where the cut starts
        self.meet_at = None          # time.time() of the meeting
        self.contact_in = None       # s until we meet the target on the current plan
        self.stepped = False         # cutting in onto the target
        self.lost_since = None       # target not seen on its circle since
        self.off_seen = 0            # ticks in a row its item was seen off the circle near its place
        self.carried = []            # units off the ring (players carry them): {'q', 'item', 'first', 'seen'}
        self.bars = None             # every player's HP bar this look (find_bars), read while the target is unseen
        self.open_miss = 0           # looks in a row the target was missing where nothing stands in front of it
        self.prev_now = None
        self.last_click = (None, 0.0)
        self.dest = None             # screen point (1080p) we last clicked: where the legend walks
        self.taken = []              # items we went for that another player took first
        self.taken_why = []
        self.priority = []           # the model's ranking of what is on offer, best first (set_priority)
        self.source = None
        self.got = None              # the item we picked up
        self.contact = None          # (angle, time, item) when we reached the target
        self.contact_gone = None
        self.got_near = 0
        self.got_rel = self.got_from = None
        self.got_seen = 0.0
        self.path = []               # [(x, y)] 1080p: pos, corners..., target (overlay)
        self.pos = None              # our feet, world
        self.pos_s = None            # our feet, screen (1080p): kept in screen points, the ring's centre is refitted
        self.gate_s = None           # where we stood in the gate (screen)
        self.det0 = None             # our detected bar's feet estimate in the gate (screen), raw
        self.det_prev = None
        self.det_still = False       # our bar has not moved since the last look
        self.outs = 0                # looks in a row that put us out of the gate
        self.off = FEET
        self.calib = None
        self.calibrated = True       # FEET is measured, not calibrated per carousel
        self._now = 0.0
        self.ends_at = None          # when the carousel ends (time.time()), from the round timer
        self.release_at = None       # when our gate opens (time.time()), from the HP schedule
        self.released = False
        self.looked = []             # (time, [(angle, item)]) the read before our release
        self.committed = False
        self.tick_dt = 0.5           # s between ticks (measured)
        self.lag = 0.0               # s from the frame's capture to our click landing: the legend starts walking then

    # -- rotation ---------------------------------------------------------------------
    @property
    def w(self):
        return RIM_SPEED / self.ring.R if self.omega is None else self.omega

    def _update_tracks(self, seen, now):
        """Free units, kept through a missed marker (turned on by the ring's speed). The ring's angular speed is
        refined from every matched unit's step against the predicted one."""
        dt = now - (self.prev_now if self.prev_now is not None else now)
        for tr in self.tracks:
            tr[0] += self.w * dt
        # a unit moves on with the ring: its next place is known to ~0.1 rad. The window never grows with the gaps
        # (with 2-3 units left a neighbour 0.8 rad on took over a taken target's track: sim, last wave)
        gap = min(0.25, math.pi / max(len(self.tracks), len(seen), 1) * 0.8)
        rs = self.ring.last_r if len(self.ring.last_r) == len(seen) else [1.0] * len(seen)
        def sure(tr):
            """The track's identified item (locked), else None: a unit is not another item."""
            return tr[1]
        # a marker read as another item than a sure track's is not that unit - unless right at its place (a misread):
        # identities swapped onto neighbours and the chase walked onto them (replays of 2026-10-10, 3 of 7 wrong pickups)
        pairs = sorted((abs(wrap(tr[0] - a)) + (0.0 if not it or not tr[1] or tr[1] == it else 0.35), i, j)
                       for i, (a, it) in enumerate(seen) for j, tr in enumerate(self.tracks)
                       if abs(wrap(tr[0] - a)) < gap
                       and not (it and sure(tr) and it != sure(tr) and abs(wrap(tr[0] - a)) > 0.06))
        got, used = {}, set()
        for _, i, j in pairs:
            if i not in got and j not in used:
                got[i] = j
                used.add(j)
        old = list(self.tracks)
        res = []
        for i, ((a, it), rn) in enumerate(zip(seen, rs)):
            if i in got:
                tr = old[got[i]]
                d = wrap(a - tr[0])
                if tr[5] >= 2 and abs(rn - 1) < 0.12:
                    res.append(d)
                err = abs(d) / max(dt, 0.2)
                tr[3] = 0.6 * tr[3] + 0.4 * err
                tr[8] = abs(rn - tr[4]) / max(dt, 0.2)       # its radius changing (radius/R per s): carried away
                if abs(rn - 1) < 0.12:
                    h = tr[9]
                    if h[0] is None:
                        tr[9] = [now, a, a]
                    else:
                        h[2] += wrap(a - wrap(h[2]))                      # the angle turned since, unwrapped
                tr[0], tr[2], tr[4], tr[5] = a, now, rn, tr[5] + 1
                if it and tr[1] is None:
                    # identified once, for good: enough reads that agree. Before that it has no item (never a target)
                    tr[7][it] = tr[7].get(it, 0) + 1
                    n, top = sum(tr[7].values()), max(tr[7], key=tr[7].get)
                    if n >= LOCK_READS and tr[7][top] >= LOCK_SHARE * n:
                        tr[1] = top
            else:
                self.n_tracks += 1
                self.tracks.append([a, None, now, 0.0, rn, 1, self.n_tracks, {it: 1} if it else {}, 0.0,
                                    [now, a, a] if abs(rn - 1) < 0.12 else [None, None, None]])
        # one turn speed for the whole ring: each unit's total turn over a long stretch (>= 1.5 s), the median over units
        # - tick-to-tick steps read 0.22..0.40 rad/s in one carousel and a unit hidden 3.5 s was then 90 px off (run
        # 20261012-194821 2-4); the running mix of steps only until three units have a long stretch
        long = [(h[2] - h[1]) / (tr[2] - h[0]) for tr in self.tracks for h in [tr[9]]
                if h[0] is not None and tr[2] - h[0] >= 1.5 and tr[1] is not None or
                (h[0] is not None and tr[2] - h[0] >= 1.5 and tr[5] >= 6)]
        long = [w for w in long if abs(w) < 1.0]
        if len(long) >= 3:
            self.omega = float(np.median(long))
        elif len(res) >= 3 and 0.1 < dt < 2.5:
            w = self.w + float(np.median(res)) / dt
            if abs(w) < 1.0:
                self.omega = w if self.omega is None else 0.7 * self.omega + 0.3 * w
        # a unit seen on the ring several times is kept while hidden (players and bars cover icons for seconds), turned on
        # with the ring - in the way, and itself when its icon shows again (run 20261012-194821 2-4: the Sparring Gloves'
        # track, 15 sightings, was dropped after 1.2 s hidden and the unit walked into us 0.9 s later)
        self.tracks = [tr for tr in self.tracks
                       if now - tr[2] < (KEEP_HIDDEN if tr[5] >= 3 and abs(tr[4] - 1) < 0.25 else TAKEN_AFTER)]
        self.seen = seen
        self.units = [(tr[0], tr[1]) for tr in self.tracks]
        self.track_of = {tr[6]: tr for tr in self.tracks}
        # every unit standing on the ring stays in the way while its icon is hidden (turned on with the ring), however
        # few times it was seen: a unit does not vanish (run 20261012-194821 2-4 robustness runs: the Sparring Gloves
        # left this list 0.9 s before it walked into us - 2 sightings, then hidden)
        self.solid = [(tr[0], tr[1], tr[4]) for tr in self.tracks
                      if tr[2] == now or abs(tr[4] - 1) < 0.25]
        # choosable: identified and in sight just now (a hidden one stays in the way, but is not chosen - chosen hidden,
        # it was given up at the next look and the next hidden one chosen, to the end of 235723 6-4)
        self.targets_ids = [(tr[0], tr[1], tr[6]) for tr in self.tracks
                            if tr[1] is not None and tr[5] >= 3 and tr[3] < 0.3 and abs(tr[4] - 1) < 0.25
                            and now - tr[2] <= SEEN_FRESH]
        self.targets = [(a, it) for a, it, _ in self.targets_ids]

    # -- where we are -----------------------------------------------------------------
    @property
    def delay(self):
        """Seconds until we can walk."""
        if self.released or self.release_at is None:
            return 0.0
        return max(0.0, self.release_at - self._now)

    def _locate(self, det, now):
        """Our feet (self.pos, world). In the gate: the detected bar moved down to the ground. Out of it: walked
        from the last place toward the clicked point at our speed; a detection within TRUST of that pulls it
        halfway (none: the walk alone). Stopped at the clicked point: we are exactly there, and the bar's offset
        to the ground is measured again."""
        dt = 0.0 if self.prev_now is None else min(1.5, max(0.0, now - self.prev_now))
        D = (lambda p: (p[0] + self.off[0], p[1] + self.off[1]))
        if not self.released:
            if det is not None and self.pos_s is not None and self.det_prev is None and                     math.dist(self.ring.world(D(det)), self.ring.world(self.pos_s)) > 150:
                det = None                                # a jump from nowhere: another bar, until seen twice
            if det is not None:
                if self.det0 is None or self.dest is None:
                    self.det0 = det                       # standing in the gate before any click
                self.pos_s = D(det)
                self.gate_s = self.gate_s or D(self.det0)
            elif self.pos_s is None:
                self.pos_s = GATE
            if self.gate_s is None:
                self.gate_s = self.pos_s
            out = math.dist(self.ring.world(self.pos_s), self.ring.world(self.gate_s)) > GATE_OUT
            self.outs = self.outs + 1 if out else 0
            late = self.release_at is not None and now > self.release_at + 1.0 and det is None
            if self.outs >= 2 or late:
                self.released = True
        else:
            P = self.ring.world(self.pos_s)
            if self.dest is not None:
                Dw = self.ring.world(self.dest)
                d = math.dist(P, Dw)
                step = min(d, self.speed * dt)
                if d > 1e-6:
                    P = (P[0] + (Dw[0] - P[0]) * step / d, P[1] + (Dw[1] - P[1]) * step / d)
            if det is not None:
                # our bar seen: we stand FEET below it - that is where we are (the reckoned walk only fills the looks
                # without a bar). A bar far from both the walk and the last bar is another player's: skipped
                Pd = self.ring.world(D(det))
                near_last = self.det_prev is not None and \
                    math.dist(Pd, self.ring.world(D(self.det_prev))) < self.speed * max(dt, 0.1) + 80
                if near_last or math.dist(P, Pd) < TRUST:
                    if math.dist(P, Pd) >= TRUST:
                        self.reanchored = getattr(self, 'reanchored', 0) + 1
                    P = Pd
            self.pos_s = self.ring.screen(P)
        self.det_still = det is not None and self.det_prev is not None and math.dist(det, self.det_prev) < 3
        self.det_prev = det
        self.pos = self.ring.world(self.pos_s)

    def _calibrate(self, det, now):
        """Waiting in our gate: one click a little toward the ring. The legend walks there inside the closed gate and
        stops exactly on the clicked point, so the bar's offset to the ground (38-88 px, by our level) is measured
        before the chase needs it. -> the click, or None."""
        if self.calibrated or self.released or self.det0 is None:
            return None
        if self.calib is None:
            g = (self.det0[0] + self.off[0], self.det0[1] + self.off[1])
            self.calib = ((g[0], g[1] - 35.0), now)
            self.last_click, self.dest = (self.calib[0], now), self.calib[0]
            return self.calib[0]
        pt, t0 = self.calib
        if det is not None and now - t0 > 0.6 and self.det_still and math.dist(det, self.det0) > 8:
            self.off, self.calibrated = (pt[0] - det[0], pt[1] - det[1]), True
            self.pos_s = pt
        elif now - t0 > 4.0:
            self.calibrated = True                            # it never moved: the guess stays
        return None

    # -- collision checks (world space) -----------------------------------------------
    def _others(self):
        """Units to keep clear of: every free unit but the target. A unit with the target's item is no obstacle:
        touching it gets the same item."""
        # each at its own distance from the centre: units stand up to ~15% off the fitted ring (replay of run 105105 5-4:
        # the Dragon's Claw at 1.13 R was touched by a crossing that cleared it 'on the ring')
        return [(a, rr) for a, it, rr in self.solid
                if self.target_angle is None or (abs(wrap(a - self.target_angle)) > 0.05
                                                 and not (self.item and it == self.item))]

    def _clear(self, P, Q, others, t0=None, onto=False):
        """Walking P -> Q, starting t0 s from now, touches none of the units at angles `others`? onto: Q is the target
        - the walk ends where it touches it (PICKUP short of it), with only a few px kept over the last stretch: the
        first unit touched is the one we get (run 130538 5-4: a unit 76 px behind the Deathblade made every cut 'unclear')."""
        if not others:
            return True
        if onto:
            L0 = math.dist(P, Q)
            if L0 <= PICKUP * 0.7:
                return True
            Q = (Q[0] + (P[0] - Q[0]) * PICKUP * 0.7 / L0, Q[1] + (P[1] - Q[1]) * PICKUP * 0.7 / L0)
        L = math.dist(P, Q)
        T = L / self.speed
        t0 = self.delay if t0 is None else t0
        s = np.linspace(0.0, 1.0, max(4, int(L / 8)) + 1)[:, None]
        x, y = P[0] + (Q[0] - P[0]) * s, P[1] + (Q[1] - P[1]) * s
        a = np.asarray([o[0] for o in others], float)[None, :] + self.w * (t0 + s * T)
        R = self.ring.R * np.asarray([o[1] for o in others], float)[None, :]
        lim = np.where(s > 0.85, PICKUP + 15, PICKUP + CLEAR) if onto else PICKUP + CLEAR
        return bool((((x - R * np.cos(a)) ** 2 + (y - R * np.sin(a)) ** 2) >= lim ** 2).all())

    @staticmethod
    def _around(P, Q, r0):
        """Shortest route P -> Q staying outside the circle of radius r0 (both outside it)."""
        dx, dy = Q[0] - P[0], Q[1] - P[1]
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, -(P[0] * dx + P[1] * dy) / L2))
        if math.hypot(P[0] + dx * t, P[1] + dy * t) >= r0 - 1e-6:
            return [P, Q], math.dist(P, Q)
        dp, dq = max(math.hypot(*P), r0), max(math.hypot(*Q), r0)
        ap, aq = math.atan2(P[1], P[0]), math.atan2(Q[1], Q[0])
        tp, tq = math.acos(min(1.0, r0 / dp)), math.acos(min(1.0, r0 / dq))
        best = None
        for sg in (1, -1):
            a1, a2 = ap + sg * tp, aq - sg * tq
            sweep = ((a2 - a1) * sg) % (2 * math.pi)
            L = math.sqrt(max(0.0, dp * dp - r0 * r0)) + math.sqrt(max(0.0, dq * dq - r0 * r0)) + r0 * sweep
            if best is None or L < best[1]:
                n = max(1, math.ceil(sweep / 0.3))
                rr = r0 / math.cos(sweep / n / 2)            # chords stay outside the circle
                pts = [P] + [(rr * math.cos(a1 + sg * sweep * k / n), rr * math.sin(a1 + sg * sweep * k / n))
                             for k in range(n + 1)] + [Q]
                best = (pts, L)
        return best

    def _plan(self, P, a0, t_max=15.0, rough=False):
        """The earliest head-on meeting (_plan_front); a straight intercept from here instead when it meets the target over
        1 s sooner and touches nothing else on the way (run 130538 5-4: the Deathblade walked away from our gate up the
        left side - every head-on meeting was 7 s off, the straight one ~3 s)."""
        front = self._plan_front(P, a0, t_max, rough)
        direct = self._plan_direct(P, a0, (front[0] - 1.0) if front else t_max, rough)
        return direct or front

    def _plan_direct(self, P, a0, t_max, rough=False):
        """Straight from P onto where the target will be when we get there -> (t, P, [P], T) or None."""
        v, lead = self.speed, max(self.delay, self.lag)
        others = self._others()
        step = 0.25 if rough else 0.1
        for k in range(1, int(max(0.0, t_max) / step) + 1):
            t = k * step
            T = self._tat(a0 + self.w * t)
            if lead + (math.dist(P, T) - PICKUP * 0.5) / v > t:
                continue
            if rough or self._clear(P, T, others, t0=lead, onto=True):
                return t, P, [P], T
            return None                                       # the first meeting in reach crosses another unit
        return None

    def _plan_front(self, P, a0, t_max=15.0, rough=False):
        """The earliest head-on meeting with the unit now at angle a0 -> (t from now, M where the cut starts,
        [world points P .. M], T the meeting point) or None. The cut M -> T runs back against the ring's turn
        from AHEAD of the meeting place and touches nothing else on the way."""
        R, v, w = self.ring.R, self.speed, self.w
        sg = 1.0 if w >= 0 else -1.0
        reach = R + PICKUP
        lane = reach + LANE_OUT
        d = max(1.0, math.hypot(*P))
        inside = d < R - PICKUP - 10
        # where the walk starts: from here; and when we stand among the units, also straight out to the lane first
        # (going on straight in is a start too - stepping out and back in flip-flopped for 10 s in the sim)
        starts = [[P]]
        if not inside and d < reach + LANE_CLEAR:
            starts.append([P, (P[0] * lane / d, P[1] * lane / d)])
        others = self._others()
        r_in = max(10.0, R - PICKUP - LANE_OUT)
        step = 0.25 if rough else 0.1
        for k in range(1, int(t_max / step) + 1):
            t = k * step
            phi = a0 + w * t
            T = self._tat(phi)
            for da in AHEAD[:1] if rough else AHEAD:
                for start in starts:
                    Q0 = start[-1]
                    lead = max(self.delay, self.lag) + math.dist(P, Q0) / v
                    q_in = math.hypot(*Q0) < R
                    # the cut starts on the outside lane, or in the open middle - reached straight across the ring
                    # through a gap that stays open while we walk it (the lane round the outside is ~5 s a half lap)
                    for M in ((self.ring.at(phi + sg * da, r_in),) if q_in else
                              (self.ring.at(phi + sg * da, lane), self.ring.at(phi + sg * da, r_in))):
                        m_in = math.hypot(*M) < R
                        if m_in:
                            pts, L = [Q0, M], math.dist(Q0, M)
                        elif math.hypot(*Q0) < reach + LANE_CLEAR - 1:
                            continue                          # from among the units the lane is reached by stepping out
                        else:
                            pts, L = self._around(Q0, M, reach + LANE_CLEAR)
                        cut = math.dist(M, T)
                        if lead + (L + cut) / v > t:
                            continue
                        if rough:
                            return t, M, start[:-1] + pts, T  # choosing: how soon, near enough
                        if m_in and not q_in and not self._clear(Q0, M, others, t0=lead):
                            continue                          # no gap open on the way in (touching the target is fine)
                        if not self._clear(M, T, others, t0=t - cut / v, onto=True):
                            continue
                        return t, M, start[:-1] + pts, T
        return None

    def _tat(self, a):
        """World point of the target at angle a: at its own distance from the centre, as last seen."""
        return self.ring.at(a, self.ring.R * getattr(self, 'target_r', 1.0))

    def _intercept(self, P, a_target, t_max=4.0):
        """The straight walk from P that meets the target earliest -> world point (on its circle) or None."""
        v, t0 = self.speed, max(self.delay, self.lag)
        for k in range(1, int((t_max + t0) / 0.05) + 1):
            tau = k * 0.05
            U = self._tat(a_target + self.w * tau)
            if math.dist(P, U) - PICKUP * 0.5 <= v * (tau - t0):
                return U
        return None

    # -- choice -----------------------------------------------------------------------
    def set_priority(self, ranked, source):
        """The model's ranking of what is on offer, best first. A chase not yet committed (cutting in, touching)
        re-chooses on the new list."""
        self.priority, self.source = list(ranked), source
        if not self.stepped and self.contact is None and not self.got:
            self.target_angle, self.item = None, None

    def _want(self, item, bench_items, roster, plan):
        if item in self.priority:
            return 100.0 - self.priority.index(item)
        return item_want(item, bench_items, roster, None if self.priority else plan)

    def _choose(self, P, bench_items, roster, plan):
        """The most wanted unit on the ring; equal wants: the sooner met. Nothing is too late - the ranking decides."""
        best = None
        left = None if self.ends_at is None else self.ends_at - self._now - 0.3
        for a, it, tid in self.targets_ids:
            r = self._plan(P, a, rough=True)
            t = r[0] if r else 15.0
            # met only after the carousel ends: not on (when nothing can be met in time, the soonest is chosen)
            late = left is not None and t > left
            score = self._want(it, bench_items, roster, plan) - 0.05 * t - (1000.0 + t if late else 0.0)
            if best is None or score > best[0]:
                best = (score, a, it, tid)
        if best is None:
            return False
        _, self.target_angle, self.item, self.target_id = best
        tr = self.track_of.get(self.target_id)
        self.target_r = tr[4] if tr is not None else 1.0
        self.stepped, self.lost_since, self.off_seen = False, None, 0
        return True

    def _rechoose(self, P, bench_items, roster, plan):
        """Not committed yet: choose again when a unit wanted more than the target has become choosable."""
        cur = self._want(self.item, bench_items, roster, plan)
        if any(it != self.item and self._want(it, bench_items, roster, plan) >= cur + 0.5 for _, it in self.targets):
            angle, item = self.target_angle, self.item
            if not self._choose(P, bench_items, roster, plan):
                self.target_angle, self.item = angle, item

    # -- the target on its circle -----------------------------------------------------
    def _follow_target(self, dt, now):
        """The target is its track: the same unit followed tick to tick on the ring (_update_tracks), whatever its icon
        reads. Seen this tick: its place. Unseen, it turns on - a bar or a player in front hides icons for seconds
        (run 20261010-121458 3-4 dropped the Recurve Bow after 4 s unseen behind two players; it turned on, untouched).
        Taken: a player started carrying its item at its place, or we stand at its place and it is not there."""
        pred = self.target_angle + self.w * dt
        tr = getattr(self, 'track_of', {}).get(self.target_id)
        if tr is not None and tr[2] == now:
            # its trajectory is the evidence (the user, 2026-10-11): turning with the ring on its circle = free, whoever
            # stands next to it; off the ring's turn or off its circle, two looks running = a player carries it
            # (a free unit reads up to ~0.2 rad/s off and 15-20% off the fitted radius near the top of the ring, where
            # the fit is a little off - run 100905 2-4's B.F. Sword - but its own radius changes slowly; a carried one
            # walks with its player at ~190 px/s, ~0.9 radius/R per s)
            if tr[5] >= 3 and (tr[3] > 0.35 or tr[8] > 0.3):
                self.off_traj = getattr(self, 'off_traj', 0) + 1
                if self.off_traj >= 2:
                    self.off_traj = 0
                    self._taken('it left its trajectory: carried', now)
                    return
            else:
                self.off_traj = 0
            self.target_angle, self.lost_since, self.off_seen, self.open_miss = tr[0], None, 0, 0
            self.target_r = 0.7 * getattr(self, 'target_r', tr[4]) + 0.3 * tr[4]
            return
        if tr is not None:
            pred = tr[0]                                      # its track turned it on
        else:
            # its track ended (unseen > TAKEN_AFTER): a unit back in sight right where it would be, with its item or
            # an unread icon, is it again
            # (unseen, its place drifts: the turn read on screen is not even round the ring - run 121458 3-4's Recurve
            # Bow came back 22 degrees behind its prediction after 5.6 s behind two players)
            hid = now - (self.lost_since or now)
            back = [t for t in getattr(self, 'tracks', []) if t[2] == now and t[1] == self.item
                    and abs(wrap(t[0] - pred)) < min(0.45, 0.2 + 0.06 * hid) and t[5] <= 3]
            if back:
                t = min(back, key=lambda t: abs(wrap(t[0] - pred)))
                self.target_id, self.target_angle, self.lost_since, self.off_seen = t[6], t[0], None, 0
                return
        self.target_angle = pred
        self.lost_since = self.lost_since or now
        U = self.ring.at(pred)
        off = [c for c in self._fresh_carried(self.item, now) if math.dist(c['q'], U) < 260
               and (self.pos is None or math.dist(c['q'], self.pos) > 110)]
        self.off_seen = self.off_seen + 1 if off else 0
        # missing where nothing can hide it: no player's bar or body over the place its icon should be (a player in
        # front hides it for seconds - run 121458 3-4's Recurve Bow; one that walked off with it does not: run 100905
        # 3-4's Negatron Cloak vanished at the top while we walked for it and the Gloves passed our gate). A look that
        # read no icon at all is a blank frame, no evidence either way (run 213127 marked the B.F. Sword 'taken' 4 times)
        if self.ring.all:
            self.open_miss = 0 if self._covered(U) else self.open_miss + 1
        if self.off_seen >= 1:
            self._taken('a player started carrying its item at its place', now)
        elif self.open_miss >= 2 and now - self.lost_since >= 0.6:
            self._taken('not at its place, with nothing in front of it', now)
        elif now - self.lost_since >= LOST_TAKEN:
            self._taken(f'not seen on its circle for {LOST_TAKEN:.0f} s', now)

    def _covered(self, U):
        """Could something stand in front of the icon of a unit at world point U? Players stand where the units they carry
        are (every off-ring icon, and every icon that does not turn with the ring), we stand where we are, and any
        player's bar seen this look. Nothing near: its icon would show."""
        mx, my = self.ring.screen(U)
        my -= UNIT_DY
        pts = [self.ring.screen(c['q']) for c in self.carried]
        pts += [self.ring.screen(self.ring.at(tr[0], tr[4] * self.ring.R)) for tr in self.tracks
                if tr[5] >= 2 and tr[3] > 0.3]
        pts = [(x, y - UNIT_DY) for x, y in pts]
        if self.pos is not None:
            x, y = self.ring.screen(self.pos)
            pts.append((x, y - 70))                            # our legend and bar over our feet
        pts += [(bx + 45, by + 40) for bx, by, *_ in (self.bars or [])]
        return any(abs(mx - x) < 120 and -110 < my - y < 120 for x, y in pts)

    def _update_carried(self, dt, now):
        """Units off the ring, each followed by proximity tick to tick (a player walks <= ~270 px/s). One seen for the
        first time just now was picked up just now; the ones already carried are never evidence about our target,
        wherever they stand (the user, 2026-10-11: players who already have a unit are excluded)."""
        reach = 60 + 300 * max(dt, 0.1)
        for q, it in self.ring.off:
            c = min((c for c in self.carried if c['item'] == it and c['seen'] < now), default=None,
                    key=lambda c: math.dist(c['q'], q))
            if c is not None and math.dist(c['q'], q) < reach:
                c['q'], c['seen'] = q, now
            else:
                self.carried.append({'q': q, 'item': it, 'first': now, 'seen': now})
        self.carried = [c for c in self.carried if now - c['seen'] < 3.0]

    def _fresh_carried(self, item, now):
        """Units of `item` a player started carrying in the last second (never at our first look)."""
        start = self.looked[0][0] if self.looked else now
        return [c for c in self.carried if c['item'] == item and now - c['first'] < 1.0 and c['first'] > start]

    def _taken(self, why, now):
        self.taken.append(self.item)
        self.taken_why.append((self.item, why, round(now - (self.lost_since or now), 2), self.off_seen))
        self.target_angle, self.item, self.lost_since, self.off_seen = None, None, None, 0
        self.stepped, self.meet, self.target_id, self.open_miss = False, None, None, 0

    def _watch_pickup(self, now):
        """We reached the target: did we pick it up? A unit we hold follows us - its marker leaves the ring and
        stays beside us. -> True once it is ours."""
        a0, t0, item = self.contact
        U = self._tat(a0 + self.w * (now - t0))              # where it would be had nobody taken it
        others = [self.ring.at(a, rr * self.ring.R) for a, rr in self._others()]
        mine = [(q[0] - self.pos[0], q[1] - self.pos[1]) for q, it in self.ring.all
                if it == item and math.dist(q, self.pos) < 90 and math.dist(q, U) > 35
                and all(math.dist(q, o) > 30 for o in others)]
        # another item following us is the unit we hold: we touched it first (run 20261012-194821 3-4: the Recurve Bow's
        # icon went out of sight behind players and was called ours, while the Chain Vest walked with us)
        alien = [((q[0] - self.pos[0], q[1] - self.pos[1]), it) for q, it in self.ring.all
                 if it not in (item, None) and math.dist(q, self.pos) < 90 and math.dist(q, U) > 35
                 and all(math.dist(q, o) > 30 for o in others)]
        if alien and not mine:
            # the same item at the same place beside us while we walk on: it is the unit we hold
            seen = getattr(self, 'alien_seen', None)
            rel, it = alien[0]
            if seen and seen[0] == it and math.dist(rel, seen[1]) < 35:
                if math.dist(self.pos, seen[2]) > FOLLOW_WALK:
                    self.got, self.path, self.contact, self.got_seen = it, [], None, now
                    self.taken_why.append((item, f'we touched {it} first: it follows us', 0.0, 0))
                    self.alien_seen = None
                    return True
            else:
                self.alien_seen = (it, rel, self.pos)
        else:
            self.alien_seen = None
        if mine and (self.got_near == 0 or min(math.dist(r, self.got_rel) for r in mine) < 35):
            r = mine[0] if self.got_near == 0 else min(mine, key=lambda r: math.dist(r, self.got_rel))
            if self.got_near == 0:
                self.got_rel, self.got_from = r, self.pos
            self.got_near += 1
        elif any(it == item and math.dist(q, self.pos) < 250 for q, it in self.ring.all):
            self.got_near = 0
        gap = min(0.3, math.pi / max(len(self.units), 1))
        a1 = a0 + self.w * (now - t0)
        if any(it in (item, None) and abs(wrap(a - a1)) < gap for a, it in self.seen):
            self.contact_gone = None
        elif any(math.dist(c['q'], self.pos) > 200 for c in self._fresh_carried(item, now)):
            self.contact, self.stepped = None, False         # walking off with someone else
            self.item = item
            self._taken('another player picked it up as we got there', now)
            return False
        else:
            self.contact_gone = self.contact_gone or now
        # picked up only on evidence - its icon following us - never because its icon is gone from the ring: a hidden or
        # taken target is gone too (run 20261012-194821 3-4: the Recurve Bow, taken by navE behind the players, was
        # called ours; we stopped in the units' band and the next unit walked into us)
        # ... and it keeps its place beside us while we walk on: a player carrying the same item can stand beside us for
        # a few looks (3-4 robustness run 2: navE with the Recurve Bow was taken for ours, standing)
        if self.got_near >= 2 and math.dist(self.pos, self.got_from) > FOLLOW_WALK:
            self.got, self.path, self.contact, self.got_seen = item, [], None, now
            if item in self.taken:
                self.taken.remove(item)
            return True
        if now - t0 > 0.6:
            # touching is picking up: nothing for 0.6 s means it was not there. Off the ring at once - standing on it the
            # next unit walks into us (replay of run 100905 5-4: the Executioner Emblem was taken half a second before
            # we got there, we followed its predicted place for 3 s and the Quicksilver walked into us)
            self.contact, self.stepped = None, False
            tr = getattr(self, 'track_of', {}).get(self.target_id)
            # taken only when seen: in sight right here and still not ours. Hidden, its place was a guess - step out and
            # keep it (it is given up when hidden too long, or seen carried): robustness run 14 of 2-4 wrote off a Rod
            # hidden behind the unit in front of it, then a B.F. Sword, and picked up nothing
            if tr is not None and tr[2] == now and math.dist(self.ring.at(tr[0], tr[4] * self.ring.R), self.pos) <= PICKUP:
                self.item = item
                self._taken('not there when we touched its place', now)
            self.step_out = True
        return False

    # -- phases -----------------------------------------------------------------------
    def phase(self, now):
        """'wait' until LOOK s before our release, 'look' while the read before it runs, 'go' after."""
        if self.committed:
            return 'go'
        if self.release_at is not None and not self.released and now < self.release_at - LOOK:
            return 'wait'
        return 'look'

    def _commit(self, now):
        """End of the read before our release: every unit seen on the ring in it made choosable."""
        if self.omega is None:
            self.omega = RIM_SPEED / self.ring.R
        last = self.looked[-1][0]
        for tr in self.tracks:
            if tr[2] == last and abs(tr[4] - 1) < 0.25:
                tr[3], tr[5] = 0.0, max(tr[5], 3)
        self.targets_ids = [(tr[0], tr[1], tr[6]) for tr in self.tracks
                            if tr[1] is not None and tr[5] >= 3 and tr[3] < 0.3 and abs(tr[4] - 1) < 0.25]
        self.targets = [(a, it) for a, it, _ in self.targets_ids]
        self.committed = True

    # -- one tick ---------------------------------------------------------------------
    def next_click(self, markers, bench_items, roster, plan, pos=None, now=None, held=None, bars=None, lag=0.0):
        """markers: [(x, y, item)] and pos (OwnTracker's feet estimate, 1080p, or None) as seen in the frame captured at
        `now`; lag: s from that capture until our click lands (the HUD read and perception before it: 0.3-1.0 s on
        2026-10-10, while the ring turns ~80 px/s - every click went where the target had been). -> point to right-click."""
        import time as _t
        now = _t.time() if now is None else now
        self.lag = max(0.0, lag)
        self.bars = bars
        if self.prev_now is not None and 0 < now - self.prev_now < 3:
            self.tick_dt = 0.7 * self.tick_dt + 0.3 * (now - self.prev_now)
        self._now = now
        dt = 0.0 if self.prev_now is None else now - self.prev_now
        click = self._tick(markers, bench_items, roster, plan, pos, now, dt)
        self.prev_now = now
        return click

    def _tick(self, markers, bench_items, roster, plan, pos, now, dt):
        ph = self.phase(now)
        if ph == 'wait':
            self.path = []
            self._locate(pos, now)
            return self._calibrate(pos, now)
        if ph == 'look' and not self.looked:
            self.tracks, self.prev_now, dt = [], None, 0.0
        self.markers_now = [(m, False) for m in markers]
        self._update_tracks(self.ring.fit(markers), now)
        self._update_carried(dt, now)
        self._locate(pos, now)
        if ph == 'look':
            calib = self._calibrate(pos, now)
            self.looked.append((now, list(self.seen)))
            span = now - self.looked[0][0]
            ready = sum(len(o[1]) >= 3 for o in self.looked) >= 3 and span >= 1.0
            if not (ready or (self.release_at is not None and now >= self.release_at - COMMIT)):
                return calib
            self._commit(now)
        P = self.pos
        if self.got:
            # keep checking: a neighbour who took it and walked off our way looks the same at first
            mine = [math.dist(q, P) for q, it in self.ring.off if it == self.got]
            if any(d < 120 for d in mine):
                self.got_seen = now
            elif any(d > 200 for d in mine) and now - self.got_seen > 0.8:
                self.taken.append(self.got)
                self.got, self.item, self.target_angle, self.stepped = None, None, None, False
                self.got_near = 0
            if self.got:
                return None
        ttr = self.track_of.get(self.target_id) if self.target_id is not None else None
        seen_ago = now - ttr[2] if ttr is not None else 99.0
        if self.target_angle is not None and self.contact is None:
            self._follow_target(dt, now)
            # a touch only on a target seen just now (or covered a moment by our own bar as we reach it): standing on a
            # hidden target's guessed place, the unit beside it walks into us (2-4 robustness run 14: 2.8 s on a hidden
            # Rod's place, then the B.F. Sword)
            if self.target_angle is not None and math.dist(P, self._tat(self.target_angle)) < PICKUP \
                    and seen_ago <= NEAR_HIDE:
                self.contact, self.got_near, self.contact_gone = (self.target_angle, now, self.item), 0, None
        if self.contact is not None:
            if self._watch_pickup(now):
                return None
            if self.contact is not None:
                # touching it: keep walking onto where it goes (our clicked point may be just short of it) - clear of
                # every other unit
                U = self._tat(self.contact[0] + self.w * (now - self.contact[1] + self.lag + self.tick_dt))
                return self._safe_click(P, U, now)
        if getattr(self, 'step_out', False):
            self.step_out = False
            d = max(1.0, math.hypot(*P))
            if abs(d - self.ring.R) < PICKUP + CLEAR:
                lane = self.ring.R + PICKUP + LANE_OUT
                return self._safe_click(P, (P[0] * lane / d, P[1] * lane / d) if d >= self.ring.R * 0.8 else
                                        (P[0] * max(10.0, self.ring.R - PICKUP - LANE_OUT) / d,
                                         P[1] * max(10.0, self.ring.R - PICKUP - LANE_OUT) / d), now)
        if not self.units:
            return None
        # plans start where we will be when this click lands: we walk on toward the last clicked point meanwhile
        # (lag 0.33 s is ~60 px of walking)
        if self.released and self.dest is not None:
            Dw = self.ring.world(self.dest)
            dd = math.dist(P, Dw)
            stp = min(dd, self.speed * self.lag)
            if dd > 1e-6:
                P = (P[0] + (Dw[0] - P[0]) * stp / dd, P[1] + (Dw[1] - P[1]) * stp / dd)
        if self.target_angle is not None and not self.stepped:
            self._rechoose(P, bench_items, roster, plan)
        if self.target_angle is None and not self._choose(P, bench_items, roster, plan):
            self.path = []
            return self._hold(P, now)
        # unseen for a second with nothing in front of it: do not walk onto the ring for it (covered - a player, or we
        # ourselves as we close in - it walks on: run 130538 5-4's Deathblade, hidden 1.5 s by a legend on it)
        unseen = self.lost_since is not None and now - self.lost_since >= 1.0 and self.open_miss >= 1
        # onto the ring only for a target seen just now: a hidden one may be gone, and its place is then where the next
        # unit walks (run 20261012-194821 3-4: navE took the hidden Recurve Bow; we got the Chain Vest that came next)
        ttr = self.track_of.get(self.target_id)
        seen_ago = now - ttr[2] if ttr is not None else 99.0
        near = self.target_angle is not None and math.dist(P, self._tat(self.target_angle)) < NEAR_FRESH \
            and seen_ago <= NEAR_HIDE
        if seen_ago > HIDDEN_GIVEUP:
            self._taken(f'hidden {seen_ago:.1f} s: taken unseen, or out of reach', now)
            self.path = []
            return self._hold(P, now)
        blind = seen_ago > SEEN_FRESH and not near
        if self.stepped and (unseen or blind):
            self.stepped = False                              # never onto the ring for a unit not seen lately
            self.path = []
            return self._hold(P, now)                         # out of the units' band, the clear way, and wait
        if self.stepped:
            # cutting in: straight onto where it meets us - while that walk touches nothing else first (the plan was
            # made on an older look; a neighbour can come between: replays of 2026-10-10, 2 of 7 wrong pickups)
            U = self._intercept(P, self.target_angle) or self._tat(self.target_angle + self.w * (self.lag + self.tick_dt))
            tau = math.dist(P, U) / self.speed
            if math.dist(P, U) > PICKUP and not self._clear(P, U, self._others(), t0=max(self.delay, self.lag), onto=True):
                self.stepped = False
        if self.stepped:
            self.contact_in = tau
            self.path = [self.ring.screen(P), self.ring.screen(U)]
            return self._click(U, now)
        r = self._plan(P, self.target_angle)
        if r is None:
            self.path = []
            return self._hold(P, now)
        t, M, route, T = r
        self.meet, self.meet_at, self.contact_in = M, now + t, t
        self.path = [self.ring.screen(q) for q in route] + [self.ring.screen(T)]
        # the cut starts once we are at its start, or would get there before the next look
        left = sum(math.dist(a, b) for a, b in zip(route, route[1:]))
        if left <= self.speed * max(0.15, self.tick_dt * 0.8) and now >= (self.release_at or 0) - 0.2 and not unseen \
                and not blind:
            self.stepped = True
            U = self._intercept(M, self.target_angle, t_max=4.0 + left / self.speed)
            if U is not None:
                self.path = [self.ring.screen(P), self.ring.screen(M), self.ring.screen(U)]
                if left > 5:
                    return self._click(M, now)
                return self._click(U, now)
        if self.delay > 0.2:
            return None                                       # clicks in a closed gate walk us about inside it
        # the next corner of the route: the farthest one walked straight without entering the units' reach
        goal = route[1] if len(route) > 1 else M
        for q in route[2:]:
            if self._clear_lane(P, q):
                goal = q
            else:
                break
        return self._safe_click(P, goal, now)

    def _clear_lane(self, P, Q):
        """The straight walk P -> Q stays outside the units' reach."""
        reach = self.ring.R + PICKUP + LANE_CLEAR - 5
        dx, dy = Q[0] - P[0], Q[1] - P[1]
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, -(P[0] * dx + P[1] * dy) / L2))
        return math.hypot(P[0] + dx * t, P[1] + dy * t) >= reach

    def _hold(self, P, now):
        """No target: stop where we are, or step out of the units' path when near it."""
        if self.last_click[0] is None or not self.released:
            return None
        r = math.hypot(*P)
        band = PICKUP + 20
        if abs(r - self.ring.R) < band:
            s = (self.ring.R + band + 15) / max(r, 1.0)
            Q = (P[0] * s, P[1] * s)
        else:
            Q = P
        return self._safe_click(P, Q, now)

    def _safe_click(self, P, Q, now):
        """Click Q unless the walk there touches another unit on the way (where each will be, as the ring turns); then
        wait where no unit reaches us, or step out of the units' band to the side that is clear. Every leg is checked,
        not only the final cut (the robustness runs of run 20261012-194821's carousels: 3 of 24 runs touched a unit
        walking a route leg - out to the lane, or back in - across the band the units walk in)."""
        others = self._others()
        t0 = max(self.delay, self.lag)
        if math.dist(P, Q) < 1.0 or self._clear(P, Q, others, t0=t0):
            return self._click(Q, now)
        r, R = max(1.0, math.hypot(*P)), self.ring.R
        if abs(r - R) >= PICKUP + CLEAR:
            return self._click(P, now)                        # outside every unit's reach: wait here for the gap
        for to in (R + PICKUP + LANE_OUT, max(10.0, R - PICKUP - LANE_OUT)):
            S = (P[0] * to / r, P[1] * to / r)
            if self._clear(P, S, others, t0=t0):
                return self._click(S, now)
        return self._click(Q, now)

    def _click(self, Q, now, force=False):
        """Right-click world point Q (re-sent each tick until we are out of the gate: clicks in a closed gate are
        lost). -> 1080p screen point or None."""
        pt = self.ring.screen(Q)
        last, t = self.last_click
        if last is not None and math.dist(last, pt) < 6 and (self.released and now - t < 1.5):
            self.dest = pt
            return None                                       # already walking there
        self.last_click = (pt, now)
        self.dest = pt
        return pt

    def view(self):
        """What the overlay draws, 1080p screen points."""
        rg = self.ring
        sc = lambda a: rg.screen(rg.at(a))
        alive = {it for _, it in self.units}
        rank = {it: i + 1 for i, it in enumerate(self.priority)}
        ranking = [(it, 'got' if it == self.got else 'target' if it == self.item and self.target_angle is not None
                    else 'free' if it in alive else 'gone') for it in self.priority
                   if it in alive or it == self.got or it == self.item or it in self.taken]
        target = marker = None
        if self.target_angle is not None:
            pred = rg.at(self.target_angle)
            same = [q for q, it in rg.all if it == self.item]
            if same:
                q = min(same, key=lambda q: math.dist(q, pred))
                if math.dist(q, pred) < 90:
                    target = rg.screen(q)
                    marker = (target[0], target[1] - UNIT_DY)
            if target is None:
                target = sc(self.target_angle)
        lane = rg.R + PICKUP + LANE_OUT
        anim = None if self.target_angle is None or self.got else {
            'cx': rg.cx, 'cy': rg.cy, 'R': rg.R, 'K': K, 'a': self.target_angle, 'w': self.w, 't': self._now,
            'label': self.item, 'rank': rank.get(self.item)}
        if anim:
            # the next two free units of the priority list too, thinner (the user, 2026-10-13: show 3, not only the
            # target): what we go for if the target is taken
            others = []
            for it in self.priority:
                angs = [a for a, u in self.units if u == it]
                if it not in (self.item, self.got) and angs and len(others) < 2:
                    others.append({'a': angs[0], 'label': it, 'rank': rank[it]})
            anim['others'] = others
        return {'ring': (rg.cx, rg.cy, rg.R, K), 'pickup': PICKUP, 'lane': lane, 'anim': anim,
                'path': list(self.path), 'ranking': ranking, 'source': self.source,
                'rank': rank.get(self.item),
                'target': target, 'target_marker': marker,
                'label': self.item,
                'pos': rg.screen(self.pos) if self.pos is not None else None,
                'stepped': self.stepped, 'got': self.got}
