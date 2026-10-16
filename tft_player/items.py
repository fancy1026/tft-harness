"""Item bench recognition and item decisions.

Recognition: the item bench is a column of 10 slots on the left; icons are
the official item art, so each slot is matched against the 50 item icons
(normalised correlation on the centre, clear of the count badge).

Decisions: slam completed items early onto the right holder - the carry gets
damage items of its type, the front-line tank gets defensive items, a
Tactician's crown/cape/shield goes on anyone. Items combine by dropping the
second component onto a unit already holding the first.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

from tft_player.catalog import CATALOG

DATA = Path(__file__).resolve().parent.parent / 'data'
SLOT_X = 32
SLOT_Y0, SLOT_STEP = 285, 53
SLOT_HALF = 19                     # slot icon ~38px at 1080p
COLUMN_STEP = 55                   # an 11th item opens a second column of 10, this far right (the user, 2026-10-08)
N_SLOTS = 20


def slot_point(i):
    """1080p reference point; 'L' = anchored to the left edge (the item bench
    hugs the left side at any aspect ratio). Slots 10-19 are the second column."""
    return SLOT_X + COLUMN_STEP * (i // 10), SLOT_Y0 + SLOT_STEP * (i % 10), 'L'


def bench_columns(frame):
    """1, or 2 when the item bench has a second column. The trait panel then moves right: its teal left
    border stands at x 132 instead of 82 (1080p; a relative colour test, so it holds under a choice screen's
    dimming). With no trait panel to see, the second column's empty slots (pure black squares) tell. A second
    column always ends in empty slots (20 items are not handled): its last slot must be black - without that,
    loading screens, carousels and covered frames read 7-10 phantom items there. Without this function, items
    11+ were never read (runs 152203 4-1..5-3: up to 4 items unseen, 221257 4-2..5-1)."""
    s = frame.shape[0] / 1080
    y0, y1 = round(300 * s), round(760 * s)
    half = round(12 * s)

    def black(k):
        x, y, _ = slot_point(k)
        c = frame[round(y * s) - half:round(y * s) + half, round(x * s) - half:round(x * s) + half]
        return c.size > 0 and c.max() <= 30

    def teal(x):
        c = frame[y0:y1, round((x - 2) * s):round((x + 3) * s)].astype(np.int16)
        r, g, b = c[..., 0], c[..., 1], c[..., 2]
        return float(((g > r * 1.3 + 5) & (b > r * 1.3 + 5)).mean())
    if not black(19):
        return 1
    t1, t2 = teal(82), teal(132)
    if t1 or t2:
        return 2 if t2 > t1 else 1
    return 2 if sum(black(k) for k in range(11, 20)) >= 3 else 1


def _load_items():
    cat = json.loads((DATA / 'item-assets' / 'catalogue.json').read_text(encoding='utf-8'))['items']
    recipes, icons, components = {}, {}, []
    for it in cat:
        name = it['name'].replace('Tear Of The Goddess', 'Tear of the Goddess')
        comps = [c.replace('Tear Of The Goddess', 'Tear of the Goddess') for c in it.get('components') or []]
        img = cv2.imread(str(DATA / 'item-assets' / Path(it['path']).name))
        if img is None:
            continue
        icons[name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if comps:
            recipes[name] = tuple(sorted(comps))
        else:
            components.append(name)
    return recipes, icons, components


RECIPES, ICONS, COMPONENTS = _load_items()
BY_PAIR = {v: k for k, v in RECIPES.items()}


def held_after_give(items, item, done=None):
    """What a unit holds after we drop `item` on it, from what it held (`items`; unread icons left out): the item
    added, or - when it completed `done` with a component it held - that component replaced by `done`. At most 3."""
    held = [i for i in items or [] if i and i != 'unknown']
    if done:
        used = next((c for c in held if c in COMPONENTS and BY_PAIR.get(tuple(sorted((c, item)))) == done), None)
        if used:
            held.remove(used)
    return (held + [done or item])[:3]

# value of a completed item on a holder type (0 = don't build it for them)
VALUE = {
    'AD': {"Infinity Edge": 10, "Giant Slayer": 9, "Last Whisper": 9, "Deathblade": 8, "Bloodthirster": 8,
           "Guinsoo's Rageblade": 8, "Kraken's Fury": 8, "Red Buff": 7, "Hand Of Justice": 7,
           "Titan's Resolve": 6, "Sterak's Gage": 6, "Edge of Night": 6, "Spear of Shojin": 6,
           "Quicksilver": 5, "Void Staff": 3},
    'AP': {"Jeweled Gauntlet": 10, "Archangel's Staff": 9, "Rabadon's Deathcap": 9, "Blue Buff": 9,
           "Spear of Shojin": 8, "Nashor's Tooth": 8, "Guinsoo's Rageblade": 7, "Morellonomicon": 7,
           "Hextech Gunblade": 7, "Void Staff": 7, "Hand Of Justice": 7, "Ionic Spark": 5,
           "Quicksilver": 5, "Adaptive Helm": 5},
    'tank': {"Gargoyle Stoneplate": 10, "Warmog's Armor": 9, "Bramble Vest": 9, "Dragon's Claw": 9,
             "Sunfire Cape": 8, "Spirit Visage": 8, "Steadfast Heart": 8, "Crownguard": 7,
             "Protector's Vow": 7, "Evenshroud": 7, "Titan's Resolve": 7, "Ionic Spark": 7,
             "Adaptive Helm": 6, "Edge of Night": 4},
}
ANYONE = {"Tactician's Crown": 12, "Tactician's Cape": 12, "Tactician's Shield": 12, "Thief's Gloves": 6}


def is_remover(name):
    return bool(name) and 'Remover' in name


def is_reforger(name):
    return bool(name) and name.startswith('Reforger')


def is_consumable(name):
    """Used up when dragged onto a champion (it is never equipped): Duplicators, Removers,
    Reforgers, Knick-Knacks ... - from the game's own text ("[Consumable - This item disappears
    when used.]"). Stat Boosters (Health Booster, Attack Speed Booster ...) have no text in the game
    data, but they are used up too: 9 of 9 hand-outs in runs 152203..213127 left the item bench and
    no unit's item row ever showed one (each was logged as 'it may be on another unit')."""
    base = re.sub(r'\s*\(.*$', '', name or '')
    # every item the game describes as "Use on a champion ..." is used up by the drop - some have no Consumable tag
    # (benchmark game 20261018-110145: a Lucky Item Chest given to Diana was logged 'missing'; -131328: a Masterwork
    # Upgrade - "Use on a champion to choose a Radiant version ..." - would have been too). The Alpha Mark is reusable
    # (is_reusable, checked first)
    return bool(base) and (bool(re.search(r'Duplicator|Remover|Reforger|\bBooster$|\bChest$', base))
                           or 'Consumable' in item_desc(base)
                           or bool(re.match(r'Use on a champion\b', item_desc(base) or '')))


def is_duplicator(name):
    return bool(name) and 'Duplicator' in name


def uses_left(name):
    m = re.search(r'\((\d+) uses left\)', name or '')
    return int(m.group(1)) if m else None


def consumable_note(name):
    """What the model is told next to a consumable on the item bench (None: not one)."""
    n = uses_left(name)
    uses = f'{n} uses left' if n else ('infinite uses' if name and 'Golden' in name else 'uses not read')
    if is_remover(name) or is_reforger(name):
        return '' if n else uses           # a count read off the tooltip is in the name already
    return None


def holder_kind(unit):
    info = unit.info
    if not info:
        return None
    if info.melee and (info.armor >= 45 or info.hp >= 800 or (info.cost == 1 and info.armor >= 40)):
        return 'tank'
    c = CATALOG.champion_raw(info.name)
    desc = ((c or {}).get('ability') or {}).get('description', '').lower()
    phys, magic = desc.count('physical damage'), desc.count('magic damage')
    if phys > magic:
        return 'AD'
    if magic > phys:
        return 'AP'
    return 'AD' if not info.melee else 'tank'


# ---- recognition ------------------------------------------------------------------

_icon_cache: dict = {}


def _icon_bank(size):
    if size not in _icon_cache:
        names, vecs = [], []
        for n, img in ICONS.items():
            v = _vec(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))
            names.append(n)
            vecs.append(v)
        _icon_cache[size] = (names, np.stack(vecs))
    return _icon_cache[size]


def _vec(icon):
    """Centre 60% of the icon (clear of the count badge), zero-mean unit-norm."""
    s = icon.shape[0]
    k0, k1 = round(s * 0.25), round(s * 0.85)
    c = cv2.resize(icon[k0:k1, round(s * 0.2):round(s * 0.8)], (16, 16), interpolation=cv2.INTER_AREA)
    v = c.astype(np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


# In-game icon crops with their names (data/item-icons-game, labels.json): the official art in
# item-assets differs from what the game draws this set (B.F. Sword scored 0.49-0.62 against it,
# Negatron Cloak 0.51, Infinity Edge 0.69; the Champion Duplicator matched "Guinsoo's" 0.55).
# Labels come from the game's own text: item-bench tooltip reads and the names of item drags in the
# run logs (votes per icon group), and for icons over units the group's art checked against those.
# Icons named at runtime (tooltip read, the model, a drag's outcome) are added in
# data/item-icons-learned, so a name is read once and then known from the picture.
# The bank is compared as fingerprints: the sample times a fixed random 256 x 768 matrix (similarities kept, the
# picture not). The release ships the fingerprints (data/item-fingerprints.npz, tools/make_release.py), no crops of
# the game; experiments/README.md section 3: the same names as the crops on 1110 of 1119 icons, none wrong.
GAME_ICONS = DATA / 'item-icons-game'
LEARNED_ICONS = DATA / 'item-icons-learned'
FINGERPRINTS = DATA / 'item-fingerprints.npz'
_P = np.random.default_rng(18).standard_normal((256, 768)).astype(np.float32) / 16.0
_game = None
_curated = None


def fingerprint(v):
    """A _vec sample (768 numbers) -> its 256-number fingerprint, unit length."""
    f = _P @ v
    return f / (np.linalg.norm(f) + 1e-6)


def _icon_fp(crop):
    return fingerprint(_vec(cv2.resize(crop, (24, 24), interpolation=cv2.INTER_AREA)))


def _folder_fps(folder, keep=None):
    try:
        labels = json.loads((folder / 'labels.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return []
    out = []
    for f, name in labels.items():
        if keep and not keep(folder / f):
            continue
        img = cv2.imread(str(folder / f))
        if img is not None:
            out.append((name, _icon_fp(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
    return out


def _load_banks():
    """The in-game bank: shipped fingerprints (curated: hand-labelled in-game icons; the rest learned) plus the icons
    in the local folders."""
    global _game, _curated
    shipped, cur = [], []
    if FINGERPRINTS.exists():
        z = np.load(FINGERPRINTS)
        for n, v, c in zip(z['names'], z['vecs'], z['curated']):
            shipped.append((str(n), v.astype(np.float32)))
            if c:
                cur.append(shipped[-1])
    game = _folder_fps(GAME_ICONS)
    _curated = cur + game
    _game = shipped + game + _folder_fps(LEARNED_ICONS)


def _game_bank():
    """[(name, fingerprint)] of in-game icons (labelled + learned)."""
    if _game is None:
        _load_banks()
    return _game


def save_fingerprints(path=FINGERPRINTS, keep=None):
    """The local folders' icons (those keep(path) accepts) as fingerprints only: the release's
    data/item-fingerprints.npz."""
    rows = [(n, v, True) for n, v in _folder_fps(GAME_ICONS, keep)] + \
        [(n, v, False) for n, v in _folder_fps(LEARNED_ICONS, keep)]
    np.savez_compressed(path, names=np.array([n for n, _, _ in rows]), vecs=np.stack([v for _, v, _ in rows]),
                        curated=np.array([c for _, _, c in rows]))
    return len(rows)


def contradicted(crop, name, match=0.85, margin=0.08):
    """Another item's name when the references (official art, the hand-labelled in-game icons - never the learned
    ones) match this crop clearly better than `name` does, else None."""
    if _curated is None:
        _load_banks()
    best = {}
    names, bank = _icon_bank(30)
    for n, sim in zip(names, bank @ _vec(cv2.resize(crop, (30, 30), interpolation=cv2.INTER_AREA))):
        best[n] = max(best.get(n, -1.0), float(sim))
    v = _icon_fp(crop)
    for n, w in _curated:
        best[n] = max(best.get(n, -1.0), float(v @ w))
    if not best:
        return None
    top = max(best, key=best.get)
    if top != name and best[top] >= match and best[top] - best.get(name, -1.0) >= margin and \
            not (name.startswith('Radiant ') and top == name[len('Radiant '):]):
        return top
    return None


def learn_icon(crop, name):
    """Keep an icon whose name was established (tooltip text, a confirmed drag outcome, the
    model naming it from the screenshot): next time it is known from the picture."""
    import hashlib
    global _game
    if crop is None or not name or crop.size == 0:
        return
    v = _icon_fp(crop)
    if any(n == name and float(v @ w) > 0.97 for n, w in _game_bank()):
        return                                  # already known
    try:
        from tft_player.pilot import wisp_names
        if name in wisp_names():
            print(f'learn_icon: not learning {name!r} - a Wisp, not an item (a B.F. Sword icon was learned as Middle Path)')
            return
    except Exception:
        pass
    other = contradicted(crop, name)
    if other:
        # the official art or a hand-labelled in-game icon clearly shows another item: the name is wrong (a tooltip
        # read off another slot, a drag that landed elsewhere) - never learned (2026-10-12: 25 of 128 learned icons
        # were mislabelled; the carousel then read a Giant's Belt as Archangel's Staff and a Chain Vest as a Rod)
        print(f'learn_icon: not learning {name!r} - the picture is {other}')
        return
    LEARNED_ICONS.mkdir(parents=True, exist_ok=True)
    f = f"{re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')}__{hashlib.md5(crop.tobytes()).hexdigest()[:8]}.png"
    cv2.imwrite(str(LEARNED_ICONS / f), cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_RGB2BGR))
    path = LEARNED_ICONS / 'labels.json'
    try:
        labels = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        labels = {}
    labels[f] = name
    path.write_text(json.dumps(labels, indent=1), encoding='utf-8')
    _game_bank().append((name, v))


WEAK_NAMED = 0.92                        # a faint-framed icon over a unit counts named this clearly (fingerprints:
                                         # the same icons score 0.922-0.930 that scored 0.931-0.935 as pictures)
GAME_MATCH, GAME_MARGIN = 0.90, 0.08     # in-game icons: same art, so a close match is required (fingerprints:
                                         # 0.90 keeps every name the crops gave at 0.85 bar 9 of 1119, none wrong)
ART_MATCH, ART_MARGIN = 0.90, 0.15       # official art: different drawing, only near-identical counts
ART_DECISIVE, ART_LEAD = 0.80, 0.30      # ... or a clear lead over every other item


_gmat = None


def _game_matrix():
    """(names, name index per row, matrix) of the in-game bank, rebuilt when it grows."""
    global _gmat
    bank = _game_bank()
    if _gmat is None or _gmat[3] != len(bank):
        names = sorted({n for n, _ in bank})
        idx = np.array([names.index(n) for n, _ in bank], dtype=np.int64)
        mat = np.stack([v for _, v in bank]) if bank else np.zeros((0, 256), np.float32)
        _gmat = (names, idx, mat, len(bank))
    return _gmat


def identify_icon(crop):
    """An item icon crop (RGB, any size) -> (name or 'unknown', score). 'unknown' = an item is
    there but no reference is close enough: never a guess."""
    name, score, _ = identify_best([crop])
    return name, score


def identify_best(crops):
    """The best-named of several crops of one icon (small shifts): (name or 'unknown', score,
    index of the crop). One matrix product for all of them."""
    V = np.stack([_vec(cv2.resize(c, (24, 24), interpolation=cv2.INTER_AREA)) for c in crops])
    gnames, gidx, gmat, _ = _game_matrix()
    best = ('unknown', -1.0, 0)
    if len(gnames):
        S = (V @ _P.T) / (np.linalg.norm(V @ _P.T, axis=1, keepdims=True) + 1e-6) @ gmat.T   # crops x references
        per = np.full((len(crops), len(gnames)), -1.0)
        for j in range(len(gnames)):
            per[:, j] = S[:, gidx == j].max(axis=1)
        srt = np.sort(per, axis=1)
        top, second = srt[:, -1], (srt[:, -2] if per.shape[1] > 1 else np.full(len(crops), -1.0))
        ok = (top >= GAME_MATCH) & (top - second >= GAME_MARGIN)
        k = int(np.argmax(np.where(ok, top, -1)))
        if ok[k]:
            return gnames[int(per[k].argmax())], float(top[k]), k
        k = int(top.argmax())
        best = ('unknown', float(top[k]), k)
    names, bank = _icon_bank(24)
    S = V @ bank.T
    srt = np.sort(S, axis=1)
    # official art: near-identical, or clearly the only candidate (a stack counter or count badge
    # over the icon lowers the score but not the lead: Titan's Resolve 0.83, 0.47 ahead). Checked
    # on held-out icons and 250 tooltip-named bench crops: no wrong name.
    ok = (((srt[:, -1] >= ART_MATCH) & (srt[:, -1] - srt[:, -2] >= ART_MARGIN))
          | ((srt[:, -1] >= ART_DECISIVE) & (srt[:, -1] - srt[:, -2] >= ART_LEAD)))
    k = int(np.argmax(np.where(ok, srt[:, -1], -1)))
    if ok[k]:
        return names[int(S[k].argmax())], float(srt[k, -1]), k
    k2 = int(srt[:, -1].argmax())
    return ('unknown', float(srt[k2, -1]), k2) if srt[k2, -1] > best[1] else best


def bench_slot_crop(frame, i, dx=0, dy=0):
    h, w = frame.shape[:2]
    s = h / 1080
    half = round(SLOT_HALF * s)
    x, y, _ = slot_point(i)
    cx, cy = round(x * s) + dx, round(y * s) + dy
    return frame[cy - half:cy + half, cx - half:cx + half]


class ItemTracker:
    """What is on the item bench, slot by slot, kept from events (the user, 2026-10-15: "the bench should have a
    tracker all the time for everything on it"). Our own give takes an item off and the game closes the gap: the
    items below move up one slot (run 20261015-003101 3-2: Tear, Tear, Chain Vest -> Tear, Chain Vest). A name read
    as text (the tooltip) or known from an event is never replaced by a picture match; the screen adds what arrives
    (orbs, carousels, a Remover's returns) and corrects names only guessed from the picture."""

    def __init__(self, n=N_SLOTS):
        self.n = n
        self.slots = [None] * n          # name, or None (empty)
        self.text = [False] * n          # the name was read as text (tooltip) or follows from our own action

    def names(self):
        return list(self.slots)

    def taken(self, i):
        """Our give took the item off slot i: the rest move up (within the first column, then the second)."""
        if 0 <= i < self.n:
            del self.slots[i], self.text[i]
            self.slots.append(None)
            self.text.append(False)

    def confirm(self, i, name):
        """Slot i holds `name`, read as text."""
        if 0 <= i < self.n:
            self.slots[i], self.text[i] = name, True

    def sync(self, read, text=None):
        """A stable read of the bench: per slot a name, 'unknown' or None; text[i]: that name was read as text.
        -> what changed, for the log ('' when nothing did)."""
        text = list(text or [False] * len(read))
        occ_t = [n is not None for n in self.slots]
        occ_r = [n is not None for n in read]
        if occ_t == occ_r:
            changed = []
            for i, n in enumerate(read):
                if n is None:
                    continue
                if text[i] or self.slots[i] in (None, 'unknown') or (not self.text[i] and n != 'unknown'):
                    if self.slots[i] != n:
                        changed.append(f'{i}: {self.slots[i]} -> {n}')
                    self.slots[i], self.text[i] = n, text[i] or (self.text[i] and self.slots[i] == n)
            return '; '.join(changed)
        k_t, k_r = sum(occ_t), sum(occ_r)
        prefix = lambda occ: all(occ[:sum(occ)])
        if prefix(occ_t) and prefix(occ_r) and k_r > k_t:
            # new items arrived below what we hold: keep ours, take the new ones as read
            for i in range(k_t, k_r):
                self.slots[i], self.text[i] = read[i], text[i]
            return f'arrived: {read[k_t:k_r]}'
        # something we did not do: start again from the screen
        old = [n for n in self.slots if n]
        self.slots = list(read) + [None] * (self.n - len(read))
        self.text = text + [False] * (self.n - len(text))
        return f'reset from the screen (tracked {old}, read {[n for n in read if n]})'


def read_item_bench(frame: np.ndarray, detail=False):
    """-> list of 20 (two columns of 10): item name, 'unknown' (an item is there, not recognised), or None
    (empty). detail=True -> list of (name, score, crop)."""
    h, w = frame.shape[:2]
    s = h / 1080
    size = 2 * round(SLOT_HALF * s)
    out = []
    used = 10 * bench_columns(frame)         # no second column: the trait panel's icons stand there
    for i in range(N_SLOTS):
        crop = bench_slot_crop(frame, i)
        if i >= used or crop.shape[:2] != (size, size) or (crop.mean() < 30 and crop.std() < 18):
            out.append((None, 0.0, None) if detail else None)
            continue
        # icons are small: the match is only sharp when aligned to ~1px, so search a little
        crops = [bench_slot_crop(frame, i, dx, dy) for dy in (-2, 0, 2) for dx in (-2, 0, 2)]
        crops = [c for c in crops if c.shape[:2] == (size, size)]
        name, score, j = identify_best(crops)
        out.append((name, score, crops[j].copy()) if detail else name)
    return out


# The item icons over a unit: up to 3 squares in a row under the left end of its health bar
# (1080p offsets from the bar point board.green_bars gives), each in a dark frame. Measured on
# recorded frames: the frame's top-left is 36.7 left / 10.0 below the bar point, frames are 25.0
# wide and touch (25.0 apart); the art inside starts 0.75 in and is 22.8 wide.
UNIT_ICON = dict(dx=-36.7, dy=10.0, step=24.98, frame=25.0, inset=0.75, art=22.8)


def _best_frame(f, x0, y0, n, rx, ry):
    """(score, x, y): the n x n square near (x0, y0) (+-rx, +-ry) whose 2-px border is darkest -
    score = share of dark pixels in that border."""
    ys, xs = y0 - ry, x0 - rx
    reg = f[max(0, ys):ys + n + 2 * ry + 1, max(0, xs):xs + n + 2 * rx + 1]
    if ys < 0 or xs < 0 or reg.shape[0] < n + 2 * ry + 1 or reg.shape[1] < n + 2 * rx + 1:
        return 0.0, x0, y0
    dark = (reg.max(axis=2) < 70).astype(np.int32)
    ii = np.pad(dark.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    def rect(y, x, m):                    # sums of every m x m square with top-left (y.., x..)
        return ii[y + m:y + m + 2 * ry + 1, x + m:x + m + 2 * rx + 1] - ii[y:y + 2 * ry + 1, x + m:x + m + 2 * rx + 1] \
            - ii[y + m:y + m + 2 * ry + 1, x:x + 2 * rx + 1] + ii[y:y + 2 * ry + 1, x:x + 2 * rx + 1]
    ring = (rect(0, 0, n) - rect(2, 2, n - 4)) / (n * n - (n - 4) ** 2)
    j = np.unravel_index(int(ring.argmax()), ring.shape)
    return float(ring[j]), xs + int(j[1]), ys + int(j[0])


EXPECT_MATCH = 0.90     # an icon is the item a give expects at this score against that item alone (identify_among):
                        # true icons 0.95-0.98, other items and empty slots <= 0.77 (run 20261018-102854 fails 01/12/14)


def read_unit_items(frame: np.ndarray, bar, expect=None):
    """Items shown over one unit: bar = (x, y) of its health bar in 1080p coordinates
    (board.green_bars). -> list of (name or 'unknown', score, art crop), left to right; [] = none.
    An icon counts only with its dark frame and a coloured inside (no frame: no item - items fill
    from the left, so the row ends there). expect: the item a give just put there - an icon that matches it
    alone at EXPECT_MATCH is named it, also in a faint frame (run 20261018-102854: a Sunfire Cape named at 0.913
    against all items, 0.978 against itself, in a 0.75 frame, ended the row: 'no items', the give 'missing')."""
    h, w = frame.shape[:2]
    s = h / 1080
    ins, art = round(UNIT_ICON['inset'] * s), round(UNIT_ICON['art'] * s)
    cx, cy = w / 2 + (bar[0] - 960) * s, bar[1] * s
    out, px, py, n = [], None, None, None
    for k in range(3):
        if px is None:
            # the bar point can be ~12 px off (where the green run starts varies), and a glowing
            # icon's frame reads 1 px wider: search sizes and a wider window for the first icon
            x0, y0 = round(cx + UNIT_ICON['dx'] * s), round(cy + UNIT_ICON['dy'] * s)
            r, px, py, n = max(_best_frame(frame, x0, y0, m, 14, 6) + (m,)
                               for m in {round((UNIT_ICON['frame'] + d) * s) for d in (-0.7, 0, 0.7)})
        else:
            r, px, py = _best_frame(frame, px + n, py, n, 2, 1)
        inner = frame[py + 3:py + n - 3, px + 3:px + n - 3]
        # a faint frame (0.55-0.75: its top edge on the blue mana bar, a glow behind it) counts only around a
        # clearly named icon: Sentinel's Protector's Vow and Aphelios's Bloodthirster were missed round after
        # round at 0.56-0.66 (runs 152104, 213127), while dark bodies there match Hand Of Justice at <= 0.91
        weak = r < 0.75
        if r < 0.55 or inner.size == 0 or inner.std() < 25 or inner.max(axis=2).mean() < 60:
            break
        # the name is only sharp when the art is aligned to ~1 px: try small shifts
        o = ins + (n - round(UNIT_ICON['frame'] * s)) // 2
        crops = [frame[py + o + dy:py + o + dy + art, px + o + dx:px + o + dx + art]
                 for dy in (-2, -1, 0, 1, 2) for dx in (-2, -1, 0, 1, 2)]
        crops = [c for c in crops if c.shape[:2] == (art, art)]
        if not crops:
            break
        name, score, j = identify_best(crops)
        if expect and name != expect:
            own = identify_among(crops, [expect])[1]
            if own >= EXPECT_MATCH and (name == 'unknown' or score < own):
                name, score = expect, own
        if weak and not (name == expect and score >= EXPECT_MATCH) and (name == 'unknown' or score < WEAK_NAMED):
            break
        out.append((name, score, crops[j].copy()))
    return out


PANEL_ITEMS = [(1728, 573), (1800, 573), (1872, 573)]     # item slots of a unit's info panel (right-anchored)


def panel_item_crops(frame, k=0):
    """Shifted crops of item slot k of an open unit info panel, or None when the slot is empty."""
    h, w = frame.shape[:2]
    s = h / 1080
    half = round(29 * s)
    px, py = PANEL_ITEMS[k]
    cx, cy = round(w - (1920 - px) * s), round(py * s)
    c0 = frame[cy - half:cy + half, cx - half:cx + half]
    if c0.shape[:2] != (2 * half, 2 * half) or (c0.mean() < 32 and c0.std() < 12):
        return None
    return [c for dy in (-2, 0, 2) for dx in (-2, 0, 2)
            if (c := frame[cy + dy - half:cy + dy + half, cx + dx - half:cx + dx + half]).shape[:2] == (2 * half, 2 * half)]


def identify_among(crops, names):
    """Which of `names` an icon shows (shifted crops of it), by the in-game and the official icons: (name, score,
    lead over the next name). A short list - a carousel's offer - separates far better than all items."""
    names = {n for n in names if n}
    if not crops or not names:
        return None, -1.0, 0.0
    V = np.stack([_vec(cv2.resize(c, (24, 24), interpolation=cv2.INTER_AREA)) for c in crops])
    scores = {}
    gnames, gidx, gmat, _ = _game_matrix()
    if len(gnames):
        # the in-game references are fingerprints (256 projections, prints.py): the crops are projected the same
        # way (identify_best) - a raw 768-number vector against them raised a shape error since 2026-10-16
        P = V @ _P.T
        S = (P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-6)) @ gmat.T
        for j, n in enumerate(gnames):
            if n in names:
                scores[n] = max(scores.get(n, -1.0), float(S[:, gidx == j].max()))
    anames, bank = _icon_bank(24)
    S = V @ bank.T
    for j, n in enumerate(anames):
        if n in names:
            scores[n] = max(scores.get(n, -1.0), float(S[:, j].max()))
    if not scores:
        return None, -1.0, 0.0
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    lead = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else -1.0)
    return ranked[0][0], ranked[0][1], lead


def read_panel_items(frame: np.ndarray):
    """Items shown in an open unit info panel: up to 3 names ('unknown' = something unrecognised)."""
    h, w = frame.shape[:2]
    s = h / 1080
    half = round(29 * s)
    names, bank = _icon_bank(38)
    out = []
    for px, py in PANEL_ITEMS:
        cx, cy = round(w - (1920 - px) * s), round(py * s)
        c0 = frame[cy - half:cy + half, cx - half:cx + half]
        if c0.shape[:2] != (2 * half, 2 * half) or (c0.mean() < 32 and c0.std() < 12):
            continue                      # an empty slot
        best, second, bj = -1.0, -1.0, 0
        for dy in (-2, 0, 2):
            for dx in (-2, 0, 2):
                c = frame[cy + dy - half:cy + dy + half, cx + dx - half:cx + dx + half]
                if c.shape[:2] != (2 * half, 2 * half):
                    continue
                sims = bank @ _vec(cv2.resize(c, (38, 38), interpolation=cv2.INTER_AREA))
                order = np.argsort(-sims)
                if sims[order[0]] > best:
                    best, second, bj = float(sims[order[0]]), float(sims[order[1]]), int(order[0])
        out.append(names[bj] if best >= 0.75 and best - second >= 0.2 else 'unknown')
    return out


# ---- decisions --------------------------------------------------------------------

def plan_strip(bench_items, board_units, plan, stripped):
    """-> (slot_index, unit, why): use a Remover to take items off a champion, or None.

    Only when the carry has room and a champion holds completed items that are worth
    clearly more on the carry than on itself; the items return to the bench and the
    normal item rules then hand them out (the carry weighs most). Once per unit per
    id set in `stripped`."""
    slot = next((i for i, n in enumerate(bench_items) if is_remover(n)), None)
    if slot is None:
        return None
    carry = next((u for u in board_units if u.info and plan.carry and u.name == plan.carry), None)
    if not carry or holder_kind(carry) not in ('AD', 'AP'):
        return None
    free = sum(1 for n in bench_items if not n)
    for u in board_units:
        if u is carry or u.id in stripped or not u.items or free < len(u.items):
            continue
        k = holder_kind(u)
        if k == holder_kind(carry):
            # same damage type: the items belong on the carry once it clearly outranks this holder
            rank = lambda x: (x.cost + 1) * {1: 1, 2: 1.8, 3: 3}.get(x.star, 1)
            gain = sum(VALUE[k].get(i, 0) for i in u.items if i in RECIPES) if rank(carry) >= 1.5 * rank(u) else 0
        else:
            gain = sum(VALUE[holder_kind(carry)].get(i, 0) - VALUE.get(k, {}).get(i, 0)
                       for i in u.items if i in RECIPES)
        room = 3 - len(carry.items)
        if gain >= 8 and room >= 1:
            return slot, u, f'remover: {", ".join(u.items)} are worth more on {carry} than on {u}'
    return None


def plan_item_move(bench_items, board_units, plan, stage):
    """-> (slot_index, unit, why) for the next component to hand out, or None.

    Priority: finish an item on a unit already holding a component; else
    build the best (item, holder) pair from two bench components; late in
    the game, hand a lone component to the carry anyway."""
    slots = [(i, n) for i, n in enumerate(bench_items) if n in COMPONENTS]
    full = [(i, n) for i, n in enumerate(bench_items) if n in RECIPES]
    holders = [u for u in board_units if u.info]
    if not holders:
        return None

    def kind(u):
        return holder_kind(u)

    def cap(u):
        return len(u.items) < 3

    def value(item, u):
        if item in ANYONE:
            return ANYONE[item]
        k = kind(u)
        v = VALUE.get(k, {}).get(item, 0)
        if not v:
            return 0
        weight = (u.cost + 1) * {1: 1, 2: 1.8, 3: 3}.get(u.star, 1)
        if plan.carry and u.name == plan.carry:
            weight *= 1.8
        return v * weight

    # 0. completed items straight from the bench (anvils, carousel)
    for i, n in full:
        best = max(((value(n, u), u) for u in holders if cap(u)), key=lambda t: t[0], default=None)
        if best and best[0] > 0:
            return i, best[1], f'equip {n}'

    # 1. finish a half-built item with the best-value completion
    done = None
    for u in holders:
        for c in (c for c in u.items if c in COMPONENTS):
            for i, n in slots:
                item = BY_PAIR.get(tuple(sorted((c, n))))
                v = value(item, u) if item else 0
                if v > 0 and (done is None or v > done[0]):
                    done = (v, i, u, f'{c}+{n} -> {item} on {u}')
    if done:
        return done[1:]

    # 2. start the best new item from two bench components
    best = None
    for a in range(len(slots)):
        for b in range(a + 1, len(slots)):
            item = BY_PAIR.get(tuple(sorted((slots[a][1], slots[b][1]))))
            if not item:
                continue
            for u in holders:
                if len(u.items) > 1 or any(c in COMPONENTS for c in u.items):
                    continue            # needs room for a full item, no half-built one pending
                v = value(item, u)
                if v > 0 and (best is None or v > best[0]):
                    best = (v, slots[a][0], u, item)
    if best:
        return best[1], best[2], f'build {best[3]} on {best[2]}'

    # 3. from stage 4 a lone component is still stats on the carry
    if stage and stage >= (4, 1) and slots:
        carry = max((u for u in holders if kind(u) in ('AD', 'AP') and cap(u)),
                    key=lambda u: (u.cost + 1) * u.star, default=None)
        if carry:
            return slots[0][0], carry, f'late: {slots[0][1]} on {carry}'
    return None


_DESC = {}

# Items of this set the game data we have does not list (no name, no text): their name is what the item's tooltip
# shows, so the tooltip read matches it and its icon is learned. The Alpha Mark (TFTSet18_Item_RiftbeastSelector,
# from 3 Riftbeast) - its text as tftips.app and blitz.gg give it (2026-10-13); it stays on the item bench all the
# time and is dragged onto a unit like a consumable (the user, 2026-10-13).
MISSING_ITEMS = {
    'Alpha Mark': ("[Stays on your item bench: use it again to move the Alpha.] Drag it onto a Riftbeast champion to "
                   "make it your Alpha Riftbeast: its unique Buff (named in its ability text) turns on. With Omega "
                   "Riftbeast you have 2 Alphas and a new one replaces the oldest. From the Riftbeast trait at 3."),
}
REUSABLE = {'Alpha Mark'}       # used by a drag onto a unit, never equipped, never used up


def is_reusable(name):
    return re.sub(r'\s*\(.*$', '', name or '') in REUSABLE


def item_desc(name):
    """The game's own description of an item (what hovering it in game shows), with its numbers
    filled in (data/sources/cdragon-*.json). '' when the game data has none. The model must not
    know less than a player: Duplicators, emblems, anvils and completed items included."""
    import json
    import re
    from pathlib import Path
    base = re.sub(r'\s*\(.*$', '', name or '').strip()
    if base in _DESC:
        return _DESC[base]
    if not _DESC:
        src = sorted((Path(__file__).resolve().parent.parent / 'data' / 'sources').glob('cdragon-tft-*.json'))
        try:
            data = json.loads(src[-1].read_text(encoding='utf-8'))['items'] if src else []
        except (OSError, ValueError, KeyError):
            data = []
        def rank(api):
            # this Set's own entry, then the set-less generic one (same mechanics every set),
            # then any older set's text
            # (never a tutorial or "Free" variant: TFT_Item_FreeBFSword has 15x the AD)
            if re.search(r'Tutorial|_Free', api):
                return 3
            return 0 if api.startswith('DA_') else 1 if re.match(r'TFT_(Item|Consumable)_', api) else 2
        best = {}
        for it in data:
            n, d, api = it.get('name'), it.get('desc') or '', it.get('apiName') or ''
            if not n or not d:
                continue
            if n not in best or (rank(api), -len(d)) < best[n][0]:
                best[n] = ((rank(api), -len(d)), it)
        for n, (_, it) in best.items():
            d = it.get('desc') or ''
            eff = {k.lower(): v for k, v in (it.get('effects') or {}).items()}

            def fill(m):
                key, mult = m.group(1).lower(), float(m.group(2) or 1)
                v = eff.get(key)
                if not isinstance(v, (int, float)):
                    return 'N'
                v = v * mult
                return str(int(v)) if float(v).is_integer() else f'{v:.1f}'
            d = re.sub(r'@([A-Za-z0-9_]+)(?:\*([0-9.]+))?@', fill, d)
            d = re.sub(r'%i:[^%]*%', '', d)            # inline icons
            d = re.sub(r'@[^@\s]*@', '', d)           # values the data does not fill (@TFTUnitProperty...@)
            d = re.sub(r'<br\s*/?>', ' ', d)
            d = re.sub(r'<[^>]+>', '', d)
            d = re.sub(r'\{\{[^}]*\}\}', '', d)
            d = re.sub(r'\s+', ' ', d).strip()
            if n.endswith(' Emblem'):
                # only what is true in every set: the trait (bonus lines differ per set)
                d = f"The holder gains the {n[:-len(' Emblem')]} trait."
            _DESC[n] = d
        _DESC.setdefault('', '')
        for n, d in MISSING_ITEMS.items():
            _DESC.setdefault(n, d)
    if base not in _DESC and base.endswith(' Emblem'):
        # an emblem the game data has no entry for (Blossom Emblem, run 20261010-010345): its trait is known
        from tft_player.catalog import CATALOG
        trait = base[:-len(' Emblem')]
        if trait in CATALOG.traits:
            _DESC[base] = f'The holder gains the {trait} trait.'
    return _DESC.get(base, '')
