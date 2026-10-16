"""Bench slots, and the choice screen a bench container opens.

Every bench slot is one of:

    unit          a green health bar stands over it (board.occupancy)
    empty         the bare stone tile: matches this slot's empty reference
    interactable  a container (anvil: component, completed item, artifact ...) - it occupies the slot;
                  it is opened by selling it and then offers a choice
    unknown       something is there that is neither (a unit whose bar is not visible, a loot orb,
                  our tactician, an effect): occupied, never a free slot
    obscured      a tooltip or panel covers the tile: nothing can be said about it

Only `empty` is a free slot (a move or a buy can land there).

Empty references are taken in each game from that game's own bench: the arena sets the tiles and it
changes from game to game, so none are stored (learn_empty, called by the player). A slot is empty
when, after a median blur (thin overlays: drag outlines) and a tint correction, under 7% of its
tile differs from the reference; recorded empties score 0.00-0.07, units and containers 0.10-0.82.
A slot without a reference yet is never 'empty'.
Containers are matched against exemplars cut at several slots (the perspective changes along the
bench: one slot-1 template scored a slot-7 anvil 0.60, the multi-slot exemplars 0.90), kept as
fingerprints (prints.py): each window of the slot where an exemplar can sit is compared with them.
Recorded containers score >= 0.93, everything else <= 0.80 (experiments/README.md section 6).

The choice screen replaces the shop with a row of option cards (brown-olive on the panel's dark
teal); choice_cards() finds them left to right, so the model can answer with an option index and the
executor clicks that card.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tft_player import prints
from tft_player.board import BENCH

OBJECTS = Path(__file__).resolve().parent.parent / 'data' / 'bench-objects'      # exemplar pictures (development)
OBJECT_PRINTS = OBJECTS.parent / 'bench-fingerprints.npz'                         # their fingerprints (the release)
CONTAINER = 0.87        # best window's fingerprint match for a container (recorded: containers >= 0.93, others <= 0.80)
EXEMPLAR = (35, 45)     # exemplar size (w, h) at half scale: 70x90 at 1080p
GRID = (12, 16)         # its sample: columns, rows
EMPTY = 0.07            # share of the tile differing from the empty reference (empties 0.00-0.07)
SHIFT = 15.0            # and the tile's colour close to the reference's (empties <= 13)
OBSCURED = 0.35         # share of near-black pixels: a tooltip / panel over the tile
_templates = None
_empty = None
CROP_Y = (690, 805)     # 1080p rows of a bench slot crop; columns are the slot centre -52..+53


def _sample(img):
    return prints.sample(cv2.resize(img, GRID, interpolation=cv2.INTER_AREA))


def templates():
    """(classes, fingerprint matrix) of the container exemplars: the shipped fingerprints and the pictures in
    data/bench-objects ('<class>.png', '<class>__s<slot>.png')."""
    global _templates
    if _templates is None:
        names, fps = [], []
        if OBJECT_PRINTS.exists():
            z = np.load(OBJECT_PRINTS)
            names += [str(n) for n in z['names']]
            fps += list(z['vecs'])
        for p in sorted(OBJECTS.glob('*.png')):
            t = cv2.imread(str(p))
            names.append(p.stem.split('__')[0])
            fps.append(prints.fingerprint(_sample(cv2.resize(t, EXEMPLAR, interpolation=cv2.INTER_AREA))))
        _templates = (names, np.stack(fps).astype(np.float32) if fps else np.zeros((0, 256), np.float32))
    return _templates


def save_fingerprints(path=OBJECT_PRINTS):
    """The exemplar pictures as fingerprints only: the release's data/bench-fingerprints.npz."""
    names, fps = templates()
    np.savez_compressed(path, names=np.array(names), vecs=fps)
    return len(names)


def learn_empty(frame, slots):
    """Each slot's empty picture from this game's own bench (the arena sets the tiles)."""
    fr = _bench_band(frame)
    refs = empty_refs()
    for i in slots:
        refs[i] = cv2.medianBlur(slot_crop(fr, i), 11).astype(np.float32)
    _memo.clear()


def forget_empty():
    """A new game: its arena may differ, so the last game's empty pictures go."""
    empty_refs().clear()
    _memo.clear()


def empty_refs():
    global _empty
    if _empty is None:
        _empty = {}
    return _empty


def _at_1080(frame):
    """RGB frame of any size -> BGR at 1080 lines (the templates' scale)."""
    h, w = frame.shape[:2]
    bgr = frame[..., ::-1]
    if h == 1080:
        return np.ascontiguousarray(bgr)
    return cv2.resize(np.ascontiguousarray(bgr), (round(w * 1080 / h), 1080), interpolation=cv2.INTER_AREA)


def slot_crop(fr, i):
    """BGR 1080p frame -> the crop of bench slot i (115 x 105)."""
    w = fr.shape[1]
    cx = round(w / 2 + (BENCH[i][0] - 960))
    return np.ascontiguousarray(fr[CROP_Y[0]:CROP_Y[1], cx - 52:cx + 53])


def occupied_share(crop, i):
    """Share of slot i's tile that differs from its empty reference (None: no reference)."""
    ref = empty_refs().get(i)
    if ref is None or crop.shape[:2] != ref.shape[:2]:
        return None
    a = cv2.medianBlur(crop, 11).astype(np.float32)
    a = a - a[40:].mean(axis=(0, 1)) + ref[40:].mean(axis=(0, 1))       # tint (lighting, effects)
    diff = np.abs(a - ref).mean(axis=2) > 30
    return float(diff[40:, 15:90].mean())


def colour_shift(crop, i):
    """How far the tile's mean colour is from the empty reference's (max over channels). Real
    empty bench tiles: <= 13; a card, panel or another arena's ground over the slot: 17-140 (the
    tint correction alone let a dark augment card over slot 3 at run27 3-2 pass as empty)."""
    ref = empty_refs().get(i)
    if ref is None:
        return None
    return float(np.abs(crop[40:].astype(np.float32).mean(axis=(0, 1)) - ref[40:].mean(axis=(0, 1))).max())


def container_match(crop):
    """(class, score) of the best container exemplar: every window of the slot (half size, 2 px apart) where an
    exemplar fits, as fingerprints, against the exemplars' fingerprints (~4 ms)."""
    names, T = templates()
    c = cv2.resize(crop, (crop.shape[1] // 2, crop.shape[0] // 2), interpolation=cv2.INTER_AREA)
    w, h = EXEMPLAR
    if not names or c.shape[0] < h or c.shape[1] < w:
        return None, -1.0
    W = np.stack([_sample(c[y:y + h, x:x + w]) for y in range(0, c.shape[0] - h + 1, 2)
                  for x in range(0, c.shape[1] - w + 1, 2)])
    F = W @ prints.matrix(T.shape[1], W.shape[1]).T
    S = (F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-6)) @ T.T
    j = int(S.max(axis=0).argmax())
    return names[j], float(S[:, j].max())


_memo: dict = {}


def _sig(crop):
    return cv2.resize(crop, (8, 8), interpolation=cv2.INTER_AREA).astype(np.int16) // 6


def _bench_band(frame):
    """The bench rows of the frame only, as BGR at 1080p scale, padded to the full 1080p width so
    slot_crop() can use frame coordinates (resizing the whole frame took most of the time)."""
    h, w = frame.shape[:2]
    s = h / 1080
    W = round(w / s)
    y0, y1 = int(CROP_Y[0] * s), int(np.ceil(CROP_Y[1] * s)) + 1
    band = np.ascontiguousarray(frame[y0:y1, :, ::-1])
    band = cv2.resize(band, (W, round((y1 - y0) / s)), interpolation=cv2.INTER_AREA) if s != 1 else band
    off = round(y0 / s)
    fr = np.zeros((CROP_Y[1] + 2, W, 3), np.uint8)
    n = min(band.shape[0], fr.shape[0] - off)
    fr[off:off + n] = band[:n]
    return fr


def classify(frame, bench_bars):
    """-> 9 entries: {'kind': 'unit' | 'empty' | 'interactable' | 'unknown' | 'obscured',
    'cls': container class or None, 'conf': container score, 'occ': occupied share}."""
    fr = _bench_band(frame)
    out = []
    for i in range(len(BENCH)):
        if bench_bars[i]:
            out.append({'kind': 'unit', 'cls': None, 'conf': 1.0, 'occ': None})
            continue
        crop = slot_crop(fr, i)
        sig = _sig(crop)
        old = _memo.get(i)
        if old is not None and old[0].shape == sig.shape and np.abs(old[0] - sig).max() <= 1:
            out.append(dict(old[1]))                 # the same pixels as last time: the same answer
            continue
        occ = occupied_share(crop, i)
        shift = colour_shift(crop, i)
        bare = occ is not None and occ < EMPTY and shift is not None and shift <= SHIFT
        dark = float((crop[40:].max(axis=2) < 40).mean())
        cls, conf = (None, 0.0) if bare else container_match(crop)
        if conf >= CONTAINER:
            kind = 'interactable'
        elif dark >= OBSCURED:
            kind = 'obscured'
        elif bare:
            kind = 'empty'
        else:
            kind = 'unknown'
        res = {'kind': kind, 'cls': cls if kind == 'interactable' else None,
               'conf': round(conf, 2), 'occ': None if occ is None else round(occ, 3)}
        _memo[i] = (sig, res)
        out.append(res)
    return out


def choice_cards(frame, fallback=True):
    """The option cards of an open choice screen: the usual strip first; if that finds nothing, the
    upper part of the cards alone - the MetaTFT companion's stat popups ("Avg Place / Popular With")
    cover the cards' lower half (run 20261006-221257 6-2: a 5-card anvil screen went unseen, the
    model was never asked and the game picked on timeout). The click point stays on the visible part.
    fallback=False when a champion shop is readable: the upper strip alone then took shop card art
    for 2 cards (run 033507 4-1, right after a Wisp was bought)."""
    cards = _choice_cards(frame, 930, 1050, 990)
    return cards or (_choice_cards(frame, 956, 976, 965) if fallback else [])


def _choice_cards(frame, y0, y1, click_y):
    """The option cards of an open choice screen, left to right, as 1080p reference
    points (centre anchor) to click - or [] when no choice screen is showing.
    The outer cards of a wide row (5 options) can be cut by a panel at the window's edge (run27
    3-2 and 7-1: our overlay covered 30-50 px of them, so the row was not recognised): an outer
    card only needs to be partly visible and in step with the others; its click point is where the
    row's spacing puts its centre."""
    h, fw = frame.shape[:2]
    s = h / 1080
    w = round(fw / s)
    strip = np.ascontiguousarray(frame[int(y0 * s):int(np.ceil(y1 * s))])
    band = (cv2.resize(strip, (w, y1 - y0), interpolation=cv2.INTER_AREA) if s != 1 else strip).astype(np.int16)
    r, g, b = band[..., 0], band[..., 1], band[..., 2]
    card = (r > 25) & (r > b + 12) & (g > b + 8)             # brown-olive card, not the dark teal panel
    on = card.mean(axis=0) > 0.35
    segs, start = [], None
    for x, v in enumerate(list(on) + [False]):
        if v and start is None:
            start = x
        if not v and start is not None:
            if x - start > w * 0.05:
                segs.append((start, x))
            start = None
    if not 2 <= len(segs) <= 6:
        return []
    # a choice panel: equal, wide cards with narrow gaps of the dark teal panel between them
    widths = [b - a for a, b in segs]
    full = max(widths)
    if full < w * 0.1:
        return []
    whole = [abs(x - full) <= full * 0.12 for x in widths]
    if not all(whole[1:-1]) or sum(whole) < max(1, len(segs) - 2):
        return []
    for k in (0, len(segs) - 1):
        if not whole[k] and widths[k] < full * 0.35:
            return []
    for (a0, b0), (a1, b1) in zip(segs, segs[1:]):
        gap = band[:, b0:a1]
        if a1 - b0 > w * 0.05 or gap.size == 0:
            return []
        teal = (gap[..., 0] < 30) & (gap[..., 2] >= gap[..., 0])
        if teal.mean() < 0.5:
            return []
    centres = [(a + b) / 2 for a, b in segs]
    pitch = full + float(np.median([a1 - b0 for (_, b0), (a1, _) in zip(segs, segs[1:])]))
    # an outer card even a few px narrower is cut at its outer edge: place it by the spacing
    if len(segs) > 1 and widths[0] < full - 4:
        centres[0] = centres[1] - pitch
    if len(segs) > 1 and widths[-1] < full - 4:
        centres[-1] = centres[-2] + pitch
    return [(960 + c - w / 2, click_y) for c in centres]
