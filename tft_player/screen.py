"""Read the TFT HUD from one frame, fast.

Regions are measured on a 1920x1080 client. The HUD scales with client height
and is anchored to the horizontal centre: x' = w/2 + (x-960)*s, y' = y*s,
s = h/1080.

OCR is RapidOCR (PP-OCR recognition only; the regions are fixed so text
detection is skipped). Each crop gets a tiny pixel fingerprint and is only
re-recognised when it changes, so a typical tick OCRs one or two crops.
"""
from __future__ import annotations

import difflib
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tft_player.catalog import CATALOG
from tft_player.meta import META

# ---- regions in 1080p pixels -------------------------------------------------
STAGE = [(755, 4, 815, 32), (810, 4, 870, 32)]     # normal bar, narrow stage-1 bar
TIMER = [(1125, 4, 1175, 32), (1070, 4, 1120, 32)]
LEVEL = (345, 880, 420, 910)
XP = (465, 882, 530, 908)
GOLD = (1015, 882, 1080, 912)
REROLL_COST = (372, 1036, 412, 1066)    # the gold cost on the Reroll button: 0 while free rerolls last
REROLL_FREE = (456, 1006, 494, 1040)    # the count badge on the Reroll button (free rerolls left)
SHOP_X0, SHOP_STEP = 555, 201.5
AUGMENT_TITLE = (830, 198, 1090, 248)               # "Choose One"
BOARD_COUNT = (880, 210, 1150, 315)                  # big "6/7" watermark shown in planning
AUGMENT_POSITIONS = ['left card', 'middle card', 'right card']     # the augment picker, by position only
AUGMENT_NAMES = [(415, 526, 700, 560), (815, 526, 1105, 560), (1220, 526, 1510, 560)]
HP_DIGITS_X = (1778, 1838)      # the digits inside our (big) HP box


def shop_name_box(i):
    x0 = SHOP_X0 + SHOP_STEP * i
    return (x0 + 2, 1043, x0 + 160, 1072)


@dataclass
class Screen:
    t: float = 0.0
    stage: str | None = None          # "3-2"
    timer: int | None = None
    level: int | None = None
    xp: tuple[int, int] | None = None
    gold: int | None = None
    hp: int | None = None
    shop: list[str | None] = field(default_factory=lambda: [None] * 5)
    shop_raw: list[str] = field(default_factory=lambda: [''] * 5)   # OCR text: wisps aren't champions
    shop_visible: bool = False
    augments: list[str] | None = None
    items: list | None = None          # item bench, filled by the player in planning
    board_count: int | None = None     # team slots used, from the game's own "n/m" watermark
    team_size: int | None = None       # its m, once two reads agree (the level unless something adds slots)
    free_rolls: int | None = None      # free rerolls left (Reroll button at 0 gold, its badge); 0 = rolls cost gold
    ms: float = 0.0

    @property
    def round(self):
        if not self.stage:
            return None
        a, b = self.stage.split('-')
        return int(a), int(b)

    def short(self):
        shop = ','.join(s or '-' for s in self.shop) if self.shop_visible else 'closed'
        aug = f' AUG={self.augments}' if self.augments else ''
        return (f"{self.stage or '?'} t={self.timer} L{self.level} xp={self.xp} "
                f"g={self.gold} hp={self.hp} shop=[{shop}]{aug} ({self.ms:.0f}ms)")


def read_board_count(text, level):
    """The game's 'n/m' over the board: n team slots used of m, the team size. -> (n, m) or None.
    m is the level unless something adds slots (a +team-size augment, a Tactician's Shield, Riftbeast 10) or
    sets it (an augment that makes it 1). OCR often reads the slash as '7' ('474') or drops it ('54'): without
    a slash, m is the level or up to 4 more, the level first."""
    t = re.sub(r'[^0-9/]', '', text or '')
    if not level:
        return None
    m = re.fullmatch(r'(\d{1,2})/(\d{1,2})', t)
    if m and 1 <= int(m.group(2)) <= level + 4 and int(m.group(1)) <= int(m.group(2)) + 2:
        return int(m.group(1)), int(m.group(2))
    for size in range(level, level + 5):
        m = re.fullmatch(rf'(\d{{1,2}}?)[/7]?{size}', t)
        if m and 0 <= int(m.group(1)) <= size + 2:
            return int(m.group(1)), size
    return None


def parse_board_count(text, level):
    """'6/7' -> 6 at level 7: the n of the game's 'n/m' (read_board_count)."""
    got = read_board_count(text, level)
    return got[0] if got else None


def match_name(text, names, cutoff=0.75):
    t = re.sub(r"[^a-z'+]", '', text.lower())       # keep '+': "Trade Sector" != "Trade Sector+"
    if len(t) < 2:
        return None
    keys = {}
    for n in names:
        keys.setdefault(re.sub(r"[^a-z'+]", '', n.lower()), n)
    if t in keys:
        return keys[t]
    hit = difflib.get_close_matches(t, keys, n=1, cutoff=cutoff)
    return keys[hit[0]] if hit else None


MEMO_FILE = Path(__file__).resolve().parent.parent / 'data' / 'ocr_memo.npz'


class OcrMemo:
    """Crops we've OCR'd before, per field: exact pixel match -> text in ~0.1 ms.
    HUD text renders identically every time, so after a game or two nearly
    every read is a lookup. Saved to disk."""

    def __init__(self, path=MEMO_FILE):
        self.path = path
        self.stacks: dict[str, np.ndarray] = {}
        self.sigs: dict[str, np.ndarray] = {}
        self.texts: dict[str, list[str]] = {}
        self.dirty = 0
        self.saved_at = time.time()
        try:
            with np.load(path, allow_pickle=False) as z:
                for k in z.files:
                    if k.endswith('|img'):
                        name = k[:-4]
                        self.stacks[name] = z[k]
                        self.sigs[name] = np.stack([self._sig(c) for c in z[k]])
                        self.texts[name] = list(z[name + '|txt'])
        except (FileNotFoundError, OSError, ValueError):
            pass

    @staticmethod
    def _sig(crop):
        """Small grayscale signature for a fast first pass."""
        return crop[::3, ::3].astype(np.int16).sum(axis=2)

    def get(self, key, crop):
        st = self.stacks.get(key)
        if st is None or st.shape[1:] != crop.shape:
            return None
        d = np.abs(self.sigs[key] - self._sig(crop)).mean(axis=(1, 2))
        i = int(d.argmin())
        if d[i] > 12:                       # summed over 3 channels
            return None
        # verify: essentially pixel-identical (one changed digit alters far more than 0.4%)
        diff = np.abs(st[i].astype(np.int16) - crop.astype(np.int16)).max(axis=2) > 40
        return self.texts[key][i] if diff.mean() < 0.004 else None

    # only fixed HUD text repeats pixel-for-pixel; crops over the moving board or
    # art (board counter, panel names, picker titles) never hit and just bloat it
    KEEP = ('level', 'xp', 'gold', 'stage', 'timer', 'stage1', 'timer1', 'hp', 'hpbox',
            'shop0', 'shop1', 'shop2', 'shop3', 'shop4', 'rrcost', 'rrbadge')
    CAP = 400

    def add(self, key, crop, text):
        if key.split('@')[0] not in self.KEEP:
            return
        st = self.stacks.get(key)
        if st is not None and st.shape[1:] != crop.shape:
            return
        if st is not None and len(st) >= self.CAP:          # oldest out
            self.stacks[key], self.sigs[key] = st[1:], self.sigs[key][1:]
            self.texts[key] = self.texts[key][1:]
            st = self.stacks[key]
        self.stacks[key] = crop[None].copy() if st is None else np.concatenate([st, crop[None]])
        sig = self._sig(crop)[None]
        self.sigs[key] = sig if st is None else np.concatenate([self.sigs[key], sig])
        self.texts.setdefault(key, []).append(text)
        self.dirty += 1
        if time.time() - self.saved_at > 60 and not getattr(self, '_saving', False):
            self.saved_at = time.time()
            self._saving = True
            import threading
            threading.Thread(target=self.save, daemon=True).start()   # never block the game loop

    def save(self):
        """Write the memo (uncompressed: compressing a large memo took ~5s)."""
        try:
            if not self.dirty:
                return
            arrays = {}
            for k in list(self.stacks):
                img, txt = self.stacks[k], list(self.texts[k])      # snapshot: OCR adds while we save
                n = min(len(img), len(txt))
                arrays[k + '|img'] = img[:n]
                arrays[k + '|txt'] = np.array(txt[:n])
            tmp = self.path.with_suffix(f'.{os.getpid()}.tmp.npz')
            np.savez(tmp, **arrays)
            tmp.replace(self.path)
            self.dirty = 0
        except (OSError, RuntimeError, ValueError):
            pass                    # best effort: a failed save must never cost a game tick
        finally:
            self.saved_at = time.time()
            self._saving = False


def _tune_onnx():
    """Every OCR model (each reader has its own) took all 16 CPU threads and busy-waited between runs: with the game,
    the video encoder and readers in other threads, a 30 ms read took 2-8 s (run 20261015-161625: 1,027 s of slow
    loop steps). 4 threads and no busy-waiting: p50 188 / p95 207 ms under load, against 479 / 1,968 ms (offline
    measure, 2 concurrent detections)."""
    from rapidocr.inference_engine.onnxruntime import main as M
    if getattr(M.OrtInferSession, '_tft_tuned', False):
        return
    orig = M.OrtInferSession._init_sess_opts

    def tuned(cfg):
        o = orig(cfg)
        o.intra_op_num_threads, o.inter_op_num_threads = 4, 1
        o.add_session_config_entry('session.intra_op.allow_spinning', '0')
        return o
    M.OrtInferSession._init_sess_opts = staticmethod(tuned)
    M.OrtInferSession._tft_tuned = True


class ScreenReader:
    def __init__(self):
        _tune_onnx()
        from rapidocr import RapidOCR
        from rapidocr.ch_ppocr_rec.typings import TextRecInput
        self._engine = RapidOCR(params={'Global.log_level': 'error'})
        self._input = TextRecInput
        self._cache: dict[tuple, tuple[np.ndarray, str]] = {}
        self.memo = OcrMemo()
        self.unit_names = list(CATALOG.units)
        # current augments first: the catalogue misses many live ones (e.g. Patient Study)
        self.augment_names = list(dict.fromkeys([*META.augments, *CATALOG.augments]))

    # -- OCR with change cache -------------------------------------------------
    def _texts(self, frame: np.ndarray, boxes: list[tuple], keys=None) -> list[str]:
        """Text per box: unchanged crop -> cached; seen-before crop -> memo;
        otherwise one batched OCR call (memoised when confident)."""
        out: list[str | None] = []
        todo = []
        h = frame.shape[0]
        for j, box in enumerate(boxes):
            crop = frame[box[1]:box[3], box[0]:box[2]]
            fp = crop.astype(np.int16)
            hit = self._cache.get(box)
            # a one-digit change touches only a few % of a crop: compare pixels, not means
            if hit is not None and hit[0].shape == fp.shape and \
                    (np.abs(hit[0] - fp).max(axis=2) > 40).mean() < 0.004:
                out.append(hit[1])
                continue
            key = f'{keys[j]}@{h}' if keys else None
            txt = self.memo.get(key, crop) if key else None
            if txt is not None:
                out.append(txt)
                self._cache[box] = (fp, txt)
                continue
            out.append(None)
            todo.append((len(out) - 1, box, crop, fp, key))
        if todo:
            res = self._engine.text_rec(self._input(img=[np.ascontiguousarray(t[2]) for t in todo]))
            for (i, box, crop, fp, key), txt, sc in zip(todo, res.txts, res.scores):
                txt = txt.strip()
                out[i] = txt
                self._cache[box] = (fp, txt)
                if key and (sc >= 0.9 or not txt):     # OCR is deterministic: same pixels, same text
                    self.memo.add(key, crop, txt)
        return out

    def read_shop(self, frame):
        """Only what a roll-down needs: (gold, shop names, raw texts) - one batched OCR call on
        six small crops, memoised (a full read() is ~0.35 s; this is the inner loop of rolling)."""
        h, w = frame.shape[:2]
        s = h / 1080
        S = lambda b: (round(w / 2 + (b[0] - 960) * s), round(b[1] * s),
                       round(w / 2 + (b[2] - 960) * s), round(b[3] * s))
        gold, *names = self._texts(frame, [S(GOLD), *[S(shop_name_box(i)) for i in range(5)]],
                                   ['gold', *[f'shop{i}' for i in range(5)]])
        g = int(gold) if re.fullmatch(r'\d{1,3}', gold) else None
        return g, [match_name(n, self.unit_names) for n in names], [n.strip() for n in names]

    def shop_sig(self, frame):
        """The shop cards' name strips, coarse pixels: changes when the shop rolls, settles when the
        new cards are drawn (no OCR - for waiting on a roll)."""
        h, w = frame.shape[:2]
        s = h / 1080
        out = []
        for i in range(5):
            b = shop_name_box(i)
            x0, y0, x1, y1 = (round(w / 2 + (b[0] - 960) * s), round(b[1] * s),
                              round(w / 2 + (b[2] - 960) * s), round(b[3] * s))
            out.append(frame[y0:y1:3, x0:x1:3].astype(np.int16))
        return out

    def read(self, frame) -> Screen:
        """frame: RGB numpy array (H, W, 3) or PIL image."""
        start = time.perf_counter()
        if not isinstance(frame, np.ndarray):
            frame = np.asarray(frame.convert('RGB'))
        h, w = frame.shape[:2]
        s = h / 1080

        def S(b):
            return (round(w / 2 + (b[0] - 960) * s), round(b[1] * s),
                    round(w / 2 + (b[2] - 960) * s), round(b[3] * s))

        scr = Screen(t=time.time())
        level, stage, timer = self._texts(frame, [S(LEVEL), S(STAGE[0]), S(TIMER[0])],
                                          ['level', 'stage', 'timer'])
        m = re.search(r'l\w{0,2}\W*(\d{1,2})', level, re.I)
        if m and 1 <= int(m.group(1)) <= 10:
            scr.level = int(m.group(1))
            scr.shop_visible = True

        if not re.search(r'[1-9]\s*-\s*[1-9]', stage):       # stage-1 bar is narrower
            stage, timer = self._texts(frame, [S(STAGE[1]), S(TIMER[1])], ['stage1', 'timer1'])
        m = re.search(r'([1-9])\s*-\s*([1-9])', stage)
        if m:
            scr.stage = f'{m.group(1)}-{m.group(2)}'
            m = re.search(r'\d{1,2}', timer)
            scr.timer = int(m.group()) if m else None

        if scr.shop_visible:
            # HUD open: xp, gold and the five shop names (one batched call at most)
            xp, gold, *names = self._texts(
                frame, [S(XP), S(GOLD), *[S(shop_name_box(i)) for i in range(5)]],
                ['xp', 'gold', *[f'shop{i}' for i in range(5)]])
            m = re.fullmatch(r'(\d{1,2})\D(\d{1,2})', xp.replace(' ', ''))
            if m:
                scr.xp = (int(m.group(1)), int(m.group(2)))
            if re.fullmatch(r'\d{1,3}', gold):
                scr.gold = int(gold)
            scr.shop = [match_name(n, self.unit_names) for n in names]
            # free rerolls (an encounter, Invested++): the Reroll button costs 0 and its badge counts them - the
            # model saw no count and spent them early (run 20261010-235723). A fixed HUD crop: cached, no OCR
            # unless it changes
            cost, badge = self._texts(frame, [S(REROLL_COST), S(REROLL_FREE)], ['rrcost', 'rrbadge'])
            cost, badge = cost.strip(), badge.strip()
            if cost == '0' and badge.isdigit():
                scr.free_rolls = int(badge)
            elif re.fullmatch(r'\d', cost) and cost != '0':
                scr.free_rolls = 0
            scr.shop_raw = [n.strip() for n in names]
            if scr.level:
                # the big "n/level" over the board: units moving behind it change its pixels every tick, so its
                # OCR ran on almost every read - 9.8% of the game loop's time in run 20261010-235723's profile.
                # Its only user (the roster check) acts on a difference that lasts 1.5 s: once a second is enough
                last = getattr(self, '_board_count', None)
                if last is None or time.time() - last[1] > 1.0 or last[2] != scr.level:
                    got = read_board_count(self._texts(frame, [S(BOARD_COUNT)], ['boardcnt'])[0], scr.level)
                    # the team size counts once two reads in a row agree (one OCR slip can't change it)
                    size = got[1] if got and last and last[3] == got[1] else None
                    last = (got[0] if got else None, time.time(), scr.level, got[1] if got else None, size)
                    self._board_count = last
                scr.board_count, scr.team_size = last[0], last[4]
        elif self._maybe_title(frame, S(AUGMENT_TITLE)):
            # HUD hidden: maybe the augment picker
            aug_title, = self._texts(frame, [S(AUGMENT_TITLE)], ['augtitle'])
            if re.search(r'choose', aug_title, re.I):
                # the card names are NOT read: the model reads them from the screenshot. An OCR
                # name that flickered ("Capital Gains I" -> "II") once made the fallback pick.
                scr.augments = AUGMENT_POSITIONS

        scr.hp = self._hp(frame, S)
        scr.ms = (time.perf_counter() - start) * 1000
        return scr

    def shop_cost(self, frame, i):
        """The gold cost printed on shop card i (Wisps have varying costs), or None."""
        h, w = frame.shape[:2]
        s = h / 1080
        x0 = SHOP_X0 + SHOP_STEP * i
        b = (x0 + 160, 1043, x0 + 196, 1072)
        box = (round(w / 2 + (b[0] - 960) * s), round(b[1] * s), round(w / 2 + (b[2] - 960) * s), round(b[3] * s))
        m = re.search(r'\d{1,2}', self._texts(frame, [box], [f'shopcost{i}'])[0] or '')
        return int(m.group()) if m else None

    def read_augments(self, frame):
        """The three augment cards' names (known names, None where unreadable). Read once the
        cards have finished flipping; used by the comp pilot (the model reads the screenshot)."""
        h, w = frame.shape[:2]
        s = h / 1080
        S = lambda b: (round(w / 2 + (b[0] - 960) * s), round(b[1] * s), round(w / 2 + (b[2] - 960) * s), round(b[3] * s))
        txts = self._texts(frame, [S(b) for b in AUGMENT_NAMES], None)
        return [match_name(t, self.augment_names, cutoff=0.7) for t in txts], txts

    def special_card(self, frame, i):
        """A Wisp in the shop (any non-champion card: Coin Flip, Bronze Spoon, ...): (description, cost) read
        from the card, cached by its pixels. The description is cream text on a dark card: its lines are
        found from the pixels and read recognition-only (~0.05-0.25 s; detection OCR took 1.3-1.7 s)."""
        h, w = frame.shape[:2]
        s = h / 1080
        x0 = SHOP_X0 + SHOP_STEP * i
        S = lambda b: (round(w / 2 + (b[0] - 960) * s), round(b[1] * s), round(w / 2 + (b[2] - 960) * s), round(b[3] * s))
        a = S((x0 + 6, 950, x0 + 192, 1040))
        crop = np.ascontiguousarray(frame[a[1]:a[3], a[0]:a[2]])
        key = crop[::6, ::6].tobytes()
        hit = getattr(self, '_cards', {}).get(key)
        if hit:
            return hit
        lines = self._card_lines(crop, s)
        desc = ' '.join(t.strip() for t in self._engine.text_rec(self._input(img=lines)).txts) if lines else ''
        desc = re.sub(r'(?<![A-Za-z])I(?![A-Za-z])', '1', desc)     # the card font draws 1 like I
        out = (desc.strip(), self.shop_cost(frame, i))
        self._cards = getattr(self, '_cards', {})
        self._cards[key] = out
        return out

    @staticmethod
    def _card_lines(crop, s):
        """The text lines of a Wisp card's description: runs of rows with cream pixels, each padded
        with the card's background, the rarity badge (drawn over the first line) painted out."""
        cream = crop.min(axis=2) > 140
        prof = cream.sum(axis=1)
        rows = prof >= max(3, round(4 * s))
        runs, k = [], 0
        while k < len(rows):
            if not rows[k]:
                k += 1
                continue
            j = k
            while j + 1 < len(rows) and (rows[j + 1] or (j + 2 < len(rows) and rows[j + 2])):
                j += 1
            if j - k + 1 >= round(6 * s) and prof[k:j + 1].max() >= round(8 * s):
                runs.append((k, j))
            k = j + 1
        if not runs or cream.all():
            return []
        bg = np.median(crop[~cream].reshape(-1, 3), axis=0).astype(crop.dtype)
        pad, out = round(3 * s), []
        for y0, y1 in runs:
            line = crop[max(0, y0 - pad):y1 + pad + 1].copy()
            line[(line.max(axis=2).astype(int) - line.min(axis=2)) > 90] = bg
            side = np.tile(bg, (line.shape[0], round(10 * s), 1))
            out.append(np.ascontiguousarray(np.hstack([side, line, side])))
        return out

    PANEL_BOX = (1690, 150, 1915, 420)      # portrait, stars, name, HP of the unit details panel

    def panel_sig(self, frame):
        """Pixels of the unit details panel, to tell when it shows a different unit."""
        h, w = frame.shape[:2]
        s = h / 1080
        b = self.PANEL_BOX
        x0, y0, x1, y1 = (round(w - (1920 - b[0]) * s), round(b[1] * s), round(w - (1920 - b[2]) * s), round(b[3] * s))
        return frame[y0:y1:3, x0:x1:3].astype(np.int16)

    TRAIT_ROWS = [274 + 53 * k for k in range(10)]      # the trait panel's rows (left edge, 1080p)
    TRAIT_SHIFT = 55          # the panel moves right when the item bench grows a second column (11+ items)

    def trait_shift(self, frame):
        """0, or TRAIT_SHIFT when the item bench has a second column (items.bench_columns). Without it,
        rows read at the one-column place gave 2-5 garbled traits on 11 of 137 recorded planning starts
        (221257 4-2..5-1, 152203 4-1..5-3)."""
        from tft_player.items import bench_columns
        return self.TRAIT_SHIFT if bench_columns(frame) == 2 else 0

    def trait_panel(self, frame):
        """The game's trait panel: {trait: units counted} for the rows shown (active traits
        first; a '4+' box hides the rest, all inactive). Left-anchored like the item bench."""
        h, w = frame.shape[:2]
        s = h / 1080
        dx = self.trait_shift(frame)
        B = lambda b: tuple(round(v * s) for v in b)
        names = self._texts(frame, [B((136 + dx, y - 13, 230 + dx, y + 11)) for y in self.TRAIT_ROWS], None)
        counts = self._texts(frame, [B((108 + dx, y - 13, 132 + dx, y + 12)) for y in self.TRAIT_ROWS], None)
        out = {}
        for n, c in zip(names, counts):
            t = match_name(n, list(CATALOG.traits), cutoff=0.7)
            m = re.fullmatch(r'\D{0,1}(\d)\D{0,1}', (c or '').strip())
            if t and m:
                out[t] = int(m.group(1))
        return out

    PANEL_NAME = (1685, 350, 1870, 377)
    PANEL_STARS = (1756, 188, 1844, 205)
    PANEL_HP = (1680, 382, 1905, 398)                 # the green health bar under the name

    def read_panel(self, frame):
        """Unit details panel (opened by right-clicking a unit): (name, stars) or (None, None)."""
        h, w = frame.shape[:2]
        s = h / 1080

        def S(b):   # the panel is anchored to the right edge
            return (round(w - (1920 - b[0]) * s), round(b[1] * s), round(w - (1920 - b[2]) * s), round(b[3] * s))
        txt, = self._texts(frame, [S(self.PANEL_NAME)], ['panelname'])
        self.last_panel_text = txt                     # kept raw: what an unexpected unit's panel says (data)
        x0, y0, x1, y1 = S(self.PANEL_HP)
        hp = frame[y0:y1, x0:x1].astype(np.int16)
        r, g, b = hp[..., 0], hp[..., 1], hp[..., 2]
        if hp.size == 0 or float(((g > 150) & (g - r > 50) & (g - b > 30)).mean()) < 0.25:
            # no unit panel is open (its green health bar sits under the name): text there is something else -
            # run 20261010-235723 4-7: an opponent's "lucky choice: Ornn" popup was read as an Ornn*** on our
            # board when a click on the BFF opened no panel. Unit panels measured 0.53-0.90, popups 0.0
            self.last_panel_text = f'(no unit panel) {txt}'
            return None, None
        if re.search(r'FURRY|\bBFF\b', (txt or '').upper()):
            # the Sprykin trait's Big Furry Friend (run 20261010-113319 4-2: its panel was matched to "Alistar")
            from tft_player.state import BFF
            return BFF, 1
        name = match_name(txt, self.unit_names)
        if not name:
            # not a champion: a board object (Training Dummy) is recorded by its own name, one star
            from tft_player.state import board_objects
            obj = match_name(txt, list(board_objects()))
            return (obj, 1) if obj else (None, None)
        x0, y0, x1, y1 = S(self.PANEL_STARS)
        strip = frame[y0:y1, x0:x1, 0] > 120          # bronze/silver/gold stars; the teal border has low red
        cols = strip.any(axis=0).astype(np.int8)
        runs = np.diff(np.concatenate(([0], cols, [0])))
        widths = np.where(runs == -1)[0] - np.where(runs == 1)[0]
        stars = int((widths >= round(6 * s)).sum()) or None
        return name, (stars if stars and stars <= 3 else None)

    SETTINGS_TITLE = (425, 145, 600, 185)

    SETTINGS_BUTTONS = (700, 898, 1420, 932)          # "EXIT TFT  SURRENDER  RESTORE DEFAULTS"
    SURRENDER_TITLE = (780, 418, 1140, 456)           # "SURRENDER GAME" confirmation dialog

    def _centred(self, frame, b):
        h, w = frame.shape[:2]
        s = h / 1080
        return (round(w / 2 + (b[0] - 960) * s), round(b[1] * s), round(w / 2 + (b[2] - 960) * s), round(b[3] * s))

    def settings_open(self, frame):
        """Is the in-game Settings menu open? Its button row (EXIT TFT / SURRENDER / RESTORE
        DEFAULTS), or its title. The title alone failed at 16:10: our MODEL THINKING panel
        covers it (run22: the surrender never saw the menu)."""
        title, buttons = self._texts(frame, [self._centred(frame, self.SETTINGS_TITLE),
                                             self._centred(frame, self.SETTINGS_BUTTONS)], ['settingstitle', None])
        buttons = buttons.upper()
        return 'SETTING' in title.upper() or 'RESTORE' in buttons or ('EXIT' in buttons and 'SURRENDER' in buttons)

    def surrender_dialog(self, frame):
        """Is the 'SURRENDER GAME' confirmation open?"""
        txt, = self._texts(frame, [self._centred(frame, self.SURRENDER_TITLE)], None)
        return 'SURRENDER' in txt.upper()

    # the game's encounter, shown in the 1-1 carousel as a centred dark banner under the stage bar,
    # one or two lines ("Lux begins your journey with 3 random emblems.", "It's a calm day ... No
    # encounter this game.")
    ENCOUNTER_BOX = (610, 74, 1310, 142)

    def encounter(self, frame):
        """The 1-1 encounter banner's text, or None while it is not shown (~2-5 s: run it off the
        tick loop). Only lines centred on the screen count: player names float beside it."""
        b = self._centred(frame, self.ENCOUNTER_BOX)
        crop = frame[b[1]:b[3], b[0]:b[2]]
        mid = crop.shape[1] / 2
        lines = sorted((y, t.strip()) for x, y, _, t in self._detect(crop, 'encounter')
                       if abs(x - mid) < crop.shape[1] * 0.2 and t.strip())
        text = ' '.join(t for _, t in lines)
        return text if len(text.split()) >= 4 and text.endswith('.') else None

    def find_text(self, frame, width=1280):
        """Full-frame text detection + recognition (~2s; background use only).
        Returns [(text, (x, y) centre in frame pixels)]."""
        h, w = frame.shape[:2]
        k = max(1, w // width)
        small = np.ascontiguousarray(frame[::k, ::k])
        res = self._engine(small, use_cls=False)
        if res.boxes is None:
            return []
        return [(t, (float(b[:, 0].mean() * k), float(b[:, 1].mean() * k))) for t, b in zip(res.txts, res.boxes)]

    @staticmethod
    def _maybe_title(frame, box):
        """Cheap check before OCR: the picker title is large cream text."""
        c = frame[box[1]:box[3], box[0]:box[2]]
        bright = (c[..., 0] > 200) & (c[..., 1] > 190) & (c[..., 2] > 160)
        return 0.04 < bright.mean() < 0.5

    # ---- more of what the screen shows, read when the model is asked (not every tick) ----
    PLAYERS_BOX = (1560, 150, 1920, 830)     # the player list, right-anchored (8 rows)
    # the game's Damage Dealt panel slides over the player list (its sword button, top right); the people button
    # under the sword brings the list back (user, run 20261014-032220: open from 1-3 on, the 2-4 carousel misread
    # our row as the top one and waited 19 s at an open gate)
    DAMAGE_BOX = (1730, 266, 1880, 300)      # "<name> / Damage Dealt" under our portrait, right-anchored
    STANDINGS_BUTTON = (1898, 146, 'R')

    def damage_panel(self, frame):
        """Is the Damage Dealt panel covering the player list?"""
        h, w = frame.shape[:2]
        s = h / 1080
        b = self.DAMAGE_BOX
        txt, = self._texts(frame, [(round(w - (1920 - b[0]) * s), round(b[1] * s), round(w - (1920 - b[2]) * s),
                                    round(b[3] * s))], ['damagepanel'])
        return 'DAMAGE' in (txt or '').upper()
    STREAK_NUM = (1135, 876, 1180, 908)      # the number beside the streak flame
    STREAK_ICON = (1108, 878, 1134, 906)     # the flame: orange/red = win streak, blue = loss streak
    ODDS_BOX = (545, 878, 850, 910)          # shop odds per cost, above the shop

    def _detect(self, crop, key):
        """Detection + recognition on one region, cached by its pixels."""
        sig = crop[::8, ::8].tobytes()
        cache = self.__dict__.setdefault('_det_cache', {})
        if cache.get(key, (None,))[0] == sig:
            return cache[key][1]
        r = self._engine(np.ascontiguousarray(crop), use_cls=False)
        out = [] if r.boxes is None else [((b[0][0] + b[2][0]) / 2, (b[0][1] + b[2][1]) / 2, b[2][1] - b[0][1], t)
                                          for b, t in zip(r.boxes, r.txts)]
        cache[key] = (sig, out)
        return out

    def players(self, frame, our_hp=None):
        """The player list top to bottom (= standing): [(name or None, hp or None)].
        Our row is the big HP number without a name."""
        h, w = frame.shape[:2]
        s = h / 1080
        b = self.PLAYERS_BOX
        x0, y0, x1, y1 = round(w - (1920 - b[0]) * s), round(b[1] * s), w, round(b[3] * s)
        items = self._detect(frame[y0:y1, x0:x1], 'players')
        rows = []
        for cx, cy, ht, t in sorted(items, key=lambda i: i[1]):
            t = t.strip()
            if not t or t.upper() in ('X', 'LEVEL', 'LEVEL5!') or t.lower().startswith('level'):
                continue
            row = next((r for r in rows if abs(r['y'] - cy) < 22 * s), None)
            if row is None:
                row = {'y': cy, 'name': None, 'hp': None, 'big': False}
                rows.append(row)
            if re.fullmatch(r'\d{1,3}', t) and int(t) <= 100:
                row['hp'] = int(t)
                row['big'] = row['big'] or ht > 26 * s
            elif re.search(r'[A-Za-z]', t) and len(t) > len(row['name'] or ''):
                row['name'] = t              # a name split in two pieces: keep the longest
        # the small HP box right of a name the list read missed: recognition only, all in
        # one batch (a detection call per box cost ~1.4 s each, 6-9 s per plan prompt)
        miss = [r for r in rows if r['hp'] is None and r['name']]
        boxes = [(round(w - (1920 - 1818) * s), round(y0 + r['y'] - 13 * s),
                  round(w - (1920 - 1866) * s), round(y0 + r['y'] + 13 * s)) for r in miss]
        for r, t in zip(miss, self._texts(frame, boxes) if boxes else []):
            m = re.fullmatch(r'\D*(\d{1,3})\D*', t or '')
            if m and int(m.group(1)) <= 100:
                r['hp'] = int(m.group(1))
        out = []
        for r in rows:
            if r['name'] is None and r['hp'] is None:
                continue
            you = r['big'] or (r['name'] is None and our_hp is not None and r['hp'] == our_hp)
            out.append(('YOU' if you else r['name'], r['hp']))
        return out

    ITEM_NAMES = None

    @classmethod
    def item_names(cls):
        """Every item name in the game data (components, completed, emblems, consumables...)."""
        if cls.ITEM_NAMES is None:
            import json as _json
            p = sorted((Path(__file__).resolve().parent.parent / 'data' / 'sources').glob('cdragon-tft-*.json'))[-1]   # newest
            from tft_player.catalog import CATALOG
            from tft_player.items import COMPONENTS, RECIPES
            names = set(COMPONENTS) | set(RECIPES)
            traits = set(CATALOG.traits)
            try:
                # this Set's items only: standard items, Set-18 items and every emblem - never a
                # bare trait name (the trait panel sits right beside the item lane)
                for it in _json.loads(p.read_text(encoding='utf-8'))['items']:
                    n, api = it.get('name') or '', it.get('apiName') or ''
                    n = re.sub(r'\s*<.*$', '', n)          # "Reforger <rules>(3 uses left!)</rules>"
                    if '_' in n:
                        continue                               # an untranslated key, not a name
                    if len(n) > 2 and not n.startswith('@') and n not in traits and (
                            n.endswith('Emblem') or api.startswith('TFT_Item_') or 'TFT18' in api
                            or (api.startswith('DA_') and 'Item' in api) or '_Consumable_' in api
                            # this Set's artifacts (run 20261016-052413 4-2: Hullcrusher, DA_Artifact_Hullcrusher, was
                            # missing, so the 4-card choice row fell to text detection and blocked the loop 9.4 s)
                            or api.startswith('DA_Artifact_')
                            or api.startswith('DA_Consumable') or api == 'DA_Reforger'
                            # Radiant items (TFT5_Item_*Radiant: run 221257, a Radiant Giant Slayer's
                            # tooltip was read 4 times and matched nothing - "unknown" for 5 rounds)
                            or (n.startswith('Radiant ') and api.endswith('Radiant'))
                            # this Set's own items (run27: the Knick-Knack reward "Attack Speed
                            # Booster", DA_SpeedBooster18, stayed unknown all game - its tooltip was
                            # read, but the name was not in this list)
                            or re.match(r'DA_.*18', api)):
                        names.add(n)
            except Exception:
                pass
            from tft_player.items import MISSING_ITEMS
            names |= set(MISSING_ITEMS)              # this set's items the game data lacks (the Alpha Mark)
            cls.ITEM_NAMES = sorted(names)
        return cls.ITEM_NAMES

    def item_tooltip(self, frame, slot_xy, before=None, title_only=False):
        """The name in the tooltip shown while an item-bench slot is hovered, or None.
        slot_xy: the slot's point in left-anchored 1080p coordinates. before: the frame just before the hover - the
        tooltip is then the part that changed, and nothing else (the trait panel beside the item bench is never
        read as an item's tooltip); nothing changed = no tooltip yet, read again later."""
        h, w = frame.shape[:2]
        s = h / 1080
        x, y = slot_xy
        # the title sits on a fixed line beside the hovered slot (run27 frames: 110-470 px right of
        # the slot point, 30 px above to 4 below it): recognition on that line alone takes tens
        # of ms, where detection over the whole tooltip took 7-14 s under load
        line = (max(0, round((x + 110) * s)), max(0, round((y - 30) * s)),
                min(w, round((x + 470) * s)), min(h, round((y + 4) * s)))
        # never the cached text: a remover's count is one digit on a long line, under the cache's change threshold
        # (run 20261014-063145: "2 uses left" read again at 3-1 and 4-1 while the slot showed 5 and 6)
        self._cache.pop(line, None)
        if before is not None and before.shape == frame.shape:
            # the line must have changed since before the hover: text that stood there already is not a tooltip (run
            # 20261015-115044 3-2: an item slot with no tooltip read our tactician's name label as its item)
            x0, y0, x1, y1 = line
            if (np.abs(frame[y0:y1, x0:x1].astype(np.int16) - before[y0:y1, x0:x1].astype(np.int16)).max(axis=2)
                    > 40).mean() < 0.05:
                self.tooltip_text = ''
                return None
        t = self._texts(frame, [line], None)[0] or ''
        self.tooltip_text = t.strip()            # what the title line says, also when no name matches
        name = match_name(re.sub(r'\s*\(.*$', '', t), self.item_names(), 0.82)
        if name:
            uses = re.search(r'\((\d+)\s*uses?', t)
            if uses and ('Remover' in name or 'Reforger' in name):
                name = f'{name} ({uses.group(1)} uses left)'
            return name
        if title_only:
            return None                          # the action path: no 7-9 s detection over the whole tooltip
        x0, x1 = max(0, round((x + 22) * s)), min(w, round((x + 520) * s))
        y0, y1 = max(0, round((y - 160) * s)), min(h, round((y + 220) * s))
        if before is not None and before.shape == frame.shape:
            diff = (np.abs(frame[y0:y1, x0:x1].astype(np.int16) - before[y0:y1, x0:x1].astype(np.int16)).max(axis=2) > 40)
            rows, cols = np.where(diff.any(axis=1))[0], np.where(diff.any(axis=0))[0]
            if diff.mean() < 0.02 or not len(rows):
                self.tooltip_text = ''
                return None                       # no tooltip appeared: nothing to read (tried again later)
            # the tooltip's box: the rows and columns that changed
            x0, x1, y0, y1 = x0 + cols[0], x0 + cols[-1] + 1, y0 + rows[0], y0 + rows[-1] + 1
        items = self._detect(frame[y0:y1, x0:x1], f'tooltip{x},{y}')
        traits = {t.lower() for t in CATALOG.traits}
        # trait names and "2 > 4 > 6" thresholds are the trait panel's, never an item's
        items = [it for it in items if it[3].strip().lower() not in traits and not re.fullmatch(r'[\d\s>/]+', it[3].strip())]
        best = None
        for _, cy, ht, t in sorted(items, key=lambda i: i[1]):
            # consumables carry a count on the title line: "Item Remover (4 uses left!)"
            name = match_name(re.sub(r'\s*\(.*$', '', t), self.item_names(), 0.82)
            uses = re.search(r'\((\d+)\s*uses?', t)
            if name and uses and ('Remover' in name or 'Reforger' in name):
                name = f'{name} ({uses.group(1)} uses left)'
            if name and (best is None or ht > best[1]):
                best = (name, ht)                 # the name is the biggest matching text
        if not best and items:
            self.tooltip_text = ' / '.join(t for *_, t in sorted(items, key=lambda i: i[1])[:3])
        return best[0] if best else None

    def streak(self, frame):
        """(n, 'win' | 'loss') from the flame beside the gold, or None."""
        h, w = frame.shape[:2]
        s = h / 1080
        S = lambda b: (round(w / 2 + (b[0] - 960) * s), round(b[1] * s), round(w / 2 + (b[2] - 960) * s), round(b[3] * s))
        txt = self._texts(frame, [S(self.STREAK_NUM)], ['streak'])[0]
        m = re.search(r'\d{1,2}', txt)
        if not m:
            return None
        x0, y0, x1, y1 = S(self.STREAK_ICON)
        icon = frame[y0:y1, x0:x1].reshape(-1, 3).astype(np.int16)
        bright = icon[icon.max(axis=1) > 120]
        if not len(bright):
            return int(m.group()), '?'
        r, g, b = bright.mean(axis=0)
        return int(m.group()), 'loss' if b > r else 'win'

    def shop_odds(self, frame):
        """Shop odds per cost 1..5 as printed above the shop, e.g. [45, 33, 20, 2, 0]."""
        h, w = frame.shape[:2]
        s = h / 1080
        b = self.ODDS_BOX
        box = (round(w / 2 + (b[0] - 960) * s), round(b[1] * s), round(w / 2 + (b[2] - 960) * s), round(b[3] * s))
        text = self._texts(frame, [box], ['odds'])[0]
        vals = [int(v) for v in re.findall(r'(\d{1,3})\s*%', text)]
        if len(vals) == 5:
            return ' / '.join(f'{v}%' for v in vals)
        return f'{text.strip()} (read incompletely)' if vals else None

    # where the gold border lines of our big HP box start (1080p x, right-anchored): 1782 (3 digits) - 1795 (1-2)
    # at 2560x1600 and 2560x1440 alike. Other players' small boxes start at 1822-1823; lines of the scenery behind
    # the list run past the search's left edge (1760). The earlier test - "no username text at x 1826-1850" - held
    # at 16:10 only: at 2560x1440 our own box reaches into that place, every read was refused and the model played
    # from 2-5 on with 90, then a stray 1 (real 76 -> 2; release run 20261016-141957)
    HP_BOX_START = (1772, 1808)

    def _hp(self, frame, S):
        """Our HP: the number in our own entry of the player list, the only one drawn as a big box with long gold
        border lines (top and bottom, ~37px at 1080p; other players' boxes are ~25px), found by those lines and
        confirmed by where they start (HP_BOX_START: run 010416 2-6 took a '7' for our 84 HP off another entry).
        The old rule - the tallest gold ring - read digits off a name ("RainDude87" -> 87 HP while we had 42)."""
        h, w = frame.shape[:2]
        s = h / 1080
        # the unit info panel (open after a unit read) covers the player list: its gold edges and
        # numbers were read as our HP (run32 3-3: "2" while we had 53). Its wide green bar marks it.
        bar = frame[round(382 * s):round(396 * s), round(w - (1920 - 1690) * s):round(w - (1920 - 1890) * s)].astype(np.int16)
        if bar.size and float(((bar[..., 1] > 150) & (bar[..., 0] < 110) & (bar[..., 2] < 110)).mean()) > 0.5:
            return None
        x0, x1 = round(w - (1920 - 1760) * s), round(w - (1920 - 1860) * s)
        y0, y1 = round(150 * s), round(0.85 * h)
        col = frame[y0:y1, x0:x1].astype(np.int16)
        r, g, b = col[..., 0], col[..., 1], col[..., 2]
        gold = (r > 120) & (g > 90) & (b < 110) & (r - b > 50)
        need = 30 * s
        lines, starts = [], {}
        for y in np.where(gold.sum(axis=1) >= need)[0]:
            d = np.diff(np.concatenate(([0], gold[y].view(np.int8), [0])))
            run_s, run_e = np.where(d == 1)[0], np.where(d == -1)[0]
            k = int((run_e - run_s).argmax())
            if run_e[k] - run_s[k] >= need:
                lines.append(y)
                starts[y] = run_s[k]
        if not lines:
            return None
        groups = np.split(np.array(lines), np.where(np.diff(lines) > 2)[0] + 1)
        ys = [(int(gp.mean()), min(starts[y] for y in gp)) for gp in groups]
        lo, hi = self.HP_BOX_START
        start = lambda a, b: 1920 - (w - x0 - min(a[1], b[1])) / s
        pair = next(((a[0], b[0]) for i, a in enumerate(ys) for b in ys[i + 1:]
                     if 28 * s <= b[0] - a[0] <= 40 * s and lo <= start(a, b) <= hi), None)
        if pair is None:
            return None                                   # no box of ours (another entry's, or the scenery)
        yc, half = y0 + (pair[0] + pair[1]) // 2, round(13 * s)
        # fixed size (memo hits) and quantised (the change cache holds when a line jitters by a pixel)
        box = (round(w - (1920 - HP_DIGITS_X[0]) * s), (yc - half) // 2 * 2,
               round(w - (1920 - HP_DIGITS_X[1]) * s), (yc - half) // 2 * 2 + 2 * half)
        txt = self._texts(frame, [box], ['hpbox'])[0]
        m = re.fullmatch(r'\D{0,2}(\d{1,3})\D{0,2}', (txt or '').strip())     # '4.' happens
        return int(m.group(1)) if m and 0 < int(m.group(1)) <= 100 else None
