"""On-screen overlay drawn over the TFT window, for people watching.

Four click-through panels anchored to corners, one job each:
  top-left      AI DECISION  - what the model is thinking (a spinner, the seconds so far and its reasoning as
                               it writes it) and then what it decided, as coloured tags with its own words;
                               everything on one page (long rows are shortened to fit; no turning pages), kept
                               until newer content replaces it. The carousel's priority list shows here too.
  top-right     TARGET COMP  - the comp as portraits: owned lit, missing greyed, the comp's star goal under
                               each (gold once reached), planned items (hidden during the carousel: it sits
                               inside the carousel's icon search)
  bottom-left   EXECUTION    - the plan as a list of steps: done / running / failed / stopped by a guard
  bottom-right  STRATEGY     - the model's own strategy for the game: one paragraph, written while we queue and
                               changed by it only when its strategy changes (with the reason under it); in flex
                               mode its comp probabilities until it commits to a comp
Each panel has a small button that collapses it to a tab (its own clickable window; the panels themselves
let every click through, WS_EX_TRANSPARENT, and never take focus, WS_EX_NOACTIVATE). They sit outside every
region perception reads. A changed row tints for about a second, then fades.

`draw_panel` renders one panel on a Tk canvas; the live overlay and `preview()` (a still over a recorded
frame) use the same code.
"""
from __future__ import annotations

import ctypes
import math
import queue
import re
import threading
import time
from pathlib import Path

GWL_EXSTYLE = -20
WS_EX_LAYERED, WS_EX_TRANSPARENT = 0x00080000, 0x00000020
WS_EX_TOOLWINDOW, WS_EX_NOACTIVATE, WS_EX_TOPMOST = 0x00000080, 0x08000000, 0x00000008

# panel boxes in 1080p reference coordinates (x centred on 960, following the centred HUD), clamped to
# the window. Clear of: stage tracker (x 735-1194, top), top-right buttons, the player list and unit
# panel (x >= 1560 below y 150), the trait tracker (left, y >= 234), the bench, shop and XP buttons,
# and a 5-option choice row (x 318-1616: run27's anvils were missed under 20-50 px of panel).
PANELS = {
    # the original size: clear of the enemy board (the user, 2026-10-08)
    'top_left': (-600, 6, 560, 240),
    # the whole top-right corner above the player list / unit panel (both read from y 150 down)
    'top_right': (1200, 4, 2600, 146),
    'bottom_left': (-600, 800, 342, 1080),
    # right of the shop (its last card ends at x 1556) and under the player list (ends ~y 790); loot orbs
    # (loot.ARENA) and the carousel (carousel.REGION) stay left of x 1560
    'bottom_right': (1575, 805, 2600, 1080),
}
# the keys, left of the stage bar at the top (the user, 2026-10-14): F7 pause / continue, F8 stop; amber while paused
# (game.PAUSED). Clear of the AI DECISION panel (x < 560) and the stage bar (from x ~735; ~770 in stage 1)
KEYS_BOX = (568, 6, 730, 34)
TITLES = {'top_left': 'AI DECISION', 'top_right': 'TARGET COMP', 'bottom_left': 'EXECUTION',
          'bottom_right': 'STRATEGY'}
BUTTON = 22                   # the collapse button, 1080p px (a small clickable square on each panel)

KEY = '#010203'            # transparent (window corners)
BG = '#0e1621'
ROW = '#16212e'            # a row's resting background
TINT = '#2b3b52'           # a changed row, fading to ROW
BORDER = '#2a3646'
TEXT = '#e6e9ef'
MUTED = '#8a94a6'
GOLD = '#c8aa6e'
CYAN = '#4fc3f7'
GREEN = '#5bd17a'
AMBER = '#f2b33d'
RED = '#e5484d'
COST = {1: '#9aa4af', 2: '#11b288', 3: '#207ac7', 4: '#c440da', 5: '#ffb93b'}
TAG_COLORS = {'SELL': '#e5484d', 'BUY': '#5bd17a', 'LEVEL': '#f2b33d', 'BOARD': '#7fb3f0',
              'ITEMS': '#e59a5b', 'OPEN': '#c8aa6e', 'NEXT': '#8a94a6', 'ECON': '#c8aa6e', 'WISP': '#b98cf0',
              'NOTE': '#8a94a6', 'NOW': '#4fc3f7', 'AUGMENT': '#c8aa6e', 'CHOICE': '#4fc3f7', 'ANVIL': '#4fc3f7',
              'CAROUSEL': '#c8aa6e',
              # execution steps: one colour per kind of action
              'XP': '#f2b33d', 'MOVE': '#7fb3f0', 'ITEM': '#e59a5b', 'ROLL': '#4fc3f7'}
STEP = {'done': ('✓', GREEN), 'run': ('→', CYAN), 'fail': ('✗', RED), 'guard': ('■', AMBER), 'wait': ('○', MUTED),
        'next': ('»', MUTED)}
ALPHA = 0.9
ROOT = Path(__file__).resolve().parent.parent


def _round_rect(c, x0, y0, x1, y1, r, **kw):
    pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1, x1 - r, y1,
           x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
    return c.create_polygon(pts, smooth=True, **kw)


def _mix(a, b, t):
    """Colour between a (t=0) and b (t=1)."""
    pa, pb = [int(a[i:i + 2], 16) for i in (1, 3, 5)], [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(x + (y - x) * t):02x}' for x, y in zip(pa, pb))


_FONTS = {}


def _font(c, px, weight='normal'):
    """Pixel-sized fonts (negative size): the same look at any Windows display scaling. Segoe UI's
    digits are fixed-width, so gold and timers don't jitter."""
    import tkinter.font as tkfont
    key = (id(c.tk), round(px), weight)          # per Tk interpreter: a font dies with its root
    if key not in _FONTS:
        _FONTS[key] = tkfont.Font(root=c, family='Segoe UI', size=-max(7, round(px)), weight=weight)
    return _FONTS[key]


def draw_keys(c, w, h, s, paused):
    """The keys box: 'F7 pause · F8 stop', or amber 'PAUSED · F7 resume · F8 stop' while paused."""
    c.delete('all')
    c.create_rectangle(0, 0, w - 1, h - 1, fill='#3a2a08' if paused else BG, outline=AMBER if paused else BORDER,
                       width=max(1, round(1.5 * s)) if paused else 1)
    parts = ([('PAUSED', AMBER, 'bold'), ('  F7', GOLD, 'bold'), (' resume', TEXT, 'normal'), ('  F8', GOLD, 'bold'),
              (' stop', TEXT, 'normal')] if paused else
             [('F7', GOLD, 'bold'), (' pause', TEXT, 'normal'), ('   F8', GOLD, 'bold'), (' stop', TEXT, 'normal')])
    for px in (14, 13, 12, 11, 10):              # the largest font that fits in one line
        fonts = [_font(c, px * s, wt) for _, _, wt in parts]
        width = sum(f.measure(t) for (t, _, _), f in zip(parts, fonts))
        if width <= w - 10 * s:
            break
    x = (w - width) / 2
    for (t, col, _), f in zip(parts, fonts):
        c.create_text(x, h / 2, text=t, anchor='w', fill=col, font=f)
        x += f.measure(t)


def _fit(font, text, width):
    """One line: cut with an ellipsis to fit width."""
    text = str(text)
    if font.measure(text) <= width:
        return text
    while text and font.measure(text + '…') > width:
        text = text[:-1]
    return text.rstrip() + '…'


_IMGS = {}

# texts too long for their place do not move (the user, 2026-10-12: "it should not roll"; 2026-10-13: the
# EXECUTION panel "still rolling < bad"): one sentence at a time, held still for a few seconds, then the next,
# and round again
_CYC = {}          # canvas -> [[item, pieces, key]]; emptied when the canvas is redrawn
_CYC_T0 = {}       # (canvas, text) -> when it was first shown: a redraw (the spinner turns) does not restart it
CYC_HOLD = 4.0     # seconds each piece is shown


def _pieces(text, fits):
    """text in pieces that each fit: its sentences (short ones together), a sentence too long cut at words."""
    out = []
    for snt in [t for t in re.split(r'(?<=[.!?;])\s+|\s+—\s+', str(text).strip()) if t]:
        if fits(snt):
            if out and fits(out[-1] + ' ' + snt):
                out[-1] += ' ' + snt
            else:
                out.append(snt)
            continue
        cur = ''
        for wd in snt.split():
            if cur and not fits(f'{cur} {wd}…'):
                out.append(cur + '…')
                cur = wd
            else:
                cur = f'{cur} {wd}'.strip()
        if cur:
            out.append(cur)
    return out or [str(text)]


def _cycle(c, x, y, text, font, width, lines=1, **kw):
    """Text in its place, never moving: whole if it fits in `lines` lines, else its pieces in turn (_pieces)."""
    text = str(text)
    if lines <= 1:
        fits = lambda t: font.measure(t) <= width
    else:
        limit = lines * font.metrics('linespace') + 1

        def fits(t):
            probe = c.create_text(x, y, text=t, font=font, width=width, anchor='nw')
            bb = c.bbox(probe)
            c.delete(probe)
            return bb[3] - bb[1] <= limit
    wrap = {'width': width} if lines > 1 else {}
    if fits(text):
        return c.create_text(x, y, text=text, font=font, **wrap, **kw)
    pieces = _pieces(text, fits)
    key = (id(c), text)
    t0 = _CYC_T0.setdefault(key, time.time())
    if len(_CYC_T0) > 400:                   # old texts are forgotten
        for k in list(_CYC_T0)[:200]:
            _CYC_T0.pop(k, None)
    item = c.create_text(x, y, text=pieces[int((time.time() - t0) / CYC_HOLD) % len(pieces)], font=font,
                         **wrap, **kw)
    _CYC.setdefault(c, []).append([item, pieces, key])
    return item


def _cycle_tick(c, now):
    for item, pieces, key in _CYC.get(c, ()):
        try:
            c.itemconfigure(item, text=pieces[int((now - _CYC_T0.get(key, now)) / CYC_HOLD) % len(pieces)])
        except Exception:
            pass


def _key(name):
    return ''.join(ch for ch in str(name) if ch.isalnum())


def _image(c, kind, name, px, grey=False):
    """A champion portrait or item icon as a Tk image (cached), or None when there is no picture."""
    k = (id(c.tk), kind, name, round(px), grey)
    if k in _IMGS:
        return _IMGS[k]
    path = (ROOT / 'data' / 'champ-icons' / f'{name}.png' if kind == 'champ'
            else ROOT / 'data' / 'item-icons-overlay' / f'{_key(name)}.png')
    img = None
    if path.exists():
        try:
            from PIL import Image, ImageEnhance, ImageTk
            im = Image.open(path).convert('RGB').resize((max(4, round(px)), max(4, round(px))), Image.LANCZOS)
            if grey:
                im = ImageEnhance.Brightness(im.convert('L').convert('RGB')).enhance(0.45)
            img = ImageTk.PhotoImage(im, master=c)
        except Exception:
            img = None
    _IMGS[k] = img
    return img


def _stars(c, cx, y, n, s, color=GOLD):
    """n small gold stars centred on cx."""
    if not n:
        return
    c.create_text(cx, y, text='★' * n, fill=color, font=_font(c, 9 * s, 'bold'), anchor='n')


def _champ_tile(c, x, y, size, name, cost, s, star=0, grey=False, flag=None, bar=None):
    """A portrait with a cost-coloured border; stars under it, or a segmented copy bar (have, need);
    flag '?' (unidentified) or '!' (doubtful) in the corner."""
    bw = max(1, round(2 * s))
    c.create_rectangle(x, y, x + size, y + size, fill=BG, outline=COST.get(cost, BORDER) if not grey else '#3a4352',
                       width=bw)
    img = _image(c, 'champ', name, size - 2 * bw, grey) if name else None
    if img is not None:
        c.create_image(x + bw, y + bw, image=img, anchor='nw')
    elif name is None:
        c.create_text(x + size / 2, y + size / 2, text='?', fill=MUTED, font=_font(c, size * 0.55, 'bold'))
    else:
        c.create_text(x + size / 2, y + size / 2, text=name[:2], fill=MUTED, font=_font(c, size * 0.4, 'bold'))
    if bar:
        have, need = bar
        seg, gap = (size - (need - 1) * 2 * s) / max(1, need), 2 * s
        for k in range(need):
            sx = x + k * (seg + gap)
            c.create_rectangle(sx, y + size + 2 * s, sx + seg, y + size + 5 * s,
                               fill=GOLD if k < have else '#3a4352', width=0)
    elif star:
        _stars(c, x + size / 2, y + size + 1 * s, star, s)
    if flag:
        r = 6 * s
        c.create_oval(x + size - 2 * r, y, x + size, y + 2 * r, fill=AMBER if flag == '!' else MUTED, width=0)
        c.create_text(x + size - r, y + r, text=flag, fill=BG, font=_font(c, 9 * s, 'bold'))


def _item_icon(c, x, y, size, name, s, held=False):
    img = _image(c, 'item', name, size)
    if img is not None:
        c.create_image(x, y, image=img, anchor='nw')
    else:
        c.create_rectangle(x, y, x + size, y + size, fill=ROW, outline=BORDER)
        c.create_text(x + size / 2, y + size / 2, text=''.join(w[0] for w in str(name).split()[:2]),
                      fill=MUTED, font=_font(c, size * 0.45, 'bold'))
    if held:
        c.create_text(x + size, y + size, text='✓', fill=GREEN, font=_font(c, 9 * s, 'bold'), anchor='se')


def _frame(c, name, x, y, w, h, s, right=''):
    """Panel background, title (small caps, muted) and right-hand info -> y of the content."""
    pad = round(10 * s)
    _round_rect(c, x + 1, y + 1, x + w - 2, y + h - 2, round(10 * s), fill=BG, outline=BORDER,
                width=max(1, round(1 * s)))
    c.create_text(x + pad, y + pad, text=TITLES[name], anchor='nw', fill=GOLD, font=_font(c, 12 * s, 'bold'))
    if right:
        color = AMBER if right.startswith('thinking') else MUTED
        c.create_text(x + w - pad, y + pad, text=right, anchor='ne', fill=color, font=_font(c, 12 * s, 'bold'))
    return y + pad + round(20 * s)


def _wrap(c, x, y, text, color, font, width, max_bottom, lines=2, cycle=False):
    """Wrapped text, at most `lines` lines and inside max_bottom -> bottom y, or None if nothing fits. Too long:
    cycle=True shows its sentences in turn (_cycle), else one line does too / more lines end in '…'."""
    text = str(text)
    lh = font.metrics('linespace')
    if lines <= 1 or max_bottom - y < 2 * lh:
        if max_bottom - y < lh:
            return None
        item = _cycle(c, x, y, text, font, width, anchor='nw', fill=color)
        return c.bbox(item)[3]
    if cycle:
        n = max(1, min(lines, int((max_bottom - y) // lh)))
        item = _cycle(c, x, y, text, font, width, n, anchor='nw', fill=color)
        return c.bbox(item)[3]
    item = c.create_text(x, y, text=text, anchor='nw', fill=color, font=font, width=width)
    limit = min(max_bottom, y + lines * lh + 1)
    if c.bbox(item)[3] <= limit:
        return c.bbox(item)[3]
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        c.itemconfigure(item, text=text[:mid].rstrip() + '…')
        if c.bbox(item)[3] <= limit:
            lo = mid
        else:
            hi = mid - 1
    if lo < 8:
        c.delete(item)
        return None
    c.itemconfigure(item, text=text[:lo].rstrip() + '…')
    return c.bbox(item)[3]


def _tint(changed, now):
    """A row's background: TINT just after it changed, fading to ROW over ~1 s."""
    if not changed:
        return None
    t = (now - changed) / 1.0
    return None if t >= 1 else _mix(TINT, BG, t)


def _chip(c, x, y, tag, s, w=None):
    """A row label: coloured caps on a dark rounded chip. -> its width."""
    f = _font(c, 11 * s, 'bold')
    w = w or f.measure(tag) + round(14 * s)
    _round_rect(c, x, y, x + w, y + round(19 * s), round(4 * s), fill=ROW, outline='')
    c.create_text(x + w / 2, y + round(10 * s), text=tag, fill=TAG_COLORS.get(tag.split(' ')[0], MUTED), font=f)
    return w


def _spinner(c, cx, cy, r, s, now):
    """A turning arc: the model is thinking."""
    c.create_oval(cx - r, cy - r, cx + r, cy + r, outline='#3a3020', width=max(2, round(2.5 * s)))
    c.create_arc(cx - r, cy - r, cx + r, cy + r, start=-(now * 300) % 360, extent=100, style='arc',
                 outline=AMBER, width=max(2, round(2.5 * s)))


def _deliberation_tail(text):
    """The model's reasoning as it streams in: the text of its leading "deliberation" object (quotes and
    braces left out), or '' when the reply has no reasoning first (a short decision)."""
    import re as _re
    i = text.find('"deliberation"')
    if i < 0:
        return ''
    body = text[i + len('"deliberation"'):]
    body = _re.sub(r'\\n|\\"', ' ', body)
    body = _re.sub(r'"[a-z_]+"\s*:', ' ', body)            # keys of a structured deliberation
    body = _re.sub(r'[{}\[\]"]', ' ', body).lstrip(' :')
    return _re.sub(r'\s+', ' ', body).strip()


def _blocks(c, d, s, width, lines=99):
    """The decision panel's content as blocks of known height: (height, draw(x, y, max_bottom)). `lines`:
    the most lines a row's text (and a card's reason) may take, to fit the panel."""
    out = []
    f_body, f_q, f_th = _font(c, 14 * s), _font(c, 13 * s, 'bold'), _font(c, 13 * s)

    def measure(text, font, w, cap=999):
        it = c.create_text(0, 0, text=text, anchor='nw', font=font, width=w)
        hgt = c.bbox(it)[3] - c.bbox(it)[1]
        c.delete(it)
        return min(hgt, cap * font.metrics('linespace'))

    th = d.get('thinking') if d.get('think_block', True) else None     # else the header alone shows it
    if th:
        q = th.get('what') or 'thinking'
        tail = _deliberation_tail(th.get('text') or '')
        hq = measure(q, f_q, width)

        def draw_think(x, y, bottom, q=q, tail=tail, hq=hq):
            c.create_text(x, y, text=q, anchor='nw', fill=AMBER, font=f_q, width=width)
            if tail:
                y2 = y + hq + round(4 * s)
                room = max(0, bottom - y2)
                lines = max(1, int(room // f_th.metrics('linespace')))
                # the newest reasoning: the tail that fits
                it = c.create_text(x, y2, text=tail, anchor='nw', fill=MUTED, font=f_th, width=width)
                while c.bbox(it)[3] > bottom and len(tail) > 20:
                    tail = tail[len(tail) // 6:]
                    c.itemconfigure(it, text='… ' + tail.lstrip())
        out.append((hq + round(4 * s) + (4 * _font(c, 13 * s).metrics('linespace') if tail else 0), draw_think))
    car = d.get('carousel')
    if car:
        rows = car.get('list') or []
        lh = round(20 * s)

        def draw_car(x, y, bottom, rows=rows, car=car):
            c.create_text(x, y, text=f"priority {car.get('version') or '-'}" + (f"  ·  {car['note']}" if car.get('note') else ''),
                          anchor='nw', fill=GOLD, font=_font(c, 12 * s, 'bold'))
            y += round(19 * s)
            for k, (it, st) in enumerate(rows):
                if y + lh > bottom:
                    break
                mark, color = {'got': ('✓', GREEN), 'target': ('→', TEXT), 'gone': ('✗', RED)}.get(st, (str(k + 1), MUTED))
                c.create_text(x, y + lh / 2, text=mark, anchor='w', fill=color, font=_font(c, 14 * s, 'bold'))
                _item_icon(c, x + round(18 * s), y + round(1 * s), round(18 * s), it, s)
                f = _font(c, 14 * s, 'bold' if st == 'target' else 'normal')
                tid = c.create_text(x + round(42 * s), y + lh / 2, text=it, anchor='w',
                                    fill=color if st != 'free' else TEXT, font=f)
                if st == 'gone':
                    bb = c.bbox(tid)
                    c.create_line(bb[0], (bb[1] + bb[3]) / 2, bb[2], (bb[1] + bb[3]) / 2, fill=RED, width=max(1, round(s)))
                y += lh
        out.append((round(19 * s) + lh * max(1, len(rows)), draw_car))
    card = d.get('card')
    if card and not car:
        f_c = _font(c, 17 * s, 'bold')
        hc = measure(card.get('choice', ''), f_c, width)
        hw = measure(card.get('why', ''), f_body, width, lines) if card.get('why') else 0

        def draw_card(x, y, bottom, card=card, hc=hc):
            _chip(c, x, y, card.get('tag', ''), s)
            y += round(25 * s)
            c.create_text(x, y, text=card.get('choice', ''), anchor='nw', fill=TEXT, font=f_c, width=width)
            if card.get('why'):
                _wrap(c, x, y + hc + round(4 * s), card['why'], MUTED, f_body, width, bottom, lines, cycle=True)
        out.append((round(25 * s) + hc + round(4 * s) + hw, draw_card))
    tagw = round(60 * s)
    for label, text, reason, changed in ([] if car else d.get('rows') or []):
        body = text + (f' — {reason}' if reason else '')
        hb = measure(body, f_body, width - tagw - round(8 * s), lines)

        def draw_row(x, y, bottom, label=label, body=body, changed=changed, hb=hb):
            tint = _tint(changed, d.get('_now') or time.time())
            if tint:
                c.create_rectangle(x - round(3 * s), y - round(2 * s), x + width + round(3 * s), y + hb + round(2 * s),
                                   fill=tint, width=0)
            if _wrap(c, x + tagw + round(8 * s), y, body, TEXT, f_body, width - tagw - round(8 * s), bottom, lines,
                     cycle=True) is None:
                return
            _chip(c, x, y + round(1 * s), label, s, tagw)
        out.append((max(hb, round(20 * s)), draw_row))
    return out


def draw_decision(c, d, x, y, w, h, s, now):
    """AI DECISION. Header: while the model thinks, a turning spinner and the seconds so far (amber);
    after, how long the answer took. A red alert line when something broke (and where to look). Then the
    thinking (the question and the model's reasoning as it writes it), the carousel's priority list, a
    one-off choice as a card, and the plan's rows (coloured tag + the model's own words). All of it on one
    page: rows are cut to fewer lines until it fits, then finished rows (✓) give way (the user, 2026-10-10:
    no turning pages)."""
    d = dict(d, _now=now)
    pad = round(10 * s)
    _round_rect(c, x + 1, y + 1, x + w - 2, y + h - 2, round(10 * s), fill=BG, outline=BORDER,
                width=max(1, round(1 * s)))
    c.create_text(x + pad, y + pad, text=TITLES['top_left'], anchor='nw', fill=GOLD, font=_font(c, 12 * s, 'bold'))
    right = x + w - pad - round((BUTTON + 6) * s)            # the collapse button sits at the far right
    th = d.get('thinking')
    stage = d.get('stage') or ''
    if th:
        secs = (now - th['t0']) if th.get('t0') else (th.get('secs') or 0)
        txt = f"thinking {secs:.1f}s"
        f = _font(c, 13 * s, 'bold')
        c.create_text(right, y + pad - round(1 * s), text=txt, anchor='ne', fill=AMBER, font=f)
        r = round(7 * s)
        _spinner(c, right - f.measure(txt) - round(12 * s), y + pad + round(8 * s), r, s, now)
        c.create_text(right - f.measure(txt) - round(26 * s), y + pad, text=stage, anchor='ne', fill=MUTED,
                      font=_font(c, 12 * s, 'bold'))
    else:
        took = d.get('took')
        c.create_text(right, y + pad, text=stage + (f"  ·  answered in {took:.1f}s" if took else ''), anchor='ne',
                      fill=MUTED, font=_font(c, 12 * s, 'bold'))
    cy = y + pad + round(22 * s)
    inner, bottom = w - 2 * pad, y + h - round(10 * s)
    for tag, text, where in (d.get('alerts') or [])[:2]:
        lh = round(22 * s)
        c.create_rectangle(x + pad, cy, x + w - pad, cy + lh, fill=RED, width=0)
        tw = _font(c, 12 * s, 'bold').measure(tag) + round(12 * s)
        c.create_text(x + pad + round(6 * s), cy + lh / 2, text=tag, anchor='w', fill='#ffffff',
                      font=_font(c, 12 * s, 'bold'))
        _cycle(c, x + pad + tw, cy + lh / 2, f"{text}" + (f"  → {where}" if where else ''), _font(c, 13 * s),
                 inner - tw - round(6 * s), anchor='w', fill='#ffffff')
        cy += lh + round(4 * s)
    # one page, compacted step by step until it all fits: rows to fewer lines; then the comp name goes (the
    # TARGET COMP panel shows it), a choice card becomes one row and the gaps close; then finished rows (✓) go
    def compact(d):
        card, rows = d.get('card'), list(d.get('rows') or [])
        if card and not d.get('carousel'):
            why = f" — {card['why']}" if card.get('why') else ''
            rows = [(card.get('tag', ''), card.get('choice', '') + why, '', None)] + rows
        return dict(d, card=None, rows=rows, title=None)

    def todo(d):
        return dict(d, rows=[r for r in d.get('rows') or [] if not str(r[1]).startswith('✓')])

    tries = [(d, lines, 6) for lines in (99, 4, 3, 2, 1)] + [(compact(d), 2, 3), (compact(d), 1, 3),
                                                             (todo(compact(d)), 1, 3)]
    for dd, lines, g in tries:
        top = cy + (round(24 * s) if dd.get('title') else 0)
        gap = round(g * s)
        blocks = _blocks(c, dd, s, inner, lines)
        if sum(hgt for hgt, _ in blocks) + gap * max(0, len(blocks) - 1) <= bottom - top:
            break
    if dd.get('title'):
        c.create_text(x + pad, cy, text=_fit(_font(c, 16 * s, 'bold'), dd['title'], inner), anchor='nw', fill=TEXT,
                      font=_font(c, 16 * s, 'bold'))
    if not blocks:
        c.create_text(x + pad, top, text=d.get('empty', 'Waiting for the first decision.'), anchor='nw', fill=MUTED,
                      font=_font(c, 14 * s))
        return
    yy = top
    for hgt, fn in blocks:
        if yy + min(hgt, round(18 * s)) > bottom:
            break
        fn(x + pad, yy, bottom)
        yy += hgt + gap


def draw_comp(c, d, x, y, w, h, s, now):
    """TARGET COMP: the fixed comp's portraits in its own order, filling the corner. Owned: full colour;
    not owned: greyed; in the shop now: a gold outline. Under each, the comp's star goal - gold once
    reached, grey until then - and the planned items, ticked when held."""
    pad = round(10 * s)
    _round_rect(c, x + 1, y + 1, x + w - 2, y + h - 2, round(10 * s), fill=BG, outline=BORDER,
                width=max(1, round(1 * s)))
    bx = x + pad + round((BUTTON + 6) * s)                  # the collapse button sits at the far left
    f_t, f_n = _font(c, 12 * s, 'bold'), _font(c, 16 * s, 'bold')
    c.create_text(bx, y + pad, text=TITLES['top_right'], anchor='nw', fill=GOLD, font=f_t)
    tw = f_t.measure(TITLES['top_right']) + round(12 * s)
    right = d.get('count', '')
    rw = _font(c, 14 * s, 'bold').measure(right) + round(4 * s)
    c.create_text(bx + tw, y + pad - round(2 * s), text=_fit(f_n, d.get('name', ''), x + w - pad - rw - bx - tw),
                  anchor='nw', fill=TEXT, font=f_n)
    c.create_text(x + w - pad, y + pad - round(1 * s), text=right, anchor='ne', fill=MUTED, font=_font(c, 14 * s, 'bold'))
    cy = y + pad + round(24 * s)
    units = list(d.get('units') or [])
    n = max(1, len(units))
    gap = round(6 * s)
    room = y + h - round(8 * s) - cy - round(32 * s)        # under a tile: the star goal and the item row
    size = min(round(60 * s), room, (w - 2 * pad - (n - 1) * gap) / n)
    for i, u in enumerate(units):
        tx = x + pad + i * (size + gap)
        _champ_tile(c, tx, cy, size, u['name'], u.get('cost'), s, grey=not u.get('owned'))
        goal, have = int(u.get('goal') or 2), int(u.get('star') or 0)
        c.create_text(tx + size / 2, cy + size + round(1 * s), text='★' * goal, anchor='n',
                      fill=GOLD if have >= goal else '#4a5568', font=_font(c, 12 * s, 'bold'))
        if u.get('shop'):
            c.create_rectangle(tx - 2 * s, cy - 2 * s, tx + size + 2 * s, cy + size + 2 * s, outline=GOLD,
                               width=max(2, round(2 * s)))
        isz = min(round(18 * s), (size - 2 * s) / 3)
        for k, (it, held) in enumerate((u.get('items') or [])[:3]):
            _item_icon(c, tx + k * (isz + 1 * s), cy + size + round(16 * s), isz, it, s, held)


def draw_execution(c, d, x, y, w, h, s, now):
    """EXECUTION: the round's steps in order - what ran (✓ done, ✗ failed + why), what runs now (→), what is
    still to come (○) and a stored next-round plan (»). Each row: its mark, the kind of action as a coloured
    tag (SELL, BUY, XP, MOVE, ITEM, ROLL, WISP, OPEN) and the step in words."""
    pad = round(10 * s)
    _round_rect(c, x + 1, y + 1, x + w - 2, y + h - 2, round(10 * s), fill=BG, outline=BORDER,
                width=max(1, round(1 * s)))
    c.create_text(x + pad, y + pad, text=TITLES['bottom_left'], anchor='nw', fill=GOLD, font=_font(c, 12 * s, 'bold'))
    c.create_text(x + w - pad - round((BUTTON + 6) * s), y + pad, text=d.get('mode', ''), anchor='ne', fill=MUTED,
                  font=_font(c, 12 * s, 'bold'))
    cy = y + pad + round(24 * s)
    inner, bottom = w - 2 * pad, y + h - round(8 * s)
    fb = _font(c, 14 * s)
    lh = round(22 * s)
    tagw = round(50 * s)
    steps = d.get('steps') or []
    if not steps:
        c.create_text(x + pad, cy, text='No steps yet this round.', anchor='nw', fill=MUTED, font=fb)
        return
    for status, tag, text, reason, changed in steps:
        if cy + lh > bottom:
            break
        mark, color = STEP.get(status, ('·', MUTED))
        tint = _tint(changed, now)
        if tint:
            c.create_rectangle(x + pad - 2 * s, cy - 1 * s, x + w - pad + 2 * s, cy + lh - 2 * s, fill=tint, width=0)
        c.create_text(x + pad, cy + round(1 * s), text=mark, anchor='nw', fill=color, font=_font(c, 14 * s, 'bold'))
        _chip(c, x + pad + round(18 * s), cy, tag, s, tagw)
        tx = x + pad + round(18 * s) + tagw + round(6 * s)
        body = text + (f' — {reason}' if reason else '')
        _cycle(c, tx, cy + round(1 * s), body, fb, x + w - pad - tx, anchor='nw',
                 fill=MUTED if status in ('done', 'next') else RED if status == 'fail' else TEXT)
        cy += lh


def draw_strategy(c, d, x, y, w, h, s, now):
    """STRATEGY: the model's strategy paragraph, wrapped over the panel (a smaller font if it does not fit,
    then cut with '…'), and under it when it last changed ("last updated: 3-2"); the paragraph tints when it
    changes. d['comps'] instead (flex, no comp committed yet): the comp probabilities as bars.
    d['pending']: the model is still preparing."""
    pad = round(10 * s)
    _round_rect(c, x + 1, y + 1, x + w - 2, y + h - 2, round(10 * s), fill=BG, outline=BORDER,
                width=max(1, round(1 * s)))
    f_t = _font(c, 12 * s, 'bold')
    c.create_text(x + pad, y + pad, text=TITLES['bottom_right'], anchor='nw', fill=GOLD, font=f_t)
    tx = x + pad + f_t.measure(TITLES['bottom_right']) + round(8 * s)
    sub = d.get('comp') or ('comp probabilities' if d.get('comps') else '')
    room = x + w - pad - round((BUTTON + 6) * s) - tx
    if sub and room > 20 * s:
        c.create_text(tx, y + pad, text=_fit(f_t, sub, room), anchor='nw', fill=TEXT, font=f_t)
    cy = y + pad + round(24 * s)
    inner, bottom = w - 2 * pad, y + h - round(8 * s)
    fb = _font(c, 12 * s)
    lh = fb.metrics('linespace')
    if d.get('comps'):
        bar = round(4 * s)
        for name, p in d['comps']:
            if cy + lh + bar > bottom:
                break
            pct = f'{round(100 * p)}%'
            pw = fb.measure(pct) + round(6 * s)
            c.create_text(x + pad, cy, text=_fit(fb, name, inner - pw), anchor='nw', fill=TEXT, font=fb)
            c.create_text(x + w - pad, cy, text=pct, anchor='ne', fill=GOLD, font=fb)
            by = cy + lh + round(1 * s)
            c.create_rectangle(x + pad, by, x + w - pad, by + bar, fill=ROW, width=0)
            c.create_rectangle(x + pad, by, x + pad + inner * max(0.0, min(1.0, p)), by + bar, fill=GOLD, width=0)
            cy = by + bar + round(6 * s)
        if d.get('pending') and cy + lh <= bottom:
            c.create_text(x + pad, cy, text='strategy: preparing…', anchor='nw', fill=MUTED, font=fb)
        return
    text = d.get('text') or ''
    if not text:
        c.create_text(x + pad, cy, text='The model is preparing its strategy…' if d.get('pending') else
                      'No strategy yet.', anchor='nw', fill=MUTED, font=fb, width=inner)
        return
    f_f = _font(c, 11 * s)
    foot_h = f_f.metrics('linespace') + round(6 * s) if d.get('foot') else 0
    body = bottom - foot_h
    changed = (d.get('rows') or [(None, None, None)])[0][-1]
    item = None
    for px in (13, 12, 11, 10):                           # the largest font the whole paragraph fits in
        font = _font(c, px * s)
        item = c.create_text(x + pad, cy, text=text, anchor='nw', fill=TEXT, font=font, width=inner)
        if c.bbox(item)[3] <= body or px == 10:
            break
        c.delete(item)
    if c.bbox(item)[3] > body:                            # still too long: cut at the last line that fits
        c.delete(item)
        _wrap(c, x + pad, cy, text, TEXT, font, inner, body, lines=99)
    tint = _tint(changed, now)
    if tint:                                              # just changed: tinted behind the paragraph
        c.create_rectangle(x + pad - 3 * s, cy - 2 * s, x + w - pad + 3 * s, body + 1 * s, fill=tint, width=0)
        for it in c.find_overlapping(x + pad, cy, x + w - pad, body):
            if c.type(it) == 'text':
                c.tag_raise(it)
    if d.get('foot'):
        _cycle(c, x + pad, bottom - foot_h + round(4 * s), d['foot'], f_f, inner, anchor='nw', fill=MUTED)


DRAW = {'top_left': draw_decision, 'top_right': draw_comp, 'bottom_left': draw_execution,
        'bottom_right': draw_strategy}


def draw_panel(c, name, data, x, y, w, h, s, now=None):
    """Draw one panel into canvas c at (x, y, w, h). s = scale (screen height / 1080). No data: nothing
    is drawn (the panel's window stays transparent)."""
    if not data:
        return
    DRAW[name](c, data, x, y, w, h, s, time.time() if now is None else now)


# the augment picker: the model's one-sentence verdict on each card at the bottom of the card (the user,
# 2026-10-12: "on the bottom of the augments, a concise sentence to show rejection or choice"), as soon as the
# model writes it, with its choice marked, until the picker closes. 1080p boxes over the cards' empty lower part
CARD_BOXES = [(421, 680, 694, 790), (821, 680, 1099, 790), (1226, 680, 1504, 790)]


def draw_cards(c, cards, to):
    """cards: per card (left, middle, right) {verdict, mark, ...} or None: the verdict in a small box at the card's
    bottom, one sentence at a time if it does not fit."""
    k = to((960, 1080))[1] / 1080
    for box, card in zip(CARD_BOXES, cards or []):
        if not card or not card.get('verdict'):
            continue
        (x0, y0), (x1, y1) = to(box[:2]), to(box[2:])
        mark = card.get('mark')
        c.create_rectangle(x0, y0, x1, y1, fill=BG, outline=GOLD if mark == 'TAKING' else BORDER,
                           width=max(2, round((3 if mark == 'TAKING' else 1.5) * k)))
        pad = round(7 * k)
        c.create_text(x0 + pad, y0 + pad, text=mark or 'MODEL', anchor='nw', fill=GOLD if mark else MUTED,
                      font=_font(c, 12 * k, 'bold'))
        font = _font(c, 14 * k)
        y = y0 + pad + round(19 * k)
        lines = max(1, int((y1 - pad - y) // font.metrics('linespace')))
        _cycle(c, x0 + pad, y, card['verdict'], font, x1 - x0 - 2 * pad, lines, anchor='nw', fill=TEXT)


def _draw_card_explanations(c, cards, to):
    """The long per-card COMBAT / ECONOMY text over the cards' pictures (not drawn: the user chose the verdict line
    at the bottom, 2026-10-12; kept for the live page)."""
    k = to((960, 1080))[1] / 1080
    for box, card in zip([(421, 296, 694, 512), (821, 296, 1099, 512), (1226, 296, 1504, 512)], cards or []):
        if not card or not (card.get('combat') or card.get('economy')):
            continue
        (x0, y0), (x1, y1) = to(box[:2]), to(box[2:])
        mark = card.get('mark')
        c.create_rectangle(x0, y0, x1, y1, fill=BG, outline=GOLD if mark == 'TAKING' else BORDER,
                           width=max(2, round((3 if mark == 'TAKING' else 1.5) * k)))
        pad = round(8 * k)
        c.create_text(x0 + pad, y0 + pad, text='MODEL' + (f'  ·  {mark}' if mark else ''), anchor='nw',
                      fill=GOLD if mark else MUTED, font=_font(c, 12 * k, 'bold'))
        y, width = y0 + pad + round(20 * k), x1 - x0 - 2 * pad
        parts = [f'{name}\n{text}' for name, text in (('COMBAT', card.get('combat')), ('ECONOMY', card.get('economy')))
                 if text]
        font = _font(c, 13 * k)
        lines = max(1, int((y1 - pad - y) // font.metrics('linespace')))
        whole = '\n'.join(parts)
        item = c.create_text(x0 + pad, y, text=whole, anchor='nw', fill=TEXT, font=font, width=width)
        bb = c.bbox(item)
        if bb[3] - bb[1] <= lines * font.metrics('linespace') + 1 or len(parts) < 2:
            continue
        c.delete(item)
        key = (id(c), whole)
        t0 = _CYC_T0.setdefault(key, time.time())
        item = c.create_text(x0 + pad, y, text=parts[int((time.time() - t0) / CYC_HOLD) % 2], anchor='nw',
                             fill=TEXT, font=font, width=width)
        _CYC.setdefault(c, []).append([item, parts, key])


def draw_path(c, d, s, to, now=None):
    """The carousel over the game: a ring on the unit we go for, with its item and rank (thin rings on the next two
    of the priority list). It is drawn
    where the ring's constant turn puts that unit now (d['anim']: its angle and the turning speed at
    the last perception tick), so it moves smoothly with the unit between reads instead of jumping
    once per read. No route lines (the user, 2026-10-07: the old path layer was noise). A thin stroke
    (< 5 px): our tactician is found by a >= 8 px flat-colour bar, item icons sit 115 px above the
    unit, outside the ring."""
    if (d or {}).get('cards'):
        return draw_cards(c, d['cards'], to)
    an = (d or {}).get('anim')
    if not an:
        return
    now = time.time() if now is None else now
    turned = an['w'] * max(0.0, min(3.0, now - an['t']))
    pk, k = d.get('pickup', 55), an['K']
    # the target: a thick ring; the next two of the priority list (an['others']): thin rings, smaller labels
    for u, main in [(an, True)] + [(o, False) for o in an.get('others') or []]:
        ang = u['a'] + turned
        ux, uy = an['cx'] + an['R'] * math.cos(ang), an['cy'] + k * an['R'] * math.sin(ang)
        r = pk if main else pk * 0.75
        pts = [to((ux + r * math.cos(t / 36 * 2 * math.pi), uy + r * k * math.sin(t / 36 * 2 * math.pi)))
               for t in range(37)]
        color = '#ffb02e' if main else '#c8aa6e'
        c.create_line(*[v for p in pts for v in p], fill=color, width=max(1, round((3 if main else 1.5) * s)),
                      smooth=True)
        if u.get('label'):
            p = to((ux, uy + r * k + 16))
            tag = f"#{u['rank']}  {u['label']}" if u.get('rank') else u['label']
            c.create_text(p[0], p[1], text=tag, fill=color,
                          font=('Segoe UI', -max(8, round((16 if main else 13) * s)), 'bold' if main else 'normal'))


class Tracker:
    """When each row last changed, so a changed row can tint and fade: rows are keyed by panel and
    label (decision rows) or by step text (checklist), and a row is 'changed' when its text differs
    from what was drawn before."""

    def __init__(self):
        self.seen, self.started = {}, set()

    def mark(self, panel, rows, key_of, text_of, now):
        first = panel not in self.started              # the first picture of a panel does not flash
        self.started.add(panel)
        out = []
        for r in rows:
            k = (panel, key_of(r))
            t = text_of(r)
            old = self.seen.get(k)
            if old is None or old[0] != t:
                self.seen[k] = (t, 0.0 if first else now)
            out.append(tuple(r) + (self.seen[k][1] or None,))
        return out


def stamp(data, tracker, now):
    """Add each row's change time to the panels' data (rows as given -> rows + (changed_at,))."""
    d = dict(data)
    if d.get('top_left'):
        tl = dict(d['top_left'])
        tl['rows'] = tracker.mark('top_left', tl.get('rows') or [], lambda r: r[0], lambda r: (r[1], r[2]), now)
        d['top_left'] = tl
    if d.get('bottom_left'):
        bl = dict(d['bottom_left'])
        bl['steps'] = tracker.mark('bottom_left', bl.get('steps') or [], lambda r: (r[1], r[2]), lambda r: r[0], now)
        d['bottom_left'] = bl
    if d.get('bottom_right') and d['bottom_right'].get('text'):
        br = dict(d['bottom_right'])                      # the paragraph as one row: it tints when it changes
        br['rows'] = tracker.mark('bottom_right', [('text', br['text'])], lambda r: r[0], lambda r: r[1], now)
        d['bottom_right'] = br
    return d


STATE_FILE = ROOT / 'data' / 'overlay-state.json'          # which panels the viewer collapsed (kept across games)
BUTTON_SIDE = {'top_left': 'right', 'top_right': 'left', 'bottom_left': 'right', 'bottom_right': 'right'}


def _load_collapsed():
    try:
        import json
        return set(json.loads(STATE_FILE.read_text(encoding='utf-8')).get('collapsed') or [])
    except Exception:
        return set()


def _save_collapsed(names):
    try:
        import json
        STATE_FILE.write_text(json.dumps({'collapsed': sorted(names)}), encoding='utf-8')
    except Exception:
        pass


def draw_button(c, name, collapsed, w, h, s):
    """The collapse button: '–' on an open panel, a small tab with the panel's name when collapsed."""
    c.delete('all')
    _round_rect(c, 1, 1, w - 1, h - 1, round(5 * s), fill='#1b2737', outline=BORDER)
    if collapsed:
        c.create_text(w / 2, h / 2, text='▸ ' + TITLES[name], fill=GOLD, font=_font(c, 11 * s, 'bold'))
    else:
        c.create_text(w / 2, h / 2 - round(1 * s), text='–', fill=TEXT, font=_font(c, 14 * s, 'bold'))


class Overlay:
    def __init__(self, client):
        self.client = client
        self.q: queue.Queue = queue.Queue()
        self.ok = True
        threading.Thread(target=self._run, daemon=True).start()

    def show(self, **panels):
        """panels: top_left / top_right / bottom_left / bottom_right (dicts, see the draw_* functions; None hides a panel)
        and path (the carousel target layer)."""
        self.q.put({k: v for k, v in panels.items() if k in PANELS or k == 'path'})

    def _to(self, p):
        """1080p reference point -> pixel in the game-window-sized path layer."""
        c = self.client
        s = c.height / 1080
        return c.width / 2 + (p[0] - 960) * s, p[1] * s

    def _box(self, box):
        c = self.client
        s = c.height / 1080
        x0, y0, x1, y1 = box
        X0 = max(0.0, c.width / 2 + (x0 - 960) * s)
        X1 = min(float(c.width), c.width / 2 + (x1 - 960) * s)
        return (round(c.left + X0), round(c.top + y0 * s), round(X1 - X0), round((y1 - y0) * s))

    def _button_box(self, name, collapsed, s):
        """Where a panel's button window goes: in its header's corner (the open panel leaves room for it),
        widened to a tab with the name when collapsed."""
        x, y, bw, bh = self._box(PANELS[name])
        size = round(BUTTON * s)
        w = round((len(TITLES[name]) * 8 + 34) * s) if collapsed else size
        top = y + round(8 * s)
        left = x + round(8 * s) if BUTTON_SIDE[name] == 'left' else x + bw - round(8 * s) - w
        return left, top, w, size

    def _run(self):
        try:
            import tkinter as tk
        except Exception:
            self.ok = False
            return
        root = tk.Tk()
        root.withdraw()
        s = self.client.height / 1080
        canvases, data, wins = {}, {k: None for k in PANELS}, {}
        collapsed = _load_collapsed() & set(PANELS)
        user32 = ctypes.windll.user32

        def styled(win, click_through):
            win.update_idletasks()
            hwnd = user32.GetParent(win.winfo_id()) or win.winfo_id()
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE) | WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE \
                | WS_EX_TOPMOST
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_TRANSPARENT if click_through else style)

        for name, box in PANELS.items():
            w = tk.Toplevel(root)
            w.overrideredirect(True)
            w.attributes('-topmost', True)
            w.attributes('-alpha', ALPHA)
            w.attributes('-transparentcolor', KEY)
            x, y, bw, bh = self._box(box)
            w.geometry(f'{bw}x{bh}+{x}+{y}')
            cv = tk.Canvas(w, width=bw, height=bh, bg=KEY, highlightthickness=0)
            cv.pack()
            canvases[name] = (cv, bw, bh)
            styled(w, True)                      # the panel lets every click through to the game
            wins[name] = w
        buttons = {}

        def place_button(name):
            bwin, bcv = buttons[name]
            bx, by, bw_, bh_ = self._button_box(name, name in collapsed, s)
            bwin.geometry(f'{bw_}x{bh_}+{bx}+{by}')
            bcv.configure(width=bw_, height=bh_)
            draw_button(bcv, name, name in collapsed, bw_, bh_, s)
            if name in collapsed:
                wins[name].withdraw()
            else:
                wins[name].deiconify()
                bwin.lift()

        def toggle(name):
            collapsed.symmetric_difference_update({name})
            _save_collapsed(collapsed)
            place_button(name)
            state['drawn'].pop(name, None)

        for name in PANELS:
            # the button: its own small window that DOES take clicks (never focus: the game stays in front)
            bwin = tk.Toplevel(root)
            bwin.overrideredirect(True)
            bwin.attributes('-topmost', True)
            bwin.attributes('-alpha', 0.95)
            bcv = tk.Canvas(bwin, width=10, height=10, bg='#1b2737', highlightthickness=0, cursor='hand2')
            bcv.pack()
            bcv.bind('<Button-1>', lambda e, n=name: toggle(n))
            buttons[name] = (bwin, bcv)
            styled(bwin, False)
        # the carousel target layer: the whole game window, click-through, empty unless data['path']
        pw = tk.Toplevel(root)
        pw.overrideredirect(True)
        pw.attributes('-topmost', True)
        pw.attributes('-transparentcolor', KEY)
        c = self.client
        pw.geometry(f'{c.width}x{c.height}+{c.left}+{c.top}')
        path_cv = tk.Canvas(pw, width=c.width, height=c.height, bg=KEY, highlightthickness=0)
        path_cv.pack()
        styled(pw, True)
        # the keys box: click-through, always shown
        kw = tk.Toplevel(root)
        kw.overrideredirect(True)
        kw.attributes('-topmost', True)
        kw.attributes('-alpha', ALPHA)
        kw.attributes('-transparentcolor', KEY)
        kx, ky, kbw, kbh = self._box(KEYS_BOX)
        kw.geometry(f'{kbw}x{kbh}+{kx}+{ky}')
        keys_cv = tk.Canvas(kw, width=kbw, height=kbh, bg=KEY, highlightthickness=0)
        keys_cv.pack()
        styled(kw, True)
        from tft_player.game import PAUSED
        data['path'] = None
        tracker = Tracker()
        state = {'drawn': {}}
        for name in PANELS:
            place_button(name)

        def poll():
            now = time.time()
            got = {}
            try:
                while True:
                    got.update(self.q.get_nowait())
            except queue.Empty:
                pass
            if got:
                data.update(stamp(got, tracker, now))
            for name, (cv, bw, bh) in canvases.items():
                d = data.get(name)
                if name in collapsed:
                    continue
                if not d:
                    if state['drawn'].get(name) != 'none':
                        state['drawn'][name] = 'none'
                        cv.delete('all')
                        buttons[name][0].withdraw()      # a hidden panel shows no button either
                    continue
                if state['drawn'].get(name) == 'none':
                    buttons[name][0].deiconify()
                # redraw on change; while a row fades or the model thinks (the spinner turns), every tick
                live = d.get('thinking') or any(
                    r[-1] and now - r[-1] < 1.1 for r in (d.get('rows') or d.get('steps') or []))
                key = repr(d) + (str(round(now, 1)) if live else '')
                if state['drawn'].get(name) != key:
                    state['drawn'][name] = key
                    cv.delete('all')
                    _CYC.pop(cv, None)
                    try:
                        draw_panel(cv, name, d, 0, 0, bw, bh, s, now)
                    except Exception:
                        pass                     # a drawing slip never stops the overlay
            for cv, _, _ in canvases.values():
                if _CYC.get(cv):
                    _cycle_tick(cv, now)
            paused = PAUSED.is_set()
            if state['drawn'].get('keys') != paused:
                state['drawn']['keys'] = paused
                draw_keys(keys_cv, kbw, kbh, s, paused)
            pd = data.get('path')
            moving = bool(pd and pd.get('anim'))
            key = repr(pd) + (str(now) if moving else '')
            if state['drawn'].get('path') != key:
                state['drawn']['path'] = key
                path_cv.delete('all')
                _CYC.pop(path_cv, None)
                draw_path(path_cv, pd, s, self._to, now)
            if _CYC.get(path_cv):
                _cycle_tick(path_cv, now)
            root.after(33 if moving else 100, poll)      # the target ring moves at ~30 fps

        poll()
        root.mainloop()


def preview(frame_path, panels, out_path, path=None):
    """Draw the panels over a recorded frame and save a PNG (design check, no game needed)."""
    import tkinter as tk
    from PIL import Image, ImageGrab, ImageTk
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)     # window pixels = screen pixels for the grab
    except Exception:
        pass
    img = Image.open(frame_path).convert('RGB')
    H = 1080
    img = img.resize((round(img.width * H / img.height), H))
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes('-topmost', True)
    root.geometry(f'{img.width}x{img.height}+0+0')
    cv = tk.Canvas(root, width=img.width, height=img.height, highlightthickness=0)
    cv.pack()
    photo = ImageTk.PhotoImage(img, master=root)
    cv.create_image(0, 0, image=photo, anchor='nw')
    client = type('C', (), {'left': 0, 'top': 0, 'width': img.width, 'height': img.height})()
    ov = Overlay.__new__(Overlay)
    ov.client = client
    now = time.time()
    panels = stamp(panels, Tracker(), now)
    # panels are separate translucent windows live; here each is drawn on a layer and blended
    for name, box in PANELS.items():
        x, y, bw, bh = ov._box(box)
        if not panels.get(name):
            continue
        layer = tk.Canvas(root, width=bw, height=bh, bg=KEY, highlightthickness=0)
        draw_panel(layer, name, panels[name], 0, 0, bw, bh, 1.0, now)
        bx, by, bbw, bbh = ov._button_box(name, False, 1.0)
        btn = tk.Canvas(layer, width=bbw, height=bbh, bg='#1b2737', highlightthickness=0)
        draw_button(btn, name, False, bbw, bbh, 1.0)
        layer.create_window(bx - x, by - y, window=btn, anchor='nw')
        cv.create_window(x, y, window=layer, anchor='nw')
    if path:
        draw_path(cv, path, 1.0, ov._to, (path.get('anim') or {}).get('t'))
    root.update()
    time.sleep(0.4)
    root.update()
    shot = ImageGrab.grab(bbox=(root.winfo_rootx(), root.winfo_rooty(),
                                root.winfo_rootx() + img.width, root.winfo_rooty() + img.height))
    # blend the panels with the frame at the live window opacity
    base = img.copy()
    for name, box in PANELS.items():
        if not panels.get(name):
            continue
        x, y, bw, bh = ov._box(box)
        crop = shot.crop((x, y, x + bw, y + bh))
        under = base.crop((x, y, x + bw, y + bh))
        mask = Image.eval(crop.convert('L'), lambda v: 255)
        key = Image.new('RGB', (bw, bh), KEY)
        from PIL import ImageChops
        diff = ImageChops.difference(crop, key).convert('L').point(lambda v: 255 if v > 3 else 0)
        blended = Image.blend(under, crop, ALPHA)
        under.paste(blended, (0, 0), diff)
        base.paste(under, (x, y))
        del mask
    if path:
        # the ring and label (drawn straight on the frame canvas): copied from the grab where they differ
        from PIL import ImageChops
        diff = ImageChops.difference(shot, img).convert('L').point(lambda v: 255 if v > 8 else 0)
        for name, box in PANELS.items():
            x, y, bw, bh = ov._box(box)
            diff.paste(0, (x, y, x + bw, y + bh))
        base.paste(shot, (0, 0), diff)
    base.save(out_path)
    root.destroy()
