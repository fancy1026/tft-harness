"""Screenshots of MetaTFT comp pages, taken by a headless Chrome in the background.

The comp list's numbers are exact in our API data (avg / pick / win / top 4), but
a comp page's averages can't be reproduced from the API, so the model gets the
page itself: the comp's box on metatft.com/comps with each of its tabs open.

Chrome is driven through its DevTools port (standard library only: a minimal
websocket client). Screenshots and their transcription are cached per comp for MAX_AGE_H hours.

    shots = comp_pages('Lunar Aphelios Nidalee')   # [(tab name, jpeg bytes), ...]
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / 'data' / 'metatft_shots'
TABS = ['Options & Quick Start', 'Units & Items', 'Pro Augments and Tips']
MAX_AGE_H = 48                 # screenshots and their transcription are taken again when older
URL = 'https://www.metatft.com/comps'


def _chrome():
    for p in (os.environ.get('TFT_CHROME'),
              r'C:\Program Files\Google\Chrome\Application\chrome.exe',
              r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'):
        if p and Path(p).exists():
            return p
    return shutil.which('chrome') or shutil.which('msedge')


class _WS:
    """Just enough of RFC 6455 for the DevTools protocol (text frames, client masking)."""

    def __init__(self, url):
        host, rest = url.split('://', 1)[1].split('/', 1)
        h, port = host.split(':')
        self.s = socket.create_connection((h, int(port)), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode()
        self.s.sendall((f'GET /{rest} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n'
                        f'Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n').encode())
        buf = b''
        while b'\r\n\r\n' not in buf:
            buf += self.s.recv(4096)
        self.buf = buf.split(b'\r\n\r\n', 1)[1]

    def send(self, text):
        data = text.encode()
        n = len(data)
        head = bytes([0x81]) + (bytes([0x80 | n]) if n < 126 else
                                bytes([0x80 | 126]) + struct.pack('>H', n) if n < 65536 else
                                bytes([0x80 | 127]) + struct.pack('>Q', n))
        mask = os.urandom(4)
        self.s.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def _read(self, n):
        while len(self.buf) < n:
            chunk = self.s.recv(1 << 16)
            if not chunk:
                raise ConnectionError('devtools closed')
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def recv(self):
        parts = []
        while True:
            b0, b1 = self._read(2)
            n = b1 & 0x7f
            if n == 126:
                n = struct.unpack('>H', self._read(2))[0]
            elif n == 127:
                n = struct.unpack('>Q', self._read(8))[0]
            parts.append(self._read(n))
            if b0 & 0x80:
                return b''.join(parts).decode('utf-8', 'replace')


_JOB = None


def _kill_with_us(proc):
    """Chrome dies with this process: without this every run left its headless Chrome
    behind (46 of them on 2026-09-29, enough load to slow screen reads from 0.3 to 9 s)."""
    global _JOB
    try:
        import win32api
        import win32con
        import win32job
        if _JOB is None:
            _JOB = win32job.CreateJobObject(None, '')
            info = win32job.QueryInformationJobObject(_JOB, win32job.JobObjectExtendedLimitInformation)
            info['BasicLimitInformation']['LimitFlags'] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            win32job.SetInformationJobObject(_JOB, win32job.JobObjectExtendedLimitInformation, info)
        h = win32api.OpenProcess(win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE, False, proc.pid)
        win32job.AssignProcessToJobObject(_JOB, h)
    except Exception as e:
        print(f'chrome not tied to this process ({e}): it may outlive it')


def _free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class Browser:
    def __init__(self, width=1500, height=1200):
        port = _free_port()          # own port: several browsers in one process must not share one
        self.profile = tempfile.mkdtemp(prefix='tft-metatft-')
        self.proc = subprocess.Popen([_chrome(), '--headless=new', f'--remote-debugging-port={port}',
                                      f'--user-data-dir={self.profile}', f'--window-size={width},{height}',
                                      '--hide-scrollbars', '--mute-audio', 'about:blank'],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        _kill_with_us(self.proc)
        for _ in range(50):
            try:
                tabs = json.loads(urllib.request.urlopen(f'http://127.0.0.1:{port}/json', timeout=1).read())
                page = next(t for t in tabs if t.get('type') == 'page')
                break
            except Exception:
                time.sleep(0.2)
        else:
            raise RuntimeError('chrome did not start')
        self.ws = _WS(page['webSocketDebuggerUrl'])
        self.n = 0
        self.cmd('Page.enable')

    def cmd(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({'id': self.n, 'method': method, 'params': params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get('id') == self.n:
                if 'error' in m:
                    raise RuntimeError(m['error'])
                return m.get('result', {})

    def js(self, expr):
        r = self.cmd('Runtime.evaluate', expression=expr, awaitPromise=True, returnByValue=True)
        return (r.get('result') or {}).get('value')

    def wait(self, expr, timeout=20):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.js(expr):
                return True
            time.sleep(0.3)
        return False

    def shot(self, rect, quality=82):
        x, y, w, h = rect
        r = self.cmd('Page.captureScreenshot', format='jpeg', quality=quality, captureBeyondViewport=True,
                     clip={'x': x, 'y': y, 'width': w, 'height': h, 'scale': 1})
        return base64.b64decode(r['data'])

    def close(self):
        try:
            self.proc.kill()
        except Exception:
            pass
        shutil.rmtree(self.profile, ignore_errors=True)


def _rect_js(sel):
    return (f"(()=>{{const e={sel}; if(!e) return null; const r=e.getBoundingClientRect();"
            f"return [r.x+scrollX, r.y+scrollY, r.width, r.height];}})()")


def _open_list(b):
    b.cmd('Page.navigate', url=URL)
    # rows show skeleton placeholders until the stats arrive
    if not b.wait("[...document.querySelectorAll('.CompRow')].filter(r => /Avg Place/.test(r.textContent)).length > 5", 40):
        raise RuntimeError('MetaTFT comp list did not load')
    time.sleep(1.5)
    # the site's sticky menu bar (and any fixed banner) would cover the comp being shot
    b.js("[...document.querySelectorAll('body *')].forEach(e => { const p = getComputedStyle(e).position;"
         " if ((p === 'fixed' || p === 'sticky') && !e.closest('.CompRowWrapper')) e.style.setProperty('display', 'none', 'important'); })")


def comp_list_shots(max_shots=3):
    """The comp list (S and A tiers first), as 1-3 screenshots."""
    b = Browser()
    try:
        _open_list(b)
        total = b.js(_rect_js("document.querySelector('.CompListContainer')"))
        if not total:
            return []
        x, y, w, h = total
        step = 1600
        return [b.shot((x, y + i * step, w, min(step, h - i * step)))
                for i in range(min(max_shots, int(h // step) + 1))]
    finally:
        b.close()


def _flat(html_text):
    """A tab's HTML -> its visible words in page order, image labels in [brackets] ('[Infinity Edge]')."""
    import html as H
    import re
    toks = re.findall(r'alt="([^"]*)"|>([^<>]+)<', html_text or '')
    return [H.unescape(f'[{a}]' if a else t.strip()) for a, t in toks if a or t.strip()]


def _rows(tokens, start, per):
    """Table rows after the heading `start`: one or more [image] labels, then `per` numbers each."""
    import re
    t = tokens[tokens.index(start) + 1:] if start in tokens else []
    res, imgs, nums = [], [], []
    for x in t:
        if x.startswith('['):
            if nums:                      # a new row starts
                if len(nums) == per:
                    res.append((imgs, nums))
                imgs, nums = [], []
            imgs.append(x.strip('[]'))
        elif imgs and re.fullmatch(r'[-+(]?[\d.]+%?\)?', x):
            nums.append(x)
        elif imgs:
            break
    if imgs and len(nums) == per:
        res.append((imgs, nums))
    return res


def transcribe(quick, items, unit_items, builds, augments):
    """The comp's three tabs as plain text: only what the pages show. Arguments: each tab's (or
    sub-tab's) HTML; augments = [(pro variant, its Pro Augments and Tips HTML)]."""
    import html as H
    import re
    q = _flat(quick)
    out = []
    # 1. the board's hexes Hex_0..27 (Hex_0 = front row, left), each followed by its unit's name
    # (only polygons that hold a unit: empty hexes have ids too)
    pos = re.findall(r'<polygon[^>]*class="teambuilder_hex_unit[^"]*"[^>]*id="Hex_(\d+)"[^>]*>.*?'
                     r'class="HexBoardText"[^>]*>([^<]*)<', quick or '', re.S)
    out += ['1. Target final board, positions (row 0 = back row, row 3 = front row; columns 0-6 left to right):']
    out += [f'- {H.unescape(n)}: row {3 - int(k) // 7}, column {int(k) % 7}' for k, n in pos]
    # the header's trait badges: class 'TraitCompact <tier>', icon '.../traits/da_18_<trait>.png', then the count
    from tft_player.catalog import CATALOG
    key = lambda x: re.sub(r'[^a-z]', '', x.lower())
    known = {key(t): t for t in CATALOG.traits}
    tr = []
    from tft_player.trait_text import trait_line, trait_name
    for tier, icon, n in re.findall(r'TraitCompact (\w+)"[^>]*>.*?/traits/([^"&]+?)\.png.*?</div></div>(\d+)<', quick or '', re.S):
        # the icon is the trait's game-data id ('da_juggernaut18', 'da_18_zyrauniquetrait' = Thornmaiden);
        # else the longest trait name inside it
        hit = sorted((k for k in known if k in key(icon)), key=len)
        nm = trait_name(icon) or (known[hit[-1]] if hit else icon)
        tr.append((nm, int(n), tier))
    # what each trait does at each level, from the game data (the page shows it only on hover)
    out += ['', 'Traits of the target final board (count on that board, badge colour; effects from the game data):']
    for nm, n, tier in tr:
        out.append(f'- ({tier}) ' + (trait_line(nm, n) or f'{nm} {n}'))
    # 2. "Levelling: | Fast 8 | Lvl 4 | [Level Up] | - | Lvl 5 | [Level Up] | 2-5 | ... | [Rerolls] | 32.7"
    i = q.index('Levelling:') if 'Levelling:' in q else len(q)
    end = q.index('Carousel Priority') if 'Carousel Priority' in q else len(q)
    lv = []
    for j in range(i, end):
        if re.fullmatch(r'Lvl \d+', q[j]) and j + 2 < end and q[j + 2] != '-':
            lv.append(f'{q[j + 2]}: {q[j][4:]}' + (f' ({q[j + 4]} rolls)' if j + 4 < end and q[j + 3] == '[Rerolls]' else ''))
    out += ['', f"2. Levelling tempo: {q[i + 1] if i + 1 < len(q) else ''}",
            'Levelling (stage: level): ' + ', '.join(lv)]
    # 3. "[item] | xN" pairs
    car, c = [], q[end + 1:]
    for k, t in enumerate(c):
        if not t.startswith('['):
            if not t.startswith('x'):
                break
            continue
        car.append(t.strip('[]') + (f' {c[k + 1]}' if k + 1 < len(c) and c[k + 1].startswith('x') else ''))
    out += ['', '3. MetaTFT carousel priority (MetaTFT\'s order; xN = the count MetaTFT lists): ' + ', '.join(car)]
    it = _flat(items)
    pair = lambda p, a: bool(re.fullmatch(r'[\d.]+%', p) and re.fullmatch(r'[\d.]+', a))
    out += ['', '4. Unit play rate (avg place, played in % of games):']
    out += [f'- {im[0]}: avg {n[0]}, {n[1]}' for im, n in _rows(it, 'Unit Play Rate', 2)]
    out += ['', '5. Item play rate (avg place, played in % of games):']
    out += [f'- {im[0]}: avg {n[0]}, {n[1]}' for im, n in _rows(it, 'Item Play Rate', 2)]
    out += ['', "6. Units + items (avg place, share of the unit's games):"]
    out += [f'- {im[0]} with {im[1]}: avg {n[0]}, {n[1]}' for im, n in _rows(_flat(unit_items), 'Unit + Item Play Rate', 2)]
    out += ['', "7. Full builds (avg place, share of the unit's games):"]
    for im, n in _rows(_flat(builds), 'Unit Build Play Rate', 2):
        its = [x for x in im[1:] if x != 'Empty Slot']
        out.append(f"- {im[0]}: {' + '.join(its) or 'no item'}"
                   f"{' (other slots empty)' if 'Empty Slot' in im else ''}: avg {n[0]}, {n[1]}")
    # the completed items named in 5-7 and their components (game data: recipes can differ from earlier sets)
    from tft_player.items import RECIPES
    named = '\n'.join(out[out.index('5. Item play rate (avg place, played in % of games):'):])
    made = [i for i in sorted(RECIPES) if i in named]
    if made:
        out += ['', 'Recipes of the completed items in 5-7 (game data):']
        out += [f'- {i} = {" + ".join(RECIPES[i])}' for i in made]
    out += ['', '8. MetaTFT pro augment tier list for this comp:']
    tips = []
    for variant, h in augments:
        a = _flat(h)
        k = a.index('Pro Augments') + 1 if 'Pro Augments' in a else len(a)
        tiers, tier = {}, None
        while k < len(a) and not a[k].startswith('Pro Tips'):
            if a[k] in ('S', 'A', 'B', 'C', 'D'):
                tier = a[k]
            elif tier and a[k].startswith('['):
                v = tiers.setdefault(tier, [])
                if a[k].strip('[]') not in v:
                    v.append(a[k].strip('[]'))
            k += 1
        if tiers:
            out += [f'{variant}:'] + [f'- {t}: ' + ', '.join(v) for t, v in tiers.items()]
        tips.append((variant, [x for x in a[k + 1:] if x.startswith('-')]))
    out += ['', '9. Pro tips (text from the MetaTFT page):']
    for variant, ts in tips:
        out += [f'{variant}:'] + [f'  {t}' for t in ts]
    return '\n'.join(out)


def _paths(name):
    safe = ''.join(ch for ch in name if ch.isalnum() or ch in ' -').strip().replace(' ', '_')
    d = CACHE / safe
    return d, [d / f'{i}.jpg' for i in range(len(TABS))], d / 'brief.txt'


def comp_pages(name, max_age_h=MAX_AGE_H):
    """[(tab, jpeg)] for one comp: its box on the comp list with each of TABS open. Taken again, with
    the transcription (comp_brief), when older than max_age_h."""
    d, files, txt = _paths(name)
    if all(f.exists() and time.time() - f.stat().st_mtime < max_age_h * 3600 for f in files + [txt]):
        return [(t, f.read_bytes()) for t, f in zip(TABS, files)]
    b = Browser()
    try:
        _open_list(b)
        norm = "(s => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase())"
        # the exact title: "Lunar Aphelios" must not match "Lunar Aphelios Nidalee"
        row = (f"[...document.querySelectorAll('.CompRowWrapper')].find(w => {norm}((w.querySelector('.Comp_Title')||{{}}).textContent)"
               f" === {norm}({json.dumps(name)}))")
        # the list renders more rows as it is scrolled: scroll until the comp is there
        found = False
        for _ in range(40):
            if b.js(f"!!({row})"):
                found = True
                break
            b.js("window.scrollBy(0, window.innerHeight)")
            time.sleep(0.6)
        if not found or not b.js(f"(()=>{{const w={row}; if(!w) return false; w.scrollIntoView(); w.querySelector('.CompRow').click(); return true;}})()"):
            raise RuntimeError(f'comp {name!r} not on the MetaTFT list')
        b.wait(f"!!({row}).querySelector('a.TabSelection')", 15)
        html_of = lambda: b.js(f"({row}).innerHTML")

        def click(text, leaf=False):
            sel = (f"[...({row}).querySelectorAll('*')].find(e=>e.children.length===0 && e.textContent.trim()==={json.dumps(text)})"
                   if leaf else f"[...({row}).querySelectorAll('a.TabSelection')].find(a=>a.textContent.trim()==={json.dumps(text)})")
            b.js(f"(()=>{{const a={sel}; if(a) a.click();}})()")
            time.sleep(1.8)                   # charts render after the click

        out, pages = [], {}
        d.mkdir(parents=True, exist_ok=True)
        for i, tab in enumerate(TABS):
            click(tab)
            img = b.shot(b.js(_rect_js(row)))
            out.append((tab, img))
            pages[tab] = html_of()
            if tab == 'Units & Items':        # the sub-tabs behind the screenshot's default 'Items' view
                click('Unit + Items', leaf=True)
                pages['Unit + Items'] = html_of()
                click('Full Builds', leaf=True)
                pages['Full Builds'] = html_of()
            if tab == 'Pro Augments and Tips':  # the pro variant the screenshot shows (the default one;
                # a comp with one variant has a title instead of a dropdown)
                shown = b.js(f"(()=>{{const r=({row}), s=r.querySelector('select'), t=r.querySelector('.CompProTitle');"
                             f" return s? s.options[s.selectedIndex].textContent.trim(): t? t.textContent.trim(): ''}})()")
                pages['augments'] = [(shown or '', pages[tab])]
        text = transcribe(pages['Options & Quick Start'], pages['Units & Items'], pages['Unit + Items'],
                          pages['Full Builds'], pages['augments'])
        for f, (_, img) in zip(files, out):
            f.write_bytes(img)
        for old in d.glob('*.jpg'):           # tabs no longer sent (3.jpg, 4.jpg of the 5-tab version)
            if old not in files:
                old.unlink()
        txt.write_text(text, encoding='utf-8')
        (d / 'pages.json').write_text(json.dumps(pages), encoding='utf-8')     # the HTML, for offline checks
        return out
    finally:
        b.close()


def comp_brief(name, max_age_h=MAX_AGE_H):
    """The transcription of the comp's screenshots (taken together with them)."""
    comp_pages(name, max_age_h)
    return _paths(name)[2].read_text(encoding='utf-8')


if __name__ == '__main__':
    import sys
    n = sys.argv[1] if len(sys.argv) > 1 else 'Lunar Aphelios Nidalee'
    t0 = time.time()
    shots = comp_pages(n, max_age_h=0)
    print(f'{n}: {len(shots)} tabs in {time.time() - t0:.1f}s ->', CACHE)
    print(comp_brief(n))
