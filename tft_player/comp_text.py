"""One comp's MetaTFT page as short text with its numbers, fetched live.

The same data as the comp's page on metatft.com (units, items, traits, early
boards, levelling, counters), from MetaTFT's API: only the significant rows - units,
items and traits played in at least MIN_PLAY of the comp's games (at most MAX_ROWS
each), builds seen in at least MIN_GAMES games - each with its average place and rate.
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from tft_player.meta import META

COMPS = 'https://api-hc.metatft.com/tft-comps-api/'
FILTER = ('queue=1100&patch=current&days=3&rank=CHALLENGER,DIAMOND,EMERALD,GRANDMASTER,MASTER,PLATINUM'
          '&permit_filter_adjustment=true')
HEADERS = {'User-Agent': 'Mozilla/5.0', 'Origin': 'https://www.metatft.com', 'Referer': 'https://www.metatft.com/'}
TRAITS_MD = Path(__file__).resolve().parent.parent / 'data' / 'reference' / 'traits.md'

MIN_PLAY = 0.2          # units, items, traits in fewer of the comp's games are left out
MIN_GAMES = 15          # boards and builds seen in fewer games are noise
MAX_ROWS = 10           # at most this many units, items, traits


def _get(url):
    req = urllib.request.Request(url, headers=HEADERS)
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def _names():
    lookup = _get(f'https://data.metatft.com/lookups/{_get(COMPS + "latest_cluster_id")["tft_set"]}_latest_en_us.json')
    names = {}
    for u in lookup['units']:
        for a in [u['apiName'], *(u.get('assetNames') or [])]:
            names[a] = u['name']
    for kind in ('traits', 'items'):
        names.update({t['apiName']: t['name'] for t in lookup[kind]})
    return names


def _breakpoints():
    """trait name -> its breakpoints (MetaTFT counts trait levels 1, 2, ...)."""
    out, cur = {}, None
    for line in TRAITS_MD.read_text(encoding='utf-8').splitlines():
        if line.startswith('## '):
            cur = line[3:].strip()
        elif cur and line.startswith('Breakpoints:'):
            out[cur] = json.loads(line.split(':', 1)[1])
    return out


def _hex(cell):
    k = int(str(cell).split('_')[-1]) - 1          # cells 1-7: back row = our row 0
    return k // 7, k % 7


def comp_text(comp) -> str:
    d = _get(f'{COMPS}comp_details?comp={comp.id}&cluster_id={comp.id[:3]}&' + FILTER)['results']
    n = _names()
    name = lambda a: n.get(a, a)
    total = (d.get('overall') or {}).get('count') or 1
    pct = lambda x: f'{x:.0%}' if x >= 0.1 else f'{x:.1%}'
    out = [f'{comp.name} - MetaTFT, ranked Platinum+, last 3 days: {total} games, '
           f"avg place {(d.get('overall') or {}).get('avg', 0):.2f}"
           + (f', top 4 {comp.top4:.0%}, win {comp.win:.0%}' if comp.win is not None else '')
           + f'. Units: {", ".join(comp.units)}; carries: {", ".join(comp.carries)}.']

    lv = [f"{x['stage']}-{x['round']}: {x['level']}" for x in d.get('levels') or [] if x.get('stage')]
    if lv:
        out += ['', 'Levelling (stage: level): ' + ', '.join(lv)]
    fl = sorted(d.get('final_levels') or [], key=lambda x: int(x['level']))
    fl_n = sum(x['count'] for x in fl) or 1
    fl = [f"level {x['level']}: {pct(x['count'] / fl_n)} of games, avg {x['avg']:.2f}" for x in fl
          if x['count'] / fl_n >= 0.05]
    if fl:
        out.append('Final level: ' + '; '.join(fl))
    rr = [f"level {k}: {r['rerolls'] / r['matches']:.1f}" for k, r in sorted((d.get('rerolls') or {}).items(), key=lambda kv: int(kv[0]))
          if int(k) >= 5 and r.get('matches', 0) >= MIN_GAMES and r['rerolls'] / r['matches'] >= 1]
    if rr:
        out.append('Rerolls per game at: ' + ', '.join(rr))

    out += ['', 'Units (avg place, played in % of games; by star level: share, avg place):']
    for u in sorted(d.get('unit_stats') or [], key=lambda u: -u['pcnt'])[:MAX_ROWS]:
        if u['pcnt'] < MIN_PLAY:
            break
        tiers = ', '.join(f"{t['tier']}* {pct(t['pcnt'])} avg {t['avg']:.2f}" for t in u.get('tiers') or []
                          if t['count'] >= MIN_GAMES)
        out.append(f"- {name(u['unit'])}: avg {u['avg']:.2f}, {pct(u['pcnt'])} | {tiers}")

    out += ['', 'Items (avg place, played in % of games; top 3 holders by MetaTFT: avg place (change vs the unit), share of the item):']
    for it in sorted(d.get('itemNames') or [], key=lambda i: -i['pcnt'])[:MAX_ROWS]:
        if it['pcnt'] < MIN_PLAY:
            break
        holders = ', '.join(f"{name(h['units'])} {h['avg']:.2f} ({h['place_change']:+.2f}) {pct(h['item_pick'])}"
                            for h in (it.get('units') or [])[:3] if h['count'] >= MIN_GAMES)
        out.append(f"- {name(it['itemNames'])}: avg {it['avg']:.2f}, {pct(it['pcnt'])} | {holders}")

    core = {u['unit'] for u in sorted(d.get('unit_stats') or [], key=lambda u: -u['pcnt'])[:MAX_ROWS]
            if u['pcnt'] >= MIN_PLAY}
    builds = {}
    for b in d.get('builds') or []:
        if b.get('unit') in core and b.get('count', 0) >= MIN_GAMES and b.get('num_items') == 3:
            builds.setdefault(b['unit'], []).append(b)
    if builds:
        out += ['', '3-item builds, lowest avg place first (avg place, games, change vs the unit):']
        for u, bs in builds.items():
            for b in sorted(bs, key=lambda b: b['avg'])[:2]:
                out.append(f"- {name(u)}: {' + '.join(name(i) for i in b['buildName'])} "
                           f"(avg {b['avg']:.2f}, {b['count']} games, {b.get('place_change', 0):+.2f})")

    bp = _breakpoints()
    out += ['', 'Traits (played in % of games; by level: avg place, share):']
    for t in sorted(d.get('traits') or [], key=lambda t: -t['count'])[:MAX_ROWS]:
        if t['count'] / total < MIN_PLAY:
            break
        tn = name(t['trait'])
        steps = bp.get(tn, [])
        lvls = ', '.join(f"{steps[x['level'] - 1] if x['level'] <= len(steps) else x['level']}: "
                         f"avg {x['avg']:.2f}, {pct(x['count'] / t['count'])}"
                         for x in sorted(t.get('levels') or [], key=lambda x: x['level']) if x['count'] >= MIN_GAMES)
        out.append(f"- {tn}: {pct(t['count'] / total)} | {lvls}")

    pos = []
    for u, v in ((d.get('positioning') or {}).get('units') or {}).items():
        if u in core:
            cells = sorted(v.get('positions') or [], key=lambda c: -c['count'])[:2]
            pos.append(f"{name(u)} {' or '.join(str(_hex(c['cell'])) for c in cells)}")
    if pos:
        out += ['', 'Positions (row, col; row 0 = back row): ' + '; '.join(pos)]

    ctr = sorted((c for c in d.get('counters') or [] if c['place_change'] >= 0.1), key=lambda c: -c['place_change'])[:3]
    if ctr:
        out += ['', 'Hardest opponents (our avg place change when they are in the lobby): ' + ', '.join(
            f"{(META.comps.get(str(c['against'])) or type('', (), {'name': c['against']})).name} {c['place_change']:+.2f}"
            for c in ctr)]

    notes = [((p.get('content') or {}).get('content') or {}).get('notes', '').strip() for p in d.get('proComps') or []]
    notes = [x for x in notes if x]
    if notes:
        out += ['', 'Pro notes:'] + notes[:2]
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(out))
