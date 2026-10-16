"""Snapshot the current MetaTFT meta into data/meta/current.json.

Run before a session (the meta moves over days, not minutes); the player only
reads the local file, never the network.

    python tools/fetch_meta.py

Contents:
  augments  name -> tier (expert list S..D), rarity, tags, resolved description
  comps     id -> name, units, traits, levelling, avg placement + games (last
            3 days, Plat+), per-comp augment tiers, best item builds per unit
MetaTFT no longer publishes augment placement stats; the tiers are its
expert-maintained lists (overall and per comp).
"""
from __future__ import annotations

import ast
import json
import re
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / 'data' / 'meta' / 'current.json'
HEADERS = {'User-Agent': 'Mozilla/5.0', 'Origin': 'https://www.metatft.com',
           'Referer': 'https://www.metatft.com/'}
STAT = 'https://api-hc.metatft.com/tft-stat-api/'
COMPS = 'https://api-hc.metatft.com/tft-comps-api/'
FILTER = 'queue=1100&patch=current&days=3&rank=CHALLENGER,DIAMOND,EMERALD,GRANDMASTER,MASTER,PLATINUM' \
         '&permit_filter_adjustment=true'


def get(url):
    req = urllib.request.Request(url, headers=HEADERS)
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def resolve_desc(desc, values):
    """<TFTCurveTable row="X" format="percent"/> -> the value at the first level."""
    def sub(m):
        row = re.search(r'row="([^"]+)"', m.group(0))
        v = (values or {}).get(row.group(1)) if row else None
        if not v:
            return 'N'
        x = v[0][1]
        if 'percentMinusOne' in m.group(0):
            return f'{round((x - 1) * 100)}%'
        if 'percent' in m.group(0):
            return f'{round(x * 100)}%'
        return f'{x:g}'
    desc = re.sub(r'<TFTCurveTable[^>]*/>', sub, desc or '')
    desc = re.sub(r'<br\s*/?>', ' ', desc)
    desc = re.sub(r'<[^>]+>|\{[^}]*\}', '', desc)
    return re.sub(r'\s+', ' ', desc).strip()


def shrunk(row, prior=4.3, weight=500):
    """Average placement pulled towards the prior: 3.3 over 78 games is weak evidence."""
    return (row['avg'] * row['games'] + prior * weight) / (row['games'] + weight)


def cell_hex(cell):
    """MetaTFT 'cell_1'..'cell_28' -> our (row, col). Cells 1-7 are the back row
    (carries sit there, tanks in 22-28), i.e. our row 0."""
    k = int(str(cell).split('_')[-1]) - 1
    return k // 7, k % 7


def item_kind(i):
    """An item's kind as metatft.com sorts it (its comps page code): emblem, artifact, support, trait or ''."""
    tags, api, parts = i.get('tags') or [], i.get('apiName') or '', i.get('composition') or []
    for tag, kind in (('Emblem', 'emblem'), ('Artifact', 'artifact'), ('Support', 'support'), ('Trait', 'trait')):
        if tag in tags:
            return kind
    if 'Emblem' in api or 'TFT_Item_Spatula' in parts or 'TFT_Item_FryingPan' in parts:
        return 'emblem'
    return ''


def situational(c, emblems, item):
    """Why metatft.com hides this comp by default, or None. Its "Situational Comps" toggle (off by default) hides
    comps with a hero augment in their name or an emblem in at least half of their games (a required emblem), and
    the page never lists a comp no unit or trait defines (centroid max below 1). Same rules as its comps page."""
    if 'Augment' in (c.get('name_string') or ''):
        return 'hero augment'
    tops = c.get('top_itemNames') or []
    try:
        tops = ast.literal_eval(tops) if isinstance(tops, str) else tops
    except (ValueError, SyntaxError):
        tops = []
    for t in tops:
        if isinstance(t, dict) and (t.get('pcnt') or 0) >= 0.5 and t.get('itemNames') in emblems:
            return f"needs {item.get(t['itemNames'], t['itemNames'])}"
    cen = c.get('centroid') or []
    try:
        cen = json.loads(cen) if isinstance(cen, str) else cen
    except ValueError:
        cen = []
    if cen and max(cen) < 1:
        return 'no defining unit or trait'
    return None


def comp_guide(cid, unit, item, trait):
    """The comp page a player reads after committing: levels, boards per level,
    positions, star goals, items, traits, pro notes - condensed for the model."""
    d = get(f'{COMPS}comp_details?comp={cid}&cluster_id={cid[:3]}&' + FILTER)['results']
    names = lambda s: [unit.get(x, x) for x in s.split('&') if x and 'Elderwood18_' not in x]
    levels = [(f"{x['stage']}-{x['round']}", x['level']) for x in d.get('levels') or [] if x.get('stage')]
    def best(opts, k=2, min_count=15):
        good = [o for o in opts if o.get('count', 0) >= min_count] or opts
        return [{'units': names(o.get('unit_list') or o.get('units_list') or ''), 'avg': round(o['avg'], 2),
                 'games': o['count']} for o in sorted(good, key=lambda o: o['avg'])[:k]]
    early = {lv: [{'units': names(o.get('unit_list') or ''), 'avg': round(o['avg'], 2), 'games': o['count'],
                   'round_win': round(o.get('win') or 0, 3)} for o in v[:2]]
             for lv, v in (d.get('early_options') or {}).items()}
    final = {lv: best(v, 1) for lv, v in (d.get('options') or {}).items()}
    pos = {}
    for u, v in ((d.get('positioning') or {}).get('units') or {}).items():
        cells = sorted(v.get('positions') or [], key=lambda c: -c['count'])[:2]
        pos[unit.get(u, u)] = [cell_hex(c['cell']) for c in cells]
    stars = {}
    for u in d.get('unit_stats') or []:
        tiers = {t['tier']: t for t in u.get('tiers') or []}
        if 3 in tiers and tiers[3]['pcnt'] >= 0.15:
            stars[unit.get(u['unit'], u['unit'])] = 3
        elif 2 in tiers and tiers[2]['pcnt'] >= 0.5:
            stars[unit.get(u['unit'], u['unit'])] = 2
    items = [item.get(i['itemNames'], i['itemNames']) for i in (d.get('itemNames') or [])[:8]]
    traits = []
    for t in (d.get('traits') or [])[:5]:
        lv = sorted(t.get('levels') or [], key=lambda x: x['level'])
        traits.append(f"{trait.get(t['trait'], t['trait'])}: " +
                      ', '.join(f"{x['level']}->{x['avg']:.2f}" for x in lv if x['count'] >= 50))
    notes = []
    for p in (d.get('proComps') or [])[:2]:
        c = p.get('content') or {}
        txt = ((c.get('content') or {}).get('notes') or '').strip()
        if txt:
            notes.append(f"{(c.get('metadata') or {}).get('title', '')}: {txt[:300]}")
    fl = d.get('final_levels') or []
    n_fl = sum(x['count'] for x in fl) or 1
    final_levels = {x['level']: round(x['count'] / n_fl, 3) for x in fl}
    rolls = {lv: round(r['rerolls'] / r['matches'], 1) for lv, r in (d.get('rerolls') or {}).items()
             if int(lv) >= 5 and r.get('matches')}
    return {'levels': levels, 'early': early, 'final': final, 'positions': pos, 'stars': stars,
            'final_levels': final_levels, 'rolls': rolls,
            'items': items, 'traits': traits, 'pro_notes': notes}


def main():
    lookup = get(f'https://data.metatft.com/lookups/{get(COMPS + "latest_cluster_id")["tft_set"]}_latest_en_us.json')
    patch = get(STAT + 'patch')
    tiers = get(STAT + 'augments_tiers')['content']['content']['tierList']
    comps_data = get(COMPS + 'comps_data?queue=1100')['results']['data']['cluster_details']
    comps_stats = get(COMPS + 'comps_stats?' + FILTER)['results']
    comp_aug = get(COMPS + 'comp_augment_tiers')['results']

    unit = {}
    for u in lookup['units']:
        for a in [u['apiName'], *(u.get('assetNames') or [])]:
            unit[a] = u['name']
    trait = {t['apiName']: t['name'] for t in lookup['traits']}
    item = {i['apiName']: i['name'] for i in lookup['items']}
    aug_name = {a['apiName']: a['name'] for a in lookup['augments']}

    augments = {}
    for a in lookup['augments']:
        augments.setdefault(a['name'], {
            'rarity': a.get('rarity'), 'tags': a.get('manual_tags') or [], 'kind': a.get('type'),
            'desc': resolve_desc(a.get('desc'), a.get('curveValues')), 'tier': None})
    for row in tiers:
        for e in row['content']:
            n = aug_name.get(e['id'])
            if n in augments and augments[n]['tier'] is None:
                augments[n]['tier'] = row['label']     # same name twice: keep the better row

    total = get('https://api-hc.metatft.com/tft-explorer-api/augments?' + FILTER)[0]['total_games']
    recent, rates = {}, {}
    for s in comps_stats:
        if s.get('cluster') and s.get('count'):
            p = s['places'][:8]
            n = max(sum(p), 1)
            recent[s['cluster']] = (sum((i + 1) * c for i, c in enumerate(p)) / n, sum(p))
            rates[s['cluster']] = {'win': round(p[0] / n, 4), 'top4': round(sum(p[:4]) / n, 4),
                                   'pick': round(sum(p) / (total / 8), 3)}

    emblems = {i['apiName'] for i in lookup['items'] if item_kind(i) == 'emblem'}
    comps = {}
    for cid, c in comps_data.items():
        def names(s, table):
            return [table.get(x.strip(), x.strip()) for x in (s or '').strip('"').split(',') if x.strip()]
        units = names(c.get('units_string'), unit)
        traits = []
        for t in (c.get('traits_string') or '').strip('"').split(','):
            m = re.fullmatch(r'\s*(.+)_(\d+)\s*', t)
            if m:
                traits.append([trait.get(m.group(1), m.group(1)), int(m.group(2))])
        label = c.get('name') or []
        label = json.loads(label) if isinstance(label, str) else label
        title = [unit.get(n['name']) or trait.get(n['name']) or n['name'] for n in label] or units[:2]
        builds = {}
        for b in c.get('builds') or []:
            u = unit.get(b['unit'], b['unit'])
            builds.setdefault(u, []).append({'items': [item.get(i, i) for i in b['buildName']],
                                             'avg': b['avg'], 'games': b['count']})
        avg, games = recent.get(cid, ((c.get('overall') or {}).get('avg'), (c.get('overall') or {}).get('count')))
        ca = comp_aug.get(cid) or {}
        comps[cid] = {
            'name': ' '.join(title), 'units': units, 'traits': traits,
            'carries': [unit.get(n['name'], n['name']) for n in label if n.get('type') == 'unit'],
            'levelling': c.get('levelling'), 'avg': round(avg, 3) if avg else None, 'games': games,
            **rates.get(cid, {}),
            'augments': {aug_name.get(a['id'], a['id']): a['tier'] for a in ca.get('augments') or []},
            'builds': {u: sorted(v, key=shrunk)[:3] for u, v in builds.items()},
            'situational': situational(c, emblems, item),
        }

    for n, cid in enumerate(comps):
        try:
            comps[cid]['guide'] = comp_guide(cid, unit, item, trait)
        except Exception as e:
            print(f'guide {cid} failed: {e}')
        if n % 10 == 9:
            print(f'guides {n + 1}/{len(comps)}')
        time.sleep(0.3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        'source': 'MetaTFT', 'fetched_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'set': lookup['_metadata'].get('set'), 'patch': patch.get('patch', '') + patch.get('b_patch_version', ''),
        'filters': 'Ranked, Platinum+, last 3 days', 'augments': augments, 'comps': comps,
    }, ensure_ascii=False, indent=1), encoding='utf-8')
    tiered = sum(1 for a in augments.values() if a['tier'])
    print(f'{OUT}: patch {patch.get("patch")}{patch.get("b_patch_version", "")}, '
          f'{len(augments)} augments ({tiered} tiered), {len(comps)} comps')


if __name__ == '__main__':
    main()
