"""Write the model's reference files: game data + MetaTFT data as plain files it
may read or search during a game (data/reference). Resources, not instructions.

    python tools/build_reference.py        (after tools/fetch_meta.py)
"""
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tft_player.catalog import CATALOG  # noqa: E402
from tft_player.items import RECIPES  # noqa: E402
from tft_player.meta import META  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / 'data' / 'reference'
SOURCES = Path(__file__).resolve().parent.parent / 'data' / 'sources'


def cdragon():
    """The newest Community Dragon TFT data (data/sources/cdragon-tft-<date>.json)."""
    return json.loads(sorted(SOURCES.glob('cdragon-tft-*.json'))[-1].read_text(encoding='utf-8'))


def clean(text):
    text = re.sub(r'<[^>]+>', '', text or '')
    return re.sub(r'\s+', ' ', text).strip()


MECHANICS = """# Game mechanics (Set 18)

Sources: op.gg Set 18 gold/XP guide, bunnymuffins Set 18 levelling guide (shop odds, pools),
League wiki (selling, streaks), TFTSense / NoxuTFT / Mobalytics Set 18 Wisp guides, and our own
games (marked "seen"). Where sources disagree it says so.

## Rounds
- Stage 1 is against monsters (PvE) and pays loot. From stage 2 each stage is rounds x-1 to x-7:
  player fights, one carousel (x-4) and a monster round (x-7) that drops loot.
- Carousel: 1-1, then 2-4, 3-4 (seen) and x-4 each stage. Lowest HP picks first.
- Augment choices at 2-1 and 3-2 (seen) and 4-2. Three options, each rerollable once (seen).
- Planning phase: about 30 seconds (seen). Losing a player fight costs HP (stage damage plus
  damage per surviving enemy unit); winning costs nothing. Out at 0 HP.

## Gold
- Round income: 1-2: 2, 1-3: 2, 1-4: 3, 2-1: 4, then 5 gold every round.
- Interest: +1 per 10 gold held (counted before the round income), up to +5 at 50 gold.
- PvP win: +1 gold. PvP loss: 0.
- Streak bonus (wins or losses in a row): 5 in a row +2, 6 or more +3. For shorter streaks the
  sources differ: op.gg (Set 18) gives +1 from 2 in a row; the wiki gives +1 from 3 in a row.
  The streak number is on screen next to the gold (a blue flame = loss streak, orange = win).

## Level and XP
- Buying XP: 4 gold for 4 XP. Passive XP each round is not confirmed by these sources.
- XP needed to reach each level: 3: 2, 4: 6, 5: 10, 6: 20, 7: 36, 8: 60, 9: 68, 10: 68.
  (Levels 3-6 also seen in our games.)
- Level = the number of units you can field.

## Shop
- 5 cards. Rerolling costs 2 gold. A new shop comes each round unless locked.
- Odds by level (1/2/3/4/5-cost %): L1 100/0/0/0/0, L2 75/25/0/0/0, L3 75/25/0/0/0,
  L4 55/30/15/0/0, L5 45/33/20/2/0 (also seen on screen), L6 30/40/25/5/0, L7 16/30/43/10/1,
  L8 15/20/32/30/3, L9 10/17/25/33/15, L10 5/10/20/40/25.
- Copies of each champion in the shared pool: 1-cost 30, 2-cost 25, 3-cost 18, 4-cost 10,
  5-cost 9. Copies other players hold are not in your shops.

## Units
- Three copies at the same star level merge into the next star; a buy that completes a set
  merges at once, even with a full bench.
- Selling: a 1-star returns its cost. A 2- or 3-star returns the gold of all its copies minus 1
  (e.g. a 3-cost 2-star sells for 8), except 1-cost units, which return the full amount.
- Bench: 9 slots. Board: rows 0-3 (row 0 is the back row next to the bench, row 3 the front row
  facing the enemy), columns 0-6 left to right. Bench units don't fight or count for traits.

## Traits
- A trait counts each different champion on the board once; duplicate copies don't add to it.
- Breakpoints and effects: traits.md. Emblems add a trait to the unit holding them.

## Items
- Components come from loot and carousels. A second component on a unit holding one combines
  them (recipes: items.md). A unit holds at most three items. Selling a unit returns its items.

## Wisps
- Every player gets them: a Wisp takes the rightmost shop slot in about every other shop.
- You can buy at most ONE Wisp per round (Blossom at high counts: two).
- A shop refresh rerolls the Wisp too. When you buy a Wisp, or when planning ends, the other
  Wisps vanish and the champion behind the slot is shown; it can still be bought.
- Cost and power rise with the stage: from 0 gold early to 45+ late (seen: 0-3 gold in stage 2).
  From stage 5 every other Wisp is a combat Wisp.
- The shop only offers Wisps you can afford with the gold you hold when the shop is drawn (patch
  18.4 notes): with little gold only cheap Wisps appear. Under 10 gold, from 4-1, the new 0-cost
  Wisps (Drizzle, Grove Armor, Returning Friend, Room for a Shroom) can appear.
- About 200 Wisps in 7 kinds: champion, combat, shop, gold/XP, item, risky, misc. Every Wisp and
  its effect (base and upgraded / Radiant): wisps.md.
- Blossom improves Wisps by breakpoint: empowered after combat, upgraded Wisps, a Wisp in every
  shop, gold for each Wisp bought, two Wisps per round (traits.md, Blossom).
- Augments that pay per Wisp: Wisp Rebate, Residual Magic, Blossom's Call (augments.md).
"""


def fill(desc, effects):
    """Put an effect's known numbers into its description; unknown variables become N."""
    def num(m):
        name, mult = m.group(1), m.group(2)
        v = effects.get(name)
        if not isinstance(v, (int, float)):
            return 'N'
        v = v * float(mult) if mult else v
        return f'{v:g}' if abs(v - round(v)) > 1e-6 else str(int(round(v)))
    return clean(re.sub(r'@([\w{}]+)(?:\*([\d.]+))?@', num, desc or '').replace('%i:', ''))


def wisps_text(raw):
    """Every Wisp in the Set: the non-champion shop cards (CDragon 'DA_' entries with the Set 18
    mechanic icon), base and upgraded version side by side."""
    src = cdragon()
    wisps = {}
    for it in src['items']:
        api = it.get('apiName') or ''
        if not api.startswith('DA_') or 'set18_mechanicicon' not in (it.get('icon') or ''):
            continue
        kind = 'upgraded' if 'upgrade' in api.lower() else 'prismatic' if 'prismatic' in api.lower() else 'base'
        text = fill(it.get('desc'), it.get('effects') or {})
        wisps.setdefault(it.get('name') or api, {}).setdefault(kind, text)
    lines = ['# Wisps', '',
             'Non-champion shop cards. Buying one applies the effect below. N = a number the game data',
             'does not give; the card in the shop shows the real value and the cost.',
             'Upgraded = the Radiant version (e.g. from the Blossom trait).', '',
             '| Wisp | Effect | Upgraded |', '|---|---|---|']
    for name, v in sorted(wisps.items()):
        base = v.get('base') or v.get('prismatic') or ''
        up = v.get('upgraded') or ''
        lines.append(f"| {name} | {base} | {up if up != base else 'same text, stronger numbers'} |")
    # Wisps seen in our own games (card text as read on screen) that the game data lacks
    seen = {}
    for log in (Path(__file__).resolve().parent.parent / 'runs').glob('*/log.jsonl'):
        for m in re.finditer(r"WISP '([^']+)' \(([0-9?]+)g\): ([^\\\n]+?)(?:\\n|\")",
                             log.read_text(encoding='utf-8', errors='replace').replace("SPECIAL CARD '", "WISP '")):
            name, cost, desc = m.groups()
            if name not in wisps and len(desc) > 8:
                seen.setdefault(name, (cost, desc.strip()))
    if seen:
        lines += ['', '## Seen in our games, missing from the game data', '',
                  '| Wisp | Cost seen | Card text (read on screen) |', '|---|---|---|']
        lines += [f'| {n} | {c} | {d} |' for n, (c, d) in sorted(seen.items())]
    return '\n'.join(lines)


STAT_NAMES = {'AD': 'Attack Damage', 'AP': 'Ability Power', 'AS': 'Attack Speed', 'Armor': 'Armor',
              'MagicResist': 'Magic Resist', 'Health': 'Health', 'Mana': 'Mana', 'CritChance': 'Crit Chance',
              'StatOmnivamp': 'Omnivamp', 'BonusDamage': 'Damage Amp'}


def tooltip(name):
    """A MetaTFT board entry that is a game tooltip, not a unit: a sentence, not a name."""
    return not CATALOG.unit(name) and (len(name.split()) > 4 or name.rstrip()[-1:] in '!.?')


def ability_text(desc, variables):
    """A champion's ability with the game data's numbers: 1/2/3-star values where they differ; N = a number
    the data does not give (most Set 18 abilities: their damage is computed in-game)."""
    var = {v['name'].lower(): v.get('value') or [] for v in variables or [] if v.get('name')}

    def num(m):
        vals = var.get(m.group(1).lower())
        if not vals or len(vals) < 4:
            return 'N'
        k = float(m.group(2)) if m.group(2) else 1.0
        stars = [round(v * k, 2) for v in vals[1:4]]
        stars = [int(v) if v == int(v) else v for v in stars]
        return str(stars[0]) if len(set(stars)) == 1 else '/'.join(map(str, stars))
    from tft_player.trait_text import ICONS
    text = re.sub(r'@([A-Za-z0-9_]+)(?:\*([\d.]+))?@', num, desc or '')
    text = re.sub(r'%i:scale(\w+)%', lambda m: f"({ICONS.get(m.group(1), m.group(1))})", text)
    return clean(text.replace('\\n', ' '))


def items_text():
    """Every component and completed item: recipe, stats and what it does (game data)."""
    from tft_player.items import COMPONENTS
    src = cdragon()
    by_name = {}
    for it in src['items']:
        api = it.get('apiName') or ''
        n = it.get('name')
        if not n or not it.get('desc'):
            continue
        # the standard item entry (TFT_Item_...) wins over encounter / augment copies
        if n not in by_name or (api.startswith('TFT_Item_') and not by_name[n]['apiName'].startswith('TFT_Item_')):
            by_name[n] = it

    def describe(n):
        it = by_name.get(n)
        if not it:
            return '', ''
        eff = it.get('effects') or {}
        stats = []
        for k, label in STAT_NAMES.items():
            v = eff.get(k)
            if isinstance(v, (int, float)) and v:
                frac = v < 1 and k in ('AD', 'BonusDamage', 'CritChance', 'StatOmnivamp')
                pct = frac or k in ('AS', 'CritChance', 'StatOmnivamp')
                v = round(v * 100, 1) if frac else v
                stats.append(f"{v:g}{'%' if pct else ''} {label}")
        text = fill(re.sub(r'<tftitemrules>.*?</tftitemrules>', '', it['desc'], flags=re.S), eff)
        text = re.sub(r'%i:\w+%|@TFTUnitProperty[^@]*@|TFT\w+%', '', text)
        text = re.sub(r'\{\{TFT_Keyword_(\w+)\}\}', '', text).strip()
        return ', '.join(stats), text

    lines = ['# Items', '', 'Stats and effects from the game data (Community Dragon). N = a number the data does not',
             'give. Two components on one unit combine into the item.', '', '## Components', '',
             '| Component | Stats |', '|---|---|']
    for c in sorted(COMPONENTS):
        st, text = describe(c)
        lines.append(f'| {c} | {st or text} |')
    lines += ['', '## Completed items', '', '| Item | Recipe | Stats | Effect |', '|---|---|---|---|']
    for item, pair in sorted(RECIPES.items()):
        st, text = describe(item)
        lines.append(f'| {item} | {pair[0]} + {pair[1]} | {st} | {text} |')
    return '\n'.join(lines)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    # empty the folder, not remove it: a model session may have it open as its working directory
    for p in (OUT.iterdir() if OUT.exists() else ()):
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    (OUT / 'metatft' / 'comps').mkdir(parents=True, exist_ok=True)
    raw = json.loads((Path(__file__).resolve().parent.parent / 'data' / 'factual-catalogue-18.json').read_text(encoding='utf-8'))
    w = lambda name, text: (OUT / name).write_text(text.strip() + '\n', encoding='utf-8')

    # patch changes newer than the game data (data/sources/patch-*.md, from the patch notes)
    patches = sorted((Path(__file__).resolve().parent.parent / 'data' / 'sources').glob('patch-*.md'))
    for p in patches:
        shutil.copy(p, OUT / p.name)
    newer = ''.join(f'- {p.name} - patch {p.stem[6:]} changes; newer than the files below where they differ\n'
                    for p in patches)

    w('README.md', f"""# Reference files - TFT Set {raw['set']}, patch {META.patch}

{newer}
Game data (Community Dragon: {sorted(SOURCES.glob('cdragon-tft-*.json'))[-1].stem[12:]}; champion list
{raw['source_date']}):
- game-mechanics.md - rounds, economy (income, interest, streaks, XP, rolling, selling),
  merging, traits, items, board, augments, carousel, Wisps
- wisps.md - every Wisp (non-champion shop card): what it does, base and upgraded
- champions.md - every champion: cost, traits, stats, ability
- traits.md - every trait: breakpoints and effects
- items.md - item recipes (component + component = item)
- augments.md - every augment: description, MetaTFT tier

MetaTFT data ({META.fetched_at}, ranked Platinum+, last 3 days):
- metatft/comps.md - all comps: average placement, games, levelling, units, carries
- metatft/comps/<id>.md - one comp's page: levelling timings,
  final boards, unit positions, star goals, item builds, trait levels, augments, pro notes
""")

    w('game-mechanics.md', MECHANICS)
    w('wisps.md', wisps_text(raw))

    game = cdragon()['sets'][str(raw['set'])]
    abilities = {re.sub(r'\s*\((\w+)\)', r' \1', c['name']): c.get('ability') or {} for c in game['champions']}
    lines = ['# Champions', '', 'Ability numbers from the game data: 1/2/3-star values; N = a number the game data does',
             'not give (shown on the unit\'s ability tooltip in game).', '',
             '| Champion | Cost | Traits | Range | HP | Armor | Ability |', '|---|---|---|---|---|---|---|']
    for name, c in sorted(raw['champions'].items(), key=lambda kv: (kv[1]['base_cost'], kv[0])):
        st = c.get('stats') or {}
        ab = abilities.get(name) or {}
        text = ability_text(ab.get('desc') or (c.get('ability') or {}).get('description'), ab.get('variables'))
        lines.append(f"| {name} | {c['base_cost']} | {', '.join(c.get('traits') or [])} | {st.get('range')} | "
                     f"{st.get('hp')} | {st.get('armor')} | {ab.get('name') or (c.get('ability') or {}).get('name', '')}: "
                     f"{text} |")
    w('champions.md', '\n'.join(lines))

    from tft_player.trait_text import CHOICES, WHERE, trait_effects  # CHOICES: what the data leaves undescribed
    lines = ['# Traits', '', 'Effects with the numbers from the game data.', '']
    for name, t in sorted(raw['traits'].items()):
        members = sorted(n for n, c in raw['champions'].items() if name in (c.get('traits') or []))
        lines += [f"## {name}", f"Breakpoints: {t.get('thresholds')}", f"Champions: {', '.join(members)}",
                  (trait_effects(name) or clean(t.get('description'))) + WHERE.get(name, '')] + \
                 ([f'Choices: {CHOICES[name]}'] if name in CHOICES else []) + ['']
    w('traits.md', '\n'.join(lines))

    w('items.md', items_text())

    lines = ['# Augments', '', 'Tier: MetaTFT expert tier list (S best).', '', '| Augment | Tier | Rarity | Description |',
             '|---|---|---|---|']
    names = sorted(set(META.augments) | set(CATALOG.augments))
    for n in names:
        m = META.augments.get(n) or {}
        desc = m.get('desc') or clean((CATALOG.augments.get(n) or [{}])[0].get('description'))
        lines.append(f"| {n} | {m.get('tier') or '-'} | {m.get('rarity') or '-'} | {clean(desc)[:260]} |")
    w('augments.md', '\n'.join(lines))

    lines = ['# MetaTFT comps', '', 'Average placement is over ranked Platinum+ games in the last 3 days (lower is better).',
             '', '| Id | Comp | Avg place | Games | Levelling | Units | Carries |', '|---|---|---|---|---|---|---|']
    for c in sorted(META.comps.values(), key=lambda c: c.avg or 9):
        lines.append(f"| {c.id} | {c.name} | {c.avg} | {c.games} | {c.levelling} | {', '.join(c.units)} | {', '.join(c.carries)} |")
        g = c.guide or {}
        page = [f'# {c.name} ({c.id})', '', f'Avg place {c.avg} over {c.games} games; levelling: {c.levelling}',
                f"Units: {', '.join(c.units)}; carries: {', '.join(c.carries)}", '']
        if g.get('levels'):
            page += ['## Levelling (stage -> level)', ', '.join(f'{st}: {lv}' for st, lv in g['levels']), '']
        if g.get('final'):
            page += ['## Final boards']
            for lv, boards in sorted(g['final'].items(), key=lambda x: int(x[0])):
                for b in boards:
                    page.append(f"- level {lv}: {', '.join(b['units'])} (avg {b['avg']}, {b['games']} games)")
            page.append('')
        # MetaTFT's board also lists a tooltip ("Drag a Sprykin over BFF and hold to set them as the rider!") as a
        # unit: dropped; other names the game data has no champion for are board objects (The Tower, Lifebloom)
        if g.get('positions'):
            page += ['## Positions (most played hexes; row 0 = back row)',
                     *[f'- {u}{"" if CATALOG.unit(u) else " (not a champion: a board object)"}: '
                       + ', '.join(f'({r},{col})' for r, col in h) for u, h in g['positions'].items() if not tooltip(u)], '']
        if g.get('stars'):
            page += ['## Star levels most played', *[f'- {u}: {s}-star' for u, s in g['stars'].items() if not tooltip(u)], '']
        if c.builds:
            page += ['## Item builds (best average placement)']
            for u, bs in c.builds.items():
                for b in bs:
                    page.append(f"- {u}: {' + '.join(b['items'])} (avg {b['avg']:.2f}, {b['games']} games)")
            page.append('')
        if g.get('items'):
            page += ['## Most played items', ', '.join(g['items']), '']
        if g.get('traits'):
            page += ['## Trait level -> avg place', *[f'- {t}' for t in g['traits']], '']
        if c.augments:
            page += ['## Augment tiers for this comp', ', '.join(f'{a}: {t}' for a, t in c.augments.items()), '']
        if g.get('pro_notes'):
            page += ['## Pro notes', *[f'- {n}' for n in g['pro_notes']], '']
        (OUT / 'metatft' / 'comps' / f'{c.id}.md').write_text('\n'.join(page) + '\n', encoding='utf-8')
    w('metatft/comps.md', '\n'.join(lines))
    n = sum(1 for _ in OUT.rglob('*.md'))
    print(f'{OUT}: {n} files, {sum(p.stat().st_size for p in OUT.rglob("*.md")) // 1024} KB')


if __name__ == '__main__':
    main()
