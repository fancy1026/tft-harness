"""Trait effects with their numbers, from the CommunityDragon game data (data/sources/cdragon-tft-*.json).

The description's @Tokens@ are filled from each level's variables; a variable stored under a hash is
found by its name's FNV-1a hash ('ADAPGain' -> {0412779a}).

    trait_line('Lunar', 2)  ->  'Lunar 2: Lunar champions and adjacent allies gain ... | levels: (2) 7% AS, ...'
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

SOURCES = Path(__file__).resolve().parent.parent / 'data' / 'sources'
SET = '18'
ICONS = {'AS': 'AS', 'AP': 'AP', 'AD': 'AD', 'Health': 'HP', 'Armor': 'Armor', 'MR': 'MR', 'Mana': 'Mana',
         'Crit': 'Crit', 'CritChance': 'Crit Chance', 'CritDamage': 'Crit Damage', 'DA': 'Damage Amp',
         'Durability': 'Durability', 'Omnivamp': 'Omnivamp', 'SV': 'Omnivamp', 'Range': 'Range'}


def _fnv(name):
    h = 0x811c9dc5
    for c in name.lower().encode():
        h = ((h ^ c) * 0x01000193) & 0xffffffff
    return '{%08x}' % h


@lru_cache(maxsize=1)
def _traits():
    src = sorted(SOURCES.glob('cdragon-tft-*.json'))[-1]
    d = json.loads(src.read_text(encoding='utf-8'))
    return {t['name']: t for t in d['sets'][SET]['traits']}


# Choices a trait offers that the game data leaves undescribed ("Choose one of four Primal Blessings").
# Source: wiki.leagueoflegends.com/en-us/TFT:Primal (checked 2026-10-07; run 221257 3-7: the model guessed
# which blessing was which from the card names alone).
CHOICES = {
    'Primal': ('one Blessing at 2 Primal, a second at 4 (Beast Within also grants a second): '
               'Blessing of the Turtle - your team heals 4% of max Health every 4 seconds; '
               'Blessing of the Bear - Primal damage executes enemies below 12% Health; '
               'Blessing of the Tiger - after 6 seconds, Primal champions gain 35% Attack Speed and your team 15%; '
               'Blessing of the Phoenix - every 15 Primal takedowns, gain a component (max 4)'),
    # game data (Rival): "Takedowns evolve Kha'Zix, permanently granting him your choice of ..." (run 185805 2-5:
    # the evolution choice went unseen)
    'Rival': ("Kha'Zix evolves after takedowns: choose one trait he gains permanently - Executioner, Rapidfire, "
              "Ravager or Spellweaver"),
    # game data: "Choose a bounty ... Upon completing a bounty, choose another." - the bounties and rewards from
    # metatft.com/tables/draven-bounties (checked 2026-10-13): 3 offered, 2 standard and 1 hard (lolchess/wiki)
    # the user's screenshot (2026-10-13): "Choose One" with two cards - convert the Essence to loot (its estimated
    # value in gold) or continue to the next threshold
    'Coven': ('at each threshold (40, 85, 130, 185, 250, 365, 500, 650, 800 Essence) a "Choose One" screen offers '
              'two cards: "Convert <n> Essence to loot" (with its estimated value in gold) or "Continue gathering '
              'Essence" (next threshold shown). Converting spends the Essence; each later reward needs more.'),
    'Bounty Seeker': ('3 bounties offered (2 standard, 1 hard); your strongest Draven works on the chosen one, then '
                      'you choose again. Standard: attacks 60 times -> 1 random 5-cost champion; casts 6 times -> 2 '
                      'random 4-cost champions; deals 10000 damage -> 7 gold; deals at least 10000 damage in one '
                      'player combat -> 15 gold; gets 8 kills -> 12 gold; deals the most damage on your team in '
                      'combat -> 2 shop rerolls; gets 14 takedowns -> 10 XP. Hard: casts 8 times -> 6 shop rerolls; '
                      'deals 35000 damage -> 1 B.F. Sword; gets 15 kills -> 1 random completed item; survives 3 '
                      'player combats -> 2 random components'),
}


# Card titles of trait choice screens other than trait names (screenshots from the user, 2026-10-13): Coven's two
# cards ("Convert 250 Essence to loot / Est. Value: 40g", "Continue gathering Essence / Next Threshold: 365" - the
# numbers change, names are matched without digits) and Draven's bounty cards ("Bounty: Draven attacks 50 times.
# Reward: 1 random 5-cost champion").
CARD_NAMES = ['Convert Essence to loot', 'Continue gathering Essence',
              'Bounty: Draven attacks 60 times. Reward: 1 random 5-cost champion',
              'Bounty: Draven casts 6 times. Reward: 2 random 4-cost champions',
              'Bounty: Draven deals 10000 damage. Reward: 7 gold',
              'Bounty: Draven deals at least 10000 damage in one player combat. Reward: 15 gold',
              'Bounty: Draven gets 8 kills. Reward: 12 gold',
              'Bounty: Draven deals the most damage on your team in combat. Reward: 2 shop rerolls',
              'Bounty: Draven gets 14 takedowns. Reward: 10 XP',
              'Bounty: Draven casts 8 times. Reward: 6 shop rerolls',
              'Bounty: Draven deals 35000 damage. Reward: 1 B.F. Sword',
              'Bounty: Draven gets 15 kills. Reward: 1 random completed item',
              'Bounty: Draven survives 3 player combats. Reward: 2 random components']


def choice_names():
    """Names that appear on trait-choice cards: every trait, the Primal Blessings, Coven's and Draven's cards."""
    return sorted(set(_traits()) | set(re.findall(r'Blessing of the \w+', CHOICES['Primal'])) | set(CARD_NAMES))


def trait_name(api_name):
    """A trait's game-data id, in any case or punctuation ('da_18_zyrauniquetrait') -> its name
    ('Thornmaiden'), or None."""
    key = lambda x: re.sub(r'[^a-z0-9]', '', x.lower())
    return {key(t.get('apiName') or ''): n for n, t in _traits().items()}.get(key(api_name))


def choices_text(traits):
    """'- Primal: one Blessing at 2 ...' lines for the traits in `traits` that offer a choice."""
    return '\n'.join(f'- {t}: {CHOICES[t]}' for t in dict.fromkeys(traits) if t in CHOICES)


def _num(v):
    v = round(v, 2)
    return str(int(v)) if v == int(v) else f'{v:g}'


def _fill(text, eff):
    def tok(m):
        name, mul = m.group(1), m.group(2)
        if name == 'MinUnits':
            return str(eff.get('minUnits', ''))
        var = {k.lower(): x for k, x in (eff.get('variables') or {}).items()}     # 'ASperAttack' vs '@ASPerAttack@'
        v = var.get(name.lower(), var.get(_fnv(name)))
        if v is None:
            return '?'
        return _num(v * float(mul)) if mul else _num(v)
    text = re.sub(r'@([A-Za-z0-9_]+)(?:\*([\d.]+))?@', tok, text)
    text = re.sub(r'%i:scale(\w+)%', lambda m: ICONS.get(m.group(1), m.group(1)), text)
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', text)).strip()


def _effects(name):
    t = _traits().get(name)
    return sorted((e for e in (t or {}).get('effects') or [] if e.get('minUnits') is not None), key=lambda e: e['minUnits'])


def trait_effects(name):
    """'<what it does> | levels: (2) ...; (3) ...' with the game data's numbers (None if unknown)."""
    t = _traits().get(name)
    if not t:
        return None
    effs = _effects(name)
    desc = t.get('desc') or ''
    rows = re.findall(r'<row>(.*?)</row>', desc, re.S)
    intro = re.sub(r'<row>.*?</row>', '', desc, flags=re.S).replace('<br>', ' ')
    out = _fill(intro, effs[0] if effs else {})
    if rows and effs:
        out += ' | levels: ' + '; '.join(_fill(r, e) for r, e in zip(rows, effs))
    return out


def level_effect(name, count):
    """The game data's text of the level a trait has at this count ('175 HP'), or None if inactive/unknown."""
    t = _traits().get(name)
    effs = _effects(name)
    rows = re.findall(r'<row>(.*?)</row>', (t or {}).get('desc') or '', re.S)
    now = max((e['minUnits'] for e in effs if e['minUnits'] <= count), default=None)
    for r, e in zip(rows, effs):
        if e['minUnits'] == now:
            return re.sub(r'^\(\d+\)\s*', '', re.sub(r'<[^>]+>', '', _fill(r, e))).strip(' ;') or None
    return None


def trait_line(name, count):
    """'<name> <count>: <what it does> | levels: (2) ..., (3) ... | now: 2, next: 3' (None if unknown)."""
    if name not in _traits():
        return None
    out = f'{name} {count}: {trait_effects(name)}'
    effs = _effects(name)
    steps = [e['minUnits'] for e in effs]
    now = max((s for s in steps if s <= count), default=None)
    nxt = min((s for s in steps if s > count), default=None)
    out += f" | now: {now if now else 'inactive'}" + (f', next: {nxt}' if nxt else ', top level')
    return out + WHERE.get(name, '')


# where on our board a trait's effect happens - what the trait text does not say (the user, 2026-10-10;
# hexes as [row, col], row 0 the back row next to the bench)
WHERE = {'Blackthorn': ' | the Blackthorn hex is [1, 3] (the centre of the board)',
         # the user, 2026-10-13 (screenshot): grown hexes show on our board by colour and look
         'Greenfather': (' | grown hexes show on your board (see the screenshot): rock = bonus Armor and Magic Resist, '
                         'water (lily pads) = bonus Mana Regen, flowers = bonus Attack Speed; other biomes have more '
                         'kinds; the God-Willow is the last one, at the centre. The unit standing on one gets it')}


# -- champion rules the trait texts state (team slots, bonus trait counts, evolutions) -------------------------

def _plain(text):
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', text or '')).strip()


def _trait_champions(trait):
    from tft_player.catalog import CATALOG
    return [n for n, u in CATALOG.units.items() if trait in u.traits]


@lru_cache(maxsize=1)
def champion_rules():
    """{champion: {'slots': n, 'bonus': {trait: n}, 'evolves': (trait, ...)}} from the game data's trait texts:
    "Elder Dragon takes up 2 team slots and grants +2 to the Riftbeast trait" (Apex Predator), "Takedowns evolve
    Kha'Zix, permanently granting him your choice of Executioner, Rapidfire, Ravager, or Spellweaver" (Rival).
    Mechanisms read off the text, not a list of names: a new set's wording that matches works the same way."""
    out = {}
    for name, t in _traits().items():
        desc = _plain(t.get('desc'))
        holders = _trait_champions(name)
        for champ, n in re.findall(r"([A-Z][\w' ]*?) takes up (\d+) team slots", desc):
            out.setdefault(champ.strip(), {})['slots'] = int(n)
        for n, trait in re.findall(r'grants \+(\d+) to the ([A-Z][\w ]*?) trait', desc):
            m = re.search(r"([A-Z][\w' ]*?) takes up", desc)
            for champ in ([m.group(1).strip()] if m else holders):
                out.setdefault(champ, {}).setdefault('bonus', {})[trait] = int(n)
        for champ, opts in re.findall(r"evolve ([A-Z][\w']*(?: [A-Z][\w']*)?), permanently granting \w+ your "
                                      r"choice of ([^.]+)\.", desc):
            names = [o.strip() for o in re.split(r',\s*(?:or\s+)?|\s+or\s+', opts) if o.strip()]
            out.setdefault(champ, {})['evolves'] = tuple(o for o in names if o in _traits())
    return out


def pair_hosts():
    """Champions an ally is paired with by dragging it onto them ("Drag an ally onto your strongest Taric to pair
    them", Emerald Aspect). The pair is a line between the two and holds through moves and rounds (the user,
    2026-10-13)."""
    return {m for t in _traits().values()
            for m in re.findall(r"[Dd]rag an ally onto your strongest ([A-Z][\w']*) to pair", _plain(t.get('desc')))}


def unit_slots(name):
    """Team slots a fielded champion uses (Elder Dragon: 2; everyone else 1)."""
    return (champion_rules().get(name) or {}).get('slots', 1)


def trait_bonus(name):
    """{trait: extra count} a champion adds beyond counting once for its own traits. "grants +2 to the Riftbeast
    trait" means Elder Dragon counts as 2 Riftbeast in all (the user, 2026-10-13, from games; MetaTFT's board of 6
    Riftbeast champions with him shows Riftbeast 7): +1 over its own count."""
    from tft_player.catalog import CATALOG
    own = set(CATALOG.unit(name).traits) if CATALOG.unit(name) else set()
    return {t: n - 1 if t in own else n for t, n in ((champion_rules().get(name) or {}).get('bonus') or {}).items()
            if (n - 1 if t in own else n) > 0}


def evolutions():
    """{champion: (trait options)} - champions that gain one trait of their choice for good (Kha'Zix)."""
    return {c: r['evolves'] for c, r in champion_rules().items() if r.get('evolves')}
