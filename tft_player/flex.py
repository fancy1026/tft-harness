"""Flex: what the game has given us, matched against every MetaTFT comp - objective data for the model's choice of
direction (no scores or advice: which owned units, items, emblems and augments each comp's own MetaTFT data uses).

    signals(units, items, augments) -> [CompMatch] (comps with any match, most matches first)
    direction_block(units, items, augments, k=8) -> the text the planning question carries until a comp is committed
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from tft_player.catalog import CATALOG
from tft_player.items import COMPONENTS, RECIPES
from tft_player.meta import META


@dataclass
class CompMatch:
    comp: object
    units: list = field(default_factory=list)        # 'Kha'Zix 2-star (carry)'
    early: list = field(default_factory=list)        # units of its MetaTFT early boards (not in the final board)
    items: list = field(default_factory=list)        # 'Hand Of Justice (Kha'Zix build)'
    buildable: list = field(default_factory=list)    # 'Spear of Shojin from Tear + B.F. Sword (Ezreal build)'
    emblems: list = field(default_factory=list)      # 'Flora Fatalis Emblem (Flora Fatalis: 4 of its units)'
    augments: list = field(default_factory=list)     # 'Omega Riftbeast: S for this comp (S in 1, A in 0 of 38)'
    missing: list = field(default_factory=list)
    weight: float = 0.0                              # how much of the comp we hold (only orders the list)


# how much each kind of match counts when ordering the list (shown to nobody; the model sees the matches)
W = {'carry': 2.0, 'unit': 0.5, 'star': 0.5, 'early': 0.5, 'carry_item': 1.5, 'item': 0.5, 'buildable': 0.5,
     'emblem': 1.5, 'augment': 3.0, 'named': 0.5}
DISTINCT = 8          # an augment counts as comp-specific when MetaTFT rates it S or A in at most this many comps


def _emblem_trait(item):
    m = re.match(r'^(.+) Emblem$', item or '')
    return m.group(1) if m and m.group(1) in CATALOG.traits else None


def _named_in(text):
    """Traits and champions an augment's description names."""
    text = text or ''
    traits = [t for t in CATALOG.traits if re.search(rf'\b{re.escape(t)}\b', text)]
    champs = [u for u in CATALOG.units if re.search(rf'\b{re.escape(u)}\b', text)] if hasattr(CATALOG, 'units') else []
    return traits, champs


def _rated(aug):
    """How many comps MetaTFT rates this augment S / A in."""
    tiers = Counter(c.augments.get(aug) for c in META.comps.values() if c.augments)
    return tiers.get('S', 0), tiers.get('A', 0), sum(1 for c in META.comps.values() if c.augments)


def _builds(comp):
    """{item: [carry]} from the comp's best builds."""
    out = {}
    for unit, rows in (comp.builds or {}).items():
        for row in rows[:1]:
            for it in row.get('items') or []:
                out.setdefault(it, []).append(unit)
    return out


def signals(units, items, augments):
    """units: [(name, stars)], items: every item we hold (bench and on units), augments: taken augment names."""
    owned = {}
    for name, stars in units:
        owned[name] = max(owned.get(name, 0), stars)
    held = Counter(items)
    comps_held = Counter({i: k for i, k in held.items() if i in COMPONENTS})
    done = [i for i in held if i not in COMPONENTS]
    out = []
    for c in META.comps.values():
        m = CompMatch(c)
        for n in c.units:
            if n in owned:
                carry = n in c.carries
                m.units.append(f"{n} {owned[n]}-star" + (' (carry)' if carry else ''))
                m.weight += (W['carry'] if carry else W['unit']) + W['star'] * (owned[n] - 1)
            else:
                m.missing.append(n)
        early = {u for boards in (c.guide.get('early') or {}).values() for b in boards for u in b.get('units', [])}
        for n in sorted(early - set(c.units)):
            if n in owned:
                m.early.append(f"{n} {owned[n]}-star")
                m.weight += W['early'] + W['star'] * (owned[n] - 1)
        builds = _builds(c)
        for it in done:
            if it in builds:
                carry = any(u in c.carries for u in builds[it])
                m.items.append(f"{it} ({', '.join(builds[it])} build)")
                m.weight += W['carry_item'] if carry else W['item']
        for it, carriers in builds.items():
            parts = RECIPES.get(it)
            if not parts or it in held:
                continue
            if all(comps_held[p] >= k for p, k in Counter(parts).items()):
                m.buildable.append(f"{it} from {' + '.join(parts)} ({', '.join(carriers)} build)")
                if any(u in c.carries for u in carriers):
                    m.weight += W['buildable']
        traits = c.traits()
        for it in done:
            t = _emblem_trait(it)
            if t and traits.get(t, 0) >= 2:
                m.emblems.append(f"{it} ({t}: {traits[t]} of its final units)")
                m.weight += W['emblem']
        for a in augments:
            tier = c.augments.get(a)
            s_, a_, n = _rated(a)
            ts, cs = _named_in((META.augments.get(a) or {}).get('desc'))
            named = [t for t in ts if traits.get(t)] + [u for u in cs if u in c.units or u in early]
            if tier in ('S', 'A') and s_ + a_ <= DISTINCT:
                m.augments.append(f"{a}: {tier} for this comp (MetaTFT rates it S in {s_} and A in {a_} of {n} comps)")
                m.weight += W['augment']
            elif named:
                m.augments.append(f"{a}: names {', '.join(named)}, which this comp plays")
                m.weight += W['named']
        if m.weight > 0:
            out.append(m)
    out.sort(key=lambda m: (-m.weight, m.comp.avg or 9))
    return out


def comp_line(c):
    builds = '; '.join(f"{u}: {', '.join(r[0]['items'])}" for u, r in (c.builds or {}).items() if r and u in c.carries)
    return (f"{c.name} [{c.id}]: avg place {c.avg:.2f}, {c.levelling}, carries {', '.join(c.carries)}"
            + (f" ({builds})" if builds else ''))


def direction_block(units, items, augments, k=8):
    rows = signals(units, items, augments)
    lines = ['WHAT YOU HOLD, MATCHED AGAINST THE METATFT COMPS (their own data: final units, carry builds, '
             'augment ratings, traits). Ordered by how much of each comp you already hold; avg place is over all games of the comp.']
    for m in rows[:k]:
        parts = [f"units {', '.join(m.units)}" if m.units else '',
                 f"early-board units {', '.join(m.early)}" if m.early else '',
                 f"items {', '.join(m.items)}" if m.items else '',
                 f"buildable now {', '.join(m.buildable[:3])}" if m.buildable else '',
                 f"emblems {', '.join(m.emblems)}" if m.emblems else '',
                 f"augments {'; '.join(m.augments)}" if m.augments else '']
        lines.append(f"- {comp_line(m.comp)}\n    yours: " + '; '.join(p for p in parts if p) +
                     f"\n    still missing: {', '.join(m.missing) or 'nothing'}")
    if len(rows) > k:
        lines.append(f"({len(rows) - k} more comps share something smaller with you; every comp's page: "
                     'metatft/comps/<id>.md)')
    return '\n'.join(lines)
