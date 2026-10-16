"""Current meta (MetaTFT snapshot from tools/fetch_meta.py), as local lookups.

The pattern: local code builds a small menu of evidence-backed candidates and
a default choice in milliseconds; the coach only compares the menu against the
game and may override.

  candidates(roster)      comps ranked by what we hold x how strong they are
  augment_value(name, w)  tier of an augment for the comps we're likely to play
  augment_decision(...)   pick index, or a card worth rerolling
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from tft_player.catalog import CATALOG

PATH = Path(__file__).resolve().parent.parent / 'data' / 'meta' / 'current.json'
TIER = {'S': 4.0, 'A': 3.0, 'B': 2.0, 'C': 1.0, 'D': 0.0}
UNTIERED = 1.5
STAR = {1: 1.0, 2: 3.0, 3: 9.0}
REROLL_GAIN = 0.2          # tier units a reroll must be expected to add


@dataclass
class Comp:
    id: str
    name: str
    units: list
    carries: list
    levelling: str | None
    avg: float | None
    games: int | None
    augments: dict
    builds: dict
    guide: dict = field(default_factory=dict)      # MetaTFT comp page: levels, boards, positions, items...
    win: float | None = None
    top4: float | None = None
    pick: float | None = None
    situational: str | None = None    # why metatft.com hides it by default (a required emblem...)

    @property
    def tier(self):
        """MetaTFT-style letter from avg place (its API has no tier field)."""
        if not self.avg or (self.games or 0) < 5000:
            return 'C'
        return 'S' if self.avg <= 4.30 else 'A' if self.avg <= 4.55 else 'B' if self.avg <= 4.75 else 'C'

    @property
    def strength(self):
        """Placement edge over an average board (4.5), shrunk for small samples."""
        if not self.avg or not self.games:
            return 0.0
        return (4.5 - self.avg) * self.games / (self.games + 3000)

    def traits(self):
        c = Counter()
        for n in self.units:
            if CATALOG.unit(n):
                c.update(CATALOG.unit(n).traits)
        return c


class Meta:
    def __init__(self, path=PATH):
        d = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        self.patch = d.get('patch')
        self.fetched_at = d.get('fetched_at')
        self.augments: dict = d.get('augments') or {}
        self.comps = {cid: Comp(cid, c['name'], c['units'], c['carries'], c.get('levelling'),
                                c.get('avg'), c.get('games'), c.get('augments') or {}, c.get('builds') or {},
                                c.get('guide') or {}, c.get('win'), c.get('top4'), c.get('pick'),
                                c.get('situational'))
                      for cid, c in (d.get('comps') or {}).items()}

    @property
    def available(self):
        return bool(self.augments)

    # -- comps -----------------------------------------------------------------
    def fit(self, comp: Comp, roster):
        """0..1: share of the comp's unit value we already hold, plus trait overlap."""
        owned = Counter()
        for u in roster.units:
            if u.name:
                owned[u.name] += STAR.get(u.star, 1)
        total = held = 0.0
        for n in comp.units:
            info = CATALOG.unit(n)
            w = (info.cost if info else 3) * (2.0 if n in comp.carries else 1.0)
            total += w
            if n in owned:
                held += w * min(1.0, 0.5 + 0.25 * owned[n])
        mine = Counter()
        for n in owned:
            if CATALOG.unit(n):
                mine.update(CATALOG.unit(n).traits)
        theirs = comp.traits()
        shared = sum(min(c, theirs[t]) for t, c in mine.items() if t in theirs)
        return min(1.0, held / max(total, 1) + 0.04 * shared)

    def candidates(self, roster, k=4, prefer=None):
        """[(comp, fit, score)] best first; `prefer` (a comp id) is always included."""
        rows = []
        for c in self.comps.values():
            f = self.fit(c, roster)
            rows.append((c, f, c.strength + 1.5 * f + (1.0 if c.id == prefer else 0.0)))
        rows.sort(key=lambda r: -r[2])
        return rows[:k]

    def comp_weights(self, cands, stage=None):
        """Belief over comps; the rest of the mass stays on 'any comp' (overall tier),
        which dominates early when what we hold says little."""
        if not cands:
            return {}, 1.0
        conf = max(f for _, f, _ in cands)
        z = [math.exp(2 * s) for _, _, s in cands]
        mass = min(0.85, 0.3 + conf) if stage is None or stage >= (3, 1) else min(0.6, 0.15 + conf)
        return {c.id: mass * zi / sum(z) for (c, _, _), zi in zip(cands, z)}, 1 - mass

    def random_top(self):
        """A random S or A tier comp of the ones metatft.com lists by default (not situational) - `--comp random`
        (the user, 2026-10-16), or None."""
        import random
        top = [c for c in self.comps.values() if c.tier in ('S', 'A') and not c.situational]
        return random.choice(top) if top else None

    def by_name(self, name):
        """A comp by its MetaTFT name (case and spacing ignored) or id."""
        key = ' '.join(str(name or '').split()).lower()
        return self.comps.get(str(name)) or next((c for c in self.comps.values() if c.name.lower() == key), None)

    def search(self, text):
        """Comps for what a user typed: its id, its exact name, or every word of it in the name (any order,
        case ignored); best average place first."""
        exact = self.by_name(text)
        if exact:
            return [exact]
        words = str(text or '').lower().split()
        hits = [c for c in self.comps.values() if words and all(w in c.name.lower() for w in words)]
        return sorted(hits, key=lambda c: c.avg or 9)

    # -- augments ----------------------------------------------------------------
    def tier(self, name, comp: Comp | None = None):
        t = comp.augments.get(name) if comp else None
        if t is None:
            t = (self.augments.get(name) or {}).get('tier')
        return TIER.get(t, UNTIERED)

    def augment_value(self, name, weights, rest):
        v = rest * self.tier(name)
        for cid, w in weights.items():
            v += w * self.tier(name, self.comps[cid])
        return v

    def pool(self, names):
        """Augments that could replace a rerolled card: same rarity, regular, tiered."""
        rar = Counter((self.augments.get(n) or {}).get('rarity') for n in names).most_common(1)[0][0]
        return [n for n, a in self.augments.items()
                if a.get('rarity') == rar and a.get('kind') == 'regular' and a.get('tier')]

    def augment_decision(self, names, rerollable, weights, rest):
        """-> ('pick', i) or ('reroll', i). A reroll replaces one card with a random
        one from the pool; do it when E[max after] beats the current max."""
        vals = [self.augment_value(n, weights, rest) for n in names]
        best = max(range(len(names)), key=lambda i: vals[i])
        pool = [self.augment_value(n, weights, rest) for n in self.pool(names)] if rerollable else []
        gains = []
        for i in rerollable:
            keep = max((v for j, v in enumerate(vals) if j != i), default=0.0)
            exp = sum(max(keep, x) for x in pool) / len(pool) if pool else 0.0
            gains.append((exp - vals[best], i))
        if gains:
            g, i = max(gains)
            if g > REROLL_GAIN:
                return ('reroll', i), vals
        return ('pick', best), vals

    def alternatives(self, names, weights, rest, k=6):
        """Strongest pool augments for our likely comps that we were not offered."""
        pool = [n for n in self.pool(names) if n not in names]
        return sorted(pool, key=lambda n: -self.augment_value(n, weights, rest))[:k]

    def describe_augment(self, name, cands=()):
        a = self.augments.get(name) or {}
        per = ', '.join(f'{c.name}: {c.augments[name]}' for c, _, _ in cands if name in c.augments)
        return (f"{name} [{a.get('rarity', '?')}, tier {a.get('tier') or '?'}"
                f"{'; ' + per if per else ''}] {a.get('desc', '')[:200]}")

    def describe_comp(self, comp: Comp, fit, roster):
        have = {u.name for u in roster.units if u.name}
        units = ', '.join(f"{n}{'*' if n in have else ''}" for n in comp.units)
        builds = '; '.join(f"{u}: {' + '.join(b[0]['items'])} (avg {b[0]['avg']:.2f}, {b[0]['games']} games)"
                           for u, b in comp.builds.items() if u in comp.carries and b)
        return (f'[{comp.id}] {comp.name} ({comp.tier} tier) - avg place {comp.avg} over {comp.games} games, '
                f'{comp.levelling or "?"}; held fit {fit:.0%}\n  units: {units}\n  carry builds: {builds or "?"}')


META = Meta()
