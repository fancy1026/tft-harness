"""Set data: champions (cost, traits, range), traits (thresholds), augments."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / 'data'


@dataclass(frozen=True)
class Unit:
    name: str
    cost: int
    traits: tuple[str, ...]
    range: float
    hp: float
    armor: float

    @property
    def melee(self):
        return self.range <= 1.5


class Catalog:
    def __init__(self, path=DATA / 'factual-catalogue-18.json'):
        d = json.loads(path.read_text(encoding='utf-8'))
        self.units: dict[str, Unit] = {}
        self._raw = d['champions']
        for name, c in d['champions'].items():
            st = c.get('stats') or {}
            self.units[name] = Unit(name, int(c['base_cost']), tuple(c.get('traits') or ()),
                                    float(st.get('range') or 1), float(st.get('hp') or 0),
                                    float(st.get('armor') or 0))
        self.traits = {k: tuple(v.get('thresholds') or ()) for k, v in d['traits'].items()}
        self.augments = dict(d['augments'])

    def unit(self, name):
        return self.units.get(name)

    def champion_raw(self, name):
        return self._raw.get(name)


CATALOG = Catalog()
