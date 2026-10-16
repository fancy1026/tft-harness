"""The TFT brain: specialists that each propose their best next action.

Called every tick during planning; returns ONE action. The loop executes it,
re-reads the screen and asks again, so nothing goes stale.

Priority order each tick:
  1. augment pick (if the picker is open)
  2. board: fill empty slots / upgrade the board (fielding units wins fights)
  3. shop: buy copies and good units within the economy's budget
  4. bench: sell junk when the bench is clogged
  5. economy: level up / roll down
  6. positioning: put units on the right hexes
The coach (LLM) only adjusts `Plan`; it never sits in this path.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from tft_player.catalog import CATALOG
from tft_player.items import plan_item_move, plan_strip
from tft_player.meta import META
from tft_player.state import Owned, Roster

STAR_POWER = {1: 1.0, 2: 2.8, 3: 7.0}


@dataclass
class Plan:
    """Strategic direction; set by the coach, or derived from the board."""
    traits: list[str] = field(default_factory=list)   # traits we're building
    units: list[str] = field(default_factory=list)    # core units to collect
    carry: str | None = None
    style: str = 'standard'                            # standard | reroll | fast8
    comp: str | None = None                            # meta comp id we're building
    augment_pick: int | None = None                    # coach's choice for current picker
    augment_name: str | None = None                    # that card's name, as the model read it
    augment_reroll: list | None = None                 # cards the model wants rerolled
    item_targets: list = field(default_factory=list)   # components the model wants from carousels

    def adopt(self, comp):
        """Take a meta comp as the direction: its units, carry and main traits."""
        self.comp = comp.id
        self.units = list(comp.units)
        self.carry = comp.carries[0] if comp.carries else self.carry
        self.traits = [t for t, c in comp.traits().most_common(3) if c >= 2]
        lv = (comp.levelling or '').lower()
        self.style = 'reroll' if lv.startswith('lvl') else 'fast8' if 'fast' in lv else 'standard'


@dataclass
class Action:
    kind: str            # buy, sell, move, level, roll, augment
    arg: object = None
    why: str = ''

    def __repr__(self):
        return f'{self.kind}({self.arg}) - {self.why}'


# ----------------------------------------------------------------------------
# economy


def level_target(stage):
    """Standard curve: 4 @2-1, 5 @2-5, 6 @3-2, 7 @4-1, 8 @4-2, 9 @5-5."""
    s, r = stage
    if s == 1:
        return 3
    if s == 2:
        return 5 if r >= 5 else 4
    if s == 3:
        return 6 if r >= 2 else 5
    if s == 4:
        return 8 if r >= 2 else 7                  # 4-2: level 8 and roll down (the standard spike)
    if s == 5:
        return 9 if r >= 5 else 8
    return 9


def gold_floor(stage, hp, level):
    """Gold we refuse to spend below (interest), relaxed as HP drops."""
    s, r = stage
    if s <= 1:
        return 0
    if hp is not None and hp <= 30:
        return 0                                  # spend everything, survive
    if hp is not None and hp <= 50:
        return 10 if s < 4 else 0
    if s == 2:
        return 10 if r <= 3 else 20
    if s == 3:
        return 30 if r <= 2 else 40
    # stage 4+: the "roll-down" windows are handled in roll logic
    return 50


class Brain:
    def __init__(self):
        self.plan = Plan()
        self.rolled_this_round = 0
        self.stripped = set()               # unit ids a Remover was used on
        self.position_moves = 0
        self.positioned = set()
        self.failed: Counter = Counter()    # (kind, arg) -> failures this round
        self.status: dict[str, str] = {}    # agent -> what it's doing / why it isn't (for the live view)
        self.protected: set[int] = set()    # unit ids the model's plan keeps: local rules never sell them

    def new_round(self, stage=None, roster=None):
        self.rolled_this_round = 0
        self.position_moves = 0
        self.positioned = set()
        self.failed.clear()
        # no coach direction yet: from stage 3 commit to the meta comp closest to what we hold
        if stage and roster and stage >= (3, 1) and not self.plan.comp and META.available:
            comp, fit, _ = META.candidates(roster, k=1)[0]
            if fit >= 0.25:
                self.plan.adopt(comp)
        if stage and roster and stage >= (3, 1) and not self.plan.traits:
            counts = Counter()
            for n in {u.name for u in roster.units if u.name}:
                counts.update(CATALOG.unit(n).traits)
            self.plan.traits = [t for t, c in counts.most_common(2) if c >= 2]

    # -- unit valuation -------------------------------------------------------
    def unit_power(self, u: Owned):
        base = (u.cost + 0.6) * STAR_POWER.get(u.star, 1)
        if u.name and u.name in self.plan.units:
            base *= 1.25
        if u.name and u.name == self.plan.carry:
            base *= 1.4
        # items are most of a unit's strength early: a benched carry wastes them
        from tft_player.items import COMPONENTS
        base += sum(1.2 if i in COMPONENTS else 3.0 for i in u.items)
        return base

    def trait_bonus(self, units):
        counts = Counter()
        for n in {u.name for u in units if u.name}:   # duplicates don't stack traits
            for t in CATALOG.unit(n).traits:
                counts[t] += 1
        bonus = 0.0
        for t, c in counts.items():
            th = CATALOG.traits.get(t) or ()
            active = [x for x in th if c >= x]
            if active:
                bonus += 1.2 * len(active) * active[-1] / max(th[0], 1)
            if t in self.plan.traits:
                bonus += 0.4 * c
        return bonus

    def best_board(self, units, slots):
        """Greedy: add the unit that raises board value most, one at a time."""
        chosen, pool = [], list(units)
        while pool and len(chosen) < slots:
            base = self.trait_bonus(chosen)
            u = max(pool, key=lambda x: self.unit_power(x) + self.trait_bonus(chosen + [x]) - base
                    - (0.8 if x.name and any(c.name == x.name for c in chosen) else 0))
            chosen.append(u)
            pool.remove(u)
        return chosen

    def why(self, agent, text):
        """Record an agent's reasoning when it decides NOT to act (shown in the live view)."""
        self.status[agent] = text
        return None

    # -- the tick ---------------------------------------------------------------
    def decide(self, scr, roster: Roster, planning: bool):
        if scr.augments:
            return None                 # the player loop handles the picker (coach + fallback)
        if not scr.shop_visible or scr.gold is None or scr.level is None or not scr.round:
            return None
        steps = [('board', self._board, planning), ('items', self._items, planning),
                 ('shop', self._shop, True), ('bench', self._bench, True),
                 ('economy', self._economy, True), ('position', self._position, planning)]
        for name, step, on in steps:
            if not on:
                self.status[name] = 'waiting for the planning phase'
                continue
            a = step(scr, roster)
            if a and self.failed[(a.kind, str(a.arg))] < 2:
                self.status[name] = f'ACT {a.kind}: {a.why}'
                return a
            if a:
                self.status[name] = f'gave up on {a.kind} ({a.why}) after 2 failed tries'
        return None

    # 2. board ------------------------------------------------------------------
    def _board(self, scr, roster):
        board, bench = roster.board(), roster.bench()
        ideal = self.best_board(board + bench, scr.level)
        ideal_ids = {u.id for u in ideal}
        incoming = [u for u in ideal if not u.on_board]
        if not incoming:
            return self.why('board', f'best board already fielded ({len(board)}/{scr.level}): '
                                     + ', '.join(repr(u) for u in board))
        u = max(incoming, key=self.unit_power)
        outgoing = [b for b in board if b.id not in ideal_ids]
        if max(len(board), roster.board_seen) < scr.level:     # the game's count, not just ours
            hex_ = self.free_hex_for(u, roster)
            return Action('move', (u, ('board', *hex_)), f'field {u} (board {len(board)}/{scr.level})')
        if outgoing:
            worst = min(outgoing, key=self.unit_power)
            # hysteresis: near-equal units would otherwise swap back and forth every tick
            if self.unit_power(u) < self.unit_power(worst) + 1.0 and worst.name:
                return self.why('board', f'{u} is not clearly better than {worst} '
                                         f'({self.unit_power(u):.1f} vs {self.unit_power(worst):.1f}): no swap')
            return Action('move', (u, worst.loc), f'swap in {u} for {worst}')
        return self.why('board', f'board full ({len(board)}/{scr.level})')

    # 2b. items -----------------------------------------------------------------
    def _items(self, scr, roster):
        bench_items = getattr(scr, 'items', None)
        if not bench_items or not any(bench_items):
            return self.why('items', 'item bench empty (or not read yet)')
        st = plan_strip(bench_items, roster.board(), self.plan, self.stripped)
        if st:
            return Action('strip', (st[0], bench_items[st[0]], st[1]), st[2])
        mv = plan_item_move(bench_items, roster.board(), self.plan, scr.round)
        if not mv:
            held = [i for i in bench_items if i]
            return self.why('items', f'holding {held}: no good holder/recipe on the board yet')
        slot, unit, why = mv
        return Action('item', (slot, bench_items[slot], unit), why)

    # 3. shop ---------------------------------------------------------------------
    def shop_value(self, name, roster, stage):
        info = CATALOG.unit(name)
        if not info:
            return 0
        owned = [u for u in roster.units if u.name == name]
        ones = len([u for u in owned if u.star == 1])
        twos = len([u for u in owned if u.star == 2])
        if twos >= 3 or (twos == 2 and ones >= 2):
            return 0                                   # 3-star done or covered
        v = 0.0
        if ones == 2:
            v += 10                                    # completes a 2-star now
        elif ones == 1:
            v += 5
        if name in self.plan.units:
            v += 4
        v += 1.5 * sum(1 for t in info.traits if t in self.plan.traits)
        s = stage[0]
        # early game: every unit is value (pairs, trait tests, free to sell back)
        if s <= 2:
            v += 2 + info.cost * 0.5
        elif not owned:
            v += info.cost * 0.6 - 1.5
        return v

    def _shop(self, scr, roster):
        stage = scr.round
        floor = gold_floor(stage, scr.hp, scr.level)
        free = len(roster.free_bench())
        best = None
        skipped = []
        for i, name in enumerate(scr.shop):
            if not name or not CATALOG.unit(name):
                continue
            cost = CATALOG.unit(name).cost
            if cost > scr.gold:
                skipped.append(f'{name}: {cost}g > {scr.gold}g')
                continue
            v = self.shop_value(name, roster, stage)
            completes = len(roster.copies(name, 1)) == 2
            if v <= 0 or (free == 0 and not completes):
                skipped.append(f'{name}: ' + ('bench full' if v > 0 else 'no value'))
                continue
            # budget: pairs/triples may dip 10 below the interest floor
            allowed = floor - (10 if v >= 5 else 0)
            if scr.gold - cost < allowed and not (completes and stage[0] >= 3):
                skipped.append(f'{name} (value {v:.0f}): would drop below the {allowed}g floor')
                continue
            if v < 3 and stage[0] >= 3:
                skipped.append(f'{name}: value {v:.1f} too low for stage {stage[0]}')
                continue
            if best is None or v > best[0]:
                best = (v, i, name)
        if best:
            return Action('buy', best[1], f'{best[2]} (value {best[0]:.1f})')
        return self.why('shop', 'nothing to buy - ' + ('; '.join(skipped) or 'shop empty'))

    # 4. bench ----------------------------------------------------------------------
    def _bench(self, scr, roster):
        bench = roster.bench()
        stage = scr.round
        keep_free = 1 if stage[0] <= 2 else 2
        if len(roster.free_bench()) >= keep_free:
            return self.why('bench', f'{9 - len(roster.free_bench())}/9 used, no need to sell')
        board = roster.board()
        def keep_score(u):
            pairs = len(roster.copies(u.name, u.star)) if u.name else 1
            return self.unit_power(u) + (4 if pairs >= 2 else 0) + (3 if u.name in self.plan.units else 0)
        # never sell blind: an unidentified unit may be a pair or a key piece.
        # With only unknowns left, do nothing - the idle step identifies them.
        junk = min((u for u in bench if u.name and u.id not in self.protected), key=keep_score, default=None)
        if junk and (not board or keep_score(junk) < max(self.unit_power(b) for b in board) + 3):
            return Action('sell', junk, f'bench clogged, sell {junk}')
        return self.why('bench', 'bench full but nothing is safe to sell (unknown or useful units)')

    # 5. economy ---------------------------------------------------------------------
    def _economy(self, scr, roster):
        stage, gold, hp, lvl = scr.round, scr.gold, scr.hp, scr.level
        floor = gold_floor(stage, hp, lvl)
        target = level_target(stage)
        if hp is not None and hp <= 40 and stage[0] >= 3:
            target = max(target, min(lvl, 8))           # low HP: stop leveling, roll for board
        if lvl < target and lvl < 9 and gold >= 4 and gold - 4 >= floor - (4 if stage[0] >= 3 else 0):
            return Action('level', None, f'level {lvl}->{target} target, gold {gold}')
        if lvl < 9 and gold >= 54:
            return Action('level', None, f'gold {gold} over interest cap')
        # rolling
        roll_floor = None
        if gold >= 52:
            roll_floor = 50                                # excess above 50 is free to roll
        if stage >= (4, 1) and lvl >= 7 and hp is not None and hp <= 60:
            roll_floor = 20
        if hp is not None and hp <= 35 and stage[0] >= 3:
            roll_floor = 0
        if stage == (4, 2) and lvl >= 8:
            roll_floor = 10                                # the standard 4-2 roll-down
        if roll_floor is not None and gold - 2 >= roll_floor and not roster.free_bench() == []:
            return Action('roll', None, f'roll gold {gold} floor {roll_floor} hp {hp}')
        return self.why('economy', f'saving: gold {gold}, interest floor {floor}, level {lvl} '
                                   f'(target {target} at {stage[0]}-{stage[1]}), hp {hp}, '
                                   f'roll floor {roll_floor}')

    # 6. positioning -------------------------------------------------------------------
    def target_hexes(self, board):
        """Melee in front, ranged in back; strongest ranged in a back corner,
        tankiest melee in the middle of the front line."""
        melee = [u for u in board if u.info and u.info.melee or (not u.info)]
        ranged = [u for u in board if u.info and not u.info.melee]
        # stable tie-break on id so equal units never trade places back and forth
        melee.sort(key=lambda u: (-(u.info.hp + 10 * u.info.armor if u.info else 0) * STAR_POWER[u.star]
                                  * (1 + 0.3 * len(u.items)), u.id))
        ranged.sort(key=lambda u: (-self.unit_power(u), u.id))
        front_order = [3, 2, 4, 1, 5, 0, 6]
        back_order = [0, 6, 1, 5, 2, 4, 3]
        want = {}
        rows_front, rows_back = [3, 2], [0, 1]
        spots_f = [(r, c) for r in rows_front for c in front_order]
        spots_b = [(r, c) for r in rows_back for c in back_order]
        for u in melee:
            want[u.id] = spots_f.pop(0) if spots_f else spots_b.pop(0)
        for u in ranged:
            want[u.id] = spots_b.pop(0) if spots_b else spots_f.pop(0)
        return want

    def free_hex_for(self, u, roster):
        occupied = {u2.loc[1:] for u2 in roster.board()}
        want = self.target_hexes(roster.board() + [u])[u.id]
        if want not in occupied:
            return want
        rows = [3, 2, 1, 0] if (not u.info or u.info.melee) else [0, 1, 2, 3]
        for r in rows:
            for c in [3, 2, 4, 1, 5, 0, 6]:
                if (r, c) not in occupied:
                    return (r, c)
        return (0, 0)

    def _position(self, scr, roster):
        if self.position_moves >= 12:        # a full re-arrange never needs more
            return None
        want = self.target_hexes(roster.board())
        for u in roster.board():
            w = want[u.id]
            if u.loc[1:] != w and u.id not in self.positioned and u.name:
                self.positioned.add(u.id)    # once per unit per round: no back-and-forth
                return Action('move', (u, ('board', *w)), f'position {u} -> {w}')
        return None
