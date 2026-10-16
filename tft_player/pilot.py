"""Comp pilot: plays one MetaTFT comp to its final board. The reasoning model makes the
judgement calls - augments and Wisps (player.pilot_handle_augment, player.pilot_wisp) - off
the action path; the rules here do the rest.

The whole game plan comes from the comp's MetaTFT page:
  - the final board (level 8: the comp's units) and star goals,
  - levelling timings (e.g. 5 @2-5, 6 @3-2, 7 @3-5, 8 @4-2),
  - item builds for its carries and tanks, and unit positions,
  - the components its items need (carousel); its augment tiers (a guide for the model, and
    the pick when the model has no answer in time).

Like brain.Brain it proposes ONE action per tick from the live state; the player
executes it, re-reads the screen and asks again. Order each tick:
  sell (room / roll-down) > buy comp units > level on schedule > buy early units >
  roll > field the board > items > positions.

Economy: interest first (never below the current 10-gold step) except for the
scheduled levels, comp units, and 2-star upgrades of fielded units. The level-8
roll-down happens at the comp's 8 timing, or later (up to 4-5) when the gold for it
is not there yet and HP allows waiting; low HP rolls earlier to stabilise.
"""
from __future__ import annotations

import difflib
import math
from collections import Counter

from tft_player.brain import STAR_POWER, Action, Brain
from tft_player.catalog import CATALOG
from tft_player.items import BY_PAIR, COMPONENTS, RECIPES, VALUE, holder_kind, is_remover

XP_TO_NEXT = {1: 2, 2: 2, 3: 6, 4: 10, 5: 20, 6: 36, 7: 56, 8: 68, 9: 68}     # 7->8: 56 on screen (not 60)
COPIES = {1: 1, 2: 3, 3: 9}
ROUNDS = [(1, 1), (1, 2), (1, 3), (1, 4)] + [(s, r) for s in range(2, 8) for r in range(1, 8)]
BASE_INCOME = {(1, 1): 0, (1, 2): 2, (1, 3): 2, (1, 4): 3, (2, 1): 4}      # then 5 a round


def carousel(r):
    return r == (1, 1) or (r[0] >= 2 and r[1] == 4)


def streak_gold(n):
    """Streak bonus for a streak of n (wins or losses) - game-mechanics.md."""
    n = abs(n or 0)
    return 3 if n >= 6 else 2 if n == 5 else 1 if n >= 3 else 0


def gold_augment(name):
    """An augment that pays gold or shop rerolls (MetaTFT/CDragon description)."""
    from tft_player.meta import META
    desc = ((META.augments.get(name or '') or {}).get('desc') or '').lower()
    return any(w in desc for w in ('gold', 'reroll', 'refresh'))


def item_name(name):
    """MetaTFT spelling -> our item names ("Warmogs Armor" -> "Warmog's Armor")."""
    if name in RECIPES or name in COMPONENTS:
        return name
    key = lambda s: s.lower().replace("'", '').replace(' ', '')
    for n in list(RECIPES) + list(COMPONENTS):
        if key(n) == key(name):
            return n
    hit = difflib.get_close_matches(name, list(RECIPES), n=1, cutoff=0.8)
    return hit[0] if hit else name


def xp_cost(level, xp, target):
    """Gold to buy XP from `level` (xp = (current, needed) as shown) up to `target`."""
    if level >= target:
        return 0
    cur, need = xp if xp else (0, XP_TO_NEXT.get(level, 99))
    total = max(0, need - cur)
    for lv in range(level + 1, target):
        total += XP_TO_NEXT.get(lv, 99)
    return 4 * math.ceil(total / 4)


def stage_key(stage):
    if isinstance(stage, str):
        a, b = stage.split('-')
        return int(a), int(b)
    return tuple(stage)


class Pilot(Brain):
    """Brain with a fixed target comp (meta.Comp)."""

    def __init__(self, comp, final_level=8):
        super().__init__()
        self.comp = comp
        self.plan.adopt(comp)
        g = comp.guide or {}
        units = [u for u in comp.units if CATALOG.unit(u)]
        # the level-8 board: the comp's units; a 9-unit list drops its last 5-cost (added at 9)
        self.final = list(units)
        while len(self.final) > final_level:
            fives = [u for u in self.final if CATALOG.unit(u).cost == 5 and u not in comp.carries]
            self.final.remove(fives[-1] if fives else self.final[-1])
        # every comp unit is wanted up to 2 stars (3 copies), never more
        self.star_goal = {u: 2 for u in self.final}
        # the roll-down is done when the carries are 2-star and one of the other 4-costs (the tanks) is
        self.carries2 = [u for u in comp.carries if u in self.final]
        self.tanks2 = [u for u in self.final if CATALOG.unit(u).cost == 4 and u not in self.carries2]
        # priority: carries first, then cost (the scarce 4-costs), then MetaTFT order
        order = {u: i for i, u in enumerate(units)}
        self.priority = sorted(self.final, key=lambda u: (u not in comp.carries, -CATALOG.unit(u).cost, order[u]))
        # levelling schedule: MetaTFT's, plus the usual 4 @2-1 when it names none earlier
        sched = [(stage_key(s), int(lv)) for s, lv in g.get('levels') or []]
        if not sched:
            sched = [((2, 5), 5), ((3, 2), 6), ((3, 5), 7), ((4, 2), 8), ((5, 5), 9)]
        # no level is bought that MetaTFT does not name (its 'Lvl 4: -' = the free XP brings it)
        self.schedule = sorted(sched)
        self.eight_at = next((s for s, lv in self.schedule if lv == 8), (4, 2))
        # items: owner -> [items], owners in priority order (carries first)
        builds = {u: [item_name(i) for i in (b[0]['items'] if b else [])]
                  for u, b in (comp.builds or {}).items() if u in self.final}
        owners = sorted(builds, key=lambda u: (u not in comp.carries, comp.carries.index(u) if u in comp.carries else 0))
        self.builds = {u: [i for i in builds[u] if i in RECIPES] for u in owners}
        # what gets built, in this order: MetaTFT's item list for the comp (most played first:
        # Infinity Edge, Guinsoo's, ...), then the rest of the carries' and tanks' best builds.
        # Nothing outside it is built (a component that only makes other items waits).
        self.item_priority = []
        for it in [item_name(i) for i in (g.get('items') or [])] + [i for b in self.builds.values() for i in b]:
            if it in RECIPES and it not in self.item_priority:
                self.item_priority.append(it)
        self.positions = {u: [tuple(h) for h in hs] for u, hs in (g.get('positions') or {}).items()}
        self.early = {u for boards in (g.get('early') or {}).values() for b in boards for u in b['units']}
        self.augment_tiers = dict(comp.augments or {})
        self.rolled_down = False          # the level-8 roll-down happened
        self.planning = True              # the current tick is in a planning phase (set by decide)
        # tuning knobs (tools/sim_pilot.py compares them)
        self.stage2_econ = True           # stage 2: no pairs, bodies only for empty slots at the interest floor
        self.roll7_above = None           # level 7 from 3-5: roll gold above this for missing 3-cost comp units
        self.rr_after = 'all'             # rounds after the level-8 roll-down, all 8 units found but the 2-star
                                          # goal not met: 'all' rolls down again, 'slow' rolls only gold above 50
        self.spike_stage = None           # the round of the level-8 roll-down
        self.streak = 0                   # current streak, + wins / - losses (the player reads it)
        self.last_plan = None             # the latest econ_plan (the augment picker hides the HUD)
        # MetaTFT: gold this comp's players roll at level 8 on average - the roll-down's budget
        self.rolls8 = float(((g.get('rolls') or {}).get('8')) or 30)
        self.roll_need = max(20, round(self.rolls8))
        self.rr_floor = 4                 # roll-down: never roll below the price of a 4-cost (a hit must be affordable)
        self.plan.item_targets = self.component_needs([])           # refreshed each planning tick

    # -- facts ------------------------------------------------------------------
    @staticmethod
    def copies(roster, name):
        return sum(COPIES.get(u.star, 1) for u in roster.units if u.name == name)

    def need(self, roster, name):
        """Copies still wanted of a final unit (0 if done or not a final unit)."""
        if name not in self.star_goal:
            return 0
        return max(0, COPIES[min(self.star_goal[name], 3)] - self.copies(roster, name))

    def missing(self, roster):
        """Final units we don't own a single copy of."""
        return [u for u in self.priority if not any(x.name == u for x in roster.units)]

    @staticmethod
    def star(roster, name):
        return max((u.star for u in roster.units if u.name == name), default=0)

    def goal_left(self, roster):
        """What the level-8 roll-down still looks for: missing units, the carries to 2 stars,
        then one of the other 4-costs (Amumu or Sentinel) to 2 stars. Empty = done."""
        left = [f'{u} (none)' for u in self.missing(roster)]
        left += [f'{u} 2-star' for u in self.carries2 if self.star(roster, u) < 2]
        if self.tanks2 and not any(self.star(roster, u) >= 2 for u in self.tanks2):
            left.append(f'{" or ".join(self.tanks2)} 2-star')
        return left

    def level_goal(self, stage):
        goal = 1
        for s, lv in self.schedule:
            if stage >= s:
                goal = max(goal, lv)
        return goal

    def hp(self, scr):
        return scr.hp if scr.hp is not None else 100

    def interest_floor(self, gold, stage):
        if stage[0] <= 1:
            return 0
        return min(50, (gold // 10) * 10)

    # -- economy plan -------------------------------------------------------------------
    def project(self, scr, until, level_to=8):
        """Walk the rounds from now to `until` (the level-8 roll-down round): income (base,
        interest, streak, a little loot gold) and passive XP each round, the scheduled levels
        bought on time, level `level_to` bought in `until`. -> (gold left for rolling at
        `until`, level reached)."""
        g, lvl = scr.gold or 0, scr.level or 1
        cur, need = scr.xp if scr.xp else (0, XP_TO_NEXT.get(lvl, 99))
        now = scr.round

        def buy_to(goal):
            nonlocal g, lvl, cur, need
            while lvl < min(goal, 10) and g >= 4:
                g, cur = g - 4, cur + 4
                while lvl < 10 and cur >= need:
                    cur, lvl = cur - need, lvl + 1
                    need = XP_TO_NEXT.get(lvl, 99)

        buy_to(min(self.level_goal(now), 7) if now < until else level_to)
        for r in ROUNDS:
            if r <= now or r > until or carousel(r):
                continue
            g += BASE_INCOME.get(r, 5) + min(5, g // 10) + streak_gold(self.streak)
            if r[1] == 7 and r[0] >= 2:
                g += 3                        # the monster round's loot, roughly
            cur += 2
            while lvl < 10 and cur >= need:
                cur, lvl = cur - need, lvl + 1
                need = XP_TO_NEXT.get(lvl, 99)
            buy_to(level_to if r == until else min(self.level_goal(r), 7))
        return g, lvl

    def econ_plan(self, scr, roster):
        """The gold question of the game: at the level-8 roll-down (MetaTFT's 8 timing, 4-2)
        how much gold is left for rolling after buying level 8, against the gold this comp's
        players roll at level 8 (MetaTFT, ~32). short > 0: behind - gold augments first (when
        no streak pays extra), no extra spending, and level 8 waits for the roll gold while
        HP allows (up to 4-5)."""
        spike = max(self.eight_at, scr.round)
        have, lvl = self.project(scr, spike)
        need = self.roll_need
        if lvl < 8:
            have = 0
        plan = {'spike': spike, 'roll_gold': have, 'roll_need': need, 'short': max(0, need - have)}
        self.last_plan = plan
        self.status['econ'] = (f"level-8 roll-down at {spike[0]}-{spike[1]}: roll gold projected {have}, "
                               f"needed ~{need} (MetaTFT's average roll at 8) -> "
                               + (f"SHORT {plan['short']}: gold augments first, no extras" if plan['short']
                                  else 'on track'))
        return plan

    # -- the tick ------------------------------------------------------------------
    def decide(self, scr, roster, planning):
        if scr.augments:
            return None
        if not scr.shop_visible or scr.gold is None or scr.level is None or not scr.round:
            return None
        stage = scr.round
        self.planning = planning
        if planning and stage < self.eight_at and stage >= (2, 1):
            self.econ_plan(scr, roster)
        steps = [('sell', self._sell, True), ('level', self._level, True), ('buy', self._buy_final, True),
                 ('early', self._buy_early, True), ('roll', self._roll, True),
                 ('board', self._board, planning), ('items', self._items, planning),
                 ('position', self._position, planning)]
        for name, step, on in steps:
            if not on:
                self.status[name] = 'waiting for the planning phase'
                continue
            a = step(scr, roster, stage)
            if a and self.failed[(a.kind, str(a.arg))] < 2:
                self.status[name] = f'ACT {a.kind}: {a.why}'
                return a
            if a:
                self.status[name] = f'gave up on {a.kind} ({a.why}) after 2 failed tries'
        return None

    # -- selling -----------------------------------------------------------------------
    def junk(self, roster, stage):
        """Bench units we can sell, least useful first: named, no items, outside the comp
        (or a surplus copy of a comp unit whose star goal is done)."""
        out = []
        for u in roster.bench():
            if not u.name or u.items or u.id in self.protected:
                continue
            if u.name in self.star_goal:
                have = self.copies(roster, u.name)
                best = max(x.star for x in roster.units if x.name == u.name)
                # a lone 1-star beside the finished 2-star is surplus; anything else is progress
                if not (best >= self.star_goal[u.name] and u.star < best and have > COPIES[self.star_goal[u.name]]):
                    continue
            out.append(u)
        def keep(u):
            pair = len(roster.copies(u.name, u.star)) >= 2
            return (u.cost * STAR_POWER[u.star] + (3 if pair else 0) + (2 if u.name in self.early else 0), u.loc)
        return sorted(out, key=keep)

    def _sell(self, scr, roster, stage):
        free = len(roster.free_bench())
        # 1. room for a final unit in the shop
        want = [n for n in scr.shop if n and self.need(roster, n) > 0 and CATALOG.unit(n).cost <= scr.gold
                and not len(roster.copies(n, 1)) >= 2]
        if want and free == 0:
            j = self.junk(roster, stage)
            if j:
                return Action('sell', j[0], f'room for {want[0]}: sell {j[0]}')
        # 2. the level-8 roll-down: every champion outside the 8 goes before anything is bought -
        #    bench first, then the board (planning only: fielded units cannot be sold in combat);
        #    their items come back to the item bench and go to the comp's units
        if self.rolling_now(scr, roster, stage) and (scr.level or 0) >= 8:
            out = [u for u in roster.bench() + (roster.board() if self.planning else [])
                   if u.name and u.name not in self.star_goal and u.id not in self.protected]
            out += [u for u in self.junk(roster, stage) if u not in out]      # surplus comp copies
            if out:
                return Action('sell', out[0], f'roll-down: sell {out[0]} (not one of the 8)')
        elif self.rolling_now(scr, roster, stage):
            j = self.junk(roster, stage)
            if j:
                return Action('sell', j[0], f'roll-down: sell {j[0]}')
        # 3. keep two slots free from stage 3 (shops bring comp units)
        keep_free = 1 if stage[0] <= 2 else 2
        if free < keep_free:
            j = self.junk(roster, stage)
            if j and (stage[0] >= 3 or free == 0):
                return Action('sell', j[0], f'bench space: sell {j[0]}')
        # 4. items of a final unit's build on a benched holder: sell it, the items come back
        a = self._release_items(scr, roster)
        if a:
            return a
        return self.why('sell', f'bench {9 - free}/9, nothing to sell')

    # -- buying -----------------------------------------------------------------------------
    def spend_floor(self, scr, stage, kind):
        """Lowest gold a purchase of this kind may leave. Stage 2: never below the current
        10-gold step (interest); later a comp unit or an upgrade may cost one step."""
        gold, hp = scr.gold, self.hp(scr)
        base = self.interest_floor(gold, stage)
        if stage[0] <= 1:
            return 0
        if stage[0] == 2:
            return base
        if kind == 'final':
            # comp units are the plan: may cost one interest step; in the roll-down every gold
            # (the rolls keep 4 so a 4-cost they show can be bought)
            if self.rolling_now(scr, None, stage):
                return 0
            return max(0, base - 10)
        if kind == 'upgrade':
            return max(0, base - 10) if stage[0] >= 3 or hp < 70 else base
        return base

    def _buy_final(self, scr, roster, stage):
        """Any copy of the comp's units that shows up, unless that unit is already 2-star
        (never more than 3 copies)."""
        for i, name in enumerate(scr.shop):
            if not name or self.need(roster, name) <= 0:
                continue
            cost = CATALOG.unit(name).cost
            merges = len(roster.copies(name, 1)) >= 2
            if not merges and not roster.free_bench():
                continue                       # _sell makes room first
            if scr.gold - cost < self.spend_floor(scr, stage, 'final') and not (merges and scr.gold >= cost):
                continue
            return Action('buy', i, f'comp unit {name} ({self.copies(roster, name) + 1}/3)')
        left = self.goal_left(roster)
        return self.why('buy', 'no comp unit to buy in the shop' + (f' (looking for {", ".join(left)})' if left else ''))

    def _buy_early(self, scr, roster, stage):
        """Units outside the comp: strength until the comp is here. Pairs and upgrades of
        fielded units, and bodies for empty board slots."""
        if stage >= self.eight_at and scr.level >= 8:
            return self.why('early', 'level 8: only comp units')
        fielded = roster.slots_used()
        # the game's own count ('n/m', team slots) when read: the roster can miss units the game placed
        seen = scr.board_count if scr.board_count is not None else roster.board_seen
        empty = (getattr(scr, 'team_size', None) or scr.level) - max(fielded, seen)
        board_names = {u.name for u in roster.board() if u.name}
        have = Counter()
        for n in board_names:
            have.update(CATALOG.unit(n).traits if CATALOG.unit(n) else ())
        short = (2, 1) <= stage < self.eight_at and self.econ_plan(scr, roster)['short'] > 0
        best = None
        for i, name in enumerate(scr.shop):
            if not name or name in self.star_goal or not CATALOG.unit(name):
                continue
            info = CATALOG.unit(name)
            ones = len(roster.copies(name, 1))
            twos = len(roster.copies(name, 2))
            if twos or info.cost > scr.gold:
                continue
            fielded_copy = any(u.name == name and u.on_board for u in roster.units)
            if ones == 2:
                kind, v = 'upgrade', 10 + info.cost
            elif ones == 1 and stage[0] <= 2 and (fielded_copy or name in self.early) and not self.stage2_econ:
                kind, v = 'pair', 5 + info.cost
            elif empty > 0 and len(roster.units) < scr.level + 2:
                kind, v = 'body', 3 + info.cost
            elif name not in board_names and self.activates(info, have) and \
                    (empty > 0 or info.cost + 0.6 >= min((self.unit_power(u) for u in roster.board()), default=0)):
                kind, v = 'trait', 4 + info.cost     # a trait goes live with what is fielded
            else:
                continue
            if name in self.early:
                v += 2
            if not roster.free_bench() and kind != 'upgrade':
                continue
            floor = self.spend_floor(scr, stage, 'upgrade' if kind == 'upgrade' else 'early')
            if kind == 'body' and empty > 0 and stage[0] <= 3 and not (self.stage2_econ and stage[0] == 2):
                floor = max(0, floor - 10)     # an empty slot loses fights
            if short and kind != 'upgrade' and empty < 2:
                continue                       # behind on gold for the roll-down: nothing extra
            if scr.gold - info.cost < floor:
                continue
            if best is None or v > best[0]:
                best = (v, i, name, kind)
        if best:
            return Action('buy', best[1], f'{best[3]}: {best[2]}')
        return self.why('early', 'nothing worth buying outside the comp')

    @staticmethod
    def activates(info, have):
        """Would fielding this champion reach a trait breakpoint the board does not have yet?"""
        for t in info.traits:
            th = CATALOG.traits.get(t) or ()
            if th and have.get(t, 0) < th[0] <= have.get(t, 0) + 1:
                return True
        return False

    # -- levels and rolls ----------------------------------------------------------------------
    # gold kept after a scheduled level-up; short of it we wait a round (HP permitting)
    # gold left after a scheduled level-up; level 8 waits (HP > 40, until 4-5) until ~20 remain to roll
    LEVEL_KEEP = {4: 0, 5: 0, 6: 0, 7: 10, 8: 20, 9: 10}

    def level_now(self, scr, stage, goal):
        """Buy XP toward `goal` now? The level is taken only when it completes with the
        reserve left (level 8: the roll-down budget), or when HP is low and it completes at all."""
        cost = xp_cost(scr.level, scr.xp, goal)
        hp = self.hp(scr)
        keep = self.LEVEL_KEEP.get(goal, 0)
        if hp <= 40 or (goal == 8 and stage >= (4, 5)):
            keep = 0
        return scr.gold >= 4 and scr.gold - cost >= keep, cost

    def _level(self, scr, roster, stage):
        lvl, gold = scr.level, scr.gold
        if lvl >= 10:
            return self.why('level', 'level 10')
        goal = self.level_goal(stage)
        if lvl < goal:
            ok, cost = self.level_now(scr, stage, goal)
            if ok:
                return Action('level', None, f'level {lvl}->{goal} on schedule (stage {stage[0]}-{stage[1]}, '
                                             f'{cost} gold to go)')
            waiting = f'level {goal} due at this stage costs {cost}, gold {gold}: waiting'
        else:
            waiting = f'level {lvl} (goal {goal})'
        # gold above 50 goes to XP once the comp is on its way (fast 8), not before 3-2
        if stage >= (3, 2) and lvl < 8 and gold >= 54 and not self.rolling_now(scr, roster, stage):
            return Action('level', None, f'gold {gold} over 50: XP toward the next level')
        if stage >= (5, 1) and lvl == 8 and gold >= 54 and not self.goal_left(roster):
            return Action('level', None, f'roll-down goal met, gold {gold}: XP toward 9')
        return self.why('level', waiting)

    def roll_floor(self, scr, stage):
        """Gold a roll must leave: enough to buy the 4-cost it may show (rolling to 0 sees
        units it cannot pay for, and the next round's shop replaces them)."""
        return self.rr_floor

    def rolling_now(self, scr, roster, stage):
        """The roll-down window: level 8 at or after the comp's 8 timing with the board not
        complete; or HP low at level 7+ in stage 4 when level 8 is out of reach."""
        if scr.level is None:
            return False
        if scr.level >= 8 and stage >= self.eight_at:
            return roster is None or bool(self.goal_left(roster))
        hp = self.hp(scr)
        if scr.level >= 7 and stage >= self.eight_at and hp <= 30:
            return not self.level_now(scr, stage, 8)[0]
        return False

    def _roll(self, scr, roster, stage):
        gold, lvl, hp = scr.gold, scr.level, self.hp(scr)
        if not roster.free_bench() and not self.junk(roster, stage):
            return self.why('roll', 'bench full of keepers')
        if self.rolling_now(scr, roster, stage):
            floor = self.roll_floor(scr, stage)
            if lvl >= 8 and self.spike_stage is None:
                self.spike_stage = stage
            if self.rr_after == 'slow' and lvl >= 8 and stage != self.spike_stage and not self.missing(roster):
                floor = max(floor, 50)         # later rounds: the 2-stars come from gold above 50
            if gold - 2 >= floor:
                self.rolled_down = True
                return Action('roll', None, f'roll-down at {lvl}: gold {gold} floor {floor}, looking for '
                                            f'{", ".join(self.goal_left(roster))}')
            return self.why('roll', f'at the roll floor ({gold} <= {floor + 1})')
        # level 7 before the 8 spike: gold above a line rolls for the comp's missing 3-costs
        if self.roll7_above is not None and lvl == 7 and stage < self.eight_at and \
                gold - 2 >= self.roll7_above and any(CATALOG.unit(u).cost <= 3 for u in self.missing(roster)):
            return Action('roll', None, f'level 7: gold {gold} over {self.roll7_above}, looking for '
                                        f'{", ".join(u for u in self.missing(roster) if CATALOG.unit(u).cost <= 3)}')
        # roll-down done: extra gold above 50 buys the other 2-stars
        if lvl >= 8 and not self.goal_left(roster) and gold >= 52 and stage >= self.eight_at:
            ups = [u for u in self.priority if self.need(roster, u) > 0]
            if ups:
                return Action('roll', None, f'gold {gold} over 50: roll for {", ".join(ups[:3])}')
        # emergency before the level-8 spike: HP low, stabilise with what we have
        if stage[0] >= 3 and hp <= 30 and gold >= 12:
            return Action('roll', None, f'hp {hp}: roll to stabilise')
        return self.why('roll', f'saving: gold {gold}, level {lvl}, hp {hp}')

    # -- the board ----------------------------------------------------------------------------
    def unit_power(self, u):
        base = (u.cost + 0.6) * STAR_POWER.get(u.star, 1)
        base += sum(1.2 if i in COMPONENTS else 3.0 for i in u.items)
        if u.name in self.star_goal:
            base *= 1.1                  # a tie-break: before the roll-down the strongest board plays
        return base

    def lineup(self, roster, level, stage=(9, 9)):
        """The units to field. From the level-8 roll-down on: the comp's 8 (best copy of each,
        priority order), others only while a slot has no comp unit for it. Before that: the
        strongest board - unit power (cost, stars, items) plus active traits, comp units only
        as a tie-break - so HP holds until the comp is there."""
        units = [u for u in roster.units]
        chosen = []
        if level >= 8 and stage >= self.eight_at:
            for name in self.priority:
                copies = [u for u in units if u.name == name]
                best = max(copies, key=lambda u: (u.star, len(u.items))) if copies else None
                if best is not None and sum(c.slots for c in chosen) + best.slots <= level:
                    chosen.append(best)       # team slots, not units: an Elder Dragon fills 2
        rest = [u for u in units if u not in chosen and not any(c.name == u.name and u.name for c in chosen)]
        while rest and sum(c.slots for c in chosen) < level:
            base = self.trait_bonus(chosen)
            u = max(rest, key=lambda x: self.unit_power(x) + self.trait_bonus(chosen + [x]) - base
                    - (3 if x.name and any(c.name == x.name for c in chosen) else 0))
            rest.remove(u)
            if sum(c.slots for c in chosen) + u.slots <= level:
                chosen.append(u)
        return chosen

    def _board(self, scr, roster, stage):
        board, level = roster.board(), getattr(scr, 'team_size', None) or scr.level     # team slots
        final = scr.level >= 8 and stage >= self.eight_at
        ideal = self.lineup(roster, level, stage)
        ids = {u.id for u in ideal}
        incoming = [u for u in ideal if not u.on_board]
        if roster.slots_used() > level and roster.free_bench():
            worst = min([u for u in board if u.id not in ids] or board, key=self.unit_power)
            return Action('move', (worst, ('bench', roster.free_bench()[0])), f'over the limit: bench {worst}')
        if not incoming:
            return self.why('board', f'lineup fielded ({len(board)}/{level})')
        u = incoming[0] if final and incoming[0].name in self.star_goal else max(incoming, key=self.unit_power)
        outgoing = [b for b in board if b.id not in ids]
        if max(roster.slots_used(), roster.board_seen) + u.slots <= level:
            hex_ = self.free_hex_for(u, roster)
            return Action('move', (u, ('board', *hex_)), f'field {u} (board {roster.slots_used()}/{level})')
        if outgoing:
            worst = min(outgoing, key=self.unit_power)
            if not final and self.unit_power(u) < self.unit_power(worst) + 1.0:
                return self.why('board', f'{u} is not clearly better than {worst}: no swap')
            if roster.free_bench():
                # bench it first: the incoming unit is then fielded on its own hex (a swap drops
                # it on the outgoing unit's hex - a backline carry in the front row)
                return Action('move', (worst, ('bench', roster.free_bench()[0])), f'make room for {u}: bench {worst}')
            return Action('move', (u, worst.loc), f'swap in {u} for {worst}')
        return self.why('board', f'board full ({len(board)}/{level})')

    def _position(self, scr, roster, stage):
        return super()._position(scr, roster)

    def target_hexes(self, board):
        """MetaTFT positions for comp units (carries first); others by role."""
        want, taken = {}, set()
        order = sorted(board, key=lambda u: (u.name not in self.star_goal,
                                             self.priority.index(u.name) if u.name in self.star_goal else 0, u.id))
        others = []
        for u in order:
            hexes = self.positions.get(u.name) or []
            h = next((h for h in hexes if h not in taken), None)
            if h is None:
                others.append(u)
                continue
            want[u.id] = h
            taken.add(h)
        front = [(r, c) for r in (3, 2) for c in (3, 2, 4, 1, 5, 0, 6)]
        back = [(r, c) for r in (0, 1) for c in (0, 6, 1, 5, 2, 4, 3)]
        for u in others:
            melee = not u.info or u.info.melee
            spots = [h for h in (front + back if melee else back + front) if h not in taken]
            want[u.id] = spots[0]
            taken.add(spots[0])
        return want

    # -- items ----------------------------------------------------------------------------------
    def owner_of(self, item, roster, exclude=()):
        """The first build owner (priority order) that wants `item` and doesn't hold it yet."""
        for owner, items in self.builds.items():
            if owner in exclude:
                continue
            want = Counter(items)
            have = Counter(i for u in roster.units if u.name == owner for i in u.items)
            if want[item] > have[item]:
                return owner
        return None

    def wanted_items(self, roster):
        """[(owner, item)] still to build, in the comp's item priority. owner: the unit whose best
        build has the item (None: on MetaTFT's list but in no build - the best holder for its
        role gets it). Items already built anywhere count."""
        held = Counter(i for u in roster.units for i in u.items if i in RECIPES)
        out = []
        for it in self.item_priority:
            owners = [o for o, items in self.builds.items() for _ in range(items.count(it))] or [None]
            for o in owners:
                if held[it] > 0:
                    held[it] -= 1
                    continue
                out.append((o, it))
        return out

    def component_needs(self, bench_items, roster=None):
        """Components to take from a carousel, best first: one that completes the next wanted
        item with a component we hold, then the wanted items' components in priority order."""
        have = Counter(n for n in bench_items or [] if n in COMPONENTS)
        wanted = self.wanted_items(roster) if roster is not None else \
            [(None, it) for it in self.item_priority]
        first, rest = [], []
        for _, it in wanted:
            a, b = RECIPES[it]
            for x, y in ((a, b), (b, a)):
                if have[x] and y not in first:
                    first.append(y)
            for c in (a, b):
                if c not in rest:
                    rest.append(c)
        return first + [c for c in rest if c not in first]

    def holder(self, roster, owner):
        """Who holds `owner`'s items now: the owner if fielded; else a fielded unit outside the
        comp (it is sold later and the items come back), same role first."""
        board = roster.board()
        own = next((u for u in sorted(board, key=lambda u: -u.star) if u.name == owner and len(u.items) < 3), None)
        if own:
            return own
        carry_items = owner in self.comp.carries or (CATALOG.unit(owner) and not CATALOG.unit(owner).melee)
        cands = [u for u in board if u.name and u.name not in self.star_goal and len(u.items) < 3
                 and not any(i in COMPONENTS for i in u.items)]
        if not cands:
            return None
        role = lambda u: (u.info and not u.info.melee) == carry_items
        return max(cands, key=lambda u: (role(u), len(u.items), self.unit_power(u)))

    def spare_holder(self, roster, item):
        """For an item outside the builds: a fielded unit outside the comp (sold later, the
        item comes back) or a comp unit without a MetaTFT build, where the item suits its role."""
        best = None
        for u in roster.board():
            if len(u.items) >= 3 or any(i in COMPONENTS for i in u.items) or u.name in self.builds:
                continue
            v = VALUE.get(holder_kind(u) or 'tank', {}).get(item, 0)
            if v <= 0:
                continue
            v += 0.5 * self.unit_power(u) + (2 if u.name in self.star_goal else 0)
            if best is None or v > best[0]:
                best = (v, u)
        return best[1] if best else None

    def _items(self, scr, roster, stage):
        """Items from the comp's list only, in its order: finish a half-built one, hand out a
        completed one, or start the next one whose two components are on the bench."""
        bench = list(scr.items or [])
        self.plan.item_targets = self.component_needs(bench, roster)      # the carousel's wishes
        if not any(bench):
            return self.why('items', 'item bench empty')
        comps = [(i, n) for i, n in enumerate(bench) if n in COMPONENTS]
        wanted = self.wanted_items(roster)
        rank = {it: k for k, (_, it) in enumerate(reversed(wanted))}     # higher = earlier in the list
        # 0. a Remover and a fielded owner whose items sit on a comp unit that isn't theirs
        a = self._remove(scr, roster, bench)
        if a:
            return a
        # 1. finish a half-built item: the unit's lone component + a bench component
        for u in roster.board():
            for c in [c for c in u.items if c in COMPONENTS]:
                options = []
                for i, n in comps:
                    it = BY_PAIR.get(tuple(sorted((c, n))))
                    if it and any(it == w and (o == u.name or o is None or u.name not in self.builds)
                                  for o, w in wanted):
                        options.append((rank[it], i, n, it))
                if options:
                    _, i, n, it = max(options)
                    return Action('item', (i, n, u), f'finish {it} on {u}')
        # 2. completed items on the bench (anvils, carousel, a sold holder)
        for i, n in enumerate(bench):
            if n in RECIPES:
                owner = self.owner_of(n, roster)
                h = self.holder(roster, owner) if owner else self.spare_holder(roster, n)
                if h:
                    return Action('item', (i, n, h), f'equip {n} ({"for " + owner if owner else "spare"}) on {h}')
        # 3. the next item in the comp's priority whose two components are on the bench
        for owner, it in wanted:
            a_, b_ = RECIPES[it]
            sa = next((i for i, n in comps if n == a_), None)
            sb = next((i for i, n in comps if n == b_ and i != sa), None)
            if sa is None or sb is None:
                continue
            h = self.holder(roster, owner) if owner else self.spare_holder(roster, it)
            if h is None or len(h.items) > 2:
                continue
            return Action('item', (sa, a_, h), f'build {it} ({self.item_priority.index(it) + 1}. on the comp list) '
                                               f'for {owner or "its best holder"} on {h}')
        return self.why('items', f'holding {[n for n in bench if n]} (no item of the comp list can be made)')

    def best_any_holder(self, roster, item):
        """A fielded comp unit with room, where the item has value for its role."""
        best = None
        for u in roster.board():
            if len(u.items) >= 3 or any(i in COMPONENTS for i in u.items):
                continue
            v = VALUE.get(holder_kind(u) or 'tank', {}).get(item, 0)
            if u.name in self.builds and item not in self.builds[u.name]:
                v *= 0.3 if len(u.items) + len([i for i in self.builds[u.name] if i not in u.items]) >= 3 else 1
            if v > 0 and (best is None or v > best[0]):
                best = (v, u)
        return best[1] if best else None

    def _remove(self, scr, roster, bench):
        slot = next((i for i, n in enumerate(bench) if is_remover(n)), None)
        if slot is None or sum(1 for n in bench if not n) < 3:
            return None
        board = roster.board()
        for owner in self.builds:
            if not any(u.name == owner for u in board):
                continue
            for u in board:
                if u.name == owner or u.id in self.stripped or not u.items:
                    continue
                mine = [i for i in u.items if i in self.builds[owner]]
                if mine and (u.name in self.star_goal or u.name in self.builds):
                    return Action('strip', (slot, bench[slot], u), f'remover: {", ".join(mine)} belong on {owner}')
        return None

    def _release_items(self, scr, roster):
        """A unit outside the comp holding items that a fielded comp unit's build wants: sell
        it and the items come back to the item bench. Benched holders always; a fielded
        holder when a benched unit can take its place. Planning only."""
        if not self.planning:
            return None                          # the item bench is read in planning only
        board = roster.board()
        board_names = {u.name for u in board}
        spare = [u for u in roster.bench() if u.name and not u.items]
        for u in roster.bench() + (board if spare else []):
            if not u.items or u.name in self.star_goal or u.id in self.protected:
                continue
            owed = [i for i in u.items if i in RECIPES and self.owner_of(i, roster) in board_names]
            if owed or (not u.on_board and any(i in COMPONENTS for i in u.items)):
                return Action('sell', u, f'free {", ".join(u.items)} for '
                                         f'{self.owner_of(owed[0], roster) if owed else "the comp"}')
        return None

    # -- augments -------------------------------------------------------------------------------
    def augment_score(self, name, gold_first=False):
        """The comp's MetaTFT tier (S 4 .. C 1). An augment it does not rate: MetaTFT's general
        tier one lower, and an augment for a trait the comp does not play (an emblem, a trait
        bonus) drops 2 more. gold_first: augments paying gold or rerolls +2.5."""
        score = {'S': 4, 'A': 3, 'B': 2, 'C': 1}
        if not name:
            return 0
        if name in self.augment_tiers:
            v = score.get(self.augment_tiers[name], 1.5)
        else:
            from tft_player.meta import META
            v = score.get((META.augments.get(name) or {}).get('tier'), 1.5) - 1
            if self.off_comp_trait(name):
                return v - 2                  # its bit of gold (Consuming Flora: 3) doesn't make it a gold augment
        return v + (2.5 if gold_first and gold_augment(name) else 0)

    def off_comp_trait(self, name):
        """Does the augment name a trait none of the comp's units has (e.g. Consuming Flora:
        a Flora Fatalis emblem for Lunar Aphelios)?"""
        from tft_player.meta import META
        desc = (META.augments.get(name) or {}).get('desc') or ''
        mine = {t for u in self.final for t in (CATALOG.unit(u).traits if CATALOG.unit(u) else ())}
        named = {t for t in CATALOG.traits if t and t in desc}
        return bool(named) and not (named & mine)

    def augment_decision(self, names, rerollable, gold_first=False):
        """('pick', i) or ('reroll', i): the best by augment_score; a card under A (3) that is
        not the best is rerolled while it can be."""
        vals = [self.augment_score(n, gold_first) for n in names]
        best = max(range(len(vals)), key=lambda i: vals[i])
        worst = [i for i in range(len(vals)) if i in rerollable and vals[i] < 3 and i != best]
        if worst and all(n for n in names):
            return ('reroll', min(worst, key=lambda i: vals[i])), vals
        return ('pick', best), vals

    def choice_value(self, name, roster, bench_items):
        """How much the comp wants an option of a choice screen (an anvil's items): an item of
        its list by its place there, a component by what it completes, anything else little."""
        if not name:
            return -1
        if name in self.item_priority:
            return 100 - self.item_priority.index(name)
        if name in COMPONENTS:
            needs = self.component_needs(bench_items, roster)
            return 50 - needs.index(name) if name in needs else 10
        return 1

    def gold_first(self, scr, roster):
        """Take gold augments first: behind on gold for the roll-down and no streak. The picker
        hides the HUD, so the plan from the last planning phase is used then."""
        if scr is not None and scr.gold is not None and scr.level is not None and scr.round:
            plan = self.econ_plan(scr, roster)
        else:
            plan = self.last_plan
        return bool(plan) and plan['short'] > 0 and abs(self.streak or 0) < 3

def wisp_names():
    """Every Wisp's name (data/reference/wisps.md)."""
    import re
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / 'data' / 'reference' / 'wisps.md'
    try:
        rows = path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return []
    return [m.group(1).strip() for m in (re.match(r'\|\s*([^|]+?)\s*\|', r) for r in rows)
            if m and m.group(1).strip() not in ('Wisp', '---')]


