"""What we own and where it is.

Identity comes from events, never from pictures: a champion we bought (its
name read off the shop card) or a name read off the info panel stays with that
unit through moves and merges until we sell it. Health-bar occupancy (read each
planning frame) only corrects positions; it never deletes or renames a named
unit. Units that appear without an action (loot, carousel) are unknown until
the info panel reads them.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from tft_player.catalog import CATALOG

_ids = itertools.count(1)

Loc = tuple  # ('bench', i) | ('board', row, col)


BFF = 'BFF'      # the Sprykin trait's Big Furry Friend: a unit on the board, not a champion, no team slot

# Placeable objects the game data has no unit entry for: the Elderwood trait's plants ("Gain placeable Elderwood
# plants": a Stonebark Tree and a Lifebloom at 3, a second Stonebark Tree at 5, the Deepwood Protector at 7) and
# the Potted Stonebark / Potted Lifebloom Wisps (a temporary one; Lifeblossom = upgraded). Effects as guides give
# them (outof.games Set 18 traits, 2026-10-13); whether they use a team slot is not confirmed - like the Training
# Dummy they are recorded as using none until a game shows otherwise.
# The user, 2026-10-13 (screenshot of an Elderwood 7 board): plants are units on the board, beside the team (10/10
# with 4 plants fielded); they cannot be sold, only moved. 3 Elderwood: a Stonebark Tree and a Lifebloom, 5: a
# second Stonebark Tree, 7: the Deepwood Protector.
_PLANT = 'an Elderwood plant on your board: uses no team slot, cannot be sold, only moved. '
PLACEABLES = {
    'Stonebark Tree': _PLANT + 'A tank tree, usually frontline: stuns nearby enemies when it dies',
    'Lifebloom': _PLANT + 'Buffs Elderwood allies (Attack Damage and Ability Power) all combat; usually backline',
    'Lifeblossom': _PLANT + 'The upgraded Lifebloom (a Wisp): buffs Elderwood allies all combat; usually backline',
    'Deepwood Protector': _PLANT + 'The strong 7-Elderwood summon: stuns and knocks up',
}


def board_objects():
    """{name: what it does} of the units in the set data that are not champions (no traits): the Training Dummy
    a Wisp gives, PvE monsters. Read off a panel, they are recorded by that name; on our board they use no team
    slot (run 20261010-235723 2-6: the game counted 4/4 with a Training Dummy as the fifth body)."""
    global _OBJECTS
    if _OBJECTS is None:
        import json
        from pathlib import Path
        _OBJECTS = {}
        try:
            src = sorted((Path(__file__).resolve().parent.parent / 'data' / 'sources').glob('cdragon-tft-*.json'))[-1]
            d = json.loads(src.read_text(encoding='utf-8'))
            for c in d['sets']['18']['champions']:
                # (Krug and Murkwolf are champions too: a PvE copy with the same name is no object)
                if not c.get('traits') and c.get('name') and (c.get('cost') or 0) <= 5                         and CATALOG.unit(c['name']) is None:
                    _OBJECTS[c['name']] = ((c.get('ability') or {}).get('desc') or '').strip()
        except (OSError, ValueError, KeyError, IndexError):
            pass
        _OBJECTS.update(PLACEABLES)
    return _OBJECTS


_OBJECTS = None


@dataclass
class Owned:
    name: str | None          # None = unknown unit
    star: int = 1
    loc: Loc = ('bench', 0)
    id: int = field(default_factory=lambda: next(_ids))
    items: list = field(default_factory=list)     # item names READ over the unit ('unknown' = an
                                                  # item is there, not recognised)
    ours: bool = False        # created by our own buy (not a reward/loot/carousel unit)
    missing: str | None = None    # not seen where tracked: why (kept, never silently deleted)
    unverified: str | None = None  # identity or place inferred, not read: why
    items_read: bool = False   # items come from a read of its icon row (False: none read yet)
    evolved: list = field(default_factory=list)   # traits it gained for good by a choice (Kha'Zix's evolution)

    @property
    def slotless(self):
        """Uses no team slot: the BFF, and board objects such as the Training Dummy (not champions)."""
        return self.name == BFF or (self.name is not None and self.name in board_objects())

    @property
    def slots(self):
        """Team slots it uses when fielded: 0 for the BFF and board objects, 2 for a champion the game data says
        takes up 2 (Elder Dragon), else 1."""
        if self.slotless:
            return 0
        from tft_player.trait_text import unit_slots
        return unit_slots(self.name) if self.name else 1

    @property
    def traits(self):
        """Its traits: the champion's own, plus the ones it evolved into."""
        own = tuple(self.info.traits) if self.info else ()
        return own + tuple(t for t in self.evolved if t not in own)

    @property
    def info(self):
        return CATALOG.unit(self.name) if self.name else None

    @property
    def cost(self):
        return self.info.cost if self.info else 1

    @property
    def on_board(self):
        return self.loc[0] == 'board' and not self.missing

    def __repr__(self):
        where = f'b{self.loc[1]}' if self.loc[0] == 'bench' else f'h{self.loc[1]}{self.loc[2]}'
        it = f"[{','.join(i.split()[0] for i in self.items)}]" if self.items else ''
        flag = '(missing)' if self.missing else '(unverified)' if self.unverified else ''
        return f"{self.name or '?'}{'*' * self.star}@{where}{it}{flag}"


def _hex_near(a, b):
    from tft_player.board import hex_xy
    (x1, y1), (x2, y2) = hex_xy(*a), hex_xy(*b)
    return abs(x1 - x2) < 140 and abs(y1 - y2) < 90


class Roster:
    def __init__(self):
        self.units: list[Owned] = []
        self._missing: dict[int, int] = {}   # unit id -> consecutive planning reads unseen
        self.board_seen = 0                  # units fielded according to the screen
        self._board_diff: list = []          # recent (observed - tracked) board counts
        self.pending_buys: list = []         # (name, time) buys not yet confirmed on screen
        self._gone_since: dict[int, float] = {}   # unit id -> when it was first missed from the bench
        self.left_bench: list = []           # named units gone from the bench (last reconcile)
        self.lost: list = []                 # units newly marked missing (last reconcile, reported)
        self.blocked: dict[int, str] = {}    # bench slot -> what occupies it that is not a unit
                                             # (a container, an unrecognised object): never free
        self.alias: dict[int, int] = {}      # unit id merged away -> the id of the unit it became
        self.stage = None                    # for messages

    # -- queries -----------------------------------------------------------
    def at(self, loc):
        return next((u for u in self.units if u.loc == loc and not u.missing), None)

    def bench(self):
        return sorted((u for u in self.units if u.loc[0] == 'bench' and not u.missing), key=lambda u: u.loc[1])

    def board(self):
        return [u for u in self.units if u.on_board]

    def fielded(self):
        """Board units that use a team slot (team size = level): all but a BFF (the Sprykin trait's Big Furry
        Friend stands on the board with a health bar and its own panel, but the game does not count it - the
        user, 2026-10-10: 'it is still 7 units, not 8')."""
        return [u for u in self.units if u.on_board and not u.slotless]

    def slots_used(self):
        """Team slots the fielded units use (Elder Dragon 2, the BFF and board objects 0)."""
        return sum(u.slots for u in self.fielded())

    def mark_used(self, item, unit, stage):
        """A reusable item (the Alpha Mark) used on a unit: the latest uses, newest first."""
        self.marks = [(item, unit.id, stage)] + [m for m in getattr(self, 'marks', []) if m[1] != unit.id][:3]

    def evolve(self, name, trait):
        """A champion gained a trait for good (Kha'Zix's evolution): every copy we hold carries it, so the
        merged or fielded one does too."""
        for u in self.units:
            if u.name == name and trait not in u.evolved:
                u.evolved.append(trait)

    def missing(self):
        return [u for u in self.units if u.missing]

    def free_bench(self):
        """Bench slots a unit can be put in or land in: no unit we track there and nothing else
        (a container occupies its slot - run27: Elise to slot 1, Alune to slot 6 refused 6x)."""
        taken = {u.loc[1] for u in self.bench()}
        return [i for i in range(9) if i not in taken and i not in self.blocked]

    def copies(self, name, star=1):
        return [u for u in self.units if u.name == name and u.star == star and not u.missing]

    def resolve(self, uid):
        """A unit id as the model knows it -> the unit now (an id merged away -> the merged unit)."""
        seen = set()
        while uid in self.alias and uid not in seen:
            seen.add(uid)
            uid = self.alias[uid]
        return next((u for u in self.units if u.id == uid), None)

    # -- our own actions -----------------------------------------------------
    def bought(self, name, ours=True, slot=None, merge=True):
        """slot: where the unit was seen landing (the slot that lit up). Else predicted: the game
        puts it in the leftmost free bench slot (a container's slot is not free), then merges.
        Also a Duplicator's copy (ours=True, it is our action)."""
        import time as _t
        seen = slot is not None
        if slot is None:
            free = self.free_bench()
            slot = free[0] if free else 8
        else:
            other = self.at(('bench', slot))
            if other is not None:
                # its slot was dark before the buy: it is not there (reconcile finds it again)
                other.missing = f'tracked at bench slot {slot}, but that slot was empty before a buy'
        new = Owned(name, 1, ('bench', slot), ours=ours)
        new.bought_at, new.bought_loc = _t.time(), ('bench', slot)
        new.sure = seen                 # landing seen: name and place certain (gallery crops)
        self.units.append(new)
        # if the game doesn't merge as we assumed, the unit shows up in this slot: it is ours
        self.recent_buys = [b for b in getattr(self, 'recent_buys', []) if _t.time() - b[2] < 6] + [(name, slot, _t.time())]
        if merge:                       # (False: it was seen landing although two copies were held - no merge)
            self._merge(name, 1)
        return new

    def _merge(self, name, star):
        group = self.copies(name, star)
        if len(group) < 3 or star >= 3:
            return
        group = group[:3]
        if getattr(self, 'in_combat', False) and any(u.on_board for u in group):
            # a copy fighting on the board merges only when the fight ends (run 033507 4-3: merges assumed
            # in combat made 1-star Nidalee, Diana and Sentinel "2-star", and auto-sell sold the copies the
            # game would have merged). set_star records the merge once the badge shows it.
            return
        keep = next((u for u in group if u.on_board), None) or min(group, key=lambda u: u.loc[1])
        # the merged unit's place is an assumption until the screen confirms it (player.confirm_merge)
        self.last_merge = {'unit': keep, 'locs': [u.loc for u in group]}
        for u in group:
            if u is not keep:
                # the game moves the copies' items to the merged unit; what it holds is read
                # from its icon row afterwards (items here are only what was last read)
                keep.items = (keep.items + u.items)[:3]
                keep.items_read = False
                keep.evolved += [t for t in u.evolved if t not in keep.evolved]
                self.alias[u.id] = keep.id
                self.units.remove(u)
        keep.star = star + 1
        keep.star_assumed = True             # until a badge or panel read confirms it
        keep.look = None                     # a merged unit is bigger: its look is taken again
        self._merge(name, star + 1)

    def settle_merges(self):
        """The fight ended: copies bought during it merge now with the copy that fought (the game merges
        them as combat ends). Run 152104 5-3: a third Aphelios bought in combat stayed three 1-stars in
        the records until a board check two rounds later. -> names merged."""
        merged = []
        for name in sorted({u.name for u in self.units if u.name}):
            for star in (1, 2):
                if len(self.copies(name, star)) >= 3:
                    self._merge(name, star)
                    merged.append(name)
        return merged

    def set_star(self, u, star, visible):
        """The screen's star badge says u is `star`. visible: the places with a health bar now. A rise is a merge
        that happened (maybe one we did not make: three Wisp/augment/monster-round copies, a copy bought in
        combat that merged when it ended): the copies it used are gone from the screen and leave the roster
        (their items move to u). A fall is a merge we assumed that did not happen: the copies it removed
        come back as units that arrived without a buy. -> (old star, [removed copies])."""
        old, gone = u.star, []
        if star > old and u.name:
            same = [x for x in self.units if x is not u and x.name == u.name and x.star == old
                    and (x.missing or x.loc not in visible)]
            for x in same[:2]:
                u.items = (u.items + x.items)[:3]
                u.items_read = False
                self.alias[x.id] = u.id
                self.units.remove(x)
                gone.append(x)
        u.star = star
        u.star_assumed = False               # the screen's badge says so
        u.look = None                        # its size changed: its look is taken again
        return old, gone

    def buy_unconfirmed(self, name):
        """A buy whose effect we didn't see in time (gold OCR is often late). If a
        unit then appears on the bench, it is this one: named, not '?'."""
        import time as _t
        self.pending_buys.append((name, _t.time()))

    def _claim_pending(self):
        import time as _t
        now = _t.time()
        self.pending_buys = [(n, t) for n, t in self.pending_buys if now - t < 15]
        return self.pending_buys.pop(0)[0] if self.pending_buys else None

    def _claim_recent(self, loc):
        """A unit in the very slot a buy of the last few seconds landed in is that buy
        (the merge we assumed did not happen): named and ours, never an unknown unit."""
        import time as _t
        now = _t.time()
        buys = [b for b in getattr(self, 'recent_buys', []) if now - b[2] < 6]
        hit = next((b for b in buys if ('bench', b[1]) == loc), None)
        if hit:
            buys.remove(hit)
            self.recent_buys = buys
            return hit[0]
        return None

    def sold(self, unit):
        if unit in self.units:
            self.units.remove(unit)
            self.vacate(unit.loc)

    def vacate(self, loc):
        """A bench slot our own action just emptied (its unit fades out for a moment): {slot: time}."""
        import time as _t
        if loc and loc[0] == 'bench':
            self.vacated = {**getattr(self, 'vacated', {}), loc[1]: _t.time()}

    def moved(self, unit, dst):
        """Dragging onto an occupied spot swaps the two units."""
        other = self.at(dst)
        if other is not None and other is not unit:
            other.loc = unit.loc
        else:
            self.vacate(unit.loc)
        unit.loc = dst

    # -- reconcile with what the screen shows (planning phase only) ------------
    def reconcile(self, bench_seen, board_seen, board_count=None, bench_kinds=None):
        """Correct our records from the screen (planning only).

        Bench: health bars sit at a fixed height, so per-slot occupancy is reliable. A tracked
        unit counts as gone from its slot only when the slot reads EMPTY three times (not when it
        is covered or holds something unrecognised: bench_kinds from bench.classify). A unit gone
        is never deleted (run27: the 4 s rule deleted Diana and Kog'Maw that were still there):
        a unit showing up in an unexplained slot takes over its identity UNVERIFIED (the player
        reads it once), else it is kept as MISSING and reported. Only our sell, a merge or a
        consumed unit removes one.
        Board: bars map to hexes only roughly (tall models, units walking back
        after combat), so positions come from our own moves. Only the COUNT is
        checked - the game's 'n/level' counter when readable, else the bar count
        - and units are added/removed only after the mismatch persists."""
        # -- bench ----------------------------------------------------------------
        import time as _t
        now = _t.time()
        seen = {('bench', i) for i, b in enumerate(bench_seen) if b}
        bench_units = self.bench()
        empty_read = bool(bench_kinds) and all(bench_kinds[u.loc[1]] == 'empty' for u in bench_units)
        if len(bench_units) >= 3 and not seen and not empty_read:
            seen = None             # every bar gone at once: covered (panel, choice screen,
                                    # transition), not evidence (run26 4-5: "[0..5] != []") - unless
                                    # every tracked slot reads as the empty tile (all units taken at once)
        self.left_bench = []        # named units gone from the bench this read
        self.lost = []              # units newly marked missing (reported to the model)
        self.taken = []             # units the game took off the board (removed; reported to the model)
        if seen is not None:
            kinds = list(bench_kinds) if bench_kinds else [None] * 9
            tracked = {u.loc for u in bench_units}
            orphans = sorted(loc for loc in seen if loc not in tracked)
            pending = False
            for u in bench_units:
                if u.loc in seen:
                    self._missing.pop(u.id, None)
                    self._gone_since.pop(u.id, None)
                    continue
                if kinds[u.loc[1]] in ('unknown', 'obscured'):
                    pending = True      # covered or not readable: no evidence either way
                    continue
                n = self._missing.get(u.id, 0) + 1
                self._missing[u.id] = n
                self._gone_since.setdefault(u.id, now)
                if n < 3:
                    pending = True      # missed reads are common (animations, tooltips, combat start)
                    continue
                self._missing.pop(u.id, None)
                self._gone_since.pop(u.id, None)
                if orphans:
                    # the same unit, moved (a slot we mis-tracked): its identity is now an inference
                    old = u.loc
                    u.loc = orphans.pop(0)
                    u.unverified = f'tracked at bench slot {old[1]}, found at slot {u.loc[1]}: to be read'
                elif not u.name:
                    self.units.remove(u)            # an unknown unit no longer there: nothing to keep
                else:
                    # a named unit gone from the bench: the game may have fielded it (combat-start
                    # auto-fill, see the board part); otherwise it is missing - kept and reported
                    self.left_bench.append(u)
                    u.missing = f'not seen at bench slot {u.loc[1]} since {self.stage or "?"}'
                    self.lost.append(u)
            if not pending:             # don't invent units while a tracked one is unaccounted for
                for loc in orphans:
                    name = self._claim_recent(loc) or self._claim_pending()
                    if name:
                        self.units.append(Owned(name, 1, loc, ours=True))
                        self._merge(name, 1)
                        continue
                    back = next((m for m in self.missing() if m.loc[0] == 'bench'), None)
                    if back is not None:
                        # a missing unit found again (identity inferred: read it)
                        back.missing, back.loc = None, loc
                        back.unverified = f'missing unit found again at bench slot {loc[1]}: to be read'
                        continue
                    self.units.append(Owned(None, 1, loc))

        # -- board ----------------------------------------------------------------
        import time as _t
        bars = len(board_seen)
        if board_count is None or abs(board_count - bars) > 1:
            return                  # only the game's 'n/level' text changes the board count:
                                    # bars alone pick up the tactician and loot orbs
        observed = board_count
        self.board_seen = observed
        board = self.board()
        # a missing unit comes back only by a read that names it: a bar at its old hex can be any unit
        # (run 010416 2-1: a missing Rek'Sai came back where Varus stood)
        # the game's count is of team slots: no BFF or board object, 2 for an Elder Dragon
        diff = observed - sum(u.slots for u in board)
        now = _t.time()
        if not diff or (self._board_diff and self._board_diff[0] != diff):
            self._board_diff = [diff, now] if diff else []
            return
        if not self._board_diff:
            self._board_diff = [diff, now]
            return
        if now - self._board_diff[1] < 1.5:
            return                  # a mismatch must last 1.5s: quick reads aren't independent
        if diff < 0:
            # drop ONE unit we think is fielded but has no bar nearby (else the weakest-evidence one)
            no_bar = [u for u in board if not any(_hex_near(u.loc[1:], h) for h in board_seen)]
            unnamed = [u for u in board if not u.name]
            victim = next((u for u in no_bar if not u.name), None) or (unnamed or [None])[0]
            if victim is None and no_bar and len(no_bar) == -diff:
                # the game's count is lower by exactly the named units that have no bar: the game took them off
                # the board without our action (an augment, a sacrifice) - removed, like a sell (the user: units
                # leave by our sell, a merge, or a game effect such as an augment), and reported
                for u in no_bar:
                    self.units.remove(u)
                    self.taken.append(u)
                self._board_diff = []
                return
            if victim is None:
                return              # every fielded unit is named: an event beats a count
            self.units.remove(victim)
        else:
            taken = {u.loc[1:] for u in board}
            free = [h for h in board_seen if h not in taken and not any(_hex_near(t, h) for t in taken)]
            free += [h for h in board_seen if h not in taken and h not in free]
            if free:
                # TFT fills empty board slots from the bench at combat start: a unit that left the
                # bench while the board count rose was MOVED, not lost (run26: Alistar tracked on
                # the bench while the game had fielded him - "Brawler screen 1 tracked 0")
                moved = next((u for u in self.left_bench if u in self.units and not u.on_board), None) \
                    or next((u for u in self.missing() if u.loc[0] == 'bench'), None)
                if moved is not None:
                    moved.loc = ('board', *free[0])
                    moved.missing = None
                    moved.unverified = 'fielded by the game at combat start (inferred from the board count)'
                    moved.auto_placed = True
                    if moved in self.lost:
                        self.lost.remove(moved)
                    self._missing.pop(moved.id, None)
                    self._gone_since.pop(moved.id, None)
                else:
                    self.units.append(Owned(None, 1, ('board', *free[0])))
        self._board_diff = []

    def __repr__(self):
        return f'board={self.board()} bench={self.bench()}'


def board_traits(units):
    """Trait counts the game's panel shows for these fielded units: each champion once (copies don't add), with
    the traits it evolved into, an emblem's trait (if it lacks it), and the extra count the game data gives some
    champions (Elder Dragon: +2 Riftbeast)."""
    from collections import Counter
    from tft_player.trait_text import trait_bonus
    per = {}
    for u in units:
        if not u.name:
            continue
        traits = set(u.traits)
        traits |= {i[:-len(' Emblem')] for i in u.items if isinstance(i, str) and i.endswith(' Emblem')}
        per.setdefault(u.name, set()).update(traits)
    out = Counter()
    for name, traits in per.items():
        out.update(traits)
        out.update(trait_bonus(name))
    return out
