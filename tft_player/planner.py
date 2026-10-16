"""The model plans every round; local code carries the plan out and checks it.

Planning phase starts -> one model call with the exact state -> a RoundPlan
(wisps, sells, buys, levels, roll-down, lineup with hexes, items) ->
the executor turns it into one action per tick, re-checking the roster each
time, so a failed or changed step is simply retried or skipped.

Combat starts -> the controller reports what it did with the plan and shows
the board we fight with. The Claude session is one conversation per stage (coach._stage_turnover).

Prompts carry the decision, what changed, the state and the reply format -
no strategy advice (the model has reference files for game/MetaTFT data).
"""
from __future__ import annotations

import json
import re
from collections import Counter

from tft_player.brain import Action
from tft_player.catalog import CATALOG
from tft_player.items import COMPONENTS, RECIPES, consumable_note, is_consumable, is_reforger, is_remover, item_desc
from tft_player.meta import META

PLAN_REPLY = 'Reply with {"execution_plan": ...} first, then {"strategy": ...} (format in your instructions).'

REVIEW_REPLY = ('Reply with ONLY this JSON: {"strategy": {...same fields as in planning...}, '
                '"problems": ["anything wrong you see on the board or in the state"]}')


def mechanics_lines(roster):
    """Board places with a trait effect, every round while the trait is active (the user, 2026-10-11: the model
    never used the Blackthorn hex or picked the BFF rider - who stands where must be in the state)."""
    from tft_player.state import BFF, board_traits
    fielded = roster.fielded()
    count = lambda trait: board_traits(fielded)[trait]
    out = []
    if count('Blackthorn') >= 2:
        from tft_player.trait_text import level_effect
        occ = roster.at(('board', 1, 3))
        gain = level_effect('Blackthorn', count('Blackthorn'))
        out.append('Blackthorn hex [1, 3]: ' + (
            f'{occ.name or "?"}{"*" * occ.star} (id {occ.id}) stands there - sacrificed at combat start'
            if occ is not None and not occ.slotless else
            'EMPTY - nothing is sacrificed at combat start, so the team gets NO Blackthorn bonus this round'
            + (f" (Blackthorn {count('Blackthorn')}: {gain} - it comes only from a sacrifice)" if gain else '')))
    bff = next((u for u in roster.units if u.name == BFF and not u.missing), None)
    snack = " The BFF's ability (Snacktime!): it eats enemies it damages under 16% max Health."
    if bff is not None:
        r, c = bff.loc[1:]
        out.append(f'BFF: at hex [{r}, {c}], no rider yet (not a champion, no team slot). A link from the BFF to one '
                   f'of your fielded Sprykin makes that one its rider (link: {{"from": [{r}, {c}], "to": <unit id>}}); '
                   'with no link, the game picks the rider when combat starts. From its first rider on, the BFF has '
                   'no hex of its own.' + snack)
    elif count('Sprykin') >= 3:
        rider = next((u for u in fielded if getattr(u, 'riding', None)), None)
        out.append('BFF: ' + (f'ridden by {rider.name or "?"}{"*" * rider.star} (id {rider.id}) at hex '
                              f'[{rider.loc[1]}, {rider.loc[2]}]' if rider else
                              'its rider is not known (one of your fielded Sprykin, shown on the BFF)')
                   + '. To change the rider: a link from the rider to another fielded Sprykin (link: {"from": '
                     '<rider id or [row, col]>, "to": <unit id>}) - the BFF goes to that champion; neither moves.'
                   + snack)
    from tft_player.trait_text import pair_hosts
    for host in [u for u in fielded if u.name in pair_hosts()]:
        ally = roster.resolve(getattr(host, 'paired', None)) if getattr(host, 'paired', None) else None
        out.append(f'{host.name} (id {host.id}): ' + (
            f'paired with {ally.name or "?"}{"*" * ally.star} (id {ally.id}) - the pair holds through moves and '
            f'rounds' if ally is not None and not ally.missing else
            f'not paired. To pair an ally: link {{"from": <ally id>, "to": {host.id}}} (the ally is dragged onto '
            f'{host.name}; a line then joins them)'))
    marks = [(i, roster.resolve(uid), st) for i, uid, st in getattr(roster, 'marks', [])]
    marks = [(i, u, st) for i, u, st in marks if u is not None and not u.missing]
    if marks:
        out.append('Alpha Mark (stays on the item bench) last used on: ' + '; '.join(
            f'{u.name or "?"}{"*" * u.star} (id {u.id}, stage {st})' for _, u, st in marks[:2])
            + ' - without Omega Riftbeast only the latest is your Alpha')
    return out


def _unit_line(u):
    if getattr(u, 'slotless', False):
        from tft_player.state import BFF, board_objects
        where = f'hex ({u.loc[1]},{u.loc[2]})' if u.loc[0] == 'board' else f'bench slot {u.loc[1]}'
        from tft_player.state import PLACEABLES
        what = ("the Sprykin trait's Big Furry Friend: not a champion, uses no team slot, gone during combat"
                if u.name == BFF else
                f'not a champion; recorded as using no team slot (not confirmed): {PLACEABLES[u.name]}'
                if u.name in PLACEABLES else f'not a champion, uses no team slot: {board_objects().get(u.name, "")}')
        return f"  [id {u.id}] {u.name} ({what}) at {where}"
    traits = '/'.join(u.traits) if u.info else '?'
    if getattr(u, 'evolved', None):
        traits += f" (evolved: +{', +'.join(u.evolved)})"
    if u.info and u.slots > 1:
        traits += f', uses {u.slots} team slots'
    held = [('an unidentified item' if i == 'unknown' else i) for i in u.items]
    if getattr(u, 'items_read', False):
        items = f" items [{', '.join(held)}]" if held else ' no items'
    else:
        items = f" items not read yet{f' (last known: {held})' if held else ''}"
    where = f'hex ({u.loc[1]},{u.loc[2]})' if u.loc[0] == 'board' else f'bench slot {u.loc[1]}'
    if getattr(u, 'riding', None) and u.loc[0] == 'board':
        where += ', riding the BFF'
    kind = ('melee' if u.info.melee else 'ranged') if u.info else '?'
    # a unit only under a silent look check (re-read in idle time) is not flagged to the model: only
    # a read that finds another champion is reported (run 033507: 38 false alarms led to ~15 board checks)
    doubt = getattr(u, 'unverified', None) and not str(u.unverified).startswith('looks different')
    flag = (f' [MISSING: {u.missing}]' if getattr(u, 'missing', None) else
            f' [UNVERIFIED: {u.unverified}]' if doubt else '')
    return (f"  [id {u.id}] {u.name or '? (unidentified)'} {'*' * u.star} "
            f"({u.info.cost if u.info else '?'}g {kind}, {traits}) at {where}{items}{flag}")


def active_traits(units):
    from tft_player.state import board_traits
    counts = board_traits(units)          # the BFF, a Training Dummy: no traits; Elder Dragon: +2 Riftbeast
    out = []
    for t, c in counts.most_common():
        th = CATALOG.traits.get(t) or ()
        nxt = next((x for x in th if x > c), None)
        out.append(f"{t} {c}{'/' + str(nxt) if nxt else ''}{' (active)' if th and c >= th[0] else ''}")
    return ', '.join(out) or 'none'


COMBAT_NOTE = 'COMBAT is running (board locked).'


def _fixed_comp():
    from tft_player.coach import forced_comp
    return forced_comp()


_TARGET_HEXES = {}


def target_hexes():
    """The fixed comp's target final board as 'Name (row,col)' from its MetaTFT page (the briefing's section 1),
    or '' - in every state, so the positions are not only in the first message (run 185805: Aphelios and
    the front line ended far from them)."""
    comp = _fixed_comp()
    if comp is None:
        return ''
    if comp.id not in _TARGET_HEXES:
        try:
            from tft_player.metatft_shots import _paths
            text = _paths(comp.name)[2].read_text(encoding='utf-8')
            sec = text.split('1. Target final board, positions', 1)[1].split('\n\n', 1)[0]
            _TARGET_HEXES[comp.id] = ', '.join(f'{n} ({r},{c})' for n, r, c in re.findall(r'^- (.+): row (\d), column (\d)$', sec, re.M))
        except Exception:
            _TARGET_HEXES[comp.id] = ''
    return _TARGET_HEXES[comp.id]


def sell_value(u):
    """Gold a unit sells for: a 1-star its cost; a 2-/3-star all its copies' gold minus 1, except
    1-cost units (full amount)."""
    cost = u.info.cost if u.info else 0
    return cost * 3 ** (u.star - 1) - (1 if u.star > 1 and cost > 1 else 0)


def interest_needs(g, roster, pvp):
    """The interest arithmetic both the model and the overlay use (the user, 2026-10-16): gold now, what the bench sells
    for (unit by unit), and per fight result the gold held when interest is paid (the next round start) and how much
    of the bench must be sold to reach the next step. -> dict, or None without gold."""
    if g is None:
        return None
    named = [u for u in roster.bench() if u.name and u.info]
    worth = sum(sell_value(u) for u in named)
    results = (('win', g + 1), ('loss', g)) if pvp else (('', g),)
    out = {'gold': g, 'worth': worth, 'units': named, 'results': []}
    for label, held in results:
        nxt = min(50, (held // 10 + 1) * 10)
        need = 0 if held >= 50 else nxt - held
        out['results'].append({'label': label, 'held': held, 'next': nxt, 'need': need,
                               'reachable': held >= 50 or need <= worth})
    return out


def econ_line(scr, roster):
    """Interest for the model, plainly (the user, 2026-10-16): '24g (interest 2). Bench sells for 8g (Elise 2g, Rakan
    3g, Varus 3g). Next interest at 30: after a win you hold 25g - sell 5g of bench units to reach it; after a loss 24g
    - sell 6g.' Interest is paid at the next round start on the gold held then; an augment can change the rule."""
    try:
        stage, rnd = (int(x) for x in str(scr.stage).split('-'))
    except (TypeError, ValueError):
        stage, rnd = 0, 0
    pvp = stage >= 2 and rnd not in (4, 7)
    n = interest_needs(scr.gold, roster, pvp)
    if n is None:
        return ''
    g = n['gold']
    if g >= 50:
        return f'Economy: {g}g, interest 5 (the maximum).'
    units = ', '.join(f'[id {u.id}] {u.name}{"*" * u.star} {sell_value(u)}g' for u in n['units'])
    bench = f'Bench sells for {n["worth"]}g ({units}).' if n['units'] else 'Nothing on the bench to sell.'
    parts = []
    for r in n['results']:
        who = (f'after a win you hold {r["held"]}g' if r['label'] == 'win' else
               f'after a loss {r["held"]}g' if r['label'] == 'loss' else f'you hold {r["held"]}g (no win gold this round)')
        if r['need'] == 0:
            parts.append(f'{who} - already there')
        elif r['reachable']:
            parts.append(f'{who} - sell {r["need"]}g of bench units to reach it')
        else:
            parts.append(f'{who} - {r["need"]}g short, more than the bench sells for')
    nxt = n['results'][0]['next']
    return (f'Economy: {g}g (interest {g // 10}). {bench} Next interest at {nxt}: ' + '; '.join(parts) + '.'
            + (' Sells for one result only: execution_plan.after_fight {"win": {"sell": [ids]}, "lose": {...}}.'
               if pvp else ''))


def is_pvp(stage):
    """A player fight: stage 2 on, except the carousel (x-4) and monster (x-7) rounds."""
    try:
        s, r = (int(x) for x in str(stage).split('-'))
    except (TypeError, ValueError):
        return False
    return s >= 2 and r not in (4, 7)


def interest_check(scr, roster):
    """The combat-start call's interest decision, in the same plain words as the planning state (econ_line): what to
    sell after a win and after a loss to reach the next step - then where the answer goes. '' without gold."""
    line = econ_line(scr, roster)
    if not line:
        return ''
    try:
        stage, rnd = (int(x) for x in str(scr.stage).split('-'))
    except (TypeError, ValueError):
        stage, rnd = 0, 0
    pvp = stage >= 2 and rnd not in (4, 7)
    return ('INTEREST CHECK: ' + line.removeprefix('Economy: ')
            + ('' if pvp else ' This round is not a player fight: bench sells and buys in execution_plan happen at once.'))


def state_text(scr, roster, plan, items, hp_log, specials, timer, extra=(), phase='planning', wisps_decided=None):
    """The exact picture for the model."""
    specials = specials or {}
    lvl = scr.level
    board, bench = roster.board(), roster.bench()
    shop = []
    for i, name in enumerate(scr.shop):
        if name and CATALOG.unit(name):
            info = CATALOG.unit(name)
            have = len([u for u in roster.units if u.name == name])
            shop.append(f"  slot {i}: {name} ({info.cost}g, {'/'.join(info.traits)})"
                        f"{f' - we own {have}' if have else ''}")
        elif i in specials:
            desc, cost = specials[i]
            shop.append(f"  slot {i}: WISP '{scr.shop_raw[i]}' ({cost if cost is not None else '?'}g): {desc}")
        elif scr.shop_raw[i]:
            shop.append(f'  slot {i}: not read - see the screenshot')
        else:
            shop.append(f'  slot {i}: empty')
    decided = wisps_decided or {}
    wisps = [f"slot {i} '{scr.shop_raw[i]}'" + (f" (already decided: {decided[i]})" if i in decided else '')
             for i in sorted(specials)]
    wisp_line = ('Wisp in shop: no' if not wisps else
                 f"Wisp in shop: {', '.join(wisps)}" + (
                     " - decide it in execution_plan.wisps." if any(i not in decided for i in specials) else '.'))
    if timer == 0:                    # combat review: no shop decision
        wisp_line = ''
    def note(it):
        # what hovering the item in game shows (a player always can): consumables, emblems,
        # completed items; components are plain (their recipes are listed below)
        c = consumable_note(it)
        n = c if c is not None else ('' if it in COMPONENTS else item_desc(it))
        return f' [{n}]' if n else ''
    bench_line = ', '.join(f'slot {k}: ' + ('an unidentified item (being read; see the screenshot)' if it == 'unknown'
                                            else f'{it}{note(it)}')
                           for k, it in enumerate(items or []) if it) or 'empty'
    comp = META.comps.get(plan.comp) if plan.comp else None
    target = f'{comp.name} ({comp.id})' if comp else 'none'
    hist = ', '.join(f'{s}: {a}->{b}' for s, a, b in hp_log[-4:]) or 'none yet'
    head = (f"Stage {scr.stage}, planning phase, ~{timer}s left.\n" if phase == 'planning' else
            f"Stage {scr.stage}. {COMBAT_NOTE}\n")
    return (head +
            f"HP {scr.hp if scr.hp is not None else 'not read yet (see the screenshot)'}. "
            f"HP change per round (lost a fight when it drops): {hist}\n"
            + (f"Gold {scr.gold}, " if scr.gold is not None else
               "Gold not read yet (see the screenshot), ")
            + f"level {lvl if lvl is not None else 'not read yet'}, xp {scr.xp}.\n"
            # (the user, 2026-10-11: they do not expire - unused ones stay for later rounds; the model spent them at
            # level 4 where its 3-cost units were rare)
            + (f"Free rerolls: {scr.free_rolls}. They do not expire: unused ones stay for later rounds. Every roll "
               f"(yours, a roll order, a roll-down) uses a free one first; the Reroll button costs 0 gold until they "
               f"are used up.\n"
               if getattr(scr, 'free_rolls', None) else '')
            + (econ_line(scr, roster) + '\n' if scr.gold is not None else '')
            + f"Board ({roster.slots_used()}/{getattr(scr, 'team_size', None) or lvl} team slots used):\n" + ('\n'.join(map(_unit_line, board)) or '  empty')
            + ''.join('\n' + line for line in mechanics_lines(roster))
            + (f"\nTarget final board hexes (MetaTFT, row,col): {target_hexes()}" if target_hexes() else '') +
            f"\nBench ({len(bench)}/9 units"
            + (f", {len(roster.blocked)} slot(s) taken by other things" if getattr(roster, 'blocked', None) else '')
            + "):\n" + ('\n'.join(map(_unit_line, bench)) or '  no units')
            + ''.join(f"\n  bench slot {k}: {what} - occupies the slot" for k, what in sorted(getattr(roster, 'blocked', {}).items()))
            + (("\nNot located this round (dropped from the records if still not found next round):\n" +'\n'.join(map(_unit_line, roster.missing())))
               if hasattr(roster, 'missing') and roster.missing() else '') +
            f"\nActive traits on board: {active_traits(board)}\n"
            + ""
            f"Shop:\n" + '\n'.join(shop) +
            (f"\n{wisp_line}" if wisp_line else '') +
            f"\nItem bench: {bench_line}"
            + ''.join('\n' + line for line in ([] if _fixed_comp() else [f"Target comp: {target}"]) + list(extra)
                      if line))


def carry_text(leftover, protected):
    """What the model must know it left unfinished."""
    out = ''
    if leftover:
        out += (f'\nNOT DONE LAST ROUND (the round ended first; the executor is continuing the '
                f'board/items part now): {leftover}')
    if protected:
        out += f'\nProtected units (your last plan wanted them; nothing sells them unless you do): {sorted(protected)}'
    return out


def plan_prompt(state, first=True, carry=''):
    """A planning decision: what is decided, what changed, the state, and a pointer to the reply format (in full
    in the system prompt, which every stage's conversation has)."""
    changes = f'Since your last decision: {carry.strip()}\n' if carry.strip() else ''
    return f'DECISION: planning phase.\n{changes}\nSTATE\n{state}\n\n{PLAN_REPLY}'


def update_prompt(state, progress, remaining):
    """Mid-round: part of the plan is done or failed and time remains."""
    return (f'DECISION: same planning phase, update.\nSince your last decision: {progress}. '
            f'Not done: {remaining or "nothing"}.\n\nSTATE\n{state}\n\n'
            + PLAN_REPLY + ' Include whatever is still to do.')


def review_prompt(state, plan, done, failed):
    """Combat starts: what the controller did with the plan, and the board we fight with."""
    return (f'Combat is starting (the attached image is the screen now).\n'
            f'Controller report on your plan: executed: {done or "nothing"}; failed or not reached: '
            f'{failed or "nothing"}.\n\nSTATE\n{state}\n\n{REVIEW_REPLY}')


def once(steps):
    """Report lines, each once: a step retried with the same result is counted, not repeated."""
    n = {}
    for s in steps:
        n[s] = n.get(s, 0) + 1
    return [s + (f' (x{k})' if k > 1 else '') for s, k in n.items()]


def buy_targets(buy):
    """A roll's "buy": a list of champions (each to a 2-star's copies, or its auto_buy star) or
    {"Champion": star} -> {name: star} for the ones given a star."""
    if isinstance(buy, dict):
        return {n: int(s) for n, s in buy.items() if CATALOG.unit(n) and isinstance(s, (int, float)) and 1 <= s <= 3}
    return {}


def parse(text):
    from tft_player.coach import json_objects
    merged = {}
    for obj in json_objects(text):
        if isinstance(obj, dict):
            merged.update(obj)             # {"execution_plan"} then {"strategy"}
    return normalize(merged) if merged else None


def normalize(ans):
    """{"execution_plan", "strategy"} (the format in the system prompt) -> the
    executor's keys. Older flat replies pass through unchanged."""
    if not isinstance(ans, dict) or 'execution_plan' not in ans:
        return ans
    ep = ans.get('execution_plan') or {}
    st = ans.get('strategy') if isinstance(ans.get('strategy'), dict) else {}
    out = {'strategy': st, 'note': st.get('now'), 'comp': st.get('target_comp')}
    for k in ('sell', 'open_container', 'wisps', 'buy', 'keep', 'fast_rr', 'next_review', 'identify',
              'auto_buy', 'auto_sell', 'after_fight', 'after_fight_sell', 'item_final'):
        if ep.get(k) is not None:
            out[k] = ep[k]
    if ep.get('xp'):
        out['level_ups'] = ep['xp']
    r = ep.get('roll')
    if isinstance(r, dict) and r.get('times') is not None:
        # reroll n times: a Wisp in a new shop stops it for a question, then it goes on
        out['roll_n'] = {'times': r.get('times'), 'buy': r.get('buy') or [],
                         'sell_others': bool(r.get('sell_others', False))}
    elif isinstance(r, dict) and not isinstance(ep.get('fast_rr'), dict):
        # the one roll key: reroll down to to_gold buying the listed champions, stop early at stop_when
        out['fast_rr'] = {'buy': r.get('buy') or [], 'gold_floor': r.get('to_gold'),
                          'stop_when': r.get('stop_when') or [], 'sell_to_make_room': bool(r.get('sell_to_make_room')),
                          'sell_others': r.get('sell_others', True) is not False,
                          'keep': ep.get('keep') or []}
    if ep.get('gold_floor') is not None:
        out['keep_gold'] = ep['gold_floor']
    if ep.get('board') is not None:
        out['board'] = [({'id': e['unit']} if 'unit' in e else {'name': e.get('name')}) | {'hex': e.get('hex')}
                        for e in ep['board'] if isinstance(e, dict)]
    if ep.get('bench') is not None:
        out['bench'] = [{'id': e.get('unit'), 'slot': e.get('slot')} for e in ep['bench'] if isinstance(e, dict)]
    if ep.get('strip') is not None:
        out['strip'] = [{'id': e.get('unit'), 'tool': e.get('tool')} for e in ep['strip'] if isinstance(e, dict)]
    if ep.get('link') is not None:
        out['link'] = [{'from': e.get('from'), 'to': e.get('to')} for e in ep['link'] if isinstance(e, dict)]
    if ep.get('items') is not None:
        # item ACTIONS: give / use (an item or consumable from the item bench), reforge (a Reforger on
        # a unit), transfer (one item from one unit to another: a Remover on the source, then gives -
        # expanded by validate(), which checks every item on the source has a destination)
        items, strip, transfer = [], list(out.get('strip') or []), []
        for e in ep['items']:
            if not isinstance(e, dict):
                continue
            if 'transfer' in e:
                transfer.append({'item': e.get('transfer'), 'from': e.get('from'), 'to': e.get('to')})
            elif 'reforge' in e:
                strip.append({'id': e.get('reforge'), 'tool': 'reforger'})
            elif 'give' in e or 'use' in e:
                items.append({'component': e.get('give') or e.get('use'), 'id': e.get('to', e.get('on'))}
                             | ({'final': e['final'], 'final_star': e.get('final_star')} if e.get('final') else {}))
            elif 'item' in e:                                  # the older {"item", "unit"} form
                items.append({'component': e.get('item'), 'id': e.get('unit')})
        out['items'] = [dict(e, k=k) for k, e in enumerate(items)]
        if strip:
            out['strip'] = strip
        if transfer:
            out['transfer'] = transfer
    if isinstance(ep.get('item_targets'), list):
        out['item_targets'] = [str(i) for i in ep['item_targets'] if i]
    if isinstance(ans.get('next_round'), dict):
        out['next_round'] = ans['next_round']          # raw; the player normalizes it when it keeps it
    return out



def validate(plan, roster, item_bench=None, names=None, level=None):
    """Check every entry of a normalized plan before it runs; drop the bad ones and say exactly
    why (run26 3-5: names in id fields failed late as "item None -> unit None"). Nothing is
    auto-corrected: writing a valid plan is part of the test. Returns the error texts."""
    from tft_player.screen import ScreenReader
    if not isinstance(plan, dict):
        return []
    # ids the model saw when it wrote the plan count too: a unit merged or re-synced since then is
    # found again by name/star when the step runs (RoundPlan.unit) - run27 5-3 rejected two sells
    ids = {u.id for u in roster.units} | set(names or ()) | set(getattr(roster, 'alias', {}))
    errors = []
    is_id = lambda v: isinstance(v, int) and not isinstance(v, bool) and v in ids

    def keep_valid(key, ok, describe):
        vals = plan.get(key)
        if vals is None:
            return
        if not isinstance(vals, list):
            errors.append(f'{key}: must be a list')
            plan.pop(key, None)
            return
        good = []
        for i, v in enumerate(vals):
            why = ok(v)
            if why:
                errors.append(f'{key}[{i}] {describe(v)}: {why}')
            else:
                good.append(v)
        plan[key] = good

    unit_why = lambda v: (None if is_id(v) else
                          f"'{v}' is a name - this field needs a unit id" if isinstance(v, str) else
                          f'no unit with id {v}')
    def sell_why(v):
        from tft_player.state import PLACEABLES
        u = roster.resolve(v) if is_id(v) else None
        if u is not None and u.name in PLACEABLES:
            return f'{u.name} is an Elderwood plant: the game does not let it be sold, only moved'
        return unit_why(v)
    keep_valid('sell', sell_why, repr)
    keep_valid('keep', unit_why, repr)

    def bench_why(e):
        if not isinstance(e.get('slot'), int) or isinstance(e.get('slot'), bool) or not 0 <= e['slot'] <= 8:
            return 'slot must be a bench slot 0-8'
        if any(isinstance(b, dict) and b.get('id') == e.get('id') for b in plan.get('board') or []):
            return 'the unit is also in "board"'
        return unit_why(e.get('id'))
    keep_valid('bench', bench_why, lambda e: f"unit {e.get('id')} -> slot {e.get('slot')}")

    def board_why(e):
        h = e.get('hex') if isinstance(e, dict) else None
        if not (isinstance(h, (list, tuple)) and len(h) == 2 and 0 <= h[0] <= 3 and 0 <= h[1] <= 6):
            return 'hex must be [row 0-3, col 0-6]'
        if 'id' in e:
            return unit_why(e['id'])
        return None if CATALOG.unit(e.get('name') or '') else f"no champion named {e.get('name')!r}"
    keep_valid('board', board_why, lambda e: str({k: v for k, v in e.items() if k != 'k'}))

    known_items = set(ScreenReader.item_names())
    on_bench = {str(i).split(' (')[0] for i in item_bench or () if i}

    def item_why(e):
        it = e.get('component') if isinstance(e, dict) else None
        if not isinstance(it, str) or not it:
            return 'no item named'
        if it not in known_items and it.split(' (')[0] not in known_items:
            return f'{it!r} is not an item name'
        # an "unknown" slot may be exactly the item named (run27 2-6: a carousel Recurve Bow not yet
        # identified got a valid Rageblade rejected): only reject what the bench surely lacks. A
        # completed item is given only if it is there - never built from components on its own
        tgt = next((u for u in roster.units if u.id == e.get('id')), None)
        if tgt is not None and any(str(i).split(' (')[0] == it.split(' (')[0] for i in tgt.items):
            return None                          # the unit already holds it: counted as done when it runs
        if 'from' not in e and item_bench is not None and 'unknown' not in on_bench and it.split(' (')[0] not in on_bench:
            return (f'{it!r} is not on the item bench (it holds: {", ".join(sorted(on_bench)) or "nothing"})'
                    + (f'; to build it, give its components ({" + ".join(RECIPES[it])}) as separate entries'
                       if it in RECIPES else ''))
        why = unit_why(e.get('id'))
        if why is None and 'from' not in e and not is_consumable(it) and not e.get('final'):
            # every item has two holders: the unit it goes on now ("to") and the champion it is meant for (the user,
            # 2026-10-14); the controller moves it to its final holder when that one is ours
            why = 'a give needs "final": the champion this item is meant for ("to" is the unit holding it now)'
        return why
    keep_valid('items', item_why, lambda e: f"{e.get('component')} -> unit {e.get('id')}")
    _expand_transfers(plan, roster, item_bench, unit_why, errors)

    def buy_why(v):
        return None if (isinstance(v, int) and 0 <= v <= 4) or (isinstance(v, str) and CATALOG.unit(v)) else             'a shop slot 0-4 (or a champion name) is needed'
    keep_valid('buy', buy_why, repr)
    lv = plan.get('level_ups')
    if lv is not None and not (isinstance(lv, int) and 0 <= lv <= 40):
        errors.append(f'xp: {lv!r} must be a whole number of XP buys')
        plan.pop('level_ups', None)
    gap = new_slot_gap(plan, roster, level)
    if gap:
        errors.append(gap)               # the xp stays; the model is asked again to name the lineup
    return errors


XP_TO_NEXT = {2: 2, 3: 6, 4: 10, 5: 20, 6: 36, 7: 60, 8: 68, 9: 68}     # xp from level L to L+1 (game-mechanics.md)


def new_slot_gap(plan, roster, level):
    """A plan that buys XP up to a new level must say who fills the new board slot(s) when we own the units
    (run 152104 5-7: levelled to 9 in the last second with no ninth unit planned). The board after the plan:
    the units fielded now, less the ones it benches or sells, plus the ones its "board" places. -> error or None."""
    n = plan.get('level_ups')
    lvl, xp = level if isinstance(level, tuple) else (level, None)
    if not n or not isinstance(n, int) or not lvl:
        return None
    cur = xp[0] if isinstance(xp, (tuple, list)) and xp and isinstance(xp[0], int) else 0
    new, pts = lvl, cur + 4 * n
    while new in XP_TO_NEXT and pts >= XP_TO_NEXT[new]:
        pts -= XP_TO_NEXT[new]
        new += 1
    if new == lvl:
        return None
    leaving = {e.get('id') for e in plan.get('bench') or [] if isinstance(e, dict)} | set(plan.get('sell') or [])
    fielded = {u.id for u in roster.board() if not u.missing} - leaving
    placed = [e for e in plan.get('board') or [] if isinstance(e, dict)]
    from tft_player.trait_text import unit_slots
    slots = {u.id: u.slots for u in roster.units}         # team slots: an Elder Dragon fills 2
    count = sum(slots.get(i, 1) for i in fielded | {e['id'] for e in placed if e.get('id') is not None}) \
        + sum(unit_slots(e['name']) for e in placed if e.get('id') is None and e.get('name'))
    owned = sum(u.slots for u in roster.units if not u.missing) + len(plan.get('buy') or [])
    if count >= min(new, owned):
        return None
    return (f'xp: the {n} XP buy(s) take us to level {new}, but the board after this plan has {count} units for '
            f'{new} slots: place the units for the new slot(s) in "board" (an owned unit id, or a champion name '
            f'bought in this plan or its roll, or still arriving; keep it from being sold with "keep")')

def _expand_transfers(plan, roster, item_bench, unit_why, errors):
    """The plan's item transfers -> a Remover on each source unit ('strip') and a give per moved item
    ('items', marked with its source). A Remover takes ALL of a unit's items off, so every item on the
    source must have a destination (another unit, the same unit, or "bench"): a source with an item
    left out is rejected whole, with the items named. Popped once expanded (validate may run twice)."""
    tr = plan.pop('transfer', None)
    if not tr:
        return
    base = lambda n: str(n or '').split(' (')[0]
    on_bench = [base(i) for i in item_bench or () if i]
    by_src = {}
    for e in tr:
        by_src.setdefault(e.get('from'), []).append(e)
    items = plan.setdefault('items', [])
    strip = plan.setdefault('strip', [])
    k = max([e.get('k', -1) for e in items] + [-1]) + 1
    for src, es in by_src.items():
        what = f"transfer from unit {src}"
        why = unit_why(src)
        u = next((x for x in roster.units if x.id == src), None)
        if not why:
            why = next((unit_why(e.get('to')) for e in es if e.get('to') != 'bench' and unit_why(e.get('to'))), None)
        if not why and item_bench is not None and 'unknown' not in on_bench and \
                not any(i.startswith(('Item Remover', 'Magnetic Remover')) for i in on_bench):
            why = 'there is no Item Remover on the item bench'
        if not why and u is not None and getattr(u, 'items_read', False):
            held = [base(i) for i in u.items]
            moved = [base(e.get('item')) for e in es]
            missing = [i for i in moved if i not in held]
            rest = list(held)
            for i in moved:
                if i in rest:
                    rest.remove(i)
            if missing:
                why = f"{u.name} (unit {src}) does not hold {', '.join(missing)} (it holds: {', '.join(held) or 'nothing'})"
            elif rest:
                why = (f"the Remover takes ALL items off {u.name} (unit {src}); also say where each of these goes: "
                       f"{', '.join(rest)} (another unit, back to unit {src}, or \"bench\")")
        if why:
            errors.append(f'{what}: {why}')
            continue
        strip.append({'id': src, 'tool': 'remover'})
        for e in es:
            if e.get('to') != 'bench':
                items.append({'component': e.get('item'), 'id': e.get('to'), 'from': src, 'k': k})
                k += 1


class RoundPlan:
    """Executes one round's plan; next_action() is called every tick."""

    def __init__(self, plan: dict, stage, shop=None, carry=False, known=(), names=None, gold=None):
        self.p = plan
        self.stage = stage
        # id -> (name, star) when the plan was written: the roster can give a unit a new id
        # mid-round (re-synced with the screen), and the plan must still find it (run23: Xayah
        # and both Akalis got new ids at 2-2, so "Tear to unit 2" and "field unit 3" found nothing)
        self.names = dict(names or {})
        self.last_gold = gold            # a level-up/roll never waits for a gold reading
        self.idle_said = None
        self.carry = carry               # last round's unfinished board/items, run until the new plan lands
        self.known = set(known)          # unit ids that existed when the plan was made
        # buys by NAME: slot numbers go stale (a misread shop, a roll), names don't
        shop = shop or [None] * 5
        self.to_buy = [shop[i] if isinstance(i, int) and 0 <= i < 5 else i for i in plan.get('buy') or []]
        self.to_buy = [n for n in self.to_buy if isinstance(n, str) and CATALOG.unit(n)]
        self.sold, self.bought, self.itemised = set(), set(), set()
        self.wisps_done = set()
        self.containers_done = set()
        self.levels_done = 0
        self.level_fails = 0
        self.rolls = 0
        self.done, self.failed = [], []
        self.keep = int(plan.get('keep_gold') or 0)
        self.tries, self.given_up, self.refused, self.linked = {}, set(), set(), set()
        self.item_later = set()          # item gives that failed this round: kept for the next round's plan
        self.item_waits = {}
        self.fast_done = False
        self.rolls_left = None           # roll n times: rerolls still to do (a Wisp question pauses it)
        self.rolled = 0                  # rerolls this plan's roll-downs made
        self.rolln_done = False
        self.buy_misses = {}
        self.benched = set()             # bench placements given up (failed)
        self.full_told = set()           # units the full board kept out (said once)

    def unit(self, roster, uid, used=()):
        """The plan's unit `uid`: by id, else the same champion (name, star) the id meant when
        the plan was written (a unit that got a new id), never one already taken."""
        u = next((x for x in roster.units if x.id == uid), None)
        if u is None and isinstance(uid, int):
            u = roster.resolve(uid)              # an id merged away -> the unit it became
        if u is not None and u.missing:
            return None                          # not located: nothing can be done to it
        if u is not None or uid not in self.names:
            return u
        name, star = self.names[uid]
        if not name:
            return None
        cands = [x for x in roster.units if x.name == name and x.id not in used and x.id not in self.names]
        cands = cands or [x for x in roster.units if x.name == name and x.id not in used]
        same = [x for x in cands if x.star == star]
        return (same or cands or [None])[0]

    def pending(self, roster):
        """Steps still to take that have not failed: units not yet placed, items not yet given,
        buys and level-ups not yet done."""
        if any(u.loc != ('board', *h) for u, h in self._lineup(roster)):
            return True
        if any((e.get('component'), e.get('id'), e.get('k')) not in self.itemised for e in self.p.get('items') or []):
            return True
        if self._bench_left(roster):
            return True
        return len(self.bought) < len(self.to_buy) or self.levels_done < int(self.p.get('level_ups') or 0)

    def moves_pending(self, roster):
        """A step that changes the board the fight is played with is still to do: a unit to put on (or move to) its
        hex, or a fielded unit to bench. Units the full board kept out do not count (the model is told)."""
        if any(u.loc != ('board', *h) and u.id not in self.full_told for u, h in self._lineup(roster)):
            return True
        return any(u.on_board for u, _ in self._bench_left(roster))

    def protected_ids(self):
        """Units this plan wants: on the planned board or explicitly kept."""
        ids = {e['id'] for e in self.p.get('board') or [] if isinstance(e.get('id'), int)}
        return ids | {i for i in self.p.get('keep') or [] if isinstance(i, int)}

    def leftover(self, roster, failed=True):
        """What the plan asked for but didn't get done (the round ran out):
        (plan dict for the executor to continue, text for the model)."""
        want = self._lineup(roster)
        missing = [(u, h) for u, h in want if u.loc != ('board', *h)]
        items = [e for e in self.p.get('items') or [] if (e.get('component'), e.get('id'), e.get('k')) not in self.itemised]
        buys = [n for k, n in enumerate(self.to_buy) if k not in self.bought]
        levels = max(0, int(self.p.get('level_ups') or 0) - self.levels_done)
        lines = [f'{u} not yet at {h}' for u, h in missing]
        lines += [f"{e.get('component')} not yet given to unit {e.get('id')}" for e in items]
        lines += [f'{n} not bought' for n in buys]
        lines += [f'{levels} level-up(s) not done'] if levels else []
        # sells of board units are held during combat: they carry over to the planning phase
        sells = [uid for uid in self.p.get('sell') or [] if uid not in self.sold and self.unit(roster, uid)]
        lines += [f'sell {self.unit(roster, uid)} not done yet (board locked in combat)' for uid in sells]
        lines += [f'failed: {f}' for f in once(self.failed)[-5:]] if failed else []      # else in progress()
        benches = self._bench_left(roster)
        lines += [f'{u} not yet at bench slot {k}' for u, k in benches]
        carry = {'board': [{'id': u.id, 'hex': list(h)} for u, h in want], 'sell': sells,
                 'items': items, 'keep': list(self.protected_ids()),
                 'bench': [{'id': u.id, 'slot': k} for u, k in benches]} if missing or items or sells or benches else None
        return carry, '; '.join(lines)

    def queued(self, roster):
        """The steps still to run, in the order the executor takes them (see _next): [(kind, text)] for the overlay."""
        need = self._board_names(roster)
        sells = [self.unit(roster, uid) for uid in self.p.get('sell') or [] if uid not in self.sold]
        sells = [u for u in sells if u is not None]
        buys = [(k, n) for k, n in enumerate(self.to_buy) if k not in self.bought]
        out = []
        for w in self.p.get('wisps') or []:
            if isinstance(w, dict) and w.get('buy') and isinstance(w.get('slot'), int) and w['slot'] not in self.wisps_done:
                out.append(('wisp', f"buy the Wisp in shop slot {w['slot']}"))
        out += [('sell', f'sell {u}') for u in sells if u.on_board]
        out += [('buy', f'buy {n}') for k, n in buys if n in need]
        levels = max(0, int(self.p.get('level_ups') or 0) - self.levels_done)
        if levels:
            out.append(('level', f'level up x{levels}'))
        out += [('move', f'{u} -> bench {k}') for u, k in self._bench_left(roster)]
        out += [('move', f'{u} -> {h}') for u, h in self._lineup(roster) if u.loc != ('board', *h)]
        for e in self.p.get('items') or []:
            if (e.get('component'), e.get('id'), e.get('k')) not in self.itemised:
                u = self.unit(roster, e.get('id'))
                out.append(('item', f"{e.get('component') or e.get('tool') or 'item'} -> {u or 'unit ' + str(e.get('id'))}"))
        out += [('sell', f'sell {u}') for u in sells if not u.on_board]
        out += [('buy', f'buy {n}') for k, n in buys if n not in need]
        if self.roll_pending():
            rn, fr = self.p.get('roll_n'), self.p.get('fast_rr')
            if isinstance(rn, dict) and not self.rolln_done:
                left = self.rolls_left if self.rolls_left is not None else rn.get('times')
                out.append(('roll', f"roll x{left}" + (f" for {', '.join(rn.get('buy') or [])}" if rn.get('buy') else '')))
            else:
                floor = (fr or {}).get('gold_floor') if isinstance(fr, dict) else (self.p.get('roll') or {}).get('down_to')
                buy = (fr or {}).get('buy') if isinstance(fr, dict) else None
                out.append(('roll', f'roll to {floor or 0}g' + (f" for {', '.join(buy)}" if buy else '')))
        return out

    def progress(self):
        """What happened to the plan so far, for the model: steps done (rolls left out - the
        model doesn't need their count, the user) and steps that failed, with their reasons."""
        done = [d for d in self.done if not d.startswith('roll(')]
        return f"done: {'; '.join(once(done)) or 'nothing'}. failed: {'; '.join(once(self.failed)) or 'nothing'}"

    def record(self, act, ok, cause=None):
        if act.kind == 'rolldown' and isinstance(act.arg, dict) and act.arg.get('times') is not None:
            self.rolled += int(act.arg.get('done') or 0)
            self.rolls_left = max(0, (self.rolls_left or 0) - int(act.arg.get('done') or 0))
            if act.arg.get('stopped') != 'wisp' or not self.rolls_left:
                self.rolln_done = True
        if act.kind == 'level' and isinstance(act.arg, int) and getattr(act, 'landed', None) is not None:
            self.levels_done -= act.arg - act.landed           # presses that did not level (gold ran out)
        # (not one that landed elsewhere - its holder records move it - or that no unit shows: it is not on the bench)
        later = (not ok and act.kind == 'item' and getattr(act, 'item_key', None) is not None
                 and not (cause or '').startswith('the game refused it')
                 and ' landed on ' not in (cause or '') and ': missing' not in (cause or ''))
        if later:
            # a give that did not land is no big deal (the user, 2026-10-12): not retried now, carried to the next
            # planning phase with the rest of this plan's unfinished steps (leftover)
            self.itemised.discard(act.item_key)
            self.item_later.add(act.item_key)
        step = f'{act.kind}({act.why})'
        if ok:
            # a retry that worked takes back the step's earlier failure (run 20261016-052413 3-5: the first E did not
            # sell, the retry did, and the model and overlay still read "failed: sell ... gold stayed 48" at 50 gold)
            self.failed = [f for f in self.failed if f != step and not f.startswith(step + ' - ')]
        (self.done if ok else self.failed).append(step + (f' - {cause}' if cause and not ok else '')
                                                  + (' - queued: it is tried again next round' if later else ''))
        if not ok:
            self.tries[repr(act)] = self.tries.get(repr(act), 0) + 1
            if act.kind == 'item' and cause and cause.startswith('the game refused it'):
                # the game said no: the same drop gets the same answer (run 20261010-113319: 3 tries a round)
                self.refused.add((act.arg[1], getattr(act.arg[2], 'id', None)))
            if act.kind == 'sell' and self.tries[repr(act)] < 2:
                # not sold: try once more (a sell that fails twice is reported, not repeated)
                uid = next((k for k in self.p.get('sell') or [] if k in self.sold and
                            self.names.get(k, (None,))[0] == getattr(act.arg, 'name', None)), None)
                if uid is not None:
                    self.sold.discard(uid)
            if act.kind == 'level' and self.level_fails < 3 and getattr(act, 'landed', None) is None:
                # (a press count that was checked is already given back above: run 20261015-003101 3-1 gave
                # one missed press back twice and bought 2 more XP)
                self.level_fails += 1
                self.levels_done -= 1            # not levelled: the step is still to do

    def snapshot(self):
        """What this plan has done so far: taken when the model is asked to update it."""
        return {'levels': self.levels_done, 'bought': set(self.bought), 'sold': set(self.sold),
                'itemised': set(self.itemised), 'wisps': set(self.wisps_done), 'rolled': self.rolled}

    def credit(self, old, snap):
        """An update replaces `old`, which kept running while the model wrote it (the action path never waits):
        the steps `old` took after `snap` - the state the model saw - count as done in this plan. The model restates
        what it saw as still to do, and run 20261015-003101 did it again: 3-1 "xp 2" ran twice (47 -> 15 gold, with
        a retry), 2-5 bought the XP twice and a Hecarim the update had dropped."""
        lv = max(0, old.levels_done - snap['levels'])
        self.levels_done = min(lv, int(self.p.get('level_ups') or 0))
        for k in sorted(old.bought - snap['bought']):
            name = old.to_buy[k] if k < len(old.to_buy) else None
            j = next((j for j, n in enumerate(self.to_buy) if n == name and j not in self.bought), None)
            if j is not None:
                self.bought.add(j)
        self.sold |= old.sold - snap['sold']
        self.wisps_done |= old.wisps_done - snap['wisps']
        for comp, uid, _ in sorted(old.itemised - snap['itemised'], key=repr):
            e = next((e for e in self.p.get('items') or [] if (e.get('component'), e.get('id')) == (comp, uid)
                      and (e.get('component'), e.get('id'), e.get('k')) not in self.itemised), None)
            if e is not None:
                self.itemised.add((comp, uid, e.get('k')))
        # a give that did not verify waits for the next round (it may have landed): an update that restates it does
        # not drop it again now (run 20261015-003101 3-2: the Tear did land on Krug, the update repeated "Tear ->
        # Krug" and the second Tear built Blue Buff instead of Protector's Vow)
        for comp, uid, _ in sorted(old.item_later, key=repr):
            e = next((e for e in self.p.get('items') or [] if (e.get('component'), e.get('id')) == (comp, uid)
                      and (e.get('component'), e.get('id'), e.get('k')) not in self.item_later), None)
            if e is not None:
                self.item_later.add((comp, uid, e.get('k')))
        rn = self.p.get('roll_n')
        rolled = old.rolled - snap['rolled']
        if isinstance(rn, dict) and rolled > 0:
            self.rolls_left = max(0, int(rn.get('times') or 0) - rolled)
        return {'levels': self.levels_done, 'buys': len(self.bought), 'rolls': rolled}

    def unsent(self, act):
        """An action this plan handed out was never sent (no input possible): count it as not
        done, so it is handed out again."""
        if act.kind == 'level':
            self.levels_done = max(0, self.levels_done - 1)
        elif act.kind == 'roll':
            self.rolls = max(0, self.rolls - 1)
        elif act.kind == 'sell' and getattr(act.arg, 'id', None) in self.sold:
            self.sold.discard(act.arg.id)
        elif act.kind == 'buy' and act.why.startswith('plan: buy '):
            name = act.why[len('plan: buy '):]
            k = max((k for k in self.bought if k < len(self.to_buy) and self.to_buy[k] == name), default=None)
            if k is not None:
                self.bought.discard(k)
        elif act.kind == 'buy' and act.why.startswith('plan: wisp'):
            self.wisps_done.discard(act.arg)
        elif act.kind == 'item' and getattr(act, 'item_key', None) in self.itemised:
            self.itemised.discard(act.item_key)      # an item give held over (the round flips): given again

    def next_action(self, scr, roster, cap, combat=False):
        """The next step, skipping any that already failed twice. cap: the board slots (the level). combat: the fight is running - the board is locked (no
        board moves, no selling or itemising board units), but buying, selling bench units, rerolling and XP
        work; the rest waits for the next planning phase."""
        self.combat = combat
        a = self._next(scr, roster, cap)
        refused = lambda a: a.kind == 'item' and (a.arg[1], getattr(a.arg[2], 'id', None)) in self.refused
        while a is not None and (self.tries.get(repr(a), 0) >= 2 or refused(a)):
            self.given_up.add(repr(a))
            a = self._next(scr, roster, cap, skip=self.given_up)
        return a

    @staticmethod
    def fast_rr_met(fr, roster):
        """Every stop_when entry has one of its champions with that star level's copies (3 for a 2-star,
        9 for a 3-star) on board + bench: a 1-star fielded and two benched count as a 2-star, merged
        or not."""
        conds = [c for c in fr.get('stop_when') or [] if isinstance(c, dict) and c.get('any')]

        def copies(n):
            return sum(3 ** (u.star - 1) for u in roster.units if u.name == n and not u.missing)
        return bool(conds) and all(any(copies(n) >= 3 ** (int(c.get('star') or 2) - 1) for n in c['any'])
                                   for c in conds)

    def _fast_rr(self, fr, scr, roster, gold):
        want = {n for n in fr.get('buy') or [] if CATALOG.unit(n)}
        floor = int(fr.get('gold_floor') or 0)
        if self.fast_rr_met(fr, roster):
            self.fast_done = True
            self.done.append('fast_rr: stop condition met')
            return None
        if fr.get('wisps') and not getattr(self, 'combat', False):
            i = next((k for k in range(5) if scr.shop_raw[k] and not scr.shop[k]), None)
            if i is not None and ('fast', self.rolls, i) not in self.wisps_done:
                self.wisps_done.add(('fast', self.rolls, i))
                return Action('buy', i, f'plan: fast_rr wisp {scr.shop_raw[i]}')
        for i, name in enumerate(scr.shop):
            if name not in want or gold - CATALOG.unit(name).cost < min(floor, gold):
                continue
            # a third copy merges on the spot: no bench space needed
            if roster.free_bench() or len(roster.copies(name, 1)) >= 2:
                return Action('buy', i, f'plan: fast_rr buy {name}')
            if fr.get('sell_to_make_room'):
                keep = {k for k in fr.get('keep') or [] if isinstance(k, int)} | self.protected_ids()
                spare = [u for u in roster.bench() if u.id not in keep and u.name not in want]
                if spare:
                    u = min(spare, key=lambda x: (x.cost * x.star, x.loc))
                    return Action('sell', u, f'plan: fast_rr make room ({u})')
        if gold - 2 >= floor:
            self.rolls += 1
            return Action('roll', None, f'plan: fast_rr roll {self.rolls} (floor {floor})')
        self.fast_done = True
        self.done.append(f'fast_rr: stopped at gold {gold} (floor {floor})')
        return None

    def _bench_left(self, roster):
        """[(unit, slot)] the plan's "bench" placements not done yet (nor given up)."""
        out = []
        for e in self.p.get('bench') or []:
            u = self.unit(roster, e.get('id'))
            k = e.get('slot')
            if u is not None and isinstance(k, int) and u.loc != ('bench', k) and (u.id, k) not in self.benched:
                out.append((u, k))
        return out

    def _lineup(self, roster):
        """[(unit, (row, col))] resolving ids, then names of units bought this round."""
        want, used = [], set()
        for e in self.p.get('board') or []:
            hex_ = tuple(e.get('hex') or ())
            if len(hex_) != 2 or not (0 <= hex_[0] <= 3 and 0 <= hex_[1] <= 6):
                continue
            u = self.unit(roster, e.get('id'), used) if 'id' in e else None
            if u is None and e.get('name'):
                u = next((x for x in sorted(roster.units, key=lambda x: (not x.on_board, x.star * -1))
                          if x.name == e['name'] and x.id not in used), None)
            if u is not None and u.id not in used:
                want.append((u, hex_))
                used.add(u.id)
        return want

    def _next(self, scr, roster, cap, skip=()):
        if scr.gold is not None:
            self.last_gold = scr.gold
        gold = scr.gold if scr.gold is not None else (self.last_gold or 0)
        ok = lambda a: a if repr(a) not in skip else None
        # 0. wisps the model chose to buy, before anything else (a Wisp can reroll the shop or give gold); a
        #    wisp slot is text that isn't a champion. Never in combat: Wisps leave the shop when it starts
        for w in ([] if getattr(self, 'combat', False) else self.p.get('wisps') or []):
            i = w.get('slot') if isinstance(w, dict) else None
            if not w.get('buy') or not isinstance(i, int) or not 0 <= i < 5 or i in self.wisps_done:
                continue
            self.wisps_done.add(i)
            if scr.shop_raw[i] and not scr.shop[i]:
                return Action('buy', i, f"plan: wisp {scr.shop_raw[i]} ({w.get('why') or ''})")
        # 0b. a choice container (an anvil) first while the choice can still be answered: the model needs
        #    ~9 s, and opened last it came with ~8 s left (run 152203 4-3: the pick was lost, the game chose).
        #    Later than that it waits for the fight, where opening works too.
        if not getattr(self, 'combat', False) and scr.timer is not None and scr.timer >= 11:
            act = self._container()
            if act:
                return act
        if getattr(self, 'combat', False):
            # the fight: the board is locked - bench sells, buys, XP and rolls work; board moves wait for the next
            # planning phase, items for the countdown before the fight (countdown_action) or the next planning phase
            return (self._sell(roster, board=False) or self._buy(scr, roster, gold) or self._level(gold)
                    or self._rolldown(gold) or self._container())
        # planning, by priority (the user, 2026-10-10): the board the fight is played with, then the items, then the
        # bench's sells and buys (they also work in the fight). What the board needs comes first: selling board
        # units, buying the units it fields (or a copy of one), the level-ups
        need = self._board_names(roster)
        bench_first = self._needed_buy_blocked(scr, roster, gold, need)
        return (self._sell(roster, bench=bench_first) or self._buy(scr, roster, gold, only=need)
                or self._level(gold) or self._board(roster, cap, ok) or self._links(roster)
                or self._items(scr, roster)
                or self._sell(roster) or self._buy(scr, roster, gold, exclude=need)
                or self._rolldown(gold) or self._container())

    def fight_give(self, scr, roster):
        """The fight is running: the plan's hand-outs not done yet go on now - the executor finds the moving unit by
        reading it off its panel first (the user, 2026-10-14: items can be given in combat; run 20261014-063145 4-6:
        Aphelios fought without the items the plan gave him, they went on at the next planning phase)."""
        self.combat = True
        a = self._items(scr, roster, gives_only=True)
        if a is not None:
            a.fight = True
        return a

    def countdown_action(self, scr, roster):
        """The 5-4-3-2-1 countdown before the fight: units stand still on their hexes (in an away fight they leave at
        2) and only items can be given - the plan's items not given in planning go on now (the user, 2026-10-10)."""
        self.combat = True
        return self._items(scr, roster)

    def _board_names(self, roster):
        """Champions the plan's board fields (by name or unit) and the ones fielded now: buying one of them changes
        the board (a unit to field, or a copy toward its next star)."""
        names = {u.name for u in roster.board() if u.name}
        for e in self.p.get('board') or []:
            if e.get('name'):
                names.add(e['name'])
            elif 'id' in e:
                u = self.unit(roster, e['id'])
                names.add((self.names.get(e['id']) or (None,))[0] or getattr(u, 'name', None))
        names.discard(None)
        return names

    def _needed_buy_blocked(self, scr, roster, gold, need):
        """A buy the board needs is in the shop but the bench is full or the gold short: the sells go first."""
        cost = 0
        for k, name in enumerate(self.to_buy):
            if k in self.bought or name not in need or name not in (scr.shop or []):
                continue
            cost += CATALOG.unit(name).cost
            if not roster.free_bench() and len(roster.copies(name, 1)) < 2:
                return True
        return cost > 0 and gold - cost < min(self.keep, gold)

    def _sell(self, roster, board=True, bench=True):
        """The plan's next sell: of a board unit (board), of a bench unit (bench)."""
        for uid in self.p.get('sell') or []:
            u = self.unit(roster, uid)
            meant = self.names.get(uid)
            if u and meant and meant[0] and u.name != meant[0] and uid not in self.sold:
                # the unit was re-read as another champion since the plan was written (run 221257 5-3:
                # "sell the spare Leona" sold Aphelios, re-read as Aphelios after the plan): not sold
                self.sold.add(uid)
                self.failed.append(f'sell unit {uid}: it is now read as {u.name}{"*" * u.star}, '
                                   f'not {meant[0]} as when you planned it - not sold')
                continue
            if u and u.star != (meant or (None, u.star))[1]:
                u = None                         # never sell a different star level than meant
            if u is None or uid in self.sold or not (board if u.on_board else bench):
                continue                         # (a fighting unit can't be sold: after the fight)
            self.sold.add(uid)
            return Action('sell', u, f'plan: sell {u}')
        return None

    def _buy(self, scr, roster, gold, only=None, exclude=()):
        """The plan's next buy, by name, in the order given (each entry once). A buy is done once clicked. A missing
        read is not an answer: without a gold or shop reading, wait; give it up only when the champion is absent from
        4 shop reads in a row, or truly unaffordable (before, one blank read dropped it for good - 2 buys a game)."""
        for k, name in enumerate(self.to_buy):
            if k in self.bought or (only is not None and name not in only) or name in exclude:
                continue
            if not getattr(scr, 'shop_visible', True) or (scr.gold is None and self.last_gold is None):
                return None
            i = next((j for j in range(5) if scr.shop[j] == name), None)
            if i is None:
                self.buy_misses[k] = self.buy_misses.get(k, 0) + 1
                if self.buy_misses[k] >= 4:
                    self.bought.add(k)
                    self.failed.append(f'buy {name}: not in the shop')
                continue
            if gold - CATALOG.unit(name).cost < min(self.keep, gold):
                self.bought.add(k)
                self.failed.append(f'buy {name}: gold {gold} is below the floor')
                continue
            if not roster.free_bench() and len(roster.copies(name, 1)) < 2:
                # the game refuses it (a 3rd copy would merge and is fine): say so at once instead of
                # clicking and waiting 2.5 s for nothing - with Cluttered Mind the bench is kept full
                self.bought.add(k)
                self.failed.append(f'buy {name}: the bench is full (9/9) - sell a unit first in the same plan')
                continue
            self.bought.add(k)
            return Action('buy', i, f'plan: buy {name}')
        return None

    def _level(self, gold):
        """Level-ups: all of them in ONE action - F pressed n times in a row (0.79 s per level-up with a screen read
        each in run26; a player just presses F n times)."""
        left = int(self.p.get('level_ups') or 0) - self.levels_done
        if left > 0 and gold >= 4:                    # explicit: not held to keep_gold
            n = max(1, min(left, gold // 4))
            self.levels_done += n
            return Action('level', n, f'plan: level up x{n}')
        return None

    def _board(self, roster, cap, ok):
        # a full board, a fielded unit to bench and a bench unit to field: ONE drag of the bench unit onto the
        # fielded unit's hex swaps the two (the user, 2026-10-10: fielding comes first - W then a drag took two
        # actions, and run 20261010-064944 2-3 fielded Lillia with 1 s left). The benched unit lands on the other's
        # old bench slot (a slot the plan names follows below), the fielded unit's own hex follows as a board move.
        # The executor reads who stands on the hex after a swap.
        # (team slots, not units. An Elder Dragon is never swapped in: the game takes him only onto a board with 2
        # slots free - the user, 2026-10-13 - so the unit leaving is benched first and he goes onto an empty hex)
        used = roster.slots_used()
        ins = [(u, h) for u, h in self._lineup(roster) if not u.on_board and u.slots == 1 and used + 1 > cap]
        if ins:
            outs = [u for u, k in self._bench_left(roster) if u.on_board and k not in roster.blocked]
            pairs = [(o, u, h) for o in outs for u, h in ins if used - o.slots + u.slots <= cap]
            if pairs:
                o, u, _ = next(((o, u, h) for o, u, h in pairs if o.loc == ('board', *h)), pairs[0])
                a = ok(Action('move', (u, o.loc), f'plan: {u} -> {tuple(o.loc[1:])} (swap with {o})'))
                if a:
                    return a
        # bench slots the plan names (a fielded unit named here leaves the board; an effect can depend on the
        # slot): a unit already in the slot is moved to another free slot first (a drop onto a unit swaps
        # the two, which the screen can't confirm). Fielded units first: they free board slots; a bench unit's
        # slot change waits until the lineup is done (it does not change the fight)
        a = self._bench_moves(roster, ok, fielded=True)
        if a:
            return a
        # lineup: the units "board" lists go to their hexes - only the model's placements. Units it does not
        # list stay where they are (the user, 2026-10-09): it benches a unit with "bench", or places a unit
        # on an occupied hex, which moves the occupant off it
        want = self._lineup(roster)
        for u, hex_ in want:
            if u.loc == ('board', *hex_):
                continue
            occupant = roster.at(('board', *hex_))
            if occupant is None and not u.on_board and roster.slots_used() + u.slots > cap:
                if u.id not in self.full_told:           # the game refuses a unit over the team size: said once
                    self.full_told.add(u.id)
                    self.failed.append(f'{u} -> {tuple(hex_)}: the board is full ({roster.slots_used()}/{cap} team '
                                       f'slots{f"; {u.name} needs {u.slots}" if u.slots > 1 else ""}) - '
                                       + (f'bench units ("bench") until {u.slots} slots are free: {u.name} cannot be '
                                          f'swapped in for a fielded unit' if u.slots > 1 else
                                          f'bench a unit ("bench") or place {u.name or "it"} on an occupied hex'))
                continue
            if occupant is not None and occupant is not u:
                # an occupant with a free planned hex of its own goes there first; otherwise ONE drag of the unit onto
                # the occupant swaps the two - the occupant takes the unit's old place (the user, 2026-10-16: click the
                # bench unit and drag it onto the board unit it replaces). The executor reads who stands there after.
                # (It used to bench the occupant first, onto a "free" slot: run 20261016-011010 2-2 dragged Rakan onto
                # the component anvil, and the Coven units were fielded late.)
                dest = next((('board', *h) for o, h in want if o is occupant
                             and roster.at(('board', *h)) is None), None)
                if u.slots > 1 and not u.on_board:
                    # an Elder Dragon is never swapped in (2 team slots): the hex is cleared first
                    if dest is None and roster.free_bench():
                        dest = ('bench', roster.free_bench()[0])
                    if dest is None:
                        continue
                if dest is not None:
                    a = ok(Action('move', (occupant, dest), f'plan: clear {hex_} - {occupant} -> {dest}'))
                    if a:
                        return a
                    if u.slots > 1 and not u.on_board:
                        continue
                a = ok(Action('move', (u, ('board', *hex_)), f'plan: {u} -> {hex_} (swap with {occupant})'))
                if a:
                    return a
                continue
            a = ok(Action('move', (u, ('board', *hex_)), f'plan: {u} -> {hex_}'))
            if a:
                return a
        return self._bench_moves(roster, ok, fielded=False)

    def _bench_moves(self, roster, ok, fielded):
        """The plan's "bench" placements of fielded units (fielded=True) or of bench units (False)."""
        for u, k in self._bench_left(roster):
            if u.on_board != fielded:
                continue
            if k in roster.blocked:
                self.benched.add((u.id, k))
                self.failed.append(f'bench {u} to slot {k}: slot {k} holds {roster.blocked[k]}')
                continue
            occ = roster.at(('bench', k))
            free = [i for i in roster.free_bench() if i != k]
            if occ is not None and occ is not u and free:
                a = ok(Action('move', (occ, ('bench', free[0])), f'plan: clear bench slot {k} - {occ}'))
                if a:
                    return a
            a = ok(Action('move', (u, ('bench', k)), f'plan: {u} -> bench slot {k}'))
            if a:
                return a
        return None

    def _links(self, roster):
        """The plan's links (drag from the source, hover on the target ~1 s, release - the BFF to its rider, Taric
        to his ally), each once, after the board is placed. from: a unit id or a board hex [row, col]; to: a unit."""
        for k, e in enumerate(self.p.get('link') or []):
            if ('link', k) in self.linked:
                continue
            self.linked.add(('link', k))
            src, to = e.get('from'), self.unit(roster, e.get('to'))
            if isinstance(src, (list, tuple)) and len(src) == 2 and all(isinstance(v, int) for v in src):
                loc = ('board', src[0], src[1])
            else:
                su = self.unit(roster, src)
                loc = su.loc if su is not None and su.on_board else None
            if loc is None or to is None or not to.on_board or to.loc == loc:
                self.failed.append(f'link {src} -> {e.get("to")}: needs a board unit or hex to drag from and '
                                   f'another unit on the board to link to')
                continue
            return Action('link', (loc, to), f'plan: link {loc[1:]} -> {to}')
        return None

    def _items(self, scr, roster, gives_only=False):
        # first the Removers / Reforgers (they return items to the bench), then the hand-outs. gives_only: in the
        # fight, where a give finds its unit by reading it, but a Remover/Reforger waits for the units to stand still
        for e in ([] if gives_only else self.p.get('strip') or []):
            key = ('strip', e.get('id'), e.get('tool'))
            if key in self.itemised:
                continue
            u = self.unit(roster, e.get('id'))
            want = is_reforger if e.get('tool') == 'reforger' else is_remover
            slot = next((k for k, it in enumerate(scr.items or []) if want(it)), None)
            if not u or not u.items:
                self.itemised.add(key)           # nothing on it to take off
                continue
            if slot is None:
                self.item_waits[key] = self.item_waits.get(key, 0) + 1
                if self.item_waits[key] > 30:
                    self.itemised.add(key)
                    self.failed.append(f'{e.get("tool")}: none on the item bench')
                continue
            self.itemised.add(key)
            return Action('strip', (slot, scr.items[slot], u), f'plan: {e.get("tool")} on {u}')
        for e in list(self.p.get('items') or []):
            key = (e.get('component'), e.get('id'), e.get('k'))
            if key in self.itemised or key in self.item_later:
                continue
            u = self.unit(roster, e.get('id'))
            if gives_only and (u is None or u.loc[0] != 'board'):
                continue                         # in the fight only fielded units can be found (bench units show no bar)
            name = e.get('component')
            bench = list(scr.items or [])
            base = lambda n: re.sub(r'\s*\(.*$', '', n or '')
            if 'from' in e and ('strip', e['from'], 'remover') not in self.itemised:
                continue                         # a transfer: its source's items come off first
            slot = next((k for k, it in enumerate(bench) if it and base(it) == base(name)), None)
            if slot is None and u and 'from' not in e and any(base(i) == base(name) for i in u.items):
                # the unit already holds it (given earlier, or a plan repeating its lineup): done
                self.itemised.add(key)
                self.done.append(f'{name} already on {u}')
                continue
            if not u or not name:
                self.itemised.add(key)           # nothing to do for this entry
                self.failed.append(f'item {name} -> unit {e.get("id")}: no such unit or item name')
                continue
            if slot is None:
                # Never substituted (a completed item that is not there is NOT given as its two
                # components: building one is the model's own instruction, one entry per
                # component). An unidentified slot may be the item: it is read now, then retried.
                # The item bench also reads empty for a moment after each drop.
                unread = any(it == 'unknown' for it in bench)
                if unread:
                    self.need_identify = True
                self.item_waits[key] = self.item_waits.get(key, 0) + 1
                if self.item_waits[key] > (80 if unread else 30):
                    self.itemised.add(key)
                    held = ', '.join(('an unidentified item' if i == 'unknown' else str(i)) for i in bench if i)
                    self.failed.append(f'item {name} -> {u}: not on the item bench (it holds: {held or "nothing"})'
                                       + (f' - {name} is a completed item: to build it, give its components '
                                          f'({" + ".join(RECIPES[name])}) as separate entries'
                                          if name in RECIPES else ''))
                continue
            self.itemised.add(key)
            a = Action('item', (slot, name, u), f'plan: {name} -> {u}')
            a.item_key = key
            return a
        return None

    def roll_pending(self):
        """The plan has a roll that has not run (or not finished) yet."""
        roll = self.p.get('roll') or {}
        if isinstance(self.p.get('roll_n'), dict) and not self.rolln_done:
            return True
        return not self.fast_done and (isinstance(self.p.get('fast_rr'), dict) or
                                       isinstance(roll.get('down_to'), (int, float)))

    def _rolldown(self, gold):
        """Roll-downs run as ONE scripted action (player.run_rolldown): D, wait for the cards, read the
        shop only, buy targets, repeat - no full screen read between rolls (run26: 1.9 s per roll; ~0.5 s now)."""
        if getattr(self, 'hold_roll', False):
            return None                          # a Wisp question is open: its answer may change the shop
        combat = getattr(self, 'combat', False)
        rn = self.p.get('roll_n')
        if isinstance(rn, dict) and not self.rolln_done:
            # reroll n times; a Wisp in a new shop stops it (the model is asked), then the rest goes on
            left = self.rolls_left if self.rolls_left is not None else int(rn.get('times') or 0)
            if left <= 0:
                self.rolln_done = True
            else:
                self.rolls_left = left
                # floor 0: an explicit roll is exempt from gold_floor (the prompt says so) - run 221257
                # 4-7 stopped after 4 of 6 rolls at the floor and left a listed Nidalee in the last shop
                spec = {'buy': [n for n in rn.get('buy') or [] if CATALOG.unit(n)], 'floor': 0,
                        'targets': buy_targets(rn.get('buy')),
                        'times': left, 'stop_when': [], 'wisps': False if combat else 'ask',
                        'keep': self.protected_ids(), 'sell_others': bool(rn.get('sell_others'))}
                return Action('rolldown', spec, f"plan: roll x{left} for {', '.join(spec['buy']) or 'nothing'}")
        try:
            stage_n = int(str(self.stage).split('-')[0])
        except ValueError:
            stage_n = 0
        # fast roll-down: every Wisp bought on the spot, no question - from stage 4 (the user: the harmful
        # ones do not appear from then on); earlier, only the first Wisp a roll brings up (the user, 2026-10-12:
        # "auto buy the first one" - run 20261012-072956 3-6 rolled past five)
        auto = False if combat else 'auto' if stage_n >= 4 else 'first'
        roll = self.p.get('roll') or {}
        fr = self.p.get('fast_rr')
        if not self.fast_done and (isinstance(fr, dict) or isinstance(roll.get('down_to'), (int, float))):
            self.fast_done = True
            if isinstance(fr, dict):
                spec = {'buy': [n for n in fr.get('buy') or [] if CATALOG.unit(n)],
                        'targets': buy_targets(fr.get('buy')),
                        'floor': int(fr.get('gold_floor') or 0), 'stop_when': fr.get('stop_when') or [],
                        'wisps': auto, 'sell_others': fr.get('sell_others', True) is not False,
                        'keep': {k for k in fr.get('keep') or [] if isinstance(k, int)} | self.protected_ids()}
            else:
                spec = {'buy': [n for n in roll.get('buy') or [] if CATALOG.unit(n)],
                        'targets': buy_targets(roll.get('buy')),
                        'floor': int(roll['down_to']), 'stop_when': [], 'wisps': auto,
                        'keep': self.protected_ids(), 'sell_others': roll.get('sell_others', True) is not False}
            return Action('rolldown', spec, f"plan: roll-down for {', '.join(spec['buy']) or 'nothing'} "
                                            f"to {spec['floor']} gold")
        return None

    def _container(self):
        """The next choice container (an anvil: no health bar, not a unit) to open - selling opens it."""
        for slot in self.p.get('open_container') or []:
            if isinstance(slot, int) and 0 <= slot <= 8 and slot not in self.containers_done:
                self.containers_done.add(slot)
                return Action('container', slot, f'plan: open the container on bench slot {slot}')
        return None
