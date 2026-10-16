"""The real-time player loop.

    capture (continuous) -> read HUD + occupancy -> update roster
      -> brain.decide -> one input -> repeat

Started by tools/play.py. Stop: F8 at any time.
"""
from __future__ import annotations

import argparse
import atexit
import json
import math
from collections import Counter
import os
import re
import threading
import time
import types
from pathlib import Path

import cv2
import numpy as np

from tft_player import board as B
from tft_player.brain import Action, Brain
from tft_player.coach import Coach, local_augment, set_stage
from tft_player.loot import BeamOrbs, LootRunner, find_beams, find_orbs, find_tactician
from tft_player.carousel import (GATE_RING_OFF, GATE_RING_ON, CarouselChaser, OwnTracker, find_bars,
                                 find_carousel_items, gate_ring, release_timer)
from tft_player.items import COMPONENTS, BY_PAIR, N_SLOTS, ItemTracker, read_item_bench, read_panel_items, slot_point, uses_left
from tft_player.screen import ScreenReader
from tft_player.state import BFF
from tft_player.live import LiveView
from tft_player.catalog import CATALOG
from tft_player.state import Roster


class NotForeground(Exception):
    pass

AUGMENT_CARDS = [(555, 450), (960, 450), (1365, 450)]
AUGMENT_REROLL = [(555, 864), (960, 864), (1365, 864)]
SETTINGS_CLOSE = (1476, 163)       # X of the in-game Settings menu (1080p reference)
# where our tactician is parked each planning phase: open ground at the arena's right edge, ~95 px from the
# nearest hexes and the bench (its default spot, left of hexes (1,0)/(0,0), hid units there: run 213127 1-4
# Hecarim, 4-3 Varus); the bar scan ends at x 1480, so its own bar is out of the way too
# where the tactician waits in a planning phase: open ground above our board's top-left corner, on the half that holds
# enemy units only in combat - it covers none of our units, bars, bench slots or item rows. The ledge in the bottom-left
# corner (the user's spot of 2026-10-10, where the game puts it after every trip; feet 426, 603) hid bench slots 0-1
# and hex (1,0) behind a large tactician: panel reads clicked it, E-sells missed, a Lifebloom and Nidalee's item row
# went unseen (runs 20261015-115044, -132144, -161625). Same click-to-feet offset as there (47 px).
PARK, PARK_FEET = (690, 337), (690, 290)
# seconds of planning left from which fielding comes first: while a plan step still has to put a unit on the board
# (or bench one), loot, reads and checks wait (the user, 2026-10-10: run 064944 moved units with 1-3 s left)
FIELD_FIRST_S = 6
RUNS = Path(__file__).resolve().parent.parent / 'runs'
REPLAY = bool(os.environ.get('TFT_REPLAY'))


def test_mode():
    """Test-game data, in runs/<ts>: the carousel frames, the per-round unit crops and the link-drag frames. Only with
    --record (a test-only option): by default a game runs exactly as the release's does (the user, 2026-10-15: the
    release is the test version, functionally and visually). Never on a replay."""
    return not REPLAY and os.environ.get('TFT_TEST_MODE') == '1'


def shots_on():
    """Screenshots go into the game's folder (round starts, fights, failed actions, the images sent to the model);
    off with --no-screenshots - the log and the model conversation are kept either way."""
    return not REPLAY and os.environ.get('TFT_NO_SCREENSHOTS') != '1'
DATA_MODE = bool(os.environ.get('TFT_DATA_MODE'))   # data-collection games (tools/play_comp.py --data)
SELL_HOVER = 0.2         # s the cursor rests on a unit before E sells it (Input.key then holds E for KEY_HOLD)
MERGE_SETTLE = 1.2       # s a newly lit slot must stay lit before a merge we expected is judged not to have happened
DATA_ROLL = bool(os.environ.get('TFT_DATA_ROLL'))   # attached to the user's game at level 8: only reroll /
                                                    # buy / photograph / sell what WE bought (--data-roll)     # tools/replay_player.py: recorded frames, clicks do nothing


class Watcher(threading.Thread):
    """Slow, non-HUD perception off the main loop: tactician position every
    frame, loot orbs when looting is on. cv2 releases the GIL, so this runs in
    parallel and the loop just reads the latest results."""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self.want_orbs = False
        self.tac = (None, None)
        self.orbs, self.orbs_at = [], 0.0
        self.beams = BeamOrbs()
        self.running = True

    def run(self):
        while self.running:
            got = self.cap.read()
            if got is None:
                time.sleep(0.05)
                continue
            frame = got[0]
            try:
                self.tac = find_tactician(frame)
                # orbs at most 4 times a second (79 ms a scan: every frame took a whole core from the loop, run
                # 20261015-161625); the router predicts arrival between scans
                if self.want_orbs and time.time() - self.orbs_at > 0.25:
                    orbs = find_orbs(frame)
                    # gold orbs hidden among our units, found by their standing light beam
                    orbs += [o for o in self.beams.update(find_beams(frame))
                             if all(math.dist(o[:2], q[:2]) > 50 for q in orbs)]
                    self.orbs, self.orbs_at = orbs, time.time()
            except Exception:
                pass
            time.sleep(0.08)                     # ~10 looks a second: the tactician walks, it does not jump


def is_carousel(scr):
    """A carousel round: every x-4 from stage 2. Not 1-1: this set opens with an encounter there, no carousel
    (the user, 2026-10-10; run 20261010-064944 ran the carousel loop through it)."""
    r = scr.round
    return bool(r) and r[0] >= 2 and r[1] == 4


class Phase:
    """The stage number changing is the start of planning; within planning the countdown only goes down, so
    once it has been seen running, a rise is the next phase (combat start, 5 -> 0)."""

    def __init__(self):
        self.stage = None
        self.planning = False
        self.last_timer = None
        self.changed = False
        self.candidate = None
        self.running = False    # the planning countdown was seen going down
        self.rise = None        # a countdown rise read once, mid-planning
        self.countdown = False  # the 5-4-3-2-1 before the fight: units stand still, only items can be given

    @staticmethod
    def _key(stage):
        a, b = stage.split('-')
        return int(a), int(b)

    @staticmethod
    def next_stage(stage):
        a, b = Phase._key(stage)
        return f'{a + 1}-1' if b >= (4 if a == 1 else 7) else f'{a}-{b + 1}'

    def update(self, scr):
        self.changed = False
        if scr.stage and scr.stage != self.stage:
            # rounds come in order: once the stage is known, only the next one can follow (anything else is a
            # misread - a dimmed 3-3 read "5-3" for a whole fight, run 20260927-151256) and counts at its first
            # read; the first stage of a run must be read twice in a row
            if self.stage:
                if scr.stage != self.next_stage(self.stage):
                    return
            elif self.candidate != scr.stage:
                self.candidate = scr.stage
                return
            self.stage, self.planning, self.changed = scr.stage, True, True
            self.running, self.rise, self.candidate, self.countdown = False, None, None, False
            self.last_timer = scr.timer
            return
        if scr.stage:
            self.candidate = None       # only a read of the current stage cancels a candidate:
                                        # a tick that reads no stage must not (it froze us at 2-2)
        t, lt = scr.timer, self.last_timer
        if t is None:
            return
        if self.countdown and (t == 0 or (lt is not None and t > lt)):
            self.countdown = False      # the fight began (units move, or leave for an away fight)
        if self.planning and lt is not None and t > lt:
            if not self.running:
                pass                    # the countdown starting (its first value), not a new phase
            elif lt <= 2 or self.rise is not None:
                # the countdown ran out and restarted: combat start (0 -> 5, or 0 -> 2 when a tick read it
                # late - the old "> 4" jump missed that and moved units into the locked board, runs 010345/185805)
                self.planning, self.rise = False, None
                self.countdown = t <= 8         # the countdown before the fight (5 s; augment rounds read 7-8)
            else:
                self.rise = t           # mid-planning: a rise counts on its second read (OCR glitch)
                return
            self.last_timer = t
            return
        if self.planning and lt is not None and t < lt:
            self.running = True
        self.rise = None
        self.last_timer = t




def loc_point(loc):
    return B.BENCH[loc[1]] if loc[0] == 'bench' else B.hex_xy(loc[1], loc[2])


class Player:
    def __init__(self, capture, inp, log_dir: Path, until=None, overlay=None, client=None, prep=None):
        """prep: the preparation thread started as we entered the queue (play_match.start_preparation; its
        .result is coach.prepare()'s); without it the briefing goes out now."""
        self.cap, self.inp = capture, inp
        self.reader = ScreenReader()
        B.TALL_HINT = self.tall_hint     # our tall units' bars map to their own hexes (board.BAR_DY)
        from tft_player.ending import EndWatcher
        # separate reader: its OCR runs in another thread
        self.ending = EndWatcher(capture, ScreenReader(), client, log=lambda **e: self.log(**e))
        self.exit_tried = 0.0
        self.reconciled_stage = None     # stage whose board/bench the roster has checked against the screen
        self.round_plan = None           # the model's plan for this planning phase (planner.RoundPlan)
        self.announced = set()           # unit ids already reported to the model as new
        self.item_names, self.item_tries = {}, {}    # item-bench icon signature -> name read from its tooltip
        self.augments_taken = []         # names, in the order picked
        self.encounter = None            # this game's encounter, as the 1-1 banner says it
        self._enc_busy = False
        self.wisps_bought = {}           # stage -> Wisps bought that round
        self.plan_pending = False        # a plan call is in flight
        self.plan_calls = 0              # plan calls this round (first + updates)
        self.carry_note = ''             # what last round's plan left undone, for the model
        self.plan_asked = None           # stage the plan was requested for
        self.next_plan = None            # (stage made at, plan): the model's next_round, run when the next planning opens
        self.next_plan_names = {}
        self.carousel_stage = None
        self.last_level = 1              # the plan never waits for a level reading
        self.key_asked = set()           # key decision points the model was reminded of
        self.pivoted = set()             # stages where the model changed comp
        self.fail_cause = None           # why the last executed step failed
        self.hp_log = []                 # (stage, hp at start of previous round, hp now)
        self.hp_at_round = None
        self.roster = Roster()
        self.brain = Brain()
        self.pilot = None
        if self.pilot:
            self.brain = self.pilot
            print(f'pilot: {self.pilot.comp.name} -> {", ".join(self.pilot.final)}')
        self.phase = Phase()
        self.live = LiveView()
        self.overlay = overlay
        self.overlay_at = 0.0
        if self.live.port:
            print(f'live view: http://localhost:{self.live.port}')
        from tft_player import coach as _coach
        if _coach.CALL_DIR != log_dir / 'calls':     # not set by coach.new_game before the queue (replays, attach)
            _coach.CALL_DIR = log_dir / 'calls'      # before the briefing thread starts: its inputs are data too
            _coach._CALL_N[0] = 0
        from tft_player.coach import CLAUDE_EFFORT, CLAUDE_MODEL, _cli, warm_up
        warm_up()
        self.comp_p = {}                 # comp name -> P(final board), from the model's last reply
        self.game_strategy = {}          # the model's strategy, section -> text (game_strategy in its replies)
        self.target = None               # comp name: the fixed or committed comp (flex: None until a commit)
        self.pending_pages = None        # (comp name, [jpeg]) to show the model at the next combat review
        self.pages_sent = set()
        from tft_player.coach import forced_comp
        self.forced = forced_comp()      # TFT_FORCE_COMP (tools/play_comp.py): one comp, fixed all game
        self.committed = None            # flex: the stage the model committed to a comp (it is self.forced from then)
        if self.forced:
            self.target = self.forced.name
        # standing orders the model sets (execution_plan.auto_buy / auto_sell), kept until it changes them:
        # auto-buy (champion -> star target) starts as the fixed comp's final board at 2 stars; auto-sell is off
        self.auto_buy = {n: 2 for n in self.forced.units if CATALOG.unit(n)} if self.forced else {}
        self.auto_sell = False
        self.item_finals = []            # [{'item', 'holder' (unit id), 'final' (champion), 'star'}]: the model's
                                         # final holders (execution_plan.item_final / a give's "final")
        self.give_finals = {}            # (unit id, item given) -> (final, star): set on the item the give makes
        self.handover_rp = None          # items moving to their final holder (it runs as the round plan meanwhile)
        self.auto_done, self.auto_tries = [], {}
        kind, _ = _cli()
        from tft_player.coach import model_name
        print('model:', f'{CLAUDE_MODEL} via {kind} (effort {CLAUDE_EFFORT})'
              if kind in ('claude', 'claude-cli') else model_name() if kind else
              'NONE - set ANTHROPIC_API_KEY to connect Claude; playing on local fallback rules only')
        self.coach = Coach(self.brain, log=lambda m: (print(m), self.log(ev='coach', msg=m)),
                           on_answer=lambda ans: self.update_comps(ans, self.phase.stage),
                           on_call=self.model_call)
        self.augment_seen = None
        self.augment_read = [None] * 3   # pilot mode: the cards' names, read by OCR
        self.surveys = {}                # stage -> surveys done (pilot mode)
        self.need_survey = None          # a reason to read every unit off the screen (pilot mode)
        self._trait_bad = 0
        self._trait_left = ()            # trait differences a survey did not explain (emblems)
        self.checked_stage = None        # the stage whose board was checked against the screen
        self.surveyed_at8 = False
        self.panel_items = []            # items shown by the last unit panel read
        self.pilot_call = None           # pilot mode: the model's current / last judgement call
        self.aug_answer = None           # pilot mode: the model's answer about the augment hand
        self.wisp_answers = {}           # pilot mode: (stage, Wisp) -> 'asked' | True | False
        self.hud_seen = None             # the last screen with the HUD up (the picker hides gold)
        self.bench_objs = {}             # bench slot -> {'kind', 'cls', 'conf'}: non-unit objects seen there
        self._bench_prev = {}
        self.bench_kinds = None          # the last read of all 9 bench slots (bench.classify)
        from tft_player import bench as _bench
        _bench.forget_empty()            # empty-slot pictures are this game's own (learn_bench_baseline)
        self.item_crops = [None] * N_SLOTS    # the item bench's icon crops (to learn a name read off a tooltip)
        self.pending_events = []         # events for the model, sent with the next update
        self.auto_opened = set()         # bench slots whose container the harness opened
        from tft_player.coach import auto_open_anvil
        self.auto_open = auto_open_anvil()   # True: containers are opened by the harness; False: the model decides
        self._auto_tried = set()         # (slot, stage) already tried
        self.news = []                   # changes, failed steps, mismatches: in the next model call
        self.container_opened = None     # (class, slot, time) of the last container we opened
        self.choice = None               # an open choice screen: {'asked', 'pick', 'cards', 'seen'}
        self.augment_clicked = 0.0
        self.augment_default = ('pick', 0)
        self.reroll_stage, self.rerolled = None, set()
        self.augment_note = None         # why the next augment question comes (the last answer could not run)
        self.aug_wait, self.aug_reasks = 0, 0    # coach.finished when the hand was asked; re-asks of the hand
        self.loot = LootRunner()
        self.chaser, self.own = CarouselChaser(), OwnTracker()
        self.bench_items = None
        self.prev_items = None
        self.items_done = set()
        self.tries = {}
        self.round_started = 0.0
        self.status_extra = {}
        self.last_frame = None
        self.tick = 0
        self.watcher = Watcher(capture)
        self.watcher.start()
        self.until = tuple(int(x) for x in until.split('-')) if until else None
        self.log_dir = log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_f = (log_dir / 'log.jsonl').open('a', encoding='utf-8')
        self.recorder = None
        # the whole game as a video: only with --video (the user, 2026-10-12: off - the carousel clips are what is
        # needed; the full video also cost capture frames)
        if not REPLAY and os.environ.get('TFT_VIDEO') == '1':
            try:
                from tft_player.game import ScreenRecorder
                self.recorder = ScreenRecorder(log_dir / 'game.mp4')
                atexit.register(self.stop_recording)
                print(f'recording: {log_dir / "game.mp4"}', flush=True)
            except Exception as e:
                print(f'RECORDING FAILED: {e!r}', flush=True)
                self.log(ev='recording_error', error=repr(e))
            try:                         # where the loop's time goes (tools/profile_report.py); --profile only: it
                # slowed the game it measured (run 20261010-235723: capture 15 fps, read p50 1.3 s)
                from tft_player.game import start_profiler
                if os.environ.get('TFT_PROFILE') == '1' and start_profiler(log_dir / 'profile.speedscope.json'):
                    print(f'profiling: {log_dir / "profile.speedscope.json"} (written when the game process ends)', flush=True)
            except Exception as e:
                print(f'PROFILER FAILED: {e!r}', flush=True)
                self.log(ev='profiler_error', error=repr(e))
        if self.forced:
            self.set_comp(self.forced.id, 'forced')     # logs, so after log_f
        from tft_player.coach import model_settings, opening_prompt
        model, effort = model_settings(kind)
        self.log(ev='model_session', backend=kind, model=model, effort=effort, system_prompt=opening_prompt())
        try:                             # which harness played: models are only comparable on the same code
            import subprocess
            root = Path(__file__).resolve().parent.parent
            git = lambda *a: subprocess.run(['git', *a], cwd=root, capture_output=True, text=True, timeout=10).stdout
            import hashlib
            # status lines are 'XY path': not stripped as a whole, or the first line loses its leading space and
            # its path its first letter (benchmark game 20261018-110145: "ools/benchmark.py")
            diff = git('diff', 'HEAD', '--', 'tft_player', 'tools')
            self.log(ev='run_config', git=git('rev-parse', '--short', 'HEAD').strip(),
                     dirty_files=[l[3:] for l in git('status', '--short', '--', 'tft_player', 'tools').splitlines()
                                  if l.strip()],
                     dirty_hash=hashlib.sha1(diff.encode('utf-8', 'replace')).hexdigest()[:10] if diff else None,
                     auto_open_anvil=self.auto_open, test_mode=test_mode(), comp=os.environ.get('TFT_FORCE_COMP'),
                     model=model, effort=effort)
        except Exception as e:
            self.log(ev='run_config_error', error=repr(e))
        if _cli()[0] and not self.pilot:     # after log_f: the answer is logged
            threading.Thread(target=self.send_briefing, args=(prep,), daemon=True).start()
        self.lat = []
        self.ending.start()

    def log(self, **ev):
        ev['t'] = round(ev.pop('at', None) or time.time(), 3)    # at: when it happened, if before it is logged
        self.log_f.write(json.dumps(ev, default=str) + '\n')
        self.log_f.flush()

    def combat_start(self, scr):
        """Combat started: the model is asked as soon as the shop has settled (send_combat_call), and
        the carousel prep (x-3) goes out now."""
        self.combat_call = (scr.stage, time.time())
        self.prepare_carousel(scr)
        self.bff_rides(scr)

    def bff_rides(self, scr):
        """The BFF (Sprykin 3) stands on the board as a unit of its own only until its first rider: a link from it
        to a Sprykin, or the game's own pick when combat starts. From then on it has no hex: it is under its rider,
        a champion on that champion's hex (the user, 2026-10-12). Its hex is kept: the game's pick shows there
        (run 20261012-071645 2-5: Teemo left (0, 3) and rode the BFF at (1, 4))."""
        bff = next((u for u in self.roster.units if u.name == BFF and not u.missing), None)
        if bff is None:
            return
        self.bff_hex = bff.loc
        self.roster.units.remove(bff)
        self.bff_assigned = True
        if not any(getattr(u, 'riding', None) for u in self.roster.units):
            self.news_event('bff', scr.stage, f'combat started: the game gave the BFF (at hex {bff.loc[1:]}) a rider - '
                                              f'one of the fielded Sprykin; the BFF has no hex of its own from now on')

    COMBAT_SHOP_WAIT = 2.0   # s: longest wait for a settled shop before the combat-start call goes anyway

    def send_combat_call(self, scr):
        """The combat-start call, sent on the first combat tick whose shop names every card: when combat
        starts each Wisp leaves the shop and the champion card under it is drawn, and a frame caught in
        between read a champion as a Wisp ('1' for Ivern, 'CV' for Kog'Maw in run 152203). Never blocks:
        until then the loop goes on; after COMBAT_SHOP_WAIT it is sent with whatever is read. The model
        hears what its plan did not get done, may order what works in combat (buy, sell bench units,
        reroll, XP) and writes the next round's plan."""
        stage, t0 = self.combat_call
        if self.phase.planning or (scr.stage and scr.stage != stage):
            self.combat_call = None      # the fight is over already
            return
        settled = scr.shop_visible and all(n or not r for n, r in zip(scr.shop, scr.shop_raw))
        if not settled and time.time() - t0 < self.COMBAT_SHOP_WAIT:
            return
        self.combat_call = None
        if not self.planner_on() or self.pilot or self.plan_pending:
            return                       # a question still running is answered in combat anyway
        rp = self.round_plan
        left = rp.leftover(self.roster)[1] if rp else ''
        orbs = len(getattr(self.watcher, 'orbs', None) or [])
        change = ('COMBAT STARTED (what the planning phase did not get done is listed below as "Not done")'
                  + (f"; {orbs} loot orb(s) still on the board, not collected" if orbs else ''))
        self.log(ev='combat_start_call', stage=scr.stage, left=left, orbs=orbs, settled=settled,
                 waited=round(time.time() - t0, 2))
        self.request_plan(scr, self.last_frame, 'update', change, interrupt=False)

    def prepare_carousel(self, scr):
        """The fight before a carousel (x-3): the model ranks every item it wants, in the background, so the
        chase starts on its choice the moment the carousel opens (the user: never ask once it is up)."""
        rnd = scr.round or (0, 0)
        if rnd[0] < 2 or rnd[1] != 3 or getattr(self, '_prepared_for', None) == (rnd[0], 4):
            return
        from tft_player import carousel as CR
        from tft_player.coach import ask_model
        self._prepared_for = (rnd[0], 4)
        target_stage = f'{rnd[0]}-4'
        targets = list(self.brain.plan.item_targets or [])
        state = (f"HP {getattr(self, '_hp', None)}, {self.roster!r}, item bench: "
                 f"{[i for i in (self.bench_items or []) if i] or 'empty'}")
        prompt = CR.prep_prompt(state)
        image = self.screen_jpeg()
        self.log(ev='carousel_prep_asked', stage=scr.stage, for_stage=target_stage)

        def run():
            t0 = time.time()
            got = {}

            def first(obj):
                r = CR.parse_prep(obj)
                if r and not got:
                    got['ranked'], got['latency'] = r, time.time() - t0
                    self.carousel_prep = {'stage': target_stage, 'ranked': r}
                    self.brain.plan.item_targets = r          # the chase's own wants follow it at once
                    self.log(ev='item_targets', stage=scr.stage, targets=r, asked='carousel_prep')
            try:
                ans = ask_model(prompt, image, on_first=first, timeout=60)
                if not got:
                    from tft_player.coach import json_objects
                    for obj in json_objects(ans or ''):
                        first(obj)
                self.model_call('carousel_prep', prompt, ans, time.time() - t0, got.get('latency'))
            except Exception as e:
                self.log(ev='carousel_prep_error', error=repr(e))
        threading.Thread(target=run, daemon=True).start()

    def combat_decide(self, scr):
        """The fight is running: the board is locked, but buying, selling bench units, rerolling and XP
        work - the plan's steps of those kinds go on now (the user: they still work in combat); board
        moves wait for the next planning phase. Wisps leave the shop when combat starts. In the countdown before
        the fight (5-4-3-2-1) the units stand still: the plan's items not given yet go first (in an away fight the
        units leave at 2; a drop after that lands on no unit of ours and the item returns to the bench)."""
        rp = self.round_plan
        if rp is not None and self.phase.countdown and scr.shop_visible and (scr.timer or 0) >= 2:
            act = rp.countdown_action(scr, self.roster)
            if act:
                self.brain.status['plan'] = f'COUNTDOWN ACT {act}'
                return act
        act = self.after_fight_orders(scr)
        if act:
            return act
        if not scr.shop_visible or not scr.stage or scr.augments:
            return None
        act = self.auto_action(scr)
        if act:
            return act
        if rp is None:
            return None
        if self.pending_events and not self.plan_pending and self.planner_on() and not self.pilot                 and getattr(self, 'combat_call', None) is None:
            # an event in combat wakes the model too (what works in combat is done at once)
            ev = '; '.join(self.pending_events)
            self.log(ev='event_sent', stage=scr.stage, what=ev)
            self.request_plan(scr, self.last_frame, 'update', ev, interrupt=False)
        if scr.level:
            self.last_level = scr.level
        rp.hold_roll = False                     # no Wisps in combat
        act = rp.next_action(scr, self.roster, self.team_cap(scr), combat=True)
        if act:
            self.brain.status['plan'] = f'COMBAT ACT {act}'
            return act
        if not self.phase.countdown and (scr.timer or 0) >= 3:
            act = rp.fight_give(scr, self.roster)
            if act:
                self.brain.status['plan'] = f'FIGHT GIVE {act}'
                return act
        if rp.failed and not rp.carry:
            self.plan_idle(rp, scr, combat=True)
        return None

    def fight_result(self, scr):
        """'win' / 'lose' once this fight's result shows, else None: the streak flame's number rises or
        its colour flips (orange = win, blue = loss), or our HP drops. Read at most every 0.4 s."""
        base = getattr(self, 'fight_base', None)
        if not base or base['stage'] != scr.stage or self.last_frame is None:
            return None
        now = time.time()
        if now - base.get('checked', 0) < 0.4:
            return base.get('result')
        base['checked'] = now
        if base.get('result'):
            return base['result']
        st, st0, hp0 = self.reader.streak(self.last_frame), base['streak'], base['hp']
        res = None
        if scr.hp is not None and hp0 is not None and scr.hp < hp0:
            res = 'lose'
        elif st and st[1] in ('win', 'loss') and (st0 is None or st0[1] != st[1] or st[0] > st0[0]):
            res = 'win' if st[1] == 'win' else 'lose'
        elif self.win_gold(scr, base):
            res = 'win'
        if res:
            base['result'] = res
            self.log(ev='fight_result', stage=scr.stage, result=res, streak=st, streak0=st0, hp=scr.hp, hp0=hp0)
        return res

    def win_gold(self, scr, base):
        """A player fight won: the gold rises by exactly 1 with no sell or buy of ours (the win gold),
        read twice in a row. The streak flame misses a win that ends a loss streak."""
        from tft_player.planner import is_pvp
        g = scr.gold
        if g is None or not is_pvp(scr.stage):
            return False
        if time.time() - getattr(self, '_acted_at', 0) < 1.5:
            base['gold_ref'], base['gold_up'] = None, 0       # our own sell/buy: a new baseline after it
            return False
        ref = base.get('gold_ref')
        if ref is None or (g != ref and g != ref + 1):
            base['gold_ref'], base['gold_up'] = g, 0
            return False
        base['gold_up'] = base.get('gold_up', 0) + 1 if g == ref + 1 else 0
        return base['gold_up'] >= 2

    def after_fight_orders(self, scr):
        """The model's orders for after this fight ({"win": {"sell": [ids], "buy": [names]}, "lose": ...}),
        one step per tick, the moment the result shows: interest is paid on the gold held when the next
        round starts, so a sell or buy made now counts. Sells (bench units) first, then buys from the
        shop on screen, each card once; what could not be done is told with the reason."""
        af = getattr(self, 'after_fight', None)
        if not af or af['stage'] != scr.stage or af.get('done'):
            return None
        res = self.fight_result(scr)
        if not res:
            return None
        o = af.get(res)
        o = o if isinstance(o, dict) else {'sell': list(o or []), 'buy': []}      # the older form: sells only
        sold, said = af.setdefault('sold', set()), af.setdefault('names', [])
        todo = [self.roster.resolve(i) for i in o.get('sell') or [] if isinstance(i, int)]
        todo = [u for u in todo if u is not None and u.loc[0] == 'bench' and not u.missing and u.id not in sold]
        if todo:
            u = todo[0]
            sold.add(u.id)
            said.append(f'sold {u}')
            return Action('sell', u, f'after-fight sell ({res}) {u}')
        if time.time() - getattr(self, '_acted_at', 0) < 0.6:
            return None                          # the gold of the last sell/buy is read first
        k = af.setdefault('bought', 0)
        buys = list(o.get('buy') or [])
        if k < len(buys) and scr.shop_visible and scr.gold is not None:
            af['bought'] = k + 1
            name = buys[k]
            slot = next((i for i, n in enumerate(scr.shop) if n == name and i not in af.setdefault('slots', set())), None)
            info = CATALOG.unit(name)
            why = ('not in the shop' if slot is None else f'{info.cost}g, gold {scr.gold}' if info.cost > scr.gold
                   else 'no bench slot' if not self.roster.free_bench() and len(self.roster.copies(name, 1)) < 2
                   else None)
            if why:
                said.append(f'not bought {name} ({why})')
                return None
            af['slots'].add(slot)
            said.append(f'bought {name}')
            return Action('buy', slot, f'after-fight buy ({res}) {name}')
        if k < len(buys):
            return None                          # the shop or gold not read on this tick
        af['done'] = res
        self.news_event('after_fight', scr.stage, f'fight result: {res}; after-fight orders: '
                                                  f'{"; ".join(said) or "none"}')
        return None

    # -- standing orders: auto-buy / auto-sell -------------------------------------------------
    def auto_need(self, name, target=None):
        """Copies of `name` still to buy for a star target (default: its auto_buy one), counted over board
        and bench: 3 copies make a 2-star, 9 a 3-star - a 1-star on the board and two on the bench are
        already a 2-star's worth, whether or not the merge was seen."""
        t = target or self.auto_buy.get(name, 0)
        if not t:
            return 0
        have = sum(3 ** (u.star - 1) for u in self.roster.units if u.name == name and not u.missing)
        have += sum(1 for n, _ in self.roster.pending_buys if n == name)
        return max(0, 3 ** (t - 1) - have)

    def auto_sell_unit(self):
        """The next bench unit auto-sell sells, or None. auto_sell True: a champion not on auto_buy, or a
        spare copy beyond its star target; a list: only those champions. Never a board unit, a unit the
        plan fields or keeps, or one whose identity is in doubt; items come back only to free bench slots."""
        if not self.auto_sell:
            return None
        rp = self.round_plan
        keep = (rp.protected_ids() if rp else set()) | set(getattr(self.brain, 'protected', None) or ())
        free_items = sum(1 for i in self.bench_items if not i) if self.bench_items is not None else 0
        for u in self.roster.bench():
            if u.missing or u.unverified or not u.name or u.id in keep or len(u.items) > free_items:
                continue
            if isinstance(self.auto_sell, list):
                if u.name in self.auto_sell:
                    return u
                continue
            t = self.auto_buy.get(u.name, 0)
            if not t:
                return u
            same = [x for x in self.roster.units if x.name == u.name and not x.missing and x is not u]
            if any(getattr(x, 'star_assumed', False) for x in same):
                continue                         # a merge not yet seen on screen: its copies may still be needed
            others = sum(3 ** (x.star - 1) for x in same)
            if others >= 3 ** (t - 1):
                return u                         # the target is met without it
        return None

    def auto_action(self, scr):
        """One auto-sell or auto-buy step on this shop, or None. No gold reserve, no question: the model set
        these orders and changes them in any reply. Each (round, slot, champion) is tried twice at most."""
        if not scr.shop_visible or scr.gold is None or scr.augments or REPLAY:
            return None
        if scr.timer is not None and scr.timer <= 1:
            return None                          # the phase is switching (run 033507 2-2: a buy at 0 s never landed)
        u = self.auto_sell_unit()
        if u is not None:
            key = (scr.stage, 'sell', u.id)
            if self.auto_tries.get(key, 0) < 2:
                self.auto_tries[key] = self.auto_tries.get(key, 0) + 1
                return Action('sell', u, f'auto-sell {u}')
        for i, name in enumerate(scr.shop):
            info = CATALOG.unit(name) if name else None
            if info is None or info.cost > scr.gold or self.auto_need(name) <= 0:
                continue
            if not self.roster.free_bench() and len(self.roster.copies(name, 1)) < 2:
                # no room (a 3rd copy would merge and needs none) - said once per stage and champion: runs
                # 20261018-092528 and -102854 kept a full bench for rounds while their comp units passed in the shop
                said = self.__dict__.setdefault('auto_blocked', set())
                if (scr.stage, name) not in said:
                    said.add((scr.stage, name))
                    self.news_event('auto_buy_blocked', scr.stage, f'auto_buy did not buy {name} (shop slot {i}): '
                                                                  f'the bench is full')
                continue
            key = (scr.stage, i, name)
            if self.auto_tries.get(key, 0) >= 2:
                continue
            self.auto_tries[key] = self.auto_tries.get(key, 0) + 1
            return Action('buy', i, f'auto-buy {name}')
        return None

    def apply_auto(self, plan, stage):
        """execution_plan.auto_buy / auto_sell from the model: merged into the standing orders; its
        after_fight_sell is armed for this stage's fight."""
        af = plan.get('after_fight')
        if not isinstance(af, dict) and isinstance(plan.get('after_fight_sell'), dict):
            af = {k: {'sell': v} for k, v in plan['after_fight_sell'].items()}
        if isinstance(af, dict):
            orders = lambda o: ({'sell': [i for i in o.get('sell') or [] if isinstance(i, int)],
                                 'buy': [str(n) for n in o.get('buy') or [] if CATALOG.unit(str(n))]}
                                if isinstance(o, dict) else {'sell': [i for i in o or [] if isinstance(i, int)], 'buy': []})
            self.after_fight = {'stage': stage, 'win': orders(af.get('win')), 'lose': orders(af.get('lose'))}
            self.log(ev='after_fight_armed', stage=stage, win=self.after_fight['win'], lose=self.after_fight['lose'])
        ab, as_ = plan.get('auto_buy'), plan.get('auto_sell')
        bad = []
        if isinstance(ab, dict):
            for n, star in ab.items():
                info = CATALOG.unit(str(n))
                if not info or not isinstance(star, int) or isinstance(star, bool) or not 0 <= star <= 3:
                    bad.append(f'auto_buy {n!r}: {star!r} (a champion name -> star 0-3)')
                elif star == 0:
                    self.auto_buy.pop(info.name, None)
                else:
                    self.auto_buy[info.name] = star
        elif ab is not None:
            bad.append('auto_buy must be {"Champion": star, ...}')
        if isinstance(as_, bool):
            self.auto_sell = as_
        elif isinstance(as_, list):
            names = [CATALOG.unit(str(n)).name for n in as_ if CATALOG.unit(str(n))]
            bad += [f'auto_sell: no champion named {n!r}' for n in as_ if not CATALOG.unit(str(n))]
            self.auto_sell = names or False
        elif as_ is not None:
            bad.append('auto_sell must be true, false or ["Champion", ...]')
        ho = plan.get('item_final')
        if isinstance(ho, list):
            for o in ho:
                uid, item = (o.get('unit'), o.get('item')) if isinstance(o, dict) else (None, None)
                final = o.get('final') if isinstance(o, dict) else None
                info = CATALOG.unit(str(final)) if final else None
                if not isinstance(uid, int) or not item or (final and not info):
                    bad.append(f'item_final {o!r}: needs "unit" (an id), "item" and "final" (a champion, or null)')
                    continue
                self.set_final(uid, str(item), info.name if info else None, o.get('star'))
        elif ho is not None:
            bad.append('item_final must be a list of {"unit", "item", "final", "star"}')
        for e in plan.get('items') or []:
            if isinstance(e, dict) and e.get('final') and isinstance(e.get('id'), int) and CATALOG.unit(str(e['final'])):
                self.give_finals[(e['id'], str(e.get('component')))] = (CATALOG.unit(str(e['final'])).name,
                                                                      e.get('final_star'))
        if ab is not None or as_ is not None or ho is not None:
            self.log(ev='auto_orders', stage=stage, auto_buy=self.auto_buy, auto_sell=self.auto_sell,
                     item_finals=self.item_finals)
        if bad:
            self.news_event('auto_rejected', stage, '; '.join(bad))

    def set_final(self, uid, item, final, star=None):
        """The model's final holder for an item a unit holds (None clears it)."""
        self.item_finals = [r for r in self.item_finals if not (r['holder'] == uid and r['item'] == item)]
        if final:
            self.item_finals.append({'item': item, 'holder': uid, 'final': final,
                                     'star': star if isinstance(star, int) and 1 <= star <= 3 else 1})

    def note_final(self, unit, given, made):
        """A give landed on `unit`: `given` from the item bench, making `made` (the same item, or the item a component
        completed). The item records follow it: an item whose holder no longer has it (taken off by a Remover, or its
        holder sold) has a new current holder now, with the same final holder; a component this unit held is part of
        `made` now; the final holder the model named with this give goes onto `made`. An item at its final holder is
        where it was meant to be: its record ends."""
        if not hasattr(self, 'item_finals'):
            return
        f = self.give_finals.pop((unit.id, given), None)
        held = unit.items or []
        for r in self.item_finals:
            h = self.roster.resolve(r['holder'])
            if r['holder'] != unit.id and r['item'] in (given, made) and (h is None or r['item'] not in (h.items or [])):
                r['holder'] = unit.id            # moved here: a transfer, or a sold holder's item given out again
                f = f or (r['final'], r['star'])
                r['item'] = made
            elif r['holder'] == unit.id and r['item'] != made and r['item'] not in held:
                f = f or (r['final'], r['star'])                 # the component it held is part of `made` now
        self.item_finals = [r for r in self.item_finals if r['holder'] != unit.id or r['item'] in held]
        if f:
            self.set_final(unit.id, made, f[0], f[1])
        self.item_finals = [r for r in self.item_finals
                            if getattr(self.roster.resolve(r['holder']), 'name', None) != r['final']]

    def comp_units(self):
        """The comp's final board, by champion name (the fixed comp, else the target comp), or None."""
        from tft_player.meta import META
        c = self.forced or (META.by_name(self.target) if getattr(self, 'target', None) else None)
        return set(c.units) if c else None

    def handover_action(self, scr):
        """Items to their final holders (the model's item_final / a give's "final"): when we own an item's final holder
        at the star the model set (name read off its panel), in a planning phase, the item moves with no question to
        the model. Its holder outside the comp's final board is sold (its items return to the item bench), the final
        holder takes its hex if it was fielded, and the item is given; a holder in the final board keeps its place - an
        Item Remover takes its items off, the item goes to the final holder, the others back onto the holder. One
        holder at a time, as a small plan of its own ahead of the round plan (set aside meanwhile, then resumed).
        -> the next action, or None."""
        from tft_player import planner
        hp = self.handover_rp
        if hp is not None:
            if self.round_plan is not hp:        # a new plan from the model replaced it
                self.handover_rp = None
                return None
            act = hp.next_action(scr, self.roster, self.team_cap(scr))
            if act:
                self.brain.status['plan'] = f'TO FINAL HOLDER {act}'
                return act
            moved, to = self._handover
            self.round_plan, self.handover_rp = self._handover_saved, None
            if not hp.failed:
                self.item_finals = [r for r in self.item_finals if r not in moved]
            what = (f"items to their final holder {to}: {', '.join(r['item'] for r in moved)} - done: "
                    f"{', '.join(hp.done) or 'nothing'}" + (f"; not done: {', '.join(hp.failed)}" if hp.failed else ''))
            self.news_event('final_holder', scr.stage, what)
            self.log(ev='final_holder_done', stage=scr.stage, moved=moved, done=hp.done, failed=hp.failed)
            return None
        if not self.item_finals or scr.timer is None or scr.timer <= 3 or self.plan_pending:
            return None
        tried = self.__dict__.setdefault('handover_tried', set())
        comp = self.comp_units()
        for r in self.item_finals:
            to = next((x for x in self.roster.units if x.name == r['final'] and x.star >= (r.get('star') or 1)
                       and not x.missing and not x.unverified), None)
            holder = self.roster.resolve(r['holder'])
            if to is None or holder is None or holder is to or (scr.stage, holder.id, to.id) in tried:
                continue
            if r['item'] not in (holder.items or []):
                continue                         # not where the record says: the state shows it, the model decides
            moved = [x for x in self.item_finals if self.roster.resolve(x['holder']) is holder and x['final'] == to.name
                     and x['item'] in (holder.items or [])]
            items = [x['item'] for x in moved]
            ep = {}
            if comp is not None and holder.name not in comp:
                ep['sell'] = [holder.id]
                if holder.loc[0] == 'board' and to.loc[0] != 'board':
                    ep['board'] = [{'unit': to.id, 'hex': [holder.loc[1], holder.loc[2]]}]
                ep['items'] = [{'give': i, 'to': to.id, 'final': to.name} for i in items]
                how = f'{holder} is not in the final board: sold'
            else:
                rest = list(holder.items or [])
                for i in items:
                    rest.remove(i)
                ep['items'] = ([{'transfer': i, 'from': holder.id, 'to': to.id} for i in items]
                               + [{'transfer': i, 'from': holder.id, 'to': holder.id} for i in rest])
                how = f'{holder} is in the final board: Item Remover'
            tried.add((scr.stage, holder.id, to.id))
            plan = planner.normalize({'execution_plan': ep})
            plan.pop('strategy', None)
            rp = planner.RoundPlan(plan, scr.stage, known=[x.id for x in self.roster.units], names=self.unit_names(),
                                   gold=scr.gold)
            # a sold holder's items reach the item bench only by the sell: not checked against the bench before it
            bad = planner.validate(rp.p, self.roster, None if 'sell' in ep else self.bench_items, rp.names)
            if bad:
                self.news_event('final_holder', scr.stage, f"items to {to.name} ({', '.join(items)} on {holder}) could not "
                                                           f"move: {'; '.join(bad)}")
                continue
            self._handover_saved, self._handover = self.round_plan, (moved, to.name)
            self.round_plan = self.handover_rp = rp
            self.log(ev='final_holder_started', stage=scr.stage, items=items, holder=repr(holder), to=repr(to), how=how,
                     plan=ep)
            print(f"{scr.stage}: {', '.join(items)} -> final holder {to} ({how})")
            return self.handover_action(scr)
        return None

    def auto_line(self):
        """The standing orders and what they did since the model's last decision, for the state."""
        buy = ', '.join(f'{n} to {s}* (' + (f'{self.auto_need(n)} more to buy' if self.auto_need(n) else 'done') + ')'
                        for n, s in sorted(self.auto_buy.items())) or 'nothing listed'
        sell = ('off' if not self.auto_sell else 'on - champions not on auto_buy and spare copies'
                if self.auto_sell is True else f"on - {', '.join(self.auto_sell)}")
        done, self.auto_done = self.auto_done[-12:], []
        ho = '; '.join(f"{r['item']} (holder: unit {r['holder']}, final holder: {r['final']}"
                       + (f" at {r['star']}*" if (r.get('star') or 1) > 1 else '') + ')'
                       for r in getattr(self, 'item_finals', [])) or 'none set'
        return (f'Standing orders (all game; change them in any reply with auto_buy / auto_sell / item_final): '
                f'Auto-buy: {buy}. Auto-sell: {sell}. Item final holders: {ho}.'
                + (f" Done by them since your last decision: {'; '.join(done)}." if done else ''))

    BURST = 50                                # frames per place (the user: 50 bench + 50 board)

    BENCH_BAR_Y = 700                         # where a bench unit's health bar sits (1080p): the crop
                                              # anchor when no bar is drawn (bench units in combat)

    def bench_pixels(self, frame):
        from tft_player import bench
        band = bench._bench_band(frame)
        return [bench.slot_crop(band, i).astype(np.int16) for i in range(9)]

    def photograph_buy(self, u):
        """Data games: a bought unit, labelled with the shop card's name, photographed 50 times on the
        bench, then sold (data_decide). Where it landed: the ONE bench slot whose pixels changed since
        just before the buy (bench units show no health bar in combat, so bars can't tell). Not
        exactly one changed slot: nothing is saved."""
        from tft_player import bench as BN
        before = getattr(self, '_bench_before', None)
        kinds = getattr(self, '_bench_before_kinds', None) or []
        slot = None
        empties = [i for i, k in enumerate(kinds) if k == 'empty']
        if empties:
            # the game's rule: a bought unit lands in the leftmost empty slot - confirmed when that
            # slot no longer reads empty (recorded buys: 60/90; the rest merged or did not land)
            i = empties[0]
            for _ in range(15):
                time.sleep(0.07)
                got = self.cap.read()
                if got is None:
                    continue
                BN._memo.clear()
                if BN.classify(got[0], B.occupancy(got[0])[0])[i]['kind'] != 'empty':
                    slot = i
                    break
            before = None                         # no fallback guess when the rule says nothing landed
        if before is not None:
            # no empty tile recognisable (an opponent's arena in combat): exactly one slot must change
            for _ in range(15):
                time.sleep(0.07)
                got = self.cap.read()
                if got is None:
                    continue
                now = self.bench_pixels(got[0])
                change = [float((np.abs(a - b).max(axis=2) > 40).mean()) for a, b in zip(before, now)]
                moved = [i for i, c in enumerate(change) if c > 0.15]
                if len(moved) == 1:
                    slot = moved[0]
                    break
        if slot is None:
            self.log(ev='look_not_taken', unit=repr(u), why='not exactly one bench slot changed after the buy')
            return
        if u.loc != ('bench', slot):
            other = self.roster.at(('bench', slot))
            if other is not None and other is not u:
                other.loc = u.loc
            u.loc = ('bench', slot)
        time.sleep(0.3)                           # the unit drops in
        from tft_player import champs
        bench_n = self.burst(u, self.BURST, 2.5)
        champs.add_progress(u.name, bench_n, 0)
        u.labelled = True
        u.photographed = True                     # bought and photographed by us: sold next (data_decide)
        self.log(ev='champ_labelled', how=f'buy: {bench_n} bench frames', unit=repr(u))
        print(f'data: {u.name} photographed - {bench_n} bench frames')

    def data_board_full(self):
        return self.roster.slots_used() >= self.team_cap()

    def move_for_photo(self, u, dst):
        src = u.loc
        try:
            self.inp.drag(loc_point(src), loc_point(dst))
        except NotForeground:
            return False
        if self.move_took(src, dst):
            self.roster.moved(u, dst)
            time.sleep(0.4)                       # the unit walks/drops into place
            return True
        return False

    def burst(self, u, frames, seconds):
        """Save up to `frames` crops of u at its place over `seconds` (each frame must show its bar)."""
        from tft_player import champs
        saved, end = 0, time.time() + seconds + 1.5     # a little slack: 50 frames are the goal
        step = seconds / frames
        while saved < frames and time.time() < end:
            t0 = time.time()
            got = self.cap.read()
            if got is not None:
                bar = self.unit_bar(got[0], u.loc)
                if bar is None and u.loc[0] == 'bench':
                    bar = (B.BENCH[u.loc[1]][0], self.BENCH_BAR_Y)    # combat: no bar drawn on the bench
                crop = champs.unit_crop(got[0], bar) if bar is not None else None
                if crop is not None:
                    champs.add_to_library(crop, u.name, u.star, burst=True, flush=False)
                    saved += 1
            time.sleep(max(0.0, step - (time.time() - t0)))
        champs.flush_labels()
        return saved

    def data_decide(self, scr, planning):
        """Data-collection games (the user, 2026-10-05): a checklist of every champion; buy only
        champions not yet crossed off (50 bench + 50 board frames), photograph, sell. Gold goes to
        levels; rerolls only from level 8. Fielding keeps one board hex free for the board photos.
        No model calls."""
        from tft_player import champs
        if not scr.shop_visible or scr.gold is None or scr.augments:
            return None
        if scr.level:
            self.last_level = scr.level
        # only what we bought and photographed is sold - never the player's own units
        # (bench or board: the board photo leaves it on the board)
        done = [u for u in self.roster.units if getattr(u, 'photographed', False) and not u.missing
                and (u.loc[0] == 'bench' or planning)]
        if done:
            return Action('sell', done[0], f'data: sell {done[0]} (photographed)')
        if not DATA_ROLL and planning and len(self.roster.board()) < max(1, (scr.level or 1) - 1):
            a = self.brain.decide(scr, self.roster, planning)
            if a and a.kind == 'move' and a.arg[1][0] == 'board':
                return a                         # field units, leaving one hex free
        gold = scr.gold
        todo = champs.still_needed()
        cards = [(i, n) for i, n in enumerate(scr.shop)
                 if n in todo and CATALOG.unit(n) and CATALOG.unit(n).cost <= gold]
        if cards and self.roster.free_bench():
            i, n = cards[0]
            return Action('buy', i, f'data: buy {n} (not yet on the list: {len(todo)} champions left)')
        if (scr.level or 0) < 8 and gold >= 4:
            return Action('level', 1, 'data: level up (rerolls only from level 8)')
        if ((scr.level or 0) >= 8 or DATA_ROLL) and gold >= 2 and planning:
            return Action('roll', None, f'data: reroll for champions still needed ({len(todo)} left)')
        return None

    def stage1_buy(self, scr):
        """Stage 1 fallback: the model decides stage-1 buys in its plan (the user, 2026-10-09: its answer comes
        2-9 s into a 14-19 s phase). Only when no plan for this round has come with 3 s left: buy every champion
        we can afford - no interest to lose below 10 gold (run23 bought nothing at 1-3 with 5 gold and 5 cards
        while the plan was late)."""
        if not scr.round or scr.round[0] != 1 or scr.gold is None or not scr.shop_visible:
            return None
        rp = self.round_plan
        if not self.planner_on() or (rp is not None and rp.stage == scr.stage and not rp.carry)                 or scr.timer is None or scr.timer > 3:
            return None
        done = self.__dict__.setdefault('_stage1_bought', set())
        for i, name in enumerate(scr.shop):
            info = CATALOG.unit(name) if name else None
            if not info or (scr.stage, i) in done or info.cost > scr.gold:
                continue
            if not self.roster.free_bench() and len(self.roster.copies(name, 1)) < 2:
                continue
            if (scr.gold - info.cost) // 10 < scr.gold // 10:
                continue                         # would cost an interest step
            done.add((scr.stage, i))
            return Action('buy', i, f'stage 1 fallback: buy {name} (no plan yet; every affordable card)')
        return None

    @staticmethod
    def effect_text(name):
        """The game's description of a Wisp or augment (data/reference wisps.md / augments.md)."""
        import re as _re
        for f, col in (('wisps.md', 1), ('augments.md', 3)):
            path = Path(__file__).resolve().parent.parent / 'data' / 'reference' / f
            try:
                for row in path.read_text(encoding='utf-8').splitlines():
                    cells = [c.strip() for c in row.strip().strip('|').split('|')]
                    if len(cells) > col and cells[0].lower() == (name or '').strip().lower():
                        return cells[col]
            except OSError:
                pass
        return ''

    def unit_change(self, name, stage):
        """A Wisp/augment that TRANSFORMS units we own (Polymorph: a random 2-cost becomes a
        3-cost, in the same slot - no new bar shows it): the units it can affect are read once,
        in idle time. Units it GIVES appear in empty slots and are read as arrivals anyway."""
        import re as _re
        desc = self.effect_text(name)
        if not _re.search(r'transform', desc, _re.I) or _re.search(r'items? transform', desc, _re.I):
            return
        costs = {int(c) for c in _re.findall(r'(\d)-cost(?= you own)', desc)}   # transformed, not the result
        if 'rightmost bench' in desc:
            slots = sorted((u.loc[1] for u in self.roster.bench()), reverse=True)[:3]
            hit = [u for u in self.roster.bench() if u.loc[1] in slots]
        else:
            hit = [u for u in self.roster.units if not costs or (u.info and u.info.cost in costs)]
        self.reread = getattr(self, 'reread', set()) | {u.id for u in hit}
        self.log(ev='unit_change_event', stage=stage, source=name, reread=[repr(u) for u in hit])
        if _re.search(r'start of every round|each round|each turn', desc, _re.I):
            self.recurring_changes = getattr(self, 'recurring_changes', set()) | {name}

    def confirm_merge(self, merge, scr):
        """A buy completed 3 copies: the game merged them into one unit. Where it stands is read
        from the screen, not assumed: of the copies' places, the one still holding a unit is it
        (bench slots by their bar; a fielded copy's hex by a bar near it)."""
        from tft_player.state import _hex_near
        time.sleep(0.35)                         # the merge animation
        got = self.cap.read()
        if got is None:
            return
        bench, board = B.occupancy(got[0], exclude=self.watcher.tac[1])
        still = [loc for loc in merge['locs'] if (loc[0] == 'bench' and bench[loc[1]]) or
                 (loc[0] == 'board' and any(_hex_near(loc[1:], h) for h in board))]
        u = merge['unit']
        if len(still) == 1 and still[0] != u.loc and self.roster.at(still[0]) in (None, u):
            self.log(ev='merge_place', stage=scr.stage, unit=repr(u), assumed=u.loc, seen=still[0])
            u.loc = still[0]
        elif len(still) != 1:
            self.reread = getattr(self, 'reread', set()) | {u.id}       # unclear: read it once
            self.log(ev='merge_unclear', stage=scr.stage, unit=repr(u), still=still)

    def plan_rejected(self, errs, stage, what=''):
        """Entries of the model's plan that were invalid and dropped: they go on the plan's failed
        list (so the model is asked again at once, with the reason) and into the news."""
        if not errs:
            return
        self.log(ev='plan_rejected', stage=stage, errors=errs)
        print(f'{stage}: {what}plan entries rejected: {"; ".join(errs)}')
        self.alert('PLAN', f'{what}entry rejected: {errs[0][:90]}' + (f' (+{len(errs) - 1})' if len(errs) > 1 else ''),
                   'model asked again')
        if self.round_plan is not None and not what:
            self.round_plan.failed += [f'rejected {e}' for e in errs]
        self.news_event('plan_rejected', stage, f'{what}plan entries rejected: ' + '; '.join(errs))

    def unit_names(self):
        """id -> (name, star) of every tracked unit: a plan keeps it to find its units again."""
        return {u.id: (u.name, u.star) for u in self.roster.units}

    def plan_idle(self, rp, scr, combat=False):
        """The plan has steps left but none can be taken now: say why, once per change (run23:
        the controller stopped silently at 2-2 and 3-1 with a level-up and a unit still to do)."""
        carry, left = rp.leftover(self.roster)
        if not left:
            return
        missing = [uid for uid in {e.get('id') for e in (rp.p.get('board') or []) + (rp.p.get('items') or [])
                                   if isinstance(e, dict) and isinstance(e.get('id'), int)}
                   if rp.unit(self.roster, uid) is None]
        why = (f"left: {left}; gold read {scr.gold} (last {rp.last_gold}), level read {scr.level} "
               f"(using {self.last_level}), board {len(self.roster.board())} units"
               + (f"; units of the plan not found: {missing}" if missing else ''))
        if why != rp.idle_said:
            rp.idle_said = why
            self.log(ev='plan_idle', stage=scr.stage, timer=scr.timer, why=why)
            print(f'{scr.stage}: PLAN IDLE - {why}')
            self.news_event('plan_idle', scr.stage, f'your plan could not be finished: {left}'
                            + (f' (units not found: {missing})' if missing else ''))
            # tell the model now, while the round runs, so it can correct the plan (sell before a
            # buy, another hex...) - not only at the next planning phase
            # at any time left, and in combat too: what still works there (buy, sell bench units, XP,
            # reroll) is done at once, the rest is kept for the planning phase (the user)
            if ((rp.failed or missing) and not self.plan_pending
                    and not getattr(rp, 'feedback_sent', False)):
                rp.feedback_sent = True
                self.log(ev='feedback_sent', stage=scr.stage, why=why)
                print(f'{scr.stage}: asking the model to correct the plan ({"combat" if combat else f"{scr.timer}s left"})')
                self.request_plan(scr, self.last_frame, 'update',            # the steps: in "failed" / "Not done"
                                  'the controller could not finish your plan'
                                  + (f' (units not found: {missing})' if missing else ''), interrupt=False)

    def input_failed(self, act, scr, before):
        """A failed action: what the input conditions were (foreground, keyboard layout, cursor and
        the window under it) and the screen just before and after, at full size - to find why
        inputs get lost (run20: 26 of 39 actions failed from 2-5 on; runs before: ~1%)."""
        if REPLAY or not hasattr(self.inp, 'diagnose'):
            return
        n = getattr(self, '_fail_n', 0) + 1
        self._fail_n = n
        diag = self.inp.diagnose()
        if re.search(r'no unit is seen at|still on the item bench after|left the item bench, but|did not move|'
                     r'not where the drop went|stands at',
                     self.fail_cause or ''):
            # failures that point at a wrong record of where our units stand. One can be a plain miss; two in a
            # round, or in two rounds running, is consistent: read the units off the screen (board_check)
            now = time.time()
            fails = [f for f in getattr(self, '_pos_fails', []) if now - f[1] < 90] + [(scr.stage, now)]
            self._pos_fails = fails[-4:]
            if len(fails) >= 2:                      # within ~90 s: this round or the one before
                self.need_survey = (f'consistent failures ({len(fails)} lately, last: '
                                    f'{(self.fail_cause or "")[:80]})')
                self._pos_fails = []
        self.log(ev='input_failed', stage=scr.stage, timer=scr.timer, act=repr(act), cause=self.fail_cause,
                 diag=diag, n=n)
        print(f'INPUT: {act.kind} failed ({self.fail_cause}) - {diag}')
        self.alert('FAILED', f'{act.kind}: {(self.fail_cause or "no effect seen")[:90]}', 'Execution')
        if n <= 40 and shots_on():        # full-size pairs: ~1 MB each
            from PIL import Image
            Image.fromarray(before).save(self.log_dir / f'fail{n:02d}-before.jpg', quality=90)
            got = self.cap.read()
            if got is not None:
                raw = getattr(self.cap, 'raw', None)
                Image.fromarray(raw if raw is not None else got[0]).save(
                    self.log_dir / f'fail{n:02d}-after.full.jpg', quality=90)

    def snap(self, frame, name, full=False):
        if not shots_on():
            return
        from PIL import Image
        im = Image.fromarray(frame)
        if full:            # native resolution (evaluation snapshots, debugging misreads)
            raw = getattr(self.cap, 'raw', None)
            (Image.fromarray(raw) if raw is not None else im).save(self.log_dir / f'{name}.full.jpg', quality=92)
        im.resize((960, 540)).save(self.log_dir / f'{name}.jpg', quality=70)

    def read(self):
        got = self.cap.read()
        if got is None:
            return None, None
        frame, t_cap = got
        self._t_cap = t_cap
        last = getattr(self, '_full_scr', None)
        now = time.time()
        if last is not None and is_carousel(last) and not last.shop_visible and now - last.t < 2.5:
            # the carousel: the chase needs fresh frames, not the HUD. A full read there OCRs the hidden
            # HUD's boxes over a turning scene every tick (0.3-0.8 s: run 213127's chase ticked every
            # 0.35-0.74 s and lost its target); between full reads the timer counts down by the clock
            import copy
            scr = copy.copy(last)
            if last.timer is not None:
                scr.timer = max(0, round(last.timer - (now - last.t)))
            return frame, scr
        scr = self.reader.read(frame)
        self._full_scr = scr
        # HP is only trustworthy with the normal HUD up (pickers, carousel and panels cover it)
        raw = scr.hp if scr.shop_visible else None
        scr.hp = self.hp_filter(raw, scr.round)
        if raw is not None and raw != scr.hp and raw != getattr(self, '_hp_logged', None):
            self._hp_logged = raw
            self.log(ev='hp_rejected', raw=raw, kept=scr.hp, stage=scr.stage)
        return frame, scr

    def model_call(self, kind, prompt, answer, latency, plan_s=None, at=None, meta=None):
        """plan_s: when the decision (first answer) arrived; latency: the whole reply; at: when it was answered,
        if earlier than now (the preparation during the queue); meta: the call's, if not the last one's."""
        stage = self.phase.stage
        from tft_player.coach import active_session
        self.log(ev='model_call', kind=kind, stage=stage, latency=round(latency, 1),
                 decision_s=round(plan_s, 1) if plan_s else None, at=at, prompt=prompt, answer=answer,
                 meta=meta or getattr(active_session(), 'last_meta', None))
        self.apply_game_strategy(answer, kind)
        if getattr(self, 'strategy_due', None) == stage and kind == 'update':
            self.strategy_due = None
            if 'game_strategy' not in str(answer or ''):
                self.log(ev='game_strategy_missing', stage=stage, what='the combat-start reply has no strategy rewrite')
        self.live.model_call(kind, stage, prompt, answer, latency)
        self.live.event(stage, 'model', f'{kind} answered in {latency:.0f}s')

    def apply_game_strategy(self, answer, kind):
        """game_strategy in a model reply: the paragraph ("text"), why it changed and, in flex, the comp it sticks
        with ("comp"). The model sends it only when its strategy changes; a reply without it keeps the last one,
        and the same text again changes nothing."""
        from tft_player.coach import json_objects
        for o in json_objects(answer if isinstance(answer, str) else ''):
            if not isinstance(o, dict):
                continue
            if isinstance(o.get('trait_choices'), dict) and o['trait_choices'] != getattr(self, 'trait_choices', None):
                self.trait_choices = o['trait_choices']          # a committed comp's brief asks for them (flex)
                self.log(ev='trait_choices', choices=self.trait_choices)
            gs = o.get('game_strategy')
            if gs is None and isinstance(o.get('strategy'), dict):
                gs = o['strategy'].get('game_strategy')          # nested in strategy: the same intent
            if isinstance(gs, str):
                gs = {'text': gs}                                # a bare paragraph
            if not isinstance(gs, dict):
                continue
            new = {k: str(gs[k]).strip() for k in ('text', 'why') if gs.get(k) is not None}
            old = self.game_strategy
            if not (new.get('text') and new['text'] != old.get('text')):
                continue
            self.game_strategy = {'text': new.get('text') or old.get('text', ''), 'why': new.get('why', ''),
                                  'stage': self.phase.stage}
            self.log(ev='game_strategy', kind=kind, stage=self.phase.stage, strategy=self.game_strategy,
                     first=not old.get('text'))

    def strategy_lines(self):
        """The model's game strategy as a state line, in every planning message."""
        gs = self.game_strategy
        if not gs.get('text'):
            return ['Your game_strategy: not written yet (format in your instructions).']
        return [f"Your game_strategy, pinned on screen - send it again when it changes, and rewrite it at every combat start: {gs['text']}"]

    def strategy_panel(self):
        """STRATEGY (bottom right): the model's strategy paragraph and its last change; in flex mode, until it
        commits to a comp ("comp"), its comp probabilities."""
        if self.pilot:
            return None
        gs = self.game_strategy
        if not self.forced and self.comp_p:
            return {'comps': sorted(self.comp_p.items(), key=lambda kv: -kv[1])[:5], 'pending': not gs.get('text')}
        foot = ''
        if gs.get('text'):
            st = gs.get('stage')
            # just when (the user, 2026-10-13): the reason was cut off and read as noise
            foot = f'last updated: {st}' if st else 'last updated: before the game'
        return {'comp': self.forced.name if self.forced else '', 'text': gs.get('text', ''),
                'foot': foot, 'pending': not gs.get('text')}

    def publish(self, scr, planning):
        """Push the current picture to the live view."""
        p = self.brain.plan
        agents = dict(self.brain.status)
        agents['loot'] = self.status_extra.get('loot', '')
        agents['carousel'] = self.status_extra.get('carousel', '')
        agents['augment'] = self.status_extra.get('augment', '')
        agents['identify'] = self.status_extra.get('identify', '')
        agents['coach'] = ('thinking...' if self.coach.busy else 'idle') + \
            (f' (last answer {self.coach.answers[-1][0]:.0f}s)' if self.coach.answers else '')
        self.live.update(
            game=dict(stage=scr.stage, timer=scr.timer, gold=scr.gold, level=scr.level, hp=scr.hp,
                      phase='planning' if planning else ('augment' if scr.augments else 'combat/other')),
            agents=agents,
            plan=dict(comp=p.comp, carry=p.carry, traits=p.traits, units=p.units, style=p.style),
            roster=repr(self.roster).replace(' bench=', '\nbench='),
            items=self.bench_items or [])
        if self.overlay and time.time() - self.overlay_at > 0.3:
            self.overlay_at = time.time()
            self.overlay.show(**self.overlay_texts(scr, planning, agents))

    def overlay_texts(self, scr, planning, agents):
        """The four overlay panels (see overlay.py): the model's decision, the target comp, the plan's
        execution and the bot's records."""
        snap = self.live.snapshot()
        if self.pilot:
            c = self.pilot_call
            recent = c and (c['answer'] is None or time.time() - c.get('t1', 0) < 90)
            old = self.pilot_call_narrative(scr, c) if recent else self.pilot_narrative(scr)
            top = {'stage': scr.stage or self.phase.stage, 'title': old.get('intent'), 'info': old.get('info', ''),
                   'thinking': bool(old.get('thinking')), 'alerts': self.alert_lines(),
                   'rows': [(t, str(v), '') for t, v in old.get('lines', [])]}
        else:
            top = self.decision_panel(scr, snap)
        if scr is not None and is_carousel(scr):
            return {}                                         # show_carousel draws the carousel's panels
        out = dict(top_left=top, top_right=self.comp_panel(scr), bottom_left=self.exec_panel(scr, planning),
                   bottom_right=self.strategy_panel())
        cards = self.augment_cards(scr, snap) if scr is not None and scr.augments and not self.pilot else None
        if cards or getattr(self, '_cards_shown', False):     # the picker closed: its explanations go too
            out['path'] = {'cards': cards} if cards else None
            self._cards_shown = bool(cards)
        return out

    def augment_cards(self, scr, snap):
        """The model's explanation of each augment card on screen, for the overlay to draw over the cards
        (the user, 2026-10-12): while it writes it (no wait for the answer), then with its choice marked
        (TAKING / REROLL / KEEP) until the picker closes. [left, middle, right] or None."""
        import re as _re
        th = self.thinking_now()
        stage = scr.stage or self.phase.stage
        earlier = [m for m in (snap.get('model') or []) if m['kind'] == 'augment' and m.get('stage') == stage]
        if th and th['what'].startswith('DECISION: augment'):
            text, a = th['text'], {}
        else:
            if not earlier:
                return None
            m, earlier = earlier[-1], earlier[:-1]
            text, a = m.get('answer') or '', self.answer_obj(m)
        tidy = lambda t: '\n'.join(ln.strip().replace('Target final board:', 'Final board:')
                                   for ln in (t or '').strip().splitlines() if ln.strip())

        def parse(txt):
            out = [None, None, None]
            parts = _re.split(r'(?m)^[ *#]*Card (\d)\s*\((?:left|middle|right)\)\**:\s*', txt or '')
            for i in range(1, len(parts) - 1, 2):
                n = int(parts[i])
                if not 0 <= n <= 2:
                    continue
                body = _re.split(r'(?m)^\s*\{', parts[i + 1])[0]               # the JSON answer is not text
                co = _re.search(r'COMBAT:(.*?)(?:ECONOMY:|VERDICT:|$)', body, _re.S)
                ec = _re.search(r'ECONOMY:(.*?)(?:VERDICT:|$)', body, _re.S)
                ve = _re.search(r'VERDICT:\s*(.+)', body)
                out[n] = {'name': body.strip().splitlines()[0].split(' - ')[0].strip(' *"') if body.strip() else '',
                          'combat': tidy(co.group(1)) if co else '', 'economy': tidy(ec.group(1)) if ec else '',
                          'verdict': ve.group(1).strip() if ve else ''}
            return out
        cards = parse(text)
        # after a reroll the answer explains only the new card: the unchanged ones keep their earlier explanation
        for m in reversed(earlier):
            for n, c in enumerate(parse(m.get('answer') or '')):
                if cards[n] is None and c is not None:
                    cards[n] = c
        if isinstance(a.get('pick'), int) and 0 <= a['pick'] <= 2 and cards[a['pick']]:
            cards[a['pick']]['mark'] = 'TAKING'
        for i in a.get('reroll') or []:
            if isinstance(i, int) and 0 <= i <= 2 and cards[i]:
                cards[i]['mark'] = 'REROLL'
        if isinstance(a.get('keep'), int) and 0 <= a['keep'] <= 2 and cards[a['keep']]:
            cards[a['keep']]['mark'] = 'KEEP'
        return cards if any(cards) else None

    # -- overlay panels ---------------------------------------------------------------------------
    PLAN_KINDS = ('plan', 'update', 'review', 'combat', 'pilot')
    CARD_KINDS = {'augment': 'AUGMENT', 'wisp': 'WISP', 'choice': 'CHOICE', 'carousel': 'CAROUSEL · v2',
                  'carousel_prep': 'CAROUSEL · v1'}

    def plain(self, text):
        """Overlay text in plain words (the user, 2026-10-16: everything shown in natural language, never code or JSON):
        'Rakan*@h33[Chain]' -> 'Rakan★', "('bench', 1)" -> 'bench slot 1', '(3, 2)' -> 'hex 3,2', 'unit 29' -> its name,
        no list indexes, quotes, braces or JSON keys. The model and the log keep the exact text."""
        t = str(text or '')
        t = re.sub(r'execution_plan\.\w+\s*\{.*\}', '', t)                   # a reply-format hint
        t = re.sub(r'\{[^{}]*\}', '', t)                                     # any JSON left
        t = re.sub(r'\b(items|board|bench|sell|buy|wisps)\[\d+\]\s*', '', t)  # "items[0] "
        names = {u.id: u.name for u in getattr(getattr(self, 'roster', None), 'units', []) if u.name}
        t = re.sub(r'\bunits? (\d+)\b', lambda m: names.get(int(m.group(1)), m.group(0)), t)
        t = re.sub(r'(\*+)@[hb]\d\d?(\[[^\]]*\])?', lambda m: '★' * len(m.group(1)), t)   # a unit: its stars only
        t = re.sub(r'\[([^\]]*)\]', lambda m: m.group(1) or 'nothing', t)       # a list: its names
        t = re.sub(r'\bplan:\s*', '', t)
        t = re.sub(r"\('bench', (\d)\)", r'bench slot \1', t)
        t = re.sub(r"\('board', (\d), (\d)\)", r'hex \1,\2', t)
        t = re.sub(r'\((\d), (\d)\)', r'hex \1,\2', t)
        t = t.replace('(unverified)', '').replace(' -> ', ' → ').replace('->', '→')
        t = re.sub(r'["`]|(?<![A-Za-z])\'|\'(?![A-Za-z])', '', t)          # quotes, not apostrophes
        t = re.sub(r'\s+', ' ', t).strip(' ;,-.:')
        return t[:1].upper() + t[1:]

    def alert(self, tag, text, where=''):
        """A problem for the red line on top of the decision panel: shown ~25 s, newest first - in plain words."""
        al = self.__dict__.setdefault('alerts', [])
        al.append((time.time(), tag, self.plain(text), where))
        del al[:-6]

    def alert_lines(self):
        now = time.time()
        return [(tag, text, where) for t, tag, text, where in reversed(getattr(self, 'alerts', [])) if now - t < 25][:2]

    def decision_panel(self, scr, snap):
        """AI DECISION: the latest round plan as rows (its own sentences), the interest numbers, and a
        one-off choice (augment, Wisp, anvil, carousel list) as a card for a few seconds. Never raw JSON."""
        from tft_player.coach import forced_comp
        calls = snap.get('model') or []
        fc = forced_comp()
        d = {'stage': scr.stage or self.phase.stage, 'alerts': self.alert_lines(),
             'thinking': self.thinking_now(), 'took': self.last_took(),
             'title': fc.name if fc else None, 'rows': []}
        plans = [m for m in calls if m['kind'] in self.PLAN_KINDS or 'execution_plan' in str(m.get('answer') or '')]
        plan = plans[-1] if plans else None
        this_round = [m for m in plans if m.get('stage') == d['stage']]
        if this_round:
            d['rows'] = self.round_rows(this_round)
            if not d['title']:
                d['title'] = self.plan_target(this_round[-1])
        if d['rows']:
            self._dec_last = (d['stage'], list(d['rows']), d['title'])
        elif getattr(self, '_dec_last', None):
            # no answer yet this round: the last plan stays up until the new one arrives (the user, 2026-10-10)
            st, d['rows'], d['title'] = self._dec_last[0], list(self._dec_last[1]), d['title'] or self._dec_last[2]
            d['stage'] = f"{d['stage']} · plan of {st}" if st != d['stage'] else d['stage']
        # while the model thinks, the header shows it; its reasoning fills the panel only before the round's
        # first answer (every update question used to swap the plan out for a few seconds)
        d['think_block'] = not d['rows']
        econ = self.econ_row(scr)
        if econ:
            at = next((k for k, r in enumerate(d['rows']) if r[0] in ('NEXT', 'NOTE')), len(d['rows']))
            d['rows'].insert(at, econ)
        # a one-off choice (augment, Wisp, anvil) shows in the same page as the plan, until a newer answer
        one = next((m for m in reversed(calls) if m['kind'] in self.CARD_KINDS), None)
        if one and (not plan or one.get('t', 0) >= plan.get('t', 0)):
            d['card'] = self.card(one)
        return d

    @staticmethod
    def thinking_now():
        """The model's question in flight, for the overlay: {what, t0, text} (text: the reply so far), or None."""
        from tft_player import coach
        lv = coach.LIVE
        if lv.get('took') is not None or not lv.get('t0') or not coach.model_busy():
            return None
        what = lv.get('what') or 'thinking'
        what = what.split(' - ')[0].rstrip('.:')[:110]
        return {'what': what, 't0': lv['t0'], 'text': (lv.get('text') or '')[-4000:]}

    @staticmethod
    def last_took():
        from tft_player import coach
        return coach.LIVE.get('took')

    @staticmethod
    def answer_obj(m):
        from tft_player.coach import json_objects
        a = {}
        for obj in json_objects(m.get('answer') or ''):
            if isinstance(obj, dict):
                a.update(obj)
        return a

    @staticmethod
    def plan_target(m):
        st = Player.answer_obj(m).get('strategy')
        return st.get('target') if isinstance(st, dict) else None

    @staticmethod
    def plan_rows(m):
        """One row per part of a round plan, in the model's own words; the action itself where it
        wrote none. -> [(LABEL, text, reason)]"""
        a = Player.answer_obj(m)
        st = a.get('strategy') if isinstance(a.get('strategy'), dict) else {}
        ep = a.get('execution_plan') if isinstance(a.get('execution_plan'), dict) else {}
        why = st.get('why') if isinstance(st.get('why'), dict) else {}
        prompt = m.get('prompt') or ''
        acts = Player.plan_parts(ep, prompt)
        rows = []
        for part, tag in (('sell', 'SELL'), ('buy', 'BUY'), ('level', 'LEVEL'), ('board', 'BOARD'),
                          ('items', 'ITEMS'), ('containers', 'OPEN')):
            said = str(why.get(part) or '').strip()
            text = said if said not in ('', '-') else acts.get(part)
            if text:
                rows.append((tag, text, ''))
            if part == 'buy':
                rows += [(t, txt, '') for t, txt in Player.wisp_lines(ep, prompt)]
        if str(why.get('next') or '').strip() not in ('', '-'):
            rows.append(('NEXT', str(why['next']), ''))
        for pr in a.get('problems') or []:
            rows.append(('NOTE', str(pr), ''))
        if not rows and st.get('now'):
            rows.append(('NOW', str(st['now']), ''))
        return rows

    ROW_ORDER = ('SELL', 'BUY', 'WISP', 'LEVEL', 'BOARD', 'ITEMS', 'OPEN', 'NEXT', 'NOTE', 'NOW')
    ROW_KIND = {'SELL': 'sell', 'BUY': 'buy', 'WISP': 'wisp', 'LEVEL': 'level', 'BOARD': 'move', 'ITEMS': 'item'}
    ROW_ACTS = {'SELL': 'sell', 'BUY': 'buy', 'WISP': 'wisps', 'LEVEL': 'xp', 'BOARD': 'board', 'ITEMS': 'items',
                'OPEN': 'open_container'}
    SAYS_NOTHING = re.compile(r"^\s*(-|unchanged|no change|same|nothing|none|n/a)\b", re.I)

    def round_rows(self, calls):
        """The round's plan as rows, one per part: the round's first plan, each later answer replacing a part only
        when it says something new for it ("Unchanged nine-unit board" keeps what was there). A part the
        executor has finished is ticked, so a done step does not read as still to do."""
        rows, notes, acts = {}, [], {}
        for m in calls:
            ep = self.answer_obj(m).get('execution_plan')
            ep = ep if isinstance(ep, dict) else {}
            # a part's words change only when its actions do: the model rewords the same plan in every update
            # (and a combat update's "the board is locked" is no new board)
            changed = set()
            for label, key in self.ROW_ACTS.items():
                v = ep.get(key)
                if v not in (None, [], {}, 0, '') and json.dumps(v, sort_keys=True, default=str) != acts.get(key):
                    acts[key] = json.dumps(v, sort_keys=True, default=str)
                    changed.add(label)
            new_notes = []
            for label, text, reason in self.plan_rows(m):
                if label == 'NOTE':
                    new_notes.append((label, text, reason))
                elif label not in rows or ((label in changed or (label not in self.ROW_ACTS and changed))
                                           and not self.SAYS_NOTHING.match(str(text))):
                    rows[label] = (label, text, reason)
            notes = new_notes or notes
        rp = self.round_plan
        if rp is not None and not rp.carry and rp.stage == calls[-1].get('stage'):
            p = rp.p
            has = {'SELL': bool(p.get('sell')), 'BUY': bool(rp.to_buy), 'LEVEL': int(p.get('level_ups') or 0) > 0,
                   'BOARD': bool(p.get('board')), 'ITEMS': bool(p.get('items')),
                   'WISP': any(isinstance(w, dict) and w.get('buy') for w in p.get('wisps') or [])}
            pending = {k for k, _ in rp.queued(self.roster)}
            for label, kind in self.ROW_KIND.items():
                if label in rows and has.get(label) and kind not in pending:
                    lb, text, reason = rows[label]
                    rows[label] = (lb, '✓ ' + str(text), reason)
        out = [rows[k] for k in self.ROW_ORDER if k in rows and k != 'NOTE']
        at = next((i for i, r in enumerate(out) if r[0] in ('NEXT', 'NOW')), len(out))
        return out[:at] + notes + out[at:]

    def econ_row(self, scr):
        """ECON: gold now, after a win / a loss, the next interest step, and the model's after-fight sells
        (once the result shows, the sells for that result)."""
        from tft_player.planner import is_pvp
        base = getattr(self, 'fight_base', None) or {}
        if not self.phase.planning and base.get('stage') == scr.stage and base.get('gold') is not None:
            g = base['gold']          # in combat: from the gold at combat start (a quick win's +1 is not counted twice)
        else:
            g = scr.gold if scr.gold is not None else getattr(self, '_last_gold_seen', None)
        if g is None or g >= 50:
            return None
        self._last_gold_seen = g
        # the user, 2026-10-16: "24g" and what happens on a win / a loss - not "win -> 25 - lose -> 24 - next 30"
        from tft_player.planner import interest_needs
        n = interest_needs(g, self.roster, is_pvp(scr.stage or self.phase.stage))
        def need(r):
            return 'hold' if r['need'] == 0 else f"sell {r['need']}g" if r['reachable'] else f"{r['need']}g short"
        rs = n['results']
        text = f"{g}g · bench {n['worth']}g · " + (
            f"{rs[0]['next']}: win {need(rs[0])} · lose {need(rs[1])}" if len(rs) == 2
            else f"{rs[0]['next']}: {need(rs[0])} (no win gold)")
        af = getattr(self, 'after_fight', None)
        reason = ''
        if af and af.get('stage') == (scr.stage or self.phase.stage):
            def fmt(o):
                o = o if isinstance(o, dict) else {'sell': list(o or []), 'buy': []}
                s = ', '.join(getattr(self.roster.resolve(i), 'name', None) or f'unit {i}' for i in o.get('sell') or [])
                b = ', '.join(o.get('buy') or [])
                return ' + '.join(x for x in (f'sell {s}' if s else '', f'buy {b}' if b else '') if x) or 'nothing'
            res = (getattr(self, 'fight_base', None) or {}).get('result')
            reason = f"win: {fmt(af.get('win'))} · lose: {fmt(af.get('lose'))}"
            if res:
                reason = f"{res}: {fmt(af.get(res))}"
            text = f"{g}g · {reason}"           # the model's orders for each result are the line (the user)
        return ('ECON', text, reason)

    def card(self, m):
        """A one-off decision as a card: what was chosen and why, in words (never the JSON)."""
        import re as _re
        a = self.answer_obj(m)
        prompt = m.get('prompt') or ''
        tag = self.CARD_KINDS.get(m['kind'], m['kind'].upper())
        if m['kind'] == 'augment':
            names = dict((int(i), n.strip()) for i, n in _re.findall(r'(?m)^\s*(\d): (.+?) - ', prompt))
            name = lambda i: names.get(i, f'card {i}') if isinstance(i, int) else str(i)
            choice = (f"Keeping {name(a['keep'])}, rerolling the others" if a.get('keep') is not None else
                      'Rerolling ' + ', '.join(name(i) for i in a['reroll']) if a.get('reroll') else
                      f"Taking {a.get('name') or name(a.get('pick'))}" if a.get('pick') is not None else 'Deciding')
        elif m['kind'] == 'wisp':
            w = _re.search(r"Wisp '([^']+)'", prompt)
            choice = f"{'Buy' if a.get('buy') else 'Skip'} {w.group(1) if w else 'the Wisp'}"
        elif m['kind'] == 'choice':
            src = _re.search(r'Source: (\w+)', prompt)
            tag = 'ANVIL' if src and 'anvil' in src.group(1) else 'CHOICE'
            choice = f"Took {a.get('name') or 'option ' + str(a.get('option'))}"
        elif m['kind'] in ('carousel', 'carousel_prep'):
            pri = a.get('carousel_priority') or a.get('item_targets') or []
            choice = ' > '.join(map(str, pri[:4])) + (' …' if len(pri) > 4 else '')
        else:
            choice = str(a.get('name') or a.get('pick') or 'answered')
        return {'tag': tag, 'choice': choice, 'why': str(a.get('why') or '')}

    def comp_panel(self, scr):
        """TARGET COMP: the fixed (or target) comp's units in its own order, as owned / missing / short of
        copies, with the comp page's items; hidden during the carousel (it sits in its icon search)."""
        if scr is not None and is_carousel(scr):
            return None
        from tft_player.coach import forced_comp
        from tft_player.meta import META
        comp = forced_comp() or (META.comps.get(self.round_plan.p.get('comp')) if self.round_plan else None)
        if comp is None:
            return None
        stars = comp.guide.get('stars') or {}
        units, owned_n = [], 0
        for name in comp.units:
            info = CATALOG.unit(name)
            mine = [u for u in self.roster.units if u.name == name and not u.missing]
            goal = int(stars.get(name) or 2)
            copies = sum(3 ** (u.star - 1) for u in mine)
            held = {i for u in mine for i in u.items}
            build = ((comp.builds.get(name) or [{}])[0].get('items') or [])[:3]
            units.append({'name': name, 'cost': info.cost if info else None, 'owned': bool(mine),
                          'star': max((u.star for u in mine), default=0), 'goal': goal,
                          'shop': name in (scr.shop or []), 'items': [(it, it in held) for it in build]})
            owned_n += bool(mine)
        lvl = scr.level or self.last_level
        return {'name': comp.name, 'count': f"{owned_n}/{len(comp.units)}" + (f" · L{lvl}" if lvl else ''),
                'units': units}

    @staticmethod
    def pretty(t):
        """A plan step as the log writes it -> words: 'move(plan: Diana*@b1 -> (3, 4))' ->
        'Diana★ bench 1 → (3,4)'."""
        import re as _re
        t = str(t)
        m = _re.match(r'^([a-z_]+)\((.*)\)( - .*)?$', t)
        if m:
            t = m.group(2) + (m.group(3) or '')
        t = t.replace('plan: ', '').replace('(unverified)', '')
        t = _re.sub(r"\[[^\]]*\]", '', t)
        t = _re.sub(r"(\*+)@h(\d)(\d)", lambda k: f"{'★' * len(k.group(1))} ({k.group(2)},{k.group(3)})", t)
        t = _re.sub(r"(\*+)@b(\d)", lambda k: f"{'★' * len(k.group(1))} bench {k.group(2)}", t)
        t = _re.sub(r"\((\d), (\d)\)", r"(\1,\2)", t)
        return t.replace(' -> ', ' → ').strip()

    STEP_TAGS = {'buy': 'BUY', 'sell': 'SELL', 'level': 'XP', 'move': 'MOVE', 'item': 'ITEM', 'container': 'OPEN',
                 'roll': 'ROLL', 'rolldown': 'ROLL', 'wisp': 'WISP'}
    ROWS = {3: 'front', 2: '2nd row', 1: '3rd row', 0: 'back'}     # board rows, front (3) to back (0)

    @classmethod
    def hex_words(cls, row, col):
        return f"{cls.ROWS.get(int(row), 'row ' + str(row))} #{int(col) + 1}"

    @classmethod
    def step_row(cls, kind, why):
        """One step in words with its kind's tag: ('move', 'plan: Elise*@b7 -> (3, 4)') ->
        ('MOVE', 'Elise★ (bench 7) → front row #5')."""
        why = str(why or '')
        tag = 'WISP' if kind == 'buy' and re.sub(r'^plan:\s*', '', why.lower()).startswith('wisp') else             cls.STEP_TAGS.get(kind, kind.upper()[:5])
        t = re.sub(r'^(plan|data|body|pilot|stage \d+):\s*', '', why)
        t = re.sub(r'^auto-', '', t)
        t = re.sub(r'\s*\((?:the model[^)]*|every affordable[^)]*|\d+ crops recorded|photographed)\)$', '', t)
        if tag == 'WISP':
            t = re.sub(r'\s*\(.*$', '', t)                  # the Wisp's name, not its card text
        t = re.sub(r'^bench (\S.*)$', lambda k: k.group(1) + ' -> bench', t)    # "bench Camille*@h00": to the bench
        t = re.sub(r'\[[^\]]*\]', '', t)                     # items a unit holds
        # units only, no positions: the panel is a short list (the user, 2026-10-10)
        t = re.sub(r'(\*+)@[hb]\d\d?', lambda k: '★' * len(k.group(1)), t)
        t = re.sub(r'\(\d, \d\)', 'board', t)
        t = re.sub(r'-> bench \d', '-> bench', t)
        t = t.replace(' -> ', ' → ').replace('->', '→').replace('(unverified)', '').strip()
        return tag, t[:1].upper() + t[1:]

    def exec_panel(self, scr, planning):
        """EXECUTION: this round's actions in the order they ran (✓ done, ✗ failed + why), then the plan's steps
        still to come (○) and a stored next_round plan; every row with its kind's coloured tag."""
        rp = self.round_plan
        acts_now = getattr(self, 'round_acts', [])
        late = any(ok is None for _, _, ok, _ in acts_now)      # board moves queued in the last second
        queued = []
        for kind, text in (rp.queued(self.roster) if rp is not None else []):
            tag, words = self.step_row(kind, text)
            queued.append(('wait', tag, words, 'queued for next round' if late and kind == 'move' else ''))
        # steps the plan gave up without an action (not in the shop, board full, ...): the newest two
        given_up = []
        for f in (rp.failed if rp is not None else [])[-2:]:
            if re.match(r'^\w+\(', f):
                continue                                     # an action that ran and failed: listed below
            body, _, why = f.partition(': ')
            kind = next((k for k in ('sell', 'buy', 'item', 'bench') if body.startswith(k)), 'move')
            given_up.append(('fail',) + self.step_row(kind, body) + (self.plain(why),))
        acts = []
        listed = any(r[1] == 'MOVE' for r in queued)
        for kind, why, ok, cause in acts_now:
            if ok is None:                                   # too late this round: listed with the queued steps
                if not listed:
                    queued.append(('wait',) + self.step_row(kind, why) + ('queued for next round',))
                continue
            reason = (str(cause or '') if kind == 'rolldown' else '') if ok else str(cause or 'failed')
            acts.append(('done' if ok else 'fail',) + self.step_row(kind, why) + (self.plain(reason),))
        acts += getattr(self, 'roll_live', None) or []     # a roll-down running now: its progress, in this list
        nxt = getattr(self, 'next_plan', None)
        nxt_rows = []
        if nxt:
            p = nxt[1]
            parts = [f"sell {len(p.get('sell') or [])}" if p.get('sell') else '',
                     'buy ' + ', '.join(map(str, p.get('buy') or [])) if p.get('buy') else '',
                     f"level x{p['level_ups']}" if p.get('level_ups') else '',
                     f"board {len(p.get('board') or [])}" if p.get('board') else '',
                     f"items {len(p.get('items') or [])}" if p.get('items') else '']
            nxt_rows.append(('next', 'NEXT', 'Next round: ' + (' · '.join(x for x in parts if x) or 'nothing to do'), ''))
        room = 10                                            # rows the panel holds
        keep = max(2, room - len(queued) - len(given_up) - len(nxt_rows))
        steps = acts[-keep:] + given_up + queued + nxt_rows
        if not steps:
            # nothing yet this round: the last list stays up until the next one has rows (the user, 2026-10-10)
            return getattr(self, '_exec_last', None) or {'mode': 'PLAN' if planning else 'COMBAT', 'steps': []}
        self._exec_last = {'mode': 'PLAN' if planning else 'COMBAT', 'steps': steps[:room]}
        return self._exec_last

    def pilot_narrative(self, scr):
        """The top panel in pilot mode: the plan for this stage of the game and each part's
        current decision (the same reasons the live view lists)."""
        p, st = self.pilot, self.pilot.status
        stage = scr.round
        left = p.goal_left(self.roster)
        if stage and stage >= p.eight_at and (scr.level or 0) >= 8:
            intent = ('Level-8 roll-down: looking for ' + ', '.join(left)) if left else 'Comp complete: the 8 fielded'
        elif stage and stage >= (2, 1):
            intent = f'Econ toward the level-8 roll-down at {p.eight_at[0]}-{p.eight_at[1]} - {p.comp.name}'
        else:
            intent = f'Stage 1: units for the early board - {p.comp.name}'
        lines = []
        for tag, key in (('ECON', 'econ'), ('LEVEL', 'level'), ('BUY', 'buy'), ('EARLY', 'early'), ('ROLL', 'roll'),
                         ('SELL', 'sell'), ('BOARD', 'board'), ('ITEMS', 'items')):
            if st.get(key):
                lines.append((tag, str(st[key])))
        if self.status_extra.get('augment'):
            lines.append(('AUGMENT', self.status_extra['augment']))
        if self.status_extra.get('choice'):
            lines.append(('CHOICE', self.status_extra['choice']))
        c = self.pilot_call
        if c and c['answer'] is not None:
            d = self.pilot_call_narrative(scr, c)
            lines.insert(0, ('MODEL', f"{c['stage']}: {d['intent']} - {c.get('why') or ''}"))
        return {'head': f"Stage {scr.stage or self.phase.stage} - pilot", 'intent': intent, 'lines': lines, 'info': ''}

    def pilot_call_narrative(self, scr, c):
        """The model's judgement call for the top panel: the question while it thinks, then its
        decision and reasoning."""
        what = {'augment': 'augment', 'wisp': 'Wisp'}.get(c['kind'], c['kind'])
        head = f"Stage {c['stage']} - model: {what}"
        if c['answer'] is None:
            return {'head': head, 'intent': f"Deciding: {c['about']}", 'lines': [],
                    'info': f"{time.time() - c['t0']:.0f}s", 'thinking': True}
        a = c['answer']
        if c['kind'] == 'augment':
            names = c.get('names') or [None] * 3
            name = lambda i: (names[i] or f'card {i}') if isinstance(i, int) and 0 <= i < 3 else str(i)
            if a.get('reroll'):
                intent = 'Rerolling ' + ', '.join(name(i) for i in a['reroll'])
            elif a.get('pick') is not None:
                intent = f"Taking {name(a['pick'])}"
            else:
                intent = 'No usable answer: the tier list decides'
            if c.get('late'):
                intent = f"Too late, the tier list took {c['late']}. The model: {intent}"
        else:
            intent = f"{'Buy' if a.get('buy') else 'Skip'} {c['about']}" if a else 'No usable answer: skip'
        return {'head': head, 'intent': intent, 'lines': [('WHY', str(c.get('why') or '...'))],
                'info': f"{c.get('t1', c['t0']) - c['t0']:.0f}s"}

    @staticmethod
    def plan_parts(ep, prompt):
        """The execution plan in words, per part, with names (the prompt's state maps slots and ids)."""
        import re as _re
        shop = dict((int(i), n) for i, n in _re.findall(r'slot (\d): ([^\s(]+(?: [^\s(]+)?) \(\d+g', prompt))
        units = dict((int(i), n.strip()) for i, n in _re.findall(r'\[id (\d+)\] ([^*\n]+?) \*', prompt))
        uname = lambda e: e.get('name') or units.get(e.get('unit'), f"unit {e.get('unit')}")
        out = {}
        if ep.get('sell'):
            out['sell'] = 'sell ' + ', '.join(units.get(i, f'unit {i}') for i in ep['sell'])
        if ep.get('buy'):
            out['buy'] = 'buy ' + ', '.join(shop.get(i, str(i)) if isinstance(i, int) else str(i) for i in ep['buy'])
        level = []
        if ep.get('xp'):
            level.append(f"level up (XP x{ep['xp']})")
        if isinstance(ep.get('roll'), dict):
            # what the roll order says, in its own terms (the user, 2026-10-11: "roll down to None gold" for a
            # roll x4 order was confusing)
            r = ep['roll']
            buy = r.get('buy')
            buy = ', '.join(buy if isinstance(buy, list) else [f'{n} {"*" * int(s)}' for n, s in buy.items()]
                            if isinstance(buy, dict) else []) if buy else ''
            how = (f"roll {r['times']} times" if r.get('times') is not None else
                   f"roll down to {r['to_gold']} gold" if r.get('to_gold') is not None else 'roll')
            level.append(how + (f' for {buy}' if buy else ''))
        if isinstance(ep.get('fast_rr'), dict):
            level.append('fast roll-down for ' + ', '.join(ep['fast_rr'].get('buy') or []))
        if level:
            out['level'] = ', '.join(level)
        if ep.get('board'):
            out['board'] = 'board: ' + ', '.join(uname(e) for e in ep['board'] if isinstance(e, dict))
        if ep.get('items'):
            uid = lambda i: units.get(i, f'unit {i}')

            def item_words(e):
                if 'transfer' in e:
                    return f"{e['transfer']}: {uid(e.get('from'))} -> " + ('the item bench' if e.get('to') == 'bench'
                                                                          else uid(e.get('to')))
                if 'reforge' in e:
                    return f"reforge {uid(e['reforge'])}"
                if 'give' in e or 'use' in e:
                    return f"{e.get('give') or e.get('use')} -> {uid(e.get('to', e.get('on')))}"
                return f"{e.get('item')} -> {uname(e)}"
            out['items'] = ', '.join(item_words(e) for e in ep['items'] if isinstance(e, dict))
        if ep.get('open_container'):
            out['containers'] = 'open the container on bench slot ' + ', '.join(map(str, ep['open_container']))
        if ep.get('item_targets'):
            out['targets'] = 'carousel/orbs: want ' + ', '.join(map(str, ep['item_targets']))
        return out

    @staticmethod
    def plan_summary(ep, prompt):
        return '; '.join(Player.plan_parts(ep, prompt).values()) or 'no actions'

    @staticmethod
    def wisp_lines(ep, prompt):
        import re as _re
        names = dict((int(k), n) for k, n in _re.findall(r"slot (\d): WISP '([^']+)'", prompt))
        return [('WISP', f"{'Buy' if w.get('buy') else 'Skip'} {names.get(w.get('slot'), 'the Wisp')} - {w.get('why') or ''}")
                for w in ep.get('wisps') or [] if isinstance(w, dict)]

    def hp_filter(self, hp, rnd=None):
        """Our HP, from our own HP box (read only with the normal HUD up). A new value counts once
        it is read twice in a row at least 0.5 s apart (a frame can catch the number mid-change, and two
        reads of one unchanged picture are one read: run 010416 took a '7' for 84); a rise needs 6 reads
        (a rise is almost always a misread). No damage cap: since 09-30 the box read is ours (the
        cap came from reading another player's ring) - it kept the state a round behind the
        screen in run26. The time of the last good read lets the round's prompt wait for it."""
        last = getattr(self, '_hp', None)
        if hp is None or not 0 < hp <= 100:
            return last
        now = time.time()
        if last is None or hp == last:
            self._hp, self._hp_run, self._hp_read_at = hp, 0, now
            return hp
        same = getattr(self, '_hp_runv', None) == hp
        run = getattr(self, '_hp_run', 0) + 1 if same else 1
        if not same:
            self._hp_run_t0 = now
        self._hp_run, self._hp_runv = run, hp
        if run >= (2 if hp < last else 6) and now - getattr(self, '_hp_run_t0', now) >= 0.5:
            self._hp, self._hp_run, self._hp_read_at = hp, 0, now
        return self._hp

    def run(self):
        """Play until the stage limit. A failing tick is logged and skipped;
        only F8 / Ctrl-C / the stage limit end the run."""
        import traceback
        from tft_player.game import Paused, Stopped
        print('player running - F7 to pause / continue, F8 to stop')
        self._last_print = ''
        errors = 0
        self._start_tick_watchdog()
        while True:
            try:
                self._tick_t0 = time.time()
                done = self._tick() == 'done'
                took = time.time() - self._tick_t0
                self._tick_t0 = None
                if took > 1.0:
                    self.log(ev='slow_tick', seconds=round(took, 2), stage=getattr(self.phase, 'stage', None),
                             planning=getattr(self.phase, 'planning', None), where=getattr(self, '_tick_where', None))
                self._tick_where = None
                if done:
                    return
            except (Stopped, KeyboardInterrupt):
                raise
            except Paused:
                time.sleep(0.1)                  # F7 came mid-action: the action is not sent; tracking goes on
            except Exception as e:
                errors += 1
                self.log(ev='tick_error', error=repr(e), trace=traceback.format_exc()[-1500:])
                if errors <= 5 or errors % 50 == 0:
                    print(f'tick error ({errors}): {e!r}')
                time.sleep(0.2)

    def _start_tick_watchdog(self):
        """What a tick that takes > 1 s is doing: the main thread's innermost game-code frames, sampled once per slow
        tick and logged with it (run 20261010-010345: rounds after a fight were first seen with ~20 s of their 30 s
        left and nothing in the log said why)."""
        import sys
        import traceback
        main_id = threading.main_thread().ident

        def watch():
            while True:
                time.sleep(0.25)
                t0 = getattr(self, '_tick_t0', None)
                if t0 is None or time.time() - t0 < 1.0 or getattr(self, '_tick_where', None):
                    continue
                fr = sys._current_frames().get(main_id)
                if fr is None:
                    continue
                stack = [f for f in traceback.extract_stack(fr) if 'tft_player' in f.filename]
                self._tick_where = [f'{Path(f.filename).stem}.{f.name}:{f.lineno}' for f in stack[-6:]]
        threading.Thread(target=watch, daemon=True).start()

    def _tick(self):
        last_print = self._last_print
        if True:
            t0 = time.perf_counter()
            if hasattr(self.inp, 'ensure_front') and not self.inp.ensure_front():
                if self.ending.game_over:
                    # the game window is gone (we left after elimination) and the client says it is over:
                    # waiting for the window would never end (run17 and run19 hung here)
                    E = self.ending
                    seen = ' '.join([E.eliminated_seen or ''] + list(getattr(E, 'words', []) or []))
                    m = re.search(r'([1-8])\s*(ST|ND|RD|TH)\s*PLACE', seen.upper())
                    self.placement_screen = (int(m.group(1)), f'elimination screen: "{m.group(0)}"') if m else                         (None, 'eliminated, place not read')
                    print(f'game over (client phase {self.ending.phase}); the game window is gone; placement {self.placement_screen}')
                    self.log(ev='game_over', phase=self.ending.phase, stage=self.phase.stage, hp=getattr(self, '_hp', None),
                             placement_screen=self.placement_screen)
                    return 'done'
                if time.time() - getattr(self, '_nofocus_logged', 0) > 5:
                    self._nofocus_logged = time.time()
                    self.log(ev='no_focus', stage=self.phase.stage)
                    print('game window not in front: waiting')
                time.sleep(0.3)             # covered and couldn't take focus: frames would be garbage
                return
            frame, scr = self.read()
            if scr is None:
                time.sleep(0.05)
                return
            self.last_frame = frame
            self.standings_shown(frame, scr)
            self.bench_reference(frame, scr)
            stalls = getattr(self.cap, 'stalls', None)
            while stalls:
                at, waited = stalls.pop(0)
                self.log(ev='capture_stall', at=at, seconds=waited, stage=self.phase.stage)
            if scr.stage:
                self.stage_seen = scr.stage
            if not REPLAY and self.ending.greyed(frame):
                self.ending.scan_now()          # an elimination greys the screen: read it at once
            if self.ending.eliminated_seen and re.search(r'PLACE', self.ending.eliminated_seen.upper()):
                from tft_player import coach
                if not coach.STOPPED:
                    coach.STOPPED = True        # no more questions to the model: the game is lost
                    self.log(ev='eliminated_stop', stage=scr.stage, text=self.ending.eliminated_seen)
                    print(f'{scr.stage}: eliminated ({self.ending.eliminated_seen}) - model calls and actions stopped')
            end = self.handle_end(frame, scr)
            if end == 'done':
                return 'done'
            if end:
                time.sleep(0.3)
                return
            from tft_player import coach as _c
            if _c.STOPPED:                      # eliminated: nothing else is played; wait for the exit button
                time.sleep(0.2)
                return
            # the menu check is OCR (~0.85s while the carousel turns: its pixels never stay still);
            # once a tick it cut the carousel loop to one action every 1-2s (run29)
            carousel = is_carousel(scr)
            check_menu = (not scr.shop_visible
                          and time.time() - getattr(self, '_menu_at', 0) > (5.0 if carousel else 1.0))
            if check_menu:
                self._menu_at = time.time()
            from tft_player.game import PAUSED as _PAUSED
            if check_menu and not _PAUSED.is_set() and self.reader.settings_open(frame):
                # the game menu blocks everything; close it with its X button
                print(f'{scr.stage}: game menu open - closing it')
                self.log(ev='menu_closed', stage=scr.stage)
                self.inp.click(SETTINGS_CLOSE)
                time.sleep(0.4)
                return
            was_planning = self.phase.planning
            self.phase.update(scr)
            if self.phase.changed:
                set_stage(self.phase.stage)          # a new stage number: the model's next question opens a new conversation
            from tft_player.game import PAUSED
            paused = PAUSED.is_set()
            self.roster.in_combat = not self.phase.planning      # merges with a fighting unit wait for the fight's end
            if not was_planning and self.phase.planning:
                merged = self.roster.settle_merges()             # ...and happen now
                if merged:
                    self.log(ev='merges_settled', stage=scr.stage, names=merged)
                    print(f'{scr.stage}: merges after the fight: {", ".join(merged)}')
            elif self.phase.planning and time.time() - getattr(self, '_merges_at', 0) > 1.0:
                # three copies of one star never stand side by side in planning - the game merges them at once. Copies
                # that came without our buy (a carousel unit, a panel read naming a unit, a star corrected from its
                # badge) never ran the buy's merge: run 20261010-235723 showed three 2-star Azirs for rounds
                self._merges_at = time.time()
                merged = self.roster.settle_merges()
                if merged:
                    self.log(ev='merges_inferred', stage=scr.stage, names=merged)
                    self.news_event('merged', scr.stage, f'three copies of {", ".join(merged)} are one unit of the next '
                                                         f'star level now (the game merges them)')
            # a carousel is no fight: its start saved a full-res snap and a unit gallery of the carousel and built a
            # "COMBAT STARTED" model call on the chase's loop (run 010345 2-4: a 2.2 s gap between chase ticks)
            if was_planning and not self.phase.planning and not self.phase.changed and not is_carousel(scr):
                self.combat_t0, self._autofill_prev, self._autofill_stage = time.time(), None, None
                # the fight's result shows as a streak change (or an HP drop): its baseline, for after-fight sells
                self.fight_base = {'stage': scr.stage, 'streak': self.reader.streak(frame), 'hp': scr.hp, 'gold': scr.gold}
                if paused:
                    pass                             # F7: no model questions; continuing asks with the state then
                elif self.forced:
                    self.combat_start(scr)
                else:
                    self.review_round(scr)
                    self.prepare_carousel(scr)       # x-3: the carousel's wants (combat_start does it in fixed-comp games)
                self.snap(frame, f'{scr.stage}-fight')       # the board we actually fought with
                self.save_board_gallery(frame, scr)          # every unit's crop with the labels we hold for it
                self.log(ev='fight', stage=scr.stage, gold=scr.gold, level=scr.level, hp=scr.hp,
                         roster=repr(self.roster), plan=repr(self.brain.plan),
                         agents=dict(self.brain.status), items=self.bench_items)
            if self.phase.changed:
                self.round_started = time.time()
                self.round_acts = []             # this round's executed actions (the overlay's list)
                if scr.hp is not None:
                    if self.hp_at_round is not None:
                        self.hp_log.append((scr.stage, self.hp_at_round, scr.hp))
                    self.hp_at_round = scr.hp
                # last round's unfinished board/items carry over: executed at once, and
                # the new plan is told about them (never sold meanwhile: protected)
                self.carry_note, self.plan_calls = '', 0
                if self.round_plan is not None:
                    from tft_player import planner
                    carry, self.carry_note = self.round_plan.leftover(self.roster)
                    self.round_plan = planner.RoundPlan(carry, scr.stage, carry=True, names=self.unit_names(),
                                                        gold=scr.gold) if carry else None
                self.brain.new_round(scr.round, self.roster)
                # a consumable's count changes without its icon changing (the game adds removers): its tooltip is read
                # again each round (run 20261014-063145: "2 uses left" all game while the slot showed 4, 5, then 6)
                for sig in [k for k, v in self.item_names.items() if 'uses left' in (v or '')]:
                    del self.item_names[sig]
                for name in getattr(self, 'recurring_changes', ()):
                    self.unit_change(name, scr.stage)
                self.items_done = set()
                ch = getattr(self, 'chaser', None)
                # the HUD often shows the carousel's stage a second after the carousel began:
                # that is the same carousel, keep chasing (run28 reset the chase mid-carousel)
                same_carousel = self.carousel_stage == scr.stage
                if ch is not None and (ch.item or ch.taken or ch.got) and not same_carousel:
                    # what the last carousel did, for the model: what we went for, what another
                    # player took first (contested - not a pathing miss), what we ended up chasing
                    went = ' -> '.join(dict.fromkeys(ch.taken + ([ch.item or ch.got] if ch.item or ch.got else [])))
                    taken = ', '.join(ch.taken)
                    self.news_event('carousel', self.carousel_stage or scr.stage,
                                    f'carousel: went for {went}' + (f' ({taken} taken by other players first)'
                                                                    if taken else ''))
                    self.log(ev='carousel_result', stage=scr.stage, taken=ch.taken, last_target=ch.item,
                             got=ch.got)
                    # confirmed by the one new unit that holds an item (check_carousel_unit): not the first new
                    # unit - loot drops arrive too (run 033507: "we got Gromp / Diana / Nidalee / Mama Beak", 4 of 4 wrong)
                    carried = getattr(self, '_carried_read', {}).get(self.carousel_stage)
                    if carried and carried[0]:
                        self.news_event('carousel', self.carousel_stage, f'carousel: we got {carried[0]} holding '
                                        f'{carried[1] or "an item not recognised"} (read off its panel as we carried it)')
                    self.carousel_confirm = {'stage': self.carousel_stage, 'went': ch.item or ch.got, 'got': ch.got,
                                             'carried': carried,
                                             'before': {u.id for u in self.roster.units}, 'at': time.time(),
                                             'offer': sorted(getattr(self, '_car_offer', {}).get(self.carousel_stage, ()))}
                if not same_carousel:
                    self.chaser, self.own = CarouselChaser(), OwnTracker()
                    if self.overlay:
                        self.overlay.show(path=None)
                self.bench_items = getattr(self, 'bench_items', None)
                self.snap(frame, f'{scr.stage}-start', full=True)
                self.log(ev='round', stage=scr.stage, screen=scr.short(), roster=repr(self.roster),
                         agents=dict(self.brain.status))
                self.live.event(scr.stage, 'round', scr.short())
                print(f'\n=== {scr.stage}  gold {scr.gold}  L{scr.level}  hp {scr.hp}  {self.roster}')
                # a unit not found again for a whole round is dropped from the records, and the model told once (the
                # user, 2026-10-12: the "Not located" list only grew). Still there after all? Then it shows up as a
                # unit without a record and is read off its panel like any other (read_new_unit, unit_arrived)
                for u in list(self.roster.units):
                    if not u.missing:
                        u.missing_rounds = 0
                        continue
                    u.missing_rounds = getattr(u, 'missing_rounds', 0) + 1
                    if u.missing_rounds >= 2:
                        self.roster.units.remove(u)
                        self.news_event('unit_dropped', scr.stage, f'{u.name or "?"}{"*" * u.star} (unit {u.id}) was '
                                        f'not found for a whole round ({u.missing}): removed from the records')
            if self.paused_tick(frame, scr, paused):
                return
            if getattr(self, 'combat_call', None):
                self.send_combat_call(scr)
            # stage limit: stop once the normal HUD is up (an open augment picker
            # blocks the Settings menu, so surrendering there fails)
            st = self.phase.stage and Phase._key(self.phase.stage)
            if self.until and st and st > self.until and scr.shop_visible and not scr.augments:
                print(f'reached {self.phase.stage}; stopping')
                return 'done'
            in_carousel = is_carousel(scr) and not scr.shop_visible
            if not in_carousel and not scr.augments:
                self.learn_bench_baseline(frame, scr)
            if DATA_ROLL and (in_carousel or scr.augments or bench_choice_open(frame)):
                return                           # the player's own game: augments, carousels, choices are theirs
            # (no choice screen opens during a carousel; its OCR check would slow the chase)
            if (not in_carousel or self.choice) and self.watch_choice(frame, scr):
                return
            if scr.augments:
                self.augment_gone = None
                self.handle_augment(scr)
                self.publish(scr, False)         # the overlay too: the model's card explanations (augment_cards)
                return
            if getattr(self, 'augment_open', None):
                self.augment_missed(scr)
            self.read_encounter(frame, scr)          # 1-1 opens with an encounter (no carousel in this set)
            if is_carousel(scr) and not scr.shop_visible:
                self.handle_carousel(frame, scr)
                self._car_overlay = True
                return
            if getattr(self, '_car_overlay', False):
                # the carousel is over: its ring and item labels leave the screen now, not at the next round
                # (run 185805: they stayed over the board after the carousel)
                self._car_overlay = False
                if self.overlay:
                    self.overlay.show(path=None)
            planning = self.phase.planning and scr.shop_visible
            self.tick += 1
            rp0 = self.round_plan
            self.rushing = bool(planning and not self.pilot and scr.timer is not None and scr.timer <= FIELD_FIRST_S
                                and rp0 is not None and rp0.moves_pending(self.roster))
            tac, tac_box = self.watcher.tac
            if planning:
                bench_seen, board_seen, places = B.bar_places(frame, exclude=tac_box)
                self.watch_lights(frame, scr)       # bench units don't walk back: no settling needed
                if not REPLAY and time.time() - getattr(self, '_notice_at', 0) > 3.0:
                    # the game's own notices ("... Rider was auto assigned! ..."): logged and passed on - data on
                    # what the game does by itself (the Sprykin rider, refused drops)
                    self._notice_at = time.time()
                    said = self.game_message(frame)
                    if said and said != getattr(self, '_notice_last', None):
                        self.news_event('game_notice', scr.stage, f'the game showed: "{said}"')
                    self._notice_last = said
                # just after a round flips units are still walking back from combat
                # units walk back for ~2.5s after a round flips; in the last second of
                # planning bench bars vanish as combat starts - neither is evidence
                settled = time.time() - self.round_started > 2.5 and scr.timer is not None and scr.timer > 1
                if not settled and scr.timer is not None and scr.timer > 1 \
                        and time.time() - getattr(self, '_bench_at', 0) > 0.3:
                    # containers from the round's first tick (a clear match counts at once; walking units do not
                    # matter to it): run 20261016-011010 2-2 moved Rakan onto a component anvil 1 s into the round,
                    # before the first bench check, so slot 1 looked free
                    self._bench_at = time.time()
                    self.bench_perception(frame, bench_seen, scr, planning=True, containers_only=True)
                if settled:
                    # every bench slot's state (unit / empty / container / unknown / obscured) before
                    # the roster is checked against it: containers occupy slots, and an unreadable
                    # slot is no evidence that a unit left it (not before the round settles: units
                    # walking back have no bar at their slot yet)
                    if time.time() - getattr(self, '_bench_at', 0) > 0.3:
                        self._bench_at = time.time()
                        self.bench_perception(frame, bench_seen, scr, planning=True)
                        if self.auto_open_container(scr):
                            return              # opened: its choice row is next
                    self.star_check(frame, scr, places)
                    before_auto = {u.id for u in self.roster.units if getattr(u, 'auto_placed', False)}
                    self.roster.stage = scr.stage
                    self.roster.reconcile(bench_seen, board_seen, scr.board_count,
                                          bench_kinds=[o['kind'] for o in self.bench_kinds] if self.bench_kinds else None)
                    self.reconciled_stage = scr.stage
                    for u in self.roster.units:
                        if getattr(u, 'auto_placed', False) and u.id not in before_auto:
                            self.news_event('game_moved_unit', scr.stage, f'{u} was moved onto the board by the '
                                                                          f'game (empty slot at combat start)')
                            self.reread = getattr(self, 'reread', set()) | {u.id}
                    for u in self.roster.lost:
                        self.news_event('unit_missing', scr.stage, f'{u.name} (unit {u.id}) is not where it was tracked '
                                                                   f'({u.missing}); kept as missing, not deleted')
                    for u in getattr(self.roster, 'taken', []):
                        self.log(ev='unit_taken', stage=scr.stage, unit=repr(u))
                        self.news_event('unit_taken', scr.stage, f'{u.name}{"*" * u.star} (unit {u.id}) left the board '
                                        f'without our action (the game counts one unit fewer and no bar is at its hex): '
                                        f'removed from the records')
                    for u in self.roster.units:
                        if u.unverified and u.id not in getattr(self, 'reread', set()) and not u.missing                                 and u.id not in getattr(self, 'reread_asked', set()):
                            # read it once (a fresh try budget: it may have been read before)
                            self.reread = getattr(self, 'reread', set()) | {u.id}
                            self.reread_asked = getattr(self, 'reread_asked', set()) | {u.id}
                            self.tries[u.id] = 0
                    if DATA_MODE and not DATA_ROLL:
                        # data games: every unit is read off its info panel once (a labelled crop each)
                        for u in self.roster.units:
                            if not getattr(u, 'labelled', False) and not u.missing                                     and u.id not in getattr(self, 'data_asked', set()):
                                self.data_asked = getattr(self, 'data_asked', set()) | {u.id}
                                self.reread = getattr(self, 'reread', set()) | {u.id}
                                self.tries[u.id] = 0
                    # items over every unit, read off the screen (never from what a drag intended)
                    if not self.rushing and time.time() - getattr(self, '_unit_items_at', 0) > 0.6:
                        self._unit_items_at = time.time()
                        self.read_roster_items(frame, places, scr)
                        self.check_carousel_unit(scr)
                    # every unit still looks like itself (a wrong name never shows in occupancy)
                    if not self.rushing and time.time() - getattr(self, '_looks_at', 0) > 2.5:
                        self._looks_at = time.time()
                        self.check_looks(frame.copy(), places, scr)
                    bc = getattr(self, 'buy_check', None)
                    if bc and time.time() > bc:
                        self.buy_check = None
                        tracked = sorted(u.loc[1] for u in self.roster.units if not u.on_board)
                        seen_slots = [i for i, b in enumerate(bench_seen) if b]
                        if tracked != seen_slots:
                            if self.pilot:
                                self.need_survey = f'bench slots tracked {tracked} != on screen {seen_slots}'
                            # model player: no re-reading of known units - reconcile() settles the
                            # slots from persistent reads (arrival -> one read; a moved unit -> moved)
                            self.news_event('roster_mismatch', scr.stage,
                                            f'after a buy the tracked bench slots {tracked} != slots with a unit on '
                                            f'screen {seen_slots} (roster: {self.roster})')
                    # pilot only: a survey re-reads every unit (4.5-16 s, blocking). The model player
                    # never does: its roster comes from confirmed actions + game rules + one read
                    # per new arrival (the user, 2026-10-02: a human never clicks every champion)
                    if self.pilot and self.check_roster(frame, scr):
                        return                  # the roster was rebuilt from the screen: decide afresh
                    if not self.pilot and not self.rushing and self.board_check(frame, scr, places):
                        return                  # a wrong board record was corrected from the screen
                detail = read_item_bench(frame, detail=True)
                items = [d[0] for d in detail]
                self.item_crops = [d[2] for d in detail]
                self.item_sigs = [self.item_sig(frame, i) if it else None for i, it in enumerate(items)]
                # a name read off the tooltip (text) for this very icon beats the picture match
                texts = [bool(sig) and sig in self.item_names for sig in self.item_sigs]
                said = getattr(self, 'item_names_model', {})       # the model's labels: below a tooltip read
                items = [self.item_names.get(sig, said.get(sig, it)) if sig else it for it, sig in zip(items, self.item_sigs)]
                prev = self.prev_items or [None] * len(items)
                raw_prev, self._raw_items = getattr(self, '_raw_items', None), items
                if items == raw_prev:            # a stable read: the tracker takes what arrived or was read as text
                    what = self.item_track.sync(items, texts)
                    if what:
                        self.log(ev='item_bench', stage=scr.stage, change=what, now=self.item_track.names())
                tracked = self.item_track.names()
                if [n is not None for n in tracked] == [n is not None for n in items]:
                    items = tracked              # same slots filled: the tracked names (text, our own gives) hold
                # act only on slots read the same twice in a row, and never re-hand an item this round
                scr.items = [it if it == pv and (i, it) not in self.items_done else None
                             for i, (it, pv) in enumerate(zip(items, prev))]
                self.prev_items = items
                self.bench_items = items
            elif scr.shop_visible and time.time() - getattr(self, '_bench_at', 0) > 1.0:
                # combat: containers can arrive now too (rewards); only containers count here -
                # on an opponent's arena every free tile reads 'unknown'
                self._bench_at = time.time()
                self.bench_perception(frame, B.occupancy(frame, exclude=tac_box)[0], scr, planning=False)
                if self.auto_open_container(scr):
                    return
            if not planning and scr.shop_visible:
                self.check_autofill(frame, scr)
            looting = not self.rushing and self.collect_loot(frame, scr, tac)
            if looting:
                self._parked = None                  # collecting moved it: parked again when the loot is done
            elif planning and not self.rushing:
                self.park_tactician(scr)
            # a unit that arrived on the bench without our buy ('?', loot, a reward) is read at once: a local
            # click, not the model's time (the user, 2026-10-09); so is one on a board hex with no record
            # ... except before this round's plan is asked for: the question goes out first and the reads run
            # while the model thinks (run 20261010-010345 1-3: three loot units read first, ~6 s, and the plan came
            # after the 15 s stage-1 round had ended)
            ask_first = planning and self.planner_on() and not self.pilot and self.plan_asked != scr.stage
            # ... and once this round's plan is here, its steps come first: arrivals are read when it has nothing to do
            # right now (run 20261012-194821 3-3: twelve failed reads of three anvils, ~1.8 s each, held the plan's
            # Wisp buy and BFF link until the last second)
            rp = self.round_plan
            plan_here = planning and rp is not None and rp.stage == scr.stage
            if scr.shop_visible and not DATA_ROLL and not ask_first and not plan_here and not self.rushing and \
                    self.identify_one(scr, combat=not planning, arrivals=True):
                return
            if not DATA_ROLL and not self.rushing and self.read_new_unit(frame, scr):
                return
            t1 = time.perf_counter()
            act = self.decide(scr, planning, frame)
            t2 = time.perf_counter()
            self.publish(scr, planning)
            if act is None:
                # item tooltips (9-14 s of OCR each) only outside planning: they delayed the
                # round's plan request until the round was nearly over
                # a plan step waiting on a fresh read (an item bench settling after a drop) comes
                # first: a 1-3 s identification click in between made run25's items take 9 s
                rp = self.round_plan
                busy = rp is not None and not rp.carry and rp.pending(self.roster)
                if plan_here and scr.shop_visible and not DATA_ROLL and not self.rushing and \
                        self.identify_one(scr, arrivals=True):
                    return                      # the plan has nothing to do this tick: an arrival is read now
                if planning and not self.pilot and self.idle_survey(scr):
                    return                      # a board check, in idle time only (stops when a plan comes)
                # reads only in time nobody else needs: while the model thinks (planning), or in
                # combat; never while plan steps are waiting, never before the model is asked
                thinking = getattr(self, 'plan_pending', False) or self.pilot or (DATA_MODE and not DATA_ROLL)
                # (a hex the model asked about is read as soon as the plan's steps are done, too)
                if planning and not busy and thinking and self.identify_one(scr):
                    return
                if not planning and scr.shop_visible and not DATA_ROLL and self.identify_one(scr, combat=True):
                    return
                # unidentified item-bench icons: read in combat, while the model thinks, and AT ONCE
                # when a plan step is waiting for an item that may be one of them
                wanted = getattr(self, 'identify_items_now', False) or (rp is not None and getattr(rp, 'need_identify', False))
                if (not planning or thinking or wanted) and self.identify_item():
                    self.identify_items_now = False
                    if rp is not None:
                        rp.need_identify = False
                    return
                time.sleep(0.03 if looting else 0.1)
                return
            line = f'{scr.stage} t={scr.timer} g={scr.gold}: {act}'
            if line != last_print:
                print(line)
                self._last_print = line
                self.live.event(scr.stage, act.kind, act.why)
            if act.kind == 'move' and 'position' in act.why:
                self.brain.position_moves += 1
            self.fail_cause, self.deferred = None, False
            act_rp = self.round_plan         # an update installed mid-action has already credited this step
            try:
                ok = self.execute(act, scr)
            except NotForeground:
                # the input was not sent (window not in front, or the cursor could not be placed):
                # the plan step is still to do
                if act_rp and act.why.startswith('plan'):
                    act_rp.unsent(act)
                self.log(ev='input_not_sent', stage=scr.stage, act=repr(act), diag=self.inp.diagnose())
                time.sleep(0.3)             # user tabbed out: wait, don't quit
                return
            if self.deferred:
                # a board move in the last second: nothing was sent and the step stays in the plan (carried to the
                # next planning phase) - no failure, the overlay lists it as queued (the user, 2026-10-10)
                if act.kind in ('sell', 'item') and act_rp is not None and act.why.startswith('plan'):
                    act_rp.unsent(act)           # handed out = counted done: it is still to do (in combat, if need be)
                row = (act.kind, act.why, None, self.fail_cause)
                ra = self.__dict__.setdefault('round_acts', [])
                if row not in ra:
                    ra.append(row)
                    self.log(ev='move_queued', stage=scr.stage, timer=scr.timer, act=repr(act))
                time.sleep(0.05)
                return
            t3 = time.perf_counter()
            self.lat.append((t1 - t0, t2 - t1, t3 - t2))
            self._acted_at = time.time()      # gold changes after this are ours, not the win gold
            self.log(ev='action', stage=scr.stage, timer=scr.timer, gold=scr.gold, level=scr.level,
                     hp=scr.hp, shop=scr.shop, act=repr(act), ok=ok, roster=repr(self.roster),
                     ms=dict(read=round((t1 - t0) * 1000), decide=round((t2 - t1) * 1000),
                             act=round((t3 - t2) * 1000)))
            self.__dict__.setdefault('round_acts', []).append(
                (act.kind, act.why, ok, act.arg.get('summary') if ok and act.kind == 'rolldown' and isinstance(act.arg, dict)
                 else self.fail_cause))
            if act_rp and act.why.startswith('plan'):
                act_rp.record(act, ok, self.fail_cause)
                if not ok:
                    self.news_event('step_failed', scr.stage,
                                    f'{act.kind}: {act.why}' + (f' - {self.fail_cause}' if self.fail_cause else ''))
            self.note_own_action(act, ok, scr)
            if act.why.startswith('auto-'):
                note = act.why[len('auto-'):] + ('' if ok else f' (failed: {self.fail_cause or "?"})')
                if ok and act.kind == 'buy':
                    # the unit it bought, so the model can sell it if it does not want it (the user, 2026-10-09)
                    name = act.why.split('auto-buy ', 1)[-1]
                    u = next((x for x in reversed(self.roster.units) if x.name == name and not x.missing), None)
                    note += f' (unit {u.id}, {u.name}{"*" * u.star}' if u else ' ('
                    note += ' - bought by your auto_buy order; it can be sold like any unit)'
                self.auto_done.append(note)
            if ok and act.kind == 'buy' and 'wisp ' in act.why:
                m = re.search(r'wisp (.+?)(?: \(|$)', act.why)
                if m:
                    self.unit_change(m.group(1), scr.stage)
                    self.expect_gifts(m.group(1), scr.stage)
            if not ok:
                self.brain.failed[(act.kind, str(act.arg))] += 1
                self.input_failed(act, scr, frame)

    # -- the model plans each round ----------------------------------------------
    PLAN_DEADLINE = 5        # seconds left when a missing plan lets local rules field units (moves only)

    def decide(self, scr, planning, frame):
        """The model's round plan when there is one; local rules only when it is
        late (or no model is connected). With a model, nothing spends gold or
        moves units except the plan: a local action while the plan is being
        made leaves the plan built on a stale state."""
        if self.pilot:
            return self.pilot_decide(scr, planning, frame)
        if DATA_MODE:
            return self.data_decide(scr, planning)
        if not self.planner_on():
            return self.brain.decide(scr, self.roster, planning)
        if not planning:
            return self.combat_decide(scr)
        if self.plan_asked != scr.stage and self.next_plan:
            self.start_next_plan(scr)                # a queued next_round plan runs first: no model call to wait for
        if getattr(self, 'trait_pending', None) and time.time() - self.round_started > 2.5:
            self.trait_check(frame, scr)             # a difference found when nothing could be read: first thing now
            return None
        act = self.stage1_buy(scr)
        if act:
            return act
        act = self.wisp_event(scr, frame)        # a Wisp: its own fast question; bought first on "buy"
        if act:
            return act
        rp = self.round_plan
        # ask on the first planning tick, with whatever this tick read: no waiting for unit reads,
        # gold or HP (the user: the model reads them off the screenshot; one screen read is
        # ~0.35 s and happens on every tick anyway). Run26 asked 3.5-19 s into the round.
        if self.plan_asked != scr.stage:
            if not self.start_next_plan(scr):
                self.request_plan(scr, frame, 'plan')
            rp = self.round_plan
        change = self.state_change(scr)          # keeps its baseline current in every case
        if change:
            # an update question for the model (WAKE) that never stops the running plan (at 3-5 a false
            # "new unit" that interrupted the plan stopped XP x8 and the roll-down)
            self.news_event('game_change', scr.stage, change)
        act = self.auto_action(scr)              # the model's standing orders: auto-buy / auto-sell
        if act:
            return act
        act = self.handover_action(scr)          # ... and items to a final holder that arrived
        if act:
            return act
        rp = self.round_plan
        wr = getattr(self, 'wisp_refreshed', None)
        if wr and not self.plan_pending and (scr.timer is None or scr.timer > 4):
            # a Wisp the model chose to buy rerolled the shop: it decides on the new shop (spending waits,
            # placing units goes on). A fast roll-down's own Wisp buys never come here: no question there
            self.wisp_refreshed = None
            self.log(ev='wisp_refresh_call', stage=scr.stage, what=wr)
            self.request_plan(scr, frame, 'update', f'{wr} - decide on the new shop', interrupt=True)
            rp = self.round_plan
        if self.pending_events and rp and not self.plan_pending and (scr.timer is None or scr.timer > 4):
            # an event (a container appeared, a unit arrived or left, a record corrected) wakes the model;
            # the running plan keeps going meanwhile
            ev = '; '.join(self.pending_events)
            self.log(ev='event_sent', stage=scr.stage, what=ev)
            self.request_plan(scr, frame, 'update', ev, interrupt=False)
        if rp:
            rp.hold_roll = getattr(self, 'wisp_hold', False)
        if rp and not self.plan_pending and self.review_due(rp, scr, plan_done=False):
            return None
        if rp and self.plan_pending and rp.p.get('_interrupted'):
            # waiting for the model's answer to the change: nothing that spends, but keep
            # placing units as the old plan wanted (else the round can end with a hole in the board)
            act = rp.next_action(scr, self.roster, self.team_cap(scr))
            return act if act and act.kind == 'move' else None
        if scr.level:
            self.last_level = scr.level
        if rp:
            self.inject_gives(rp)
            act = rp.next_action(scr, self.roster, self.team_cap(scr))
            if act:
                self.brain.status['plan'] = (f"{'CARRY-OVER ' if rp.carry else ''}"
                                             f"{'PRE-PLANNED ' if getattr(rp, 'pre', None) else ''}ACT {act}")
                return act
            if rp.hold_roll and rp.roll_pending():
                self.brain.status['plan'] = 'roll waits for the Wisp answer'
                return None                      # not finished: the roll is still to come
            # plan finished: the model is asked again only if it asked for it (next_review)
            if not self.plan_pending and self.review_due(rp, scr, plan_done=True):
                return None
            self.brain.status['plan'] = 'plan done' if not rp.carry else 'carry-over done, waiting for the plan'
            if not rp.carry:
                self.plan_idle(rp, scr)
            return None
        # no fallback (the user, 2026-10-02): a late plan is a design problem to fix, never covered
        # by local rules - at 4-3 in run26 their swaps undid the model's board
        self.brain.status['plan'] = 'waiting for the model'
        return None

    def pilot_decide(self, scr, planning, frame):
        """Pilot mode: name new units first (a comp unit from a carousel or an orb only
        counts once named), open containers, then the pilot's action."""
        if planning and getattr(self, '_streak_read', None) != scr.stage:
            st = self.reader.streak(frame)          # the economy plan counts streak gold
            if st:
                self._streak_read = scr.stage
                self.pilot.streak = st[0] if st[1] == 'win' else -st[0] if st[1] == 'loss' else 0
        if planning and self.checked_stage != scr.stage and time.time() - self.round_started < 9:
            self.brain.status['identify'] = 'checking the board against the trait panel first'
            return None                          # check_roster runs once the units have settled
        if planning:
            unknown = [u for u in self.roster.units if u.name is None and self.tries.get(u.id, 0) < 2]
            if unknown and (scr.timer is None or scr.timer > 6):
                self.brain.status['identify'] = f'naming {len(unknown)} unit(s) first'
                return None                  # the idle step right-clicks them
            slot = next((i for i, o in self.bench_objs.items() if o['kind'] == 'interactable'), None)
            if slot is not None and not self.choice and (scr.timer or 0) > 8 and \
                    (slot, scr.stage) not in getattr(self, '_opened', set()):
                self._opened = getattr(self, '_opened', set()) | {(slot, scr.stage)}
                return Action('container', slot, f"pilot: open the {self.bench_objs[slot]['cls']}")
        if scr.shop_visible and scr.gold is not None:
            self.hud_seen = scr
        act = self.pilot.decide(scr, self.roster, planning)
        if planning and not self.wisps_bought.get(scr.stage, 0):
            buy = self.pilot_wisp(frame, scr)    # asks about a new Wisp at once (the model takes seconds)
            act = act or buy                     # a Wisp may reroll the shop: the pilot's buys go first
        return act

    def pilot_wisp(self, frame, scr):
        """Wisps in the shop: the model decides buy or skip, one question per Wisp and round.
        Returns the buy once it said yes (and the gold is there)."""
        from tft_player.coach import _cli
        if not _cli()[0]:
            return None                          # no model, no Wisps
        for slot, name, cost in self.wisp_offers(frame, scr):
            key = (scr.stage, name)
            ans = self.wisp_answers.get(key)
            if ans is None:
                c = self.pilot_call
                if c and c['answer'] is None and time.time() - c['t0'] < 60:
                    return None                  # one question at a time
                self.wisp_answers[key] = 'asked'
                prompt = (f"DECISION: Wisp, stage {scr.stage} (~{scr.timer or '?'}s left). Shop slot {slot}: "
                          f"{name}, costs {cost if cost is not None else '?'} gold. Effect: "
                          f"{self.wisp_effects().get(name, 'not in the reference - see the screenshot')} "
                          f"(N = the value on the card). One Wisp can be bought per round.\n\n"
                          f"STATE\n{self.pilot_state(scr)}\n\n"
                          'Reply {"buy": true} or {"buy": false}, then {"why": "..."}.')

                def done(decision, key=key):
                    self.wisp_answers[key] = (decision or {}).get('buy') is True
                self.pilot_ask('wisp', f'{name} ({cost if cost is not None else "?"} gold)', prompt,
                               self.screen_jpeg(), done)
                return None
            if ans is True and scr.gold is not None and (cost or 0) <= scr.gold:
                self.wisp_answers[key] = False   # bought once
                return Action('buy', slot, f'wisp {name} for {cost} gold (the model: buy)')
        return None

    def wisp_effects(self):
        """Wisp name -> effect text (data/reference/wisps.md). Its tables differ: the effect column is
        the one headed 'Effect' or 'Card text' (the second table has the cost seen in between)."""
        if not hasattr(self, '_wisp_fx'):
            self._wisp_fx = {}
            path = Path(__file__).resolve().parent.parent / 'data' / 'reference' / 'wisps.md'
            col = 1
            for row in path.read_text(encoding='utf-8').splitlines() if path.exists() else []:
                if not row.startswith('|'):
                    continue
                cells = [c.strip() for c in row.strip().strip('|').split('|')]
                if cells[0] == 'Wisp':
                    col = next((k for k, c in enumerate(cells) if c.startswith(('Effect', 'Card text'))), 1)
                elif len(cells) > col and cells[0] != '---' and 'on the card in the screenshot' not in cells[col]:
                    self._wisp_fx[cells[0]] = cells[col]
        return self._wisp_fx

    def wisp_specials(self, scr, frame):
        """{shop slot: (description, cost)} for the Wisps in the shop: the description and cost read off
        the card (~0.05-0.25 s once per card, then cached by its pixels) - the game's own text, with its
        numbers, valid after any patch. The Wisp table only when no text is found on the card."""
        from tft_player.pilot import wisp_names
        from tft_player.screen import match_name
        if not hasattr(self, '_wisp_names'):
            self._wisp_names = wisp_names()
        specials = {}
        for i in range(5):
            if not (scr.shop_raw[i] and not scr.shop[i]):
                continue
            desc, cost = self.reader.special_card(frame, i) if frame is not None else ('', None)
            if desc:
                specials[i] = (f'card text: "{desc}"', cost)
                continue
            raw = scr.shop_raw[i]
            name = match_name(raw, self._wisp_names, cutoff=0.75)
            if not name and len(raw) >= 4:
                # a long title is cut off in the name box ("Bunch-o" for Bunch-o'-Belts)
                pre = [n for n in self._wisp_names if n.lower().startswith(raw.lower())]
                name = pre[0] if len(pre) == 1 else None
            fx = self.wisp_effects().get(name) if name else None
            specials[i] = ((f'{name} (Wisp table): "{fx}"' + (' (N = the value on the card in the screenshot)'
                                                              if re.search(r'\bN\b', fx) else '')
                            if fx else 'its text is on the card in the screenshot'), cost)
        return specials

    WISP_WAIT = 15.0     # s: the longest the plan's roll waits for a Wisp answer (~8 s to the first word at 4-3)

    def wisp_event(self, scr, frame):
        """A Wisp in the shop (planning): asked at once - a question in the game's conversation, ahead of any
        other waiting question (priority, like a choice screen) - with its card text and cost and the state,
        and bought before anything else on "buy". Sells, buys, XP, board and items go on meanwhile;
        only the roll waits (self.wisp_hold), at most WISP_WAIT s. A card counts as a Wisp once it read
        the same on two ticks in a row (a champion card mid-redraw reads as text that is no champion).
        Only in a round that runs a scheduled next_round script: otherwise the round's state call decides
        its Wisps first (execution_plan.wisps, bought by the plan). Returns the Wisp buy, or None."""
        q = self.__dict__.setdefault('wisp_q', {})
        prev = getattr(self, '_wisp_prev', None)
        self._wisp_prev = (scr.stage, list(scr.shop_raw), list(scr.shop))
        rp = self.round_plan
        scripted = rp is not None and getattr(rp, 'pre', None) and rp.stage == scr.stage
        if not scr.shop_visible:
            return None
        for i in range(5):
            raw = scr.shop_raw[i]
            if not scripted or not raw or scr.shop[i] or not prev or prev[0] != scr.stage or prev[1][i] != raw \
                    or prev[2][i]:
                continue
            key = (scr.stage, i, raw)
            if key not in q:
                q[key] = {'asked': time.time(), 'buy': None, 'done': False}
                self.ask_wisp(scr, frame, i, key)
        now, hold, act = time.time(), False, None
        for (stage, i, raw), w in q.items():
            if stage != scr.stage or w['done'] or w.get('via') == 'plan':
                continue                         # a state call's Wisp is bought by its plan
            here = scr.shop_raw[i] == raw and not scr.shop[i]
            if w['buy'] is None:
                hold = hold or (here and now - w['asked'] < self.WISP_WAIT)
            elif w['buy'] and here and act is None:
                w['done'] = True
                act = Action('buy', i, f'wisp {raw} (the model: buy)')
            else:
                w['done'] = True                 # skipped, or the card is gone
        self.wisp_hold = hold
        return act

    def ask_wisp_now(self, scr, frame, i):
        """A Wisp in a new shop during a roll n times: asked at once; the rest of the rolls waits for the
        answer (wisp_event holds the roll while it is open)."""
        fresh = self.reader.read(frame)
        fresh.stage = fresh.stage or scr.stage
        if not (fresh.shop_raw and fresh.shop_raw[i]) or fresh.shop[i]:
            return                               # a full read names a champion there: no Wisp
        key = (scr.stage, i, fresh.shop_raw[i])
        q = self.__dict__.setdefault('wisp_q', {})
        if key not in q:
            q[key] = {'asked': time.time(), 'buy': None, 'done': False}
            self.ask_wisp(fresh, frame, i, key)

    def wisps_answered(self, plan, stage):
        """A state call's reply decided the Wisps it was shown (execution_plan.wisps; the plan buys them)."""
        ws = {w.get('slot'): bool(w.get('buy')) for w in plan.get('wisps') or [] if isinstance(w, dict)}
        for (st, i, raw), w in getattr(self, 'wisp_q', {}).items():
            if st == stage and w.get('via') == 'plan' and w['buy'] is None:
                w['buy'], w['done'] = ws.get(i, False), True

    def ask_wisp(self, scr, frame, i, key):
        """The Wisp question (the game's conversation, priority): the card's text and cost and the state;
        {"buy": true|false} first."""
        from tft_player import planner
        from tft_player.coach import ask_model
        raw = scr.shop_raw[i]
        desc, cost = self.wisp_specials(scr, frame).get(i, ('its text is on the card in the screenshot', None))
        state = planner.state_text(scr, self.roster, self.brain.plan, self.bench_items, self.hp_log,
                                   {i: (desc, cost)}, scr.timer)
        state = '\n'.join(l for l in state.splitlines() if not l.startswith('Wisp in shop:'))
        prompt = (f"EVENT: WISP IN THE SHOP, stage {scr.stage}, ~{scr.timer if scr.timer is not None else '?'} s "
                  f"of planning left.\n"
                  f"Shop slot {i}: Wisp '{raw}', costs {cost if cost is not None else '?'} gold. {desc}\n"
                  f"Wisps bought this round: {self.wisps_bought.get(scr.stage, 0)}. Your round plan runs on meanwhile (sells, buys, XP, board, items); its roll waits "
                  f"for this answer.\n\nSTATE\n{state}\n\n"
                  'Reply {"buy": true} or {"buy": false} first, then {"why": "<one sentence>"}.')
        image = self.screen_jpeg() if 'screenshot' in desc else None      # the card's text was not read
        q, t0, got = self.wisp_q, time.time(), {}
        self.log(ev='wisp_asked', stage=scr.stage, slot=i, wisp=raw, cost=cost, desc=desc)
        print(f'{scr.stage}: Wisp {raw!r} in shop slot {i} ({cost if cost is not None else "?"} gold) -> asking the model')

        def first(obj):
            if isinstance(obj, dict) and 'buy' in obj and q[key]['buy'] is None:
                q[key]['buy'] = obj['buy'] is True
                got['t'] = time.time() - t0

        def run():
            try:
                text = ask_model(prompt, image, on_first=first, timeout=45, priority=True)
            except Exception as e:
                text = f'FAILED: {e}'
            if q[key]['buy'] is None:
                from tft_player.coach import json_objects
                for o in json_objects(text or ''):
                    first(o)
            if q[key]['buy'] is None:
                q[key]['buy'] = False            # no answer: not bought, and the roll goes on
            self.model_call('wisp', prompt, text, time.time() - t0, got.get('t'))
            print(f"[wisp {time.time() - t0:.1f}s] {'BUY' if q[key]['buy'] else 'skip'} {raw}: {str(text or '')[:140]}")
        threading.Thread(target=run, daemon=True).start()

    def wisp_offers(self, frame, scr):
        """[(slot, Wisp name, cost)] for the Wisps in the shop (a card whose text is no champion)."""
        from tft_player.pilot import wisp_names
        from tft_player.screen import match_name
        if not hasattr(self, '_wisp_names'):
            self._wisp_names = wisp_names()
        out = []
        for i in range(5):
            if scr.shop[i] is None and scr.shop_raw[i]:
                name = match_name(scr.shop_raw[i], self._wisp_names, cutoff=0.75)
                if name:
                    out.append((i, name, self.reader.shop_cost(frame, i)))
        return out

    def team_cap(self, scr=None):
        """Team slots: the level (the user, 2026-10-09), unless the game's own 'n/m' over the board shows another
        team size - a +team-size augment, a Tactician's Shield, Riftbeast 10 (the user, 2026-10-13). What it adds is
        kept for the frames that don't read it, and a level-up keeps it. The state shows the same number."""
        lvl = (scr.level if scr is not None else None) or self.last_level or 1
        size = getattr(scr, 'team_size', None) if scr is not None else None
        if size and scr.level:
            self.team_extra = size - scr.level
        cap = max(1, lvl + getattr(self, 'team_extra', 0))
        if scr is not None and getattr(scr, 'team_size', None) is None and getattr(self, 'team_extra', 0):
            scr.team_size = cap
        return cap

    def note_own_action(self, act, ok, scr):
        """What our own action changes: the shop (buy, roll), gold (sell) and the
        units. Anything else that changes is the game's doing."""
        self.expect = getattr(self, 'expect', {})
        if not ok:
            return
        now = time.time()
        if act.kind == 'buy':
            self.expect['shop'] = now
            self.buy_check = now + 1.5           # then: does the tracked bench match the screen?
            if scr.shop[act.arg] is None and scr.shop_raw[act.arg]:
                self.wisps_bought[scr.stage] = self.wisps_bought.get(scr.stage, 0) + 1
                # remember the rest of the shop and the gold: the Wisp matters to the plan only
                # if its effect changes them (Starting Town rerolls the shop; a Training Dummy doesn't)
                others = tuple(n for k, n in enumerate(scr.shop) if k != act.arg)
                self.pending_change = (f"Wisp '{scr.shop_raw[act.arg]}' bought", now, act.arg, others, scr.gold)
        elif act.kind == 'roll':
            self.expect['shop'] = now
        elif act.kind == 'sell':
            self.expect['gold'] = now

    def state_change(self, scr):
        """A significant change the model hasn't seen: a Wisp we bought, the shop
        changing without our buy/roll, gold rising without a sell, a unit we didn't
        buy. Returns a description (and resets), or None."""
        if not scr.shop_visible or scr.gold is None:
            return None
        cur = (tuple(scr.shop), scr.stage, scr.gold)
        last, self.last_seen = getattr(self, 'last_seen', None), cur
        if last and last[1] != cur[1]:
            last = None                  # a new round: income and the new shop are expected
        # our own buy/roll/sell explains a change seen within 2s of it (OCR can lag a tick)
        expect = {k for k, t in getattr(self, 'expect', {}).items() if time.time() - t < 2.0}
        why = []
        pc = getattr(self, 'pending_change', None)
        if pc and time.time() - pc[1] < 1.5:
            return None                  # let the Wisp's effect show (new shop, gold) first
        if pc:
            self.pending_change = None
            label, _, slot, others, gold0 = pc
            now_others = tuple(n for k, n in enumerate(scr.shop) if k != slot)
            effects = []
            if now_others != others:
                effects.append('the shop changed')
            if scr.gold is not None and gold0 is not None and scr.gold > gold0:
                effects.append(f'gold rose to {scr.gold}')
            if effects:
                why.append(f"{label}: {' and '.join(effects)}")
            if now_others != others:
                self.wisp_refreshed = f'{label}: it rerolled the shop'
            expect.add('shop')           # the Wisp's own slot showing its champion is not news
        if last and last[0] != cur[0] and 'shop' not in expect:
            why.append('the shop changed (not by our buy or roll)')
        if last and cur[2] > last[2] and 'gold' not in expect:
            why.append(f'gold went up {last[2]} -> {cur[2]} without a sell')
        known = set(getattr(self.round_plan, 'known', ()) or ())
        new = [u for u in self.roster.units
               if known and u.id not in known and not u.ours and u.id not in self.announced]
        if new:
            self.announced |= {u.id for u in new}
            why.append(f"new unit(s) we didn't buy: {', '.join(map(repr, new))}")
        return '; '.join(why) or None

    def screen_extras(self, frame, scr):
        """Everything else the screen shows that the model should know, as state lines."""
        out = []          # the trait-panel check's note: the controller's own work, logged only
        try:
            # the player list is NOT read: its detection OCR took 1-11 s in the main loop before
            # every plan (1-3 and 1-4 on 2026-09-29 were asked with ~3 s left). The model reads
            # the standings from the screenshot it gets with every question.
            st = self.reader.streak(frame) if scr.shop_visible else None
            out.append(f'Streak: {st[0]} {st[1]}' if st else 'Streak: none shown')
            odds = self.reader.shop_odds(frame) if scr.shop_visible else None
            if odds:
                out.append(f'Shop odds by cost 1-5: {odds}')
        except Exception as e:
            self.log(ev='extras_error', error=repr(e))
        gl = self.gift_line(scr.stage)
        if gl:
            out.append(gl)
        objs = getattr(self, 'bench_objs', {})
        if objs:
            out.append('Bench containers (not units): ' + '; '.join(
                f"slot {k}: {o['cls']}" if o['kind'] == 'interactable'
                else f"slot {k}: unrecognised object - not a unit, not a known container"
                for k, o in sorted(objs.items())))
        try:
            if frame is not None and self.phase.planning and time.time() - self.round_started > 4.0:
                pl = B.bar_places(frame, exclude=self.watcher.tac[1])[2]
                seen = {l[1:] for _, l in pl if l[0] == 'board'}
                tracked = {u.loc[1:] for u in self.roster.board() if not u.missing}
                if seen != tracked:          # the controller checks it; not the model's to act on
                    self.log(ev='bars_differ', stage=scr.stage, extra=sorted(seen - tracked),
                             absent=sorted(tracked - seen))
        except Exception as e:
            self.log(ev='extras_error', error=repr(e))
        if self.encounter:
            out.append(f"This game's encounter (1-1 banner): {self.encounter}")
        out.append('Augments taken: ' + (', '.join(self.augments_taken) or 'none'))
        out.append(f'Wisps bought this round: {self.wisps_bought.get(scr.stage, 0)}')
        out.append(self.auto_line())
        return out

    def send_briefing(self, prep=None):
        """Rules, traits, items, augment tiers and every MetaTFT comp. prep: the preparation sent as we entered
        the queue - its answer is taken when it comes; without it (the game was already running) or when it
        failed, the briefing goes out now."""
        from tft_player.coach import ask_model, briefing_text
        if prep is not None:
            prep.join()
            r = prep.result or {}
            if r.get('answer') is not None:
                answer = r['answer'] + (f"\n[cut off: {r['cut']}]" if r.get('cut') else '')
                return self.take_briefing(r['prompt'], answer, r['seconds'], at=r['t'], meta=r.get('meta'))
            print(f"preparation failed ({r.get('error')}): the briefing goes now", flush=True)
        t0 = time.time()
        text, images = briefing_text(), None     # fixed comp: its MetaTFT page as text, no screenshots
        if self.forced:
            self.pages_sent.add(self.forced.name)
        try:
            answer = ask_model(text, images)
        except Exception as e:
            answer = f'FAILED: {e}'
        self.take_briefing(text, answer, time.time() - t0)

    def take_briefing(self, text, answer, latency, at=None, meta=None):
        """The briefing's answer: logged, and the trait choices it names are kept for the choice screen."""
        if self.forced:
            self.pages_sent.add(self.forced.name)
        self.model_call('briefing', text, answer, latency, at=at, meta=meta)
        print(f'[briefing {latency:.0f}s] {str(answer)[:80]}')
        from tft_player.coach import json_objects
        for obj in json_objects(answer if isinstance(answer, str) else ''):
            if isinstance(obj, dict) and isinstance(obj.get('trait_choices'), dict):
                self.trait_choices = obj['trait_choices']
                self.log(ev='trait_choices', choices=self.trait_choices)
            if isinstance(obj, dict) and isinstance(obj.get('strategy'), dict):
                self.update_comps(obj['strategy'], self.phase.stage)        # flex: P(comp) from the preparation

    KEY_STAGES = ()          # reminders to revise a game plan: off (no game plan is asked for)

    def update_comps(self, st, stage=None):
        """Flex: the model's P over comps (the overlay shows it until a commit) and its commit ("commit": a comp
        name) - from then on the game is a fixed-comp game on that comp (commit_comp). Before a commit no comp is
        the target: nothing is bought, sold or handed over for a comp the model has not chosen."""
        if self.forced or not isinstance(st, dict):
            return                       # fixed or committed comp: no P, nothing to choose
        self.take_comp_p(st, stage)
        cm = st.get('commit')
        if isinstance(cm, str) and cm.strip() and cm.strip().lower() not in ('null', 'none'):
            self.commit_comp(cm.strip(), stage)

    def take_comp_p(self, st, stage):
        """The reply's P over comps, normalised (comps + other = 1), for the overlay and the log."""
        if not isinstance(st.get('comps'), list):
            return
        ps = {}
        for c in st['comps']:
            if isinstance(c, dict) and c.get('name'):
                try:
                    ps[str(c['name'])] = max(0.0, float(c.get('p') or 0))
                except (TypeError, ValueError):
                    pass
        other = st.get('other')
        total = sum(ps.values()) + (float(other) if isinstance(other, (int, float)) else 0.0)
        if not ps or total <= 0:
            return
        p = {n: q / total for n, q in ps.items()}          # normalised: comps + other = 1
        if p != self.comp_p:
            self.comp_p = p
            self.log(ev='comp_p', stage=stage, p=p)

    def commit_comp(self, name, stage):
        """Flex: the model's commit. The comp becomes this game's fixed comp, as if the game had started with it
        (tools/play_comp.py --comp): target comp, auto_buy to its units, its brief ahead of the next question, the
        fixed-comp prompts and panels from then on. A later change of comp is a pivot."""
        from tft_player.meta import META
        comp = META.by_name(name)
        if comp is None:
            self.news_event('commit_refused', stage, f'"commit": no MetaTFT comp is named {name!r} - nothing is '
                                                     f'committed; use the name exactly as in the briefing')
            return
        self.forced, self.target, self.committed = comp, comp.name, stage
        os.environ['TFT_FORCE_COMP'] = comp.name         # forced_comp(): every prompt and panel follows it
        self.auto_buy = {n: 2 for n in comp.units if CATALOG.unit(n)}
        self.set_comp(comp.id, 'commit')
        self.log(ev='commit', stage=stage, comp=comp.name, comp_id=comp.id, p=self.comp_p)
        print(f'{stage}: COMMIT {comp.name} (P {self.comp_p})', flush=True)
        self.news_event('commit', stage, f'you committed to {comp.name}: it is the target comp from now on (auto_buy: '
                                         f'its units to 2-star; its MetaTFT page comes with the next question)')
        threading.Thread(target=self.load_commit_brief, args=(comp,), daemon=True).start()

    def load_commit_brief(self, comp):
        """The committed comp's brief - the fixed-comp briefing (its MetaTFT page as text, trait choices) - for the
        next question; our own text page of it when the site's page cannot be read."""
        from tft_player import coach
        t0 = time.time()
        try:
            text = coach.forced_briefing()
        except Exception as e:
            text = f'Target comp for this game: {comp.name} ({comp.id}).'
            self.log(ev='commit_brief_error', comp=comp.name, error=repr(e))
        if "MetaTFT page" not in text:
            f = coach.REFERENCE / 'metatft' / 'comps' / f'{comp.id}.md'
            if f.exists():
                text += '\n\nIts MetaTFT page as text:\n' + f.read_text(encoding='utf-8')
        if self.forced is comp:
            coach.commit_brief(text)
            self.pages_sent.add(comp.name)
            self.log(ev='commit_brief', comp=comp.name, chars=len(text), secs=round(time.time() - t0, 1))

    def fetch_pages(self, name):
        """The target comp's MetaTFT pages (5 tabs), captured in the background."""
        from tft_player.metatft_shots import comp_pages
        t0 = time.time()
        try:
            comp_pages(name)
            from tft_player.metatft_shots import comp_brief
            text = comp_brief(name)          # the page as text; the screenshots repeat it
        except Exception as e:
            self.log(ev='pages_error', comp=name, error=repr(e))
            print(f'MetaTFT pages for {name} failed: {e}')
            # MetaTFT's list hides some low-play comps: send our text page of it instead
            from tft_player.coach import REFERENCE
            from tft_player.meta import META
            comp = META.by_name(name)
            f = REFERENCE / 'metatft' / 'comps' / f'{comp.id}.md' if comp else None
            if f and f.exists() and name == self.target and not self.forced:     # fixed comp: screenshots only
                self.pending_pages = (name, f.read_text(encoding='utf-8'))
            return
        if name == self.target:
            self.pending_pages = (name, text)
            self.log(ev='pages_ready', comp=name, secs=round(time.time() - t0, 1))

    def direction_lines(self):
        """Flex, before a commit: what we hold - units, items, emblems, augments - matched against every MetaTFT comp
        (tft_player/flex.py; experiments/README.md section 8: with it the model chose the comps the data supports
        and held back its commit when nothing pointed anywhere)."""
        from tft_player.flex import direction_block
        units = [(u.name, u.star or 1) for u in self.roster.units if u.name and u.info and not u.missing]
        items = ([i for i in (self.bench_items or []) if i and i != 'unknown'] +
                 [i for u in self.roster.units if not u.missing for i in (u.items or []) if i and i != 'unknown'])
        try:
            return ['', direction_block(units, items, list(self.augments_taken or []))]
        except Exception as e:
            self.log(ev='direction_error', error=repr(e))
            return []

    def comps_line(self):
        if not self.comp_p:
            return 'Comp probabilities: none given yet.'
        ps = ', '.join(f'P({n}) = {p:.2f}' for n, p in sorted(self.comp_p.items(), key=lambda kv: -kv[1]))
        return f'Your comp probabilities (last reply): {ps}. Not committed yet ("commit" in strategy).'

    def inherit_carry(self, plan):
        """Last round's unfinished steps are never dropped by a newer plan (the user: they run first and are
        completed): a plan without a board or items keeps the carried ones; carried sells and keeps stay."""
        carry = self.round_plan.p if self.round_plan is not None and self.round_plan.carry else None
        if not carry:
            return
        if plan.get('board') is None and carry.get('board'):
            plan['board'] = carry['board']
        if plan.get('bench') is None and carry.get('bench'):
            plan['bench'] = carry['bench']
        if carry.get('items'):
            have = {(e.get('component'), e.get('id')) for e in plan.get('items') or []}
            plan['items'] = list(plan.get('items') or []) + [e for e in carry['items']
                                                             if (e.get('component'), e.get('id')) not in have]
        plan['sell'] = list(dict.fromkeys(list(plan.get('sell') or []) + list(carry.get('sell') or [])))
        plan['keep'] = list(dict.fromkeys(list(plan.get('keep') or []) + list(carry.get('keep') or [])))

    def start_next_plan(self, scr):
        """The model's next_round plan (written last planning phase) runs as this round's plan at
        once, with no model call; the model is asked when it completes (or at its own next_review)."""
        from tft_player import planner
        if not self.next_plan or self.next_plan[0] == scr.stage:
            return False
        made, plan = self.next_plan
        self.next_plan = None
        plan = dict(plan)
        plan.setdefault('next_review', [{'plan_complete': True}])
        # last round's unfinished steps go first and are never dropped (the user: 4-2's lineup and items
        # were still undone at 4-6)
        self.inherit_carry(plan)
        self.plan_asked = scr.stage
        self.round_plan = planner.RoundPlan(plan, scr.stage, known=[u.id for u in self.roster.units],
                                            names=self.next_plan_names, gold=scr.gold)
        self.round_plan.pre = made
        self.apply_auto(plan, scr.stage)
        self.plan_rejected(planner.validate(self.round_plan.p, self.roster, self.bench_items, self.round_plan.names), scr.stage)
        self.brain.protected = self.round_plan.protected_ids()
        self.news_event('next_round_started', scr.stage, f'your next_round plan from {made} ran from the start of this '
                                                         f'planning phase (gold {scr.gold}, level {scr.level})')
        self.log(ev='next_round_started', stage=scr.stage, made=made, plan=plan)
        print(f'{scr.stage}: running the next_round plan written at {made}')
        return True

    def keep_next_plan(self, plan, stage):
        nr = plan.get('next_round') if isinstance(plan, dict) else None
        if not isinstance(nr, dict):
            return
        from tft_player import planner
        nr = planner.normalize({'execution_plan': nr})
        # item-bench presence is checked when it starts, against that round's bench (run27 3-1)
        self.plan_rejected(planner.validate(nr, self.roster, None, self.unit_names()), stage, 'next_round ')
        nr.pop('strategy', None)
        self.next_plan = (stage, nr)
        self.next_plan_names = self.unit_names()
        self.log(ev='next_round_planned', stage=stage, plan=nr)
        print(f'{stage}: next_round plan kept for the next planning phase: {json.dumps(nr)[:200]}')

    def request_plan(self, scr, frame, kind='plan', change=None, interrupt=True):
        """kind 'plan': the round's first plan (with last round's leftovers);
        'update': a re-plan with the progress so far (and what changed)."""
        from tft_player import planner
        from tft_player.coach import ask_model
        self.plan_asked = scr.stage
        if not getattr(self, 'rushing', False):
            self.trait_check(frame, scr)     # the board's identities against the trait panel, fixed before asking
        self.plan_pending = True
        self.plan_calls += 1
        self.pending_events = []         # the state below lists what they reported
        # no Wisps in combat: they leave the shop when it starts
        specials = self.wisp_specials(scr, frame) if self.phase.planning else {}
        wq = self.__dict__.setdefault('wisp_q', {})
        decided = {k[1]: ('buy' if w['buy'] else 'skip') for k, w in wq.items()
                   if k[0] == scr.stage and w['buy'] is not None and k[1] in specials and k[2] == scr.shop_raw[k[1]]}
        for i in specials:
            if i not in decided:                 # this reply decides it (execution_plan.wisps): no own question
                wq[(scr.stage, i, scr.shop_raw[i])] = {'asked': time.time(), 'buy': None, 'done': False, 'via': 'plan'}
        phase = 'planning' if self.phase.planning else 'combat'
        state = planner.state_text(scr, self.roster, self.brain.plan, self.bench_items, self.hp_log,
                                   specials, scr.timer, self.screen_extras(frame, scr)
                                   + ([] if self.forced else [self.comps_line()])      # fixed comp: no P line
                                   + self.strategy_lines()
                                   + ([] if self.forced or phase != 'planning' else self.direction_lines()),
                                   phase=phase, wisps_decided=decided)
        if kind == 'update':
            rp = self.round_plan
            progress = rp.progress() if rp else 'no plan yet'          # failures here; "Not done" without them
            if change:
                progress = f'CHANGED: {change}. Plan so far - {progress}'
                if rp and interrupt:
                    rp.p['_interrupted'] = True
            prompt = planner.update_prompt(state, progress, rp.leftover(self.roster, failed=False)[1] if rp else '')
            if phase == 'combat':
                prompt = prompt.replace('DECISION: same planning phase, update.',
                                        'DECISION: update during COMBAT (see the first STATE line).', 1)
                if str(change or '').startswith('COMBAT STARTED'):
                    # the combat-start call: the interest decision, as numbers, before the state
                    base = getattr(self, 'fight_base', None) or {}
                    at = (types.SimpleNamespace(gold=base['gold'], stage=scr.stage)       # gold at combat start: a
                          if base.get('stage') == scr.stage and base.get('gold') is not None else scr)   # fast win's +1
                    check = planner.interest_check(at, self.roster)
                    if check:          # it replaces the state's Economy line (the same numbers)
                        prompt = '\n'.join(l for l in prompt.split('\n') if not l.startswith('Economy: '))
                        prompt = prompt.replace('\n\nSTATE\n', f'\n\n{check}\n\nSTATE\n', 1)
                    # ... and the strategy rewritten every combat (the user, 2026-10-13: run 233635 kept its queue-time
                    # strategy all game, though from 3-7 it played otherwise)
                    if self.game_strategy.get('text'):
                        prompt += ('\nSTRATEGY REWRITE (every combat): send {"game_strategy": {"text": ..., "why": ...}} '
                                   'in this reply - your whole strategy paragraph as it stands now, after what this game '
                                   'has shown; "why": what changed, or "unchanged".')
                        self.strategy_due = scr.stage
        else:
            prompt = planner.plan_prompt(state, first=not getattr(self, 'planned_once', False),
                                         carry=planner.carry_text(self.carry_note, self.brain.protected))
            self.planned_once = True
        news = self.take_news()
        if news:                             # what happened since the model last decided
            first, _, rest = prompt.partition('\n')
            prompt = f'{first}\nEVENTS SINCE YOUR LAST DECISION: {news}\n{rest}'
        stage, t0 = scr.stage, time.time()
        shop_at_plan = list(scr.shop)
        known = [u.id for u in self.roster.units]
        names, gold0 = self.unit_names(), scr.gold       # what each id meant when the model saw the state
        lvl0 = (scr.level or self.last_level, scr.xp)    # a plan that levels up must field the new slot
        image = self.screen_jpeg()
        old_rp = self.round_plan if kind == 'update' else None
        snap = old_rp.snapshot() if old_rp is not None else None     # what the old plan had done when asked

        def credit(new):
            """The old plan kept running while the model wrote this update: its steps since the prompt are done."""
            if old_rp is not None and old_rp is not new:
                got = new.credit(old_rp, snap)
                if any(got.values()):
                    self.log(ev='update_credited', stage=stage, **got)

        early = {}

        def install(obj):
            """The first answer (execution_plan) runs the moment it is written;
            the strategy is still being written."""
            plan = planner.normalize(obj) if isinstance(obj, dict) and 'execution_plan' in obj else None
            if plan and stage in (self.phase.stage, getattr(self, 'stage_seen', None)):
                errs = planner.validate(plan, self.roster, self.bench_items, names, level=lvl0)
                self.apply_plan_meta(plan, stage)
                self.inherit_carry(plan)
                self.round_plan = planner.RoundPlan(plan, stage, shop_at_plan, known=known, names=names, gold=gold0)
                credit(self.round_plan)
                self.plan_rejected(errs, stage)
                self.brain.protected = self.round_plan.protected_ids()
                self.plan_pending = False
                self.wisps_answered(plan, stage)
                early['t'] = time.time() - t0
                print(f'[{kind} plan in {early["t"]:.1f}s: executing while the strategy is written]')
                # the panel shows the decision now, not when the whole reply is done (run23: it
                # looked frozen for the 10-20 s the reasoning took)
                self.live.model_call(kind, stage, prompt, json.dumps(obj), early['t'])

        def run():
            try:
                text = ask_model(prompt, image, on_first=install)
            except Exception as e:
                text = f'FAILED: {e}'
            plan = planner.parse(text)
            full_errs = planner.validate(plan, self.roster, self.bench_items, names, level=lvl0) if plan else []
            # (the early install reported its own errors; the full answer only when it installs)
            from tft_player.coach import json_objects
            for o in json_objects(text):
                if isinstance(o, dict) and 'deliberation' in o:
                    d = o['deliberation']
                    self.last_ledger = (stage, json.dumps(d, ensure_ascii=False)[:1200] if not isinstance(d, str) else d[:1200])
            if plan and stage in (self.phase.stage, getattr(self, 'stage_seen', None)):
                if not early:            # no streamed first answer: install the whole reply now
                    self.inherit_carry(plan)
                    self.round_plan = planner.RoundPlan(plan, stage, shop_at_plan, known=known, names=names, gold=gold0)
                    credit(self.round_plan)
                    self.brain.protected = self.round_plan.protected_ids()
                    self.plan_rejected(full_errs, stage)
                elif self.round_plan is not None:
                    self.round_plan.p['strategy'] = plan.get('strategy')
                    # the early-installed plan must equal the complete answer (run26: a streamed plan
                    # cut to 2 of 7 board units ran the whole round): correct the lineup parts
                    for k in ('board', 'items', 'strip', 'keep'):
                        if plan.get(k) is not None and plan.get(k) != self.round_plan.p.get(k):
                            self.log(ev='plan_corrected', stage=stage, key=k)
                            self.round_plan.p[k] = plan[k]
                self.update_comps(plan.get('strategy'), stage)
                self.apply_plan_meta(plan, stage)
                self.keep_next_plan(plan, stage)
                self.wisps_answered(plan, stage)
            self.plan_pending = False
            self.model_call(kind, prompt, text, time.time() - t0, plan_s=early.get('t'))
            print(f'[{kind} {time.time() - t0:.0f}s] {(plan or {}).get("note") or (text or "")[:200]}')
            for w in (plan or {}).get('wisps') or []:
                if isinstance(w, dict):
                    print(f"  wisp slot {w.get('slot')}: {'BUY' if w.get('buy') else 'skip'} - {w.get('why')}")
        threading.Thread(target=run, daemon=True).start()

    def apply_plan_meta(self, plan, stage):
        """What the model's reply sets besides this round's actions: the components it wants
        from carousels and orbs (kept until it changes them), a pivot to another comp (with
        its reason), and a revised game plan (logged)."""
        from tft_player.items import COMPONENTS
        t = plan.get('item_targets')
        if isinstance(t, list):
            want = [i for i in t if i in COMPONENTS]
            if want != self.brain.plan.item_targets:
                self.brain.plan.item_targets = want
                self.log(ev='item_targets', stage=stage, targets=want, asked=t)
        self.apply_identify(plan.get('identify'), stage)
        self.apply_auto(plan, stage)
        st = plan.get('strategy')
        if not isinstance(st, dict):
            return
        if st.get('game_plan'):
            self.log(ev='game_plan', stage=stage, text=st['game_plan'])
        pv = st.get('pivot')
        if not (self.forced and isinstance(pv, dict) and pv.get('to')):
            return
        if not self.committed:                                           # the chosen comp is played all game
            # (a flex game chose its comp itself: it may change it)
            if stage not in self.pivoted:
                self.pivoted.add(stage)
                self.news_event('pivot_refused', stage, f'comp changes are switched off in this harness mode: '
                                                               f'the target comp stays {self.forced.name}')
            return
        from tft_player.meta import META
        comp = META.by_name(str(pv['to']))
        if not comp:
            self.news_event('pivot_refused', stage,
                            f"no MetaTFT comp named {pv.get('to')!r}: still playing {self.forced.name}")
        elif comp.id != self.forced.id and stage not in self.pivoted:
            self.pivoted.add(stage)
            old, self.forced = self.forced, comp
            os.environ['TFT_FORCE_COMP'] = comp.name
            self.target = comp.name
            self.auto_buy = {n: 2 for n in comp.units if CATALOG.unit(n)}
            self.set_comp(comp.id, 'pivot')
            self.log(ev='pivot', stage=stage, old=old.name, new=comp.name, why=pv.get('why'))
            print(f"{stage}: PIVOT {old.name} -> {comp.name}: {pv.get('why')}")
            self.news_event('pivot', stage, f"you changed comp from {old.name} to {comp.name} ({pv.get('why')}); "
                                            f"auto_buy: its units to 2-star; its MetaTFT page comes with the next question")
            threading.Thread(target=self.load_commit_brief, args=(comp,), daemon=True).start()

    def apply_identify(self, ids, stage):
        """The model's names for units shown as "?" and item slots shown as "unknown" (read off the
        screenshot): perception that failed is fixed by the model, never left unnamed."""
        from tft_player.screen import ScreenReader
        for e in ids if isinstance(ids, list) else []:
            if not isinstance(e, dict) or not isinstance(e.get('name'), str):
                continue
            if 'unit' in e:
                u = next((x for x in self.roster.units if x.id == e['unit']), None)
                info = CATALOG.unit(e['name'])
                if u is not None and info and (u.name is None or u.name != info.name):
                    old = u.name
                    u.name = info.name
                    if isinstance(e.get('star'), int) and 1 <= e['star'] <= 3:
                        u.star = e['star']
                    # used at once, but a guess off the screenshot until its panel confirms it: read again in idle
                    # time (run 20261015-161625 2-6/2-7: the model named a unit Kayle, Veigar, then Kayle again)
                    u.unverified = 'named by the model from the screenshot'
                    self.reread = getattr(self, 'reread', set()) | {u.id}
                    self.tries[u.id] = 0
                    self.log(ev='model_identified', stage=stage, unit=u.id, old=old, name=u.name, star=u.star)
                    print(f'{stage}: the model named unit {u.id}: {u.name} {"*" * u.star}')
            elif isinstance(e.get('item_slot'), int) and 0 <= e['item_slot'] <= 9:
                name = next((n for n in ScreenReader.item_names() if n.lower() == e['name'].lower()), None)
                sig = (getattr(self, 'item_sigs', None) or [None] * N_SLOTS)[e['item_slot']]
                if name and sig is not None:
                    # the model's name is a label until the tooltip is read, never a text read: run 20261015-132144
                    # 5-5 named a carousel item "Kraken's Fury" (it went for one), the give skipped the tooltip
                    # check, and Nidalee got the Inferno Emblem it really was
                    self.__dict__.setdefault('item_names_model', {})[sig] = name
                    self.log(ev='model_identified', stage=stage, item_slot=e['item_slot'], name=name)
                    print(f'{stage}: the model named item slot {e["item_slot"]}: {name}')
                elif not name:
                    self.news_event('identify_failed', stage, f'item slot {e["item_slot"]}: "{e["name"]}" is not '
                                                              f'an item name in the game data - the slot stays unknown')

    @staticmethod
    def planner_on():
        """The model plans rounds (and owns the target comp) when one is connected."""
        from tft_player.coach import _cli
        return bool(_cli()[0])

    def set_comp(self, cid, source):
        """The target comp is part of the game state; the model changes it."""
        from tft_player.meta import META
        comp = META.comps.get(str(cid)) if cid else None
        if comp and comp.id != self.brain.plan.comp:
            old = self.brain.plan.comp
            self.brain.plan.adopt(comp)
            print(f'target comp: {old} -> {comp.id} {comp.name} ({source})')
            self.log(ev='comp_change', old=old, new=comp.id, name=comp.name, source=source)

    def review_round(self, scr):
        """Combat started: show the model the board we fight with (runs during combat)."""
        from tft_player import planner
        from tft_player.coach import _cli, ask_model
        rp = self.round_plan
        if not rp or not _cli()[0]:
            return
        if self.forced:
            return       # fixed comp: no target to change, pages already in the briefing - nothing to review
        state = planner.state_text(scr, self.roster, self.brain.plan, self.bench_items, self.hp_log, {}, 0)
        prompt = planner.review_prompt(state, rp.p, rp.done, rp.failed)
        image, t0 = self.screen_jpeg(), time.time()
        if self.target and self.target not in self.pages_sent and not self.pending_pages                 and time.time() - getattr(self, 'pages_tried', 0) > 60:
            self.pages_tried = time.time()          # an earlier capture failed: try again
            threading.Thread(target=self.fetch_pages, args=(self.target,), daemon=True).start()
        if self.pending_pages and isinstance(self.pending_pages[1], str):
            name, page = self.pending_pages          # no page on the site: our text page
            self.pending_pages = None
            self.pages_sent.add(name)
            prompt = f'MetaTFT data of the target comp, {name}:\n{page}\n\n' + prompt

        def run():
            try:
                text = ask_model(prompt, image)
            except Exception as e:
                text = f'FAILED: {e}'
            self.model_call('review', prompt, text, time.time() - t0)
            r = planner.parse(text) or {}
            self.update_comps(r.get('strategy'), self.phase.stage)
            self.apply_plan_meta(r, self.phase.stage)
            print(f"[review {time.time() - t0:.0f}s] {(r.get('strategy') or {}).get('now') or ''} problems: {r.get('problems') or 'none'}")
        threading.Thread(target=run, daemon=True).start()

    def handle_end(self, frame, scr):
        """'done' once the client says the game is over. When we are out, leave:
        click the elimination screen's exit button, or Settings -> EXIT TFT.
        Leaving is only ever done on evidence we are eliminated - exiting a game
        we are still in would forfeit it."""
        from tft_player.ending import EXIT_TFT_BUTTON
        E = self.ending
        if E.game_over:
            # placement from the game itself: the elimination screen's "Nth place" when we were
            # knocked out; a game that ended while we were still in it is a win (1st) - the client
            # closes the game window at once then, so there is no screen to read (run27: 1st place
            # recorded as null, since match history never answered)
            seen = ' '.join([E.eliminated_seen or ''] + list(getattr(E, 'words', []) or []))
            m = re.search(r'([1-8])\s*(ST|ND|RD|TH)\s*PLACE', seen.upper())     # "4THPLACE" (run29)
            if not m and not E.eliminated_seen and not getattr(self, 'exit_clicked', False) and not REPLAY:
                # knocked out in the last fight, the match ends at the same moment: the end screen is up but its
                # background read (3.5-4 s) has not come in yet (benchmark game 20261018-153420: logged 1st, then
                # "2NDPLACE" one event later) - it is read now; a win closes the window and shows nothing
                try:
                    E._scan()
                except Exception as e:
                    self.log(ev='end_scan_error', error=repr(e))
                seen = ' '.join([E.eliminated_seen or ''] + list(getattr(E, 'words', []) or []))
                m = re.search(r'([1-8])\s*(ST|ND|RD|TH)\s*PLACE', seen.upper())
            if m:
                self.placement_screen = (int(m.group(1)), f'elimination screen: "{m.group(0)}"')
            elif not getattr(self, 'exit_clicked', False) and not E.eliminated_seen:
                self.placement_screen = (1, 'the game ended while we were still in it')
            else:
                self.placement_screen = (None, f'eliminated, place not read ({E.eliminated_seen!r})')
            print(f'game over (client phase {E.phase}); placement from the screen: {self.placement_screen}')
            self.log(ev='game_over', phase=E.phase, stage=self.phase.stage, hp=getattr(self, '_hp', None),
                     placement_screen=self.placement_screen)
            return 'done'
        st = self.phase.stage and Phase._key(self.phase.stage)
        carousel = is_carousel(scr)    # its scan costs the chase CPU
        E.want_scan = bool(st and st >= (3, 1) and not scr.shop_visible and not scr.augments and not carousel)
        if time.time() - self.exit_tried < 5:
            return None
        if E.exit_button:
            x, y = E.exit_button
            ref = (960 + (x - frame.shape[1] / 2) * 1080 / frame.shape[0], y * 1080 / frame.shape[0])
            print(f'{self.phase.stage}: eliminated - clicking exit at {ref}')
            self.snap(frame, f'{self.phase.stage}-eliminated', full=True)
            self.log(ev='exit_click', how='button', xy=ref, text=E.eliminated_seen)   # not at=: log() reads at as the time
            self.exit_clicked = True
            self.inp.click(ref)
            self.exit_tried, E.exit_button = time.time(), None
            return 'exit'
        hp = getattr(self, '_hp', None)
        if E.eliminated_seen and hp is not None and hp <= 45:
            if not self.reader.settings_open(frame):
                self.log(ev='exit_click', how='menu', text=E.eliminated_seen, hp=hp)
                self.exit_clicked = True
                self.inp.key('escape')
            else:
                print(f'{self.phase.stage}: eliminated - Settings -> EXIT TFT')
                self.inp.click(EXIT_TFT_BUTTON)
            self.exit_tried = time.time() - 3.5
            return 'exit'
        return None

    def collect_loot(self, frame, scr, tac):
        """Right-click toward orbs; the tactician walks while we keep playing."""
        if scr.stage is None and not getattr(self, '_stage_seen', False):
            # no stage read yet this game: the HUD is not where we look (run 20261010-002419: a stale window crop,
            # 25 minutes of clicks at 'orbs' and nothing else) - act on nothing until the screen reads
            return False
        self._stage_seen = True
        s, r = scr.round or (0, 0)
        pve = s == 1 or r == 7 or (s == 2 and r <= 1)
        # orbs are rare in PvP rounds: the watcher scans every frame in PvE, every ~2s otherwise - and every frame
        # while an orb is in sight. A scan is acted on while it is younger than a tick: with 0.5 s and ~1 s ticks,
        # PvP scans were nearly always stale when looked at (run 20261015-014311: the 5-7 orbs lay there all of
        # 6-1 and 6-2, no click)
        age = time.time() - self.watcher.orbs_at
        self.watcher.want_orbs = pve or bool(self.watcher.orbs) or age > 2.0
        if age > 2.5:
            return self.loot.target is not None
        orbs = self.watcher.orbs
        click = self.loot.next_click(orbs, tac, time.time())
        if self.loot.target is not None:
            self._orbs_seen_at = time.time()        # a live orb to walk to: no parking for a while
        if click:
            click = self.ground_point(click)
        self.status_extra['loot'] = (f'{len(orbs)} orb(s) visible, walking to {tuple(round(v) for v in self.loot.target)}'
                                     if orbs and self.loot.target else
                                     f'{len(orbs)} orb(s) visible' if orbs else 'no orbs')
        if click:
            try:
                self.inp.click(click, 'right')
            except NotForeground:
                return False
            self._loot_at = time.time()
            self.log(ev='loot', stage=scr.stage, click=[round(click[0]), round(click[1])],
                     orbs=len(orbs), tactician=tac and [round(tac[0]), round(tac[1])],
                     speed=round(self.loot.speed))
        return bool(orbs)

    def item_sig(self, frame, i):
        """A slot's icon pixels, coarse: the same item keeps the same signature."""
        import cv2
        from tft_player.items import SLOT_HALF, slot_point
        s = frame.shape[0] / 1080
        x, y, _ = slot_point(i)
        c = frame[round((y - SLOT_HALF) * s):round((y + SLOT_HALF) * s), round((x - SLOT_HALF) * s):round((x + SLOT_HALF) * s)]
        if not c.size:
            return None
        small = cv2.resize(c, (8, 8), interpolation=cv2.INTER_AREA) // 32
        return small.tobytes()

    def identify_item(self):
        """Idle time: hover one item-bench slot whose item has no name read yet, and read
        the name from its tooltip (text, not the icon). Hover only - nothing is clicked."""
        from tft_player.items import slot_point
        sigs = getattr(self, 'item_sigs', None) or []
        # two tries per item and round (run 221257: two failed reads at 4-6 left a Radiant item
        # "unknown" for the rest of the game - it was never hovered again)
        stage = self.phase.stage
        todo = [i for i, sig in enumerate(sigs) if sig and sig not in self.item_names
                and self.item_tries.get((sig, stage), 0) < 2]
        if not todo:
            return False
        if getattr(self, '_item_ocr_busy', False):
            return False                          # one tooltip read at a time
        i = todo[0]
        sig = sigs[i]
        self.item_tries[(sig, stage)] = self.item_tries.get((sig, stage), 0) + 1
        x, y, _ = slot_point(i)
        try:
            # approach from the screen's left edge: tooltips need motion, and the old start point (x 200) lay on the
            # trait panel - its trait tooltip was then read as the item's ("2 > 4 > 6 / Spellweaver", 15 times in the
            # games of 2026-10-10)
            self.inp.hover((4, y, 'L'))
            time.sleep(0.05)
            pre = self.cap.read()
            self.inp.hover((x, y, 'L'))
            time.sleep(0.4)                       # the tooltip fades in
            got = self.cap.read()
            self.inp.hover((960, 540))            # off the item bench again
        except NotForeground:
            return False
        if not got:
            return True
        frame = got[0]
        before = pre[0] if pre else None

        crop = (getattr(self, 'item_crops', None) or [None] * N_SLOTS)[i]

        def read():
            # off the loop: the title line is read in tens of ms, but when that fails the
            # detection over the whole tooltip takes 7-9 s
            from tft_player.items import learn_icon
            try:
                self.reader.tooltip_text = ''
                name = self.reader.item_tooltip(frame, (x, y), before=before)
                text = getattr(self.reader, 'tooltip_text', '')
                if name:
                    self.item_names[sig] = name
                    if crop is not None:
                        learn_icon(crop, re.sub(r'\s*\(.*$', '', name))   # known from the picture from now on
                elif text:
                    # read, but no item of that name in the data: the model gets the text and can name
                    # the slot itself (identify), instead of a bare "unknown"
                    self.news_event('item_unrecognised', stage,
                                    f'item bench slot {i}: its tooltip reads "{text[:80]}", which matches no '
                                    f'item name - name it with identify {{"item_slot": {i}, "name": ...}}')
                self.log(ev='item_read', slot=i, name=name, text=None if name else text)
                self.status_extra['identify'] = f'item slot {i}: {name or "tooltip not read"}'
            finally:
                self._item_ocr_busy = False
        self._item_ocr_busy = True
        threading.Thread(target=read, daemon=True).start()
        return True

    # events that change what the model sees: each wakes it with an update question when no question is
    # running and time allows (the user, 2026-10-09: no fixed number of calls a round; more when events need it)
    WAKE = {'game_change', 'unit_taken', 'unit_corrected', 'star_corrected', 'game_moved_unit', 'duplicated',
            'rolldown', 'after_fight', 'unit_missing', 'unit_found'}

    # step_failed / plan_idle: in the plan report ("failed", "Not done"); roster_mismatch: tracking, fixed by the
    # controller - the state shows the result
    LOG_ONLY = {'step_failed', 'plan_idle', 'roster_mismatch'}

    def news_event(self, kind, stage, what):
        """Record something that happened; the next model call gets it. A WAKE kind also asks for that call."""
        self.log(ev=kind, stage=stage, what=what)
        if kind in self.LOG_ONLY:
            return
        line = f'{stage} {kind}: {what}'
        if line not in self.news:
            self.news.append(line)
            if kind in self.WAKE:
                self.pending_events.append(f'{kind}: {what}')

    def take_news(self):
        news, self.news = self.news[-12:], []
        return '; '.join(news)

    def review_due(self, rp, scr, plan_done):
        """The model's own next_review conditions for this round: when one holds, it is asked
        again (once per plan). Nothing else makes a mid-round model call."""
        conds = rp.p.get('next_review') if isinstance(rp.p, dict) else None
        if not conds or getattr(rp, 'review_fired', False):
            return False
        if not isinstance(conds, list):
            conds = [conds]
        met = None
        for c in conds:
            if not isinstance(c, dict):
                continue
            if c.get('plan_complete') and plan_done:
                met = 'the plan is complete'
            elif isinstance(c.get('seconds_left'), (int, float)) and scr.timer is not None \
                    and scr.timer <= c['seconds_left']:
                met = f"{scr.timer}s left"
            elif isinstance(c.get('gold_at_least'), (int, float)) and scr.gold is not None \
                    and scr.gold >= c['gold_at_least']:
                met = f'gold is {scr.gold}'
            elif isinstance(c.get('unit'), str) and isinstance(c.get('star'), int) and any(
                    u.name == c['unit'] and u.star >= c['star'] for u in self.roster.units):
                met = f"{c['unit']} is {c['star']}-star"
            if met:
                break
        if not met or (scr.timer is not None and scr.timer <= 2):
            return False
        rp.review_fired = True
        self.log(ev='review_requested', stage=scr.stage, condition=met)
        print(f'{scr.stage}: next_review met ({met}) -> asking the model')
        self.request_plan(scr, self.last_frame, 'update', f'your next_review condition is met: {met}', interrupt=False)
        return True

    def bench_names(self, frame, detail=False):
        """The item bench now: names (a tooltip-read name for this very icon beats the picture)."""
        d = read_item_bench(frame, detail=True)
        names = []
        said = getattr(self, 'item_names_model', {})
        for i, (n, _, _) in enumerate(d):
            sig = self.item_sig(frame, i) if n else None
            names.append(self.item_names.get(sig, said.get(sig, n)) if sig else n)
        track = self.__dict__.get('_item_track')
        tracked = track.names() if track is not None else None
        if tracked and [n is not None for n in tracked] == [n is not None for n in names]:
            names = tracked                      # the same slots filled: the tracked names hold
        return (names, d) if detail else names

    @property
    def item_track(self):
        """The item bench slot by slot, kept from our own actions (items.ItemTracker)."""
        return self.__dict__.setdefault('_item_track', ItemTracker())

    def confirm_slot(self, i):
        """The name on item-bench slot i, read as text from its tooltip's title line (the user, 2026-10-15: confirm
        the component's text before making the item - fast and accurate). ~0.3 s: a glide in from the screen's edge
        (tooltips need motion), the fade-in, one line of OCR. None when no title was read in time."""
        x, y, _ = slot_point(i)
        self.inp.hover((4, y, 'L'), steps=2, duration=0.03)
        pre = self.cap.read()                    # the screen with no tooltip: what the tooltip changes is read
        self.inp.hover((x, y, 'L'), steps=4, duration=0.06)
        end, name = time.time() + 0.8, None
        time.sleep(0.15)
        while time.time() < end and name is None:
            got = self.cap.read()
            if got is not None:
                name = self.reader.item_tooltip(got[0], (x, y), before=pre[0] if pre else None, title_only=True)
            if name is None:
                time.sleep(0.06)
        if name:
            self.item_track.confirm(i, name)
            sig = (getattr(self, 'item_sigs', None) or [None] * N_SLOTS)[i]
            if sig:
                self.item_names[sig] = name
        self.log(ev='item_confirm', slot=i, name=name)
        return name

    def find_in_fight(self, unit, frame):
        """Where `unit` stands now in a fight (units move): right-click our units (green bars, nearest to where it
        was placed first) and read each one's panel until it is `unit` (name and star). -> the point to drop on (its
        body, under its bar, found again after the read), or None."""
        tac_box = self.watcher.tac[1] if getattr(self, 'watcher', None) else None
        inside = lambda b: tac_box is not None and tac_box[0] - 20 <= b[0] <= tac_box[2] + 20 and \
            tac_box[1] - 30 <= b[1] <= tac_box[3] + 10
        bars = [b for b in B.green_bars(frame, y_range=(120, 900)) if not inside(b)]
        if not bars:
            return None
        home = loc_point(unit.loc)
        bars.sort(key=lambda b: (b[0] - home[0]) ** 2 + (b[1] + 45 - home[1]) ** 2)
        pts = [(b[0], b[1] + 45) for b in bars]
        for k, pt in enumerate(pts[:6]):
            name, stars, _ = self.read_fresh(pt, pts[:k] + pts[k + 1:])
            if name == unit.name and (not stars or stars == unit.star):
                got = self.cap.read()
                if got is None:
                    return pt
                now = [b for b in B.green_bars(got[0], y_range=(120, 900)) if not inside(b)]
                near = min(now, key=lambda b: (b[0] - pt[0]) ** 2 + (b[1] + 45 - pt[1]) ** 2, default=None)
                if near is not None and abs(near[0] - pt[0]) < 120 and abs(near[1] + 45 - pt[1]) < 120:
                    return (near[0], near[1] + 45)
                return pt
        return None

    def fight_row(self, frame, point, expect=None):
        """The item row of the unit standing at `point` in a fight (the bar nearest it), or None."""
        from tft_player.items import read_unit_items
        bars = B.green_bars(frame, y_range=(120, 900))
        bar = min(bars, key=lambda b: (b[0] - point[0]) ** 2 + (b[1] + 45 - point[1]) ** 2, default=None)
        if bar is None or abs(bar[0] - point[0]) > 80 or abs(bar[1] + 45 - point[1]) > 80:
            return None
        return read_unit_items(frame, bar, expect)

    def unit_row(self, frame, unit, expect=None):
        """(bar, [(name, score, crop)]) for one tracked unit, or (None, None) when its bar is not
        found where it is tracked (or two units could own it)."""
        from tft_player.items import read_unit_items
        from tft_player.state import _hex_near
        _, _, places = B.bar_places(frame, exclude=self.watcher.tac[1])
        bar = next((b for b, loc in places if loc == unit.loc), None)
        if bar is None and unit.on_board:
            near = [b for b, loc in places if loc[0] == 'board' and _hex_near(loc[1:], unit.loc[1:])]
            bar = near[0] if len(near) == 1 else None
        return (bar, read_unit_items(frame, bar, expect)) if bar is not None else (None, None)

    def give_item(self, a, scr):
        """Drag an item onto a unit, by the item's IDENTITY (its slot is looked up on a fresh read
        just before the drag: run27 2-1 took the Gloves from a stored slot number and built
        Infinity Edge where the state logged Edge of Night). Success only when the screen shows it:
        the item left the bench and the unit's icon row shows it (or the item it completed); a
        consumable (Duplicator) must be gone, and a Duplicator's copy is added to the roster.
        The unit's items are what its row shows afterwards - never what was meant."""
        from tft_player.items import BY_PAIR, COMPONENTS, is_consumable, is_duplicator, is_reusable, learn_icon
        slot, item, unit = a.arg
        base = lambda n: re.sub(r'\s*\(.*$', '', n or '')
        if is_consumable(item) or 'Chest' in item:
            self.last_used = (base(item), unit, time.time())     # a choice screen may follow: its source
        got = self.cap.read()
        if got is None:
            self.fail_cause = 'no frame to read the item bench from'
            return False
        names, detail = self.bench_names(got[0], detail=True)
        here = [k for k, n in enumerate(names) if n and base(n) == base(item)]
        src = slot if slot in here else (here[0] if here else None)
        track = self.__dict__.get('_item_track') if isinstance(self, Player) else None
        if track is not None and not REPLAY and not is_reusable(item):
            # confirmed by its tooltip text before the drag (a slot already read as text since it last changed is
            # not hovered again); a slot that turns out to hold another item is renamed and the next one tried
            cands = ([src] if src is not None else []) + [k for k in here if k != src]                 + [k for k, n in enumerate(names) if n == 'unknown']
            src, wrong = None, []
            for k in cands[:4]:
                if track.text[k] and base(track.slots[k]) == base(item):
                    src = k
                    break
                try:
                    said = self.confirm_slot(k)
                except NotForeground:
                    raise
                if said is None or base(said) == base(item):
                    src = k                      # matches, or no title read in time: act on the picture's name
                    break
                wrong.append(f'slot {k} holds {said}')
            if src is None and wrong:
                self.fail_cause = f'{item} not found: its tooltip text says {"; ".join(wrong)}'
                return False
            if wrong:
                self.log(ev='item_misread', stage=scr.stage, item=item, wrong=wrong, used=src)
            got = self.cap.read() or got          # the bench as it is now, for the checks below
            names, detail = self.bench_names(got[0], detail=True)
        if src is None:
            if 'unknown' in names:
                self.identify_items_now = True       # read the unidentified slots now (any phase)
                self.fail_cause = (f'{item} is not among the identified items on the item bench; '
                                   f'unidentified slots {[k for k, n in enumerate(names) if n == "unknown"]} are being read')
            else:
                self.fail_cause = f'{item} is not on the item bench (it holds: {", ".join(n for n in names if n) or "nothing"})'
            return False
        fight = getattr(a, 'fight', False)
        target = None
        if fight:
            target = self.find_in_fight(unit, got[0])
            if target is None:
                self.fail_cause = f'{unit} was not found among our units in the fight (read off their panels)'
                return False
            before = list(unit.items or [])
        else:
            bar, before = self.unit_row(got[0], unit)
            before = [n for n, _, _ in before] if before is not None else None
        # a component dropped on a unit holding a matching component completes an item. The row read before the
        # drop can miss an icon: the tracked items count too (run 20261015-003101 2-1: Negatron Cloak on Rek'Sai's
        # Chain Vest made Gargoyle Stoneplate, judged 'failed' and queued again)
        held = list(before or []) + [c for c in unit.items or [] if c not in (before or [])]
        done = next((BY_PAIR[tuple(sorted((c, item)))] for c in held
                     if item in COMPONENTS and c in COMPONENTS and tuple(sorted((c, item))) in BY_PAIR), None)
        expect = done or item
        # any item this component completes also shows it landed
        makes = {v for k, v in BY_PAIR.items() if item in k} if item in COMPONENTS else set()
        raw_name = lambda f, k, n: base(self.item_names.get(self.item_sig(f, k), n)) if n else None
        raw0 = [d[0] for d in detail] if detail else list(names)
        n_here = sum(1 for k, n in enumerate(raw0) if raw_name(got[0], k, n) == base(item))
        src_was = raw_name(got[0], src, raw0[src])
        inp = self.inp
        how, stage = 'fight', getattr(scr, 'stage', None)
        if target is None:
            target, how = Player.drop_point(self, unit, bar, stage)
            if how is None:
                return False
        self._last_drop = (unit.id, how, stage)
        inp.drag(slot_point(src), target or loc_point(unit.loc))
        if is_reusable(item):
            # it stays on the item bench (the Alpha Mark - the user, 2026-10-13): the drop is the use, and only the
            # game's own refusal fails it
            said = None
            for _ in range(8):
                time.sleep(0.12)
                got2 = self.cap.read()
                said = self.game_message(got2[0]) if got2 is not None and not REPLAY else None
                if said:
                    break
            if said:
                self.fail_cause = f'the game refused it: "{said}" ({item} on {unit})'
                self.log(ev='game_refused', stage=scr.stage, item=item, unit=repr(unit), said=said)
                return False
            self.roster.mark_used(base(item), unit, scr.stage)
            self.log(ev='mark_used', stage=scr.stage, item=item, unit=repr(unit))
            return True
        # the new icon animates in: read at the first changed frame it was missing or matched another item, and
        # 9 of 26 'failed' hand-outs (runs 221257, 152104, 213127) showed the expected item on the unit a moment
        # later. So: done as soon as the row shows the expected item, else the row read last, 2 s after the drop
        left, after, frame2, said, said_at = False, None, None, None, 0.0
        # the icon can take over 2 s to show: 'failed' hand-outs of run 20261010-121458 (fail08, 10, 16) showed the expected
        # item in the frame saved just after the 2 s window (Archangel's on Kog'Maw, Adaptive Helm on Amumu ...); 4 s was
        # still short in run 20261014-063145 (4-3 Bloodthirster on Master Yi, 4-6 Kraken's on Aphelios: both rows read
        # empty in the window and right in the frame saved after it). The wait ends as soon as the item shows.
        end = time.time() + 7.0
        while time.time() < end:
            time.sleep(0.08)
            got2 = self.cap.read()
            if got2 is None:
                continue
            frame2 = got2[0]
            now = read_item_bench(frame2)
            # by count, not by slot: the item bench closes up, and a second copy slides into the emptied slot (run
            # 20261015-003101 3-2: a Tear landed on Krug, the other Tear moved up, 'still on the bench' - the retry
            # built Blue Buff on Krug)
            if n_here:
                left = sum(1 for k, n in enumerate(now) if raw_name(frame2, k, n) == base(item)) < n_here
            else:                                # the picture never saw it as this item (named by text): the slot
                left = raw_name(frame2, src, now[src]) != src_was
            if not left and said is None and time.time() - said_at > 0.4:
                said, said_at = self.game_message(frame2), time.time()
                if said:
                    break                        # the game refused it, in its own words: no need to wait on
            if left:
                if is_consumable(item):
                    break
                row = self.fight_row(frame2, target, expect) if fight else self.unit_row(frame2, unit, expect)[1]
                if row is not None:
                    after = row
                    names_ = [n for n, _, _ in row]
                    if expect in names_ or (not done and any(n in makes and n not in (before or []) for n in names_)):
                        break
        if not left:
            from tft_player.items import item_desc
            rule = item_desc(base(item))
            if said:
                # run 20261010-113319 6-1/6-2: Guinsoo's dropped on a 3-item Azir four times; the game said why
                # each time and the log only guessed
                self.fail_cause = (f'the game refused it: "{said}" ({item} is still on the item bench; {unit} '
                                   f'holds {unit.items or "no items"})')
                self.log(ev='game_refused', stage=scr.stage, item=item, unit=repr(unit), said=said)
                self.items_from_panel_after_refusal(unit, item, said, scr)
                return False
            self.fail_cause = ('the item is still on the item bench after the drop (missed the unit, 3 items '
                               'already, or the game refused it' + (f': "{rule}"' if is_consumable(item) and rule else '') + ')')
            return False
        self.items_done.add((src, item))
        if track is not None:
            track.taken(src)                     # off the bench; the items below move up
        if is_consumable(item):
            if is_duplicator(item) and unit.name:
                # the game puts a 1-star copy on the bench (leftmost free slot) and merges 3 copies
                self.roster.last_merge = None
                copy = self.roster.bought(unit.name, ours=True)
                self.news_event('duplicated', scr.stage, f'{item} used on {unit}: a 1-star {unit.name} copy '
                                                         f'(unit {copy.id}) was added on the bench')
                if self.roster.last_merge and not REPLAY:
                    self.confirm_merge(self.roster.last_merge, scr)
            return True
        if after is None and not fight and not REPLAY:
            # the row is hidden (our tactician's bar over it, run 20261015-132144 5-6): its info panel shows the items
            name, _, held_now = self.read_fresh(target or loc_point(unit.loc), [])
            if name == unit.name and held_now:
                after = [(n, 1.0, None) for n in held_now]
                self.log(ev='items_from_panel', stage=scr.stage, unit=repr(unit), items=held_now)
        if after is None and not fight and not REPLAY:
            found = self.locate_item(item, makes, unit)
            if found[0] is not None:              # another unit shows it: it went there
                return self.misplaced(item, makes, unit, scr, f"{unit}'s item row could not be read", found=found)
        if after is None:
            self.fail_cause = (f'{item} left the item bench, but {unit}\'s item row could not be read - where it '
                               f'went is unverified')
            unit.items_read = False
            self._unit_items_at = 0              # read every unit's items at once
            return False
        row = [n for n, _, _ in after]
        # the icon that appeared is the item we gave: compared by what the row holds, not by place - the game can put
        # the new item first (run 20261016-052413 4-3: Flickerblades showed as ['unknown', 'Last Whisper', 'Blue
        # Buff'], every place differed, nothing was learned, and the give was reported missing and queued again)
        added = Counter(row) - Counter(before or [])
        k = row.index('unknown') if row.count('unknown') == 1 else None
        if (k is not None and dict(added) == {'unknown': 1} and after[k][2] is not None
                and len(row) == len(before or []) + (0 if done else 1)):
            learn_icon(after[k][2], expect)
            row[k] = expect
            self.log(ev='icon_learned', stage=scr.stage, name=expect, how='drag outcome')
        if expect not in row and not done:
            made = next((n for n in row if n in makes and n not in (before or [])), None)
            if made:
                expect = done = made         # it met a component the records did not have and completed it
        if expect not in row:
            unit.items, unit.items_read = row, True
            self._unit_items_at = 0
            if not fight and not REPLAY:
                return self.misplaced(item, makes, unit, scr, f'{unit} shows {row or "no items"} (expected {expect})')
            self.fail_cause = (f'{item} left the item bench, but {unit} shows {row or "no items"} '
                               f'(expected {expect}): it may be on another unit')
            return False
        # what it holds now follows from our own drop: what it held, plus the item (or the item it completed). A row
        # read with fewer icons than that missed some - items leave a unit only by a Remover, a sell or a temporary
        # item running out (run 20261013-220557 5-1/5-2: a Rod on 3-star Pebbles read as "Needlessly Large Rod"
        # alone, and Jeweled Gauntlet + Blue Buff were gone from the records - the model planned with them missing)
        from tft_player.items import held_after_give
        predicted = held_after_give(unit.items, item, done)
        if len([n for n in row if n != 'unknown']) < len(predicted) and all(n in predicted for n in row if n != 'unknown'):
            self.log(ev='items_kept', stage=scr.stage, unit=repr(unit), read=row, kept=predicted)
            row = predicted
        unit.items, unit.items_read = row, True
        Player.note_final(self, unit, base(item), expect)
        return True

    def items_from_panel_after_refusal(self, unit, item, said, scr):
        """The game refused a give: what the unit holds is read off its info panel (all its items) and the records and
        the model are told - its item row can read short (benchmark game 20261018-134910 5-2, 6-2: 3-star Veigar
        recorded with [Giant Slayer] only; the game refused four gives, "Item not valid on target", and the model went
        on planning them)."""
        if REPLAY or not unit.on_board and unit.loc[0] != 'bench':
            return
        try:
            got = self.cap.read()
            bars = B.green_bars(got[0]) if got else None
            others = [self.unit_point(o.loc, bars) for o in self.roster.units
                      if o.name and o.name != unit.name and not o.missing and o.loc[0] in ('board', 'bench')][:2]
            name, _, held = self.read_fresh(self.unit_point(unit.loc, bars), others)
        except NotForeground:
            return
        self.log(ev='items_after_refusal', stage=scr.stage, unit=repr(unit), read=name, items=held)
        if name != unit.name or held is None:
            return
        if list(held) != list(unit.items or []):
            unit.items, unit.items_read = list(held), True
        self.news_event('item_refused', scr.stage, f'the game refused {item} on {unit} ("{said}"); its info panel shows '
                                                   f'{", ".join(held) if held else "no items"}')

    def hold_item_over(self, a, scr):
        """An item for a fielded unit in the last 2 s of planning waits: the round flips under the drop, the units fly
        off and the read finds no items (benchmark game 20261018-131328 3-5: Sparring Gloves to Pebbles at 1 s,
        'missing'). -> True when held over (the plan gives it again)."""
        unit = a.arg[2]
        # (a bench unit too: benchmark game 20261018-150046 3-6, a Giant's Belt for Alistar on the bench at 1 s)
        if (getattr(self.phase, 'planning', False) and scr.timer is not None and scr.timer <= 2
                and not getattr(a, 'fight', False) and getattr(unit, 'loc', ('',))[0] in ('board', 'bench')):
            self.fail_cause = 'queued: the round flips in the last seconds'
            self.deferred = True
            return True
        return False

    def drop_point(self, unit, bar, stage):
        """Where an item for this unit is dropped: (point, 'body') just under its bar, (None, 'centre') for its hex
        centre, or (None, None) - not now, fail_cause says why. A point whose drop landed on another unit this round
        is not used again (release run 20261016-141957 4-6: a Recurve Bow for Xayah at (0,0) landed on Nidalee at
        (1,0), and the Negatron Cloak dropped on the same point made Kraken's Fury on Nidalee)."""
        stage_missed, tags = self.__dict__.setdefault('drop_missed', {}).get(unit.id, (None, set()))
        tags = tags if stage_missed == stage else set()
        if bar is not None and 'body' not in tags:
            # onto the unit's own body, just under its bar - not its hex centre, where a neighbour's big model can
            # reach (run 20261015-132144 5-2: a Negatron Cloak for Aphelios at (0,2) went to Kog'Maw at (0,1)). With a
            # unit right behind it, not above its own row: there, under the bar is that unit's hex (4-6 above: the
            # drop was 31 px above Xayah's hex centre, 39 px from Nidalee's)
            point = (bar[0], bar[1] + 45)
            others = [o for o in (self.roster.board() if getattr(self, 'roster', None) else []) if o is not unit]
            if unit.on_board:
                hx, hy = B.hex_xy(*unit.loc[1:])
                # a tall model's bar stands far above its body: under the bar is air above its head (benchmark game
                # 20261018-131328 4-3, 5-1: 3-star Pebbles' bar ~120 px over its hex centre, two Sparring Gloves
                # stayed on the item bench) - the drop stays within 40 px of the hex centre
                point = (point[0], max(point[1], hy - 40))
                if any(0 < hy - B.hex_xy(*o.loc[1:])[1] < 100 and abs(B.hex_xy(*o.loc[1:])[0] - hx) < 90
                       for o in others):
                    point = (point[0], max(point[1], hy - 25))
            return point, 'body'
        if 'centre' in tags:
            self.fail_cause = (f'drops for {unit} landed on another unit this round (dropped '
                               f'{" and ".join("under its bar" if t == "body" else "on its hex centre" for t in sorted(tags))}):'
                               f' not given again until the next round')
            return None, None
        return None, 'centre'

    def locate_item(self, item, makes, target):
        """Where a give that missed `target` went: the unit whose item row (or, when the row is covered, its info panel)
        now shows `item` - or an item it completed there (`makes`) - more often than its records do.
        -> (unit, item name) or (None, None). The units nearest the target first: a drop lands on a neighbour."""
        got = self.cap.read()
        if got is None:
            return None, None
        frame, wanted, hidden = got[0], {item} | set(makes or ()), []
        near = lambda u: (0 if u.loc[0] == target.loc[0] else 1,
                          sum(abs(a - b) for a, b in zip(u.loc[1:], target.loc[1:])))
        units = sorted((u for u in self.roster.units if u is not target and not u.missing
                        and u.loc[0] in ('board', 'bench')), key=near)
        gained = lambda names, u: [n for n in names if n in wanted and names.count(n) > (u.items or []).count(n)]
        for u in units:
            _, row = self.unit_row(frame, u)
            if row is None:
                hidden.append(u)
                continue
            names = [n for n, _, _ in row]
            new = gained(names, u)
            if new:
                u.items, u.items_read = names, True
                return u, new[0]
        for u in hidden[:3]:                     # rows covered (our tactician's bar, an effect): their panels
            name, _, held = self.read_fresh(loc_point(u.loc), [])
            new = gained(held, u) if name == u.name else []
            if new:
                u.items, u.items_read = list(held), True
                return u, new[0]
        return None, None

    def misplaced(self, item, makes, target, scr, seen, found=None):
        """A give left the item bench but its target does not show it: the item is found (locate_item) and gets its
        current holder (the unit it landed on) and its target holder (the unit the plan named) in the item-holder
        records (item_finals, the user's design 2026-10-14), so handover_action moves it there - an Item Remover, or a
        sell of a holder outside the comp. Run 20261015-132144 5-2: a Negatron Cloak for Aphelios went to Kog'Maw and
        stayed there all game. Not found anywhere: reported missing. -> False (the give itself did not land)."""
        stage = getattr(scr, 'stage', None)
        holder, got = found or self.locate_item(item, makes, target)
        if holder is not None:
            last = getattr(self, '_last_drop', None)
            if last and last[0] == target.id and last[1] in ('body', 'centre'):
                # the drop point that reached a neighbour is not used for this unit again this round (give_item)
                missed = self.__dict__.setdefault('drop_missed', {})
                st, tags = missed.get(target.id, (None, set()))
                missed[target.id] = (last[2], (tags if st == last[2] else set()) | {last[1]})
            self.set_final(holder.id, got, target.name, target.star)
            self.fail_cause = (f'{item} landed on {holder} (unit {holder.id}) as {got}, not on {target}: it is moved '
                               f'to {target.name} (its target holder)')
            self.news_event('item_misplaced', stage, f'{item} (given to {target}, unit {target.id}) landed on '
                            f'{holder} (unit {holder.id}) as {got}; it is being moved to {target.name}')
            self.log(ev='item_misplaced', stage=stage, item=item, got=got, holder=repr(holder), target=repr(target))
        else:
            self.fail_cause = (f'{item} left the item bench, but {seen} and no other unit shows it: missing '
                               f'(the item reads go on)')
            self.news_event('item_missing', stage, f'{item} (given to {target}, unit {target.id}) left the item '
                            f'bench but no unit shows it')
            self.log(ev='item_missing', stage=stage, item=item, target=repr(target))
        return False

    def game_message(self, frame):
        """The game's own one-line notice over the bench ("Item not valid on target. Check item description to
        equip appropriately."), or None. One OCR crop, ~0.1 s; only read after an action did not land."""
        h, w = frame.shape[:2]
        s = h / 1080
        box = (int(w / 2 - 480 * s), int(800 * s), int(w / 2 + 480 * s), int(852 * s))
        try:
            txt = (self.reader._texts(frame, [box])[0] or '').strip()
        except Exception:
            return None
        # a real notice is mostly whole words; a tooltip or the board behind reads as fragments (run
        # 20261010-235723 2-1: "them f r 1.5 scod age   mag am nd sg")
        words = txt.split()
        real = [w for w in words if re.fullmatch(r"[A-Za-z][a-z']{2,}[.,!?:]?", w)]
        ok = (re.fullmatch(r"[A-Za-z0-9 .,'!?:()-]{12,}", txt) and len(real) >= 4
              and len(real) >= 0.7 * len(words) and '  ' not in txt)
        return txt if ok else None

    def selected_here(self, frame, loc):
        """Is the selection ring (bright cyan, drawn at the feet of the unit whose panel is open) at
        loc? Measured on recorded reads: the ring covers >4% of a band around the unit's feet."""
        h, w = frame.shape[:2]
        s = h / 1080
        x, y = (B.BENCH[loc[1]][0], 775) if loc[0] == 'bench' else B.hex_xy(loc[1], loc[2])
        cx, cy = w / 2 + (x - 960) * s, y * s
        reg = frame[int(cy - 30 * s):int(cy + 30 * s), int(cx - 70 * s):int(cx + 70 * s)].astype(np.int16)
        if reg.size == 0:
            return True                          # off screen: no evidence against it
        r, g, b = reg[..., 0], reg[..., 1], reg[..., 2]
        return float(((b > 190) & (g > 170) & (b - r > 35)).mean()) > 0.04

    def unit_bar(self, frame, loc, places=None):
        """The health bar of the unit at loc in this frame (board: the one bar near the hex), or None."""
        from tft_player.state import _hex_near
        places = places if places is not None else B.bar_places(frame, exclude=self.watcher.tac[1])[2]
        bar = next((b for b, l in places if l == loc), None)
        if bar is None and loc[0] == 'board':
            near = [b for b, l in places if l[0] == 'board' and _hex_near(l[1:], loc[1:])]
            bar = near[0] if len(near) == 1 else None
        return bar

    def remember_look(self, u, frame, label=False):
        """Store u's appearance (champs.embed of its crop): later reads of its slot/hex are compared
        with it. label=True: its name was just read off its info panel, so the crop also goes into
        the champion library (data/champ-library) for naming by sight."""
        from tft_player import champs
        if frame is None or (REPLAY and not os.environ.get('TFT_LOOKS')):
            return
        bar = self.unit_bar(frame, u.loc)
        if bar is None and label:
            got = self.cap.read()                 # the frame before the click had no bar there: now
            frame = got[0] if got else frame
            bar = self.unit_bar(frame, u.loc)
        if bar is None and u.loc[0] == 'bench':
            # bench units show no health bar in combat (run32: 26 reads lost): the slot's fixed point,
            # if the slot is not empty
            from tft_player import bench as BN
            crop0 = BN.slot_crop(BN._bench_band(frame), u.loc[1])
            occ = BN.occupied_share(crop0, u.loc[1])
            if occ is None or occ >= BN.EMPTY:
                bar = (B.BENCH[u.loc[1]][0], self.BENCH_BAR_Y)
        crop = champs.unit_crop(frame, bar) if bar is not None else None
        if crop is None:
            self.log(ev='look_not_taken', unit=repr(u), why='no health bar at its place' if bar is None else 'crop off screen')
            u.look = None
            return
        try:
            u.look, u.look_loc, u.look_misses = champs.embed([crop])[0], u.loc[0], 0
            if label and u.name:
                champs.add_to_library(crop, u.name, u.star)
        except Exception as e:                   # model file missing / onnx error: say so, go on
            self.log(ev='look_error', error=repr(e))
            u.look = None

    # same unit, same place (tools/eval_champs.py, 365 same-unit / 2892 different-champion pairs,
    # 960x600 recordings): bench 0.75 -> 5.4% of same units flagged, 2.1% of other champions pass;
    # board 0.70 -> 8.4% flagged, 7.3% pass. Two low readings in a row are needed to act.
    LOOK_SAME = {'bench': 0.75, 'board': 0.70}
    LOOK_SAME_MOVED = 0.60    # after a move (bench <-> board view differs): looser, not measured

    GALLERY_EVERY = 20        # s between two gallery crops of one unit (different poses, rounds)

    def grow_gallery(self, u, crop, vec, scr):
        """The recognition library grows from normal games: a unit whose name is certain (its buy
        seen landing, or read off its info panel) and that still looks like itself gives a crop now
        and then, bench and board, until its champion has champs.NEED per place."""
        from tft_player import champs
        place = u.loc[0]
        if not (u.name and getattr(u, 'sure', False) and not u.unverified and not u.missing):
            return
        if time.time() - getattr(u, 'gallery_at', 0) < self.GALLERY_EVERY or not champs.wants(u.name, place):
            return
        u.gallery_at = time.time()
        champs.add_to_library(crop, u.name, u.star, vec=vec)
        champs.add_progress(u.name, place == 'bench', place == 'board')
        self.log(ev='gallery', stage=scr.stage, unit=repr(u), place=place)

    def check_looks(self, frame, places, scr):
        """Every unit still looks like itself: its crop now against the one stored when its identity
        was set (a wrong name never shows in occupancy - run27 tracked Kayle as Xayah all game).
        Runs in a background thread (~20 ms per unit); two low readings in a row mark the unit
        UNVERIFIED and it is read off its info panel once. A unit with no stored look gets one."""
        from tft_player import champs
        from tft_player.state import _hex_near
        if getattr(self, '_looks_busy', False) or (REPLAY and not os.environ.get('TFT_LOOKS')):
            return
        todo = []
        for bar, loc in places:
            u = self.roster.at(loc)
            if u is None and loc[0] == 'board':
                near = [x for x in self.roster.board() if _hex_near(x.loc[1:], loc[1:])]
                u = near[0] if len(near) == 1 else None
            if u is None or u.missing:
                continue
            crop = champs.unit_crop(frame, bar)
            if crop is not None:
                todo.append((u, crop))
        if not todo:
            return
        self._looks_busy = True

        def run():
            try:
                vecs = champs.embed([c for _, c in todo])
                for (u, crop), v in zip(todo, vecs):
                    if getattr(u, 'look', None) is None:
                        u.look, u.look_loc, u.look_misses = v, u.loc[0], 0
                        bought = getattr(u, 'bought_at', None)
                        if bought and time.time() - bought < 30 and u.loc == getattr(u, 'bought_loc', None) \
                                and u.name and not u.unverified and getattr(u, 'sure', False):
                            champs.add_to_library(crop, u.name, u.star)
                            u.labelled = True
                            self.log(ev='champ_labelled', how='buy landing', unit=repr(u))
                        continue
                    sim = max(float(np.dot(w, v)) for w in [u.look] + list(getattr(u, 'looks_ok', [])))
                    moved = getattr(u, 'look_loc', None) != u.loc[0]
                    if sim >= (self.LOOK_SAME_MOVED if moved else self.LOOK_SAME[u.loc[0]]):
                        u.look_misses = 0
                        if moved:
                            u.look, u.look_loc = v, u.loc[0]       # passed after a move: the new view
                        self.grow_gallery(u, crop, v, scr)
                        continue
                    if getattr(u, 'look_trusted', None) == u.star:
                        # read off its panel at this star level: a pose, an item glow or an effect, not
                        # another champion - the view is kept as one more reference, no re-read
                        u.looks_ok = (list(getattr(u, 'looks_ok', [])) + [v])[-4:]
                        u.look_misses = 0
                        continue
                    u.look_misses = getattr(u, 'look_misses', 0) + 1
                    if u.look_misses >= 2 and not u.unverified:
                        # checked silently, in idle time (identify_one): only a read that finds another
                        # champion is reported (unit_corrected) - the model is not sent false alarms
                        u.unverified = (f'looks different from when it was identified (similarity '
                                        f'{sim:.2f}): to be read')
                        self.reread = getattr(self, 'reread', set()) | {u.id}
                        self.tries[u.id] = 0
                        self.log(ev='look_changed', stage=scr.stage, unit=repr(u), sim=round(sim, 3))
                        u.look_before_read, u.name_before_read = u.look, u.name
                        u.look = None                      # taken again after the read
            except Exception as e:
                self.log(ev='look_error', error=repr(e))
            finally:
                self._looks_busy = False
        threading.Thread(target=run, daemon=True).start()

    def read_roster_items(self, frame, places, scr):
        """Every unit's items, read from the icon row over its health bar (items.read_unit_items).
        A changed reading must repeat once before it replaces the old one (a unit still walking,
        an icon half-drawn). 'unknown' = an item is there, not recognised."""
        from tft_player.items import read_unit_items
        from tft_player.state import _hex_near
        t0 = time.perf_counter()
        pend = getattr(self, '_items_pending', {})
        # a few units a step, in turn - a unit waiting for its confirming second read first: every unit's row each
        # 0.6 s took 4.3 s a time (run 20261015-161625: 325 s of the game, the loop blocked meanwhile)
        todo = []
        for bar, loc in places:
            u = self.roster.at(loc)
            if u is None and loc[0] == 'board':
                near = [x for x in self.roster.board() if _hex_near(x.loc[1:], loc[1:])]
                u = near[0] if len(near) == 1 else None      # two candidates: no guess
            if u is not None:
                todo.append((bar, loc, u))
        turn = self.__dict__.setdefault('_items_turn', {})
        todo.sort(key=lambda t: (t[2].id not in pend, t[2].items_read, turn.get(t[2].id, 0)))
        for bar, loc, u in todo[:self.ITEM_ROWS_PER_STEP]:
            turn[u.id] = time.time()
            got = [n for n, _, _ in read_unit_items(frame, bar)]
            if 'unknown' in got and u.items_read and len(got) >= len(u.items):
                # an icon not recognised is not another item: a name we knew that is missing from the reading
                # fills it (run 033507: Aphelios' Kraken's Fury read "unknown" for 13 reads after Cybernetic
                # Implants changed the icons). Components combining shrink the count: then no filling.
                spare = list(u.items)
                for n in got:
                    if n in spare:
                        spare.remove(n)
                spare = [n for n in spare if n != 'unknown']
                got = [spare.pop(0) if n == 'unknown' and spare else n for n in got]
            if u.items_read and got == u.items:
                pend.pop(u.id, None)
                continue
            if pend.get(u.id) != got:
                pend[u.id] = got                 # seen once: wait for a second identical read
                continue
            pend.pop(u.id, None)
            if got != u.items or not u.items_read:
                self.log(ev='unit_items', stage=scr.stage, unit=repr(u), was=u.items, read=got)
            u.items, u.items_read = got, True
        self._items_pending = pend
        self.lat_items = time.perf_counter() - t0

    ITEM_ROWS_PER_STEP = 3                   # item rows read per loop step (read_roster_items)

    def open_container(self, slot, scr):
        """Open the container on a bench slot (E over it sells it, which opens its choice row)."""
        obj = self.bench_objs.get(slot)
        if not obj or obj['kind'] != 'interactable':
            self.log(ev='container_refused', slot=slot, seen=obj)
            print(f'refused: bench slot {slot} holds no recognised container ({obj})')
            return False
        self.inp.key('e', at=loc_point(('bench', slot)))      # selling a container opens it
        self.container_opened = (obj['cls'], slot, time.time())
        self.auto_opened.add(slot)
        # a new choice screen comes: never the one answered before (run29 4-2: the second anvil's
        # screen was taken for the first, already answered one; the game picked on timeout)
        self.choice = None
        self.log(ev='container_open', stage=scr.stage, slot=slot, cls=obj['cls'], conf=obj['conf'])
        self.bench_objs.pop(slot, None)
        time.sleep(0.2)
        return True

    def auto_open_container(self, scr):
        """A container is on the bench: if the model is free RIGHT NOW, hold it (coach.reserve_model) and open
        the container, so that the choice question goes out the moment its options show - in planning or in
        the fight. Never open and then wait for the model (the user, 2026-10-05). The held model is given
        back when the choice call ends, or when no choice screen came."""
        from tft_player import coach
        held = getattr(self, 'model_reserved', None)
        if held and not self.choice and time.time() - held > 8.0:          # opened, but no choice screen came
            self.model_reserved = None
            coach.release_model()
            self.log(ev='model_released', stage=scr.stage, why='no choice screen after opening')
        if not self.auto_open or self.choice or self.plan_pending or coach.STOPPED or held:
            return False
        if scr.timer is not None and scr.timer <= 1 and self.phase.planning:
            return False
        for slot, o in sorted(self.bench_objs.items()):
            if o['kind'] != 'interactable' or (slot, scr.stage) in self._auto_tried:
                continue
            if not coach.reserve_model():
                return False                    # the model is busy: look again on the next read
            self._auto_tried.add((slot, scr.stage))
            print(f"{scr.stage}: {o['cls']} on bench slot {slot}: the model is free - opening it")
            self.model_reserved = time.time()
            try:
                ok = self.open_container(slot, scr)
            except NotForeground:
                ok = False
                self._auto_tried.discard((slot, scr.stage))
            if not ok:
                self.model_reserved = None
                coach.release_model()
            return ok
        return False

    def bench_perception(self, frame, bench_seen, scr, planning=True, containers_only=False):
        """Every bench slot's state (bench.classify: unit / empty / container / unknown / obscured).
        Non-unit objects must read the same twice before they count; a new container is an event
        for the model. Slots holding something that is not a unit are blocked for moves and buys
        (roster.blocked) - containers always; an unrecognised object only when seen in planning on
        our own board (in combat on an opponent's arena empty tiles look different)."""
        from tft_player import bench
        try:
            seen = bench.classify(frame, bench_seen)
        except Exception as e:
            self.log(ev='bench_error', error=repr(e))
            return
        self.bench_kinds = seen
        # an 'unknown' slot where we track a unit is that unit (its bar was missed), not an object;
        # in combat, unknowns are the opponent arena's tiles: only containers are objects then
        # ... and only when this read shows our own bench (one slot matching our empty tile):
        # run29 3-6 reported 4 "objects" while the camera was on an opponent's arena
        own = any(o['kind'] == 'empty' for o in seen)
        now = {i: o for i, o in enumerate(seen)
               if o['kind'] == 'interactable'
               or (o['kind'] == 'unknown' and planning and own and not containers_only
                   and self.roster.at(('bench', i)) is None)}
        prev = self._bench_prev
        # a clear container match counts at once (so the round's first plan already has it);
        # anything weaker must read the same twice
        # an unrecognised object must read the same 3 times in a row (a unit passing, a hover)
        runs = getattr(self, '_bench_runs', {})
        runs = {i: (runs.get(i, 0) + 1 if prev.get(i, {}).get('kind') == o['kind'] and prev.get(i, {}).get('cls') == o['cls']
                    else 1) for i, o in now.items()}
        self._bench_runs = runs
        stable = {i: o for i, o in now.items()
                  if (o['kind'] == 'interactable' and (o['conf'] >= 0.9 or runs[i] >= 2))
                  or (o['kind'] == 'unknown' and runs[i] >= 3)}
        self._bench_prev = now
        for i, o in stable.items():
            old = self.bench_objs.get(i)
            if old and old['kind'] == o['kind'] and old['cls'] == o['cls']:
                continue
            self.log(ev='bench_object', stage=scr.stage, slot=i, kind=o['kind'], cls=o['cls'], conf=o['conf'])
            if o['kind'] == 'interactable':
                print(f"{scr.stage}: bench slot {i}: {o['cls']} ({o['conf']:.2f})"
                      + ('' if self.auto_open else ' -> telling the model'))
                if not self.auto_open:       # the model reports it and decides when to open
                    self.pending_events.append(f"EVENT CHOICE_CONTAINER_AVAILABLE: {o['cls']} on bench slot {i} "
                                               f"(execution_plan.open_container).")
            else:
                print(f"{scr.stage}: bench slot {i}: unrecognised object ({o['conf']:.2f})")
        keep_other = lambda o: containers_only and o['kind'] != 'interactable'   # a containers-only read keeps them
        for i in [i for i in self.bench_objs if i not in now and seen[i]['kind'] != 'obscured'
                  and (planning or self.bench_objs[i]['kind'] == 'interactable') and not keep_other(self.bench_objs[i])]:
            self.log(ev='bench_object_gone', stage=scr.stage, slot=i, was=self.bench_objs[i])
        # a covered slot is no evidence that its object left; in combat an unrecognised object
        # seen in planning is kept (the opponent's arena makes every free tile 'unknown')
        self.bench_objs = {i: o for i, o in self.bench_objs.items()
                           if i in now or seen[i]['kind'] == 'obscured' or keep_other(o)
                           or (not planning and o['kind'] == 'unknown' and seen[i]['kind'] == 'unknown')}
        self.bench_objs.update(stable)
        blocked = {}
        for i, o in self.bench_objs.items():
            if o['kind'] == 'interactable':
                blocked[i] = f"a container ({o['cls']})"
            elif o['kind'] == 'unknown' and self.roster.at(('bench', i)) is None and (planning or i in self.roster.blocked):
                # (a tracked unit there whose bar was missed is that unit, not an object)
                blocked[i] = 'something that is not a unit (unrecognised)'
        if blocked != self.roster.blocked:
            self.log(ev='bench_blocked', stage=scr.stage, slots=blocked)
        self.roster.blocked = blocked

    CHOICE_TITLE = (860, 812, 1060, 846)        # "Choose One" above a row of option cards

    def read_choice(self, frame):
        """An open 'Choose One' row (an item anvil, a reward): [(point to click, option name)], or
        [] when none is up. The names sit at fixed places for 2-5 cards; otherwise the row is
        searched with text detection (slow)."""
        from tft_player.screen import match_name
        h, w = frame.shape[:2]
        s = h / 1080
        S = lambda b: (round(w / 2 + (b[0] - 960) * s), round(b[1] * s), round(w / 2 + (b[2] - 960) * s), round(b[3] * s))
        title, = self.fresh_texts(frame, [S(self.CHOICE_TITLE)])
        if 'choose' not in (title or '').lower():
            return []
        from tft_player.trait_text import choice_names
        names = list(self.reader.item_names()) + choice_names()     # items, and trait choices (Kha'Zix, Primal)
        for n in (5, 4, 3, 2):
            centres = [962 + (k - (n - 1) / 2) * 265 for k in range(n)]
            txt = self.fresh_texts(frame, [S((c - 125, 955, c + 125, 995)) for c in centres])
            got = [match_name(t, names, 0.75) for t in txt]
            if all(got):
                return [((c, 915), g) for c, g in zip(centres, got)]
        # otherwise text detection over the cards: every card's lines, grouped by place, matched as a whole first
        # (then its biggest line). A card whose text matches no name is still a card, named None: the count must be
        # right (run 20261013-220557 5-5: Draven's 3 bounty cards were read as 2 - "10,000 damage." as Damage Amp,
        # "components" as 1 component - and the model's pick of the third card was refused as out of range)
        x0, y0 = round(w / 2 + (250 - 960) * s), round(900 * s)
        crop = frame[y0:round(1010 * s), x0:round(w / 2 + (1670 - 960) * s)]
        cards = []
        for cx, cy, ht, t in sorted(self.reader._detect(crop, 'choice'), key=lambda r: r[0]):
            if not t.strip():
                continue
            x = (x0 + cx - w / 2) / s + 960
            card = next((c for c in cards if abs(x - sum(c['x']) / len(c['x'])) < 100), None)
            if card is None:
                card = {'x': [], 'lines': []}
                cards.append(card)
            card['x'].append(x)
            card['lines'].append((cy, ht, t))
        out = []
        for c in cards:
            if not 330 <= sum(c['x']) / len(c['x']) <= 1595:
                continue                          # outside any card (5 cards: centres 432-1492): board text
            whole = ' '.join(t for _, _, t in sorted(c['lines']))
            biggest = max(c['lines'], key=lambda l: l[1])[2]
            out.append(((sum(c['x']) / len(c['x']), 915), match_name(whole, names, 0.75) or match_name(biggest, names, 0.75)))
        return sorted(out, key=lambda o: o[0][0])

    def fresh_texts(self, frame, boxes):
        """OCR of these boxes now, never the reader's cache: its 'crop barely changed' test kept "Choose One" and
        the four Blessing names after the screen closed, when a dark shop tooltip took the dark banner's place (run
        20261015-132144 4-3: a phantom second Primal screen, the model asked twice, moves blocked for 6 s)."""
        for b in boxes:
            self.reader._cache.pop(b, None)
        return self.reader._texts(frame, boxes, None)

    def pilot_watch_choice(self, frame, scr):
        """Pilot mode: a 'Choose One' row is up - take the option the comp wants most (its item
        list, then components by what they complete). True while it is up."""
        c = self.choice
        if time.time() - getattr(self, '_pchoice_at', 0) < 0.4:
            return bool(c)
        self._pchoice_at = time.time()
        opts = self.read_choice(frame)
        if not opts:
            if c and time.time() - c['seen'] > 2.0:
                self.log(ev='choice_closed', stage=self.phase.stage, picked=c.get('clicked'))
                self.choice = None
            return False
        if c is None:
            c = self.choice = {'seen': time.time(), 'opts': [n for _, n in opts]}
            self.log(ev='choice_open', stage=self.phase.stage, source='pilot', options=c['opts'])
        c['seen'] = time.time()
        if not c.get('clicked'):
            vals = [self.pilot.choice_value(n, self.roster, self.bench_items) if n else -1e9 for _, n in opts]
            k = max(range(len(opts)), key=lambda i: vals[i])
            try:
                self.inp.click(opts[k][0])
            except NotForeground:
                return True
            c['clicked'] = opts[k][1]
            self.note_evolution(c['opts'], opts[k][1])
            self.log(ev='choice_pick', stage=self.phase.stage, option=k, name=opts[k][1], options=c['opts'])
            print(f'{self.phase.stage}: choice - {opts[k][1]} (of {", ".join(c["opts"])})')
            self.status_extra['choice'] = f'took {opts[k][1]} of {", ".join(c["opts"])}'
            time.sleep(0.5)
        return True

    def watch_choice(self, frame, scr):
        """EVENT CHOICE_SCREEN_OPEN: a row of option cards replaced the shop. The model sees the
        screenshot and the state and answers with an option index; that card is clicked.
        True while a choice screen is up (nothing else is done then)."""
        if self.pilot:
            return self.pilot_watch_choice(frame, scr)
        from tft_player import bench
        if not self.choice and time.time() - getattr(self, '_choice_at', 0) < 0.3:
            return False
        self._choice_at = time.time()
        cards = bench.choice_cards(frame, fallback=not any(scr.shop or ()))
        # the "Choose One" title decides that a choice is up, and the option names come off the cards
        # (run 185805: shop art read as 2 cards five times - option 0 clicked on the shop; a 5-card
        # anvil read as 6 - "option 4 Giant Slayer" clicked Blue Buff; a component anvil's cards unseen)
        h, w = frame.shape[:2]
        s = h / 1080
        box = self.CHOICE_TITLE
        tbox = (round(w / 2 + (box[0] - 960) * s), round(box[1] * s), round(w / 2 + (box[2] - 960) * s), round(box[3] * s))
        # read fresh only while a choice is open (to see it close): a fresh OCR every 0.3 s for nothing cost 9 % of the
        # loop; an opening is seen by the cached read, which follows any real change of the crop
        title, = self.fresh_texts(frame, [tbox]) if self.choice else self.reader._texts(frame, [tbox], None)
        named = []
        if 'choose' not in (title or '').lower():
            cards = []
        else:
            named = (self.choice or {}).get('named') or self.read_choice(frame)     # [(1080p point, option name)]
            if named and len(named) != len(cards):
                cards = [p for p, _ in named]            # the names read off the cards beat the colour scan
        c = self.choice
        if not cards:
            # still waiting for the model's answer: the screen stays "open" (run 152203 4-3: the cards were
            # not read for 2 s, the record was dropped, and the model's answer - Infinity Edge, shown in
            # the panel - found no choice to click; the game picked a Void Staff on timeout)
            waiting = c and c['pick'] is None and not c.get('clicked') and time.time() - c['asked'] < 25
            if c and time.time() - c['seen'] > 2.0 and not waiting:
                self.log(ev='choice_closed', stage=self.phase.stage, picked=c.get('clicked'))
                self.choice = None                 # the screen is gone
            return False
        if c is not None and c.get('clicked') and time.time() - c.get('clicked_at', time.time()) > 2.5:
            # cards still up 2.5 s after our click: a NEW choice screen (another container, a reward). Its cards are
            # read afresh: the old record's names and places are another row (release run 20261016-141957 3-6: the
            # second Primal screen has 3 cards, the first had 4 - Bear was clicked at the 4-card place, in the gap)
            self.log(ev='choice_closed', stage=self.phase.stage, picked=c.get('clicked'), how='replaced by a new one')
            c = self.choice = None
            named = self.read_choice(frame)
            cards = bench.choice_cards(frame, fallback=not any(scr.shop or ()))
            if named and len(named) != len(cards):
                cards = [p for p, _ in named]
            if not cards:
                return True
        lp = getattr(self, '_late_pick', None)
        if c is None and lp and time.time() - lp[0] < 20:
            # an answer that came after its record was dropped, and the cards are still up: use it
            self._late_pick = None
            c = self.choice = {'asked': lp[0], 'pick': (lp[1], lp[2]), 'cards': len(cards), 'seen': time.time()}
        if c is None:
            c = self.choice = {'asked': time.time(), 'pick': None, 'cards': len(cards), 'seen': time.time(),
                               'named': named}
            names = [n for _, n in named] if len(named) == len(cards) and all(n for _, n in named) else None
            planned = self.planned_choice(names)
            if planned is not None:
                # the model planned this choice in its briefing (the Primal Blessing): taken at once, no model call -
                # it is told (the user, 2026-10-15: 9 s for a planned pick; it should take under 1 s)
                c['pick'], c['planned'] = planned, True
                self.log(ev='choice_planned', stage=self.phase.stage, option=planned[0], name=planned[1], of=names)
                self.news_event('choice_taken', self.phase.stage, f'choice screen ({", ".join(names)}): took '
                                f'{planned[1]}, as you planned in your briefing')
            else:
                self.ask_choice(scr, len(cards), names)
        c['seen'] = time.time()
        if c['pick'] is not None and not c.get('clicked'):
            i, name = c['pick']
            from tft_player.screen import match_name
            by_name = [k for k, (_, n) in enumerate(c.get('named') or [])
                       if name and n and match_name(str(name), [n], 0.75)] if len(c.get('named') or []) == len(cards) else []
            if len(by_name) == 1 and by_name[0] != i:
                self.log(ev='choice_by_name', option=i, name=name, card=by_name[0])
                i = by_name[0]                       # the card that carries the name the model chose
            if 0 <= i < len(cards):
                self.inp.click(cards[i])
                # off the shop at once: a cursor left on a shop card opens its tooltip there, over the cards' place
                self.inp.hover((960, 330), steps=2, duration=0.03)
                c['clicked'], c['clicked_at'] = name or i, time.time()
                card_names = [n for _, n in c.get('named') or []]
                if len(card_names) == len(cards):
                    self.note_evolution(card_names, card_names[i])
                self.log(ev='choice_pick', stage=self.phase.stage, option=i, name=name, of=len(cards))
                print(f'{self.phase.stage}: choice - option {i} of {len(cards)}: {name}')
                time.sleep(0.5)
            else:
                self.log(ev='choice_bad_index', option=i, of=len(cards))
                c['pick'] = None
        return True

    def add_pending_give(self, name, uid):
        """The model took an item on a choice screen "to" a unit: it is given to that unit as soon as it
        is on the item bench (run 221257 4-3: a Radiant item meant for Nidalee sat on the bench for rounds)."""
        from tft_player.screen import ScreenReader, match_name
        item = match_name(name, ScreenReader.item_names(), 0.8)
        if not item:
            self.news_event('give_not_planned', self.phase.stage,
                            f'"{name}" is not an item name in the game data: give it in a plan once it is on the item bench')
            return
        self.pending_gives = getattr(self, 'pending_gives', []) + [{'item': item, 'id': uid, 'plans': 0, 'at': time.time()}]
        self.log(ev='pending_give', stage=self.phase.stage, item=item, unit=uid)

    def inject_gives(self, rp):
        """Each pending give goes into the running round plan's items once (into at most 3 plans, within
        4 minutes); it is dropped when the unit holds the item."""
        base = lambda n: re.sub(r'\s*\(.*$', '', n or '')
        keep = []
        for g in getattr(self, 'pending_gives', []):
            u = self.roster.resolve(g['id'])
            if u is None or any(base(i) == g['item'] for i in u.items) or g['plans'] >= 3 \
                    or time.time() - g['at'] > 240:
                continue
            keep.append(g)
            done = getattr(rp, 'gives_injected', set())
            if (g['item'], g['id']) not in done:
                rp.gives_injected = done | {(g['item'], g['id'])}
                g['plans'] += 1
                items = rp.p.setdefault('items', [])
                k = max([e.get('k', -1) for e in items] + [-1]) + 1
                items.append({'component': g['item'], 'id': u.id, 'k': k})
                self.log(ev='give_injected', stage=self.phase.stage, item=g['item'], unit=u.id)
        self.pending_gives = keep

    def planned_choice(self, names):
        """(option index, name) of the first planned trait choice (briefing trait_choices, e.g. Primal: Tiger, then
        Bear) on this screen that was not taken yet, or None."""
        if not names:
            return None
        taken = self.__dict__.setdefault('trait_taken', [])
        for trait, want in (getattr(self, 'trait_choices', None) or {}).items():
            for name in (want if isinstance(want, list) else [want]):
                if name in names and (trait, name) not in taken:
                    taken.append((trait, name))
                    return names.index(name), name
        return None

    def trait_choice_text(self):
        """The trait choices (e.g. Primal Blessings) the model planned in its briefing, as text."""
        tc = getattr(self, 'trait_choices', None) or {}
        return '; '.join(f"{t}: {', then '.join(map(str, v)) if isinstance(v, list) else v}" for t, v in tc.items())

    def trait_option_text(self, names):
        """Trait options on a choice screen (Kha'Zix's evolution): each trait's effect, from the game data."""
        try:
            from tft_player.trait_text import trait_line
            lines = [trait_line(x, 1) for x in names or [] if x in CATALOG.traits]
            return ('Trait effects (game data): ' + ' | '.join(l for l in lines if l) + '\n') if any(lines) else ''
        except Exception:
            return ''

    def ask_choice(self, scr, n, names=None):
        from tft_player import planner
        from tft_player.coach import ask_model
        self.trait_check(self.last_frame, scr)       # an item goes to a named unit: the board's names confirmed first
        co = self.container_opened
        lu = getattr(self, 'last_used', None)
        if co and time.time() - co[2] < 20:
            source = f'{co[0]} (you opened it on bench slot {co[1]})'
        elif lu and time.time() - lu[2] < 30:
            source = f'{lu[0]} (you used it on {lu[1].name or "?"}, unit {lu[1].id})'
        else:
            source = "not a container we opened (for example an augment's reward)"
        planned = self.trait_choice_text()
        # a short question of its own (the conversation already holds the game): the options are on the
        # screenshot; only what decides between them is repeated here
        board = ', '.join(f"{u.name or '?'}{'*' * u.star}" + (f"[{', '.join(u.items)}]" if u.items else '')
                          for u in self.roster.board()) or 'none'
        held = ', '.join(i for i in (self.bench_items or []) if i and i != 'unknown') or 'none'
        prompt = (f'CHOICE SCREEN, stage {self.phase.stage}. Source: {source}.\n'
                  f'The screenshot shows {n} options in a row where the shop was, numbered 0 (left) to {n - 1} '
                  f'(right). The game has a timer on it.\n'
                  + (f"The options, as read off the cards: {', '.join(f'{k} {x}' for k, x in enumerate(names))}.\n"
                     if names else '')
                  + self.trait_option_text(names)
                  + f'Board: {board}. Item bench: {held}.\n'
                  + (f'Your planned trait choices (from your briefing): {planned}.\n' if planned else '')
                  + 'Reply with two JSON objects and nothing else: first {"option": <index>, "name": "<the option '
                  'you take>"} - for an item, add "to": <unit id> and the controller gives it to that unit as soon '
                  'as it is on the item bench - then {"why": "<1-3 sentences>"}.')
        image, t0 = self.screen_jpeg(), time.time()
        self.log(ev='choice_open', stage=self.phase.stage, source=source, options=n)
        print(f'{self.phase.stage}: choice screen ({n} options, source: {source}) -> asking the model')

        def pick(obj):
            if isinstance(obj, dict) and isinstance(obj.get('option'), int):
                to = obj.get('to')
                if isinstance(to, int) and not isinstance(to, bool) and isinstance(obj.get('name'), str):
                    self.add_pending_give(obj['name'], to)
                if self.choice is not None:
                    self.choice['pick'] = (obj['option'], obj.get('name'))
                else:           # the record was dropped meanwhile: keep the answer for the screen's next read
                    self._late_pick = (time.time(), obj['option'], obj.get('name'))
                    self.log(ev='choice_late_pick', stage=self.phase.stage, option=obj['option'], name=obj.get('name'))

        def run():
            try:
                text = ask_model(prompt, image, on_first=pick, timeout=45, priority=True)
            except Exception as e:
                text = f'FAILED: {e}'
            finally:
                if getattr(self, 'model_reserved', None):          # the model held for this question
                    from tft_player import coach
                    self.model_reserved = None
                    coach.release_model()
            self.model_call('choice', prompt, text, time.time() - t0)
            print(f"[choice {time.time() - t0:.0f}s] {str(text)[:160]}")
        threading.Thread(target=run, daemon=True).start()

    # -- the roster against the screen (pilot mode) ---------------------------------------------
    SURVEYS_PER_ROUND = 2

    def check_roster(self, frame, scr):
        """Is the tracked board what the game shows? The trait panel counts the fielded units'
        traits; a count that differs twice in a row (units still moving read once wrong), a
        bench-slot mismatch after a buy, or the first planning phase at level 8 (the roll-down
        needs exact copies) -> survey: every unit's name read off its info panel. True when
        the roster was rebuilt (the caller skips this tick)."""
        if REPLAY or (scr.timer is not None and scr.timer < 8):
            self.checked_stage = scr.stage
            return False                        # a recording does not answer clicks
        if getattr(getattr(self, 'phase', None), 'rise', None) is not None:
            return False                        # the countdown just rose: maybe the fight began (see idle_survey)
        done = self.surveys.get(scr.stage, 0)
        if done >= self.SURVEYS_PER_ROUND:
            self.checked_stage = scr.stage
            return False
        reason = None
        eight_at = self.pilot.eight_at if self.pilot else (4, 2)
        unknown = [u for u in self.roster.units if u.name is None and self.tries.get(u.id, 0) >= 2]
        if (scr.level or 0) >= 8 and not self.surveyed_at8 and (scr.round or (0, 0)) >= eight_at:
            self.surveyed_at8 = True
            reason = 'level-8 roll-down: exact copies'
        elif self.need_survey:
            reason = self.need_survey
        elif unknown:
            reason = f'{len(unknown)} unit(s) still unnamed after two reads'      # never act on a "?"
        elif time.time() - getattr(self, '_trait_at', 0) > (0.5 if self.checked_stage != scr.stage else 1.5):
            self._trait_at = time.time()
            seen = self.reader.trait_panel(frame)
            self.learn_evolution(seen)
            mine = self.tracked_traits()
            bad = self.trait_diff(seen, mine)
            self._trait_bad = (self._trait_bad + 1) if bad and seen else 0
            if not bad or not seen:
                self.checked_stage = scr.stage      # the board matches: act on it
            if bad and tuple(bad) == self._trait_left:
                self._trait_bad = 0             # what a survey could not explain (an emblem's trait): not again
                self.checked_stage = scr.stage
            if self._trait_bad >= 2:
                reason = 'trait panel != tracked board: ' + ', '.join(
                    f'{t} screen {seen.get(t, 0)} tracked {mine.get(t, 0)}' for t in bad[:4])
        if not reason:
            return False
        self.need_survey = None
        self.surveys[scr.stage] = done + 1
        self.survey(scr, reason)
        self.checked_stage = scr.stage
        # a trait difference that survives the survey is not a tracking error: remember it
        got = self.cap.read()
        self._trait_left = tuple(self.trait_diff(self.reader.trait_panel(got[0]), self.tracked_traits())) if got else ()
        return True

    def board_check(self, frame, scr, places):
        """Model player: our health bars against where the roster says our units stand. Not every round -
        only when something is wrong: a step failed in a way that points at a wrong record (input_failed
        sets need_survey), or the hexes on screen and in the records differ for a lasting moment (a second
        Varus the game fielded, a unit never moved where we thought). Then every unit is read
        off its info panel and the roster takes the places the screen shows (survey) - a few seconds
        that save the game (run 152203: Nidalee stood at (0,2) while the roster said (1,0), eight item
        drops hit empty ground and two fights were played without the carry)."""
        if REPLAY or scr.timer is None or scr.stage is None:
            return False
        seen = {l[1:] for _, l in places if l[0] == 'board'}
        tracked = {u.loc[1:] for u in self.roster.board() if not u.missing}
        if seen == tracked:
            self._board_bad_since = None
        elif time.time() - self.round_started > 4.0:         # units have walked back to their hexes
            since = getattr(self, '_board_bad_since', None) or time.time()
            self._board_bad_since = since
            if time.time() - since > 1.5 and not self.need_survey:        # a lasting difference, not one bad read
                self.need_survey = (f'the board differs from the records: on screen only {sorted(seen - tracked)}, '
                                    f'tracked only {sorted(tracked - seen)}')
            if time.time() - since > 1.5 and self.phase.planning:
                self.find_rider(scr, tracked - seen, seen - tracked)
            if time.time() - since > 3.0 and self.phase.planning:
                # an unverified or unnamed record on a hex with no health bar for 3 s is no unit: set it aside now,
                # not at an idle survey that a busy round never reaches (run 185805: a phantom "?" at (0,0) from 4-2
                # to 5-3 - three moves failed on it and it held Aphelios's corner)
                # (a BFF is gone whenever Sprykin drops below 3: it comes back by a read of its panel)
                # a named unit too when the trait panel agrees it is not fielded (trait_gone): two witnesses
                gone = getattr(self, 'trait_gone', set())
                for u in [u for u in self.roster.board() if u.loc[1:] in tracked - seen
                          and (u.unverified or not u.name or u.slotless or u.id in gone)]:
                    u.missing = 'no unit is seen at its hex (no health bar for 3 s)'
                    self.news_event('unit_missing', scr.stage, f'{u.name or "?"} (unit {u.id}) at hex {u.loc[1:]}: no unit '
                                                               f'is seen there - set aside (kept, not deleted)')
        return False                                          # the check itself waits for idle time (idle_survey)

    def find_rider(self, scr, empty_hexes, new_hexes):
        """The game's own rider pick at combat start ("... Rider was auto assigned!"): the Sprykin it picked leaves
        its hex and rides the BFF on the BFF's hex (run 20261012-071645 2-5; run 20261010-235723 4-5). A fielded
        Sprykin missing from its hex while the BFF's old hex shows a unit no record holds is the rider: its place
        becomes that hex."""
        # also when the BFF was only seen with its rider on it (run 20261012-194821 3-2: Sprykin 3 came during 3-1, the
        # game seated Veigar at combat start, and the unit at (1, 4) that opens no panel was recorded as the BFF while
        # Veigar's own hex (0, 3) stood empty): a recorded BFF plus one Sprykin gone from its hex = that Sprykin rides it
        bff = next((u for u in self.roster.units if u.name == BFF and not u.missing), None)
        hexes = bff.loc if bff is not None else getattr(self, 'bff_hex', None)
        if hexes is None and not getattr(self, 'bff_assigned', False) and len(new_hexes) == 1 \
                and self.sprykin_fielded() >= 3:
            # ... and when the BFF itself was never seen: the game seats a Sprykin on it in the planning phase that
            # Sprykin 3 comes (run 20261012-233635 2-5: Rammus made Sprykin 3, Veigar went onto the BFF between Azir
            # and its own hex, and the bot knew only at 2-6) - one Sprykin gone, one new hex = that Sprykin on the BFF
            hexes = ('board',) + tuple(next(iter(new_hexes)))
        if hexes is None or any(getattr(u, 'riding', None) for u in self.roster.units):
            return
        if bff is None and hexes[1:] not in new_hexes:
            return
        gone = [u for u in self.roster.fielded() if u.loc[1:] in empty_hexes and u.info
                and 'Sprykin' in u.info.traits and not u.unverified]
        if len(gone) == 1:
            old = gone[0].loc[1:]
            if bff is not None:
                self.roster.units.remove(bff)
            self.bff_hex, self.bff_assigned = hexes, True
            gone[0].loc, gone[0].riding = hexes, True
            self.news_event('bff_rider', scr.stage, f'{gone[0]} rides the BFF (the game chose it: it left its hex '
                                                    f'{old} for the BFF at {hexes[1:]})')

    def idle_survey(self, scr):
        """A requested board check, only in idle time (the user, 2026-10-07: only when something is wrong
        AND nothing else is to do; it stops the moment the model's plan comes): no plan steps waiting, a
        planning phase with time left. -> True when it ran (even if a plan cut it short)."""
        if not self.need_survey or scr.timer is None or scr.timer < 9 or REPLAY:
            return False
        if getattr(getattr(self, 'phase', None), 'rise', None) is not None:
            # a countdown rise read once counts as the fight on its second read: until then the "time left" may be the
            # fight's clock (benchmark game 20261018-174944 3-5: a 7 s roll-down ended in combat, the next tick read 30
            # s, a board check ran in the fight, saw no bench bars and set six bench units missing - dropped at 3-7)
            return False
        if time.time() - self.round_started < 2.5:
            return False                                      # units still walking back after the fight
        if self.plan_asked != scr.stage:
            # this round's plan (a queued next_round, or the question) starts first: at 5-3 of run 185805 a
            # 7.5 s survey ran before the queued Amumu-for-Rakan swap, which then started with 3 s left
            return False
        rp = self.round_plan                                  # (called only when the plan had no step to take)
        if self.surveys.get(scr.stage, 0) >= 1:
            self.need_survey = None
            return False
        reason = self.need_survey
        self.need_survey = None
        self.surveys[scr.stage] = self.surveys.get(scr.stage, 0) + 1
        plan0 = rp

        def interrupted():
            # a new plan with steps to do arrived: the check gives way
            cur = self.round_plan
            return cur is not None and cur is not plan0 and cur.pending(self.roster)
        self.survey(scr, reason, stop=interrupted)
        self._board_bad_since = None
        return True

    def tracked_traits(self):
        """The traits the game's panel should show for the board we track: each fielded champion counted once,
        plus the trait of an emblem a unit holds (if it lacks that trait), the trait a unit evolved into
        (Kha'Zix) and the extra count the game data gives (Elder Dragon: +2 Riftbeast)."""
        from tft_player.state import board_traits
        return board_traits(self.roster.board())

    def learn_evolution(self, seen):
        """Kha'Zix evolves into one trait of his choice for good (Rival, game data). The choice we click is recorded
        (note_evolution); one the game made without us (its timer) shows on the panel: exactly one of his options
        one higher than the tracked board, everything else equal, with a fielded Kha'Zix that has not evolved.
        Then he evolved into that one. -> the trait learned, or None."""
        from tft_player.trait_text import evolutions
        if not seen:
            return None
        mine = self.tracked_traits()
        diff = {t: seen.get(t, 0) - mine.get(t, 0) for t in set(seen) | set(mine) if seen.get(t, 0) != mine.get(t, 0)}
        if len(seen) >= 9:                                    # 9+ rows: the rest hide behind '4+'
            diff = {t: d for t, d in diff.items() if t in seen}
        if len(diff) != 1:
            return None
        (t, d), = diff.items()
        for name, opts in evolutions().items():
            if d == 1 and t in opts and any(u.name == name and not u.evolved for u in self.roster.board()):
                self.roster.evolve(name, t)
                self.log(ev='evolved', stage=self.phase.stage, unit=name, trait=t, how='trait panel')
                self.news_event('evolved', self.phase.stage, f'{name} evolved into {t} (the trait panel shows it)')
                return t
        return None

    def note_evolution(self, options, chosen):
        """A choice screen whose options are an evolving champion's traits (Kha'Zix: Executioner, Rapidfire,
        Ravager, Spellweaver) and the one clicked: that champion now has it for good."""
        from tft_player.trait_text import evolutions
        opts = {o for o in options or [] if o}
        for name, choices in evolutions().items():
            if chosen in choices and len(opts) >= 2 and opts <= set(choices) \
                    and any(u.name == name for u in self.roster.units):
                self.roster.evolve(name, chosen)
                self.log(ev='evolved', stage=self.phase.stage, unit=name, trait=chosen, how='choice clicked')
                return True
        return False

    def trait_equation(self, seen):
        """The board as an equation, traits as the variables: the game's trait panel (seen) against the traits of
        the board we track. -> (bad traits, hypotheses). A hypothesis is ONE record wrong in a way that balances
        every compared trait: ('replace', unit, champion) - that unit is really that champion; ('add', None,
        champion) - an untracked unit of that champion is fielded; ('remove', unit, None) - that unit is not
        fielded. Hypotheses only decide what is read first; the read decides."""
        mine = self.tracked_traits()
        full = len(seen) < 9                                  # 9+ rows: the rest hide behind '4+'
        keys = set(seen) | ({t for t, c in mine.items() if c} if full else set())
        bad = sorted(t for t in keys if seen.get(t, 0) != mine.get(t, 0))
        if not bad:
            return bad, []
        board = [u for u in self.roster.board() if u.name]
        names = Counter(u.name for u in board)
        from tft_player.trait_text import trait_bonus

        def tr(n, unit=None):
            """A champion's count on the panel: its traits once (and what this unit evolved into), plus the
            game data's extra (Elder Dragon: +2 Riftbeast)."""
            own = set(CATALOG.unit(n).traits) if n and CATALOG.unit(n) else set()
            own |= set(unit.evolved) if unit is not None else set()
            out = Counter(own)
            out.update({t: k for t, k in trait_bonus(n).items() if t in own})
            return out

        def balanced(m):
            return all(seen.get(t, 0) == m.get(t, 0) for t in (set(seen) | ({t for t, c in m.items() if c} if full else set())))
        hyps = []
        for u in board:
            less = Counter(mine)
            if names[u.name] == 1:
                less.subtract(tr(u.name, u))                  # its traits leave only if it is its champion's only copy
            if balanced(less):
                hyps.append(('remove', u, None))
            for c in CATALOG.units:
                if c == u.name:
                    continue
                more = Counter(less)
                if c not in names:
                    more.update(tr(c))
                if balanced(more):
                    hyps.append(('replace', u, c))
        for c in CATALOG.units:
            if c not in names:
                more = Counter(mine)
                more.update(tr(c))
                if balanced(more):
                    hyps.append(('add', None, c))
        return bad, hyps

    def trait_check(self, frame, scr):
        """Confirmation before a model call that uses the board's champions (round plans and updates, a choice that
        gives an item to a unit - the user, 2026-10-08): the game's trait panel against the board we track. Equal:
        the board's identities are confirmed. Different: the units the difference points at are read now (at most
        3; first the ones a single-error hypothesis names) and their records fixed before the question goes out.
        With too little time left (< 6 s, or units still walking back), the model is told and the check runs
        first thing in the next planning phase. A difference that survives the reads (an augment's extra trait,
        a hidden '4+' row) is remembered, not read again. Never in combat (the user: no checks once the
        champions move)."""
        if REPLAY or frame is None or not self.phase.planning:
            return
        seen = self.reader.trait_panel(frame)
        if not seen:
            return
        # the panel must hold still: one read under an opening choice screen gave 2 of 5 traits (run 010416 2-1)
        time.sleep(0.15)
        got = self.cap.read()
        if not got or self.reader.trait_panel(got[0]) != seen:
            return
        t0 = time.time()
        self.learn_evolution(seen)
        bad, hyps = self.trait_equation(seen)
        # records the panel says are not fielded: with no health bar at their hex either, they are set aside
        # (board_check) - a named unit too (run 20261013-220557 4-5: the sold Murkwolf's record stayed on (2, 3))
        self.trait_gone = {u.id for k, u, _ in hyps if k == 'remove'}
        if not bad:
            self.trait_note, self.trait_pending = None, None
            return
        if (scr.stage, tuple(bad)) == getattr(self, '_trait_left', ()):
            return                                            # read this round and still unexplained: not again this round
        hyp_text = '; '.join(f"{'unit ' + str(u.id) + ' at ' + str(tuple(u.loc[1:])) + ' is ' + c if k == 'replace' else 'an untracked ' + c + ' is fielded' if k == 'add' else 'unit ' + str(u.id) + ' is not fielded'}"
                             for k, u, c in hyps[:3])
        mine = self.tracked_traits()
        diff = ', '.join(f'{t} screen {seen.get(t, 0)} / tracked {mine.get(t, 0)}' for t in bad)
        settled = self.phase.planning and time.time() - self.round_started > 2.5
        if not settled or scr.timer is None or scr.timer < 6:
            # nothing can be read now: the model is told, and the check comes first in the next planning phase
            self.trait_note = (f'Trait panel check: the board differs from the records ({diff})'
                               + (f'; most likely {hyp_text}' if hyp_text else '') + '. Being checked.')
            self.trait_pending = scr.stage
            self.log(ev='trait_check', stage=scr.stage, bad=bad, hyps=hyp_text, read=[], deferred=True)
            return
        # what to read: the units the hypotheses name, then the rest the difference points at
        over = {t for t in bad if mine.get(t, 0) > seen.get(t, 0)}
        under = {t for t in bad if seen.get(t, 0) > mine.get(t, 0)}
        traits = lambda u: set(CATALOG.unit(u.name).traits) if u.name and CATALOG.unit(u.name) else set()
        board = self.roster.board()
        named = [u for k, u, c in hyps if u is not None]
        rest = sorted([u for u in board if not u.name or u.unverified or traits(u) & over or (under and not traits(u) & under)],
                      key=lambda u: (not (u.unverified or not u.name), not (traits(u) & over)))
        suspects, ids = [], set()
        for u in named + rest:                              # in order, each unit once
            if u.id not in ids:
                ids.add(u.id)
                suspects.append(u)
        bars = B.green_bars(frame)
        _, _, places = B.bar_places(frame, exclude=self.watcher.tac[1], bars=bars)
        tracked = {u.loc for u in board}
        untracked = [loc for _, loc in places if loc[0] == 'board' and loc not in tracked]
        if any(k == 'add' for k, _, _ in hyps):
            todo = [('hex', l) for l in untracked[:2]] + [('unit', u) for u in suspects]
        else:
            todo = [('unit', u) for u in suspects] + [('hex', l) for l in untracked]
        todo = todo[:3]
        fixed, reads = [], []
        try:
            k = 0
            while k < len(todo) and k < 6:
                kind, x = todo[k]
                k += 1
                loc = x.loc if kind == 'unit' else x
                name, stars = self.trusted_read(loc, bars)
                reads.append(f'{loc[1:]}: {name or "no read"}{"*" * (stars or 0)}')
                if not name:
                    continue
                u = self.roster.at(loc)
                if u is None:
                    from tft_player.state import Owned
                    back = next((m for m in self.roster.missing() if m.name == name), None)
                    u = back or Owned(name, stars or 1, loc, ours=True)
                    if back is None:
                        self.roster.units.append(u)
                    u.missing, u.loc = None, loc
                    fixed.append(f'{name} found at {loc[1:]}')
                elif u.name != name or (stars and stars != u.star):
                    if u.name and not u.unverified and u.name != name:
                        again = self.trusted_read(loc, bars)[0]
                        reads.append(f'{loc[1:]} again: {again or "no read"}')
                        if again != name:
                            continue                          # a confirmed unit is renamed only by two agreeing reads
                    fixed.append(f'unit {u.id} at {loc[1:]} is {name}{"*" * (stars or u.star)}, not '
                                 f'{u.name or "?"}{"*" * u.star}')
                    # the champion read here was recorded on another unit: that record is wrong too - read it next
                    # (run 185805 4-3: Aphelios read where Vi was recorded; the recorded "Aphelios" was Malphite)
                    for o in board:
                        if o is not u and o.name == name and o.loc != loc and o.id not in ids:
                            ids.add(o.id)
                            todo.insert(k, ('unit', o))
                u.name, u.star = name, stars or u.star
                u.unverified, u.sure = None, True
                if not self.trait_equation(seen)[0]:
                    break                                     # balanced: the board is confirmed
        except NotForeground:
            return
        bad2, _ = self.trait_equation(seen)
        self._trait_left = (scr.stage, tuple(bad2))
        self.trait_pending = None
        self.trait_note = (f'Trait panel check: still differs after reading ({", ".join(bad2)}) - maybe an '
                           f'augment or emblem trait, or a hidden row.') if bad2 else None
        self.log(ev='trait_check', stage=scr.stage, screen=dict(seen), tracked=dict(mine), bad=bad, hyps=hyp_text,
                 read=[repr(x) if kind == 'unit' else list(x) for kind, x in todo], reads=reads, fixed=fixed, left=bad2,
                 seconds=round(time.time() - t0, 2))
        if fixed:
            self.news_event('unit_corrected', scr.stage, 'trait panel check: ' + '; '.join(fixed))
            self.alert('RECORD', 'trait check fixed ' + '; '.join(fixed)[:80], 'State')
        print(f'{scr.stage}: trait check {bad} -> read {len(todo)}' + (f', fixed: {fixed}' if fixed else '')
              + (f', left {bad2}' if bad2 else ''))

    def trusted_read(self, loc, bars):
        """A unit's panel read that counts only when it can be this unit's: the click on its own bar's body; a
        name that a CONFIRMED unit on a neighbouring hex has (not the one recorded here) is a neighbour's
        panel (run 152104 5-5) - tried once more higher on the body, then not trusted. -> (name, stars)."""
        from tft_player.state import _hex_near
        others = [self.unit_point(u.loc, bars) for u in self.roster.units if u.name and u.loc != loc and not u.missing][:2]
        name, stars, _ = self.read_fresh(self.unit_point(loc, bars), others)
        if not name or loc[0] != 'board':
            return name, stars
        here = self.roster.at(loc)
        neigh = {u.name for u in self.roster.board() if u.loc != loc and _hex_near(u.loc[1:], loc[1:])
                 and u.name and not u.unverified}
        if name in neigh and (here is None or here.name != name):
            px, py = self.unit_point(loc, bars)
            name2, stars2, fresh = self.read_unit((px, py - 25))
            if fresh and name2 and name2 not in neigh:
                return name2, stars2
            self.log(ev='read_untrusted', loc=list(loc), read=name, neighbours=sorted(neigh))
            return None, None
        return name, stars

    @staticmethod
    def trait_diff(seen, mine):
        """Traits whose count on the game's panel differs from the tracked board's. With 9+ rows
        shown more may be hidden ('4+'), so then only the rows shown count."""
        if not seen:
            return []
        keys = set(seen) | {t for t, c in mine.items() if c >= min(CATALOG.traits.get(t) or (1,))}
        return sorted(t for t in keys if seen.get(t, 0) != mine.get(t, 0) and (t in seen or len(seen) < 9))

    def confirm_before_field(self, unit, src, scr):
        """The user, 2026-10-09: a bench unit is clicked and read before it is dragged onto the board. -> True
        when it reads as another champion: its record is fixed (and the other records of that champion are
        read too - a swap), this unit's remaining plan steps (its hex, its items) are dropped, the model is
        told (an update question), and nothing is dragged. A read that gives no name changes nothing.
        (run 185805 4-3: "Aphelios" from bench slot 4 was Malphite; it was fielded and got Guinsoo's.)"""
        others = [self.unit_point(o.loc) for o in self.roster.units
                  if o.name and o.name != unit.name and o.loc[0] == 'bench' and not o.missing][:2]
        try:
            name, stars, _ = self.read_fresh(self.unit_point(src), others)
        except NotForeground:
            return False
        self.log(ev='confirm_before_field', stage=scr.stage, unit=repr(unit), read=name, stars=stars)
        if not name or name == unit.name:
            if name:
                unit.unverified, unit.sure = None, True
                if stars:
                    unit.star = stars
            return False
        old = f'{unit.name}{"*" * unit.star}'
        unit.name, unit.unverified, unit.sure = name, None, True
        if stars:
            unit.star = stars
        self.reread = getattr(self, 'reread', set()) | {o.id for o in self.roster.units if o is not unit and not o.missing
                                                         and (o.name == name or o.unverified)}
        rp = self.round_plan
        if rp is not None:
            for key in ('board', 'items'):
                if isinstance(rp.p.get(key), list):
                    rp.p[key] = [e for e in rp.p[key] if not (isinstance(e, dict) and e.get('id') == unit.id)]
        self.news_event('unit_corrected', scr.stage, f'unit {unit.id} on bench slot {src[1]} is {name}{"*" * unit.star}, '
                                                      f'not {old} (read before it was moved onto the board): not moved; '
                                                      f'its hex and items in your plan are dropped')
        self.fail_cause = f'unit {unit.id} reads {name}, not {old}: not moved onto the board'
        return True

    def bench_sell_blocked(self, frame, unit):
        """Before E on a bench unit, its slot is looked at: in 14 of 19 'failed' sells of four games on 2026-10-18 the
        slot had no health bar before the E. -> 'wait' when no bench bar is drawn at all yet (a planning phase opening:
        4-7 at 29 s showed a full bench with no bars), 'stale' when other bars are drawn but none in its slot and its
        tile does not read as a unit (the unit had gone or moved: the same frame showed a Brambleback where Cinderling
        was tracked), else None. Only in a planning phase: in combat no bench bar is ever drawn, and 130 of 163 bench
        sells in combat (2026-10-15..18) went through."""
        from tft_player import bench as BN
        if not getattr(getattr(self, 'phase', None), 'planning', False):
            return None
        bars = B.bar_places(frame, exclude=self.watcher.tac[1] if getattr(self, 'watcher', None) else None)[0]
        if not any(bars):
            return 'wait' if any(u.loc[0] == 'bench' and u is not unit for u in self.roster.units) else None
        if not bars[unit.loc[1]] and BN.classify(frame, bars)[unit.loc[1]]['kind'] != 'unit':
            return 'stale'
        return None

    def occupant_differs(self, occupant, dst, scr):
        """Before a bench unit is swapped onto an occupied hex, the unit standing there is read: the swap sends it to the
        bench. -> True when it is another champion than the records say: nothing is dragged, the board is read again
        (a survey) and the model is told. A read that gives no name changes nothing. (Benchmark game 20261018-115603
        4-1: a swap meant to bench Scuttlecrab benched Pebbles, the 3-item carry, for the fight - the records had
        her at another hex since a 3-7 correction.)"""
        try:
            got = self.cap.read()
            name, stars, _ = self.read_fresh(self.unit_point(dst, B.green_bars(got[0]) if got else None), [])
        except NotForeground:
            return False
        self.log(ev='confirm_occupant', stage=scr.stage, unit=repr(occupant), read=name, stars=stars)
        if not name or name == occupant.name:
            if name:
                occupant.unverified, occupant.sure = None, True
            return False
        self.need_survey = f'hex {dst[1:]} holds {name}, not {occupant.name} (read before a swap)'
        self.fail_cause = (f'hex {dst[1:]} holds {name}, not {occupant}: not swapped (it would have gone to the bench); '
                           f'the board is read again')
        self.news_event('unit_corrected', scr.stage, f'hex {dst[1:]} holds {name}{"*" * (stars or 1)}, not {occupant} '
                                                      f'(read before a swap onto it): not swapped; the board is read again')
        return True

    def unit_point(self, loc, bars=None):
        """Where to click, sell or drag a unit: its body just under its health bar - on the bench too (the bench slot's
        point is at its feet, below a small model: E there sold nothing and panel reads read nothing - run
        20261015-161625: LeBlanc at 6-3, Kayle at 2-6). Without a bar: body height over the slot / hex."""
        if loc[0] == 'bench':
            own = [b for b, place in B.bar_places(None, bars=bars or [])[2] if place == loc]
            if own:
                return (own[0][0], own[0][1] + 45)
            return (B.BENCH[loc[1]][0], self.BENCH_BAR_Y + 45)
        hx, hy = B.hex_xy(loc[1], loc[2])
        # only the bar the occupancy read gives this hex: the nearest bar by x can be a neighbour's
        # (run 152104 5-5: (1,0)'s point took 2-star Aphelios's bar at (0,0), 28 px off vs Nidalee's
        # own 30 px, and three board checks read Nidalee as "Aphelios")
        own = [b for b, place in B.bar_places(None, bars=bars or [])[2] if place == loc]
        if own:
            bx, by = own[0]
            # a tall model's bar is far over its hex: 45 px under it is between Kayle's wings (161404: read nothing)
            return (bx, by + (45 if hy - by <= 110 else round((hy - by) * 0.55)))
        return (hx, hy - 15)

    def tall_hint(self):
        """For board.bar_places, in a planning phase: {hex: bar height} of our fielded units whose champion and star
        level carry a bar too high for nearest-hex matching (board.BAR_DY). Fights move units: nothing then."""
        if not getattr(getattr(self, 'phase', None), 'planning', False) or not B.BAR_DY:
            return {}
        return {u.loc[1:]: B.BAR_DY[(u.name, u.star)] for u in self.roster.board()
                if not u.missing and (u.name, u.star) in B.BAR_DY}

    def learn_tall_bars(self, found, bars):
        """A board check's reads: a unit read at its hex centre (no bar of its own) under a bar 95-175 px up that is
        unplaced or gave an empty read at the hex it was matched to is a tall model - that bar is its own. Its height
        is learned for the champion and star level (board.BAR_DY) and the empty read is dropped: no '?' record.
        (Benchmark game 20261018-161404: Kayle 2-star at (0,0), bar 142 px up; four board checks 3-6..4-6, two
        phantom '?' units.) -> the reads to keep."""
        places = B.bar_places(None, bars=bars or [])[2]
        own = {loc for _, loc in places}
        empty = {tuple(f[0]) for f in found if f[0][0] == 'board' and not f[1]}
        placed = {tuple(b) for b, _ in places}
        drop = set()
        for f in found:
            loc, name = tuple(f[0]), f[1]
            if loc[0] != 'board' or not name or loc in own:
                continue
            rec = self.roster.at(loc)
            stars = f[2] or (rec.star if rec is not None and rec.name == name else 1)    # (a badge can go unread)
            hx, hy = B.hex_xy(*loc[1:])
            above = [(b, pl) for b, pl in places if pl in empty and pl not in drop] + \
                    [(b, None) for b in bars or [] if tuple(b) not in placed]
            cands = [(b, pl) for b, pl in above if abs(b[0] - hx) < 45 and 95 <= hy - b[1] <= 175]
            if len(cands) != 1:
                continue
            b, pl = cands[0]
            B.BAR_DY[(name, stars)] = round(hy - b[1])
            if pl is not None:
                drop.add(pl)
            self.log(ev='tall_bar', unit=name, stars=stars, hex=list(loc[1:]), height=round(hy - b[1]),
                     read_as=list(pl[1:]) if pl else None)
        return [f for f in found if tuple(f[0]) not in drop]

    def tactician_off(self, point):
        """Our tactician stands over `point` (1080p; a unit we are about to click, read, drag or sell): it is walked to
        PARK first - a click there hit the tactician, an E-sell missed (run 20261015-161625 2-6: four panel reads of
        bench slot 1 read nothing, two sells did not land). Waits at most 1.2 s for it to move off."""
        if REPLAY:
            return
        box = self.watcher.tac[1] if getattr(self, 'watcher', None) else None
        covers = lambda b: b is not None and b[0] - 50 <= point[0] <= b[2] + 50 and b[1] - 20 <= point[1] <= b[3] + 170
        if not covers(box):
            return
        try:
            self.inp.click(PARK, 'right')
        except NotForeground:
            return
        self._parked = self.phase.stage
        self.log(ev='tactician_moved_off', point=[round(point[0]), round(point[1])])
        end = time.time() + 1.2
        while time.time() < end and covers(self.watcher.tac[1]):
            time.sleep(0.08)

    def read_unit(self, point):
        """Right-click a unit and read its info panel -> (name, stars, fresh). fresh: the panel
        changed after the click; unchanged means it already showed this champion (a duplicate)
        or the click missed. The panel's items are kept in self.panel_items."""
        self.tactician_off(point)
        got = self.cap.read()
        before = self.reader.panel_sig(got[0]) if got else None
        self.inp.click(point, 'right')
        for k in range(10):
            time.sleep(0.07)
            got = self.cap.read()
            if not got:
                continue
            sig = self.reader.panel_sig(got[0])
            changed = before is None or sig.shape != before.shape or \
                (np.abs(sig - before).max(axis=2) > 40).mean() >= 0.05
            if changed or k >= 6:
                name, stars = self.reader.read_panel(got[0])
                if name or k >= 6:
                    self.panel_items = read_panel_items(got[0]) if name else []
                    return name, stars, changed
        return None, None, False

    def read_fresh(self, point, others):
        """read_unit that never takes a stale panel: when the panel did not change (the same
        champion as before, or a missed click), show another unit first (`others`: points of
        units with a different name) and read again. -> (name, stars, items) or (None, None, [])."""
        name, stars, fresh = self.read_unit(point)
        if fresh:
            return name, stars, list(self.panel_items)
        for other in others[:2]:
            self.read_unit(other)
            name, stars, fresh = self.read_unit(point)
            if fresh:
                return name, stars, list(self.panel_items)
        return None, None, []

    def apply_partial_survey(self, found, scr, reason):
        """A board check stopped by the model's plan: each place read takes the unit read there (a tracked
        unit of that name and star level moves to it), unread places keep their records."""
        from tft_player.state import Owned
        changed = []
        for loc, name, stars, items in found:
            if not name:
                continue
            here = self.roster.at(loc)
            if here is not None and here.name == name and (stars is None or here.star == stars):
                here.unverified = None                   # confirmed: otherwise unchanged
                continue
            cands = [u for u in self.roster.units if u.name == name and (stars is None or u.star == stars)
                     and u is not here]
            u = min(cands, key=lambda x: (x.loc[0] != loc[0], x.loc)) if cands else None
            if here is not None and here is not u:
                # the record that stood here is somewhere else: kept, as not located (found again later)
                here.missing = f'a board check read {name} at its place'
            if u is None:
                u = Owned(name, stars or 1, loc, ours=True)
                self.roster.units.append(u)
            u.loc, u.unverified = loc, None
            u.items = [i for i in items if i != 'unknown']
            changed.append(f'{name} at {loc[1:]}')
        print(f'{scr.stage}: board check ({reason}) stopped for the plan after {len(found)} reads'
              + (f'; corrected: {", ".join(changed)}' if changed else ''))
        self.log(ev='survey_partial', stage=scr.stage, reason=reason, read=len(found), changed=changed)

    def check_borrowed_panels(self, found, bars):
        """A click can open a neighbour's panel (a bigger model in front of the hex). Two places that
        read the same champion, stars and items, where the records had a different confirmed unit on
        one of them: that place is read again at its hex centre. If it still shows the neighbour,
        its record stays (marked unverified) - a doubtful read never renames a confirmed unit."""
        def same(a, b):
            return a[1] and a[1] == b[1] and a[2] == b[2] and sorted(a[3]) == sorted(b[3]) \
                and (a[3] or (a[2] or 1) >= 2)       # equal items, or two 2-stars: one is borrowed
        for f in found:
            twin = next((g for g in found if g is not f and same(f, g)), None)
            rec = self.roster.at(f[0])
            if twin is None or rec is None or not rec.name or rec.name == f[1] or rec.unverified or rec.missing:
                continue
            hx, hy = B.hex_xy(*f[0][1:]) if f[0][0] == 'board' else B.BENCH[f[0][1]]
            px, py = self.unit_point(f[0], bars)
            # higher on its own body (away from a model in front), then the hex centre
            for point in ((px, py - 25), (hx, hy - 15)):
                self.read_unit(self.unit_point(twin[0], bars))          # show the twin, so a change is seen
                name, stars, fresh = self.read_unit(point)
                if fresh and name and (name, stars) != (twin[1], twin[2]):
                    f[1], f[2], f[3] = name, stars, list(self.panel_items)
                    break
            else:
                note = (f'a board check read {twin[1]}{"*" * (twin[2] or 1)} here, the same panel as at '
                        f'{tuple(twin[0][1:])}: kept as {rec.name}{"*" * rec.star}')
                f[1], f[2], f[3] = rec.name, rec.star, list(rec.items)
                f.append(note)
            self.log(ev='borrowed_panel', loc=list(f[0]), twin=list(twin[0]), record=rec.name, now=f[1],
                     kept=len(f) > 4)
            where = tuple(f[0][1:]) if f[0][0] == 'board' else f'bench {f[0][1]}'
            self.state_note = (time.time(), f'{rec.name} {where}: ' + ('read as its neighbour - kept' if len(f) > 4
                               else f're-read as {f[1]}'), 'muted' if len(f) > 4 else 'amber')

    def survey_records(self, found):
        """A finished board check's reads -> (new records, records dropped). Only the wrong records change (the
        user, 2026-10-09): a place read as the champion recorded there, or read without a name, keeps its record
        as it is; a place read as another champion takes that champion's record from elsewhere (or a new one);
        a record whose place showed nothing is kept as missing."""
        from tft_player.state import Owned
        old = list(self.roster.units)
        new, rest = [], []
        for f in found:                                   # 1. records the reads confirm: unchanged
            loc, name, stars, items, *kept = f
            here = next((x for x in old if x.loc == loc and not x.missing), None)
            if here is not None and (not name or kept or (here.name == name and (stars is None or here.star == stars))):
                if kept:
                    here.unverified = kept[0]             # its read was another unit's panel: the record stays
                elif name:
                    here.unverified = None
                    if stars:
                        here.star_assumed = False         # its star level read off its panel
                old.remove(here)
                new.append(here)
            else:
                rest.append(f)
        for loc, name, stars, items, *kept in rest:       # 2. the wrong or missing ones
            cands = [u for u in old if name and u.name == name and (stars is None or u.star == stars)]
            u = min(cands, key=lambda x: (x.loc[0] != loc[0], len(x.items) == 0)) if cands else None
            if u is None:
                u = next((x for x in old if x.loc == loc and not x.name), None)    # a '?' there takes the name
            if u is None:
                u = Owned(name, stars or 1, loc, ours=True)
            else:
                old.remove(u)
            u.loc, u.missing, u.name = loc, None, name or u.name
            if kept:
                u.unverified = kept[0]
            elif name:
                u.items = [i for i in items if i != 'unknown']      # what the panel shows it holds
                u.unverified = None
                if stars:
                    u.star, u.star_assumed = stars, False
            new.append(u)
        # 3. a unit is removed only by a sell, a merge or a game effect we saw: one this check did not find is
        # kept, as not located - a bar and a click can both miss (a monster or the tactician in front); a later
        # read names it again. Only nameless '?' records are dropped.
        for u in list(old):
            if u.name:
                u.missing = 'not seen in a board check (no health bar, no panel at its place)'
                new.append(u)
                old.remove(u)
                self.state_note = (time.time(), f'{u.name} {tuple(u.loc[1:]) if u.on_board else "bench " + str(u.loc[1])}: '
                                   'unseen - kept as missing', 'amber')
        return new, old

    def survey(self, scr, reason, stop=None):
        """Read every fielded and benched unit (health bars say where) off its info panel and
        rebuild the roster from what the game shows; tracked units keep ids and items when
        they match by name and stars."""
        t0 = time.time()
        tac_box = self.watcher.tac[1]
        # health bars flicker (a monster or effect in front, an animation): one frame can miss a
        # fielded unit, and a missed unit used to leave the records (run 152104 4-7: 2-star Sentinel
        # at (3,4) dropped, then Diana was moved onto its hex). Occupancy is the union of a few frames.
        seen = Counter()                                          # place -> frames that showed a bar there
        bars = None
        for k in range(3):
            got = self.cap.read()
            if not got:
                continue
            b_seen, h_seen = B.occupancy(got[0], exclude=tac_box)
            seen.update([('bench', i) for i, b in enumerate(b_seen) if b] + [('board', r, c) for r, c in h_seen])
            fb = [b for b in B.green_bars(got[0])
                  if not (tac_box and tac_box[0] <= b[0] <= tac_box[2] + 40 and tac_box[1] <= b[1] <= tac_box[3])]
            if bars is None or len(fb) > len(bars):
                bars = fb                                     # click points from the fullest frame
            if k < 2:
                time.sleep(0.08)
        if bars is None:
            return
        # a place seen in 2 of 3 frames holds a unit; one seen once (a unit still walking back after the
        # fight - run 213127 2-7 made two phantom '?' units of that) and a confirmed unit's place that no
        # bar showed are only probed: ONE click, and no panel there means no unit (a probe costs ~0.6 s;
        # read_fresh's retries made empty probes ~3 s each - the 4-3 check took 16.9 s)
        firm = sorted((l for l, n in seen.items() if n >= 2), key=lambda l: (l[0] != 'board', l[1:]))
        board_seen = {l[1:] for l in firm if l[0] == 'board'}
        probes = [l for l, n in seen.items() if n == 1]
        probes += [u.loc for u in self.roster.units if u.on_board and u.name and not u.missing and not u.unverified
                   and u.loc not in firm and u.loc not in probes]
        probes = sorted(set(probes), key=lambda l: (l[0] != 'board', l[1:]))[:4]
        found, partial = [], False
        try:
            for loc in firm:
                if stop is not None and stop():
                    partial = True                               # the model's plan came: it goes first
                    break
                shown = found[-1][1] if found else None          # the champion the panel shows now
                others = [self.unit_point(f[0], bars) for f in found if f[1] and f[1] != shown]
                name, stars, items = self.read_fresh(self.unit_point(loc, bars), others)
                found.append([loc, name, stars, items])
            for loc in ([] if partial else probes):
                if stop is not None and stop():
                    partial = True
                    break
                name, stars, fresh = self.read_unit(self.unit_point(loc, bars))
                if fresh and name:                               # the panel changed to a unit standing there
                    found.append([loc, name, stars, list(self.panel_items)])
                    if loc[0] == 'board':
                        board_seen.add(loc[1:])
            if not partial:
                self.check_borrowed_panels(found, bars)
                found = self.learn_tall_bars(found, bars)
        except NotForeground:
            return
        if partial:
            # cut short: only the places read are corrected; nothing is removed or marked missing
            self.apply_partial_survey(found, scr, reason)
            return
        if not any(l[0] == 'bench' for l in seen):
            # no bench health bar in any frame (a fight began, or the bars are not drawn yet): the bench was not read,
            # so its records stay as they are - not missing (174944 3-5: six bench units dropped a round later)
            read = {tuple(f[0]) for f in found}
            found += [[u.loc, u.name, u.star, list(u.items), 'not read: no bench health bar during a board check']
                      for u in self.roster.units if u.loc[0] == 'bench' and not u.missing and u.name
                      and tuple(u.loc) not in read]
        before = repr(self.roster)
        new, old = self.survey_records(found)
        renamed = [(f[0], f[1]) for f in found if f[1] and (r := self.roster.at(f[0])) is not None and r.name
                   and r.name != f[1] and not r.unverified and len(f) <= 4]
        if renamed:
            loc, name = renamed[0]
            self.alert('RECORD', f'board check: {self.roster.at(loc).name} at {loc[1:]} now read as {name}', 'State')
        kept = [u for u in new if u.missing and u.missing.startswith('not seen in a board check')]
        if kept:
            self.alert('RECORD', 'board check could not find ' + ', '.join(f'{u.name}{"*" * u.star}' for u in kept[:3])
                       + ' - kept as not located', 'State')
        old_locs = {u.id: (u.loc, u) for u in self.roster.units}
        self.roster.units = new
        self.roster.board_seen = len(board_seen)
        moved = [f'{u.name or "?"} (unit {u.id}) is at {tuple(u.loc[1:]) if u.on_board else "bench " + str(u.loc[1])}, '
                 f'not at {tuple(old_locs[u.id][0][1:]) if old_locs[u.id][0][0] == "board" else "bench " + str(old_locs[u.id][0][1])}'
                 for u in new if u.id in old_locs and old_locs[u.id][0] != u.loc]
        if moved:
            self.news_event('survey', scr.stage, 'board check: places corrected: ' + '; '.join(moved))
        elif 'model asked' in str(reason):           # the answer to the model's own request
            self.news_event('survey', scr.stage, f'the board check you asked for: all {len(new)} places match the records')
        took = time.time() - t0
        print(f'{scr.stage}: survey ({reason}) in {took:.1f}s -> {self.roster}')
        self.log(ev='survey', stage=scr.stage, reason=reason, seconds=round(took, 1), before=before,
                 after=repr(self.roster), found=[(list(f[0]), f[1], f[2], f[3]) for f in found],
                 lost=[repr(u) for u in old])
        self.status_extra['identify'] = f'survey ({reason}): {len(found)} units read'

    def identify_one(self, scr, combat=False, arrivals=False):
        """Read ONE unit off its info panel: a unit that arrived without our buy ('?'), or one an
        event may have changed (self.reread: a Wisp/augment that transforms units, a sell or move
        that did not confirm). Known units are never read again. arrivals: only '?' units on the bench,
        read at once (local time, the user 2026-10-09); the rest only in otherwise idle time - while the
        model thinks, or in combat (bench units only: fighting units move)."""
        reread = getattr(self, 'reread', set())
        unknown = [u for u in self.roster.units
                   if (u.name is None or (u.id in reread and not arrivals)) and self.tries.get(u.id, 0) < 4
                   and not u.missing and not (combat and u.on_board) and not (arrivals and u.on_board)]
        if not unknown or (not combat and scr.timer is not None and scr.timer < (1 if arrivals else 4)):
            return False
        # least-tried first: consecutive clicks go to different units, so the panel
        # switches between them (two copies of one champion look identical)
        u = min(unknown, key=lambda x: (self.tries.get(x.id, 0), x.on_board, x.loc))
        got = self.cap.read()
        if got and u.on_board and not REPLAY:
            # a fielded unit is read only where its health bar shows (run 221257 5-2: Aphelios on hex
            # (0,0) was clicked as the planning phase ended, no bar was there, the panel showed a
            # Leona 2-star, and the model then benched and stripped that "Leona" and sold it at 5-3)
            # (the bar must be assigned to THIS hex: the old 75 px / 140 px window took the bar of the unit on the
            # neighbouring hex - run 20261010-235723 4-3: a stale Azir record at (0,0) was "read" off the real Azir***
            # at (1,0), which was then recorded a second time)
            if u.loc not in {loc for _, loc in B.bar_places(got[0], exclude=self.watcher.tac[1])[2]}:
                return False
        self.tries[u.id] = self.tries.get(u.id, 0) + 1
        try:
            pre = got[0] if got else None         # the unit before the click: no selection ring yet
            before = self.reader.panel_sig(got[0]) if got else None
            # a fielded unit is clicked on its body under its health bar (the hex centre missed
            # the 1-1 carousel unit twice in the first pilot game)
            self.inp.click(self.unit_point(u.loc, B.green_bars(got[0]) if got else None), 'right')
            name, stars = None, None
            # The panel may still show the unit clicked before (or the click missed):
            # read it only once it has visibly changed after our click. An unchanged
            # panel names nobody - the unit stays '?' and is tried again, never guessed.
            for _ in range(10):
                time.sleep(0.08)
                got = self.cap.read()
                if not got:
                    continue
                sig = self.reader.panel_sig(got[0])
                if before is not None and sig.shape == before.shape and \
                        (np.abs(sig - before).max(axis=2) > 40).mean() < 0.05:
                    continue                     # unchanged: not (yet) this unit's panel
                name, stars = self.reader.read_panel(got[0])
                # the same champion as the panel showed before: maybe still that unit's panel (it can
                # change by itself, e.g. a merge to 2-star) - then the selection ring must be here.
                # (Only then: on recordings the ring test alone missed 62% of good reads.)
                if name and name == getattr(self, '_last_panel_name', None) and not self.selected_here(got[0], u.loc):
                    # the panel changed but shows another unit (run29 4-2: bench slot 3 read as the
                    # board Kog'Maw, whose panel had just changed to 2-star)
                    self.log(ev='read_not_selected', stage=scr.stage, unit=repr(u), read=name)
                    name, stars = None, None
                    continue
                if name:
                    self._last_panel_name = name
                    # the panel shows its items too: a carousel unit arrives holding one, and this read is what
                    # confirms the pick (25 of 25 carousels of 2026-10-10 were "could not be told": the names
                    # were read here, the items never)
                    self.panel_items = read_panel_items(got[0])
                    break
            # the panel is left open: it closes on our next click. (Escape is never
            # used - if the panel already closed, Escape opens the game menu.)
        except NotForeground:
            return False
        if name:
            g = getattr(u, 'gift', None)
            if g:
                u.gift = None
                if name != g['name']:
                    # not this gift: gifts landing together were named in slot order - the unit given this one's
                    # name takes the other name (read again too); else the gift is still to come
                    o = next((o for o in self.roster.units if o is not u and getattr(o, 'gift', None)
                              and o.gift['name'] == name), None)
                    if o is not None:
                        o.gift, o.name = g, g['name']
                        self.reread = reread | {o.id}
                        self.tries[o.id] = 0
                    else:
                        self.gifts = getattr(self, 'gifts', []) + [g]
            elif u.name is None:
                g = next((g for g in getattr(self, 'gifts', []) if g['name'] == name), None)
                if g:
                    self.gifts.remove(g)                                # a gift that arrived unnamed
            if u.id in reread:
                reread.discard(u.id)
                if u.name and (u.name != name or (stars and stars != u.star)):
                    self.news_event('unit_corrected', scr.stage, f'unit {u.id} is {name} {stars or u.star}-star, '
                                                                  f'not {u.name} {u.star}-star')
                    if u.name != name:
                        # a swap: the other records of this champion, and the unverified ones, are read too
                        # (run 185805 4-2: "Aphelios" was Malphite, and the real Aphelios was recorded as Vi)
                        self.reread = reread | {o.id for o in self.roster.units if o is not u and not o.missing
                                                and (o.name == name or o.unverified)}
            u.name = name
            u.unverified = None                  # read off its own info panel now
            u.sure = True
            if not u.on_board or not u.items_read:
                u.items, u.items_read = list(self.panel_items), True
            getattr(self, 'reread_asked', set()).discard(u.id)
            if stars:
                u.star = stars
                u.star_assumed = False
            old_look = getattr(u, 'look_before_read', None)
            confirmed_same = old_look is not None and u.name == getattr(u, 'name_before_read', None)
            self.remember_look(u, pre, label=True)   # its appearance now that its name is confirmed
            u.labelled = True
            if confirmed_same:
                # the read said it IS the same unit: the earlier look stays a valid reference too
                # (run29: Kog'Maw on hex (0,1) was flagged 5 times and read as itself each time)
                u.looks_ok = (list(getattr(u, 'looks_ok', [])) + [old_look])[-2:]
            # read off its own panel: its looks are not questioned again at this star level (run 033507:
            # 61 of 91 re-reads named the same unit again - Kog'Maw 8 flags, Shen 4)
            u.look_trusted = u.star
            u.look_before_read = None
            # the game merges any 3 same-star copies, so 3 in our roster means a
            # read was wrong: look at the others again
            dup = [x for x in self.roster.copies(name, u.star) if x is not u]
            if len(dup) >= 2:
                for x in dup:
                    if self.tries.get(x.id, 0) < 2:     # re-check once more at most - no loops
                        x.name = None
            print(f'{scr.stage} identified {u}')
        self.log(ev='identify', stage=scr.stage, unit=repr(u), name=name, stars=stars)
        self.status_extra['identify'] = (f'read {u} from its info panel' if name
                                         else f'could not read the panel for {u}')
        self.live.event(scr.stage, 'identify', self.status_extra['identify'])
        return True

    def check_carousel_unit(self, scr):
        """What the carousel gave: the ONE unit that arrived since it ended (not bought) holding an item,
        named and with its item row read - loot drops arrive meanwhile but hold nothing. Several such
        units: the one holding what we went for. Reported once; nothing found by the end of the next
        planning phase is reported as unknown, never guessed."""
        cc = getattr(self, 'carousel_confirm', None)
        if not cc:
            return
        new = [u for u in self.roster.units if u.id not in cc['before'] and not u.ours and not u.missing]
        carried = cc.get('carried')
        if carried and carried[0]:
            # read off its panel while we carried it: the unit that arrives with that name is it - no second click
            u = next((u for u in new if u.name == carried[0]), None)
            if u is None and time.time() - cc['at'] < 75:
                return
            if u is not None and carried[1]:
                u.items, u.items_read = [carried[1]], True
            self.carousel_confirm = None
            self.log(ev='carousel_confirmed', stage=cc['stage'], unit=carried[0], items=[carried[1]] if carried[1] else [],
                     chaser_got=cc['got'], went_for=cc['went'], match=(carried[1] in (cc['got'], cc['went'])) if carried[1] else None,
                     how='carried panel', on_bench=u is not None)
            return
        # backup when the carried unit could not be read: look, don't wait: each new unit is clicked once and the item in its panel read - matched against what this
        # carousel offered, else named by its tooltip (the user, 2026-10-12; in run 20261010-235723 all 5 carousels
        # ended "could not be told": the unit came back, its panel item was never read, and Kayle with our Sparring
        # Gloves was sold as a plain unit)
        checked = cc.setdefault('checked', set())
        todo = sorted((u for u in new if u.name and not u.slotless and u.id not in checked and not u.items_read),
                      key=lambda u: (u.on_board, u.loc))
        if todo and not REPLAY and self.phase.planning and (scr.timer or 0) >= 4 and not cc.get('reading'):
            u = todo[0]
            checked.add(u.id)
            self.read_carousel_item(u, cc, scr)
        held = [u for u in new if u.name and u.items_read and len(u.items) == 1]
        pick = None
        if len(held) == 1:
            pick = held[0]
        elif held:
            pick = next((u for u in held if u.items[0] in (cc['got'], cc['went'])), None)
        if pick is not None:
            item = pick.items[0] if pick.items[0] != 'unknown' else 'an item not recognised'
            ok = item in (cc['got'], cc['went']) if (cc['got'] or cc['went']) else None
            what = f"carousel: we got {pick.name} holding {item}"
        elif time.time() - cc['at'] > 75:                    # a planning phase and a fight have passed
            pick, item, ok = None, None, None
            what = "carousel: which unit it gave could not be told (no new unit holding an item was seen)"
        else:
            return
        self.carousel_confirm = None
        print(f"{scr.stage} {what} (went for: {cc['went'] or 'not sure'})")
        self.log(ev='carousel_confirmed', stage=cc['stage'], unit=pick.name if pick else None, items=[item] if item else [],
                 chaser_got=cc['got'], went_for=cc['went'], match=ok)
        self.news_event('carousel', cc['stage'], what)

    def read_carousel_item(self, u, cc, scr):
        """Click u, read the item in its panel: by its icon among the carousel's offer when that is clear, else by the
        tooltip the game shows while the item is hovered (read off the loop). Sets u.items when named; a unit whose
        panel shows no item is no carousel unit (a loot drop)."""
        from tft_player.items import PANEL_ITEMS, identify_among, learn_icon, panel_item_crops
        try:
            got = self.cap.read()
            bars = B.green_bars(got[0]) if got and u.on_board else None
            point = self.unit_point(u.loc, bars)
            name, stars, fresh = self.read_unit(point)
            got = self.cap.read()
        except NotForeground:
            return
        if not got or not name:
            cc['checked'].discard(u.id)              # no panel: tried again on a later tick
            return
        crops = panel_item_crops(got[0])
        if crops is None:
            u.items, u.items_read = [], True        # holds nothing: not what the carousel gave
            return
        item, score, lead = identify_among(crops, cc.get('offer') or [])
        self.log(ev='carousel_item_icon', stage=scr.stage, unit=repr(u), item=item, score=round(score, 2),
                 lead=round(lead, 2), offer=cc.get('offer'))
        if item and score >= 0.75 and lead >= 0.2:
            u.items, u.items_read = [item], True
            return
        # not clear from the icon: hover the item and read the game's own tooltip
        px, py = PANEL_ITEMS[0]
        try:
            self.inp.hover((px - 200, py, 'R'))
            self.inp.hover((px, py, 'R'))
            time.sleep(0.4)                          # the tooltip fades in
            tip = self.cap.read()
            self.inp.hover((960, 540))
        except NotForeground:
            return
        if not tip:
            return
        frame, crop = tip[0], crops[len(crops) // 2]

        def read():
            try:
                from tft_player.screen import match_name
                h, w = frame.shape[:2]
                s = h / 1080
                # the tooltip opens beside the item, inside the screen: the right part of the screen around the panel
                x0, y0, x1, y1 = round(w - 820 * s), round(300 * s), w, round(1000 * s)
                texts = self.reader._detect(frame[y0:y1, x0:x1], 'panel_item_tip')
                found = [(ht, n) for _, _, ht, t in texts
                         if (n := match_name(re.sub(r'\s*\(.*$', '', t), self.reader.item_names(), 0.82))]
                name = max(found)[1] if found else None   # the title is the biggest item name in the tooltip
                self.log(ev='carousel_item_tooltip', stage=scr.stage, unit=repr(u), name=name,
                         texts=[t for *_, t in texts][:8])
                if name:
                    u.items, u.items_read = [name], True
                    learn_icon(crop, name)           # known by its picture from now on
            finally:
                cc['reading'] = False
        cc['reading'] = True
        threading.Thread(target=read, daemon=True).start()

    def read_encounter(self, frame, scr):
        """1-1: this game's encounter, as its banner says it - read off the loop, retried on later frames."""
        if scr.stage != '1-1' or self.encounter or self._enc_busy:
            return
        self._enc_busy = True

        def read(f=frame):                   # 2-5 s of OCR
            try:
                if not hasattr(self, '_enc_reader'):
                    self._enc_reader = ScreenReader()     # its own: OCR runs in this thread
                self.encounter = self._enc_reader.encounter(f)
                if self.encounter:
                    self.log(ev='encounter', text=self.encounter)
                    print(f'encounter: {self.encounter}')
            except Exception as e:
                self.log(ev='encounter_error', error=repr(e))
            finally:
                self._enc_busy = False
        threading.Thread(target=read, daemon=True).start()

    def paused_tick(self, frame, scr, paused):
        """F7 pause: the game is read and the board tracked up to here (stage, HP, merges, round changes), nothing
        is clicked and the model is not asked. On continue the plan from before the pause is dropped and the model
        gets the state as it is now: the round's first question again, told about the pause. -> True while paused."""
        was = getattr(self, '_paused_at', None)
        if paused:
            if was is None:
                self._paused_at = (time.time(), scr.stage)
                self.log(ev='paused', stage=scr.stage)
                self.live.event(scr.stage, 'paused', 'F7: paused - no input, no model questions')
            time.sleep(0.1)
            return True
        if was is not None:
            self._paused_at = None
            secs = round(time.time() - was[0])
            self.log(ev='resumed', stage=scr.stage, paused_at=was[1], seconds=secs)
            self.live.event(scr.stage, 'resumed', f'F7: continuing after {secs} s')
            self.round_plan, self.next_plan, self.plan_asked = None, None, None
            self.news_event('pause_over', scr.stage, f'the user paused the bot at {was[1]} for {secs} s (no input, no '
                                                 f'questions); the plan from before is dropped - this state is as '
                                                 f'the game is now')
            if not self.phase.planning:
                self.request_plan(scr, frame, 'plan')    # in a fight: ask now (planning asks on its first tick)
        return False

    def bench_located(self, scr):
        """A bench action (E sell, a drag from or to the bench, an item onto a bench unit) only where the bench is: at
        home. In planning we are home; in a fight we may be on the opponent's arena, where the camera shows the bench
        elsewhere and its units have no bars to find them by (run 20261014-063145: six sells and two item drops landed
        on empty tiles). There the step waits for the next planning phase. -> True when we are home."""
        ph = getattr(self, 'phase', None)
        if ph is None or ph.planning or REPLAY:
            return True
        # the fight's first seconds (units teleport, the camera may move) and its last one (they go home, a choice may
        # open): 2026-10-18 combat sells at +2.2 s (171815 4-5) and at timer 0 (171815 4-1, 161404 3-1) pressed E on
        # an empty bench; one 18 s later in the same fight sold
        if time.time() - getattr(self, 'round_started', 0.0) < 3.0 or (scr.timer is not None and scr.timer <= 1):
            self.fail_cause = 'the fight is starting or ending: the bench is used a moment later'
            self.deferred = True
            return False
        got = self.cap.read()
        sim = B.home_similarity(got[0]) if got else None
        if sim is None or sim >= B.HOME_MIN:
            if got is None or not self.bench_not_ours(got[0]):
                return True
            sim = None                           # an arena like ours, but our bench units are not on its bench
        if getattr(self, '_away_logged', None) != scr.stage:
            self._away_logged = scr.stage
            self.log(ev='away_fight', stage=scr.stage, similarity=None if sim is None else round(sim, 2))
        self.fail_cause = 'an away fight: the bench is not where it is at home - done in the next planning phase'
        self.deferred = True
        return False

    def bench_not_ours(self, frame):
        """In a fight: every bench tile our tracked bench units stand on (two or more, each with its empty picture)
        looks empty - an opponent's arena with the same scenery as ours, or our units still teleporting. A bench
        action there would press on nothing."""
        from tft_player import bench as BN
        slots = {u.loc[1] for u in self.roster.bench() if not u.missing}
        kinds = BN.classify(frame, [False] * 9)
        seen = [kinds[i]['kind'] for i in slots if not self.on_tactician(i)
                and not (kinds[i]['kind'] == 'unknown' and kinds[i]['occ'] is None)]     # (no empty picture yet)
        return len(seen) >= 2 and all(k == 'empty' for k in seen)

    def bench_reference(self, frame, scr):
        """Our home arena's scenery, once, from a stage-1 planning phase (stage 1 is played at home)."""
        if B._HOME_REF[0] is None and not REPLAY and scr.stage and scr.stage.startswith('1-') and self.phase.planning \
                and scr.shop_visible and time.time() - getattr(self, 'round_started', time.time()) > 3.0:
            B._HOME_REF[0] = B.arena_sig(frame)
            self.log(ev='home_reference', stage=scr.stage)

    def standings_shown(self, frame, scr):
        """The player list, not the game's Damage Dealt panel, on the right: the carousel's release order, the HP
        standings in every prompt and players alive are read from it. Checked every 2 s in planning and carousels;
        an open panel is closed with the people button under its sword (a few tries per round)."""
        now = time.time()
        if REPLAY or now - getattr(self, '_standings_at', 0) < 2.0 or not (scr.shop_visible or is_carousel(scr)):
            return
        self._standings_at = now
        if not self.reader.damage_panel(frame):
            return
        tries = self._standings_tries = (self._standings_tries if getattr(self, '_standings_stage', None) == scr.stage
                                         else 0) + 1
        self._standings_stage = scr.stage
        self.log(ev='damage_panel', stage=scr.stage, try_=tries)
        if tries <= 3:
            self.inp.click(ScreenReader.STANDINGS_BUTTON)

    def handle_carousel(self, frame, scr):
        """Chase the unit carrying the item we want most."""
        self.carousel_stage = scr.stage
        now = time.time()
        # two reads: the offer for the model's ranking, then the route's read just before our gate opens -
        # in between nothing is read (the chase waits; the CPU stays free)
        rk = getattr(self, '_rank', None)
        waiting = (self.chaser.phase(now) == 'wait' and rk is not None and rk.get('stage') == scr.stage
                   and rk.get('asked'))               # the first read went to the model
        # (no player bars, no 'carried' test: a carried unit leaves the ring, and the chase sees that - and the chase
        # ticked every 1.0-1.4 s in the games of 2026-10-10, walking to where its target had been)
        markers = [] if waiting else find_carousel_items(frame)
        held = self._held = []
        # players' bars (~60 ms) only while the target is unseen: they tell a hidden target from a taken one
        bars = self._bars = find_bars(frame) if self.chaser.lost_since is not None else None
        if scr.timer and 0 < scr.timer <= 36 and self.chaser.release_at is not None:
            self.chaser.ends_at = now + scr.timer                # the carousel ends at timer 0
        if not self.chaser.released:
            # our gate's countdown circle: seen, then gone = the gate is open, whatever the schedule said
            g = gate_ring(frame)
            # (seen = for 1.5 s on end: the carousel's first frames flash blue for ~0.5 s)
            gs = self._gate if getattr(self, '_gate', {}).get('stage') == scr.stage else {'stage': scr.stage, 'first': None,
                                                                                            'last': None, 'off': 0}
            self._gate = gs
            if g >= GATE_RING_ON:
                gs['first'], gs['last'], gs['off'] = gs['first'] or now, now, 0
            elif g <= GATE_RING_OFF and gs['first'] is not None:
                if gs['last'] - gs['first'] < 1.5:
                    gs['first'] = None                    # a flash, not the circle
                else:
                    gs['off'] += 1
                if gs['off'] == 2 and (self.chaser.release_at is None or self.chaser.release_at > now):
                    self.log(ev='carousel_gate_open', stage=scr.stage, timer=scr.timer, ring=g,
                             schedule_in=None if self.chaser.release_at is None else round(self.chaser.release_at - now, 1))
                    self.chaser.release_at = now
        if self.chaser.release_at is None and scr.timer and not self.reader.damage_panel(frame):
            # our gate opens on a fixed schedule by HP order: plan the route from that moment
            rt = release_timer(frame)
            # (the first carousel frames can still show the last round's timer: "2")
            if rt and rt[0] < scr.timer <= 36:
                self.chaser.release_at = now + (scr.timer - rt[0])
                self.log(ev='carousel_release', stage=scr.stage, timer=scr.timer, release_timer=rt[0],
                         row=rt[1], alive=rt[2])
        self.carousel_rank(markers, scr, now)
        tac = self.own.find(frame, now, hp=getattr(self, '_hp', None))   # our HP bar: how far we moved
        # everything above was seen in the frame captured at t_cap: the chase predicts from then, and its click lands
        # ~0.06 s after now
        t_cap = getattr(self, '_t_cap', None) or now
        click = self.chaser.next_click(markers, self.bench_items, self.roster, self.brain.plan,
                                       pos=tac, now=t_cap, held=held, bars=bars, lag=time.time() + 0.06 - t_cap)
        if click:
            if self.chaser.item and self.chaser.target_angle is not None and not getattr(self, '_announced', None) == scr.stage:
                print(f'{scr.stage} carousel: going for {self.chaser.item}')
                self.live.event(scr.stage, 'carousel', f'going for {self.chaser.item} '
                                f'(seen: {[m[2] for m in markers]})')
                self.log(ev='carousel', stage=scr.stage, want=self.chaser.item,
                         seen=[m[2] for m in markers])
                self._announced = scr.stage
                self.news_event('carousel', scr.stage,
                                f'the controller went for {self.chaser.item} (on offer: {[m[2] for m in markers]})')
            try:
                self.inp.click(click, 'right')
            except NotForeground:
                pass
        self.show_carousel(scr, markers, tac, click)
        self.record_carousel(frame, scr, markers, tac, click, now)
        self.read_carried(scr)

    def read_carried(self, scr):
        """Once we hold a unit: click the unit we carry and read its panel - the champion and the item it brings (the
        user, 2026-10-12: "just click on the dragged component and you could see what it is"). Once per carousel."""
        ch = self.chaser
        done = self.__dict__.setdefault('_carried_read', {})
        if not ch.got or scr.stage in done or REPLAY:
            return
        mine = sorted((math.dist(q, ch.pos), q) for q, it in ch.ring.off if it == ch.got and ch.pos is not None)
        if not mine or mine[0][0] > 120:
            return                                     # the carried unit's icon is not seen beside us yet
        q = mine[0][1]
        done[scr.stage] = None
        # the eye icon over the floating item icon opens the unit's panel (the user, 2026-10-12: "once grab it,
        # click on the eye icon on top to confirm it"): 40 px above the item icon (run 20261012-071645 2-4); the
        # unit's body under the icon (right, then left button) if the eye opens none
        tries = [((q[0], q[1] - 40), 'left'), ((q[0], q[1] + 40), 'right'), ((q[0], q[1] + 40), 'left')]
        from tft_player.items import identify_among, panel_item_crops
        try:
            got = self.cap.read()
            before = self.reader.panel_sig(got[0]) if got else None
            name = item = None
            for point, button in tries:
                self.inp.click(point, button)
                for _ in range(8):
                    time.sleep(0.08)
                    got = self.cap.read()
                    if not got:
                        continue
                    sig = self.reader.panel_sig(got[0])
                    if before is not None and sig.shape == before.shape and                             (np.abs(sig - before).max(axis=2) > 40).mean() < 0.05:
                        continue                       # no new panel yet
                    name, _ = self.reader.read_panel(got[0])
                    if name:
                        break
                if name:
                    crops = panel_item_crops(got[0])
                    offer = sorted(getattr(self, '_car_offer', {}).get(scr.stage, ()))
                    it, score, lead = identify_among(crops, offer) if crops else (None, -1.0, 0.0)
                    item = it if it and score >= 0.75 and lead >= 0.2 else None
                    self.log(ev='carousel_carried', stage=scr.stage, button=button, xy=point, unit=name,
                             item=item, icon=it,
                             score=round(score, 2), lead=round(lead, 2), chaser_got=ch.got)
                    self.save_panel_shot(f'{scr.stage}_carousel', f'{name}-{item or "unread"}')
                    break
            done[scr.stage] = (name, item)
            if not name:
                self.log(ev='carousel_carried', stage=scr.stage, unit=None, note='no panel opened on the carried unit')
        except NotForeground:
            done.pop(scr.stage, None)

    def carousel_rank(self, markers, scr, now):
        """The model ranks what this carousel offers (components early, completed items later),
        in the background: the chase starts on the targets it set before and switches to its
        ranking the moment the answer's first JSON object is written."""
        from tft_player.coach import _cli, ask_model
        from tft_player import carousel as CR
        st = getattr(self, '_rank', None)
        if st is None or st['stage'] != scr.stage:
            st = self._rank = {'stage': scr.stage, 'asked': False, 'ranked': None, 'applied': False, 'source': None,
                               'version': None, 'v1': False}
        # v1: the list the model prepared in the fight before - the WHOLE list, in its order, from the first
        # tick (the first view shows only part of the ring: run 221257 2-4 / 5-4 lost items cut from it)
        prep = getattr(self, 'carousel_prep', None)
        if not st['v1'] and prep and prep['stage'] == scr.stage:
            st['v1'] = True
            if not st['ranked']:
                st['ranked'], st['source'], st['version'] = list(prep['ranked']), 'prepared in the fight before', 'v1'
        if st['ranked'] and not st['applied']:
            st['applied'] = True
            late = bool(self.chaser.got or self.chaser.stepped or self.chaser.contact is not None)
            if st['version'] == 'v2' and late:
                # committed to a unit already: the v1 chase goes on (the user: switch only if in time)
                st['source'] = f"{st['source']} - too late, v1 kept"
                self.chaser.source = st['source']
            else:
                self.chaser.set_priority(st['ranked'], st['source'])
            print(f"{scr.stage} carousel: priority {st['version']} {st['ranked']} ({st['source']})")
            self.live.event(scr.stage, 'carousel', f"priority {st['version']}: {', '.join(st['ranked'])}")
            self.log(ev='carousel_ranking', stage=scr.stage, ranked=st['ranked'], source=st['source'],
                     version=st['version'], late=late, on_offer=[m[2] for m in markers])
        # v2: once the offer is seen, the model ranks EVERY item on it, seeing which champion carries
        # it - asked even when v1 covers the offer (run 152104 5-4: v1 had only Deathblade on offer;
        # once it was taken nothing ranked was left)
        if st['asked'] or not _cli()[0] or sum(1 for m in markers if m[2]) < 5:
            return
        st['asked'] = True
        offer = list(markers)
        targets = list(self.chaser.priority or self.brain.plan.item_targets or [])
        state = (f"HP {getattr(self, '_hp', None)}, {self.roster!r}, item bench: "
                 f"{[i for i in (self.bench_items or []) if i] or 'empty'}")
        prompt = CR.priority_prompt(offer, state, targets)
        image = self.carousel_jpeg(markers)

        def run():
            t0 = time.time()

            def first(obj):
                r = CR.parse_priority(obj, offer)
                if r and st['version'] != 'v2':
                    st['latency'], st['ranked'], st['version'] = time.time() - t0, r, 'v2'
                    st['source'], st['applied'] = f'ranked on the offer in {time.time() - t0:.1f}s', False
            try:
                ans = ask_model(prompt, image, on_first=first, timeout=40)
                if st['version'] != 'v2':
                    from tft_player.coach import json_objects
                    for obj in json_objects(ans or ''):
                        first(obj)
                self.model_call('carousel', prompt, ans, time.time() - t0, st.get('latency'))
            except Exception as e:
                self.log(ev='carousel_rank_error', error=repr(e))
        threading.Thread(target=run, daemon=True).start()

    def record_carousel(self, frame, scr, markers, tac, click, now):
        """Every carousel tick: the frame and what the chaser saw and decided, for checking the
        real game against tools/sim_carousel.py (pickup range, release, speed, rotation)."""
        # the items this carousel offered: what the unit we bring back can hold (check_carousel_unit)
        self.__dict__.setdefault('_car_offer', {}).setdefault(scr.stage, set()).update(m[2] for m in markers if m[2])
        try:
            ch = self.chaser
            d = self.log_dir / 'carousel' / scr.stage
            d.mkdir(parents=True, exist_ok=True)
            n = getattr(self, '_car_n', 0) + 1
            self._car_n = n
            name = f'{n:05d}_{now:.2f}.jpg'
            if not hasattr(self, '_car_q'):                  # saving takes ~50ms: off the carousel loop
                import queue
                self._car_q = queue.Queue(maxsize=30)

                def saver():
                    while True:
                        path, img = self._car_q.get()
                        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
                threading.Thread(target=saver, daemon=True).start()
            # one full-size frame a second (the raw recorder below keeps 8 a second): the chase now ticks ~3x faster
            if test_mode() and not self._car_q.full() and now - getattr(self, '_car_saved', 0) >= 1.0:
                self._car_saved = now
                self._car_q.put((str(d / name), frame))
            else:
                name = None
            self._car_live = time.time()
            # (not while the game video records: game.mp4 already holds the carousel, and this recorder takes frames
            # from the same capture stream as the chase - in video mode each read waits for a NEW frame, so the two
            # took turns. Run 20261010-235723: chase ticks 0.6-1.1 s apart, the recorder thread busy 90% meanwhile)
            if not getattr(self, '_car_rec_on', False) and hasattr(self.cap, 'cam') and not getattr(self, 'recorder', None) and test_mode():
                # the whole carousel at a steady 8 fps (the tick runs ~1.6/s): what happened between two ticks,
                # for replaying the chase (the user, 2026-10-09: record carousel rounds to fix them). Started again
                # after a pause in the ticks (run 010345 2-4: one 2.3 s gap ended it after 6 s, once per stage)
                self._car_rec_on = True
                raw_dir = d / 'raw'
                raw_dir.mkdir(exist_ok=True)

                def recorder():
                    while time.time() - getattr(self, '_car_live', 0) < 2.0:
                        t = time.time()
                        try:
                            f = self.cap.read_raw()     # (never cam.get_latest_frame: it takes frames from the loop)
                            if f is not None:
                                f = np.asarray(f)
                                h, w = f.shape[:2]
                                small = cv2.resize(f, (round(w * 540 / h), 540), interpolation=cv2.INTER_AREA)
                                cv2.imwrite(str(raw_dir / f'{t:.3f}.jpg'), cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                                            [cv2.IMWRITE_JPEG_QUALITY, 80])
                        except Exception:
                            pass
                        time.sleep(max(0.0, 0.125 - (time.time() - t)))
                    self._car_rec_on = False
                threading.Thread(target=recorder, daemon=True).start()
            r = lambda p: None if p is None else [round(float(p[0]), 1), round(float(p[1]), 1)]
            self.log(ev='carousel_tick', stage=scr.stage, frame=name, t=round(now, 3),
                     markers=[[round(x, 1), round(y, 1), it] for x, y, it in markers], held=getattr(self, '_held', None),
                     tac=r(tac), color=self.own.color, click=r(click), item=ch.item,
                     target=r(ch.view()['target']), stepped=ch.stepped, contact_in=ch.contact_in,
                     ring=[round(float(v), 1) for v in (ch.ring.cx, ch.ring.cy, ch.ring.R)],
                     omega=None if ch.omega is None else round(ch.omega, 3), speed=round(ch.speed),
                     free=len(ch.units), taken=list(ch.taken), taken_why=getattr(ch, 'taken_why', []), got=ch.got,
                     pos=r(ch.pos_s), released=ch.released, feet_off=r(ch.off), lost=ch.lost_since is not None,
                     path=[r(p) for p in ch.path])
        except Exception as e:
            self.log(ev='carousel_record_error', error=repr(e))

    def show_carousel(self, scr, markers, tac, click):
        """For people watching: the decision panel shows the model's priority list as the chase works
        through it (taken, target, grabbed - frozen once grabbed); the comp and execution panels are hidden
        (the comp panel sits in the carousel's icon search); the target layer rings the unit we go for."""
        if not self.overlay:
            return
        ch = self.chaser
        view = ch.view()
        st = getattr(self, '_rank', None) or {}
        version = st.get('version') if st.get('stage') == scr.stage else None
        listed = list(ch.priority or [])
        rank = {it: stt for it, stt in view['ranking']}
        items = [(it, rank.get(it, 'free')) for it in listed if it in rank]
        if ch.item and ch.item not in listed and not ch.got:
            items.append((ch.item, 'target'))             # a unit the model did not rank: shown as such
        frozen = getattr(self, '_car_frozen', None)
        if ch.got and (frozen is None or frozen[0] != scr.stage):
            self._car_frozen = (scr.stage, items)
        elif frozen and frozen[0] == scr.stage and ch.got:
            items = frozen[1]
        note = ('grabbed ' + ch.got) if ch.got else ('not in the model\'s list' if ch.item and ch.item not in listed else '')
        phase = ('picked up' if ch.got else 'waiting for our gate' if ch.delay > 0.5 else
                 'stepping in' if ch.stepped else 'walking to it' if ch.item else 'reading the carousel')
        from tft_player.coach import forced_comp
        fc = forced_comp()
        # built directly (no parsing of the model's answers): this runs every chase tick
        top = {'stage': scr.stage, 'title': fc.name if fc else None, 'alerts': self.alert_lines(),
               'thinking': self.thinking_now(), 'took': self.last_took(),
               'carousel': {'version': version, 'list': items, 'note': note or phase}}
        view['markers'] = [(m[0], m[1]) for m in markers]
        if tac is None:
            view['pos'] = None
        self.overlay.show(top_left=top, top_right=None, bottom_left=None, path=view)

    def carousel_jpeg(self, markers):
        """The carousel for the model, each unit's item written beside its icon, so it sees which
        champion carries which item."""
        import io
        from PIL import Image, ImageDraw
        raw = getattr(self.cap, 'raw', None)
        if raw is None:
            return self.screen_jpeg()
        im = Image.fromarray(raw).convert('RGB')
        s = im.height / 1080
        d = ImageDraw.Draw(im)
        for x, y, it in markers:                     # markers are 1080p reference points (x centred on 960)
            if not it:
                continue
            tx, ty = im.width / 2 + (x - 960 + 22) * s, (y - 12) * s
            box = d.textbbox((tx, ty), it)
            d.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(0, 0, 0))
            d.text((tx, ty), it, fill=(255, 230, 120))
        im.thumbnail((1568, 1568))
        buf = io.BytesIO()
        im.save(buf, 'JPEG', quality=85)
        return buf.getvalue()

    def screen_jpeg(self):
        """The live screen for the model: native frame, 1568px on the long side."""
        import io
        from PIL import Image
        raw = getattr(self.cap, 'raw', None)
        if raw is None:
            got = self.cap.read()
            raw = got[0] if got else None
        if raw is None:
            return None
        im = Image.fromarray(raw)
        im.thumbnail((1568, 1568))
        buf = io.BytesIO()
        im.save(buf, 'JPEG', quality=85)
        return buf.getvalue()

    def ground_point(self, orb):
        """A right-click ON a unit opens its info panel instead of walking. Click
        the ground beside the orb (still inside pickup range), as far from our
        units as possible."""
        import math
        units = [loc_point(u.loc) for u in self.roster.units]
        units += [(bx, by + 55) for bx, by in B.green_bars(self.last_frame)] if self.last_frame is not None else []
        if not units:
            return orb
        best, bd = orb, min(math.dist(orb, u) for u in units)
        if bd > 70:
            return orb
        for r in (28, 40):
            for k in range(8):
                a = k * math.pi / 4
                p = (orb[0] + r * math.cos(a), orb[1] + r * math.sin(a) * 0.8)
                d = min(math.dist(p, u) for u in units)
                if d > bd:
                    best, bd = p, d
        return best

    # -- pilot mode: the model's judgement calls (augments, Wisps) ----------------------------------
    def pilot_state(self, scr):
        """The game as the model needs it for a judgement call."""
        u = lambda x: f"{x.name or '?'} {x.star}-star" + (f" [{', '.join(x.items)}]" if x.items else '')
        h = scr if scr.gold is not None else (self.hud_seen or scr)
        seen = '' if h is scr else f' (read at {h.stage}: the picker hides the HUD)'
        plan = self.pilot.last_plan
        econ = (f"level-8 roll-down planned for {plan['spike'][0]}-{plan['spike'][1]}; projected gold left for "
                f"rolling then: {plan['roll_gold']}; this comp's players roll ~{plan['roll_need']} gold at "
                f"level 8 (MetaTFT average)" if plan else 'no projection yet')
        st = self.pilot.streak
        items = [i for i in (self.bench_items or []) if i]
        return (f"Stage {scr.stage or self.phase.stage}. Gold {h.gold if h.gold is not None else '?'}, "
                f"level {h.level or '?'}" + (f" (xp {h.xp[0]}/{h.xp[1]})" if h.xp else '') + seen +
                f". HP {scr.hp if scr.hp is not None else '?'}. "
                f"Streak: {f'{st} wins' if st > 0 else f'{-st} losses' if st < 0 else 'none'}.\n"
                f"Economy: {econ}.\n"
                f"Board: {', '.join(u(x) for x in self.roster.board()) or 'empty'}\n"
                f"Bench: {', '.join(u(x) for x in self.roster.bench()) or 'empty'}\n"
                f"Item bench: {', '.join(items) or 'empty'}\n"
                f"Comp units not owned yet: {', '.join(self.pilot.missing(self.roster)) or 'none'}\n"
                + (f"This game's encounter (1-1 banner): {self.encounter}\n" if self.encounter else '') +
                f"Augments taken: {', '.join(self.augments_taken) or 'none'}")

    def pilot_ask(self, kind, about, prompt, image, done, **info):
        """One judgement call to the model in the background. done(decision) as soon as the
        decision is written (its "why" follows); done(None) when the call fails. The question,
        then the answer, show top-left."""
        from tft_player.coach import ask_model, json_objects
        call = dict(info, kind=kind, about=about, stage=self.phase.stage, t0=time.time(), answer=None, why=None)
        self.pilot_call = call

        def decided(obj):
            if isinstance(obj, dict) and call['answer'] is None and set(obj) - {'why'}:
                call['answer'], call['t1'] = obj, time.time()
                done(obj)

        def run():
            try:
                text = ask_model(prompt, image, on_first=decided) or ''
            except Exception as e:
                text = f'FAILED: {e}'
            objs = [o for o in json_objects(text) if isinstance(o, dict)]
            call['why'] = next((o['why'] for o in objs if o.get('why')), None) or (text[:300] if not objs else '')
            if call['answer'] is None:
                dec = next((o for o in objs if set(o) - {'why'}), None)
                call['answer'], call['t1'] = dec or {}, time.time()
                done(dec)
            self.model_call(kind, prompt, text, time.time() - call['t0'], call['t1'] - call['t0'])
            print(f"[model {kind} {call['t1'] - call['t0']:.0f}s] {call['answer']} - {str(call['why'])[:160]}")
        threading.Thread(target=run, daemon=True).start()

    def pilot_handle_augment(self, scr):
        """The augment picker in pilot mode: the cards are read off the screen, and the model,
        given their text, MetaTFT's tiers (a guide) and the game, takes one or rerolls. Only when
        it has no answer in time does the comp's tier ranking decide."""
        from tft_player.coach import _cli
        from tft_player.meta import META
        stage = scr.stage or self.phase.stage
        if self.reroll_stage != stage:
            self.reroll_stage, self.rerolled = stage, set()
        if time.time() - self.augment_clicked < 1.5:
            time.sleep(0.1)                      # picker closing / a rerolled card still flipping
            return
        rerollable = [i for i in range(3) if i not in self.rerolled]
        hand = (stage, tuple(sorted(self.rerolled)))
        if self.augment_seen != hand:
            self.augment_seen, self.aug_answer = hand, None
            self.augment_default, why = self.pilot_augment(rerollable)
            names = list(self.augment_read)
            print(f'{stage} augments {names} - tier-list fallback {self.augment_default}')
            self.log(ev='augment_options', stage=stage, options=names, default=self.augment_default, why=why)
            if _cli()[0]:
                lines = []
                for i, n in enumerate(names):
                    a = META.augments.get(n or '') or {}
                    lines.append(f"{i}: {n or 'name not read - see the screenshot'}: {a.get('desc') or '?'} "
                                 f"[MetaTFT tier for this comp: {self.pilot.augment_tiers.get(n, 'not rated')}; "
                                 f"over all comps: {a.get('tier') or 'not rated'}]")
                prompt = (f"DECISION: augment, stage {stage} (~{scr.timer or '?'}s left). "
                          f"Cards (0 = left, 1 = middle, 2 = right):\n" + '\n'.join(lines) +
                          f"\nCards you can still reroll (each once): {rerollable or 'none'}.\n\n"
                          f"STATE\n{self.pilot_state(scr)}\n\n"
                          'Reply {"pick": <card>} or {"reroll": [<cards>]}, then {"why": "..."}.')

                def done(decision, hand=hand):
                    if self.augment_seen == hand:
                        self.aug_answer = decision or {}
                self.pilot_ask('augment', ' / '.join(n or '?' for n in names), prompt, self.screen_jpeg(), done,
                               names=names)
        ans = self.aug_answer
        c = self.pilot_call
        waited = time.time() - c['t0'] if c and c['kind'] == 'augment' else 99
        if ans is None and _cli()[0] and (scr.timer is None or scr.timer > 6) and waited < 45:
            time.sleep(0.15)                     # the model is deciding
            return
        ans = ans or {}
        reroll = ans.get('reroll')
        reroll = [reroll] if isinstance(reroll, int) else reroll if isinstance(reroll, list) else []
        reroll = [i for i in reroll if i in rerollable]
        if reroll and (scr.timer is None or scr.timer > 10):
            return self.reroll_augment(stage, self.augment_read, reroll, 'model')
        pick, src = ans.get('pick'), 'model'
        if not isinstance(pick, int) or not 0 <= pick <= 2:
            kind, i = self.augment_default
            pick = i if kind == 'pick' else \
                self.pilot.augment_decision(self.augment_read, [], self.pilot.gold_first(None, self.roster))[0][1]
            src = 'tier list (no model answer in time)' if _cli()[0] else 'tier list (no model)'
        taken = self.augment_read[pick] or f'card {pick}'
        if src != 'model' and c and c['kind'] == 'augment' and c['answer'] is None:
            c['late'] = taken                    # an answer arriving now is shown as too late
        print(f'{stage} t={scr.timer}: augment {pick} {taken} ({src})')
        self.status_extra['augment'] = f'took {taken} ({src})'
        self.live.event(stage, 'augment', self.status_extra['augment'])
        self.log(ev='augment', stage=stage, options=self.augment_read, pick=pick, name=taken, source=src)
        self.augments_taken.append(taken)
        self.execute(Action('augment', pick, src), scr)
        self.augment_clicked = time.time()
        time.sleep(0.5)

    AUGMENT_GONE_S = 2.0     # s the augment screen stays closed before a pick that never came is reported

    def augment_missed(self, scr):
        """The augment screen closed without our pick (it came after the timer: benchmark game 20261018-123359 3-2
        and 4-2 answered with 11-12 s left in 10.5-10.7 s) - the game took a card on its own and nothing said which.
        The model is told, and its augment list says so, instead of carrying on without it. Which card the game took is
        not read (the augments panel on the screenshot shows it)."""
        if not getattr(self, 'augment_open', None):
            return
        now = time.time()
        if getattr(self, 'augment_gone', None) is None:
            self.augment_gone = now
            return
        if now - self.augment_gone < self.AUGMENT_GONE_S:
            return
        o, self.augment_open, self.augment_gone = self.augment_open, None, None
        names = [n or '?' for n in o['names']]
        self.augments_taken.append(f"one of {' / '.join(names)} (the game's own pick)")
        self.log(ev='augment_missed', stage=o['stage'], options=names)
        self.news_event('augment_missed', o['stage'], f"your augment pick at {o['stage']} did not reach the game in "
                                                      f"time: the game took one of the cards on its own ({', '.join(names)}) "
                                                      f"- which one is not read; the augments panel on the screenshot shows it")

    def handle_augment(self, scr):
        """Local default first (meta tiers, ms): reroll weak cards at once, then ask
        the coach about the settled hand and take its pick if it arrives in time."""
        if self.pilot:
            return self.pilot_handle_augment(scr)
        names, p = scr.augments, self.brain.plan
        stage = scr.stage or self.phase.stage          # the HUD is hidden behind the picker
        if self.reroll_stage != stage:
            self.reroll_stage, self.rerolled = stage, set()
        if time.time() - self.augment_clicked < 1.5:
            time.sleep(0.1)             # picker closing / rerolled card still animating
            return
        rerollable = [i for i in range(len(names)) if i not in self.rerolled]
        hand = (stage, tuple(sorted(self.rerolled)))   # a hand = the stage and the cards rerolled so far
        if self.augment_seen != hand:
            if self.planner_on() and self.coach.busy:
                # still writing about the last hand (its "why"): ask about this one once it is free
                if (scr.timer or 0) > 5:
                    time.sleep(0.2)
                    return
            self.augment_seen = hand
            self.augment_open = {'stage': stage, 'names': list(names)}
            p.augment_pick = p.augment_reroll = None
            if self.pilot:
                self.augment_default, why = self.pilot_augment(rerollable)
            else:
                self.augment_default, why = local_augment(names, self.roster, stage, rerollable, p)
            print(f'{stage} augments {names} - local {self.augment_default} ({why})')
            self.log(ev='augment_options', stage=stage, options=names, default=self.augment_default, why=why)
            # without a model, local rules may reroll; with one, the model decides
            if not self.planner_on() and self.augment_default[0] == 'reroll' and (scr.timer is None or scr.timer > 8):
                return self.reroll_augment(stage, names, [self.augment_default[1]], 'local')
            if self.planner_on():
                time.sleep(0.6)          # rerolled cards finish their flip before the screenshot
            self.aug_wait, self.aug_reasks = self.coach.finished, 0
            self.coach.ask(scr, self.roster, augments=names, rerollable=rerollable, image=self.screen_jpeg(),
                           default=self.augment_default, note=self.augment_note)
            self.augment_note = None
        want = [i for i in (p.augment_reroll or []) if i in rerollable]
        bad = [i for i in (p.augment_reroll or []) if i not in rerollable]
        if bad:
            # each card rerolls once: a reroll asked for a card that used it is told to the model, never dropped silently
            self.augment_note = (f'Your last reply asked to reroll card {", ".join(map(str, bad))}, which has no reroll '
                                 f'left (each card rerolls once).')
            self.log(ev='augment_reroll_invalid', stage=stage, asked=p.augment_reroll, rerollable=rerollable)
            p.augment_reroll = want or None
        if want and (scr.timer is None or scr.timer > 12):
            return self.reroll_augment(stage, names, want, 'coach')
        pick = p.augment_pick
        if pick is None:
            # the model is still answering about this hand: wait for it
            if self.coach.busy and self.coach.pending_augment and (scr.timer or 0) > 5:
                time.sleep(0.25)
                return
            if self.planner_on() and not self.pilot:
                if not want and self.coach.finished > self.aug_wait and self.aug_reasks < 2 and \
                        (scr.timer is None or scr.timer > 8):
                    # its answer about this hand came and holds no pick or reroll that can be done: say why, ask again
                    note = self.augment_note or 'Your last reply held no "pick" or "reroll" that could be carried out.'
                    self.augment_note, p.augment_reroll = None, None
                    self.aug_wait, self.aug_reasks = self.coach.finished, self.aug_reasks + 1
                    self.log(ev='augment_reask', stage=stage, why=note)
                    self.coach.ask(scr, self.roster, augments=names, rerollable=rerollable, image=self.screen_jpeg(),
                                   default=self.augment_default, note=note)
                    return
                # no fallback (the user): the model decides; if it never answers, the game's own
                # timer picks a card, as it would for a player who didn't click
                self.status_extra['augment'] = 'waiting for the model'
                time.sleep(0.25)
                return
            kind, i = self.augment_default
            if kind == 'pick':
                pick = i
            elif self.pilot:
                pick = self.pilot.augment_decision(self.augment_read, [], self.pilot.gold_first(None, self.roster))[0][1]
            else:
                pick = local_augment(names, self.roster, stage, [], p)[0][1]
            src = 'local'
        else:
            src = 'coach'
            # a reroll left on another card is free information: use it before taking the pick, and ask again with
            # the new card (the user, 2026-10-11: "the last reroll is always used if it is not picked")
            others = [i for i in rerollable if i != pick]
            if others and (scr.timer is None or scr.timer > 12):
                p.augment_pick = None
                self.log(ev='augment_reroll_rest', stage=stage, pick=pick, reroll=others)
                return self.reroll_augment(stage, names, others, 'unused rerolls before the pick')
        taken = (getattr(p, 'augment_name', None) if src == 'coach' else None) or \
            (self.augment_read[pick] if self.pilot and self.augment_read[pick] else names[pick])
        p.augment_name = None
        print(f'{stage} t={scr.timer}: augment {pick} {taken} ({src})')
        self.status_extra['augment'] = f'picked {taken} ({names[pick]}, {src})'
        self.live.event(stage, 'augment', self.status_extra['augment'])
        self.log(ev='augment', stage=stage, options=names, pick=pick, name=taken, source=src)
        self.unit_change(taken, stage)
        self.expect_gifts(taken, stage)
        self.augments_taken.append(taken)
        self.augment_open = None
        self.execute(Action('augment', pick, src), scr)
        self.augment_clicked = time.time()
        p.augment_pick = None
        self.aug_wait = self.coach.finished          # this answer is carried out: not one to ask again about
        if self.overlay and getattr(self, '_cards_shown', False):
            # the verdict boxes go with the pick, not at the next overlay refresh: what follows can block the loop
            # (run 20261016-052413 4-2: they stayed over Portable Forge's artifact row, "TAKING" over the first card)
            self.overlay.show(path=None)
            self._cards_shown = False
        time.sleep(0.5)

    def pilot_augment(self, rerollable):
        """Pilot mode: read the three names (after the flip) and rank them by the comp's
        MetaTFT augment tiers. A card that was not rerolled keeps the name read before, so
        one unreadable card never turns the pick into a blind one."""
        stage = self.reroll_stage
        if getattr(self, '_aug_known_stage', None) != stage:
            self._aug_known_stage, self._aug_known = stage, {}
        for i in self.rerolled:
            if i in self._aug_known and i not in getattr(self, '_aug_seen_rerolled', set()):
                del self._aug_known[i]          # a new card sits there now
        self._aug_seen_rerolled = set(self.rerolled)
        names, raw = [None] * 3, []
        for _ in range(4):
            time.sleep(0.4)              # a rerolled card flips for a moment
            got = self.cap.read()
            if got is None:
                continue
            names, raw = self.reader.read_augments(got[0])
            if all(names):
                break
        names = [n or self._aug_known.get(i) for i, n in enumerate(names)]
        self._aug_known.update({i: n for i, n in enumerate(names) if n})
        self.augment_read = names
        gold_first = self.pilot.gold_first(None, self.roster)
        decision, vals = self.pilot.augment_decision(names, rerollable, gold_first)
        return decision, (f'pilot tiers {list(zip(names, vals))}{" (behind on gold, no streak: gold augments first)" if gold_first else ""}'
                          f' (read {raw})')

    def reroll_augment(self, stage, names, cards, src):
        """Reroll one or several cards at once (the model can keep the best and
        reroll the rest in one go); it is asked again about the new hand."""
        print(f'{stage}: reroll augment cards {cards} ({[names[i] for i in cards]}) - {src}')
        self.log(ev='augment_reroll', stage=stage, options=names, cards=cards, source=src)
        self.brain.plan.augment_reroll = None
        for i in cards:
            self.rerolled.add(i)             # each card rerolls once, whether or not the click took
            try:
                self.inp.click(AUGMENT_REROLL[i])
            except NotForeground:
                return
            time.sleep(0.25)
        self.augment_clicked = time.time()

    def run_rolldown(self, spec, scr):
        """A whole roll-down as one script (the user: the target is clear, no check per action):
            read the shop -> buy every wanted card at once (logged, no waiting) -> press D ->
            wait only until the 5 card strips have redrawn (pixels, ~0.1 ms per look) -> read the 5
            names (+ gold, one small OCR batch) -> repeat; stop at the stop condition or the floor.
        Gold is counted as we spend and re-synced from each shop read. ONE check at the end: the
        next planning read reconciles the bench with what we logged. Run26: 1.9 s per roll.
        spec['times']: reroll that many times at most (roll n times). spec['wisps']: 'auto' (fast roll-down):
        a Wisp is bought on the spot and its shop read again before the next roll; 'ask' (roll n times): a
        Wisp in a new shop stops the script and the model is asked (spec['stopped'] = 'wisp',
        spec['done'] = the rolls made); a Wisp click the game refuses means its Wisp limit is reached."""
        from tft_player import planner
        want, floor = set(spec.get('buy') or []), int(spec.get('floor') or 0)
        # copy targets: a listed champion is bought until board + bench hold its star level's copies
        # (given star, else its auto_buy star, else 2) - never every copy that shows up (run 152104)
        targets = {n: int((spec.get('targets') or {}).get(n) or self.auto_buy.get(n) or 2) for n in want}
        refused = self.__dict__.setdefault('wisp_refused', {})
        stopped, skipped = None, []
        keep, stop = set(spec.get('keep') or ()), {'stop_when': spec.get('stop_when') or []}
        t_start = time.time()
        rolls, bought, sold, wisps, missed = 0, [], [], 0, []
        shops = []                               # every shop seen: names as read, '-' = unread (a missed target shows here)
        gold = scr.gold if scr.gold is not None else getattr(self.round_plan, 'last_gold', None)

        # 0. sell first, when the model's plan says so (sell_others: fast_rr's default, a plain roll's not):
        # bench units outside the buy list, auto_buy, keep and the plan's board make room and gold before the
        # first roll (run 152104 4-1: a full bench skipped Sentinels while rolling on). Bench units only, so
        # this works in combat too.
        free_items = sum(1 for i in self.bench_items if not i) if self.bench_items is not None else 0
        for u in (list(self.roster.bench()) if spec.get('sell_others') else []):
            if (u.id in keep or not u.name or u.unverified or u.missing or u.name in want
                    or self.auto_buy.get(u.name) or len(u.items) > free_items):
                continue
            self.inp.key('e', at=loc_point(u.loc))
            self.roster.sold(u)
            sold.append(u.name)
            free_items -= len(u.items)
            if gold is not None:
                gold += u.cost * (3 ** (u.star - 1)) - (1 if u.star > 1 else 0)
            time.sleep(0.05)
        if sold:
            time.sleep(0.15)                     # let the gold and the shop settle before reading them

        def shop_now():
            got = self.cap.read()
            if got is None:
                return None, [None] * 5, [''] * 5, None
            g, names, raw = self.reader.read_shop(got[0])
            return g, names, raw, got[0]

        def strip_changed(a_, b_):
            return (np.abs(a_ - b_) > 40).mean() > 0.1

        def card_gone(i, sig0):
            return self.wait_card(i, sig0, 0.45, strip_changed)

        def redrawn(sig0, need, timeout=1.0):
            """Wait until at least `need` card strips differ from sig0 and the shop stands still."""
            prev, t0 = None, time.time()
            while time.time() - t0 < timeout:
                time.sleep(0.03)
                got = self.cap.read()
                if got is None or sig0 is None:
                    continue
                sig = self.reader.shop_sig(got[0])
                changed = sum(1 for a_, b_ in zip(sig, sig0) if (np.abs(a_ - b_) > 40).mean() > 0.1)
                if changed >= need and prev is not None and all(
                        (np.abs(a_ - b_) > 40).mean() < 0.02 for a_, b_ in zip(sig, prev)):
                    return True
                prev = sig
            return False

        def show(now=''):
            # the roll-down runs as one script (no ticks): it pushes its own progress to the overlay
            if not self.overlay:
                return
            total = spec.get('times')
            steps = ([('done', 'SELL', 'Sold ' + ', '.join(sold), 'room first')] if sold else []) + [
                ('done' if now.startswith('done') else 'run', 'ROLL', f"Roll {rolls}{'/' + str(total) if total else ''} · gold {gold}", '')] +                 ([('done', 'BUY', 'Bought ' + ', '.join(bought[-6:]), '')] if bought else []) +                 ([('guard', 'BUY', ', '.join(skipped) + ' skipped', 'bench full - rolling stopped')] if skipped else [])
            # its rows join the round's list (one panel, no separate roll-down page - the user, 2026-10-10)
            self.roll_live = steps
            self.overlay.show(bottom_left=self.exec_panel(scr, True))

        g, names, raw, frame = shop_now()
        gold = g if g is not None else gold
        if gold is None:
            self.fail_cause = 'roll-down: gold could not be read'
            return False
        show('reading the shop')
        free = scr.free_rolls or 0               # free rerolls go first: they cost no gold, so no gold floor stops them
        while True:                              # stops only below: the model's conditions, or the game's
            shops.append([n or ('?' if r else '-') for n, r in zip(names, raw)])
            # 1. buy the cards still short of their copy target in this shop - clicks only, logged as bought
            for i, name in enumerate(names):
                info = CATALOG.unit(name) if name else None
                if info is None:
                    continue
                listed = name in want and self.auto_need(name, targets[name]) > 0
                # auto-buy's champions too, while short of their star target (no floor: no gold reserve)
                auto = not listed and self.auto_need(name) > 0 and info.cost <= gold
                if not (listed or auto):
                    continue
                if info.cost > gold:
                    missed.append(f'{name} (costs {info.cost}, gold {gold})')
                    continue
                # the floor stops the rolling, not the buying of a card rolled into (run 185805 6-3: Sentinel,
                # wanted for its 2-star, was passed at 32 gold with a floor of 30 and no word to the model)
                if not self.roster.free_bench() and len(self.roster.copies(name, 1)) < 2:
                    skipped.append(name)         # bench full and no merge: the card cannot be bought
                    continue
                # confirmed before it is recorded and before any roll (the user: slower is fine): a click the
                # game took late was rolled away by the next D and still logged as bought (run 213127: 11
                # 'bench slots tracked != on screen after a buy'). A card still there gets one more click -
                # an empty slot ignores a click, so a late first click cannot buy twice
                sig_now = self.reader.shop_sig(frame) if frame is not None else None
                # the bench before the click: the slot that lights up is where this unit landed (a Wisp's or an
                # augment's unit that came meanwhile is already lit here, so it is not taken for ours)
                lights0 = self.bench_lights(frame) if callable(getattr(self, 'bench_lights', None)) else None
                if lights0 is not None:
                    # a unit that came since the last action (the Wisp just bought, an augment's delivery) is
                    # recorded now - unnamed, read off its panel - before our buy, as a normal buy does
                    self.adopt_unexplained(lights0, scr)
                took = False
                for attempt in range(2):
                    self.inp.click(B.SHOP_CLICK[i])
                    took = card_gone(i, sig_now)
                    if took:
                        break
                if not took:
                    missed.append(name)
                    continue
                if lights0 is not None:
                    slot = self.landing_slot(name, lights0, scr)
                    self.note_landing(self.roster.bought(name, slot=slot, merge=not self._no_merge))
                    self.copies_doubted(name, scr)
                else:
                    self.roster.bought(name)
                bought.append(name)
                gold -= info.cost
                got = self.cap.read()
                frame = got[0] if got else frame         # the next card's 'before' picture
            if skipped:
                stopped = 'bench_full'           # rolling on would only skip more wanted cards
                self.news_event('rolldown', scr.stage, f'roll-down stopped after {rolls} rolls: the bench is full '
                                f'and {", ".join(skipped)} could not be bought (no copy to merge with). Sell or '
                                f'field bench units, then roll again if you want to.')
                break
            # Wisps (a Wisp: text that is not a champion)
            mode = spec.get('wisps')
            wslots = [] if refused.get(scr.stage) else [i for i, (n, r) in enumerate(zip(names, raw)) if r and not n]
            if wslots and (mode == 'auto' or (mode == 'first' and rolls > 0 and not wisps)):
                i = wslots[0]
                cost = self.reader.shop_cost(frame, i) if frame is not None else None
                if cost is None or gold - cost >= min(floor, gold):
                    sig0 = self.reader.shop_sig(frame) if frame is not None else None
                    self.inp.click(B.SHOP_CLICK[i])
                    if redrawn(sig0, 1):
                        wisps += 1
                        self.log(ev='rolldown_wisp', stage=scr.stage, slot=i, card=raw[i], cost=cost)   # which Wisp, for what it did
                        # the model hears which Wisp the roll bought and what it does (a Wisp can add champions:
                        # run 185805 4-2, a Malphite) - a unit it adds is recorded unnamed and read before the next buy
                        try:
                            from tft_player.screen import match_name
                            fx = self.wisp_effects()
                            wname = match_name(raw[i], list(fx), 0.75) if raw[i] else None
                        except Exception:
                            fx, wname = {}, None
                        self.news_event('rolldown_wisp', scr.stage, f"the roll-down bought the Wisp '{wname or raw[i] or '?'}'"
                                        f"{f' ({cost}g)' if cost is not None else ''}"
                                        f"{': ' + fx[wname] if wname in fx else ''}")
                        self.wisps_bought[scr.stage] = self.wisps_bought.get(scr.stage, 0) + 1
                        gold -= cost or 0
                        g, names, raw, frame = shop_now()      # its effect may have rerolled the shop
                        if g is not None:
                            gold = g
                        continue                 # buy from this shop before rolling again
                    refused[scr.stage] = True    # the game took no more Wisps this round
            elif wslots and mode == 'ask' and rolls > 0:
                stopped = 'wisp'                 # a Wisp in a new shop: the model decides, then the rest
                self.ask_wisp_now(scr, frame, wslots[0])
                break
            # 2. stop? Only the model's own conditions (the user, 2026-10-09: "until 2-star X and 2-star Y, or until
            #    20 gold" is the only rule): its stop_when, its gold floor, its number of rolls
            if stop['stop_when'] and planner.RoundPlan.fast_rr_met(stop, self.roster):
                stopped = 'stop_when_met'
                break
            if not free and gold - 2 < floor:
                stopped = 'gold_floor'
                break
            if spec.get('times') is not None and rolls >= int(spec['times']):
                break
            # 3. roll, and wait just until the new cards are drawn
            sig0 = self.reader.shop_sig(frame) if frame is not None else None
            raw0 = list(raw)
            self.inp.key('d')
            rolls += 1
            cost = 0 if free else 2
            free = max(0, free - 1)
            gold -= cost
            show('rolling')
            gold_before = gold + cost
            changed = redrawn(sig0, 3)
            g, names, raw, frame = shop_now()
            if not changed:
                # a free reroll (an encounter, Invested++) spends no gold, so only the cards tell: the redraw can take
                # over 1 s on a busy machine (run 20261010-235723 2-3 / 3-2: rolls that changed the shop were taken
                # for refused ones). Wait a little more, then compare the card texts themselves
                changed = redrawn(sig0, 3, timeout=1.2)
                g, names, raw, frame = shop_now()
                changed = changed or sum(1 for a_, b_ in zip(raw, raw0) if a_ and b_ and a_ != b_) >= 2
            if g is not None:
                gold = g                         # re-sync (part of the same small OCR batch)
            # the game did not take the roll (no new cards, no gold spent): it cannot roll now - stop and say so
            if not changed and g is not None and g >= gold_before:
                rolls -= 1
                stopped = 'roll_refused'
                self.news_event('rolldown', scr.stage, f'roll-down stopped after {rolls} rolls: the game did not '
                                f'take a roll (the shop did not change, gold stayed {g}).')
                break
        took = time.time() - t_start
        spec['done'], spec['stopped'] = rolls, stopped
        show(f'done: {rolls} rolls' + (f', stopped ({stopped.replace("_", " ")})' if stopped else ''))
        self.roll_live = None                             # the finished roll shows as its round_acts row
        spec['summary'] = (f'{rolls} rolls, bought {", ".join(bought[-6:]) or "nothing"}'
                           + (f', stopped: {stopped.replace("_", " ")}' if stopped else ''))
        if skipped:
            self.alert('SKIP', f'{", ".join(skipped)} not bought: bench full', 'Execution')
        if missed:                                       # a wanted card was in a shop and not bought: the model hears it
            self.news_event('rolldown', scr.stage, f'roll-down: wanted cards seen but not bought: {", ".join(missed)}')
        self.log(ev='rolldown', stage=scr.stage, rolls=rolls, bought=bought, sold=sold, wisps=wisps, shops=shops,
                 want=sorted(want), targets=targets, skipped=skipped, missed=missed, floor=floor, gold_end=gold,
                 seconds=round(took, 1), per_roll=round(took / max(1, rolls), 2), stopped=stopped)
        print(f'{scr.stage}: roll-down - {rolls} rolls in {took:.1f}s, bought {bought or "nothing"}'
              + (f', sold {sold}' if sold else '') + (f', stopped: {stopped}' if stopped else '')
              + (f' (skipped {skipped})' if skipped else ''))
        self.buy_check = time.time() + 1.5      # the ONE check: bench slots vs the log, next planning read
        return True

    def wait_for(self, cond, timeout=2.0):
        """Poll fresh frames until cond(screen) holds; the HUD lags an input by a few frames and one
        read takes ~0.8s, so the default leaves time for two reads."""
        end = time.time() + timeout
        while time.time() < end:
            time.sleep(0.06)
            _, after = self.read()
            if after is not None and cond(after):
                return True
        return False

    # -- bench lights: one per slot, on while a unit's health bar is there ------------------
    def bench_lights(self, frame=None):
        """Per bench slot: True = something is there (the tile differs from its empty image),
        False = empty, None = no evidence (covered by a banner or panel, or a container)."""
        from tft_player import bench as BN
        if frame is None:
            got = self.cap.read()
            if got is None:
                return None
            frame = got[0]
        lights = [True if o['kind'] in ('unit', 'unknown') else False if o['kind'] == 'empty' else None
                  for o in BN.classify(frame, [False] * 9)]
        # our tactician standing on a slot fills its tile too: no evidence there (run 20261005-022645
        # 3-3: it stood on slot 0 and was read as an arriving unit 8 times)
        return [None if self.on_tactician(i) else x for i, x in enumerate(lights)]

    def on_tactician(self, i):
        tac_box = self.watcher.tac[1]
        return tac_box is not None and tac_box[3] > 600 and tac_box[0] - 60 <= B.BENCH[i][0] <= tac_box[2] + 60

    def learn_bench_baseline(self, frame, scr):
        """Each bench slot's 'off' light for this game's arena (no pictures are stored: the arena changes
        the tiles). Stage 1-2 before the first shop: every bench slot is empty. Later - the player started
        mid-game, or a slot was busy at 1-2 - a slot is taken once it is visibly free. A slot counts when
        it has no health bar, nothing tracked, no tactician on it, and looks the same as the read before
        (nothing passing through; after 1-2: for 2 s), and holds no container and no tooltip or panel."""
        from tft_player import bench as BN
        got = self.__dict__.setdefault('_base_got', set())
        # not 1-1: the game opens on the lobby-wide intro view (all eight tacticians on one map), and
        # its ground taken as 'empty' lit 5 bench slots as units for the whole of both games of 2026-10-10
        if len(got) == 9 or not scr.stage or scr.stage == '1-1':
            return
        later = scr.stage != '1-2'
        if later:
            if not self.phase.planning or not scr.shop_visible or time.time() - getattr(self, '_base_at', 0) < 1.0:
                return                   # after 1-2: in planning with our own HUD up (not a carousel), once a second
            self._base_at = time.time()
        tac_box = self.watcher.tac[1]
        bars = B.bar_places(frame, exclude=tac_box)[0]
        fr = BN._bench_band(frame)
        crops = [BN.slot_crop(fr, i).astype(np.int16) for i in range(9)]
        prev, self._base_prev = getattr(self, '_base_prev', None), crops
        if prev is None:
            return
        ok = [i for i in range(9) if i not in got and not bars[i] and not self.on_tactician(i)
              and self.roster.at(('bench', i)) is None and np.abs(crops[i] - prev[i]).mean() < 4]
        # no anvil and no tooltip or panel over it: those are things on the slot, not its bare tile (also at 1-2:
        # run 20261016-052413 had the opening encounter's anvil on slot 1, learned as 'empty' - the anvil then
        # read as a free slot)
        ok = [i for i in ok if BN.container_match(BN.slot_crop(fr, i))[1] < BN.CONTAINER
              and float((BN.slot_crop(fr, i)[40:].max(axis=2) < 40).mean()) < BN.OBSCURED]
        if later:
            # after 1-2 a slot's picture is taken only once it has looked the same for 2 s: a glow on a free slot
            # (hover, sell flash) outlasts one read (run 20261016-052413 2-6: a glow on slot 1 became its 'empty'
            # picture, and the bare slot then read as taken)
            cand, now, steady = self.__dict__.setdefault('_base_cand', {}), time.time(), []
            for i in range(9):
                if i not in ok:
                    cand.pop(i, None)
                elif i in cand and np.abs(crops[i] - cand[i][0]).mean() < 4:
                    if now - cand[i][1] >= 2.0:
                        steady.append(i)
                        cand.pop(i, None)
                else:
                    cand[i] = (crops[i], now)
            ok = steady
        if ok:
            BN.learn_empty(frame, ok)
            got.update(ok)
            self.log(ev='bench_baseline', stage=scr.stage, slots=ok, have=sorted(got))

    LIT_FOR = 1.5                             # s a slot must stay lit before an unexplained unit counts

    def watch_lights(self, frame, scr):
        """Planning, every read: a slot lit without a break for LIT_FOR s that no tracked unit and no buy of the
        last seconds explains is a unit that came without our action - recorded and read. Two lit reads in a row
        were not enough (run 20261014-063145: the 4-4 carousel's champion walking over slots 4 and 5 to the bench,
        and a unit fading out after its sell at 4-6, became '?' units no read could name - and Aphelios, bought into
        slot 4, then shared it with one)."""
        now = time.time()
        lights = self.bench_lights(frame)
        since = self.__dict__.setdefault('_lit_since', {})
        for i, t in getattr(self.roster, 'vacated', {}).items():
            if now - t < 3.0 and since.get(i, now) < t:
                since[i] = t                     # lit since our sell/move emptied it, not before: the fade-out
        for i, lit in enumerate(lights or []):
            if lit:
                since.setdefault(i, now)
            else:
                since.pop(i, None)
        recent = [b for b in getattr(self.roster, 'recent_buys', []) if now - b[2] < 3]
        pending = [b for b in self.roster.pending_buys if now - b[1] < 15]
        if lights is None or recent or pending:
            return
        self.adopt_unexplained([x if not x else now - since.get(i, now) >= Player.LIT_FOR for i, x in enumerate(lights)],
                               scr)

    def expect_gifts(self, augment, stage):
        """The champions an augment's (or Wisp's) own text says it gives ('... Gain a Camille, an Elise, and a
        Caitlyn.'): recorded as expected, told to the model at once, and the units that then arrive without a buy take
        these names instead of being unknown (the user, 2026-10-16). A gift at a later round or level stays expected
        until it lands. From the card's text in the game data - no augment is named in the code."""
        from tft_player.coach import augment_text
        text = re.sub(r'<[^>]+>', ' ', augment_text(augment, full=True) or self.effect_text(augment) or '')
        names = sorted(CATALOG.units, key=len, reverse=True)
        gifts = []
        timing = re.compile(r'\b(level|stage|rounds?|after|every|when|each|combats?)\b', re.I)
        for sentence in re.split(r'(?<=[.!])\s+', text):
            # 'Gain ...' is what we get; '<unit> gains ...' is an effect on a unit, not a gift
            m = re.search(r'\b[Gg]ain\b', sentence)
            if not m:
                continue
            before, sentence = sentence[:m.start()], sentence[m.end():]
            found = []
            for n in names:
                for f in re.finditer(r'\b' + re.escape(n) + r'\b', sentence):
                    if not any(a <= f.start() < b for a, b, _ in found):
                        found.append((f.start(), f.end(), n))
            found.sort()
            for k, (_, end, n) in enumerate(found):
                # when: said before 'Gain' ('At level 7, gain ...') for all, or right after the name ('a Vi now
                # and a Nidalee after 3 rounds')
                after = sentence[end:found[k + 1][0] if k + 1 < len(found) else len(sentence)]
                after = re.split(r'\b(and|with)\b', after)[0] if not timing.search(before) else after
                when = before if timing.search(before) else after if timing.search(after) else ''
                when = re.sub(r'\s+', ' ', when).strip(' ,.;')
                gifts.append({'name': n, 'from': augment, 'later': bool(when), 'when': when, 'stage': stage})
        if gifts:
            self.gifts = getattr(self, 'gifts', []) + gifts
            self.log(ev='gifts_expected', stage=stage, augment=augment, names=[g['name'] for g in gifts])
            self.news_event('gifts_expected', stage, f"{augment} gives "
                            + ', '.join(g['name'] + (f" ({g['when']})" if g['when'] else '') for g in gifts)
                            + ': place them by name in "board", they are fielded the moment they land')

    def gift_line(self, stage=None):
        """The state's line for champions an augment gives that have not landed yet. A gift due now that has not
        landed two rounds later is dropped (it went elsewhere or the text was misread); later gifts stay."""
        from tft_player.coach import stage_key
        now = stage_key(stage) if stage else None
        if now:
            def due_passed(g):
                k = stage_key(g['stage'])
                return not g['later'] and k and (now[0] * 10 + now[1]) - (k[0] * 10 + k[1]) >= 2 + 3 * (now[0] > k[0])
            for g in [g for g in getattr(self, 'gifts', []) if due_passed(g)]:
                self.gifts.remove(g)
                self.log(ev='gift_dropped', stage=stage, name=g['name'], augment=g['from'])
        gifts = getattr(self, 'gifts', [])
        if not gifts:
            return None
        return ('Champions your augments give, not on the bench yet: '
                + ', '.join(f"{g['name']} ({g['from']}{', ' + g['when'] if g['when'] else ''})" for g in gifts)
                + '. Place them by name in "board"; they are fielded the moment they land.')

    def adopt_unexplained(self, lights, scr):
        """A lit slot no tracked unit explains got its unit without a buy of ours (loot orb,
        carousel, augment): record it unnamed - identify_one reads it off its info panel.
        Only while no tracked bench unit's slot reads empty (else it may be a tracked unit we
        mis-placed: reconcile settles that)."""
        from tft_player.state import Owned
        tracked = {u.loc[1] for u in self.roster.bench()}
        covered = [i for i, x in enumerate(lights) if x is None and not self.on_tactician(i)]
        if False not in lights:
            return                  # not one slot looks like our empty tile: another arena is on screen
                                    # (away fights: run 20261005-022645 4-2 read 6 'arrivals' there)
        if covered or any(lights[i] is False for i in tracked):
            return                  # part of the bench covered (banner, panel): its neighbours misread too
        # a slot our own sell or move just emptied stays lit while the unit fades out: no arrival there for 3 s (run
        # 20261016-011010 4-2: a "unit we did not buy" in bench slot 3, 0.4 s after Yunara was sold from it - eight
        # panel reads of nothing followed). Nothing is adopted in the last 2 s of planning either (units teleport).
        now = time.time()
        vacated = getattr(self.roster, 'vacated', {})
        timer = getattr(scr, 'timer', None)
        late = getattr(getattr(self, 'phase', None), 'planning', False) and timer is not None and timer <= 2
        if late and not REPLAY:
            return
        for i, lit in enumerate(lights):
            if lit and i not in tracked and i not in self.roster.blocked and not self.on_tactician(i) \
                    and now - vacated.get(i, 0) > 3.0:
                u = Owned(None, 1, ('bench', i))
                gifts = getattr(self, 'gifts', [])
                # an expected gift: its name from the augment's text (several at once: in slot order, read again in
                # idle time - a read that differs corrects it and the gift stays expected). A gift due later is not
                # given to a unit that may have come from a loot orb.
                g = next((g for g in gifts if not g['later']), None) or                     (gifts[0] if gifts and time.time() - getattr(self, '_loot_at', 0) > 30 else None)
                if g:
                    gifts.remove(g)
                    u.gift = g
                    u.name = g['name']
                    u.unverified = f"named from {g['from']}'s text: to be read"
                    self.reread = getattr(self, 'reread', set()) | {u.id}
                    self.tries[u.id] = 0
                    self.log(ev='gift_arrived', stage=scr.stage, slot=i, name=u.name, augment=g['from'])
                    print(f"{scr.stage}: bench slot {i}: {u.name} from {g['from']}")
                self.roster.units.append(u)
                src = 'loot' if time.time() - getattr(self, '_loot_at', 0) < 30 else 'no action of ours'
                self.log(ev='unit_arrived', stage=scr.stage, slot=i, source=src)
                print(f'{scr.stage}: bench slot {i} lit without a buy ({src}): reading it')

    def save_board_gallery(self, frame, scr):
        """At the start of each fight: a crop of every unit on screen (board and bench: whatever has a health
        bar) with the labels the harness holds - champion, star level (tracked and read off the badge), items,
        place, flags - in runs/<ts>/units/<stage>/. Units the screen shows that we do not track, and tracked
        units no bar shows, are listed too: that is where the harness is wrong (tools/units_report.py makes the
        contact sheet). The disk writes run off the game loop. Only with --record."""
        if REPLAY or not test_mode():
            return
        try:
            from tft_player import stars
            raw = getattr(self.cap, 'raw', None)
            src = raw if raw is not None else frame
            h, w = src.shape[:2]
            s = h / 1080
            d = self.log_dir / 'units' / scr.stage
            places = B.bar_places(frame, exclude=self.watcher.tac[1])[2]
            last = self.__dict__.setdefault('_gal_last', {})        # unit key -> (labels, file) last saved
            rows, jobs = [], []
            for (bx, by), loc in sorted(places, key=lambda p: p[1]):
                u = self.roster.at(loc)
                badge = stars.read_star(frame, (bx, by))
                tracked = None if u is None else {
                    'id': u.id, 'name': u.name, 'star': u.star, 'items': list(u.items),
                    'unverified': u.unverified or None, 'auto_placed': getattr(u, 'auto_placed', False)}
                # a new crop only when what the harness holds for the unit changed (champion, star level,
                # badge, items, flags); otherwise the row points at the crop already saved
                key = ('unit', u.id) if u is not None else ('screen', loc)
                sig = json.dumps([tracked and {k: v for k, v in tracked.items() if k != 'id'}, badge], default=str)
                prev = last.get(key)
                if prev and prev[0] == sig:
                    fn, new = prev[1], False
                else:
                    fn, new = f"{scr.stage}/{loc[0]}-{'-'.join(map(str, loc[1:]))}.jpg", True
                    X, Y = round(w / 2 + (bx - 960) * s), round(by * s)
                    crop = src[max(0, Y - round(22 * s)):Y + round(160 * s),
                               max(0, X - round(75 * s)):X + round(75 * s)].copy()
                    jobs.append((self.log_dir / 'units' / fn, crop))
                    last[key] = (sig, fn)
                rows.append({'file': fn, 'new_crop': new, 'where': list(loc), 'badge_star': badge, 'tracked': tracked})
            seen = {tuple(r['where']) for r in rows}
            for u in self.roster.units:
                if u.loc not in seen:
                    rows.append({'file': None, 'where': list(u.loc), 'badge_star': None,
                                 'tracked': {'id': u.id, 'name': u.name, 'star': u.star, 'items': list(u.items),
                                             'missing': u.missing or None, 'unverified': u.unverified or None}})
            meta = {'stage': scr.stage, 'level': scr.level, 'gold': scr.gold, 'hp': getattr(self, '_hp', None),
                    'units': rows}

            def write():
                d.mkdir(parents=True, exist_ok=True)
                for path, crop in jobs:
                    cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
                (d / 'labels.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
            threading.Thread(target=write, daemon=True).start()
        except Exception as e:
            self.log(ev='gallery_error', error=repr(e))

    def star_check(self, frame, scr, places):
        """Planning, every ~0.5 s: the star badge on each unit's health bar against the tracked star level. A
        different badge read twice in a row corrects the roster - a merge the rules did not see (three copies
        given at once, a copy bought in the fight that merged when it ended) or one they assumed that did not
        happen (run 152203 3-6: a 2-star Vi that was 1-star for a round). The model is told."""
        if REPLAY or time.time() - getattr(self, '_star_at', 0) < 0.5:
            return
        self._star_at = time.time()
        from tft_player import stars
        pend = self.__dict__.setdefault('_star_pend', {})
        read = stars.read_stars(frame, places)
        visible = set(read)
        for loc, star in read.items():
            u = self.roster.at(loc)
            if u is None or star is None or u.star == star or (star == 3 and u.star != 2):
                # (a 3 is taken only for a unit we track as 2-star: the gold badge is new to us)
                if u is not None:
                    pend.pop(u.id, None)
                continue
            prev = pend.get(u.id)
            if prev is None or prev[0] != star:
                pend[u.id] = (star, time.time())
                continue
            if time.time() - prev[1] < 0.4:
                continue
            pend.pop(u.id, None)
            if star > u.star:
                # a rise is a merge: it needs two more copies at the old star level (tracked, missing, or
                # units not named yet). Without them it is a misread badge (run 033507: a lone Vi read as
                # 2-star 3 times, a Kog'Maw with one copy twice - each "corrected" back a round later)
                # ... and the copies a merge used are gone from the screen: one still standing there was not used
                # (run 20261015-014311 5-1: two 1-star Aphelios side by side on the bench, one badge read as 2-star,
                # the copies counted were the other Aphelios and an unnamed board unit - both still there; the
                # roll-down then passed an Aphelios as "2-star reached" and the model sold the real copy)
                copies = [x for x in self.roster.units if x is not u and (x.name is None or
                                                                         (x.name == u.name and x.star == u.star))
                          and (x.missing or x.loc not in visible)]
                if len(copies) < 2:
                    self.log(ev='star_rise_ignored', stage=scr.stage, unit=repr(u), read=star, copies=len(copies))
                    continue
            elif not getattr(u, 'star_assumed', False):
                # a drop is only an assumed merge that did not happen: a star level we SAW (badge or panel)
                # does not fall on a badge read alone (item glows, effects) - its panel is read once instead
                if u.id not in getattr(self, 'reread', set()):
                    self.reread = getattr(self, 'reread', set()) | {u.id}
                    self.tries[u.id] = 0
                    self.log(ev='star_drop_check', stage=scr.stage, unit=repr(u), read=star)
                continue
            old, gone = self.roster.set_star(u, star, visible)
            self.log(ev='star_corrected', stage=scr.stage, unit=repr(u), was=old, now=star,
                     used=[repr(x) for x in gone])
            what = (f'{u.name or "a unit"} (unit {u.id}) is {star}-star on the screen (we had {old}-star)'
                    + (f'; its copies {", ".join(str(x.id) for x in gone)} are gone (merged)' if gone else
                       '; a merge we assumed did not happen, the copies it used are still there' if star < old else ''))
            print(f'{scr.stage}: {what}')
            self.news_event('star_corrected', scr.stage, what)
            if star < old:
                self.reread = getattr(self, 'reread', set()) | {u.id}
            else:
                u.star_badge_at = time.time()
        # a unit that arrived already upgraded (three copies given at once) has no name yet when its badge is
        # read: the copies it used turn up as missing units once it is named - they are merged away
        for u in list(self.roster.units):
            if u.name and u.star >= 2 and time.time() - getattr(u, 'star_badge_at', 0) < 120:
                gone = [m for m in self.roster.units if m is not u and m.name == u.name and m.star == u.star - 1
                        and m.missing][:2]
                for m in gone:
                    u.items = (u.items + m.items)[:3]
                    u.items_read = False
                    self.roster.alias[m.id] = u.id
                    self.roster.units.remove(m)
                if gone:
                    self.log(ev='merged_copies_removed', stage=scr.stage, unit=repr(u), copies=[repr(m) for m in gone])

    def sprykin_fielded(self):
        """Distinct Sprykin champions on the board (the trait counts each champion once)."""
        return len({u.name for u in self.roster.fielded() if u.info and 'Sprykin' in u.info.traits})

    def save_panel_shot(self, tag, text):
        """Data for what a panel says when a unit came without our action (the BFF, a carousel or augment unit):
        the full frame with the panel open, in runs/<ts>/panels/<tag>.jpg, written off the loop."""
        got = self.cap.read() if shots_on() else None
        if got is None or REPLAY:
            return
        frame = getattr(self.cap, 'raw', None)
        frame = frame if frame is not None else got[0]
        path = self.log_dir / 'panels' / f'{tag}_{re.sub(r"[^A-Za-z0-9]+", "-", text or "none")[:30]}.jpg'

        def save(f=frame.copy()):
            try:
                from PIL import Image
                path.parent.mkdir(exist_ok=True)
                Image.fromarray(f).save(path, quality=90)
            except Exception as e:
                self.log(ev='panel_shot_error', error=repr(e))
        threading.Thread(target=save, daemon=True).start()

    def read_new_unit(self, frame, scr):
        """A health bar on a board hex where nothing is recorded (the 1-1 carousel champion, a unit the game
        gives us): click it and read its panel at once (the user, 2026-10-09 - run 010416 knew of the 1-1 unit
        only at 1-3). Planning only (units stand still); a place seen on two ticks; each place once a round. A
        click that opens no champion panel (our tactician, an orb) records nothing. -> True when it clicked."""
        if REPLAY or frame is None or not self.phase.planning or time.time() - self.round_started < 1.0:
            return False
        if scr.timer is not None and scr.timer < 1:
            return False
        bars = B.green_bars(frame)
        _, _, places = B.bar_places(frame, bars=bars)
        tracked = {u.loc for u in self.roster.units if not u.missing}
        new = {loc for _, loc in places if loc[0] == 'board' and loc not in tracked}
        prev, self._untracked_prev = getattr(self, '_untracked_prev', set()), new
        tried = self.__dict__.setdefault('_untracked_tried', set())
        todo = sorted(loc for loc in new & prev if (self.phase.stage, loc) not in tried)
        if not todo:
            return False
        loc = todo[0]
        tried.add((self.phase.stage, loc))
        try:
            name, stars, fresh = self.read_unit(self.unit_point(loc, bars))
        except NotForeground:
            return True
        raw = getattr(self.reader, 'last_panel_text', None)
        self.log(ev='new_unit_read', stage=scr.stage, loc=list(loc), name=name, stars=stars, fresh=fresh, panel_text=raw)
        self.save_panel_shot(f'{scr.stage}_new_{loc[1]}{loc[2]}', raw)
        if not name and loc[0] == 'board' and self.roster.at(loc) is None and self.sprykin_fielded() >= 3 \
                and getattr(self, 'bff_assigned', False):
            # the BFF has had its rider: a unit there that opens no panel is the rider with the BFF under it - a
            # champion (a team slot), named when a read or the model names it (run 20261012-072956 3-1..4-6: the rider
            # Teemo at (1, 4) was recorded as an unnamed unit)
            from tft_player.state import Owned
            u = next((m for m in self.roster.missing() if m.info and 'Sprykin' in m.info.traits), None)
            if u is None:
                u = Owned(None, 1, loc, ours=True)
                self.roster.units.append(u)
            u.loc, u.missing, u.unverified, u.riding = loc, None, None, True
            self.news_event('bff_rider', scr.stage, f'a unit at hex {loc[1:]} opens no panel: the BFF rider '
                                                    f'({u.name or "not named yet"}, unit {u.id})')
            return True
        if not name and loc[0] == 'board' and self.roster.at(loc) is None and self.sprykin_fielded() >= 3:
            # before its first rider, a unit on the board that opens no panel while Sprykin is active is the BFF
            # (run 20261010-235723 4-5..4-7: three clicks on it opened nothing). Recorded without a team slot
            from tft_player.state import BFF, Owned
            u = next((m for m in self.roster.units if m.name == BFF), None)
            if u is None:
                u = Owned(BFF, 1, loc)
                self.roster.units.append(u)
            u.loc, u.missing, u.unverified = loc, None, None
            self.news_event('bff', scr.stage, f'the BFF (Sprykin) stands at hex {loc[1:]} (unit {u.id})')
            return True
        if not (fresh and name) or self.roster.at(loc) is not None:
            return True
        from tft_player.state import Owned
        # a missing record of that champion is this unit (a read names it); so is a fielded record of the same
        # champion and star level whose own hex shows no unit (a stale place: run 20261010-235723 4-3 recorded the
        # one Azir*** twice, and the board looked full); else it is a new one
        # so is a bench record of that champion whose slot is empty now: the game fielded it into an empty board slot
        # at combat start, unseen (release run 20261016-141957 3-3: Sejuani, fielded at 3-2, became a second record
        # while the first stayed "on the bench" - the model's sell order for her lost its unit, Vi met a full board)
        occupied = {l for _, l in places}
        same = lambda m: m.name == name and (stars is None or m.star == stars)
        lights = self.bench_lights(frame) if any(same(m) for m in self.roster.bench()) else None
        from_bench = next((m for m in self.roster.bench() if same(m) and lights and lights[m.loc[1]] is False), None)
        u = next((m for m in self.roster.missing() if same(m)), None) or from_bench \
            or next((m for m in self.roster.board() if same(m) and m.loc not in occupied
                     and not getattr(m, 'riding', None)), None)
        # a unit we did not put there: from the bench (or a new record) - the game's own placement, not the model's
        by_game = u is None or u.loc[0] == 'bench'
        was = u.loc if u is not None else None
        if u is None:
            u = Owned(name, stars or 1, loc, ours=True)
            self.roster.units.append(u)
        u.loc, u.missing, u.unverified = loc, None, None
        if by_game:
            u.auto_placed = True
        u.items = [i for i in self.panel_items if i != 'unknown']
        print(f'{scr.stage}: unit at {loc[1:]} where none was recorded: read as {name}{"*" * (stars or 1)}'
              + (f' (unit {u.id}, from bench slot {was[1]})' if was and was[0] == 'bench' else ''))
        self.news_event('unit_found', scr.stage, f'{u.name}{"*" * u.star} (unit {u.id}) stands at hex {loc[1:]}: '
                        + (f'the game fielded it from bench slot {was[1]} (an empty board slot at combat start)'
                           if was and was[0] == 'bench' else 'read off its panel (it was not in the records)'))
        return True

    def check_autofill(self, frame, scr):
        """The first seconds of a fight: a bench slot where a unit is tracked goes dark with no operation of
        ours (a sell, a move and a merge update the roster at once) - the game fielded that unit into an empty
        board slot at combat start (run 152203: a second Varus fielded this way, the records never knew). The
        game's order (the user, 2026-10-09; run 010416 1-3 and 1-4 agree): bench units left to right go to the
        free hexes from the bottom left, to the right - (0,0), (0,1) ... (0,6), then the next row. So their
        hexes are recorded at once, no read needed."""
        t0 = getattr(self, 'combat_t0', None)
        if t0 is None or self._autofill_stage == scr.stage or REPLAY:
            return
        age = time.time() - t0
        if age > 5.0:
            self._autofill_stage = scr.stage
            return
        if age < 0.6:
            return                              # the fill happens as combat starts; let the picture settle
        lights = self.bench_lights(frame)
        if lights is None or False not in lights:
            return                              # unreadable, or another arena is on screen
        prev, self._autofill_prev = self._autofill_prev, lights
        if prev is None:
            return
        gone = [u for u in self.roster.bench() if not u.missing and lights[u.loc[1]] is False and prev[u.loc[1]] is False]
        if not gone:
            return
        self._autofill_stage = scr.stage
        gone.sort(key=lambda u: u.loc[1])
        free = [('board', r, c) for r in range(4) for c in range(7) if self.roster.at(('board', r, c)) is None]
        # the game fields only up to the team size, and each fielded unit shows a health bar on its hex: dark bench
        # slots beyond that are not fielded units (run 20261010-010345 2-2/2-3: 4 recorded fielded at a full board)
        cap = self.team_cap(scr) or 0
        room = max(0, cap - self.roster.slots_used())
        try:
            barred = {l for _, l in B.bar_places(frame, exclude=self.watcher.tac[1])[2] if l[0] == 'board'}
        except Exception:
            barred = None
        pairs = [(u, loc) for u, loc in zip(gone, free) if barred is None or loc in barred][:room]
        if not pairs:
            self.log(ev='auto_fill_refused', stage=scr.stage, dark=[u.loc[1] for u in gone], room=room)
            return
        moves = []
        for u, loc in pairs:
            moves.append(f"{u.name or '?'} (unit {u.id}) bench slot {u.loc[1]} -> hex {loc[1:]}")
            u.loc, u.auto_placed = loc, True
            if u.name is None:
                self.reread = getattr(self, 'reread', set()) | {u.id}      # a '?' is still to be named
        self.log(ev='auto_fill', stage=scr.stage, units=moves)
        print(f'{scr.stage}: the game fielded {"; ".join(moves)} (its order: bottom left, to the right)')
        self.news_event('game_moved_unit', scr.stage, f'the game fielded units from the bench at combat start (empty '
                        f'board slots): {"; ".join(moves)}.')

    def landing_slot(self, name, lights0, scr):
        """Where a bought unit landed: the slot that lit up after the click. None when a merge is
        expected (two copies owned: no new light) or nothing lit in time (then predicted)."""
        self._landing_note, self._no_merge = None, False
        if lights0 is None or REPLAY:
            return None
        merge = len(self.roster.copies(name, 1)) >= 2
        # a merge is expected with two copies held: no slot lights then. One that lights all the same means the game
        # did NOT merge - a "copy" is another champion (run 20261015-161625 3-3: an "Ornn" that was a Veigar made a
        # merge we assumed, the new Ornn was then taken for a unit we did not buy, and the stars went wrong for rounds)
        end = time.time() + (0.7 if merge else 1.5)
        lit = None                               # (slot, when): a new light while a merge is expected
        while time.time() < end:
            lights = self.bench_lights()
            new = [i for i in range(9) if lights and lights[i] and lights0[i] is False]
            if merge and lit is not None:
                if lights and lights[lit[0]] is False:
                    # the bought copy showed in a slot for a moment and flew into the merge (benchmark game
                    # 20261018-134910: all 9 'did not happen' verdicts were merges - each left ghost copies that were
                    # dropped a round later, and Veigar's 3-star read was ignored for want of copies)
                    self.log(ev='merge_after_flash', stage=scr.stage, unit=name, slot=lit[0],
                             secs=round(time.time() - lit[1], 2))
                    return None
                if time.time() - lit[1] >= getattr(self, 'MERGE_SETTLE', MERGE_SETTLE):
                    self._no_merge = True
                    self.log(ev='merge_did_not_happen', stage=scr.stage, unit=name, slot=lit[0])
                    return lit[0]
                time.sleep(0.05)
                continue
            if new and merge:
                lit = (new[0], time.time())     # still lit after MERGE_SETTLE: no merge
                end = max(end, time.time() + getattr(self, 'MERGE_SETTLE', MERGE_SETTLE) + 0.2)
                continue
            if new:
                if len(new) > 1:
                    # another unit arrived with ours (a Wisp's unit, an augment's delivery): which is ours is
                    # read, not assumed (run 185805 4-2: Malphite took the slot the bought Aphelios was recorded in)
                    self._landing_note = f'landed while another unit arrived (bench slots {new} lit up together)'
                    self.log(ev='buy_landing_unclear', stage=scr.stage, unit=name, slots=new)
                return new[0]
            time.sleep(0.05)
        if merge and lit is not None:
            self._no_merge = True                # lit until the end of the wait: no merge
            self.log(ev='merge_did_not_happen', stage=scr.stage, unit=name, slot=lit[0])
            return lit[0]
        if merge:
            return None                          # nothing lit: the merge happened
        self._landing_note = 'its landing slot was not seen (slot predicted)'
        self.log(ev='buy_landing_unseen', stage=scr.stage, unit=name)
        return None

    def copies_doubted(self, name, scr):
        """After a buy the game did not merge although we held two copies: those records are read off their panels
        again - one of them is another champion."""
        if not getattr(self, '_no_merge', False):
            return
        self._no_merge = False
        doubted = [u for u in self.roster.units if u.name == name and u.star == 1 and not getattr(u, 'sure', False)]
        for u in doubted:
            u.unverified = f'the game did not merge a {name} bought next to it: read again'
            self.reread = getattr(self, 'reread', set()) | {u.id}
            self.tries[u.id] = 0
        self.news_event('unit_corrected', scr.stage, f'a {name} bought next to two recorded copies did not merge: '
                        f'those records ({", ".join(str(u.id) for u in doubted)}) are read again')

    def note_landing(self, unit):
        """A bought unit whose landing was unclear: unverified, and read off its panel in idle time."""
        note = getattr(self, '_landing_note', None)
        if note and unit is not None and unit in self.roster.units:
            unit.unverified = note
            self.reread = getattr(self, 'reread', set()) | {unit.id}

    # -- execution + verification ----------------------------------------------------
    def execute(self, a: Action, scr) -> bool:
        inp = self.inp
        if a.kind in ('sell', 'move', 'item', 'strip'):
            # a unit under our tactician cannot be clicked, dragged or sold: walk it off first
            unit = a.arg if a.kind == 'sell' else a.arg[0] if a.kind == 'move' else a.arg[2] if a.kind in ('item', 'strip') and len(a.arg) > 2 else None
            locs = [getattr(unit, 'loc', None)] + ([a.arg[1]] if a.kind == 'move' else [])
            for loc in locs:
                if isinstance(loc, tuple) and loc and loc[0] in ('bench', 'board'):
                    self.tactician_off(loc_point(loc)[:2])
        if a.kind == 'buy':
            name = scr.shop[a.arg]            # None for a wisp: nothing lands on our bench by name
            gold0 = scr.gold
            if DATA_MODE:
                got0 = self.cap.read()
                self._bench_before = self.bench_pixels(got0[0]) if got0 else None
                from tft_player import bench as BN
                BN._memo.clear()
                self._bench_before_kinds = [o['kind'] for o in BN.classify(got0[0], B.occupancy(got0[0])[0])] if got0 else None
            lights0 = self.bench_lights() if name else None
            if lights0 is not None:
                self.adopt_unexplained(lights0, scr)
            inp.click(B.SHOP_CLICK[a.arg])
            raw0 = scr.shop_raw[a.arg]
            # a Wisp can cost 0 and leaves the champion behind it in the slot: bought
            # when that slot's text changes
            ok = self.wait_for(lambda s: s.gold is not None and s.gold < gold0 or
                               (s.shop[a.arg] is None if name else s.shop_raw[a.arg] != raw0), timeout=2.5)   # one screen read takes ~0.8s
            if not ok:
                self.fail_cause = 'neither gold nor the shop slot changed after the click (not enough gold, bench full, or the shop moved on)'
            if ok and name:
                self.roster.last_merge = None
                slot = self.landing_slot(name, lights0, scr)
                new = self.roster.bought(name, slot=slot, merge=not getattr(self, '_no_merge', False))
                self.copies_doubted(name, scr)
                self.note_landing(new)
                if self.roster.last_merge and not REPLAY:
                    self.confirm_merge(self.roster.last_merge, scr)
                elif DATA_MODE and new in self.roster.units:
                    self.photograph_buy(new)
            elif name:
                self.roster.buy_unconfirmed(name)   # the unit that shows up on the bench is this one
            return ok
        if a.kind in ('sell', 'move', 'item', 'container'):
            on_bench = (a.kind == 'sell' and a.arg.loc[0] == 'bench') or (a.kind == 'container') or \
                (a.kind == 'move' and 'bench' in (a.arg[0].loc[0], a.arg[1][0])) or \
                (a.kind == 'item' and getattr(a.arg[2], 'loc', ('',))[0] == 'bench')
            if on_bench and not self.bench_located(scr):
                return False
        if a.kind == 'sell':
            # a sell can't be undone: a unit whose identity is in doubt is read first (pilot: every unit)
            if (self.pilot or a.arg.unverified) and a.arg.name and not REPLAY:
                got = self.cap.read()
                bars = B.green_bars(got[0]) if got else None
                pt = self.unit_point(a.arg.loc, bars)
                others = [self.unit_point(u.loc, bars) for u in self.roster.units
                          if u.name and u.name != a.arg.name and u is not a.arg]
                name, stars, _ = self.read_fresh(pt, others)
                fresh = name is not None
                if name != a.arg.name or (stars and stars != a.arg.star):
                    # not the unit we meant to sell: the roster is wrong there - sell nothing, re-read all
                    self.fail_cause = f'the unit there reads {name or "nothing"} {stars or "?"}-star, not {a.arg}'
                    self.log(ev='sell_refused', stage=scr.stage, unit=repr(a.arg), read=name, stars=stars, fresh=fresh)
                    self.need_survey = f'sell check: {a.arg.loc} holds {name}, not {a.arg.name}'
                    if name and fresh:
                        a.arg.name, a.arg.star = name, stars or a.arg.star
                    return False
                a.arg.unverified = None              # read off its own panel: it is the unit meant
            if getattr(getattr(self, 'phase', None), 'planning', False) and scr.timer is not None and scr.timer <= 2:
                # the round flips: units teleport and an E lands on nothing (run 20261015-161625 2-6, 5-2: "nothing
                # was sold" at 1 s; benchmark games 20261018-102854/-110145: bench sells at 2 s) - it waits for the
                # next moment the bench holds still (a bench unit sells in combat too)
                self.fail_cause = 'queued: the round flips in the last seconds'
                self.deferred = True
                return False
            got = None if REPLAY else self.cap.read()
            if a.arg.loc[0] == 'bench' and got is not None:
                why = self.bench_sell_blocked(got[0], a.arg)
                if why == 'wait':
                    self.fail_cause = 'no bench health bar is drawn yet: tried again in a moment'
                    self.deferred = True
                    return False
                if why == 'stale':
                    a.arg.unverified = f'not seen at bench slot {a.arg.loc[1]} where it is tracked: to be read'
                    self.reread = getattr(self, 'reread', set()) | {a.arg.id}
                    self.fail_cause = (f'bench slot {a.arg.loc[1]} is empty on screen, where {a.arg} is tracked - '
                                       f'nothing was sold; it is looked for')
                    return False
            gold0 = (self.reader.read_shop(got[0])[0] if got else None) or scr.gold   # the gold right before E
            if not REPLAY:
                # the game sells the unit it has under the cursor: give its hover time to move here first (run
                # 20261013-220557 4-5: E on Amumu at bench slot 1 sold the fielded Murkwolf 2-star)
                spot = self.unit_point(a.arg.loc, B.green_bars(got[0]) if got else None)
                inp.move(spot)
                # the game picks the hovered unit a frame or two after the cursor arrives; at 0.08 s under load E
                # found nothing under it (flex game 20261018-112700 2-2: Rek'Sai hovered in the frame after, not sold)
                time.sleep(SELL_HOVER)
            else:
                spot = loc_point(a.arg.loc)
            inp.key('e', at=spot)
            # checked, like every other action: run26 counted 3 sells that the game never did ("Your
            # bench is full!"), so the roster had room the game didn't and every buy failed. The fast
            # evidence is the unit's place going empty (bars, ~0.15 s); the gold text lags and costs OCR,
            # so it is only the fallback
            took = REPLAY or self.sell_seen(gold0)
            if took and took is not True and a.arg.name and a.arg.info:
                # the gold it brought tells which unit went: another amount means another unit was sold
                from tft_player.planner import sell_value
                want = sell_value(a.arg)
                if took - gold0 != want:
                    self.log(ev='sell_value_mismatch', stage=scr.stage, unit=repr(a.arg), expected=want,
                             got=took - gold0)
                    self.news_event('sell_mismatch', scr.stage,
                                    f'selling {a.arg} brought {took - gold0} gold, not its {want}: maybe another unit '
                                    f'was sold - the board and bench are read again')
                    self.need_survey = f'sell of {a.arg} brought {took - gold0}g, not {want}g'
            if not took:
                self.fail_cause = f'nothing was sold: gold stayed {gold0} and {a.arg.loc} still holds a unit'
                if self.pilot:
                    self.need_survey = f'a sell of {a.arg} did not take'
                else:
                    self.reread = getattr(self, 'reread', set()) | {a.arg.id}   # read that one unit, once
                # its place may be stale: the retry reads its panel first (benchmark game 20261018-102854 3-5: Caitlyn
                # sold twice at (1,1) while she stood at (0,2))
                a.arg.unverified = a.arg.unverified or 'a sell at this place did not take'
                return False
            self.roster.sold(a.arg)
            return True
        if a.kind == 'container':
            if a.arg in self.auto_opened and not self.bench_objs.get(a.arg):
                return True                      # already opened by the harness: nothing left to do
            ok = self.open_container(a.arg, scr)
            if not ok:
                self.fail_cause = f'bench slot {a.arg} holds no recognised container'
            return ok
        if a.kind == 'move':
            unit, dst = a.arg
            src = unit.loc
            if self.phase.planning and scr.timer is not None and scr.timer <= 1 and 'board' in (src[0], dst[0]):
                # the board locks as combat starts: a drag in the last second does not land (the user: 1 s, not
                # earlier - every second of planning counts). It is kept for the next round.
                self.fail_cause = 'queued for the next planning phase (the board locks at combat start)'
                self.deferred = True
                return False
            if not REPLAY:
                got = self.cap.read()
                if got is not None and self.unit_bar(got[0], src) is None and src[0] == 'board':
                    # run29 3-2: "Rammus at (0,0)" was dragged twice from an empty hex
                    unit.unverified = f'not seen at hex ({src[1]},{src[2]}) where it is tracked: to be read'
                    self.reread = getattr(self, 'reread', set()) | {unit.id}
                    self.fail_cause = f'no unit is seen at {src}, where {unit} is tracked - nothing was dragged'
                    return False
            cap = self.team_cap(scr)
            if src[0] == 'bench' and dst[0] == 'board' and self.roster.at(dst) is None and cap:
                got = self.cap.read()
                bars = B.occupancy(got[0], exclude=self.watcher.tac[1])[1] if got else set()
                # bars count units, the cap counts team slots: an Elder Dragon shows one bar and uses 2, and needs 2;
                # a unit that uses none (Elderwood plants, the BFF, a Training Dummy) shows a bar and uses 0 (run
                # 20261015-115044 3-5: 5 champions + a Stonebark Tree read "6/6 full", Sentinel was never fielded)
                n_board = len(bars) - sum(1 for u in self.roster.board() if u.slotless and u.loc[1:] in bars)
                n_board += sum(u.slots - 1 for u in self.roster.board() if u.slots > 1)
                # the game's own count (the n of its "n/m") leaves plants out even when our records lost them (run
                # 20261015-161625 5-6..6-1: "9/9 full" with two untracked plants); it can lag a move by a second, the
                # bars cannot - the lower of the two, and the drag is checked on screen anyway
                if getattr(scr, 'board_count', None) is not None:
                    n_board = min(n_board, scr.board_count)
                if n_board + unit.slots > cap:
                    # the board is full (run 152203: two untracked Varus held slots): a drop on an empty
                    # hex is refused, so say so instead of dragging
                    self.fail_cause = (f'the board is full ({n_board}/{cap} units on screen): put a unit on the '
                                       f'bench or sell one first, or drag onto an occupied hex to swap')
                    return False
            if dst[0] == 'bench' and not REPLAY and self.roster.at(dst) is None:
                # the slot is looked at before the drop: a container or an object there takes no unit (run 20261016-011010
                # 2-2: Rakan dragged onto a component anvil the records thought was a free slot)
                got = self.cap.read()
                if got is not None:
                    from tft_player import bench as BN
                    bench_seen = B.bar_places(got[0], exclude=self.watcher.tac[1])[0]
                    kinds = BN.classify(got[0], bench_seen)
                    if kinds[dst[1]]['kind'] != 'empty':
                        if kinds[dst[1]]['kind'] == 'interactable':
                            self.roster.blocked = {**self.roster.blocked, dst[1]: f"a container ({kinds[dst[1]]['cls']})"}
                        free = [i for i, k in enumerate(kinds) if k['kind'] == 'empty' and self.roster.at(('bench', i)) is None]
                        if not free:
                            self.fail_cause = (f"bench slot {dst[1]} holds {kinds[dst[1]]['cls'] or 'something'} and no "
                                               f"bench slot is free")
                            return False
                        self.log(ev='bench_slot_taken', stage=scr.stage, slot=dst[1], kind=kinds[dst[1]]['kind'],
                                 cls=kinds[dst[1]]['cls'], instead=free[0])
                        dst = ('bench', free[0])
            if src[0] == 'board' and dst[0] == 'bench' and self.w_to_bench(unit, src):
                return True                      # W with the cursor on the unit: faster and exact
            # (in the last seconds a unit already read off its own panel is not read again: fielding comes first)
            read_first = not (getattr(self, 'rushing', False) and getattr(unit, 'sure', False))
            if (src[0] == 'bench' and dst[0] == 'board' and unit.name and not REPLAY and read_first
                    and self.confirm_before_field(unit, src, scr)):
                return False                     # it is another champion: records fixed, the model is asked again
            occupant = self.roster.at(dst) if dst[0] == 'board' else None
            if (src[0] == 'bench' and occupant is not None and occupant is not unit and occupant.name and not REPLAY
                    and not (getattr(self, 'rushing', False) and getattr(occupant, 'sure', False))
                    and self.occupant_differs(occupant, dst, scr)):
                return False                     # a swap would bench another unit than the plan meant
            pre = self.board_now()
            start = loc_point(src)
            if src[0] == 'bench' and not REPLAY:          # a small bench unit is not under its slot's point
                got = self.cap.read()
                start = self.unit_point(src, B.green_bars(got[0]) if got else None)
            inp.drag(start, loc_point(dst))
            took = self.move_took(src, dst, pre=pre)
            if took == 'elsewhere':
                took = self.fix_landing(unit, dst)
            elif took is True and dst[0] == 'board' and pre is not None and not REPLAY                     and self.roster.at(dst) is not None and self.roster.at(dst) is not unit and unit.name:
                # a swap: the bars look the same before and after, so read who stands there now (a click)
                if self.swap_left_another(unit, src, dst):
                    return False
            if not took:
                got = self.cap.read()
                said = self.game_message(got[0]) if got is not None and not REPLAY else None
                self.fail_cause = getattr(self, 'landing_note', None) or \
                    (f'the unit did not move; the game said: "{said}"' if said else
                     'the unit did not move: the drop missed or the target was refused')
                self.landing_note = None
                return False                     # roster unchanged: it still matches the screen
            self.roster.moved(unit, dst)
            return True
        if a.kind == 'item':
            if self.hold_item_over(a, scr):
                return False
            return self.give_item(a, scr)
        if a.kind == 'link':
            loc, target = a.arg
            if self.phase.planning and scr.timer is not None and scr.timer <= 2:
                self.fail_cause = 'too late in planning for a link (it takes ~1.5 s): kept for the next round'
                self.deferred = True
                return False
            # data for checking a link on screen (the progress bar, what changes once linked): frames from just
            # before the press to 1.5 s after the release, in runs/<ts>/links/<n>_<t>.jpg
            shots, stop = [], threading.Event()

            def film():
                t0 = time.time()
                while not stop.is_set() and time.time() - t0 < 4.0:
                    raw = getattr(self.cap, 'raw', None)
                    if raw is not None:
                        shots.append((time.time() - t0, raw.copy()))
                    time.sleep(0.2)
            filmer = threading.Thread(target=film, daemon=True)
            filmer.start()
            inp.drag_hover(loc_point(loc), loc_point(target.loc), hover=1.0)
            time.sleep(1.5)
            stop.set()
            filmer.join(timeout=1)
            n = self.__dict__.setdefault('_links_n', [0])
            n[0] += 1
            folder, k = self.log_dir / 'links', n[0]

            def save(shots=shots):
                from PIL import Image
                folder.mkdir(exist_ok=True)
                for t, f in shots:
                    Image.fromarray(f).resize((f.shape[1] * 1080 // f.shape[0], 1080)).save(
                        folder / f'{k:02d}_{scr.stage}_{t:4.1f}s.jpg', quality=85)
            if test_mode():                      # data for learning the link's look: test games only
                threading.Thread(target=save, daemon=True).start()
            got = self.cap.read()
            said = self.game_message(got[0]) if got is not None and not REPLAY else None
            # no on-screen check of the link yet (its look is to be learned from a recorded game): the game's own
            # notice, if it shows one, is passed on
            self.log(ev='link', stage=scr.stage, src=list(loc), to=repr(target), said=said)
            # the BFF goes to the target (the user, 2026-10-12: drag the rider, hover the other Sprykin, "and then the
            # BFF went to the new champion"); from the BFF itself (before its first rider) it leaves its own hex
            src = self.roster.at(loc)
            from tft_player.trait_text import pair_hosts
            hosts = pair_hosts()
            if target.name in hosts or (src is not None and src.name in hosts):
                # an ally dragged onto Taric: the two are paired (a line between them) and stay paired through moves
                # and rounds (the user, 2026-10-13) - this is no BFF rider
                host, ally = (target, src) if target.name in hosts else (src, target)
                if ally is not None:
                    for r in self.roster.units:
                        if getattr(r, 'paired', None) in (host.id, ally.id):
                            r.paired = None
                    host.paired, ally.paired = ally.id, host.id
                    self.log(ev='paired', stage=scr.stage, host=repr(host), ally=repr(ally))
                if said:
                    self.news_event('link', scr.stage, f'link {loc[1:]} -> {target}: the game said "{said}"')
                return True
            if src is not None and src.name == BFF:
                self.bff_hex, self.bff_assigned = src.loc, True
                self.roster.units.remove(src)
            for r in self.roster.units:
                if getattr(r, 'riding', None) and r is not target:
                    r.riding = None
            target.riding = True
            if said:
                self.news_event('link', scr.stage, f'link {loc[1:]} -> {target}: the game said "{said}"')
            return True
        if a.kind == 'strip':
            slot, tool, unit = a.arg
            got = self.cap.read()
            before = sum(1 for n in read_item_bench(got[0]) if n)
            # onto the unit's body, as item gives are (2026-10-16): the hex centre or the bench slot's floor point
            # missed (benchmark game 20261018-150046: Ornn on bench slot 5, Shen at (3,4), Rek'Sai at (3,2) - "the
            # items did not come back", three times)
            bars = B.green_bars(got[0]) if got is not None and not REPLAY else None
            if unit.on_board:
                own = next((b for b, place in B.bar_places(None, bars=bars or [])[2] if place == unit.loc), None)
                point = Player.drop_point(self, unit, own, scr.stage)[0] if own is not None else None
            else:
                point = self.unit_point(unit.loc, bars) if bars is not None else None
            inp.drag(slot_point(slot), point or loc_point(unit.loc))
            took = False
            for _ in range(20):                  # the items come back to the bench a moment later (up to ~2 s)
                time.sleep(0.1)
                got = self.cap.read()
                if got is None:
                    continue
                now = read_item_bench(got[0])
                n_now = sum(1 for n in now if n)
                # the items are back; a used-up consumable leaves its own slot empty
                if n_now >= before + len(unit.items) or (now[slot] is None and n_now >= before + len(unit.items) - 1):
                    took = True
                    break
            if not took:
                self.fail_cause = 'the items did not come back to the item bench (missed the unit, or the tool could not be used)'
                return False
            self.brain.stripped.add(unit.id)
            self.news_event('strip', scr.stage, f'{tool} used on {unit}: {", ".join(unit.items)} back on the item bench')
            unit.items.clear()
            sig = (getattr(self, 'item_sigs', None) or [None] * N_SLOTS)[slot]
            n = uses_left(self.item_names.get(sig, tool))
            if sig in self.item_names and n:
                self.item_names[sig] = re.sub(r'\(\d+ uses', f'({n - 1} uses', self.item_names[sig])
            return True
        if a.kind == 'level':
            # n presses in a row, ONE check at the end (the user: like a player pressing F n times)
            n = a.arg if isinstance(a.arg, int) and a.arg > 0 else 1
            gold0 = scr.gold if scr.gold is not None else getattr(self.round_plan, 'last_gold', None)
            for _ in range(n):
                inp.key('f')
                time.sleep(0.06)
            # the HUD shows the result a few frames later - sometimes later than 0.25 s (run 20261015-003101 3-1:
            # 27 -> 27 read, 23 a moment later, and the 'missed' press was bought again): read until the gold shows
            # all n presses, or stops changing, up to 1.2 s
            gold1, end, prev = None, time.time() + 1.2, None
            time.sleep(0.2)
            while time.time() < end:
                got = self.cap.read()
                g = self.reader.read_shop(got[0])[0] if got else None
                if g is not None:
                    gold1 = g
                    if gold0 is None or gold0 - g >= 4 * n or (g == prev and gold0 - g > 0):
                        break
                    prev = g
                time.sleep(0.1)
            if gold0 is not None and gold1 is not None:
                a.landed = max(0, min(n, round((gold0 - gold1) / 4)))
            else:
                a.landed = n                     # unread: the next state shows the level anyway
            if a.landed < n:
                self.fail_cause = f'{n - a.landed} of {n} XP buys did not land (gold {gold0} -> {gold1})'
            return a.landed > 0
        if a.kind == 'rolldown':
            return self.run_rolldown(a.arg, scr)
        if a.kind == 'roll':
            raw0 = list(scr.shop_raw)
            inp.key('d')
            self.brain.rolled_this_round += 1
            # the next decision must see the NEW shop: wait until the cards changed and read the
            # same twice in a row (a card caught mid-flip reads as garbage, and the shop would be
            # rolled past without its champions being looked at)
            seen = []

            def settled(s):
                seen.append(list(s.shop_raw))
                return (s.shop_visible and sum(1 for x, y in zip(s.shop_raw, raw0) if x != y) >= 3
                        and len(seen) >= 2 and seen[-1] == seen[-2])
            if not self.wait_for(settled, timeout=1.5):
                self.log(ev='roll_unsettled', stage=scr.stage, before=raw0, reads=seen[-3:])
            return True
        if a.kind == 'augment':
            inp.click(AUGMENT_CARDS[a.arg])
            time.sleep(0.4)
            return True
        return False

    def wait_card(self, i, sig0, timeout, changed):
        """Shop card i was bought: its strip differs from sig0 (the card left the slot), on two frames in a
        row. sig0 None (no picture before): accepted after a short wait."""
        if sig0 is None or REPLAY:
            time.sleep(0.1)
            return True
        end, streak = time.time() + timeout, 0
        while time.time() < end:
            time.sleep(0.03)
            got = self.cap.read()
            if got is None:
                continue
            sig = self.reader.shop_sig(got[0])
            streak = streak + 1 if changed(sig[i], sig0[i]) else 0
            if streak >= 2:
                return True
        return False

    def park_tactician(self, scr):
        """Once a planning phase, when no loot is left to collect and the units have walked back: walk our
        tactician to PARK (one right-click on the ground), so it hides no unit or bar. Usually it already stands
        there (the game puts it there after the trip home): no click then."""
        if REPLAY or getattr(self, '_parked', None) == scr.stage or time.time() - self.round_started < 1.5:
            return
        if scr.timer is not None and scr.timer < 4:
            return
        if self.loot.target is not None or time.time() - getattr(self, '_orbs_seen_at', 0) < 4.5:
            # an orb is still out (orbs blink out of detection, and PvP rounds scan every ~2 s): parking now walks the
            # tactician away from it (run 20261010-064944 2-3: parked twice on the way, the orb took 8 s)
            return
        self._parked = scr.stage
        feet = self.watcher.tac[0] if getattr(self, 'watcher', None) is not None else None
        if feet is not None and math.dist(feet, PARK_FEET) < 60:
            return                               # already there
        try:
            self.inp.click(PARK, 'right')
        except NotForeground:
            return
        self.log(ev='parked', stage=scr.stage)

    def sell_seen(self, gold0, timeout=1.5):
        """-> the gold after the sell, or False. A sell happened only when the gold rose (the user, 2026-10-12: "sell should only be confirmed by the change of
        the gold, not by whether it pressed E"). The unit's bar going is no proof: at combat start every unit flies to the
        other board and every bar goes at once (run 20261012-194821 2-5: E on Rek'Sai at t=1 hit an empty slot, the
        empty slot was taken for a sell, and Rek'Sai stayed on the bench). -> True / False (no rise in time: not sold,
        tried again)."""
        if gold0 is None:
            return False
        end = time.time() + timeout
        while time.time() < end:
            time.sleep(0.08)
            got = self.cap.read()
            if got is None:
                continue
            g = self.reader.read_shop(got[0])[0]
            if g is not None and g > gold0:
                return g                          # the gold after: what it brought tells which unit went
        return False

    def board_now(self):
        """The board hexes our health bars show right now (a fresh frame)."""
        got = self.cap.read()
        return set(B.occupancy(got[0], exclude=self.watcher.tac[1])[1]) if got else None

    def w_to_bench(self, unit, src):
        """W with the cursor on a fielded unit sends it to the bench (the game's own hotkey: no drag, no
        hex to miss). True when a new bench bar appeared and a board bar went; the unit's slot is then
        the one the screen shows. False -> the caller drags instead."""
        got = self.cap.read()
        if got is None or REPLAY:
            return False
        tac = self.watcher.tac[1]
        bench0, board0 = B.occupancy(got[0], exclude=tac)
        bars = B.green_bars(got[0])
        try:
            self.inp.key('w', at=self.unit_point(src, bars))
        except NotForeground:
            return False
        streak, end = 0, time.time() + 0.8
        while time.time() < end:
            time.sleep(0.07)
            got = self.cap.read()
            if got is None:
                continue
            bench, board = B.occupancy(got[0], exclude=tac)
            new = [i for i in range(9) if bench[i] and not bench0[i]]
            streak = streak + 1 if new and len(board) < len(board0) else 0
            if streak >= 2:
                self.roster.moved(unit, ('bench', new[0]))
                self.log(ev='w_to_bench', unit=repr(unit), slot=new[0])
                return True
        self.log(ev='w_failed', unit=repr(unit), src=src)
        return False

    def fix_landing(self, unit, dst):
        """The drop landed on another hex than dst (move_took saw where). Read the unit there off its info
        panel (a click, ~0.4 s) to be sure it is this one, record the true place, and drag it to dst.
        -> True when it now stands on dst."""
        landed = self.landed_at
        got = self.cap.read()
        if landed is None or got is None:
            return False
        bars = B.green_bars(got[0])
        name, _, _ = self.read_unit(self.unit_point(('board', *landed), bars))
        if unit.name and name and name != unit.name:
            self.need_survey = f'a drop to {dst[1:]} left {unit.name} unseen; {name} stands at {landed}'
            self.landing_note = f'the unit is not where the drop went; {name} stands at {landed}'
            return False
        unit.loc = ('board', *landed)
        self.log(ev='landed_elsewhere', unit=repr(unit), aimed=dst, landed=landed)
        pre = self.board_now()
        self.inp.drag(loc_point(unit.loc), loc_point(dst))
        again = self.move_took(unit.loc, dst, pre=pre)
        if again == 'elsewhere':
            unit.loc = ('board', *self.landed_at)
            self.landing_note = f'the unit stands at {self.landed_at}, not {dst[1:]} (dragged twice)'
            return False
        return bool(again)

    def move_took(self, src, dst, timeout=0.7, pre=None):
        """Did a drag src -> dst happen? Checked on fresh frames from our units'
        health bars. A drag onto an occupied spot swaps two units, which bars
        can't tell apart from no change: accepted as done."""
        from tft_player.state import _hex_near
        if self.roster.at(dst) is not None:
            time.sleep(0.08)
            return True
        end = time.time() + timeout
        streak = 0                       # a bar can blink out for one frame: need two in a row
        others_benched = any(u.loc[0] == 'bench' and u.loc != src for u in self.roster.units if not u.missing)
        while time.time() < end:
            time.sleep(0.07)
            got = self.cap.read()
            if got is None:
                continue
            bench, board = B.occupancy(got[0], exclude=self.watcher.tac[1])
            lights = None
            if dst[0] == 'bench' and others_benched and not any(bench):
                # no bench bar at all while other units stand there: the round flipped to combat (bench units show
                # no bars in a fight) - the tiles tell (benchmark game 20261018-161404 1-4: Akali went from slot 2 to
                # 3 at 1 s, "did not move" kept her at 2, and five combat sells pressed E on the empty slot)
                lights = self.bench_lights(got[0])
            took = (src[0] == 'bench' and dst[0] == 'board' and not bench[src[1]]) or \
                   (dst[0] == 'bench' and lights is None and bench[dst[1]] and (src[0] != 'bench' or not bench[src[1]])) or \
                   (dst[0] == 'bench' and lights is not None and lights[dst[1]] is True
                    and (src[0] != 'bench' or lights[src[1]] is False)) or \
                   (src[0] == 'board' and dst[0] == 'board' and any(_hex_near(dst[1:], h) for h in board)
                    and not any(h == src[1:] for h in board)) or \
                   (src[0] == 'board' and dst[0] == 'board' and pre is not None and dst[1:] in board
                    and dst[1:] not in pre)
            # (the last: a bar new on exactly dst is the unit arriving, even when src still shows one - a tall
            # neighbour's bar maps onto src: run 20261018-161404 5-2, Vi stood on (3,3) and "did not move" twice,
            # Kayle's bar at (0,0) read as (1,0))
            streak = streak + 1 if took else 0
            if streak >= 2:
                # it moved - but WHERE? a bar on exactly dst, or a new bar somewhere else (the drop went to
                # another hex: the roster must not be told it is on dst)
                self.landed_at = None
                if dst[0] == 'board' and pre is not None and dst[1:] not in board:
                    new = sorted(board - pre)
                    if len(new) == 1:
                        self.landed_at = new[0]
                        return 'elsewhere'
                return True
        self.log(ev='move_failed', src=src, dst=dst)
        return False

    def swap_left_another(self, unit, src, dst):
        """After a swap onto an occupied hex (bars look the same before and after) the unit on dst is read off its
        panel. -> True when it is another champion: the records are checked by a survey. Only a panel that changed
        for this click counts: a click that missed leaves the panel of the unit selected before (benchmark game
        20261018-161404 1-4: Kayle did land on (0,0), the read gave the benched Leona selected earlier, "after the
        swap Leona stands at (0,0)" failed the step and its retry swapped Kayle back)."""
        time.sleep(0.15)
        got = self.cap.read()
        if got is None:
            return False
        bars = B.green_bars(got[0])
        # the swapped-out unit now stands on src: shown first when the panel did not change (another champion)
        occupant = next((u for u in self.roster.units if u.loc == dst and u is not unit), None)
        others = [self.unit_point(src, bars)] if occupant is not None and occupant.name != unit.name else []
        name = self.read_fresh(self.unit_point(dst, bars), others)[0]
        if name and name != unit.name:
            self.need_survey = f'a swap onto {dst[1:]} left {name} there, not {unit.name}'
            self.fail_cause = f'after the swap {name} stands at {dst[1:]}, not {unit.name}: the records are being checked'
            return True
        return False

    def stop_recording(self):
        if self.recorder:
            self.recorder.stop()
            self.recorder = None

    def latency_report(self):
        if not self.lat:
            return ''
        a = np.array(self.lat) * 1000
        return ' | '.join(f'{k} p50 {np.percentile(a[:, i], 50):.0f} p95 {np.percentile(a[:, i], 95):.0f}ms'
                          for i, k in enumerate(('read', 'decide', 'act')))


def bench_choice_open(frame):
    from tft_player import bench
    return bool(bench.choice_cards(frame))


