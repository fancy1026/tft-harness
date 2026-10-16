"""Queue TFT matches and play them to the end, recording the official placement.

    python tools/play_match.py                 one full game
    python tools/play_match.py --games 5       five in a row

Per game:
1. queue + accept the ready check through the League client's local API
2. wait for the TFT window, run the player until the game is over
3. dismiss end-of-game screens, read our placement from match history and
   append it to runs/games.jsonl
F8 stops all game input at any time.
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings('ignore')
import tft_player.game  # noqa: E402,F401  (arms the F8 kill switch from the start, queueing included)

ROOT = Path(__file__).resolve().parent.parent


def gold(p):
    r, g, b = int(p[0]), int(p[1]), int(p[2])
    return r > 75 and r - g >= 20 and g - b >= 20


def menu_state(frame):
    """Settings / surrender-confirmation anchors, measured on 1920x1080 (centred, height-scaled)."""
    h, w = frame.shape[:2]
    s = h / 1080
    px = lambda x, y: frame[round(y * s), round(w / 2 + (x - 960) * s)]
    if gold(px(650, 390)) and gold(px(1268, 693)):
        return 'confirmation'
    if gold(px(400, 115)) and not gold(px(650, 390)):
        return 'settings'
    return 'unknown'


def surrender(cap, inp, client, log):
    """Escape -> Settings menu -> Surrender -> confirm. The menu and the confirmation are
    recognised by OCR of their text (the menu's button row: our overlay covers its title at
    16:10); success by the client's gameflow phase, never by a click alone."""
    from tft_player.screen import ScreenReader
    reader = ScreenReader()

    def screen():
        got = cap.read()
        return got[0] if got is not None else None

    def ended():
        return client.phase() in ('PreEndOfGame', 'EndOfGame', 'WaitingForStats')

    for _ in range(4):
        if ended():
            return 'surrendered'
        f = screen()
        if f is not None and reader.surrender_dialog(f):
            inp.click((1069, 624))      # confirm
            for _ in range(20):
                if ended():
                    return 'surrendered'
                time.sleep(0.5)
            continue
        if f is not None and reader.settings_open(f):
            inp.click((955, 915))       # Surrender (greyed out before 3-2)
            time.sleep(0.8)
            continue
        inp.key('escape')
        time.sleep(0.8)
    if ended():
        return 'surrendered'
    f = screen()
    if f is None or not (reader.settings_open(f) or reader.surrender_dialog(f)):
        return 'settings menu not seen'
    return 'no end-of-game phase after confirming'


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=1, help='full games to play in a row')
    ap.add_argument('--ranked', action='store_true', help='queue TFT Ranked (1100) instead of Normal (1090)')
    ap.add_argument('--normal', action='store_true', help=argparse.SUPPRESS)     # the default now
    ap.add_argument('--no-screenshots', action='store_true',
                    help="do not save screenshots in the game's folder (runs/<ts>); its log and the model "
                         'conversation are saved either way')
    ap.add_argument('--no-overlay', action='store_true', help='no on-screen explanation panels')
    args = ap.parse_args()
    os.environ['TFT_TEST_MODE'] = '0'
    if args.no_screenshots:
        os.environ['TFT_NO_SCREENSHOTS'] = '1'

    print(json.dumps({'code': str(ROOT)}), flush=True)     # which version is playing
    try:                                    # F8 also leaves the model's conversation open for questions
        from tft_player import game
        from tft_player.coach import _cli, active_session
        kind, exe = _cli()
        if kind in ('claude-cli', 'codex'):
            game.STOP_HOOKS.append(lambda: active_session().open_terminal(exe, locked=False))
    except Exception as e:
        print('F8 reopen hook not set:', e)
    from tft_player.lobby import LeagueClient
    from tft_player.game import overwolf_running
    client = LeagueClient()
    # the game needs no MetaTFT app (the user, 2026-10-15): its data comes from fetch_meta as text
    if overwolf_running():
        print('WARNING: Overwolf (MetaTFT companion) is running. Its in-game overlay can take the game input: '
              'clicks and keys (run20 lost ~30 inputs). Close it before the game.', flush=True)
    for n in range(args.games):
        print(f'===== game {n + 1}/{args.games}', flush=True)
        rec = play_one(client, args)
        print(json.dumps(rec), flush=True)
        if rec.get('result', '').startswith('stopped'):
            break
    try:                                    # keep the model's conversation: ask it why, in a terminal
        from tft_player.coach import _cli, active_session
        kind, exe = _cli()
        if kind in ('claude-cli', 'codex') and active_session().open_terminal(exe):
            print('model conversation reopened in a terminal window - ask it about the game', flush=True)
    except Exception as e:
        print('could not reopen the conversation:', e)
    client.close()


META_MAX_AGE_H = 48


def update_meta(max_age_h=META_MAX_AGE_H):
    """The MetaTFT data and the model's reference files, downloaded again when older than max_age_h
    (plain downloads, no model calls); META is reloaded in place. Returns what happened, in a few words."""
    import subprocess
    cur = ROOT / 'data' / 'meta' / 'current.json'
    if cur.exists() and time.time() - cur.stat().st_mtime < max_age_h * 3600:
        return f'data fresh ({(time.time() - cur.stat().st_mtime) / 3600:.1f} h old)'
    for tool in ('fetch_meta.py', 'build_reference.py'):
        r = subprocess.run([sys.executable, str(ROOT / 'tools' / tool)], cwd=ROOT, capture_output=True,
                           text=True, encoding='utf-8', errors='replace')
        if r.returncode:
            return f'{tool} failed: {(r.stderr or r.stdout).strip().splitlines()[-1:]}'
    from tft_player.meta import META
    META.__init__()                  # every module holds this same object
    return f'data refreshed: patch {META.patch}, {len(META.comps)} comps'


def refresh_meta():
    """While we queue, in the background (plain downloads, no model calls): the MetaTFT data and the
    model's reference files when older than META_MAX_AGE_H, and - for a fixed-comp game - the comp's
    page screenshots and their transcription when older than metatft_shots.MAX_AGE_H. The thread's
    .result says what happened; a refreshed META is reloaded in place."""
    import threading
    t = threading.Thread(daemon=True)

    def run():
        t0, said = time.time(), [update_meta()]
        from tft_player.coach import forced_comp
        comp = forced_comp()
        if comp:
            try:
                from tft_player.metatft_shots import comp_pages
                comp_pages(comp.name)            # no-op while younger than its MAX_AGE_H
                said.append(f'{comp.name} pages ready')
            except Exception as e:
                said.append(f'{comp.name} pages failed: {e}')
        t.result = f"{'; '.join(said)} ({time.time() - t0:.0f}s)"

    t.run = run
    t.result = 'running'
    t.start()
    return t


PREP_DATA_WAIT = 20      # s: longest the preparation waits for the MetaTFT refresh before it starts on the old data


def start_preparation(refresh):
    """The model's preparation, in the background while we queue (coach.prepare): it starts once the data
    refresh is done (it reads that data) and is cut off at the game's first question. The thread's .result
    is prepare()'s. Before: the briefing went out in the loading screen and was answered in 4-16 s with at
    most one file read, while 1-2 ran (10 games, 2026-10-10)."""
    import threading
    from tft_player.coach import prepare
    t = threading.Thread(daemon=True)

    def run():
        # the data on disk is good enough to prepare with: waiting for the whole refresh (~3 min) let the game's
        # first question arrive first and cut the preparation at once (run 20261018-092528: no briefing at 1-3)
        refresh.join(timeout=PREP_DATA_WAIT)
        print('model preparing (while we queue) ...' + ('' if not refresh.is_alive() else
              ' with the data on disk (the refresh is still running)'), flush=True)
        t.result = prep = prepare()
        meta = prep.get('meta') or {}
        print(json.dumps({'preparation_s': round(prep['seconds']), 'turns': meta.get('num_turns'),
                          'cut': prep['cut'], 'error': prep['error']}), flush=True)

    t.run = run
    t.result = None
    t.start()
    return t


def play_one(client, args):
    from tft_player import coach
    from tft_player.lobby import enter_match
    from tft_player.setup import INI, ensure_settings
    started_ms = int(time.time() * 1000)
    run_dir = ROOT / 'runs' / time.strftime('%Y%m%d-%H%M%S')
    coach.new_game(run_dir / 'calls')       # this game's own conversation (STOPPED from the last game cleared)
    if client.phase() in ('Lobby', 'None', None, 'EndOfGame', 'PreEndOfGame', 'WaitingForStats'):
        changed = ensure_settings()      # game not running: safe to set its display config
        print(json.dumps({'settings': str(INI), 'changed': changed}), flush=True)
    print(json.dumps({'phase': client.phase(), 'queue': client.queue_id()}), flush=True)
    refresh = prep = None
    if client.phase() not in ('InProgress', 'GameStart', 'Reconnect'):
        if os.environ.get('TFT_FORCE_COMP'):
            # a fixed comp is checked against fresh data before we queue for it: MetaTFT renames and
            # drops comps between patches, and a comp it no longer has has no page to brief the model
            print(json.dumps({'meta': update_meta()}), flush=True)
            if not coach.forced_comp():
                return {'result': f"stopped: no comp named {os.environ['TFT_FORCE_COMP']!r} on MetaTFT any more "
                                  f"- choose one again (python tools/play.py)"}
        refresh = refresh_meta()            # ~3 min of MetaTFT downloads, no model tokens: runs while we queue
        if coach._cli()[0]:
            prep = start_preparation(refresh)
        res = enter_match(client, emit=lambda e: print(json.dumps(e), flush=True), timeout=1800,
                          queue_id=1100 if args.ranked else 1090)   # Normal by default (the user, 2026-10-14)
        if not res.get('entered'):
            coach.cut_preparation('no game started')
            return {'result': f'stopped: could not enter a match: {res}'}

    from tft_player.game import Capture, Input, Stopped, find_client, settled_client
    from tft_player.player import Player
    t0 = time.time()
    client_win = None
    while time.time() - t0 < 240:
        client_win = find_client()
        if client_win:
            break
        time.sleep(1)
    if not client_win:
        return {'result': 'stopped: TFT window never appeared'}
    if refresh is not None:
        refresh.join(timeout=max(0, 90 - (time.time() - t0)))     # the loading screen gives it time
        print(json.dumps({'meta_refresh': refresh.result if not refresh.is_alive() else 'still running: old data'}),
              flush=True)
    client_win = settled_client(timeout=60) or client_win         # measured once its size holds (fullscreen)
    print(f'TFT client {client_win.width}x{client_win.height} at {client_win.left},{client_win.top}', flush=True)
    from tft_player.game import bring_to_front
    print('focus:', bring_to_front(client_win.hwnd), flush=True)
    time.sleep(2)
    cap = Capture(client_win)
    inp = Input(client_win, auto_focus=True)     # dedicated test machine: take focus when covered
    from tft_player.overlay import Overlay
    overlay = None if args.no_overlay else Overlay(client_win)
    p = Player(cap, inp, run_dir, None, overlay=overlay, client=client, prep=prep)
    result = 'stopped'
    try:
        p.run()
        result = 'game over' if p.ending.game_over else 'stage limit reached'
        if result == 'stage limit reached':
            for attempt in range(6):
                time.sleep(1)
                result = surrender(cap, inp, client, p.log)
                print('surrender:', result, flush=True)
                p.log(ev='surrender', result=result, attempt=attempt)
                if result == 'surrendered':
                    break
                # greyed-out Surrender (too early) or a missed click: close the menu, wait, retry
                from tft_player.screen import ScreenReader
                got = cap.read()
                if got is not None and ScreenReader().settings_open(got[0]):
                    inp.click((1476, 163))      # close the menu before retrying
                time.sleep(8)
    except Stopped as e:
        result = f'stopped: {e}'
        print(result, flush=True)
    finally:
        p.ending.running = False
        print('latency:', p.latency_report(), flush=True)
        p.reader.memo.save()                  # keep what this game's OCR learned
        p.stop_recording()
        cap.close()
        # the whole model conversation of this game, as a file in the run folder
        try:
            import subprocess
            subprocess.run([sys.executable, str(ROOT / 'tools' / 'transcript.py'), str(run_dir),
                            '--out', str(run_dir / 'conversation.md')], timeout=60)
        except Exception as e:
            print('conversation file failed:', e)
    rec = {'run_dir': str(run_dir), 'result': result, 'started_ms': started_ms,
           'seconds': round(time.time() - started_ms / 1000),
           'last_stage': p.phase.stage, 'comp': p.brain.plan.comp, 'carry': p.brain.plan.carry}
    from tft_player.meta import META
    played = META.comps.get(str(p.brain.plan.comp))
    rec.update(comp_name=played.name if played else None, patch=META.patch,    # MetaTFT renumbers comps each patch
               model=coach.model_name())
    if not result.startswith('stopped'):
        for _ in range(60):                     # wait for the game to close
            if client.phase() not in ('InProgress', 'GameStart', 'Reconnect'):
                break
            time.sleep(2)
        time.sleep(3)
        try:
            client.dismiss_end_of_game()
        except Exception as e:
            print('dismiss failed:', e)
        from tft_player.ending import fetch_placement
        rec.update(fetch_placement(client, started_ms, tries=3) or {'placement': None})     # the end screen usually gives it first
    # the place the game itself showed (or a win: the game ended with us still in it)
    scr_place = getattr(p, 'placement_screen', None)
    if scr_place:
        rec['placement_screen'], rec['placement_screen_how'] = scr_place
        if rec.get('placement') is None and scr_place[0] is not None:
            rec['placement'], rec['placement_source'] = scr_place[0], 'screen'
    with open(ROOT / 'runs' / 'games.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec) + '\n')
    p.log(ev='game_result', **rec)
    print('run dir:', run_dir, flush=True)
    try:
        import subprocess
        subprocess.run([sys.executable, str(ROOT / 'tools' / 'report.py'), str(run_dir)])
    except Exception as e:
        print('report failed:', e)
    return rec


if __name__ == '__main__':
    main()
