"""End of game: notice it, leave it, record the official placement.

A game ends in one of two ways:
  - we win (or the last opponent dies): the client's gameflow phase leaves
    InProgress by itself;
  - we are eliminated: the game keeps running (spectating) until we exit.

The watcher polls the gameflow phase (League client local API, no input) and,
from stage 3 on while the shop HUD is hidden, runs full-frame text detection
looking for the post-elimination buttons. It never clicks by itself: the
player's loop asks it what to do and does the one click.

The placement always comes from Riot's match history afterwards, never from
the screen.
"""
from __future__ import annotations

import re
import threading
import time

import numpy as np

PLAYING = ('GameStart', 'InProgress', 'Reconnect')
# buttons that only exist once we are out. NOT the Settings menu's "EXIT TFT",
# which is there all game and would abandon a game we are still in.
EXIT_WORDS = ('EXIT NOW', 'EXIT GAME', 'LEAVE GAME')
EXIT_TFT_BUTTON = (756, 916)        # Settings menu, left of Surrender (1920x1080 reference)
SCAN_EVERY = 5.0


class EndWatcher(threading.Thread):
    def __init__(self, cap, reader, client=None, log=print):
        super().__init__(daemon=True)
        self.cap, self.reader, self.client, self.log = cap, reader, client, log
        self.phase = None
        self.game_over = False          # gameflow left InProgress
        self.exit_button = None         # (x, y) in frame pixels of a visible exit button
        self.eliminated_seen = None     # text that told us we are out
        self.want_scan = False          # set by the loop: stage >= 3 and HUD hidden
        self.scan_asked = False         # set by the loop on a greyed screen: scanned by this thread, at once
        self.scanned_at = 0.0
        self.running = True

    def run(self):
        last_phase_at = 0.0
        while self.running:
            now = time.time()
            if self.client and now - last_phase_at > 2.0:
                last_phase_at = now
                try:
                    self.phase = self.client.phase()
                except Exception:
                    self.phase = None
                if self.phase and self.phase not in PLAYING:
                    self.game_over = True
            if self.scan_asked and now - self.scanned_at > 0.4:
                self.scan_asked = False
                self.scanned_at = now
                self._scan()
            elif self.want_scan and now - self.scanned_at > SCAN_EVERY:
                self.scanned_at = now
                self._scan()
            time.sleep(0.05 if self.scan_asked else 0.25)

    GREY_BELOW = 14.0               # mean colour spread of the screen once the elimination overlay greys it
                                    # (normal play 40, eliminations 5-8; run 152203 asked the model at 5-4)
    DARK_BELOW = 20.0               # mean brightness: eliminations 36-48, the black scene changes 7-8
                                    # (carousel -> board: run 185805 took those for eliminations)

    def greyed(self, frame):
        """The elimination (and end) overlay turns the whole screen grey: a pixel check, ~1 ms, so
        the text scan runs the moment it appears instead of every 5 s and never in carousel rounds.
        A black scene change is not grey."""
        small = frame[::16, ::16].astype(np.int16)
        return (float((small.max(axis=2) - small.min(axis=2)).mean()) < self.GREY_BELOW
                and float(small.mean()) > self.DARK_BELOW)

    def scan_now(self):
        """Look for 'YOU FINISHED nTH PLACE' / EXIT NOW now (not when the timer comes round). The scan
        (full-screen OCR, 3.5-4 s) runs on this thread: in the game loop it froze the loop at every black
        scene change (run 185805: each round after a carousel was seen 8-13 s late)."""
        self.scan_asked = True

    def _scan(self):
        got = self.cap.read()
        if got is None:
            return
        frame = got[0]
        h, w = frame.shape[:2]
        hits = self.reader.find_text(frame)
        # the elimination panel is in the middle of the screen; our own overlay panels (which
        # show the model's text) sit at the edges: run29's model wrote "-0.34 AVG PLACE" there and
        # the bot took it for an elimination and opened the Settings menu 91 times while alive
        central = lambda xy: abs(xy[0] - w / 2) < 0.3 * w and 0.15 * h < xy[1] < 0.85 * h
        words = [(t.upper(), xy) for t, xy in hits if central(xy)]
        button = next((xy for t, xy in words if t.strip() == 'EXIT' or any(w in t for w in EXIT_WORDS)), None)
        out = next((t for t, _ in words if re.search(r'[1-8]\s*(ST|ND|RD|TH)\s*PLACE', t)
                    or 'ELIMINATED' in t), None)
        if button or out:
            self.words = [t for t, _ in words][:40]     # the place ("4TH") is read from these
            if (button, out) != (self.exit_button, self.eliminated_seen):
                self.log(ev='end_screen', button=button, text=out, words=[t for t, _ in words][:30])
            self.exit_button, self.eliminated_seen = button, out
        else:
            self.exit_button = None


def fetch_placement(client, since_ms, tries=20, wait=6.0):
    """Our placement in the newest TFT match that started after since_ms, from
    the client's match history (it appears a little after the game ends)."""
    me = client.summoner() or {}
    puuid = me.get('puuid')
    if not puuid:
        return None
    for _ in range(tries):
        status, body = client.request('GET', f'/lol-match-history/v1/products/tft/{puuid}/matches?begin=0&count=5')
        for g in (body or {}).get('games', []) if status == 200 else []:
            j = g.get('json') or g
            info = j.get('info') or j          # 2026-10: the match fields sit in 'json' itself, no 'info'
            # game_datetime (a string) is when the whole match ended, not when it started: the match before ours
            # can end after ours began (benchmark run 20261018-094821 was given the flex game's 8th place, which
            # ended 14 min into it) - compare the match's start
            start = int(info.get('game_datetime') or 0) - int(float(info.get('game_length') or 0) * 1000)
            if start < since_ms:
                continue
            p = next((x for x in info.get('participants', []) if x.get('puuid') == puuid), None)
            if p:
                return {'placement': p.get('placement'), 'level': p.get('level'),
                        'last_round': p.get('last_round'), 'gold_left': p.get('gold_left'),
                        'time_eliminated': p.get('time_eliminated'),
                        'queue_id': info.get('queue_id') or info.get('queueId'),
                        'game_datetime': info.get('game_datetime'),
                        'traits': [(t.get('name'), t.get('num_units'), t.get('tier_current'))
                                   for t in p.get('traits', []) if t.get('tier_current', 0) > 0],
                        'units': [(u.get('character_id'), u.get('tier'), u.get('itemNames'))
                                  for u in p.get('units', [])]}
        time.sleep(wait)
    return None
