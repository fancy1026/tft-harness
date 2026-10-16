"""Queue and ready-check through the League client's local API. No desktop input.

(Ported from the integration checkout's tft_runtime/lobby.py.)

Getting into a match is a client-session concern, not a gameplay one, so it is kept
entirely off the input path: this module never imports the actuator, never takes the
controller lock, and never needs the window focused. Clicking the launcher's
Chromium UI was the alternative and is both fragile and explicitly restricted by
tft_controller, which permits one diagnostic launcher click only.

Authentication comes from the client's own lockfile, which the running client owns;
nothing is stored, copied into the repository, or sent anywhere.
"""
import base64
from pathlib import Path
import time

DEFAULT_LOCKFILE = Path(r'C:\Riot Games\League of Legends\lockfile')
# Riot's gameflow states, in the order a queued match passes through them.
QUEUED = ('Matchmaking', 'ReadyCheck', 'ChampSelect', 'GameStart', 'InProgress')
PLAYING = ('GameStart', 'InProgress', 'Reconnect')
TFT_QUEUES = {1090: 'TFT Normal', 1100: 'TFT Ranked', 1130: 'TFT Hyper Roll',
              1160: 'TFT Double Up', 1170: 'TFT Fortune'}


class LobbyError(RuntimeError):
    pass


class LeagueClient:
    """Read-only client state plus the two actions needed to enter a match."""

    def __init__(self, lockfile=None, client=None):
        self.lockfile = Path(lockfile or DEFAULT_LOCKFILE)
        self._client = client

    @property
    def client(self):
        if self._client is None:
            self._client = self._connect()
        return self._client

    def _connect(self):
        import httpx
        if not self.lockfile.exists():
            raise LobbyError(f'League client is not running: no lockfile at {self.lockfile}')
        parts = self.lockfile.read_text().split(':')
        if len(parts) < 5: raise LobbyError('Unreadable lockfile format')
        _, _, port, password, protocol = parts[:5]
        token = base64.b64encode(f'riot:{password}'.encode()).decode()
        # The client serves a self-signed certificate on loopback only.
        return httpx.Client(base_url=f'{protocol}://127.0.0.1:{port}', verify=False, timeout=10,
                            headers={'Authorization': f'Basic {token}'})

    def request(self, method, path, json=None):
        response = self.client.request(method, path, json=json)
        body = None
        if response.content:
            try: body = response.json()
            except ValueError: body = response.text
        return response.status_code, body

    def phase(self):
        """Current gameflow phase, or None when the client will not say."""
        status, body = self.request('GET', '/lol-gameflow/v1/gameflow-phase')
        return body if status == 200 and isinstance(body, str) else None

    def lobby(self):
        status, body = self.request('GET', '/lol-lobby/v2/lobby')
        return body if status == 200 and isinstance(body, dict) else None

    def summoner(self):
        status, body = self.request('GET', '/lol-summoner/v1/current-summoner')
        return body if status == 200 and isinstance(body, dict) else None

    def queue_id(self):
        lobby = self.lobby() or {}
        return (lobby.get('gameConfig') or {}).get('queueId')

    def create_lobby(self, queue_id):
        status, body = self.request('POST', '/lol-lobby/v2/lobby', json={'queueId': queue_id})
        if status not in (200, 204):
            raise LobbyError(f'Could not create a lobby for queue {queue_id} ({status}): {body}')
        return True

    def start_search(self):
        status, body = self.request('POST', '/lol-lobby/v2/lobby/matchmaking/search')
        if status not in (200, 204):
            raise LobbyError(f'Could not start matchmaking ({status}): {body}')
        return True

    def cancel_search(self):
        return self.request('DELETE', '/lol-lobby/v2/lobby/matchmaking/search')[0] in (200, 204)

    def accept_ready_check(self):
        status, body = self.request('POST', '/lol-matchmaking/v1/ready-check/accept')
        if status not in (200, 204):
            raise LobbyError(f'Could not accept the ready check ({status}): {body}')
        return True

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None


    def dismiss_end_of_game(self):
        """Leave honor/stats screens. Does not click the desktop."""
        honor = self.request('POST', '/lol-honor-v2/v1/honor-player', json={'honorInt': 0})[0]
        again = self.request('POST', '/lol-lobby/v2/play-again')[0]
        return honor in (200, 204, 400) and again in (200, 204)


def enter_match(client, emit=print, timeout=1800, poll=1.0, require_tft=True, now=time.monotonic,
                queue_id=None):
    """Queue if needed, accept every ready check, and return once a game is live.

    Declines by other players or a dodge return the lobby to Lobby, so searching
    restarts rather than waiting out the timeout. Reports each phase transition.
    """
    if client.phase() in ('PreEndOfGame', 'EndOfGame', 'WaitingForStats'):
        client.dismiss_end_of_game()
        emit({'type': 'end_of_game_dismissed', 'phase': client.phase()})
        time.sleep(poll)
    if queue_id is not None and client.queue_id() != queue_id:
        if client.phase() in ('Matchmaking', 'ReadyCheck'):
            # a search in another queue is still running: the client refuses a new lobby until it stops (423
            # "Gameflow prevented a lobby", 2026-10-15: Normal -> Ranked)
            client.request('DELETE', '/lol-lobby/v2/lobby/matchmaking/search')
            emit({'type': 'search_cancelled', 'queue_id': client.queue_id()})
            time.sleep(poll)
        client.create_lobby(queue_id)          # never search in a queue we didn't ask for
        emit({'type': 'lobby_created', 'queue_id': queue_id})
        time.sleep(poll)
    lobby = client.lobby()
    if queue_id is not None and client.queue_id() != queue_id:
        raise LobbyError(f'Lobby is queue {client.queue_id()}, not {queue_id}')
    if lobby is None:
        raise LobbyError('No lobby is open; create a TFT lobby in the client first')
    queue = client.queue_id()
    if require_tft and queue not in TFT_QUEUES:
        raise LobbyError(f'Lobby queue {queue} is not a TFT queue; open a TFT lobby first')
    emit({'type': 'lobby_ready', 'queue_id': queue, 'queue': TFT_QUEUES.get(queue, 'unknown'),
          'can_start': lobby.get('canStartActivity')})
    deadline = now()+timeout
    seen, accepted, searches = None, 0, 0
    since, beat = now(), now()
    while now() < deadline:
        try:
            phase = client.phase()
        except Exception as error:               # the client did not answer: say so, keep polling
            emit({'type': 'client_error', 'error': repr(error)})
            time.sleep(poll)
            continue
        if phase != seen:
            emit({'type': 'gameflow_phase', 'phase': phase, 'accepted': accepted})
            seen, since = phase, now()
        elif now() - beat > 30:
            beat = now()                         # a heartbeat: a stall is visible in the log (run30)
            emit({'type': 'still_waiting', 'phase': phase, 'for_s': round(now() - since), 'accepted': accepted})
        if phase == 'ReadyCheck' and now() - since > 60:
            # a ready check lasts ~10 s: one stuck for a minute is restarted (run30 sat there 9 min)
            emit({'type': 'ready_check_stuck', 'for_s': round(now() - since)})
            try:
                client.request('DELETE', '/lol-lobby/v2/lobby/matchmaking/search')
            except Exception as error:
                emit({'type': 'cancel_failed', 'error': repr(error)})
            since = now()
        if phase in PLAYING:
            return {'entered': True, 'phase': phase, 'accepted': accepted, 'searches': searches}
        if phase == 'ReadyCheck':
            # Accept repeatedly: one call can land before the check is registered.
            try:
                client.accept_ready_check(); accepted += 1
            except LobbyError as error:
                emit({'type': 'ready_check_retry', 'error': str(error)})
        elif phase in ('PreEndOfGame', 'EndOfGame', 'WaitingForStats'):
            try:
                client.dismiss_end_of_game()
                emit({'type': 'end_of_game_dismissed', 'phase': phase})
            except LobbyError as error:
                emit({'type': 'end_of_game_retry', 'error': str(error)})
        elif phase in ('Lobby', 'None', None):
            if phase == 'Lobby':
                try:
                    client.start_search(); searches += 1
                    emit({'type': 'matchmaking_started', 'searches': searches})
                except LobbyError as error:
                    emit({'type': 'matchmaking_retry', 'error': str(error)})
        time.sleep(poll)
    return {'entered': False, 'phase': seen, 'accepted': accepted, 'searches': searches,
            'reason': 'timeout'}
