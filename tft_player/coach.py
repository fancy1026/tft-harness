"""The coach: an LLM that sets strategic direction, never in the action path.

Runs the Cursor (or Codex) CLI in a background thread. A call takes ~20s
(CLI start + model), so it is asked early - when the augment picker opens, or
during combat - and its answer only updates `brain.plan`. If it is late, the
local default (meta.py) has already acted.

Local code builds the menu - candidate comps and augment options with MetaTFT
evidence - and a default choice; the coach compares the menu against the game.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

from tft_player.catalog import CATALOG
from tft_player.meta import META

CATALOG_SET = 18

def _cli():
    """Backend for the model (TFT_COACH_BACKEND, set by use_model). 'claude': the Claude Code CLI, which keeps
    one conversation per stage and signs in with the user's Claude login or ANTHROPIC_API_KEY. 'openrouter':
    any OpenRouter model, also one conversation per stage. 'codex': the Codex CLI on the user's ChatGPT login, one
    conversation per stage too."""
    want = os.environ.get('TFT_COACH_BACKEND', 'claude').lower()
    if want == 'claude':
        cli = claude_cli()
        return ('claude-cli', cli) if cli else (None, None)
    if want == 'codex':
        exe = codex_cli()
        return ('codex', exe) if exe else (None, None)
    if want == 'openrouter' and api_key():
        return 'openrouter', OR_MODEL
    return None, None


OR_MODEL = os.environ.get('TFT_OR_MODEL', 'openai/gpt-6-sol')
OR_EFFORT = os.environ.get('TFT_OR_EFFORT', 'low')
MODEL_SPECS = ('claude[:<model id>] (Claude login), claude-key[:<model id>] (Anthropic API key), codex[:<model id>] '
               '(ChatGPT login, Codex CLI), openai:<model id> (OpenAI API key) or openrouter:<model id> (OpenRouter '
               'API key)')
# the chat-completions loop (_ORSession) talks to OpenRouter or to OpenAI directly: same messages, tools and stream
API_HOSTS = {'openrouter': ('https://openrouter.ai/api/v1', 'OPENROUTER_API_KEY', 'OpenRouter'),
             'openai': ('https://api.openai.com/v1', 'OPENAI_API_KEY', 'OpenAI')}


def api_host():
    return os.environ.get('TFT_API_HOST', 'openrouter')


def api_key():
    return user_env(API_HOSTS[api_host()][1])


def use_model(spec, effort=None):
    """The game's model, for this process and the processes it starts. spec: 'claude' or 'claude:<model id>'
    (Claude Code CLI) or 'openrouter:<model id>' (OPENROUTER_API_KEY; ids as on openrouter.ai/models).
    effort: low / medium / high. Returns why the model can't play, or None."""
    global CLAUDE_MODEL, CLAUDE_EFFORT, OR_MODEL, OR_EFFORT
    backend, _, mid = str(spec).partition(':')
    backend, mid = backend.strip().lower(), mid.strip()
    if backend in ('claude', 'claude-key'):
        if not claude_cli():
            return 'Claude Code was not found: install it from claude.com/claude-code'
        if backend == 'claude-key':
            key = anthropic_key()
            if not key:
                return 'ANTHROPIC_API_KEY is not set'
            why = check_anthropic_key(key)
            if why:
                return why
            os.environ['ANTHROPIC_API_KEY'] = key
        # 'claude' plays on the Claude login: a key left in the environment would bill the API instead (cli_env)
        os.environ['TFT_CLAUDE_AUTH'] = 'key' if backend == 'claude-key' else 'login'
        os.environ['TFT_COACH_BACKEND'] = 'claude'
        os.environ['TFT_CLAUDE_MODEL'] = CLAUDE_MODEL = mid or CLAUDE_MODEL
        if effort:
            os.environ['TFT_CLAUDE_EFFORT'] = CLAUDE_EFFORT = effort
        return None
    if backend in ('codex', 'chatgpt'):
        return _use_codex(mid, effort)
    if backend == 'openai':
        return _use_openai(mid, effort)
    if backend != 'openrouter':
        return f'unknown model {spec!r}: use {MODEL_SPECS}'
    if not mid:
        return 'name the model: openrouter:<model id>, as listed on openrouter.ai/models'
    if not openrouter_key():
        return 'OPENROUTER_API_KEY is not set'
    try:
        import urllib.request
        models = json.loads(urllib.request.urlopen('https://openrouter.ai/api/v1/models', timeout=15).read())['data']
    except (OSError, ValueError, KeyError) as e:
        return f'could not reach OpenRouter to check the model: {e}'
    m = next((x for x in models if x.get('id') == mid), None)
    if m is None:
        return f'OpenRouter has no model {mid!r} (ids as on openrouter.ai/models, e.g. openai/gpt-6-sol)'
    if 'image' not in ((m.get('architecture') or {}).get('input_modalities') or []):
        return f'{mid} does not take images: the model gets the game screen as screenshots'
    if 'tools' not in (m.get('supported_parameters') or []):
        return f'{mid} does not support tool calls: the model reads its reference files through tools'
    try:
        req = urllib.request.Request('https://openrouter.ai/api/v1/key',
                                     headers={'Authorization': 'Bearer ' + openrouter_key()})
        left = json.loads(urllib.request.urlopen(req, timeout=15).read())['data'].get('limit_remaining')
    except (OSError, ValueError, KeyError) as e:
        return f'OpenRouter did not accept the key: {e}'
    if left is not None and left < 1:
        return f'the OpenRouter key has ${left:.2f} of credit left - add credit on openrouter.ai/settings/credits'
    os.environ['TFT_COACH_BACKEND'], os.environ['TFT_API_HOST'] = 'openrouter', 'openrouter'
    os.environ['TFT_OR_MODEL'] = OR_MODEL = mid
    if effort:
        os.environ['TFT_OR_EFFORT'] = OR_EFFORT = effort
    return None


def _use_openai(mid, effort):
    """An OpenAI model through the chat-completions loop. OpenAI's model list says nothing about images or
    tools: a model without them fails at its first question."""
    global OR_MODEL, OR_EFFORT
    import urllib.error
    import urllib.request
    if not mid:
        return 'name the model: openai:<model id>, as listed on platform.openai.com/docs/models'
    key = user_env('OPENAI_API_KEY')
    if not key:
        return 'OPENAI_API_KEY is not set'
    try:
        req = urllib.request.Request(API_HOSTS['openai'][0] + '/models', headers={'Authorization': 'Bearer ' + key})
        ids = {m.get('id') for m in json.loads(urllib.request.urlopen(req, timeout=15).read())['data']}
    except urllib.error.HTTPError as e:
        return f'the OpenAI API did not accept the key (HTTP {e.code})'
    except (OSError, ValueError, KeyError) as e:
        return f'could not reach the OpenAI API to check the model: {e}'
    if mid not in ids:
        return f'OpenAI has no model {mid!r} for this key (ids as on platform.openai.com/docs/models)'
    os.environ['TFT_COACH_BACKEND'], os.environ['TFT_API_HOST'] = 'openrouter', 'openai'
    os.environ['TFT_OR_MODEL'] = OR_MODEL = mid
    if effort:
        os.environ['TFT_OR_EFFORT'] = OR_EFFORT = effort
    return None


def _use_codex(mid, effort):
    """A ChatGPT model through the Codex CLI on the user's ChatGPT login (the Codex app ships the CLI)."""
    global CODEX_MODEL, CODEX_EFFORT
    exe = codex_cli()
    if not exe:
        return 'Codex was not found: install the Codex app (or the Codex CLI) and log in with your ChatGPT account'
    try:
        out = subprocess.run([exe, 'login', 'status'], capture_output=True, timeout=30,
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        said = (out.stdout + out.stderr).decode('utf-8', 'replace').lower()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f'Codex did not start: {e}'
    if 'logged in' not in said or 'not logged in' in said:
        return 'Codex is not logged in: run "codex login" (or log in in the Codex app)'
    os.environ['TFT_COACH_BACKEND'] = 'codex'
    os.environ['TFT_CODEX_MODEL'] = CODEX_MODEL = mid or CODEX_MODEL
    if effort:
        os.environ['TFT_CODEX_EFFORT'] = CODEX_EFFORT = effort
    return None


def model_name():
    """The game's model as a --model spec ('claude:claude-opus-5-5', 'openrouter:openai/gpt-6-sol'), or None."""
    kind = _cli()[0]
    cli = 'claude-key' if os.environ.get('TFT_CLAUDE_AUTH') == 'key' else 'claude'
    return {'claude-cli': f'{cli}:{CLAUDE_MODEL}', 'claude': f'claude-api:{CLAUDE_MODEL}',
            'openrouter': f'{api_host()}:{OR_MODEL}', 'codex': f'codex:{CODEX_MODEL}'}.get(kind, kind)


def model_settings(kind=None):
    """(model, effort) the game's backend runs with - for the run log (a Codex game was logged as Claude's model
    and effort: release run 20261016-141957)."""
    kind = kind or _cli()[0]
    return {'openrouter': (OR_MODEL, OR_EFFORT), 'codex': (CODEX_MODEL, CODEX_EFFORT)}.get(kind, (CLAUDE_MODEL, CLAUDE_EFFORT))


def user_env(name):
    """A variable from the environment, or the Windows user environment (set since this terminal opened)."""
    k = os.environ.get(name)
    if not k and os.name == 'nt':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as h:
                k = winreg.QueryValueEx(h, name)[0]
        except OSError:
            k = None
    return k


def save_user_env(name, value):
    """Keep a key for later runs: the Windows user environment (as setx does), and this process."""
    os.environ[name] = value
    if os.name == 'nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment', 0, winreg.KEY_SET_VALUE) as h:
            winreg.SetValueEx(h, name, 0, winreg.REG_SZ, value)


def openrouter_key():
    return user_env('OPENROUTER_API_KEY')


def anthropic_key():
    return user_env('ANTHROPIC_API_KEY')


def check_anthropic_key(key):
    """Does the Anthropic API accept the key? -> why not, or None."""
    import urllib.error
    import urllib.request
    req = urllib.request.Request('https://api.anthropic.com/v1/models',
                                 headers={'x-api-key': key, 'anthropic-version': '2023-06-01'})
    try:
        urllib.request.urlopen(req, timeout=15).read()
    except urllib.error.HTTPError as e:
        return f'the Anthropic API did not accept the key (HTTP {e.code})'
    except OSError as e:
        return f'could not reach the Anthropic API to check the key: {e}'
    return None


TERMINAL_FLAGS = (getattr(__import__('subprocess'), 'CREATE_NEW_CONSOLE', 0) | 0x01000000      # CREATE_BREAKAWAY_FROM_JOB
                  | getattr(__import__('subprocess'), 'CREATE_NEW_PROCESS_GROUP', 0))


def cli_env():
    """The environment for the model's CLI: without the markers a parent Claude Code session
    leaves (games started from a Claude Code session made the model a "child session", whose
    transcript is not saved - the user's post-game discussion was lost, 2026-10-02)."""
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith('CLAUDE_CODE_') or k in ('CLAUDECODE', 'CLAUDE_AGENT_SDK_VERSION'))}
    env['CLAUDE_CODE_FORCE_SESSION_PERSISTENCE'] = '1'
    # questions come every 5-30 s: the 5-minute cache is enough, and its writes cost 5/8 of the 1-hour cache's,
    # which the CLI picks on a Claude login (experiments/token-cost: -23% of a game's cost)
    env['CLAUDE_CODE_PROMPT_CACHE_TTL'] = '5m'
    if os.environ.get('TFT_CLAUDE_AUTH') == 'login':
        env.pop('ANTHROPIC_API_KEY', None)
        env.pop('ANTHROPIC_AUTH_TOKEN', None)
    return env


_CLAUDE_EXE = []


def claude_cli():
    """Claude Code CLI bundled with the desktop app (uses the user's Claude login). Found once: the player asks
    every tick (planner_on), and a PATH search plus a glob there showed up in slow ticks up to 20 s."""
    if not _CLAUDE_EXE:
        found = _find_claude_cli()
        if not found:
            return None
        _CLAUDE_EXE.append(found)
    return _CLAUDE_EXE[0]


def _find_claude_cli():
    found = shutil.which('claude')
    if found:
        return found
    native = Path.home() / '.local' / 'bin' / 'claude.exe'     # `claude update` installs here (2.1.295 knows Haiku 5.5)
    if native.exists():
        return str(native)
    # <version>/claude.exe, or <version>/<build hash>/claude.exe since the app's 2.1.28x updates
    # (the old one-level glob found nothing and run21 played on fallback rules without a model)
    base = Path(os.environ.get('APPDATA', '')) / 'Claude' / 'claude-code'
    version = lambda p: next(([int(x) if x.isdigit() else 0 for x in d.name.split('.')]
                              for d in [p.parent, *p.parents] if d.parent == base), [0])
    exes = sorted(base.glob('**/claude.exe'), key=lambda p: (version(p), p.stat().st_mtime))
    return str(exes[-1]) if exes else None


_CODEX_EXE = []


def codex_cli():
    """The Codex CLI: on PATH (npm install -g @openai/codex), or the one the Codex app ships
    (%LOCALAPPDATA%/OpenAI/Codex/bin/<build>/codex.exe, newest build). Found once."""
    if not _CODEX_EXE:
        found = shutil.which('codex')
        if not found:
            base = Path(os.environ.get('LOCALAPPDATA', '')) / 'OpenAI' / 'Codex' / 'bin'
            exes = sorted(base.glob('*/codex.exe'), key=lambda p: p.stat().st_mtime)
            found = str(exes[-1]) if exes else None
        if not found:
            return None
        _CODEX_EXE.append(found)
    return _CODEX_EXE[0]


def codex_sandbox():
    """The Windows sandbox mode the user's Codex is set up with (config.toml [windows] sandbox): the game's
    calls skip the user's config (no plugins or MCP servers in a game), but without this setting the read-only
    sandbox refuses every shell command and the model cannot read its reference files."""
    return (_codex_config().get('windows') or {}).get('sandbox')


def codex_rollout_answer(thread_id, since):
    """The last answer Codex recorded in its own session file for this thread (sessions/YYYY/MM/DD/rollout-*-<thread
    id>.jsonl) - its turn's final agent message - if that turn completed after `since` (time.time()); else None."""
    if not thread_id:
        return None
    from datetime import datetime
    home = Path(os.environ.get('CODEX_HOME') or Path.home() / '.codex') / 'sessions'
    try:
        f = max(home.rglob(f'rollout-*-{thread_id}.jsonl'), key=lambda x: x.stat().st_mtime)
        lines = f.read_text(encoding='utf-8', errors='replace').splitlines()
    except (ValueError, OSError):
        return None
    for line in reversed(lines):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        p = r.get('payload') or {}
        if p.get('type') == 'task_complete':
            try:
                at = datetime.fromisoformat(str(r.get('timestamp')).replace('Z', '+00:00')).timestamp()
            except ValueError:
                return None
            return p.get('last_agent_message') if at >= since - 1 else None
    return None


def codex_service_tier():
    """The service tier the user's Codex is set up with ("priority" = Fast: about 2x speed for more of the plan's
    usage); the game's calls skip the user's config, so it is passed back in (2026-10-18: flex questions answered
    in 7-25 s with it)."""
    return _codex_config().get('service_tier')


def _codex_config():
    try:
        import tomllib
        home = Path(os.environ.get('CODEX_HOME') or Path.home() / '.codex')
        return tomllib.loads((home / 'config.toml').read_text(encoding='utf-8'))
    except Exception:
        return {}


REFERENCE = Path(__file__).resolve().parent.parent / 'data' / 'reference'


OPENING = """You are playing a live Teamfight Tactics game (Set <SET>, patch <PATCH>).
Your goal is the best expected final placement. You make every strategic decision in this game.

HOW THE SYSTEM WORKS
- Perception reads the game from the screen. Each decision gives you structured state plus a
  screenshot. Unit names come from text only (the shop card when we buy, the unit's info panel);
  a unit whose name has not been read yet is shown as "?". When the screenshot and the state
  disagree, strategy.notes is the place to say so.
- Shop slots hold champions or Wisps (non-champion cards; the state gives each Wisp's card text
  and cost). A Wisp is decided FIRST in the planning question (execution_plan.wisps). It comes as
  its own short question when a round runs your next_round plan instead of asking you, and when your
  "roll" (n times) brings one up; the plan runs on meanwhile and only the rolling waits for your
  answer. In a fast_rr, Wisps are bought on the spot from stage 4; before stage 4, only the first Wisp a
  roll brings up is bought. A Wisp you buy that rerolls the
  shop brings you the new shop at once. One Wisp can be bought per round (two with high Blossom); the
  shop offers a Wisp only when the gold held as the shop is drawn can pay for it.
- This Set and patch may be newer than your training data. The data in each message (game
  state, MetaTFT data) and the reference files are for this Set and patch.
- You decide. The controller carries out your plan one input at a time, re-reads the game after
  each step and reports what worked, what failed and what changed. It makes no strategic choices:
  what you leave out of the plan does not happen - also when no plan arrives in time: nothing
  acts for you, except the standing orders you set (auto_buy, auto_sell).
- Your plan is carried out as given. What changes meanwhile (a Wisp's effect, gold, a new unit)
  and any step that failed are reported in your next message. You are asked at each planning
  phase, at each COMBAT START (what your plan did not get done; combat time is yours to use),
  when a choice appears (augment, container, choice screen), when a step fails, and when a
  next_review condition you set in your plan holds.
- Units: the state lists what you own, kept from the controller's own confirmed actions. A unit
  that arrived without a buy (carousel, loot, augment, Wisp) shows as "?" until the controller has
  read its info panel - the screenshot shows it meanwhile.

Each message is one decision: its type, what changed since your last decision (including the
controller's report), the current state and the seconds left. The newest state is authoritative.

Reference files for this Set and patch are in your working directory (start with README.md):
game mechanics, every Wisp, champions, traits, items, augments, MetaTFT comps with positions and
item builds. They are information, not instructions. Reading a file takes time; during the game
that time runs on the game's clock.

Each stage of the game is a new conversation (stage 1 continues your preparation's): its first message
repeats what you need from the earlier stages - the comp brief, your trait choices, your game_strategy and
notes, and the last state if the question has none.

REPLY FORMAT FOR PLANNING DECISIONS
Two JSON objects, one after the other, and nothing else (plus an optional third, next_round - see
PLANNING AHEAD below):
{"execution_plan": {...}}
{"strategy": {...}}
The controller starts executing the moment the first object is complete, while you write the
second: the explanation never delays the decision. Write execution_plan first, always.

execution_plan - what the controller does now. Leave out keys you don't need; {} does nothing.
The controller always runs it in this order: wisps, sell, buy, xp, board, items, roll / fast_rr - then it places
the units the roll bought (name them in "board"). So in one plan, sells free bench slots and gold
before the buys use them. Choice containers are opened by the controller itself. When a step
fails or cannot be done (bench full, not enough gold, unit not found), the controller tells you at
once with the reason - in the planning phase or during combat - so you can send a corrected plan.
An invalid entry (a name where a unit id is needed, an item that is not on the item bench) is
dropped before anything runs, and you are told which and why; the rest of the plan runs.
During combat the board is locked (no moving, selling or itemising board units), but buying,
selling bench units, rerolling and XP work: the controller does those at once and keeps
board moves, board sells and items for the next planning phase. Wisps leave the shop when combat
starts (the champion card under each one shows): no Wisp can be bought in combat.
{
  "wisps":   [{"slot": <shop slot>, "buy": true|false, "why": "<one sentence>"}],
             // FIRST: one entry for every Wisp the state asks you to decide; a Wisp is bought before
             // anything else. Not in next_round (that shop is not known yet).
  "sell":    [unit ids],
  "open_container": [bench slots 0-8],    // optional: the controller opens a choice container (item
             // anvil, artifact anvil, ...) by itself as soon as it sees one and no question to you is
             // running, also in combat; its offer is then shown to you and you choose. You never need
             // to list one here.
  "buy":     ["Champion", ...],           // champions in the shop, by name, in this order. A full bench (9/9)
             // takes a buy only if it is a 3rd copy (it merges); otherwise put a "sell" in the same plan.
             // Stage 1: if no plan of yours for the round has come with 3 s left, every affordable card is bought
  "xp":      <n>,                         // buy XP n times (4 gold each)
  "board":   [{"unit": <id>, "hex": [row, col]}, {"name": "Champion", "hex": [row, col]}],
             // where units go: each listed unit is moved to that hex ("name" for units bought in this plan or
             // its roll, and for units still arriving - an augment's or a Wisp's: placed the moment they are on
             // the bench). Units you do not list stay where they are. The board holds at most `level` units: a
             // unit placed on an occupied hex swaps with the unit standing there (one drag; that unit takes the
             // placed unit's old spot, or its own listed hex if free); to take a unit off the board otherwise,
             // list it in "bench".
  "bench":   [{"unit": <id>, "slot": <bench slot 0-8>}],
             // put these units in these bench slots (from the bench or the board; done before "board"); a
             // unit already in that slot is moved to another free slot first.
  "link":    [{"from": <unit id> | [row, col], "to": <unit id>}],
             // press on the source, hover on the target ~1 s, release - the game's drag-and-hold link (the BFF
             // or its current rider to a new rider, Taric to his ally). Done after "board"; from: a board unit or a
             // board hex.
  "items":   [{"give": "<item on the item bench>", "to": <unit id>, "final": "Champion"},
              {"use": "<consumable on the item bench>", "on": <unit id>},
              {"reforge": <unit id>},
              {"transfer": "<item>", "from": <unit id>, "to": <unit id> | "bench"}],
             // ACTIONS, each once (a give whose item is no longer on the item bench but already on
             // that unit counts as done). give: two components on one unit build the completed item -
             // name each component; only what is on the item bench can be given. reforge: a Reforger
             // turns ALL the unit's items into random ones of a similar type. transfer: a Remover
             // takes ALL of the source's items off, so give EVERY item on the source a destination
             // (another unit, the same unit to put it back, or "bench"). Not during combat.
  "roll":    {"times": <n>, "buy": ["Champion", ...] | {"Champion": <star 1-3>, ...}, "sell_others": false},
             // reroll n times, buying the listed champions until each has its star level's copies on
             // board + bench (1 / 3 / 9 copies; a plain list means its auto_buy star, else 2). A Wisp in a
             // new shop stops the rolling and comes to you as its own short question; then the rest is rolled.
  "fast_rr": {"buy": ["Champion", ...] | {"Champion": <star>, ...}, "gold_floor": <g>,
              "stop_when": [{"any": ["Champion", ...], "star": <2|3>}], "sell_others": true},
             // a fast roll-down instead: reroll while gold >= g + 2, buying as above; stops early when every
             // stop_when entry has one of its champions with that star level's copies (merged or not).
             // From stage 4 every Wisp that shows up is bought on the spot, with no question.
             // sell_others (fast_rr: true unless you set false; roll: false unless you set true): before
             // the first roll, every BENCH unit outside buy, keep, board and auto_buy is sold (also in
             // combat). Rolling stops only at your own conditions (stop_when, gold_floor, times), or when
             // the game stops it: the bench is full and a wanted card cannot be bought, or the game does
             // not take a roll (you are told which).
  "identify": [{"unit": <id>, "name": "Champion", "star": <1-3>}, {"item_slot": <0-9>, "name": "<item>"}],
             // name a unit shown as "?" or an item-bench slot shown as unknown, read off the screenshot.
             // Where units stand is checked by the controller itself (trait panel, health bars).
  "keep":    [unit ids],                  // never sold by the controller
  "auto_buy": {"Champion": <star 1-3>, ...},
             // STANDING ORDER, kept all game until you change it: whenever the shop (also during a roll)
             // shows a listed champion and we own fewer copies than its star level needs (1 / 3 / 9), the
             // controller buys it at once - planning or combat, no gold reserve, no question. Adds to or
             // changes the list; 0 takes a champion off it. The state shows the list and what it bought.
  "auto_sell": true | false | ["Champion", ...],
             // STANDING ORDER, off until you turn it on: true = every BENCH unit whose champion is not on
             // auto_buy, and every spare copy beyond its auto_buy star level, is sold as soon as seen; a list =
             // only those champions. Never board units, units in your board/keep, or units whose identity
             // is unverified. false turns it off.
  "after_fight": {"win":  {"sell": [unit ids], "buy": ["Champion", ...]},
                  "lose": {"sell": [unit ids], "buy": ["Champion", ...]}},
             // orders for the moment THIS round's fight result shows (our HP drops, the streak flame
             // changes, or the +1 win gold arrives) - before the next round starts and pays interest on
             // the gold held then. Only the orders for the actual result run: its sells first (bench
             // units), then its buys from the shop on screen (one card per name; the shop is replaced
             // when the next planning phase opens). You are told what was done.
  "item_final": [{"unit": <id>, "item": "<item it holds>", "final": "Champion", "star": <1-3>}],
             // HOLDERS: every item has two - the unit holding it now (its current holder) and the champion it is
             // meant for (its final holder). A give names both: "to" is the current holder, "final" the final holder
             // (also for a component: the item it builds keeps it); add "final_star" (1-3, default 1) for the star the
             // final holder must have. A transfer changes only the current holder. item_final changes an item's final
             // holder ("final": null clears it). The moment we own an item's final holder at that star (its name read
             // off its panel) in a planning phase, the controller moves the item to it with no question to you: a
             // current holder that is not in the comp's final board is sold (its items return to the item bench, the
             // final holder takes its hex if the holder was fielded, and the item is given to the final holder); one
             // that is in it keeps its place - an Item Remover takes all its items off, the item goes to the final
             // holder and the others back onto it. The state lists every item's holders; you are told what moved.
  "gold_floor": <g>,  "gold_floor": <g>,  "gold_floor": <g>,                      // no spending below g, except explicit xp / roll
  "next_review": [{"plan_complete": true} | {"seconds_left": <n>} | {"gold_at_least": <n>} |
                  {"unit": "Champion", "star": <1-3>}]
             // optional: ask me again this planning phase as soon as any of these holds.
             // Without it you are asked again at the next planning phase (or a choice event).
}
Hexes: row 0 is the back row (next to the bench), row 3 the front row; columns 0-6 left to right.
Unit ids stay the same all game. Shop slots are valid only for the state they came with.

PLANNING AHEAD (optional, any round): a third object after strategy,
{"next_round": {...same keys as execution_plan...}}
is carried out the moment the NEXT planning phase opens, instead of asking you first: no thinking
time is spent there, execution starts at once. Without it you are asked at the start of the next
planning phase as usual. Name champions in "buy" (no shop slots: that shop is not known yet) and
use "name" in "board" for units not owned yet. When it is complete you are asked again with what
it did, unless you set its own "next_review" ([] = don't ask me this round). A newer next_round
replaces an older one; a choice or augment in between does not cancel it.

strategy - where you are going and why. Every reply must include it.
{
  "comps":     [{"name": "<MetaTFT comp name, exactly as in the briefing>", "p": 0.3}, ...],  // 1-3 comps
  "other":     0.4,
  "commit":    null,   // or "<MetaTFT comp name, exactly as in the briefing>": the comp you play from now on
                      // (COMMIT below)
  "why": {   // for the people watching: plain words, one short sentence per part, in this order.
             // Champion and item names, places like "front row, left" - no unit ids, no hex
             // numbers, no JSON words. "-" for a part that does nothing this round.
    "sell":       "<what you sell and why>",
    "buy":        "<what you buy and why>",
    "level":      "<XP and rolls: what and why>",
    "board":      "<who is fielded where and why - every round, also when nothing moves>",
    "items":      "<which items go on whom and why>",
    "containers": "<anvils or other bench containers: open now or keep, and why>",
    "next":       "<what you expect to do next round>"
  },
  "notes":     "<anything to remember, including doubts about perception>"
}
P = the probability that this comp is the FINAL board of this game. The listed comps plus "other" must sum
to 1. Until you commit, the overlay shows these probabilities.
COMMIT - "commit" names the comp you play from now on; null = not decided yet. A commit makes it this game's
target comp, as in a fixed-comp game: its MetaTFT page and target board hexes are sent to you, auto_buy is
set to its units (to 2-star), and the overlay shows it as TARGET COMP. Before a commit nothing is bought or
placed for any comp beyond what your plans say. After a commit you can still change comp: "pivot":
{"to": "<comp>", "why": "..."} in strategy.

Augment decisions and combat reviews give their own short reply format; there too the
decision comes first and the reason after it."""


AUTO_OPEN_TEXT = {
    'order': ("""The controller always runs it in this order: wisps, sell, buy, xp, board, items, roll / fast_rr - then it places
the units the roll bought (name them in "board"). So in one plan, sells free bench slots and gold
before the buys use them. Choice containers are opened by the controller itself.""",
              """The controller always runs it in this order: wisps, sell, buy, xp, board, items, roll / fast_rr - then it places
the units the roll bought (name them in "board") - and open_container. So in one plan, sells free bench
slots and gold before the buys use them. A container is opened first while 11 s or more are left in
the planning phase (you need ~9 s to choose from its offer), otherwise last."""),
    'open': ("""  "open_container": [bench slots 0-8],    // optional: the controller opens a choice container (item
             // anvil, artifact anvil, ...) by itself as soon as it sees one and no question to you is
             // running, also in combat; its offer is then shown to you and you choose. You never need
             // to list one here.""",
             """  "open_container": [bench slots 0-8],    // choice containers on the bench (the state lists them:
             // item anvil, artifact anvil, ...). Opening one sells it; its offer is then shown to
             // you and you choose. Opening now or later is your decision."""),
}


def auto_open_anvil():
    """Choice containers (anvils) on the bench: True = the controller opens them itself the moment it sees
    one and no question is running (default when the comp is fixed: the model only chooses among the
    offered options); False = the model is told and decides when to open (execution_plan.open_container)."""
    return forced_comp() is not None


def forced_comp():
    """TFT_FORCE_COMP (tools/play_comp.py): the one comp this game plays, or None."""
    name = os.environ.get('TFT_FORCE_COMP')
    return META.by_name(name) if name else None


FORCED_STRATEGY = """strategy - why. Every reply must include it.
{
  "why": {   // for the people watching: plain words, one short sentence per part, in this order.
             // Champion and item names, places like "front row, left" - no unit ids, no hex
             // numbers, no JSON words. "-" for a part that does nothing this round.
    "sell":       "<what you sell and why>",
    "buy":        "<what you buy and why>",
    "level":      "<XP and rolls: what and why>",
    "board":      "<who is fielded where and why - every round, also when nothing moves>",
    "items":      "<which items go on whom and why>",
    "containers": "<anvils or other bench containers: open now or keep, and why>",
    "next":       "<what you expect to do next round>"
  },
  "notes":     "<anything to remember, including doubts about perception>"
}

"""


PILOT_OPENING = """You make the judgement calls in a live Teamfight Tactics game (Set <SET>, patch <PATCH>).
A rule-based controller plays one comp this game: <COMP>.
Its final board at level 8: <UNITS>.
Traits of that board: <TRAITS>.
Item builds (MetaTFT): <BUILDS>.
The controller does the economy, the levelling (MetaTFT timings: <LEVELS>), buying the comp's units,
the level-8 roll-down, the board, positions and items.

You decide:
- augments: three cards are offered; you may reroll each card once, or take one;
- Wisps (non-champion shop cards): buy or skip.
Each question brings the game state, the options and their data, and a screenshot.

Reference files for this Set and patch are in your working directory (README.md first): augments,
Wisps, champions, traits, items, MetaTFT comps. The questions have a clock.

Reply with the decision as one JSON object first, then {"why": "<2-4 sentences>"}. Nothing else."""


def pilot_opening(pilot):
    """The standing brief in pilot mode: the comp, its board, traits and builds, the model's calls."""
    count = Counter(t for u in pilot.final for t in (CATALOG.unit(u).traits if CATALOG.unit(u) else ()))
    traits = ', '.join(f"{t} {n} (breakpoints {'/'.join(map(str, CATALOG.traits.get(t) or ()))})"
                       for t, n in count.most_common())
    builds = '; '.join(f"{u}: {', '.join(items)}" for u, items in pilot.builds.items()) or 'none'
    levels = ', '.join(f'{s[0]}-{s[1]}: {lv}' for s, lv in pilot.schedule)
    return (PILOT_OPENING.replace('<SET>', str(CATALOG_SET)).replace('<PATCH>', str(META.patch))
            .replace('<COMP>', pilot.comp.name).replace('<UNITS>', ', '.join(pilot.final))
            .replace('<TRAITS>', traits).replace('<BUILDS>', builds).replace('<LEVELS>', levels))


GAME_STRATEGY = """GAME STRATEGY - your strategy for this game, one paragraph in your own words. You write it while you
prepare (before the game); it is pinned on screen for the people watching (panel STRATEGY, bottom right)
and shown to you in every planning message. Send it again when your strategy changes (a new
direction, comp, tempo or item plan), and in every combat-start message (STRATEGY REWRITE: the whole
paragraph as it stands now, after what this game has shown), as one more object after strategy:
{"game_strategy": {<COMP>"text": "<the whole paragraph, at most about 400 characters>",
                   "why": "<what changed, one short sentence, or 'unchanged' (not in the first one)>"}}
Leave it out otherwise: the last one stays.<FLEX>

"""
# flex: the comp is chosen by strategy.commit (OPENING), not here
GAME_STRATEGY_FLEX = ('', '\nUntil you commit, the panel shows your comp probabilities; then it shows this paragraph.')


def opening_prompt():
    """The model's standing brief for the whole game (the user's design).
    Fixed comp: no comp probabilities or target to choose, and one line naming the comp."""
    text = OPENING
    if not auto_open_anvil():
        for k in ('order', 'open'):
            text = text.replace(AUTO_OPEN_TEXT[k][0], AUTO_OPEN_TEXT[k][1], 1)
    else:
        text = text.replace(AUTO_OPEN_TEXT['open'][0] + '\n', '', 1)
    comp = forced_comp()
    gs = GAME_STRATEGY.replace('<COMP>', '' if comp else GAME_STRATEGY_FLEX[0]).replace(
        '<FLEX>', '' if comp else GAME_STRATEGY_FLEX[1])
    i = text.index('Augment decisions and combat reviews')
    text = text[:i] + gs + text[i:]
    if comp:
        a = text.index('strategy - where you are going and why.')
        text = text[:a] + FORCED_STRATEGY + text[text.index('GAME STRATEGY - your strategy'):]
        text += (f"\n\nTarget comp for this game: {comp.name} ({comp.id}). auto_buy starts as the units of its "
                 f'target final board at 2 stars ({", ".join(comp.units)}); auto_sell starts off. Both can be '
                 f'changed at any time.')
    return text.replace('<SET>', str(CATALOG_SET)).replace('<PATCH>', str(META.patch))


SYSTEM_PROMPT = opening_prompt


class _Session:
    """The Claude Code CLI conversation of the current stage, kept open: each question is a new message in it,
    and the first question of a new stage number starts a new one with a digest (_stage_turnover). If the
    process dies, the next question starts a fresh session."""

    def __init__(self):
        self._proc = None
        self._results = None
        self._lock = threading.Lock()        # one question at a time
        self._write = threading.Lock()       # stdin: a question, or an interrupt while one is in flight
        self.cut = False                     # the reply in flight was interrupted (cut_preparation)
        self._partial = None                 # text chunk callback of the question in flight
        self.session_id = None               # of the last answer: lets a terminal resume it

    def _start(self, exe):
        import queue
        # the reference files are the working directory; the model may read/search them
        d = str(REFERENCE) if REFERENCE.exists() else tempfile.mkdtemp(prefix='tft-claude-')
        cmd = [exe, '-p', '--input-format', 'stream-json', '--output-format', 'stream-json', '--verbose',
               '--model', CLAUDE_MODEL, '--effort', CLAUDE_EFFORT, '--strict-mcp-config',
               '--system-prompt', opening_prompt(),
               '--tools', 'Read,Grep,Glob', '--allowedTools', 'Read,Grep,Glob',
               '--include-partial-messages']      # text as it is written: act on the first answer
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, cwd=d, env=cli_env(),
                                      creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self._results = queue.Queue()
        proc, results = self._proc, self._results

        def reader():
            for line in proc.stdout:
                if b'"text_delta"' in line:
                    try:
                        ev = json.loads(line.decode('utf-8', 'replace')).get('event') or {}
                        chunk = ev.get('delta', {}).get('text', '')
                        LIVE['text'] += chunk                 # the reply as it is written, for the overlay
                        if self._partial:
                            self._partial(chunk)
                    except Exception:
                        pass
                elif b'"type":"result"' in line:
                    try:
                        results.put(json.loads(line.decode('utf-8', 'replace')))
                    except ValueError:
                        pass
            results.put(None)                  # process ended

        threading.Thread(target=reader, daemon=True).start()

    def warm(self, exe):
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._start(exe)

    def close(self):
        """End the conversation's process: the next question starts a new conversation."""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
            self._proc = None

    def interrupt(self):
        """Cut the reply in flight short (the CLI's interrupt request): it ends at once as an error result,
        and the conversation goes on with what the model read so far."""
        with self._write:
            if self._proc is not None and self._proc.poll() is None:
                self.cut = True
                req = {'type': 'control_request', 'request_id': f'cut-{time.time():.3f}', 'request': {'subtype': 'interrupt'}}
                self._proc.stdin.write((json.dumps(req) + chr(10)).encode())
                self._proc.stdin.flush()

    def ask(self, exe, content, timeout=90, on_first=None, call_n=None):
        """on_first(obj): called (from the reader thread) with the reply's first
        complete JSON object as soon as it has been written - the decision - while
        the model is still writing the rest (the explanation)."""
        with self._lock:
            content, new = _stage_turnover(content, call_n)
            if new and self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()               # the last stage's conversation ends
            if new or self._proc is None or self._proc.poll() is not None:
                self._start(exe)
            self._partial = _first_object_watcher(on_first) if on_first else None
            LIVE.update(what=_question_line(content), t0=time.time(), text='', took=None)
            msg = json.dumps({'type': 'user', 'message': {'role': 'user', 'content': content}})
            self.cut = False
            try:
                with self._write:
                    self._proc.stdin.write((msg + chr(10)).encode())
                    self._proc.stdin.flush()
                r = self._results.get(timeout=timeout)
            except Exception as e:
                self._proc = None
                raise RuntimeError(f'session failed: {e}')
            if r is None:
                self._proc = None
                raise RuntimeError('session ended')
            self._partial = None
            LIVE['took'] = time.time() - LIVE['t0']
            self.session_id = r.get('session_id') or self.session_id      # also after an error: the conversation exists
            cut = self.cut and r.get('is_error')
            if r.get('is_error') and not cut:
                raise RuntimeError(r.get('result'))
            # turns > 1 means the model used its tools (read/searched the reference files)
            self.last_meta = {k: r.get(k) for k in ('num_turns', 'duration_ms', 'duration_api_ms', 'total_cost_usd',
                                                    'usage', 'modelUsage', 'stop_reason', 'session_id')}
            if CALL_DIR is not None:
                try:
                    (CALL_DIR / f'{call_n or 0:04d}-result.json').write_text(json.dumps(r, ensure_ascii=False), encoding='utf-8')
                except Exception:
                    pass
            self.session_id = r.get('session_id') or self.session_id
            return LIVE['text'] if cut else r.get('result', '')      # cut: the text written until then

    def open_terminal(self, exe, locked=True):
        """After the game: stop the headless process and resume this very conversation in a
        console window, so the model can be asked why it did what it did. Returns True if opened."""
        import contextlib
        with (self._lock if locked else contextlib.nullcontext()):     # F8: a question may hold the lock
            if not self.session_id:
                return False
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            self._proc = None
            d = str(REFERENCE) if REFERENCE.exists() else tempfile.gettempdir()
            # its own console, outside the launcher's job: stopping the launcher (or its time limit)
            # must not close the conversation the user is reading (run27's terminal was killed)
            cmd = [exe, '--resume', self.session_id, '--model', CLAUDE_MODEL,
                              '--effort', CLAUDE_EFFORT, '--strict-mcp-config',
                              '--system-prompt', opening_prompt(), '--tools', 'Read,Grep,Glob']
            try:
                subprocess.Popen(cmd, cwd=d, env=cli_env(), creationflags=TERMINAL_FLAGS)
            except OSError:              # the launcher's job forbids breakaway: a plain new console
                subprocess.Popen(cmd, cwd=d, env=cli_env(),
                                 creationflags=getattr(subprocess, 'CREATE_NEW_CONSOLE', 0))
            return True


_SESSION = _Session()


def json_objects(text, repair=True):
    """The complete top-level JSON objects in text, in order (strings respected). repair=False:
    only objects whose closing brace is in the text (a reply still streaming in must never have
    its unfinished object closed for it - run26 3-6: a plan cut to 2 of its 7 board units was
    installed, and the controller benched Varus, Kog'Maw and Shen).
    A reply that forgot closing brackets is repaired: an object that starts at the very start
    of a line while another is still open closes the open one first (run19: the plan line
    lacked its last brace, so every plan from 4-5 on was lost); the end of the text closes
    whatever is open (seen with GPT-6 Sol)."""
    out, stack, start, in_str, esc = [], [], None, False, False
    nl = chr(10)
    src = text or ''

    def add(chunk):
        try:
            out.append(json.loads(re.sub(r'//[^' + nl + r']*', '', chunk)))
        except ValueError:
            pass

    for i, ch in enumerate(src):
        if in_str:
            if esc:
                esc = False
            elif ch == chr(92):
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and stack:
            in_str = True
        elif repair and ch == '{' and stack and i > 0 and src[i - 1] == nl:
            # a new top-level object begins on its own line: close the one left open
            add(src[start:i].rstrip() + ''.join(reversed(stack)))
            stack, start = ['}'], i
        elif ch in '{[':
            if not stack:
                start = i
            if ch == '{' or stack:
                stack.append('}' if ch == '{' else ']')
        elif ch in '}]' and stack:
            stack.pop()
            if not stack:
                add(src[start:i + 1])
    if repair and stack and start is not None:
        add(src[start:].rstrip() + ''.join(reversed(stack)))
    return out


def _first_object_watcher(on_first):
    """Feed text chunks; calls on_first once, with the first complete JSON object."""
    buf, fired = [], [False]

    def feed(chunk):
        if fired[0] or not chunk:
            return
        buf.append(chunk)
        if '}' in chunk:
            # the deliberation comes first in a planning reply; the decision is the object after it
            objs = [o for o in json_objects(''.join(buf), repair=False)
                    if not (isinstance(o, dict) and set(o) == {'deliberation'})]
            if objs:
                fired[0] = True
                try:
                    on_first(objs[0])
                except Exception as e:
                    print('first-answer handler failed:', e)
    return feed


def _images(x):
    """One JPEG, a list of them, or None -> list."""
    if x is None:
        return []
    return [i for i in x if i] if isinstance(x, (list, tuple)) else [x]


PREP_HEAD = ('PREPARATION - sent as we enter the queue for the game. The queue takes about 1-3 minutes, then the '
             'game loads for about 1 minute; no game clock runs until then. When the game asks its first question '
             '(in round 1-2), this reply is cut off: the files you have read stay in the conversation (until stage 2, '
             'which starts a new one), what you have not written yet is lost.\n\n')
# what the preparation is for (the user, 2026-10-12): read everything on the comp, then write the game strategy
PREP_TASK_FIXED = ('\n\nPREPARATION TASK: read all the information on this comp in the reference files - its MetaTFT '
                   'page (metatft/comps/<ID>.md), each of its champions and their abilities (champions.md), its traits '
                   '(traits.md), the items of its builds (items.md) and the augments (augments.md). Then write your '
                   'strategy for this game as one paragraph: {"game_strategy": {"text": "..."}} (format in your '
                   'instructions). Topics: tempo, augments, items, board and the comp\'s mechanics - in your own words '
                   'and weight.')
PREP_TASK_FLEX = ('\n\nPREPARATION TASK: read the reference files (README.md lists them) together with the MetaTFT comp '
                  'ranking above. Then write your overall strategy for this game as one paragraph: {"game_strategy": '
                  '{"text": "..."}} (no "comp" yet; format in your instructions), and your comp probabilities: {"strategy": '
                  '{"comps": [{"name": "<MetaTFT comp>", "p": 0.3}, ...], "other": 0.4, "commit": null}} ("commit": a comp '
                  'name instead if you already commit to one).')


def briefing_text(before_queue=False):
    """The first message of every game: the rules, traits, items, the augment tier list and
    every MetaTFT comp with its exact stats. Read in full now; the per-comp pages come later,
    when a direction is chosen. Fixed comp: only that comp (see forced_briefing).
    before_queue: sent before we queue (prepare), with no game clock; otherwise the game is
    already running (the player was started into it)."""
    comp = forced_comp()
    if comp:
        return (PREP_HEAD + forced_briefing() + PREP_TASK_FIXED.replace('<ID>', comp.id) if before_queue
                else forced_briefing())
    ref = REFERENCE
    read = lambda f: (ref / f).read_text(encoding='utf-8') if (ref / f).exists() else f'({f} missing)'
    tiers = {}
    for n, a in sorted(META.augments.items()):
        if a.get('tier'):
            tiers.setdefault(a['tier'], []).append(n)
    aug = '\n'.join(f"{t}: {', '.join(tiers[t])}" for t in ('S', 'A', 'B', 'C') if t in tiers)
    comps = sorted((c for c in META.comps.values() if c.avg), key=lambda c: c.avg)
    rows = [f"- {c.name} [{c.id}]: avg place {c.avg:.2f}, pick {c.pick}, win {c.win:.1%}, top 4 {c.top4:.1%}, "
            f"{c.levelling}; units: {', '.join(c.units)}"
            if c.win is not None else f"- {c.name} [{c.id}]: avg place {c.avg:.2f}, {c.levelling}; units: {', '.join(c.units)}"
            for c in comps]
    return ((PREP_HEAD + 'BRIEFING - data for this game.\n\n' if before_queue else
             'BRIEFING - data for this game, sent while it runs.\n\n') +
            '=== GAME MECHANICS ===\n' + read('game-mechanics.md') + '\n\n'
            '=== TRAITS ===\n' + read('traits.md') + '\n\n'
            '=== ITEMS ===\n' + read('items.md') + '\n\n'
            f'=== AUGMENT TIER LIST (MetaTFT expert list, S best; patch {META.patch}) ===\n' + aug + '\n\n'
            f'=== METATFT COMPS (Ranked, Platinum+, last 3 days, patch {META.patch}; best average place first; '
            'pick = times played per lobby) ===\n' + '\n'.join(rows) + '\n\n'
            'Each comp also has a detailed MetaTFT page (positioning, levelling and rolls, carousel priority, '
            'unit and item stats, traits, augments). When a comp becomes your target, its page is sent to you '
            'as text. '
            'More text data: the reference files in your working directory (metatft/comps/<id>.md).' +
            (PREP_TASK_FLEX if before_queue else '\n\nReply with ONLY {"ready": true}.'))


_FORCED_TEXT = {}


def comp_curve_line(comp):
    """The comp's own levelling line, as MetaTFT text fetched for the briefing (cached)."""
    m = re.search(r'^Levelling[^\n]*$', _FORCED_TEXT.get(comp.id, ''), re.M)
    return m.group(0) if m else ''


def forced_briefing():
    """Fixed comp: one line and the transcription of the comp's three MetaTFT page screenshots
    (metatft_shots.comp_brief; the screenshots themselves are attached by the player)."""
    comp = forced_comp()
    try:
        # the transcription of the attached screenshots (taken together, refreshed after 48 h)
        from tft_player.metatft_shots import comp_brief
        text = ("The comp's MetaTFT page (tabs Options & Quick Start, Units & Items, Pro Augments and Tips) "
                "as text:\n\n" + comp_brief(comp.name))
    except Exception as e:
        print(f'MetaTFT comp transcription failed ({e}): screenshots only')
        text = ''
    _FORCED_TEXT[comp.id] = text
    from tft_player.trait_text import choices_text
    traits = [t for u in comp.units for t in (CATALOG.unit(u).traits if CATALOG.unit(u) else ())]
    choices = choices_text(traits)
    if choices:
        # what each option does, and the model's choice made now - at the choice screen it has seconds
        text += ('\n\nTrait choices on the target final board (the game shows only the names on the cards):\n' + choices +
                 '\nThe choice screen has a short timer, so the picks are asked now. End your reply with '
                 '{"trait_choices": {"<trait>": ["<first pick>", "<second pick>"]}} using the names on the cards; '
                 'they are shown to you again when the choice appears.')
    return f'Target comp for this game: {comp.name} ({comp.id}).' + (f'\n\n{text}' if text else '')


STOPPED = False          # set when we are eliminated: ask_model answers None at once
CALL_DIR = None          # set by the player: every model call's screenshots and raw result go here (benchmark data)
_CALL_N = [0]


def _save_text(n, name, text):
    """A diagnostic file next to call n's inputs (calls/NNNN-<name>)."""
    if CALL_DIR is None or not n:
        return
    try:
        (CALL_DIR / f'{n:04d}-{name}').write_text(text, encoding='utf-8')
    except OSError:
        pass


def _save_call_inputs(prompt, imgs):
    """Save the screenshots (and prompt) of a model call as calls/NNNN-*.jpg; returns the call number."""
    if CALL_DIR is None:
        return None
    try:
        _CALL_N[0] += 1
        n = _CALL_N[0]
        CALL_DIR.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(imgs if os.environ.get('TFT_NO_SCREENSHOTS') != '1' else ()):   # --no-screenshots
            (CALL_DIR / f'{n:04d}-{i}.jpg').write_bytes(img)
        (CALL_DIR / f'{n:04d}-prompt.txt').write_text(prompt, encoding='utf-8')
        return n
    except Exception:
        return None


def _claude_cli_call(exe, prompt, image_jpeg=None, on_first=None, timeout=90):
    """Claude through the Claude Code CLI (your Claude subscription), as a new
    message in the game's open conversation. The screenshot is an inline image
    block; a short system prompt and no tools keep answers at ~3-6s."""
    import base64
    content = []
    for img in _images(image_jpeg):
        content.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg',
                                                    'data': base64.standard_b64encode(img).decode()}})
    content.append({'type': 'text', 'text': prompt})
    n = _save_call_inputs(prompt, _images(image_jpeg))
    try:
        return _SESSION.ask(exe, content, timeout=timeout, on_first=on_first, call_n=n)
    finally:
        if n:
            _SESSION.last_meta = dict(getattr(_SESSION, 'last_meta', None) or {}, call=n)


class _ORSession:
    """ONE OpenRouter conversation per game (any model there, e.g. GPT-6 Sol), with
    the same system prompt, the screenshot on each question (only the latest kept
    in the history), streaming (on_first gets the decision as soon as it is
    written) and two tools to read/search the reference files."""

    TOOLS = [
        {'type': 'function', 'function': {
            'name': 'read_file', 'description': 'Read a reference file (path relative to the reference folder).',
            'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}},
        {'type': 'function', 'function': {
            'name': 'grep', 'description': 'Search the reference files for a regular expression; returns matching lines.',
            'parameters': {'type': 'object', 'properties': {'pattern': {'type': 'string'}}, 'required': ['pattern']}}},
    ]

    def __init__(self):
        self.messages = None
        self._lock = threading.Lock()

    def _tool(self, name, args):
        base = REFERENCE.resolve()
        if name == 'read_file':
            p = (base / str(args.get('path', ''))).resolve()
            if base not in p.parents and p != base or not p.is_file():
                return 'no such file'
            return p.read_text(encoding='utf-8')[:20000]
        if name == 'grep':
            try:
                rx = re.compile(str(args.get('pattern', '')), re.I)
            except re.error as e:
                return f'bad pattern: {e}'
            hits = []
            for f in sorted(base.rglob('*.md')):
                for line in f.read_text(encoding='utf-8').splitlines():
                    if rx.search(line):
                        hits.append(f'{f.relative_to(base).as_posix()}: {line[:300]}')
                        if len(hits) >= 60:
                            return '\n'.join(hits)
            return '\n'.join(hits) or 'no match'
        return 'unknown tool'

    def _stream(self, on_text):
        import urllib.error
        import urllib.request
        # max_tokens as on the Anthropic path: without it OpenRouter reserves the model's whole output length
        # against the key's credit and refuses the call (402) long before the credit is used up
        host = api_host()
        base, _, label = API_HOSTS[host]
        body = {'model': OR_MODEL, 'messages': self.messages, 'stream': True, 'tools': self.TOOLS}
        if host == 'openai':                       # OpenAI's names for the same two settings
            body['max_completion_tokens'] = 16000
            if OR_EFFORT:
                body['reasoning_effort'] = OR_EFFORT
        else:
            body['max_tokens'] = 16000
            if OR_EFFORT:
                body['reasoning'] = {'effort': OR_EFFORT}
        req = urllib.request.Request(base + '/chat/completions', json.dumps(body).encode(),
                                     {'Authorization': 'Bearer ' + api_key(),
                                      'Content-Type': 'application/json'})
        text, calls = [], {}
        try:
            r = urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f'{label} {e.code}: {e.read().decode("utf-8", "replace")[:300]}') from None
        with r:
            for raw in r:
                line = raw.decode('utf-8', 'replace').strip()
                if not line.startswith('data:') or line == 'data: [DONE]':
                    continue
                try:
                    ch = json.loads(line[5:]).get('choices') or [{}]
                except ValueError:
                    continue
                d = ch[0].get('delta') or {}
                if d.get('content'):
                    text.append(d['content'])
                    on_text(d['content'])
                for tc in d.get('tool_calls') or []:
                    c = calls.setdefault(tc.get('index', 0), {'id': '', 'name': '', 'args': ''})
                    c['id'] = tc.get('id') or c['id']
                    fn = tc.get('function') or {}
                    c['name'] += fn.get('name') or ''
                    c['args'] += fn.get('arguments') or ''
        return ''.join(text), [calls[k] for k in sorted(calls)]

    def ask(self, content, on_first=None):
        with self._lock:
            content, new = _stage_turnover(content)
            if self.messages is None or new:
                self.messages = [{'role': 'system', 'content': opening_prompt()}]
            # keep only the newest screenshot in the history (size and cost)
            for m in self.messages:
                if isinstance(m.get('content'), list):
                    m['content'] = [c for c in m['content'] if c.get('type') != 'image_url'] or ''
            self.messages.append({'role': 'user', 'content': content})
            feed = _first_object_watcher(on_first) if on_first else (lambda t: None)
            t0, turns = time.time(), 0
            while True:
                turns += 1
                text, calls = self._stream(feed)
                msg = {'role': 'assistant', 'content': text}
                if calls:
                    msg['tool_calls'] = [{'id': c['id'], 'type': 'function',
                                          'function': {'name': c['name'], 'arguments': c['args']}} for c in calls]
                self.messages.append(msg)
                if not calls or turns > 6:
                    self.last_meta = {'num_turns': turns, 'duration_ms': round((time.time() - t0) * 1000)}
                    return text
                for c in calls:
                    try:
                        args = json.loads(c['args'] or '{}')
                    except ValueError:
                        args = {}
                    self.messages.append({'role': 'tool', 'tool_call_id': c['id'], 'content': self._tool(c['name'], args)})


_OR = _ORSession()


class _CodexSession:
    """The Codex CLI conversation of the current stage (ChatGPT login). Each question is one `codex exec` run that
    resumes the conversation by its id (never "the last session": the user's own Codex sessions stay apart). The
    game's system prompt replaces Codex's own instructions; the user's config is not loaded (no plugins or MCP
    servers); the model reads the reference files with its shell in a read-only sandbox. Codex prints each message
    whole, not as it is written: the decision is acted on when its message is complete."""

    def __init__(self):
        self.thread_id = None
        self._proc = None
        self._lock = threading.Lock()
        self.cut = False
        self.session_id = None               # = thread_id: lets a terminal resume it
        self.last_meta = None
        self._instr = None

    def _base(self, exe):
        cmd = [exe, 'exec', '--json', '--ignore-user-config', '--ignore-rules', '--skip-git-repo-check',
               '-s', 'read-only', '-c', 'approval_policy="never"', '-m', CODEX_MODEL,
               '-c', f'model_reasoning_effort="{CODEX_EFFORT}"',
               '-c', f'model_instructions_file="{self._instr.as_posix()}"']
        sb = codex_sandbox()
        if sb:
            cmd += ['-c', f'windows.sandbox="{sb}"']
        tier = codex_service_tier()
        if tier:
            cmd += ['-c', f'service_tier="{tier}"']
        return cmd

    def close(self):
        with self._lock:
            self.thread_id = None

    def interrupt(self):
        """Cut the reply in flight short: its process ends; the conversation goes on from what it holds."""
        p = self._proc
        if p is not None and p.poll() is None:
            self.cut = True
            p.kill()

    def ask(self, exe, content, timeout=90, on_first=None, call_n=None):
        with self._lock:
            content, new = _stage_turnover(content, call_n)
            if new or self.thread_id is None or self._instr is None:
                self.thread_id = None
                self._instr = Path(tempfile.gettempdir()) / f'tft-codex-instructions-{os.getpid()}.md'
                self._instr.write_text(opening_prompt(), encoding='utf-8')
            text = next((c.get('text') or '' for c in content if c.get('type') == 'text'), '')
            imgs = [c['jpeg'] for c in content if c.get('type') == 'jpeg']
            d = Path(tempfile.mkdtemp(prefix='tft-codex-'))
            files = []
            for i, img in enumerate(imgs):
                f = d / f'screen-{i}.jpg'
                f.write_bytes(img)
                files.append(str(f))
            cmd = self._base(exe)
            if self.thread_id:
                cmd += ['resume', self.thread_id]
            for f in files:
                cmd += ['-i', f]
            cmd += ['-']
            feed = _first_object_watcher(on_first) if on_first else None
            LIVE.update(what=_question_line(content), t0=time.time(), text='', took=None)
            self.cut, said, usage, turns, err, proc = False, [], None, 1, None, None
            raw, began = [], time.time()
            cwd = str(REFERENCE) if REFERENCE.exists() else str(d)
            try:
                self._proc = proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                     stderr=subprocess.DEVNULL, cwd=cwd,
                                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                proc.stdin.write(text.encode('utf-8'))
                proc.stdin.close()
                timer = threading.Timer(timeout, proc.kill)
                timer.start()
                try:
                    for line in proc.stdout:
                        if len(raw) < 400:
                            raw.append(line.decode('utf-8', 'replace')[:2000])
                        try:
                            ev = json.loads(line.decode('utf-8', 'replace'))
                        except ValueError:
                            continue
                        kind = ev.get('type')
                        item = ev.get('item') or {}
                        if kind == 'thread.started':
                            self.thread_id = self.session_id = ev.get('thread_id') or self.thread_id
                        elif kind == 'item.completed' and item.get('type') == 'agent_message':
                            chunk = (item.get('text') or '') + '\n'
                            said.append(chunk)
                            LIVE['text'] += chunk
                            if feed:
                                feed(chunk)
                        elif kind == 'item.completed' and item.get('type') == 'command_execution':
                            turns += 1                 # a file read or search
                        elif kind == 'turn.completed':
                            usage = ev.get('usage')
                        elif kind in ('turn.failed', 'error'):
                            err = (ev.get('error') or {}).get('message') or ev.get('message') or err
                    proc.wait(timeout=10)
                finally:
                    timer.cancel()
            except OSError as e:
                if not self.cut:             # cut while its question was still being written: a cut, not a fault
                    raise RuntimeError(f'Codex did not start: {e}')
            finally:
                self._proc = None
                for f in files:
                    try:
                        os.remove(f)
                    except OSError:
                        pass
                try:
                    d.rmdir()
                except OSError:
                    pass
            LIVE['took'] = time.time() - LIVE['t0']
            answer = ''.join(said).strip()
            if not answer and not self.cut:
                # run 20261018-092528 call 9: the turn completed with an answer in Codex's own session file, but none
                # reached us - take it from there, and keep what we did read for the diagnosis
                _save_text(call_n, 'codex-stdout.txt', ''.join(raw))
                answer = codex_rollout_answer(self.thread_id, began) or ''
                if not answer:
                    raise RuntimeError(err or f'Codex ended without an answer (exit code {proc.returncode})')
                print(f'Codex answer read from its session file (none on its output): {answer[:80]}', flush=True)
            self.last_meta = {'num_turns': turns, 'duration_ms': round(LIVE['took'] * 1000), 'usage': usage,
                              'session_id': self.thread_id}
            if CALL_DIR is not None:
                try:
                    (CALL_DIR / f'{call_n or 0:04d}-result.json').write_text(
                        json.dumps(dict(self.last_meta, result=answer), ensure_ascii=False), encoding='utf-8')
                except Exception:
                    pass
            return answer

    def open_terminal(self, exe, locked=True):
        """After the game: resume this conversation in a console window (the user's own Codex settings)."""
        import contextlib
        with (self._lock if locked else contextlib.nullcontext()):
            if not self.thread_id:
                return False
            cmd = [exe, 'resume', self.thread_id, '-s', 'read-only']
            d = str(REFERENCE) if REFERENCE.exists() else tempfile.gettempdir()
            try:
                subprocess.Popen(cmd, cwd=d, creationflags=TERMINAL_FLAGS)
            except OSError:
                subprocess.Popen(cmd, cwd=d, creationflags=getattr(subprocess, 'CREATE_NEW_CONSOLE', 0))
            return True


_CODEX = _CodexSession()


def _codex_call(exe, prompt, image_jpeg=None, on_first=None, timeout=90):
    content = [{'type': 'text', 'text': prompt}] + [{'type': 'jpeg', 'jpeg': img} for img in _images(image_jpeg)]
    n = _save_call_inputs(prompt, _images(image_jpeg))
    try:
        return _CODEX.ask(exe, content, timeout=timeout, on_first=on_first, call_n=n)
    finally:
        if n:
            _CODEX.last_meta = dict(_CODEX.last_meta or {}, call=n)


def active_session():
    """The conversation object of the game's model backend (its last_meta, open_terminal)."""
    kind = _cli()[0]
    return _OR if kind == 'openrouter' else _CODEX if kind == 'codex' else _SESSION


def _openrouter_call(prompt, image_jpeg=None, on_first=None):
    import base64
    content = [{'type': 'text', 'text': prompt}]
    for img in _images(image_jpeg):
        content.append({'type': 'image_url', 'image_url': {
            'url': 'data:image/jpeg;base64,' + base64.standard_b64encode(img).decode()}})
    n = _save_call_inputs(prompt, _images(image_jpeg))        # benchmark data: the screenshots and prompt of every call
    try:
        return _OR.ask(content, on_first)
    finally:
        if n:
            _OR.last_meta = dict(getattr(_OR, 'last_meta', None) or {}, call=n)


# the question in flight, for the overlay: what is asked, since when, the reply as it streams in, and how
# long the last one took (None while one is running)
LIVE = {'what': None, 't0': 0.0, 'text': '', 'took': None}


def _question_line(content):
    """The first line of a question's text: what the model is being asked ('DECISION: augment, stage 4-2')."""
    try:
        text = next((c.get('text') for c in content if isinstance(c, dict) and c.get('type') == 'text'), '')             if isinstance(content, list) else str(content)
    except Exception:
        text = ''
    return next((ln.strip() for ln in (text or '').splitlines() if ln.strip()), '')[:140]


_PRIORITY = [0]              # urgent questions waiting or running (a choice screen is open and timed)
_RESERVED = [0]              # the model is held for an urgent question about to be asked (a container just opened)


def reserve_model():
    """Hold the model for an urgent question that is about to be asked: True only if nothing is running or
    reserved, and from then on no other question starts until release_model(). The caller then does what
    produces the question (opens the container) - it never opens and then waits for the model."""
    if _RESERVED[0] or _PRIORITY[0] or model_busy():
        return False
    _RESERVED[0] = 1
    return True


def release_model():
    _RESERVED[0] = 0


def plain(text):
    """The roster's log shorthand (Varus*@b3, Kha'Zix*@h33[Gargoyle]) in the words the state uses."""
    text = re.sub(r'(\*+)@b(\d)', r'\1 at bench slot \2', text)
    return re.sub(r'(\*+)@h(\d)(\d)', r'\1 at hex (\2,\3)', text)


def ask_model(prompt, image_jpeg=None, on_first=None, timeout=None, priority=False):
    """One question to the game's model (blocking). None when no model is connected.
    on_first(obj) gets the reply's first JSON object (the decision) as soon as it is
    written; the call itself returns the whole reply (decision + explanation).
    priority: an urgent question (a choice screen is open and the game's own timer runs): other
    questions wait for it instead of taking the conversation first."""
    if STOPPED:
        return None              # eliminated: nothing is asked any more
    from tft_player.game import PAUSED
    if PAUSED.is_set():
        return None              # F7: no questions while paused; on continue the player asks again
    prompt = plain(prompt)
    _remember(prompt)
    if _COMMIT['brief'] and not priority:
        # flex: the comp just committed to - its brief goes ahead of the next question (a timed choice waits for
        # the one after); later stages' digests carry it in place of the flex briefing
        prompt, _COMMIT['brief'] = _COMMIT['brief'] + COMMIT_SEP + prompt, None
    cut_preparation("the game's first question came")
    if _PREP['cut'] and not _PREP['told']:
        _PREP['told'] = True
        prompt = (f"(Your preparation reply was cut off: {_PREP['cut']}. What you had not written is lost; if your "
                  f"game_strategy is not written yet, send it in this reply.)\n\n" + prompt)
    if priority:
        _PRIORITY[0] += 1
        try:
            return _remember(answer=_ask(prompt, image_jpeg, on_first, timeout))
        finally:
            _PRIORITY[0] -= 1
    end = time.time() + 25
    while (_PRIORITY[0] > 0 or _RESERVED[0]) and time.time() < end:
        time.sleep(0.05)
    return _remember(answer=_ask(prompt, image_jpeg, on_first, timeout))


def _ask(prompt, image_jpeg, on_first, timeout):
    kind, exe = _cli()
    if kind == 'claude-cli':
        return _claude_cli_call(exe, prompt, image_jpeg, on_first, **({'timeout': timeout} if timeout else {}))
    if kind == 'claude':
        return _claude_call(prompt, image_jpeg)
    if kind == 'openrouter':
        return _openrouter_call(prompt, image_jpeg, on_first)
    if kind == 'codex':
        return _codex_call(exe, prompt, image_jpeg, on_first, **({'timeout': timeout} if timeout else {}))
    return None


def model_busy():
    """True while a question to the model is in flight (the conversation answers one at a time)."""
    try:
        return _SESSION._lock.locked() or _OR._lock.locked() or _CODEX._lock.locked()
    except Exception:
        return False


_PREP = {'running': False, 'cut': None, 'told': True}   # the preparation in flight; why it was cut off

# One conversation per stage (experiments/token-cost, 2026-10-16): the game's first conversation (preparation and
# stage 1) lasts until stage 2; each new stage number opens a new conversation whose first message carries what
# the earlier ones held - the comp brief, the trait picks, the model's game_strategy and notes, the last state.
# About half a game's model cost; in replays the decisions stayed within the one-conversation game's own variation.
_STAGE = {'now': None, 'conv': None}     # the game's stage (from the player); the stage number the conversation began in
_MEMO = {}                               # what a new stage's conversation is told: brief, traits, strategy, notes, ...
_STATE_BLOCK = re.compile(r'\nSTATE\n.*?(?=\n\nReply with|\Z)', re.S)


_COMMIT = {'brief': None}        # flex: the committed comp's brief, waiting for the next question
COMMIT_SEP = '\n\n=== THE QUESTION ===\n'


def commit_brief(text):
    """Flex: the model committed to a comp (now forced_comp()). Its brief (forced_briefing: the comp's MetaTFT page
    and trait choices) goes ahead of the next question and replaces the flex briefing in later digests."""
    _MEMO['brief'] = text
    _MEMO.pop('comps', None)
    _COMMIT['brief'] = text


def set_stage(stage):
    """The game's stage as the player reads it ('3-2'); a question in a new stage number opens a new conversation."""
    _STAGE['now'] = stage


def _remember(prompt=None, answer=None):
    """Keep what a later stage's first message repeats: from questions the briefing, the last state and the target
    comp's MetaTFT data (flex); from answers the trait picks, game_strategy, notes and comp probabilities."""
    if prompt:
        if prompt.startswith((PREP_HEAD, 'BRIEFING', 'Target comp for this game')):
            brief = prompt.replace(PREP_HEAD, '')
            for cut in ('\n\nTrait choices on the target final board', '\n\nPREPARATION TASK', '\n\nReply with ONLY'):
                brief = brief[:brief.find(cut)] if cut in brief else brief
            _MEMO['brief'] = brief.strip()
        if prompt.startswith('MetaTFT data of the target comp,'):
            _MEMO['page'] = prompt.split('\n\nDECISION:', 1)[0]
        m = _STATE_BLOCK.search(prompt)
        if m:
            _MEMO['state'] = m.group(0).strip()
    for o in json_objects(answer or '') if isinstance(answer, str) else ():
        if not isinstance(o, dict):
            continue
        if isinstance(o.get('trait_choices'), dict):
            _MEMO['traits'] = o['trait_choices']
        if isinstance(o.get('game_strategy'), dict) and o['game_strategy'].get('text'):
            _MEMO['strategy'] = o['game_strategy']['text']
        st = o.get('strategy')
        if isinstance(st, dict):
            _MEMO['notes'] = st.get('notes') or _MEMO.get('notes')
            if isinstance(st.get('comps'), list):
                _MEMO['comps'] = {'comps': st['comps'], 'other': st.get('other'), 'target': st.get('target')}
    return answer


def stage_digest(stage, with_state=True):
    """The first message of a new stage's conversation, before its first question."""
    m = _MEMO
    parts = [f'NEW CONVERSATION FOR STAGE {stage} - each stage of this game is played in a new conversation, to keep '
             'messages short. The earlier stages are not repeated here; this is what you need from them.',
             m.get('brief') or '(no briefing was sent)',
             f"Your trait choices (made in your preparation): {json.dumps(m.get('traits'))}\n"
             f"Your game_strategy as it stands (pinned on screen): {m.get('strategy')}\n"
             f"Your notes in your last reply: {m.get('notes')}"
             + (f"\nYour comp probabilities in your last reply: {json.dumps(m['comps'])}" if m.get('comps') else '')]
    if m.get('page'):
        parts.append(m['page'])
    if with_state and m.get('state'):
        parts.append(f"The last game state you were sent (before this question):\n{m['state']}")
    parts.append('The reference files are in your working directory as before (README.md first).')
    return '\n\n'.join(parts) + '\n\n=== THE QUESTION ===\n'


def _stage_turnover(content, call_n=None):
    """(content, True) with the digest in front of the question's text when it is the first question of a new stage
    number - the caller then starts a new conversation; otherwise (content, False). Called inside the session's
    lock, so no other question can slip in between."""
    now = _STAGE['now']
    if not now or _PREP['running'] or not isinstance(content, list):
        return content, False
    n = str(now).split('-')[0]
    if _STAGE['conv'] in (None, n):
        _STAGE['conv'] = n
        return content, False
    _STAGE['conv'] = n
    text = next((c.get('text') or '' for c in content if c.get('type') == 'text'), '')
    if _MEMO.get('brief') and text.startswith(_MEMO['brief'] + COMMIT_SEP):
        # a commit brief ahead of this question: the new conversation's digest carries it already
        content = [dict(c, text=c['text'][len(_MEMO['brief'] + COMMIT_SEP):])
                   if c.get('type') == 'text' and c.get('text') == text else c for c in content]
        text = text[len(_MEMO['brief'] + COMMIT_SEP):]
    digest = stage_digest(n, with_state='\nSTATE\n' not in text)
    if CALL_DIR is not None and call_n:
        try:
            (CALL_DIR / f'{call_n:04d}-digest.txt').write_text(digest, encoding='utf-8')
        except OSError:
            pass
    out, done = [], False
    for c in content:
        if c.get('type') == 'text' and not done:
            c, done = dict(c, text=digest + c['text']), True
        out.append(c)
    return out, True


def new_game(call_dir):
    """Before each game: a new conversation (the last game's process ends), questions are answered again
    (STOPPED is set at elimination) and every call's files go to this game's run folder."""
    global STOPPED, CALL_DIR
    STOPPED, CALL_DIR, _CALL_N[0] = False, call_dir, 0
    _PREP.update(running=False, cut=None, told=True)
    _STAGE.update(now=None, conv=None)
    _MEMO.clear()
    _COMMIT['brief'] = None
    _SESSION.close()
    _CODEX.close()
    _OR.messages = None


PREP_WAIT = 3600         # s: only a hung CLI reaches this - the game's first question cuts the preparation off


def prepare():
    """The briefing, sent as we enter the queue (no game clock yet). It runs until the model is done or the
    game's first question comes - then it is cut off (cut_preparation) and the conversation goes on with
    what the model read. Returns {'prompt', 'answer', 'error', 'cut', 'seconds', 't', 'meta'}; answer None
    when it failed."""
    t0, text, error, answer = time.time(), briefing_text(before_queue=True), None, None
    _PREP.update(running=True, cut=None, told=False)
    _remember(plain(text))
    try:
        answer = _remember(answer=_ask(plain(text), None, None, PREP_WAIT))
    except Exception as e:
        error = str(e)
    finally:
        _PREP['running'] = False
    if answer is None and not error:
        error = 'no answer (no model connected)'
    meta = getattr(active_session(), 'last_meta', None)
    return {'prompt': text, 'answer': answer, 'error': error, 'cut': _PREP['cut'], 'seconds': time.time() - t0,
            't': time.time(), 'meta': meta}


def cut_preparation(why):
    """End the preparation reply now (the game needs the model, or its time is up); a no-op when none runs."""
    if _PREP['running'] and not _PREP['cut']:
        _PREP['cut'] = why
        if _cli()[0] == 'claude-cli':
            _SESSION.interrupt()
        elif _cli()[0] == 'codex':
            _CODEX.interrupt()


def warm_up():
    """Open the game's conversation now so the first question is fast too."""
    kind, exe = _cli()
    if kind == 'claude-cli':
        threading.Thread(target=_SESSION.warm, args=(exe,), daemon=True).start()


CLAUDE_MODEL = os.environ.get('TFT_CLAUDE_MODEL', 'claude-opus-5-5')
CLAUDE_EFFORT = os.environ.get('TFT_CLAUDE_EFFORT', 'low')        # latency matters: low by default
CODEX_MODEL = os.environ.get('TFT_CODEX_MODEL', 'gpt-6.1-sol')
CODEX_EFFORT = os.environ.get('TFT_CODEX_EFFORT', 'high')


def _claude_call(prompt, image_jpeg=None):
    """One request to Claude with the current screenshot attached. Returns the
    answer text. Server-side refusal fallbacks are on (a TFT screenshot should
    never trip a classifier, but a decline then completes on a fallback model
    instead of stalling the game)."""
    import base64
    import anthropic
    client = anthropic.Anthropic(max_retries=1, timeout=90.0)
    content = []
    if image_jpeg is not None:
        content.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg',
                                                    'data': base64.standard_b64encode(image_jpeg).decode()}})
    content.append({'type': 'text', 'text': prompt})
    resp = client.beta.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        thinking={'type': 'adaptive'},
        output_config={'effort': CLAUDE_EFFORT},
        betas=['server-side-fallback-2026-07-01'],
        extra_body={'fallbacks': 'default'},
        system=opening_prompt(),
        messages=[{'role': 'user', 'content': content}],
    )
    if resp.stop_reason == 'refusal':
        raise RuntimeError(f'refused: {getattr(resp, "stop_details", None)}')
    return ''.join(b.text for b in resp.content if b.type == 'text')


def augment_text(name, full=False):
    """The augment's card text (game data; the screenshot shows the card too)."""
    n = None if full else 220
    if name in META.augments:
        return META.augments[name].get('desc', '')[:n]
    entries = CATALOG.augments.get(name) or []
    desc = entries[0].get('description', '') if entries else ''
    return re.sub(r'@\w+@', 'N', desc)[:n]


def keyword_augment(names, roster):
    """Last resort when an augment is unknown to the meta snapshot."""
    traits = Counter(t for u in roster.units if u.info for t in u.info.traits)
    def score(n):
        d = augment_text(n).lower() + ' ' + (n or '').lower()
        s = 0.0
        for t, c in traits.items():
            if t.lower() in d:
                s += 2 + c
        for kw, v in (('gold', 1.5), ('item', 1.2), ('component', 1.2), ('emblem', 1.0),
                      ('health', 0.8), ('attack speed', 0.8), ('ability power', 0.6),
                      ('reroll', 0.8), ('xp', 0.8)):
            if kw in d:
                s += v
        if re.search(r'after \d+|in \d+ rounds|at stage [4-7]', d):
            s -= 1.0            # delayed payoff
        return s
    scores = [score(n) for n in names]
    return max(range(len(names)), key=lambda i: scores[i])


def stage_key(stage):
    try:
        a, b = str(stage).split('-')
        return int(a), int(b)
    except ValueError:
        return None


def local_augment(names, roster, stage, rerollable, plan):
    """Default decision: ('pick', i) or ('reroll', i), plus a reason."""
    known = [n in META.augments for n in names]
    if not META.available or not any(known):
        return ('pick', keyword_augment(names, roster)), 'keywords'
    cands = META.candidates(roster, prefer=plan.comp)
    w, rest = META.comp_weights(cands, stage_key(stage))
    # never reroll off a misread: only when all three names are recognised
    decision, vals = META.augment_decision(names, rerollable if all(known) else [], w, rest)
    return decision, 'meta ' + '/'.join(f'{v:.2f}' for v in vals)


# the augment answer format (the user, 2026-10-12): each card's COMBAT and ECONOMY value in numbers, then the
# decision; the example's card texts are copied from the cards on screen (run 20261010-235723, stage 4-2)
AUGMENT_EXPLAIN = (
    'Before deciding, explain every card on the screen, using the numbers printed on the card:\n'
    'COMBAT: the exact bonuses on your current board and on the {final}.\n'
    'ECONOMY: what it gives now and later.\n'
    'VERDICT: one short sentence - why you take this card, or why not (shown on screen under the card).\n'
    'Write "none" for a part where the card gives nothing.\n\n'
    'Example of the format, from an earlier game (its board and gold are not yours):\n\n'
    'Card 1 (middle): Bronze For Life II - "Your team gains 2.5% Damage Amp and 3 Armor and Magic Resist for '
    'each Bronze-tier trait."\n'
    'COMBAT:\n'
    '  Current board: +12.5% Damage Amp, +15 Armor and MR on all 7 units.\n'
    '  Target final board: +15% Damage Amp, +18 Armor and MR on all 7 units.\n'
    'ECONOMY: none.\n'
    'VERDICT: not taken - Invested++ gives more in that game.\n\n'
    'Card 0 (left): Invested++ - "Gain 45 gold. At the start of each round, gain 1 Shop reroll for every 10 '
    'gold above 50 (max 80 gold)."\n'
    'COMBAT: none.\n'
    'ECONOMY:\n'
    '  Now: +45 gold (16 -> 61).\n'
    '  Later: +4 interest a round, and 1-3 free rerolls a round while holding 60-80 gold.\n'
    'VERDICT: taken - 45 gold now and free rerolls while holding 60+.\n\n')


def card_sections(text):
    """{card index: its explanation block ('Card 1 (middle): ... VERDICT: ...')} from an augment answer."""
    parts = re.split(r'(?m)^(?=[ *#]*Card \d\s*\((?:left|middle|right)\))', text or '')
    out = {}
    for part in parts:
        m = re.match(r'[ *#]*Card (\d)', part)
        if m and 0 <= int(m.group(1)) <= 2:
            out[int(m.group(1))] = re.split(r'(?m)^\s*\{', part)[0].strip()      # the JSON answer is not part of it
    return out


class Coach:
    def __init__(self, brain, log=print, model=None, on_call=None, on_answer=None):
        self.brain, self.log = brain, log
        self.on_call = on_call          # (kind, prompt, raw_answer, latency) for the live view / log
        self.on_answer = on_answer      # the parsed answer (e.g. its comp probabilities)
        self.model = model or os.environ.get('TFT_COACH_MODEL') or 'auto'
        self.kind, self.exe = _cli()
        self.busy = False
        self.pending_augment = None       # augment names the current call is about
        self.answers = []
        self.finished = 0                 # augment calls ended (answered, unusable or failed)
        # this stage's card explanations so far and the cards the last answer rerolled: after a reroll only the new
        # card is explained again (the user, 2026-10-16)
        self.aug_memory = {'stage': None, 'cards': {}, 'reroll': []}

    @property
    def available(self):
        return self.exe is not None

    def ask(self, scr, roster, augments=None, rerollable=(), default=None, image=None, note=None):
        """An augment decision, as a message in the game's conversation: the three
        augments as displayed, and which to reroll (or which to take). note: why the
        question comes again (the last answer could not be carried out)."""
        if not self.available or self.busy or not augments:
            return False
        self._image = image
        # no card names: the model reads the cards from the screenshot, by position
        cards = ('The attached screenshot shows the augment choice: three cards, 0 = left, 1 = middle, '
                 '2 = right. Read the cards from the screenshot.')
        forced = forced_comp()
        why = '"why": "<one sentence>"'
        if len(rerollable) == len(augments):
            ask = ('Choose the ONE card you keep. The other two are then rerolled (free) and you choose again '
                   'from the new three.' + ('' if forced else ' Also choose your path now: P for 1-3 comps '
                   '(P = the probability it is the final board of this game; comps + other = 1), and "commit": '
                   'a comp name if you commit to it now (COMMIT in your instructions), else null.'))
            decision = '{"keep": <0-2>, "name": "<the kept card\'s name as shown>", ' + why + '}'
        elif rerollable:
            # run23: "Rerolls are used" was said while the kept card still had its reroll
            ask = (f'Cards you can still reroll (each once): {", ".join(str(i) for i in rerollable)}'
                   + (' - the card you kept still has its own reroll' if len(augments) - len(rerollable) == 2 else '')
                   + '. Reroll one of them, or take a card. Taking a card while another one still has its '
                   'reroll: that reroll is used first (it is free) and you are asked again with the new card.')
            decision = ('{"reroll": [<card>]} or {"pick": <0-2>, "name": "<the card\'s name as shown>", '
                        + why + '}')
        else:
            ask = 'No rerolls left. Which augment do you take?'
            decision = '{"pick": <0-2>, "name": "<the card\'s name as shown>", ' + why + '}'
        # the explanation comes first, as plain text: the decision object is still the reply's first JSON
        # object, so on_first applies it the moment it is written
        explain = AUGMENT_EXPLAIN.format(final='target final board' if forced else 'final board you are building toward')
        mem = getattr(self, 'aug_memory', None) or {'stage': None, 'cards': {}, 'reroll': []}
        if mem['stage'] != scr.stage:
            self.aug_memory = mem = {'stage': scr.stage, 'cards': {}, 'reroll': []}
        new = [i for i in mem['reroll'] if 0 <= i < len(augments)]
        kept = [i for i in range(len(augments)) if i not in new and i in mem['cards']]
        if new and kept:
            # after a reroll only the new card(s) are explained: the unchanged cards' explanations stand (the user,
            # 2026-10-16 - the model wrote all three again)
            names = lambda ix: ' and '.join(f'card {i}' for i in ix)
            explain = explain.replace('explain every card on the screen',
                                      f'explain only the new {names(new)} (just rerolled)', 1)
            explain = (f'{names(kept).capitalize()} did not change: your explanation of '
                       f'{"it" if len(kept) == 1 else "them"} from the last question stands - do not write it again:\n\n'
                       + '\n\n'.join(mem['cards'][i] for i in kept) + '\n\n' + explain)
        ask += '\n\n' + explain
        if forced:                                  # fixed comp: no path to choose, no P
            ask += f'Then end your reply with one JSON object:\n{decision}'
        else:
            ask += ('Then end your reply with two JSON objects:\n' + decision + '\nthen {"comps": [{"name": '
                    '"<MetaTFT comp>", "p": 0.3}, ...], "other": 0.4, "commit": null, '
                    '"why": "<3-5 sentences>"}')
        if forced:
            traits = Counter(t for u in forced.units for t in (CATALOG.unit(u).traits if CATALOG.unit(u) else ()))
            ask = (f'Target final board of {forced.name} (MetaTFT): {", ".join(forced.units)}; its traits: '
                   + ', '.join(f'{t} {n}' for t, n in traits.most_common()) + '.\n' + ask)
        prompt = (f'DECISION: augment, stage {scr.stage} (~{scr.timer}s left).\n' + (f'{note}\n' if note else '') +
                  f'{cards}\n\n{ask}')
        self.busy = True
        self.pending_augment = augments
        self._kind = 'augment'
        threading.Thread(target=self._run, args=(prompt, augments), daemon=True).start()
        return True

    def _run(self, prompt, augments):
        t0 = time.time()
        try:
            if self.kind == 'claude':
                text = _claude_call(prompt + '\n\nThe attached image is the live game screen right now.',
                                    getattr(self, '_image', None))
            elif self.kind in ('claude-cli', 'openrouter', 'codex'):
                # the pick/reroll is applied the moment its object is written, after the per-card explanation
                text = ask_model(prompt + '\n\nThe attached image is the live game screen right now.',
                                 getattr(self, '_image', None), on_first=lambda obj: self._apply(obj, augments))
            else:
              with tempfile.TemporaryDirectory(prefix='tft-coach-', ignore_cleanup_errors=True) as d:
                if self.kind == 'cursor':
                    cmd = [self.exe, '--print', '--output-format', 'json', '--workspace', d,
                           '--trust', '--mode', 'ask'] + (['--model', self.model] if self.model else [])
                else:
                    cmd = [self.exe, 'exec', '--skip-git-repo-check', '--sandbox', 'read-only', '-']
                out = subprocess.run(cmd, input=prompt.encode(), capture_output=True, cwd=d,
                                     timeout=60, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                text = out.stdout.decode('utf-8', 'replace')
                if out.returncode != 0:
                    raise RuntimeError(out.stderr.decode('utf-8', 'replace').strip()[-300:])
            try:
                text = json.loads(text).get('result', text)
            except Exception:
                pass
            ans = {}
            for obj in json_objects(text):
                if isinstance(obj, dict):
                    ans.update(obj)
            if self._kind == 'augment':
                mem = self.__dict__.setdefault('aug_memory', {'stage': None, 'cards': {}, 'reroll': []})
                mem['cards'].update(card_sections(text))
                aug = ans.get('augment') if isinstance(ans.get('augment'), dict) else ans
                if isinstance(aug.get('keep'), int):
                    mem['reroll'] = [i for i in range(len(augments)) if i != aug['keep']]
                else:
                    r = aug.get('reroll')
                    mem['reroll'] = [r] if isinstance(r, int) else [i for i in r or [] if isinstance(i, int)]
            self._apply(ans, augments)
            if self.on_answer:
                self.on_answer(ans)
            self.answers.append((time.time() - t0, ans))
            self.log(f'[coach {time.time() - t0:.0f}s] ' + json.dumps(ans, ensure_ascii=True))
            if self.on_call:
                self.on_call(self._kind, prompt, text, time.time() - t0)
        except Exception as e:
            self.log(f'[coach failed after {time.time() - t0:.0f}s] {e}')
            if self.on_call:
                self.on_call(self._kind, prompt, f'FAILED: {e}', time.time() - t0)
        finally:
            self.finished += 1
            self.busy = False

    def _apply(self, ans, augments):
        p = self.brain.plan
        aug = ans.get('augment') if isinstance(ans.get('augment'), dict) else ans
        if isinstance(aug.get('keep'), int) and augments and 0 <= aug['keep'] < len(augments):
            aug = {'reroll': [i for i in range(len(augments)) if i != aug['keep']]}
        if augments and self.pending_augment == augments:
            pick, reroll = aug.get('pick'), aug.get('reroll')
            reroll = [reroll] if isinstance(reroll, int) else reroll if isinstance(reroll, list) else []
            reroll = [i for i in reroll if isinstance(i, int) and 0 <= i < len(augments)]
            if reroll:
                p.augment_reroll = reroll
            elif isinstance(pick, int) and 0 <= pick < len(augments):
                p.augment_pick = pick
            if isinstance(aug.get('name'), str):
                p.augment_name = aug['name']        # the card as the model read it
