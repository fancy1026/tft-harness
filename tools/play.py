"""Play TFT with the model: pick a comp on metatft.com/comps (or flex), confirm it, and the game is queued and played.

    python tools/play.py                     MetaTFT data updated, then type a comp's name from metatft.com/comps
    python tools/play.py flex                flex: the model picks among the MetaTFT comps during the game
    python tools/play.py random              a random S or A tier comp
    python tools/play.py vanguard rengar     a comp by its name on metatft.com/comps (words in any order)
    python tools/play.py 426004              a comp by its MetaTFT number
    python tools/play.py ... --yes           no confirmation question
    python tools/play.py ... --model openrouter:openai/gpt-6-sol    the model, without the menu: claude[:<id>]
                                             (Claude login), claude-key[:<id>] (Anthropic API key),
                                             codex[:<id>] (ChatGPT login), openai:<id> or openrouter:<id>
Without --model it asks which model plays: the Claude subscription (Claude Code), an API key - Anthropic,
OpenAI or OpenRouter, told apart by the key - asked for the first time and kept in the Windows user environment,
or the ChatGPT subscription (the Codex app or Codex CLI, logged in).
The chosen comp is shown (units, items, levels) for confirmation, then its MetaTFT page is read into the
comp's prompt file (data/metatft_shots/<comp>/brief.txt, with the page screenshots) before the game starts.
Options after the choice go to play_comp.py / play_match.py: --games 3, --ranked, --no-screenshots ...
"""
import os
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tft_player.meta import META  # noqa: E402
from tft_player.metatft_shots import URL  # noqa: E402


def ask(text):
    try:
        return input(text).strip()
    except EOFError:
        sys.exit('no choice made')


def playable():
    """The comps metatft.com lists by default. Its situational ones (an emblem in most of their games, or a hero
    augment) are hidden there and not offered here: a fixed comp is played from the first round, without them."""
    return [c for c in META.comps.values() if not c.situational]


def by_place():
    return sorted(playable(), key=lambda c: c.avg or 9)


def update():
    """MetaTFT data no older than the game's own refresh allows (play_match.META_MAX_AGE_H)."""
    from tools.play_match import update_meta
    print('MetaTFT data: checking ...', flush=True)
    said = update_meta()
    if not META.comps:
        sys.exit(f'no MetaTFT data ({said}) - check the internet connection and try again')
    print(f'MetaTFT data: patch {META.patch}, {len(playable())} comps - {said}\n', flush=True)


# an API key tells its provider by how it starts
KEYS = [('claude-key', 'sk-ant-', 'ANTHROPIC_API_KEY', 'Anthropic'),
        ('openrouter', 'sk-or-', 'OPENROUTER_API_KEY', 'OpenRouter'),
        ('openai', 'sk-', 'OPENAI_API_KEY', 'OpenAI')]


def key_kind(key):
    return next((k for k in KEYS if key.startswith(k[1])), None)


def choose_model():
    """Which model plays: asked every start, nothing picked for the user. 1: Claude Code on the user's Claude
    login. 2: an API key - Anthropic, OpenAI or OpenRouter, told apart by how the key starts; a saved one can be
    picked, a new one is pasted (not shown) and, if wanted, kept for later runs (Windows user environment, as setx
    does). 3: the Codex CLI on the user's ChatGPT login. -> a --model spec that use_model accepted."""
    import getpass
    from tft_player.coach import CLAUDE_MODEL, CODEX_MODEL, save_user_env, use_model, user_env
    print('Which model plays?')
    print('  1  Claude subscription   Claude Code, installed and logged in')
    print('  2  API key               an Anthropic (sk-ant-...), OpenAI (sk-...) or OpenRouter (sk-or-...) key')
    print('  3  ChatGPT subscription  the Codex app (or Codex CLI), installed and logged in')
    refused = set()
    while True:
        got = ask('Model [1-3]: ')
        if got == '1':
            kind, typed = 'claude', False
        elif got == '3':
            kind, typed = 'codex', False
        elif got == '2':
            saved = [k for k in KEYS if k[0] not in refused and user_env(k[2])]
            kind = None
            if saved:
                for i, k in enumerate(saved, 1):
                    print(f'  {chr(96 + i)}  your saved {k[3]} key')
                pick = ask(f'Key [{"/".join(chr(97 + i) for i in range(len(saved)))}, Enter = paste a new one]: ').lower()
                if len(pick) == 1 and 0 <= ord(pick) - 97 < len(saved):
                    kind, typed = saved[ord(pick) - 97][0], False
            if kind is None:
                try:
                    key = getpass.getpass('Paste your API key (hidden): ').strip()
                except EOFError:
                    sys.exit('no key given')
                k = key_kind(key)
                if not k:
                    print('  not an Anthropic, OpenAI or OpenRouter key\n')
                    continue
                kind, typed = k[0], True
                os.environ[k[2]] = key
                refused.discard(kind)
        else:
            continue
        if kind == 'openrouter':
            mid = ask('OpenRouter model id, as on openrouter.ai/models (e.g. openai/gpt-6-sol): ')
        elif kind == 'openai':
            mid = ask('OpenAI model id, as on platform.openai.com/docs/models (e.g. gpt-6-sol): ')
        elif kind == 'codex':
            mid = ask(f'ChatGPT model [Enter = {CODEX_MODEL}]: ') or CODEX_MODEL
        else:
            mid = ask(f'Claude model [Enter = {CLAUDE_MODEL}]: ') or CLAUDE_MODEL
        spec = f'{kind}:{mid}'
        why = use_model(spec)
        if why:
            print(f'  {why}\n')
            if 'not accept' in why or 'not set' in why:
                refused.add(kind)                     # a refused key is not offered again
            continue
        var = next((k[2] for k in KEYS if k[0] == kind), None)
        if typed and ask('Keep this key for next time? [Enter = yes, n = no]: ').lower() != 'n':
            save_user_env(var, os.environ[var])
        print(f'Model: {spec}\n')
        return spec


def show_list():
    print(f'MetaTFT comps, patch {META.patch} (Platinum+, last 3 days, best average place first)\n')
    for i, c in enumerate(by_place(), 1):
        print(f'  {i:>2}  {c.tier}  {c.name:<34} avg ' + (f'{c.avg:.2f}' if c.avg else '?'))
    print()


def resolve(text):
    """The comp for what the user typed, or None after saying why."""
    found = META.search(text)
    hits = [c for c in found if not c.situational]
    if not hits and found:
        print(' / '.join(f'{c.name} is a situational comp on MetaTFT ({c.situational}), hidden there by default '
                         f'and not offered here' for c in found) + ' - pick another')
        return None
    if len(hits) == 1:
        return hits[0]
    if hits:
        print(f'{text!r} fits {len(hits)} comps: ' + ', '.join(c.name for c in hits) + ' - type the whole name')
    else:
        near = sorted((c for c in playable() if set(text.lower().split()) & set(c.name.lower().split())),
                      key=lambda c: c.avg or 9)
        print(f'no MetaTFT comp named {text!r}' + (
            '; close: ' + ', '.join(c.name for c in near) if near else ' - "list" shows them all'))
    return None


def choose():
    """A comp the user types, checked against the MetaTFT data, or 'flex'."""
    while True:
        got = ask('Comp: ')
        if not got:
            continue
        if got.lower() == 'list':
            show_list()
        elif got.lower() == 'flex':
            return 'flex'
        elif got.lower() == 'random':
            c = META.random_top()
            if c:
                return c
        elif got.isdigit() and 1 <= int(got) <= len(playable()):      # a line number from "list"
            return by_place()[int(got) - 1]
        else:
            c = resolve(got)
            if c:
                return c


def describe(c):
    """The comp as the MetaTFT data has it: units with stars, item builds, level timings."""
    g = c.guide or {}
    stars = g.get('stars') or {}
    avg = f'{c.avg:.2f}' if c.avg else '?'
    out = [f'{c.name}  (MetaTFT {c.id}, tier {c.tier}, average place {avg} over {c.games or 0:,} games, '
           f'patch {META.patch})',
           '  Units:  ' + ', '.join(f'{n} {stars.get(n, 1)}*' + (' (carry)' if n in c.carries else '')
                                    for n in c.units)]
    builds = [(n, b[0]['items']) for n, b in c.builds.items() if b]
    for i, (n, items) in enumerate(builds):
        out.append(('  Items:  ' if i == 0 else '          ') + f'{n}: ' + ', '.join(items))
    levels = g.get('levels') or []
    if levels:
        out.append('  Levels: ' + ', '.join(f'{lv} at {stage}' for stage, lv in levels) +
                   (f'  ({c.levelling})' if c.levelling else ''))
    return '\n'.join(out)


def confirm(c):
    print('\n' + describe(c) + '\n')
    return ask('Play this comp? [Enter = yes, n = choose another]: ').lower() in ('', 'y', 'yes')


def prepare(c):
    """The comp's prompt file: its MetaTFT page (three tabs) as screenshots and as text, read by a headless
    Chrome/Edge. A game without it would brief the model on a comp it can't see, so no page, no game."""
    from tft_player.metatft_shots import _paths, comp_pages
    print(f"Reading the MetaTFT page of {c.name} ...", flush=True)
    t0 = time.time()
    try:
        comp_pages(c.name)              # cached for metatft_shots.MAX_AGE_H
    except Exception as e:
        sys.exit(f'could not read the MetaTFT page of {c.name}: {e} - not starting the game')
    _, shots, brief = _paths(c.name)
    print(f'Comp prompt file: {brief.relative_to(ROOT)} (+ {len(shots)} page screenshots, {time.time() - t0:.0f}s)')


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    args = sys.argv[1:]
    yes = '--yes' in args
    args = [a for a in args if a != '--yes']
    k = next((i for i, a in enumerate(args) if a.startswith('--')), len(args))
    words, rest = ' '.join(args[:k]), args[k:]
    if '--model' in rest:                   # checked before the comp is chosen: a wrong name fails at once
        from tft_player.coach import use_model
        i = rest.index('--model')
        why = use_model(rest[i + 1]) if i + 1 < len(rest) else 'name the model after --model'
        if why:
            sys.exit(f'--model: {why}')
    else:
        rest += ['--model', choose_model()]
    update()
    told = False
    while True:
        if not words:
            if not told:
                print(f'Pick a comp on {URL} and type its name as shown there (words in any order, case '
                      f'ignored),\ne.g. "Lunar Aphelios Nidalee". "list" shows all comps here; "random" picks a random S or A tier comp; "flex" lets the model '
                      f'pick during the game.\n')
                told = True
            choice = choose()
        elif words.lower() == 'flex':
            choice = 'flex'
        elif words.lower() == 'random':
            choice = META.random_top() or sys.exit('no S or A tier comp in the MetaTFT data')
        else:
            choice = resolve(words) or sys.exit(1)
        if choice == 'flex' or yes or confirm(choice):
            break
        words = ''
        print()
    if choice != 'flex':
        prepare(choice)
    print('\nPlaying: ' + ('flex' if choice == 'flex' else f'{choice.name}'), flush=True)
    print('F7  pause / continue: while paused the bot sends no input and asks the model nothing; it keeps\n'
          '    watching, and on continue it plans again from the game as it is then. Do not change units while paused.\n'
          'F8  stop: all input stops and the bot closes at once. The match goes on - you can play it yourself.\n',
          flush=True)
    # by name: MetaTFT renumbers its comps each patch (425002 -> 426004 on 2026-10-12)
    sys.argv = [str(ROOT / 'tools' / 'play_comp.py'), '--comp', 'flex' if choice == 'flex' else choice.name, *rest]
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
