"""Play one comp, with the default (original) player - one model conversation per stage.

The system prompt is the usual one with no comp probabilities, plus one line naming the
comp; the briefing is that line, the comp's MetaTFT page as short text with its numbers,
and screenshots of its page tabs. The target is fixed for the whole game.

The model makes the decisions (round plans, augments, Wisps, items) and may write the next
round's plan ahead (next_round): it runs the moment that planning phase opens, with no model call.

    python tools/play_comp.py                                  Lunar Aphelios Nidalee, one game
    python tools/play_comp.py --comp "Flora Fatalis Azir"   by exact name (or MetaTFT number - numbers
                                                               change when MetaTFT reclusters, e.g. 425013 -> 426003)
    python tools/play_comp.py --list                           the comps and their numbers
    python tools/play_comp.py --comp flex                      no fixed comp: the model picks among the MetaTFT comps
    python tools/play_comp.py --comp random                    a random S or A tier comp (MetaTFT's default list)
    --model claude | claude:<model id> | codex[:<model id>] | openrouter:<model id>
                                                               the model (default since 2026-10-18: codex:gpt-6-luna,
                                                               the ChatGPT login through the Codex CLI - the user's
                                                               test model; claude: Claude Code, claude-opus-5-5)
    --effort low | medium | high                               how much it reasons per call (default high;
                                                               medium for the default gpt-6-luna)
Other arguments are passed to play_match.py. tools/play.py is the menu in front of this.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

args = sys.argv[1:]
if '--list' in args:
    from tft_player.meta import META
    for c in sorted(META.comps.values(), key=lambda c: c.avg):
        print(f'{c.id}  avg {c.avg:.2f}  {c.name}')
    sys.exit()
comp = 'Lunar Aphelios Nidalee'                  # by name: MetaTFT renumbers its comps (425002 -> 426004, 2026-10-12)
if '--comp' in args:
    i = args.index('--comp')
    comp = args[i + 1]
    del args[i:i + 2]
if comp.lower() == 'random':
    from tft_player.meta import META
    pick = META.random_top()
    if pick is None:
        sys.exit('no S or A tier comp in the MetaTFT data (run tools/fetch_meta.py)')
    comp = pick.name
    print(f'random S/A comp: {pick.name} (tier {pick.tier}, average place {pick.avg:.2f})', flush=True)
flex = comp.lower() == 'flex'
if flex:
    os.environ.pop('TFT_FORCE_COMP', None)
else:
    os.environ['TFT_FORCE_COMP'] = comp

def option(name, default):
    if name not in args:
        return default
    i = args.index(name)
    value = args[i + 1]
    del args[i:i + 2]
    return value


# the test model (the user, 2026-10-18: use the ChatGPT subscription; gpt-6-luna chose flex comps as well as
# gpt-6.1-sol, deciding in 7 s at medium effort - experiments/README.md section 8). At high effort its live answers
# grew to 60-100 s (run 20261018-092528: every plan after its round), so the test default is medium.
DEFAULT_MODEL, DEFAULT_EFFORT = 'codex:gpt-6-luna', 'medium'
model = option('--model', DEFAULT_MODEL)      # claude, claude:<model id>, codex[:<id>], openrouter:<model id>
# high: the model reasons through each decision (the user, 2026-10-11 - at medium it "barely thought" and reacted
# to each state); TFT_CLAUDE_EFFORT / TFT_OR_EFFORT override
effort = option('--effort', os.environ.get('TFT_OR_EFFORT' if model.startswith(('openrouter', 'openai')) else
                                          'TFT_CODEX_EFFORT' if model.startswith(('codex', 'chatgpt')) else
                                          'TFT_CLAUDE_EFFORT', DEFAULT_EFFORT if model == DEFAULT_MODEL else 'high'))
os.environ.setdefault('TFT_CLAUDE_MODEL', 'claude-opus-5-5')
os.environ['TFT_TEST_OPEN_CONTAINERS'] = '1'          # test games: open containers when seen (exercises the choice path)

from tft_player.coach import _cli, forced_comp, model_name, use_model  # noqa: E402

why = use_model(model, effort)
if why:
    sys.exit(f'no model: {why} - not starting a game')
print(f'model: {model_name()}, effort {effort} ({_cli()[1]})', flush=True)
c = forced_comp()
if flex:
    print('flex: the model picks among the MetaTFT comps during the game', flush=True)
elif not c:
    sys.exit(f'no MetaTFT comp named {comp!r}')
else:
    print(f'forced comp: {c.name} ({c.id}); units: {", ".join(c.units)}', flush=True)
sys.argv = [sys.argv[0]] + args
from tools.play_match import main  # noqa: E402

main()
# the game is over and its record printed: end now - a leftover non-daemon thread (live page, capture, overlay)
# kept the process alive for hours after the game (2026-10-09, twice)
sys.stdout.flush()
os._exit(0)
