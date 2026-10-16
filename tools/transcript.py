"""The model conversation of one game as a readable transcript.

    python tools/transcript.py runs/<run> [--until 2-1] [--out transcript.md]

Every model call is shown as the orchestrator's message (sent at answer time
minus latency) and the model's reply, with game-clock and wall-clock stamps.
--until S stops after the first review (combat start) of stage S.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--until')
    ap.add_argument('--out')
    args = ap.parse_args()
    from tft_player.coach import CLAUDE_EFFORT, CLAUDE_MODEL, opening_prompt
    rows = [json.loads(l) for l in open(Path(args.run) / 'log.jsonl', encoding='utf-8')]
    t0 = rows[0]['t']
    sess = next((r for r in rows if r.get('ev') == 'model_session'), None)
    if sess and sess.get('backend') == 'openrouter' and str(sess.get('model', '')).startswith('claude'):
        sess['model'], sess['effort'] = 'openai/gpt-6-sol', 'low'   # runs before 2026-09-27 logged the wrong name
    model = (f"{sess['model']}, effort {sess['effort']}, via {sess['backend']}" if sess else
             f'{CLAUDE_MODEL}, effort {CLAUDE_EFFORT} (from current settings: not logged in this run)')
    system = sess['system_prompt'] if sess else '(not logged in this run)'
    out = [f'# Model conversation - {Path(args.run).name}', '',
           f'Model: {model} - one conversation per stage', '',
           '**System prompt:**', '', system, '']
    clock = lambda t: f"+{t - t0:6.1f}s  ({time.strftime('%H:%M:%S', time.localtime(t))})"
    for r in rows:
        if r.get('ev') == 'round':
            out += [f'---', f'### Round {r["stage"]}  {clock(r["t"])}', f'Screen: `{r["screen"]}`', '']
        if r.get('ev') != 'model_call':
            continue
        sent = r['t'] - r['latency']
        out += [f'**orchestrator -> model** [{r["kind"]}, stage {r["stage"]}]  {clock(sent)}', '',
                '```', r['prompt'].strip(), '```', '',
                f'**AI model ({sess["model"] if sess else CLAUDE_MODEL})** [{r["kind"]}]  {clock(r["t"])}  latency {r["latency"]}s', '',
                '```', (r['answer'] or '').strip(), '```', '']
        if args.until and r['kind'] == 'review' and r['stage'] == args.until:
            break
    text = '\n'.join(out)
    if args.out:
        Path(args.out).write_text(text, encoding='utf-8')
        print(f'{args.out}: {text.count("orchestrator ->")} model calls')
    else:
        print(text)


if __name__ == '__main__':
    main()
