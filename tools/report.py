"""Post-game report: round by round, what the player saw, did and why.

    python tools/report.py runs/<timestamp>        -> runs/<timestamp>/report.html

Per round: the board at round start and at the fight, every agent's
reasoning at those moments, the actions with their reasons and outcomes,
identifications, augment/carousel/loot decisions, and every model call with
its full prompt and answer.
"""
import html
import json
import sys
from pathlib import Path

CSS = """
:root{--bg:#f6f7f9;--panel:#fff;--line:#e3e6ec;--text:#1b1f27;--dim:#5d667a;--acc:#9a7418;--ok:#1f8a58;--bad:#b5532a}
@media (prefers-color-scheme: dark){:root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--text:#e6e8ee;--dim:#8b93a7;--acc:#d8b25a;--ok:#5fbf8f;--bad:#e0875a}}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,Segoe UI,sans-serif}
main{max-width:1200px;margin:0 auto;padding:16px}
h1{font-size:20px}h2{font-size:18px;color:var(--acc);margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:10px}
figure{margin:0}figure img{width:100%;border-radius:8px;border:1px solid var(--line)}figcaption{color:var(--dim);font-size:12px}
.box{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin:8px 0}
.box b{color:var(--acc)}table{border-collapse:collapse;width:100%}td{padding:3px 6px;border-top:1px solid var(--line);vertical-align:top}
.ok{color:var(--ok)}.bad{color:var(--bad)}pre{white-space:pre-wrap;word-break:break-word;font:12px/1.4 ui-monospace,Consolas,monospace}
details summary{cursor:pointer;color:var(--dim)}
"""


def esc(x):
    return html.escape(str(x if x is not None else ''))


def agents_box(title, agents):
    if not agents:
        return ''
    rows = ''.join(f'<tr><td><b>{esc(k)}</b></td><td>{esc(v)}</td></tr>' for k, v in agents.items())
    return f'<div class=box><div>{esc(title)}</div><table>{rows}</table></div>'


def main():
    run = Path(sys.argv[1])
    events = [json.loads(line) for line in (run / 'log.jsonl').open(encoding='utf-8')]
    rounds, order = {}, []
    for e in events:
        st = e.get('stage') or 'pre'
        if st not in rounds:
            rounds[st] = []
            order.append(st)
        rounds[st].append(e)
    out = [f'<!doctype html><meta charset=utf-8><title>TFT run {esc(run.name)}</title><style>{CSS}</style><main>',
           f'<h1>TFT run {esc(run.name)}</h1>']
    for st in order:
        evs = rounds[st]
        out.append(f'<h2>{esc(st)}</h2><div class=row>')
        for name, cap in ((f'{st}-start.jpg', 'round start'), (f'{st}-fight.jpg', 'board it fought with')):
            if (run / name).exists():
                out.append(f'<figure><img src="{esc(name)}" loading=lazy><figcaption>{cap}</figcaption></figure>')
        out.append('</div>')
        for e in evs:
            if e['ev'] == 'round':
                out.append(f"<div class=box><b>read</b> {esc(e.get('screen'))}<br><b>tracked</b> {esc(e.get('roster'))}</div>")
                out.append(agents_box('agents at round start', e.get('agents')))
        acts = [e for e in evs if e['ev'] in ('action', 'identify', 'augment', 'augment_options',
                                              'augment_reroll', 'carousel', 'loot', 'menu_closed', 'surrender')]
        if acts:
            rows = []
            for e in acts:
                if e['ev'] == 'action':
                    ok = '<span class=ok>ok</span>' if e.get('ok') else '<span class=bad>not confirmed</span>'
                    rows.append(f"<tr><td>t={esc(e.get('timer'))}</td><td>{esc(e.get('act'))}</td><td>{ok}</td>"
                                f"<td>g={esc(e.get('gold'))} L{esc(e.get('level'))}</td></tr>")
                else:
                    detail = {k: v for k, v in e.items() if k not in ('ev', 't', 'stage')}
                    rows.append(f"<tr><td>{esc(e['ev'])}</td><td colspan=3>{esc(json.dumps(detail, ensure_ascii=False)[:400])}</td></tr>")
            out.append(f"<div class=box><div>actions and events</div><table>{''.join(rows)}</table></div>")
        for e in evs:
            if e['ev'] == 'model_call':
                out.append(f"<div class=box><b>model ({esc(e.get('kind'))}, {esc(e.get('latency'))}s)</b>"
                           f"<pre>{esc(e.get('answer'))}</pre><details><summary>full prompt</summary>"
                           f"<pre>{esc(e.get('prompt'))}</pre></details></div>")
            if e['ev'] == 'fight':
                out.append(agents_box(f"agents when the fight started (gold {e.get('gold')}, level {e.get('level')}, "
                                      f"hp {e.get('hp')})", e.get('agents')))
    out.append('</main>')
    dst = run / 'report.html'
    dst.write_text('\n'.join(out), encoding='utf-8')
    print(dst)


if __name__ == '__main__':
    main()
