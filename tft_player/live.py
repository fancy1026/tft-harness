"""Live explanation view: what every agent is doing and why, while it plays.

The player calls update()/event(); a tiny local HTTP server serves a page
(http://localhost:8765) that polls the state every ~0.7s. Nothing leaves the
machine. The same events go to the run log, from which tools/report.py builds
the post-game report.
"""
from __future__ import annotations

import http.server
import json
import threading
import time
from collections import deque

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>TFT player live</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--text:#e6e8ee;--dim:#8b93a7;--acc:#d8b25a;--ok:#5fbf8f;--warn:#e0875a}
@media (prefers-color-scheme: light){:root{--bg:#f6f7f9;--panel:#fff;--line:#e3e6ec;--text:#1b1f27;--dim:#5d667a;--acc:#9a7418;--ok:#1f8a58;--warn:#b5532a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,Segoe UI,sans-serif}
header{display:flex;gap:18px;flex-wrap:wrap;align-items:baseline;padding:12px 16px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg)}
header b{font-size:22px;color:var(--acc)}header span{color:var(--dim)}header em{font-style:normal;color:var(--text);font-weight:600}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px;padding:12px 16px}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px;min-width:0}
h2{margin:0 0 6px;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim)}
.agent{padding:5px 0;border-top:1px solid var(--line)}.agent:first-of-type{border-top:0}
.agent b{display:inline-block;min-width:76px;color:var(--acc)}.act{color:var(--ok)}
pre{white-space:pre-wrap;word-break:break-word;margin:4px 0;font:12px/1.4 ui-monospace,Consolas,monospace;color:var(--text)}
.tl{max-height:420px;overflow:auto}.tl div{padding:2px 0;border-top:1px dashed var(--line);font-size:13px}
.tl i{color:var(--dim);font-style:normal;margin-right:6px}.wide{grid-column:1/-1}.warn{color:var(--warn)}
</style></head><body>
<header id=h></header><main>
<section><h2>Agents - doing / why not</h2><div id=agents></div></section>
<section><h2>Plan (model / meta)</h2><div id=plan></div></section>
<section><h2>Board &amp; bench (as tracked)</h2><pre id=roster></pre><h2>Item bench</h2><pre id=items></pre></section>
<section class=wide><h2>Model calls - full reasoning</h2><div id=model></div></section>
<section class=wide><h2>Timeline</h2><div class=tl id=tl></div></section>
</main><script>
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function tick(){try{const r=await fetch('state.json?'+Date.now());const s=await r.json();
const g=s.game||{};h.innerHTML=`<b>${esc(g.stage||'?')}</b><span>timer <em>${esc(g.timer)}</em></span><span>gold <em>${esc(g.gold)}</em></span><span>level <em>${esc(g.level)}</em></span><span>HP <em>${esc(g.hp)}</em></span><span>phase <em>${esc(g.phase)}</em></span><span class=${s.age>5?'warn':''}>updated ${esc(s.age?.toFixed(1))}s ago</span>`;
agents.innerHTML=Object.entries(s.agents||{}).map(([k,v])=>`<div class=agent><b>${esc(k)}</b><span class=${String(v).startsWith('ACT')?'act':''}>${esc(v)}</span></div>`).join('');
const p=s.plan||{};plan.innerHTML=`<pre>${esc(JSON.stringify(p,null,1))}</pre>`;
roster.textContent=s.roster||'';items.textContent=JSON.stringify(s.items||[]);
model.innerHTML=(s.model||[]).slice().reverse().map(m=>`<div class=agent><b>${esc(m.kind)}</b> ${esc(m.stage)} - ${esc(m.latency)}s<pre>${esc(m.answer)}</pre><details><summary>prompt</summary><pre>${esc(m.prompt)}</pre></details></div>`).join('')||'<span class=dim>no model calls yet</span>';
tl.innerHTML=(s.timeline||[]).slice().reverse().map(e=>`<div><i>${esc(e.stage)} ${esc(e.agent)}</i>${esc(e.text)}</div>`).join('');
}catch(e){}setTimeout(tick,700)}tick();
</script></body></html>"""


class LiveView:
    def __init__(self, port=8765):
        self.state = {'game': {}, 'agents': {}, 'plan': {}, 'roster': '', 'items': [],
                      'model': [], 'timeline': []}
        self._tl = deque(maxlen=400)
        self._model = deque(maxlen=30)
        self._lock = threading.Lock()
        self.updated = time.time()
        self.port = port
        try:
            self._serve()
        except OSError:
            self.port = None             # port busy: run without the page

    def update(self, **kw):
        with self._lock:
            self.state.update(kw)
            self.updated = time.time()

    def event(self, stage, agent, text):
        with self._lock:
            self._tl.append({'t': round(time.time(), 2), 'stage': stage, 'agent': agent, 'text': text})

    def model_call(self, kind, stage, prompt, answer, latency):
        with self._lock:
            self._model.append({'kind': kind, 'stage': stage, 'prompt': prompt, 'answer': answer,
                                'latency': round(latency, 1), 't': time.time()})

    def snapshot(self):
        with self._lock:
            s = dict(self.state)
            s['timeline'] = list(self._tl)
            s['model'] = list(self._model)
            s['age'] = time.time() - self.updated
            return s

    def _serve(self):
        view = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith('/state.json'):
                    body = json.dumps(view.snapshot(), default=str).encode()
                    ctype = 'application/json'
                else:
                    body, ctype = PAGE.encode(), 'text/html; charset=utf-8'
                self.send_response(200)
                self.send_header('Content-Type', ctype)
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        srv = http.server.ThreadingHTTPServer(('127.0.0.1', self.port), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
