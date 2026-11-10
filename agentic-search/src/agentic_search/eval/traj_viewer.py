#!/usr/bin/env python3
"""Trajectory viewer for rollout_eval output.

Usage:
    python3 traj_viewer.py                         # serves /data/eval on :8080
    python3 traj_viewer.py --root /some/path
    python3 traj_viewer.py --root /data/eval --port 9000 --host 0.0.0.0

Each sub-directory under --root is one eval run; it is expected to contain a
``*_trajectories.jsonl`` (the name rollout_eval.py writes). The page lists the
runs, lets you pick one, filter by data_source (all / one of the 7 Search-R1
benchmarks), and step through trajectories one at a time.

Only the Python standard library is used, so it runs in the bare eval image.
"""

import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path("/data/eval")
BENCHES = [
    "searchR1_nq",
    "searchR1_triviaqa",
    "searchR1_popqa",
    "searchR1_hotpotqa",
    "searchR1_2wikimultihopqa",
    "searchR1_musique",
    "searchR1_bamboogle",
]

# folder name -> {"path": Path, "rows": list[dict], "mtime": float}
_cache = {}


def _runs():
    """Sub-directories that hold a trajectory jsonl."""
    if not ROOT.is_dir():
        return []
    out = []
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir():
            continue
        files = sorted(d.glob("*_trajectories.jsonl")) or sorted(d.glob("*.jsonl")) or sorted(d.glob("*.json"))
        if files:
            out.append((d.name, files[0]))
    return out


def _load(name):
    """Read one run's jsonl, caching by mtime. Falls back to index -> bench.

    Older runs were written before ``data_source`` was recorded. test.jsonl is
    a fixed concatenation of 7 benchmarks (500 each, bamboogle 125), so the
    row's ``index`` (the dataset row id) maps to a benchmark deterministically.
    """
    runs = dict(_runs())
    if name not in runs:
        return None
    path = runs[name]
    mtime = path.stat().st_mtime
    hit = _cache.get(name)
    if hit and hit["mtime"] == mtime:
        return hit["rows"]

    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row["_pos"] = len(rows)
            if not row.get("data_source"):
                idx = row.get("index")
                if isinstance(idx, int) and 0 <= idx < 3125:
                    row["data_source"] = BENCHES[min(idx // 500, 6)]
                else:
                    row["data_source"] = "unknown"
            rows.append(row)

    _cache[name] = {"path": path, "mtime": mtime, "rows": rows}
    return rows


def _summary(rows):
    out = []
    for r in rows:
        out.append(
            {
                "i": r["_pos"],
                "index": r.get("index"),
                "data_source": r.get("data_source"),
                "num_turns": r.get("num_turns"),
                "format": r.get("format"),
                "EM": r.get("EM"),
                "score": r.get("score"),
                "rm_score": r.get("rm_score"),
                "answer": r.get("answer"),
            }
        )
    return out


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trajectory viewer</title>
<style>
:root{
  --bg:#f7f7f5; --panel:#ffffff; --line:#e4e4e0; --txt:#22221f; --muted:#77776f;
  --accent:#3b6d11; --accent-bg:#eaf3de; --chip:#f1efe8; --warn:#a32d2d; --warn-bg:#fcebeb;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--txt);
  font:13px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB",sans-serif;
  display:flex;flex-direction:column;overflow:hidden}
header{flex:0 0 auto;display:flex;align-items:center;gap:14px;flex-wrap:wrap;
  padding:12px 20px;background:var(--panel);border-bottom:1px solid var(--line)}
h1{font-size:15px;font-weight:500;margin:0 8px 0 0}
label{font-size:12px;color:var(--muted);margin-right:6px}
select,input[type=number]{font:inherit;font-size:13px;padding:5px 9px;border:1px solid var(--line);
  border-radius:7px;background:var(--panel);color:var(--txt);outline:none}
select:focus,input:focus{border-color:var(--accent)}
button{font:inherit;font-size:13px;padding:5px 11px;border:1px solid var(--line);border-radius:7px;
  background:var(--panel);color:var(--txt);cursor:pointer}
button:hover{background:var(--chip)}
button:disabled{opacity:.4;cursor:default}
#root-path{font:12px var(--mono);color:var(--muted)}
main{flex:1;min-height:0;padding:16px 20px;display:grid;
  grid-template-columns:300px minmax(0,1fr);gap:16px}
.side{min-height:0;overflow:auto;display:flex;flex-direction:column;gap:14px}
.side .card{margin-bottom:0}
.trace{min-height:0;display:flex;flex-direction:column}
.trace .card{flex:1;min-height:0;margin-bottom:0;display:flex;flex-direction:column}
.tools{display:flex;align-items:center;gap:12px}
#copy{border-color:var(--accent);color:var(--accent)}
#copy.ok{background:var(--accent-bg)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-bottom:14px}
.q{font-size:14px;font-weight:500;margin:0 0 8px}
.kv{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12px;color:var(--muted)}
.kv b{font-weight:500;color:var(--txt)}
.metrics{display:flex;flex-wrap:wrap;gap:8px}
.metric{background:var(--chip);border-radius:8px;padding:7px 12px;min-width:92px}
.metric .k{font-size:11px;color:var(--muted);display:block}
.metric .v{font:500 16px var(--mono)}
.metric.good .v{color:var(--accent)}
.metric.bad .v{color:var(--warn)}
.cardhead{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:10px}
.cardhead h2{font-size:13px;font-weight:500;margin:0}
pre{margin:0;font:12.5px/1.65 var(--mono);white-space:pre-wrap;word-break:break-word;
  background:var(--panel);padding:0;flex:1;min-height:0;overflow:auto}
.tog{display:flex;align-items:center;gap:5px;font-size:12px;color:var(--muted);cursor:pointer}
footer{flex:0 0 auto;background:var(--panel);border-top:1px solid var(--line);
  padding:10px 20px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.nav{display:flex;align-items:center;gap:6px}
.nav input{width:78px;text-align:center}
#count{font:12px var(--mono);color:var(--muted);min-width:86px}
.spacer{flex:1}
#status{font-size:12px;color:var(--muted)}
.padnote{color:var(--muted);font-style:italic}
@media(max-width:900px){
  main{grid-template-columns:1fr;overflow:auto}
  .side{overflow:visible}
  .trace .card{min-height:60vh}
}
</style>
</head>
<body>
<header>
  <h1>Trajectory viewer</h1>
  <label for="folder">run</label><select id="folder"></select>
  <span id="root-path"></span>
  <span class="spacer"></span>
  <span id="status"></span>
</header>

<main>
  <aside class="side">
    <div class="card">
      <div class="q" id="question">—</div>
      <div class="kv" id="kv"></div>
    </div>
    <div class="card">
      <div class="cardhead"><h2>Metrics</h2></div>
      <div class="metrics" id="metrics"></div>
    </div>
  </aside>
  <section class="trace">
    <div class="card">
      <div class="cardhead">
        <h2>Trajectory</h2>
        <div class="tools">
          <label class="tog"><input type="checkbox" id="hidepad" checked> 折叠 padding</label>
          <button id="copy" title="复制当前轨迹原文（不受折叠开关影响）">复制轨迹</button>
        </div>
      </div>
      <pre id="traj">—</pre>
    </div>
  </section>
</main>

<footer>
  <div class="nav">
    <button id="prev" title="上一条 (←)">←</button>
    <input type="number" id="goto" min="1" value="1">
    <button id="next" title="下一条 (→)">→</button>
  </div>
  <span id="count">— / —</span>
  <label for="bench">bench</label><select id="bench"></select>
  <span class="spacer"></span>
  <span style="font-size:12px;color:var(--muted)">← / → 翻页</span>
</footer>

<script>
let runs=[], rows=[], filtered=[], cur=0, folder='', bench='all', currentRaw='';

const $=id=>document.getElementById(id);
const esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

async function api(url){
  const r=await fetch(url);
  if(!r.ok) throw new Error(r.status+' '+await r.text());
  return r.json();
}

async function init(){
  const meta=await api('/api/meta');
  $('root-path').textContent=meta.root;
  $('folder').innerHTML=meta.runs.map(x=>`<option value="${esc(x.name)}">${esc(x.name)} (${x.count})</option>`).join('');
  const b=['all',...meta.benches];
  $('bench').innerHTML=b.map(x=>`<option value="${esc(x)}">${x==='all'?'all':esc(x.replace('searchR1_',''))}</option>`).join('');
  if(!meta.runs.length){ $('status').textContent='no trajectories found'; return; }
  await pickFolder($('folder').value);
}

async function pickFolder(name){
  folder=name;
  const d=await api('/api/summary?folder='+encodeURIComponent(name));
  rows=d.rows;
  const bs=['all',...new Set(rows.map(r=>r.data_source).filter(x=>x&&x!=='unknown'))];
  const keep=$('bench').value;
  $('bench').innerHTML=bs.map(x=>`<option value="${esc(x)}">${x==='all'?'all':esc(x.replace('searchR1_',''))}</option>`).join('');
  bench=bs.includes(keep)?keep:'all';
  $('bench').value=bench;
  applyFilter(true);
}

function applyFilter(reset){
  filtered = bench==='all' ? rows.map(r=>r.i) : rows.filter(r=>r.data_source===bench).map(r=>r.i);
  if(reset) cur=0;
  if(cur>=filtered.length) cur=0;
  render();
}

async function render(){
  const n=filtered.length;
  $('goto').max=n||1;
  $('prev').disabled=n===0||cur<=0;
  $('next').disabled=n===0||cur>=n-1;
  $('count').textContent=n?`${cur+1} / ${n}`:'0 / 0';
  $('goto').value=n?cur+1:0;
  if(!n){
    currentRaw='';
    $('question').textContent='—'; $('kv').innerHTML=''; $('metrics').innerHTML='';
    $('traj').textContent='（该 bench 没有轨迹）'; $('status').textContent='';
    return;
  }
  const pos=filtered[cur];
  const d=await api('/api/item?folder='+encodeURIComponent(folder)+'&pos='+pos);
  const r=d.row;
  $('status').textContent=`row ${pos} · ${r.data_source||'unknown'}`;
  $('question').textContent=r.question||'—';
  const gt=Array.isArray(r.ground_truth)?r.ground_truth.join(' | '):r.ground_truth;
  const bits=[
    ['index',r.index],['bench',(r.data_source||'').replace('searchR1_','')],
    ['num_turns',r.num_turns],['mode',r.mode],['ground_truth',gt],
    ['chars',(r.trajectory||'').length]
  ];
  $('kv').innerHTML=bits.map(([k,v])=>`<span>${esc(k)}: <b>${esc(v)}</b></span>`).join('');

  const skip=new Set(['index','mode','question','num_turns','ground_truth','trajectory','data_source','_pos']);
  const metrics=Object.keys(r).filter(k=>!skip.has(k));
  $('metrics').innerHTML=metrics.map(k=>{
    const v=r[k];
    const num=typeof v==='number';
    let cls='';
    if(num&&(k==='EM'||k==='format'||k==='answer')) cls=v>0?'good':'bad';
    const shown=num?Number(v.toFixed(4)):(v==null?'—':String(v));
    return `<div class="metric ${cls}"><span class="k">${esc(k)}</span><span class="v">${esc(shown)}</span></div>`;
  }).join('')||'<span style="color:var(--muted)">no metrics</span>';

  currentRaw=r.trajectory||'';
  let t=currentRaw;
    if($('hidepad').checked){
      const pad=/<\|endoftext\|>/g;
      const hits=(t.match(pad)||[]).length;
      if(hits) t=t.replace(/(?:<\|endoftext\|>[\r\n]*){2,}/g,`\n[… ${hits} padding tokens 已折叠 …]\n`);
    }
  $('traj').textContent=t;
  $('traj').scrollTop=0;
}

function go(delta){
  const n=filtered.length;
  if(!n) return;
  cur=Math.max(0,Math.min(n-1,cur+delta));
  render();
}

$('folder').addEventListener('change',e=>pickFolder(e.target.value));
$('bench').addEventListener('change',e=>{bench=e.target.value;applyFilter(true);});
$('prev').addEventListener('click',()=>go(-1));
$('next').addEventListener('click',()=>go(1));
$('goto').addEventListener('change',e=>{
  const n=filtered.length, v=parseInt(e.target.value,10);
  if(n&&Number.isFinite(v)){cur=Math.max(0,Math.min(n-1,v-1));render();}
});
$('hidepad').addEventListener('change',()=>render());
async function copyText(s){
  try{
    if(navigator.clipboard&&window.isSecureContext){
      await navigator.clipboard.writeText(s);
      return true;
    }
  }catch(e){}
  const ta=document.createElement('textarea');
  ta.value=s;
  ta.setAttribute('readonly','');
  ta.style.cssText='position:fixed;top:0;left:0;opacity:0';
  document.body.appendChild(ta);
  ta.select();
  let ok=false;
  try{ok=document.execCommand('copy');}catch(e){}
  document.body.removeChild(ta);
  return ok;
}
$('copy').addEventListener('click',async()=>{
  const ok=await copyText(currentRaw);
  const b=$('copy');
  b.textContent=ok?'已复制':'复制失败';
  b.classList.toggle('ok',ok);
  setTimeout(()=>{b.textContent='复制轨迹';b.classList.remove('ok');},1500);
});
window.addEventListener('keydown',e=>{
  if(document.activeElement&&document.activeElement.tagName==='INPUT') return;
  if(e.key==='ArrowLeft'){e.preventDefault();go(-1);}
  if(e.key==='ArrowRight'){e.preventDefault();go(1);}
});
init().catch(e=>{$('status').textContent='error: '+e.message;});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False))

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/" or u.path == "/index.html":
                return self._send(200, HTML, "text/html; charset=utf-8")

            if u.path == "/api/meta":
                runs = []
                for name, _ in _runs():
                    rows = _load(name) or []
                    runs.append({"name": name, "count": len(rows)})
                return self._json({"root": str(ROOT), "runs": runs, "benches": BENCHES})

            if u.path == "/api/summary":
                name = q.get("folder", [""])[0]
                rows = _load(name)
                if rows is None:
                    return self._json({"error": "unknown folder"}, 404)
                return self._json({"rows": _summary(rows)})

            if u.path == "/api/item":
                name = q.get("folder", [""])[0]
                pos = int(q.get("pos", ["0"])[0])
                rows = _load(name)
                if rows is None:
                    return self._json({"error": "unknown folder"}, 404)
                if not 0 <= pos < len(rows):
                    return self._json({"error": "out of range"}, 404)
                return self._json({"row": rows[pos]})

            return self._send(404, "not found", "text/plain; charset=utf-8")
        except Exception as exc:
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def main():
    global ROOT
    ap = argparse.ArgumentParser(description="Browse rollout_eval trajectory jsonl files")
    ap.add_argument("--root", default=os.environ.get("TRAJ_ROOT", "/data/eval"),
                    help="directory containing one sub-dir per eval run (default /data/eval)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    ROOT = Path(args.root).expanduser().resolve()
    print(f"trajectory viewer: http://{args.host}:{args.port}/  root={ROOT}", flush=True)
    if not ROOT.is_dir():
        print(f"warning: {ROOT} does not exist yet", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
