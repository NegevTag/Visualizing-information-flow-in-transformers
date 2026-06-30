"""Small Flask UI for the SAE feature lens (chip + linked-highlight version).

Interactive view of `sae_feature_lens.ipynb`. For a prompt + the last position, the residual stream
decomposes into one contribution vector per source token; we decode each into its top Llama Scope
(JumpReLU) SAE features.

Layout (full width)
-------------------
  * an editable **prompt** box + a horizontal **layer slider** with clickable per-layer ticks;
  * **left**   -- *all* distinct features at this layer, sorted/labelled **by count**;
  * **center** -- one row per source token, each feature a colored **chip** (no bars);
  * **right (top)** -- the **top-5 logits** (logit lens at the selected layer), *not* color-coded.

Encodings / interaction
-----------------------
  * **color = feature identity** (stable golden-angle hue), low opacity; same feature = same color.
  * **magnitude = chip text size** (not color strength); exact activation in the hover tooltip.
  * **hover any feature** -> it stays lit everywhere, everything else dims (linked highlight).
  * **hide features you don't care about**: click a chip, or the × on a left-panel row. Hidden
    features vanish from every panel and the choice persists across layers/prompts/restarts
    (localStorage); restore via the "hidden (N)" tray.

Caching
-------
  * the model run (per prompt), unembedding matrix, last-RMS weight, SAE weights (HF), and
    Neuronpedia descriptions are all cached on disk;
  * additionally the *decoded per-layer payload* is cached to `.sae_payload_cache/`, so a
    previously-viewed (prompt, layer) renders instantly on restart -- without even loading the SAE.

Run (from the backend project so `uv` uses its env):
    cd real/backend && uv run ../../project_scratchpad/sae_feature_lens_ui.py
then open http://127.0.0.1:8050 .

CLAUDE_WRITTEN
"""

from __future__ import annotations

import sys

# Windows consoles default to cp1252; nnsight's remote status display writes unicode (e.g. '◉'),
# which otherwise crashes stdout with UnicodeEncodeError. Force UTF-8 *before* importing anything
# that captures stdout (nnsight/wandb).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import hashlib
import json
import time
from pathlib import Path

import torch

# --- make the backend package + this folder importable (mirrors the notebook) ----------------
HERE = Path(__file__).resolve().parent
BACKEND_SRC = HERE.parents[0] / "real" / "backend" / "src"
assert BACKEND_SRC.exists(), f"backend src not found: {BACKEND_SRC}"
sys.path.insert(0, str(BACKEND_SRC))
sys.path.insert(0, str(HERE))

from flask import Flask, Response, jsonify, request  # noqa: E402

from api_checks.api_cache import APICache  # noqa: E402
from api_checks.position import LLMResidualPosition  # noqa: E402
from info_flow.config import Config  # noqa: E402
from sae_feature_lens import SAEFeatureLens  # noqa: E402

# --- config ----------------------------------------------------------------------------------
DEFAULT_PROMPT = "I hate this person. I think he is so"  # cached run -> no remote call
N_SHOW = 6                 # how many top features to display per source (display only; not SAE gating)
PAYLOAD_DIR = HERE / ".sae_payload_cache"  # decoded per-(prompt,layer) results, cached across runs

config = Config()
MODEL = config.info_flow_model
api_cache = APICache(hf_token=config.hf_token, cache_path=Path(config.result_cache_path))
calculator = api_cache.get_infomration_calculator(MODEL)
lens = SAEFeatureLens(device="cpu")

# One run at startup just to learn the layer count for the slider (same model => constant).
_run0 = api_cache.get_full_run_results(MODEL, DEFAULT_PROMPT)
N_LAYERS = _run0.dimentions.layers

# Logit-lens weights. unembed is disk-cached; last_rms does a one-off remote trace (then disk-cached).
# If remote is unavailable, degrade gracefully (disable the logits panel) instead of crashing.
try:
    unembed = api_cache.get_unembedding_matrix(MODEL)   # (vocab, d_model)
    last_rms = api_cache.get_last_rms_weight(MODEL)      # final RMSNorm weight
    LOGIT_LENS = True
except Exception as e:  # noqa: BLE001
    print(f"[warn] logit-lens disabled (weight fetch failed): {type(e).__name__}: {e}")
    unembed, last_rms, LOGIT_LENS = None, None, False

print(f"model={MODEL} | layers={N_LAYERS} | default prompt={DEFAULT_PROMPT!r}")


def _hue(idx: int) -> float:
    """Stable hue (deg) for a feature id, via the golden angle -> well-spread colors. CLAUDE_WRITTEN"""
    return round((idx * 137.508) % 360.0, 1)


def _disp(tok: str) -> str:
    s = tok.strip()
    return "BOS" if tok.startswith("<|begin_of_text|>") else (s or "·")


def _payload_path(prompt: str, layer: int) -> Path:
    key = hashlib.md5(f"{MODEL}|{prompt}|{layer}|{N_SHOW}".encode()).hexdigest()
    return PAYLOAD_DIR / f"{key}.json"


_mem_cache: dict[tuple[str, int], dict] = {}  # (prompt, layer) -> payload, in-process


def _compute(prompt: str, layer: int):
    """Generator yielding ('log', message) progress lines, then ('result', payload).

    Three cache levels: in-memory dict, on-disk payload json, then full compute (which itself uses
    the cached run + cached SAE weights + cached Neuronpedia descriptions). CLAUDE_WRITTEN
    """
    layer = max(0, min(N_LAYERS - 1, layer))
    mem_key = (prompt, layer)
    if mem_key in _mem_cache:
        yield "log", f"layer {layer}: cached (memory)"
        yield "result", _mem_cache[mem_key]
        return

    path = _payload_path(prompt, layer)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            _mem_cache[mem_key] = payload
            yield "log", f"layer {layer}: loaded from disk cache (no SAE load needed)"
            yield "result", payload
            return
        except (OSError, json.JSONDecodeError):
            pass  # fall through and recompute

    t0 = time.time()
    # Ensure the run for this prompt exists (new prompt = remote model run; cached prompt = disk load).
    run_file = api_cache.results_cache_path / f"{api_cache._get_result_key_name(MODEL, prompt)}.pt"
    yield "log", ("loading cached run for this prompt…" if run_file.exists()
                  else "running model on new prompt (remote — this can take a while)…")
    run = api_cache.get_full_run_results(MODEL, prompt)
    tokens = calculator.calc_tokens(prompt)
    last_pos = run.dimentions.prompt_len - 1
    pos = LLMResidualPosition(layer=layer, token_position=last_pos, is_mlp=True)
    contrib = run.contributions[pos].float()             # (S, d_model)
    resid_norm = run.precise[pos].float().norm().item()  # rescale target (see sae_feature_lens.py)

    yield "log", f"layer {layer}: loading SAE into memory (weights cached on disk after first ever load)…"
    ts = time.time()
    lens.load_sae(layer)  # warm the lru cache; later top_features calls are instant
    yield "log", f"SAE in memory in {time.time() - ts:.1f}s — decoding {len(tokens)} sources + Neuronpedia"

    sources, counts, desc_of, maxact_of = [], {}, {}, {}
    for i, tok in enumerate(tokens):
        feats = []
        for idx, act in lens.top_features(contrib[i], layer, k=N_SHOW, normalize_to=resid_norm):
            idx, act = int(idx), float(act)
            desc = lens.describe(layer, idx)
            feats.append({"id": idx, "act": act, "desc": desc, "hue": _hue(idx)})
            counts[idx] = counts.get(idx, 0) + 1
            desc_of[idx] = desc
            maxact_of[idx] = max(maxact_of.get(idx, 0.0), act)
        sources.append({"i": i, "token": _disp(tok), "feats": feats})
        yield "log", f"source {i + 1}/{len(tokens)}  {_disp(tok)!r}  -> {len(feats)} feats"

    max_act = max((f["act"] for s in sources for f in s["feats"]), default=1.0)
    all_features = sorted(
        (
            {"id": idx, "count": c, "desc": desc_of[idx], "hue": _hue(idx), "max_act": maxact_of[idx]}
            for idx, c in counts.items()
        ),
        key=lambda r: (-r["count"], -r["max_act"]),
    )

    top_logits = []
    if LOGIT_LENS:
        yield "log", "logit lens (last RMSNorm + unembedding)…"
        full = run.precise[pos]
        logit_probs = calculator.calc_top_perdictions_from_vector(
            full, unembed, prediction_num=5, with_last_rms=True, rms_weight=last_rms
        )
        top_logits = [{"token": tok, "prob": float(p)} for tok, p in logit_probs.items()]

    payload = {"layer": layer, "prompt": prompt, "max_act": max_act,
               "sources": sources, "all_features": all_features, "top_logits": top_logits}
    _mem_cache[mem_key] = payload
    try:
        PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass
    yield "log", f"done in {time.time() - t0:.1f}s"
    yield "result", payload


# --- page (vanilla HTML/CSS/JS; data fetched from /api/stream) --------------------------------
PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>SAE feature lens</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;color:#222}
 header{padding:10px 18px;display:flex;align-items:center;gap:14px;border-bottom:1px solid #eee;flex-wrap:wrap}
 header .ttl{font-size:13px;color:#274060;font-weight:600;flex:0 0 auto}
 .sliderbox{flex:1 1 360px}
 .srow2{display:flex;align-items:center;gap:12px}
 #lab{flex:0 0 70px;font-size:14px;color:#274060;font-weight:700;font-variant-numeric:tabular-nums}
 #slider{flex:1 1 auto}
 #ticks{display:flex;justify-content:space-between;font-size:9px;color:#bbb;margin:3px 0 0 82px}
 .tick{cursor:pointer;padding:0 1px}
 .tick.cur{color:#274060;font-weight:700}
 .tick:hover{color:#555}
 .promptbox{display:flex;align-items:center;gap:6px;flex:0 0 auto}
 .promptbox label{font-size:11px;color:#999}
 #prompt-in{width:340px;font-size:12px;padding:5px 8px;border:1px solid #cbd5e3;border-radius:6px}
 #run{font-size:12px;padding:5px 12px;border:1px solid #aac;border-radius:6px;background:#eef3fb;cursor:pointer}
 #run:hover{background:#e0eaf7}
 #wrap{display:flex;gap:22px;padding:16px 18px;align-items:flex-start}
 #all{flex:0 0 290px;max-height:84vh;overflow:auto;border-right:1px solid #f0f0f0;padding-right:14px}
 #sources{flex:1 1 auto;min-width:0}
 #logits{flex:0 0 220px;border-left:1px solid #f0f0f0;padding-left:14px}
 .hd{font-size:11px;color:#999;margin-bottom:8px}
 .srow{display:flex;align-items:flex-start;gap:10px;margin-bottom:9px;padding-bottom:8px;
       border-bottom:1px solid #f5f5f5}
 .tok{flex:0 0 110px;text-align:right;font-weight:600;color:#274060;font-size:13px;padding-top:4px}
 .chips{display:flex;flex-wrap:wrap;gap:6px;align-items:baseline}
 .chip{padding:3px 9px;border-radius:11px;line-height:1.15;white-space:nowrap;cursor:pointer;
       border:1px solid rgba(0,0,0,.07);transition:filter .08s,opacity .08s}
 .afeat{display:flex;align-items:center;gap:8px;padding:3px 4px;border-radius:6px;font-size:12px;
        cursor:default;transition:filter .08s,opacity .08s}
 .sw{width:12px;height:12px;border-radius:3px;flex:0 0 auto;border:1px solid rgba(0,0,0,.12)}
 .cnt{flex:0 0 20px;text-align:right;color:#999;font-size:11px;font-variant-numeric:tabular-nums}
 .desc{overflow:hidden;text-overflow:ellipsis}
 .lrow{display:flex;justify-content:space-between;gap:10px;font-size:12px;padding:3px 0;
       border-bottom:1px solid #f5f5f5}
 .ltok{font-family:ui-monospace,Consolas,monospace;color:#333}
 .lp{color:#888;font-variant-numeric:tabular-nums}
 .afeat .desc{flex:1 1 auto;min-width:0}
 .afeat .x{margin-left:auto;color:#ccc;cursor:pointer;font-size:14px;padding:0 3px;visibility:hidden}
 .afeat:hover .x{visibility:visible}
 .afeat .x:hover{color:#c33}
 .tray{margin-top:14px;border-top:1px dashed #e0e0e0;padding-top:8px}
 .hrow{display:flex;align-items:center;gap:8px;font-size:11px;color:#9aa;padding:2px 4px}
 .hrow .desc{flex:1 1 auto;overflow:hidden;text-overflow:ellipsis}
 .lnk{color:#3a6ea5;cursor:pointer;font-size:11px;white-space:nowrap}
 .lnk:hover{text-decoration:underline}
 #wrap.dimmed .chip:not(.hot),#wrap.dimmed .afeat:not(.hot){filter:grayscale(1);opacity:.10}
 .chip.hot,.afeat.hot{outline:2px solid #333;outline-offset:1px}
 #wrap.loading{opacity:.5}
 #log{position:fixed;right:14px;bottom:14px;width:360px;max-height:42vh;overflow:auto;
      background:rgba(18,26,38,.93);color:#cfe0f3;font:11px/1.45 ui-monospace,Consolas,monospace;
      padding:8px 11px;border-radius:8px;box-shadow:0 6px 20px rgba(0,0,0,.28);display:none;z-index:60}
 #log.show{display:block}
 #log .ln{white-space:pre-wrap}
 #log .done{color:#8fe6b0}
</style></head><body>
<header>
 <span class="ttl">SAE feature lens</span>
 <div class="sliderbox">
   <div class="srow2"><span id="lab">layer __DEFAULT__</span>
     <input id="slider" type="range" min="0" max="__MAX__" step="1" value="__DEFAULT__"></div>
   <div id="ticks"></div>
 </div>
 <div class="promptbox"><label>prompt</label>
   <input id="prompt-in" type="text" spellcheck="false"><button id="run">run ▶</button></div>
</header>
<div id="wrap"><div id="all"></div><div id="sources"></div><div id="logits"></div></div>
<div id="log"></div>
<script>
 const slider=document.getElementById('slider'), lab=document.getElementById('lab'),
       all=document.getElementById('all'), sources=document.getElementById('sources'),
       logits=document.getElementById('logits'), wrap=document.getElementById('wrap'),
       ticks=document.getElementById('ticks'),
       promptIn=document.getElementById('prompt-in'), runBtn=document.getElementById('run');
 const N=(+slider.max)+1;
 const hsla=(h,a)=>`hsla(${h},65%,52%,${a})`;
 const CHIP_A=0.14;                                            // low, subtle fill opacity
 const fsize=(act,mx)=>(11+8*(mx>0?Math.min(act/mx,1):0)).toFixed(1)+'px';  // magnitude -> text size
 const DEFAULT_PROMPT=__PROMPT_JSON__;
 promptIn.value=DEFAULT_PROMPT;
 let CUR_PROMPT=DEFAULT_PROMPT, LAST=null;

 // hidden ("don't care") features: {id: desc}, persisted across layers/prompts/restarts
 const HID_KEY='sae_hidden_feats';
 let hidden=JSON.parse(localStorage.getItem(HID_KEY)||'{}');
 const isHidden=id=>Object.prototype.hasOwnProperty.call(hidden,String(id));
 const saveHidden=()=>localStorage.setItem(HID_KEY,JSON.stringify(hidden));
 function hide(id,desc){hidden[String(id)]=desc||('feature '+id); saveHidden(); if(LAST)render(LAST);}
 function unhide(id){delete hidden[String(id)]; saveHidden(); if(LAST)render(LAST);}
 function showAll(){hidden={}; saveHidden(); if(LAST)render(LAST);}

 // layer-number ticks
 for(let l=0;l<N;l++){const s=document.createElement('span');s.className='tick';s.textContent=l;s.dataset.l=l;ticks.appendChild(s);}
 ticks.addEventListener('click',e=>{const l=e.target.dataset.l; if(l===undefined)return;
   slider.value=l; lab.textContent='layer '+l; mark(); load(l);});
 function mark(){ticks.querySelectorAll('.tick').forEach(t=>t.classList.toggle('cur',+t.dataset.l===+slider.value));}

 // log panel helpers
 const logbox=document.getElementById('log');
 const showLog=()=>logbox.classList.add('show');
 const clearLog=()=>{logbox.innerHTML='';};
 const hideLogSoon=()=>setTimeout(()=>logbox.classList.remove('show'),1800);
 function logline(m){const e=document.createElement('div');
   e.className='ln'+(/^(done|cached|layer \d+: loaded)/.test(m)?' done':''); e.textContent=m;
   logbox.appendChild(e); logbox.scrollTop=logbox.scrollHeight;}

 function render(d){
   LAST=d; wrap.classList.remove('dimmed');
   wrap.querySelectorAll('.hot').forEach(n=>n.classList.remove('hot'));
   // center: per-source chips (hidden features filtered out)
   sources.innerHTML='';
   for(const s of d.sources){
     const row=document.createElement('div'); row.className='srow';
     const t=document.createElement('div'); t.className='tok'; t.textContent=s.token; row.appendChild(t);
     const ch=document.createElement('div'); ch.className='chips';
     for(const f of s.feats){
       if(isHidden(f.id)) continue;
       const c=document.createElement('span'); c.className='chip'; c.dataset.feat=f.id;
       c.style.background=hsla(f.hue,CHIP_A);
       c.style.fontSize=fsize(f.act,d.max_act);
       c.textContent=f.desc;
       c.title=`feature ${f.id}\n${f.desc}\nact ${f.act.toFixed(2)}\n(click to hide)`;
       c.addEventListener('click',()=>hide(f.id,f.desc));
       ch.appendChild(c);
     }
     row.appendChild(ch); sources.appendChild(row);
   }
   // left: all features by count, hidden ones filtered out
   all.innerHTML='';
   const visible=d.all_features.filter(f=>!isHidden(f.id));
   const nHidden=Object.keys(hidden).length;
   const hd=document.createElement('div'); hd.className='hd';
   hd.textContent=`all features (${visible.length} shown`+(nHidden?`, ${nHidden} hidden`:'')+`) — by count`;
   all.appendChild(hd);
   for(const f of visible){
     const a=document.createElement('div'); a.className='afeat'; a.dataset.feat=f.id;
     const sw=document.createElement('span'); sw.className='sw'; sw.style.background=hsla(f.hue,.85); a.appendChild(sw);
     const cn=document.createElement('span'); cn.className='cnt'; cn.textContent=f.count; a.appendChild(cn);
     const tx=document.createElement('span'); tx.className='desc'; tx.textContent=f.desc;
     tx.title=`feature ${f.id}\n${f.desc}\ncount ${f.count}`; a.appendChild(tx);
     const x=document.createElement('span'); x.className='x'; x.textContent='×'; x.title='hide this feature';
     x.addEventListener('click',ev=>{ev.stopPropagation(); hide(f.id,f.desc);}); a.appendChild(x);
     all.appendChild(a);
   }
   if(nHidden){  // hidden tray with per-feature restore + show all
     const tray=document.createElement('div'); tray.className='tray';
     const th=document.createElement('div'); th.className='hd';
     th.appendChild(document.createTextNode(`hidden (${nHidden}) — `));
     const sa=document.createElement('a'); sa.className='lnk'; sa.textContent='show all';
     sa.addEventListener('click',showAll); th.appendChild(sa); tray.appendChild(th);
     for(const id of Object.keys(hidden)){
       const hr=document.createElement('div'); hr.className='hrow';
       const tx=document.createElement('span'); tx.className='desc'; tx.textContent=hidden[id];
       tx.title=hidden[id]; hr.appendChild(tx);
       const r=document.createElement('a'); r.className='lnk'; r.textContent='restore';
       r.addEventListener('click',()=>unhide(id)); hr.appendChild(r);
       tray.appendChild(hr);
     }
     all.appendChild(tray);
   }
   // right: top-5 logits (not color-coded)
   logits.innerHTML='';
   const lh=document.createElement('div'); lh.className='hd';
   lh.textContent=`top-5 logits — logit lens @ layer ${d.layer}`; logits.appendChild(lh);
   for(const t of d.top_logits){
     const r=document.createElement('div'); r.className='lrow';
     const tk=document.createElement('span'); tk.className='ltok'; tk.textContent=JSON.stringify(t.token);
     const pb=document.createElement('span'); pb.className='lp'; pb.textContent=(100*t.prob).toFixed(1)+'%';
     r.appendChild(tk); r.appendChild(pb); logits.appendChild(r);
   }
   wrap.classList.remove('loading');
 }

 // stream a (prompt, layer) over SSE: show log lines live, then render the result
 let es=null;
 function load(L){
   lab.textContent='layer '+L; mark(); wrap.classList.add('loading');
   showLog(); clearLog(); logline('layer '+L+': requesting…');
   if(es) es.close();
   es=new EventSource('/api/stream?layer='+L+'&prompt='+encodeURIComponent(CUR_PROMPT));
   es.addEventListener('log',e=>logline(e.data));
   es.addEventListener('result',e=>{render(JSON.parse(e.data)); es.close(); es=null; hideLogSoon();});
   es.onerror=()=>{logline('(stream error — see server console)'); if(es){es.close(); es=null;} wrap.classList.remove('loading');};
 }
 function runPrompt(){const p=promptIn.value.trim(); if(!p)return; CUR_PROMPT=p; load(slider.value);}
 runBtn.addEventListener('click',runPrompt);
 promptIn.addEventListener('keydown',e=>{if(e.key==='Enter')runPrompt();});

 // linked highlight: hover a feature anywhere -> dim everything that isn't that feature
 wrap.addEventListener('mouseover',e=>{const el=e.target.closest('[data-feat]'); if(!el)return;
   wrap.classList.add('dimmed');
   wrap.querySelectorAll('[data-feat="'+el.dataset.feat+'"]').forEach(n=>n.classList.add('hot'));});
 wrap.addEventListener('mouseout',e=>{const el=e.target.closest('[data-feat]'); if(!el)return;
   wrap.classList.remove('dimmed');
   wrap.querySelectorAll('.hot').forEach(n=>n.classList.remove('hot'));});
 slider.addEventListener('input',()=>{lab.textContent='layer '+slider.value; mark();});
 slider.addEventListener('change',()=>load(slider.value));
 load(slider.value);
</script></body></html>"""

app = Flask(__name__)


@app.route("/")
def index():
    return (PAGE
            .replace("__MAX__", str(N_LAYERS - 1))
            .replace("__DEFAULT__", str(N_LAYERS - 1))
            .replace("__PROMPT_JSON__", json.dumps(DEFAULT_PROMPT)))


def _args() -> tuple[str, int]:
    prompt = request.args.get("prompt", DEFAULT_PROMPT) or DEFAULT_PROMPT
    layer = max(0, min(N_LAYERS - 1, int(request.args.get("layer", N_LAYERS - 1))))
    return prompt, layer


@app.route("/api/features")
def api_features():
    prompt, layer = _args()
    payload = None
    for kind, val in _compute(prompt, layer):
        if kind == "result":
            payload = val
    return jsonify(payload)


@app.route("/api/stream")
def api_stream():
    """Server-Sent Events: 'log' events while computing, then one 'result' event. CLAUDE_WRITTEN"""
    prompt, layer = _args()

    def gen():
        for kind, val in _compute(prompt, layer):
            if kind == "log":
                yield f"event: log\ndata: {val}\n\n"
            else:
                yield f"event: result\ndata: {json.dumps(val)}\n\n"

    return Response(
        gen(), mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)
