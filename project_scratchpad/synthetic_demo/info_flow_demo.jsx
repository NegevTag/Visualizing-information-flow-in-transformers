import { useState, useMemo } from "react";

function makeRNG(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

const TOKENS  = ["the", "cat", "sat", "on", "the₂"];
const N       = 5;
const NLAYERS = 10;
const COLORS    = ["#4A90D9","#27AE74","#E05A2B","#C9960A","#8B5CF6"];
const COLORS_BG = ["#D6E9FA","#BBE9D8","#FAD5C5","#FDE8A0","#DDD6FE"];

const JUMP_LAYER = 6; // the layer where "on" suddenly attends strongly to "cat"

function norm(arr) {
  const s = arr.reduce((a, b) => a + b, 0);
  return s < 1e-9 ? arr : arr.map(v => Math.max(0, v) / s);
}

function lerp(a, b, t) { return a + (b - a) * t; }

// Small deterministic jitter to make curves look organic, not perfectly linear
function jitter(rng, v, scale) { return Math.max(0, v + (rng() - 0.5) * scale); }

function buildBaseLayers() {
  const rng = makeRNG(137);
  const layers = [];

  for (let l = 0; l < NLAYERS; l++) {
    const t = l / (NLAYERS - 1); // 0→1 across layers

    // pos 0 "the": pure blue, always
    const the = [1, 0, 0, 0, 0];

    // pos 1 "cat": causal — only sees itself; stays pure
    const cat = norm([
      jitter(rng, lerp(0.00, 0.04, t), 0.01),
      jitter(rng, lerp(1.0, 0.90, t), 0.02),
      0, 0, 0,
    ]);

    // pos 2 "sat": steadily builds cat+sat, modest the bleed
    const satCatShare = lerp(0.00, 0.44, t);
    const satSatShare = lerp(1.00, 0.46, t);
    const satTheShare = lerp(0.00, 0.08, t);
    const sat = norm([
      jitter(rng, satTheShare, 0.01),
      jitter(rng, satCatShare, 0.025),
      jitter(rng, satSatShare, 0.025),
      0, 0,
    ]);

    // pos 3 "on": slow continuous build, then jump at JUMP_LAYER to cat-dominant
    let onThe, onCat, onSat, onOn;
    if (l === 0) {
      onThe = 0; onCat = 0; onSat = 0; onOn = 1;
    } else if (l < JUMP_LAYER) {
      const s = l / JUMP_LAYER;
      onThe = lerp(0.00, 0.07, s);
      onCat = lerp(0.00, 0.15, s);  // slow rise
      onSat = lerp(0.00, 0.20, s);
      onOn  = lerp(1.00, 0.58, s);
    } else {
      // sudden jump: cat shoots up
      const s = (l - JUMP_LAYER) / Math.max(1, NLAYERS - 1 - JUMP_LAYER);
      onCat = lerp(0.58, 0.65, s);  // big jump from ~15% → 58%
      onSat = lerp(0.20, 0.18, s);
      onOn  = lerp(0.16, 0.10, s);
      onThe = lerp(0.06, 0.07, s);
    }
    const on = norm([
      jitter(rng, onThe, 0.012),
      jitter(rng, onCat, 0.018),
      jitter(rng, onSat, 0.018),
      jitter(rng, onOn,  0.018),
      0,
    ]);

    // pos 4 "the₂": mixed — sat+cat dominant, not blue-heavy
    const t2Sat  = lerp(0.00, 0.36, t);
    const t2Cat  = lerp(0.00, 0.28, t);
    const t2On   = lerp(0.00, 0.16, t);
    const t2The  = lerp(0.00, 0.09, t);
    const t2Self = lerp(1.00, 0.11, t);
    const the2 = norm([
      jitter(rng, t2The,  0.015),
      jitter(rng, t2Cat,  0.025),
      jitter(rng, t2Sat,  0.025),
      jitter(rng, t2On,   0.015),
      jitter(rng, t2Self, 0.020),
    ]);

    layers.push([the, cat, sat, on, the2]);
  }
  return layers;
}

const BASE_LAYERS = buildBaseLayers();

function buildMLPLayer(prevDist, layerIdx) {
  const rng = makeRNG(400 + layerIdx * 23);
  return prevDist.map(posRow => {
    const nr     = [...posRow];
    const ranked = nr.map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v);
    if (ranked.length > 1) {
      const bi = ranked[1].i;
      nr[bi]   = Math.min(0.98, nr[bi] * (1.10 + rng() * 0.04));
      const s  = nr.reduce((a, b) => a + b, 0);
      return nr.map(v => v / s);
    }
    return nr;
  });
}

function hideSelfRow(row, pos) {
  const nr = [...row];
  nr[pos]  = 0;
  const s  = nr.reduce((a, b) => a + b, 0);
  if (s < 1e-6) return nr;
  return nr.map(v => v / s);
}

const MONO = "'JetBrains Mono','Fira Mono','Consolas',monospace";

function Checkbox({ checked, onChange, label }) {
  return (
    <label style={{ display:"flex", alignItems:"center", gap:6, cursor:"pointer", userSelect:"none" }}>
      <input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)}
        style={{ accentColor:"#333", width:12, height:12, cursor:"pointer" }}/>
      <span style={{ fontFamily:MONO, fontSize:11, color: checked ? "#111" : "#888" }}>{label}</span>
    </label>
  );
}

function Bar({ row, posIdx, height, selected, isMLP }) {
  return (
    <div style={{ height, display:"flex", overflow:"hidden", background:"#f0f0f0" }}>
      {selected !== null ? (
        <>
          <div style={{ width:`${row[selected]*100}%`, background:COLORS[selected], transition:"width .3s" }}/>
          <div style={{ flex:1 }}/>
        </>
      ) : (
        row
          .map((v, ti) => ({ v, ti }))
          .filter(d => d.v > 0.004)
          .sort((a, b) => b.v - a.v)
          .map(({ v, ti }) => (
            <div key={ti} style={{ width:`${v*100}%`, background:COLORS[ti], transition:"width .3s",
              opacity: isMLP ? 0.7 : 1 }}/>
          ))
      )}
    </div>
  );
}

export default function AttentionFlow() {
  const [hideSelf, setHideSelf] = useState(false);
  const [showMLP,  setShowMLP]  = useState(false);
  const [selected, setSelected] = useState(null);

  const rows = useMemo(() => {
    const list = [{ type:"attn", idx:0, dist:BASE_LAYERS[0] }];
    for (let l = 1; l < NLAYERS; l++) {
      if (showMLP) list.push({ type:"mlp", idx:l-1, dist:buildMLPLayer(BASE_LAYERS[l-1], l) });
      list.push({ type:"attn", idx:l, dist:BASE_LAYERS[l] });
    }
    return [...list].reverse();
  }, [showMLP]);

  const CW = 90;
  const AH = 11;
  const MH = 4;
  const LW = 38;

  return (
    <div style={{ padding:"32px 36px 40px", fontFamily:MONO, maxWidth:660, margin:"0 auto",
      background:"#fff", color:"#111" }}>

      {/* title */}
      <div style={{ marginBottom:24, borderBottom:"1px solid #ddd", paddingBottom:14 }}>
        <div style={{ fontSize:13, fontWeight:700, letterSpacing:".01em", marginBottom:4 }}>
          Token information flow
        </div>
        <div style={{ fontSize:10, color:"#888", lineHeight:1.7 }}>
          Each bar shows the mixture of source-token information carried at that position after each layer.
          Layers run bottom (L0 = input) to top (L9 = output).
          Click a token name to isolate its contribution.
        </div>
      </div>

      {/* token legend — color-coded words */}
      <div style={{ display:"flex", gap:20, marginBottom:18, alignItems:"center" }}>
        <span style={{ fontSize:9, color:"#bbb", textTransform:"uppercase", letterSpacing:".08em", flexShrink:0 }}>trace:</span>
        {TOKENS.map((tok, i) => {
          const active = selected === i;
          const dim    = selected !== null && !active;
          return (
            <button key={i} onClick={() => setSelected(active ? null : i)} style={{
              fontFamily:MONO, fontSize:12, fontWeight: active ? 700 : 500,
              color: dim ? "#ddd" : COLORS[i],
              background:"none", border:"none", padding:0,
              cursor:"pointer", outline:"none",
              borderBottom: active ? `2px solid ${COLORS[i]}` : "2px solid transparent",
              transition:"color .15s",
            }}>{tok}</button>
          );
        })}
        {selected !== null && (
          <button onClick={() => setSelected(null)} style={{
            fontFamily:MONO, fontSize:10, color:"#aaa", background:"none",
            border:"none", cursor:"pointer", padding:0, marginLeft:"auto",
          }}>× clear</button>
        )}
      </div>

      {/* grid */}
      <div style={{ overflowX:"auto", marginBottom:20 }}>
        {rows.map((row, ri) => {
          const isMLP = row.type === "mlp";
          const rh    = isMLP ? MH : AH;
          return (
            <div key={ri}>
              {/* mark the jump layer with a subtle rule + annotation */}
              <div style={{ display:"flex", alignItems:"center", marginBottom: isMLP ? 1 : 16 }}>
                <div style={{
                  width:LW, flexShrink:0, textAlign:"right", paddingRight:8,
                  fontFamily:MONO,
                  fontSize: isMLP ? 8 : 10,
                  color: isMLP ? "#bbb" : "#999",
                  fontWeight: 400,
                }}>
                  {isMLP ? "mlp" : `L${row.idx}`}
                </div>
                {Array.from({ length:N }, (_, pos) => {
                  const raw = row.dist[pos];
                  const r   = hideSelf ? hideSelfRow(raw, pos) : raw;
                  return (
                    <div key={pos} style={{ width:CW, padding:"0 2px" }}>
                      <Bar row={r} posIdx={pos} height={rh} selected={selected} isMLP={isMLP}/>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}

        {/* axis labels */}
        <div style={{ display:"flex", marginLeft:LW, marginTop:6 }}>
          {TOKENS.map((tok, i) => (
            <button key={i} onClick={() => setSelected(selected === i ? null : i)} style={{
              width:CW, fontFamily:MONO, fontSize:11, fontWeight:700,
              color: selected !== null && selected !== i ? "#ccc" : COLORS[i],
              background:"none", border:"none", cursor:"pointer",
              textAlign:"center", padding:"2px 0",
              transition:"color .15s",
            }}>{tok}</button>
          ))}
        </div>
      </div>

      {/* controls */}
      <div style={{ borderTop:"1px solid #e8e8e8", paddingTop:14, display:"flex", gap:24, alignItems:"center" }}>
        <Checkbox checked={hideSelf} onChange={setHideSelf} label="hide self-contribution"/>
        <Checkbox checked={showMLP}  onChange={setShowMLP}  label="show MLP sublayers"/>
        <span style={{ marginLeft:"auto", fontSize:9, color:"#ccc", letterSpacing:".04em" }}>
          seed 137 · synthetic
        </span>
      </div>
    </div>
  );
}
