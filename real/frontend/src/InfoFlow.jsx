import { useState, useMemo } from "react";
import ZoomPanVanilla from "./ZoomPanVanilla.jsx";

// MVP: prompt -> GET http://127.0.0.1:8000/?prompt=... -> render attention + MLP norm bars.
// Backend has CORS enabled, so we hit it directly (no vite proxy).
// Backend returns:
//   { attention_norms: [layer][position][source],
//     mlp_norms:        [layer][position][source],
//     tokens: [str] }

const MONO = "'JetBrains Mono','Fira Mono','Consolas',monospace";

// HSL palette so it scales to any prompt length. Muted, Tufte-ish.
// MLP gets a washed-out variant so the eye separates attention (primary
// information mixing) from MLP (secondary, per-position transform).
function colorFor(i, n, isMLP = false) {
  const h = (i * 360) / Math.max(n, 1);
  return isMLP ? `hsl(${h}, 30%, 72%)` : `hsl(${h}, 50%, 50%)`;
}

function normalizeRow(row) {
  const s = row.reduce((a, b) => a + b, 0);
  if (s < 1e-9) return row.map(() => 0);
  return row.map((v) => Math.max(0, v) / s);
}

// Drop contributions from hidden source indices and renormalize so the bar
// still fills its slot — visible tokens absorb the hidden mass proportionally.
function applyHidden(row, hidden) {
  if (!hidden || hidden.size === 0) return row;
  const masked = row.map((v, i) => (hidden.has(i) ? 0 : v));
  return normalizeRow(masked);
}

function Bar({ row, height, selected, n, isMLP }) {
  // MLP rows: same hue family as the source token but desaturated + lighter,
  // plus a thin dashed top border, so they read as a secondary information
  // band rather than another attention row.
  const bg = isMLP ? "#fafafa" : "#f0f0f0";
  return (
    <div
      style={{
        height,
        display: "flex",
        overflow: "hidden",
        background: bg,
        borderTop: isMLP ? "1px dashed #ddd" : "none",
      }}
    >
      {selected !== null ? (
        <>
          <div
            style={{
              width: `${(row[selected] ?? 0) * 100}%`,
              background: colorFor(selected, n, isMLP),
              transition: "width .3s",
            }}
          />
          <div style={{ flex: 1 }} />
        </>
      ) : (
        row
          .map((v, ti) => ({ v, ti }))
          .filter((d) => d.v > 0.004)
          .sort((a, b) => b.v - a.v)
          .map(({ v, ti }) => (
            <div
              key={ti}
              style={{
                width: `${v * 100}%`,
                background: colorFor(ti, n, isMLP),
                transition: "width .3s",
              }}
            />
          ))
      )}
    </div>
  );
}

export default function InfoFlow() {
  const [prompt, setPrompt] = useState("the cat sat");
  const [data, setData] = useState(null); // {tokens, attention_norms, mlp_norms, top_perdictions}
  // Per-source contributions to the top output logit (list[float], length N).
  // Fetched in parallel from /top_logit_contributions.
  const [topLogitContribs, setTopLogitContribs] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);
  // Source indices whose contribution is suppressed and redistributed.
  // First token hidden by default (BOS / leading token tends to dominate).
  const [hidden, setHidden] = useState(() => new Set([0]));

  // Visual swap: when true, attention rows get the thin/washed treatment and
  // MLP rows get the thick/saturated one. Row positions and label text stay
  // put — only height/font-size/color-saturation flip.
  const [swap, setSwap] = useState(false);

  function toggleHidden(i) {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else {
        next.add(i);
        // If we just hid the currently-traced token, clear the trace.
        if (selected === i) setSelected(null);
      }
      return next;
    });
  }

  async function run() {
    setLoading(true);
    setError(null);
    setSelected(null);
    try {
      // Sequential: backend caches the model run, so / must complete (and
      // populate the cache) before /top_logit_contributions can hit it cheaply.
      const url = (p) => `http://127.0.0.1:8000${p}?prompt=${encodeURIComponent(prompt)}`;
      const mainRes = await fetch(url("/"));
      if (!mainRes.ok) throw new Error(`HTTP ${mainRes.status}`);
      const json = await mainRes.json();
      setData(json);
      // top_logit_contributions is best-effort — don't fail the whole view if it errors.
      const logitRes = await fetch(url("/top_logit_contributions"));
      setTopLogitContribs(logitRes.ok ? await logitRes.json() : null);
      // New prompt = new tokens; reset hide state to "first token hidden".
      setHidden(new Set([0]));
    } catch (e) {
      setError(String(e));
      setData(null);
      setTopLogitContribs(null);
    } finally {
      setLoading(false);
    }
  }

  // Build a flat list of rows: [attn_0, mlp_0, attn_1, mlp_1, ...]; then reverse
  // so output sits on top and input on bottom, matching the synthetic demo.
  const rows = useMemo(() => {
    if (!data) return [];
    const { attention_norms, mlp_norms } = data;
    const L = attention_norms.length;
    const list = [];
    for (let l = 0; l < L; l++) {
      list.push({
        type: "attn",
        layer: l,
        dist: attention_norms[l].map((r) => applyHidden(normalizeRow(r), hidden)),
      });
      list.push({
        type: "mlp",
        layer: l,
        dist: mlp_norms[l].map((r) => applyHidden(normalizeRow(r), hidden)),
      });
    }
    return list.reverse();
  }, [data, hidden]);

  const tokens = data?.tokens ?? [];
  const N = tokens.length;
  // 1.5× wider than the previous 40–90 range. With zoom + pan available now,
  // it's fine if not all columns fit on screen at once.
  const CW = N > 0 ? Math.max(60, Math.min(135, Math.floor(900 / N))) : 90;
  const AH = 11;
  const MH = 4;
  const LW = 48;

  return (
    <div
      style={{
        padding: "32px 36px 40px",
        fontFamily: MONO,
        maxWidth: 900,
        margin: "0 auto",
        background: "#fff",
        color: "#111",
      }}
    >
      {/* title */}
      <div style={{ marginBottom: 20, borderBottom: "1px solid #ddd", paddingBottom: 14 }}>
        <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>
          Token information flow
        </div>
        <div style={{ fontSize: 10, color: "#888", lineHeight: 1.7 }}>
          Each bar = source-token mixture at that position after each sublayer.
          Bottom = layer 0 attn (input side), top = last layer MLP (output side).
        </div>
      </div>

      {/* prompt input */}
      <div style={{ display: "flex", gap: 8, marginBottom: 20, alignItems: "center" }}>
        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
          placeholder="prompt"
          style={{
            flex: 1,
            fontFamily: MONO,
            fontSize: 12,
            padding: "6px 10px",
            border: "1px solid #ccc",
            outline: "none",
            background: "#fafafa",
          }}
        />
        <button
          onClick={run}
          disabled={loading || !prompt.trim()}
          style={{
            fontFamily: MONO,
            fontSize: 11,
            padding: "6px 14px",
            border: "1px solid #333",
            background: loading ? "#eee" : "#fff",
            cursor: loading ? "wait" : "pointer",
          }}
        >
          {loading ? "running…" : "run"}
        </button>
        {/* vertical attn/mlp style swap. Top half = "attn primary" (default),
            bottom half = "mlp primary". Click anywhere to toggle. */}
        <div
          onClick={() => setSwap((s) => !s)}
          title="swap attention/MLP visual styles"
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            cursor: "pointer",
            userSelect: "none",
            fontFamily: MONO,
            fontSize: 8,
            color: "#888",
            lineHeight: 1,
            marginLeft: 4,
          }}
        >
          <div style={{ marginBottom: 2, fontWeight: !swap ? 700 : 400, color: !swap ? "#333" : "#bbb" }}>A</div>
          <div
            style={{
              width: 16,
              height: 30,
              borderRadius: 8,
              background: "#eee",
              border: "1px solid #ccc",
              position: "relative",
            }}
          >
            <div
              style={{
                position: "absolute",
                left: 2,
                top: swap ? 16 : 2,
                width: 10,
                height: 10,
                borderRadius: 5,
                background: "#333",
                transition: "top .18s ease",
              }}
            />
          </div>
          <div style={{ marginTop: 2, fontWeight: swap ? 700 : 400, color: swap ? "#333" : "#bbb" }}>M</div>
        </div>
      </div>

      {error && (
        <div style={{ fontFamily: MONO, fontSize: 11, color: "#c0392b", marginBottom: 16 }}>
          {error}
        </div>
      )}

      {data && N > 0 && (
        <>
          {/* token legend — click to trace */}
          <div style={{ display: "flex", gap: 14, marginBottom: 14, alignItems: "center", flexWrap: "wrap" }}>
            <span style={{ fontSize: 9, color: "#bbb", textTransform: "uppercase", letterSpacing: ".08em" }}>
              trace:
            </span>
            {tokens.map((tok, i) => {
              const active = selected === i;
              const isHidden = hidden.has(i);
              const dim = selected !== null && !active;
              return (
                <div key={i} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
                  <button
                    onClick={() => !isHidden && setSelected(active ? null : i)}
                    disabled={isHidden}
                    style={{
                      fontFamily: MONO,
                      fontSize: 12,
                      fontWeight: active ? 700 : 500,
                      color: isHidden ? "#ccc" : dim ? "#ddd" : colorFor(i, N),
                      textDecoration: isHidden ? "line-through" : "none",
                      background: "none",
                      border: "none",
                      padding: 0,
                      cursor: isHidden ? "default" : "pointer",
                      borderBottom: active ? `2px solid ${colorFor(i, N)}` : "2px solid transparent",
                    }}
                  >
                    {tok}
                  </button>
                  <button
                    onClick={() => toggleHidden(i)}
                    title={isHidden ? "show" : "hide"}
                    style={{
                      fontFamily: MONO,
                      fontSize: 9,
                      lineHeight: 1,
                      color: isHidden ? "#888" : "#ccc",
                      background: "none",
                      border: "none",
                      padding: 0,
                      cursor: "pointer",
                    }}
                  >
                    {isHidden ? "●" : "○"}
                  </button>
                </div>
              );
            })}
            {selected !== null && (
              <button
                onClick={() => setSelected(null)}
                style={{
                  fontFamily: MONO,
                  fontSize: 10,
                  color: "#aaa",
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  marginLeft: "auto",
                }}
              >
                × clear
              </button>
            )}
          </div>

          {/* Top next-token predictions: legend swatches, same color, stacked
              vertically above the last token's column (where the next token
              would appear). Most probable token sits at the top. */}
          {data.top_perdictions && Object.keys(data.top_perdictions).length > 0 && (
            <div
              style={{
                display: "flex",
                justifyContent: "flex-end",
                marginBottom: 10,
              }}
            >
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: 3,
                  fontFamily: MONO,
                  fontSize: 10,
                  color: "#333",
                  alignItems: "flex-start",
                }}
              >
                {Object.entries(data.top_perdictions)
                  .sort((a, b) => b[1] - a[1])
                  .map(([tok, p], i) => (
                    <div
                      key={i}
                      title={p.toFixed(4)}
                      style={{ display: "flex", alignItems: "center", gap: 4 }}
                    >
                      <div style={{ width: 8, height: 8, background: "#555" }} />
                      <span>{tok}</span>
                      <span style={{ color: "#888" }}>{p.toFixed(3)}</span>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* grid — wrapped in zoom/pan surface (mouse wheel zooms centered
              on the cursor, click-drag pans, reset button top-right). */}
          <ZoomPanVanilla>
            {/* Top-logit contributions: a single bar sitting above the last
                (rightmost) column, sized to match an attention row. Sources
                that are hidden in the rest of the grid are also hidden here. */}
            {topLogitContribs && topLogitContribs.length === N && (
              <div
                style={{ display: "flex", alignItems: "center", marginBottom: 8 }}
              >
                <div
                  style={{
                    width: LW,
                    flexShrink: 0,
                    textAlign: "right",
                    paddingRight: 8,
                    fontFamily: MONO,
                    fontSize: 10,
                    color: "#333",
                    fontWeight: 700,
                  }}
                >
                  logit
                </div>
                {tokens.map((_, pos) => (
                  <div key={pos} style={{ width: CW, padding: "0 2px" }}>
                    {pos === N - 1 && (
                      <Bar
                        row={applyHidden(normalizeRow(topLogitContribs), hidden)}
                        height={AH}
                        selected={selected}
                        n={N}
                        isMLP={false}
                      />
                    )}
                  </div>
                ))}
              </div>
            )}
            {rows.map((row, ri) => {
              const isMLP = row.type === "mlp";
              // styleAsMLP drives the visual treatment (height, text size,
              // color saturation). Position and label text stay tied to the
              // actual row type — only design flips.
              const styleAsMLP = swap ? !isMLP : isMLP;
              const rh = styleAsMLP ? MH : AH;
              return (
                <div
                  key={ri}
                  style={{ display: "flex", alignItems: "center", marginBottom: isMLP ? 8 : 1 }}
                >
                  <div
                    style={{
                      width: LW,
                      flexShrink: 0,
                      textAlign: "right",
                      paddingRight: 8,
                      fontFamily: MONO,
                      fontSize: styleAsMLP ? 8 : 10,
                      color: styleAsMLP ? "#bbb" : "#999",
                    }}
                  >
                    {isMLP ? `mlp${row.layer}` : `L${row.layer}`}
                  </div>
                  {row.dist.map((posRow, pos) => (
                    <div key={pos} style={{ width: CW, padding: "0 2px" }}>
                      <Bar row={posRow} height={rh} selected={selected} n={N} isMLP={styleAsMLP} />
                    </div>
                  ))}
                </div>
              );
            })}

            {/* color stripe above bottom token labels — same width as the
                token's column, in the token's hue. Visually ties the legend
                color to the column it sits over.
                Uses a real spacer div (not marginLeft) so it lines up with the
                bars: row labels are content-box, so LW + paddingRight pushes
                bars 8px past LW — a margin of just LW would be misaligned. */}
            <div style={{ display: "flex", marginTop: 4, alignItems: "center" }}>
              <div style={{ width: LW, flexShrink: 0, paddingRight: 8 }} />
              {tokens.map((_, i) => {
                const isHidden = hidden.has(i);
                const dim = selected !== null && selected !== i;
                return (
                  <div key={i} style={{ width: CW, padding: "0 2px" }}>
                    <div
                      style={{
                        height: 2,
                        background: isHidden || dim ? "#eee" : colorFor(i, N),
                      }}
                    />
                  </div>
                );
              })}
            </div>

            {/* token axis labels at bottom */}
            <div style={{ display: "flex", marginTop: 4, alignItems: "center" }}>
              <div style={{ width: LW, flexShrink: 0, paddingRight: 8 }} />
              {tokens.map((tok, i) => {
                const isHidden = hidden.has(i);
                return (
                  // Wrapper matches the bar cell (width: CW, padding: 0 2px)
                  // so labels line up exactly with the columns above.
                  <div key={i} style={{ width: CW, padding: "0 2px" }}>
                    <button
                      onClick={() => !isHidden && setSelected(selected === i ? null : i)}
                      disabled={isHidden}
                      style={{
                        width: "100%",
                        fontFamily: MONO,
                        fontSize: 11,
                        fontWeight: 700,
                        color: isHidden
                          ? "#ccc"
                          : selected !== null && selected !== i
                          ? "#ccc"
                          : colorFor(i, N),
                        textDecoration: isHidden ? "line-through" : "none",
                        background: "none",
                        border: "none",
                        cursor: isHidden ? "default" : "pointer",
                        textAlign: "center",
                        padding: "2px 0",
                      }}
                    >
                      {tok}
                    </button>
                  </div>
                );
              })}
            </div>
          </ZoomPanVanilla>
        </>
      )}
    </div>
  );
}
