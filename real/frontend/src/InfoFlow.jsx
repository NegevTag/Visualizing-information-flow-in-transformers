import { useState, useMemo } from "react";

// MVP: prompt -> GET /api?prompt=... -> render attention + MLP norm bars.
// Backend returns:
//   { attention_norms: [layer][position][source],
//     mlp_norms:        [layer][position][source],
//     tokens: [str] }

const MONO = "'JetBrains Mono','Fira Mono','Consolas',monospace";

// HSL palette so it scales to any prompt length. Muted, Tufte-ish.
function colorFor(i, n) {
  const h = (i * 360) / Math.max(n, 1);
  return `hsl(${h}, 50%, 50%)`;
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
  return (
    <div style={{ height, display: "flex", overflow: "hidden", background: "#f0f0f0" }}>
      {selected !== null ? (
        <>
          <div
            style={{
              width: `${(row[selected] ?? 0) * 100}%`,
              background: colorFor(selected, n),
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
                background: colorFor(ti, n),
                transition: "width .3s",
                opacity: isMLP ? 0.7 : 1,
              }}
            />
          ))
      )}
    </div>
  );
}

export default function InfoFlow() {
  const [prompt, setPrompt] = useState("the cat sat");
  const [data, setData] = useState(null); // {tokens, attention_norms, mlp_norms}
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);
  // Source indices whose contribution is suppressed and redistributed.
  // First token hidden by default (BOS / leading token tends to dominate).
  const [hidden, setHidden] = useState(() => new Set([0]));

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
      const res = await fetch(`/api?prompt=${encodeURIComponent(prompt)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
      // New prompt = new tokens; reset hide state to "first token hidden".
      setHidden(new Set([0]));
    } catch (e) {
      setError(String(e));
      setData(null);
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
  const CW = N > 0 ? Math.max(40, Math.min(90, Math.floor(600 / N))) : 60;
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

          {/* grid */}
          <div style={{ overflowX: "auto" }}>
            {rows.map((row, ri) => {
              const isMLP = row.type === "mlp";
              const rh = isMLP ? MH : AH;
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
                      fontSize: isMLP ? 8 : 10,
                      color: isMLP ? "#bbb" : "#999",
                    }}
                  >
                    {isMLP ? `mlp${row.layer}` : `L${row.layer}`}
                  </div>
                  {row.dist.map((posRow, pos) => (
                    <div key={pos} style={{ width: CW, padding: "0 2px" }}>
                      <Bar row={posRow} height={rh} selected={selected} n={N} isMLP={isMLP} />
                    </div>
                  ))}
                </div>
              );
            })}

            {/* token axis labels at bottom */}
            <div style={{ display: "flex", marginLeft: LW, marginTop: 6 }}>
              {tokens.map((tok, i) => {
                const isHidden = hidden.has(i);
                return (
                  <button
                    key={i}
                    onClick={() => !isHidden && setSelected(selected === i ? null : i)}
                    disabled={isHidden}
                    style={{
                      width: CW,
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
                );
              })}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
