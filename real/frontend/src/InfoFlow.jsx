import { useEffect, useMemo, useRef, useState } from "react";
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

  // Grouping mode. Three states, all sharing the same source-space rendering
  // path (grouping is just a per-token → per-group index map):
  //   'none'   — no grouping. Bars show per-token contributions.
  //   'words'  — server-computed word grouping (POST /group_by_words).
  //   'custom' — client-editable mask, applied via POST /apply_mask.
  //
  // wordsData / customData cache the currently-applied grouped payload for
  // each mode so switching between modes is instant once fetched.
  const [mode, setMode] = useState("words");
  const [wordsData, setWordsData] = useState(null); // {attention_norms, mlp_norms, group}
  const [customData, setCustomData] = useState(null); // last-applied custom result
  // Currently-being-edited per-token group ids in custom mode. null when not
  // in custom mode (or before seeding).
  const [customMask, setCustomMask] = useState(null);
  const [groupLoading, setGroupLoading] = useState(false);
  // Custom-mode editing state.
  //  - pendingSelection: token indices currently rubber-band-selected but not
  //    yet assigned to a group.
  //  - dragRect: live rectangle {x0,y0,x1,y1} in trace-row coordinates while
  //    a rubber-band drag is in progress; null otherwise.
  //  - ctxMenu: right-click menu state {x,y,tokenIdx}, or null when closed.
  const [pendingSelection, setPendingSelection] = useState(() => new Set());
  const [dragRect, setDragRect] = useState(null);
  const [ctxMenu, setCtxMenu] = useState(null);
  const traceRowRef = useRef(null);
  const tokenRefs = useRef([]);

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
    // New prompt = new tokens = every cached grouped payload is stale.
    setWordsData(null);
    setCustomData(null);
    setCustomMask(null);
    setPendingSelection(new Set());
    setCtxMenu(null);
    // If user was in 'custom' with a mask specific to old tokens, snap back
    // to 'words' (the sensible default) rather than leaving a stale mask.
    if (mode === "custom") setMode("words");
    try {
      const url = `http://127.0.0.1:8000/?prompt=${encodeURIComponent(prompt)}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
      // New prompt = new tokens; reset hide state to "first token hidden".
      setHidden(new Set([0]));
      // Fetch the grouped payload for the current mode so the initial render
      // is already grouped. Sequential because these endpoints read
      // app.state.args.prompt which is set by `/`.
      const nextMode = mode === "custom" ? "words" : mode;
      if (nextMode === "words") {
        const gres = await fetch("http://127.0.0.1:8000/group_by_words", { method: "POST" });
        if (!gres.ok) throw new Error(`HTTP ${gres.status}`);
        setWordsData(await gres.json());
      }
    } catch (e) {
      setError(String(e));
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  // Switch grouping mode. Fetches on demand and caches per-mode; source-index
  // state (`selected`, `hidden`) is reset because index meanings differ across
  // masks.
  async function changeMode(next) {
    if (next === mode) return;
    setSelected(null);
    setHidden(new Set([0]));
    setPendingSelection(new Set());
    setCtxMenu(null);
    if (next === "none") {
      setMode("none");
      // Fire-and-forget: don't block UI, don't care if it fails.
      fetch("http://127.0.0.1:8000/ungroup", { method: "POST" }).catch(() => {});
      return;
    }
    if (next === "words") {
      if (!wordsData) {
        setGroupLoading(true);
        try {
          const res = await fetch("http://127.0.0.1:8000/group_by_words", { method: "POST" });
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          setWordsData(await res.json());
        } catch (e) {
          setError(String(e));
          setGroupLoading(false);
          return;
        }
        setGroupLoading(false);
      }
      setMode("words");
      return;
    }
    if (next === "custom") {
      // Seed the editable mask from whatever mask is currently active. If
      // words mode is on, start from the word mask (very common workflow:
      // words + a small tweak). Otherwise start from all-singletons.
      const seed =
        mode === "words" && wordsData
          ? wordsData.group.slice()
          : (data?.tokens ?? []).map((_, i) => i);
      setCustomMask(seed);
      // Auto-apply the seed so bars reflect the current custom mask without
      // requiring a manual click when the seed is non-trivial.
      setGroupLoading(true);
      try {
        const res = await fetch("http://127.0.0.1:8000/apply_mask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(seed),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        setCustomData(await res.json());
      } catch (e) {
        setError(String(e));
        setGroupLoading(false);
        return;
      }
      setGroupLoading(false);
      setMode("custom");
      return;
    }
  }

  // Reassign the currently-targeted tokens (pendingSelection if non-empty,
  // else the token that was right-clicked) into a group. `target === 'new'`
  // creates a fresh group id; otherwise it's an existing group id from the
  // current customMask.
  //
  // After assignment we renumber groups by first-occurrence so ids stay
  // contiguous [0, k). This keeps the palette dense and stable.
  function assignToGroup(target) {
    if (!customMask) return;
    const toks =
      pendingSelection.size > 0 ? [...pendingSelection] : ctxMenu ? [ctxMenu.tokenIdx] : [];
    if (toks.length === 0) return;
    const next = customMask.slice();
    const newId = target === "new" ? Math.max(-1, ...next) + 1 : target;
    for (const i of toks) next[i] = newId;
    // Renumber by first-occurrence so ids are contiguous starting at 0.
    const seen = new Map();
    let k = 0;
    for (const g of next) if (!seen.has(g)) seen.set(g, k++);
    const normalized = next.map((g) => seen.get(g));
    setCustomMask(normalized);
    setPendingSelection(new Set());
    setCtxMenu(null);
  }

  // Rubber-band: pointerdown on the trace-row background (not on a button)
  // starts a drag; pointermove updates the rect; pointerup hit-tests each
  // token wrapper's bbox against the rect and stores intersecting indices in
  // pendingSelection. Only active in custom mode.
  const onTraceRowPointerDown = (e) => {
    if (mode !== "custom") return;
    if (e.button !== 0) return; // left button only
    if (e.target.closest("button")) return; // let buttons handle their clicks
    const rect = traceRowRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setDragRect({ x0: x, y0: y, x1: x, y1: y });
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onTraceRowPointerMove = (e) => {
    if (!dragRect) return;
    const rect = traceRowRef.current.getBoundingClientRect();
    setDragRect((r) => r && { ...r, x1: e.clientX - rect.left, y1: e.clientY - rect.top });
  };
  const onTraceRowPointerUp = (e) => {
    if (!dragRect) return;
    const rect = traceRowRef.current.getBoundingClientRect();
    const minX = Math.min(dragRect.x0, dragRect.x1) + rect.left;
    const maxX = Math.max(dragRect.x0, dragRect.x1) + rect.left;
    const minY = Math.min(dragRect.y0, dragRect.y1) + rect.top;
    const maxY = Math.max(dragRect.y0, dragRect.y1) + rect.top;
    const hit = new Set();
    for (let i = 0; i < tokenRefs.current.length; i++) {
      const el = tokenRefs.current[i];
      if (!el) continue;
      const b = el.getBoundingClientRect();
      if (b.left < maxX && b.right > minX && b.top < maxY && b.bottom > minY) hit.add(i);
    }
    setPendingSelection(hit);
    setDragRect(null);
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {}
  };

  // Close the context menu on any click / escape outside of it. Bound only
  // while the menu is open to avoid a global always-on listener.
  useEffect(() => {
    if (!ctxMenu) return;
    const close = (e) => {
      // Ignore clicks inside the menu itself (they're the assign action).
      if (e.target && e.target.closest && e.target.closest("[data-ctx-menu]")) return;
      setCtxMenu(null);
    };
    const key = (e) => e.key === "Escape" && setCtxMenu(null);
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", key);
    };
  }, [ctxMenu]);

  // Apply the current customMask by hitting /apply_mask. Used by the "apply"
  // button in custom mode after the user has edited the mask.
  async function applyCustomMask() {
    if (mode !== "custom" || !customMask) return;
    setGroupLoading(true);
    setSelected(null);
    setHidden(new Set([0]));
    try {
      const res = await fetch("http://127.0.0.1:8000/apply_mask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(customMask),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setCustomData(await res.json());
    } catch (e) {
      setError(String(e));
    } finally {
      setGroupLoading(false);
    }
  }

  // Grouping-aware source-space derivation.
  //
  // - `grouping`: length-N list where grouping[i] = source index (group id) of
  //   token i, or null if ungrouped.
  // - `numSources`: size of the source dim in the norm arrays (num_groups when
  //   grouped, num_tokens otherwise). Used for palette scaling and Bar's `n`.
  // - `tokenToSource[i]`: source index that token i maps to. Identity when
  //   ungrouped; equals `grouping[i]` when grouped. All color / click / hide
  //   logic on the token-side of the UI maps through this so tokens in the
  //   same word share color and interactions.
  const tokens = data?.tokens ?? [];
  const N = tokens.length;
  // Effective grouped payload for the current mode. null in 'none' mode or
  // when the mode's cache hasn't been fetched yet (falls back to ungrouped).
  const effectiveGrouped =
    mode === "words" ? wordsData : mode === "custom" ? customData : null;
  const grouping = effectiveGrouped ? effectiveGrouped.group : null;
  const numSources = grouping ? Math.max(...grouping) + 1 : N;
  const tokenToSource = grouping ?? tokens.map((_, i) => i);
  const attnN = effectiveGrouped ? effectiveGrouped.attention_norms : data?.attention_norms;
  const mlpN = effectiveGrouped ? effectiveGrouped.mlp_norms : data?.mlp_norms;

  // Custom mode: "dirty" iff customMask (what the user has edited) differs
  // from what /apply_mask last returned. Drives the "apply" button state.
  const customDirty =
    mode === "custom" &&
    customMask &&
    (!customData ||
      customData.group.length !== customMask.length ||
      customMask.some((g, i) => customData.group[i] !== g));

  // Build a flat list of rows: [attn_0, mlp_0, attn_1, mlp_1, ...]; then reverse
  // so output sits on top and input on bottom, matching the synthetic demo.
  const rows = useMemo(() => {
    if (!attnN || !mlpN) return [];
    const L = attnN.length;
    const list = [];
    for (let l = 0; l < L; l++) {
      list.push({
        type: "attn",
        layer: l,
        dist: attnN[l].map((r) => applyHidden(normalizeRow(r), hidden)),
      });
      list.push({
        type: "mlp",
        layer: l,
        dist: mlpN[l].map((r) => applyHidden(normalizeRow(r), hidden)),
      });
    }
    return list.reverse();
  }, [attnN, mlpN, hidden]);

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
          {/* grouping controls — mutually-exclusive radio picks which mask
              indexes the source dim of the norm arrays. Custom exposes an
              editable per-token mask (right-click on a token or the trace-row
              rubber-band to select tokens, then assign to a new/existing
              group; "apply" pushes the mask to the backend). */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 14,
              marginBottom: 12,
              fontFamily: MONO,
              fontSize: 10,
              color: "#666",
            }}
          >
            <span style={{ color: "#bbb", textTransform: "uppercase", letterSpacing: ".08em" }}>
              group:
            </span>
            {[
              ["none", "none"],
              ["words", "by words"],
              ["custom", "custom"],
            ].map(([val, label]) => (
              <label
                key={val}
                style={{ display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}
              >
                <input
                  type="radio"
                  name="group-mode"
                  value={val}
                  checked={mode === val}
                  disabled={groupLoading}
                  onChange={() => changeMode(val)}
                  style={{ margin: 0 }}
                />
                {label}
              </label>
            ))}
            {groupLoading && <span style={{ color: "#aaa" }}>…</span>}
            {mode === "custom" && (
              <button
                onClick={applyCustomMask}
                disabled={!customDirty || groupLoading}
                title={customDirty ? "apply mask to backend" : "no changes since last apply"}
                style={{
                  fontFamily: MONO,
                  fontSize: 10,
                  padding: "3px 10px",
                  border: `1px solid ${customDirty ? "#333" : "#ddd"}`,
                  background: customDirty ? "#fff" : "#fafafa",
                  color: customDirty ? "#111" : "#bbb",
                  cursor: customDirty ? "pointer" : "default",
                  marginLeft: "auto",
                }}
              >
                {customDirty ? "apply •" : "apply"}
              </button>
            )}
          </div>

          {/* token legend — click to trace. In custom mode, this row also
              accepts rubber-band selection (drag on empty space) + right-click
              on tokens to open the "add to group" menu. */}
          <div
            ref={traceRowRef}
            onPointerDown={onTraceRowPointerDown}
            onPointerMove={onTraceRowPointerMove}
            onPointerUp={onTraceRowPointerUp}
            onPointerCancel={onTraceRowPointerUp}
            style={{
              position: "relative",
              display: "flex",
              gap: 14,
              marginBottom: 14,
              alignItems: "center",
              flexWrap: "wrap",
              cursor: mode === "custom" ? (dragRect ? "crosshair" : "cell") : "default",
              userSelect: mode === "custom" ? "none" : "auto",
            }}
          >
            <span style={{ fontSize: 9, color: "#bbb", textTransform: "uppercase", letterSpacing: ".08em" }}>
              trace:
            </span>
            {tokens.map((tok, i) => {
              // In grouped mode, s is the group id for token i, so tokens in
              // the same word share color, selected state, and hidden state.
              const s = tokenToSource[i];
              const active = selected === s;
              const isHidden = hidden.has(s);
              const dim = selected !== null && !active;
              const pending = pendingSelection.has(i);
              return (
                <div
                  key={i}
                  ref={(el) => {
                    tokenRefs.current[i] = el;
                  }}
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    gap: 2,
                    padding: "2px 4px",
                    // Dashed outline for tokens picked by the rubber-band; a
                    // transparent border in the default state so the layout
                    // doesn't jump when the outline appears.
                    border: pending ? "1px dashed #369" : "1px dashed transparent",
                    background: pending ? "rgba(50, 100, 200, 0.06)" : "transparent",
                  }}
                >
                  <button
                    onClick={() => !isHidden && setSelected(active ? null : s)}
                    onContextMenu={(e) => {
                      if (mode !== "custom") return; // let the browser show its menu
                      e.preventDefault();
                      setCtxMenu({ x: e.clientX, y: e.clientY, tokenIdx: i });
                    }}
                    disabled={isHidden}
                    style={{
                      fontFamily: MONO,
                      fontSize: 12,
                      fontWeight: active ? 700 : 500,
                      color: isHidden ? "#ccc" : dim ? "#ddd" : colorFor(s, numSources),
                      textDecoration: isHidden ? "line-through" : "none",
                      background: "none",
                      border: "none",
                      padding: 0,
                      cursor: isHidden ? "default" : "pointer",
                      borderBottom: active ? `2px solid ${colorFor(s, numSources)}` : "2px solid transparent",
                    }}
                  >
                    {tok}
                  </button>
                  <button
                    onClick={() => toggleHidden(s)}
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
            {/* live rubber-band rectangle (custom mode only, while dragging) */}
            {dragRect && (
              <div
                style={{
                  position: "absolute",
                  left: Math.min(dragRect.x0, dragRect.x1),
                  top: Math.min(dragRect.y0, dragRect.y1),
                  width: Math.abs(dragRect.x1 - dragRect.x0),
                  height: Math.abs(dragRect.y1 - dragRect.y0),
                  background: "rgba(50, 100, 200, 0.08)",
                  border: "1px dashed #369",
                  pointerEvents: "none",
                }}
              />
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
                      <Bar row={posRow} height={rh} selected={selected} n={numSources} isMLP={styleAsMLP} />
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
                const s = tokenToSource[i];
                const isHidden = hidden.has(s);
                const dim = selected !== null && selected !== s;
                return (
                  <div key={i} style={{ width: CW, padding: "0 2px" }}>
                    <div
                      style={{
                        height: 2,
                        background: isHidden || dim ? "#eee" : colorFor(s, numSources),
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
                const s = tokenToSource[i];
                const isHidden = hidden.has(s);
                return (
                  // Wrapper matches the bar cell (width: CW, padding: 0 2px)
                  // so labels line up exactly with the columns above.
                  <div key={i} style={{ width: CW, padding: "0 2px" }}>
                    <button
                      onClick={() => !isHidden && setSelected(selected === s ? null : s)}
                      disabled={isHidden}
                      style={{
                        width: "100%",
                        fontFamily: MONO,
                        fontSize: 11,
                        fontWeight: 700,
                        color: isHidden
                          ? "#ccc"
                          : selected !== null && selected !== s
                          ? "#ccc"
                          : colorFor(s, numSources),
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

      {/* Right-click "add to group" menu. Rendered at the app root (not inside
          the trace row) so it isn't clipped by the flex layout. Positioned
          in viewport coordinates (position: fixed). */}
      {ctxMenu && customMask && (() => {
        const targets =
          pendingSelection.size > 0 ? [...pendingSelection] : [ctxMenu.tokenIdx];
        // Existing groups in the *pending* customMask, sorted by id for a
        // stable menu order. numCustomSources drives the swatch palette so
        // menu colors match what the bars will show once applied.
        const existing = [...new Set(customMask)].sort((a, b) => a - b);
        const numCustomSources = existing.length;
        const menuItem = (label, onClick, swatch) => (
          <button
            onMouseDown={(e) => e.stopPropagation()}
            onClick={onClick}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              width: "100%",
              padding: "4px 10px",
              fontFamily: MONO,
              fontSize: 10,
              textAlign: "left",
              background: "none",
              border: "none",
              cursor: "pointer",
            }}
            onMouseOver={(e) => (e.currentTarget.style.background = "#f5f5f5")}
            onMouseOut={(e) => (e.currentTarget.style.background = "none")}
          >
            {swatch}
            <span>{label}</span>
          </button>
        );
        return (
          <div
            data-ctx-menu
            style={{
              position: "fixed",
              top: ctxMenu.y,
              left: ctxMenu.x,
              minWidth: 140,
              background: "#fff",
              border: "1px solid #ccc",
              boxShadow: "0 2px 8px rgba(0,0,0,0.10)",
              zIndex: 1000,
              padding: "4px 0",
            }}
          >
            <div style={{ padding: "2px 10px", fontSize: 9, color: "#999", fontFamily: MONO }}>
              {targets.length === 1 ? "assign token" : `assign ${targets.length} tokens`} to:
            </div>
            {menuItem(
              "new group",
              () => assignToGroup("new"),
              <span style={{ width: 8, height: 8, border: "1px dashed #888", display: "inline-block" }} />
            )}
            {existing.map((g) => {
              // "current group" hint when the right-clicked token is already
              // in group g and no multi-selection is active.
              const isCurrent = targets.length === 1 && customMask[targets[0]] === g;
              return menuItem(
                `group ${g}${isCurrent ? " (current)" : ""}`,
                () => assignToGroup(g),
                <span
                  style={{
                    width: 8,
                    height: 8,
                    background: colorFor(g, numCustomSources),
                    display: "inline-block",
                  }}
                />
              );
            })}
          </div>
        );
      })()}
    </div>
  );
}
