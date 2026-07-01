# Backup — top-logit-contributions bar feature

Removed on 2026-07-01 because the backend implementation had a bug. Everything
needed to restore it lives here. All snippets are exactly as they were in
`real/frontend/src/InfoFlow.jsx` at time of removal.

## Backend endpoint (what the frontend expected)

- Route: `GET /top_logit_contributions?prompt=...`
- Response: `list[float]` of length N (num tokens). Each entry = contribution
  of that source token to the top output logit at the last position.

## Frontend pieces to restore

### 1. State hook

Add near the other `useState` calls:

```jsx
// Per-source contributions to the top output logit (list[float], length N).
// Fetched sequentially after /, so the backend's model-run cache is warm.
const [topLogitContribs, setTopLogitContribs] = useState(null);
```

### 2. Sequential fetch inside `run()`

After the `setData(json)` line, before `setHidden(...)`:

```jsx
// top_logit_contributions is best-effort — don't fail the whole view if it errors.
const logitRes = await fetch(url("/top_logit_contributions"));
setTopLogitContribs(logitRes.ok ? await logitRes.json() : null);
```

And in the `catch` block:

```jsx
setTopLogitContribs(null);
```

### 3. Render — logit row inside `<ZoomPanVanilla>`, above `rows.map`

```jsx
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
```

## Notes for restoring

- Restore *after* /-and-cache flow is confirmed working. Endpoint expects the
  backend cache to already have the run result for the current prompt.
- The `aggregateByGroup` helper (currently in InfoFlow.jsx) was added for a
  possible grouped-mode aggregation of this row. If grouping-by-words is on,
  decide the grouped-mode semantics before restoring (see chat 2026-07-01).
