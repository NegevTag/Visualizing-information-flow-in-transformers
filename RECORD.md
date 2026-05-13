# Research notebook

Append-only log of what I tried, what worked, what didn't, and *why* — including
failed approaches and the lesson learned. Newest entries at the bottom.

---

## 2026-05-13 — SwiGLU lesson (Q3 design discussion, MLP handling)

> When asked how to handle the MLP nonlinearity, I jumped to a generic textbook
> tool — full Jacobian linearization at the actual activation — and offered it
> as the recommended option. I did this **without first writing out SwiGLU's
> explicit form on the page**. When the supervisor asked for a formal
> explanation, I had to write $\mathrm{MLP}(x) = W_{\mathrm{down}} \cdot
> (\mathrm{SiLU}(W_{\mathrm{gate}} \cdot x) \odot (W_{\mathrm{up}} \cdot x))$
> properly. At that point the structure-specific option — *freeze the gate*,
> i.e. treat $s^* = \mathrm{SiLU}(W_{\mathrm{gate}} \cdot x^*)$ as a constant so
> the MLP becomes the linear map $W_{\mathrm{down}} \cdot \mathrm{diag}(s^*)
> \cdot W_{\mathrm{up}}$, exactly linear, exact at $x^*$ — became immediate.
> This option (frozen-gate, A′) is strictly simpler than the Jacobian, is
> structurally symmetric with the frozen-QK approach we were already using
> elsewhere, and did **not** appear in my original menu.
>
> **Rule for future me: before reaching for a generic mathematical tool, read
> the actual math of the operation under analysis. Write out the architecture's
> explicit form first. Look for structure-specific simplifications first;
> generic tools second.** The "selection × write" factorization in SwiGLU
> (gate selects features, $W_{\mathrm{up}}/W_{\mathrm{down}}$ write them)
> mirrors the QK/OV split in attention — that pattern is everywhere in modern
> decoder transformers and is the first thing to look for.

---

## 2026-05-13 — Phase 0 executed (repo hygiene)

Moved synthetic React demo (`info_flow_demo.jsx`, `index.html`, `package.json`,
`package-lock.json`, `vite.config.js`, `src/`, `node_modules/`) into
`synthetic_demo/`. Deleted the duplicated `real/docs/decomposition_math.md`
in favor of the LaTeX source. Created this file plus `DECISIONS.md` and
top-level `README.md`. Branch: `phase/0-repo-hygiene`. Next: Phase 1 nnsight
probe.
