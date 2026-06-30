# Related work: per-token decomposition / context-mixing in Transformers

**Purpose.** Place our `real/backend/src/api_checks/model_calculator.py` in the literature. Our method makes a Llama **decoder** forward pass linear in the input embeddings by **freezing every nonlinearity** at its real value (attention softmax, the SwiGLU gate `g_l`, and the RMSNorm scale `rms_factor`), giving an **exact** decomposition of the residual stream into per-source-token contributions of shape `(layer, position, source, d_model)`.

All summaries below were written from the actual paper PDFs (read page-by-page, not WebFetch). The common ancestor is **Kobayashi et al. (2020/2021)**, who first rewrote the attention output as a sum over source tokens of transformed value vectors and showed the *norm* of each — not the raw attention weight — is what matters. Everyone here builds on that.

The one distinction that matters most for us: **does the method keep an exact per-source decomposition all the way through (like ours), or does it collapse to scalar scores and stitch layers together with lossy attention rollout?**

| Method | What the contribution is | Covers the MLP? | Exact? | Setting |
|---|---|---|---|---|
| **ALTI** ('22) | scalar (L1 proximity) + rollout | no | no | BERT encoder |
| **GlobEnc** ('22) | scalar (L2 norm) + rollout | no | no | BERT encoder |
| **Value Zeroing** ('23) | scalar (cosine change under perturbation) | yes (re-runs layer) | no | BERT encoder |
| **DecompX** ('23) | **full vector, propagated** | yes (linearized) | **yes** | BERT encoder |
| **ALTI-Logit** ('23) | scalar logit update + rollback | no (MLP lumped) | no | **GPT-2 decoder** |
| **our code** | **full vector, propagated** | **yes (gate-freeze)** | **yes** | **Llama decoder** |

---

## DecompX (Modarressi et al., ACL 2023) — arXiv:2306.02873

**What it does.** The one that matches our method. Decomposes each token representation into an exact sum of per-input-token vectors and propagates those vectors through the entire encoder — including the nonlinear FFN and the classification head — by freezing the nonlinearities (attention weights fixed, LayerNorm denominator frozen, activation replaced by its exact per-point slope). The contributions sum exactly to the true representation at every step.

**Difference from ours.** Essentially none in *method* — ours is DecompX generalized to a decoder LLM. The differences are architectural: we use RMSNorm (no mean-centering) instead of LayerNorm; the SwiGLU **gate-freeze** is cleaner than their GELU slope trick (the gate is natively multiplicative, so freezing it is exact with no approximation); Llama is bias-free so we skip their bias-distribution machinery; and we handle GQA, which they never face. They stop at a classification head; we project to next-token logits. Notably, DecompX explicitly listed decoder LLMs (GPT-2, T5) as **out of scope** — that's the gap we fill.

**Did they draw conclusions?** Only empirical ones (their decomposition is more faithful than gradient/rollout baselines; FFN, biases, and the head each matter). They **inherit** the token-decomposition framing and never argue *why* tokens are the right unit.

---

## ALTI (Ferrando et al., EMNLP 2022) — arXiv:2203.04212

**What it does.** Measures how much each source token contributes to each token's representation in the attention block, then aggregates across layers with attention rollout. Its signature choice: instead of the L2-norm of a contribution vector, it uses **L1/Manhattan proximity** of the transformed source vector to the actual output — they argue L2 is unreliable because Transformer representations are highly anisotropic and a few outlier dimensions dominate squared norms.

**Difference from ours.** Big. ALTI (a) throws the vector away and keeps a **scalar** per token pair, (b) **ignores the MLP** entirely, and (c) stitches layers with **lossy rollout**. We keep the full vector, decompose the MLP exactly, and never roll back. Their useful warning for us: if we ever reduce our `(…, d_model)` contributions to a single importance number, prefer L1/cosine over L2.

**Did they draw conclusions?** Methodological, not conceptual: *raw attention weights are misleading, and vector orientation (not just norm) must be accounted for*. No claim about tokens being computational units.

---

## Value Zeroing (Mohebbi et al., EACL 2023) — arXiv:2301.12971

**What it does.** A **perturbation** method, not a decomposition. To measure how much token *j* contributes to token *i*, it zeros out *j*'s **value vector** and re-runs the layer, measuring the cosine change in *i*'s output. Zeroing only the value (not the input) is the clever part — it leaves the attention distribution and the token's own identity intact, isolating the mixing contribution. Because it re-runs the real layer, it implicitly covers every component including the FFN.

**Difference from ours.** Orthogonal in mechanism — it's a difference of two forward passes, not an additive decomposition, so contributions don't sum to anything and it isn't exact. This makes it a good **independent cross-check**: which source tokens Value Zeroing flags as important should qualitatively agree with which ones our exact decomposition gives large mass to. Their value-only idea also rhymes with our freezing the attention pattern (hold QK/softmax fixed, let only the value pathway carry per-source mass).

**Did they draw conclusions?** The closest of the group to a conceptual stance: by keeping residuals intact they argue a token's representation is a *carrier of accumulated context*, not an atomic unit, and they criticize norm-based mixing ratios as residual-dominated. But still no positive argument that tokens are *the* unit of computation.

---

## GlobEnc + ALTI-Logit (same lineage)

**GlobEnc (Modarressi et al., NAACL 2022) — arXiv:2205.03286.** The direct predecessor of DecompX (same first author). Extends Kobayashi's norm-based decomposition to the *whole encoder layer* (adds the second residual and output LayerNorm), scores contributions by **L2-norm**, and aggregates with rollout. It cannot decompose the FFN (non-linear) so it omits the FFN's direct effect. *Difference from ours:* scalar + rollout + no real FFN decomposition — superseded by DecompX, which it directly led to. *Conclusions:* methodological only ("incorporate the whole encoder layer," residuals are essential); no token-unit claim.

**ALTI-Logit (Ferrando et al., ACL 2023) — arXiv:2305.12535. The most relevant paper to us.** The only one targeting **decoder LLMs** (GPT-2, OPT, BLOOM) and attributing **next-token logits**. It uses the same frozen-LayerNorm linearization we do, and its per-token-per-layer logit projection through the unembedding is exactly what our `calc_logits_contributions` computes.

*Difference from ours — and we are strictly more exact.* Two gaps: (1) it **lumps the MLP** as a single per-position term and never splits it across source tokens — our SwiGLU gate-freeze does exactly that; (2) because intermediate residual streams are *mixtures* of input tokens, ALTI-Logit must **approximately trace contributions back** to the inputs using ALTI's rollout matrices. We never face this — by carrying the `source` axis exactly through every layer we keep input-token attribution exact, with no rollback. In one line: **our code is "ALTI-Logit made exact."**

*Did they draw conclusions?* This is the only paper to state the token-unit assumption **explicitly** ("each residual stream preserves its token identity throughout the layers") — and it does so precisely because the assumption *breaks*, which is why it adds ALTI to patch it. Honest, but still a pragmatic assumption they work around, not an argument that tokens are the right unit.

---

## Bottom line for our project

The frozen-nonlinearity decomposition itself is **not new** (DecompX, 2023). What's unoccupied is the **exact** version on a **modern decoder LLM (RMSNorm + SwiGLU + GQA)** attributing **logits** per source token: DecompX is encoder-only and skipped decoders; ALTI-Logit is a decoder but approximate (lumped MLP + rollback). Our code is the intersection — "DecompX-exact, in ALTI-Logit's decoder/logit regime."

The conceptual question you raised — *why are tokens the right unit of computation?* — is answered by **none** of them. They all inherit the token basis from Kobayashi. ALTI-Logit is the only one even to name the assumption, and only because it fails. Our exactness removes the *measurement* approximation but not the *interpretive* question; that argument belongs to the residual-stream / mechanistic-interpretability literature (Elhage et al., "the residual stream is the communication channel") and is worth recording in `DECISIONS.md` before we lean on the visualization.

### Sources
- DecompX — https://arxiv.org/abs/2306.02873 · code https://github.com/mohsenfayyaz/DecompX
- ALTI — https://arxiv.org/abs/2203.04212 · code https://github.com/mt-upc/transformer-contributions
- Value Zeroing — https://arxiv.org/abs/2301.12971
- GlobEnc — https://arxiv.org/abs/2205.03286
- ALTI-Logit — https://arxiv.org/abs/2305.12535
- Kobayashi et al. 2020 / 2021 — the foundational decomposition this family builds on
