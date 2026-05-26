# Phase 1 probe

Three scripts that, together, verify nnsight on a Llama-architecture model exposes everything the per-source residual decomposition needs (see `real/docs/decomposition_math.tex`). Throwaway in spirit — but kept in the repo for reproducibility.

## Files

- `_common.py` — shared config: env-var-driven model/prompt/device, save-tensor helper that re-flushes a manifest after every write.
- `01_load_check.py` — load the model only. Prints architecture (L, H, H_kv, d, d_ff) and writes `run_config.json` / `model_config.json`. ~30–60s. Use this to confirm auth, attn implementation, and that the model fits in memory before paying for a full forward pass.
- `02_capture.py` — load + one forward pass + write every needed tensor to its own `.pt` under `probe_output/<model>/`. Manifest flushed after every save; if killed mid-run, prior files are still valid.
- `03_inspect.py` — read whatever `.pt` files exist on disk. No model load. Prints shapes, basic stats, and sanity checks (attention rows sum to 1, residual norm trend, causality, RMSNorm scale positivity).

## Run

Environment overrides:

```
INFO_FLOW_MODEL   default local : meta-llama/Llama-3.2-3B (gated; needs HF login)
                  default remote: meta-llama/Meta-Llama-3.1-8B (on NDIF)
INFO_FLOW_PROMPT  default: "The cat sat" (3 content tokens)
INFO_FLOW_DEVICE  default: cuda > mps > cpu (ignored in remote mode)
INFO_FLOW_REMOTE  set to "1" to run on NDIF instead of locally
NDIF_API_KEY      required when INFO_FLOW_REMOTE=1 — get one at https://login.ndif.us
```

From `real/backend/`:

```bash
# quick load-only smoke test
INFO_FLOW_MODEL=Qwen/Qwen2.5-0.5B uv run python tests/scratchpad/probe/01_load_check.py

# full capture (one forward pass; writes per-tensor .pt files)
INFO_FLOW_MODEL=Qwen/Qwen2.5-0.5B uv run python tests/scratchpad/probe/02_capture.py

# inspection (no model load; reads disk)
INFO_FLOW_MODEL=Qwen/Qwen2.5-0.5B uv run python tests/scratchpad/probe/03_inspect.py
```

PowerShell equivalent for env var:

```powershell
$env:INFO_FLOW_MODEL = "Qwen/Qwen2.5-0.5B"
uv run python tests/scratchpad/probe/01_load_check.py
```

## Running on NDIF (remote, no local GPU needed)

NDIF hosts the model on their servers. Local memory + compute is essentially zero — only the captured tensors come back. Sign up at `https://login.ndif.us` for a free API key, then:

PowerShell:
```powershell
$env:NDIF_API_KEY    = "<your_key>"
$env:INFO_FLOW_REMOTE = "1"
$env:INFO_FLOW_MODEL = "meta-llama/Meta-Llama-3.1-8B"
uv run python tests/scratchpad/probe/01_load_check.py
uv run python tests/scratchpad/probe/02_capture.py
uv run python tests/scratchpad/probe/03_inspect.py
```

bash:
```bash
export NDIF_API_KEY="..."
export INFO_FLOW_REMOTE=1
export INFO_FLOW_MODEL=meta-llama/Meta-Llama-3.1-8B
uv run python tests/scratchpad/probe/01_load_check.py
uv run python tests/scratchpad/probe/02_capture.py
uv run python tests/scratchpad/probe/03_inspect.py
```

Caveats:
- NDIF picks the device/dtype/attn implementation. We can't force `attn_implementation="eager"` remotely, which means the strategy for capturing attention probabilities may differ from the local path.
- NDIF hosts the Llama-3.1 family + DeepSeek-R1 at time of writing — not Llama-3.2. Check `https://nnsight.net/status/` for the live model list.

## Running on Colab (GPU)

CPU on a 3B model is painful. Colab gives us a free T4 (or A100 / L4 on Pro) which makes the forward pass and weight download an order of magnitude faster. Paste into one cell:

```python
!git clone https://github.com/NegevTag/Visualizing-information-flow-in-transformers.git iflow
%cd iflow
!git checkout phase/1-nnsight-probe
%cd real/backend
!pip install -q -e .

# If running on a gated model (Llama, Mistral...), authenticate first:
# !huggingface-cli login   # paste a `Read` token

import os
os.environ["INFO_FLOW_MODEL"]  = "Qwen/Qwen2.5-0.5B"     # or "meta-llama/Llama-3.2-3B"
os.environ["INFO_FLOW_PROMPT"] = "The cat sat"

!python tests/scratchpad/probe/01_load_check.py
!python tests/scratchpad/probe/02_capture.py
!python tests/scratchpad/probe/03_inspect.py
```

Pull captured tensors back to local (after the run):

```python
!tar czf probe_output.tgz tests/scratchpad/probe/probe_output
from google.colab import files
files.download("probe_output.tgz")
```

## Output layout

```
tests/scratchpad/probe/probe_output/<model-slug>/
    run_config.json
    model_config.json
    manifest.json                 # rewritten after every save
    embed_tokens.output.pt
    model.norm.input.pt
    L00.pre_attn_resid.pt
    L00.pre_mlp_resid.pt
    L00.post_layer_out.pt
    L00.attn_sublayer_out.pt
    L00.attn_weights.pt           # [B, H, S, S]
    L00.gate_post_silu.pt         # [B, S, d_ff]
    L00.rmsnorm_r1.pt             # [B, S]
    L00.rmsnorm_r2.pt
    L01.…
    …
```
