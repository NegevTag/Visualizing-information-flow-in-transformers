"""
Step 02 — capture: run one forward pass on PROMPT, save every quantity needed
by the decomposition to its own .pt file under probe_output/<model>/.

Captured per layer:
    pre_attn_resid    : input to input_layernorm        =  x^(l)_p
    pre_mlp_resid     : input to post_attention_layernorm = post-attention residual
    post_layer_out    : output of the whole transformer layer = x^(l+1)_p
    attn_sublayer_out : the attention sublayer's output tensor
    attn_weights      : the per-head attention probabilities  A^(l,h)_{p,s}
    gate_post_silu    : mlp.act_fn(...) output            =  g^(l)_p ∈ R^{d_ff}
    rmsnorm_r1        : 1 / sqrt(mean(pre_attn^2) + eps)
    rmsnorm_r2        : 1 / sqrt(mean(pre_mlp ^2) + eps)

Plus one-off:
    embed_tokens.output : token embeddings  e_p
    model.norm.input    : input to the final RMSNorm (= last layer output)

Each tensor is written to a separate .pt file with the manifest re-flushed
after every successful write. If the script crashes mid-loop, everything
written so far is still valid; rerun and it will pick up where it left off
(it doesn't actually resume — it just re-overwrites files; but earlier writes
are not lost).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running directly:
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from _common import (  # noqa: E402
    MODEL_NAME,
    PROMPT,
    REMOTE,
    section,
    ensure_out_dir,
    print_banner,
    materialise,
    unwrap_first,
    save_tensor,
    write_json,
    configure_ndif,
    make_model,
)


def main() -> int:
    ensure_out_dir()
    banner = print_banner()
    write_json("run_config", banner)
    configure_ndif()

    # -- Load ---------------------------------------------------------------
    section("LOAD MODEL")
    t0 = time.time()
    try:
        model = make_model()
    except Exception as e:
        print(f"FAILED to load model: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    load_time = time.time() - t0
    print(f"loaded in {load_time:.1f}s" + ("  (remote — config only)" if REMOTE else ""))

    cfg = model.config
    L = cfg.num_hidden_layers
    H = cfg.num_attention_heads
    H_KV = getattr(cfg, "num_key_value_heads", H)
    d = cfg.hidden_size
    d_ff = cfg.intermediate_size
    print(f"L={L}  H_q={H}  H_kv={H_KV}  d={d}  d_ff={d_ff}")
    write_json(
        "model_config",
        {"L": L, "H": H, "H_KV": H_KV, "d": d, "d_ff": d_ff,
         "load_time_sec": load_time},
    )

    # -- Trace forward pass (one pass; save all proxies) --------------------
    section("TRACE FORWARD PASS")
    # NOTE: In transformers 5.x, the attention sublayer dispatches through
    # `attention_interface(...)` which can bypass the `self_attn.__call__`
    # Envoy nnsight wraps. Capturing `self_attn.output` directly raises a
    # MissedProviderError. Instead we capture the underlying nn.Linear
    # submodules (q/k/v/o_proj) — they ARE always called — and (if available)
    # the layer's tuple output, which carries `attn_weights` when
    # `output_attentions=True` is honored by the layer's forward.
    captures: dict[str, list] = {
        "pre_attn_resid":  [None] * L,
        "pre_mlp_resid":   [None] * L,
        "post_layer_out":  [None] * L,
        "q_proj":          [None] * L,
        "k_proj":          [None] * L,
        "v_proj":          [None] * L,
        "o_proj":          [None] * L,
        "gate_post_silu":  [None] * L,
    }
    embed_proxy = None
    final_norm_in_proxy = None

    t0 = time.time()
    # `remote=True` ships the trace to NDIF; otherwise it runs locally.
    with model.trace(PROMPT, remote=REMOTE):
        embed_proxy = model.model.embed_tokens.output.save()
        for li in range(L):
            layer = model.model.layers[li]
            captures["pre_attn_resid"][li] = layer.input_layernorm.input.save()
            captures["pre_mlp_resid"][li]  = layer.post_attention_layernorm.input.save()
            captures["post_layer_out"][li] = layer.output.save()
            # attention submodule projections (these always get called)
            captures["q_proj"][li] = layer.self_attn.q_proj.output.save()
            captures["k_proj"][li] = layer.self_attn.k_proj.output.save()
            captures["v_proj"][li] = layer.self_attn.v_proj.output.save()
            captures["o_proj"][li] = layer.self_attn.o_proj.output.save()
            captures["gate_post_silu"][li] = layer.mlp.act_fn.output.save()
        final_norm_in_proxy = model.model.norm.input.save()
    trace_time = time.time() - t0
    print(f"trace completed in {trace_time:.2f}s")

    # -- Persist to disk layer by layer; manifest re-flushed each time ------
    section("WRITE CAPTURES")
    print("[embeddings & final-norm input]")
    save_tensor("embed_tokens.output", materialise(embed_proxy))
    save_tensor("model.norm.input", unwrap_first(materialise(final_norm_in_proxy)))

    for li in range(L):
        print(f"[layer {li:02d}]")
        pre_attn = unwrap_first(materialise(captures["pre_attn_resid"][li]))
        pre_mlp  = unwrap_first(materialise(captures["pre_mlp_resid"][li]))
        post     = unwrap_first(materialise(captures["post_layer_out"][li]))
        gate     = materialise(captures["gate_post_silu"][li])
        attn_out = materialise(captures["attn_output"][li])

        save_tensor(f"L{li:02d}.pre_attn_resid", pre_attn)
        save_tensor(f"L{li:02d}.pre_mlp_resid", pre_mlp)
        save_tensor(f"L{li:02d}.post_layer_out", post)
        save_tensor(f"L{li:02d}.gate_post_silu", gate)

        # self_attn.output is (attn_out, attn_weights, past_kv) when output_attentions=True
        if isinstance(attn_out, tuple):
            if len(attn_out) >= 1 and isinstance(attn_out[0], torch.Tensor):
                save_tensor(f"L{li:02d}.attn_sublayer_out", attn_out[0])
            if len(attn_out) >= 2 and isinstance(attn_out[1], torch.Tensor):
                save_tensor(f"L{li:02d}.attn_weights", attn_out[1])
        elif isinstance(attn_out, torch.Tensor):
            save_tensor(f"L{li:02d}.attn_sublayer_out", attn_out)

        # Derived: RMSNorm scale factors r1, r2 = 1 / sqrt(mean(x^2) + eps)
        eps1 = model.model.layers[li].input_layernorm.variance_epsilon
        eps2 = model.model.layers[li].post_attention_layernorm.variance_epsilon
        r1 = 1.0 / (pre_attn.to(torch.float32).pow(2).mean(-1) + eps1).sqrt()
        r2 = 1.0 / (pre_mlp.to(torch.float32).pow(2).mean(-1) + eps2).sqrt()
        save_tensor(f"L{li:02d}.rmsnorm_r1", r1)
        save_tensor(f"L{li:02d}.rmsnorm_r2", r2)

    print(f"\nload={load_time:.1f}s  trace={trace_time:.2f}s")
    print("DONE (capture).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
