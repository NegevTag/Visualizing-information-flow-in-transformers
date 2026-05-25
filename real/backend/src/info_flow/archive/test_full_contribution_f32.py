import hashlib
from pathlib import Path

import torch
from info_flow.config import Config
from info_flow.ex4_full_contribution_f32 import ModelInformationCalculator, _get_model
from icecream import ic


def _cache_path(tag: str, prompt: str, model_name: str) -> Path:
    # Cache key = hash(model + prompt) so changing either invalidates the cache.
    key = hashlib.sha1(f"{model_name}||{prompt}".encode()).hexdigest()[:16]
    cache_dir = Path(__file__).resolve().parent / ".cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{tag}_{key}.pt"


def get_real_activations(model, prompt: str, model_name: str):
    """Real per-layer attn/mlp outputs + post-RMSNorm-1 residual stream.
    Always cached — recomputed only if cache miss."""
    cache = _cache_path("real", prompt, model_name)
    if cache.exists():
        print(f"[cache] loading real outputs from {cache.name}")
        blob = torch.load(cache, weights_only=False)
        return blob["mlp"], blob["attn"]

    with model.trace(prompt, remote=True):
        mlp_outs: list = list().save()
        attn_outs: list = list().save()
        for layer in model.model.layers:
            attn_outs.append(layer.post_attention_layernorm.input.save()[0])
            mlp_outs.append(layer.output[0].save())
    mlp_outs = [t.detach().cpu() for t in mlp_outs]
    attn_outs = [t.detach().cpu() for t in attn_outs]
    torch.save({"mlp": mlp_outs, "attn": attn_outs,}, cache)
    print(f"[cache] saved real outputs to {cache.name}")
    return mlp_outs, attn_outs


def get_my_contributions(information_calculator, prompt: str, model_name: str, use_cache: bool):
    """My decomposition. Honors USE_CACHE — skips cache and recomputes if False."""
    cache = _cache_path("mine", prompt, model_name)
    if use_cache and cache.exists():
        print(f"[cache] loading my contributions from {cache.name}")
        blob = torch.load(cache, weights_only=False)
        return blob["post_mlp"], blob["post_attn"]

    contributions = information_calculator.calc(prompt)
    post_mlp = [t.detach().cpu() for t in contributions.post_mlp_contribution]
    post_attn = [t.detach().cpu() for t in contributions.post_attention_contribution]
    torch.save({"post_mlp": post_mlp, "post_attn": post_attn}, cache)
    print(f"[cache] saved my contributions to {cache.name}")
    return post_mlp, post_attn


def _prompt_use_cache() -> bool:
    # Only matters if a cache file already exists for this (model, prompt).
    ans = input("Use cached 'mine' contributions if available? [Y/n]: ").strip().lower()
    return ans in ("", "y", "yes")


if __name__ == "__main__":
    USE_CACHE = _prompt_use_cache()  # only affects my contributions; real activations are always cached
    config = Config()
    model = _get_model(config.info_flow_model, config.hf_token)
    information_calculator = ModelInformationCalculator(model=model)
    prompt = "The cat sat on the mat, and then afterword, he decided that "

    layers_actual_mlp_output, layers_actual_attention_output = get_real_activations(
        model, prompt, config.info_flow_model
    )
    post_mlp_contribution, post_attention_contribution = get_my_contributions(
        information_calculator, prompt, config.info_flow_model, use_cache=USE_CACHE
    )

    for l in range(len(model.model.layers)):  # (prompt_len(query),prompt_len(key),d_model)
        my_layer_mlp_residual = post_mlp_contribution[l].sum(dim=1)
        real_layer_mlp_residual = layers_actual_mlp_output[l].float()

        my_layer_attention_residual = post_attention_contribution[l].sum(dim=1)
        real_layer_attention_residual = layers_actual_attention_output[l].float()

        assert torch.is_same_size(my_layer_mlp_residual, real_layer_mlp_residual), "Mlp not same size"
        assert torch.is_same_size(my_layer_attention_residual, real_layer_attention_residual), "Attention not the same size"

        
        diff_attention = my_layer_attention_residual - real_layer_attention_residual
        rel_norm_diff_attn = ic(diff_attention.norm()/real_layer_attention_residual.norm())
        ic(rel_norm_diff_attn)
        ic((diff_attention.abs() -config.default_rtol*real_layer_attention_residual.abs()).max())
        assert rel_norm_diff_attn < 4e-3

        assert torch.allclose(
            my_layer_attention_residual,
            real_layer_attention_residual,
            atol=config.default_atol,
            rtol=config.default_rtol,
        ), f"Attention not close full diff {((my_layer_attention_residual-real_layer_attention_residual).abs() - config.default_rtol * real_layer_attention_residual.abs()).max()}"
        
        diff_mlp = my_layer_mlp_residual - real_layer_mlp_residual
        rel_norm_diff_mlp = diff_mlp.norm()/real_layer_mlp_residual.norm()
        ic(rel_norm_diff_mlp)
        ic((diff_mlp.abs() -config.default_rtol*real_layer_mlp_residual.abs()).max())
        assert rel_norm_diff_mlp < 4e-3
        assert torch.allclose(
            my_layer_mlp_residual,
            real_layer_mlp_residual,
            atol=config.default_atol,
            rtol=config.default_rtol,
        ), f"MLP not close max diff {((my_layer_mlp_residual-real_layer_mlp_residual).abs() - config.default_rtol * real_layer_mlp_residual.abs()).max()}"
        
        print(f"Layer {l} passed")
        
        
