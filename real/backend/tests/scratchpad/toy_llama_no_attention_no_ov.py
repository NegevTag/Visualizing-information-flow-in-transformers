"""Tiny Llama-architecture model for compile-checks (~27k params, no download).

Same architectural shape as real Llama (GQA, RMSNorm, SiLU, RoPE) so any code
that compiles against this will compile against Llama-3.1-8B too.

Usage:
    from tests.scratchpad.toy_llama import ToyLlama
    model = ToyLlama.build()                       # defaults
    model = ToyLlama.build(num_hidden_layers=4)    # override anything
"""

from toy_llama import ToyLlama
import torch
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
import nnsight




# Toy llama move delete
class ToyLlamaNoAttentionNoOV(ToyLlama):
    """Builder for a tiny Llama wrapped in nnsight.LanguageModel."""
    @classmethod
    def build_hf_model(cls, config_overrides, tokenizer):
        hf_model  = super().build_hf_model(config_overrides,tokenizer)
        with torch.no_grad():
            for layer in hf_model.model.layers:
                layer.self_attn.v_proj.weight.zero_()
                layer.self_attn.o_proj.weight.zero_()
                layer.mlp.gate_proj.weight.zero_()
                layer.mlp.up_proj.weight.zero_()
                layer.mlp.down_proj.weight.zero_()
        return hf_model