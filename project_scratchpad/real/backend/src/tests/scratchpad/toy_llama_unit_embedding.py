"""Tiny Llama-architecture model for compile-checks (~27k params, no download).

Same architectural shape as real Llama (GQA, RMSNorm, SiLU, RoPE) so any code
that compiles against this will compile against Llama-3.1-8B too.

Usage:
    from tests.scratchpad.toy_llama import ToyLlama
    model = ToyLlama.build()                       # defaults
    model = ToyLlama.build(num_hidden_layers=4)    # override anything
"""

from tests.scratchpad.toy_llama import ToyLlama
import torch
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
import nnsight

"""Tiny Llama-architecture model for compile-checks (~27k params, no download).

Same architectural shape as real Llama (GQA, RMSNorm, SiLU, RoPE) so any code
that compiles against this will compile against Llama-3.1-8B too.

Usage:
    from tests.scratchpad.toy_llama import ToyLlama
    model = ToyLlama.build()                       # defaults
    model = ToyLlama.build(num_hidden_layers=4)    # override anything
"""

from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizer
import nnsight
import torch
import einops as ein

LlamaDType = torch.bfloat16



class EmbedUnitVecs(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, input_ids):  # [batch, p_len]
        unit_vecs = torch.eye(input_ids.shape[1], self.d_model, dtype=LlamaDType)  # [p_len,d_model]
        return ein.repeat(unit_vecs, "pl dm-> b pl dm", b=len(input_ids))


# Toy llama move delete
class ToyLllamaUnitEmbedding(ToyLlama):
    DEFAULT_CONFIG = dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=10,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        attn_implementation="eager",
    )
    """Builder for a tiny Llama wrapped in nnsight.LanguageModel."""

    @classmethod
    def build_hf_model(cls, config_overrides, tokenizer):
        hf_model = super().build_hf_model(config_overrides, tokenizer)
        with torch.no_grad():
            hf_model.model.embed_tokens = EmbedUnitVecs(hf_model.config.hidden_size)
        return hf_model
