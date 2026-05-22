"""Tiny Llama-architecture model for compile-checks (~27k params, no download).

Same architectural shape as real Llama (GQA, RMSNorm, SiLU, RoPE) so any code
that compiles against this will compile against Llama-3.1-8B too.

Usage:
    from tests.scratchpad.toy_llama import ToyLlama
    model = ToyLlama.build()                       # defaults
    model = ToyLlama.build(num_hidden_layers=4)    # override anything
"""

from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
import nnsight


class ToyLlama:
    """Builder for a tiny Llama wrapped in nnsight.LanguageModel."""

    # Defaults: 2 layers, GQA ratio 2, eager attention so attn weights are materialised.
    DEFAULT_CONFIG = dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        attn_implementation="eager",
    )

    TOKENIZER_NAME = "hf-internal-testing/llama-tokenizer"

    @classmethod
    def build(cls, **config_overrides) -> nnsight.LanguageModel:
        """Build a tiny Llama, wrap in nnsight. Pass kwargs to override DEFAULT_CONFIG."""
        tok = AutoTokenizer.from_pretrained(cls.TOKENIZER_NAME)
        tok.pad_token = tok.eos_token
        # vocab_size must match the tokenizer so token IDs index valid embedding rows.
        cfg_kwargs = {**cls.DEFAULT_CONFIG, **config_overrides, "vocab_size": tok.vocab_size}
        config = LlamaConfig(**cfg_kwargs)
        hf_model = LlamaForCausalLM(config)
        return nnsight.LanguageModel(hf_model, tokenizer=tok)
