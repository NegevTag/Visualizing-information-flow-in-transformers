"""Compare mod2 (parity) certainty of Llama-3.1 8B vs 70B, run remotely on NDIF."""

import torch
from dotenv import dotenv_values
from nnsight import CONFIG, LanguageModel

env = dotenv_values("real/backend/.env.local")
CONFIG.API.APIKEY = env["NDIF_API_KEY"]
HF_TOKEN = env["HF_TOKEN"]

MODELS = ["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-70B"]

# prompt -> correct mod2 answer
PROMPTS = {
    "(0 + 1 + 0 + 1)mod2 = ": 0,
    "(1 + 1 + 0 + 1)mod2 = ": 1,
    "(1 + 1 + 1 + 1)mod2 = ": 0,
}

for model_name in MODELS:
    model = LanguageModel(model_name, token=HF_TOKEN)
    tok = model.tokenizer
    id0 = tok.encode(" 0", add_special_tokens=False)[-1]  # token for the digit answer
    id1 = tok.encode(" 1", add_special_tokens=False)[-1]

    print(f"\n=== {model_name} ===")
    for prompt, answer in PROMPTS.items():
        with model.trace(prompt, remote=True):
            logits = model.lm_head.output[0, -1].save()
        probs = torch.softmax(logits.float(), dim=-1)
        p0, p1 = probs[id0].item(), probs[id1].item()
        p_correct = p0 if answer == 0 else p1
        print(f"{prompt}  true={answer}  P(0)={p0:.3f} P(1)={p1:.3f}  P(correct)={p_correct:.3f}")
        # what does the model ACTUALLY want to predict here?
        top_p, top_i = probs.topk(5)
        print("    top5:", [(tok.decode([i]), round(p, 3)) for i, p in zip(top_i.tolist(), top_p.tolist())])
