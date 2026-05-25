"""Quick one-off: do I have permission to trace Llama-3.1-405B-Instruct on NDIF?

Runs the supervisor's exact snippet from a real file so `inspect.getsource()`
inside nnsight works (heredoc / -c fails because no source on disk).
"""

import os
import sys

from nnsight import CONFIG, LanguageModel

CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])

print("Constructing LanguageModel('meta-llama/Llama-3.1-405B-Instruct')...")
model = LanguageModel("meta-llama/Llama-3.1-405B-Instruct")
print("  ok — wrapper built")

print("Tracing 'The Eiffel Tower is in the city of' (remote=True)...")
try:
    with model.trace("The Eiffel Tower is in the city of", remote=True):
        out = model.output.save()
except Exception as e:
    print(f"  trace FAILED: {type(e).__name__}: {e}")
    sys.exit(2)

val = getattr(out, "value", out)
print(f"  ok — trace completed")
print(f"  model.output type: {type(val).__name__}")
if hasattr(val, "logits"):
    print(f"  logits.shape: {tuple(val.logits.shape)}")
    print(f"  logits.dtype: {val.logits.dtype}")
