"""
Shared config / helpers for the Phase 1 nnsight probe scripts.

Reads environment overrides:
    INFO_FLOW_MODEL    : HuggingFace model id    (default: meta-llama/Llama-3.2-3B)
    INFO_FLOW_PROMPT   : input prompt            (default: "The cat sat", 3 content tokens)
    INFO_FLOW_DEVICE   : torch device override   (default: auto cuda > mps > cpu)

Output layout (one directory per model so we can run several models without collision):
    tests/scratchpad/probe/probe_output/<model-slug>/
        run_config.json     # model / device / dtype / versions snapshot
        model_config.json   # L, H, d, d_ff, ...
        manifest.json       # per-tensor stats; rewritten after every save_tensor()
        <name>.pt           # one file per saved tensor (see 02_capture.py)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Config (env-driven, no CLI args by design — easier to run from any shell)
# ---------------------------------------------------------------------------

MODEL_NAME: str = os.environ.get("INFO_FLOW_MODEL", "meta-llama/Llama-3.2-3B")
PROMPT: str = os.environ.get("INFO_FLOW_PROMPT", "The cat sat")

_THIS_DIR = Path(__file__).resolve().parent
MODEL_SLUG = MODEL_NAME.replace("/", "__")
OUT_DIR: Path = _THIS_DIR / "probe_output" / MODEL_SLUG
MANIFEST_PATH: Path = OUT_DIR / "manifest.json"


def pick_device() -> str:
    """cuda > mps > cpu, with INFO_FLOW_DEVICE override."""
    override = os.environ.get("INFO_FLOW_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    """bf16 on accelerators (memory), fp32 on cpu (bf16 ops are slow on CPU)."""
    if device in ("cuda", "mps"):
        return torch.bfloat16
    return torch.float32


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    sys.stdout.flush()


def stats(t: torch.Tensor) -> dict:
    """Cheap shape/dtype/device + finite check + min/max/mean (cast to f32 for stats)."""
    t_f32 = t.detach().to(torch.float32)
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "min": float(t_f32.min().item()),
        "max": float(t_f32.max().item()),
        "mean": float(t_f32.mean().item()),
        "finite": bool(torch.isfinite(t_f32).all().item()),
    }


# ---------------------------------------------------------------------------
# nnsight proxy unwrapping
# ---------------------------------------------------------------------------

def materialise(x):
    """After a trace block, nnsight saved-proxies expose `.value` (older API)
    or are themselves the tensor (newer API). Handle both."""
    return getattr(x, "value", x)


def unwrap_first(x):
    """`.input` proxies materialise to a tuple of positional args (typically of
    length 1). Pull the first element if so."""
    if isinstance(x, tuple) and len(x) > 0:
        return x[0]
    return x


# ---------------------------------------------------------------------------
# Disk I/O — every save flushes the manifest so partial runs stay usable
# ---------------------------------------------------------------------------

# In-process manifest; persisted after every successful tensor write.
_MANIFEST: dict[str, dict] = {}


def _load_manifest() -> None:
    """Restore previous-run manifest contents so resume-after-crash is sensible."""
    global _MANIFEST
    if MANIFEST_PATH.exists():
        try:
            _MANIFEST = json.loads(MANIFEST_PATH.read_text())
        except Exception:
            _MANIFEST = {}
    else:
        _MANIFEST = {}


def _persist_manifest() -> None:
    MANIFEST_PATH.write_text(json.dumps(_MANIFEST, indent=2))


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _load_manifest()


def save_tensor(name: str, t: torch.Tensor | None) -> None:
    """Save one tensor to its own .pt file; update manifest immediately.

    Called many times during 02_capture.py. Each call is independent — if
    the script dies before the next one, the file just written is still good.
    """
    if t is None:
        print(f"  SKIP {name}: None")
        return
    path = OUT_DIR / f"{name}.pt"
    torch.save(t.detach().cpu(), path)
    info = stats(t)
    _MANIFEST[name] = info
    _persist_manifest()
    print(
        f"  saved {path.name:42s} shape={tuple(info['shape'])} "
        f"dtype={info['dtype']} finite={info['finite']}"
    )
    sys.stdout.flush()


def write_json(name: str, payload: dict) -> None:
    """Write a metadata JSON to OUT_DIR/<name>.json."""
    (OUT_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2))


def read_json(name: str) -> dict | None:
    """Read OUT_DIR/<name>.json or return None if missing/invalid."""
    p = OUT_DIR / f"{name}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Environment / version banner — printed at the top of each script
# ---------------------------------------------------------------------------

def print_banner() -> dict:
    """Print + return a dict of versions/config for the manifest."""
    import nnsight
    import transformers
    device = pick_device()
    dtype = pick_dtype(device)
    banner = {
        "model": MODEL_NAME,
        "device": device,
        "dtype": str(dtype),
        "prompt": PROMPT,
        "torch": torch.__version__,
        "nnsight": nnsight.__version__,
        "transformers": transformers.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    section("PROBE CONFIG")
    for k, v in banner.items():
        print(f"  {k:12s}: {v}")
    print(f"  out_dir     : {OUT_DIR}")
    sys.stdout.flush()
    return banner
