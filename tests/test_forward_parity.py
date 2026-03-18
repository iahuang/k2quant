"""Test harness: compare QuantizableMoEBlock forward vs HF original.

Measures per-layer output divergence to determine if our reimplemented
forward path introduces floating-point differences that compound across
layers during calibration.

Three tests:
  1. Single-layer forward parity (output comparison)
  2. Routing parity (logits, softmax, top-k indices/weights)
  3. Accumulated divergence across all layers

Usage:
    python tests/test_forward_parity.py
"""

import os
import sys
import copy

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from k2quant.models.qwen_moe import QwenMoEBlock

# ── Configuration ────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B"
DEVICE = "cuda"
CACHE_DIR = "/workspace/kbvq/hf_cache"
os.environ["HF_HOME"] = CACHE_DIR

SEQ_LEN = 128  # tokens per test input


def stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Compute comparison stats between two tensors."""
    diff = (a.float() - b.float()).abs()
    a_flat = a.float().reshape(-1)
    b_flat = b.float().reshape(-1)
    cos = F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()
    denom = a.float().abs().clamp(min=1e-8)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": (diff / denom).max().item(),
        "cosine": cos,
    }


def fmt(s: dict) -> str:
    return (
        f"max_abs={s['max_abs']:.2e}  mean_abs={s['mean_abs']:.2e}  "
        f"max_rel={s['max_rel']:.2e}  cosine={s['cosine']:.10f}"
    )


# ── Test 1: Single-layer forward parity ──────────────────────────────────

def test_single_layer(model, layer_idx=0):
    print(f"\n{'='*70}")
    print(f"Test 1: Single-layer forward parity (layer {layer_idx})")
    print(f"{'='*70}")

    hf_mlp = model.model.layers[layer_idx].mlp

    # Create our block from the same HF module
    hf_params = next(hf_mlp.parameters())
    qmoe = QwenMoEBlock.from_hf_module(
        hf_mlp, device=hf_params.device, dtype=hf_params.dtype
    )

    # Random input matching what the decoder layer would pass
    x = torch.randn(1, SEQ_LEN, model.config.hidden_size,
                     device=DEVICE, dtype=torch.float16)

    with torch.no_grad():
        hf_out = hf_mlp(x)
        if isinstance(hf_out, tuple):
            hf_out = hf_out[0]

        our_out = qmoe(x)

    s = stats(hf_out, our_out)
    print(f"  Output: {fmt(s)}")
    return s


# ── Test 2: Routing parity ───────────────────────────────────────────────

def test_routing(model, layer_idx=0):
    print(f"\n{'='*70}")
    print(f"Test 2: Routing parity (layer {layer_idx})")
    print(f"{'='*70}")

    hf_mlp = model.model.layers[layer_idx].mlp
    hf_params = next(hf_mlp.parameters())
    qmoe = QwenMoEBlock.from_hf_module(
        hf_mlp, device=hf_params.device, dtype=hf_params.dtype
    )

    x = torch.randn(1, SEQ_LEN, model.config.hidden_size,
                     device=DEVICE, dtype=torch.float16)
    x_flat = x.reshape(-1, model.config.hidden_size)

    with torch.no_grad():
        # HF routing path: gate is a custom module
        hf_logits, hf_scores, hf_indices = hf_mlp.gate(x_flat)

        # Our routing path
        our_logits = F.linear(x_flat, qmoe.router)
        our_softmax = F.softmax(our_logits, dim=-1, dtype=torch.float)
        our_scores, our_indices = torch.topk(our_softmax, k=qmoe.top_k, dim=-1)
        our_scores = our_scores.to(our_softmax.dtype)

    # Router logits (pre-softmax)
    s_logits = stats(hf_logits, our_logits)
    print(f"  Logits (pre-softmax): {fmt(s_logits)}")

    # Softmax outputs — HF gate returns post-softmax as first element
    s_softmax = stats(hf_logits, our_softmax)
    print(f"  Softmax outputs:      {fmt(s_softmax)}")

    # Top-k indices match?
    # Sort within each token's top-k to ignore ordering
    hf_sorted = hf_indices.sort(dim=-1).values
    our_sorted = our_indices.sort(dim=-1).values
    idx_match = (hf_sorted == our_sorted).all().item()
    idx_match_pct = (hf_sorted == our_sorted).float().mean().item() * 100
    print(f"  Top-k indices match:  {idx_match} ({idx_match_pct:.1f}%)")

    # Top-k weights
    s_weights = stats(hf_scores, our_scores)
    print(f"  Top-k weights:        {fmt(s_weights)}")


# ── Test 3: Accumulated divergence ───────────────────────────────────────

def test_accumulated(model):
    print(f"\n{'='*70}")
    print(f"Test 3: Accumulated divergence across layers")
    print(f"{'='*70}")

    num_layers = model.config.num_hidden_layers

    # Random input tokens (use fixed seed for reproducibility)
    torch.manual_seed(42)
    input_ids = torch.randint(0, model.config.vocab_size, (1, SEQ_LEN),
                              device=DEVICE)

    # ── Pass A: Original HF model, record MoE block inputs ──
    print("  Pass A: Recording MoE inputs with original HF forward...")
    hf_inputs = {}
    hooks = []

    for li in range(num_layers):
        storage = []
        hf_inputs[li] = storage

        def make_hook(s):
            def fn(_module, args, _kwargs, output):
                s.append(args[0].detach().clone())
                return output
            return fn

        h = model.model.layers[li].mlp.register_forward_hook(
            make_hook(storage), with_kwargs=True
        )
        hooks.append(h)

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    # ── Swap all MoE blocks ──
    print("  Swapping all MoE blocks to QwenMoEBlock...")
    original_mlps = {}
    for li in range(num_layers):
        hf_mlp = model.model.layers[li].mlp
        original_mlps[li] = hf_mlp
        hf_params = next(hf_mlp.parameters())
        qmoe = QwenMoEBlock.from_hf_module(
            hf_mlp, device=hf_params.device, dtype=hf_params.dtype
        )
        model.model.layers[li].mlp = qmoe

    # ── Pass B: Our blocks, record MoE block inputs ──
    print("  Pass B: Recording MoE inputs with our QwenMoEBlock forward...")
    our_inputs = {}
    hooks = []

    for li in range(num_layers):
        storage = []
        our_inputs[li] = storage

        def make_hook(s):
            def fn(_module, args, _kwargs, output):
                s.append(args[0].detach().clone())
                return output
            return fn

        h = model.model.layers[li].mlp.register_forward_hook(
            make_hook(storage), with_kwargs=True
        )
        hooks.append(h)

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    # ── Restore original modules ──
    for li in range(num_layers):
        model.model.layers[li].mlp = original_mlps[li]

    # ── Compare ──
    print(f"\n  {'Layer':<8} {'Max Abs':>12} {'Mean Abs':>12} {'Max Rel':>12} {'Cosine':>16}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*16}")

    for li in range(num_layers):
        a = hf_inputs[li][0]
        b = our_inputs[li][0]
        s = stats(a, b)
        print(
            f"  {li:<8} {s['max_abs']:>12.2e} {s['mean_abs']:>12.2e} "
            f"{s['max_rel']:>12.2e} {s['cosine']:>16.10f}"
        )


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")

    test_single_layer(model, layer_idx=0)
    test_single_layer(model, layer_idx=12)
    test_single_layer(model, layer_idx=23)
    test_routing(model, layer_idx=0)
    test_accumulated(model)


if __name__ == "__main__":
    main()
