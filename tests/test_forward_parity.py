"""Test harness: compare QuantizableExperts dispatch vs HF original.

Measures per-layer output divergence to verify that our experts-only
replacement produces identical results to HF's experts.forward().

Since we now only replace the experts sub-module (not the full MoE block),
routing and shared expert remain HF's original code. This test verifies
that the expert dispatch loop itself is bit-identical.

Three tests:
  1. Single-layer experts dispatch parity
  2. Routing parity (should be trivially identical — we don't touch routing)
  3. Accumulated divergence across all layers (experts-only swap)

Usage:
    python tests/test_forward_parity.py
"""

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from k2quant.models.qwen_moe import QwenExperts

# -- Configuration --
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


# -- Test 1: Single-layer experts dispatch parity --

def test_single_layer(model, layer_idx=0):
    print(f"\n{'='*70}")
    print(f"Test 1: Single-layer MoE parity (layer {layer_idx})")
    print(f"{'='*70}")

    hf_mlp = model.model.layers[layer_idx].mlp

    # Swap only experts, keep HF routing and shared expert
    hf_experts_orig = hf_mlp.experts
    hf_params = next(hf_experts_orig.parameters())
    qe = QwenExperts.from_hf_module(
        hf_mlp, device=hf_params.device, dtype=hf_params.dtype
    )

    # Random input matching what the decoder layer would pass
    x = torch.randn(1, SEQ_LEN, model.config.hidden_size,
                     device=DEVICE, dtype=torch.float16)

    with torch.no_grad():
        # Pass A: original HF forward
        hf_out = hf_mlp(x)
        if isinstance(hf_out, tuple):
            hf_out = hf_out[0]

        # Pass B: swap experts, run same MoE block
        hf_mlp.experts = qe
        our_out = hf_mlp(x)
        if isinstance(our_out, tuple):
            our_out = our_out[0]

        # Restore original
        hf_mlp.experts = hf_experts_orig

    s = stats(hf_out, our_out)
    print(f"  Output: {fmt(s)}")
    return s


# -- Test 2: Routing parity (trivial — we don't touch routing) --

def test_routing(model, layer_idx=0):
    print(f"\n{'='*70}")
    print(f"Test 2: Routing parity (layer {layer_idx}) — should be bit-identical")
    print(f"{'='*70}")

    hf_mlp = model.model.layers[layer_idx].mlp

    x = torch.randn(1, SEQ_LEN, model.config.hidden_size,
                     device=DEVICE, dtype=torch.float16)
    x_flat = x.reshape(-1, model.config.hidden_size)

    with torch.no_grad():
        # HF routing
        hf_logits, hf_scores, hf_indices = hf_mlp.gate(x_flat)

    print(f"  Routing is HF's original code — no comparison needed.")
    print(f"  Gate output shape: logits={hf_logits.shape}, "
          f"scores={hf_scores.shape}, indices={hf_indices.shape}")


# -- Test 3: Accumulated divergence across all layers --

def test_accumulated(model):
    print(f"\n{'='*70}")
    print(f"Test 3: Accumulated divergence across layers (experts-only swap)")
    print(f"{'='*70}")

    num_layers = model.config.num_hidden_layers

    # Random input tokens
    torch.manual_seed(42)
    input_ids = torch.randint(0, model.config.vocab_size, (1, SEQ_LEN),
                              device=DEVICE)

    # -- Pass A: Original HF model, record MoE block inputs --
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

    # -- Swap all experts sub-modules --
    print("  Swapping all experts to QwenExperts...")
    original_experts = {}
    for li in range(num_layers):
        hf_mlp = model.model.layers[li].mlp
        hf_experts = hf_mlp.experts
        original_experts[li] = hf_experts
        hf_params = next(hf_experts.parameters())
        qe = QwenExperts.from_hf_module(
            hf_mlp, device=hf_params.device, dtype=hf_params.dtype
        )
        hf_mlp.experts = qe

    # -- Pass B: Our experts, record MoE block inputs --
    print("  Pass B: Recording MoE inputs with QwenExperts dispatch...")
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

    # -- Restore original experts --
    for li in range(num_layers):
        model.model.layers[li].mlp.experts = original_experts[li]

    # -- Compare --
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


# -- Main --

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
