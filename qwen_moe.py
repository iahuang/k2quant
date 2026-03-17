"""Qwen1.5-MoE-A2.7B: 2-bit quantization with k2quant.

End-to-end example that quantizes all MoE expert layers using the
IDRE + VPTQ + BCOS pipeline, then evaluates WikiText2 perplexity.

Result: PPL ≈ 7.56 at 2-bit (vs 6.71 FP16 baseline).

This script contains all Qwen-specific code:
  - Model weight path navigation (gate_up_proj, down_proj)
  - Activation collection via forward hooks
  - Router-based expert dispatch for down_proj calibration
  - Fused gate_up_proj splitting for independent BCOS
  - Patched forward with BCOS corrections
"""

import os
import sys

# Must be set before importing faiss (via kbvq) to prevent OpenMP thread explosion.
os.environ["OMP_NUM_THREADS"] = "1"

# Add repo root to path so `import kbvq` works when running from examples/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gc
import time
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import k2quant

# ── Configuration ────────────────────────────────────────────────────────
# Model
MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B"
DEVICE = "cuda"
CACHE_DIR = "/workspace/kbvq/hf_cache"

# Calibration
CALIB_NSAMPLES = 256
CALIB_SEQLEN = 4096
MAX_CALIB_TOKENS = 4096  # subsample activations per layer

# Evaluation
EVAL_SEQLEN = 4096

# Quantization
cfg = k2quant.QuantConfig(
    k_factor=1 / 8,        # IDRE rank: k=256 for ic=2048 (~77% SVD energy)
    vq_bits=2,              # 2 bits per scalar weight
    vq_d=4,                 # 4-dim subvectors → K=256 centroids (8-bit index)
    vq_kmeans_niter=20,     # k-means iterations
    vptq_block_size=128,    # GPTQ block size
    vptq_damp_percent=0.01, # Hessian damping
    vq_num_threads=24,      # parallel CPU k-means workers
    seed=42,
)

os.environ["HF_HOME"] = CACHE_DIR


# ── Patched forward with BCOS on all projections ────────────────────────
# This is Qwen-specific: it knows about gate_up_proj (fused), down_proj,
# the expert routing structure (one_hot + expert_mask), and top_k_weights.
def patched_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Qwen MoE forward with BCOS correction on gate, up, and down projections."""
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]

        # gate_up_proj with split BCOS correction
        gu_out = F.linear(current_state, self.gate_up_proj[expert_idx])
        gate, up = gu_out.chunk(2, dim=-1)

        # Apply BCOS independently to gate and up halves.
        # Must be separate because gate (→ SiLU) and up have different
        # output distributions. Applying joint BCOS corrupts the interaction.
        if hasattr(self, "_bcos_scale_gate"):
            s_g = self._bcos_scale_gate[expert_idx]
            b_g = self._bcos_bias_gate[expert_idx]
            gate = (1 + s_g) * gate + b_g

            s_u = self._bcos_scale_up[expert_idx]
            b_u = self._bcos_bias_up[expert_idx]
            up = (1 + s_u) * up + b_u

        current_hidden_states = self.act_fn(gate) * up

        # down_proj with BCOS correction
        current_hidden_states = F.linear(
            current_hidden_states, self.down_proj[expert_idx]
        )

        if hasattr(self, "_bcos_scale_down"):
            s = self._bcos_scale_down[expert_idx]
            b = self._bcos_bias_down[expert_idx]
            current_hidden_states = (1 + s) * current_hidden_states + b

        current_hidden_states = (
            current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        )
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )

    return final_hidden_states


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("=" * 70)
    print("KBVQ-MoE: 2-bit Quantization of Qwen1.5-MoE-A2.7B")
    print("=" * 70)

    # ── 1. Load tokenizer ────────────────────────────────────────────
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True
    )

    # ── 2. Load calibration data ─────────────────────────────────────
    print("\n[2/5] Loading calibration data...")
    calib_data = k2quant.get_calibration_data(
        tokenizer,
        nsamples=CALIB_NSAMPLES,
        seqlen=CALIB_SEQLEN,
        seed=cfg.seed,
        cache_dir=CACHE_DIR,
    )
    print(f"  Shape: {calib_data.shape}")  # (256, 4096)

    # ── 3. Load model ────────────────────────────────────────────────
    print("\n[3/5] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )
    model.eval()

    # ── 4. Collect activations & quantize ────────────────────────────
    print("\n[4/5] Quantizing expert weights...")

    # 4a. Collect MoE block input activations for all layers.
    # Qwen-specific: hooks on model.model.layers[li].mlp
    print("  Collecting calibration activations...")
    num_layers = model.config.num_hidden_layers
    moe_inputs: dict[int, list] = {}

    hooks = []
    for li in range(num_layers):
        moe_block = model.model.layers[li].mlp
        storage: list = []
        moe_inputs[li] = storage

        def make_hook(s):
            def fn(module, args, kwargs, output):
                s.append(args[0].detach().cpu())
                return output
            return fn

        h = moe_block.register_forward_hook(make_hook(storage), with_kwargs=True)
        hooks.append(h)

    batch_size = 2
    with torch.no_grad():
        for i in range(0, len(calib_data), batch_size):
            batch = calib_data[i : i + batch_size].to(DEVICE)
            model(batch)
            if (i // batch_size) % 32 == 0:
                print(f"    Batch {i // batch_size + 1}/{len(calib_data) // batch_size}")

    for h in hooks:
        h.remove()
    print("  Activations collected.")

    # 4b. Quantize layer by layer.
    print("  Quantizing layers...")
    t_start = time.time()

    for li in range(num_layers):
        t0 = time.time()
        # Qwen-specific: access expert weight tensors
        experts = model.model.layers[li].mlp.experts

        # Subsample and prepare calibration activations for this layer
        acts = torch.cat(moe_inputs[li], dim=0).reshape(-1, model.config.hidden_size)
        moe_inputs[li] = None  # free memory
        if acts.shape[0] > MAX_CALIB_TOKENS:
            perm = torch.randperm(
                acts.shape[0], generator=torch.Generator().manual_seed(cfg.seed)
            )
            acts = acts[perm[:MAX_CALIB_TOKENS]]
        X = acts.float().to(DEVICE)  # (b, ic=2048)

        # ── gate_up_proj: IDRE + VPTQ + split BCOS ──────────────────
        # Qwen-specific: gate_up_proj is fused (60, 2816, 2048) where
        # 2816 = 2 * 1408 (gate + up concatenated along oc dimension).
        W_gu_orig = experts.gate_up_proj.data.float()  # (60, 2816, 2048)
        W_gu_vq = k2quant.w_quantize_and_reconstruct(W_gu_orig, X, cfg)  # (60, 2816, 2048)

        # Compute BCOS for gate and up halves separately.
        # They have different output distributions (gate → SiLU, up → linear),
        # so joint BCOS would corrupt the gating mechanism.
        inter = W_gu_orig.shape[1] // 2  # 1408
        W_gate_orig = W_gu_orig[:, :inter, :]  # (60, 1408, 2048)
        W_up_orig = W_gu_orig[:, inter:, :]    # (60, 1408, 2048)
        W_gate_vq = W_gu_vq[:, :inter, :]      # (60, 1408, 2048)
        W_up_vq = W_gu_vq[:, inter:, :]        # (60, 1408, 2048)

        bcos_s_gate, bcos_b_gate = k2quant.bcos(W_gate_vq, W_gate_orig, X)
        bcos_s_up, bcos_b_up = k2quant.bcos(W_up_vq, W_up_orig, X)

        experts.gate_up_proj = nn.Parameter(W_gu_vq.half().to(DEVICE))
        experts._bcos_scale_gate = bcos_s_gate.to(DEVICE)
        experts._bcos_bias_gate = bcos_b_gate.to(DEVICE)
        experts._bcos_scale_up = bcos_s_up.to(DEVICE)
        experts._bcos_bias_up = bcos_b_up.to(DEVICE)

        # ── down_proj calibration: router-dispatched activations ─────
        # Qwen-specific: use the actual router to determine which tokens
        # go to which expert, then compute per-expert intermediate
        # activations (SiLU(gate) * up). This gives accurate calibration
        # for down_proj — previous approach of averaging across arbitrary
        # experts produced wrong intermediate distributions (PPL 11.74 → 9.16).
        with torch.no_grad():
            router = model.model.layers[li].mlp.gate
            router_logits = F.linear(X.half(), router.weight)
            routing_weights = F.softmax(router_logits, dim=-1)
            _, top_k_indices = torch.topk(
                routing_weights, k=router.top_k, dim=-1
            )

            all_intermediates = []
            n_experts = W_gu_orig.shape[0]
            for ei in range(n_experts):
                mask = (top_k_indices == ei).any(dim=-1)
                if mask.sum() == 0:
                    continue
                tokens = X[mask].half()
                gu_out = F.linear(tokens, W_gu_orig[ei].half().to(DEVICE))
                gate_e, up_e = gu_out.chunk(2, dim=-1)
                intermediate = F.silu(gate_e) * up_e
                all_intermediates.append(intermediate)

            X_down = torch.cat(all_intermediates, dim=0).float()  # (b', 1408)
            del all_intermediates, router_logits, routing_weights, top_k_indices
            if X_down.shape[0] > MAX_CALIB_TOKENS:
                perm = torch.randperm(
                    X_down.shape[0],
                    generator=torch.Generator().manual_seed(cfg.seed),
                )
                X_down = X_down[perm[:MAX_CALIB_TOKENS]]

        # ── down_proj: IDRE + VPTQ + BCOS ───────────────────────────
        W_d_orig = experts.down_proj.data.float()  # (60, 2048, 1408)
        W_d_vq = k2quant.w_quantize_and_reconstruct(W_d_orig, X_down, cfg)  # (60, 2048, 1408)

        bcos_s, bcos_b = k2quant.bcos(W_d_vq, W_d_orig, X_down)

        experts.down_proj = nn.Parameter(W_d_vq.half().to(DEVICE))
        experts._bcos_scale_down = bcos_s.to(DEVICE)
        experts._bcos_bias_down = bcos_b.to(DEVICE)

        # Monkey-patch the forward to apply BCOS corrections
        experts.forward = types.MethodType(patched_experts_forward, experts)

        # Free memory
        del W_gu_orig, W_gu_vq, W_d_orig, W_d_vq, X, X_down
        del W_gate_orig, W_up_orig, W_gate_vq, W_up_vq
        gc.collect()
        torch.cuda.empty_cache()

        print(f"    Layer {li:2d}/{num_layers - 1} done in {time.time() - t0:.1f}s")

    print(f"\n  Total quantization: {time.time() - t_start:.1f}s")

    del moe_inputs, calib_data
    gc.collect()
    torch.cuda.empty_cache()

    # ── 5. Evaluate ──────────────────────────────────────────────────
    print("\n[5/5] Evaluating WikiText2 perplexity...")
    ppl = k2quant.evaluate_perplexity(
        model, tokenizer,
        seqlen=EVAL_SEQLEN,
        device=DEVICE,
        cache_dir=CACHE_DIR,
    )
    print(f"\n{'=' * 70}")
    print(f"  WikiText2 PPL (2-bit, seqlen={EVAL_SEQLEN}): {ppl:.2f}")
    print(f"  Baseline: FP16 -> 6.71  |  Paper 2-bit -> 9.61")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
