"""Qwen1.5-MoE-A2.7B: 2-bit quantization with k2quant.

End-to-end example that quantizes all MoE expert layers using the
IDRE + VPTQ + BCOS pipeline, then evaluates WikiText2 perplexity.

Result: PPL ≈ 7.56 at 2-bit (vs 6.71 FP16 baseline).
"""

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import k2quant
from k2quant.models.qwen_moe import QwenMoEBlock
from k2quant.util import get_calibration_data, evaluate_perplexity

# ── Configuration ────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B"
DEVICE = "cuda"
CACHE_DIR = "/workspace/kbvq/hf_cache"

cfg = k2quant.QuantConfig(
    k_factor=1 / 8,
    vq_bits=2,
    vq_d=4,
    vq_kmeans_niter=20,
    vptq_block_size=128,
    vptq_damp_percent=0.01,
    vq_num_threads=24,
    seed=42,
)

os.environ["HF_HOME"] = CACHE_DIR


def main():
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("=" * 70)
    print("k2quant: 2-bit Quantization of Qwen1.5-MoE-A2.7B")
    print("=" * 70)

    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise RuntimeError(f"{MODEL_NAME} did not load a fast tokenizer")

    print("\n[2/4] Loading calibration data...")
    calib_data = get_calibration_data(
        tokenizer, nsamples=256, seqlen=4096, seed=cfg.seed, cache_dir=CACHE_DIR
    )

    print("\n[3/4] Loading and quantizing model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )
    model.eval()

    k2quant.quantize_model(
        model, calib_data, cfg,
        QwenMoEBlock,
        get_moe_parent_and_attr=lambda m, i: (m.model.layers[i], "mlp"),
        num_layers=model.config.num_hidden_layers,
        device=DEVICE, max_calib_tokens=4096, batch_size=2,
    )

    print("\n[4/4] Evaluating WikiText2 perplexity...")
    ppl = evaluate_perplexity(
        model, tokenizer, seqlen=4096, device=DEVICE, cache_dir=CACHE_DIR
    )
    print(f"\n{'=' * 70}")
    print(f"  WikiText2 PPL (2-bit, seqlen=4096): {ppl:.2f}")
    print(f"  Baseline: FP16 -> 6.71  |  Paper 2-bit -> 9.61")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
