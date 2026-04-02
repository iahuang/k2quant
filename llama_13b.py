import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import k2quant
from k2quant.models.llama import LlamaMLP
from k2quant.util import get_calibration_data, evaluate_perplexity

# -- Configuration --
MODEL_NAME = "NousResearch/Llama-2-13b-hf"
DEVICE = "cuda"
CACHE_DIR = "./hf_cache"
OUTPUT_DIR = "./llama_13b_2bit"

cfg = k2quant.QuantConfig(
    k_factor=1 / 128,
    vq_bits=2,
    vq_d=4,
    vq_kmeans_niter=20,
    vptq_block_size=128,
    vptq_damp_percent=0.01,
    seed=42,
)

os.environ["HF_HOME"] = CACHE_DIR


def _tensor_bytes(t: torch.Tensor) -> int:
    return t.nelement() * t.element_size()


def main():
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("=" * 70)
    print("k2quant: 2-bit Quantization of LLaMA-2 13B")
    print("=" * 70)

    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
    )

    print("\n[2/5] Loading calibration data...")
    calib_data = get_calibration_data(
        tokenizer, nsamples=256, seqlen=4096, seed=cfg.seed, cache_dir=CACHE_DIR
    )

    print("\n[3/5] Loading and quantizing model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE_DIR,
    )
    model.eval()

    # Measure original MLP weight sizes before quantization
    orig_mlp_bytes = 0
    num_layers = model.config.num_hidden_layers
    for li in range(num_layers):
        mlp = model.model.layers[li].mlp
        for p in mlp.parameters():
            orig_mlp_bytes += _tensor_bytes(p)

    compressed_tensors = k2quant.quantize_dense_model(
        model, calib_data, cfg,
        LlamaMLP,
        get_mlp_block=lambda m, i: m.model.layers[i].mlp,
        set_mlp_block=lambda m, i, mlp: setattr(m.model.layers[i], "mlp", mlp),
        num_layers=num_layers,
        device=DEVICE, max_calib_tokens=4096, batch_size=2,
    )

    print("\n[4/5] Saving quantized safetensors...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect non-MLP parameters (embeddings, attention, norms)
    non_mlp_tensors = {}
    for name, param in model.state_dict().items():
        # Skip MLP weights — they're replaced by compressed tensors
        if ".mlp." in name:
            continue
        non_mlp_tensors[name] = param.contiguous().cpu()

    # Merge non-MLP params with compressed MLP tensors
    all_tensors = {}
    all_tensors.update(non_mlp_tensors)
    all_tensors.update(compressed_tensors)

    output_path = os.path.join(OUTPUT_DIR, "model.safetensors")
    save_file(all_tensors, output_path)

    # Compute sizes
    non_mlp_bytes = sum(_tensor_bytes(t) for t in non_mlp_tensors.values())
    compressed_bytes = sum(_tensor_bytes(t) for t in compressed_tensors.values())
    total_orig_bytes = orig_mlp_bytes + non_mlp_bytes
    total_quant_bytes = compressed_bytes + non_mlp_bytes
    file_bytes = os.path.getsize(output_path)

    print(f"  Saved to: {output_path}")
    print(f"\n  --- Size Breakdown ---")
    print(f"  Original MLP weights:       {orig_mlp_bytes / 1e9:.2f} GB")
    print(f"  Compressed MLP weights:     {compressed_bytes / 1e9:.2f} GB")
    print(f"  Non-MLP parameters:         {non_mlp_bytes / 1e9:.2f} GB")
    print(f"  ---")
    print(f"  Original total (params):    {total_orig_bytes / 1e9:.2f} GB")
    print(f"  Quantized total (params):   {total_quant_bytes / 1e9:.2f} GB")
    print(f"  Reduction:                  {(1 - total_quant_bytes / total_orig_bytes) * 100:.1f}%")
    print(f"  Output file size:           {file_bytes / 1e9:.2f} GB")

    print("\n[5/5] Evaluating WikiText2 perplexity...")
    ppl = evaluate_perplexity(
        model, tokenizer, seqlen=4096, device=DEVICE, cache_dir=CACHE_DIR
    )
    print(f"\n{'=' * 70}")
    print(f"  WikiText2 PPL (2-bit, seqlen=4096): {ppl:.2f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
