# k2quant

A post-training quantization toolkit for compressing mixture-of-experts (MoE) language models to 2 bits per parameter with minimal accuracy loss.

k2quant combines [KBVQ-MoE (Xu et al., 2026)](https://arxiv.org/abs/2602.11184) with a modified variant of [VPTQ (Liu et al., 2024)](https://arxiv.org/abs/2409.17066), composing input-driven low-rank factorization with second-order vector quantization. It is, to our knowledge, the first publicly available implementation of KBVQ-MoE.

For a detailed write-up, see the [accompanying blog post](https://ianhuang.dev/blog/quantization-llm-inference).

## Approach

KBVQ-MoE factors each expert's FFN weights into a shared low-rank component (kept at full precision) and a per-expert residual (compressed via VQ). k2quant extends this by replacing the naive VQ step with a variant of VPTQ, with the following modifications:

- **Hessian-ordered error propagation.** Quantization errors are propagated over columns sorted by inverse Hessian diagonal, prioritizing corrections to the most sensitive dimensions first. This technique was present in VPTQ's reference implementation but undocumented in the paper; it empirically improves perplexity.
- **No residual VQ.** The residual quantization stage from the original VPTQ pipeline is removed, as it was found to yield diminishing returns in this setting.

## Performance

### Optimized C++ quantization kernel

The VPTQ quantization step (K-means clustering and error propagation) is implemented in C++ with pybind11, using OpenBLAS for matrix operations and template-specialized distance computations tuned for small vector dimensions.

On LLaMA-2 13B (dense VPTQ-only, as a standard benchmark), this achieves **~50 minutes end-to-end on a single A100**, compared to ~4 hours on a 4x A100 cluster as reported by the original VPTQ authors—a roughly 20x improvement in compute-cost efficiency.

### Quantization quality

On `Qwen1.5-MoE-A2.7B` (WikiText2 perplexity, lower is better):

| Method                     | PPL — FP16 | PPL — 2-bit ↓ | Degradation ↓ |
| -------------------------- | ---------- | ------------- | ------------- |
| KBVQ-MoE (reported)        | 7.22       | 9.61          | +2.39         |
| KBVQ-MoE + VPTQ (reported) | 7.22       | 8.78          | +1.56         |
| **k2quant (KBVQ only)**    | 7.49       | 9.03          | +1.54         |
| **k2quant (KBVQ + VPTQ)**  | 7.49       | **8.61**      | **+1.12**     |

### Results across models

Perplexity measured on WikiText2. Some weights (e.g. embeddings, attention matrices) are left at original precision due to high sensitivity and small relative size.

| Model               | Total / Active Params | PPL — FP16 | PPL — Quantized ↓ | Size — FP16 | Size — Quantized ↓      |
| ------------------- | --------------------- | ---------- | ----------------- | ----------- | ----------------------- |
| `Qwen1.5-MoE-A2.7B` | 14.3B / 2.7B          | 7.22       | 7.50              | 29 GB       | 7.1 GB (4x compression) |

## Project structure

```
k2quant/          # Python package — pipeline, model adapters, quantization logic
vptq_kernel/      # C++ kernel — K-means clustering, error propagation (pybind11)
tests/            # Unit tests
```

## Next steps

- Inference integration: write dequantization kernels for frameworks like [MLX](https://github.com/ml-explore/mlx), [vLLM](https://github.com/vllm-project/vllm), and [SGLang](https://github.com/sgl-project/sglang).
- Support for additional models and memory/disk offloading for models larger than VRAM budget.
- Further studies on the impact of Hessian-ordered error propagation, sub-2-bit quantization, and other configurations.
