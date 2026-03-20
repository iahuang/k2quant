# k2quant

Minimally lossy 2-bit post-training quantization of MoE language models using [KBVQ-MoE (Xu et al., 2026)](https://arxiv.org/abs/2602.11184) and [VPTQ (Liu et al., 2024)](https://arxiv.org/abs/2409.17066).

k2quant extends the original findings of KBVQ-MoE by swapping the VQ step for a variant of VPTQ which makes the following modification:

- Propagates errors over the columns of the weight matrix sorted by the inverse of the Hessian. This is a unique contribution of k2quant and empirically improves perplexity.
- Removes the residual VQ step from the pipeline, as this was found to contribute diminishing returns.
- Weighted K-means algorithm is approximated by oversampling proportional to the Hessian diagonal. This is a technical limitation due to the use of faiss for k-means, which does not support weighted k-means.

## Comparison to Published Results

On the `Qwen1.5-MoE-A2.7B` model, we compare the perplexity (PPL) of the FP16 baseline and the results reported in the original paper. The authors evaluate their results using KBVQ-MoE using a naive VQ algorithm and using VPTQ.

| Method                           | PPL (WikiText2) ↓ |
| -------------------------------- | ----------------- |
| _FP16 Baseline_                  | 7.22              |
| KBVQ-MoE (reported)              | 9.61              |
| KBVQ-MoE + VPTQ (reported)       | 8.78              |
| **k2quant (no column ordering)** | 8.13              |
| **k2quant**                      | **7.56**          |

## Results on All Tested Models

Perplexity measured on the WikiText2 dataset. Model size informs minimum VRAM requirement. Some weights (e.g. embeddings, attention matrices) are left in their original precision, due to their high sensitivity and small relative size.

| Model               | Total / Active Parameters | PPL - FP16 | PPL - Quantized ↓ | Size - FP16 | Size - Quantized ↓ |
| ------------------- | ------------------------- | ---------- | ----------------- | ----------- | ------------------ |
| `Qwen1.5-MoE-A2.7B` | 14.3B / 2.7B              | 7.22       | 7.56              | 29 GB       | ?                  |
| `Qwen3.5-35B-A3B`   | 35B / 3B                  | 7.17       | 8.08              | 67 GB       | 22.2 GB (~3x)      |
| `Mixtral-8x7B-v0.1` | 46.7B / 12.9B             | ?          | ?                 | ?           | ?                  |
