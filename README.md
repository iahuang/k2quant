# k2quant

Minimally lossy 2-bit post-training quantization of MoE language models using [KBVQ-MoE (Xu et al., 2026)](https://arxiv.org/abs/2602.11184), [VPTQ (Liu et al., 2024)](https://arxiv.org/abs/2409.17066), and [GPTVQ (van Baalen et al., 2024)](https://arxiv.org/abs/2402.15319).

k2quant extends the original findings of KBVQ-MoE by swapping the VQ step for a variant of VPTQ which makes the following modifications:
- Uses a GPTQ-style error propagation step rather than the per-column error propagation used in VPTQ.
- Propagates errors over the columns of the weight matrix sorted by the inverse of the Hessian. This is a unique contribution of k2quant and empirically improves perplexity.

