"""KBVQ: KLT-guided SVD with Bias-Corrected Vector Quantization.

Model-agnostic library for post-training quantization of Mixture-of-Experts
weight matrices using IDRE + VPTQ + BCOS.

Typical usage:
    import kbvq

    cfg = kbvq.QuantConfig(vq_bits=2, vq_d=4, k_factor=1/8)
    W_vq = kbvq.quantize_weight(W_experts, X_calib, cfg)
    scale, bias = kbvq.compute_bcos_params(W_vq, W_experts, X_calib)
"""

import os as _os

# Prevent OpenMP thread explosion when using ThreadPoolExecutor with faiss.
# faiss CPU k-means uses OpenMP internally; running it inside a thread pool
# without this causes O(n_threads^2) threads.
_os.environ.setdefault("OMP_NUM_THREADS", "1")

from kbvq.bcos import compute_bcos
from kbvq.calibration import get_calibration_data
from kbvq.config import QuantConfig
from kbvq.eval import evaluate_perplexity
from kbvq.export import QuantizedWeight
from kbvq.idre import idre, klt_decomposition
from kbvq.pipeline import compute_bcos_params, quantize_weight, quantize_weight_compressed
from kbvq.vptq import VPTQResult, vptq_quantize
from kbvq.vq import vq_reconstruct

__all__ = [
    "QuantConfig",
    "QuantizedWeight",
    "quantize_weight",
    "quantize_weight_compressed",
    "compute_bcos_params",
    "idre",
    "klt_decomposition",
    "vptq_quantize",
    "VPTQResult",
    "vq_reconstruct",
    "compute_bcos",
    "get_calibration_data",
    "evaluate_perplexity",
]
