"""KBVQ: KLT-guided SVD with Bias-Corrected Vector Quantization.

Model-agnostic library for post-training quantization of Mixture-of-Experts
weight matrices using IDRE + VPTQ + BCOS.

Typical usage:
    import kbvq

    cfg = kbvq.QuantConfig(vq_bits=2, vq_d=4, k_factor=1/8)
    W_vq = kbvq.quantize_weight(W_experts, X_calib, cfg)
    scale, bias = kbvq.compute_bcos_params(W_vq, W_experts, X_calib)
"""


from .vq import vq_quantize, vq_reconstruct, VQResult
from .config import QuantConfig
from .kbvq import bcos, idre

__all__ = [
    "vq_quantize",
    "vq_reconstruct",
    "VQResult",
    "QuantConfig",
    "bcos",
    "idre",
]
