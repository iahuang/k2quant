"""Per-projection quantization: IDRE + VPTQ + BCOS in one call.

Combines w_quantize_and_reconstruct and bcos into a single function,
with support for split BCOS correction across sub-projections.
"""

from __future__ import annotations

import dataclasses

import torch

from .core import QuantConfig, bcos
from .quant import w_quantize_and_reconstruct


@dataclasses.dataclass
class BCOSLayout:
    """Describes how to split a weight matrix for independent BCOS correction.

    For fused projections (e.g., gate_up_proj), BCOS must be applied
    independently to each sub-projection because they have different
    output distributions (gate → SiLU, up → linear).

    For unfused projections, use a single split covering the whole oc
    dimension — BCOS is computed once over the entire matrix.

    Attributes:
        split_sizes: Sizes of each sub-projection along the oc dimension.
            Must sum to the total oc of the weight matrix.
        names: Name for each sub-projection, used as keys in
            ProjectionResult.bcos_params.

    Examples:
        Fused gate+up (Qwen): BCOSLayout([1408, 1408], ["gate", "up"])
        Unfused down:          BCOSLayout([2048], ["proj"])
    """

    split_sizes: list[int]
    names: list[str]

    def __post_init__(self):
        if len(self.split_sizes) != len(self.names):
            raise ValueError(
                f"split_sizes ({len(self.split_sizes)}) and names "
                f"({len(self.names)}) must have the same length"
            )


@dataclasses.dataclass
class ProjectionResult:
    """Result of quantizing one projection.

    Attributes:
        W_vq: Reconstructed quantized weight. (n_experts, oc, ic). float32.
        bcos_params: BCOS correction factors keyed by sub-projection name.
            Scale and bias are (n_experts, sub_oc). float16.
    """

    W_vq: torch.Tensor
    bcos_params: dict[str, tuple[torch.Tensor, torch.Tensor]]


def quantize_projection(
    W_orig: torch.Tensor,
    X_calib: torch.Tensor,
    cfg: QuantConfig,
    bcos_layout: BCOSLayout,
) -> ProjectionResult:
    """Quantize a single projection: IDRE + VPTQ + BCOS.

    Args:
        W_orig: Original weight matrices. (n_experts, oc, ic). float32.
        X_calib: Calibration activations. (b, ic). float32.
        cfg: Quantization configuration.
        bcos_layout: Describes how to split the weight for independent
            BCOS correction per sub-projection.

    Returns:
        ProjectionResult with quantized weights and BCOS correction factors.
    """
    W_vq = w_quantize_and_reconstruct(W_orig, X_calib, cfg)

    if sum(bcos_layout.split_sizes) != W_orig.shape[1]:
        raise ValueError(
            f"BCOSLayout split_sizes sum ({sum(bcos_layout.split_sizes)}) "
            f"!= oc ({W_orig.shape[1]})"
        )

    bcos_params: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    offset = 0
    for size, name in zip(bcos_layout.split_sizes, bcos_layout.names):
        W_sub_orig = W_orig[:, offset : offset + size, :]
        W_sub_vq = W_vq[:, offset : offset + size, :]
        s, b = bcos(W_sub_vq, W_sub_orig, X_calib)
        bcos_params[name] = (s, b)
        offset += size

    return ProjectionResult(W_vq=W_vq, bcos_params=bcos_params)
