from __future__ import annotations

import dataclasses
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from .config import QuantConfig

_bin_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'bin'))
if _bin_dir not in sys.path:
    sys.path.insert(0, _bin_dir)

import vptq as _vptq

@dataclasses.dataclass
class VQResult:
    """Result of VQ for a batch of experts.

    Stores everything needed to reconstruct quantized weights.
    Row-axis (VPTQ): indices (n_experts, oc_padded // V, ic), codebooks (n_experts, K, V)
    """

    main_indices: torch.Tensor
    """Main codebook indices. int32."""

    main_codebooks: torch.Tensor
    """Main codebook centroids. float16."""

    oc_pad: int
    """Number of zero-padding rows added to make oc divisible by V."""

    oc_padded: int
    """Padded output channel dimension: oc + oc_pad."""

    ic_pad: int
    """Always 0 for row axis."""

    ic_padded: int
    """Same as ic for row axis."""

    col_invperm: torch.Tensor
    """Inverse column permutation to undo column ordering.
    Shape: (ic,). int64. Apply as W_recon[:, :, col_invperm]."""


def _prepare_vq_inputs(
    W_quant: torch.Tensor,
    H: torch.Tensor,
    cfg: QuantConfig,
) -> tuple:
    """Prepare inputs for the C++ VQ kernel.

    Pads rows, computes Hinv, applies column ordering.

    Returns:
        (W_quant, Hinv_np, h_diag_np, oc_pad, oc_padded, ic_padded,
         col_invperm, K, V)
    """
    n, oc, ic = W_quant.shape
    K = cfg.codebook_size
    V = cfg.vq_d

    oc_pad = (V - oc % V) % V
    if oc_pad > 0:
        W_quant = nn.functional.pad(W_quant, (0, 0, 0, oc_pad))
    oc_padded = oc + oc_pad
    ic_padded = ic

    damp = cfg.vptq_damp_percent * H.diagonal().mean()
    H_damp = H + damp * torch.eye(ic, device=H.device, dtype=H.dtype)
    H_inv = torch.linalg.inv(H_damp.double()).float()
    Hinv = torch.linalg.cholesky(H_inv.double(), upper=True).float()

    col_perm = torch.argsort(Hinv.diagonal())
    col_invperm = torch.argsort(col_perm)
    Hinv = Hinv[col_perm][:, col_perm]
    W_quant = W_quant[:, :, col_perm]
    h_diag_np = H.diagonal()[col_perm].cpu().float().numpy()
    Hinv_np = Hinv.cpu().numpy()

    return (W_quant, Hinv_np, h_diag_np, oc_pad, oc_padded, ic_padded,
            col_invperm, K, V)


def _vq_quantize_cpp(
    W_quant: torch.Tensor,
    Hinv_np: np.ndarray,
    h_diag_np: np.ndarray,
    V: int,
    K: int,
    cfg: QuantConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Call the C++ VPTQ kernel."""
    n = W_quant.shape[0]
    oc_padded = W_quant.shape[1]

    W_np = np.ascontiguousarray(W_quant.cpu().float().numpy())
    indices_np, codebooks_np = _vptq.vptq_quantize(
        W_np, Hinv_np, h_diag_np,
        V, K, cfg.vq_kmeans_niter, cfg.vptq_block_size,
    )

    n_row_subvecs = oc_padded // V
    indices_np = indices_np.reshape(n, n_row_subvecs, -1)
    codebooks_np = codebooks_np.reshape(n, K, V)
    return indices_np, codebooks_np


def vq_quantize(
    W_quant: torch.Tensor,
    H: torch.Tensor,
    cfg: QuantConfig,
) -> VQResult:
    """
    VPTQ-based vector quantization (paper-faithful row-axis mode).

    Hessian-weighted k-means + column-by-column error propagation
    using the compiled C++ kernel.

    Args:
        W_quant: Weight residuals after IDRE.
            Shape: (n_experts, oc, ic). float32.
        H: Hessian approximation X^T X / n_samples.
            Shape: (ic, ic). float32. Shared across all experts.
        cfg: Quantization configuration (QuantConfig).

    Returns:
        VQResult with all indices, codebooks, and column permutation info.
    """
    n, oc, ic = W_quant.shape

    (W_quant, Hinv_np, h_diag_np, oc_pad, oc_padded, ic_padded,
     col_invperm, K, V) = _prepare_vq_inputs(W_quant, H, cfg)

    t0 = time.time()
    indices_np, codebooks_np = _vq_quantize_cpp(
        W_quant, Hinv_np, h_diag_np, V, K, cfg)
    print(f"[vq_quantize] {n} experts, V={V}, K={K}: {time.time() - t0:.1f}s")

    return VQResult(
        main_indices=torch.from_numpy(indices_np.copy()),
        main_codebooks=torch.from_numpy(codebooks_np.copy()).half(),
        oc_pad=oc_pad,
        oc_padded=oc_padded,
        ic_pad=0,
        ic_padded=ic_padded,
        col_invperm=col_invperm,
    )


def vq_reconstruct(
    result: VQResult,
    n_experts: int,
    oc: int,
    ic: int,
) -> torch.Tensor:
    """
    Reconstruct weight matrices from VQ quantization results.

    Args:
        result: VQResult from vq_quantize().
        n_experts: Number of experts.
        oc: Output channels (rows per expert, before padding).
        ic: Input channels (columns per expert, before padding).

    Returns:
        W_recon: Reconstructed weights.
            Shape: (n_experts, oc, ic). float16.
    """
    indices = result.main_indices  # (n, n_row_subvecs, ic)
    codebooks = result.main_codebooks  # (n, K, V)
    oc_padded = result.oc_padded

    W_recon = torch.zeros(n_experts, oc_padded, ic, dtype=codebooks.dtype)
    for ei in range(n_experts):
        for col in range(ic):
            # indices[ei, :, col] -> (n_row_subvecs,) centroid indices
            # codebooks[ei][indices] -> (n_row_subvecs, V)
            W_recon[ei, :, col] = codebooks[ei][indices[ei, :, col].long()].reshape(
                oc_padded
            )

    # Undo column permutation
    if result.col_invperm is not None:
        W_recon = W_recon[:, :, result.col_invperm.cpu()]

    # Strip row padding
    if result.oc_pad > 0:
        W_recon = W_recon[:, :oc, :]

    return W_recon