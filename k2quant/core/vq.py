"""
Vector Quantization (VQ) implementing VPTQ (Liu et al., 2024).

Channel-independent second-order optimization: subvectors span V contiguous
output rows for each input column. Each column is fully quantized before
error propagates to the next, so every centroid is chosen on fully-corrected
weights. This is faithful to Algorithm 1 in the VPTQ paper.

Column ordering: low-sensitivity columns are processed first (ascending
diag(H_inv)), so their errors propagate to less important columns.
"""

from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor

import faiss
import numpy as np
import torch
import torch.nn as nn

from .config import QuantConfig


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


def _vptq_one_expert_row(args: tuple) -> tuple:
    """VPTQ quantization for a single expert (paper-faithful row-axis mode).

    Subvectors span V contiguous output rows for each input column.
    Each column is fully quantized (all its V-length subvectors assigned
    to centroids) before error propagation to the next column. This means
    every centroid choice is made on fully-corrected weights.

    Args:
        args: Tuple of (W_np, Hinv_np, h_diag_np, V, K, niter, block_size).

    Returns:
        Tuple of (indices, centroids).
    """
    W_np, Hinv_np, h_diag_np, V, K, niter, block_size = args
    oc, ic = W_np.shape  # single expert: (oc_padded, ic)
    n_row_subvecs = oc // V

    # ── 1. Hessian-weighted k-means ──────────────────────────────────────
    # Approximate weighted k-means via oversampling: columns with higher
    # diag(H) values contribute more training vectors to the codebook.
    if h_diag_np.sum() > 0:
        norm_w = h_diag_np / (h_diag_np.mean() + 1e-10)
        repeat_counts = np.clip(np.round(norm_w).astype(int), 1, 4)
    else:
        repeat_counts = np.ones(ic, dtype=int)

    train_parts = []
    for col in range(ic):
        col_subvecs = W_np[:, col].reshape(n_row_subvecs, V)  # (n_row_subvecs, V)
        for _ in range(repeat_counts[col]):
            train_parts.append(col_subvecs)
    train_data = np.vstack(train_parts).astype(np.float32)

    km = faiss.Kmeans(V, K, niter=niter, verbose=False, gpu=False)
    km.train(train_data)
    centroids = km.centroids.copy().astype(np.float32)  # (K, V)

    search_index = faiss.IndexFlatL2(V)
    search_index.add(centroids)

    # ── 2. Column-by-column error propagation (VPTQ Algorithm 1) ─────────
    # For each column, slice all oc rows into V-length subvectors, assign
    # to nearest centroid. Then propagate the full-column error to all
    # remaining columns before moving on.
    W = W_np.copy().astype(np.float64)
    indices = np.zeros((n_row_subvecs, ic), dtype=np.int32)
    Hinv = Hinv_np.astype(np.float64)
    centroids_f64 = centroids.astype(np.float64)

    for i1 in range(0, ic, block_size):
        i2 = min(i1 + block_size, ic)
        count = i2 - i1

        W1 = W[:, i1:i2].copy()  # (oc, count)
        Err1 = np.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]  # (count, count)

        # Process one column at a time
        for j in range(count):
            col_vec = W1[:, j]  # (oc,)
            sv = col_vec.reshape(n_row_subvecs, V).astype(np.float32)
            _, idx = search_index.search(sv, 1)
            idx = idx.reshape(-1)
            indices[:, i1 + j] = idx

            q_col = centroids_f64[idx].reshape(oc)

            err = (W1[:, j] - q_col) / Hinv1[j, j]
            Err1[:, j] = err
            if j + 1 < count:
                W1[:, j + 1 :] -= (
                    err[:, np.newaxis] * Hinv1[j, j + 1 : count][np.newaxis, :]
                )

        if i2 < ic:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    return indices, centroids


def vq_quantize(
    W_quant: torch.Tensor,
    H: torch.Tensor,
    cfg: QuantConfig,
) -> VQResult:
    """
    VPTQ-based vector quantization (paper-faithful row-axis mode).

    Hessian-weighted k-means + column-by-column error propagation.

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
    K = cfg.codebook_size
    V = cfg.vq_d  # subvector dimension = V rows

    # ── Padding: pad oc to be divisible by V ──────────────────────────────
    oc_pad = (V - oc % V) % V
    ic_pad = 0
    if oc_pad > 0:
        W_quant = nn.functional.pad(W_quant, (0, 0, 0, oc_pad))  # pad rows
    oc_padded = oc + oc_pad
    ic_padded = ic

    # ── Compute upper Cholesky of H^{-1} (shared across all experts) ────
    damp = cfg.vptq_damp_percent * H.diagonal().mean()
    H_damp = H + damp * torch.eye(ic, device=H.device, dtype=H.dtype)
    H_inv = torch.linalg.inv(H_damp.double()).float()
    Hinv = torch.linalg.cholesky(H_inv.double(), upper=True).float()

    # ── Column ordering ──────────────────────────────────────────────────
    # Sort columns by ascending diag(H_inv): process easy/low-sensitivity
    # columns first. Their quantization errors propagate to columns that
    # are less affected by perturbation, while high-sensitivity columns
    # are quantized last with the benefit of all prior error corrections.
    col_perm = torch.argsort(Hinv.diagonal())
    col_invperm = torch.argsort(col_perm)
    Hinv = Hinv[col_perm][:, col_perm]
    W_quant = W_quant[:, :, col_perm]
    h_diag_np = H.diagonal()[col_perm].cpu().float().numpy()

    Hinv_np = Hinv.cpu().numpy()

    # ── Dispatch per-expert VPTQ to thread pool ──────────────────────────
    work_items = []
    for ei in range(n):
        W_np = W_quant[ei].cpu().float().numpy()
        work_items.append(
            (
                W_np,
                Hinv_np,
                h_diag_np,
                V,
                K,
                cfg.vq_kmeans_niter,
                cfg.vptq_block_size,
            )
        )

    with ThreadPoolExecutor(max_workers=cfg.vq_num_threads) as pool:
        results = list(pool.map(_vptq_one_expert_row, work_items))

    # ── Collect results ──────────────────────────────────────────────────
    all_indices = []
    all_codebooks = []

    for idx, cb in results:
        all_indices.append(torch.from_numpy(idx))
        all_codebooks.append(torch.from_numpy(cb).half())

    return VQResult(
        main_indices=torch.stack(all_indices),
        main_codebooks=torch.stack(all_codebooks),
        oc_pad=oc_pad,
        oc_padded=oc_padded,
        ic_pad=ic_pad,
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
