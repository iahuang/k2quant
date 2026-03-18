"""
Vector Quantization (VQ) modeled after VPTQ (Liu et al., 2024) and GPTVQ (van Baalen et al., 2024).

Hessian-weighted k-means initialization + GPTQ-style column-by-column
error propagation. Operates on per-expert weight matrices independently,
parallelized across experts via ThreadPoolExecutor.

Important notes:

 -  Subvectors span d contiguous input columns for one
    output channel row. This is GPTVQ-style grouped quantization: the
    centroid locks in all d column values at once, so columns j+1..j+d-1
    within each group do NOT benefit from error correction before their
    values are committed. Intra-group scalar error propagation updates
    the remaining workspace but cannot revisit the centroid choice.
    Despite this theoretical disadvantage, empirically better for perplexity.

 -  Column ordering: low-sensitivity columns are processed first (ascending
    diag(H_inv)), so their errors propagate to less important columns. This contribution
    is unique to this implementation, but it empirically provides a significant improvement in
    perplexity.

-   Residual VQ: This part is intentionally omitted for simplicity. Empirical testing found that it
    provides little to no benefit in the context of KBVQ-MoE.
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
    Index and codebook shapes:
      indices (n_experts, oc, ic_padded // d), codebooks (n_experts, K, d)
    """

    main_indices: torch.Tensor
    """Main codebook indices. int32. Shape depends on vq_axis."""

    main_codebooks: torch.Tensor
    """Main codebook centroids. float16. Shape depends on vq_axis."""

    oc_pad: int
    """Number of zero-padding rows added to make oc divisible by V (row axis).
    Always 0 for col axis."""

    oc_padded: int
    """Padded output channel dimension: oc + oc_pad."""

    ic_pad: int
    """Number of zero-padding columns added to make ic divisible by d (col axis).
    Always 0 for row axis."""

    ic_padded: int
    """Padded input channel dimension: ic + ic_pad."""

    col_invperm: torch.Tensor
    """Inverse column permutation to undo GPTQ column ordering.
    Shape: (ic_padded,) for col axis, (ic,) for row axis. int64.
    Apply as W_recon[:, :, col_invperm]."""


def _vptq_one_expert_col(args: tuple) -> tuple:
    """VPTQ quantization for a single expert, col-axis mode (GPTVQ-style).

    Subvectors span d contiguous input columns for each output row.
    The centroid assignment is atomic over d columns — all d values are
    committed by one lookup. Intra-group scalar error propagation then
    updates the remaining workspace, but cannot revisit the centroid
    choice for columns j+1..j+d-1 within the group.

    Args:
        args: Tuple of (W_np, Hinv_np, h_diag_np, d, K, niter,
              block_size).

    Returns:
        Tuple of (indices, centroids).
    """
    W_np, Hinv_np, h_diag_np, d, K, niter, block_size = args
    oc, ic = W_np.shape  # single expert: (oc, ic_padded)
    n_subvecs = ic // d

    # ── 1. Hessian-weighted k-means ──────────────────────────────────────
    # NOTE: The paper describes true weighted k-means where each training
    # vector receives a Hessian-derived scalar weight in the loss function.
    # This is approximated via oversampling: subvector positions spanning
    # higher diag(H) columns contribute more training vectors to the
    # codebook. This avoids modifying faiss's optimized k-means
    # implementation. The approximation biases centroid placement toward
    # important regions but does not minimize the exact weighted objective.
    #
    # Compute per-subvector importance from Hessian diagonal.
    # Cap at 4x to prevent a single subvector from dominating training.
    # This cap is not in the paper — chosen empirically.
    subvec_weights = h_diag_np.reshape(n_subvecs, d).mean(axis=1)  # (n_subvecs,)
    flat = W_np.reshape(oc, n_subvecs, d)  # (oc, n_subvecs, d)

    if subvec_weights.sum() > 0:
        norm_w = subvec_weights / (subvec_weights.mean() + 1e-10)
        repeat_counts = np.clip(np.round(norm_w).astype(int), 1, 4)
    else:
        repeat_counts = np.ones(n_subvecs, dtype=int)

    train_parts = []
    for sv in range(n_subvecs):
        vecs = flat[:, sv, :]  # (oc, d)
        for _ in range(repeat_counts[sv]):
            train_parts.append(vecs)
    train_data = np.vstack(train_parts).astype(np.float32)

    km = faiss.Kmeans(d, K, niter=niter, verbose=False, gpu=False)
    km.train(train_data)
    centroids = km.centroids.copy().astype(np.float32)  # (K, d)

    search_index = faiss.IndexFlatL2(d)
    search_index.add(centroids)

    # ── 2. GPTQ-style error propagation ──────────────────────────────────
    W = W_np.copy().astype(np.float64)
    indices = np.zeros((oc, n_subvecs), dtype=np.int32)
    Hinv = Hinv_np.astype(np.float64)

    for i1 in range(0, ic, block_size):
        i2 = min(i1 + block_size, ic)
        count = i2 - i1

        W1 = W[:, i1:i2].copy()  # (oc, count)
        Err1 = np.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]  # (count, count)

        # Process subvectors (groups of d columns) within this block
        for j in range(0, count - d + 1, d):
            sv = W1[:, j : j + d].astype(np.float32)  # (oc, d)
            _, idx = search_index.search(sv, 1)  # (oc, 1)
            idx = idx.reshape(-1)  # (oc,)

            sv_idx = (i1 + j) // d
            indices[:, sv_idx] = idx
            q_sv = centroids[idx].astype(np.float64)  # (oc, d)

            # Propagate errors column by column within subvector
            for c in range(d):
                col = j + c
                err = (W1[:, col] - q_sv[:, c]) / Hinv1[col, col]  # (oc,)
                Err1[:, col] = err
                if col + 1 < count:
                    W1[:, col + 1 :] -= (
                        err[:, np.newaxis] * Hinv1[col, col + 1 : count][np.newaxis, :]
                    )

        # Inter-block error propagation
        if i2 < ic:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    return indices, centroids


def vq_quantize(
    W_quant: torch.Tensor,
    H: torch.Tensor,
    cfg: QuantConfig,
) -> VQResult:
    """
    VPTQ-based vector quantization.

    Hessian-weighted k-means + GPTQ error propagation + column ordering.

    Quantizes a batch of expert weight matrices with output-aware error
    minimization. Each expert is processed independently in a thread pool.

    Args:
        W_quant: Weight residuals after IDRE.
            Shape: (n_experts, oc, ic). float32.
        H: Hessian approximation X^T X / n_samples.
            Shape: (ic, ic). float32. Shared across all experts.
        cfg: Quantization configuration (QuantConfig).

    Returns:
        VPTQResult with all indices, codebooks, and column permutation info.
    """
    n, oc, ic = W_quant.shape
    K = cfg.codebook_size
    d = cfg.vq_d

    # ── Padding ───────────────────────────────────────────────────────────
    # Pad ic to be divisible by d
    ic_pad = (d - ic % d) % d
    oc_pad = 0
    if ic_pad > 0:
        W_quant = nn.functional.pad(W_quant, (0, ic_pad))
    ic_padded = ic + ic_pad
    oc_padded = oc

    # ── Hessian padding (col axis only) ──────────────────────────────────
    if ic_pad > 0:
        H_padded = torch.zeros(ic_padded, ic_padded, dtype=H.dtype, device=H.device)
        H_padded[:ic, :ic] = H
        for p in range(ic, ic_padded):
            H_padded[p, p] = H.diagonal().mean() * cfg.vptq_damp_percent
        H = H_padded

    # ── Compute upper Cholesky of H^{-1} (shared across all experts) ────
    # Damping prevents singularity when some input features have zero variance.
    ic_h = ic_padded
    damp = cfg.vptq_damp_percent * H.diagonal().mean()
    H_damp = H + damp * torch.eye(ic_h, device=H.device, dtype=H.dtype)
    H_inv = torch.linalg.inv(H_damp.double()).float()  # (ic_h, ic_h)
    # Upper Cholesky factorization for numerical stability in GPTQ loop.
    Hinv = torch.linalg.cholesky(H_inv.double(), upper=True).float()  # (ic_h, ic_h)

    # ── Column ordering ──────────────────────────────────────────────────
    # Sort columns by ascending diag(H_inv): process easy/low-sensitivity
    # columns first. Their quantization errors propagate to columns that
    # are less affected by perturbation, while high-sensitivity columns
    # are quantized last with the benefit of all prior error corrections.
    col_perm = torch.argsort(Hinv.diagonal())  # (ic_h,)
    col_invperm = torch.argsort(col_perm)  # (ic_h,)
    Hinv = Hinv[col_perm][:, col_perm]
    W_quant = W_quant[:, :, col_perm]
    h_diag_np = H.diagonal()[col_perm].cpu().float().numpy()  # (ic_h,)

    Hinv_np = Hinv.cpu().numpy()

    # ── Dispatch per-expert VPTQ to thread pool ──────────────────────────
    worker_fn = _vptq_one_expert_col

    work_items = []
    for ei in range(n):
        W_np = W_quant[ei].cpu().float().numpy()
        work_items.append(
            (
                W_np,
                Hinv_np,
                h_diag_np,
                d,
                K,
                cfg.vq_kmeans_niter,
                cfg.vptq_block_size,
            )
        )

    with ThreadPoolExecutor(max_workers=cfg.vq_num_threads) as pool:
        results = list(pool.map(worker_fn, work_items))

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

    Looks up each subvector's centroid from the codebook, optionally adds
    the residual codebook contribution, undoes column permutation, and
    strips padding.

    Args:
        result: VPTQResult from vptq_quantize().
        n_experts: Number of experts.
        oc: Output channels (rows per expert, before padding).
        ic: Input channels (columns per expert, before padding).

    Returns:
        W_recon: Reconstructed weights.
            Shape: (n_experts, oc, ic). float16.
    """

    return _reconstruct_col(result, n_experts, oc, ic)


def _reconstruct_col(
    result: VQResult,
    n_experts: int,
    oc: int,
    ic: int,
) -> torch.Tensor:
    """Reconstruct from col-axis indices: (n, oc, n_subvecs) + codebook (n, K, d)."""
    indices = result.main_indices  # (n, oc, n_subvecs)
    codebooks = result.main_codebooks  # (n, K, d)
    ic_padded = result.ic_padded

    W_recon = torch.zeros(n_experts, oc, ic_padded, dtype=codebooks.dtype)
    for ei in range(n_experts):
        recon = codebooks[ei][indices[ei].long()]  # (oc, n_subvecs, d)
        W_recon[ei] = recon.reshape(oc, ic_padded)

    # Undo column permutation
    if result.col_invperm is not None:
        W_recon = W_recon[:, :, result.col_invperm.cpu()]

    # Strip column padding
    if result.ic_pad > 0:
        W_recon = W_recon[:, :, :ic]

    return W_recon
