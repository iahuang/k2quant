"""
Vector Quantization (VQ) modeled after VPTQ (Liu et al., 2024) and GPTVQ (van Baalen et al., 2024).

Hessian-weighted k-means initialization + GPTQ-style column-by-column
error propagation.

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
"""

from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor

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


# ── Weighted k-means in PyTorch ───────────────────────────────────────────



def _batched_weighted_kmeans(
    data: torch.Tensor,
    weights: torch.Tensor,
    K: int,
    niter: int,
    seed: int = 0,
    max_batch: int = 8,
) -> torch.Tensor:
    """Batched weighted k-means — runs B independent k-means in parallel.

    Processes experts in chunks of max_batch to limit GPU memory.
    The main memory cost per chunk is the (batch, N, K) distance tensor.

    Args:
        data: Training vectors. Shape: (B, N, d). float32.
        weights: Per-vector weights. Shape: (B, N). float32.
        K: Number of centroids per batch element.
        niter: Number of iterations.
        seed: Random seed.
        max_batch: Max experts to process in one GPU batch.

    Returns:
        centroids: Shape: (B, K, d). float32. Same device as data.
    """
    B = data.shape[0]
    if B <= max_batch:
        return _batched_weighted_kmeans_core(data, weights, K, niter, seed)

    # Chunk over experts
    chunks = []
    for start in range(0, B, max_batch):
        end = min(start + max_batch, B)
        chunk = _batched_weighted_kmeans_core(
            data[start:end], weights[start:end], K, niter, seed + start
        )
        chunks.append(chunk)
    return torch.cat(chunks, dim=0)


def _batched_weighted_kmeans_core(
    data: torch.Tensor,
    weights: torch.Tensor,
    K: int,
    niter: int,
    seed: int,
) -> torch.Tensor:
    """Core batched weighted k-means for a single chunk of experts."""
    B, N, d = data.shape

    # Initialize centroids via weighted random sampling, per batch element
    gen = torch.Generator(device=data.device).manual_seed(seed)
    probs = weights / weights.sum(dim=1, keepdim=True)  # (B, N)
    all_centroids = []
    for b in range(B):
        init_idx = torch.multinomial(probs[b], K, replacement=False, generator=gen)
        all_centroids.append(data[b, init_idx])
    centroids = torch.stack(all_centroids)  # (B, K, d)

    for _ in range(niter):
        # Assignment: (B, N, K) distances
        dists = torch.cdist(data, centroids, p=2.0).square()  # (B, N, K)
        assignments = dists.argmin(dim=2)  # (B, N)
        del dists

        # Update: weighted mean per cluster
        one_hot = torch.zeros(B, N, K, device=data.device, dtype=data.dtype)
        one_hot.scatter_(2, assignments.unsqueeze(2), 1.0)
        weighted_one_hot = one_hot * weights.unsqueeze(2)  # (B, N, K)
        del one_hot

        cluster_weights = weighted_one_hot.sum(dim=1)  # (B, K)
        # (B, K, N) @ (B, N, d) -> (B, K, d)
        new_centroids = torch.bmm(weighted_one_hot.transpose(1, 2), data)
        del weighted_one_hot

        nonempty = cluster_weights > 0  # (B, K)
        divisor = cluster_weights.unsqueeze(2).clamp(min=1e-10)
        new_centroids = new_centroids / divisor
        new_centroids = torch.where(
            nonempty.unsqueeze(2), new_centroids, centroids
        )
        centroids = new_centroids

    return centroids


# ── GPTQ error propagation (per-expert, CPU) ─────────────────────────────


def _gptq_with_codebook(
    W_np: np.ndarray,
    Hinv_np: np.ndarray,
    centroids_np: np.ndarray,
    d: int,
    block_size: int,
) -> np.ndarray:
    """GPTQ-style quantization using a pre-trained codebook.

    Sequentially assigns subvectors to nearest centroids while propagating
    quantization error through the Hessian inverse. This is inherently
    sequential per expert (error from column j affects column j+1).

    Args:
        W_np: Weight matrix for one expert. Shape: (oc, ic_padded). float64.
        Hinv_np: Upper Cholesky of H^{-1}. Shape: (ic_padded, ic_padded). float64.
        centroids_np: Codebook centroids. Shape: (K, d). float32.
        d: Subvector dimension.
        block_size: GPTQ block size.

    Returns:
        indices: Codebook assignments. Shape: (oc, n_subvecs). int32.
    """
    oc, ic = W_np.shape
    n_subvecs = ic // d

    # Build a simple brute-force index for nearest centroid lookup.
    # For K=256, d=4 this is a (oc, K) distance matrix — trivial.
    # Using numpy to avoid faiss dependency in this function.
    W = W_np.copy()  # already float64
    indices = np.zeros((oc, n_subvecs), dtype=np.int32)
    Hinv = Hinv_np  # already float64
    centroids_f64 = centroids_np.astype(np.float64)

    for i1 in range(0, ic, block_size):
        i2 = min(i1 + block_size, ic)
        count = i2 - i1

        W1 = W[:, i1:i2].copy()  # (oc, count)
        Err1 = np.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]  # (count, count)

        for j in range(0, count - d + 1, d):
            sv = W1[:, j : j + d].astype(np.float32)  # (oc, d)

            # Nearest centroid lookup: (oc, K) distances
            # sv: (oc, d), centroids: (K, d) -> dists: (oc, K)
            dists = ((sv[:, np.newaxis, :] - centroids_np[np.newaxis, :, :]) ** 2).sum(
                axis=2
            )
            idx = dists.argmin(axis=1).astype(np.int32)  # (oc,)

            sv_idx = (i1 + j) // d
            indices[:, sv_idx] = idx
            q_sv = centroids_f64[idx]  # (oc, d)

            for c in range(d):
                col = j + c
                err = (W1[:, col] - q_sv[:, c]) / Hinv1[col, col]
                Err1[:, col] = err
                if col + 1 < count:
                    W1[:, col + 1 :] -= (
                        err[:, np.newaxis]
                        * Hinv1[col, col + 1 : count][np.newaxis, :]
                    )

        if i2 < ic:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    return indices


# ── Main entry point ──────────────────────────────────────────────────────


def vq_quantize(
    W_quant: torch.Tensor,
    H: torch.Tensor,
    cfg: QuantConfig,
) -> VQResult:
    """
    VPTQ-based vector quantization.

    Hessian-weighted k-means + GPTQ error propagation + column ordering.

    Two-phase approach:
    1. K-means codebook training: batched across all experts on GPU with
       true weighted k-means (no oversampling approximation).
    2. GPTQ error propagation: per-expert on CPU (inherently sequential).

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
    d = cfg.vq_d

    # ── Padding ───────────────────────────────────────────────────────────
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
    ic_h = ic_padded
    damp = cfg.vptq_damp_percent * H.diagonal().mean()
    H_damp = H + damp * torch.eye(ic_h, device=H.device, dtype=H.dtype)
    H_inv = torch.linalg.inv(H_damp.double()).float()
    Hinv = torch.linalg.cholesky(H_inv.double(), upper=True).float()

    # ── Column ordering ──────────────────────────────────────────────────
    col_perm = torch.argsort(Hinv.diagonal())
    col_invperm = torch.argsort(col_perm)
    Hinv = Hinv[col_perm][:, col_perm]
    W_quant = W_quant[:, :, col_perm]
    h_diag = H.diagonal()[col_perm]  # (ic_h,) — keep as tensor for GPU k-means

    # ── Phase 1: Batched weighted k-means (GPU) ──────────────────────────
    n_subvecs = ic_padded // d
    # Reshape weights into subvectors: (n, oc, n_subvecs, d)
    flat = W_quant.reshape(n, oc, n_subvecs, d)
    # Training data: (n, oc * n_subvecs, d)
    train_data = flat.reshape(n, oc * n_subvecs, d)

    # Per-subvector Hessian weights, broadcast across oc rows
    subvec_weights = h_diag.reshape(n_subvecs, d).mean(dim=1)  # (n_subvecs,)
    subvec_weights = subvec_weights.clamp(min=0)
    # Expand to match training data: each of oc rows gets the same
    # subvector weight pattern -> (oc * n_subvecs,)
    per_vec_weights = subvec_weights.repeat(oc)  # (oc * n_subvecs,)
    # Broadcast to all experts: (n, oc * n_subvecs)
    per_vec_weights = per_vec_weights.unsqueeze(0).expand(n, -1)

    all_centroids = _batched_weighted_kmeans(
        train_data, per_vec_weights, K, cfg.vq_kmeans_niter, seed=cfg.seed
    )  # (n, K, d)

    # ── Phase 2: GPTQ error propagation (CPU, parallel across experts) ───
    Hinv_np = Hinv.cpu().numpy().astype(np.float64)
    centroids_cpu = all_centroids.cpu().float().numpy()

    work_items = []
    for ei in range(n):
        W_np = W_quant[ei].cpu().float().numpy().astype(np.float64)
        work_items.append((W_np, Hinv_np, centroids_cpu[ei], d, cfg.vptq_block_size))

    with ThreadPoolExecutor(max_workers=cfg.vq_num_threads) as pool:
        results = list(pool.map(lambda args: _gptq_with_codebook(*args), work_items))

    # ── Collect results ──────────────────────────────────────────────────
    all_indices = [torch.from_numpy(idx) for idx in results]

    return VQResult(
        main_indices=torch.stack(all_indices),
        main_codebooks=all_centroids.half().cpu(),
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

    Looks up each subvector's centroid from the codebook, undoes column
    permutation, and strips padding.

    Args:
        result: VQResult from vq_quantize().
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
