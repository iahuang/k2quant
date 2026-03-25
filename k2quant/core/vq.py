from __future__ import annotations

import dataclasses
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn

from .config import QuantConfig

_bin_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'bin'))
if _bin_dir not in sys.path:
    sys.path.insert(0, _bin_dir)

import vptq as _vptq
# _vptq = None

import faiss

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
        Tuple of (indices, centroids, kmeans_train_seconds, errprop_seconds).
    """
    W_np, Hinv_np, h_diag_np, V, K, niter, block_size = args
    oc, ic = W_np.shape  # single expert: (oc_padded, ic)
    n_row_subvecs = oc // V

    # ── 1. Hessian-weighted k-means ──────────────────────────────────────
    train_parts = []
    for col in range(ic):
        col_subvecs = W_np[:, col].reshape(n_row_subvecs, V)  # (n_row_subvecs, V)
        train_parts.append(col_subvecs)
    train_data = np.vstack(train_parts).astype(np.float32)
    if h_diag_np.sum() > 0:
        train_weights = np.repeat(h_diag_np.astype(np.float32), n_row_subvecs)
    else:
        train_weights = np.ones(train_data.shape[0], dtype=np.float32)

    km = faiss.Kmeans(V, K, niter=niter, verbose=False, gpu=False)
    t_km0 = time.perf_counter()
    km.train(train_data, weights=train_weights)
    t_kmeans = time.perf_counter() - t_km0
    centroids = km.centroids.copy().astype(np.float32)  # (K, V)

    t_ep0 = time.perf_counter()
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
    t_errprop = time.perf_counter() - t_ep0

    return indices, centroids, t_kmeans, t_errprop


def _prepare_vq_inputs(
    W_quant: torch.Tensor,
    H: torch.Tensor,
    cfg: QuantConfig,
) -> tuple:
    """Shared input preparation for both C++ and Python VQ paths.

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


def _vq_quantize_python(
    W_quant: torch.Tensor,
    Hinv_np: np.ndarray,
    h_diag_np: np.ndarray,
    V: int,
    K: int,
    cfg: QuantConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-Python fallback using faiss k-means."""


    n = W_quant.shape[0]

    work_items = []
    for ei in range(n):
        W_np = W_quant[ei].cpu().float().numpy()
        work_items.append((W_np, Hinv_np, h_diag_np, V, K,
                           cfg.vq_kmeans_niter, cfg.vptq_block_size))

    with ThreadPoolExecutor(max_workers=cfg.vq_num_threads) as pool:
        results = list(pool.map(_vptq_one_expert_row, work_items))

    all_indices = [r[0] for r in results]
    all_codebooks = [r[1] for r in results]
    total_kmeans = sum(r[2] for r in results)
    total_errprop = sum(r[3] for r in results)
    print(
        f"[_vq_quantize_python] k-means train (sum over threads): {total_kmeans:.3f}s, "
        f"error propagation (sum over threads): {total_errprop:.3f}s"
    )
    return np.stack(all_indices), np.stack(all_codebooks)


def _vq_quantize_hybrid(
    W_quant: torch.Tensor,
    Hinv_np: np.ndarray,
    h_diag_np: np.ndarray,
    V: int,
    K: int,
    cfg: QuantConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Hybrid: FAISS k-means + C++ error propagation."""
    n = W_quant.shape[0]
    oc_padded = W_quant.shape[1]
    ic = W_quant.shape[2]
    n_row_subvecs = oc_padded // V

    def train_one(ei: int) -> np.ndarray:
        W_np = W_quant[ei].cpu().float().numpy()

        train_parts = []
        for col in range(ic):
            col_subvecs = W_np[:, col].reshape(n_row_subvecs, V)
            train_parts.append(col_subvecs)
        train_data = np.vstack(train_parts).astype(np.float32)

        if h_diag_np.sum() > 0:
            train_weights = np.repeat(h_diag_np.astype(np.float32), n_row_subvecs)
        else:
            train_weights = np.ones(train_data.shape[0], dtype=np.float32)

        init_centroids = _vptq.kmeanspp_init(train_data, K)

        km = faiss.Kmeans(V, K, niter=cfg.vq_kmeans_niter, verbose=False, gpu=False)
        km.train(train_data, weights=train_weights, init_centroids=init_centroids)
        return km.centroids.copy().astype(np.float32)  # (K, V)

    t_km = time.perf_counter()
    with ThreadPoolExecutor(max_workers=cfg.vq_num_threads) as pool:
        all_centroids = list(pool.map(train_one, range(n)))
    print(f"[_vq_quantize_hybrid] k-means: {time.perf_counter() - t_km:.3f}s")

    centroids_np = np.stack(all_centroids)  # (n, K, V)
    centroids_flat = np.ascontiguousarray(centroids_np.reshape(n * K, V))

    t_ep = time.perf_counter()
    W_np = np.ascontiguousarray(W_quant.cpu().float().numpy())
    indices_flat = _vptq.vptq_errprop(W_np, Hinv_np, centroids_flat, V, K, cfg.vptq_block_size)
    print(f"[_vq_quantize_hybrid] error propagation: {time.perf_counter() - t_ep:.3f}s")

    indices_np = indices_flat.reshape(n, n_row_subvecs, ic)
    return indices_np, centroids_np


def vq_quantize(
    W_quant: torch.Tensor,
    H: torch.Tensor,
    cfg: QuantConfig,
) -> VQResult:
    """
    VPTQ-based vector quantization (paper-faithful row-axis mode).

    Hessian-weighted k-means + column-by-column error propagation.
    Uses the compiled C++ kernel when available, otherwise falls back
    to a pure-Python implementation with faiss.

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
    if _vptq is not None and hasattr(_vptq, 'vptq_errprop'):
        indices_np, codebooks_np = _vq_quantize_hybrid(
            W_quant, Hinv_np, h_diag_np, V, K, cfg)
        backend = 'hybrid'
    elif _vptq is not None:
        indices_np, codebooks_np = _vq_quantize_cpp(
            W_quant, Hinv_np, h_diag_np, V, K, cfg)
        backend = 'C++'
    else:
        indices_np, codebooks_np = _vq_quantize_python(
            W_quant, Hinv_np, h_diag_np, V, K, cfg)
        backend = 'Python'
    print(f"[vq_quantize] {n} experts, V={V}, K={K}: {time.time() - t0:.1f}s ({backend})")

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