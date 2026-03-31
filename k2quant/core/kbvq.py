from __future__ import annotations

import torch


def bcos(
    W_vq: torch.Tensor,
    W_orig: torch.Tensor,
    X: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute BCOS affine correction factors.

    Finds scale s and bias b such that:
        y_corrected = (1 + s) * y_vq + b ≈ y_orig

    where y_orig = W_orig @ X^T and y_vq = W_vq @ X^T.

    Args:
        W_vq: Quantized weight matrices.
            (n_experts, oc, ic). float32.
        W_orig: Original weight matrices.
            (n_experts, oc, ic). float32.
        X: Calibration activations.
            (b, ic). float32.

    Returns:
        scale: Per-expert, per-channel scale correction (s).
            (n_experts, oc). float16.
            Applied as (1 + s) * y_vq. s=0 means no scaling.
        bias: Per-expert, per-channel bias correction (b).
            (n_experts, oc). float16.

    Notes:
        std of y_vq is clamped to min=1e-8 to avoid division by zero
        when a quantized channel collapses to a constant (degenerate centroid).
    """
    # y_orig[n, b, o] = sum_i W_orig[n, o, i] * X[b, i]
    y = torch.einsum("noi,bi->nbo", W_orig, X)  # (n, b, oc)
    y_vq = torch.einsum("noi,bi->nbo", W_vq, X)  # (n, b, oc)

    # Scale: match standard deviations across batch
    s = (y.std(dim=1) / y_vq.std(dim=1).clamp(min=1e-8)) - 1  # (n, oc)

    # Bias: match means after scaling
    b = y.mean(dim=1) - (1 + s) * y_vq.mean(dim=1)  # (n, oc)

    return s.half(), b.half()


def klt_decomposition(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    KLT (Karhunen-Loeve Transform) of input activations.

    Args:
        X: Calibration activations. (b, ic) for b samples.

    Returns:
        U_X: Forward transform. (ic, ic).
            Maps weights into KLT space: W_hat = W @ U_X.
        U_X_inv: Inverse transform. (ic, ic).
            Maps back to weight space.

    Notes:
        Eigenvalues are clamped to min=1e-8 to handle dead input features
        (near-zero variance channels) without division-by-zero.
    """

    b, _ = X.shape

    # Sample covariance: C = X^T X / (b - 1)
    C = (1 / (b - 1)) * X.T @ X  # (ic, ic)
    # Symmetric eigendecomposition (C is PSD)
    L, U = torch.linalg.eigh(C)  # L: (ic,), U: (ic, ic)
    L = L.clamp(min=1e-8)
    L_diag = torch.diag(L**0.5)  # (ic, ic)
    # U_X = eigenvectors scaled by sqrt(eigenvalues)
    U_X = U @ L_diag  # (ic, ic)
    U_X_inv = torch.linalg.inv(L_diag) @ U.T  # (ic, ic)

    return U_X, U_X_inv


def idre(
    X: torch.Tensor,
    W: torch.Tensor,
    k_factor: float = 1 / 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Input-Driven Redundancy Elimination (IDRE).

    Args:
        X: Calibration activations. (b, ic).
        W: Expert weight matrices. (n_experts, oc, ic).
        k_factor: Rank fraction. k = int(ic * k_factor).

    Returns:
        V_k: Per-expert coefficients. (n_experts, oc, k). Same dtype/device as W.
        basis: Shared low-rank basis. (k, ic). Same dtype/device as W.
            Reconstruct via: W_share = V_k @ basis
    """

    n, oc, ic = W.shape
    k = max(1, int(ic * k_factor))

    U_X, U_X_inv = klt_decomposition(X)  # both (ic, ic)

    W_hat = W @ U_X  # (n, oc, ic)

    W_bar = W_hat.reshape(-1, ic)  # (n*oc, ic)
    U, S, Vh = torch.linalg.svd(W_bar.T, full_matrices=False)
    # U: (ic, min(ic, n*oc)), S: (min(...),), Vh: (min(...), n*oc)

    U_k = U[:, :k]  # (ic, k)
    S_k = torch.diag(S[:k])  # (k, k)
    V_k = (Vh[:k, :].T @ S_k).reshape(n, oc, k)  # (n, oc, k)

    # basis = U_k^T @ U_X_inv — shared across all experts
    basis = U_k.T @ U_X_inv  # (k, ic)

    return V_k, basis
