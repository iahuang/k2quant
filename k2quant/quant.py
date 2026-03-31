from __future__ import annotations
import dataclasses

import torch
from .core import QuantConfig, VQResult, idre, vq_quantize, vq_reconstruct


def w_quantize_and_reconstruct(
    W: torch.Tensor,
    X_calib: torch.Tensor,
    cfg: QuantConfig,
) -> torch.Tensor:
    """IDRE + VPTQ pipeline, returning reconstructed weights.

    Convenience wrapper around w_quantize() that immediately
    reconstructs the full weight.

    Args:
        W: Expert weight matrices.
            Shape: (n_experts, oc, ic). float32.
        X_calib: Calibration activations.
            Shape: (b, ic). float32. On same device as W.
        cfg: Quantization configuration (QuantConfig).

    Returns:
        W_vq: Reconstructed quantized weights.
            Shape: (n_experts, oc, ic). float32. Same device as W.
    """
    qw = w_quantize(W, X_calib, cfg)
    return qw.reconstruct().to(W.device)


def w_quantize(
    W: torch.Tensor,
    X_calib: torch.Tensor,
    cfg: QuantConfig,
) -> WeightQuantization:
    """
    IDRE + VPTQ pipeline, returning the compressed representation.

    Args:
        W: Expert weight matrices.
            Shape: (n_experts, oc, ic). float32.
        X_calib: Calibration activations.
            Shape: (b, ic). float32. On same device as W.
        cfg: Quantization configuration (QuantConfig).

    Returns:
        WeightQuantization holding all compressed data. BCOS fields are None —
        call compute_bcos_params() separately and assign to .bcos_scale/.bcos_bias.
    """
    n, oc, ic = W.shape

    # Step 1: IDRE — extract shared low-rank component in factored form
    V_k, basis = idre(X_calib, W, k_factor=cfg.k_factor)
    W_share = V_k @ basis  # (n, oc, ic) — reconstruct for residual computation
    W_quant = W - W_share  # (n, oc, ic) — residual to be quantized

    # Step 2: Hessian from calibration activations
    H = (X_calib.T @ X_calib) / X_calib.shape[0]  # (ic, ic)

    # Step 3: VPTQ quantization of residual
    vq = vq_quantize(W_quant, H, cfg)

    return WeightQuantization(
        idre_vk=V_k.half(),
        idre_basis=basis.half(),
        vq=vq,
        n_experts=n,
        oc=oc,
        ic=ic,
    )


@dataclasses.dataclass
class WeightQuantization:
    # IDRE shared component — stored in factored form for compression
    idre_vk: torch.Tensor
    """Per-expert coefficients. Shape: (n_experts, oc, k). float16."""

    idre_basis: torch.Tensor
    """Shared low-rank basis. Shape: (k, ic). float16."""

    # VPTQ compressed residual
    vq: VQResult
    """Indices, codebooks, and column permutation for the quantized residual."""

    # BCOS correction (set after quantization + BCOS computation)
    bcos_scale: torch.Tensor | None = None
    """Per-channel scale. Shape: (n_experts, oc). float16. None until computed."""

    bcos_bias: torch.Tensor | None = None
    """Per-channel bias. Shape: (n_experts, oc). float16. None until computed."""

    # Original dimensions (needed for reconstruction)
    n_experts: int = 0
    oc: int = 0
    ic: int = 0

    def reconstruct(self) -> torch.Tensor:
        """Reconstruct the full quantized weight from compressed data.

        Returns:
            W_vq: Reconstructed weight.
                Shape: (n_experts, oc, ic). float32.
        """
        W_share = (self.idre_vk.float() @ self.idre_basis.float())
        W_quant_vq = vq_reconstruct(self.vq, self.n_experts, self.oc, self.ic)
        return W_share + W_quant_vq.float().to(self.idre_vk.device)

    def to_tensors(self, prefix: str) -> dict[str, torch.Tensor]:
        """Flatten to a dict of named tensors for safetensors serialization.

        All values are contiguous CPU tensors. Scalar metadata is stored
        as a 1-D int64 tensor under {prefix}.meta.

        Args:
            prefix: Key prefix, e.g. "layers.0.gate_up". Each tensor is
                stored as "{prefix}.{field_name}".

        Returns:
            Dict mapping "{prefix}.{name}" -> Tensor.
        """
        p = prefix

        tensors = {
            f"{p}.idre_vk": self.idre_vk.contiguous().cpu(),
            f"{p}.idre_basis": self.idre_basis.contiguous().cpu(),
            f"{p}.main_indices": self.vq.main_indices.contiguous().cpu(),
            f"{p}.main_codebooks": self.vq.main_codebooks.contiguous().cpu(),
            f"{p}.col_invperm": self.vq.col_invperm.contiguous().cpu(),
            f"{p}.meta": torch.tensor(
                [
                    self.n_experts,
                    self.oc,
                    self.ic,
                    self.vq.oc_pad,
                    self.vq.oc_padded,
                    self.vq.ic_pad,
                    self.vq.ic_padded,
                ],
                dtype=torch.int64,
            ),
        }

        if self.bcos_scale is not None:
            tensors[f"{p}.bcos_scale"] = self.bcos_scale.contiguous().cpu()
        if self.bcos_bias is not None:
            tensors[f"{p}.bcos_bias"] = self.bcos_bias.contiguous().cpu()
        return tensors

    @classmethod
    def from_tensors(
        cls, tensors: dict[str, torch.Tensor], prefix: str
    ) -> WeightQuantization:
        """Reconstruct a WeightQuantization from a flat tensor dict.

        Inverse of to_tensors(). Use after loading from safetensors.

        Args:
            tensors: Dict of tensors (e.g. from safetensors.load_file()).
            prefix: Same prefix used in to_tensors().

        Returns:
            WeightQuantization with all fields populated.
        """
        p = prefix
        meta = tensors[f"{p}.meta"]
        meta_list = meta.tolist()
        n_experts, oc, ic = meta_list[0], meta_list[1], meta_list[2]
        oc_pad, oc_padded = meta_list[3], meta_list[4]
        ic_pad, ic_padded = meta_list[5], meta_list[6]

        vq = VQResult(
            main_indices=tensors[f"{p}.main_indices"],
            main_codebooks=tensors[f"{p}.main_codebooks"],
            oc_pad=oc_pad,
            oc_padded=oc_padded,
            ic_pad=ic_pad,
            ic_padded=ic_padded,
            col_invperm=tensors[f"{p}.col_invperm"],
        )

        return cls(
            idre_vk=tensors[f"{p}.idre_vk"],
            idre_basis=tensors[f"{p}.idre_basis"],
            vq=vq,
            bcos_scale=tensors.get(f"{p}.bcos_scale"),
            bcos_bias=tensors.get(f"{p}.bcos_bias"),
            n_experts=n_experts,
            oc=oc,
            ic=ic,
        )
