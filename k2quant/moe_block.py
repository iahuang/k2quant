"""QuantizableExperts: drop-in replacement for HF fused experts with BCOS.

Replaces HuggingFace's experts sub-module within an MoE block, preserving
the parent block's routing and shared expert code unchanged. This avoids
compound float16 divergence from reimplementing the full MoE forward.

The forward signature matches HF's experts calling convention:
    experts(hidden_states, top_k_index, top_k_weights) -> final_hidden_states

Standardized weight layout:
    gate_up_proj: (num_experts, 2*intermediate_size, hidden_size)
    down_proj:    (num_experts, hidden_size, intermediate_size)

Subclasses implement from_hf_module() to copy weights from a specific
HF architecture into this layout. The dispatch loop and BCOS application
are shared across all MoE architectures.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .projection import BCOSLayout


class QuantizableExperts(nn.Module):
    """Drop-in replacement for HF fused experts with native BCOS support.

    After quantization, BCOS correction params are set directly on this
    module. The forward() checks for their presence and applies them
    inline — no forward patching required.

    Attributes:
        num_experts: Number of experts.
        hidden_size: Model hidden dimension (input to gate_up, output of down).
        intermediate_size: Expert intermediate dimension.
        act_fn: Activation function applied to the gate output (e.g. F.silu).
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        act_fn: Callable = F.silu,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.act_fn = act_fn

        # Expert weights — standardized layout
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size,
                        device=device, dtype=dtype)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size,
                        device=device, dtype=dtype)
        )

        # BCOS correction params — None until quantization sets them.
        self.bcos_scale_gate: Optional[torch.Tensor] = None
        self.bcos_bias_gate: Optional[torch.Tensor] = None
        self.bcos_scale_up: Optional[torch.Tensor] = None
        self.bcos_bias_up: Optional[torch.Tensor] = None
        self.bcos_scale_down: Optional[torch.Tensor] = None
        self.bcos_bias_down: Optional[torch.Tensor] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Expert dispatch with optional BCOS correction.

        Matches HF's experts.forward() calling convention so the parent
        MoE block's routing and shared expert code remain unchanged.

        Args:
            hidden_states: Flattened input activations. (num_tokens, hidden_size).
            top_k_index: Expert indices per token. (num_tokens, top_k). int64.
            top_k_weights: Routing weights per token. (num_tokens, top_k). float.

        Returns:
            Output activations. (num_tokens, hidden_size).
        """
        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = F.one_hot(
                top_k_index, num_classes=self.num_experts
            )  # (tokens, top_k, num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)  # (num_experts, top_k, tokens)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx >= self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            # gate_up_proj with split BCOS
            gu_out = F.linear(current_state, self.gate_up_proj[expert_idx])
            gate, up = gu_out.chunk(2, dim=-1)

            if self.bcos_scale_gate is not None:
                gate = (1 + self.bcos_scale_gate[expert_idx]) * gate + self.bcos_bias_gate[expert_idx]
                up = (1 + self.bcos_scale_up[expert_idx]) * up + self.bcos_bias_up[expert_idx]

            current_hidden_states = self.act_fn(gate) * up

            # down_proj with BCOS
            current_hidden_states = F.linear(
                current_hidden_states, self.down_proj[expert_idx]
            )

            if self.bcos_scale_down is not None:
                current_hidden_states = (
                    (1 + self.bcos_scale_down[expert_idx]) * current_hidden_states
                    + self.bcos_bias_down[expert_idx]
                )

            # Weight by routing score
            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        return final_hidden_states

    def set_bcos_params(
        self,
        gate_up_bcos: dict[str, tuple[torch.Tensor, torch.Tensor]],
        down_bcos: dict[str, tuple[torch.Tensor, torch.Tensor]],
        device: str,
    ) -> None:
        """Set BCOS correction parameters from quantization results."""
        gate_up_layout = self.get_gate_up_bcos_layout()
        if len(gate_up_layout.names) == 2:
            self.bcos_scale_gate = gate_up_bcos[gate_up_layout.names[0]][0].to(device)
            self.bcos_bias_gate = gate_up_bcos[gate_up_layout.names[0]][1].to(device)
            self.bcos_scale_up = gate_up_bcos[gate_up_layout.names[1]][0].to(device)
            self.bcos_bias_up = gate_up_bcos[gate_up_layout.names[1]][1].to(device)
        else:
            name = gate_up_layout.names[0]
            self.bcos_scale_gate = gate_up_bcos[name][0].to(device)
            self.bcos_bias_gate = gate_up_bcos[name][1].to(device)
            self.bcos_scale_up = None
            self.bcos_bias_up = None

        down_name = self.get_down_bcos_layout().names[0]
        self.bcos_scale_down = down_bcos[down_name][0].to(device)
        self.bcos_bias_down = down_bcos[down_name][1].to(device)

    def get_gate_up_bcos_layout(self) -> BCOSLayout:
        """Return the BCOS layout for gate_up_proj.

        Default: fused gate+up with independent BCOS per half.
        """
        return BCOSLayout(
            split_sizes=[self.intermediate_size, self.intermediate_size],
            names=["gate", "up"],
        )

    def get_down_bcos_layout(self) -> BCOSLayout:
        """Return the BCOS layout for down_proj.

        Default: single unfused projection.
        """
        return BCOSLayout(
            split_sizes=[self.hidden_size],
            names=["down"],
        )

    @classmethod
    def from_hf_module(
        cls,
        hf_moe_block: nn.Module,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> QuantizableExperts:
        """Create a QuantizableExperts by copying weights from an HF MoE block.

        Subclasses must override this to handle architecture-specific
        weight names and layouts.

        Args:
            hf_moe_block: The HuggingFace MoE block containing experts.
            device: Target device for the new module's parameters.
            dtype: Target dtype for the new module's parameters.

        Returns:
            A new QuantizableExperts with weights copied from the HF experts.
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement from_hf_module() to convert "
            f"HuggingFace experts into the standardized layout."
        )

    @staticmethod
    def get_routing_info(
        hf_moe_block: nn.Module,
    ) -> tuple[torch.Tensor, int]:
        """Return routing info for down_proj calibration.

        Subclasses must override this.

        Args:
            hf_moe_block: The HuggingFace MoE block.

        Returns:
            (router_weight, top_k) tuple.
        """
        raise NotImplementedError
