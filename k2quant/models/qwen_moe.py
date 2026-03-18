"""QwenExperts: QuantizableExperts for Qwen1.5-MoE models.

The only model-specific code is from_hf_module() and get_routing_info(),
which know how to extract weights and routing from Qwen's HF MoE block:
    - experts.gate_up_proj -> gate_up_proj (already fused)
    - experts.down_proj -> down_proj
    - gate.weight -> router (for down_proj calibration routing)
    - gate.top_k -> top_k
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..moe_block import QuantizableExperts


class QwenExperts(QuantizableExperts):
    """QuantizableExperts for Qwen1.5-MoE-A2.7B and similar architectures."""

    @classmethod
    def from_hf_module(
        cls,
        hf_moe_block: nn.Module,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> QwenExperts:
        """Convert a Qwen HF MoE block's experts to QwenExperts.

        Expected HF module structure:
            hf_moe_block.experts.num_experts: int
            hf_moe_block.experts.gate_up_proj: (num_experts, 2*inter, hidden)
            hf_moe_block.experts.down_proj: (num_experts, hidden, inter)
            hf_moe_block.experts.act_fn: activation function
            hf_moe_block.gate.weight: (num_experts, hidden_size)
        """
        experts = hf_moe_block.experts
        block = cls(
            num_experts=experts.num_experts,
            hidden_size=hf_moe_block.gate.weight.shape[1],
            intermediate_size=experts.down_proj.shape[2],
            act_fn=experts.act_fn,
            device=device,
            dtype=dtype,
        )
        block.gate_up_proj.data.copy_(experts.gate_up_proj.data)
        block.down_proj.data.copy_(experts.down_proj.data)

        return block

    @staticmethod
    def get_routing_info(
        hf_moe_block: nn.Module,
    ) -> tuple[torch.Tensor, int]:
        """Return (router_weight, top_k) from Qwen's gate module."""
        gate = hf_moe_block.gate
        return gate.weight, gate.top_k
