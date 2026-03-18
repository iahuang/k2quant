"""QwenMoEBlock: QuantizableMoEBlock for Qwen1.5-MoE models.

The only model-specific code is from_hf_module(), which knows how to
extract weights from Qwen's HuggingFace MoE implementation:
    - experts.gate_up_proj → gate_up_proj (already fused)
    - experts.down_proj → down_proj
    - gate.weight → router
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..moe_block import QuantizableMoEBlock


class QwenMoEBlock(QuantizableMoEBlock):
    """QuantizableMoEBlock for Qwen1.5-MoE-A2.7B and similar architectures."""

    @classmethod
    def from_hf_module(
        cls,
        hf_moe: nn.Module,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> QwenMoEBlock:
        """Convert a Qwen HF MoE module to a QwenMoEBlock.

        Expected HF module structure:
            hf_moe.experts.num_experts: int
            hf_moe.gate.weight: (num_experts, hidden_size)
            hf_moe.gate.top_k: int
            hf_moe.experts.gate_up_proj: (num_experts, 2*inter, hidden)
            hf_moe.experts.down_proj: (num_experts, hidden, inter)
            hf_moe.experts.act_fn: activation function
        """
        block = cls(
            num_experts=hf_moe.experts.num_experts,
            hidden_size=hf_moe.gate.weight.shape[1],
            intermediate_size=hf_moe.experts.down_proj.shape[2],
            top_k=hf_moe.gate.top_k,
            act_fn=hf_moe.experts.act_fn,
            device=device,
            dtype=dtype,
        )
        block.gate_up_proj.data.copy_(hf_moe.experts.gate_up_proj.data)
        block.down_proj.data.copy_(hf_moe.experts.down_proj.data)
        block.router.data.copy_(hf_moe.gate.weight.data)
        return block
