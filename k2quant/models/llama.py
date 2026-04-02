from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..dense_block import QuantizableMLP


class LlamaMLP(QuantizableMLP):
    """QuantizableMLP for LLaMA-2 and similar dense architectures.

    Supports any HF LlamaForCausalLM variant (LLaMA-2 7B/13B/70B,
    Code Llama, etc.) whose MLP has gate_proj, up_proj, down_proj.
    """

    @classmethod
    def from_hf_module(
        cls,
        hf_mlp: nn.Module,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> LlamaMLP:
        """Convert a HuggingFace LLaMA MLP block to LlamaMLP.

        Expected HF module structure:
            hf_mlp.gate_proj.weight: (intermediate_size, hidden_size)
            hf_mlp.up_proj.weight:   (intermediate_size, hidden_size)
            hf_mlp.down_proj.weight: (hidden_size, intermediate_size)
            hf_mlp.act_fn:           activation function (SiLU)
        """
        gate_w = hf_mlp.gate_proj.weight.data
        up_w = hf_mlp.up_proj.weight.data
        down_w = hf_mlp.down_proj.weight.data

        intermediate_size, hidden_size = gate_w.shape

        block = cls(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            act_fn=hf_mlp.act_fn,
            device=device,
            dtype=dtype,
        )

        # Fuse gate + up into (1, 2*inter, hidden)
        fused = torch.cat([gate_w, up_w], dim=0)
        block.gate_up_proj.data.copy_(fused.unsqueeze(0))
        block.down_proj.data.copy_(down_w.unsqueeze(0))

        return block
