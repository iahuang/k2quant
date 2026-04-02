from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .projection import BCOSLayout


class QuantizableMLP(nn.Module):
    """Base class for dense (non-MoE) MLP blocks.

    Stores weights in the same (1, oc, ic) layout used by QuantizableExperts
    so that quantize_projection() works unchanged.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        act_fn: Callable = F.silu,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.act_fn = act_fn

        # Weights — (1, oc, ic) to match the n_experts dimension
        self.gate_up_proj = nn.Parameter(
            torch.empty(1, 2 * intermediate_size, hidden_size,
                        device=device, dtype=dtype)
        )
        self.down_proj = nn.Parameter(
            torch.empty(1, hidden_size, intermediate_size,
                        device=device, dtype=dtype)
        )

        # BCOS correction params — None until quantization sets them.
        self.bcos_scale_gate: Optional[torch.Tensor] = None
        self.bcos_bias_gate: Optional[torch.Tensor] = None
        self.bcos_scale_up: Optional[torch.Tensor] = None
        self.bcos_bias_up: Optional[torch.Tensor] = None
        self.bcos_scale_down: Optional[torch.Tensor] = None
        self.bcos_bias_down: Optional[torch.Tensor] = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size) or (tokens, hidden_size).

        Returns:
            Output with same leading dimensions.
        """
        # gate_up
        gu_out = F.linear(hidden_states, self.gate_up_proj[0])
        gate, up = gu_out.chunk(2, dim=-1)

        if self.bcos_scale_gate is not None:
            gate = (1 + self.bcos_scale_gate) * gate + self.bcos_bias_gate
            up = (1 + self.bcos_scale_up) * up + self.bcos_bias_up

        intermediate = self.act_fn(gate) * up

        # down
        out = F.linear(intermediate, self.down_proj[0])

        if self.bcos_scale_down is not None:
            out = (1 + self.bcos_scale_down) * out + self.bcos_bias_down

        return out

    def set_bcos_params(
        self,
        gate_up_bcos: dict[str, tuple[torch.Tensor, torch.Tensor]],
        down_bcos: dict[str, tuple[torch.Tensor, torch.Tensor]],
        device: str,
    ) -> None:
        """Set BCOS correction parameters from quantization results."""
        gate_up_layout = self.get_gate_up_bcos_layout()
        if len(gate_up_layout.names) == 2:
            # Squeeze the n_experts=1 dim for dense forward
            self.bcos_scale_gate = gate_up_bcos[gate_up_layout.names[0]][0][0].to(device)
            self.bcos_bias_gate = gate_up_bcos[gate_up_layout.names[0]][1][0].to(device)
            self.bcos_scale_up = gate_up_bcos[gate_up_layout.names[1]][0][0].to(device)
            self.bcos_bias_up = gate_up_bcos[gate_up_layout.names[1]][1][0].to(device)
        else:
            name = gate_up_layout.names[0]
            self.bcos_scale_gate = gate_up_bcos[name][0][0].to(device)
            self.bcos_bias_gate = gate_up_bcos[name][1][0].to(device)
            self.bcos_scale_up = None
            self.bcos_bias_up = None

        down_name = self.get_down_bcos_layout().names[0]
        self.bcos_scale_down = down_bcos[down_name][0][0].to(device)
        self.bcos_bias_down = down_bcos[down_name][1][0].to(device)

    def get_gate_up_bcos_layout(self) -> BCOSLayout:
        return BCOSLayout(
            split_sizes=[self.intermediate_size, self.intermediate_size],
            names=["gate", "up"],
        )

    def get_down_bcos_layout(self) -> BCOSLayout:
        return BCOSLayout(
            split_sizes=[self.hidden_size],
            names=["down"],
        )

    @classmethod
    def from_hf_module(
        cls,
        hf_mlp: nn.Module,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> QuantizableMLP:
        """Convert a HuggingFace MLP module into a QuantizableMLP.

        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement from_hf_module() to convert "
            f"HuggingFace MLP into the standardized layout."
        )
