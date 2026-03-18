"""QuantizableMoEBlock: drop-in MoE expert block with native BCOS support.

Replaces HuggingFace MoE modules with a standardized nn.Module that
owns the expert dispatch loop and applies BCOS corrections natively
in forward() — no monkeypatching needed.

Standardized weight layout:
    gate_up_proj: (num_experts, 2*intermediate_size, hidden_size)
    down_proj:    (num_experts, hidden_size, intermediate_size)
    router:       (num_experts, hidden_size)

Subclasses implement from_hf_module() to copy weights from a specific
HF architecture into this layout. The forward path and BCOS application
are shared across all MoE architectures.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .projection import BCOSLayout


class QuantizableMoEBlock(nn.Module):
    """Base MoE expert block with native BCOS support.

    After quantization, BCOS correction params are set directly on this
    module. The forward() checks for their presence and applies them
    inline — no forward patching required.

    Attributes:
        num_experts: Number of experts.
        hidden_size: Model hidden dimension (input to gate_up, output of down).
        intermediate_size: Expert intermediate dimension.
        top_k: Number of experts activated per token.
        act_fn: Activation function applied to the gate output (e.g. F.silu).
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        act_fn: Callable = F.silu,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.act_fn = act_fn

        # Expert weights — standardized layout
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        self.router = nn.Parameter(
            torch.empty(num_experts, hidden_size)
        )

        # BCOS correction params — None until quantization sets them.
        # Not nn.Parameters (not trained), just tensors on the right device.
        self.bcos_scale_gate: Optional[torch.Tensor] = None
        self.bcos_bias_gate: Optional[torch.Tensor] = None
        self.bcos_scale_up: Optional[torch.Tensor] = None
        self.bcos_bias_up: Optional[torch.Tensor] = None
        self.bcos_scale_down: Optional[torch.Tensor] = None
        self.bcos_bias_down: Optional[torch.Tensor] = None

        # Calibration collection state
        self._collecting: bool = False
        self._collected_inputs: list[torch.Tensor] = []

    def start_collecting(self) -> None:
        """Enable calibration input collection in forward()."""
        self._collecting = True
        self._collected_inputs = []

    def stop_collecting(self) -> list[torch.Tensor]:
        """Disable collection and return captured inputs."""
        self._collecting = False
        inputs = self._collected_inputs
        self._collected_inputs = []
        return inputs

    def set_bcos_params(
        self,
        gate_up_bcos: dict[str, tuple[torch.Tensor, torch.Tensor]],
        down_bcos: dict[str, tuple[torch.Tensor, torch.Tensor]],
        device: str,
    ) -> None:
        """Set BCOS correction parameters from quantization results.

        Args:
            gate_up_bcos: BCOS params for gate_up_proj, keyed by layout names.
                For fused (default): {"gate": (s, b), "up": (s, b)}.
                For unfused: {"gate_up": (s, b)}.
            down_bcos: BCOS params for down_proj, keyed by layout names.
                Default: {"down": (s, b)}.
            device: Device to place tensors on.
        """
        gate_up_layout = self.get_gate_up_bcos_layout()
        if len(gate_up_layout.names) == 2:
            # Fused: separate gate and up corrections
            self.bcos_scale_gate = gate_up_bcos[gate_up_layout.names[0]][0].to(device)
            self.bcos_bias_gate = gate_up_bcos[gate_up_layout.names[0]][1].to(device)
            self.bcos_scale_up = gate_up_bcos[gate_up_layout.names[1]][0].to(device)
            self.bcos_bias_up = gate_up_bcos[gate_up_layout.names[1]][1].to(device)
        else:
            # Unfused: single correction applied to both gate and up
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
        Override for models with separate gate/up projections
        (return a single-entry layout) or different split ratios.
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MoE forward with optional BCOS correction.

        Args:
            hidden_states: Input activations. (num_tokens, hidden_size).

        Returns:
            Output activations. (num_tokens, hidden_size).
        """
        if self._collecting:
            self._collected_inputs.append(hidden_states.detach().cpu())

        # Routing
        router_logits = F.linear(hidden_states, self.router)
        routing_weights = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, k=self.top_k, dim=-1
        )
        # Normalize top-k weights to sum to 1
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Expert dispatch
        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = F.one_hot(
                top_k_indices, num_classes=self.num_experts
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

    def compute_routed_calibration(
        self,
        X_layer: torch.Tensor,
        W_gate_up_orig: torch.Tensor,
    ) -> torch.Tensor:
        """Compute down_proj calibration activations via router dispatch.

        Routes tokens through the router, dispatches to experts using
        original (pre-quantization) gate_up weights, and computes
        SiLU(gate) * up intermediates.

        Args:
            X_layer: Layer input activations. (b, hidden_size). float32.
            W_gate_up_orig: Original gate_up_proj weights before quantization.
                (num_experts, 2*intermediate_size, hidden_size). float32.

        Returns:
            Intermediate activations for down_proj calibration.
            (b', intermediate_size). float32.
        """
        device = self.router.device

        with torch.no_grad():
            router_logits = F.linear(X_layer.half(), self.router)
            routing_weights = F.softmax(router_logits, dim=-1)
            _, top_k_indices = torch.topk(
                routing_weights, k=self.top_k, dim=-1
            )

            all_intermediates = []
            for ei in range(self.num_experts):
                mask = (top_k_indices == ei).any(dim=-1)
                if mask.sum() == 0:
                    continue
                tokens = X_layer[mask].half()
                gu_out = F.linear(tokens, W_gate_up_orig[ei].half().to(device))
                gate_e, up_e = gu_out.chunk(2, dim=-1)
                intermediate = self.act_fn(gate_e) * up_e
                all_intermediates.append(intermediate)

            X_down = torch.cat(all_intermediates, dim=0).float()

        return X_down

    @classmethod
    def from_hf_module(cls, hf_module: nn.Module) -> QuantizableMoEBlock:
        """Create a QuantizableMoEBlock by copying weights from an HF module.

        Subclasses must override this to handle architecture-specific
        weight names and layouts.

        Args:
            hf_module: The HuggingFace MoE module to convert.

        Returns:
            A new QuantizableMoEBlock with weights copied from hf_module.
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement from_hf_module() to convert "
            f"HuggingFace modules into the standardized layout."
        )
