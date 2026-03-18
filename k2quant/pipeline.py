"""High-level quantization pipeline for MoE models.

Provides quantize_model() as the single entry point: collects calibration
activations on the original HF model, swaps HF MoE modules for
QuantizableMoEBlocks, and quantizes all expert projections with BCOS.
"""

from __future__ import annotations

import gc
import time
from typing import Callable, Optional

import torch
import torch.nn as nn

from .core import QuantConfig
from .moe_block import QuantizableMoEBlock
from .projection import quantize_projection


def _collect_activations_via_hooks(
    model: nn.Module,
    get_moe_parent_and_attr: Callable[[nn.Module, int], tuple[nn.Module, str]],
    num_layers: int,
    calib_data: torch.Tensor,
    device: str,
    batch_size: int,
    log_fn: Callable[[str], None],
) -> dict[int, list[torch.Tensor]]:
    """Collect MoE block inputs using hooks on the original HF modules.

    Runs calibration forward passes through the untouched HF model so that
    activations at each layer are bit-identical to what the original model
    produces. This avoids compound divergence from our reimplemented forward.
    """
    layer_inputs: dict[int, list[torch.Tensor]] = {}
    hooks = []

    for li in range(num_layers):
        parent, attr = get_moe_parent_and_attr(model, li)
        hf_moe = getattr(parent, attr)
        storage: list[torch.Tensor] = []
        layer_inputs[li] = storage

        def make_hook(s):
            def fn(_module, args, _kwargs, output):
                s.append(args[0].detach().cpu())
                return output
            return fn

        h = hf_moe.register_forward_hook(make_hook(storage), with_kwargs=True)
        hooks.append(h)

    n_batches = len(calib_data) // batch_size
    with torch.no_grad():
        for i in range(0, len(calib_data), batch_size):
            batch = calib_data[i : i + batch_size].to(device)
            model(batch)
            batch_idx = i // batch_size
            if batch_idx % 32 == 0:
                log_fn(f"    Batch {batch_idx + 1}/{n_batches}")

    for h in hooks:
        h.remove()

    return layer_inputs


def _swap_moe_blocks(
    model: nn.Module,
    block_cls: type[QuantizableMoEBlock],
    get_moe_parent_and_attr: Callable[[nn.Module, int], tuple[nn.Module, str]],
    num_layers: int,
) -> list[tuple[str, QuantizableMoEBlock]]:
    """Replace HF MoE modules with QuantizableMoEBlocks, return the new blocks."""
    parent_names = {id(m): n for n, m in model.named_modules()}
    blocks = []
    for li in range(num_layers):
        parent, attr = get_moe_parent_and_attr(model, li)
        hf_moe = getattr(parent, attr)
        hf_params = next(hf_moe.parameters())
        qmoe = block_cls.from_hf_module(
            hf_moe, device=hf_params.device, dtype=hf_params.dtype
        )
        setattr(parent, attr, qmoe)
        parent_name = parent_names.get(id(parent), "")
        name = f"{parent_name}.{attr}" if parent_name else attr
        blocks.append((name, qmoe))
        print(f"   {li} Swapped {name}")
        del hf_moe
    gc.collect()
    torch.cuda.empty_cache()
    return blocks


def quantize_model(
    model: nn.Module,
    calib_data: torch.Tensor,
    cfg: QuantConfig,
    block_cls: type[QuantizableMoEBlock],
    get_moe_parent_and_attr: Callable[[nn.Module, int], tuple[nn.Module, str]],
    num_layers: int,
    *,
    device: str = "cuda",
    max_calib_tokens: int = 4096,
    batch_size: int = 2,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Quantize all MoE expert layers of a model in-place.

    Collects calibration activations on the original HF model (ensuring
    bit-identical activations), then swaps HF MoE modules for
    QuantizableMoEBlocks and quantizes each block's projections with BCOS.

    Args:
        model: The HuggingFace model to quantize (modified in-place).
        calib_data: Calibration token IDs. (nsamples, seqlen). int64.
        cfg: Quantization configuration.
        block_cls: QuantizableMoEBlock subclass with from_hf_module().
        get_moe_parent_and_attr: Callable that returns (parent_module, attr_name)
            for the MoE block at each layer index.
            E.g. lambda m, i: (m.model.layers[i], "mlp")
        num_layers: Number of MoE layers to process.
        device: Device for computation.
        max_calib_tokens: Maximum calibration tokens per block.
        batch_size: Batch size for calibration forward passes.
        log_fn: Optional progress logger.
    """
    if log_fn is None:
        log_fn = print

    # Step 1: Collect calibration activations on the original HF model
    log_fn("  Collecting calibration activations (original HF forward)...")
    layer_inputs = _collect_activations_via_hooks(
        model, get_moe_parent_and_attr, num_layers,
        calib_data, device, batch_size, log_fn,
    )
    log_fn("  Activations collected.")

    # Step 2: Swap HF modules for quantizable blocks
    log_fn("  Swapping MoE blocks...")
    moe_blocks = _swap_moe_blocks(
        model, block_cls, get_moe_parent_and_attr, num_layers
    )
    num_blocks = len(moe_blocks)
    log_fn(f"  Swapped {num_blocks} MoE blocks.")

    # Step 3: Quantize block by block
    log_fn("  Quantizing blocks...")
    t_start = time.time()

    for bi, (name, block) in enumerate(moe_blocks):
        t0 = time.time()
        hidden_size = block.hidden_size

        # Prepare activations: concatenate, reshape, subsample
        acts = torch.cat(layer_inputs[bi], dim=0).reshape(-1, hidden_size)
        del layer_inputs[bi]
        if acts.shape[0] > max_calib_tokens:
            perm = torch.randperm(
                acts.shape[0], generator=torch.Generator().manual_seed(cfg.seed)
            )
            acts = acts[perm[:max_calib_tokens]]
        X_layer = acts.float().to(device)

        # ── gate_up_proj: IDRE + VPTQ + BCOS ─────────────────────────
        W_gu_orig = block.gate_up_proj.data.float()

        gu_result = quantize_projection(
            W_gu_orig, X_layer, cfg,
            bcos_layout=block.get_gate_up_bcos_layout(),
        )

        block.gate_up_proj = nn.Parameter(gu_result.W_vq.half().to(device))

        # ── down_proj calibration: router-dispatched intermediates ───
        X_down = block.compute_routed_calibration(X_layer, W_gu_orig)
        if X_down.shape[0] > max_calib_tokens:
            perm = torch.randperm(
                X_down.shape[0],
                generator=torch.Generator().manual_seed(cfg.seed),
            )
            X_down = X_down[perm[:max_calib_tokens]]

        # ── down_proj: IDRE + VPTQ + BCOS ───────────────────────────
        W_d_orig = block.down_proj.data.float()

        d_result = quantize_projection(
            W_d_orig, X_down, cfg,
            bcos_layout=block.get_down_bcos_layout(),
        )

        block.down_proj = nn.Parameter(d_result.W_vq.half().to(device))
        block.set_bcos_params(gu_result.bcos_params, d_result.bcos_params, device)

        # Free memory
        del W_gu_orig, W_d_orig, X_layer, X_down, gu_result, d_result
        gc.collect()
        torch.cuda.empty_cache()

        log_fn(f"    Block {bi:2d}/{num_blocks - 1} ({name}) done in {time.time() - t0:.1f}s")

    log_fn(f"\n  Total quantization: {time.time() - t_start:.1f}s")

    del layer_inputs
    gc.collect()
    torch.cuda.empty_cache()
