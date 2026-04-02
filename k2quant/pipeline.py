from __future__ import annotations

import gc
import time
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import QuantConfig
from .dense_block import QuantizableMLP
from .moe_block import QuantizableExperts
from .projection import quantize_projection


def _collect_activations(
    model: nn.Module,
    moe_blocks: list[nn.Module],
    calib_data: torch.Tensor,
    device: str,
    batch_size: int,
    log_fn: Callable[[str], None],
) -> dict[int, list[torch.Tensor]]:
    """
    Collect MoE block input activations through unmodified HF forward.
    """
    block_inputs: dict[int, list[torch.Tensor]] = {}
    hooks = []

    for li, moe_block in enumerate(moe_blocks):
        storage: list[torch.Tensor] = []
        block_inputs[li] = storage

        def make_hook(s):
            def fn(_module, args, _kwargs, output):
                s.append(args[0].detach().cpu())
                return output
            return fn

        h = moe_block.register_forward_hook(make_hook(storage), with_kwargs=True)
        hooks.append(h)

    n_batches = len(calib_data) // batch_size
    with torch.no_grad():
        for i in range(0, len(calib_data), batch_size):
            batch = calib_data[i : i + batch_size].to(device)
            model(batch)
            batch_idx = i // batch_size

            log_fn(f"    Batch {batch_idx + 1}/{n_batches}")

    for h in hooks:
        h.remove()

    return block_inputs


def quantize_model(
    model: nn.Module,
    calib_data: torch.Tensor,
    cfg: QuantConfig,
    experts_cls: type[QuantizableExperts],
    get_moe_block: Callable[[nn.Module, int], nn.Module],
    num_layers: int,
    *,
    device: str = "cuda",
    max_calib_tokens: int = 4096,
    batch_size: int = 2,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict[str, torch.Tensor]:
    """
    Quantize all MoE expert layers of a model in-place.

    Args:
        model: The HuggingFace model to quantize (modified in-place).
        calib_data: Calibration token IDs. (nsamples, seqlen). int64.
        cfg: Quantization configuration.
        experts_cls: QuantizableExperts subclass with from_hf_module().
        get_moe_block: Callable that returns the MoE block module at each
            layer index. E.g. lambda m, i: m.model.layers[i].mlp
        num_layers: Number of MoE layers to process.
        device: Device for computation.
        max_calib_tokens: Maximum calibration tokens per block.
        batch_size: Batch size for calibration forward passes.
        log_fn: Optional progress logger.

    Returns:
        A flat dict mapping tensor names to CPU tensors for all quantized
        expert projections, suitable for safetensors serialization.
    """
    if log_fn is None:
        log_fn = print

    # Step 1: Collect calibration activations through unmodified HF forward
    log_fn("  Collecting calibration activations...")
    moe_blocks = [get_moe_block(model, i) for i in range(num_layers)]
    block_inputs = _collect_activations(
        model, moe_blocks, calib_data, device, batch_size, log_fn
    )
    log_fn("  Activations collected.")

    # Step 2: Swap experts and quantize layer by layer
    log_fn("  Quantizing layers...")
    t_start = time.time()
    all_compressed: dict[str, torch.Tensor] = {}

    for li in range(num_layers):
        t0 = time.time()
        moe_block = moe_blocks[li]

        # Create our experts module from the HF MoE block
        hf_params = next(moe_block.experts.parameters())
        qe = experts_cls.from_hf_module(
            moe_block, device=hf_params.device, dtype=hf_params.dtype
        )
        hidden_size = qe.hidden_size

        acts = torch.cat(block_inputs[li], dim=0).reshape(-1, hidden_size)
        del block_inputs[li]
        if acts.shape[0] > max_calib_tokens:
            perm = torch.randperm(
                acts.shape[0], generator=torch.Generator().manual_seed(cfg.seed)
            )
            acts = acts[perm[:max_calib_tokens]]
        X_layer = acts.float().to(device)

        # -- gate_up_proj: IDRE + VPTQ + BCOS --
        W_gu_orig = qe.gate_up_proj.data.float()

        gu_result = quantize_projection(
            W_gu_orig, X_layer, cfg,
            bcos_layout=qe.get_gate_up_bcos_layout(),
        )

        qe.gate_up_proj = nn.Parameter(gu_result.W_vq.half().to(device))

        # -- down_proj calibration: router-dispatched intermediates --
        router_weight, top_k = experts_cls.get_routing_info(moe_block)

        with torch.no_grad():
            router_logits = F.linear(X_layer.half(), router_weight)
            routing_weights = F.softmax(router_logits, dim=-1)
            _, top_k_indices = torch.topk(routing_weights, k=top_k, dim=-1)

            all_intermediates = []
            for ei in range(qe.num_experts):
                mask = (top_k_indices == ei).any(dim=-1)
                if mask.sum() == 0:
                    continue
                tokens = X_layer[mask].half()
                gu_out = F.linear(tokens, W_gu_orig[ei].half().to(device))
                gate_e, up_e = gu_out.chunk(2, dim=-1)
                intermediate = qe.act_fn(gate_e) * up_e
                all_intermediates.append(intermediate)

            X_down = torch.cat(all_intermediates, dim=0).float()
            if X_down.shape[0] > max_calib_tokens:
                perm = torch.randperm(
                    X_down.shape[0],
                    generator=torch.Generator().manual_seed(cfg.seed),
                )
                X_down = X_down[perm[:max_calib_tokens]]

        # -- down_proj: IDRE + VPTQ + BCOS --
        W_d_orig = qe.down_proj.data.float()

        d_result = quantize_projection(
            W_d_orig, X_down, cfg,
            bcos_layout=qe.get_down_bcos_layout(),
        )

        qe.down_proj = nn.Parameter(d_result.W_vq.half().to(device))
        qe.set_bcos_params(gu_result.bcos_params, d_result.bcos_params, device)

        # Collect compressed tensors (CPU) before freeing results
        all_compressed.update(gu_result.wq.to_tensors(f"layers.{li}.gate_up"))
        all_compressed.update(d_result.wq.to_tensors(f"layers.{li}.down"))

        # Replace only the experts sub-module on the HF MoE block
        moe_block.experts = qe

        # Free memory
        del W_gu_orig, W_d_orig, X_layer, X_down, gu_result, d_result
        gc.collect()
        torch.cuda.empty_cache()

        log_fn(f"    Layer {li:2d}/{num_layers - 1} done in {time.time() - t0:.1f}s")

    log_fn(f"\n  Total quantization: {time.time() - t_start:.1f}s")

    del block_inputs
    gc.collect()
    torch.cuda.empty_cache()

    return all_compressed


def _collect_mlp_activations(
    model: nn.Module,
    mlp_blocks: list[nn.Module],
    calib_data: torch.Tensor,
    device: str,
    batch_size: int,
    log_fn: Callable[[str], None],
) -> dict[int, list[torch.Tensor]]:
    """Collect MLP input activations through unmodified HF forward."""
    block_inputs: dict[int, list[torch.Tensor]] = {}
    hooks = []

    for li, mlp_block in enumerate(mlp_blocks):
        storage: list[torch.Tensor] = []
        block_inputs[li] = storage

        def make_hook(s):
            def fn(_module, args, _kwargs, output):
                s.append(args[0].detach().cpu())
                return output
            return fn

        h = mlp_block.register_forward_hook(make_hook(storage), with_kwargs=True)
        hooks.append(h)

    n_batches = len(calib_data) // batch_size
    with torch.no_grad():
        for i in range(0, len(calib_data), batch_size):
            batch = calib_data[i : i + batch_size].to(device)
            model(batch)
            batch_idx = i // batch_size
            log_fn(f"    Batch {batch_idx + 1}/{n_batches}")

    for h in hooks:
        h.remove()

    return block_inputs


def quantize_dense_model(
    model: nn.Module,
    calib_data: torch.Tensor,
    cfg: QuantConfig,
    mlp_cls: type[QuantizableMLP],
    get_mlp_block: Callable[[nn.Module, int], nn.Module],
    set_mlp_block: Callable[[nn.Module, int, nn.Module], None],
    num_layers: int,
    *,
    device: str = "cuda",
    max_calib_tokens: int = 4096,
    batch_size: int = 2,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict[str, torch.Tensor]:
    """
    Quantize all dense MLP layers of a model in-place.

    Like quantize_model() but for non-MoE architectures (e.g. LLaMA).
    No expert routing is needed — each layer has a single MLP.

    Args:
        model: The HuggingFace model to quantize (modified in-place).
        calib_data: Calibration token IDs. (nsamples, seqlen). int64.
        cfg: Quantization configuration.
        mlp_cls: QuantizableMLP subclass with from_hf_module().
        get_mlp_block: Callable that returns the MLP module at each
            layer index. E.g. lambda m, i: m.model.layers[i].mlp
        set_mlp_block: Callable that replaces the MLP module at each
            layer index. E.g. lambda m, i, mlp: setattr(m.model.layers[i], 'mlp', mlp)
        num_layers: Number of layers to process.
        device: Device for computation.
        max_calib_tokens: Maximum calibration tokens per block.
        batch_size: Batch size for calibration forward passes.
        log_fn: Optional progress logger.

    Returns:
        A flat dict mapping tensor names to CPU tensors for all quantized
        projections, suitable for safetensors serialization.
    """
    if log_fn is None:
        log_fn = print

    # Step 1: Collect calibration activations through unmodified HF forward
    log_fn("  Collecting calibration activations...")
    mlp_blocks = [get_mlp_block(model, i) for i in range(num_layers)]
    block_inputs = _collect_mlp_activations(
        model, mlp_blocks, calib_data, device, batch_size, log_fn
    )
    log_fn("  Activations collected.")

    # Step 2: Quantize layer by layer
    log_fn("  Quantizing layers...")
    t_start = time.time()
    all_compressed: dict[str, torch.Tensor] = {}

    for li in range(num_layers):
        t0 = time.time()
        mlp_block = mlp_blocks[li]

        # Create our MLP module from the HF block
        hf_params = next(mlp_block.parameters())
        qm = mlp_cls.from_hf_module(
            mlp_block, device=hf_params.device, dtype=hf_params.dtype
        )
        hidden_size = qm.hidden_size

        acts = torch.cat(block_inputs[li], dim=0).reshape(-1, hidden_size)
        del block_inputs[li]
        if acts.shape[0] > max_calib_tokens:
            perm = torch.randperm(
                acts.shape[0], generator=torch.Generator().manual_seed(cfg.seed)
            )
            acts = acts[perm[:max_calib_tokens]]
        X_layer = acts.float().to(device)

        # -- gate_up_proj: IDRE + VPTQ + BCOS --
        W_gu_orig = qm.gate_up_proj.data.float()

        gu_result = quantize_projection(
            W_gu_orig, X_layer, cfg,
            bcos_layout=qm.get_gate_up_bcos_layout(),
        )

        qm.gate_up_proj = nn.Parameter(gu_result.W_vq.half().to(device))

        # -- down_proj calibration: forward all tokens (no routing) --
        with torch.no_grad():
            gu_out = F.linear(X_layer.half(), W_gu_orig[0].half().to(device))
            gate_e, up_e = gu_out.chunk(2, dim=-1)
            X_down = (qm.act_fn(gate_e) * up_e).float()
            if X_down.shape[0] > max_calib_tokens:
                perm = torch.randperm(
                    X_down.shape[0],
                    generator=torch.Generator().manual_seed(cfg.seed),
                )
                X_down = X_down[perm[:max_calib_tokens]]

        # -- down_proj: IDRE + VPTQ + BCOS --
        W_d_orig = qm.down_proj.data.float()

        d_result = quantize_projection(
            W_d_orig, X_down, cfg,
            bcos_layout=qm.get_down_bcos_layout(),
        )

        qm.down_proj = nn.Parameter(d_result.W_vq.half().to(device))
        qm.set_bcos_params(gu_result.bcos_params, d_result.bcos_params, device)

        # Collect compressed tensors (CPU) before freeing results
        all_compressed.update(gu_result.wq.to_tensors(f"layers.{li}.gate_up"))
        all_compressed.update(d_result.wq.to_tensors(f"layers.{li}.down"))

        # Replace the MLP module on the HF layer
        set_mlp_block(model, li, qm)

        # Free memory
        del W_gu_orig, W_d_orig, X_layer, X_down, gu_result, d_result
        gc.collect()
        torch.cuda.empty_cache()

        log_fn(f"    Layer {li:2d}/{num_layers - 1} done in {time.time() - t0:.1f}s")

    log_fn(f"\n  Total quantization: {time.time() - t_start:.1f}s")

    del block_inputs
    gc.collect()
    torch.cuda.empty_cache()

    return all_compressed
