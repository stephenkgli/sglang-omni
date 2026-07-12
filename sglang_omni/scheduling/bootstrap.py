# SPDX-License-Identifier: Apache-2.0
"""Generic SGLang bootstrap utilities for model-specific schedulers."""

from __future__ import annotations

from typing import Any

from sglang.srt.model_executor.cuda_graph_config import Backend, Phase


def create_sglang_infrastructure(
    server_args: Any,
    gpu_id: int,
    *,
    tp_rank: int = 0,
    nccl_port: int | None = None,
    model_arch_override: str | None = None,
    weight_prefix: str | None = None,
    capture_hidden_layers: list[int] | None = None,
    total_gpu_memory_fraction: float | None = None,
):
    """Create SGLang worker, memory pools, tree cache, and prefill/decode managers."""
    from sglang_omni.model_runner.model_worker import ModelWorker, ModelWorkerConfig
    from sglang_omni.scheduling.sglang_backend import (
        DecodeManager,
        PrefillManager,
        create_tree_cache,
    )

    model_worker = ModelWorker(
        config=ModelWorkerConfig(
            model_arch_override=model_arch_override,
            weight_prefix=weight_prefix,
            nccl_port=nccl_port,
            total_gpu_memory_fraction=total_gpu_memory_fraction,
        ),
        server_args=server_args,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
    )

    if capture_hidden_layers:
        from sglang_omni.model_runner._hidden_capture import (
            install_hidden_capture_hooks,
        )

        model = model_worker.model_runner.model
        install_hidden_capture_hooks(model, capture_hidden_layers)

    # note (kaige): SGLang 0.5.15 separates model loading, pool/backend init,
    # and graph capture. Stages finish model-specific setup before the latter.
    model_worker.model_runner.alloc_memory_pool()
    model_worker.model_runner.init_attention_backends()

    req_to_token_pool, token_to_kv_pool_allocator = model_worker.get_memory_pool()

    tree_cache = create_tree_cache(
        server_args,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        server_args.page_size,
    )

    enable_overlap = not server_args.disable_overlap_schedule

    prefill_mgr = PrefillManager(
        page_size=server_args.page_size,
        chunked_prefill_size=server_args.chunked_prefill_size,
        max_prefill_tokens=server_args.max_prefill_tokens,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=model_worker.model_config,
        enable_overlap=enable_overlap,
    )

    decode_mgr = DecodeManager(
        server_args=server_args,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        on_retract=lambda req: prefill_mgr.add_one_request(req),
    )

    return (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_worker.model_config,
    )


def is_cuda_graph_enabled(server_args: Any, phase: str) -> bool:
    """Return whether one SGLang forward phase uses a CUDA graph backend."""
    if phase not in Phase.ALL:
        raise ValueError(f"unknown CUDA graph phase: {phase}")
    return server_args.cuda_graph_config[phase].backend != Backend.DISABLED


def has_enabled_cuda_graph(server_args: Any) -> bool:
    """Return whether either SGLang forward phase uses a CUDA graph backend."""
    return any(is_cuda_graph_enabled(server_args, phase) for phase in Phase.ALL)


def create_sglang_infrastructure_defer_cuda_graph(
    server_args: Any,
    gpu_id: int,
    **kwargs: Any,
):
    """Build pools/backends while leaving graph capture to the stage caller."""
    infrastructure = create_sglang_infrastructure(server_args, gpu_id, **kwargs)
    return has_enabled_cuda_graph(server_args), infrastructure
