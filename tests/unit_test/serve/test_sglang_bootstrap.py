# SPDX-License-Identifier: Apache-2.0
"""SGLang bootstrap helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
    PhaseConfig,
)

from sglang_omni.scheduling import bootstrap


def test_infrastructure_initializes_pool_and_attention_before_consumers(
    monkeypatch,
) -> None:
    from sglang_omni.model_runner import model_worker as model_worker_module
    from sglang_omni.scheduling import sglang_backend

    events: list[str] = []
    req_pool = SimpleNamespace()
    kv_pool = SimpleNamespace()

    class FakeRunner:
        model = SimpleNamespace()

        def alloc_memory_pool(self) -> None:
            events.append("alloc_memory_pool")

        def init_attention_backends(self) -> None:
            assert events[-1] == "alloc_memory_pool"
            events.append("init_attention_backends")

    class FakeWorker:
        def __init__(self, *, config, server_args, gpu_id, tp_rank) -> None:
            del config, server_args, gpu_id, tp_rank
            events.append("model_worker")
            self.model_runner = FakeRunner()
            self.model_config = SimpleNamespace()

        def get_memory_pool(self):
            assert events[-1] == "init_attention_backends"
            events.append("get_memory_pool")
            return req_pool, kv_pool

    def fake_create_tree_cache(*args):
        del args
        events.append("tree_cache")
        return SimpleNamespace()

    class FakePrefillManager:
        def __init__(self, **kwargs) -> None:
            assert kwargs["req_to_token_pool"] is req_pool
            assert kwargs["token_to_kv_pool_allocator"] is kv_pool
            events.append("prefill_manager")

        def add_one_request(self, request) -> None:
            del request

    class FakeDecodeManager:
        def __init__(self, **kwargs) -> None:
            assert kwargs["token_to_kv_pool_allocator"] is kv_pool
            events.append("decode_manager")

    monkeypatch.setattr(model_worker_module, "ModelWorker", FakeWorker)
    monkeypatch.setattr(sglang_backend, "create_tree_cache", fake_create_tree_cache)
    monkeypatch.setattr(sglang_backend, "PrefillManager", FakePrefillManager)
    monkeypatch.setattr(sglang_backend, "DecodeManager", FakeDecodeManager)

    server_args = SimpleNamespace(
        page_size=1,
        disable_overlap_schedule=False,
        chunked_prefill_size=4096,
        max_prefill_tokens=4096,
    )
    infrastructure = bootstrap.create_sglang_infrastructure(server_args, 0)

    assert events == [
        "model_worker",
        "alloc_memory_pool",
        "init_attention_backends",
        "get_memory_pool",
        "tree_cache",
        "prefill_manager",
        "decode_manager",
    ]
    assert infrastructure[0].model_runner.model is not None


def test_defer_cuda_graph_reports_requested_graph_capture(monkeypatch) -> None:
    server_args = SimpleNamespace(
        cuda_graph_config=CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.FULL),
            prefill=PhaseConfig(backend=Backend.DISABLED),
        )
    )
    seen: list[str] = []

    def fake_create_sglang_infrastructure(server_args, gpu_id, **kwargs):
        seen.append(server_args.cuda_graph_config.decode.backend)
        return ("infra", gpu_id, kwargs)

    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure",
        fake_create_sglang_infrastructure,
    )

    want_cuda_graph, infrastructure = (
        bootstrap.create_sglang_infrastructure_defer_cuda_graph(
            server_args,
            3,
            model_arch_override="TestModel",
        )
    )

    assert want_cuda_graph is True
    assert seen == [Backend.FULL]
    assert infrastructure == ("infra", 3, {"model_arch_override": "TestModel"})


def test_defer_cuda_graph_reports_fully_disabled_config(monkeypatch) -> None:
    server_args = SimpleNamespace(
        cuda_graph_config=CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.DISABLED),
            prefill=PhaseConfig(backend=Backend.DISABLED),
        )
    )
    seen: list[str] = []

    def fake_create_sglang_infrastructure(server_args, gpu_id, **kwargs):
        del gpu_id, kwargs
        seen.append(server_args.cuda_graph_config.decode.backend)
        return object()

    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure",
        fake_create_sglang_infrastructure,
    )

    want_cuda_graph, _ = bootstrap.create_sglang_infrastructure_defer_cuda_graph(
        server_args,
        0,
    )

    assert want_cuda_graph is False
    assert seen == [Backend.DISABLED]


def test_cuda_graph_phase_query_is_phase_aware() -> None:
    server_args = SimpleNamespace(
        cuda_graph_config=CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.DISABLED),
            prefill=PhaseConfig(backend=Backend.TC_PIECEWISE),
        )
    )

    assert bootstrap.is_cuda_graph_enabled(server_args, Phase.DECODE) is False
    assert bootstrap.is_cuda_graph_enabled(server_args, Phase.PREFILL) is True
    assert bootstrap.has_enabled_cuda_graph(server_args) is True

    with pytest.raises(ValueError, match="unknown CUDA graph phase"):
        bootstrap.is_cuda_graph_enabled(server_args, "unknown")
