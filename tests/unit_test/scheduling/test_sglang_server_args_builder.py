# SPDX-License-Identifier: Apache-2.0
"""Omni policy for SGLang's phase-aware CUDA graph configuration."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
    PhaseConfig,
)
from sglang.srt.server_args import ServerArgs

from sglang_omni.scheduling.sglang_backend import server_args_builder
from sglang_omni.scheduling.sglang_backend.server_args_builder import OmniServerArgs


def test_omni_architecture_is_constructor_only_state() -> None:
    server_args = OmniServerArgs(
        model_path="dummy",
        omni_model_architecture="TestArchitecture",
    )

    assert server_args._omni_model_architecture == "TestArchitecture"
    assert "omni_model_architecture" not in dataclasses.asdict(server_args)


def test_prefill_capability_lookup_is_explicit_per_architecture() -> None:
    supports_prefill = server_args_builder._supports_sglang_tc_piecewise_prefill

    assert supports_prefill("HiggsMultimodalQwen3ForConditionalGeneration") is True
    assert supports_prefill("MossTranscribeDiarizeForConditionalGeneration") is True
    assert supports_prefill("Qwen3TTSForConditionalGeneration") is False
    assert supports_prefill(None) is False


def _omni_args(model_config: object, *, locked: set[tuple[str, str]] | None = None):
    server_args = object.__new__(OmniServerArgs)
    server_args.cuda_graph_config = CudaGraphConfig(
        decode=PhaseConfig(backend=Backend.FULL),
        prefill=PhaseConfig(backend=Backend.BREAKABLE),
    )
    server_args._cuda_graph_config_locked = locked or set()
    server_args._omni_model_architecture = "TestArchitecture"
    server_args.get_model_config = lambda: model_config
    return server_args


def test_supported_omni_model_selects_tc_piecewise_and_keeps_upstream_checks(
    monkeypatch,
) -> None:
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["TestArchitecture"]),
        is_multimodal=True,
        is_multimodal_piecewise_cuda_graph_supported=False,
        is_piecewise_cuda_graph_disabled_model=True,
    )
    server_args = _omni_args(model_config)
    observed: list[tuple[str, bool, bool]] = []

    monkeypatch.setattr(
        server_args_builder,
        "_supports_sglang_tc_piecewise_prefill",
        lambda _architecture: True,
    )
    monkeypatch.setattr(
        ServerArgs,
        "_apply_cuda_graph_compatibility",
        lambda self: observed.append(
            (
                self.cuda_graph_config.prefill.backend,
                self.get_model_config().is_multimodal_piecewise_cuda_graph_supported,
                self.get_model_config().is_piecewise_cuda_graph_disabled_model,
            )
        ),
    )

    server_args._apply_cuda_graph_compatibility()

    assert observed == [(Backend.TC_PIECEWISE, True, False)]
    assert server_args.cuda_graph_config.prefill.backend == Backend.TC_PIECEWISE
    assert model_config.is_multimodal_piecewise_cuda_graph_supported is False
    assert model_config.is_piecewise_cuda_graph_disabled_model is True


def test_unsupported_omni_model_keeps_prefill_graph_disabled(monkeypatch) -> None:
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["TestArchitecture"]),
        is_multimodal=True,
        is_multimodal_piecewise_cuda_graph_supported=False,
        is_piecewise_cuda_graph_disabled_model=False,
    )
    server_args = _omni_args(model_config)
    observed: list[str] = []
    monkeypatch.setattr(
        server_args_builder,
        "_supports_sglang_tc_piecewise_prefill",
        lambda _architecture: False,
    )
    monkeypatch.setattr(
        ServerArgs,
        "_apply_cuda_graph_compatibility",
        lambda self: observed.append(self.cuda_graph_config.prefill.backend),
    )

    server_args._apply_cuda_graph_compatibility()

    assert server_args.cuda_graph_config.prefill.backend == Backend.DISABLED
    assert observed == [Backend.DISABLED]


def test_supported_capability_does_not_override_upstream_architecture_blacklist(
    monkeypatch,
) -> None:
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]),
        is_multimodal=True,
        is_multimodal_piecewise_cuda_graph_supported=False,
        is_piecewise_cuda_graph_disabled_model=True,
    )
    server_args = _omni_args(model_config)
    server_args._omni_model_architecture = "DeepseekV4ForCausalLM"
    observed: list[bool] = []
    monkeypatch.setattr(
        server_args_builder,
        "_supports_sglang_tc_piecewise_prefill",
        lambda _architecture: True,
    )
    monkeypatch.setattr(
        ServerArgs,
        "_apply_cuda_graph_compatibility",
        lambda self: observed.append(
            self.get_model_config().is_piecewise_cuda_graph_disabled_model
        ),
    )

    server_args._apply_cuda_graph_compatibility()

    assert observed == [True]
    assert model_config.is_piecewise_cuda_graph_disabled_model is True


def test_explicit_prefill_backend_retains_sglang_precedence(monkeypatch) -> None:
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["TestArchitecture"]),
        is_multimodal=True,
        is_multimodal_piecewise_cuda_graph_supported=False,
        is_piecewise_cuda_graph_disabled_model=False,
    )
    server_args = _omni_args(
        model_config,
        locked={(Phase.PREFILL, "backend")},
    )
    observed: list[str] = []
    monkeypatch.setattr(
        ServerArgs,
        "_apply_cuda_graph_compatibility",
        lambda self: observed.append(self.cuda_graph_config.prefill.backend),
    )

    server_args._apply_cuda_graph_compatibility()

    assert observed == [Backend.BREAKABLE]
    assert server_args.cuda_graph_config.prefill.backend == Backend.BREAKABLE


def test_supported_model_requires_multimodal_static_input_buffer(monkeypatch) -> None:
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["TestArchitecture"]),
        is_multimodal=False,
        is_multimodal_piecewise_cuda_graph_supported=False,
        is_piecewise_cuda_graph_disabled_model=False,
    )
    server_args = _omni_args(model_config)
    monkeypatch.setattr(
        server_args_builder,
        "_supports_sglang_tc_piecewise_prefill",
        lambda _architecture: True,
    )

    with pytest.raises(ValueError, match="multimodal classification"):
        server_args._apply_cuda_graph_compatibility()
