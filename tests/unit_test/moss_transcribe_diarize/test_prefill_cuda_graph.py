# SPDX-License-Identifier: Apache-2.0
"""MOSS-TD native prefill CUDA graph model contract."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.cuda_graph_config import Backend

from sglang_omni.models.model_capabilities import get_model_capabilities
from sglang_omni.models.moss_transcribe_diarize import sglang_model, stages
from sglang_omni.models.moss_transcribe_diarize.sglang_model import (
    MossTranscribeDiarizeForConditionalGeneration as MossModel,
)
from sglang_omni.models.moss_transcribe_diarize.stages import (
    create_sglang_moss_transcribe_diarize_executor,
)


def test_moss_td_prefill_cuda_graph_is_enabled_by_default() -> None:
    signature = inspect.signature(create_sglang_moss_transcribe_diarize_executor)

    assert signature.parameters["enable_prefill_cuda_graph"].default is True
    assert signature.parameters["prefill_graph_token_buckets"].default is None
    capabilities = get_model_capabilities(
        "MossTranscribeDiarizeForConditionalGeneration"
    )
    assert capabilities is not None
    assert capabilities.supports_sglang_tc_piecewise_prefill is True


@pytest.mark.parametrize(
    ("enable_prefill_cuda_graph", "expected_backend", "expected_request_slots"),
    [(True, Backend.FULL, 16), (False, None, None)],
)
def test_moss_td_resolves_full_prefill_backend_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    enable_prefill_cuda_graph: bool,
    expected_backend: str | None,
    expected_request_slots: int | None,
) -> None:
    captured: dict[str, object] = {}

    class StopAfterServerArgs(Exception):
        pass

    monkeypatch.setattr(
        stages.AutoProcessor,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(tokenizer=object()),
    )

    def capture_server_args(
        model_path: str,
        *,
        context_length: int,
        **overrides: object,
    ) -> SimpleNamespace:
        captured.update(overrides)
        backend = overrides.get("cuda_graph_backend_prefill", Backend.DISABLED)
        return SimpleNamespace(
            max_running_requests=overrides["max_running_requests"],
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(
                    backend=backend,
                    full_prefill_max_req=None,
                )
            ),
        )

    def capture_validated_server_args(
        *,
        model_name: str,
        server_args: SimpleNamespace,
    ) -> None:
        captured["full_prefill_max_req"] = (
            server_args.cuda_graph_config.prefill.full_prefill_max_req
        )
        raise StopAfterServerArgs

    monkeypatch.setattr(stages, "build_sglang_server_args", capture_server_args)
    monkeypatch.setattr(
        stages,
        "validate_generation_batch_policy",
        capture_validated_server_args,
    )

    with pytest.raises(StopAfterServerArgs):
        create_sglang_moss_transcribe_diarize_executor(
            "fake-model",
            context_length=128,
            max_new_tokens=8,
            enable_prefill_cuda_graph=enable_prefill_cuda_graph,
        )

    assert captured.get("cuda_graph_backend_prefill") == expected_backend
    assert captured.get("disable_prefill_cuda_graph") is (
        None if enable_prefill_cuda_graph else True
    )
    assert captured["full_prefill_max_req"] == expected_request_slots


def test_moss_td_preserves_explicit_full_prefill_request_slots() -> None:
    server_args = SimpleNamespace(
        max_running_requests=16,
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(
                backend=Backend.FULL,
                full_prefill_max_req=4,
            )
        ),
    )

    stages._default_full_prefill_request_slots(server_args)

    assert server_args.cuda_graph_config.prefill.full_prefill_max_req == 4


def test_moss_td_full_prefill_embeds_the_padded_token_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MossModel.__new__(MossModel)
    torch.nn.Module.__init__(model)
    model.language_model = torch.nn.Identity()

    raw_input_ids = torch.tensor([1, 2, 3])
    static_input_ids = torch.tensor([1, 2, 3, 0])
    raw_forward_batch = SimpleNamespace()
    context = SimpleNamespace(
        forward_batch=SimpleNamespace(input_ids=static_input_ids),
        num_tokens=4,
        raw_num_tokens=3,
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        sglang_model,
        "get_tc_piecewise_forward_context",
        lambda: context,
    )

    def capture_mm_routine(**kwargs: object) -> torch.Tensor:
        captured.update(kwargs)
        return torch.empty(0)

    monkeypatch.setattr(
        sglang_model,
        "general_mm_embed_routine",
        capture_mm_routine,
    )

    model.forward(
        raw_input_ids,
        torch.arange(raw_input_ids.shape[0]),
        raw_forward_batch,
    )

    assert captured["input_ids"] is static_input_ids
    assert captured["forward_batch"] is raw_forward_batch


def test_moss_td_language_model_alias_is_not_registered_twice() -> None:
    model = MossModel.__new__(MossModel)
    torch.nn.Module.__init__(model)
    language_model = torch.nn.Linear(4, 4, bias=False)
    model.language_model = language_model

    assert model.model is language_model
    assert "model" not in model._modules
    assert set(model.state_dict()) == {"language_model.weight"}

    model.model = language_model

    assert model.model is language_model
    assert "model" not in model._modules

    with pytest.raises(AssertionError, match="may only alias"):
        model.model = torch.nn.Linear(4, 4, bias=False)
