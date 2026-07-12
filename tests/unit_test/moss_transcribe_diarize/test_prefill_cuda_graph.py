# SPDX-License-Identifier: Apache-2.0
"""MOSS-TD native prefill CUDA graph model contract."""

from __future__ import annotations

import inspect

import pytest
import torch

from sglang_omni.models.model_capabilities import get_model_capabilities
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
