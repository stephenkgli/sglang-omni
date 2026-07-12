# SPDX-License-Identifier: Apache-2.0
"""Higgs TTS piecewise prefill CUDA-graph contract tests."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("sglang")

from sglang.srt.managers.schedule_batch import MultimodalInputs  # noqa: E402

from sglang_omni.models.higgs_tts import model as higgs_model_module  # noqa: E402
from sglang_omni.models.higgs_tts.model import (  # noqa: E402
    HiggsPrefillEmbeddingInputs,
    HiggsTTSModel,
)
from sglang_omni.models.higgs_tts.model_runner import HiggsTTSModelRunner  # noqa: E402


class _FakeQwenModel(nn.Module):
    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(32, hidden_size)
        self.layers = nn.ModuleList()
        with torch.no_grad():
            values = torch.arange(32 * hidden_size, dtype=torch.float32)
            self.embed_tokens.weight.copy_(values.reshape(32, hidden_size))

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        del input_ids, forward_batch
        assert input_embeds is not None
        return input_embeds + positions.to(input_embeds.dtype).unsqueeze(-1)


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeQwenModel()
        self.config = SimpleNamespace(vocab_size=11)


class _FakeFusedEmbedding(nn.Module):
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        values = codes.to(torch.float32).sum(dim=-1, keepdim=True)
        return values.repeat(1, 4)


def _make_higgs_model() -> HiggsTTSModel:
    model = HiggsTTSModel.__new__(HiggsTTSModel)
    nn.Module.__init__(model)
    model.backbone = _FakeBackbone()
    return model


def _request(
    request_id: str,
    *,
    extend_input_len: int,
    inflight_middle_chunks: int = 0,
    is_finished: bool = False,
    reference_codes: list[list[int]] | None = None,
    origin_input_ids: list[int] | None = None,
):
    req = SimpleNamespace(
        extend_range=SimpleNamespace(length=extend_input_len),
        inflight_middle_chunks=inflight_middle_chunks,
        origin_input_ids=(
            list(origin_input_ids)
            if origin_input_ids is not None
            else [1] * extend_input_len
        ),
        sampling_params=SimpleNamespace(sampling_seed=17),
        finished=lambda: is_finished,
    )
    data = SimpleNamespace(
        req=req,
        reference_codes_delayed=reference_codes,
    )
    return SimpleNamespace(request_id=request_id, data=data)


def test_piecewise_cuda_graph_factory_defaults() -> None:
    from sglang_omni.models.higgs_tts.stages import create_sglang_tts_engine_executor

    signature = inspect.signature(create_sglang_tts_engine_executor)

    assert signature.parameters["enable_prefill_cuda_graph"].default is True
    assert signature.parameters["prefill_graph_token_buckets"].default is None


def test_piecewise_cuda_graph_model_alias_is_not_registered_twice() -> None:
    model = _make_higgs_model()
    qwen_model = model.backbone.model

    assert model.model is qwen_model
    assert "model" not in model._modules
    assert set(model.state_dict()) == {"backbone.model.embed_tokens.weight"}

    model.model = qwen_model

    assert model.model is qwen_model
    assert "model" not in model._modules
    assert set(model.state_dict()) == {"backbone.model.embed_tokens.weight"}

    with pytest.raises(AssertionError, match="may only alias"):
        model.model = nn.Linear(4, 4, bias=False)


def test_before_prefill_keeps_input_embeds_unset_and_packages_reference_audio() -> None:
    runner = object.__new__(HiggsTTSModelRunner)
    seed_calls: list[tuple[str, int | None]] = []
    runner.model = SimpleNamespace(
        multimodal_embedding=SimpleNamespace(
            modality_embedding_0=_FakeFusedEmbedding()
        ),
        set_request_seed=lambda request_id, seed: seed_calls.append((request_id, seed)),
    )
    request = _request(
        "ref",
        extend_input_len=5,
        reference_codes=[[1, 2], [3, 4]],
        origin_input_ids=[5, -100, -100, 6, 7],
    )
    forward_batch = SimpleNamespace(
        batch_size=1,
        input_ids=torch.tensor([5, -100, -100, 6, 7]),
        input_embeds=None,
        mm_inputs=[None],
        extend_seq_lens_cpu=[5],
        extend_prefix_lens_cpu=[0],
    )

    runner.before_prefill(forward_batch, None, [request])

    assert forward_batch.input_embeds is None
    assert seed_calls == [("ref", 17)]
    assert len(forward_batch.mm_inputs) == 1
    overrides = forward_batch.mm_inputs[0]
    assert isinstance(overrides, HiggsPrefillEmbeddingInputs)
    assert overrides.positions.tolist() == [1, 2]
    assert torch.equal(
        overrides.embeddings,
        torch.tensor([[3.0] * 4, [7.0] * 4]),
    )


def test_before_prefill_keeps_mm_inputs_batch_aligned_for_mixed_requests() -> None:
    runner = object.__new__(HiggsTTSModelRunner)
    runner.model = SimpleNamespace(
        multimodal_embedding=SimpleNamespace(
            modality_embedding_0=_FakeFusedEmbedding()
        ),
        set_request_seed=lambda _request_id, _seed: None,
    )
    requests = [
        _request("zero-shot", extend_input_len=2, origin_input_ids=[5, 6]),
        _request(
            "ref",
            extend_input_len=3,
            reference_codes=[[1, 2], [3, 4]],
            origin_input_ids=[7, -100, -100],
        ),
    ]
    forward_batch = SimpleNamespace(
        batch_size=2,
        input_ids=torch.tensor([5, 6, 7, -100, -100]),
        input_embeds=None,
        mm_inputs=[None, None],
        extend_seq_lens_cpu=[2, 3],
        extend_prefix_lens_cpu=[0, 0],
    )

    runner.before_prefill(forward_batch, None, requests)

    assert len(forward_batch.mm_inputs) == 2
    assert forward_batch.mm_inputs[0] is None
    overrides = forward_batch.mm_inputs[1]
    assert isinstance(overrides, HiggsPrefillEmbeddingInputs)
    assert isinstance(overrides, MultimodalInputs)
    assert overrides.positions.tolist() == [3, 4]
    assert torch.equal(
        overrides.embeddings,
        torch.tensor([[3.0] * 4, [7.0] * 4]),
    )


def test_before_prefill_uses_eager_embeddings_for_branched_radix_prefix() -> None:
    runner = object.__new__(HiggsTTSModelRunner)
    model = _make_higgs_model()
    model.multimodal_embedding = SimpleNamespace(
        modality_embedding_0=_FakeFusedEmbedding()
    )
    model.set_request_seed = lambda _request_id, _seed: None
    runner.model = model
    request = _request(
        "branched",
        extend_input_len=2,
        origin_input_ids=[5, 6, 7],
    )
    forward_batch = SimpleNamespace(
        batch_size=1,
        input_ids=torch.tensor([6, 7]),
        input_embeds=None,
        mm_inputs=[None],
        extend_seq_lens_cpu=[2],
        extend_prefix_lens_cpu=[1],
    )

    runner.before_prefill(forward_batch, None, [request])

    assert forward_batch.mm_inputs is None
    assert torch.equal(
        forward_batch.input_embeds,
        model.backbone.model.embed_tokens(torch.tensor([6, 7])),
    )


def test_before_prefill_keeps_exact_prompt_cache_hit_graph_eligible() -> None:
    runner = object.__new__(HiggsTTSModelRunner)
    runner.model = SimpleNamespace(
        multimodal_embedding=SimpleNamespace(
            modality_embedding_0=_FakeFusedEmbedding()
        ),
        set_request_seed=lambda _request_id, _seed: None,
    )
    request = _request(
        "exact",
        extend_input_len=1,
        origin_input_ids=[5, 6, 7],
    )
    forward_batch = SimpleNamespace(
        batch_size=1,
        input_ids=torch.tensor([7]),
        input_embeds=None,
        mm_inputs=[None],
        extend_seq_lens_cpu=[1],
        extend_prefix_lens_cpu=[2],
    )

    runner.before_prefill(forward_batch, None, [request])

    assert forward_batch.input_embeds is None
    assert forward_batch.mm_inputs == [None]


def test_reference_embeddings_use_absolute_prefix_offset_after_radix_hit() -> None:
    runner = object.__new__(HiggsTTSModelRunner)
    runner.model = SimpleNamespace(
        multimodal_embedding=SimpleNamespace(modality_embedding_0=_FakeFusedEmbedding())
    )
    request = _request(
        "cached-ref",
        extend_input_len=3,
        reference_codes=[[1, 10], [2, 20], [3, 30], [4, 40]],
        origin_input_ids=[8, -100, -100, -100, -100, 9],
    )
    forward_batch = SimpleNamespace(
        input_ids=torch.tensor([-100, -100, 9]),
        extend_seq_lens_cpu=[3],
        extend_prefix_lens_cpu=[3],
    )

    overrides = runner._build_prefill_embedding_overrides(forward_batch, [request])

    assert len(overrides) == 1
    assert isinstance(overrides[0], HiggsPrefillEmbeddingInputs)
    assert overrides[0].positions.tolist() == [0, 1]
    assert torch.equal(
        overrides[0].embeddings,
        torch.tensor([[33.0] * 4, [44.0] * 4]),
    )


def test_prefill_embeddings_use_sglang_stable_address_during_piecewise_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _make_higgs_model()
    overrides = HiggsPrefillEmbeddingInputs(
        positions=torch.tensor([1, 3]),
        embeddings=torch.tensor([[100.0] * 4, [200.0] * 4]),
    )
    input_ids = torch.tensor([2, -100, 3, -100])
    stable_input_embeds = torch.empty((4, 4))
    forward_batch = SimpleNamespace(
        batch_size=1,
        input_embeds=stable_input_embeds,
        mm_inputs=[overrides],
    )

    monkeypatch.setattr(
        higgs_model_module,
        "is_in_tc_piecewise_cuda_graph",
        lambda: False,
    )
    eager = model._build_prefill_input_embeds(input_ids, forward_batch)
    expected = model.backbone.model.embed_tokens(torch.tensor([2, 0, 3, 0]))
    expected[1] = 100
    expected[3] = 200
    assert torch.equal(eager, expected)
    assert input_ids.tolist() == [2, -100, 3, -100]

    monkeypatch.setattr(
        higgs_model_module,
        "is_in_tc_piecewise_cuda_graph",
        lambda: True,
    )
    first = model._build_prefill_input_embeds(input_ids, forward_batch)
    first_address = first.data_ptr()
    second = model._build_prefill_input_embeds(input_ids, forward_batch)

    assert input_ids.tolist() == [2, 0, 3, 0]
    assert first_address == stable_input_embeds.data_ptr()
    assert second.data_ptr() == first_address
    assert torch.equal(second, expected)


def test_prefill_forward_returns_hidden_states_without_sampling() -> None:
    model = _make_higgs_model()

    def fail_if_sampled(*args, **kwargs):
        del args, kwargs
        raise AssertionError("prefill sampler must run outside model.forward")

    model.decode_codebooks_batch = fail_if_sampled
    forward_mode = SimpleNamespace(
        is_decode=lambda: False,
        is_extend=lambda: True,
    )
    forward_batch = SimpleNamespace(
        forward_mode=forward_mode,
        extend_seq_lens=torch.tensor([2, 1]),
        mm_inputs=None,
    )
    input_ids = torch.tensor([1, 2, 3])
    positions = torch.tensor([0, 1, 0])

    output = model.forward(input_ids, positions, forward_batch)

    assert output.hidden_states.shape == (2, 4)
    assert output.next_token_logits.shape == (2, 11)
    assert torch.count_nonzero(output.next_token_logits) == 0


def test_prefill_sampler_runs_only_for_final_chunks() -> None:
    runner = object.__new__(HiggsTTSModelRunner)
    sample_calls: list[tuple[torch.Tensor, list[str], list[str]]] = []

    def decode_codebooks_batch(hidden_states, request_ids, gen_params):
        sample_calls.append((hidden_states.clone(), request_ids, gen_params))

    runner.model = SimpleNamespace(
        _gen_params_for_batch=lambda sampling_info, batch_size: [
            f"param-{index}" for index in range(batch_size)
        ],
        decode_codebooks_batch=decode_codebooks_batch,
    )
    requests = [
        _request("middle", extend_input_len=1, inflight_middle_chunks=1),
        _request("final", extend_input_len=1),
        _request("finished", extend_input_len=1, is_finished=True),
    ]
    hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = SimpleNamespace(logits_output=SimpleNamespace(hidden_states=hidden_states))

    runner._sample_prefill_codebooks(
        result,
        SimpleNamespace(sampling_info=object()),
        requests,
    )

    assert len(sample_calls) == 1
    sampled_hidden, sampled_ids, sampled_params = sample_calls[0]
    assert torch.equal(sampled_hidden, hidden_states[1:2])
    assert sampled_ids == ["final"]
    assert sampled_params == ["param-1"]


def test_chunked_prefill_rows_do_not_advance_generation_steps() -> None:
    runner = object.__new__(HiggsTTSModelRunner)
    requests = [
        _request("middle", extend_input_len=1, inflight_middle_chunks=2),
        _request("final", extend_input_len=1),
    ]

    assert runner.finalize_skip_rids(SimpleNamespace(requests=requests)) == {"middle"}
