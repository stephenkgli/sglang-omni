# SPDX-License-Identifier: Apache-2.0
"""Higgs native prefill CUDA graph model contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.models.higgs_tts.model import HiggsGenParams, HiggsTTSModel
from sglang_omni.models.higgs_tts.model_runner import HiggsTTSModelRunner


class _FakeInnerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input_embeds: torch.Tensor | None = None

    def forward(self, input_ids, positions, forward_batch, input_embeds):
        del positions, forward_batch
        self.last_input_embeds = input_embeds
        return torch.arange(
            input_ids.shape[0] * 4,
            dtype=torch.float32,
            device=input_ids.device,
        ).reshape(input_ids.shape[0], 4)


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeInnerModel()
        self.config = SimpleNamespace(vocab_size=8)


class _ExtendMode:
    @staticmethod
    def is_decode() -> bool:
        return False

    @staticmethod
    def is_extend() -> bool:
        return True


def _bare_higgs_model() -> HiggsTTSModel:
    model = HiggsTTSModel.__new__(HiggsTTSModel)
    nn.Module.__init__(model)
    model.backbone = _FakeBackbone()
    model._pending_prefill = None
    return model


def test_higgs_language_model_aliases_are_not_registered_twice() -> None:
    model = _bare_higgs_model()
    backbone = model.backbone

    assert model.model is backbone
    assert model.language_model is backbone
    assert set(model._modules) == {"backbone"}

    model.model = backbone
    model.language_model = backbone

    assert model.model is backbone
    assert model.language_model is backbone
    assert set(model._modules) == {"backbone"}

    with pytest.raises(AssertionError, match="may only alias"):
        model.model = nn.Linear(4, 4)


def test_higgs_prefill_copies_staged_embeds_into_static_graph_buffer() -> None:
    model = _bare_higgs_model()
    staged = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    static_input_embeds = torch.zeros((4, 4), dtype=torch.float32)
    sampled: dict[str, object] = {}

    def fake_decode(hidden_states, req_ids, gen_params):
        sampled["hidden_states"] = hidden_states
        sampled["req_ids"] = req_ids
        sampled["gen_params"] = gen_params
        return torch.zeros((len(req_ids), 8), dtype=torch.float32)

    model.decode_codebooks_batch = fake_decode
    gen_params = [HiggsGenParams(temperature=0.7)]
    model.stage_prefill(
        input_embeds=staged,
        req_ids=["req-1"],
        gen_params=gen_params,
        sample_row_indices=(0,),
    )
    forward_batch = SimpleNamespace(
        forward_mode=_ExtendMode(),
        input_embeds=static_input_embeds,
        extend_seq_lens=torch.tensor([2]),
    )

    output = model.forward(
        torch.tensor([1, 2, 0, 0]),
        torch.arange(4),
        forward_batch,
    )

    assert torch.equal(static_input_embeds[:2], staged)
    assert torch.count_nonzero(static_input_embeds[2:]) == 0
    assert model.backbone.model.last_input_embeds is static_input_embeds
    assert sampled["req_ids"] == ["req-1"]
    assert sampled["gen_params"] == gen_params
    assert model._pending_prefill is None
    assert output.next_token_logits.shape == (1, 8)


def test_higgs_prefill_middle_chunk_does_not_touch_sampler_state() -> None:
    model = _bare_higgs_model()

    def fail_decode(*_args, **_kwargs):
        raise AssertionError("middle chunk must not run the sampler")

    model.decode_codebooks_batch = fail_decode
    model.stage_prefill(
        input_embeds=torch.ones((2, 4), dtype=torch.float32),
        req_ids=["chunked"],
        gen_params=[HiggsGenParams()],
        sample_row_indices=(),
    )
    forward_batch = SimpleNamespace(
        forward_mode=_ExtendMode(),
        input_embeds=torch.zeros((4, 4), dtype=torch.float32),
        extend_seq_lens=torch.tensor([2]),
    )

    output = model.forward(
        torch.tensor([1, 2, 0, 0]),
        torch.arange(4),
        forward_batch,
    )

    assert torch.count_nonzero(output.next_token_logits) == 0
    assert output.next_token_logits.shape == (1, 8)
    assert model._pending_prefill is None


def test_higgs_prefill_samples_only_final_chunk_rows() -> None:
    model = _bare_higgs_model()
    sampled: dict[str, object] = {}

    def fake_decode(hidden_states, req_ids, gen_params):
        sampled["hidden_states"] = hidden_states
        sampled["req_ids"] = req_ids
        sampled["gen_params"] = gen_params
        return torch.zeros((len(req_ids), 8), dtype=torch.float32)

    model.decode_codebooks_batch = fake_decode
    gen_params = [HiggsGenParams(temperature=0.5), HiggsGenParams(temperature=0.8)]
    model.stage_prefill(
        input_embeds=torch.ones((4, 4), dtype=torch.float32),
        req_ids=["chunked", "final"],
        gen_params=gen_params,
        sample_row_indices=(1,),
    )
    forward_batch = SimpleNamespace(
        forward_mode=_ExtendMode(),
        input_embeds=torch.zeros((4, 4), dtype=torch.float32),
        extend_seq_lens=torch.tensor([2, 2]),
    )

    output = model.forward(
        torch.tensor([1, 2, 3, 4]),
        torch.arange(4),
        forward_batch,
    )

    assert sampled["req_ids"] == ["final"]
    assert sampled["gen_params"] == [gen_params[1]]
    assert torch.equal(
        sampled["hidden_states"],
        torch.tensor([[12.0, 13.0, 14.0, 15.0]]),
    )
    assert output.next_token_logits.shape == (2, 8)


def test_higgs_prefill_capture_warmup_does_not_touch_sampler_state() -> None:
    model = _bare_higgs_model()

    def fail_decode(*_args, **_kwargs):
        raise AssertionError("capture warmup must not run the eager sampler")

    model.decode_codebooks_batch = fail_decode
    forward_batch = SimpleNamespace(
        forward_mode=_ExtendMode(),
        input_embeds=torch.zeros((4, 4), dtype=torch.float32),
        extend_seq_lens=torch.tensor([4]),
    )

    output = model.forward(
        torch.tensor([1, 2, 3, 4]),
        torch.arange(4),
        forward_batch,
    )

    assert torch.count_nonzero(output.next_token_logits) == 0
    assert output.next_token_logits.shape == (1, 8)
    assert model._pending_prefill is None


def test_higgs_runner_keeps_live_forward_batch_input_embeds_empty() -> None:
    runner = object.__new__(HiggsTTSModelRunner)
    staged: dict[str, object] = {}

    class FakeModel:
        def set_request_seed(self, request_id, seed) -> None:
            staged["seed"] = (request_id, seed)

        def _gen_params_for_batch(self, sampling_info, batch_size):
            staged["sampling"] = (sampling_info, batch_size)
            return [HiggsGenParams()]

        def stage_prefill(self, **kwargs) -> None:
            staged["prefill"] = kwargs

    runner.model = FakeModel()
    runner._build_prefill_input_embeds = lambda _batch, _requests: torch.ones((2, 4))
    sampling_info = object()
    forward_batch = SimpleNamespace(input_embeds=object(), sampling_info=sampling_info)
    request = SimpleNamespace(
        request_id="req-1",
        data=SimpleNamespace(
            req=SimpleNamespace(
                sampling_params=SimpleNamespace(sampling_seed=7),
            )
        ),
    )

    runner.before_prefill(
        forward_batch,
        SimpleNamespace(chunked_req=None),
        [request],
    )

    assert forward_batch.input_embeds is None
    assert forward_batch.req_ids == ["req-1"]
    assert staged["seed"] == ("req-1", 7)
    assert staged["sampling"] == (sampling_info, 1)
    assert staged["prefill"]["req_ids"] == ["req-1"]
    assert staged["prefill"]["sample_row_indices"] == (0,)


def test_higgs_runner_excludes_schedule_batch_chunked_request() -> None:
    runner = object.__new__(HiggsTTSModelRunner)
    seeded: list[tuple[str, int]] = []
    staged: dict[str, object] = {}

    class FakeModel:
        def set_request_seed(self, request_id, seed) -> None:
            seeded.append((request_id, seed))

        def _gen_params_for_batch(self, _sampling_info, batch_size):
            return [HiggsGenParams() for _ in range(batch_size)]

        def stage_prefill(self, **kwargs) -> None:
            staged.update(kwargs)

    runner.model = FakeModel()
    runner._build_prefill_input_embeds = lambda _batch, _requests: torch.ones((4, 4))
    chunked_req = SimpleNamespace(sampling_params=SimpleNamespace(sampling_seed=1))
    final_req = SimpleNamespace(sampling_params=SimpleNamespace(sampling_seed=2))
    requests = [
        SimpleNamespace(request_id="chunked", data=SimpleNamespace(req=chunked_req)),
        SimpleNamespace(request_id="final", data=SimpleNamespace(req=final_req)),
    ]
    forward_batch = SimpleNamespace(input_embeds=object(), sampling_info=object())

    runner.before_prefill(
        forward_batch,
        SimpleNamespace(chunked_req=chunked_req),
        requests,
    )

    assert seeded == [("final", 2)]
    assert staged["sample_row_indices"] == (1,)
    assert forward_batch.input_embeds is None


def test_higgs_post_prefill_ignores_middle_chunk_outputs() -> None:
    runner = object.__new__(HiggsTTSModelRunner)
    runner.model = SimpleNamespace(
        _rid_to_row={"chunked": 0},
        _output_codes={"chunked": [torch.tensor([1, 2, 3])]},
        _sampler_pool=SimpleNamespace(generation_done=torch.tensor([False])),
    )
    req = SimpleNamespace(rid="chunked", finished=lambda: False)
    data = SimpleNamespace(
        req=req,
        output_codes=[],
        return_omni_rollout=False,
        return_logprob=False,
    )
    requests = [SimpleNamespace(request_id="chunked", data=data)]
    schedule_batch = SimpleNamespace(chunked_req=req)
    result = SimpleNamespace(
        logits_output=SimpleNamespace(next_token_logits=torch.zeros(1, 8))
    )

    runner.post_prefill(result, None, schedule_batch, requests)

    assert data.output_codes == []
    assert result.next_token_ids.tolist() == [0]
    assert runner.finalize_skip_rids(SimpleNamespace(batch_data=schedule_batch)) == {
        "chunked"
    }
