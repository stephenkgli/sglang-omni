# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.mem_cache.radix_cache import RadixKey

from sglang_omni.models.higgs_tts import request_builders
from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.streaming_vocoder import INITIAL_CODEC_CHUNK_FRAMES_PARAM


def test_higgs_reference_audio_namespaces_radix_cache() -> None:
    prompt = [-100, -100, 42]
    first = request_builders.build_sglang_higgs_request(
        HiggsTtsState(
            prompt_token_ids=prompt,
            reference_codes_delayed=torch.tensor([[1, 2], [3, 4]]),
        ),
        request_id="first",
    ).req
    second = request_builders.build_sglang_higgs_request(
        HiggsTtsState(
            prompt_token_ids=prompt,
            reference_codes_delayed=torch.tensor([[5, 6], [7, 8]]),
        ),
        request_id="second",
    ).req

    assert first.origin_input_ids == second.origin_input_ids
    assert first.extra_key != second.extra_key
    with pytest.raises(ValueError, match="matching extra_key"):
        RadixKey(first.origin_input_ids, first.extra_key).match(
            RadixKey(second.origin_input_ids, second.extra_key)
        )


def test_higgs_scheduler_adapters_clamp_cap_and_record_engine_time(
    monkeypatch,
) -> None:
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(
        request_builders,
        "_perf_counter",
        lambda: next(ticks),
    )
    request_builder, result_adapter = request_builders.make_higgs_scheduler_adapters(
        max_new_tokens_cap=2048,
    )
    state = HiggsTtsState(
        prompt_token_ids=[1, 2, 3],
        max_new_tokens=4096,
    )
    payload = StagePayload(
        request_id="req-higgs",
        request=OmniRequest(
            inputs={"references": ["large-reference"]},
            params={
                "stream": True,
                INITIAL_CODEC_CHUNK_FRAMES_PARAM: 8,
                "ref_audio": "large-reference",
            },
            metadata={"frontend_only": True},
        ),
        data=state.to_dict(),
    )

    data = request_builder(payload)
    assert data.stage_payload.data == {}
    assert data.stage_payload.request.inputs is None
    assert data.stage_payload.request.params == {
        "stream": True,
        INITIAL_CODEC_CHUNK_FRAMES_PARAM: 8,
    }
    assert data.stage_payload.request.metadata == {}
    data.output_codes.append(torch.tensor([1, 2, 3], dtype=torch.long))
    result = result_adapter(data)

    assert data.max_new_tokens == 2048
    assert data.req.sampling_params.max_new_tokens == 2048
    assert torch.equal(
        result.data["output_codes_delayed"],
        torch.tensor([[1, 2, 3]], dtype=torch.long),
    )
    assert result.data["completion_tokens"] == 1
    assert result.data["engine_time_s"] == 2.5
    # The result adapter rebuilds the state from the engine's own contract, so
    # frontend fields cannot ride the vocoder hop even though the inbound
    # payload carried them.
    assert result.data["num_codebooks"] == data.num_codebooks
    assert result.data["codebook_size"] == data.codebook_size
    assert result.data["prompt_token_ids"] == []
    assert "reference_codes_delayed" not in result.data


def test_higgs_result_adapter_reads_output_code_buffer() -> None:
    _, result_adapter = request_builders.make_higgs_scheduler_adapters()
    state = HiggsTtsState(
        prompt_token_ids=[1, 2, 3],
        max_new_tokens=8,
        num_codebooks=3,
    )
    payload = StagePayload(
        request_id="req-higgs",
        request=OmniRequest(inputs={}),
        data=state.to_dict(),
    )
    data = request_builders.build_sglang_higgs_request(
        state, request_id=payload.request_id
    )
    data.stage_payload = payload
    data.output_codes.append(torch.tensor([9, 9, 9], dtype=torch.long))
    output_code_buffer = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    data.output_code_buffer = output_code_buffer
    data.output_code_count = 2

    result = result_adapter(data)

    output_codes = result.data["output_codes_delayed"]
    assert torch.equal(
        output_codes,
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
    )
    assert output_codes.device.type == "cpu"
    assert output_codes.is_contiguous()
    assert output_codes.data_ptr() != output_code_buffer.data_ptr()
    output_code_buffer.zero_()
    assert output_codes.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert result.data["completion_tokens"] == 2


def test_higgs_result_adapter_propagates_conversion_failure(
    monkeypatch,
) -> None:
    _, result_adapter = request_builders.make_higgs_scheduler_adapters()
    state = HiggsTtsState(prompt_token_ids=[1], max_new_tokens=8)
    payload = StagePayload(
        request_id="req-higgs-failed-result",
        request=OmniRequest(inputs={}),
        data=state.to_dict(),
    )
    data = request_builders.build_sglang_higgs_request(
        state, request_id=payload.request_id
    )
    data.stage_payload = payload

    def fail_result_conversion(_state, _data) -> None:
        raise RuntimeError("result conversion failed")

    monkeypatch.setattr(
        request_builders,
        "apply_higgs_result",
        fail_result_conversion,
    )

    with pytest.raises(RuntimeError, match="result conversion failed"):
        result_adapter(data)


def test_higgs_reference_fingerprint_matches_legacy_byte_packing() -> None:
    """The digest is a radix-cache namespace key, so it must survive dtype and
    layout changes. Pinned against the original two-bytes-per-code encoding."""
    rows = [
        [0, 1, 1023, 1024, 1025],
        [1025, 1024, 1023, 1, 0],
    ]
    expected_fingerprint = "1cc3f6dbe1b650d2b1b4ab25b32bce4f"

    # Non-contiguous int32 view: strides must not leak into the hashed bytes.
    backing = torch.tensor(rows, dtype=torch.int32).repeat_interleave(2, dim=1)
    strided_codes = backing[:, ::2]
    assert not strided_codes.is_contiguous()

    strided_data = request_builders.build_sglang_higgs_request(
        HiggsTtsState(reference_codes_delayed=strided_codes)
    )
    contiguous_data = request_builders.build_sglang_higgs_request(
        HiggsTtsState(reference_codes_delayed=torch.tensor(rows, dtype=torch.long))
    )

    assert strided_data.req.extra_key == expected_fingerprint
    assert contiguous_data.req.extra_key == expected_fingerprint
    # Aliased, not copied: the engine reads these rows throughout prefill.
    assert strided_data.reference_codes_delayed.data_ptr() == strided_codes.data_ptr()


def test_higgs_request_builder_rejects_non_tensor_reference_codes() -> None:
    """Delayed rows are tensor-only on the wire; a nested list means an upstream
    stage regressed to the list handoff and must fail at the boundary."""
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        request_builders.build_sglang_higgs_request(
            HiggsTtsState(reference_codes_delayed=[[1, 2], [3, 4]])
        )
