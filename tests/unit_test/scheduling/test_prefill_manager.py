# SPDX-License-Identifier: Apache-2.0
"""Chunked-prefill accounting against the pinned SGLang request contract."""

from __future__ import annotations

from types import SimpleNamespace

from sglang_omni.scheduling.sglang_backend import prefill


def test_chunked_request_uses_inflight_middle_chunk_counter(monkeypatch) -> None:
    request = SimpleNamespace(
        rid="chunked",
        _input_embeds_are_projected=False,
        inflight_middle_chunks=0,
        extend_input_len=4,
        init_next_round_input=lambda: None,
    )

    class FakePrefillAdder:
        def __init__(self, **_kwargs) -> None:
            self.can_run_list = []
            self.new_chunked_req = None

        def add_chunked_req(self, req):
            self.can_run_list.append(req)
            return req

    class FakeBatch:
        def __init__(self, chunked_req) -> None:
            self.chunked_req = chunked_req
            self.prepared = False

        def prepare_for_extend(self) -> None:
            self.prepared = True

    monkeypatch.setattr(prefill, "PrefillAdder", FakePrefillAdder)
    monkeypatch.setattr(
        prefill.ScheduleBatch,
        "init_new",
        lambda **kwargs: FakeBatch(kwargs["chunked_req"]),
    )

    tree_cache = SimpleNamespace(cache_unfinished_req=lambda *_args, **_kwargs: None)
    manager = prefill.PrefillManager(
        page_size=1,
        chunked_prefill_size=4,
        max_prefill_tokens=8,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        tree_cache=tree_cache,
        model_config=None,
        enable_overlap=False,
    )
    manager.chunked_req = request

    batch = manager.schedule_next_batch(None, num_allocatable_reqs=1)

    assert request.inflight_middle_chunks == 1
    assert batch.chunked_req is request
    assert batch.prepared is True
