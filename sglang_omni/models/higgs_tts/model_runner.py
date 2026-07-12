# SPDX-License-Identifier: Apache-2.0
"""Higgs TTS model runner — phase-aware AR base-runner subclass.

Decode-mode hooks gather sampler-pool state into ``_cg_active_*`` shadow
buffers before the captured forward and scatter results back after, so
the graph itself only ever does ``_cg_active_*[:bs]`` slicing — no
``pool[row_indices]`` gather/scatter under capture (capture-time
``row_indices`` are all-zero placeholders → duplicate-index UB).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.higgs_tts.model import (
    HiggsPrefillEmbeddingInputs,
    _flat_sampling_attr,
)
from sglang_omni.models.higgs_tts.sampler import K_MAX, selected_token_logprobs
from sglang_omni.models.higgs_tts.text_tokenizer import AUDIO_PLACEHOLDER_ID
from sglang_omni.models.higgs_tts.utils import EOC_ID
from sglang_omni.scheduling.messages import OutgoingMessage

logger = logging.getLogger(__name__)


class HiggsTTSModelRunner(ModelRunner):
    """ModelRunner for :class:`HiggsTTSModel`."""

    def __init__(self, tp_worker: Any, output_processor: Any) -> None:
        super().__init__(tp_worker, output_processor)
        self._outbox: Any | None = None
        self._vocoder_target = "vocoder"
        # Ping-pong pinned host buffers for the async-decode rollout-logprob D2H.
        self._logprob_host_buffers: list[torch.Tensor] | None = None
        self._logprob_slot = 0

    def _next_logprob_host_staging(self, device_buf: torch.Tensor) -> torch.Tensor:
        if self._logprob_host_buffers is None:
            self._logprob_host_buffers = [
                torch.empty(
                    device_buf.shape,
                    dtype=device_buf.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                for _ in range(2)
            ]
        buf = self._logprob_host_buffers[self._logprob_slot]
        self._logprob_slot ^= 1
        return buf

    def set_stream_outbox(self, outbox: Any) -> None:
        self._outbox = outbox

    def before_prefill(self, forward_batch, schedule_batch, requests):
        del schedule_batch
        assert forward_batch.batch_size == len(requests), (
            f"Higgs prefill batch_size={forward_batch.batch_size} does not match "
            f"{len(requests)} scheduled requests"
        )
        assert forward_batch.input_embeds is None, (
            "Higgs prefill must keep ForwardBatch.input_embeds unset so SGLang "
            "piecewise CUDA graph can run"
        )
        existing_mm_inputs = forward_batch.mm_inputs
        assert existing_mm_inputs is None or (
            len(existing_mm_inputs) == len(requests)
            and all(item is None for item in existing_mm_inputs)
        ), "Higgs prefill received unexpected SGLang multimodal inputs"

        for req in requests:
            self.model.set_request_seed(
                req.request_id, req.data.req.sampling_params.sampling_seed
            )
        embedding_overrides = self._build_prefill_embedding_overrides(
            forward_batch, requests
        )
        if self._has_branched_radix_prefill(forward_batch):
            forward_batch.input_embeds = self._materialize_prefill_input_embeds(
                forward_batch.input_ids,
                embedding_overrides,
            )
            forward_batch.mm_inputs = None
        else:
            forward_batch.mm_inputs = embedding_overrides

    def post_prefill(self, result, forward_batch, schedule_batch, requests):
        del schedule_batch
        self._sample_prefill_codebooks(result, forward_batch, requests)
        self._collect_step_outputs(result, requests, forward_batch)

    def finalize_skip_rids(self, scheduler_output) -> set[str]:
        return {
            request.request_id
            for request in scheduler_output.requests
            if request.data.req.inflight_middle_chunks > 0
        }

    def before_decode(
        self,
        forward_batch,
        schedule_batch,
        requests,
        *,
        is_lookahead: bool = False,
    ):
        del schedule_batch
        forward_batch.req_ids = [req.request_id for req in requests]
        self._populate_cg_buffers(forward_batch, requests, is_lookahead=is_lookahead)

    def post_decode(self, result, forward_batch, schedule_batch, requests):
        del schedule_batch
        self._collect_step_outputs_cg(result, forward_batch, requests)

    def post_decode_launch(self, result, forward_batch, requests):
        """Async-decode GPU half: scatter + pack (GPU->GPU), then a
        non-blocking copy of the staging snapshot into a pinned host staging buffer.
        Returns the buffer; the base runner records the event right after, so
        ``event.query()`` then means "this snapshot is on the host".
        """
        if len(requests) == 0:
            return None
        n_real = len(requests)
        bs = int(forward_batch.batch_size)
        if bs < n_real:
            raise ValueError(
                f"forward_batch.batch_size ({bs}) < len(requests) ({n_real})"
            )
        staging = self._decode_pack_gpu(n_real)
        collect_staging = self.model._cg_collect_staging
        host_buf = self._next_host_staging(collect_staging.shape, collect_staging.dtype)
        host_buf[:n_real].copy_(staging[:n_real], non_blocking=True)
        logprob_host = None
        if self._should_capture_rollout_logprobs(requests):
            logprobs_BN = self._decode_step_logprobs(result, n_real)
            logprob_host = self._next_logprob_host_staging(logprobs_BN)
            logprob_host[:n_real].copy_(logprobs_BN[:n_real], non_blocking=True)
        # Set next_token_ids (cb0) from GPU state now, with NO host sync, so the
        # AR input chain (next step's input_ids = this step's output_ids) is
        # available at launch — the host collect (post_decode_resolve) lags by
        # one step under lookahead. For Higgs the decode input_ids is masked by
        # _decode_step_embeds_cg (rows with codes use _cg_active_last_codes), so
        # this only feeds the upstream bookkeeping. clamp>=0 keeps STOP_CODE(-1)
        # rows in embed_tokens range; the host collect later overwrites with the
        # skip-aware cb0 for output reporting.
        result.next_token_ids = (
            self.model._cg_codes_BN[:n_real, 0].clamp_min(0).to(torch.long).clone()
        )
        return host_buf, logprob_host

    def post_decode_resolve(
        self, host_buf, result, forward_batch, schedule_batch, requests
    ):
        """Async-decode host half: read the already-copied pinned snapshot and
        run the per-request collect loop. Mirrors the tail of
        ``_collect_step_outputs_cg`` (shares ``_decode_collect_host``).
        """
        del forward_batch, schedule_batch
        if len(requests) == 0:
            return
        n_real = len(requests)
        host_buf, logprob_host = host_buf
        logprobs_cpu = None if logprob_host is None else logprob_host[:n_real]
        self._decode_collect_host(
            host_buf[:n_real],
            logprobs_cpu,
            result,
            requests,
            next_token_device=None,
        )

    def _populate_cg_buffers(
        self, forward_batch, requests, *, is_lookahead: bool = False
    ) -> None:
        """Fill the model's CG buffers for one decode step.

        Padding rows (``batch_size > len(requests)``) point at the
        reserved padding row, which is reset every step so it can't
        leak state into real rows.
        """
        model = self.model
        bs = int(forward_batch.batch_size)
        n_real = len(requests)
        if bs < n_real:
            raise ValueError(
                f"forward_batch.batch_size ({bs}) < len(requests) ({n_real})"
            )

        model._sampler_pool.reset_row(model._padding_row)

        rows_py: list[int] = [model.acquire_row(req.request_id) for req in requests]
        rows_py.extend([model._padding_row] * (bs - n_real))
        model._cg_row_indices[:bs] = torch.tensor(
            rows_py, dtype=torch.long, device=model._cg_row_indices.device
        )

        if self._async_enabled and is_lookahead and n_real > 0:
            # Async-lookahead overrun guard (GPU-side, no host sync): a request
            # that finished via EOC at the prior step is still in this batch
            # with pool.generation_done=True. Running the normal decode forward
            # for such a done row trips a device-side gather assert, so route it
            # to the reset padding row — its overrun output is discarded by the
            # collect's finished()/was_done skip anyway. Length-finish rows have
            # generation_done=False and are untouched.
            #
            # Only the lookahead launch path can carry such an overrun (the
            # 1-wasted-step lag). On a fast-path (sync) decode step finished reqs
            # are filtered out before the step, so no generation_done row is ever
            # present and this gather+torch.where would be pure wasted GPU work.
            rows_t_real = model._cg_row_indices[:n_real]
            done = model._sampler_pool.generation_done[rows_t_real]
            model._cg_row_indices[:n_real] = torch.where(
                done, torch.full_like(rows_t_real, model._padding_row), rows_t_real
            )

        temps, top_ps, top_ks = self._extract_decode_sampling_params(
            forward_batch, n_real
        )
        temps.extend([1.0] * (bs - n_real))
        top_ps.extend([1.0] * (bs - n_real))
        model._cg_temperature[:bs] = torch.tensor(
            temps, dtype=torch.float32, device=model._cg_temperature.device
        )
        model._cg_top_p[:bs] = torch.tensor(
            top_ps, dtype=torch.float32, device=model._cg_top_p.device
        )

        top_k_vals = [(tk if (tk is not None and tk > 0) else K_MAX) for tk in top_ks]
        top_k_vals.extend([K_MAX] * (bs - n_real))
        model._cg_top_k_buf[:bs] = torch.tensor(
            top_k_vals, dtype=torch.long, device=model._cg_top_k_buf.device
        )

        rows_t = model._cg_row_indices[:bs]
        pool = model._sampler_pool
        model._cg_active_delay_count[:bs] = pool.delay_count[rows_t]
        model._cg_active_eoc_countdown[:bs] = pool.eoc_countdown[rows_t]
        model._cg_active_generation_done[:bs] = pool.generation_done[rows_t]
        model._cg_active_last_codes[:bs] = pool.last_codes[rows_t]
        model._cg_active_seeds[:bs] = pool.seeds[rows_t]
        model._cg_active_step_count[:bs] = pool.step_count[rows_t]

    @staticmethod
    def _extract_decode_sampling_params(forward_batch, n_real: int):
        """Pull per-row temperature / top_p / top_k off sglang's
        ``sampling_info`` with safe defaults. ``top_k`` values outside
        ``(0, K_MAX)`` (including sglang's ``TOP_K_ALL`` sentinel for
        unspecified top_k) are normalized to ``None`` — the downstream
        buffer maps that to ``K_MAX`` = no-op filter.
        """
        sampling_info = getattr(forward_batch, "sampling_info", None)
        if sampling_info is None or n_real == 0:
            return ([1.0] * n_real, [1.0] * n_real, [None] * n_real)

        temps_raw = _flat_sampling_attr(sampling_info, "temperatures") or [1.0] * n_real
        top_ps_raw = _flat_sampling_attr(sampling_info, "top_ps") or [1.0] * n_real
        top_ks_raw = _flat_sampling_attr(sampling_info, "top_ks")

        temps = [float(t) for t in temps_raw[:n_real]]
        top_ps = [float(t) for t in top_ps_raw[:n_real]]
        if top_ks_raw is None:
            top_ks: list[int | None] = [None] * n_real
        else:
            top_ks = [
                int(t) if (t is not None and 0 < int(t) < K_MAX) else None
                for t in top_ks_raw[:n_real]
            ]
        return temps, top_ps, top_ks

    def _collect_step_outputs_cg(
        self, result: Any, forward_batch: Any, requests: list
    ) -> None:
        """Synchronous collect: scatter + pack (GPU->GPU), one blocking D2H,
        then the host collect loop. Used when async decode is off; behavior is
        identical to the pre-split implementation (now factored into
        ``_decode_pack_gpu`` + ``_decode_collect_host``, which the async
        ``post_decode_launch`` / ``post_decode_resolve`` also reuse).
        """
        if len(requests) == 0:
            return
        n_real = len(requests)
        bs = int(forward_batch.batch_size)
        if bs < n_real:
            raise ValueError(
                f"forward_batch.batch_size ({bs}) < len(requests) ({n_real})"
            )
        staging = self._decode_pack_gpu(n_real)
        combined_cpu = staging[:n_real].cpu()  # one blocking D2H (sync path)
        logprobs_cpu = None
        if self._should_capture_rollout_logprobs(requests):
            logprobs_cpu = self._decode_step_logprobs(result, n_real)[:n_real].cpu()
        self._decode_collect_host(
            combined_cpu,
            logprobs_cpu,
            result,
            requests,
            next_token_device=result.logits_output.next_token_logits.device,
        )

    def _decode_pack_gpu(self, n_real: int) -> torch.Tensor:
        """Scatter shadow sampler state back into the pool and pack the three
        collect tensors (codes / was_done / generation_done) into the staging
        buffer. All GPU->GPU; returns the device staging buffer.
        """
        model = self.model
        rows_t = model._cg_row_indices[:n_real]
        pool = model._sampler_pool
        pool.delay_count[rows_t] = model._cg_active_delay_count[:n_real]
        pool.eoc_countdown[rows_t] = model._cg_active_eoc_countdown[:n_real]
        pool.generation_done[rows_t] = model._cg_active_generation_done[:n_real]
        pool.last_codes[rows_t] = model._cg_active_last_codes[:n_real]
        pool.step_count[rows_t] = model._cg_active_step_count[:n_real]

        # Note(Jiaxin): pack the 3 tensors so a single D2H pulls them all back.
        num_codebooks = model._cg_codes_BN.shape[1]
        staging = model._cg_collect_staging
        staging[:n_real, :num_codebooks] = model._cg_codes_BN[:n_real]
        staging[:n_real, num_codebooks] = model._cg_was_done[:n_real]
        staging[:n_real, num_codebooks + 1] = model._cg_active_generation_done[:n_real]
        return staging

    def _decode_collect_host(
        self,
        combined_cpu: torch.Tensor,
        logprobs_cpu: torch.Tensor | None,
        result: Any,
        requests: list,
        *,
        next_token_device: torch.device | None,
    ) -> None:
        """Host-side collect loop over an already-D2H'd staging snapshot:
        append per-request codes, mark finishes, build ``result.next_token_ids``.
        Skips chunked and already-done rows (the latter is what makes the
        one-step-lookahead overrun harmless — see r1_idempotency_check.md).

        ``next_token_device`` is set for synchronous decode because those ids
        feed the next step. Async resolve passes ``None``: launch already
        published GPU codebook-0, and resolve only needs a CPU tensor for
        output processing.
        """
        model = self.model
        num_codebooks = model._cg_codes_BN.shape[1]
        codes_BN_cpu = combined_cpu[:, :num_codebooks]
        was_done_cpu = combined_cpu[:, num_codebooks].bool().tolist()
        gen_done_after_cpu = combined_cpu[:, num_codebooks + 1].bool().tolist()
        cb0_per_row: list[int] = []
        for b, sched_req in enumerate(requests):
            data = sched_req.data
            req = data.req
            if req.inflight_middle_chunks > 0:
                cb0_per_row.append(0)
                continue
            # Already finished in an earlier step? Skip its append. Under async
            # lookahead the finished req gets one extra (wasted) forward before
            # being dropped; this prevents leaking that overrun token. Catches
            # length finishes too (which `_cg_was_done`, an EOC-only flag, does
            # not). No-op for the sync path: a req is never finished() at its
            # own collect (finish is set later, in process_batch_result).
            if req.finished():
                cb0_per_row.append(0)
                continue
            if was_done_cpu[b]:
                cb0_per_row.append(0)
                continue
            codes_N = codes_BN_cpu[b].to(torch.long).clone()
            data.output_codes.append(codes_N)
            if logprobs_cpu is not None and self._request_captures_rollout_logprobs(
                sched_req
            ):
                data.output_logprobs.append(logprobs_cpu[b].to(torch.float32).clone())
            data.generation_done = bool(gen_done_after_cpu[b])
            self._emit_code_chunk(sched_req, codes_N)
            self._mark_sampler_finished(req, data.generation_done)
            cb0_per_row.append(int(codes_N[0].item()))

        if next_token_device is None:
            result.next_token_ids = torch.tensor(cb0_per_row, dtype=torch.long)
        else:
            result.next_token_ids = torch.tensor(
                cb0_per_row,
                dtype=torch.long,
                device=next_token_device,
            )

    def _build_prefill_embedding_overrides(
        self,
        forward_batch: Any,
        requests: list,
    ) -> list[HiggsPrefillEmbeddingInputs | None]:
        input_ids = forward_batch.input_ids
        placeholder_mask = input_ids == AUDIO_PLACEHOLDER_ID
        extend_seq_lens = forward_batch.extend_seq_lens_cpu
        extend_prefix_lens = forward_batch.extend_prefix_lens_cpu
        assert len(extend_seq_lens) == len(requests)
        assert len(extend_prefix_lens) == len(requests)

        overrides: list[HiggsPrefillEmbeddingInputs | None] = []
        offset = 0
        for sched_req, extend_len_raw, prefix_len_raw in zip(
            requests, extend_seq_lens, extend_prefix_lens
        ):
            extend_len = int(extend_len_raw)
            req = sched_req.data.req
            assert req.extend_range is not None
            assert extend_len == int(req.extend_range.length)
            end = offset + extend_len
            overrides.append(
                self._build_request_prefill_embedding_override(
                    sched_req,
                    placeholder_mask[offset:end],
                    flattened_offset=offset,
                    prefix_len=int(prefix_len_raw),
                    device=input_ids.device,
                )
            )
            offset = end

        assert offset == input_ids.shape[0], (
            f"Higgs flattened prefill has {input_ids.shape[0]} tokens but request "
            f"extend lengths sum to {offset}"
        )
        return overrides

    @staticmethod
    def _has_branched_radix_prefill(forward_batch: Any) -> bool:
        """Return whether this batch extends a cached prefix with a new branch.

        SGLang deliberately replays one token for an exact prompt cache hit.
        More than one extend token after a non-empty prefix means the request
        branched from another prompt. Higgs keeps this case eager because the
        tc_piecewise Qwen3 path can move the seeded first codec draw across a
        probability boundary and make the decoder echo the reference prompt.
        """
        prefix_lens = forward_batch.extend_prefix_lens_cpu
        extend_lens = forward_batch.extend_seq_lens_cpu
        assert len(prefix_lens) == len(extend_lens)
        return any(
            int(prefix_len) > 0 and int(extend_len) > 1
            for prefix_len, extend_len in zip(prefix_lens, extend_lens)
        )

    def _materialize_prefill_input_embeds(
        self,
        input_ids: torch.Tensor,
        embedding_overrides: list[HiggsPrefillEmbeddingInputs | None],
    ) -> torch.Tensor:
        placeholder_mask = input_ids == AUDIO_PLACEHOLDER_ID
        safe_input_ids = torch.where(
            placeholder_mask,
            torch.zeros_like(input_ids),
            input_ids,
        )
        input_embeds = self.model.backbone.model.embed_tokens(safe_input_ids)
        for overrides in embedding_overrides:
            if overrides is None:
                continue
            input_embeds.index_copy_(
                0,
                overrides.positions,
                overrides.embeddings.to(input_embeds.dtype),
            )
        return input_embeds

    def _build_request_prefill_embedding_override(
        self,
        sched_req: Any,
        placeholder_mask: torch.Tensor,
        *,
        flattened_offset: int,
        prefix_len: int,
        device: torch.device,
    ) -> HiggsPrefillEmbeddingInputs | None:
        local_positions = placeholder_mask.nonzero(as_tuple=True)[0]
        num_placeholders = local_positions.numel()
        if num_placeholders == 0:
            return None

        data = sched_req.data
        codes_rows = data.reference_codes_delayed
        if not codes_rows:
            raise ValueError(
                f"Higgs request {sched_req.request_id!r} contains audio "
                "placeholders without reference codes"
            )
        code_start = self._reference_code_offset(sched_req, prefix_len)
        code_end = code_start + num_placeholders
        if code_end > len(codes_rows):
            raise ValueError(
                f"Higgs request {sched_req.request_id!r} needs {code_end} "
                f"reference-code rows but only {len(codes_rows)} are available"
            )
        codes = torch.tensor(
            codes_rows[code_start:code_end],
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            embeddings = self.model.multimodal_embedding.modality_embedding_0(codes)
        return HiggsPrefillEmbeddingInputs(
            positions=local_positions + flattened_offset,
            embeddings=embeddings,
        )

    @staticmethod
    def _reference_code_offset(sched_req: Any, prefix_len: int) -> int:
        origin_input_ids = sched_req.data.req.origin_input_ids
        if prefix_len < 0 or prefix_len > len(origin_input_ids):
            raise ValueError(
                f"Higgs request {sched_req.request_id!r} has invalid prefill "
                f"prefix length {prefix_len} for {len(origin_input_ids)} input tokens"
            )
        return sum(
            token_id == AUDIO_PLACEHOLDER_ID
            for token_id in origin_input_ids[:prefix_len]
        )

    def _sample_prefill_codebooks(
        self,
        result: Any,
        forward_batch: Any,
        requests: list,
    ) -> None:
        final_indices = [
            index
            for index, request in enumerate(requests)
            if request.data.req.inflight_middle_chunks == 0
            and not request.data.req.finished()
        ]
        if not final_indices:
            return

        hidden_states = result.logits_output.hidden_states
        if hidden_states is None:
            raise RuntimeError("Higgs prefill requires last hidden states")
        if hidden_states.ndim == 3:
            hidden_states = hidden_states[:, -1, :]
        if hidden_states.shape[0] != len(requests):
            raise ValueError(
                f"Higgs prefill returned {hidden_states.shape[0]} hidden-state rows "
                f"for {len(requests)} requests"
            )

        all_gen_params = self.model._gen_params_for_batch(
            forward_batch.sampling_info,
            len(requests),
        )
        self.model.decode_codebooks_batch(
            hidden_states[final_indices],
            [requests[index].request_id for index in final_indices],
            [all_gen_params[index] for index in final_indices],
        )

    def _collect_step_outputs(
        self,
        result: Any,
        requests: list,
        forward_batch: Any | None = None,
    ) -> None:
        """Pull per-request newly emitted codes from the model into
        ``data.output_codes`` and overwrite ``result.next_token_ids``
        with codebook-0 so the base runner skips its text-vocab sampler.
        """
        batch_size = len(requests)
        if batch_size == 0:
            return

        model = self.model
        logprobs_BN = None
        if self._should_capture_rollout_logprobs(requests):
            logprobs_BN = self._prefill_step_logprobs(result, requests, forward_batch)
        cb0_per_row: list[int] = []
        for b, sched_req in enumerate(requests):
            data = sched_req.data
            req = data.req
            rid = sched_req.request_id
            row = model._rid_to_row.get(rid)
            codes_log = model._output_codes.get(rid)
            if (
                req.inflight_middle_chunks > 0
                or row is None
                or not codes_log
                or req.finished()
            ):
                cb0_per_row.append(0)
                continue
            codes_N = codes_log[-1]
            data.output_codes.append(codes_N.detach().cpu().clone())
            if logprobs_BN is not None and self._request_captures_rollout_logprobs(
                sched_req
            ):
                data.output_logprobs.append(logprobs_BN[b].detach().cpu().clone())
            data.generation_done = bool(model._sampler_pool.generation_done[row].item())
            self._emit_code_chunk(sched_req, data.output_codes[-1])
            self._mark_sampler_finished(req, data.generation_done)
            cb0_per_row.append(int(codes_N[0].item()))

        result.next_token_ids = torch.tensor(
            cb0_per_row,
            dtype=torch.long,
            device=result.logits_output.next_token_logits.device,
        )

    @staticmethod
    def _request_captures_rollout_logprobs(sched_req: Any) -> bool:
        data = sched_req.data
        return bool(data.return_omni_rollout and data.return_logprob)

    def _should_capture_rollout_logprobs(self, requests: list) -> bool:
        return any(self._request_captures_rollout_logprobs(req) for req in requests)

    def _decode_step_logprobs(self, result: Any, n_real: int) -> torch.Tensor:
        model = self.model
        hidden_states = result.logits_output.hidden_states
        if hidden_states.ndim == 3:
            hidden_states = hidden_states[:, -1, :]
        logits_BNV = model.modality_head.generate(hidden_states[:n_real]).to(
            torch.float32
        )
        codes_BN = model._cg_codes_BN[:n_real].clamp_min(0)
        return selected_token_logprobs(
            logits_BNV,
            codes_BN,
            temperature=model._cg_temperature[:n_real],
            top_k_buf=model._cg_top_k_buf[:n_real],
        )

    def _prefill_step_logprobs(
        self, result: Any, requests: list, forward_batch: Any | None
    ) -> torch.Tensor:
        del forward_batch
        model = self.model
        hidden_states = result.logits_output.hidden_states
        if hidden_states.ndim == 3:
            hidden_states = hidden_states[:, -1, :]
        logits_BNV = model.modality_head.generate(hidden_states[: len(requests)]).to(
            torch.float32
        )
        codes = []
        temps = []
        top_ks = []
        for sched_req in requests:
            rid = sched_req.request_id
            codes_log = model._output_codes.get(rid)
            if codes_log:
                codes.append(codes_log[-1])
            else:
                codes.append(
                    torch.zeros(
                        model._num_codebooks,
                        dtype=torch.long,
                        device=logits_BNV.device,
                    )
                )
            sp = sched_req.data.req.sampling_params
            temps.append(float(getattr(sp, "temperature", 1.0)))
            top_k = getattr(sp, "top_k", None)
            top_ks.append(
                int(top_k) if (top_k is not None and int(top_k) > 0) else K_MAX
            )
        codes_BN = torch.stack(
            [c.to(device=logits_BNV.device, dtype=torch.long) for c in codes]
        )
        temperature = torch.tensor(temps, dtype=torch.float32, device=logits_BNV.device)
        top_k_buf = torch.tensor(top_ks, dtype=torch.long, device=logits_BNV.device)
        return selected_token_logprobs(
            logits_BNV,
            codes_BN.clamp_min(0),
            temperature=temperature,
            top_k_buf=top_k_buf,
        )

    @staticmethod
    def _mark_sampler_finished(req: Any, generation_done: bool) -> None:
        """Bridge Higgs sampler completion into upstream SGLang finish state."""
        if generation_done and req.finished_reason is None:
            req.finished_reason = FINISH_MATCHED_TOKEN(EOC_ID)

    def _emit_code_chunk(self, sched_req: Any, codes_N: torch.Tensor) -> None:
        if self._outbox is None:
            return
        metadata = sched_req.data.stream_metadata
        if metadata is None:
            return
        self._outbox.put(
            OutgoingMessage(
                request_id=sched_req.request_id,
                type="stream",
                target=self._vocoder_target,
                data=codes_N,
                metadata=metadata,
            )
        )


__all__ = ["HiggsTTSModelRunner"]
