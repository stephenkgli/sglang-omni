# SGLang 0.5.15 bump and prefill CUDA graph regression

Date: 2026-07-13

Target worktree: `/private/tmp/sglang-omni-prefill-cg-main-20260712`

Target branch: `codex/bump-sglang-prefill-cuda-graph`

AutoDL instance: `af9bv589z3-7cbf32ff`

GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB

## Revisions

- Upstream baseline: `14d1d5841210fb216b0e3e8c00450183eea40841`
- Bump-only: `39fef56a`
- MOSS-TD PCG: `ca0a2e5e`
- Initial Higgs PCG: `400aa263`
- SGLang 0.5.15 compatibility follow-up: `5c3c53ec`
- Higgs PCG correctness follow-up: `43128cae`
- Final tested revision: `43128cae5d949fd52f752e849d2b3c41206f62b6`

## Artifact roots

- Prior Higgs P0 investigation: `/root/autodl-tmp/sglang-eval-lab/runs/higgs-pcg-seedtts-20260712-v1`
- This regression: `/root/autodl-tmp/sglang-eval-lab/runs/sglang-0.5.15-pcg-regression-20260713-v1`
- Raw SeedTTS input: `/root/autodl-tmp/datasets/seed-tts-eval-arrow/data/en-00000-of-00001.parquet`
- Raw Movies800Time input: `/root/autodl-tmp/huggingface/hub/datasets--zhaochenyang20--movies800time/snapshots/382168daa4e9d764318a20c1365fd0af0d00d785`

## Issue ledger

### I-001: Higgs prefill performed request-state sampling inside model forward

- Severity: P0 correctness
- Symptom: dummy PCG capture has no real request state; putting Higgs sampling in the captured model forward couples capture/replay to mutable request metadata and chunk state.
- Root cause: the initial integration staged request IDs, sampling parameters, and reference embeddings into model-owned pending state, then sampled inside `HiggsTTSModel.forward()`.
- Fix: model forward now returns last hidden states with zero text logits during prefill. `HiggsTTSModelRunner.post_prefill()` performs request-aware codebook sampling after replay.
- Validation: the final target suite passed 1,929 tests. Three full SeedTTS rounds completed 3,264/3,264 requests with no WER>50% outliers after the partial-prefix guard described in C-016.

### I-002: Shared prefill/decode input buffer retained audio placeholder IDs

- Severity: P0 correctness
- Symptom: decode CUDA Graph could fail in text embedding with an invalid `-100` index when a real decode batch was smaller than its graph bucket.
- Root cause: SGLang shares the named `input_ids` CUDA-graph buffer across prefill and decode. Higgs reference-audio prefill copied `-100` placeholders into that buffer; decode overwrote only real rows, leaving stale sentinels in padded rows.
- Fix: while inside SGLang tc-piecewise prefill, build reference embeddings first and then replace placeholder IDs in the shared `input_ids` buffer with legal token ID 0. CPU/radix-cache token state remains unchanged.
- Validation: predecessor reproduction passed 64/64 after the fix; SGLang 0.5.15 validation pending.

### I-003: Reference-code progress was stored as mutable per-request consumption

- Severity: P0 correctness
- Symptom: chunked prefill and partial radix-cache hits could select the wrong reference-code rows.
- Root cause: `num_ref_codes_consumed` depended on one linear execution history and could not represent an arbitrary cached absolute prefix.
- Fix: derive the code offset from the number of audio placeholders in `origin_input_ids[:extend_prefix_len]` for each replayed chunk. Keep `mm_inputs` aligned one-to-one with request rows.
- Validation: unit coverage for mixed zero-shot/reference batches and partial cached prefixes, plus the two-request live Radix-cache reproducer and three full SeedTTS rounds.

## Environment and dependency issues

### E-001: Local macOS Python cache and version mismatch

- Symptom: the first `py_compile` used Xcode Python 3.9 and attempted to write under a sandboxed user cache.
- Root cause: the default `python3` is below the project requirement and has an unsuitable bytecode cache root.
- Resolution: use the Codex workspace Python and `PYTHONPYCACHEPREFIX=/tmp/sglang-omni-pycache`.
- Bypassed: no code or test behavior; only the local interpreter/cache location changed.

### E-002: Data disk cannot hold a normal second 9.9 GiB venv

- Observation: `/root/autodl-tmp` has 9.7 GiB free; the existing baseline venv is 9.9 GiB.
- Resolution: hard-link cloned the old environment, which initially consumed about 15 MiB, then ran the dependency delta in the clone. The old environment remained on SGLang 0.5.12.post1; the new environment reports SGLang 0.5.15. The data disk has about 4 GiB free after the resolved stack was installed.

### E-003: `uv pip install -e .` did not honor project overrides

- Symptom: a fresh editable install failed on the pre-existing `s3prl`/`descript-audiotools` protobuf constraints.
- Root cause: the pip-compatible command does not resolve with `[tool.uv].override-dependencies` as project `uv sync` does.
- Resolution: use project-level `uv sync` for the environment build. No dependency constraint was bypassed.

### E-004: grpcio 1.82 requires protobuf 7

- Symptom: after the SGLang bump, `grpcio-health-checking`/`grpcio-reflection` 1.82.1 required protobuf >=7.35.1 while this repository intentionally overrides protobuf to <7.
- Root cause: unconstrained grpc transitive dependencies selected the newest 1.82 line.
- Resolution: constrain `grpcio`, `grpcio-health-checking`, and `grpcio-reflection` to >=1.81.1,<1.82. Version 1.81.1 satisfies SGLang's >=1.81.1 minimum and protobuf >=6.33.5,<7. `pip check` then showed only the same pre-existing `descript-audiotools 0.7.2` protobuf waiver present in the baseline environment.

### E-005: Initial targeted pytest path was stale

- Symptom: pytest exited 4 with no tests collected for `tests/unit_test/utils/test_cuda_graph_batch_validator.py`.
- Root cause: the test actually lives at `tests/unit_test/model_runner/test_cuda_graph_batch_validator.py`.
- Resolution: reran with the repository path; 136 tests passed.

### E-006: Host SOCKS proxy lacked the optional `socksio` package

- Symptom: two URL-policy tests attempted to construct an HTTPX client from `ALL_PROXY=socks5://127.0.0.1:7890` and raised ImportError before rejecting a blocked/private URL.
- Root cause: the connector constructed the client before applying its URL security policy.
- Resolution: defer sync/async client construction until after the first URL policy check. This fixes the ordering invariant; no proxy variable was removed for the validating rerun.

### E-007: Blackwell FA4 experiment was not usable for MOSS-TTS Local varlen attention

- Symptom: the first FA4 attempt exposed positional-signature mismatch, then CUTE compilation failures for ragged output layout and Python local-window arguments.
- Root cause: SGLang 0.5.15's FA4/CUTE path is not a drop-in replacement for this packed varlen/local-window call on SM120.
- Resolution: removed the FA4 experiment. Match SGLang 0.5.15's documented FA3 support boundary (SM80/SM90 families) and fall back to the model's SDPA path on Blackwell. This changes SM120 from a runtime crash to a correct fallback.

### E-008: Temporary test indentation damage during mechanical patching

- Symptom: a local `py_compile` immediately reported malformed indentation in the Ming test fake while adapting CUDA-graph lifecycle methods.
- Root cause: an intermediate patch inserted the fake method at the wrong nesting level.
- Resolution: corrected the indentation before remote sync; local `py_compile` and later pytest runs passed.

### E-009: FlashInfer JIT could not find `ninja` with an incomplete environment PATH

- Symptom: an early full-suite run reported six Higgs CUDA-graph test failures while importing FlashInfer JIT helpers.
- Root cause: the command invoked the isolated environment's Python by absolute path but did not prepend that environment's `bin` directory, so the installed `ninja` executable was not discoverable.
- Resolution: all subsequent commands use `PATH="$ENV/bin:$PATH"` in addition to the absolute Python path. The six focused tests and both full suites then passed. No source behavior was bypassed.

### E-010: Historical MOSS benchmark config contained experimental factory arguments

- Symptom: the first bump-only MOSS server launch rejected `enable_cross_request_encoder_batching` as an unexpected factory argument.
- Root cause: the seed config came from an unrelated experimental cross-request batching worktree rather than the current public `main` factory contract.
- Resolution: regenerated the three benchmark configs from current-main arguments and removed all cross-request/coalescing experiment fields. The A/B matrix now differs only in source revision and the explicit prefill-CUDA-graph switch.

### E-011: A stale failed server caused automatic port fallback

- Symptom: one diagnostic restart logged that port 18100 was occupied and silently bound to 34189; the benchmark still targeted 18100 and therefore exercised the previous failed process. The resulting 16 failures were discarded.
- Root cause: per-batch failures do not terminate the stage, and terminating only the parent CLI process did not immediately reap the spawned GPU stage process.
- Resolution: before every subsequent variant, verify the exact bound port in the server log, enumerate both parent and child PIDs, confirm the GPU process list is empty, and then require stable health on the requested port. Stale parent/resource-tracker/stage processes were terminated before restarting.

## SGLang 0.5.15 compatibility defects found and fixed

### C-001: Req finish-state API rename

- Failure: FishAudio test failed after 106 passes; production dLLM scheduler and the test still called `Req.check_finished()`.
- Cause: SGLang 0.5.15 renamed the state transition to `Req.update_finish_state()`.
- Fix: update both call sites. Focused rerun: 70 passed.

### C-002: CUDA graph runner lifecycle changed

- Failure: Ming TP test fake lacked `init_cuda_graphs`; later MOSS-TTS-Local test fake lacked phase-aware `cuda_graph_config` and runner initialization.
- Cause: the new ModelRunner initializes graph runners explicitly and indexes configuration by prefill/decode phase.
- Fix: update the fakes to model the new lifecycle and assert initialization calls; focused reruns passed.

### C-003: Scheduler parallel-state contract changed

- Failure: upstream idle/flush logic accessed `self.ps.pp_size`; the composed `OmniScheduler` only carried legacy scalar rank fields. Its existing `compute_dp_attention_world_info` unpack also expected three values although 0.5.15 returns four.
- Fix: build the real immutable `ParallelState`, retain all attention TP/CP/DP ranks, and refresh it before binding process groups.

### C-004: Scheduler metrics initialization was split

- Failure: `OmniScheduler` still called removed `init_metrics()`, while upstream flush now requires `metrics_reporter.reset_metrics()`.
- Fix: call `init_metrics_collector()` followed by `init_metrics_reporter()` and update the scheduler test doubles.

### C-005: Decode sequence-length sum became lazy

- Failure: Qwen3-Omni talker rollback required an integer `seq_lens_sum`, but SGLang 0.5.15 sets it to `None` after `prepare_for_decode()` and recomputes it in `ForwardBatch.init_new()`.
- Fix: preserve the new `None` invariant during rollback while still undoing allocation, request counters, tensor lengths, and req-to-token writes. The contract test now invokes the real upstream method.

### C-006: MOSS-TTS Local selected an unusable FA3 fallback on Blackwell

- Failure: SGLang's FA3 wrapper falls back to top-level FA2 outside SM80/SM90, but the Blackwell environment only contains the CUTE namespace and no FA2 `flash_attn_varlen_func`.
- Fix: gate the packed path to the architectures SGLang 0.5.15 actually supports and otherwise use the existing dense SDPA path.

### C-007: URL security validation happened after transport construction

- Failure: see E-006.
- Fix: establish the security boundary before any transport setup or network operation, for sync and async paths.

### C-008: Composed OmniScheduler did not initialize the SGLang 0.5.15 scheduling contract

- Failure: the first bump-only real-model server reached HTTP readiness, but its scheduler thread had already crashed on `AttributeError: _pending_chunked_abort_req`; the pipeline failure watcher terminated it seconds later.
- Cause: `OmniScheduler` delegates the 0.5.15 scheduling loop through composition while manually reproducing constructor state from the older Scheduler. The new loop requires chunk-abort state, a `NewTokenRatioTracker`, min-free-slot admission state, the IPC sender contract, and a DP-attention adapter.
- Fix: initialize the chunk-abort marker and min-free-slot gate, invoke the upstream 0.5.15 schedule-policy initializer, reset its tracker while idle, provide the new IPC channel contract, and construct the real `SchedulerDPAttnAdapter`. A regression test now executes one actual upstream idle scheduling iteration instead of checking fields only.
- Operational lesson: one immediate `/health` response is insufficient for a multiprocess pipeline. Readiness checks now require the endpoint to remain healthy for at least ten seconds while both coordinator and stage processes remain alive and the log has no scheduler-thread crash.

### C-009: Custom request builders violated the new Req token-storage type

- Failure: after C-008, the first 16 real MOSS requests crashed in `Req._refresh_fill_ids()` with `TypeError: can only concatenate list (not "array.array") to list`; the attempted 800-sample round consequently returned 800 failures and was discarded.
- Cause: SGLang 0.5.15 changed `Req.output_ids` and `full_untruncated_fill_ids` to signed-64-bit `array.array` storage and declares the same contract for `origin_input_ids`. Omni's model-specific builders still pass Python lists, so the new append-only refresh path cannot concatenate them.
- Fix: normalize both padded and unpadded prompt storage to `array("q")` once at the shared request-builder/scheduler boundary, preserving aliasing when both fields originally reference the same sequence. This covers all custom AR model builders without duplicating conversion in every model.
- Validation: a focused contract test verifies typecode, alias preservation, and concatenation with an upstream-style output array; all later live MOSS and Higgs requests used the normalized storage.

### C-010: Scheduler components split out of the old metrics initializer were missing

- Failure: the next live warmup failed on missing `load_inquirer`; the same 0.5.15 result path also requires a pool-stats observer, batch-result processor, output-streamer contract, KV-event publisher, and load-snapshot state.
- Cause: replacing removed `init_metrics()` with only `init_metrics_collector()` and `init_metrics_reporter()` covered construction but not the components now consumed by prefill accounting and result processing.
- Fix: initialize the real upstream pool observer, load inquirer, and batch-result processor in dependency order. Route its output-streamer calls back to Omni's stage outbox and use an explicit no-op KV publisher because this scheduler disables KV events.
- Validation: the scheduler construction test now asserts all three real components exist; live requests advanced past scheduling and into model forward.

### C-011: ScheduleBatch no longer exposes get_model_worker_batch

- Failure: after C-010, every forward failed on `ScheduleBatch.get_model_worker_batch()`.
- Cause: 0.5.15 moved the conversion into `ForwardBatch.init_new(ScheduleBatch, ModelRunner)` and made capture-hidden overrides one-shot fields on `ScheduleBatch` itself.
- Fix: pass `ScheduleBatch` directly in the shared AR model runner and dLLM scheduler, move capture-hidden overrides to the schedule batch, and remove the obsolete intermediate object from synchronous and asynchronous bookkeeping.
- Validation: 56 focused model-runner, async-decode, Qwen3-TTS, Voxtral, scheduler, and FishAudio tests passed.

### C-012: Custom run_batch bypassed deferred forward-input materialization

- Failure: the 0.5.15 eager registry tried to copy a 120-3195-token source into a zero-length destination because `ForwardBatch.input_ids` remained `None`.
- Cause: 0.5.15 keeps prefill IDs in pinned CPU `ScheduleBatch.prefill_input_ids_cpu` and materializes them at forward entry through `resolve_forward_inputs()`. Decode and mixed-chunk inputs are relayed through the always-on `FutureMap`. Omni overrides `run_batch`, so it bypassed both the materialization and relay steps.
- Fix: initialize the upstream overlap/FutureMap state, resolve forward inputs before every custom sync or async launch, publish sampled tokens back to the FutureMap, and clear the consumed schedule-batch input exactly as upstream does. No eager-copy fallback was enabled.
- Validation: 36 focused relay/model-runner/scheduler tests passed; a 16-request MOSS warmup completed 16/16, followed by three 800-request rounds with zero failures.

## Test matrix and results

### Environment contract

- Command: new-environment import/version probe for SGLang, Transformers, `PrefillCudaGraphRunner`, tc-piecewise context, Higgs multimodal input type, and `OmniServerArgs` MRO.
- Dataset: none.
- Raw input location: none.
- Result: passed; SGLang 0.5.15 and Transformers 5.12.1 loaded from the isolated environment.

### Targeted bump and PCG unit tests

- Command: `python -m pytest -q tests/unit_test/scheduling tests/unit_test/server_args tests/unit_test/model_runner/test_cuda_graph_batch_validator.py tests/unit_test/moss_transcribe_diarize tests/unit_test/higgs_tts ...` (exact log: `logs/pytest-targeted-bump-pcg-rerun.log`).
- Dataset: synthetic unit-test fixtures.
- Raw input location: repository test fixtures under `tests/unit_test/`.
- Result: 136 passed, 18 warnings.

### Bump-only full suite, discovery run

- Command: `python -m pytest tests/ -q -m "not benchmark"`.
- Dataset: repository unit/integration fixtures; no external benchmark corpus.
- Raw input location: `tests/`.
- Initial result: 1907 passed, 30 skipped, 42 deselected, 6 failed. The six failures produced C-003/C-005/C-006/C-007; all are covered by focused reruns.

### Focused compatibility rerun after fixes

- Command: exact ten-node pytest invocation recorded in `logs/pytest-six-failure-fixes-r3.log`.
- Dataset: synthetic unit-test fixtures.
- Raw input location: the named files under `tests/unit_test/`.
- Result: 8 passed, 2 skipped. The two numeric FA3 tests are intentionally skipped on SM120; the architecture contract test passed and proves the SDPA fallback selection.

### Final bump-only full suite

- Command: `PATH="$ENV/bin:$PATH" "$ENV/bin/python" -m pytest tests/ -q -m "not benchmark"`.
- Dataset: repository unit/integration fixtures; no external benchmark corpus.
- Raw input location: `tests/`.
- Result: 1911 passed, 32 skipped, 42 deselected, 26 warnings in 37.00 seconds.

### Final target full suite before live-model testing

- Command: `PATH="$ENV/bin:$PATH" "$ENV/bin/python" -m pytest tests/ -q -m "not benchmark"`.
- Dataset: repository unit/integration fixtures; no external benchmark corpus.
- Raw input location: `tests/`.
- Result: 1924 passed, 32 skipped, 42 deselected, 26 warnings in 39.47 seconds.

### Scheduler 0.5.15 idle-loop contract regression

- Command: `PATH="$ENV/bin:$PATH" PYTHONPATH="$BUMP_SRC" "$ENV/bin/python" -m pytest tests/unit_test/pipeline/test_scheduler.py::test_omni_scheduler_initializes_upstream_queue_limit -q`.
- Dataset: synthetic scheduler state.
- Raw input location: `tests/unit_test/pipeline/test_scheduler.py`.
- Result: 1 passed in 8.16 seconds; the test executes the real upstream `get_next_batch_to_run()` idle path.

### Scheduler 0.5.15 request-storage contract regression

- Command: `PATH="$ENV/bin:$PATH" PYTHONPATH="$BUMP_SRC" "$ENV/bin/python" -m pytest tests/unit_test/pipeline/test_scheduler.py::test_omni_scheduler_initializes_upstream_queue_limit tests/unit_test/pipeline/test_scheduler.py::test_omni_scheduler_normalizes_req_token_storage_for_sglang -q`.
- Dataset: synthetic scheduler and SGLang Req token-storage state.
- Raw input location: `tests/unit_test/pipeline/test_scheduler.py`.
- Result: 2 passed in 8.08 seconds.

## Final completion addendum (authoritative)

The earlier full-suite counts above are intermediate checkpoints. The final
source bytes include all fixes below plus pre-commit formatting; the final target
suite is recorded under "Final verification after Review".

### C-013: Every model-specific prefill path still read removed `Req.extend_input_len`

- Severity: P0 correctness.
- Failure: real Higgs startup/warmup advanced past scheduler compatibility and then failed because SGLang 0.5.15 represents the active extend interval as `Req.extend_range`.
- Root cause: Fish, MOSS-TTS, MOSS-TTS-Local, Qwen3-TTS, Qwen3-Omni talker, Voxtral-TTS, Higgs, the shared output processor, and the prefill manager still read the removed scalar.
- Fix: require `req.extend_range` and read its `.length` at every model-specific forward boundary. Tests and fakes were updated to construct the real 0.5.15 contract.
- Validation: the final target suite exercises every migrated model family; live MOSS and Higgs runs completed without this API failure.

### C-014: Higgs bump-only warmup leaked dummy capture rows into sampler state

- Severity: P0 correctness.
- Failure: four synthetic capture rows were allocated during Higgs graph warmup and survived into the Python request/sampler dictionaries.
- Root cause: the old Higgs prefill sampler ran inside `model.forward()`, so CUDA-graph warmup executed request-state mutation without real requests.
- Fix: the target implementation moves all request-aware prefill sampling to `HiggsTTSModelRunner.post_prefill()`. The bump-only diagnostic tree received a narrow dummy-prefill guard so it could serve as a valid version-only control; that diagnostic workaround is not part of the target design.
- Validation: the final target has no capture-time request IDs or sampler-pool allocation; 3,264 final SeedTTS requests completed with stable sampler state.

### C-015: Scheduler thread ran with grad enabled after capture

- Severity: P0 correctness and P1 performance.
- Failure: Higgs PCG capture completed under no-grad, but the first runtime forward recompiled and eventually raised `PCG capture stream is not set`.
- Root cause: SGLang's scheduler entry point is decorated by `DynamicGradMode`; Omni's composed schedulers bypassed that entry point and started their custom loops with the ambient thread grad mode.
- Fix: decorate both `OmniScheduler.start()` and `DllmScheduler.start()` with SGLang's `DynamicGradMode`.
- Validation: live servers no longer recompile the captured path; MOSS logged real prefill graph replay and Higgs completed the full graph-on matrix.

### C-016: Higgs partial Radix-cache branch produced stable prompt echo under PCG

- Severity: P0 correctness.
- Symptom: with seed 17, sample `common_voice_en_23718813-common_voice_en_23718814` generated the reference prompt itself (6.28 s audio, WER 116.67%) only with PCG enabled. The error repeated in all three full rounds.
- Initial hypothesis, falsified: a request-row/sampler-row mix-up. The wrong text was the current request's own reference transcript, not another batched request, and a concurrency-1 reproducer still failed.
- Isolation sequence:
  1. PCG off: correct 4.24 s output.
  2. PCG on, cold 205-token prompt: correct and graph replayed.
  3. PCG on, exact cache hit (`204 + 1` replay): correct and graph replayed.
  4. PCG on, branched partial cache hit (`189 + 16`): prompt echo and graph replayed.
- Root cause boundary: only a non-empty cached prefix extended by more than the one-token exact-hit replay was unsafe for Higgs' seeded first codec draw through the tc-piecewise Qwen3 path. The numerical shift crossed a sampling boundary and then autoregressively amplified.
- Rejected alternative: forcing the SGLang `breakable` backend captured the inner Qwen layer and bypassed Higgs' outer placeholder-to-reference-embedding preparation; the first request hit a device-side gather assertion. This backend was not retained.
- Fix: `HiggsTTSModelRunner.before_prefill()` materializes correct full input embeddings and clears `mm_inputs` only when any row has `prefix_len > 0 && extend_len > 1`. SGLang's `PrefillCudaGraphRunner.can_run_graph()` then selects eager for that batch. Cold prompts and exact full-prompt hits remain graph eligible.
- Minimal-reproducer validation: graph decisions became cold=true, exact-hit=true, branched=false; the failing sample returned the correct transcript at 4.40 s.
- Full validation: final PCG-on WER averaged 0.8931%, zero WER>50% outliers, and the original sample had WER 0. Prefill batches were 2,183 graph / 93 eager (95.9% graph eligibility, including the reproducer and three full rounds).

### C-017: Legal custom request builders may omit `origin_input_ids_unpadded`

- Severity: P0 correctness at the shared request boundary.
- Failure: nine unit tests failed after token-storage normalization assumed every custom `Req`-like object carried the optional unpadded alias.
- Root cause: SGLang's concrete `Req` has the field, but Omni unit/custom builders legitimately supply only `origin_input_ids` before scheduler admission.
- Fix: normalize `origin_input_ids` to signed-64-bit `array('q')`; treat a missing unpadded field as an alias to the same sequence and preserve aliasing after conversion.
- Validation: focused contract coverage plus the final target and bump-only full suites.

### C-018: Scheduler tests modeled obsolete batch handoff and finalizer signatures

- Severity: P2 test maintainability; observable as four failures after the production migration.
- Failure: fakes still expected direct `batch.input_ids` chaining and the removed `model_worker_batch` finalizer argument.
- Root cause: SGLang 0.5.15 relays sampled IDs through `FutureMap`, resolves forward inputs at launch, and constructs `ForwardBatch` directly from `ScheduleBatch`.
- Fix: tests now exercise the FutureMap publish/consume contract and the current finalizer signature rather than recreating removed private state.
- Validation: 92 focused scheduler/MOSS tests passed before the final full suite.

### E-012: Higgs benchmark initially used the wrong checkpoint path

- Symptom: the first launch pointed at a display/model alias that was not present on the data disk.
- Resolution: use the immutable local snapshot `/root/autodl-tmp/huggingface/hub/models--bosonai--higgs-tts-3-4b/snapshots/7556c17e05201fccd9c8cc120bc216dcc7b5d561` for every retained Higgs run.
- Disposition: no result from the failed launch was included.

### E-013: The SeedTTS driver did not accept attempted inline ASR server flags

- Symptom: an attempted combined generate/transcribe command rejected unsupported ASR-launch flags.
- Root cause: `benchmark_tts_seedtts.py` supports a generation server or a transcription-only pass against an already-running ASR server, not both server lifecycles in one invocation.
- Resolution: run generation with `--generate-only`, then launch Qwen3-ASR separately and run `--transcribe-only` on the saved audio.
- Disposition: no partial metric was retained.

### E-014: `breakable` PCG backend was incompatible with Higgs' outer multimodal wrapper

- Symptom: capture succeeded in 5.32 s and used 0.79 GB, but the first real request hit a device-side gather assertion.
- Root cause: the backend captured a Qwen inner layer boundary that did not include Higgs placeholder replacement.
- Resolution: reject this backend for Higgs and retain `tc_piecewise` plus the precise partial-prefix eager gate in C-016.

### E-015: Optional NIXL/Mooncake relay imports warned in both environments

- Symptom: server logs reported no importable `nixl` module and an old Conda `libstdc++` lacking `GLIBCXX_3.4.30` for Mooncake.
- Comparison: both the SGLang 0.5.12 baseline and 0.5.15 environment install `nixl-cu13` but cannot `import nixl`; the Conda library-path issue is host-environment state, not introduced by the bump.
- Scope: every retained run explicitly used the `shm` relay backend, so neither optional transport participated in correctness or performance measurements. No source fallback was added and no relay result is claimed.

### E-016: Dependency checker retains the repository's existing protobuf waiver

- New environment command: `/root/autodl-tmp/sglang-eval-lab/envs/omni-cu13-py312-sglang-0.5.15-20260713/bin/python -m pip check`.
- Baseline command: `/root/autodl-tmp/sglang-eval-lab/envs/omni-cu13-py312-sm120-20260706/bin/python -m pip check`.
- Both results: `descript-audiotools 0.7.2` declares `protobuf<3.20` while the repository forces protobuf 6.33.6. This is pre-existing and unchanged. The new grpc 1.82/protobuf 7 conflict was fixed by the 1.81.1 pins in E-004.

## Exact retained benchmark commands

All commands used the new environment below unless the variant explicitly says
baseline 0.5.12. The source path was switched among `main-baseline-20260712`,
`bump-only-20260713`, and `bump-pcg-full-20260713` without changing the workload.

```bash
ENV=/root/autodl-tmp/sglang-eval-lab/envs/omni-cu13-py312-sglang-0.5.15-20260713
SRC=/root/autodl-tmp/sglang-eval-lab/src/sglang-omni/bump-pcg-full-20260713
RUN=/root/autodl-tmp/sglang-eval-lab/runs/sglang-0.5.15-pcg-regression-20260713-v1
PATH="$ENV/bin:$PATH" PYTHONPATH="$SRC" \
  "$ENV/bin/sgl-omni" serve \
  --config "$RUN/manifest/configs/moss-full-pcg-on.json" \
  --host 127.0.0.1 --port 18100
```

MOSS warmup and each retained 800-sample round used the same driver; only
`NAME` changed (`moss-main-sglang-0.5.12-r1..r3`,
`moss-bump-only-sglang-0.5.15-final-r1..r3`, `moss-full-pcg-off-r1..r3`,
`moss-full-pcg-on-r1..r3`). The corresponding server config and source/env were
selected before each command.

```bash
NAME=moss-full-pcg-on-r1
PATH="$ENV/bin:$PATH" PYTHONPATH="$SRC" "$ENV/bin/python" \
  -m benchmarks.eval.benchmark_asr_transcribe_diarize \
  --dataset movies800times \
  --max-concurrency 16 \
  --use-existing-server \
  --base-url http://127.0.0.1:18100 \
  --model-path /root/autodl-tmp/models/OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --output-dir "$RUN/results/moss/$NAME" \
  --disable-tqdm
```

- Dataset: Movies800Time, 800 samples per round.
- Original raw data: `/root/autodl-tmp/huggingface/hub/datasets--zhaochenyang20--movies800time/snapshots/382168daa4e9d764318a20c1365fd0af0d00d785`.
- Raw per-request results: `$RUN/results/moss/$NAME/transcribe_diarize_asr_results.json`.
- Speed metrics: `$RUN/results/moss/$NAME/transcribe_diarize_speed_results.json`.

Higgs server:

```bash
PATH="$ENV/bin:$PATH" PYTHONPATH="$SRC" \
  "$ENV/bin/sgl-omni" serve \
  --config "$RUN/manifest/configs/higgs-full-pcg-on.json" \
  --host 127.0.0.1 --port 18081
```

Generation (each of the three retained rounds):

```bash
NAME=higgs-final-seed17-pcg-on-prefix-gate-r1
PATH="$ENV/bin:$PATH" PYTHONPATH="$SRC" "$ENV/bin/python" \
  "$SRC/benchmarks/eval/benchmark_tts_seedtts.py" \
  --use-existing-server --generate-only \
  --host 127.0.0.1 --port 18081 \
  --model bosonai/higgs-tts-3-4b \
  --meta /root/autodl-tmp/datasets/seed-tts-eval-arrow \
  --lang en --ref-format flat \
  --output-dir "$RUN/results/higgs/$NAME" \
  --concurrency 16 --disable-tqdm --seed 17
```

ASR server and transcription pass:

```bash
PATH="$ENV/bin:$PATH" PYTHONPATH="$SRC" "$ENV/bin/sgl-omni" serve \
  --model-path /root/autodl-tmp/models/Qwen3-ASR-1.7B \
  --host 127.0.0.1 --port 18081 \
  --allowed-local-media-path /tmp

PATH="$ENV/bin:$PATH" PYTHONPATH="$SRC" "$ENV/bin/python" \
  "$SRC/benchmarks/eval/benchmark_tts_seedtts.py" \
  --use-existing-server --transcribe-only \
  --host 127.0.0.1 --port 18081 \
  --model bosonai/higgs-tts-3-4b \
  --meta /root/autodl-tmp/datasets/seed-tts-eval-arrow \
  --lang en --ref-format flat \
  --output-dir "$RUN/results/higgs/$NAME" \
  --asr-model-path /root/autodl-tmp/models/Qwen3-ASR-1.7B \
  --asr-concurrency 24 --disable-tqdm --seed 17
```

- Dataset: SeedTTS English, 1,088 samples per round.
- Original raw data: `/root/autodl-tmp/datasets/seed-tts-eval-arrow/data/en-00000-of-00001.parquet`.
- Generated audio and JSON: `$RUN/results/higgs/$NAME/`.
- Minimal reproducer source: the same parquet, original rows 638 and 639; derived two-row file `/tmp/higgs-seedtts-pair/data/en-00000-of-00001.parquet`.
- The earliest unseeded exploratory audio directories were deleted when the data disk became tight; their logs and metric JSON were retained, and no unseeded number is used for the final gate.

## Performance and accuracy results

### MOSS-Transcribe-Diarize, 3 x 800 requests

| Variant | Avg QPS | Mean latency s | P95 s | CER no speaker | Delta QPS vs old |
|---|---:|---:|---:|---:|---:|
| main + SGLang 0.5.12.post1 | 25.465 | 0.548 | 1.147 | 21.643% | baseline |
| bump-only + SGLang 0.5.15 | 24.967 | 0.558 | 1.187 | 21.653% | -1.96% |
| target, prefill PCG off | 25.185 | 0.553 | 1.165 | 21.653% | -1.10% |
| target, prefill PCG on | 26.798 | 0.515 | 1.094 | 21.680% | +5.23% |

- Every retained variant completed 2,400/2,400 requests.
- Graph-on vs graph-off: +6.41% QPS and about 7% lower mean latency.
- Accuracy spread was about 0.03 percentage point, with no request failures.
- Prefill capture: 11.11 s, 0.38 GB; runtime logs confirmed the native prefill graph path.

### Higgs SeedTTS, 3 x 1,088 requests, seed 17

| Variant | Avg QPS | Mean latency s | P95 s | Codec tok/s | Avg WER | WER>50% |
|---|---:|---:|---:|---:|---:|---:|
| main + SGLang 0.5.12.post1 | 11.227 | 1.417 | 2.007 | 1254.0 | 0.9350% | 0 |
| bump-only + SGLang 0.5.15 | 10.796 | output length differed | not comparable | 1242.4 | 1.6160% | 4 |
| target, prefill PCG off | 11.030 | 1.442 | 2.050 | 1232.6 | 0.9127% | 0 |
| final target, PCG on + prefix gate | 11.308 | 1.405 | 2.016 | 1259.5 | 0.8931% | 0 |

- Final PCG-on vs target PCG-off: +2.52% QPS, +2.18% codec tok/s, -2.59% mean latency, -1.68% P95.
- Final PCG-on vs old main: +0.72% QPS, +0.44% codec tok/s, -0.87% mean latency, +0.45% P95. The P95 change is treated as noise, not an improvement claim.
- Final correctness: 3,264/3,264 generated and transcribed; no WER>50% outlier. The original prompt-echo sample is correct with WER 0.
- Bump-only output totals differed across rounds (125,305 / 124,904 / 125,432), so QPS is not a fair version comparison; codec tok/s is the primary performance normalization and was -0.93% versus old main.

## Final verification after Review

The retained GPU benchmark matrix was completed after C-016 and all runtime
compatibility fixes. The only later source edits were deterministic formatting,
replacement of defensive Higgs `getattr` calls with explicit 0.5.15 attributes,
and a startup-only assertion that SGLang's `model` alias is the existing Qwen
backbone. They do not change GPU math, batching, graph eligibility, or sampler
state. The full target suite below was rerun on the exact final commit bytes.

Final full target command:

```bash
PATH="$ENV/bin:$PATH" \
PYTHONPATH=/root/autodl-tmp/sglang-eval-lab/src/sglang-omni/bump-pcg-full-20260713 \
"$ENV/bin/python" -m pytest tests/ -q -m "not benchmark" -p no:cacheprovider
```

- Dataset: repository unit/integration fixtures; no external benchmark corpus.
- Original raw input: `/root/autodl-tmp/sglang-eval-lab/src/sglang-omni/bump-pcg-full-20260713/tests/`.
- Result: 1,929 passed, 32 skipped, 42 deselected, 26 warnings in 37.59 s.
- Raw log: `$RUN/logs/pytest-target-final-r3.log`.

Final bump-only command was identical with `PYTHONPATH` and working directory set
to `/root/autodl-tmp/sglang-eval-lab/src/sglang-omni/bump-only-20260713`.

- Dataset: repository unit/integration fixtures; no external benchmark corpus.
- Original raw input: `/root/autodl-tmp/sglang-eval-lab/src/sglang-omni/bump-only-20260713/tests/`.
- Result: 1,914 passed, 32 skipped, 42 deselected, 26 warnings in 37.52 s.
- Raw log: `$RUN/logs/pytest-bump-only-final.log`.

Final static validation:

- `git diff --check`: passed.
- `rg -n '\.extend_input_len|\.check_finished\(|get_model_worker_batch|\.init_metrics\(' sglang_omni tests`: no obsolete call sites.
- `pre-commit run --all-files`: first pass formatted seven files; after Review edits Black formatted one file; third pass passed every hook without modifying files.

## Human-maintainer and model-history review note

The review corpus sweep covered all touched paths and then widened to the
upstream scheduler, CUDA-graph, multimodal, cache, and model-runner surfaces.
The recurring maintainer requirements were: gate experimental PCG support per
model, preserve cache-branch semantics, test real accuracy and performance, and
avoid treating a capture success as correctness evidence. Those requirements
directly caused the SeedTTS WER gate, the cache-branch reproducer, and rejection
of the `breakable` backend.

Model-history references consulted:

- `sglang/qwen3-core/README.en.md`: Qwen3 runner/attention compatibility and weight-layout risk.
- `sglang/qwen-vlm-omni-asr/README.en.md`: PR #13055 (piecewise multimodal PCG plus nightly accuracy), PR #15320 (ViT PCG), PR #22038 (cache-aware multimodal behavior), and PR #22073 (Qwen3-ASR accuracy/performance validation).

Resulting decision: keep Omni support capability-gated to MOSS-TD and Higgs,
retain every upstream compatibility rule, add model-specific correctness tests,
and require real dataset A/B evidence before enabling by default.

## Final review disposition

- P0 correctness: all discovered blockers C-001 through C-018 are fixed; no unresolved P0 finding remains.
- P1 performance: no measured regression on the normalized metrics. MOSS PCG improves QPS; Higgs final PCG-on is neutral-to-positive versus both target-off and old main.
- P2 maintainability: SGLang-version adaptation is centralized in phase-aware graph policy, scheduler construction, and request-storage normalization. Higgs' `model` alias now fails fast unless SGLang assigns the existing Qwen backbone.
- P3 style: final pre-commit passed.
- P4 process: unit coverage and copy-pasteable regression commands are recorded above. GPU validation is limited to one RTX PRO 6000 Blackwell; H100/A100 coverage remains for CI rather than being claimed here.
