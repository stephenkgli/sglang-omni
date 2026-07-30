# SPDX-License-Identifier: Apache-2.0
"""CLI override tests for prefill admission coalescing.

``--prefill-coalesce-requests/--prefill-coalesce-wait-ms`` write the typed
``runtime.scheduling`` block of stages whose factories declare support.
"""

from __future__ import annotations

import pytest
import typer

from sglang_omni.cli.serve import apply_prefill_coalesce_cli_overrides
from sglang_omni.config import PipelineConfig, SchedulingConfig
from sglang_omni.config.runtime import (
    resolve_factory_signature_args,
    resolve_stage_factory_arg_defaults,
    resolve_stage_static_factory_args,
)
from sglang_omni.models.fun_asr.config import FunASRPipelineConfig
from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
from sglang_omni.models.moss_transcribe_diarize.config import (
    MossTranscribeDiarizePipelineConfig,
)
from sglang_omni.models.moss_tts_local.config import MossTTSLocalPipelineConfig
from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig
from sglang_omni.utils.imports import import_string


def _ar_stage_args(config: PipelineConfig, stage_name: str) -> dict[str, object]:
    stage = next(s for s in config.stages if s.name == stage_name)
    return resolve_factory_signature_args(
        import_string(stage.factory),
        resolve_stage_static_factory_args(stage, config),
        defaults=resolve_stage_factory_arg_defaults(stage, config),
    )


@pytest.mark.parametrize(
    ("config_cls", "stage_name"),
    [
        (HiggsTtsPipelineConfig, "tts_engine"),
        (MossTTSLocalPipelineConfig, "tts_engine"),
        (MossTranscribeDiarizePipelineConfig, "asr"),
        (FunASRPipelineConfig, "asr"),
        (Qwen3OmniPipelineConfig, "thinker"),
    ],
)
def test_cli_sets_coalesce_args(config_cls, stage_name):
    config = config_cls(model_path="dummy")
    apply_prefill_coalesce_cli_overrides(
        config, prefill_coalesce_requests=32, prefill_coalesce_wait_ms=300.0
    )
    stage = next(s for s in config.stages if s.name == stage_name)
    assert stage.runtime.scheduling.prefill_coalesce_requests == 32
    assert stage.runtime.scheduling.prefill_coalesce_wait_ms == 300.0
    args = _ar_stage_args(config, stage_name)
    assert args["prefill_coalesce_requests"] == 32
    assert args["prefill_coalesce_wait_ms"] == 300.0


def test_typed_scheduling_field_reaches_factory_args():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    stage = next(s for s in config.stages if s.name == "tts_engine")
    stage.runtime.scheduling = SchedulingConfig(prefill_coalesce_requests=16)
    args = _ar_stage_args(config, "tts_engine")
    assert args["prefill_coalesce_requests"] == 16
    # Note (maydomine): An omitted typed value must preserve the factory default.
    assert (
        "prefill_coalesce_wait_ms" not in args
        or args["prefill_coalesce_wait_ms"] == 60.0
    )


def test_omitting_both_flags_is_a_noop():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    before = _ar_stage_args(config, "tts_engine")
    apply_prefill_coalesce_cli_overrides(
        config, prefill_coalesce_requests=None, prefill_coalesce_wait_ms=None
    )
    assert _ar_stage_args(config, "tts_engine") == before


def test_rejects_pipeline_without_supporting_factory():
    # Note (maydomine): Reject unsupported factories instead of silently
    # accepting ineffective CLI flags.
    from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig

    config = Qwen3TTSPipelineConfig(model_path="dummy")
    with pytest.raises(typer.BadParameter):
        apply_prefill_coalesce_cli_overrides(
            config, prefill_coalesce_requests=32, prefill_coalesce_wait_ms=None
        )


def test_cli_discovers_legacy_full_stage_yaml():
    dumped = HiggsTtsPipelineConfig(model_path="dummy").model_dump()
    tts_stage = next(
        stage for stage in dumped["stages"] if stage["name"] == "tts_engine"
    )
    tts_stage["runtime"].pop("scheduling")
    config = HiggsTtsPipelineConfig(**dumped)

    stage = next(stage for stage in config.stages if stage.name == "tts_engine")
    assert stage.runtime.scheduling is None
    apply_prefill_coalesce_cli_overrides(
        config,
        prefill_coalesce_requests=32,
        prefill_coalesce_wait_ms=None,
    )
    assert stage.runtime.scheduling.prefill_coalesce_requests == 32


def test_rejects_invalid_values():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    with pytest.raises(typer.BadParameter):
        apply_prefill_coalesce_cli_overrides(
            config, prefill_coalesce_requests=-1, prefill_coalesce_wait_ms=None
        )
    for bad_wait in (0.0, float("nan"), float("inf")):
        with pytest.raises(typer.BadParameter):
            apply_prefill_coalesce_cli_overrides(
                config,
                prefill_coalesce_requests=None,
                prefill_coalesce_wait_ms=bad_wait,
            )


def test_legacy_factory_args_are_validated(caplog):
    # Note (maydomine): Preserve #1071 configs while directing users to the
    # typed replacement.
    config = HiggsTtsPipelineConfig(model_path="dummy")
    stage = next(s for s in config.stages if s.name == "tts_engine")
    stage.factory_args = {
        **(stage.factory_args or {}),
        "prefill_coalesce_requests": 32.0,
    }
    with caplog.at_level("WARNING", logger="sglang_omni.config.runtime"):
        args = _ar_stage_args(config, "tts_engine")
    assert args["prefill_coalesce_requests"] == 32
    assert any("deprecated" in record.message for record in caplog.records)


def test_wait_only_cli_respects_legacy_request_count(caplog):
    config = HiggsTtsPipelineConfig(model_path="dummy")
    stage = next(s for s in config.stages if s.name == "tts_engine")
    stage.factory_args = {
        **(stage.factory_args or {}),
        "prefill_coalesce_requests": 32,
    }
    with caplog.at_level("WARNING", logger="sglang_omni.cli.serve"):
        apply_prefill_coalesce_cli_overrides(
            config,
            prefill_coalesce_requests=None,
            prefill_coalesce_wait_ms=300,
        )
    assert not any(
        "alone does not enable" in record.message for record in caplog.records
    )


@pytest.mark.parametrize("legacy_source", ["factory_args", "runtime_overrides"])
def test_cli_value_overrides_legacy_source(legacy_source):
    config = HiggsTtsPipelineConfig(model_path="dummy")
    stage = next(s for s in config.stages if s.name == "tts_engine")
    if legacy_source == "factory_args":
        stage.factory_args["prefill_coalesce_requests"] = 32
    else:
        config.runtime_overrides[stage.name] = {"prefill_coalesce_requests": 32}

    apply_prefill_coalesce_cli_overrides(
        config,
        prefill_coalesce_requests=64,
        prefill_coalesce_wait_ms=None,
    )

    assert _ar_stage_args(config, "tts_engine")["prefill_coalesce_requests"] == 64


@pytest.mark.parametrize("legacy_source", ["factory_args", "runtime_overrides"])
def test_legacy_explicit_null_is_rejected(legacy_source):
    config = HiggsTtsPipelineConfig(model_path="dummy")
    stage = next(s for s in config.stages if s.name == "tts_engine")
    if legacy_source == "factory_args":
        stage.factory_args["prefill_coalesce_requests"] = None
    else:
        config.runtime_overrides[stage.name] = {"prefill_coalesce_requests": None}

    with pytest.raises(ValueError, match="cannot be null"):
        _ar_stage_args(config, "tts_engine")


def test_typed_and_legacy_values_conflict():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    stage = next(s for s in config.stages if s.name == "tts_engine")
    stage.runtime.scheduling = SchedulingConfig(prefill_coalesce_requests=16)
    stage.factory_args = {
        **(stage.factory_args or {}),
        "prefill_coalesce_requests": 32,
    }
    with pytest.raises(ValueError, match="through both"):
        _ar_stage_args(config, "tts_engine")


def test_typed_scheduling_rejects_unsupported_factory():
    from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig

    config = Qwen3TTSPipelineConfig(model_path="dummy")
    stage = next(s for s in config.stages if s.name == "tts_engine")
    stage.runtime.scheduling = SchedulingConfig(prefill_coalesce_requests=16)
    with pytest.raises(ValueError, match="does not support"):
        _ar_stage_args(config, "tts_engine")
