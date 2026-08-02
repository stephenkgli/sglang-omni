# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from benchmarks.eval.verify_higgs_tts_full_pipeline_mps import (
    expected_gpu_processes,
    parse_stage_group_pids,
    validate_full_pipeline_attachment,
)

_PIPELINE_CONFIG = {
    "stages": [
        {
            "name": "preprocessing",
            "process": "tts_frontend",
            "gpu": None,
            "num_replicas": 2,
            "tp_size": 1,
        },
        {
            "name": "audio_encoder",
            "process": "tts_frontend",
            "gpu": 0,
            "num_replicas": 2,
            "tp_size": 1,
        },
        {
            "name": "tts_engine",
            "process": "pipeline",
            "gpu": 0,
            "num_replicas": 2,
            "tp_size": 1,
        },
        {
            "name": "vocoder",
            "process": "vocoder",
            "gpu": 0,
            "num_replicas": 1,
            "tp_size": 1,
        },
    ]
}

_SERVER_LOG = """
INFO StageGroup tts_frontend@r0: spawned 1 process(es) (pids=[101])
INFO StageGroup tts_frontend@r1: spawned 1 process(es) (pids=[102])
INFO StageGroup pipeline@r0: spawned 1 process(es) (pids=[201])
INFO StageGroup pipeline@r1: spawned 1 process(es) (pids=[202])
INFO StageGroup vocoder: spawned 1 process(es) (pids=[301])
"""


def test_parse_stage_group_pids_rejects_inconsistent_count() -> None:
    with pytest.raises(ValueError, match="declared 2 process"):
        parse_stage_group_pids(
            "StageGroup pipeline@r0: spawned 2 process(es) (pids=[201])"
        )


def test_expected_gpu_processes_expands_replicas_and_skips_cpu_stage() -> None:
    assert expected_gpu_processes(_PIPELINE_CONFIG) == {
        "pipeline@r0": ["tts_engine@r0"],
        "pipeline@r1": ["tts_engine@r1"],
        "tts_frontend@r0": ["audio_encoder@r0"],
        "tts_frontend@r1": ["audio_encoder@r1"],
        "vocoder": ["vocoder"],
    }


def test_expected_gpu_processes_deduplicates_colocated_stages() -> None:
    config = {
        "stages": [
            {
                "name": "tts_engine",
                "process": "pipeline",
                "gpu": 0,
                "num_replicas": 1,
                "tp_size": 1,
            },
            {
                "name": "vocoder",
                "process": "pipeline",
                "gpu": 0,
                "num_replicas": 1,
                "tp_size": 1,
            },
        ]
    }

    assert expected_gpu_processes(config) == {"pipeline": ["tts_engine", "vocoder"]}


def test_full_pipeline_mps_attachment_accepts_every_gpu_process() -> None:
    result = validate_full_pipeline_attachment(
        pipeline_config=_PIPELINE_CONFIG,
        server_log=_SERVER_LOG,
        server_clients={9001: [101, 102, 201, 202, 301]},
        pid_is_live=lambda _pid: True,
    )

    assert result["exact_full_pipeline_attachment"] is True
    assert result["mps_client_pids"] == [101, 102, 201, 202, 301]
    assert result["violations"] == []


@pytest.mark.parametrize("missing_pid", [102, 201, 301])
def test_full_pipeline_mps_attachment_rejects_missing_gpu_process(
    missing_pid: int,
) -> None:
    client_pids = [101, 102, 201, 202, 301]
    client_pids.remove(missing_pid)
    result = validate_full_pipeline_attachment(
        pipeline_config=_PIPELINE_CONFIG,
        server_log=_SERVER_LOG,
        server_clients={9001: client_pids},
        pid_is_live=lambda _pid: True,
    )

    assert result["exact_full_pipeline_attachment"] is False
    assert result["violations"] == [
        f"configured GPU processes missing from MPS clients: [{missing_pid}]"
    ]


def test_full_pipeline_mps_attachment_rejects_cpu_only_coordinator() -> None:
    result = validate_full_pipeline_attachment(
        pipeline_config=_PIPELINE_CONFIG,
        server_log=_SERVER_LOG,
        server_clients={9001: [101, 102, 201, 202, 301, 401]},
        pid_is_live=lambda _pid: True,
    )

    assert result["exact_full_pipeline_attachment"] is False
    assert result["violations"] == [
        "processes outside the configured GPU process set attached to MPS: [401]"
    ]
