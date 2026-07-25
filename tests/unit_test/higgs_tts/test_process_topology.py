# SPDX-License-Identifier: Apache-2.0

from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig


def test_higgs_tts_engine_projects_only_vocoder_payload() -> None:
    config = HiggsTtsPipelineConfig(model_path="fake-model")
    tts_engine = next(stage for stage in config.stages if stage.name == "tts_engine")

    assert tts_engine.project_payload == {
        "vocoder": (
            "sglang_omni.models.higgs_tts.request_builders."
            "project_tts_engine_to_vocoder"
        )
    }
