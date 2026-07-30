# SPDX-License-Identifier: Apache-2.0
"""SchedulingConfig validation: types and ranges are enforced at the schema
layer, before int()/float() truncation can silently change meaning."""

from __future__ import annotations

import pytest

from sglang_omni.config.schema import SchedulingConfig


def test_normalizes_values():
    cfg = SchedulingConfig(prefill_coalesce_requests=32, prefill_coalesce_wait_ms=300)
    assert cfg.prefill_coalesce_requests == 32
    assert cfg.prefill_coalesce_wait_ms == 300.0
    assert SchedulingConfig().prefill_coalesce_requests is None


def test_accepts_integral_floats():
    # Note (maydomine): YAML round-tripping may produce a lossless 32.0.
    cfg = SchedulingConfig(prefill_coalesce_requests=32.0)
    assert cfg.prefill_coalesce_requests == 32


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prefill_coalesce_requests": None},
        {"prefill_coalesce_wait_ms": None},
        {"prefill_coalesce_requests": -1},
        {"prefill_coalesce_wait_ms": 0.0},
        {"prefill_coalesce_wait_ms": float("nan")},
        {"prefill_coalesce_wait_ms": float("inf")},
        # Note (maydomine): These values would otherwise coerce into valid but
        # unintended settings.
        {"prefill_coalesce_requests": True},
        {"prefill_coalesce_requests": False},
        {"prefill_coalesce_wait_ms": True},
        {"prefill_coalesce_requests": -0.5},
        {"prefill_coalesce_requests": 2.9},
        {"prefill_coalesce_requests": float("inf")},
        {"prefill_coalesce_requests": float("-inf")},
        {"prefill_coalesce_requests": "32"},
        {"prefill_coalesce_wait_ms": "300"},
    ],
)
def test_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        SchedulingConfig(**kwargs)


def test_warns_on_one(caplog):
    # Note (maydomine): Warn on the ambiguous off value 1, while explicit 0
    # remains silent.
    with caplog.at_level("WARNING", logger="sglang_omni.config.schema"):
        assert (
            SchedulingConfig(prefill_coalesce_requests=1).prefill_coalesce_requests == 1
        )
    assert any("disables coalescing" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level("WARNING", logger="sglang_omni.config.schema"):
        assert (
            SchedulingConfig(prefill_coalesce_requests=0).prefill_coalesce_requests == 0
        )
    assert not caplog.records
