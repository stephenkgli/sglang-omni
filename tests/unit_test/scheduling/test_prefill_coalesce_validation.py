# SPDX-License-Identifier: Apache-2.0

import pytest

from sglang_omni.scheduling.prefill_coalesce import validate_prefill_coalesce_args


def test_validate_prefill_coalesce_args_normalizes_values():
    assert validate_prefill_coalesce_args(32, 300) == (32, 300.0)
    assert validate_prefill_coalesce_args(None, None) == (None, None)


@pytest.mark.parametrize(
    ("requests", "wait_ms"),
    [
        (-1, 60.0),
        (0, 0.0),
        (0, float("nan")),
        (0, float("inf")),
        # bools and non-integral counts must fail before int()/float()
        # truncation can silently change their meaning
        (True, 60.0),
        (False, 60.0),
        (0, True),
        (-0.5, 60.0),
        (2.9, 60.0),
        # int(inf) raises OverflowError, which is not a ValueError, so callers
        # wrapping this in typer.BadParameter would leak a traceback instead
        (float("inf"), 60.0),
        (float("-inf"), 60.0),
        ("32", 60.0),
        (0, "300"),
    ],
)
def test_validate_prefill_coalesce_args_rejects_invalid_values(requests, wait_ms):
    with pytest.raises(ValueError):
        validate_prefill_coalesce_args(requests, wait_ms)


def test_validate_prefill_coalesce_args_accepts_integral_floats():
    # YAML often parses `32` as an int, but `32.0` should also work since it
    # loses nothing under int().
    assert validate_prefill_coalesce_args(32.0, 300) == (32, 300.0)


def test_validate_prefill_coalesce_args_warns_on_one(caplog):
    # 1 passes validation but the gate only engages at >= 2, so it must warn
    # the user that coalescing stays disabled; 0 stays silent (explicit off).
    with caplog.at_level("WARNING", logger="sglang_omni.scheduling.prefill_coalesce"):
        assert validate_prefill_coalesce_args(1, 60.0) == (1, 60.0)
    assert any("disables coalescing" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level("WARNING", logger="sglang_omni.scheduling.prefill_coalesce"):
        assert validate_prefill_coalesce_args(0, 60.0) == (0, 60.0)
    assert not caplog.records
