# SPDX-License-Identifier: Apache-2.0
"""Shared validation for prefill admission coalescing."""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


def validate_prefill_coalesce_args(
    prefill_coalesce_requests: int | float | None,
    prefill_coalesce_wait_ms: float | None,
) -> tuple[int | None, float | None]:
    """Validate and normalize coalescing arguments from any config entrypoint.

    Both 0 and 1 leave the admission gate disabled: the gate only engages at
    >= 2, since a batch of one has nothing to coalesce with. 0 is the explicit
    off switch; 1 is accepted but warns, as it is usually a misconfiguration.
    """
    requests = _normalize_requests(prefill_coalesce_requests)
    if requests == 1:
        logger.warning(
            "prefill_coalesce_requests=1 disables coalescing: the admission "
            "gate only engages at >= 2 (a batch of one has nothing to "
            "coalesce with). Use 0 to disable explicitly, or >= 2 to enable."
        )
    return requests, _normalize_wait_ms(prefill_coalesce_wait_ms)


def _normalize_requests(value: int | float | None) -> int | None:
    if value is None:
        return None
    # bool is an int subclass, so `true` would otherwise mean 1.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"prefill_coalesce_requests must be a number, got {value!r}")
    # Reject non-integral and non-finite counts before int() changes their
    # meaning: YAML `-0.5` would land as 0 (silently off) and `2.9` as 2, while
    # int(inf) raises OverflowError, which callers do not catch.
    if isinstance(value, float) and not (math.isfinite(value) and value.is_integer()):
        raise ValueError(
            f"prefill_coalesce_requests must be a finite integer, got {value!r}"
        )
    requests = int(value)
    if requests < 0:
        raise ValueError("prefill_coalesce_requests must be >= 0")
    return requests


def _normalize_wait_ms(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"prefill_coalesce_wait_ms must be a number, got {value!r}")
    try:
        wait_ms = float(value)
    except OverflowError as exc:
        raise ValueError("prefill_coalesce_wait_ms must be a finite value > 0") from exc
    if not (math.isfinite(wait_ms) and wait_ms > 0):
        raise ValueError("prefill_coalesce_wait_ms must be a finite value > 0")
    return wait_ms
