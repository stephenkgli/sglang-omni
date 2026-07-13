# SPDX-License-Identifier: Apache-2.0
"""Generation-stage batch policy helpers for SGLang-backed stages."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from sglang.srt.model_executor.cuda_graph_config import Backend

_MISSING = object()


def set_default_full_prefill_request_slots(server_args: Any) -> None:
    """Cover configured concurrency for FULL prefill unless explicitly set."""
    prefill_config = server_args.cuda_graph_config.prefill
    if (
        prefill_config.backend == Backend.FULL
        and prefill_config.full_prefill_max_req is None
    ):
        prefill_config.full_prefill_max_req = server_args.max_running_requests


def build_default_cuda_graph_bs(max_bs: int) -> list[int]:
    max_bs = int(max_bs)
    if max_bs < 1:
        raise ValueError("max_bs must be >= 1")

    values = [1, 2, 4, 8, 12]
    values.extend(range(16, 257, 8))
    values.extend(range(272, 512, 16))
    values.extend(range(512, max_bs + 1, 32))
    values = [bs for bs in values if bs <= max_bs]
    if not values or values[-1] != max_bs:
        values.append(max_bs)
    return values


def build_generation_batch_overrides(
    *,
    max_running_requests: int,
    cuda_graph_max_bs: int | None = None,
    torch_compile_max_bs: int | None = None,
    enable_prefill_cuda_graph: bool = False,
    prefill_cuda_graph_backend: str | None = None,
    prefill_graph_token_buckets: list[int] | None = None,
    server_args_overrides: Mapping[str, Any] | None = None,
    **stage_defaults: Any,
) -> dict[str, Any]:
    """Resolve Omni serving limits into SGLang's phase-aware graph fields."""
    incoming = dict(server_args_overrides or {})
    max_running_requests = _normalize_positive_int(
        "max_running_requests",
        incoming.pop("max_running_requests", max_running_requests),
    )
    cuda_graph_max_bs = (
        max_running_requests if cuda_graph_max_bs is None else cuda_graph_max_bs
    )
    cuda_graph_max_bs = _normalize_positive_int(
        "cuda_graph_max_bs",
        _pop_graph_alias(
            incoming,
            new_name="cuda_graph_max_bs_decode",
            legacy_name="cuda_graph_max_bs",
            default=cuda_graph_max_bs,
        ),
    )
    torch_compile_max_bs = (
        max_running_requests if torch_compile_max_bs is None else torch_compile_max_bs
    )
    torch_compile_max_bs = _normalize_positive_int(
        "torch_compile_max_bs",
        incoming.pop("torch_compile_max_bs", torch_compile_max_bs),
    )
    cuda_graph_bs = _pop_graph_alias(
        incoming,
        new_name="cuda_graph_bs_decode",
        legacy_name="cuda_graph_bs",
        default=_MISSING,
    )

    if prefill_graph_token_buckets is not None:
        conflicting_prefill_fields = sorted(
            field
            for field in ("cuda_graph_bs_prefill", "cuda_graph_max_bs_prefill")
            if field in incoming or field in stage_defaults
        )
        if conflicting_prefill_fields:
            raise ValueError(
                "prefill_graph_token_buckets cannot be combined with "
                + ", ".join(conflicting_prefill_fields)
            )

    prefill_overrides = _build_prefill_cuda_graph_overrides(
        enabled=enable_prefill_cuda_graph,
        backend=prefill_cuda_graph_backend,
        token_buckets=prefill_graph_token_buckets,
    )
    if incoming.get("disable_prefill_cuda_graph") is True:
        prefill_overrides.pop("cuda_graph_backend_prefill", None)

    overrides = {
        **prefill_overrides,
        **stage_defaults,
        **incoming,
        "max_running_requests": max_running_requests,
        "cuda_graph_max_bs_decode": cuda_graph_max_bs,
        "torch_compile_max_bs": torch_compile_max_bs,
    }
    if cuda_graph_bs is _MISSING:
        overrides["cuda_graph_bs_decode"] = build_default_cuda_graph_bs(
            cuda_graph_max_bs
        )
    else:
        overrides["cuda_graph_bs_decode"] = cuda_graph_bs

    return overrides


def _pop_graph_alias(
    values: dict[str, Any],
    *,
    new_name: str,
    legacy_name: str,
    default: Any,
) -> Any:
    if new_name in values and legacy_name in values:
        raise ValueError(f"Specify only one of {new_name} and {legacy_name}")
    if new_name in values:
        return values.pop(new_name)
    return values.pop(legacy_name, default)


def _build_prefill_cuda_graph_overrides(
    *,
    enabled: bool,
    backend: str | None,
    token_buckets: list[int] | None,
) -> dict[str, Any]:
    if not isinstance(enabled, bool):
        raise TypeError("enable_prefill_cuda_graph must be a bool")
    if not enabled:
        if token_buckets is not None:
            raise ValueError(
                "prefill_graph_token_buckets must be None when "
                "enable_prefill_cuda_graph is False"
            )
        if backend is not None:
            raise ValueError(
                "prefill_cuda_graph_backend must be None when "
                "enable_prefill_cuda_graph is False"
            )
        return {"disable_prefill_cuda_graph": True}

    if backend not in (
        None,
        Backend.FULL,
        Backend.BREAKABLE,
        Backend.TC_PIECEWISE,
    ):
        raise ValueError(
            "prefill_cuda_graph_backend must be one of full, breakable, "
            "or tc_piecewise"
        )

    overrides = {"cuda_graph_backend_prefill": backend} if backend is not None else {}
    if token_buckets is None:
        return overrides
    normalized = _normalize_prefill_graph_token_buckets(token_buckets)
    overrides.update(
        cuda_graph_bs_prefill=normalized,
        cuda_graph_max_bs_prefill=max(normalized),
    )
    return overrides


def _normalize_prefill_graph_token_buckets(token_buckets: list[int]) -> list[int]:
    if (
        not isinstance(token_buckets, list)
        or not token_buckets
        or any(
            isinstance(bucket, bool) or not isinstance(bucket, int) or bucket < 1
            for bucket in token_buckets
        )
    ):
        raise ValueError(
            "prefill_graph_token_buckets must be a non-empty list of positive integers"
        )
    return sorted(set(token_buckets))


def validate_generation_batch_policy(
    *,
    model_name: str,
    server_args: Any,
    model_buffer_bs: int | None = None,
) -> None:
    errors: list[str] = []

    max_running_requests = _validate_positive_int(
        "max_running_requests",
        server_args.max_running_requests,
        errors,
    )
    cuda_graph_config = server_args.cuda_graph_config
    decode_graph_config = cuda_graph_config.decode
    cuda_graph_enabled = decode_graph_config.backend != Backend.DISABLED

    cuda_graph_max_bs: int | None = None
    cuda_graph_bs: tuple[int, ...] | None = None
    if cuda_graph_enabled:
        cuda_graph_max_bs = _validate_positive_int(
            "cuda_graph_max_bs",
            decode_graph_config.max_bs,
            errors,
            required=True,
        )
        cuda_graph_bs_value = decode_graph_config.bs
        if cuda_graph_bs_value is None:
            errors.append("cuda_graph_bs must be explicit when CUDA graph is enabled")
        else:
            cuda_graph_bs = _normalize_cuda_graph_bs(cuda_graph_bs_value, errors)

        if cuda_graph_max_bs is not None and cuda_graph_bs is not None:
            if max(cuda_graph_bs) != cuda_graph_max_bs:
                errors.append(
                    "max(cuda_graph_bs) must match cuda_graph_max_bs "
                    f"({max(cuda_graph_bs)} != {cuda_graph_max_bs})"
                )

        if (
            max_running_requests is not None
            and cuda_graph_max_bs is not None
            and cuda_graph_max_bs < max_running_requests
        ):
            errors.append(
                "cuda_graph_max_bs must cover max_running_requests "
                f"({cuda_graph_max_bs} < {max_running_requests})"
            )

    torch_compile_enabled = bool(server_args.enable_torch_compile)
    torch_compile_max_bs = _validate_positive_int(
        "torch_compile_max_bs",
        server_args.torch_compile_max_bs,
        errors,
        required=torch_compile_enabled,
    )
    if (
        torch_compile_enabled
        and max_running_requests is not None
        and torch_compile_max_bs is not None
        and torch_compile_max_bs < max_running_requests
    ):
        errors.append(
            "torch_compile_max_bs must cover max_running_requests "
            f"({torch_compile_max_bs} < {max_running_requests})"
        )

    normalized_model_buffer_bs: int | None = None
    if model_buffer_bs is not None:
        normalized_model_buffer_bs = int(model_buffer_bs)
        if normalized_model_buffer_bs < 1:
            errors.append("model_buffer_bs must be >= 1")
        if (
            max_running_requests is not None
            and normalized_model_buffer_bs < max_running_requests
        ):
            errors.append(
                "model_buffer_bs must cover max_running_requests "
                f"({normalized_model_buffer_bs} < {max_running_requests})"
            )

    if errors:
        raise ValueError(
            f"{model_name} invalid generation batch policy: " + "; ".join(errors)
        )


def _validate_positive_int(
    field: str,
    value: Any,
    errors: list[str],
    *,
    required: bool = True,
) -> int | None:
    if value is None:
        if required:
            errors.append(f"{field} must be explicit")
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        errors.append(f"{field} must be an integer")
        return None
    if normalized < 1:
        errors.append(f"{field} must be >= 1")
        return None
    return normalized


def _normalize_positive_int(field: str, value: Any) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if normalized < 1:
        raise ValueError(f"{field} must be >= 1")
    return normalized


def _normalize_cuda_graph_bs(
    value: Iterable[Any],
    errors: list[str],
) -> tuple[int, ...] | None:
    if isinstance(value, (str, bytes)):
        errors.append("cuda_graph_bs must be a sequence of positive integers")
        return None

    try:
        normalized = tuple(int(item) for item in value)
    except (TypeError, ValueError):
        errors.append("cuda_graph_bs must be a sequence of positive integers")
        return None

    if not normalized:
        errors.append("cuda_graph_bs must be non-empty")
        return None
    if any(item < 1 for item in normalized):
        errors.append("cuda_graph_bs values must be >= 1")
        return None
    if tuple(sorted(set(normalized))) != normalized:
        errors.append("cuda_graph_bs must be strictly increasing")
        return None
    return normalized
