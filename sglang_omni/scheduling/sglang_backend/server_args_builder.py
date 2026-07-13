# SPDX-License-Identifier: Apache-2.0
"""Shared ServerArgs construction for SGLang AR engines."""
from __future__ import annotations

import dataclasses
from typing import Any

from sglang.srt.model_executor.cuda_graph_config import Backend, Phase
from sglang.srt.server_args import ServerArgs


@dataclasses.dataclass
class OmniServerArgs(ServerArgs):
    """SGLang arguments with explicit Omni prefill-graph capability gating."""

    omni_model_architecture: dataclasses.InitVar[str | None] = None

    def __post_init__(self, omni_model_architecture: str | None) -> None:
        self._omni_model_architecture = omni_model_architecture
        super().__post_init__()

    def _apply_cuda_graph_compatibility(self) -> None:
        locked = self._cuda_graph_config_locked
        if (Phase.PREFILL, "backend") in locked:
            super()._apply_cuda_graph_compatibility()
            return

        architecture = self._omni_model_architecture
        if not _supports_sglang_full_prefill(architecture):
            # note (kaige): Preserve Omni's decode-only default until a model
            # declares and tests the native FULL prefill runner contract.
            self.cuda_graph_config.prefill.backend = Backend.DISABLED
            super()._apply_cuda_graph_compatibility()
            return
        assert architecture is not None

        model_config = self.get_model_config()
        if not model_config.is_multimodal:
            raise ValueError(
                f"{architecture} requires SGLang multimodal classification "
                "for the prefill CUDA graph input buffer"
            )
        self.cuda_graph_config.prefill.backend = Backend.FULL
        super()._apply_cuda_graph_compatibility()


def _supports_sglang_full_prefill(architecture: str | None) -> bool:
    from sglang_omni.models.model_capabilities import get_model_capabilities

    if architecture is None:
        return False
    capabilities = get_model_capabilities(architecture)
    return bool(capabilities is not None and capabilities.supports_sglang_full_prefill)


def build_sglang_server_args(
    model_path: str,
    context_length: int,
    *,
    model_architecture: str | None = None,
    chunked_prefill_size: int | None = None,
    max_prefill_tokens: int = 16384,
    max_running_requests: int = 16,
    mem_fraction_static: float | None = None,
    **overrides: Any,
) -> ServerArgs:
    """Build ServerArgs with shared defaults for all SGLang AR engines."""
    kwargs: dict[str, Any] = {
        "model_path": model_path,
        "trust_remote_code": True,
        "tp_size": 1,
        "pp_size": 1,
        "chunked_prefill_size": chunked_prefill_size,
        "max_prefill_tokens": max_prefill_tokens,
        "max_running_requests": max_running_requests,
        "random_seed": 123,
        "context_length": context_length,
    }
    if mem_fraction_static is not None:
        kwargs["mem_fraction_static"] = mem_fraction_static
    kwargs.update(overrides)
    if kwargs.get("mem_fraction_static") is None:
        kwargs.pop("mem_fraction_static", None)
    return OmniServerArgs(
        omni_model_architecture=model_architecture,
        **kwargs,
    )


def apply_encoder_mem_reserve(
    server_args: ServerArgs,
    encoder_mem_reserve: float,
) -> None:
    """Subtract Qwen external encoder headroom from an auto-selected SGLang budget."""
    if not 0.0 <= encoder_mem_reserve < 1.0:
        raise ValueError("encoder_mem_reserve must be in [0, 1)")
    if encoder_mem_reserve == 0:
        return

    current = server_args.mem_fraction_static
    if current is None:
        return

    reserved = current - encoder_mem_reserve
    if reserved < 0.1:
        raise ValueError(
            f"auto mem_fraction_static {current:.3f} minus encoder_mem_reserve "
            f"{encoder_mem_reserve:.3f} = {reserved:.3f} is below the safe "
            "floor 0.1; lower encoder_mem_reserve or pin mem_fraction_static "
            "explicitly."
        )
    server_args.mem_fraction_static = round(reserved, 3)
