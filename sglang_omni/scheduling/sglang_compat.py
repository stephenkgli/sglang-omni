# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from array import array
from typing import Any


def normalize_req_token_storage(req: Any) -> None:
    origin_input_ids = req.origin_input_ids
    unpadded_input_ids = getattr(req, "origin_input_ids_unpadded", origin_input_ids)
    if isinstance(origin_input_ids, array) and origin_input_ids.typecode == "q":
        normalized_origin_input_ids = origin_input_ids
    else:
        normalized_origin_input_ids = array("q", origin_input_ids)
    req.origin_input_ids = normalized_origin_input_ids

    if unpadded_input_ids is origin_input_ids:
        req.origin_input_ids_unpadded = normalized_origin_input_ids
    elif not (
        isinstance(unpadded_input_ids, array) and unpadded_input_ids.typecode == "q"
    ):
        req.origin_input_ids_unpadded = array("q", unpadded_input_ids)
