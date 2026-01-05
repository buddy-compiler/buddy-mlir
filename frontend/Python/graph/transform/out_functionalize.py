# ===- out_functionalize.py ---------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# 可选 transform：将 ATen 的 out overload（如 `*.out/*_out`）功能化为对应的
# functional overload，并移除 out-like Tensor kwargs。
# 该 transform 用于提升导入鲁棒性/覆盖率，但不代表支持真实 out buffer 的
# alias/复用语义。
#
# ===---------------------------------------------------------------------------

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type

from .. import Graph
from ..operation import Op


def _parse_torch_op(torch_op: str) -> Optional[Tuple[str, str]]:
    if not isinstance(torch_op, str) or "." not in torch_op:
        return None
    base, overload = torch_op.rsplit(".", 1)
    if not base or not overload:
        return None
    return base, overload


def _candidate_functional_keys(base: str, overload: str) -> List[str]:
    if overload == "out":
        return [f"{base}.default", f"{base}.Tensor", f"{base}.Scalar"]
    if overload.endswith("_out"):
        stripped = overload[: -len("_out")]
        candidates: List[str] = []
        if stripped:
            candidates.append(f"{base}.{stripped}")
        candidates.append(f"{base}.default")
        return candidates
    return []


def functionalize_out_overloads(
    graph: Graph, *, ops_map: Dict[str, Type[Op]]
) -> None:
    for node in list(graph.body):
        torch_op = getattr(node, "_torch_op", None)
        parsed = _parse_torch_op(torch_op)
        if parsed is None:
            continue
        base, overload = parsed
        if overload != "out" and not overload.endswith("_out"):
            continue

        out_kw_names = getattr(node, "_torch_out_kwarg_names", None)
        if not isinstance(out_kw_names, list):
            out_kw_names = []
        if not out_kw_names and not any(
            k in node.kwargs for k in ("out", "values", "indices")
        ):
            continue

        target_key: Optional[str] = None
        for cand in _candidate_functional_keys(base, overload):
            if cand in ops_map:
                target_key = cand
                break
        if target_key is None:
            continue

        new_node = ops_map[target_key]()
        new_node.name = node.name
        graph.displace_node(node, new_node)

        for k in list(out_kw_names) + ["out", "values", "indices"]:
            new_node.kwargs.pop(k, None)

        new_node._torch_op = target_key
        new_node._torch_out_kwarg_names = []
