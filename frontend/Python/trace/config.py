# ===- config.py ---------------------------------------------------------------
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

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TraceConfig:
    trace_config: dict
    matched_nodes: set[str] = field(default_factory=set, init=False)


def normalize_trace_meta(key: str, meta):
    extra = set(meta) - {"id", "id_path", "tag", "cycle", "tensor"}
    if extra:
        raise ValueError(f"unsupported trace fields for {key}: {sorted(extra)}")
    trace_id = meta["id"]
    path = (
        meta["id_path"]
        if "id_path" in meta
        else ([trace_id] if type(trace_id) is int else trace_id)
    )
    if (
        not isinstance(path, list)
        or not path
        or any(type(i) is not int for i in path)
    ):
        raise ValueError(
            f"trace id for {key} must be an integer or integer path"
        )
    if type(meta["tag"]) is not str or not meta["tag"]:
        raise ValueError(f"trace tag for {key} must be a non-empty string")
    for mode in ("cycle", "tensor"):
        if type(meta[mode]) is not bool:
            raise TypeError(f"trace {mode} for {key} must be a boolean")
    return {
        "id": path[0],
        "id_path": path,
        "tag": meta["tag"],
        "cycle": meta["cycle"],
        "tensor": meta["tensor"],
    }


def load_trace_config(path: Path) -> dict:
    data = tomllib.loads(path.read_text())
    if set(data["trace"]) != {"node"}:
        raise ValueError(
            "trace config must contain only [[trace.node]] entries"
        )
    result = {}
    ids = set()
    for item in data["trace"]["node"]:
        if set(item) != {"node", "id", "tag", "cycle", "tensor"}:
            raise ValueError(
                "trace node requires node, id, tag, cycle and tensor"
            )
        node = item["node"]
        if type(node) is not str or not node:
            raise ValueError("trace node must be a non-empty string")
        meta = normalize_trace_meta(
            node, {k: v for k, v in item.items() if k != "node"}
        )
        key = tuple(meta["id_path"])
        if node in result or key in ids:
            raise ValueError(f"duplicate trace node or id: {node}")
        result[node] = meta
        ids.add(key)
    return result
