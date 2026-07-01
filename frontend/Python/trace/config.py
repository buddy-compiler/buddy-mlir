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

from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@dataclass
class TraceConfig:
    """
    Trace settings for Buddy graph based tensor-value tracing.

    Args:
        trace_config: Maps Buddy graph node names to trace metadata. Each entry
            must contain an integer `id` and may contain a string `tag`.

    Internal state:
        matched_nodes: Buddy graph node names that matched entries in
            `trace_config` during trace insertion. The transform uses this to
            reject stale or misspelled trace config entries.
    """

    trace_config: dict
    matched_nodes: set[str] = field(default_factory=set, init=False)


def normalize_trace_meta(key: str, meta):
    if isinstance(meta, int):
        return {"id": meta, "id_path": [meta]}
    if not isinstance(meta, dict):
        raise TypeError(
            f"trace.nodes[{key!r}] must be an int or dict, got {type(meta).__name__}"
        )
    extra_keys = set(meta) - {"id", "id_path", "tag", "extend"}
    if extra_keys:
        names = ", ".join(sorted(extra_keys))
        raise TypeError(f"trace.nodes[{key!r}] has unsupported keys: {names}")

    trace_id = meta.get("id")
    trace_id_path = meta.get("id_path")
    if trace_id_path is not None:
        if not (
            isinstance(trace_id_path, list)
            and trace_id_path
            and all(isinstance(item, int) for item in trace_id_path)
        ):
            raise TypeError(
                f"trace.nodes[{key!r}] requires non-empty integer id_path"
            )
        if trace_id is not None and not isinstance(trace_id, int):
            raise TypeError(f"trace.nodes[{key!r}] requires integer id")
        flat_id = trace_id_path[0] if trace_id is None else trace_id
        id_path = trace_id_path
    elif isinstance(trace_id, int):
        flat_id = trace_id
        id_path = [trace_id]
    elif (
        isinstance(trace_id, list)
        and trace_id
        and all(isinstance(item, int) for item in trace_id)
    ):
        flat_id = trace_id[0]
        id_path = trace_id
    else:
        raise TypeError(f"trace.nodes[{key!r}] requires integer id or id path")

    result = {"id": flat_id, "id_path": id_path}
    if "tag" in meta:
        tag = meta["tag"]
        if not isinstance(tag, str):
            raise TypeError(f"trace.nodes[{key!r}]['tag'] must be a string")
        result["tag"] = tag
    if "extend" in meta:
        result["extend"] = meta["extend"]
    return result


def _parse_id(node: str, trace_id):
    if isinstance(trace_id, int):
        return trace_id, (trace_id,)
    if (
        isinstance(trace_id, list)
        and trace_id
        and all(isinstance(item, int) for item in trace_id)
    ):
        return trace_id[0], tuple(trace_id)
    raise ValueError(f"trace id for {node} must be an integer or id path")


def _parse_extend(node: str, value):
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"trace.extend for {node} must be a list")

    result = []
    seen = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"trace.extend for {node} must contain tables")
        extra_keys = set(item) - {"dialect", "granularity", "include"}
        if extra_keys:
            names = ", ".join(sorted(extra_keys))
            raise ValueError(
                f"unsupported trace.extend fields for {node}: {names}"
            )
        dialect = item.get("dialect")
        granularity = item.get("granularity", "op")
        include = item.get("include", [])
        if dialect not in {"linalg", "buckyball"}:
            raise ValueError(
                f"unsupported trace.extend dialect for {node}: {dialect}"
            )
        if granularity != "op":
            raise ValueError(
                f"unsupported trace.extend granularity for {node}: {granularity}"
            )
        if not isinstance(include, list) or not all(
            isinstance(name, str) and name for name in include
        ):
            raise ValueError(
                f"trace.extend include for {node} must be a string list"
            )
        if dialect in seen:
            raise ValueError(
                f"duplicate trace.extend dialect for {node}: {dialect}"
            )
        seen.add(dialect)
        result.append(
            {
                "dialect": dialect,
                "granularity": granularity,
                "include": include,
            }
        )
    return result


def load_trace_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"trace config not found: {path}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    trace_data = data.get("trace")
    if not isinstance(trace_data, dict):
        raise ValueError("trace config must contain [trace] entries")

    extra_keys = set(trace_data) - {"node", "extend"}
    if extra_keys:
        names = ", ".join(sorted(extra_keys))
        raise ValueError(f"unsupported trace fields: {names}")

    items = trace_data.get("node")
    if not isinstance(items, list):
        raise ValueError("trace config must contain [[trace.node]] entries")
    extend = _parse_extend("trace", trace_data.get("extend"))

    result = {}
    ids = set()
    for item in items:
        extra_keys = set(item) - {"node", "id", "tag"}
        if extra_keys:
            names = ", ".join(sorted(extra_keys))
            raise ValueError(f"unsupported trace.node fields: {names}")
        node = item.get("node")
        flat_id, id_path = _parse_id(node, item.get("id"))
        tag = item.get("tag")
        if not isinstance(node, str) or not node:
            raise ValueError("trace.node must be a non-empty string")
        if not isinstance(tag, str) or not tag:
            raise ValueError(f"trace tag for {node} must be a non-empty string")
        if node in result:
            raise ValueError(f"duplicate trace node: {node}")
        if id_path in ids:
            raise ValueError(f"duplicate trace id: {list(id_path)}")
        ids.add(id_path)
        result[node] = {
            "id": flat_id,
            "id_path": list(id_path),
            "tag": tag,
            "extend": extend,
        }
    return result
