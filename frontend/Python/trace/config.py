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
        return {"id": meta}
    if not isinstance(meta, dict):
        raise TypeError(
            f"trace.nodes[{key!r}] must be an int or dict, got {type(meta).__name__}"
        )
    extra_keys = set(meta) - {"id", "tag"}
    if extra_keys:
        names = ", ".join(sorted(extra_keys))
        raise TypeError(f"trace.nodes[{key!r}] has unsupported keys: {names}")

    trace_id = meta.get("id")
    if not isinstance(trace_id, int):
        raise TypeError(f"trace.nodes[{key!r}] requires integer id")

    result = {"id": trace_id}
    if "tag" in meta:
        tag = meta["tag"]
        if not isinstance(tag, str):
            raise TypeError(f"trace.nodes[{key!r}]['tag'] must be a string")
        result["tag"] = tag
    return result


def load_trace_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"trace config not found: {path}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    items = data.get("trace")
    if not isinstance(items, list):
        raise ValueError("trace config must contain [[trace]] entries")

    result = {}
    ids = set()
    for item in items:
        extra_keys = set(item) - {"node", "id", "tag"}
        if extra_keys:
            names = ", ".join(sorted(extra_keys))
            raise ValueError(f"unsupported trace fields: {names}")
        node = item.get("node")
        trace_id = item.get("id")
        tag = item.get("tag")
        if not isinstance(node, str) or not node:
            raise ValueError("trace.node must be a non-empty string")
        if not isinstance(trace_id, int):
            raise ValueError(f"trace id for {node} must be an integer")
        if not isinstance(tag, str) or not tag:
            raise ValueError(f"trace tag for {node} must be a non-empty string")
        if node in result:
            raise ValueError(f"duplicate trace node: {node}")
        if trace_id in ids:
            raise ValueError(f"duplicate trace id: {trace_id}")
        ids.add(trace_id)
        result[node] = {"id": trace_id, "tag": tag}
    return result
