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


@dataclass
class TraceConfig:
    """
    Trace settings shared by FX graph dumping and tensor-value tracing.

    Args:
        trace_config: Maps Buddy graph node names to trace metadata. Each entry
            must contain an integer `id` and may contain a string `tag`.
        trace_file_dir: Directory for generated trace files. When set,
            `trace.fx.txt` and `pytorch-ckpt.jsonl` are written under this
            directory.

    Internal state:
        matched_nodes: Buddy graph node names that matched entries in
            `trace_config` during trace insertion. The transform uses this to
            reject stale or misspelled trace config entries.
    """
    trace_config: dict
    trace_file_dir: Path | None = None
    matched_nodes: set[str] = field(default_factory=set, init=False)

    def __post_init__(self):
        self.trace_file_dir = Path(self.trace_file_dir) if self.trace_file_dir else None

    @property
    def fx_path(self) -> Path | None:
        if self.trace_file_dir is None:
            return None
        return self.trace_file_dir / "trace.fx.txt"

    @property
    def torch_path(self) -> Path | None:
        if self.trace_file_dir is None:
            return None
        return self.trace_file_dir / "pytorch-ckpt.jsonl"


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
