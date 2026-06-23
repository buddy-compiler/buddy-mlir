# ===- fx.py -------------------------------------------------------------------
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

import json

import torch

from .config import TraceConfig, normalize_trace_meta


def format_fx_target(gm_node) -> str:
    target = gm_node.target
    if hasattr(target, "__name__"):
        return target.__name__
    return str(target)


def format_fx_meta(gm_node) -> tuple[str, str]:
    tensor_meta = gm_node.meta.get("tensor_meta")
    if tensor_meta is not None:
        if hasattr(tensor_meta, "shape") and hasattr(tensor_meta, "dtype"):
            return (
                str(list(tensor_meta.shape)),
                str(tensor_meta.dtype).removeprefix("torch."),
            )
        if isinstance(tensor_meta, (tuple, list)):
            shapes = [str(list(item.shape)) for item in tensor_meta]
            dtypes = [str(item.dtype).removeprefix("torch.") for item in tensor_meta]
            return ";".join(shapes), ";".join(dtypes)

    val = gm_node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return str(list(val.shape)), str(val.dtype).removeprefix("torch.")
    if isinstance(val, (tuple, list)) and all(
        isinstance(item, torch.Tensor) for item in val
    ):
        shapes = [str(list(item.shape)) for item in val]
        dtypes = [str(item.dtype).removeprefix("torch.") for item in val]
        return ";".join(shapes), ";".join(dtypes)
    return "", ""


def dump_fx_graph(
    trace: TraceConfig | None,
    gm,
    *,
    param_nodes: list,
    buffer_nodes: list,
    input_nodes: list,
    other_nodes: list,
) -> None:
    if trace is None or trace.fx_path is None:
        return

    trace.trace_file_dir.mkdir(parents=True, exist_ok=True)
    nodes = list(gm.graph.nodes)
    node_kinds = {}
    for node in param_nodes:
        node_kinds[node.name] = "param"
    for node in buffer_nodes:
        node_kinds[node.name] = "buffer"
    for node in input_nodes:
        node_kinds[node.name] = "input"
    for node in other_nodes:
        node_kinds[node.name] = "other"

    input_deps = {}
    for node in nodes:
        kind = node_kinds.get(node.name, "other")
        if kind == "input":
            input_deps[node.name] = True
            continue
        if kind in {"param", "buffer"}:
            input_deps[node.name] = False
            continue

        def depends_on_input(arg):
            if isinstance(arg, torch.fx.Node):
                return input_deps.get(arg.name, False)
            if isinstance(arg, (list, tuple)):
                return any(depends_on_input(item) for item in arg)
            if isinstance(arg, dict):
                return any(depends_on_input(item) for item in arg.values())
            return False

        input_deps[node.name] = depends_on_input(node.args) or depends_on_input(
            node.kwargs
        )

    rows = []
    for node in nodes:
        shape, dtype = format_fx_meta(node)
        users = ",".join(str(user) for user in node.users)
        rows.append(
            [
                node.name,
                node_kinds.get(node.name, "other"),
                node.op,
                format_fx_target(node),
                shape,
                dtype,
                "yes" if input_deps.get(node.name, False) else "no",
                users,
            ]
        )

    headers = [
        "name",
        "kind",
        "op",
        "target",
        "shape",
        "dtype",
        "input_dep",
        "users",
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    lines = [
        "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))),
        "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    for row in rows:
        lines.append(
            "  ".join(row[i].ljust(widths[i]) for i in range(len(headers)))
        )
    trace.fx_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_fx_trace(trace: TraceConfig | None, gm, inputs: list[torch.Tensor]) -> None:
    if trace is None or trace.torch_path is None:
        return
    if not trace.trace_config:
        raise ValueError("trace output requires trace nodes")

    trace.trace_file_dir.mkdir(parents=True, exist_ok=True)

    class TraceInterpreter(torch.fx.Interpreter):
        def __init__(self, module):
            super().__init__(module)
            self.records = []

        def run_node(self, node):
            result = super().run_node(node)
            if node.name in trace.trace_config:
                meta = normalize_trace_meta(node.name, trace.trace_config[node.name])
                if not isinstance(result, torch.Tensor):
                    raise TypeError(f"trace node {node.name} did not produce a tensor")
                array = result.detach().cpu().to(torch.float32).contiguous().numpy()
                self.records.append(
                    {
                        "id": meta["id"],
                        "tag": meta.get("tag", node.name),
                        "shape": list(array.shape),
                        "values": array.reshape(-1).tolist(),
                    }
                )
            return result

    with torch.no_grad():
        interpreter = TraceInterpreter(gm)
        interpreter.run(*inputs)

    with trace.torch_path.open("w", encoding="utf-8") as f:
        for record in sorted(interpreter.records, key=lambda item: item["id"]):
            f.write(json.dumps(record) + "\n")
