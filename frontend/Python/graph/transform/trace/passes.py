# ===- passes.py ---------------------------------------------------------------
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

from ....trace.config import TraceConfig, normalize_trace_meta
from .lowering import trace_op_result, trace_op_start


def _wrap_registry(ops_registry: dict):
    wrapped = {}
    for op_name, lower in ops_registry.items():

        def traced_lower(node, *args, _lower=lower, **kwargs):
            trace_op_start(node)
            op_ret = _lower(node, *args, **kwargs)
            return trace_op_result(node, op_ret)

        wrapped[op_name] = traced_lower
    return wrapped


class TraceInsertionPass:
    def __init__(self, trace: TraceConfig):
        self.trace = trace

    def __call__(self, graph):
        matched_nodes = set()
        for node in graph.body:
            if node.name not in self.trace.trace_config:
                continue
            if node.__class__.__name__ in {"GetItemOp", "PlaceholderOp"}:
                raise TypeError(
                    f"Trace node {node.name} does not lower to an MLIR op"
                )
            node.trace_meta = normalize_trace_meta(
                node.name, self.trace.trace_config[node.name]
            )
            matched_nodes.add(node.name)

        missing = set(self.trace.trace_config) - matched_nodes
        if missing:
            names = ", ".join(sorted(missing))
            raise KeyError(
                f"trace nodes did not match buddy graph nodes: {names}"
            )

        self.trace.matched_nodes = matched_nodes
        graph._ops_registry = _wrap_registry(graph._ops_registry)


def trace_insertion(trace: TraceConfig):
    return TraceInsertionPass(trace)


__all__ = [
    "TraceInsertionPass",
    "trace_insertion",
]
