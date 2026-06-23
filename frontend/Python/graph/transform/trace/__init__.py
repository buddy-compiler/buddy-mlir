# ===- __init__.py -------------------------------------------------------------
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
from ....trace.fx import write_fx_trace
from ... import Graph


def trace_insertion(trace: TraceConfig, gm=None, inputs=None):
    def _trace_insertion(graph: Graph):
        matched_nodes = set()
        for node in graph.body:
            if node.name not in trace.trace_config:
                continue
            node._trace_meta = normalize_trace_meta(
                node.name, trace.trace_config[node.name]
            )
            matched_nodes.add(node.name)

        missing = set(trace.trace_config) - matched_nodes
        if missing:
            names = ", ".join(sorted(missing))
            raise KeyError(f"trace nodes did not match buddy graph nodes: {names}")

        trace.matched_nodes = matched_nodes
        if gm is not None and inputs is not None:
            write_fx_trace(trace, gm, list(inputs))

    return _trace_insertion
