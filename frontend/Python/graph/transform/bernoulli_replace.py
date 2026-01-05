# ===- bernoulli_replace.py ---------------------------------------------------
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
# Transform pass to replace BernoulliOp with runtime RNG external calls.
#
# ===---------------------------------------------------------------------------

from .. import Graph
from ..operation import BernoulliOp, CallExternalOp, FullLikeOp, FullOp, OpType

import re


_ARG_POS_RE = re.compile(r"^arg(\d+)_")


def _arg_pos(name):
    if not isinstance(name, str):
        return None
    m = _ARG_POS_RE.match(name)
    if not m:
        return None
    return int(m.group(1))


def replace_bernoulli_with_runtime_rng(graph: Graph):
    for op in list(graph.body):
        if not isinstance(op, BernoulliOp):
            continue

        shape = op.tensor_meta.get("shape", [])
        rank = len(list(shape)) if shape is not None else 0
        arity = len(op.args)

        call_func_name = f"buddy_bernoulli_f32_r{rank}_{arity}"
        if arity == 1:
            src = op.args[0]
            if (
                isinstance(src, str)
                and src in graph.node_table
                and isinstance(graph.node_table[src], (FullLikeOp, FullOp))
            ):
                full_like = graph.node_table[src]
                if len(full_like.args) >= 2 and isinstance(
                    full_like.args[1], float
                ):
                    call_func_name = (
                        f"buddy_bernoulli_f32_f64rng_r{rank}_{arity}"
                    )

        call_args = list(op.args)
        if arity == 2:
            p0 = _arg_pos(call_args[0])
            p1 = _arg_pos(call_args[1])
            if p0 is not None and p1 is not None and p0 > p1:
                call_args = [call_args[1], call_args[0]]

        call_op = CallExternalOp(
            call_func_name=call_func_name,
            args=list(op.args),
            args_index=[0] * arity,
            tensor_meta={
                "shape": op.tensor_meta.get("shape", []),
                "dtype": op.tensor_meta.get("dtype", None),
            },
            # Keep the original op name so any alias/extra outputs referencing
            # the original node remain valid (e.g. .out / in-place variants).
            name=op.name,
        )

        if hasattr(op, "_parents"):
            call_op._parents = op._parents.copy()
        if hasattr(op, "_children"):
            call_op._children = op._children.copy()

        graph.displace_node(op, call_op)
        call_op._arguments = call_args
        call_op._op_type = OpType.Unfusable
