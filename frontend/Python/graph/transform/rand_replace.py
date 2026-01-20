# ===- rand_replace.py --------------------------------------------------------
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
# Runtime RNG transform passes (Bernoulli / Geometric / Rand).
#
# ===---------------------------------------------------------------------------

import re

from .. import Graph
from ..graph import TensorDType
from ..operation import (
    BernoulliOp,
    CallExternalOp,
    ExponentialOp,
    FullLikeOp,
    FullOp,
    GeometricOp,
    OpType,
    RandOp,
)


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


def replace_geometric_with_runtime_rng(graph: Graph):
    for op in list(graph.body):
        if not isinstance(op, GeometricOp):
            continue

        if len(op.args) < 2:
            continue

        like_name = op.args[0]
        p = op.args[1]
        if not isinstance(like_name, str):
            continue
        if not isinstance(p, (float, int)):
            continue

        shape = op.tensor_meta.get("shape", [])
        rank = len(list(shape)) if shape is not None else 0
        call_func_name = f"buddy_geometric_f32_r{rank}_1"

        prob_name = f"{op.name}_p_full_like"
        if prob_name in graph.node_table:
            continue

        insert_pos = graph.body.index(op)

        full_like = FullLikeOp()
        full_like.name = prob_name
        full_like.add_argument(like_name)
        full_like.add_argument(float(p))
        full_like.tensor_meta["shape"] = (
            list(shape) if shape is not None else []
        )
        full_like.tensor_meta["dtype"] = op.tensor_meta.get("dtype", None)
        full_like.add_parent(like_name)
        full_like.add_children(op.name)

        like_node = graph.node_table.get(like_name)
        if like_node is not None and hasattr(like_node, "_children"):
            if full_like.name not in like_node._children:
                like_node._children.append(full_like.name)

        graph.node_table[full_like.name] = full_like
        graph.body.insert(insert_pos, full_like)

        call_op = CallExternalOp(
            call_func_name=call_func_name,
            args=[full_like.name],
            args_index=[0],
            tensor_meta={
                "shape": op.tensor_meta.get("shape", []),
                "dtype": op.tensor_meta.get("dtype", None),
            },
            name=op.name,
        )

        if hasattr(op, "_parents"):
            call_op._parents = [full_like.name]
        if hasattr(op, "_children"):
            call_op._children = op._children.copy()

        graph.displace_node(op, call_op)
        call_op._arguments = [full_like.name]
        if hasattr(call_op, "_parents"):
            call_op._parents = [full_like.name]
        if like_node is not None and hasattr(like_node, "_children"):
            while call_op.name in like_node._children:
                like_node._children.remove(call_op.name)
        call_op._op_type = OpType.Unfusable


def replace_rand_with_runtime_rng(graph: Graph):
    for op in list(graph.body):
        if not isinstance(op, RandOp):
            continue

        shape = op.tensor_meta.get("shape", [])
        rank = len(list(shape)) if shape is not None else 0
        dtype = op.tensor_meta.get("dtype", None)
        if dtype != TensorDType.Float32:
            continue

        call_func_name = f"buddy_rand_f32_like_r{rank}_1"

        like_op = FullOp()
        like_base_name = f"{op.name}__rand_like"
        like_name = like_base_name
        suffix = 2
        while like_name in graph.node_table:
            like_name = f"{like_base_name}_v{suffix}"
            suffix += 1
        like_op.name = like_name
        like_op.tensor_meta = {
            "shape": op.tensor_meta.get("shape", []),
            "dtype": op.tensor_meta.get("dtype", None),
        }

        size = op.args[0] if len(op.args) > 0 else list(shape)
        like_op._arguments = [size, 0.0]
        like_op._keyword_arguments = {}
        for key in ("dtype", "device"):
            if key in op.kwargs:
                like_op._keyword_arguments[key] = op.kwargs[key]

        call_op = CallExternalOp(
            call_func_name=call_func_name,
            args=[like_op.name],
            args_index=[0],
            tensor_meta={
                "shape": op.tensor_meta.get("shape", []),
                "dtype": op.tensor_meta.get("dtype", None),
            },
            name=op.name,
        )

        body_idx = graph.body.index(op)
        if hasattr(op, "_parents"):
            call_op._parents = op._parents.copy()
        if hasattr(op, "_children"):
            call_op._children = op._children.copy()

        graph.displace_node(op, call_op)
        graph.body.insert(body_idx, like_op)
        graph.node_table[like_op.name] = like_op

        call_op._arguments = [like_op.name]
        call_op._args_index = [0]
        call_op._keyword_arguments = {}
        call_op._op_type = OpType.Unfusable

        if like_op.name not in call_op._parents:
            call_op.add_parent(like_op.name)
        if call_op.name not in like_op._children:
            like_op.add_children(call_op.name)


def replace_exponential_with_runtime_rng(graph: Graph):
    for op in list(graph.body):
        if not isinstance(op, ExponentialOp):
            continue

        dtype = op.tensor_meta.get("dtype", None)
        if dtype != TensorDType.Float32:
            continue

        if not op.args or not isinstance(op.args[0], str):
            continue
        like_name = op.args[0]

        lambd = 1.0
        if len(op.args) >= 2 and isinstance(op.args[1], (float, int)):
            lambd = float(op.args[1])

        shape = op.tensor_meta.get("shape", [])
        rank = len(list(shape)) if shape is not None else 0
        call_func_name = f"buddy_exponential_f32_r{rank}_1"

        full_like = FullLikeOp()
        base_name = f"{op.name}__exponential_lambda_full_like"
        full_like_name = base_name
        suffix = 2
        while full_like_name in graph.node_table:
            full_like_name = f"{base_name}_v{suffix}"
            suffix += 1
        full_like.name = full_like_name
        full_like.add_argument(like_name)
        full_like.add_argument(lambd)
        full_like.tensor_meta["shape"] = (
            list(shape) if shape is not None else []
        )
        full_like.tensor_meta["dtype"] = dtype
        full_like.add_parent(like_name)
        full_like.add_children(op.name)

        like_node = graph.node_table.get(like_name)
        if like_node is not None and hasattr(like_node, "_children"):
            if full_like.name not in like_node._children:
                like_node._children.append(full_like.name)

        insert_pos = graph.body.index(op)
        graph.node_table[full_like.name] = full_like
        graph.body.insert(insert_pos, full_like)

        call_op = CallExternalOp(
            call_func_name=call_func_name,
            args=[full_like.name],
            args_index=[0],
            tensor_meta={
                "shape": op.tensor_meta.get("shape", []),
                "dtype": dtype,
            },
            name=op.name,
        )

        if hasattr(op, "_parents"):
            call_op._parents = [full_like.name]
        if hasattr(op, "_children"):
            call_op._children = op._children.copy()

        graph.displace_node(op, call_op)
        call_op._arguments = [full_like.name]
        call_op._args_index = [0]
        call_op._keyword_arguments = {}
        call_op._op_type = OpType.Unfusable
