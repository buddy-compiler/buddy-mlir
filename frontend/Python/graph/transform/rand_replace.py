# ===- rand_replace.py ---------------------------------------------------------
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
# Runtime RNG transform passes (Bernoulli / Geometric / Rand / Randn).
#
# ===---------------------------------------------------------------------------

import re

from .. import Graph
from ..graph import TensorDType
from ..operation import (
    BernoulliOp,
    CallExternalOp,
    CauchyOp,
    ExponentialOp,
    FullLikeOp,
    FullOp,
    GeometricOp,
    LogNormalOp,
    MultinomialOp,
    NormalOp,
    OpType,
    PoissonOp,
    RandOp,
    RandLikeOp,
    RandintLikeOp,
    RandnOp,
    RandnLikeOp,
    RreluWithNoiseOp,
    UniformOp,
)


_ARG_POS_RE = re.compile(r"^arg(\d+)_")


def _arg_pos(name):
    if not isinstance(name, str):
        return None
    m = _ARG_POS_RE.match(name)
    if not m:
        return None
    return int(m.group(1))


def _insert_full_like_const(
    graph: Graph, op, like_name: str, value, dtype, suffix: str
) -> str:
    full_like = FullLikeOp()
    base_name = f"{op.name}__{suffix}"
    full_like_name = base_name
    ver = 2
    while full_like_name in graph.node_table:
        full_like_name = f"{base_name}_v{ver}"
        ver += 1
    full_like.name = full_like_name
    full_like.add_argument(like_name)
    full_like.add_argument(value)
    full_like.tensor_meta["shape"] = list(op.tensor_meta.get("shape", []))
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
    return full_like.name


def _insert_full_const(
    graph: Graph, op, shape, value, dtype, suffix: str
) -> str:
    full_op = FullOp()
    base_name = f"{op.name}__{suffix}"
    full_name = base_name
    ver = 2
    while full_name in graph.node_table:
        full_name = f"{base_name}_v{ver}"
        ver += 1
    full_op.name = full_name
    full_op.tensor_meta = {"shape": list(shape), "dtype": dtype}
    full_op._arguments = [list(shape), value]
    full_op._keyword_arguments = {}

    insert_pos = graph.body.index(op)
    graph.node_table[full_op.name] = full_op
    graph.body.insert(insert_pos, full_op)
    return full_op.name


def _replace_with_external_call(
    graph: Graph, op, call_func_name: str, call_args: list[str]
):
    call_op = CallExternalOp(
        call_func_name=call_func_name,
        args=call_args,
        args_index=[0] * len(call_args),
        tensor_meta={
            "shape": op.tensor_meta.get("shape", []),
            "dtype": op.tensor_meta.get("dtype", None),
        },
        name=op.name,
    )

    if hasattr(op, "_parents"):
        call_op._parents = op._parents.copy()
    if hasattr(op, "_children"):
        call_op._children = op._children.copy()

    graph.displace_node(op, call_op)
    call_op._arguments = list(call_args)
    call_op._args_index = [0] * len(call_args)
    call_op._keyword_arguments = {}
    call_op._op_type = OpType.Unfusable


def replace_bernoulli_with_runtime_rng(graph: Graph):
    for op in list(graph.body):
        if not isinstance(op, BernoulliOp):
            continue

        dtype = op.tensor_meta.get("dtype", None)
        if dtype != TensorDType.Float32:
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
            a0, a1 = call_args[0], call_args[1]
            if isinstance(a0, str) and isinstance(a1, (float, int)):
                prob_name = _insert_full_like_const(
                    graph,
                    op,
                    a0,
                    float(a1),
                    dtype,
                    "bernoulli_prob_full_like",
                )
                call_args = [prob_name]
                call_func_name = f"buddy_bernoulli_f32_f64rng_r{rank}_1"
            elif isinstance(a1, str) and isinstance(a0, (float, int)):
                prob_name = _insert_full_like_const(
                    graph,
                    op,
                    a1,
                    float(a0),
                    dtype,
                    "bernoulli_prob_full_like",
                )
                call_args = [prob_name]
                call_func_name = f"buddy_bernoulli_f32_f64rng_r{rank}_1"
            else:
                p0 = _arg_pos(call_args[0])
                p1 = _arg_pos(call_args[1])
                if p0 is not None and p1 is not None and p0 > p1:
                    call_args = [call_args[1], call_args[0]]

        call_op = CallExternalOp(
            call_func_name=call_func_name,
            args=call_args,
            args_index=[0] * len(call_args),
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
        call_op._args_index = [0] * len(call_args)
        call_op._keyword_arguments = {}
        call_op._op_type = OpType.Unfusable


def replace_geometric_with_runtime_rng(graph: Graph):
    for op in list(graph.body):
        if not isinstance(op, GeometricOp):
            continue

        dtype = op.tensor_meta.get("dtype", None)
        if dtype != TensorDType.Float32:
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


def replace_randn_with_runtime_rng(graph: Graph):
    for op in list(graph.body):
        if not isinstance(op, RandnOp):
            continue

        shape = op.tensor_meta.get("shape", [])
        rank = len(list(shape)) if shape is not None else 0
        dtype = op.tensor_meta.get("dtype", None)
        if dtype != TensorDType.Float32:
            continue

        call_func_name = f"buddy_randn_f32_like_r{rank}_1"

        like_op = FullOp()
        like_base_name = f"{op.name}__randn_like"
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


def replace_rand_like_with_runtime_rng(graph: Graph):
    for op in list(graph.body):
        if not isinstance(op, RandLikeOp):
            continue

        dtype = op.tensor_meta.get("dtype", None)
        if dtype != TensorDType.Float32:
            continue
        if not op.args or not isinstance(op.args[0], str):
            continue

        shape = op.tensor_meta.get("shape", [])
        rank = len(list(shape)) if shape is not None else 0
        like_name = op.args[0]
        call_func_name = f"buddy_rand_f32_like_r{rank}_1"

        call_op = CallExternalOp(
            call_func_name=call_func_name,
            args=[like_name],
            args_index=[0],
            tensor_meta={
                "shape": op.tensor_meta.get("shape", []),
                "dtype": op.tensor_meta.get("dtype", None),
            },
            name=op.name,
        )

        if hasattr(op, "_parents"):
            call_op._parents = op._parents.copy()
        if hasattr(op, "_children"):
            call_op._children = op._children.copy()

        graph.displace_node(op, call_op)
        call_op._arguments = [like_name]
        call_op._args_index = [0]
        call_op._keyword_arguments = {}
        call_op._op_type = OpType.Unfusable


def replace_randn_like_with_runtime_rng(graph: Graph):
    for op in list(graph.body):
        if not isinstance(op, RandnLikeOp):
            continue

        dtype = op.tensor_meta.get("dtype", None)
        if dtype != TensorDType.Float32:
            continue
        if not op.args or not isinstance(op.args[0], str):
            continue

        shape = op.tensor_meta.get("shape", [])
        rank = len(list(shape)) if shape is not None else 0
        like_name = op.args[0]
        call_func_name = f"buddy_randn_f32_like_r{rank}_1"

        call_op = CallExternalOp(
            call_func_name=call_func_name,
            args=[like_name],
            args_index=[0],
            tensor_meta={
                "shape": op.tensor_meta.get("shape", []),
                "dtype": op.tensor_meta.get("dtype", None),
            },
            name=op.name,
        )

        if hasattr(op, "_parents"):
            call_op._parents = op._parents.copy()
        if hasattr(op, "_children"):
            call_op._children = op._children.copy()

        graph.displace_node(op, call_op)
        call_op._arguments = [like_name]
        call_op._args_index = [0]
        call_op._keyword_arguments = {}
        call_op._op_type = OpType.Unfusable


def replace_normal_with_runtime_rng(graph: Graph):
    for op in list(graph.body):
        if not isinstance(op, NormalOp):
            continue

        dtype = op.tensor_meta.get("dtype", None)
        if dtype != TensorDType.Float32:
            continue

        shape = op.tensor_meta.get("shape", [])
        rank = len(list(shape)) if shape is not None else 0
        args = list(op.args)
        if not args:
            continue

        mean_name = None
        std_name = None

        if (
            len(args) >= 2
            and isinstance(args[0], str)
            and isinstance(args[1], (float, int))
        ):
            if len(args) >= 3 and isinstance(args[2], (float, int)):
                like_name = args[0]
                mean_name = _insert_full_like_const(
                    graph,
                    op,
                    like_name,
                    float(args[1]),
                    dtype,
                    "normal_mean_full_like",
                )
                std_name = _insert_full_like_const(
                    graph,
                    op,
                    like_name,
                    float(args[2]),
                    dtype,
                    "normal_std_full_like",
                )
            else:
                like_name = args[0]
                mean_name = like_name
                std_name = _insert_full_like_const(
                    graph,
                    op,
                    like_name,
                    float(args[1]),
                    dtype,
                    "normal_std_full_like",
                )
        elif (
            len(args) >= 2
            and isinstance(args[0], (float, int))
            and isinstance(args[1], str)
        ):
            like_name = args[1]
            mean_name = _insert_full_like_const(
                graph,
                op,
                like_name,
                float(args[0]),
                dtype,
                "normal_mean_full_like",
            )
            std_name = like_name
        elif (
            len(args) >= 2
            and isinstance(args[0], str)
            and isinstance(args[1], str)
        ):
            mean_name = args[0]
            std_name = args[1]
        elif (
            len(args) >= 3
            and isinstance(args[0], (float, int))
            and isinstance(args[1], (float, int))
            and isinstance(args[2], (list, tuple))
        ):
            shape_like = list(args[2])
            like_name = _insert_full_const(
                graph, op, shape_like, 0.0, dtype, "normal_shape_like"
            )
            mean_name = _insert_full_like_const(
                graph,
                op,
                like_name,
                float(args[0]),
                dtype,
                "normal_mean_full_like",
            )
            std_name = _insert_full_like_const(
                graph,
                op,
                like_name,
                float(args[1]),
                dtype,
                "normal_std_full_like",
            )
        else:
            continue

        if mean_name is None or std_name is None:
            continue

        call_func_name = f"buddy_normal_f32_r{rank}_2"
        _replace_with_external_call(
            graph, op, call_func_name, [mean_name, std_name]
        )


def replace_log_normal_with_runtime_rng(graph: Graph):
    for op in list(graph.body):
        if not isinstance(op, LogNormalOp):
            continue

        dtype = op.tensor_meta.get("dtype", None)
        if dtype != TensorDType.Float32:
            continue

        if not op.args or not isinstance(op.args[0], str):
            continue

        like_name = op.args[0]
        mean = float(op.args[1]) if len(op.args) >= 2 else 1.0
        std = float(op.args[2]) if len(op.args) >= 3 else 2.0

        mean_name = _insert_full_like_const(
            graph, op, like_name, mean, dtype, "log_normal_mean_full_like"
        )
        std_name = _insert_full_like_const(
            graph, op, like_name, std, dtype, "log_normal_std_full_like"
        )

        shape = op.tensor_meta.get("shape", [])
        rank = len(list(shape)) if shape is not None else 0
        call_func_name = f"buddy_log_normal_f32_r{rank}_2"
        _replace_with_external_call(
            graph, op, call_func_name, [mean_name, std_name]
        )


def replace_poisson_with_runtime_rng(graph: Graph):
    for op in list(graph.body):
        if not isinstance(op, PoissonOp):
            continue

        dtype = op.tensor_meta.get("dtype", None)
        if dtype != TensorDType.Float32:
            continue

        if not op.args or not isinstance(op.args[0], str):
            continue

        shape = op.tensor_meta.get("shape", [])
        rank = len(list(shape)) if shape is not None else 0
        call_func_name = f"buddy_poisson_f32_r{rank}_1"
        _replace_with_external_call(graph, op, call_func_name, [op.args[0]])


def replace_multinomial_with_runtime_rng(graph: Graph):
    for op in list(graph.body):
        if not isinstance(op, MultinomialOp):
            continue

        dtype = op.tensor_meta.get("dtype", None)
        if dtype != TensorDType.Int64:
            continue

        if len(op.args) < 2 or not isinstance(op.args[0], str):
            continue

        num_samples = op.args[1]
        if not isinstance(num_samples, (int, float)):
            continue

        num_samples_name = _insert_full_const(
            graph,
            op,
            [],
            int(num_samples),
            TensorDType.Int64,
            "multinomial_num_samples_scalar",
        )

        shape = op.tensor_meta.get("shape", [])
        rank = len(list(shape)) if shape is not None else 0
        call_func_name = f"buddy_multinomial_f32_i64_r{rank}_2"
        _replace_with_external_call(
            graph, op, call_func_name, [op.args[0], num_samples_name]
        )


def replace_randint_like_with_runtime_rng(graph: Graph):
    for op in list(graph.body):
        if not isinstance(op, RandintLikeOp):
            continue

        dtype = op.tensor_meta.get("dtype", None)
        if dtype != TensorDType.Int64:
            continue

        if not op.args or not isinstance(op.args[0], str):
            continue

        like_name = op.args[0]
        low_val = 0
        high_name = None

        if (
            len(op.args) >= 3
            and isinstance(op.args[1], (int, float))
            and isinstance(op.args[2], (int, float))
        ):
            low_val = int(op.args[1])
            high_name = _insert_full_const(
                graph,
                op,
                [],
                int(op.args[2]),
                TensorDType.Int64,
                "randint_high_scalar",
            )
        elif len(op.args) >= 2 and isinstance(op.args[1], (int, float)):
            high_name = _insert_full_const(
                graph,
                op,
                [],
                int(op.args[1]),
                TensorDType.Int64,
                "randint_high_scalar",
            )
        elif len(op.args) >= 2 and isinstance(op.args[1], str):
            high_name = op.args[1]
        else:
            continue

        low_name = _insert_full_const(
            graph, op, [], int(low_val), TensorDType.Int64, "randint_low_scalar"
        )

        shape = op.tensor_meta.get("shape", [])
        rank = len(list(shape)) if shape is not None else 0
        call_func_name = f"buddy_randint_like_i64_r{rank}_3"
        _replace_with_external_call(
            graph, op, call_func_name, [like_name, low_name, high_name]
        )


def replace_uniform_with_runtime_rng(graph: Graph):
    for op in list(graph.body):
        if not isinstance(op, UniformOp):
            continue

        dtype = op.tensor_meta.get("dtype", None)
        if dtype != TensorDType.Float32:
            continue

        if not op.args or not isinstance(op.args[0], str):
            continue

        like_name = op.args[0]
        from_v = float(op.args[1]) if len(op.args) >= 2 else 0.0
        to_v = float(op.args[2]) if len(op.args) >= 3 else 1.0

        from_name = _insert_full_const(
            graph, op, [], from_v, TensorDType.Float32, "uniform_from_scalar"
        )
        to_name = _insert_full_const(
            graph, op, [], to_v, TensorDType.Float32, "uniform_to_scalar"
        )

        shape = op.tensor_meta.get("shape", [])
        rank = len(list(shape)) if shape is not None else 0
        call_func_name = f"buddy_uniform_f32_r{rank}_3"
        _replace_with_external_call(
            graph, op, call_func_name, [like_name, from_name, to_name]
        )


def replace_cauchy_with_runtime_rng(graph: Graph):
    for op in list(graph.body):
        if not isinstance(op, CauchyOp):
            continue

        dtype = op.tensor_meta.get("dtype", None)
        if dtype != TensorDType.Float32:
            continue

        if not op.args or not isinstance(op.args[0], str):
            continue

        like_name = op.args[0]
        median_v = float(op.args[1]) if len(op.args) >= 2 else 0.0
        sigma_v = float(op.args[2]) if len(op.args) >= 3 else 1.0

        median_name = _insert_full_const(
            graph, op, [], median_v, TensorDType.Float32, "cauchy_median_scalar"
        )
        sigma_name = _insert_full_const(
            graph, op, [], sigma_v, TensorDType.Float32, "cauchy_sigma_scalar"
        )

        shape = op.tensor_meta.get("shape", [])
        rank = len(list(shape)) if shape is not None else 0
        call_func_name = f"buddy_cauchy_f32_r{rank}_3"
        _replace_with_external_call(
            graph, op, call_func_name, [like_name, median_name, sigma_name]
        )


def replace_rrelu_with_noise_with_runtime_rng(graph: Graph):
    for op in list(graph.body):
        if not isinstance(op, RreluWithNoiseOp):
            continue

        dtype = op.tensor_meta.get("dtype", None)
        shape = op.tensor_meta.get("shape", [])

        is_functional = False
        if isinstance(dtype, tuple):
            if len(dtype) != 2:
                continue
            if (
                dtype[0] != TensorDType.Float32
                or dtype[1] != TensorDType.Float32
            ):
                continue
            is_functional = True
            rank = len(list(shape[0])) if shape and shape[0] is not None else 0
        else:
            if dtype != TensorDType.Float32:
                continue
            rank = len(list(shape)) if shape is not None else 0

        if len(op.args) < 2:
            continue
        if not isinstance(op.args[0], str) or not isinstance(op.args[1], str):
            continue

        self_name = op.args[0]
        noise_name = op.args[1]
        lower = float(op.args[2]) if len(op.args) >= 3 else 0.125
        upper = float(op.args[3]) if len(op.args) >= 4 else (1.0 / 3.0)
        training = bool(op.args[4]) if len(op.args) >= 5 else False

        lower_name = _insert_full_const(
            graph, op, [], lower, TensorDType.Float32, "rrelu_lower_scalar"
        )
        upper_name = _insert_full_const(
            graph, op, [], upper, TensorDType.Float32, "rrelu_upper_scalar"
        )
        training_name = _insert_full_const(
            graph,
            op,
            [],
            1 if training else 0,
            TensorDType.Int64,
            "rrelu_training_scalar",
        )

        if is_functional:
            call_func_name = f"buddy_rrelu_with_noise_functional_f32_r{rank}_5"
        else:
            call_func_name = f"buddy_rrelu_with_noise_f32_r{rank}_5"
        _replace_with_external_call(
            graph,
            op,
            call_func_name,
            [self_name, noise_name, lower_name, upper_name, training_name],
        )


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
