# ===- eliminate_weight_transpose.py -------------------------------------------
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
# Eliminate transpose operations on function arguments (weights) that are only
# used once. Instead of transposing the weight, we directly use the weight with
# the transposed shape.
#
# ===---------------------------------------------------------------------------

from .. import Graph
from ..operation import *
import torch


def eliminate_transpose(graph: Graph):
    """
    Eliminate transpose operations on function arguments (weights) that are only
    used once. This optimization directly modifies the weight shape instead of
    generating a transpose operation.

    This handles:
    - TransposeOp: transpose.int operations that swap two specific dimensions
    - TOp: t.default operations that transpose 2D tensors (always [1, 0] perm)
    - PermuteOp: permute operations with a permutation list

    Args:
        graph (Graph): The Graph to be optimized.
    """
    if not hasattr(graph, "_transposed_params"):
        graph._transposed_params = {}

    nodes_to_remove = []
    param_changed = set()

    for node in graph.body:
        is_transpose_op = isinstance(node, TransposeOp)
        is_t_op = isinstance(node, TOp)
        is_permute_op = isinstance(node, PermuteOp)

        if not (is_transpose_op or is_t_op or is_permute_op):
            continue

        if len(node.args) < 1:
            continue
        input_arg_name = str(node.args[0])

        if input_arg_name not in graph.node_table:
            continue

        input_node = graph.node_table[input_arg_name]

        if not isinstance(input_node, PlaceholderOp):
            continue

        if (
            len(input_node._children) != 1
            or input_node._children[0] != node.name
        ):
            continue

        current_shape = list(input_node.tensor_meta["shape"])

        if is_t_op:
            if len(current_shape) != 2:
                continue
            transposed_shape = [current_shape[1], current_shape[0]]
        elif is_permute_op:
            if len(node.args) < 2:
                continue

            perm = node.args[1]
            if not isinstance(perm, (list, tuple)):
                continue

            perm = list(perm)

            if len(perm) != len(current_shape) or set(perm) != set(
                range(len(current_shape))
            ):
                continue

            transposed_shape = [current_shape[i] for i in perm]
        elif is_transpose_op:
            if len(node.args) < 3:
                continue

            dim1 = int(node.args[1])
            dim2 = int(node.args[2])

            if dim1 < 0:
                dim1 += len(current_shape)
            if dim2 < 0:
                dim2 += len(current_shape)

            if (
                dim1 < 0
                or dim1 >= len(current_shape)
                or dim2 < 0
                or dim2 >= len(current_shape)
            ):
                continue

            transposed_shape = current_shape.copy()
            transposed_shape[dim1], transposed_shape[dim2] = (
                transposed_shape[dim2],
                transposed_shape[dim1],
            )
        else:
            continue

        node_output_shape = list(node.tensor_meta["shape"])
        if node_output_shape != transposed_shape:
            continue

        param_idx = None
        transpose_info = None
        node_dtype = input_node.tensor_meta.get("dtype")

        for idx, tensor_meta in enumerate(graph.params_shapes):
            if (
                list(tensor_meta.shape) == current_shape
                and tensor_meta.dtype == node_dtype
                and idx not in param_changed
                and idx > 0
            ):
                param_idx = idx
                param_changed.add(idx)
                break

        if param_idx is None:
            continue

        input_node.tensor_meta["shape"] = torch.Size(list(transposed_shape))

        if is_t_op:
            transpose_info = {"type": "t", "dims": [1, 0]}
        elif is_permute_op:
            transpose_info = {"type": "permute", "perm": perm}
        elif is_transpose_op:
            transpose_info = {"type": "transpose", "dims": [dim1, dim2]}

        if param_idx is not None:
            tensor_meta = graph.params_shapes[param_idx]
            if (
                list(tensor_meta.shape) == current_shape
                and tensor_meta.dtype == node_dtype
            ):
                tensor_meta.shape = torch.Size(list(transposed_shape))

                if (
                    hasattr(graph, "_params_ref")
                    and graph._params_ref is not None
                ):
                    if param_idx < len(graph._params_ref):
                        param_tensor = graph._params_ref[param_idx]
                        is_bf16 = param_tensor.dtype == torch.bfloat16
                        if is_bf16:
                            param_tensor_data = (
                                param_tensor.detach().float().clone()
                            )
                        else:
                            param_tensor_data = param_tensor.detach().clone()

                        if transpose_info["type"] == "t":
                            param_tensor_data = param_tensor_data.T
                        elif transpose_info["type"] == "transpose":
                            dim1, dim2 = transpose_info["dims"]
                            param_tensor_data = param_tensor_data.swapaxes(
                                dim1, dim2
                            )
                        elif transpose_info["type"] == "permute":
                            param_tensor_data = param_tensor_data.permute(
                                transpose_info["perm"]
                            )

                        if is_bf16:
                            param_tensor_data = param_tensor_data.bfloat16()

                        from torch.nn import Parameter

                        if isinstance(param_tensor, Parameter):
                            new_param = Parameter(
                                param_tensor_data,
                                requires_grad=param_tensor.requires_grad,
                            )
                            graph._params_ref[param_idx] = new_param
                        else:
                            graph._params_ref[param_idx] = param_tensor_data

                graph._transposed_params[param_idx] = transpose_info

        transpose_children = list(node._children)

        for child_name in transpose_children:
            if child_name not in graph.node_table:
                continue
            child_node = graph.node_table[child_name]
            for i, arg in enumerate(child_node.args):
                if str(arg) == node.name:
                    child_node.args[i] = input_arg_name
            if node.name in child_node._parents:
                idx = child_node._parents.index(node.name)
                child_node._parents[idx] = input_arg_name
                input_node._children.append(child_name)

        if node.name in input_node._children:
            input_node._children.remove(node.name)

        node._children.clear()
        nodes_to_remove.append(node)

    for node in nodes_to_remove:
        if node.name in graph.node_table:
            del graph.node_table[node.name]
        if node in graph.body:
            graph.body.remove(node)
