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

    Note: This function records transpose information in graph._transposed_params
    which maps parameter indices to transpose operations that need to be applied
    when storing the weight data.
    """
    # Initialize transpose tracking if not exists
    if not hasattr(graph, "_transposed_params"):
        graph._transposed_params = {}

    # We'll iterate through all nodes to find transpose nodes
    nodes_to_remove = []
    # Store information about eliminated nodes for summary
    eliminated_info = []

    for node in graph.body:
        is_transpose_op = isinstance(node, TransposeOp)
        is_t_op = isinstance(node, TOp)
        is_permute_op = isinstance(node, PermuteOp)

        if not (is_transpose_op or is_t_op or is_permute_op):
            continue

        # Get input argument name
        if len(node.args) < 1:
            continue
        input_arg_name = str(node.args[0])

        # Check if the input is in the graph
        if input_arg_name not in graph.node_table:
            continue

        input_node = graph.node_table[input_arg_name]

        # Check if input is a PlaceholderOp (function argument/weight)
        if not isinstance(input_node, PlaceholderOp):
            continue

        # Check if this argument is only used in this transpose operation
        # (i.e., has only one child and it's this transpose node)
        if len(input_node._children) != 1:
            continue

        if input_node._children[0] != node.name:
            continue

        # Get the current shape
        current_shape = list(input_node.tensor_meta["shape"])

        # Calculate the transposed shape based on the operation type
        if is_t_op:
            # TOp always transposes 2D tensors using [1, 0] permutation
            if len(current_shape) != 2:
                continue
            transposed_shape = [current_shape[1], current_shape[0]]
        elif is_permute_op:
            # PermuteOp uses a permutation list
            if len(node.args) < 2:
                continue

            # Get permutation list (second argument)
            perm = node.args[1]
            if not isinstance(perm, (list, tuple)):
                continue

            perm = list(perm)

            # Validate permutation
            if len(perm) != len(current_shape):
                continue

            # Check if it's a valid permutation (contains 0 to len-1 exactly once)
            if set(perm) != set(range(len(current_shape))):
                continue

            # Calculate the transposed shape
            transposed_shape = [current_shape[i] for i in perm]
        elif is_transpose_op:
            # TransposeOp swaps two specific dimensions
            if len(node.args) < 3:
                continue

            # Get transpose dimensions
            dim1 = int(node.args[1])
            dim2 = int(node.args[2])

            # Validate dimensions
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

            # Calculate the transposed shape
            transposed_shape = current_shape.copy()
            transposed_shape[dim1], transposed_shape[dim2] = (
                transposed_shape[dim2],
                transposed_shape[dim1],
            )
        else:
            continue

        # Verify the transposed shape matches the output shape
        node_output_shape = list(node.tensor_meta["shape"])
        if node_output_shape != transposed_shape:
            continue

        # IMPORTANT: Only optimize weights (parameters), not inputs
        # Check if this PlaceholderOp is in _fake_params (it's a weight)
        # If not in _fake_params, it's a real input and should NOT be optimized
        param_idx = None
        transpose_info = None

        # First, try to find this node in _fake_params by matching shape and dtype
        # This ensures we only optimize weights, not inputs
        current_shape_tuple = tuple(current_shape)
        node_dtype = input_node.tensor_meta.get("dtype")

        for idx, tensor_meta in enumerate(graph._fake_params):
            if (
                list(tensor_meta.shape) == current_shape
                and tensor_meta.dtype == node_dtype
            ):
                param_idx = idx
                break

        # If not found in _fake_params, this is an input, skip optimization
        if param_idx is None:
            continue

        # Now we know this is a weight, proceed with optimization
        # Update the input node's shape to the transposed shape
        input_node.tensor_meta["shape"] = transposed_shape

        # Prepare transpose operation information
        if is_t_op:
            transpose_info = {"type": "t", "dims": [1, 0]}
        elif is_permute_op:
            transpose_info = {"type": "permute", "perm": perm}
        elif is_transpose_op:
            transpose_info = {"type": "transpose", "dims": [dim1, dim2]}

        # Update the corresponding TensorMeta in graph._fake_params
        # We already found param_idx above, so update it
        if param_idx is not None:
            tensor_meta = graph._fake_params[param_idx]
            # Verify it matches before updating (safety check)
            if (
                list(tensor_meta.shape) == current_shape
                and tensor_meta.dtype == node_dtype
            ):
                tensor_meta.shape = tuple(transposed_shape)

                # Directly modify the actual parameter tensor if available
                # This is the key: modify weights in-place during transform
                if (
                    hasattr(graph, "_params_ref")
                    and graph._params_ref is not None
                ):
                    if param_idx < len(graph._params_ref):
                        param_tensor = graph._params_ref[param_idx]
                        # For bfloat16, convert to float32 for transpose operations
                        is_bf16 = param_tensor.dtype == torch.bfloat16
                        if is_bf16:
                            param_tensor_data = (
                                param_tensor.detach().float().clone()
                            )
                        else:
                            param_tensor_data = param_tensor.detach().clone()

                        # Apply transpose operation
                        if transpose_info["type"] == "t":
                            # 2D transpose [1, 0]
                            param_tensor_data = param_tensor_data.T
                        elif transpose_info["type"] == "transpose":
                            # Swap two dimensions
                            dim1, dim2 = transpose_info["dims"]
                            param_tensor_data = param_tensor_data.swapaxes(
                                dim1, dim2
                            )
                        elif transpose_info["type"] == "permute":
                            # Apply permutation
                            param_tensor_data = param_tensor_data.permute(
                                transpose_info["perm"]
                            )

                        # Convert back to bf16 if needed
                        if is_bf16:
                            param_tensor_data = param_tensor_data.bfloat16()

                        # Update the parameter tensor
                        # If shape changed (due to transpose), we need to replace the tensor
                        if param_tensor.shape != param_tensor_data.shape:
                            # Shape changed, need to replace the entire parameter tensor
                            # Check if it's a Parameter or regular Tensor
                            from torch.nn import Parameter

                            if isinstance(param_tensor, Parameter):
                                # Create a new Parameter with the transposed data
                                new_param = Parameter(
                                    param_tensor_data,
                                    requires_grad=param_tensor.requires_grad,
                                )
                                # Replace in the params list
                                graph._params_ref[param_idx] = new_param
                            else:
                                # Regular tensor, replace directly
                                graph._params_ref[param_idx] = param_tensor_data
                        else:
                            # Shape unchanged, can copy in-place
                            param_tensor.data.copy_(param_tensor_data)

                # Record the transpose operation (for reference, though params are already modified)
                graph._transposed_params[param_idx] = transpose_info
            else:
                # Shape mismatch - this shouldn't happen, but skip if it does
                # This could happen if the same PlaceholderOp appears multiple times
                # or if the matching logic had an error
                continue

        # Save children list before modifying
        transpose_children = list(node._children)

        # Replace all uses of this transpose node with the input node
        # Update children to use the input node directly
        for child_name in transpose_children:
            if child_name not in graph.node_table:
                continue
            child_node = graph.node_table[child_name]
            # Replace the argument in the child node
            for i, arg in enumerate(child_node.args):
                if str(arg) == node.name:
                    child_node.args[i] = input_arg_name
            # Update parent relationship
            if node.name in child_node._parents:
                idx = child_node._parents.index(node.name)
                child_node._parents[idx] = input_arg_name
                input_node._children.append(child_name)

        # Remove transpose node from input_node's children
        # (it was the only child before, now we're replacing it with transpose's children)
        # Check if it exists before removing to avoid errors
        if node.name in input_node._children:
            input_node._children.remove(node.name)

        # Clear transpose node's children (already moved to input_node)
        # Save info before clearing
        node_children = (
            list(node._children) if hasattr(node, "_children") else []
        )
        node_parents = list(node._parents) if hasattr(node, "_parents") else []
        node._children.clear()

        # Store information about this eliminated node for summary
        eliminated_info.append(
            {
                "node": node,
                "node_name": node.name,
                "op_type": type(node).__name__,
                "input_shape": current_shape,
                "output_shape": transposed_shape,
                "dtype": node_dtype,
                "args": list(node.args) if hasattr(node, "args") else [],
                "children": node_children,
                "parents": node_parents,
                "input_node_name": input_arg_name,
                "input_node_shape_after": transposed_shape,
                "param_idx": param_idx,
                "transpose_info": transpose_info,
            }
        )

        # Remove transpose node from graph
        nodes_to_remove.append(node)

    # Remove the transpose nodes
    for node in nodes_to_remove:
        if node.name in graph.node_table:
            del graph.node_table[node.name]
        if node in graph.body:
            graph.body.remove(node)

    # Print summary of eliminated transpose operations
    if eliminated_info:
        print("\n=== Eliminate Transpose Summary ===")
        print(f"Total transpose nodes eliminated: {len(eliminated_info)}")
        print("\nDetailed information:")
        for i, info in enumerate(eliminated_info, 1):
            print(f"\n  [{i}] Node: {info['node_name']}")
            print(f"      Operation Type: {info['op_type']}")
            print(f"      Input Shape (before): {info['input_shape']}")
            print(f"      Output Shape: {info['output_shape']}")
            print(f"      Dtype: {info['dtype']}")
            print(f"      Arguments: {info['args']}")
            print(f"      Children: {info['children']}")
            print(f"      Parents: {info['parents']}")
            print(f"      Input Node: {info['input_node_name']}")
            print(
                f"      Input Node Shape (after optimization): {info['input_node_shape_after']}"
            )
            if info["param_idx"] is not None:
                print(f"      Parameter Index: {info['param_idx']}")
                print(f"      Transpose Info: {info['transpose_info']}")
        print("\n=== End of Summary ===\n")
