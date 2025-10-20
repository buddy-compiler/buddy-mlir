# ===- onednn_replace.py -------------------------------------------------------
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
# Transform pass to replace MatmulOp with oneDNN library calls.
#
# ===---------------------------------------------------------------------------

from .. import Graph
from ..operation import MatmulOp, CallOp, CallExternalOp, OpType


def replace_matmul_with_onednn(graph: Graph):
    """
    Replace MatmulOp nodes with CallExternalOp nodes that call oneDNN library functions.

    This transform pass identifies MatmulOp operations in the graph and replaces
    them with CallExternalOp operations that invoke the oneDNN matmul implementation.

    Args:
        graph (Graph): The computation graph to transform.

    Returns:
        None: Modifies the input graph in place.

    Example:
        Before:
            %result = MatmulOp(%input, %weight)

        After:
            %result = CallExternalOp("onednn_matmul_f32", %input, %weight)
    """
    # Iterate over a copy of the body to avoid modification during iteration
    for op in list(graph.body):
        if isinstance(op, MatmulOp):
            # Get output shape and dtype from the MatmulOp
            output_shape = op.tensor_meta.get("shape", [])
            output_dtype = op.tensor_meta.get("dtype", "float32")

            # Create a new CallExternalOp to replace the MatmulOp
            call_op = CallExternalOp(
                call_func_name="onednn_matmul_f32",
                args=[op.args[0], op.args[1]],
                args_index=[0, 0],
                tensor_meta={
                    "shape": output_shape,
                    "dtype": output_dtype,
                },
                name=f"onednn_{op.name}",
            )

            # Copy other important attributes
            if hasattr(op, '_parents'):
                call_op._parents = op._parents.copy()
            if hasattr(op, '_children'):
                call_op._children = op._children.copy()

            # Replace the MatmulOp with CallExternalOp in the graph
            graph.displace_node(op, call_op)

            # IMPORTANT: displace_node overwrites _op_type, so we need to restore it
            call_op._op_type = OpType.Unfusable


def replace_matmul_with_onednn_selective(graph: Graph, min_size=None):
    """
    Selectively replace MatmulOp nodes with oneDNN calls based on size criteria.

    This variant only replaces MatmulOp operations that meet certain size
    requirements, allowing for hybrid execution where small matmuls use TOSA
    and large matmuls use oneDNN.

    Args:
        graph (Graph): The computation graph to transform.
        min_size (int, optional): Minimum matrix size to use oneDNN. If None,
                                  replace all MatmulOp operations.

    Returns:
        None: Modifies the input graph in place.
    """
    for op in list(graph.body):
        if isinstance(op, MatmulOp):
            # Check if we should replace this matmul
            should_replace = True

            if min_size is not None and hasattr(op, 'tensor_meta'):
                shape = op.tensor_meta.get('shape', [])
                if len(shape) >= 2:
                    # For 2D: [M, N], for 3D: [B, M, N]
                    matrix_size = shape[-2] * shape[-1]
                    if matrix_size < min_size:
                        should_replace = False
                        print(f"[oneDNN Transform] Skipping {op.name} (size {matrix_size} < {min_size})")

            if should_replace:
                # Get output shape and dtype from the MatmulOp
                output_shape = op.tensor_meta.get("shape", [])
                output_dtype = op.tensor_meta.get("dtype", "float32")

                # Create a new CallExternalOp to replace the MatmulOp
                call_op = CallExternalOp(
                    call_func_name="onednn_matmul_f32",
                    args=[op.args[0], op.args[1]],
                    args_index=[0, 0],
                    tensor_meta={
                        "shape": output_shape,
                        "dtype": output_dtype,
                    },
                    name=f"onednn_{op.name}",
                )

                if hasattr(op, '_parents'):
                    call_op._parents = op._parents.copy()
                if hasattr(op, '_children'):
                    call_op._children = op._children.copy()

                graph.displace_node(op, call_op)

                # IMPORTANT: displace_node overwrites _op_type, so we need to restore it
                call_op._op_type = OpType.Unfusable

                print(f"[oneDNN Transform] Replaced {op.name} (MatmulOp) with {call_op.name} (CallExternalOp)")
                if hasattr(op, 'tensor_meta') and 'shape' in op.tensor_meta:
                    print(f"  Output shape: {op.tensor_meta['shape']}")

