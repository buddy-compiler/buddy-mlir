# ===- runtime_matmul_replace.py -----------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# Replace MatmulOp with the standard runtime ABI: buddy_matmul_f32
# (MemRef<float,2> in buddy/Core/Container.h). Backends link one implementation
# (e.g. buddy_matmul_torq for TORQ-Tile).
#
# ===---------------------------------------------------------------------------

from .. import Graph
from ..operation import CallExternalOp, MatmulOp, OpType


def replace_matmul_with_buddy_runtime(graph: Graph):
    """
    Replace MatmulOp with CallExternalOp(\"buddy_matmul_f32\", ...).

    The MLIR lowering uses llvm.emit_c_interface → _mlir_ciface_buddy_matmul_f32
    """
    for op in list(graph.body):
        if isinstance(op, MatmulOp):
            output_shape = op.tensor_meta.get("shape", [])
            output_dtype = op.tensor_meta.get("dtype", "float32")

            call_op = CallExternalOp(
                call_func_name="buddy_matmul_f32",
                args=[op.args[0], op.args[1]],
                args_index=[0, 0],
                tensor_meta={
                    "shape": output_shape,
                    "dtype": output_dtype,
                },
                name=f"buddy_matmul_{op.name}",
            )

            if hasattr(op, "_parents"):
                call_op._parents = op._parents.copy()
            if hasattr(op, "_children"):
                call_op._children = op._children.copy()

            graph.displace_node(op, call_op)
            call_op._op_type = OpType.Unfusable
