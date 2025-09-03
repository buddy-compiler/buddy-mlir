# ===- func.py -----------------------------------------------------------------
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
# The registry of mappings from Buddy node to MLIR func dialect operations.
#
# ===---------------------------------------------------------------------------

from typing import Tuple
import functools
from mlir.dialects import func, memref
from mlir import ir
from ..graph import FuncOp, CallOp, PlaceholderOp
from .utils import *


def func_op(node: FuncOp, symbol_table: Dict[Tuple[str, int], ir.Operation]):
    """
    Import the buddy FuncOp.
    From Buddy FuncOp to MLIR FUNC Func operation.
    """
    arguments = []
    for arg in node.args:
        shape = list(arg.shape)
        mlir_dtype = mlir_element_type_get(arg.dtype)
        stride = []
        for dim, dim_size in enumerate(shape):
            stride.append(
                functools.reduce(lambda x, y: x * y, shape[dim + 1 :] + [1])
            )
        memref_attr = ir.Attribute.parse(
            "strided<{}, offset: ?>".format(stride)
        )
        arguments.append(ir.MemRefType.get(shape, mlir_dtype, memref_attr))
    results = []
    for i, shape in enumerate(node.tensor_meta["shape"]):
        mlir_dtype = mlir_element_type_get(node.tensor_meta["dtype"][i])
        results.append(ir.MemRefType.get(shape, mlir_dtype))
    function_type = ir.FunctionType.get(inputs=arguments, results=results)
    op = func.FuncOp(name=node.name, type=function_type, visibility="private")
    return op


def call_op(node: CallOp, symbol_table: Dict[Tuple[str, int], ir.Operation]):
    """
    Import the buddy CallOp.
    From Buddy CallOp to MLIR FUNC call operation.
    """
    arguments = []
    for i, arg in enumerate(node.args):
        input_node = symbol_table.get((str(arg), node._args_index[i]))
        memref_type = ir.MemRefType(input_node.type)
        stride = []
        shape = memref_type.shape
        for dim, dim_size in enumerate(shape):
            stride.append(
                functools.reduce(lambda x, y: x * y, shape[dim + 1 :] + [1])
            )
        memref_attr = ir.Attribute.parse(
            "strided<{}, offset: ?>".format(stride)
        )
        dest = ir.MemRefType.get(shape, memref_type.element_type, memref_attr)
        cast_op = memref.CastOp(dest, input_node)
        arguments.append(cast_op)
    results = []
    for i, shape in enumerate(node.tensor_meta["shape"]):
        mlir_dtype = mlir_element_type_get(node.tensor_meta["dtype"][i])
        results.append(ir.MemRefType.get(shape, mlir_dtype))
    func_symbol = ir.FlatSymbolRefAttr.get(node.call_func_name)
    op = func.call(results, func_symbol, arguments)
    return op


def param_extract(
    node: PlaceholderOp,
    offset,
    params_mlir_node,
):
    """
    Extract param from packed params.

    Note: This function extract slice from packed params tensor, and expand
    shape by param node shape.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the memref.expand_shape op.
    """
    dtype_mapping = {
        TensorDType.Float16: ir.F16Type.get(),
        TensorDType.Float32: ir.F32Type.get(),
        TensorDType.Int64: ir.IntegerType.get_signless(64),
    }
    memref_element_type = dtype_mapping[node.tensor_meta["dtype"]]
    if len(node.tensor_meta["shape"]) == 0:
        output_shape = [1]
    else:
        output_shape = list(node.tensor_meta["shape"])
    static_output_shape = ir.DenseI64ArrayAttr.get(output_shape)
    subview_size = functools.reduce(lambda x, y: x * y, output_shape)
    offset_attr = ir._denseI64ArrayAttr([offset], None)
    size_attr = ir._denseI64ArrayAttr([subview_size], None)
    stride = [1]
    stride_attr = ir._denseI64ArrayAttr(stride, None)
    memref_attr = ir.Attribute.parse("strided<[1], offset: {}>".format(offset))
    if offset == 0:
        memref_type = ir.MemRefType.get([subview_size], memref_element_type)
    else:
        memref_type = ir.MemRefType.get(
            [subview_size], memref_element_type, memref_attr
        )
    memref_subview_op = memref.SubViewOp(
        memref_type,
        params_mlir_node,
        [],
        [],
        [],
        offset_attr,
        size_attr,
        stride_attr,
    )
    if len(output_shape) == 1:
        return memref_subview_op
    stride = []
    for dim, dim_size in enumerate(output_shape):
        stride.append(
            functools.reduce(lambda x, y: x * y, output_shape[dim + 1 :] + [1])
        )
    memref_attr = ir.Attribute.parse(
        "strided<{}, offset: {}>".format(stride, offset)
    )
    if offset == 0:
        memref_type = ir.MemRefType.get(output_shape, memref_element_type)
    else:
        memref_type = ir.MemRefType.get(
            output_shape, memref_element_type, memref_attr
        )
    axis = ir.ArrayAttr.get(
        [
            ir.IntegerAttr.get(ir.IntegerType.get_signless(64), i)
            for i in range(len(output_shape))
        ],
        None,
    )
    axis = ir.ArrayAttr.get([axis], None)
    expand_shape_op = memref.ExpandShapeOp(
        memref_type, memref_subview_op.result, axis, [], static_output_shape
    )
    return expand_shape_op


ops_registry = {
    "FuncOp": func_op,
    "CallOp": call_op,
    "param.extract": param_extract,
}
