# ===- tosa.py -----------------------------------------------------------------
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
# The registry of mappings from Torch node to MLIR tosa dialect operations.
#
# ===---------------------------------------------------------------------------

import array
from typing import Dict, List, Union, Tuple

import mlir.ir as ir
from mlir.dialects import tosa, tensor
import torch


def _normalize_binary_operator_shape(shp1, shp2):
    shp1 = list(shp1)
    shp2 = list(shp2)
    while len(shp1) < len(shp2):
        shp1.insert(0, 1)
    while len(shp2) < len(shp1):
        shp2.insert(0, 1)

    return shp1, shp2


def _gen_arith_binary_op(input1, input2, op_func):
    input1, input2 = _normalize_binary_operator_args(input1, input2)

    input1_shape = ir.RankedTensorType(input1.type).shape
    input2_shape = ir.RankedTensorType(input2.type).shape

    norm_input1_shape, norm_input2_shape = _normalize_binary_operator_shape(
        input1_shape, input2_shape
    )

    broadcasted_result_shp = []
    for dim1, dim2 in zip(norm_input1_shape, norm_input2_shape):
        broadcasted_result_shp.append(max(dim1, dim2))
    if input1_shape != norm_input1_shape:
        input1 = tosa.ReshapeOp(
            input1, memoryview(array.array("i", norm_input1_shape))
        ).result
    if input2_shape != norm_input2_shape:
        input2 = tosa.ReshapeOp(
            input2, memoryview(array.array("i", norm_input2_shape))
        ).result

    result_element_type = ir.RankedTensorType(input1.type).element_type
    result_tensor_type = ir.RankedTensorType.get(
        broadcasted_result_shp, result_element_type
    )
    op = op_func(result_tensor_type, input1, input2)
    return op


def _scalar_to_tensor(
    scalar: Union[float, int], element_type: ir.Type, shape: List[int]
):
    element = (
        ir.FloatAttr.get(element_type, float(scalar))
        if str(element_type) == "f32"
        else ir.IntegerAttr.get(element_type, int(scalar))
    )
    attr = ir.DenseElementsAttr.get_splat(
        ir.RankedTensorType.get(shape, element_type), element
    )
    return tosa.ConstOp(attr).results[0]


def _normalize_binary_operator_args(arg1, arg2):
    if isinstance(arg1, ir.Value) and (
        isinstance(arg2, float) or isinstance(arg2, int)
    ):
        arg2 = _scalar_to_tensor(
            arg2,
            ir.RankedTensorType(arg1.type).element_type,
            ir.RankedTensorType(arg1.type).shape,
        )
        return arg1, arg2
    elif isinstance(arg2, ir.Value) and (
        isinstance(arg1, float) or isinstance(arg1, int)
    ):
        arg1 = _scalar_to_tensor(
            arg1,
            ir.RankedTensorType(arg2.type).element_type,
            ir.RankedTensorType(arg2.type).shape,
        )
        return arg1, arg2
    elif isinstance(arg1, ir.Value) and isinstance(arg2, ir.Value):
        return arg1, arg2
    elif (isinstance(arg1, float) or isinstance(arg1, int)) and (
        isinstance(arg2, float) or isinstance(arg2, int)
    ):
        if isinstance(arg1, float) or isinstance(arg2, float):
            arg1 = _scalar_to_tensor(arg1, ir.F32Type.get(), [1])
            arg2 = _scalar_to_tensor(arg2, ir.F32Type.get(), [1])
        else:
            arg1 = _scalar_to_tensor(arg1, ir.IntegerType.get_signless(32), [1])
            arg2 = _scalar_to_tensor(arg2, ir.IntegerType.get_signless(32), [1])
        return arg1, arg2
    else:
        raise ValueError(
            "Invalid input types %s and %s" % (type(arg1), type(arg2))
        )


def add_op(node, symbol_table):
    input1 = symbol_table.get((str(node.args[0]), 0), node.args[0])
    input2 = symbol_table.get((str(node.args[1]), 0), node.args[1])
    return _gen_arith_binary_op(input1, input2, tosa.AddOp)


def mul_op(node, symbol_table):
    def _inner_op(result_type, input1, input2):
        return tosa.MulOp(
            result_type,
            input1,
            input2,
            ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0),
        )

    input1 = symbol_table.get((str(node.args[0]), 0), node.args[0])
    input2 = symbol_table.get((str(node.args[1]), 0), node.args[1])

    return _gen_arith_binary_op(input1, input2, _inner_op)


ops_registry = {
    "add.Tensor": add_op,
    "mul.Tensor": mul_op,
}
