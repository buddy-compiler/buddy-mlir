# ===- utils.py ----------------------------------------------------------------
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
# The element utils of mlir element type and attribute.
#
# ===---------------------------------------------------------------------------

import buddy_mlir.ir as ir
from buddy_mlir.dialects import arith, linalg, tensor, tosa

from ..graph import TensorDType
from ..graph.sizes import broadcast_sizes

LARGE_SPLAT_TENSOR_ELEMENT_THRESHOLD = 10_000
LARGE_ZERO_TENSOR_ELEMENT_THRESHOLD = LARGE_SPLAT_TENSOR_ELEMENT_THRESHOLD


def _static_element_count(shape: list[int]) -> int | None:
    if not shape:
        return 1
    count = 1
    for dim in shape:
        try:
            dim = int(dim)
        except (TypeError, ValueError):
            return None
        if dim < 0:
            return None
        count *= dim
    return count


def zero_tensor_value(
    shape: list[int],
    element_type: ir.Type,
    zero_attr: ir.Attribute,
    element_threshold: int = LARGE_ZERO_TENSOR_ELEMENT_THRESHOLD,
) -> ir.Value:
    """Create a zero tensor value without materializing huge f32 literals."""
    return splat_tensor_value(shape, element_type, zero_attr, element_threshold)


def splat_tensor_value(
    shape: list[int],
    element_type: ir.Type,
    element_attr: ir.Attribute,
    element_threshold: int = LARGE_SPLAT_TENSOR_ELEMENT_THRESHOLD,
) -> ir.Value:
    """Create a splat tensor without materializing huge f32 literals. A size
    may be an index value (a dynamic size)."""
    if any(isinstance(d, ir.Value) for d in shape):
        scalar = arith.ConstantOp(element_type, element_attr).result
        empty = tensor.EmptyOp(shape, element_type).result
        return linalg.fill(scalar, outs=[empty])
    tensor_type = ir.RankedTensorType.get(shape, element_type)
    element_count = _static_element_count(shape)
    if (
        element_count is not None
        and element_count >= element_threshold
        and isinstance(element_type, ir.F32Type)
    ):
        scalar = arith.ConstantOp(element_type, element_attr).result
        empty = tensor.EmptyOp(shape, element_type).result
        return linalg.fill(scalar, outs=[empty])

    attr = ir.DenseElementsAttr.get_splat(tensor_type, element_attr)
    return arith.ConstantOp(tensor_type, attr).result


def mlir_element_type_get(type_name):
    """
    Get the mlir element type base on TensorDType's enum type.
    Args:
        type_name: The TensorDType's enum type.
    """
    match type_name:
        case TensorDType.Float16:
            return ir.F16Type.get()
        case TensorDType.BFloat16:
            return ir.BF16Type.get()
        case TensorDType.Float32:
            return ir.F32Type.get()
        case TensorDType.Float64:
            return ir.F64Type.get()
        case TensorDType.Complex64:
            return ir.ComplexType.get(ir.F32Type.get())
        case TensorDType.Complex128:
            return ir.ComplexType.get(ir.F64Type.get())
        case TensorDType.Int8:
            return ir.IntegerType.get_signless(8)
        case TensorDType.Int32:
            return ir.IntegerType.get_signless(32)
        case TensorDType.Int64:
            return ir.IntegerType.get_signless(64)
        case TensorDType.Bool:
            return ir.IntegerType.get_signless(1)
        case _:
            raise NotImplementedError(f"Unsupported dtype {type_name}")


def mlir_element_attr_get(type_name, value):
    """
    Get the mlir element attribute base on TensorDType's enum type and value.
    Args:
        type_name: The TensorDType's enum type.
        value: The real value for mlir element attribute.
    """
    match type_name:
        case TensorDType.Float16:
            return ir.FloatAttr.get(ir.F16Type.get(), value)
        case TensorDType.BFloat16:
            return ir.FloatAttr.get(ir.BF16Type.get(), value)
        case TensorDType.Float32:
            return ir.FloatAttr.get(ir.F32Type.get(), value)
        case TensorDType.Float64:
            return ir.FloatAttr.get(ir.F64Type.get(), value)
        case TensorDType.Int8:
            return ir.IntegerAttr.get(
                ir.IntegerType.get_signless(8), int(value)
            )
        case TensorDType.Int32:
            return ir.IntegerAttr.get(
                ir.IntegerType.get_signless(32), int(value)
            )
        case TensorDType.Int64:
            return ir.IntegerAttr.get(
                ir.IntegerType.get_signless(64), int(value)
            )
        case TensorDType.Bool:
            return ir.IntegerAttr.get(
                ir.IntegerType.get_signless(1), int(value)
            )
        case _:
            raise NotImplementedError(
                f"Unsupported dtype for attribute {type_name}"
            )


# The scalar operation of a TOSA elementwise operation, on floats and on
# integers (bitwise and logical operations: integers only).
dynamic_elementwise_ops = {
    tosa.AddOp: (arith.AddFOp, arith.AddIOp),
    tosa.SubOp: (arith.SubFOp, arith.SubIOp),
    tosa.MulOp: (arith.MulFOp, arith.MulIOp),
    tosa.MaximumOp: (arith.MaximumFOp, arith.MaxSIOp),
    tosa.MinimumOp: (arith.MinimumFOp, arith.MinSIOp),
    tosa.BitwiseAndOp: (None, arith.AndIOp),
    tosa.BitwiseOrOp: (None, arith.OrIOp),
    tosa.BitwiseXorOp: (None, arith.XOrIOp),
    tosa.LogicalAndOp: (None, arith.AndIOp),
    tosa.LogicalOrOp: (None, arith.OrIOp),
    tosa.LogicalXorOp: (None, arith.XOrIOp),
}


def dynamic_elementwise(result_type, operands, scalar):
    """An elementwise operation over operands that share dynamic sizes, as a
    linalg.generic computing `scalar` of their elements: a size of the trace
    is the same symbol in every operand, so only a static size of 1
    broadcasts. tosa-to-linalg would check every shared dynamic size for 1 at
    run time instead."""
    rank = result_type.rank
    maps = []
    for value in operands:
        dims = [
            ir.AffineExpr.get_constant(0)
            if extent == 1 and result_type.get_dim_size(i) != 1
            else ir.AffineExpr.get_dim(i)
            for i, extent in enumerate(ir.RankedTensorType(value.type).shape)
        ]
        maps.append(ir.AffineMapAttr.get(ir.AffineMap.get(rank, 0, dims)))
    maps.append(ir.AffineMapAttr.get(ir.AffineMap.get_identity(rank)))
    empty = tensor.EmptyOp(
        broadcast_sizes(result_type, operands), result_type.element_type
    )
    generic = linalg.GenericOp(
        [result_type],
        operands,
        [empty.result],
        ir.ArrayAttr.get(maps),
        ir.ArrayAttr.get(
            [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * rank
        ),
    )
    types = [ir.RankedTensorType(v.type).element_type for v in operands]
    block = ir.Block.create_at_start(
        generic.region, [*types, result_type.element_type]
    )
    with ir.InsertionPoint(block):
        linalg.YieldOp([scalar(*block.arguments[:-1])])
    return generic


def dynamic_binary(result_type, input1, input2, op_func):
    """dynamic_elementwise of a TOSA elementwise operation."""
    floating, integer = dynamic_elementwise_ops[op_func]
    element = result_type.element_type
    operation = integer if isinstance(element, ir.IntegerType) else floating
    return dynamic_elementwise(
        result_type, [input1, input2], lambda a, b: operation(a, b).result
    )
