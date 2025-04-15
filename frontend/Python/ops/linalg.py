# ===- linalg.py ---------------------------------------------------------------
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
# The registry of mappings from Buddy Graph to MLIR linalg dialect operations.
#
# ===---------------------------------------------------------------------------

from typing import Dict, Tuple, List

import mlir.ir as ir
from mlir.dialects import tosa, linalg, arith, tensor, math
import copy, array, sys
import numpy
import functools

from ..graph import *
from ..graph.graph import TensorDType
from .utils import *


def add_op(node: AddOp, symbol_table: Dict[Tuple[str, int], ir.Operation]):
    """
    Import tensor add operation.
    From buddy AddOp to MLIR arith `constant` operation.

    Note: this function init an output tensor according input range.

    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation representing the result tensor of two input nodes' add
        result.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    shape = list(node.tensor_meta["shape"])
    if isinstance(node.args[1], str):
        input2 = symbol_table.get((str(node.args[1]), 0))
    else:
        data = [node.args[1]]
        input2_shape = numpy.array(data).shape
        tensor_type = ir.RankedTensorType.get(input2_shape, mlir_dtype)
        element = mlir_element_attr_get(dtype, node.args[1])
        attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
        input2 = arith.ConstantOp(tensor_type, attr).result
    if input1 is None or input2 is None:
        return
    add_result_tensor_type = ir.RankedTensorType.get(shape, mlir_dtype)
    op = tosa.AddOp(
        add_result_tensor_type,
        input1,
        input2,
    )
    return op.result


def arange_op(
    node: ArangeOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import tensor arange operation.
    From buddy ArangeOp to MLIR arith `constant` operation.

    Note: this function init an output tensor according input range.

    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation representing the result tensor of ranging the start
        and end from input node.
    """
    if len(node.args) == 2:
        start = int(node.args[0])
        end = int(node.args[1])
    else:
        start = 0
        end = int(node.args[0])
    stride = 1
    dtype = node.tensor_meta["dtype"]
    shape = list(node.tensor_meta["shape"])
    dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(shape, dtype)
    attr = ir.DenseElementsAttr.get(
        numpy.array([i for i in range(start, end, stride)]),
        signless=True,
        type=tensor_type,
    )
    op = arith.ConstantOp(tensor_type, attr)

    return op


def unsqueeze_op(
    node: UnsqueezeOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the unsqueeze operation.
    From buddy UnsqueezeOp to MLIR TOSA `reshape` operation.

    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the reshape op.
    """
    input_node = symbol_table.get((str(node.args[0]), 0))
    if input_node is None:
        return
    axis = int(node.args[1])
    input_shape = ir.RankedTensorType(input_node.type).shape
    input_shape.insert(axis, 1)
    tensor_type = ir._denseI64ArrayAttr(
        numpy.array(input_shape, dtype=numpy.int64), None
    )
    op = tosa.ReshapeOp(input_node, tensor_type)

    return op


def view_op(
    node: ViewOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor view operation.
    From buddy ViewOp to MLIR TOSA `reshape` operation.

    Note: If the new shape contains one and only one `-1`, the size of the new
    shape will be inferred automatically.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the reshape op.
    """
    input_node = symbol_table.get((str(node.args[0]), 0))
    if input_node is None:
        return
    output_shape = list(node.args[1])
    input_shape = list(ir.RankedTensorType(input_node.type).shape)

    nums = 1
    for i in input_shape:
        nums *= i
    for i in output_shape:
        if i != -1:
            nums //= i
    for i, s in enumerate(output_shape):
        if s == -1:
            output_shape[i] = nums

    tensor_type = ir._denseI64ArrayAttr(
        numpy.array(output_shape, dtype=numpy.int64), None
    )
    op = tosa.ReshapeOp(input_node, tensor_type)

    return op


def embedding_op(
    node: EmbeddingOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the embedding operation.
    From buddy EmbeddingOp to MLIR linalg `generic` operation.

    Note: In this op, input node1's value is as index to get input node2's row
    slice.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    input2 = symbol_table.get((str(node.args[1]), 0))
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(output_shape, dtype)
    output = tensor.EmptyOp(output_shape, dtype)
    generic_map = ir.AffineMap.get_permutation([0, 1, 2])
    op = linalg.GenericOp(
        [tensor_type],
        [input2],
        [output],
        ir.ArrayAttr.get(
            [
                ir.AffineMapAttr.get(generic_map.get_submap([0, 1])),
                ir.AffineMapAttr.get(generic_map.get_submap([0, 1, 2])),
            ]
        ),
        ir.ArrayAttr.get(
            [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * 3
        ),
    )
    block = ir.Block.create_at_start(
        op.region,
        [
            ir.RankedTensorType(input2.type).element_type,
            ir.RankedTensorType(output.result.type).element_type,
        ],
    )
    index1 = arith.IndexCastOp(ir.IndexType.get(), block.arguments[0])
    index2 = linalg.IndexOp(ir._i64Attr(2, None))
    value = tensor.ExtractOp(input1, [index1.result, index2.result])
    block.append(index1)
    block.append(index2)
    block.append(value)
    block.append(linalg.YieldOp([value.result]))

    return op


def ones_op(
    node: OnesOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor ones operation.
    From buddy OnesOp to MLIR arith `constant` operation.

    Note: This op, input node1's value is as index to get input node2's row
    slice.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the arith.constant op.
    """
    output_shape = list(node.args[0])
    dtype = node.tensor_meta["dtype"]
    element = mlir_element_attr_get(dtype, 1)
    tensor_type = ir.RankedTensorType.get(output_shape, element.type)
    attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
    op = arith.ConstantOp(tensor_type, attr)

    return op


def full_op(
    node: FullOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor full operation.
    From buddy FullOp to MLIR arith `constant` operation.

    Note: This op, input node1's value is the shape of output tensor, input
    node2's value is the value of all elements in output tensor.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the arith.constant op.
    """
    output_shape = list(node.args[0])
    value = node.args[1]
    dtype = node.tensor_meta["dtype"]
    element = mlir_element_attr_get(dtype, value)
    tensor_type = ir.RankedTensorType.get(output_shape, element.type)
    attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
    op = arith.ConstantOp(tensor_type, attr)

    return op


def lt_op(
    node: LessThanOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor less than operation.
    From buddy LessThanOp to MLIR arith `constant` operation.

    Note: This op, campare two input nodes, and output bool tensor to represent
    compare result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    input2 = symbol_table.get((str(node.args[1]), 0))
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    value = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 2)
    shp1 = list(ir.RankedTensorType(ir.Value(input1).type).shape)
    shp2 = list(ir.RankedTensorType(ir.Value(input2).type).shape)
    dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(output_shape, dtype)
    output = tensor.EmptyOp(output_shape, dtype)
    if len(shp1) < len(shp2):
        if int(shp1[-1]) > 1 and shp2[-1] == 1:
            generic_map = ir.AffineMap.get_permutation(
                [i for i in range(len(shp2) + 1)]
            )
            op = linalg.GenericOp(
                [tensor_type],
                [input1, input2],
                [output],
                ir.ArrayAttr.get(
                    [
                        ir.AffineMapAttr.get(
                            generic_map.get_submap(
                                [
                                    i
                                    for i in range(
                                        len(shp2) - len(shp1), len(shp2)
                                    )
                                ]
                            )
                        ),
                        ir.AffineMapAttr.get(
                            generic_map.get_submap(
                                [i for i in range(0, len(shp2) - 1)]
                                + [len(shp2)]
                            )
                        ),
                        ir.AffineMapAttr.get(
                            generic_map.get_submap(
                                [i for i in range(0, len(shp2))]
                            )
                        ),
                    ]
                ),
                ir.ArrayAttr.get(
                    [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
                    * len(shp2)
                    + [ir.Attribute.parse("#linalg.iterator_type<reduction>")]
                ),
            )
            block = ir.Block.create_at_start(
                op.region,
                [
                    ir.RankedTensorType(input2.type).element_type,
                    ir.RankedTensorType(input2.type).element_type,
                    dtype,
                ],
            )
            if (
                str(ir.RankedTensorType(input2.type).element_type).find("i")
                != -1
            ):
                cmpop = arith.CmpIOp(
                    value, block.arguments[0], block.arguments[1]
                )
            else:
                cmpop = arith.CmpFOp(
                    value, block.arguments[0], block.arguments[1]
                )
            block.append(cmpop)
            block.append(linalg.YieldOp([cmpop.result]))

    return op


def masked_fill_op(
    node: MaskedFillOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor masked fill operation.
    From buddy MaskedFillOp to MLIR linalg `generic` operation.

    Note: This op, input node2 is a bool tensor. Select input node1's value or
    input node3's value by true or false in input node2's value.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    input2 = symbol_table.get((str(node.args[1]), 0))
    if input1 is None or input2 is None:
        return
    dtype = node.tensor_meta["dtype"]
    value = node.args[2]
    attr = mlir_element_attr_get(dtype, value)
    dtype = mlir_element_type_get(dtype)
    value = arith.ConstantOp(dtype, attr)
    output_shape = list(node.tensor_meta["shape"])
    tensor_type = ir.RankedTensorType.get(output_shape, dtype)
    output = tensor.EmptyOp(output_shape, dtype)
    generic_map = ir.AffineMap.get_permutation(
        [i for i in range(len(output_shape))]
    )
    op = linalg.GenericOp(
        [tensor_type],
        [input1, input2],
        [output],
        ir.ArrayAttr.get(
            [
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [i for i in range(len(output_shape))]
                    )
                ),
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [i for i in range(len(output_shape))]
                    )
                ),
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [i for i in range(len(output_shape))]
                    )
                ),
            ]
        ),
        ir.ArrayAttr.get(
            [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
            * len(output_shape)
        ),
    )
    block = ir.Block.create_at_start(
        op.region,
        [
            ir.RankedTensorType(input1.type).element_type,
            ir.RankedTensorType(input2.type).element_type,
            ir.RankedTensorType(output.result.type).element_type,
        ],
    )
    select_op = arith.SelectOp(block.arguments[1], value, block.arguments[0])
    block.append(select_op)
    block.append(linalg.YieldOp([select_op.result]))

    return op


def slice_op(
    node: SliceOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor slice operation.
    From buddy SliceOp to MLIR tensor `extract_slice` operation.

    Note: This op, get the slice of input node1.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the tensor.extract_slice op.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    if input1 is None:
        return
    dim = int(node.args[1])
    start = int(node.args[2])
    end = int(node.args[3])
    input_shape = ir.RankedTensorType(input1.type).shape
    if end > input_shape[dim]:
        end = input_shape[dim]
    if len(node.args) < 5:
        step = 1
    else:
        step = node.args[4]
    offset = [0 for x in input_shape]
    offset[dim] = start
    offset_attr = ir._denseI64ArrayAttr(offset, None)
    output_shape = list(node.tensor_meta["shape"])
    size_attr = ir._denseI64ArrayAttr(output_shape, None)
    stride = [1 for x in output_shape]
    stride[dim] = step
    stride_attr = ir._denseI64ArrayAttr(stride, None)
    dtype = node.tensor_meta["dtype"]
    dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(output_shape, dtype)

    op = tensor.ExtractSliceOp(
        tensor_type, input1, [], [], [], offset_attr, size_attr, stride_attr
    )

    return op


def expand_op(
    node: ExpandOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor expand operation.
    From buddy ExpandOp to MLIR tensor `extract_slice` operation.

    Note: This op, based on expand shape, create a new tensor and extract slice
    from origin tensor.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the tensor.extract_slice op.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    assert isinstance(node.args[1], list)

    if input1 is None:
        return
    input_shape = ir.RankedTensorType(input1.type).shape
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    dtype = mlir_element_type_get(dtype)
    empty_tensor = tensor.EmptyOp(output_shape, dtype)
    if list(input_shape) == list(node.args[1]):
        offset_attr = ir._denseI64ArrayAttr([0 for x in input_shape], None)
        size_attr = ir._denseI64ArrayAttr(output_shape, None)
        stride_attr = ir._denseI64ArrayAttr([1 for x in input_shape], None)
        tensor_type = ir.RankedTensorType.get(output_shape, dtype)
        extract_tensor = tensor.ExtractSliceOp(
            tensor_type, input1, [], [], [], offset_attr, size_attr, stride_attr
        )
        op = tensor.InsertSliceOp(
            extract_tensor.result,
            empty_tensor.result,
            [],
            [],
            [],
            offset_attr,
            size_attr,
            stride_attr,
        )
    else:
        for i in range(len(input_shape) - 1, -1, -1):
            if input_shape[i] != output_shape[i]:
                for j in range(output_shape[i]):
                    offset = [0 for x in input_shape]
                    offset_attr = ir._denseI64ArrayAttr(offset, None)
                    size_attr = ir._denseI64ArrayAttr(
                        [1] * (i + 1) + [x for x in output_shape[i + 1 :]], None
                    )
                    stride_attr = ir._denseI64ArrayAttr([1] * len(offset), None)
                    tensor_type = ir.RankedTensorType.get(
                        [1] * (i + 1) + [x for x in output_shape[i + 1 :]],
                        dtype,
                    )
                    extract_tensor = tensor.ExtractSliceOp(
                        tensor_type,
                        input1,
                        [],
                        [],
                        [],
                        offset_attr,
                        size_attr,
                        stride_attr,
                    )
                    offset[i] = j
                    offset_attr = ir._denseI64ArrayAttr(offset, None)
                    op = tensor.InsertSliceOp(
                        extract_tensor.result,
                        empty_tensor.result,
                        [],
                        [],
                        [],
                        offset_attr,
                        size_attr,
                        stride_attr,
                    )
                    empty_tensor = op
    return op


def to_copy_op(
    node: ToCopyOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor copy operation.
    From buddy ToCopyOp to MLIR linalg `generic`
    operation.

    Note: This op, will convert input node's value type, such as float32 to
    bool.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    if input1 is None:
        return
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]

    if dtype == TensorDType.Bool:
        if str(ir.RankedTensorType(input1.type).element_type) == "f32":
            tensor_type = ir.RankedTensorType.get(
                output_shape, ir.IntegerType.get_signless(1)
            )
            output = tensor.EmptyOp(
                output_shape, ir.IntegerType.get_signless(1)
            )
            generic_map = ir.AffineMap.get_permutation(
                [i for i in range(len(output_shape))]
            )
            op = linalg.GenericOp(
                [tensor_type],
                [input1],
                [output],
                ir.ArrayAttr.get(
                    [
                        ir.AffineMapAttr.get(
                            generic_map.get_submap(
                                [i for i in range(len(output_shape))]
                            )
                        ),
                        ir.AffineMapAttr.get(
                            generic_map.get_submap(
                                [i for i in range(len(output_shape))]
                            )
                        ),
                    ]
                ),
                ir.ArrayAttr.get(
                    [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
                    * len(output_shape)
                ),
            )
            block = ir.Block.create_at_start(
                op.region,
                [
                    ir.RankedTensorType(input1.type).element_type,
                    ir.RankedTensorType(output.result.type).element_type,
                ],
            )
            fptosi_op = arith.FPToSIOp(
                ir.IntegerType.get_signless(32), block.arguments[0]
            )
            trunc_op = arith.TruncIOp(
                ir.IntegerType.get_signless(1), fptosi_op.result
            )
            block.append(fptosi_op)
            block.append(trunc_op)
            block.append(linalg.YieldOp([trunc_op.result]))
    elif dtype == TensorDType.Float32:
        if str(ir.RankedTensorType(input1.type).element_type) == "i1":
            tensor_type = ir.RankedTensorType.get(
                output_shape, ir.F32Type.get()
            )
            output = tensor.EmptyOp(output_shape, ir.F32Type.get())
            generic_map = ir.AffineMap.get_permutation(
                [i for i in range(len(output_shape))]
            )
            op = linalg.GenericOp(
                [tensor_type],
                [input1],
                [output],
                ir.ArrayAttr.get(
                    [
                        ir.AffineMapAttr.get(
                            generic_map.get_submap(
                                [i for i in range(len(output_shape))]
                            )
                        ),
                        ir.AffineMapAttr.get(
                            generic_map.get_submap(
                                [i for i in range(len(output_shape))]
                            )
                        ),
                    ]
                ),
                ir.ArrayAttr.get(
                    [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
                    * len(output_shape)
                ),
            )
            block = ir.Block.create_at_start(
                op.region,
                [
                    ir.RankedTensorType(input1.type).element_type,
                    ir.RankedTensorType(output.result.type).element_type,
                ],
            )
            exti_op = arith.ExtUIOp(
                ir.IntegerType.get_signless(32), block.arguments[0]
            )
            sitofp_op = arith.SIToFPOp(ir.F32Type.get(), exti_op.result)
            block.append(exti_op)
            block.append(sitofp_op)
            block.append(linalg.YieldOp([sitofp_op.result]))

    return op


def rsub_op(
    node: RsubOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor rsub operation.
    From buddy RsubOp to MLIR linalg `generic` operation.

    Note: This op, compute input node1 rsub input node2
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    value = node.args[1]
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    if not isinstance(value, str):
        value = arith.ConstantOp(
            mlir_dtype, mlir_element_attr_get(dtype, value)
        )
        generic_map = ir.AffineMap.get_permutation(
            [i for i in range(len(output_shape))]
        )
        tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
        output = tensor.EmptyOp(output_shape, mlir_dtype)
        op = linalg.GenericOp(
            [tensor_type],
            [input1],
            [output],
            ir.ArrayAttr.get(
                [
                    ir.AffineMapAttr.get(
                        generic_map.get_submap(
                            [i for i in range(len(output_shape))]
                        )
                    ),
                    ir.AffineMapAttr.get(
                        generic_map.get_submap(
                            [i for i in range(len(output_shape))]
                        )
                    ),
                ]
            ),
            ir.ArrayAttr.get(
                [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
                * len(output_shape)
            ),
        )
        block = ir.Block.create_at_start(
            op.region,
            [
                ir.RankedTensorType(input1.type).element_type,
                ir.RankedTensorType(output.result.type).element_type,
            ],
        )
        if str(ir.RankedTensorType(input1.type).element_type).find("i") != -1:
            sub_op = arith.SubIOp(value.result, block.arguments[0])
        else:
            sub_op = arith.SubFOp(value.result, block.arguments[0])
        block.append(sub_op)
        block.append(linalg.YieldOp([sub_op.result]))

    return op


def pow_op(
    node: PowOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor copy operation.
    From buddy PowOp to MLIR linalg `generic`
    operation.

    Note: This op, compute input node's power result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    if input1 is None:
        return
    value = node.args[1]
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    dtype = mlir_element_type_get(dtype)
    if not isinstance(value, str):
        if abs(int(value) - float(value)) < 1e-6:
            generic_map = ir.AffineMap.get_permutation(
                [i for i in range(len(output_shape))]
            )
            tensor_type = ir.RankedTensorType.get(output_shape, dtype)
            output = tensor.EmptyOp(output_shape, dtype)
            value = arith.ConstantOp(
                ir.IntegerType.get_signless(32),
                ir.IntegerAttr.get(ir.IntegerType.get_signless(32), value),
            )
            op = linalg.GenericOp(
                [tensor_type],
                [input1],
                [output],
                ir.ArrayAttr.get(
                    [
                        ir.AffineMapAttr.get(
                            generic_map.get_submap(
                                [i for i in range(len(output_shape))]
                            )
                        ),
                        ir.AffineMapAttr.get(
                            generic_map.get_submap(
                                [i for i in range(len(output_shape))]
                            )
                        ),
                    ]
                ),
                ir.ArrayAttr.get(
                    [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
                    * len(output_shape)
                ),
            )
            block = ir.Block.create_at_start(
                op.region,
                [
                    ir.RankedTensorType(input1.type).element_type,
                    ir.RankedTensorType(output.result.type).element_type,
                ],
            )
            if (
                str(ir.RankedTensorType(input1.type).element_type).find("i")
                != -1
            ):
                powi_op = math.IPowIOp(block.arguments[0], value.result)
            else:
                powi_op = math.FPowIOp(block.arguments[0], value.result)
            block.append(powi_op)
            block.append(linalg.YieldOp([powi_op.result]))

    return op


def mean_op(
    node: MeanOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor copy operation.
    From buddy MeanOp to MLIR linalg `generic` operation.

    Note: This op, compute input node's mean result in a specified dim.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    if input1 is None:
        return
    dims = list(node.args[1])
    keep_dim = bool(node.args[2])
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    element = mlir_element_attr_get(dtype, 0.0)
    attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
    output = arith.ConstantOp(tensor_type, attr)
    assert len(dims) == 1
    for dim in dims:
        if dim < 0:
            dim = len(list(ir.RankedTensorType(input1.type).shape)) + dim
        if keep_dim:
            generic_map = ir.AffineMap.get_permutation(
                [i for i in range(len(output_shape) + 1)]
            )
            tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
            output_map = [i for i in range(len(output_shape))]
            output_map[dim] = len(output_shape)
            loop_type = [
                ir.Attribute.parse("#linalg.iterator_type<parallel>")
            ] * (len(output_shape) + 1)
            loop_type[dim] = ir.Attribute.parse(
                "#linalg.iterator_type<reduction>"
            )
            op = linalg.GenericOp(
                [tensor_type],
                [input1],
                [output],
                ir.ArrayAttr.get(
                    [
                        ir.AffineMapAttr.get(
                            generic_map.get_submap(
                                [i for i in range(len(output_shape))]
                            )
                        ),
                        ir.AffineMapAttr.get(
                            generic_map.get_submap(output_map)
                        ),
                    ]
                ),
                ir.ArrayAttr.get(loop_type),
            )
            block = ir.Block.create_at_start(
                op.region,
                [
                    ir.RankedTensorType(input1.type).element_type,
                    ir.RankedTensorType(output.result.type).element_type,
                ],
            )
            value = arith.ConstantOp(
                mlir_dtype,
                mlir_element_attr_get(
                    dtype, list(ir.RankedTensorType(input1.type).shape)[dim]
                ),
            )
            if (
                str(ir.RankedTensorType(input1.type).element_type).find("i")
                != -1
            ):
                block_div_op = arith.DivSIOp(block.arguments[0], value.result)
                block_add_op = arith.AddIOp(
                    block_div_op.result, block.arguments[1]
                )
            else:
                block_div_op = arith.DivFOp(block.arguments[0], value.result)
                block_add_op = arith.AddFOp(
                    block_div_op.result, block.arguments[1]
                )
            block.append(value)
            block.append(block_div_op)
            block.append(block_add_op)
            block.append(linalg.YieldOp([block_add_op.result]))

    return op


def rsqrt_op(
    node: RsqrtOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor rsqrt operation.
    From buddy RsqrtOp to MLIR linalg `generic` operation.

    Note: This op, compute input node's rsqrt result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    assert len(node.args) == 1
    input1 = symbol_table.get((str(node.args[0]), 0))
    if input1 is None:
        return

    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    output = tensor.EmptyOp(output_shape, mlir_dtype)
    generic_map = ir.AffineMap.get_permutation(
        [i for i in range(len(output_shape))]
    )
    op = linalg.GenericOp(
        [tensor_type],
        [input1],
        [output],
        ir.ArrayAttr.get(
            [
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [i for i in range(len(output_shape))]
                    )
                ),
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [i for i in range(len(output_shape))]
                    )
                ),
            ]
        ),
        ir.ArrayAttr.get(
            [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
            * len(output_shape)
        ),
    )
    block = ir.Block.create_at_start(
        op.region,
        [
            ir.RankedTensorType(input1.type).element_type,
            ir.RankedTensorType(output.result.type).element_type,
        ],
    )
    math_rsqrt_op = math.RsqrtOp(block.arguments[0])
    block.append(math_rsqrt_op)
    block.append(linalg.YieldOp([math_rsqrt_op.result]))

    return op


def mul_op(
    node: MulOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor mul operation.
    From buddy MulOp to MLIR linalg `generic` operation.

    Note: This op, compute input node's mul result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    assert len(node.args) == 2
    input1 = symbol_table.get((str(node.args[0]), 0))
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    shape = list(node.tensor_meta["shape"])
    if isinstance(node.args[1], str):
        input2 = symbol_table.get((str(node.args[1]), 0))
    else:
        data = [node.args[1]]
        input2_shape = numpy.array(data).shape
        tensor_type = ir.RankedTensorType.get(input2_shape, mlir_dtype)
        element = mlir_element_attr_get(dtype, node.args[1])
        attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
        input2 = arith.ConstantOp(tensor_type, attr).result

    input1_dtype = ir.RankedTensorType(input1.type).element_type
    input2_dtype = ir.RankedTensorType(input2.type).element_type
    if input1_dtype != mlir_dtype:
        input1 = tosa.CastOp(
            ir.RankedTensorType.get(
                ir.RankedTensorType(input1.type).shape,
                mlir_dtype,
            ),
            input1,
        )
    if input2_dtype != mlir_dtype:
        input2 = tosa.CastOp(
            ir.RankedTensorType.get(
                ir.RankedTensorType(input2.type).shape,
                mlir_dtype,
            ),
            input2,
        )

    if input1 is None or input2 is None:
        return
    mul_result_tensor_type = ir.RankedTensorType.get(shape, mlir_dtype)
    op = tosa.MulOp(
        mul_result_tensor_type,
        input1,
        input2,
    )
    return op.result


def t_op(
    node: TOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor tanspose operation.
    From buddy TransposeOp to MLIR linalg `generic` operation.

    Note: This op, compute input node's transpose result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    assert len(node.args) == 1
    input1 = symbol_table.get((str(node.args[0]), 0))
    if input1 is None:
        return

    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    perm = ir._denseI64ArrayAttr([1, 0], None)
    output = tensor.EmptyOp(output_shape, mlir_dtype)
    op = linalg.transpose(input=input1, outs=[output], permutation=perm)

    return op.result[0]


def matmul_transpose_b_op(
    node: TransposeMatmulFusedOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    input1 = symbol_table.get((str(node.args[0]), 0))
    input2 = symbol_table.get((str(node.args[1]), 0))

    if input1 is None or input2 is None:
        return
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    element = mlir_element_attr_get(dtype, 0.0)
    attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
    result_buffer = arith.ConstantOp(tensor_type, attr).result
    op = linalg.matmul_transpose_b(input1, input2, outs=[result_buffer])
    return op


def transpose_op(
    node: TransposeOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor transpose operation.
    From buddy TransposeSpecificDimOp to MLIR linalg `generic`
    operation.

    Note: This op, compute input node's transpose result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    assert len(node.args) == 3
    input1 = symbol_table.get((str(node.args[0]), 0))
    if input1 is None:
        return
    dim1 = int(node.args[1])
    dim2 = int(node.args[2])
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    output_perm = [i for i in range(len(output_shape))]
    output_perm[dim2], output_perm[dim1] = output_perm[dim1], output_perm[dim2]
    perm = ir._denseI64ArrayAttr(output_perm, None)
    output = tensor.EmptyOp(output_shape, mlir_dtype)
    op = linalg.transpose(input=input1, outs=[output], permutation=perm)

    return op.result[0]


def index_op(
    node: IndexOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor index operation.
    From buddy IndexOp to MLIR linalg `generic`
    operation.

    Note: This op, get input node slice result by input index.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    assert len(node.args) == 2
    input1 = symbol_table.get((str(node.args[0]), 0))
    if input1 is None:
        return
    input1_shape = ir.RankedTensorType(input1.type).shape
    input2 = node.args[1]
    input2_dim_sum = 0
    for i in range(len(input2)):
        input2_dim_sum += len(symbol_table.get((str(input2[i]), 0)).type.shape)
    output_shape = list(node.tensor_meta["shape"])
    input_shape = input1.type.shape
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    if len(input2) < len(input1_shape):
        tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
        output = tensor.EmptyOp(output_shape, mlir_dtype)
        generic_map = ir.AffineMap.get_permutation(
            [i for i in range(max(len(output_shape), len(input_shape)))]
        )
        input_map = []
        for i in range(len(input2)):
            input2_shape = symbol_table.get((str(input2[i]), 0)).type.shape
            input_map.append(
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [j for j in range(i, i + len(input2_shape))]
                    )
                )
            )
        if len(input_shape) > len(output_shape):
            input_map.append(
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [
                            j
                            for j in range(
                                len(input_shape) - len(output_shape),
                                len(input_shape),
                            )
                        ]
                    )
                )
            )
        else:
            input_map.append(
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [j for j in range(len(output_shape))]
                    )
                )
            )
        operands = [symbol_table.get((str(i), 0)) for i in input2]
        op = linalg.GenericOp(
            [tensor_type],
            operands,
            [output],
            ir.ArrayAttr.get(input_map),
            ir.ArrayAttr.get(
                [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
                * max(len(output_shape), len(input_shape))
            ),
        )
        arguments = [
            ir.RankedTensorType(i.type).element_type for i in operands
        ] + [ir.RankedTensorType(output.result.type).element_type]
        block = ir.Block.create_at_start(op.region, arguments)
        index = []
        for i in block.arguments[:-1]:
            indexcast_op = arith.IndexCastOp(ir.IndexType.get(), i)
            block.append(indexcast_op)
            index.append(indexcast_op.result)
        for i in range(
            input2_dim_sum, max(len(input_shape), len(output_shape))
        ):
            index_op = linalg.IndexOp(ir._i64Attr(i, None))
            block.append(index_op)
            index.append(index_op.result)
        value = tensor.ExtractOp(input1, index)
        block.append(value)
        block.append(linalg.YieldOp([value.result]))

    return op


def neg_op(
    node: NegOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor neg operation.
    From buddy NegOp to MLIR linalg `negf` operation.

    Note: This op, compute input node's neg result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    assert len(node.args) == 1
    input1 = symbol_table.get((str(node.args[0]), 0))
    if input1 is None:
        return
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    output = tensor.EmptyOp(output_shape, mlir_dtype)
    op = linalg.negf(input1, outs=output)

    return op


def cat_op(
    node: CatOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor concate operation.
    From buddy CatOp to MLIR tensor `insert_slice`
    operation.

    Note: This op, concate two input tensor.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the tensor.insert_slice op.
    """
    assert len(node.args) == 2
    input1 = symbol_table.get((str(node.args[0][0]), 0))
    input2 = symbol_table.get((str(node.args[0][1]), 0))
    dim = int(node.args[1])
    if input1 is None or input2 is None:
        return

    output_shape = list(node.tensor_meta["shape"])
    if dim < 0:
        dim = len(output_shape) + dim
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    output = tensor.EmptyOp(output_shape, mlir_dtype)
    offset = [0 for x in output_shape]
    offset_attr = ir._denseI64ArrayAttr(offset, None)
    input1_shape = ir.RankedTensorType(input1.type).shape
    size_attr = ir._denseI64ArrayAttr(input1_shape, None)
    stride_attr = ir._denseI64ArrayAttr([1] * len(offset), None)
    insert_input1 = tensor.InsertSliceOp(
        input1,
        output.result,
        [],
        [],
        [],
        offset_attr,
        size_attr,
        stride_attr,
    )
    offset[dim] += input1_shape[dim]
    offset_attr = ir._denseI64ArrayAttr(offset, None)
    input2_shape = ir.RankedTensorType(input2.type).shape
    size_attr = ir._denseI64ArrayAttr(input2_shape, None)
    insert_input2 = tensor.InsertSliceOp(
        input2,
        insert_input1.result,
        [],
        [],
        [],
        offset_attr,
        size_attr,
        stride_attr,
    )

    return insert_input2


def squeeze_op(
    node: SqueezeOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor squeeze operation.
    From buddy SqueezeOp to MLIR linalg `generic` operation.

    Note: This op, reduce the input tensor's shape dims by specified dim.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    assert len(node.args) == 2
    input1 = symbol_table.get((str(node.args[0]), 0))
    dim = int(node.args[1])
    if input1 is None:
        return

    output_shape = list(node.tensor_meta["shape"])
    input1_shape = ir.RankedTensorType(input1.type).shape
    if dim < 0:
        dim = len(input1_shape) + dim
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    output = tensor.EmptyOp(output_shape, mlir_dtype)
    if input1_shape[dim] != 1:
        offset = [0 for x in output_shape]
        offset_attr = ir._denseI64ArrayAttr(offset, None)
        size_attr = ir._denseI64ArrayAttr(input1_shape, None)
        stride_attr = ir._denseI64ArrayAttr([1] * len(offset), None)
        op = tensor.InsertSliceOp(
            input1,
            output.result,
            [],
            [],
            [],
            offset_attr,
            size_attr,
            stride_attr,
        )
    else:
        output_map = ir.AffineMap.get(
            len(output_shape),
            0,
            [ir.AffineExpr.get_dim(i) for i in range(len(output_shape))],
        )
        input1_map = []
        loop_index = 0
        for i in range(len(input1_shape)):
            if len(input1_map) == dim:
                input1_map.append(ir.AffineExpr.get_constant(0))
            else:
                input1_map.append(ir.AffineExpr.get_dim(loop_index))
                loop_index += 1
        input1_map = ir.AffineMap.get(len(output_shape), 0, input1_map)
        op = linalg.GenericOp(
            [tensor_type],
            [input1],
            [output],
            ir.ArrayAttr.get(
                [
                    ir.AffineMapAttr.get(input1_map),
                    ir.AffineMapAttr.get(output_map),
                ]
            ),
            ir.ArrayAttr.get(
                [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
                * len(output_shape)
            ),
        )
        block = ir.Block.create_at_start(
            op.region,
            [
                ir.RankedTensorType(input1.type).element_type,
                ir.RankedTensorType(output.result.type).element_type,
            ],
        )
        block.append(linalg.YieldOp([block.arguments[0]]))

    return op


def batch_matmul_op(
    node: BatchMatmulOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor batch matmul operation.
    From buddy BatchMatmulOp to MLIR linalg `batch_matmul`
    operation.

    Note: This op, compute input node's batch matrix multiplication result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.batch_matmul op.
    """
    assert len(node.args) == 2
    input1 = symbol_table.get((str(node.args[0]), 0))
    input2 = symbol_table.get((str(node.args[1]), 0))
    if input1 is None or input2 is None:
        return

    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    element = mlir_element_attr_get(dtype, 0)
    attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
    zero_fill = arith.ConstantOp(tensor_type, attr).result
    op = linalg.batch_matmul(input1, input2, outs=[zero_fill])

    return op


def div_op(
    node: DivOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor divsion operation.
    From buddy DivOp to MLIR linalg `generic` operation.

    Note: This op, compute input node's division result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    assert len(node.args) == 2
    input1 = symbol_table.get((str(node.args[0]), 0))
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    shape = list(node.tensor_meta["shape"])
    if isinstance(node.args[1], str):
        input2 = symbol_table.get((str(node.args[1]), 0))
    else:
        data = [node.args[1]]
        input2_shape = numpy.array(data).shape
        tensor_type = ir.RankedTensorType.get(input2_shape, mlir_dtype)
        element = mlir_element_attr_get(dtype, node.args[1])
        attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
        input2 = arith.ConstantOp(tensor_type, attr).result
    if input1 is None or input2 is None:
        return
    div_result_tensor_type = ir.RankedTensorType.get(shape, mlir_dtype)
    op = tosa.MulOp(
        div_result_tensor_type,
        input1,
        tosa.ReciprocalOp(input2.type, input2).result,
    )
    return op.result


def softmax_op(
    node: SoftmaxOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor softmax operation.
    From buddy SoftmaxOp to MLIR linalg `generic` operation.

    Note: This op, compute input node's softmax result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    assert len(node.args) == 3
    assert node.args[2] == False
    input1 = symbol_table.get((str(node.args[0]), 0))
    dim = int(node.args[1])
    if input1 is None:
        return
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    if dim < 0:
        dim += len(output_shape)
    mlir_dtype = mlir_element_type_get(dtype)
    max_vals = tosa.ReduceMaxOp(input1, dim)
    sub_op_output = ir.RankedTensorType.get(input1.type.shape, mlir_dtype)
    input1 = tosa.SubOp(sub_op_output, input1, max_vals)
    # tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    # output = tensor.EmptyOp(output_shape, mlir_dtype)
    # op = linalg.softmax(
    #     [tensor_type],
    #     input1,
    #     output,
    #     ir.IntegerAttr.get(ir.IntegerType.get_signless(64), dim),
    # )
    # print(op, flush=True)
    sum_tensor_shape = copy.deepcopy(output_shape)
    sum_tensor_shape[dim] = 1
    sum_tensor_type = ir.RankedTensorType.get(sum_tensor_shape, mlir_dtype)
    element = mlir_element_attr_get(dtype, 0)
    attr = ir.DenseElementsAttr.get_splat(sum_tensor_type, element)
    sum_tensor = arith.ConstantOp(sum_tensor_type, attr).result
    input1_map = [ir.AffineExpr.get_dim(i) for i in range(len(output_shape))]
    input1_map = ir.AffineMap.get(len(output_shape), 0, input1_map)
    sum_tensor_map = [
        ir.AffineExpr.get_dim(i) for i in range(len(output_shape))
    ]
    sum_tensor_map[dim] = ir.AffineExpr.get_constant(0)
    sum_tensor_map = ir.AffineMap.get(len(output_shape), 0, sum_tensor_map)
    loop_type = [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * len(
        output_shape
    )
    loop_type[dim] = ir.Attribute.parse("#linalg.iterator_type<reduction>")
    sum_tensor_op = linalg.GenericOp(
        [sum_tensor_type],
        [input1],
        [sum_tensor],
        ir.ArrayAttr.get(
            [
                ir.AffineMapAttr.get(input1_map),
                ir.AffineMapAttr.get(sum_tensor_map),
            ]
        ),
        ir.ArrayAttr.get(loop_type),
    )
    block = ir.Block.create_at_start(
        sum_tensor_op.region,
        [
            mlir_dtype,
            mlir_dtype,
        ],
    )
    exp_op = math.ExpOp(block.arguments[0])
    add_op = arith.AddFOp(exp_op.result, block.arguments[1])
    block.append(exp_op)
    block.append(add_op)
    block.append(linalg.YieldOp([add_op.result]))
    result_tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    result_tensor = tensor.EmptyOp(output_shape, mlir_dtype)
    result_tensor_map = [
        ir.AffineExpr.get_dim(i) for i in range(len(output_shape))
    ]
    result_tensor_map = ir.AffineMap.get(
        len(output_shape), 0, result_tensor_map
    )
    op = linalg.GenericOp(
        [result_tensor_type],
        [input1, sum_tensor_op.result],
        [result_tensor.result],
        ir.ArrayAttr.get(
            [
                ir.AffineMapAttr.get(input1_map),
                ir.AffineMapAttr.get(sum_tensor_map),
                ir.AffineMapAttr.get(result_tensor_map),
            ]
        ),
        ir.ArrayAttr.get(
            [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
            * len(output_shape)
        ),
    )
    block = ir.Block.create_at_start(
        op.region,
        [
            mlir_dtype,
            mlir_dtype,
            mlir_dtype,
        ],
    )
    exp_op = math.ExpOp(block.arguments[0])
    div_op = arith.DivFOp(exp_op.result, block.arguments[1])
    block.append(exp_op)
    block.append(div_op)
    block.append(linalg.YieldOp([div_op.result]))

    return op


def clone_op(
    node: CloneOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor clone operation.
    From buddy CloneOp to MLIR tensor `extract_slice`
    operation.

    Note: This op, clone input tensor to a new tensor.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the tensor.extract_slice op.
    """
    assert len(node.args) == 1
    input1 = symbol_table.get((str(node.args[0]), 0))
    if input1 is None:
        return

    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    offset = [0 for x in output_shape]
    offset_attr = ir._denseI64ArrayAttr(offset, None)
    size_attr = ir._denseI64ArrayAttr(output_shape, None)
    stride = [1 for x in output_shape]
    stride_attr = ir._denseI64ArrayAttr(stride, None)
    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    op = tensor.ExtractSliceOp(
        tensor_type, input1, [], [], [], offset_attr, size_attr, stride_attr
    )

    return op


def silu_op(
    node: SiluOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor silu activation operation.
    From Buddy SiluOp to MLIR linalg `generic` operation.

    Note: This op, compute input node's silu activation result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    assert len(node.args) == 1
    input1 = symbol_table.get((str(node.args[0]), 0))
    if input1 is None:
        return

    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    output = tensor.EmptyOp(output_shape, mlir_dtype)
    generic_map = ir.AffineMap.get_permutation(
        [i for i in range(len(output_shape))]
    )
    op = linalg.GenericOp(
        [tensor_type],
        [input1],
        [output],
        ir.ArrayAttr.get(
            [
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [i for i in range(len(output_shape))]
                    )
                ),
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [i for i in range(len(output_shape))]
                    )
                ),
            ]
        ),
        ir.ArrayAttr.get(
            [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
            * len(output_shape)
        ),
    )
    block = ir.Block.create_at_start(
        op.region,
        [
            ir.RankedTensorType(input1.type).element_type,
            ir.RankedTensorType(output.result.type).element_type,
        ],
    )
    neg_op = arith.NegFOp(block.arguments[0])
    exp_op = math.ExpOp(neg_op.result)
    one_op = arith.ConstantOp(mlir_dtype, mlir_element_attr_get(dtype, 1))
    add_op = arith.AddFOp(one_op.result, exp_op.result)
    div_op = arith.DivFOp(block.arguments[0], add_op.result)
    block.append(neg_op)
    block.append(exp_op)
    block.append(one_op)
    block.append(add_op)
    block.append(div_op)
    block.append(linalg.YieldOp([div_op.result]))

    return op


def where_op(
    node: WhereOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor where operation.
    From Buddy WhereOp to MLIR linalg `generic` operation.

    Note: This op, compute input node's silu activation result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    assert len(node.args) == 3
    input1 = symbol_table.get((str(node.args[0]), 0))
    input2 = symbol_table.get((str(node.args[1]), 0))
    input3 = symbol_table.get((str(node.args[2]), 0))
    if input1 is None or input2 is None or input3 is None:
        return
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    output = tensor.EmptyOp(output_shape, mlir_dtype)

    if not isinstance(input2.type, ir.RankedTensorType):
        input2 = tensor.SplatOp(tensor_type, input2, []).result
    if not isinstance(input3.type, ir.RankedTensorType):
        input3 = tensor.SplatOp(tensor_type, input3, []).result

    generic_map = ir.AffineMap.get_permutation(
        [i for i in range(len(output_shape))]
    )
    op = linalg.GenericOp(
        [tensor_type],
        [input1, input2, input3],
        [output],
        ir.ArrayAttr.get(
            [
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [i for i in range(len(output_shape))]
                    )
                ),
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [i for i in range(len(output_shape))]
                    )
                ),
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [i for i in range(len(output_shape))]
                    )
                ),
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [i for i in range(len(output_shape))]
                    )
                ),
            ]
        ),
        ir.ArrayAttr.get(
            [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
            * len(output_shape)
        ),
    )
    block = ir.Block.create_at_start(
        op.region,
        [
            ir.RankedTensorType(input1.type).element_type,
            ir.RankedTensorType(input2.type).element_type,
            ir.RankedTensorType(input3.type).element_type,
            ir.RankedTensorType(output.result.type).element_type,
        ],
    )
    select_op = arith.SelectOp(
        block.arguments[0], block.arguments[1], block.arguments[2]
    )
    block.append(select_op)
    block.append(linalg.YieldOp([select_op.result]))

    return op


def scalar_tensor_op(node: ScalarTensorOp, symbol_table):
    """
    Import the tensor Scalar_Tensor operation.
    From Buddy ScalarTensorOp to MLIR arith `ConstantOp` operation.
    """
    assert len(node.args) == 1
    dtype = node.tensor_meta["dtype"]
    attr = mlir_element_attr_get(dtype, node.args[0])
    op = arith.ConstantOp(dtype, attr)

    return op


def split_op(node: SplitOp, symbol_table):
    """
    Split the input tensor into smaller tensors along the specified dimension.

    Args:
        node (SplitOp): The split operation node with metadata.
        symbol_table: Mapping of variable names to tensor references.

    Returns:
        List[Tensor]: List of split tensors.
    """
    # Get the input tensor and parameters
    input_tensor = symbol_table.get((str(node.args[0]), 0), node.args[0])
    split_size = node.args[1]  # Size of each split tensor
    input_shape = input_tensor.type.shape
    dim = node.args[2]  # Dimension to split along
    if dim < 0:
        dim += len(input_shape)

    split_count = (input_shape[dim] + split_size - 1) // split_size  # Round up
    tensor_rank = len(input_shape)
    default_sizes = list(input_shape)
    default_strides = [1] * tensor_rank
    splits = []

    for i in range(split_count):
        # Calculate the offset along the specified dimension
        offsets = [0] * tensor_rank
        offsets[dim] = i * split_size
        offsets_attr = ir._denseI64ArrayAttr(offsets, None)

        # Set the size along the split dimension;
        # the last slice may be smaller than split_size
        sizes = list(default_sizes)
        sizes[dim] = min(split_size, input_shape[dim] - i * split_size)
        sizes_attr = ir._denseI64ArrayAttr(sizes, None)

        # The stride for each dimension is set to 1 by default
        strides = list(default_strides)
        strides_attr = ir._denseI64ArrayAttr(strides, None)

        output_shape = list(node.tensor_meta["shape"][i])
        dtype = node.tensor_meta["dtype"][i]
        mlir_dtype = mlir_element_type_get(dtype)
        tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)

        slice_op = tensor.ExtractSliceOp(
            tensor_type,
            input_tensor,
            [],
            [],
            [],
            offsets_attr,
            sizes_attr,
            strides_attr,
        )
        splits.append(slice_op.result)

    return splits


def max_op(node: MaxOp, symbol_table):
    """
    Computes the maximum value from the input tensor and returns it as a tensor.

    Args:
        node: The operation node containing input tensor information.
        symbol_table: A table mapping identifiers to tensor values.

    Returns:
        A tensor containing the maximum value extracted from the input tensor.
    """
    input1 = symbol_table.get((str(node.args[0]), 0), node.args[0])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    output_shape = node.tensor_meta["shape"]
    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    input_shape = ir.RankedTensorType(input1.type).shape

    total_size = 1
    for x in input_shape:
        total_size *= x
    reshape_op = tosa.ReshapeOp(
        input1, memoryview(array.array("i", [total_size]))
    )

    argmax_result = ir.RankedTensorType.get([], ir.IntegerType.get_signless(64))
    argmax_op = tosa.ArgMaxOp(argmax_result, reshape_op.result, 0)
    index_value = tensor.ExtractOp(argmax_op, [])
    index = arith.IndexCastOp(ir.IndexType.get(), index_value)
    max_value = tensor.ExtractOp(reshape_op, index)
    output = tensor.FromElementsOp(tensor_type, max_value)

    return output


def gt_op(node: GtOp, symbol_table):
    """
    Compares an input tensor with a scalar value to determine element-wise greater than.

    Parameters:
    - node: The operation node containing arguments and metadata.
    - symbol_table: A mapping of tensor names to their corresponding MLIR objects.

    Returns:
    - cmp_op: A comparison operation result indicating where the input tensor's elements
              are greater than the scalar.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0), node.args[0])
    input_dtype = ir.RankedTensorType(input_tensor.type).element_type
    input_shape = ir.RankedTensorType(input_tensor.type).shape
    tensor_type = ir.RankedTensorType.get(input_shape, input_dtype)
    scalar = arith.ConstantOp(input_dtype, node.args[1])
    rhs = tensor.SplatOp(tensor_type, scalar, [])
    if str(input_dtype).find("i") != -1:
        cmp_op = arith.CmpIOp(4, input_tensor, rhs)
    else:
        cmp_op = arith.CmpFOp(2, input_tensor, rhs)

    return cmp_op


def ge_op(
    node: GeOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor greater equal operation.
    From buddy GreaterEqualOp to MLIR arith `constant` operation.
    Note: This op, campare two input nodes, and output bool tensor to represent
    compare result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.
    Returns:
        op: The operation return the linalg.generic op.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0), node.args[0])
    input_dtype = ir.RankedTensorType(input_tensor.type).element_type
    input_shape = ir.RankedTensorType(input_tensor.type).shape
    tensor_type = ir.RankedTensorType.get(input_shape, input_dtype)

    scalar = arith.ConstantOp(input_dtype, node.args[1])
    rhs = tensor.SplatOp(tensor_type, scalar, [])

    if str(input_dtype).find("i") != -1:
        cmp_op = arith.CmpIOp(5, input_tensor, rhs)
    else:
        cmp_op = arith.CmpFOp(3, input_tensor, rhs)

    return cmp_op


def greater_than_op(
    node: GreaterThanOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor greater than operation.
    From buddy GreaterThanOp to MLIR arith `constant` operation.
    Note: This op, campare two input nodes, and output bool tensor to represent
    compare result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.
    Returns:
        op: The operation return the linalg.generic op.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    input2 = symbol_table.get((str(node.args[1]), 0))
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    # value = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 4)
    shp1 = list(ir.RankedTensorType(ir.Value(input1).type).shape)
    shp2 = list(ir.RankedTensorType(ir.Value(input2).type).shape)
    dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(output_shape, dtype)
    output = tensor.EmptyOp(output_shape, dtype)
    if len(shp1) < len(shp2):
        if int(shp1[-1]) > 1 and shp2[-1] == 1:
            generic_map = ir.AffineMap.get_permutation(
                [i for i in range(len(shp2) + 1)]
            )
            op = linalg.GenericOp(
                [tensor_type],
                [input1, input2],
                [output],
                ir.ArrayAttr.get(
                    [
                        ir.AffineMapAttr.get(
                            generic_map.get_submap(
                                [
                                    i
                                    for i in range(
                                        len(shp2) - len(shp1), len(shp2)
                                    )
                                ]
                            )
                        ),
                        ir.AffineMapAttr.get(
                            generic_map.get_submap(
                                [i for i in range(0, len(shp2) - 1)]
                                + [len(shp2)]
                            )
                        ),
                        ir.AffineMapAttr.get(
                            generic_map.get_submap(
                                [i for i in range(0, len(shp2))]
                            )
                        ),
                    ]
                ),
                ir.ArrayAttr.get(
                    [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
                    * len(shp2)
                    + [ir.Attribute.parse("#linalg.iterator_type<reduction>")]
                ),
            )
            block = ir.Block.create_at_start(
                op.region,
                [
                    ir.RankedTensorType(input2.type).element_type,
                    ir.RankedTensorType(input2.type).element_type,
                    dtype,
                ],
            )
            if (
                str(ir.RankedTensorType(input2.type).element_type).find("i")
                != -1
            ):
                cmpop = arith.CmpIOp(4, block.arguments[0], block.arguments[1])
            else:
                cmpop = arith.CmpFOp(2, block.arguments[0], block.arguments[1])
            block.append(cmpop)
            block.append(linalg.YieldOp([cmpop.result]))

    return op


def unsafe_index_op(
    node: UnsafeIndexOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor _unsafe_index operation.
    From buddy UnsafeIndexOp to MLIR linalg `generic`
    operation.
    Note: This op, get input node slice result by input index.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.
    Returns:
        op: The operation return the linalg.generic op.
    """
    assert len(node.args) == 2
    input1 = symbol_table.get((str(node.args[0]), 0))
    if input1 is None:
        return
    input1_shape = ir.RankedTensorType(input1.type).shape
    input2 = node.args[1]
    have_none = False
    for i in input2:
        if i == None:
            have_none = True
            break
    input2_dim_sum = 0
    for i in range(len(input2)):
        input2_dim_sum += (
            len(symbol_table.get((str(input2[i]), 0)).type.shape)
            if input2[i] != None
            else 0
        )
    output_shape = list(node.tensor_meta["shape"])
    input_shape = input1.type.shape
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    if len(input2) < len(input1_shape):
        tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
        output = tensor.EmptyOp(output_shape, mlir_dtype)
        generic_map = ir.AffineMap.get_permutation(
            [i for i in range(max(len(output_shape), len(input_shape)))]
        )
        input_map = []
        for i in range(len(input2)):
            input2_shape = symbol_table.get((str(input2[i]), 0)).type.shape
            input_map.append(
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [j for j in range(i, i + len(input2_shape))]
                    )
                )
            )
        if len(input_shape) > len(output_shape):
            input_map.append(
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [
                            j
                            for j in range(
                                len(input_shape) - len(output_shape),
                                len(input_shape),
                            )
                        ]
                    )
                )
            )
        else:
            input_map.append(
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [j for j in range(len(output_shape))]
                    )
                )
            )
        operands = [symbol_table.get((str(i), 0)) for i in input2]
        op = linalg.GenericOp(
            [tensor_type],
            operands,
            [output],
            ir.ArrayAttr.get(input_map),
            ir.ArrayAttr.get(
                [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
                * max(len(output_shape), len(input_shape))
            ),
        )
        arguments = [
            ir.RankedTensorType(i.type).element_type for i in operands
        ] + [ir.RankedTensorType(output.result.type).element_type]
        block = ir.Block.create_at_start(op.region, arguments)
        index = []
        for i in block.arguments[:-1]:
            indexcast_op = arith.IndexCastOp(ir.IndexType.get(), i)
            block.append(indexcast_op)
            index.append(indexcast_op.result)
        for i in range(
            input2_dim_sum, max(len(input_shape), len(output_shape))
        ):
            index_op = linalg.IndexOp(ir._i64Attr(i, None))
            block.append(index_op)
            index.append(index_op.result)
        value = tensor.ExtractOp(input1, index)
        block.append(value)
        block.append(linalg.YieldOp([value.result]))
    else:
        tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
        output = tensor.EmptyOp(output_shape, mlir_dtype)
        generic_map = ir.AffineMap.get_permutation(
            [i for i in range(max(len(output_shape), len(input_shape)))]
        )
        input_map = []
        for i in range(len(input2)):
            if input2[i] == None:
                continue
            input2_shape = symbol_table.get((str(input2[i]), 0)).type.shape
            if have_none:
                input_map.append(
                    ir.AffineMapAttr.get(
                        generic_map.get_submap([j for j in range(i, i + 1)])
                    )
                )
        if len(input_shape) > len(output_shape):
            input_map.append(
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [
                            j
                            for j in range(
                                len(input_shape) - len(output_shape),
                                len(input_shape),
                            )
                        ]
                    )
                )
            )
        else:
            input_map.append(
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [j for j in range(len(output_shape))]
                    )
                )
            )
        if have_none:
            operands = []
            for i in input2:
                if i == None:
                    continue
                input2_ = symbol_table.get((str(i), 0))
                input2_shape = input2_.type.shape
                if i != None and len(input2_shape) > 1:
                    total_size = 1
                    for x in input2_shape:
                        total_size *= x
                    reshape_op = tosa.ReshapeOp(
                        input2_, memoryview(array.array("i", [total_size]))
                    )
                operands.append(reshape_op.result)

        else:
            operands = [symbol_table.get((str(i), 0)) for i in input2]
        op = linalg.GenericOp(
            [tensor_type],
            operands,
            [output],
            ir.ArrayAttr.get(input_map),
            ir.ArrayAttr.get(
                [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
                * max(len(output_shape), len(input_shape))
            ),
        )
        arguments = [
            ir.RankedTensorType(i.type).element_type for i in operands
        ] + [ir.RankedTensorType(output.result.type).element_type]
        block = ir.Block.create_at_start(op.region, arguments)
        index = []
        None_count = 0
        for i in range(len(input2)):
            if input2[i] == None:
                None_count += 1
                index_op = linalg.IndexOp(ir._i64Attr(i, None))
                block.append(index_op)
                index.append(index_op.result)
            else:
                indexcast_op = arith.IndexCastOp(
                    ir.IndexType.get(), block.arguments[i - None_count]
                )
                block.append(indexcast_op)
                index.append(indexcast_op.result)
        value = tensor.ExtractOp(input1, index)
        block.append(value)
        block.append(linalg.YieldOp([value.result]))
    return op


def equal_op(
    node: EqualOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor equal operation.
    Converts Buddy EqualOp to the MLIR arith `CmpIOp` or `CmpFOp` operation.

    This operation compares two input tensors and produces a boolean tensor
    representing the comparison result.

    Args:
        node: The input graph node containing operation details.
        symbol_table: A dictionary mapping symbols to their corresponding
                      operations.

    Returns:
        op: A linalg.generic operation that performs element-wise equality
            comparison.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0), node.args[0])
    input_dtype = ir.RankedTensorType(input_tensor.type).element_type
    input_shape = ir.RankedTensorType(input_tensor.type).shape
    tensor_type = ir.RankedTensorType.get(input_shape, input_dtype)
    if str(input_dtype).find("i") == -1:
        scalar = arith.ConstantOp(input_dtype, float(node.args[1]))
    else:
        scalar = arith.ConstantOp(input_dtype, node.args[1])
    rhs = tensor.SplatOp(tensor_type, scalar, [])
    if str(input_dtype).find("i") != -1:
        cmp_op = arith.CmpIOp(0, input_tensor, rhs)
    else:
        cmp_op = arith.CmpFOp(1, input_tensor, rhs)

    return cmp_op


def copy_op(node: CopyOp, symbol_table):
    """
    Import the tensor copy operation.
    Converts Buddy CopyOp to an equivalent MLIR linalg.generic operation.

    This operation copies data from the source tensor to the destination tensor.

    Args:
        node: The input graph node containing operation details.
        symbol_table: A dictionary mapping symbols to their corresponding
                      operations.

    Returns:
        op: A linalg.generic operation that performs element-wise copying
            from input to output.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    input2 = symbol_table.get((str(node.args[1]), 0))

    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    output = tensor.EmptyOp(output_shape, mlir_dtype)
    generic_map = ir.AffineMap.get_permutation(
        [i for i in range(len(output_shape))]
    )
    op = linalg.GenericOp(
        [tensor_type],
        [input2],
        [input1],
        ir.ArrayAttr.get(
            [
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [i for i in range(len(output_shape))]
                    )
                ),
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [i for i in range(len(output_shape))]
                    )
                ),
            ]
        ),
        ir.ArrayAttr.get(
            [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
            * len(output_shape)
        ),
    )
    block = ir.Block.create_at_start(
        op.region,
        [
            ir.RankedTensorType(input1.type).element_type,
            ir.RankedTensorType(output.result.type).element_type,
        ],
    )
    block.append(linalg.YieldOp([block.arguments[0]]))

    return op


def slice_scatter_op(node: SliceScatterOp, symbol_table):
    """
    Scatter a source tensor into a slice of the input tensor.

    Args:
        node (SliceScatterOp): The slice_scatter operation node.
        symbol_table: Mapping of variable names to tensor references.

    Returns:
        Tensor: The resulting tensor after inserting the source tensor.
    """
    # Retrieve input tensor and scatter-related parameters
    input_tensor = symbol_table.get((str(node.args[0]), 0), node.args[0])
    source_tensor = symbol_table.get((str(node.args[1]), 0), node.args[1])
    dim = node.args[2]  # The dimension to insert into
    start = node.args[3]  # Start index
    end = node.args[4]  # End index

    input_shape = input_tensor.type.shape
    if dim < 0:
        dim += len(input_shape)  # Handle negative indices

    if end == 9223372036854775807:
        end = input_shape[dim]  # Adjust end index if it is set to max value

    tensor_rank = len(input_shape)
    default_sizes = list(input_shape)
    default_strides = [1] * tensor_rank

    # 1. Compute slice offsets
    offsets = [0] * tensor_rank
    offsets[dim] = start  # Offset only in the target dimension
    offsets_attr = ir._denseI64ArrayAttr(offsets, None)

    # 2. Compute slice sizes
    sizes = list(default_sizes)
    sizes[dim] = end - start  # Modify only the target dimension size
    sizes_attr = ir._denseI64ArrayAttr(sizes, None)

    # 3. Compute slice strides
    strides = list(default_strides)
    strides_attr = ir._denseI64ArrayAttr(strides, None)

    # 4. Extract target slice
    slice_op = tensor.ExtractSliceOp(
        source_tensor.type,  # Target type is the same as source_tensor
        input_tensor,
        [],
        [],
        [],
        offsets_attr,
        sizes_attr,
        strides_attr,
    )

    # 5. Insert source_tensor into the target position
    insert_op = tensor.InsertSliceOp(
        source_tensor,
        input_tensor,
        [],
        [],
        [],
        offsets_attr,
        sizes_attr,
        strides_attr,
    )

    return insert_op.result


ops_registry = {
    "TransposeMatmulFusedOp": matmul_transpose_b_op,
    "ArangeOp": arange_op,
    "UnsqueezeOp": unsqueeze_op,
    "ViewOp": view_op,
    "EmbeddingOp": embedding_op,
    "OnesOp": ones_op,
    "FullOp": full_op,
    "LessThanOp": lt_op,
    "MaskedFillOp": masked_fill_op,
    "SliceOp": slice_op,
    "ExpandOp": expand_op,
    "ToCopyOp": to_copy_op,
    "RsubOp": rsub_op,
    "PowOp": pow_op,
    "MeanOp": mean_op,
    "RsqrtOp": rsqrt_op,
    "MulOp": mul_op,
    "TOp": t_op,
    "TransposeOp": transpose_op,
    "IndexOp": index_op,
    "NegOp": neg_op,
    "CatOp": cat_op,
    "SqueezeOp": squeeze_op,
    "BatchMatmulOp": batch_matmul_op,
    "DivOp": div_op,
    "SoftmaxOp": softmax_op,
    "CloneOp": clone_op,
    "SiluOp": silu_op,
    "AddOp": add_op,
    "WhereOp": where_op,
    "ScalarTensorOp": scalar_tensor_op,
    "SplitOp": split_op,
    "MaxOp": max_op,
    "GtOp": gt_op,
    "GeOp": ge_op,
    "GreaterThanOp": greater_than_op,
    "UnsafeIndexOp": unsafe_index_op,
    "EqualOp": equal_op,
    "CopyOp": copy_op,
    "SliceScatterOp": slice_scatter_op,
}
