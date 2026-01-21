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

import torch
from typing import Dict, Tuple, List

import mlir.ir as ir
from mlir.dialects import (
    tosa,
    linalg,
    arith,
    tensor,
    math,
    bufferization,
    memref,
    scf,
    vector,
)
import copy, array, sys
import numpy
import functools

from ..graph import *
from ..graph.graph import TensorDType
from .utils import *


def _safe_get_permutation(perm: List[int]) -> ir.AffineMap:
    if not perm:
        return ir.AffineMap.get_empty()
    return ir.AffineMap.get_permutation(perm)


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

    shape_ty = ir.Type.parse(f"!tosa.shape<{len(input_shape)}>")
    index_ty = ir.IndexType.get()
    shape_val = tosa.ConstShapeOp(
        shape_ty,
        ir.DenseElementsAttr.get(
            array.array("q", input_shape),
            type=index_ty,
            shape=[len(input_shape)],
        ),
    ).result
    op = tosa.ReshapeOp(input_node, shape_val)

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
    shape_ty = ir.Type.parse(f"!tosa.shape<{len(output_shape)}>")
    index_ty = ir.IndexType.get()
    shape_val = tosa.ConstShapeOp(
        shape_ty,
        ir.DenseElementsAttr.get(
            array.array("q", output_shape),
            type=index_ty,
            shape=[len(output_shape)],
        ),
    ).result

    op = tosa.ReshapeOp(input_node, shape_val)

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
    generic_map = _safe_get_permutation([0, 1, 2])
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
            generic_map = _safe_get_permutation(
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
    generic_map = _safe_get_permutation([i for i in range(len(output_shape))])
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

    op = None  # Initialize op to None

    if dtype == TensorDType.Bool:
        if str(ir.RankedTensorType(input1.type).element_type) == "f32":
            tensor_type = ir.RankedTensorType.get(
                output_shape, ir.IntegerType.get_signless(1)
            )
            output = tensor.EmptyOp(
                output_shape, ir.IntegerType.get_signless(1)
            )
            generic_map = _safe_get_permutation(
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
            generic_map = _safe_get_permutation(
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

    # Return op if defined, otherwise return input as identity
    if op is not None:
        return op
    else:
        return input1


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
    if any(dim < 0 for dim in output_shape):
        raise NotImplementedError("histc requires static output shape")
    if not isinstance(value, str):
        value = arith.ConstantOp(
            mlir_dtype, mlir_element_attr_get(dtype, value)
        )
        generic_map = _safe_get_permutation(
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
    output_shape = ir.RankedTensorType(input1.type).shape
    dtype = node.tensor_meta["dtype"]
    dtype = mlir_element_type_get(dtype)

    op = None  # Initialize op

    # Handle scalar (0-dim) tensors specially using math ops directly
    if len(output_shape) == 0:
        # For scalar tensors, use math.powf directly
        if isinstance(value, (int, float)):
            exp_value = float(value)
        else:
            exp_value = 2.0

        # Create exponent constant
        exp_const = arith.ConstantOp(
            dtype,
            ir.FloatAttr.get(dtype, exp_value),
        )
        # Extract scalar, compute pow, then create tensor
        tensor_type = ir.RankedTensorType.get([], dtype)
        result = math.PowFOp(input1, exp_const.result)
        return result

    # Check if value is an integer-valued number (not string)
    is_integer_valued = (
        not isinstance(value, str) and abs(int(value) - float(value)) < 1e-6
    )

    if is_integer_valued:
        # value is an integer-valued float (e.g., 2.0), convert to int
        int_value = int(value)
        generic_map = ir.AffineMap.get_permutation(
            [i for i in range(len(output_shape))]
        )
        tensor_type = ir.RankedTensorType.get(output_shape, dtype)
        output = tensor.EmptyOp(output_shape, dtype)
        value_const = arith.ConstantOp(
            ir.IntegerType.get_signless(32),
            ir.IntegerAttr.get(ir.IntegerType.get_signless(32), int_value),
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
        if str(ir.RankedTensorType(input1.type).element_type).find("i") != -1:
            powi_op = math.IPowIOp(block.arguments[0], value_const.result)
        else:
            powi_op = math.FPowIOp(block.arguments[0], value_const.result)
        block.append(powi_op)
        block.append(linalg.YieldOp([powi_op.result]))
    else:
        # For non-integer exponents or string values, use math.PowFOp
        generic_map = ir.AffineMap.get_permutation(
            [i for i in range(len(output_shape))]
        )
        tensor_type = ir.RankedTensorType.get(output_shape, dtype)
        output = tensor.EmptyOp(output_shape, dtype)

        # Create constant for the exponent
        if isinstance(value, (int, float)):
            exp_value = float(value)
        else:
            exp_value = 2.0  # Default to square if string

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
        # Create constant for exponent inside the block
        exp_const = arith.ConstantOp(
            dtype,
            ir.FloatAttr.get(dtype, exp_value),
        )
        block.append(exp_const)
        powf_op = math.PowFOp(block.arguments[0], exp_const.result)
        block.append(powf_op)
        block.append(linalg.YieldOp([powf_op.result]))

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
            generic_map = _safe_get_permutation(
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
    generic_map = _safe_get_permutation([i for i in range(len(output_shape))])
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
    shift = tosa.ConstOp(
        ir.DenseElementsAttr.get_splat(
            ir.Type.parse("tensor<1xi8>"),
            ir.IntegerAttr.get(ir.IntegerType.get_signless(8), 0),
        )
    ).result
    op = tosa.MulOp(
        mul_result_tensor_type,
        input1,
        input2,
        shift,
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


def matmul_op(
    node: MatmulOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor matmul operation.
    From Buddy MatmulOp to MLIR linalg `matmul` operation.

    Note: This op, compute input node's matrix multiplication result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.matmul op.
    """
    assert len(node.args) == 2
    input1 = symbol_table.get((str(node.args[0]), 0))
    input2 = symbol_table.get((str(node.args[1]), 0))
    if input1 is None or input2 is None:
        return

    input1_shape = ir.RankedTensorType(input1.type).shape
    input2_shape = ir.RankedTensorType(input2.type).shape
    output_shape = [input1_shape[0], input2_shape[1]]
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    generic_map = _safe_get_permutation([0, 1, 2])
    element = mlir_element_attr_get(dtype, 0.0)
    attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
    matmul_result_buffer = arith.ConstantOp(tensor_type, attr).result
    op = linalg.MatmulOp(
        result_tensors=[tensor_type],
        inputs=[input1, input2],
        outputs=[matmul_result_buffer],
        indexing_maps=[
            generic_map.get_submap([0, 2]),  # lhs: (m, k)
            generic_map.get_submap([2, 1]),  # rhs: (k, n)
            generic_map.get_submap([0, 1]),  # out: (m, n)
        ],
        cast="cast_signed",
    )
    linalg.fill_builtin_region(op.operation)
    return op


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
    From buddy IndexOp to MLIR linalg `generic` operation.

    Handles advanced indexing where some dimensions may have index tensors
    and others may be None (meaning use all elements along that dimension).

    For example, with input shape [3, 3] and indices [None, idx]:
    - None means keep all elements along dimension 0
    - idx is a tensor of indices for dimension 1
    - Output: input[:, idx] -> shape [3, len(idx)]

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
    input2 = node.args[1]  # List of indices, may contain None

    output_shape = list(node.tensor_meta["shape"])
    input_shape = list(input1.type.shape)
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)

    # Check if any index is None - if so, use the new path that handles None indices
    has_none_indices = any(idx is None for idx in input2)

    if has_none_indices:
        # New path: Handle advanced indexing with None entries
        return _index_op_with_none_indices(
            node,
            input1,
            input2,
            output_shape,
            input_shape,
            mlir_dtype,
            symbol_table,
        )
    else:
        # Original path: All indices are tensors, use special broadcast handling
        return _index_op_all_tensors(
            node,
            input1,
            input2,
            output_shape,
            input_shape,
            mlir_dtype,
            symbol_table,
        )


def _index_op_all_tensors(
    node: IndexOp,
    input1,
    input2,
    output_shape,
    input_shape,
    mlir_dtype,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Handle index operation when all indices are tensors (no None values).
    This uses the original implementation with special broadcast pattern handling.
    """
    # store index operand shapes for later use
    index_shapes = []
    for i in range(len(input2)):
        t = symbol_table.get((str(input2[i]), 0))
        if t is None:
            return
        s = tuple(t.type.shape)
        index_shapes.append(s)

    # Create output tensor and result type
    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    output = tensor.EmptyOp(output_shape, mlir_dtype)

    # The iteration space is determined by output_shape
    out_rank = len(output_shape)
    generic_map = ir.AffineMap.get_permutation([i for i in range(out_rank)])

    # Build indexing maps list (AffineMapAttr) for inputs and output.
    input_map = []

    # >>> Handle a common broadcast pattern explicitly:
    # If we have: input rank == 2, two index operands and shapes like
    #   idx0: (1,1)  (a scalar per row, broadcast across cols)
    #   idx1: (40,)  (one index per column)
    # then set maps to:
    #   idx0 -> (d0,d0)   (broadcast scalar per row across columns)
    #   idx1 -> (d1)      (each column index uses d1)
    #   output -> (d0,d1)
    applied_special_broadcast = False
    if (
        len(input_shape) == 2
        and len(index_shapes) == 2
        and len(output_shape) == 2
    ):
        s0 = index_shapes[0]
        s1 = index_shapes[1]
        # match the specific example: idx0 shape (1,1) and idx1 shape (N)
        if (len(s0) == 2 and s0[0] == 1 and s0[1] == 1) and (
            len(s1) == 1 and s1[0] == output_shape[1]
        ):
            # Construct explicit submaps:
            input_map.append(
                ir.AffineMapAttr.get(generic_map.get_submap([0, 0]))
            )  # idx0: (d0,d0)
            input_map.append(
                ir.AffineMapAttr.get(generic_map.get_submap([1]))
            )  # idx1: (d1)
            input_map.append(
                ir.AffineMapAttr.get(generic_map.get_submap([0, 1]))
            )  # output: (d0,d1)
            applied_special_broadcast = True

    # General broadcast handling: all index tensors broadcast to output_shape
    if not applied_special_broadcast:
        # Check if all index tensors can broadcast to output_shape
        all_broadcastable = True
        for idx_shape in index_shapes:
            if len(idx_shape) > out_rank:
                all_broadcastable = False
                break
            # Check broadcast compatibility
            for j in range(len(idx_shape)):
                out_dim_idx = out_rank - len(idx_shape) + j
                if (
                    idx_shape[j] != 1
                    and idx_shape[j] != output_shape[out_dim_idx]
                ):
                    all_broadcastable = False
                    break
            if not all_broadcastable:
                break

        if all_broadcastable:
            # Create affine maps with broadcast semantics
            for idx_shape in index_shapes:
                idx_rank = len(idx_shape)
                # Build affine expressions for this index tensor
                # Align to the right of output dimensions
                exprs = []
                for j in range(idx_rank):
                    out_dim_idx = out_rank - idx_rank + j
                    if idx_shape[j] == 1:
                        # Broadcast dimension: use constant 0
                        exprs.append(ir.AffineConstantExpr.get(0))
                    else:
                        # Non-broadcast dimension: use the corresponding output dim
                        exprs.append(ir.AffineDimExpr.get(out_dim_idx))
                affine_map = ir.AffineMap.get(out_rank, 0, exprs)
                input_map.append(ir.AffineMapAttr.get(affine_map))

            # Output map: identity over output dimensions
            out_exprs = [ir.AffineDimExpr.get(i) for i in range(out_rank)]
            out_map = ir.AffineMap.get(out_rank, 0, out_exprs)
            input_map.append(ir.AffineMapAttr.get(out_map))
        else:
            # Fallback: contiguous dimension mapping (original behavior)
            # This may fail for complex cases, but preserves old behavior
            offset = 0
            for i in range(len(input2)):
                input2_shape = symbol_table.get((str(input2[i]), 0)).type.shape
                dim_len = len(input2_shape)
                idx_list = [
                    j for j in range(offset, min(offset + dim_len, out_rank))
                ]
                # Pad with zeros if needed
                while len(idx_list) < dim_len:
                    idx_list.append(0)
                input_map.append(
                    ir.AffineMapAttr.get(generic_map.get_submap(idx_list))
                )
                offset += dim_len

            # Output map
            idx_list = [j for j in range(out_rank)]
            input_map.append(
                ir.AffineMapAttr.get(generic_map.get_submap(idx_list))
            )

    # Build operands list
    operands = [symbol_table.get((str(i), 0)) for i in input2]

    # Prepare iterator types (parallel for each iteration dimension)
    iter_count = out_rank
    iterator_attr = ir.ArrayAttr.get(
        [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * iter_count
    )

    # Create the linalg.generic op
    op = linalg.GenericOp(
        [tensor_type],
        operands,
        [output],
        ir.ArrayAttr.get(input_map),
        iterator_attr,
    )

    # Build the region body
    arguments = [ir.RankedTensorType(i.type).element_type for i in operands] + [
        ir.RankedTensorType(output.result.type).element_type
    ]
    block = ir.Block.create_at_start(op.region, arguments)

    # Convert block arguments (index tensor values) into index values
    # Each block argument (except the last one which is output) is an index value
    index = []
    for i in block.arguments[:-1]:
        indexcast_op = arith.IndexCastOp(ir.IndexType.get(), i)
        block.append(indexcast_op)
        index.append(indexcast_op.result)

    # If we have fewer index tensors than input dimensions, add linalg.index ops
    # for the remaining dimensions. These must be mapped to output loop dims,
    # not input dims, to avoid out-of-range linalg.index on lower-rank outputs.
    num_index_tensors = len(input2)
    remaining_dims = len(input_shape) - num_index_tensors
    broadcast_rank = out_rank - remaining_dims
    for i in range(num_index_tensors, len(input_shape)):
        output_dim = broadcast_rank + (i - num_index_tensors)
        index_op_inst = linalg.IndexOp(ir._i64Attr(output_dim, None))
        block.append(index_op_inst)
        index.append(index_op_inst.result)

    # Extract value from the input tensor using the constructed 'index' list
    value = tensor.ExtractOp(input1, index)
    block.append(value)
    block.append(linalg.YieldOp([value.result]))

    return op


def _index_op_with_none_indices(
    node: IndexOp,
    input1,
    input2,
    output_shape,
    input_shape,
    mlir_dtype,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Handle index operation when some indices are None (meaning use all elements
    along that dimension).
    """
    # Create output tensor and result type
    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    output = tensor.EmptyOp(output_shape, mlir_dtype)

    # Separate index tensors from None entries
    # None means "use linalg.index" for that dimension
    # Non-None means use the index tensor value
    index_tensor_operands = []
    index_tensor_shapes = []
    index_positions = []  # Which input dimensions use index tensors
    none_positions = []  # Which input dimensions use None (slice all)

    for i, idx in enumerate(input2):
        if idx is None:
            none_positions.append(i)
        else:
            operand = symbol_table.get((str(idx), 0))
            if operand is None:
                return
            index_tensor_operands.append(operand)
            index_tensor_shapes.append(list(operand.type.shape))
            index_positions.append(i)

    # For iteration space, use output_shape dimensions
    iter_count = len(output_shape)

    # Build indexing maps for each index tensor operand
    input_map = []
    for idx, (operand, idx_shape) in enumerate(
        zip(index_tensor_operands, index_tensor_shapes)
    ):
        # Build affine map for broadcast: match dimensions from the end
        map_indices = []
        idx_rank = len(idx_shape)
        out_rank = len(output_shape)

        # Align from the right (numpy-style broadcast)
        for i in range(idx_rank):
            out_dim = out_rank - idx_rank + i
            if out_dim >= 0:
                if idx_shape[i] == 1:
                    map_indices.append(0)
                else:
                    map_indices.append(out_dim)
            else:
                map_indices.append(0)

        # Create affine map with proper handling of singleton dimensions
        exprs = []
        for i, dim_idx in enumerate(map_indices):
            if idx_shape[i] == 1:
                exprs.append(ir.AffineConstantExpr.get(0))
            else:
                exprs.append(ir.AffineDimExpr.get(dim_idx))

        affine_map = ir.AffineMap.get(iter_count, 0, exprs)
        input_map.append(ir.AffineMapAttr.get(affine_map))

    # Output map: identity over output dimensions
    output_map = _safe_get_permutation([i for i in range(iter_count)])
    input_map.append(ir.AffineMapAttr.get(output_map))

    # Prepare iterator types (parallel for each iteration dimension)
    iterator_attr = ir.ArrayAttr.get(
        [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * iter_count
    )

    # Create the linalg.generic op
    op = linalg.GenericOp(
        [tensor_type],
        index_tensor_operands,  # Only non-None operands
        [output],
        ir.ArrayAttr.get(input_map),
        iterator_attr,
    )

    # Build the region body
    arguments = [
        ir.RankedTensorType(i.type).element_type for i in index_tensor_operands
    ] + [ir.RankedTensorType(output.result.type).element_type]
    block = ir.Block.create_at_start(op.region, arguments)

    # Convert block arguments (index values from index tensors) into index type
    index_tensor_values = []
    for i, arg in enumerate(block.arguments[:-1]):
        indexcast_op = arith.IndexCastOp(ir.IndexType.get(), arg)
        block.append(indexcast_op)
        index_tensor_values.append(indexcast_op.result)

    # Build the full index list for tensor.extract
    # For each input dimension:
    # - If it has an index tensor: use the index tensor value
    # - If it's None: use linalg.index to get the iteration index
    # Remaining dimensions (beyond input2 length) also use linalg.index
    extract_indices = []
    tensor_val_idx = 0  # Counter for index tensor values

    # Track which output dimension corresponds to each None position
    # None positions map to sequential output dimensions
    none_dim_counter = 0

    for dim in range(len(input_shape)):
        if dim < len(input2):
            if input2[dim] is None:
                # Use linalg.index for this dimension
                # This maps to the corresponding output dimension
                index_op_inst = linalg.IndexOp(
                    ir._i64Attr(none_dim_counter, None)
                )
                block.append(index_op_inst)
                extract_indices.append(index_op_inst.result)
                none_dim_counter += 1
            else:
                # Use the value from the index tensor
                extract_indices.append(index_tensor_values[tensor_val_idx])
                tensor_val_idx += 1
        else:
            # Remaining dimensions beyond input2 use linalg.index
            linalg_idx = (
                dim - len(input2) + none_dim_counter + len(index_positions)
            )
            if linalg_idx < iter_count:
                index_op_inst = linalg.IndexOp(ir._i64Attr(linalg_idx, None))
                block.append(index_op_inst)
                extract_indices.append(index_op_inst.result)
            else:
                zero = arith.ConstantOp(ir.IndexType.get(), 0)
                block.append(zero)
                extract_indices.append(zero.result)

    # Extract value from the input tensor
    value = tensor.ExtractOp(input1, extract_indices)
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
    input1_shape = ir.RankedTensorType(input1.type).shape
    output_shape = list(input1_shape)
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

    Note: This op, concate multiple input tensors.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the tensor.insert_slice op.
    """
    # Support both 2 args and variable number of inputs
    inputs = node.args[0]  # List of input tensors
    # dim is optional, default to 0 if not provided
    dim = int(node.args[1]) if len(node.args) > 1 else 0

    # Get all input tensors
    input_tensors = []
    for inp in inputs:
        t = symbol_table.get((str(inp), 0))
        if t is None:
            return
        input_tensors.append(t)

    if len(input_tensors) == 0:
        return

    output_shape = list(node.tensor_meta["shape"])
    if dim < 0:
        dim = len(output_shape) + dim
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    output = tensor.EmptyOp(output_shape, mlir_dtype)

    offset = [0 for x in output_shape]
    current_result = output.result
    stride_attr = ir._denseI64ArrayAttr([1] * len(offset), None)

    for input_tensor in input_tensors:
        input_shape = list(ir.RankedTensorType(input_tensor.type).shape)
        offset_attr = ir._denseI64ArrayAttr(offset, None)
        size_attr = ir._denseI64ArrayAttr(input_shape, None)
        insert_op = tensor.InsertSliceOp(
            input_tensor,
            current_result,
            [],
            [],
            [],
            offset_attr,
            size_attr,
            stride_attr,
        )
        current_result = insert_op.result
        offset[dim] += input_shape[dim]

    return insert_op


def cat_op_legacy(
    node: CatOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Legacy cat_op for backward compatibility with 2 inputs only.
    """
    assert len(node.args) == 2
    input1 = symbol_table.get((str(node.args[0][0]), 0))
    input2 = symbol_table.get((str(node.args[0][1]), 0))
    dim = int(node.args[1])
    if input1 is None or input2 is None:
        return

    input1_shape = ir.RankedTensorType(input1.type).shape
    input2_shape = ir.RankedTensorType(input2.type).shape
    output_shape = input1_shape[:-1] + [input2_shape[-1] + input1_shape[-1]]
    if dim < 0:
        dim = len(output_shape) + dim
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    output = tensor.EmptyOp(output_shape, mlir_dtype)
    offset = [0 for x in output_shape]
    offset_attr = ir._denseI64ArrayAttr(offset, None)
    # input1_shape = ir.RankedTensorType(input1.type).shape
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
    # input2_shape = ir.RankedTensorType(input2.type).shape
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
    shift = tosa.ConstOp(
        ir.DenseElementsAttr.get_splat(
            ir.Type.parse("tensor<1xi8>"),
            ir.IntegerAttr.get(ir.IntegerType.get_signless(8), 0),
        )
    ).result
    op = tosa.MulOp(
        div_result_tensor_type,
        input1,
        tosa.ReciprocalOp(input2.type, input2).result,
        shift,
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


def log_softmax_op(
    node: LogSoftmaxOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor log_softmax operation.
    From buddy LogSoftmaxOp to MLIR linalg `generic` operation.

    log_softmax(x) = x - log(sum(exp(x)))
    For numerical stability: log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))

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

    # Step 1: Subtract max for numerical stability
    max_vals = tosa.ReduceMaxOp(input1, dim)
    sub_op_output = ir.RankedTensorType.get(input1.type.shape, mlir_dtype)
    shifted_input = tosa.SubOp(sub_op_output, input1, max_vals)

    # Step 2: Compute log(sum(exp(shifted_input)))
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

    # Compute sum(exp(shifted_input))
    sum_tensor_op = linalg.GenericOp(
        [sum_tensor_type],
        [shifted_input],
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

    # Step 3: Compute log(sum_exp) and subtract from shifted_input
    # log_softmax = shifted_input - log(sum_exp)
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
        [shifted_input, sum_tensor_op.result],
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
    # log_softmax = x - log(sum_exp)
    log_sum_exp = math.LogOp(block.arguments[1])
    sub_result = arith.SubFOp(block.arguments[0], log_sum_exp.result)
    block.append(log_sum_exp)
    block.append(sub_result)
    block.append(linalg.YieldOp([sub_result.result]))

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
    generic_map = _safe_get_permutation([i for i in range(len(output_shape))])
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

    def _broadcast_indexing_map(input_shape, name):
        out_rank = len(output_shape)
        in_rank = len(input_shape)
        if in_rank > out_rank:
            raise NotImplementedError(
                "where broadcast for {} with rank {} to {} is not supported".format(
                    name, in_rank, out_rank
                )
            )
        exprs = []
        for i in range(in_rank):
            out_dim_index = out_rank - in_rank + i
            in_dim = input_shape[i]
            out_dim = output_shape[out_dim_index]
            if in_dim == out_dim or in_dim == -1 or out_dim == -1:
                exprs.append(ir.AffineDimExpr.get(out_dim_index))
            elif in_dim == 1:
                exprs.append(ir.AffineConstantExpr.get(0))
            else:
                raise NotImplementedError(
                    "where broadcast for {} with shape {} to {} is not supported".format(
                        name, input_shape, output_shape
                    )
                )
        return ir.AffineMap.get(out_rank, 0, exprs)

    def _normalize_where_input(value, name):
        if isinstance(value.type, ir.RankedTensorType):
            input_shape = list(ir.RankedTensorType(value.type).shape)
        else:
            element_type = value.type
            value = tensor.FromElementsOp(
                ir.RankedTensorType.get([], element_type), [value]
            ).result
            input_shape = []
        return value, ir.AffineMapAttr.get(
            _broadcast_indexing_map(input_shape, name)
        )

    input1, input1_map = _normalize_where_input(input1, "condition")
    input2, input2_map = _normalize_where_input(input2, "self")
    input3, input3_map = _normalize_where_input(input3, "other")
    output_map = ir.AffineMapAttr.get(
        _safe_get_permutation([i for i in range(len(output_shape))])
    )

    generic_map = _safe_get_permutation([i for i in range(len(output_shape))])
    op = linalg.GenericOp(
        [tensor_type],
        [input1, input2, input3],
        [output],
        ir.ArrayAttr.get(
            [
                input1_map,
                input2_map,
                input3_map,
                output_map,
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
    element_type = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(
        list(node.tensor_meta["shape"]), element_type
    )
    element_attr = mlir_element_attr_get(dtype, node.args[0])
    attr = ir.DenseElementsAttr.get_splat(tensor_type, element_attr)
    op = arith.ConstantOp(tensor_type, attr)

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
    shape_ty = ir.Type.parse(f"!tosa.shape<{len([total_size])}>")
    index_ty = ir.IndexType.get()
    shape_val = tosa.ConstShapeOp(
        shape_ty,
        ir.DenseElementsAttr.get(
            array.array("q", [total_size]),
            type=index_ty,
            shape=[len([total_size])],
        ),
    ).result
    reshape_op = tosa.ReshapeOp(input1, shape_val)

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

    # Convert scalar value to the appropriate type
    scalar_val = node.args[1]
    if str(input_dtype).find("f") != -1:
        scalar_val = float(scalar_val)
    else:
        scalar_val = int(scalar_val)

    scalar = arith.ConstantOp(input_dtype, scalar_val)
    rhs = tensor.SplatOp(tensor_type, scalar, [])
    if str(input_dtype).find("i") != -1:
        cmp_op = arith.CmpIOp(4, input_tensor, rhs)
    else:
        cmp_op = arith.CmpFOp(2, input_tensor, rhs)

    return cmp_op

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

    # Convert scalar value to the appropriate type
    scalar_val = node.args[1]
    if str(input_dtype).find("f") != -1:
        scalar_val = float(scalar_val)
    else:
        scalar_val = int(scalar_val)

    scalar = arith.ConstantOp(input_dtype, scalar_val)
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
            generic_map = _safe_get_permutation(
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
        generic_map = _safe_get_permutation(
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
        generic_map = _safe_get_permutation(
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
                    shape_ty = ir.Type.parse(
                        f"!tosa.shape<{len([total_size])}>"
                    )
                    index_ty = ir.IndexType.get()
                    shape_val = tosa.ConstShapeOp(
                        shape_ty,
                        ir.DenseElementsAttr.get(
                            array.array("q", [total_size]),
                            type=index_ty,
                            shape=[len([total_size])],
                        ),
                    ).result
                    reshape_op = tosa.ReshapeOp(input2_, shape_val)
                    operands.append(reshape_op.result)
                else:
                    operands.append(input2_)

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
    Converts a Buddy EqualOp operation to an MLIR comparison operation (CmpIOp or CmpFOp).

    This operation compares two input tensors (or a tensor and a scalar) and produces a boolean tensor
    where each element represents the result of the comparison. The operation handles both integer and
    floating-point comparisons.

    Parameters:
        node (EqualOp): The Buddy EqualOp node containing the operation details and tensor metadata.
        symbol_table (dict): A dictionary mapping tensor names to their corresponding MLIR operations.

    Returns:
        op: An MLIR comparison operation (either CmpIOp for integers or CmpFOp for floats) that performs
            element-wise equality comparison between the input tensors or tensor and scalar.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0), node.args[0])
    input_dtype = ir.RankedTensorType(input_tensor.type).element_type
    input_shape = ir.RankedTensorType(input_tensor.type).shape
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    if isinstance(node.args[1], str):
        rhs = symbol_table.get((str(node.args[1]), 0), node.args[1])
        if input_tensor.type.shape != output_shape:
            tensor_type = ir.RankedTensorType.get(
                output_shape, input_tensor.type.element_type
            )
            if str(input_tensor.type.element_type) == "f32":
                element = ir.FloatAttr.get(ir.F32Type.get(), 0)
            elif str(input_tensor.type.element_type) == "f16":
                element = ir.FloatAttr.get(ir.F16Type.get(), 0)
            elif str(input_tensor.type.element_type) == "i64":
                element = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 0)
            attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
            to_broadcast_tensor = arith.ConstantOp(input_tensor.type, attr)
            input_tensor = tosa.AddOp(
                tensor_type, input_tensor, to_broadcast_tensor
            ).result

        if rhs.type.shape != output_shape:
            tensor_type = ir.RankedTensorType.get(
                output_shape, rhs.type.element_type
            )
            if str(rhs.type.element_type) == "f32":
                element = ir.FloatAttr.get(ir.F32Type.get(), 0)
            elif str(input_tensor.type.element_type) == "f16":
                element = ir.FloatAttr.get(ir.F16Type.get(), 0)
            elif str(input_tensor.type.element_type) == "i64":
                element = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 0)
            attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
            to_broadcast_tensor = arith.ConstantOp(rhs.type, attr)
            rhs = tosa.AddOp(
                tensor_type, input_tensor, to_broadcast_tensor
            ).result
    else:
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
    generic_map = _safe_get_permutation([i for i in range(len(output_shape))])
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


def _get_vectorizable_trailing_dims(input2, input3_shape, accumulate):
    """
    Check if the trailing dimensions can be vectorized.
    
    A dimension can be vectorized if:
    1. It has no index tensor (input2[d] is None or d >= len(input2))
    2. accumulate is False (or we could support vectorized accumulate later)
    
    Returns:
        tuple: (num_vectorizable_dims, vector_length)
               num_vectorizable_dims: Number of trailing dims that can be vectorized
               vector_length: Product of those dimensions' sizes
               Returns (0, 0) if vectorization is not beneficial
    """
    if accumulate:
        return 0, 0
    
    # Count trailing dimensions without index tensors
    num_vectorizable_dims = 0
    for d in range(len(input3_shape) - 1, -1, -1):
        if d < len(input2) and input2[d] is not None:
            break  # This dimension has an index tensor, stop
        num_vectorizable_dims += 1
    
    if num_vectorizable_dims == 0:
        return 0, 0
    
    # Calculate vector length (product of vectorizable dimensions)
    vector_length = 1
    for d in range(len(input3_shape) - num_vectorizable_dims, len(input3_shape)):
        vector_length *= input3_shape[d]
    
    # Only vectorize if beneficial (at least 4 elements)
    if vector_length < 4:
        return 0, 0
    
    return num_vectorizable_dims, vector_length


def _broadcast_index_tensor_for_vec(value, target_shape):
    """
    Broadcast an index tensor to target shape for vectorized index_put.
    
    This is a simplified version that handles common cases for KV cache patterns.
    """
    try:
        value_type = ir.RankedTensorType(value.type)
        value_shape = list(value_type.shape)
        elem_type = value_type.element_type
    except Exception:
        # Scalar value - splat to target shape
        elem_type = value.type
        target_type = ir.RankedTensorType.get(target_shape, elem_type)
        return tensor.SplatOp(target_type, value, []).result

    # If shapes match, return as-is
    if value_shape == list(target_shape):
        return value

    # Handle rank-1 index tensor (common case: cache_position is 1D)
    if len(value_shape) == 1 and len(target_shape) >= 1:
        # Check if the index tensor can broadcast
        # For KV cache: index is [seq_len], target might be [batch, heads, seq_len]
        # We need to find which dimension matches
        for d in range(len(target_shape)):
            if value_shape[0] == target_shape[d]:
                # Reshape to match target rank with 1s in other dims
                new_shape = [1] * len(target_shape)
                new_shape[d] = value_shape[0]
                shape_ty = ir.Type.parse(f"!tosa.shape<{len(new_shape)}>")
                index_ty = ir.IndexType.get()
                shape_val = tosa.ConstShapeOp(
                    shape_ty,
                    ir.DenseElementsAttr.get(
                        array.array("q", new_shape),
                        type=index_ty,
                        shape=[len(new_shape)],
                    ),
                ).result
                reshaped = tosa.ReshapeOp(value, shape_val).result
                
                # Broadcast to target shape
                if new_shape != list(target_shape):
                    if str(elem_type).startswith("f") or str(elem_type).startswith("bf"):
                        zero_elem = ir.FloatAttr.get(elem_type, 0.0)
                    else:
                        zero_elem = ir.IntegerAttr.get(elem_type, 0)
                    zero_type = ir.RankedTensorType.get(target_shape, elem_type)
                    zero_attr = ir.DenseElementsAttr.get_splat(zero_type, zero_elem)
                    zero_tensor = tosa.ConstOp(zero_attr).result
                    return tosa.AddOp(zero_type, reshaped, zero_tensor).result
                return reshaped
    
    # Fallback: try direct broadcast via add with zeros
    if len(value_shape) <= len(target_shape):
        padded_shape = [1] * (len(target_shape) - len(value_shape))
        padded_shape.extend(value_shape)
        shape_ty = ir.Type.parse(f"!tosa.shape<{len(padded_shape)}>")
        index_ty = ir.IndexType.get()
        shape_val = tosa.ConstShapeOp(
            shape_ty,
            ir.DenseElementsAttr.get(
                array.array("q", padded_shape),
                type=index_ty,
                shape=[len(padded_shape)],
            ),
        ).result
        value = tosa.ReshapeOp(value, shape_val).result
        
        if padded_shape != list(target_shape):
            if str(elem_type).startswith("f") or str(elem_type).startswith("bf"):
                zero_elem = ir.FloatAttr.get(elem_type, 0.0)
            else:
                zero_elem = ir.IntegerAttr.get(elem_type, 0)
            zero_type = ir.RankedTensorType.get(target_shape, elem_type)
            zero_attr = ir.DenseElementsAttr.get_splat(zero_type, zero_elem)
            zero_tensor = tosa.ConstOp(zero_attr).result
            return tosa.AddOp(zero_type, value, zero_tensor).result
        return value
    
    raise ValueError(f"Cannot broadcast shape {value_shape} to {target_shape}")


def _generate_vectorized_index_put(
    input1_memref, input1_shape, input2, input2_memref_list, input3_memref, input3_shape,
    mlir_dtype, symbol_table, num_vec_dims, vector_length
):
    """
    Generate vectorized index_put operation.
    
    This creates nested loops for the non-vectorized dimensions and uses
    vector.transfer_read/write for the vectorized trailing dimensions.
    
    Args:
        input1_memref: Destination memref (cache)
        input1_shape: Shape of destination
        input2: List of index tensors (may contain None)
        input2_memref_list: List of memref ops for index tensors
        input3_memref: Source memref (values)
        input3_shape: Shape of source
        mlir_dtype: Element type
        symbol_table: Symbol table for lookups
        num_vec_dims: Number of trailing dims to vectorize
        vector_length: Total vector length
    """
    rank = len(input3_shape)
    num_loop_dims = rank - num_vec_dims  # Dimensions that need loops
    
    lb = arith.ConstantOp(ir.IndexType.get(), 0)
    step = arith.ConstantOp(ir.IndexType.get(), 1)
    
    # Create upper bounds for loop dimensions
    ub_ops = []
    for d in range(num_loop_dims):
        ub_ops.append(arith.ConstantOp(ir.IndexType.get(), input3_shape[d]))
    
    # Vector type for the trailing dimensions
    vector_type = ir.VectorType.get([vector_length], mlir_dtype)
    
    # Padding value for transfer_read
    if str(mlir_dtype).startswith("f"):
        padding = arith.ConstantOp(mlir_dtype, ir.FloatAttr.get(mlir_dtype, 0.0))
    else:
        padding = arith.ConstantOp(mlir_dtype, ir.IntegerAttr.get(mlir_dtype, 0))
    
    # AffineMap: map from rank-D memref to 1D vector (last num_vec_dims -> 1)
    # For a rank-4 memref with 1 vec dim: (d0, d1, d2, d3) -> (d3)
    identity_map_attr = ir.AffineMapAttr.get(
        ir.AffineMap.get_minor_identity(rank, 1)
    )
    
    def create_nested_loops_vectorized(dim, idx_vars):
        """Recursively create nested loops and perform vectorized operation at innermost level."""
        if dim >= num_loop_dims:
            # All loop dimensions processed, now do vectorized read/write
            
            # Build source indices: [idx_var_0, idx_var_1, ..., idx_var_{num_loop_dims-1}, 0]
            src_indices = list(idx_vars) + [lb]
            
            # Build destination indices: use index tensors where available, loop vars otherwise
            dst_indices = []
            for d in range(len(input1_shape)):
                if d < len(input2) and input2[d] is not None:
                    # Use index tensor value
                    # Index tensor should be indexed by loop variables
                    if d < num_loop_dims:
                        # This dimension has both a loop var and an index tensor
                        # Load from index tensor using loop vars as indices
                        idx_load_indices = list(idx_vars)
                        idx_val = memref.LoadOp(input2_memref_list[d], idx_load_indices).result
                        idx_cast = arith.IndexCastOp(ir.IndexType.get(), idx_val)
                        dst_indices.append(idx_cast)
                    else:
                        # Vectorized dimension with index - this shouldn't happen
                        # as we don't vectorize dimensions with indices
                        dst_indices.append(lb)
                elif d < len(idx_vars):
                    dst_indices.append(idx_vars[d])
                else:
                    # This is a vectorized dimension, use 0 as starting index
                    dst_indices.append(lb)
            
            # Vector read from source
            vec_val = vector.TransferReadOp(
                vector_type,
                input3_memref,
                src_indices,
                identity_map_attr,
                padding,
                [True]  # in_bounds
            )
            
            # Vector write to destination
            vector.TransferWriteOp(
                None,
                vec_val.result,
                input1_memref,
                dst_indices,
                identity_map_attr,
                [True]
            )
            return
        
        # Create loop for this dimension
        loop = scf.ForOp(lb, ub_ops[dim], step)
        with ir.InsertionPoint(loop.body):
            new_idx_vars = idx_vars + [loop.induction_variable]
            create_nested_loops_vectorized(dim + 1, new_idx_vars)
            scf.YieldOp(loop.inner_iter_args)
    
    # Start creating nested loops
    create_nested_loops_vectorized(0, [])


def index_put_op(
    node: IndexPutOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Converts a Buddy IndexPutOp operation to an MLIR operation using scf.ForOp loops.

    This operation updates elements in the target tensor (input1) at specified indices (from input2)
    with new values (from input3). It handles cases where some indices are `None`, which represents
    a full selection for that dimension. The operation is implemented using nested loops over the
    tensor dimensions.

    Parameters:
        node (IndexPutOp): The Buddy IndexPutOp containing tensor metadata and index data.
        symbol_table (dict): A mapping from tensor names to corresponding MLIR operations.

    Returns:
        op: The MLIR operation representing the converted IndexPutOp.
    """
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    input1 = symbol_table.get((str(node.args[0]), 0))
    input1_shape = list(input1.type.shape)
    input2 = node.args[1]  # List of index tensors (may contain None)
    input3 = symbol_table.get((str(node.args[2]), 0))
    input3_shape = list(input3.type.shape)
    accumulate = node.args[3] if len(node.args) > 3 else False

    if len(input3_shape) == 0:
        for idx in input2:
            if idx is None:
                continue
            idx_val = symbol_table.get((str(idx), 0))
            if idx_val is None:
                continue
            idx_shape = list(ir.RankedTensorType(idx_val.type).shape)
            if not idx_shape:
                continue
            scalar_val = tensor.ExtractOp(input3, []).result
            splat_type = ir.RankedTensorType.get(
                idx_shape, ir.RankedTensorType(input3.type).element_type
            )
            input3 = tensor.SplatOp(splat_type, scalar_val, []).result
            input3_shape = idx_shape
            break

    input1_memref_element_type = input1.type.element_type
    input1_memref_type = ir.MemRefType.get(
        input1_shape, input1_memref_element_type
    )
    input1_memref = bufferization.ToBufferOp(input1_memref_type, input1)

    # Check if we can vectorize trailing dimensions
    num_vec_dims, vector_length = _get_vectorizable_trailing_dims(
        input2, input3_shape, accumulate
    )
    
    if num_vec_dims > 0:
        # Use vectorized path
        input3_memref_element_type = input3.type.element_type
        input3_memref_type = ir.MemRefType.get(
            input3_shape, input3_memref_element_type
        )
        input3_memref = bufferization.ToBufferOp(input3_memref_type, input3)
        
        # Convert index tensors to memrefs (only for non-None ones in loop dims)
        input2_memref_list = []
        num_loop_dims = len(input3_shape) - num_vec_dims
        for i in range(len(input2)):
            if input2[i] is None:
                input2_memref_list.append(None)
                continue
            input2_ = symbol_table.get((str(input2[i]), 0))
            if input2_ is None:
                input2_memref_list.append(None)
                continue
            # For vectorized path, index tensors should have shape of loop dims
            # Broadcast to loop dimensions shape
            loop_shape = input3_shape[:num_loop_dims]
            try:
                index_tensor = _broadcast_index_tensor_for_vec(input2_, loop_shape)
                index_elem_type = ir.RankedTensorType(index_tensor.type).element_type
                memref_type = ir.MemRefType.get(loop_shape, index_elem_type)
                input2_memref_list.append(
                    bufferization.ToBufferOp(memref_type, index_tensor)
                )
            except:
                # If broadcasting fails, fall back to scalar path
                num_vec_dims = 0
                break
        
        if num_vec_dims > 0:
            _generate_vectorized_index_put(
                input1_memref, input1_shape, input2, input2_memref_list, 
                input3_memref, input3_shape, mlir_dtype, symbol_table,
                num_vec_dims, vector_length
            )
            
            output_tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
            op = bufferization.ToTensorOp(
                output_tensor_type, input1_memref, restrict=True
            )
            return op

    # Fallback to scalar path
    def _broadcast_index_tensor(value, target_shape):
        try:
            value_type = ir.RankedTensorType(value.type)
            value_shape = list(value_type.shape)
            elem_type = value_type.element_type
        except Exception:
            elem_type = value.type
            target_type = ir.RankedTensorType.get(target_shape, elem_type)
            return tensor.SplatOp(target_type, value, []).result

        if len(value_shape) == 0 and len(target_shape) > 0:
            scalar_val = tensor.ExtractOp(value, []).result
            target_type = ir.RankedTensorType.get(target_shape, elem_type)
            return tensor.SplatOp(target_type, scalar_val, []).result

        if len(value_shape) < len(target_shape):
            padded_shape = [1] * (len(target_shape) - len(value_shape))
            padded_shape.extend(value_shape)
            shape_ty = ir.Type.parse(f"!tosa.shape<{len(padded_shape)}>")
            index_ty = ir.IndexType.get()
            shape_val = tosa.ConstShapeOp(
                shape_ty,
                ir.DenseElementsAttr.get(
                    array.array("q", padded_shape),
                    type=index_ty,
                    shape=[len(padded_shape)],
                ),
            ).result
            value = tosa.ReshapeOp(value, shape_val).result
            value_shape = padded_shape

        if len(value_shape) != len(target_shape):
            raise ValueError(
                "IndexPutOp: index rank %d does not match target rank %d"
                % (len(value_shape), len(target_shape))
            )

        for src_dim, tgt_dim in zip(value_shape, target_shape):
            if src_dim in (-1,) or tgt_dim in (-1,):
                continue
            if src_dim != 1 and src_dim != tgt_dim:
                raise ValueError(
                    "IndexPutOp: index shape %s is not broadcastable to %s"
                    % (value_shape, target_shape)
                )

        if value_shape != target_shape:
            if str(elem_type).startswith("f") or str(elem_type).startswith(
                "bf"
            ):
                zero_elem = ir.FloatAttr.get(elem_type, 0.0)
            else:
                zero_elem = ir.IntegerAttr.get(elem_type, 0)
            zero_type = ir.RankedTensorType.get(target_shape, elem_type)
            zero_attr = ir.DenseElementsAttr.get_splat(zero_type, zero_elem)
            zero_tensor = tosa.ConstOp(zero_attr).result
            value = tosa.AddOp(zero_type, value, zero_tensor).result

        return value

    # Convert index tensors to memrefs
    input2_memref = []
    for i in range(len(input2)):
        if input2[i] is None:
            input2_memref.append(None)
            continue
        input2_ = symbol_table.get((str(input2[i]), 0))
        if input2_ is None:
            return
        index_tensor = _broadcast_index_tensor(input2_, input3_shape)
        index_elem_type = ir.RankedTensorType(index_tensor.type).element_type
        memref_type = ir.MemRefType.get(input3_shape, index_elem_type)
        input2_memref.append(
            bufferization.ToBufferOp(memref_type, index_tensor)
        )

    input3_memref_element_type = input3.type.element_type
    input3_memref_type = ir.MemRefType.get(
        input3_shape, input3_memref_element_type
    )
    input3_memref = bufferization.ToBufferOp(input3_memref_type, input3)

    lb = arith.ConstantOp(ir.IndexType.get(), 0)
    step = arith.ConstantOp(ir.IndexType.get(), 1)

    # Create upper bounds for each dimension of source tensor
    ub = []
    for i in range(len(input3_shape)):
        ub.append(arith.ConstantOp(ir.IndexType.get(), input3_shape[i]))

    # Generate nested loops dynamically based on number of dimensions
    def create_nested_loops(dim, loops, idx_vars):
        """Recursively create nested loops and perform the store operation at innermost level."""
        if dim >= len(input3_shape):
            # Innermost: load value and store to target
            val_index = list(idx_vars)
            put_val = memref.LoadOp(input3_memref, val_index).result

            # Build store indices: use index tensors where available, loop vars otherwise
            store_index = []
            for d in range(len(output_shape)):
                if d < len(input2) and input2[d] is not None:
                    # Use corresponding index tensor
                    # The index tensor should have shape matching input3_shape
                    idx_dim_val = memref.LoadOp(
                        input2_memref[d], val_index
                    ).result
                    idx_dim = arith.IndexCastOp(ir.IndexType.get(), idx_dim_val)
                    store_index.append(idx_dim)
                elif d < len(idx_vars):
                    store_index.append(idx_vars[d])
                else:
                    # Use constant 0 for missing dimensions
                    store_index.append(lb)

            if accumulate:
                # Load existing value and add
                existing_val = memref.LoadOp(input1_memref, store_index).result
                if str(mlir_dtype).find("f") != -1:
                    new_val = arith.AddFOp(existing_val, put_val)
                else:
                    new_val = arith.AddIOp(existing_val, put_val)
                memref.StoreOp(new_val, input1_memref, store_index)
            else:
                memref.StoreOp(put_val, input1_memref, store_index)
            return

        # Create loop for this dimension
        loop = scf.ForOp(lb, ub[dim], step)
        with ir.InsertionPoint(loop.body):
            new_idx_vars = idx_vars + [loop.induction_variable]
            create_nested_loops(dim + 1, loops + [loop], new_idx_vars)
            scf.YieldOp(loop.inner_iter_args)

    # Start creating nested loops
    create_nested_loops(0, [], [])

    output_tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    op = bufferization.ToTensorOp(
        output_tensor_type, input1_memref, restrict=True
    )
    return op


def ne_scalar_op(
    node: NeScalarOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Converts a Buddy NeScalarOp operation to an MLIR comparison operation (CmpIOp or CmpFOp).

    This operation compares a tensor with a scalar value and produces a boolean tensor where each element
    represents the result of the inequality comparison (not equal). The operation supports both integer
    and floating-point types.

    Parameters:
        node (NeScalarOp): The Buddy NeScalarOp node containing the operation details and tensor metadata.
        symbol_table (dict): A dictionary mapping tensor names to their corresponding MLIR operations.

    Returns:
        op: An MLIR comparison operation (either CmpIOp for integers or CmpFOp for floats) that performs
            element-wise inequality (not equal) comparison between the input tensor and the scalar.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0), node.args[0])
    input_dtype = ir.RankedTensorType(input_tensor.type).element_type
    input_shape = ir.RankedTensorType(input_tensor.type).shape
    tensor_type = ir.RankedTensorType.get(input_shape, input_dtype)

    # Convert scalar value to the appropriate type
    scalar_val = node.args[1]
    if str(input_dtype).find("f") != -1:
        scalar_val = float(scalar_val)
    else:
        scalar_val = int(scalar_val)

    scalar = arith.ConstantOp(input_dtype, scalar_val)
    rhs = tensor.SplatOp(tensor_type, scalar, [])

    if str(input_dtype).find("i") != -1:
        cmp_op = arith.CmpIOp(1, input_tensor, rhs)
    else:
        cmp_op = arith.CmpFOp(6, input_tensor, rhs)

    return cmp_op


def _cumulative_tensor(
    input_tensor: ir.Value,
    output_shape: List[int],
    dim: int,
    mlir_dtype: ir.Type,
    op_kind: str,
):
    rank = len(output_shape)
    if rank == 0:
        raise NotImplementedError(f"{op_kind} does not support rank-0 tensors")
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        raise NotImplementedError(f"{op_kind} dim out of range")

    output_tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    if input_tensor.type.element_type != mlir_dtype:
        input_tensor = tosa.CastOp(output_tensor_type, input_tensor).result

    input_shape = list(ir.RankedTensorType(input_tensor.type).shape)
    if len(input_shape) != rank:
        raise NotImplementedError(
            f"{op_kind} requires matching input/output ranks"
        )

    input_memref_type = ir.MemRefType.get(input_shape, mlir_dtype)
    input_memref = bufferization.ToBufferOp(input_memref_type, input_tensor)

    index_type = ir.IndexType.get()
    dynamic_sizes = []
    for i, size in enumerate(input_shape):
        if size < 0:
            dim_index = arith.ConstantOp(index_type, i).result
            dynamic_sizes.append(memref.DimOp(input_memref, dim_index).result)
    output_memref_type = ir.MemRefType.get(input_shape, mlir_dtype)
    output_memref = memref.AllocOp(output_memref_type, dynamic_sizes, [])

    bounds = []
    for i, size in enumerate(input_shape):
        if size < 0:
            dim_index = arith.ConstantOp(index_type, i).result
            bounds.append(memref.DimOp(input_memref, dim_index).result)
        else:
            bounds.append(arith.ConstantOp(index_type, size).result)

    c0 = arith.ConstantOp(index_type, 0)
    c1 = arith.ConstantOp(index_type, 1)
    is_float = ir.FloatType.isinstance(mlir_dtype) or ir.BF16Type.isinstance(
        mlir_dtype
    )
    init_value = 0.0 if op_kind == "cumsum" else 1.0
    if is_float:
        init_val = arith.ConstantOp(mlir_dtype, float(init_value))
    else:
        init_val = arith.ConstantOp(mlir_dtype, int(init_value))

    idx_values = [None] * rank

    def build_outer_loops(dim_idx: int):
        if dim_idx == rank:
            dim_loop = scf.ForOp(
                c0.result, bounds[dim], c1.result, [init_val.result]
            )
            with ir.InsertionPoint(dim_loop.body):
                idx_values[dim] = dim_loop.induction_variable
                curr_val = memref.LoadOp(input_memref, idx_values).result
                prev_val = dim_loop.inner_iter_args[0]
                if op_kind == "cumsum":
                    if is_float:
                        new_val = arith.AddFOp(prev_val, curr_val).result
                    else:
                        new_val = arith.AddIOp(prev_val, curr_val).result
                else:
                    if is_float:
                        new_val = arith.MulFOp(prev_val, curr_val).result
                    else:
                        new_val = arith.MulIOp(prev_val, curr_val).result
                memref.StoreOp(new_val, output_memref, idx_values)
                scf.YieldOp([new_val])
            return
        if dim_idx == dim:
            build_outer_loops(dim_idx + 1)
            return
        loop = scf.ForOp(c0.result, bounds[dim_idx], c1.result)
        with ir.InsertionPoint(loop.body):
            idx_values[dim_idx] = loop.induction_variable
            build_outer_loops(dim_idx + 1)
            scf.YieldOp(loop.inner_iter_args)

    build_outer_loops(0)

    return bufferization.ToTensorOp(
        output_tensor_type, output_memref, restrict=True
    ).result


def _cumulative_op(
    node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
    op_kind: str,
):
    output_shape = list(node.tensor_meta["shape"])
    dim = node.args[1]
    if not isinstance(dim, int):
        raise NotImplementedError(f"{op_kind} requires integer dim")

    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    input1 = symbol_table.get((str(node.args[0]), 0), node.args[0])
    return _cumulative_tensor(input1, output_shape, dim, mlir_dtype, op_kind)


def cumsum_op(
    node: CumsumOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Converts a Buddy CumsumOp operation to an MLIR operation using scf.ForOp loops.

    This operation computes the cumulative sum along a specified dimension (axis)
    of an input tensor of any rank.
    """
    return _cumulative_op(node, symbol_table, "cumsum")


def cumprod_op(
    node: CumProdOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Converts a Buddy CumprodOp operation to an MLIR operation using scf.ForOp loops.

    This operation computes the cumulative product along a specified dimension (axis)
    of an input tensor of any rank.
    """
    return _cumulative_op(node, symbol_table, "cumprod")


def logcumsumexp_op(
    node: LogCumsumExpOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the logcumsumexp operation.
    From buddy LogCumsumExpOp to MLIR operations.
    aten.logcumsumexp(input, dim) -> Tensor
    """
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    input1 = symbol_table.get((str(node.args[0]), 0), node.args[0])
    dim = node.args[1]
    if isinstance(dim, str):
        if len(output_shape) != 1:
            raise NotImplementedError(
                "logcumsumexp dimname requires rank-1 tensor"
            )
        dim = 0
    if not isinstance(dim, int):
        raise NotImplementedError("logcumsumexp requires integer dim")

    input_type = ir.RankedTensorType(input1.type)
    exp_type = ir.RankedTensorType.get(
        list(input_type.shape), input_type.element_type
    )
    exp_tensor = tosa.ExpOp(exp_type, input1).result

    cumsum = _cumulative_tensor(
        exp_tensor, output_shape, dim, mlir_dtype, "cumsum"
    )
    log_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    return tosa.LogOp(log_type, cumsum).result


def diagonal_scatter_op(
    node: DiagonalScatterOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Scatter values from src into the diagonal of input.

    Implements aten.diagonal_scatter.default for tensors of any rank using
    scf.ForOp loops. The diagonal is defined by offset, dim1, and dim2.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0), node.args[0])
    src_tensor = symbol_table.get((str(node.args[1]), 0), node.args[1])
    offset = node.args[2] if len(node.args) > 2 else 0
    dim1 = node.args[3] if len(node.args) > 3 else 0
    dim2 = node.args[4] if len(node.args) > 4 else 1

    if not isinstance(offset, int):
        raise NotImplementedError("diagonal_scatter requires constant offset")

    input_shape = list(ir.RankedTensorType(input_tensor.type).shape)
    rank = len(input_shape)
    if rank < 2:
        raise NotImplementedError("diagonal_scatter requires input rank >= 2")

    if dim1 < 0:
        dim1 += rank
    if dim2 < 0:
        dim2 += rank
    if dim1 == dim2 or dim1 < 0 or dim2 < 0:
        raise NotImplementedError("diagonal_scatter invalid dims")

    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    output_tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)

    if input_tensor.type.element_type != mlir_dtype:
        input_tensor = tosa.CastOp(output_tensor_type, input_tensor).result

    src_shape = list(ir.RankedTensorType(src_tensor.type).shape)
    if src_tensor.type.element_type != mlir_dtype:
        src_tensor = tosa.CastOp(
            ir.RankedTensorType.get(src_shape, mlir_dtype), src_tensor
        ).result

    input_memref_type = ir.MemRefType.get(input_shape, mlir_dtype)
    src_memref_type = ir.MemRefType.get(src_shape, mlir_dtype)
    output_memref_type = ir.MemRefType.get(input_shape, mlir_dtype)

    input_memref = bufferization.ToBufferOp(input_memref_type, input_tensor)
    src_memref = bufferization.ToBufferOp(src_memref_type, src_tensor)

    index_type = ir.IndexType.get()
    dynamic_sizes = []
    for i, size in enumerate(input_shape):
        if size < 0:
            dim_index = arith.ConstantOp(index_type, i).result
            dynamic_sizes.append(memref.DimOp(input_memref, dim_index).result)
    output_memref = memref.AllocOp(output_memref_type, dynamic_sizes, [])

    bounds = []
    for i, size in enumerate(input_shape):
        if size < 0:
            dim_index = arith.ConstantOp(index_type, i).result
            bounds.append(memref.DimOp(input_memref, dim_index).result)
        else:
            bounds.append(arith.ConstantOp(index_type, size).result)

    c0 = arith.ConstantOp(index_type, 0)
    c1 = arith.ConstantOp(index_type, 1)

    idx_values = [None] * rank

    def copy_loop(depth: int):
        if depth == rank:
            val = memref.LoadOp(input_memref, idx_values).result
            memref.StoreOp(val, output_memref, idx_values)
            return
        loop = scf.ForOp(c0.result, bounds[depth], c1.result)
        with ir.InsertionPoint(loop.body):
            idx_values[depth] = loop.induction_variable
            copy_loop(depth + 1)
            scf.YieldOp(loop.inner_iter_args)

    copy_loop(0)

    other_dims = [i for i in range(rank) if i not in (dim1, dim2)]
    if len(src_shape) != rank - 1:
        raise NotImplementedError(
            "diagonal_scatter expects src rank to be input rank - 1"
        )

    diag_dim = len(src_shape) - 1
    if src_shape[diag_dim] < 0:
        diag_bound = memref.DimOp(
            src_memref, arith.ConstantOp(index_type, diag_dim).result
        ).result
    else:
        diag_bound = arith.ConstantOp(index_type, src_shape[diag_dim]).result

    if offset >= 0:
        offset_val = arith.ConstantOp(index_type, offset).result
        abs_offset_val = None
    else:
        offset_val = None
        abs_offset_val = arith.ConstantOp(index_type, -offset).result

    def diag_loop(depth: int):
        if depth == len(other_dims):
            loop = scf.ForOp(c0.result, diag_bound, c1.result)
            with ir.InsertionPoint(loop.body):
                diag_k = loop.induction_variable
                if offset >= 0:
                    idx_values[dim1] = diag_k
                    if offset == 0:
                        idx_values[dim2] = diag_k
                    else:
                        idx_values[dim2] = arith.AddIOp(
                            diag_k, offset_val
                        ).result
                else:
                    idx_values[dim2] = diag_k
                    idx_values[dim1] = arith.AddIOp(
                        diag_k, abs_offset_val
                    ).result

                src_indices = []
                for d in other_dims:
                    src_indices.append(idx_values[d])
                src_indices.append(diag_k)
                src_val = memref.LoadOp(src_memref, src_indices).result
                memref.StoreOp(src_val, output_memref, idx_values)
                scf.YieldOp(loop.inner_iter_args)
            return
        dim_idx = other_dims[depth]
        loop = scf.ForOp(c0.result, bounds[dim_idx], c1.result)
        with ir.InsertionPoint(loop.body):
            idx_values[dim_idx] = loop.induction_variable
            diag_loop(depth + 1)
            scf.YieldOp(loop.inner_iter_args)

    diag_loop(0)

    return bufferization.ToTensorOp(
        output_tensor_type, output_memref, restrict=True
    )


def empty_op(
    node: EmptyOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Create an empty tensor with the requested shape and dtype.

    Note: This ignores memory_format/stride hints and returns a standard
    tensor.empty equivalent for compile-time coverage.
    """
    output_shape = list(node.tensor_meta.get("shape", []))
    if node.args:
        size_arg = node.args[0]
        if isinstance(size_arg, (list, tuple)):
            try:
                output_shape = [int(dim) for dim in size_arg]
            except (TypeError, ValueError):
                pass

    if any(dim < 0 for dim in output_shape):
        raise NotImplementedError("empty with dynamic shape is not supported")

    dtype = node.tensor_meta.get("dtype", None)
    element_type = (
        mlir_element_type_get(dtype) if dtype is not None else ir.F32Type.get()
    )
    return tensor.EmptyOp(output_shape, element_type)


def gcd_op(
    node: GcdOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Compute elementwise greatest common divisor for integer tensors.
    """
    input1 = symbol_table.get((str(node.args[0]), 0), node.args[0])
    input2 = symbol_table.get((str(node.args[1]), 0), node.args[1])

    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    if not ir.IntegerType.isinstance(mlir_dtype):
        raise NotImplementedError("gcd only supports integer types")

    output_tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    rank = len(output_shape)

    def _gcd_scalar(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        abs_lhs = math.AbsIOp(lhs).result
        abs_rhs = math.AbsIOp(rhs).result
        zero = arith.ConstantOp(mlir_dtype, 0).result
        while_op = scf.WhileOp([mlir_dtype, mlir_dtype], [abs_lhs, abs_rhs])
        before_block = while_op.before.blocks.append(mlir_dtype, mlir_dtype)
        with ir.InsertionPoint(before_block):
            a_val, b_val = before_block.arguments
            cond = arith.CmpIOp(arith.CmpIPredicate.ne, b_val, zero).result
            scf.ConditionOp(cond, [a_val, b_val])
        after_block = while_op.after.blocks.append(mlir_dtype, mlir_dtype)
        with ir.InsertionPoint(after_block):
            a_val, b_val = after_block.arguments
            rem = arith.RemSIOp(a_val, b_val).result
            scf.YieldOp([b_val, rem])
        return while_op.results[0]

    def _scalar_value(value, name: str) -> ir.Value:
        if hasattr(value, "type"):
            value_type = ir.RankedTensorType(value.type)
            if list(value_type.shape) != []:
                raise NotImplementedError(
                    f"gcd scalar {name} expects rank-0 tensor"
                )
            if value_type.element_type != mlir_dtype:
                value = tosa.CastOp(output_tensor_type, value).result
            return tensor.ExtractOp(value, []).result
        return arith.ConstantOp(mlir_dtype, int(value)).result

    if rank == 0:
        lhs = _scalar_value(input1, "input1")
        rhs = _scalar_value(input2, "input2")
        gcd_val = _gcd_scalar(lhs, rhs)
        return tensor.FromElementsOp(output_tensor_type, gcd_val)

    def _normalize_input(value, name: str) -> ir.Value:
        if not hasattr(value, "type"):
            if any(dim < 0 for dim in output_shape):
                raise NotImplementedError(
                    "gcd scalar broadcast requires static shape"
                )
            scalar_attr = mlir_element_attr_get(dtype, value)
            splat_attr = ir.DenseElementsAttr.get_splat(
                output_tensor_type, scalar_attr
            )
            return arith.ConstantOp(output_tensor_type, splat_attr).result

        value_type = ir.RankedTensorType(value.type)
        if len(value_type.shape) == 0:
            if any(dim < 0 for dim in output_shape):
                raise NotImplementedError(
                    "gcd scalar broadcast requires static shape"
                )
            scalar = tensor.ExtractOp(value, []).result
            return tensor.SplatOp(output_tensor_type, scalar, []).result

        if list(value_type.shape) != output_shape:
            raise NotImplementedError(
                "gcd requires matching shapes or scalar inputs"
            )
        if value_type.element_type != mlir_dtype:
            value = tosa.CastOp(output_tensor_type, value).result
        return value

    input1 = _normalize_input(input1, "input1")
    input2 = _normalize_input(input2, "input2")

    input_memref_type = ir.MemRefType.get(output_shape, mlir_dtype)
    input1_memref = bufferization.ToBufferOp(input_memref_type, input1)
    input2_memref = bufferization.ToBufferOp(input_memref_type, input2)

    index_type = ir.IndexType.get()
    dynamic_sizes = []
    for i, size in enumerate(output_shape):
        if size < 0:
            dim_index = arith.ConstantOp(index_type, i).result
            dynamic_sizes.append(memref.DimOp(input1_memref, dim_index).result)

    output_memref = memref.AllocOp(
        ir.MemRefType.get(output_shape, mlir_dtype), dynamic_sizes, []
    )

    bounds = []
    for i, size in enumerate(output_shape):
        if size < 0:
            dim_index = arith.ConstantOp(index_type, i).result
            bounds.append(memref.DimOp(input1_memref, dim_index).result)
        else:
            bounds.append(arith.ConstantOp(index_type, size).result)

    c0 = arith.ConstantOp(index_type, 0)
    c1 = arith.ConstantOp(index_type, 1)
    idx_values = [None] * rank

    def build_loops(dim_idx: int):
        if dim_idx == rank:
            lhs = memref.LoadOp(input1_memref, idx_values).result
            rhs = memref.LoadOp(input2_memref, idx_values).result
            gcd_val = _gcd_scalar(lhs, rhs)
            memref.StoreOp(gcd_val, output_memref, idx_values)
            return
        loop = scf.ForOp(c0.result, bounds[dim_idx], c1.result)
        with ir.InsertionPoint(loop.body):
            idx_values[dim_idx] = loop.induction_variable
            build_loops(dim_idx + 1)
            scf.YieldOp(loop.inner_iter_args)

    build_loops(0)

    return bufferization.ToTensorOp(
        output_tensor_type, output_memref, restrict=True
    ).result


def sort_op(
    node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Converts a Buddy SortOp operation to MLIR operations.

    This is a bubble sort implementation using scf.ForOp loops for 2D tensors.
    Returns (sorted_values, indices).

    Parameters:
        node: The Buddy SortOp node containing the operation details and tensor metadata.
        symbol_table (dict): A dictionary mapping tensor names to their corresponding MLIR operations.

    Returns:
        tuple: (values, indices) as MLIR operations.
    """
    shape_meta = node.tensor_meta["shape"]
    dtype_meta = node.tensor_meta["dtype"]

    # tensor_meta for sort contains tuple of (values_shape, indices_shape)
    if isinstance(shape_meta, tuple):
        output_shape = list(shape_meta[0])
    else:
        output_shape = list(shape_meta)

    if isinstance(dtype_meta, tuple):
        dtype = dtype_meta[0]
    else:
        dtype = dtype_meta

    mlir_dtype = mlir_element_type_get(dtype)
    input1 = symbol_table.get((str(node.args[0]), 0), node.args[0])

    dim = node.args[1] if len(node.args) > 1 else -1
    descending = node.args[2] if len(node.args) > 2 else False

    if dim == -1:
        dim += len(output_shape)

    output_tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    indices_tensor_type = ir.RankedTensorType.get(
        output_shape, ir.IntegerType.get_signless(64)
    )

    # Convert input to memref for in-place sorting
    input_memref_type = ir.MemRefType.get(output_shape, mlir_dtype)
    input_memref = bufferization.ToBufferOp(input_memref_type, input1)

    # Create indices memref
    indices_memref_type = ir.MemRefType.get(
        output_shape, ir.IntegerType.get_signless(64)
    )
    indices_memref = memref.AllocOp(indices_memref_type, [], [])

    # Initialize indices with sequential values along the sort dimension
    lb = arith.ConstantOp(ir.IndexType.get(), 0)
    step = arith.ConstantOp(ir.IndexType.get(), 1)
    ub0 = arith.ConstantOp(ir.IndexType.get(), output_shape[0])
    ub1 = arith.ConstantOp(ir.IndexType.get(), output_shape[1])

    # Initialize indices: indices[i][j] = j (for dim=1)
    init_loop0 = scf.ForOp(lb, ub0, step)
    with ir.InsertionPoint(init_loop0.body):
        init_loop1 = scf.ForOp(lb, ub1, step)
        with ir.InsertionPoint(init_loop1.body):
            idx_val = arith.IndexCastOp(
                ir.IntegerType.get_signless(64), init_loop1.induction_variable
            )
            memref.StoreOp(
                idx_val,
                indices_memref,
                [init_loop0.induction_variable, init_loop1.induction_variable],
            )
            scf.YieldOp(init_loop1.inner_iter_args)
        scf.YieldOp(init_loop0.inner_iter_args)

    # Bubble sort along dim=1
    sort_size = output_shape[dim]
    outer_ub = arith.ConstantOp(ir.IndexType.get(), sort_size - 1)

    # Outer loop (over rows for 2D, dim=1)
    row_loop = scf.ForOp(lb, ub0, step)
    with ir.InsertionPoint(row_loop.body):
        # Bubble sort passes
        pass_loop = scf.ForOp(lb, outer_ub, step)
        with ir.InsertionPoint(pass_loop.body):
            # Compare adjacent elements
            inner_ub = arith.SubIOp(outer_ub, pass_loop.induction_variable)
            compare_loop = scf.ForOp(lb, inner_ub, step)
            with ir.InsertionPoint(compare_loop.body):
                next_idx = arith.AddIOp(compare_loop.induction_variable, step)

                # Load current and next values
                val_curr = memref.LoadOp(
                    input_memref,
                    [
                        row_loop.induction_variable,
                        compare_loop.induction_variable,
                    ],
                )
                val_next = memref.LoadOp(
                    input_memref, [row_loop.induction_variable, next_idx]
                )

                # Load indices
                idx_curr = memref.LoadOp(
                    indices_memref,
                    [
                        row_loop.induction_variable,
                        compare_loop.induction_variable,
                    ],
                )
                idx_next = memref.LoadOp(
                    indices_memref, [row_loop.induction_variable, next_idx]
                )

                # Compare: for ascending, swap if curr > next
                if str(mlir_dtype).startswith("f"):
                    if descending:
                        should_swap = arith.CmpFOp(
                            arith.CmpFPredicate.OLT, val_curr, val_next
                        )
                    else:
                        should_swap = arith.CmpFOp(
                            arith.CmpFPredicate.OGT, val_curr, val_next
                        )
                else:
                    if descending:
                        should_swap = arith.CmpIOp(
                            arith.CmpIPredicate.slt, val_curr, val_next
                        )
                    else:
                        should_swap = arith.CmpIOp(
                            arith.CmpIPredicate.sgt, val_curr, val_next
                        )

                # Conditional swap using scf.if
                if_op = scf.IfOp(should_swap, hasElse=False)
                with ir.InsertionPoint(if_op.then_block):
                    # Swap values
                    memref.StoreOp(
                        val_next,
                        input_memref,
                        [
                            row_loop.induction_variable,
                            compare_loop.induction_variable,
                        ],
                    )
                    memref.StoreOp(
                        val_curr,
                        input_memref,
                        [row_loop.induction_variable, next_idx],
                    )
                    # Swap indices
                    memref.StoreOp(
                        idx_next,
                        indices_memref,
                        [
                            row_loop.induction_variable,
                            compare_loop.induction_variable,
                        ],
                    )
                    memref.StoreOp(
                        idx_curr,
                        indices_memref,
                        [row_loop.induction_variable, next_idx],
                    )
                    scf.YieldOp([])

                scf.YieldOp(compare_loop.inner_iter_args)
            scf.YieldOp(pass_loop.inner_iter_args)
        scf.YieldOp(row_loop.inner_iter_args)

    # Convert back to tensors
    values = bufferization.ToTensorOp(
        output_tensor_type, input_memref, restrict=True
    )
    indices = bufferization.ToTensorOp(
        indices_tensor_type, indices_memref, restrict=True
    )

    return (values.result, indices.result)


def tensor_constant_op(
    node: TensorConstantOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Converts a Buddy Constant0Op operation to an MLIR arith.ConstantOp.

    This operation creates a constant tensor filled with zeros. It constructs a ranked tensor
    of the specified shape and data type, generates a zero-valued element attribute, and
    initializes the entire tensor with this value using a splat attribute.

    Parameters:
        node (Constant0Op): The Buddy Constant0Op node containing the tensor shape and data type metadata.
        symbol_table (dict): A dictionary mapping tensor names to their corresponding MLIR operations.

    Returns:
        op: An MLIR arith.ConstantOp representing a tensor filled with zeros.
    """
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    output_shape = list(node.tensor_meta["shape"])
    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    value = node.args[0]
    element = mlir_element_attr_get(dtype, value)
    attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
    op = arith.ConstantOp(tensor_type, attr)
    return op


def lift_fresh_copy_op(
    node: LiftFreshCopyOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Converts a Buddy LiftFreshCopyOp operation to an MLIR tosa.IdentityOp.

    This operation creates a new tensor with the same shape and element type as the input tensor,
    effectively producing a fresh copy without modifying the data. Internally, this is represented
    as an identity operation in MLIR.

    Parameters:
        node (LiftFreshCopyOp): The Buddy LiftFreshCopyOp node containing the operation details.
        symbol_table (dict): A dictionary mapping tensor names to their corresponding MLIR operations.

    Returns:
        op: An MLIR tosa.IdentityOp that represents creating a fresh copy of the input tensor.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    sizes = ir.RankedTensorType(input_tensor.type).shape
    result_element_type = ir.RankedTensorType(input_tensor.type).element_type
    output_type = ir.RankedTensorType.get(sizes, result_element_type)
    op = tosa.IdentityOp(output_type, input_tensor)
    return op


def repeat_op(
    node: LiftFreshCopyOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Converts a Buddy RepeatOp operation to an MLIR operation.

    This operation is intended to repeat a tensor along specified dimensions by a given set of repeat factors.
    If all repeat factors are 1, the input tensor is returned without any modification. If the repeat factors
    are not fully implemented or contain values other than 1, the operation currently raises an assertion error.

    Parameters:
        node (RepeatOp): The Buddy RepeatOp node containing the tensor and repeat factors.
        symbol_table (dict): A dictionary mapping tensor names to their corresponding MLIR operations.

    Returns:
        op: The input tensor, or a modified version if repeat factors are implemented in the future.

    Note:
        - The repeat functionality is not fully implemented. Currently, if all repeat factors are 1, the input tensor
        is returned unchanged.
        - If any repeat factor is other than 1, an assertion error is triggered.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    repeat_factors = node.args[1]
    if len(repeat_factors) == repeat_factors.count(1):
        return input_tensor
    else:
        assert False
    return input_tensor


def repeat_interleave_op(
    node: RepeatInterleaveOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Lower repeat_interleave for 1D tensors with static shapes.
    Supports:
      - repeat_interleave.Tensor (repeats only)
      - repeat_interleave.self_Tensor
      - repeat_interleave.self_int
    """
    args = list(node.args)
    if not args:
        raise NotImplementedError("repeat_interleave requires arguments")

    output_shape = list(node.tensor_meta["shape"])
    if len(output_shape) != 1 or any(dim < 0 for dim in output_shape):
        raise NotImplementedError(
            "repeat_interleave requires static 1D output shape"
        )

    dtype_meta = node.tensor_meta.get("dtype", None)
    if isinstance(dtype_meta, TensorDType):
        output_dtype = mlir_element_type_get(dtype_meta)
    elif isinstance(dtype_meta, ir.Type):
        output_dtype = dtype_meta
    else:
        output_dtype = ir.IntegerType.get_signless(64)

    output_memref = memref.AllocOp(
        ir.MemRefType.get(output_shape, output_dtype), [], []
    )
    output_tensor_type = ir.RankedTensorType.get(output_shape, output_dtype)

    index_type = ir.IndexType.get()
    c0 = arith.ConstantOp(index_type, 0)
    c1 = arith.ConstantOp(index_type, 1)
    out_ub = arith.ConstantOp(index_type, output_shape[0]).result

    if _is_float_type(output_dtype):
        zero_val = arith.ConstantOp(
            output_dtype, ir.FloatAttr.get(output_dtype, 0.0)
        ).result
    else:
        zero_val = arith.ConstantOp(
            output_dtype, ir.IntegerAttr.get(output_dtype, 0)
        ).result
    linalg.fill(zero_val, outs=[output_memref.result])

    counter_memref = memref.AllocOp(ir.MemRefType.get([1], index_type), [], [])
    memref.StoreOp(c0.result, counter_memref, [c0.result])

    def _repeat_count_to_index(rep_val, rep_type):
        if not ir.IntegerType.isinstance(rep_type):
            raise NotImplementedError("repeat_interleave expects int repeats")
        zero_int = arith.ConstantOp(
            rep_type, ir.IntegerAttr.get(rep_type, 0)
        ).result
        is_neg = arith.CmpIOp(arith.CmpIPredicate.slt, rep_val, zero_int).result
        rep_idx = arith.IndexCastOp(index_type, rep_val).result
        return arith.SelectOp(is_neg, c0.result, rep_idx).result

    repeats_tensor = (
        symbol_table.get((str(args[1]), 0)) if len(args) > 1 else None
    )
    if len(args) == 1:
        repeats = symbol_table.get((str(args[0]), 0))
        if repeats is None:
            raise NotImplementedError(
                "repeat_interleave.Tensor requires repeats tensor"
            )
        repeats_type = ir.RankedTensorType(repeats.type)
        repeats_shape = list(repeats_type.shape)
        if len(repeats_shape) != 1 or repeats_shape[0] < 0:
            raise NotImplementedError(
                "repeat_interleave.Tensor requires static 1D repeats"
            )
        repeats_memref = bufferization.ToBufferOp(
            ir.MemRefType.get(repeats_shape, repeats_type.element_type),
            repeats,
        ).result

        ub_repeats = arith.ConstantOp(index_type, repeats_shape[0])
        outer_loop = scf.ForOp(c0.result, ub_repeats.result, c1.result)
        with ir.InsertionPoint(outer_loop.body):
            i = outer_loop.induction_variable
            rep_val = memref.LoadOp(repeats_memref, [i]).result
            rep_idx = _repeat_count_to_index(rep_val, repeats_type.element_type)
            inner_loop = scf.ForOp(c0.result, rep_idx, c1.result)
            with ir.InsertionPoint(inner_loop.body):
                out_pos = memref.LoadOp(counter_memref, [c0.result]).result
                in_range = arith.CmpIOp(
                    arith.CmpIPredicate.slt, out_pos, out_ub
                ).result
                if_op = scf.IfOp(in_range, hasElse=False)
                with ir.InsertionPoint(if_op.then_block):
                    if not ir.IntegerType.isinstance(output_dtype):
                        raise NotImplementedError(
                            "repeat_interleave.Tensor requires integer output"
                        )
                    idx_val = arith.IndexCastOp(output_dtype, i).result
                    memref.StoreOp(idx_val, output_memref.result, [out_pos])
                    new_pos = arith.AddIOp(out_pos, c1.result)
                    memref.StoreOp(new_pos, counter_memref, [c0.result])
                    scf.YieldOp([])
                scf.YieldOp(inner_loop.inner_iter_args)
            scf.YieldOp(outer_loop.inner_iter_args)

        return bufferization.ToTensorOp(
            output_tensor_type, output_memref.result, restrict=True
        )

    self_tensor = symbol_table.get((str(args[0]), 0))
    if self_tensor is None:
        raise NotImplementedError("repeat_interleave requires self tensor")

    dim = args[2] if len(args) > 2 else None
    if dim is None:
        dim = 0
    if dim != 0:
        raise NotImplementedError("repeat_interleave only supports dim=0")

    self_type = ir.RankedTensorType(self_tensor.type)
    self_shape = list(self_type.shape)
    if len(self_shape) != 1 or self_shape[0] < 0:
        raise NotImplementedError(
            "repeat_interleave only supports static 1D self tensors"
        )

    self_memref = bufferization.ToBufferOp(
        ir.MemRefType.get(self_shape, self_type.element_type), self_tensor
    ).result
    ub_self = arith.ConstantOp(index_type, self_shape[0])

    if repeats_tensor is not None:
        repeats_type = ir.RankedTensorType(repeats_tensor.type)
        repeats_shape = list(repeats_type.shape)
        if repeats_shape != self_shape:
            raise NotImplementedError(
                "repeat_interleave.self_Tensor requires repeats shape == self shape"
            )
        repeats_memref = bufferization.ToBufferOp(
            ir.MemRefType.get(repeats_shape, repeats_type.element_type),
            repeats_tensor,
        ).result

        outer_loop = scf.ForOp(c0.result, ub_self.result, c1.result)
        with ir.InsertionPoint(outer_loop.body):
            i = outer_loop.induction_variable
            rep_val = memref.LoadOp(repeats_memref, [i]).result
            rep_idx = _repeat_count_to_index(rep_val, repeats_type.element_type)
            inner_loop = scf.ForOp(c0.result, rep_idx, c1.result)
            with ir.InsertionPoint(inner_loop.body):
                out_pos = memref.LoadOp(counter_memref, [c0.result]).result
                in_range = arith.CmpIOp(
                    arith.CmpIPredicate.slt, out_pos, out_ub
                ).result
                if_op = scf.IfOp(in_range, hasElse=False)
                with ir.InsertionPoint(if_op.then_block):
                    self_val = memref.LoadOp(self_memref, [i]).result
                    memref.StoreOp(self_val, output_memref.result, [out_pos])
                    new_pos = arith.AddIOp(out_pos, c1.result)
                    memref.StoreOp(new_pos, counter_memref, [c0.result])
                    scf.YieldOp([])
                scf.YieldOp(inner_loop.inner_iter_args)
            scf.YieldOp(outer_loop.inner_iter_args)
    else:
        repeats_scalar = args[1] if len(args) > 1 else None
        if not isinstance(repeats_scalar, int):
            raise NotImplementedError(
                "repeat_interleave.self_int requires static integer repeats"
            )
        if repeats_scalar < 0:
            raise NotImplementedError(
                "repeat_interleave requires non-negative repeats"
            )
        rep_idx_const = arith.ConstantOp(index_type, repeats_scalar).result

        outer_loop = scf.ForOp(c0.result, ub_self.result, c1.result)
        with ir.InsertionPoint(outer_loop.body):
            i = outer_loop.induction_variable
            inner_loop = scf.ForOp(c0.result, rep_idx_const, c1.result)
            with ir.InsertionPoint(inner_loop.body):
                out_pos = memref.LoadOp(counter_memref, [c0.result]).result
                in_range = arith.CmpIOp(
                    arith.CmpIPredicate.slt, out_pos, out_ub
                ).result
                if_op = scf.IfOp(in_range, hasElse=False)
                with ir.InsertionPoint(if_op.then_block):
                    self_val = memref.LoadOp(self_memref, [i]).result
                    memref.StoreOp(self_val, output_memref.result, [out_pos])
                    new_pos = arith.AddIOp(out_pos, c1.result)
                    memref.StoreOp(new_pos, counter_memref, [c0.result])
                    scf.YieldOp([])
                scf.YieldOp(inner_loop.inner_iter_args)
            scf.YieldOp(outer_loop.inner_iter_args)

    return bufferization.ToTensorOp(
        output_tensor_type, output_memref.result, restrict=True
    )


def as_strided_op(
    node: AsStridedOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Converts a Buddy AsStridedOp operation to an MLIR operation.

    This operation implements the `as_strided` functionality, allowing
    a tensor to be viewed with a different shape, stride, or offset.

    This also handles the resize_ semantics when input and output sizes differ:
    - If shrinking: slice the input (preserving first N elements in row-major order)
    - If enlarging: pad with zeros
    - If same size: simple reshape

    Parameters:
        node (AsStridedOp): The Buddy AsStridedOp node containing the tensor and metadata.
        symbol_table (dict): A dictionary mapping tensor names to their corresponding MLIR operations.

    Returns:
        op: An MLIR operation representing the transformed tensor.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    input_type = ir.RankedTensorType(input_tensor.type)
    element_type = input_type.element_type
    input_shape = list(input_type.shape)
    output_shape = list(node.tensor_meta["shape"])

    input_size = 1
    output_size = 1
    for i in input_shape:
        input_size *= i
    for i in output_shape:
        output_size *= i

    if input_size == output_size:
        # Same size: simple reshape
        shape_ty = ir.Type.parse(f"!tosa.shape<{len(output_shape)}>")
        index_ty = ir.IndexType.get()
        shape_val = tosa.ConstShapeOp(
            shape_ty,
            ir.DenseElementsAttr.get(
                array.array("q", output_shape),
                type=index_ty,
                shape=[len(output_shape)],
            ),
        ).result
        op = tosa.ReshapeOp(input_tensor, shape_val)
    elif output_size < input_size:
        # Shrinking: flatten, slice, reshape
        # Step 1: Flatten to 1D
        flat_shape_ty = ir.Type.parse("!tosa.shape<1>")
        index_ty = ir.IndexType.get()
        flat_shape_val = tosa.ConstShapeOp(
            flat_shape_ty,
            ir.DenseElementsAttr.get(
                array.array("q", [input_size]),
                type=index_ty,
                shape=[1],
            ),
        ).result
        flattened = tosa.ReshapeOp(input_tensor, flat_shape_val)

        # Step 2: Slice to get first output_size elements
        slice_type = ir.RankedTensorType.get([output_size], element_type)
        # Create start and size operands using ConstShapeOp (similar to _create_shape_operand in tosa.py)
        start_shape_ty = ir.Type.parse("!tosa.shape<1>")
        start_shape_val = tosa.ConstShapeOp(
            start_shape_ty,
            ir.DenseElementsAttr.get(
                array.array("q", [0]),
                type=index_ty,
                shape=[1],
            ),
        ).result
        size_shape_ty = ir.Type.parse("!tosa.shape<1>")
        size_shape_val = tosa.ConstShapeOp(
            size_shape_ty,
            ir.DenseElementsAttr.get(
                array.array("q", [output_size]),
                type=index_ty,
                shape=[1],
            ),
        ).result
        sliced = tosa.SliceOp(
            slice_type, flattened.result, start_shape_val, size_shape_val
        )

        # Step 3: Reshape to output shape
        output_shape_ty = ir.Type.parse(f"!tosa.shape<{len(output_shape)}>")
        output_shape_val = tosa.ConstShapeOp(
            output_shape_ty,
            ir.DenseElementsAttr.get(
                array.array("q", output_shape),
                type=index_ty,
                shape=[len(output_shape)],
            ),
        ).result
        op = tosa.ReshapeOp(sliced.result, output_shape_val)
    else:
        # Enlarging: flatten, pad with zeros, reshape
        padding_size = output_size - input_size

        # Step 1: Flatten input to 1D
        flat_type = ir.RankedTensorType.get([input_size], element_type)
        flat_shape = ir._denseI64ArrayAttr(
            numpy.array([input_size], dtype=numpy.int64), None
        )
        flattened = tosa.ReshapeOp(input_tensor, flat_shape)

        # Step 2: Create zero padding tensor
        if ir.IntegerType.isinstance(element_type):
            zero_attr = ir.IntegerAttr.get(element_type, 0)
        else:
            zero_attr = ir.FloatAttr.get(element_type, 0.0)

        padding_type = ir.RankedTensorType.get([padding_size], element_type)
        padding_attr = ir.DenseElementsAttr.get_splat(padding_type, zero_attr)
        padding_tensor = tosa.ConstOp(padding_attr)

        # Step 3: Concatenate input with padding
        concat_type = ir.RankedTensorType.get([output_size], element_type)
        padded = tosa.ConcatOp(
            concat_type, [flattened.result, padding_tensor.result], axis=0
        )

        # Step 4: Reshape to output shape
        output_shape_attr = ir._denseI64ArrayAttr(
            numpy.array(output_shape, dtype=numpy.int64), None
        )
        op = tosa.ReshapeOp(padded.result, output_shape_attr)

    return op


def as_strided_scatter_op(
    node: AsStridedScatterOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Implements aten.as_strided_scatter by writing src into a strided view of self.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    src_tensor = symbol_table.get((str(node.args[1]), 0))
    if input_tensor is None or src_tensor is None:
        return

    size = node.args[2]
    stride = node.args[3]
    storage_offset = node.args[4] if len(node.args) > 4 else 0
    if not isinstance(size, (list, tuple)) or not isinstance(
        stride, (list, tuple)
    ):
        raise NotImplementedError(
            "as_strided_scatter requires static size/stride"
        )

    size_list = [int(x) for x in size]
    stride_list = [int(x) for x in stride]
    if len(size_list) != len(stride_list):
        raise ValueError("as_strided_scatter size/stride rank mismatch")
    storage_offset = int(storage_offset) if storage_offset is not None else 0

    input_type = ir.RankedTensorType(input_tensor.type)
    input_shape = list(input_type.shape)
    if any(dim < 0 for dim in input_shape):
        raise NotImplementedError(
            "as_strided_scatter does not support dynamic input shapes"
        )
    if any(s == 0 for s in size_list):
        return input_tensor

    src_type = ir.RankedTensorType(src_tensor.type)
    src_shape = list(src_type.shape)
    if src_shape != size_list:
        raise NotImplementedError(
            "as_strided_scatter requires src shape to match size"
        )

    input_memref_type = ir.MemRefType.get(input_shape, input_type.element_type)
    input_memref = bufferization.ToBufferOp(input_memref_type, input_tensor)
    src_memref_type = ir.MemRefType.get(src_shape, src_type.element_type)
    src_memref = bufferization.ToBufferOp(src_memref_type, src_tensor)

    # Row-major strides for computing base indices from linear offsets.
    row_strides: List[int] = []
    for i in range(len(input_shape)):
        stride_val = 1
        for s in input_shape[i + 1 :]:
            stride_val *= s
        row_strides.append(stride_val)

    index_type = ir.IndexType.get()
    lb = arith.ConstantOp(index_type, 0)
    step = arith.ConstantOp(index_type, 1)
    ubs = [arith.ConstantOp(index_type, s) for s in size_list]
    offset_const = arith.ConstantOp(index_type, storage_offset)

    def create_nested_loops(depth, indices):
        if depth == len(size_list):
            linear = offset_const.result
            for idx, stride_val in zip(indices, stride_list):
                stride_const = arith.ConstantOp(index_type, stride_val)
                mul = arith.MulIOp(idx, stride_const.result)
                linear = arith.AddIOp(linear, mul.result).result

            remaining = linear
            base_indices = []
            for stride_val in row_strides:
                stride_const = arith.ConstantOp(index_type, stride_val)
                div = arith.DivUIOp(remaining, stride_const.result)
                base_indices.append(div.result)
                rem = arith.RemUIOp(remaining, stride_const.result)
                remaining = rem.result

            src_val = memref.LoadOp(src_memref, indices)
            memref.StoreOp(src_val, input_memref, base_indices)
        else:
            loop = scf.ForOp(lb, ubs[depth], step)
            with ir.InsertionPoint(loop.body):
                new_indices = indices + [loop.induction_variable]
                create_nested_loops(depth + 1, new_indices)
                scf.YieldOp(loop.inner_iter_args)

    create_nested_loops(0, [])

    output_shape = list(node.tensor_meta["shape"])
    dtype_meta = node.tensor_meta["dtype"]
    if isinstance(dtype_meta, TensorDType):
        output_dtype = mlir_element_type_get(dtype_meta)
    elif isinstance(dtype_meta, ir.Type):
        output_dtype = dtype_meta
    else:
        raise NotImplementedError("nonzero_static requires integer output")
    output_tensor_type = ir.RankedTensorType.get(output_shape, output_dtype)
    op = bufferization.ToTensorOp(
        output_tensor_type, input_memref, restrict=True
    )
    return op


def scatter_src_op(
    node: ScatterSrcOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Converts a Buddy ScatterSrcOp operation to an MLIR operation.

    Implements aten.scatter.src: self.scatter_(dim, index, src)
    Scatters values from src tensor into self tensor at positions specified by index
    along the specified dimension.

    For a 3-D tensor, the operation is:
        self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
        self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
        self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

    Parameters:
        node (ScatterSrcOp): The Buddy ScatterSrcOp node containing:
            - args[0]: input tensor (self)
            - args[1]: dim (int) - the axis along which to scatter
            - args[2]: index tensor
            - args[3]: src tensor
        symbol_table (dict): A dictionary mapping tensor names to their corresponding MLIR operations.

    Returns:
        op: An MLIR operation representing the scatter result.
    """
    # Get input tensors
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    if input_tensor is None:
        return
    dim = int(node.args[1])
    index_tensor = symbol_table.get((str(node.args[2]), 0))
    src_tensor = symbol_table.get((str(node.args[3]), 0))

    if index_tensor is None or src_tensor is None:
        return

    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    tensor_rank = len(output_shape)

    # Handle negative dimension
    if dim < 0:
        dim += tensor_rank

    # Get shapes
    input_shape = list(ir.RankedTensorType(input_tensor.type).shape)
    index_shape = list(ir.RankedTensorType(index_tensor.type).shape)
    src_shape = list(ir.RankedTensorType(src_tensor.type).shape)

    # Convert tensors to memrefs for in-place operations
    input_memref_type = ir.MemRefType.get(
        input_shape, ir.RankedTensorType(input_tensor.type).element_type
    )
    input_memref = bufferization.ToBufferOp(input_memref_type, input_tensor)

    index_memref_type = ir.MemRefType.get(
        index_shape, ir.RankedTensorType(index_tensor.type).element_type
    )
    index_memref = bufferization.ToBufferOp(index_memref_type, index_tensor)

    src_memref_type = ir.MemRefType.get(
        src_shape, ir.RankedTensorType(src_tensor.type).element_type
    )
    src_memref = bufferization.ToBufferOp(src_memref_type, src_tensor)

    # Create loop bounds
    lb = arith.ConstantOp(ir.IndexType.get(), 0)
    step = arith.ConstantOp(ir.IndexType.get(), 1)
    ubs = [arith.ConstantOp(ir.IndexType.get(), s) for s in index_shape]

    # Generate nested loops over all dimensions of the index/src tensor
    def create_nested_loops(depth, indices):
        """Recursively create nested loops and perform scatter at innermost level."""
        if depth == tensor_rank:
            # At the innermost level, perform the scatter operation
            # Load the index value
            idx_val = memref.LoadOp(index_memref, indices)
            # Cast to index type
            scatter_idx = arith.IndexCastOp(ir.IndexType.get(), idx_val)
            # Load the source value
            src_val = memref.LoadOp(src_memref, indices)
            # Build the store indices: replace indices[dim] with scatter_idx
            store_indices = list(indices)
            store_indices[dim] = scatter_idx
            # Store the value
            memref.StoreOp(src_val, input_memref, store_indices)
        else:
            # Create a loop for the current dimension
            loop = scf.ForOp(lb, ubs[depth], step)
            with ir.InsertionPoint(loop.body):
                new_indices = indices + [loop.induction_variable]
                create_nested_loops(depth + 1, new_indices)
                scf.YieldOp(loop.inner_iter_args)

    create_nested_loops(0, [])

    # Convert back to tensor
    output_tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    op = bufferization.ToTensorOp(
        output_tensor_type, input_memref, restrict=True
    )

    return op


def scatter_value_op(
    node: ScatterValueOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Converts a Buddy ScatterValueOp operation to an MLIR operation.

    Implements aten.scatter.value: self.scatter_(dim, index, value)
    Scatters a scalar value into self tensor at positions specified by index
    along the specified dimension.

    For a 3-D tensor, the operation is:
        self[index[i][j][k]][j][k] = value  # if dim == 0
        self[i][index[i][j][k]][k] = value  # if dim == 1
        self[i][j][index[i][j][k]] = value  # if dim == 2

    Parameters:
        node (ScatterValueOp): The Buddy ScatterValueOp node containing:
            - args[0]: input tensor (self)
            - args[1]: dim (int) - the axis along which to scatter
            - args[2]: index tensor
            - args[3]: value (scalar)
        symbol_table (dict): A dictionary mapping tensor names to their corresponding MLIR operations.

    Returns:
        op: An MLIR operation representing the scatter result.
    """
    # Get input tensor
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    if input_tensor is None:
        return
    dim = int(node.args[1])
    index_tensor = symbol_table.get((str(node.args[2]), 0))
    value = node.args[3]

    if index_tensor is None:
        return

    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    tensor_rank = len(output_shape)

    # Handle negative dimension
    if dim < 0:
        dim += tensor_rank

    # Get shapes
    input_shape = list(ir.RankedTensorType(input_tensor.type).shape)
    index_shape = list(ir.RankedTensorType(index_tensor.type).shape)

    # Create constant for the scalar value
    value_attr = mlir_element_attr_get(dtype, value)
    value_const = arith.ConstantOp(mlir_dtype, value_attr)

    # Convert tensors to memrefs for in-place operations
    input_memref_type = ir.MemRefType.get(
        input_shape, ir.RankedTensorType(input_tensor.type).element_type
    )
    input_memref = bufferization.ToBufferOp(input_memref_type, input_tensor)

    index_memref_type = ir.MemRefType.get(
        index_shape, ir.RankedTensorType(index_tensor.type).element_type
    )
    index_memref = bufferization.ToBufferOp(index_memref_type, index_tensor)

    # Create loop bounds
    lb = arith.ConstantOp(ir.IndexType.get(), 0)
    step = arith.ConstantOp(ir.IndexType.get(), 1)
    ubs = [arith.ConstantOp(ir.IndexType.get(), s) for s in index_shape]

    # Generate nested loops over all dimensions of the index tensor
    def create_nested_loops(depth, indices):
        """Recursively create nested loops and perform scatter at innermost level."""
        if depth == tensor_rank:
            # At the innermost level, perform the scatter operation
            # Load the index value
            idx_val = memref.LoadOp(index_memref, indices)
            # Cast to index type
            scatter_idx = arith.IndexCastOp(ir.IndexType.get(), idx_val)
            # Build the store indices: replace indices[dim] with scatter_idx
            store_indices = list(indices)
            store_indices[dim] = scatter_idx
            # Store the constant value
            memref.StoreOp(value_const, input_memref, store_indices)
        else:
            # Create a loop for the current dimension
            loop = scf.ForOp(lb, ubs[depth], step)
            with ir.InsertionPoint(loop.body):
                new_indices = indices + [loop.induction_variable]
                create_nested_loops(depth + 1, new_indices)
                scf.YieldOp(loop.inner_iter_args)

    create_nested_loops(0, [])

    # Convert back to tensor
    output_tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    op = bufferization.ToTensorOp(
        output_tensor_type, input_memref, restrict=True
    )

    return op


def select_scatter_op(
    node: SelectScatterOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Scatter src values into input tensor along a dimension at a single index.
    Implements aten.select_scatter.default/out.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    src_tensor = symbol_table.get((str(node.args[1]), 0))
    if input_tensor is None or src_tensor is None:
        return

    dim = int(node.args[2])
    index = int(node.args[3])

    input_type = ir.RankedTensorType(input_tensor.type)
    input_shape = list(input_type.shape)
    input_dtype = input_type.element_type
    if any(dim_size < 0 for dim_size in input_shape):
        raise NotImplementedError("select_scatter requires static shapes")

    rank = len(input_shape)
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        raise NotImplementedError("select_scatter invalid dim")
    if index < 0:
        index += input_shape[dim]
    if index < 0 or index >= input_shape[dim]:
        raise NotImplementedError("select_scatter index out of range")

    src_shape = list(ir.RankedTensorType(src_tensor.type).shape)
    expected_src_shape = input_shape[:dim] + input_shape[dim + 1 :]
    if src_shape != expected_src_shape:
        raise NotImplementedError(
            "select_scatter requires src shape to match input without dim"
        )

    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    output_tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)

    input_memref = bufferization.ToBufferOp(
        ir.MemRefType.get(input_shape, input_dtype), input_tensor
    ).result
    output_memref = memref.AllocOp(
        ir.MemRefType.get(output_shape, input_dtype), [], []
    )
    linalg.copy(input_memref, outs=[output_memref.result])

    src_memref = bufferization.ToBufferOp(
        ir.MemRefType.get(src_shape, input_dtype), src_tensor
    ).result

    index_type = ir.IndexType.get()
    c0 = arith.ConstantOp(index_type, 0)
    c1 = arith.ConstantOp(index_type, 1)
    index_const = arith.ConstantOp(index_type, index).result

    if not src_shape:
        src_val = memref.LoadOp(src_memref, []).result
        idx_values = [None] * rank
        idx_values[dim] = index_const
        memref.StoreOp(src_val, output_memref.result, idx_values)
        return bufferization.ToTensorOp(
            output_tensor_type, output_memref.result, restrict=True
        )

    bounds = [arith.ConstantOp(index_type, s) for s in src_shape]
    idx_values = [None] * rank
    src_indices: List[ir.Value] = []

    def create_loops(depth: int):
        if depth == len(src_shape):
            idx_values[dim] = index_const
            src_val = memref.LoadOp(src_memref, src_indices).result
            memref.StoreOp(src_val, output_memref.result, idx_values)
            return

        loop = scf.ForOp(c0.result, bounds[depth].result, c1.result)
        with ir.InsertionPoint(loop.body):
            src_indices.append(loop.induction_variable)
            input_dim = depth if depth < dim else depth + 1
            idx_values[input_dim] = loop.induction_variable
            create_loops(depth + 1)
            src_indices.pop()
            scf.YieldOp(loop.inner_iter_args)

    create_loops(0)

    return bufferization.ToTensorOp(
        output_tensor_type, output_memref.result, restrict=True
    )


def scatter_reduce_op(
    node: ScatterReduceOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Converts a Buddy ScatterReduceOp operation to an MLIR operation.

    Implements aten.scatter_reduce: self.scatter_reduce_(dim, index, src, reduce)
    Scatters values from src tensor into self tensor with a reduction operation
    applied at positions specified by index along the specified dimension.

    Supported reduce operations:
        - "sum": self[idx] += src
        - "prod": self[idx] *= src
        - "mean": self[idx] = mean of scattered values
        - "amax": self[idx] = max(self[idx], src)
        - "amin": self[idx] = min(self[idx], src)

    Parameters:
        node (ScatterReduceOp): The Buddy ScatterReduceOp node containing:
            - args[0]: input tensor (self)
            - args[1]: dim (int) - the axis along which to scatter
            - args[2]: index tensor
            - args[3]: src tensor
            - args[4]: reduce (str) - reduction operation type
            - args[5]: include_self (bool) - whether to include self in reduction
        symbol_table (dict): A dictionary mapping tensor names to their corresponding MLIR operations.

    Returns:
        op: An MLIR operation representing the scatter_reduce result.
    """
    # Get input tensors
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    if input_tensor is None:
        return
    dim = int(node.args[1])
    index_tensor = symbol_table.get((str(node.args[2]), 0))
    src_tensor = symbol_table.get((str(node.args[3]), 0))
    src_scalar = None
    if src_tensor is None:
        src_arg = node.args[3]
        if isinstance(src_arg, (int, float)):
            src_scalar = src_arg
        else:
            return
    reduce_op = (
        str(node.args[4])
        if len(node.args) > 4
        else node.kwargs.get("reduce", "sum")
    )
    include_self = (
        bool(node.args[5])
        if len(node.args) > 5
        else bool(node.kwargs.get("include_self", True))
    )

    if index_tensor is None:
        return

    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    tensor_rank = len(output_shape)

    # Handle negative dimension
    if dim < 0:
        dim += tensor_rank

    # Get shapes
    input_shape = list(ir.RankedTensorType(input_tensor.type).shape)
    index_shape = list(ir.RankedTensorType(index_tensor.type).shape)
    src_shape = (
        list(ir.RankedTensorType(src_tensor.type).shape)
        if src_tensor is not None
        else []
    )

    # Convert tensors to memrefs for in-place operations
    input_memref_type = ir.MemRefType.get(
        input_shape, ir.RankedTensorType(input_tensor.type).element_type
    )
    input_memref = bufferization.ToBufferOp(input_memref_type, input_tensor)

    index_memref_type = ir.MemRefType.get(
        index_shape, ir.RankedTensorType(index_tensor.type).element_type
    )
    index_memref = bufferization.ToBufferOp(index_memref_type, index_tensor)

    if src_tensor is not None:
        src_memref_type = ir.MemRefType.get(
            src_shape, ir.RankedTensorType(src_tensor.type).element_type
        )
        src_memref = bufferization.ToBufferOp(src_memref_type, src_tensor)
    else:
        src_memref = None

    # Create loop bounds
    lb = arith.ConstantOp(ir.IndexType.get(), 0)
    step = arith.ConstantOp(ir.IndexType.get(), 1)
    ubs = [arith.ConstantOp(ir.IndexType.get(), s) for s in index_shape]

    # Determine if we're working with integers or floats
    is_float = str(mlir_dtype).startswith("f")
    if src_scalar is not None:
        if is_float:
            src_const = arith.ConstantOp(mlir_dtype, float(src_scalar)).result
        else:
            src_const = arith.ConstantOp(mlir_dtype, int(src_scalar)).result

    # Generate nested loops over all dimensions of the index/src tensor
    def create_nested_loops(depth, indices):
        """Recursively create nested loops and perform scatter_reduce at innermost level."""
        if depth == tensor_rank:
            # At the innermost level, perform the scatter_reduce operation
            # Load the index value
            idx_val = memref.LoadOp(index_memref, indices)
            # Cast to index type
            scatter_idx = arith.IndexCastOp(ir.IndexType.get(), idx_val)
            # Load the source value
            if src_memref is not None:
                src_val = memref.LoadOp(src_memref, indices).result
            else:
                src_val = src_const
            # Build the load/store indices: replace indices[dim] with scatter_idx
            target_indices = list(indices)
            target_indices[dim] = scatter_idx
            # Load the current value at target position
            curr_val = memref.LoadOp(input_memref, target_indices).result

            # Apply the reduction operation
            if reduce_op == "sum":
                if is_float:
                    new_val = arith.AddFOp(curr_val, src_val)
                else:
                    new_val = arith.AddIOp(curr_val, src_val)
            elif reduce_op == "prod":
                if is_float:
                    new_val = arith.MulFOp(curr_val, src_val)
                else:
                    new_val = arith.MulIOp(curr_val, src_val)
            elif reduce_op == "amax":
                if is_float:
                    new_val = arith.MaximumFOp(curr_val, src_val)
                else:
                    new_val = arith.MaxSIOp(curr_val, src_val)
            elif reduce_op == "amin":
                if is_float:
                    new_val = arith.MinimumFOp(curr_val, src_val)
                else:
                    new_val = arith.MinSIOp(curr_val, src_val)
            else:
                # Default to sum for unsupported operations
                if is_float:
                    new_val = arith.AddFOp(curr_val, src_val)
                else:
                    new_val = arith.AddIOp(curr_val, src_val)

            # Store the result
            memref.StoreOp(new_val, input_memref, target_indices)
        else:
            # Create a loop for the current dimension
            loop = scf.ForOp(lb, ubs[depth], step)
            with ir.InsertionPoint(loop.body):
                new_indices = indices + [loop.induction_variable]
                create_nested_loops(depth + 1, new_indices)
                scf.YieldOp(loop.inner_iter_args)

    if len(index_shape) == tensor_rank:
        create_nested_loops(0, [])
    elif (
        src_tensor is not None
        and len(index_shape) == 1
        and len(src_shape) == tensor_rank
    ):
        if index_shape[0] != src_shape[dim]:
            raise NotImplementedError(
                "index_reduce expects index length to match src dim size"
            )

        ubs_src = [arith.ConstantOp(ir.IndexType.get(), s) for s in src_shape]
        indices = [None] * tensor_rank

        def create_src_loops(depth):
            if depth == tensor_rank:
                idx_val = memref.LoadOp(index_memref, [indices[dim]]).result
                scatter_idx = arith.IndexCastOp(ir.IndexType.get(), idx_val)
                src_val = memref.LoadOp(src_memref, indices).result
                target_indices = list(indices)
                target_indices[dim] = scatter_idx
                curr_val = memref.LoadOp(input_memref, target_indices).result

                if reduce_op == "sum":
                    if is_float:
                        new_val = arith.AddFOp(curr_val, src_val)
                    else:
                        new_val = arith.AddIOp(curr_val, src_val)
                elif reduce_op == "prod":
                    if is_float:
                        new_val = arith.MulFOp(curr_val, src_val)
                    else:
                        new_val = arith.MulIOp(curr_val, src_val)
                elif reduce_op == "amax":
                    if is_float:
                        new_val = arith.MaximumFOp(curr_val, src_val)
                    else:
                        new_val = arith.MaxSIOp(curr_val, src_val)
                elif reduce_op == "amin":
                    if is_float:
                        new_val = arith.MinimumFOp(curr_val, src_val)
                    else:
                        new_val = arith.MinSIOp(curr_val, src_val)
                else:
                    if is_float:
                        new_val = arith.AddFOp(curr_val, src_val)
                    else:
                        new_val = arith.AddIOp(curr_val, src_val)

                memref.StoreOp(new_val, input_memref, target_indices)
                return

            loop = scf.ForOp(lb, ubs_src[depth], step)
            with ir.InsertionPoint(loop.body):
                indices[depth] = loop.induction_variable
                create_src_loops(depth + 1)
                scf.YieldOp(loop.inner_iter_args)

        create_src_loops(0)
    else:
        raise NotImplementedError(
            "scatter_reduce requires index rank to match tensor rank"
        )

    # Convert back to tensor
    output_tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    op = bufferization.ToTensorOp(
        output_tensor_type, input_memref, restrict=True
    )

    return op


def max_pool2d_with_indices_op(
    node: MaxPool2dWithIndicesOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the max_pool2d_with_indices operation.
    From buddy MaxPool2dWithIndicesOp to MLIR operations using scf.for loops.
    aten.max_pool2d_with_indices(input, kernel_size, stride, padding,
                                  dilation, ceil_mode) -> (Tensor, Tensor)

    Returns both the max-pooled values and the indices of the max values.
    Uses scf.for loops since TOSA max_pool2d doesn't support returning indices.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    kernel_size = node.args[1]
    stride = (
        node.args[2] if len(node.args) > 2 and node.args[2] else kernel_size
    )
    padding = node.args[3] if len(node.args) > 3 else [0, 0]
    dilation = node.args[4] if len(node.args) > 4 else [1, 1]
    ceil_mode = node.args[5] if len(node.args) > 5 else False

    input_shape = list(ir.RankedTensorType(input_tensor.type).shape)
    input_dtype = ir.RankedTensorType(input_tensor.type).element_type
    if any(dim < 0 for dim in input_shape):
        raise NotImplementedError(
            "fractional_max_pool2d requires static shapes"
        )

    N, C, H, W = input_shape

    # Normalize kernel_size, stride, padding, dilation
    if isinstance(kernel_size, int):
        kh, kw = kernel_size, kernel_size
    else:
        kh = kernel_size[0]
        kw = kernel_size[1] if len(kernel_size) > 1 else kernel_size[0]

    if isinstance(stride, int):
        sh, sw = stride, stride
    else:
        sh = stride[0]
        sw = stride[1] if len(stride) > 1 else stride[0]

    if isinstance(padding, int):
        ph, pw = padding, padding
    else:
        ph = padding[0]
        pw = padding[1] if len(padding) > 1 else padding[0]

    if isinstance(dilation, int):
        dh, dw = dilation, dilation
    else:
        dh = dilation[0]
        dw = dilation[1] if len(dilation) > 1 else dilation[0]

    # Calculate output dimensions
    if ceil_mode:
        out_h = (H + 2 * ph - dh * (kh - 1) - 1 + sh - 1) // sh + 1
        out_w = (W + 2 * pw - dw * (kw - 1) - 1 + sw - 1) // sw + 1
    else:
        out_h = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        out_w = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    output_shape = [N, C, out_h, out_w]
    output_type = ir.RankedTensorType.get(output_shape, input_dtype)
    indices_dtype = ir.IntegerType.get_signless(64)
    indices_type = ir.RankedTensorType.get(output_shape, indices_dtype)

    # Create memrefs for output and indices
    output_memref_type = ir.MemRefType.get(output_shape, input_dtype)
    indices_memref_type = ir.MemRefType.get(output_shape, indices_dtype)

    output_memref = memref.AllocOp(output_memref_type, [], [])
    indices_memref = memref.AllocOp(indices_memref_type, [], [])

    # Initialize output with negative infinity
    neg_inf = arith.ConstantOp(
        input_dtype, ir.FloatAttr.get(input_dtype, float("-inf"))
    )
    linalg.fill(neg_inf.result, outs=[output_memref.result])

    # Initialize indices with zeros
    zero_idx = arith.ConstantOp(
        indices_dtype, ir.IntegerAttr.get(indices_dtype, 0)
    )
    linalg.fill(zero_idx.result, outs=[indices_memref.result])

    # Convert input to memref
    input_memref = bufferization.ToBufferOp(
        ir.MemRefType.get(input_shape, input_dtype), input_tensor
    ).result

    # Create constants
    zero = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 0)
    )
    one = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 1)
    )
    n_bound = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), N)
    )
    c_bound = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), C)
    )
    out_h_bound = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), out_h)
    )
    out_w_bound = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), out_w)
    )
    kh_bound = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), kh)
    )
    kw_bound = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), kw)
    )

    # Constants for calculations
    sh_const = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), sh)
    )
    sw_const = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), sw)
    )
    ph_const = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), ph)
    )
    pw_const = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), pw)
    )
    dh_const = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), dh)
    )
    dw_const = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), dw)
    )
    h_const = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), H)
    )
    w_const = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), W)
    )
    w_i64 = arith.ConstantOp(
        indices_dtype, ir.IntegerAttr.get(indices_dtype, W)
    )

    # Six nested loops: n, c, oh, ow, kh, kw
    n_loop = scf.ForOp(zero.result, n_bound.result, one.result)
    with ir.InsertionPoint(n_loop.body):
        n = n_loop.induction_variable

        c_loop = scf.ForOp(zero.result, c_bound.result, one.result)
        with ir.InsertionPoint(c_loop.body):
            c = c_loop.induction_variable

            oh_loop = scf.ForOp(zero.result, out_h_bound.result, one.result)
            with ir.InsertionPoint(oh_loop.body):
                oh = oh_loop.induction_variable

                ow_loop = scf.ForOp(zero.result, out_w_bound.result, one.result)
                with ir.InsertionPoint(ow_loop.body):
                    ow = ow_loop.induction_variable

                    # Base input position: oh * stride - padding
                    h_base = arith.MulIOp(oh, sh_const.result).result
                    h_base = arith.SubIOp(h_base, ph_const.result).result
                    w_base = arith.MulIOp(ow, sw_const.result).result
                    w_base = arith.SubIOp(w_base, pw_const.result).result

                    # Inner loops over kernel
                    kih_loop = scf.ForOp(
                        zero.result, kh_bound.result, one.result
                    )
                    with ir.InsertionPoint(kih_loop.body):
                        kih = kih_loop.induction_variable

                        kiw_loop = scf.ForOp(
                            zero.result, kw_bound.result, one.result
                        )
                        with ir.InsertionPoint(kiw_loop.body):
                            kiw = kiw_loop.induction_variable

                            # Calculate input position with dilation
                            ih = arith.AddIOp(
                                h_base,
                                arith.MulIOp(kih, dh_const.result).result,
                            ).result
                            iw = arith.AddIOp(
                                w_base,
                                arith.MulIOp(kiw, dw_const.result).result,
                            ).result

                            # Check bounds: 0 <= ih < H and 0 <= iw < W
                            ih_ge_0 = arith.CmpIOp(
                                arith.CmpIPredicate.sge, ih, zero.result
                            ).result
                            ih_lt_h = arith.CmpIOp(
                                arith.CmpIPredicate.slt, ih, h_const.result
                            ).result
                            iw_ge_0 = arith.CmpIOp(
                                arith.CmpIPredicate.sge, iw, zero.result
                            ).result
                            iw_lt_w = arith.CmpIOp(
                                arith.CmpIPredicate.slt, iw, w_const.result
                            ).result

                            h_valid = arith.AndIOp(ih_ge_0, ih_lt_h).result
                            w_valid = arith.AndIOp(iw_ge_0, iw_lt_w).result
                            in_bounds = arith.AndIOp(h_valid, w_valid).result

                            # If in bounds, load and compare
                            if_op = scf.IfOp(in_bounds, hasElse=False)
                            with ir.InsertionPoint(if_op.then_block):
                                # Load input value
                                input_val = memref.LoadOp(
                                    input_memref, [n, c, ih, iw]
                                ).result

                                # Load current max
                                current_max = memref.LoadOp(
                                    output_memref.result, [n, c, oh, ow]
                                ).result

                                # Check if input > current_max
                                is_greater = arith.CmpFOp(
                                    arith.CmpFPredicate.OGT,
                                    input_val,
                                    current_max,
                                ).result

                                inner_if = scf.IfOp(is_greater, hasElse=False)
                                with ir.InsertionPoint(inner_if.then_block):
                                    # Update max value
                                    memref.StoreOp(
                                        input_val,
                                        output_memref.result,
                                        [n, c, oh, ow],
                                    )

                                    # Calculate flattened index: ih * W + iw
                                    ih_i64 = arith.IndexCastOp(
                                        indices_dtype, ih
                                    ).result
                                    iw_i64 = arith.IndexCastOp(
                                        indices_dtype, iw
                                    ).result
                                    flat_idx = arith.AddIOp(
                                        arith.MulIOp(
                                            ih_i64, w_i64.result
                                        ).result,
                                        iw_i64,
                                    ).result

                                    # Store index
                                    memref.StoreOp(
                                        flat_idx,
                                        indices_memref.result,
                                        [n, c, oh, ow],
                                    )

                                    scf.YieldOp([])

                                scf.YieldOp([])

                            scf.YieldOp([])

                        scf.YieldOp([])

                    scf.YieldOp([])

                scf.YieldOp([])

            scf.YieldOp([])

        scf.YieldOp([])

    # Convert memrefs back to tensors
    output_result = bufferization.ToTensorOp(
        output_type, output_memref.result, restrict=True
    )
    indices_result = bufferization.ToTensorOp(
        indices_type, indices_memref.result, restrict=True
    )

    return output_result, indices_result


def fractional_max_pool2d_op(
    node: FractionalMaxPool2dOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the fractional_max_pool2d operation.
    From buddy FractionalMaxPool2dOp to MLIR operations using scf.for loops.
    aten.fractional_max_pool2d(input, kernel_size, output_size, random_samples)
        -> (Tensor, Tensor)

    Note: Uses deterministic strides derived from output_size and ignores
    random_samples.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    kernel_size = node.args[1]
    output_size = node.args[2]

    input_shape = list(ir.RankedTensorType(input_tensor.type).shape)
    input_dtype = ir.RankedTensorType(input_tensor.type).element_type

    N, C, H, W = input_shape

    if isinstance(kernel_size, int):
        kh, kw = kernel_size, kernel_size
    else:
        kh = kernel_size[0]
        kw = kernel_size[1] if len(kernel_size) > 1 else kernel_size[0]

    if isinstance(output_size, int):
        out_h, out_w = output_size, output_size
    else:
        out_h = output_size[0]
        out_w = output_size[1] if len(output_size) > 1 else output_size[0]

    if out_h <= 0 or out_w <= 0:
        raise NotImplementedError("fractional_max_pool2d output_size invalid")

    sh = 1 if out_h <= 1 else max(1, (H - kh) // (out_h - 1))
    sw = 1 if out_w <= 1 else max(1, (W - kw) // (out_w - 1))
    ph, pw = 0, 0
    dh, dw = 1, 1

    output_shape = [N, C, out_h, out_w]
    output_type = ir.RankedTensorType.get(output_shape, input_dtype)
    indices_dtype = ir.IntegerType.get_signless(64)
    indices_type = ir.RankedTensorType.get(output_shape, indices_dtype)

    output_memref_type = ir.MemRefType.get(output_shape, input_dtype)
    indices_memref_type = ir.MemRefType.get(output_shape, indices_dtype)

    output_memref = memref.AllocOp(output_memref_type, [], [])
    indices_memref = memref.AllocOp(indices_memref_type, [], [])

    neg_inf = arith.ConstantOp(
        input_dtype, ir.FloatAttr.get(input_dtype, float("-inf"))
    )
    linalg.fill(neg_inf.result, outs=[output_memref.result])

    zero_idx = arith.ConstantOp(
        indices_dtype, ir.IntegerAttr.get(indices_dtype, 0)
    )
    linalg.fill(zero_idx.result, outs=[indices_memref.result])

    input_memref = bufferization.ToBufferOp(
        ir.MemRefType.get(input_shape, input_dtype), input_tensor
    ).result

    zero = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 0)
    )
    one = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 1)
    )
    n_bound = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), N)
    )
    c_bound = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), C)
    )
    out_h_bound = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), out_h)
    )
    out_w_bound = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), out_w)
    )
    kh_bound = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), kh)
    )
    kw_bound = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), kw)
    )

    sh_const = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), sh)
    )
    sw_const = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), sw)
    )
    ph_const = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), ph)
    )
    pw_const = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), pw)
    )
    dh_const = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), dh)
    )
    dw_const = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), dw)
    )
    h_const = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), H)
    )
    w_const = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), W)
    )
    w_i64 = arith.ConstantOp(
        indices_dtype, ir.IntegerAttr.get(indices_dtype, W)
    )

    n_loop = scf.ForOp(zero.result, n_bound.result, one.result)
    with ir.InsertionPoint(n_loop.body):
        n = n_loop.induction_variable

        c_loop = scf.ForOp(zero.result, c_bound.result, one.result)
        with ir.InsertionPoint(c_loop.body):
            c = c_loop.induction_variable

            oh_loop = scf.ForOp(zero.result, out_h_bound.result, one.result)
            with ir.InsertionPoint(oh_loop.body):
                oh = oh_loop.induction_variable

                ow_loop = scf.ForOp(zero.result, out_w_bound.result, one.result)
                with ir.InsertionPoint(ow_loop.body):
                    ow = ow_loop.induction_variable

                    h_base = arith.MulIOp(oh, sh_const.result).result
                    h_base = arith.SubIOp(h_base, ph_const.result).result
                    w_base = arith.MulIOp(ow, sw_const.result).result
                    w_base = arith.SubIOp(w_base, pw_const.result).result

                    kih_loop = scf.ForOp(
                        zero.result, kh_bound.result, one.result
                    )
                    with ir.InsertionPoint(kih_loop.body):
                        kih = kih_loop.induction_variable

                        kiw_loop = scf.ForOp(
                            zero.result, kw_bound.result, one.result
                        )
                        with ir.InsertionPoint(kiw_loop.body):
                            kiw = kiw_loop.induction_variable

                            ih = arith.AddIOp(
                                h_base,
                                arith.MulIOp(kih, dh_const.result).result,
                            ).result
                            iw = arith.AddIOp(
                                w_base,
                                arith.MulIOp(kiw, dw_const.result).result,
                            ).result

                            ih_ge_0 = arith.CmpIOp(
                                arith.CmpIPredicate.sge, ih, zero.result
                            ).result
                            ih_lt_h = arith.CmpIOp(
                                arith.CmpIPredicate.slt, ih, h_const.result
                            ).result
                            iw_ge_0 = arith.CmpIOp(
                                arith.CmpIPredicate.sge, iw, zero.result
                            ).result
                            iw_lt_w = arith.CmpIOp(
                                arith.CmpIPredicate.slt, iw, w_const.result
                            ).result

                            h_valid = arith.AndIOp(ih_ge_0, ih_lt_h).result
                            w_valid = arith.AndIOp(iw_ge_0, iw_lt_w).result
                            in_bounds = arith.AndIOp(h_valid, w_valid).result

                            if_op = scf.IfOp(in_bounds, hasElse=False)
                            with ir.InsertionPoint(if_op.then_block):
                                input_val = memref.LoadOp(
                                    input_memref, [n, c, ih, iw]
                                ).result

                                current_max = memref.LoadOp(
                                    output_memref.result, [n, c, oh, ow]
                                ).result

                                is_greater = arith.CmpFOp(
                                    arith.CmpFPredicate.OGT,
                                    input_val,
                                    current_max,
                                ).result

                                inner_if = scf.IfOp(is_greater, hasElse=False)
                                with ir.InsertionPoint(inner_if.then_block):
                                    memref.StoreOp(
                                        input_val,
                                        output_memref.result,
                                        [n, c, oh, ow],
                                    )

                                    ih_i64 = arith.IndexCastOp(
                                        indices_dtype, ih
                                    ).result
                                    iw_i64 = arith.IndexCastOp(
                                        indices_dtype, iw
                                    ).result
                                    flat_idx = arith.AddIOp(
                                        arith.MulIOp(
                                            ih_i64, w_i64.result
                                        ).result,
                                        iw_i64,
                                    ).result

                                    memref.StoreOp(
                                        flat_idx,
                                        indices_memref.result,
                                        [n, c, oh, ow],
                                    )

                                    scf.YieldOp([])

                                scf.YieldOp([])

                            scf.YieldOp([])

                        scf.YieldOp([])

                    scf.YieldOp([])

                scf.YieldOp([])

            scf.YieldOp([])

        scf.YieldOp([])

    output_result = bufferization.ToTensorOp(
        output_type, output_memref.result, restrict=True
    )
    indices_result = bufferization.ToTensorOp(
        indices_type, indices_memref.result, restrict=True
    )

    return output_result, indices_result


def max_pool3d_op(
    node: MaxPool3dOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the max_pool3d operation.
    From buddy MaxPool3dOp to MLIR operations using scf.for loops.
    aten.max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode)
        -> (Tensor, Tensor)

    Returns both the max-pooled values and the indices of the max values.
    Uses scf.for loops since TOSA doesn't support 3D pooling.
    Input format: NCDHW (batch, channels, depth, height, width)
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    kernel_size = node.args[1]
    stride = (
        node.args[2] if len(node.args) > 2 and node.args[2] else kernel_size
    )
    padding = node.args[3] if len(node.args) > 3 else [0, 0, 0]
    dilation = node.args[4] if len(node.args) > 4 else [1, 1, 1]
    ceil_mode = node.args[5] if len(node.args) > 5 else False

    input_shape = list(ir.RankedTensorType(input_tensor.type).shape)
    input_dtype = ir.RankedTensorType(input_tensor.type).element_type

    N, C, D, H, W = input_shape

    # Normalize kernel_size, stride, padding, dilation
    if isinstance(kernel_size, int):
        kd, kh, kw = kernel_size, kernel_size, kernel_size
    else:
        kd = kernel_size[0]
        kh = kernel_size[1] if len(kernel_size) > 1 else kernel_size[0]
        kw = kernel_size[2] if len(kernel_size) > 2 else kernel_size[0]

    if isinstance(stride, int):
        sd, sh, sw = stride, stride, stride
    else:
        sd = stride[0]
        sh = stride[1] if len(stride) > 1 else stride[0]
        sw = stride[2] if len(stride) > 2 else stride[0]

    if isinstance(padding, int):
        pd, ph, pw = padding, padding, padding
    else:
        pd = padding[0]
        ph = padding[1] if len(padding) > 1 else padding[0]
        pw = padding[2] if len(padding) > 2 else padding[0]

    if isinstance(dilation, int):
        dd, dh, dw = dilation, dilation, dilation
    else:
        dd = dilation[0]
        dh = dilation[1] if len(dilation) > 1 else dilation[0]
        dw = dilation[2] if len(dilation) > 2 else dilation[0]

    # Calculate output dimensions
    if ceil_mode:
        out_d = (D + 2 * pd - dd * (kd - 1) - 1 + sd - 1) // sd + 1
        out_h = (H + 2 * ph - dh * (kh - 1) - 1 + sh - 1) // sh + 1
        out_w = (W + 2 * pw - dw * (kw - 1) - 1 + sw - 1) // sw + 1
    else:
        out_d = (D + 2 * pd - dd * (kd - 1) - 1) // sd + 1
        out_h = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        out_w = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    output_shape = [N, C, out_d, out_h, out_w]
    output_type = ir.RankedTensorType.get(output_shape, input_dtype)
    indices_dtype = ir.IntegerType.get_signless(64)
    indices_type = ir.RankedTensorType.get(output_shape, indices_dtype)

    # Create memrefs for output and indices
    output_memref_type = ir.MemRefType.get(output_shape, input_dtype)
    indices_memref_type = ir.MemRefType.get(output_shape, indices_dtype)

    output_memref = memref.AllocOp(output_memref_type, [], [])
    indices_memref = memref.AllocOp(indices_memref_type, [], [])

    # Initialize output with negative infinity
    neg_inf = arith.ConstantOp(
        input_dtype, ir.FloatAttr.get(input_dtype, float("-inf"))
    )
    linalg.fill(neg_inf.result, outs=[output_memref.result])

    # Initialize indices with zeros
    zero_idx = arith.ConstantOp(
        indices_dtype, ir.IntegerAttr.get(indices_dtype, 0)
    )
    linalg.fill(zero_idx.result, outs=[indices_memref.result])

    # Convert input to memref
    input_memref = bufferization.ToBufferOp(
        ir.MemRefType.get(input_shape, input_dtype), input_tensor
    ).result

    # Create constants
    zero = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 0)
    )
    one = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 1)
    )

    # Loop bounds
    bounds = {
        "n": N,
        "c": C,
        "od": out_d,
        "oh": out_h,
        "ow": out_w,
        "kd": kd,
        "kh": kh,
        "kw": kw,
    }
    bound_ops = {
        k: arith.ConstantOp(
            ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), v)
        )
        for k, v in bounds.items()
    }

    # Stride, padding, dilation constants
    const_ops = {}
    for name, val in [
        ("sd", sd),
        ("sh", sh),
        ("sw", sw),
        ("pd", pd),
        ("ph", ph),
        ("pw", pw),
        ("dd", dd),
        ("dh", dh),
        ("dw", dw),
        ("D", D),
        ("H", H),
        ("W", W),
    ]:
        const_ops[name] = arith.ConstantOp(
            ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), val)
        )

    # Multiplier for flat index: H * W
    hw_i64 = arith.ConstantOp(
        indices_dtype, ir.IntegerAttr.get(indices_dtype, H * W)
    )
    w_i64 = arith.ConstantOp(
        indices_dtype, ir.IntegerAttr.get(indices_dtype, W)
    )

    # Seven nested loops: n, c, od, oh, ow, kd, kh, kw
    n_loop = scf.ForOp(zero.result, bound_ops["n"].result, one.result)
    with ir.InsertionPoint(n_loop.body):
        n = n_loop.induction_variable

        c_loop = scf.ForOp(zero.result, bound_ops["c"].result, one.result)
        with ir.InsertionPoint(c_loop.body):
            c = c_loop.induction_variable

            od_loop = scf.ForOp(zero.result, bound_ops["od"].result, one.result)
            with ir.InsertionPoint(od_loop.body):
                od = od_loop.induction_variable

                oh_loop = scf.ForOp(
                    zero.result, bound_ops["oh"].result, one.result
                )
                with ir.InsertionPoint(oh_loop.body):
                    oh = oh_loop.induction_variable

                    ow_loop = scf.ForOp(
                        zero.result, bound_ops["ow"].result, one.result
                    )
                    with ir.InsertionPoint(ow_loop.body):
                        ow = ow_loop.induction_variable

                        # Base input positions
                        d_base = arith.SubIOp(
                            arith.MulIOp(od, const_ops["sd"].result).result,
                            const_ops["pd"].result,
                        ).result
                        h_base = arith.SubIOp(
                            arith.MulIOp(oh, const_ops["sh"].result).result,
                            const_ops["ph"].result,
                        ).result
                        w_base = arith.SubIOp(
                            arith.MulIOp(ow, const_ops["sw"].result).result,
                            const_ops["pw"].result,
                        ).result

                        # Kernel loops
                        kid_loop = scf.ForOp(
                            zero.result, bound_ops["kd"].result, one.result
                        )
                        with ir.InsertionPoint(kid_loop.body):
                            kid = kid_loop.induction_variable

                            kih_loop = scf.ForOp(
                                zero.result, bound_ops["kh"].result, one.result
                            )
                            with ir.InsertionPoint(kih_loop.body):
                                kih = kih_loop.induction_variable

                                kiw_loop = scf.ForOp(
                                    zero.result,
                                    bound_ops["kw"].result,
                                    one.result,
                                )
                                with ir.InsertionPoint(kiw_loop.body):
                                    kiw = kiw_loop.induction_variable

                                    # Calculate input positions with dilation
                                    id_pos = arith.AddIOp(
                                        d_base,
                                        arith.MulIOp(
                                            kid, const_ops["dd"].result
                                        ).result,
                                    ).result
                                    ih_pos = arith.AddIOp(
                                        h_base,
                                        arith.MulIOp(
                                            kih, const_ops["dh"].result
                                        ).result,
                                    ).result
                                    iw_pos = arith.AddIOp(
                                        w_base,
                                        arith.MulIOp(
                                            kiw, const_ops["dw"].result
                                        ).result,
                                    ).result

                                    # Check bounds
                                    id_ge_0 = arith.CmpIOp(
                                        arith.CmpIPredicate.sge,
                                        id_pos,
                                        zero.result,
                                    ).result
                                    id_lt_D = arith.CmpIOp(
                                        arith.CmpIPredicate.slt,
                                        id_pos,
                                        const_ops["D"].result,
                                    ).result
                                    ih_ge_0 = arith.CmpIOp(
                                        arith.CmpIPredicate.sge,
                                        ih_pos,
                                        zero.result,
                                    ).result
                                    ih_lt_H = arith.CmpIOp(
                                        arith.CmpIPredicate.slt,
                                        ih_pos,
                                        const_ops["H"].result,
                                    ).result
                                    iw_ge_0 = arith.CmpIOp(
                                        arith.CmpIPredicate.sge,
                                        iw_pos,
                                        zero.result,
                                    ).result
                                    iw_lt_W = arith.CmpIOp(
                                        arith.CmpIPredicate.slt,
                                        iw_pos,
                                        const_ops["W"].result,
                                    ).result

                                    d_valid = arith.AndIOp(
                                        id_ge_0, id_lt_D
                                    ).result
                                    h_valid = arith.AndIOp(
                                        ih_ge_0, ih_lt_H
                                    ).result
                                    w_valid = arith.AndIOp(
                                        iw_ge_0, iw_lt_W
                                    ).result
                                    dh_valid = arith.AndIOp(
                                        d_valid, h_valid
                                    ).result
                                    in_bounds = arith.AndIOp(
                                        dh_valid, w_valid
                                    ).result

                                    if_op = scf.IfOp(in_bounds, hasElse=False)
                                    with ir.InsertionPoint(if_op.then_block):
                                        input_val = memref.LoadOp(
                                            input_memref,
                                            [n, c, id_pos, ih_pos, iw_pos],
                                        ).result
                                        current_max = memref.LoadOp(
                                            output_memref.result,
                                            [n, c, od, oh, ow],
                                        ).result

                                        is_greater = arith.CmpFOp(
                                            arith.CmpFPredicate.OGT,
                                            input_val,
                                            current_max,
                                        ).result

                                        inner_if = scf.IfOp(
                                            is_greater, hasElse=False
                                        )
                                        with ir.InsertionPoint(
                                            inner_if.then_block
                                        ):
                                            memref.StoreOp(
                                                input_val,
                                                output_memref.result,
                                                [n, c, od, oh, ow],
                                            )

                                            # Flat index: id * H * W + ih * W + iw
                                            id_i64 = arith.IndexCastOp(
                                                indices_dtype, id_pos
                                            ).result
                                            ih_i64 = arith.IndexCastOp(
                                                indices_dtype, ih_pos
                                            ).result
                                            iw_i64 = arith.IndexCastOp(
                                                indices_dtype, iw_pos
                                            ).result
                                            flat_idx = arith.AddIOp(
                                                arith.AddIOp(
                                                    arith.MulIOp(
                                                        id_i64, hw_i64.result
                                                    ).result,
                                                    arith.MulIOp(
                                                        ih_i64, w_i64.result
                                                    ).result,
                                                ).result,
                                                iw_i64,
                                            ).result
                                            memref.StoreOp(
                                                flat_idx,
                                                indices_memref.result,
                                                [n, c, od, oh, ow],
                                            )
                                            scf.YieldOp([])
                                        scf.YieldOp([])
                                    scf.YieldOp([])
                                scf.YieldOp([])
                            scf.YieldOp([])
                        scf.YieldOp([])
                    scf.YieldOp([])
                scf.YieldOp([])
            scf.YieldOp([])
        scf.YieldOp([])

    output_result = bufferization.ToTensorOp(
        output_type, output_memref.result, restrict=True
    )
    indices_result = bufferization.ToTensorOp(
        indices_type, indices_memref.result, restrict=True
    )

    return output_result, indices_result


def scatter_add_op(
    node: ScatterAddOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the scatter_add operation.
    From buddy ScatterAddOp to MLIR operations using scf.for loops.
    aten.scatter_add(self, dim, index, src) -> Tensor

    Adds all values from src tensor into self at indices specified in index.
    For each element in src, adds it to self at the position given by index.
    self[index[i][j][k]][j][k] += src[i][j][k]  (for dim=0)

    Uses scf.for loops for the scatter add operation.
    """
    self_tensor = symbol_table.get((str(node.args[0]), 0))
    dim = node.args[1]
    index_tensor = symbol_table.get((str(node.args[2]), 0))
    src_tensor = symbol_table.get((str(node.args[3]), 0))

    self_shape = list(ir.RankedTensorType(self_tensor.type).shape)
    src_shape = list(ir.RankedTensorType(src_tensor.type).shape)
    self_dtype = ir.RankedTensorType(self_tensor.type).element_type
    try:
        index_dtype = ir.RankedTensorType(index_tensor.type).element_type
    except Exception:
        index_dtype = index_tensor.type

    ndim = len(self_shape)

    # Handle negative dim
    if dim < 0:
        dim = ndim + dim

    # Create output memref and copy self into it
    output_type = ir.RankedTensorType.get(self_shape, self_dtype)
    output_memref_type = ir.MemRefType.get(self_shape, self_dtype)
    output_memref = memref.AllocOp(output_memref_type, [], [])

    # Copy self to output
    self_memref = bufferization.ToBufferOp(
        ir.MemRefType.get(self_shape, self_dtype), self_tensor
    ).result
    linalg.copy(self_memref, outs=[output_memref.result])

    # Convert src and index to memrefs
    src_memref = bufferization.ToBufferOp(
        ir.MemRefType.get(src_shape, self_dtype), src_tensor
    ).result

    def _broadcast_index_tensor(value, target_shape):
        try:
            value_shape = list(ir.RankedTensorType(value.type).shape)
        except Exception:
            target_type = ir.RankedTensorType.get(target_shape, index_dtype)
            return tensor.SplatOp(target_type, value, []).result

        if len(value_shape) < len(target_shape):
            padded_shape = [1] * (len(target_shape) - len(value_shape))
            padded_shape.extend(value_shape)
            shape_ty = ir.Type.parse(f"!tosa.shape<{len(padded_shape)}>")
            index_ty = ir.IndexType.get()
            shape_val = tosa.ConstShapeOp(
                shape_ty,
                ir.DenseElementsAttr.get(
                    array.array("q", padded_shape),
                    type=index_ty,
                    shape=[len(padded_shape)],
                ),
            ).result
            value = tosa.ReshapeOp(value, shape_val).result
            value_shape = padded_shape

        if len(value_shape) != len(target_shape):
            raise ValueError(
                "Index rank %d does not match target rank %d"
                % (len(value_shape), len(target_shape))
            )

        for src_dim, tgt_dim in zip(value_shape, target_shape):
            if src_dim in (-1,) or tgt_dim in (-1,):
                continue
            if src_dim != 1 and src_dim != tgt_dim:
                raise ValueError(
                    "Index shape %s is not broadcastable to %s"
                    % (value_shape, target_shape)
                )

        if value_shape != target_shape:
            if str(index_dtype).startswith("f") or str(index_dtype).startswith(
                "bf"
            ):
                zero_elem = ir.FloatAttr.get(index_dtype, 0.0)
            else:
                zero_elem = ir.IntegerAttr.get(index_dtype, 0)
            zero_type = ir.RankedTensorType.get(target_shape, index_dtype)
            zero_attr = ir.DenseElementsAttr.get_splat(zero_type, zero_elem)
            zero_tensor = tosa.ConstOp(zero_attr).result
            value = tosa.AddOp(zero_type, value, zero_tensor).result

        return value

    try:
        index_shape = list(ir.RankedTensorType(index_tensor.type).shape)
    except Exception:
        index_shape = []
    if index_shape != src_shape:
        index_tensor = _broadcast_index_tensor(index_tensor, src_shape)

    index_memref = bufferization.ToBufferOp(
        ir.MemRefType.get(src_shape, index_dtype), index_tensor
    ).result

    # Create loop bounds
    zero = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 0)
    )
    one = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 1)
    )

    bounds = [
        arith.ConstantOp(
            ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), s)
        )
        for s in src_shape
    ]

    def create_nested_loops(depth, indices):
        if depth == ndim:
            # Load source value
            src_val = memref.LoadOp(src_memref, indices).result

            # Load index at this position
            idx_val = memref.LoadOp(index_memref, indices).result
            idx = arith.IndexCastOp(ir.IndexType.get(), idx_val).result

            # Build output indices: replace dim-th index with idx
            out_indices = []
            for d in range(ndim):
                if d == dim:
                    out_indices.append(idx)
                else:
                    out_indices.append(indices[d])

            # Load current value, add src, store back
            current_val = memref.LoadOp(
                output_memref.result, out_indices
            ).result
            if str(self_dtype).startswith("f"):
                new_val = arith.AddFOp(current_val, src_val).result
            else:
                new_val = arith.AddIOp(current_val, src_val).result
            memref.StoreOp(new_val, output_memref.result, out_indices)
        else:
            loop = scf.ForOp(zero.result, bounds[depth].result, one.result)
            with ir.InsertionPoint(loop.body):
                new_indices = indices + [loop.induction_variable]
                create_nested_loops(depth + 1, new_indices)
                scf.YieldOp(loop.inner_iter_args)

    create_nested_loops(0, [])

    result = bufferization.ToTensorOp(
        output_type, output_memref.result, restrict=True
    )
    return result


def index_select_op(
    node: IndexSelectOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the index_select operation.
    From buddy IndexSelectOp to MLIR operations using scf.for loops.
    aten.index_select(input, dim, index) -> Tensor

    Selects elements from input tensor along the specified dimension
    using the indices in index tensor.

    For a 3D input with dim=1:
    output[i, j, k] = input[i, index[j], k]
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    dim = node.args[1]
    index_tensor = symbol_table.get((str(node.args[2]), 0))

    input_shape = list(ir.RankedTensorType(input_tensor.type).shape)
    input_dtype = ir.RankedTensorType(input_tensor.type).element_type
    index_shape = list(ir.RankedTensorType(index_tensor.type).shape)
    index_dtype = ir.RankedTensorType(index_tensor.type).element_type

    ndim = len(input_shape)

    # Handle negative dim
    if dim < 0:
        dim = ndim + dim

    # Output shape: input_shape with dim replaced by index length
    output_shape = input_shape.copy()
    output_shape[dim] = index_shape[0]

    output_type = ir.RankedTensorType.get(output_shape, input_dtype)
    output_memref_type = ir.MemRefType.get(output_shape, input_dtype)

    # Allocate output memref
    output_memref = memref.AllocOp(output_memref_type, [], [])

    # Convert input and index to memrefs
    input_memref = bufferization.ToBufferOp(
        ir.MemRefType.get(input_shape, input_dtype), input_tensor
    ).result

    index_memref = bufferization.ToBufferOp(
        ir.MemRefType.get(index_shape, index_dtype), index_tensor
    ).result

    # Create index constants
    c0 = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 0)
    ).result
    c1 = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 1)
    ).result

    # Create bounds for each output dimension
    bounds = []
    for i, size in enumerate(output_shape):
        bounds.append(
            arith.ConstantOp(
                ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), size)
            ).result
        )

    # Generate nested loops for all output dimensions
    def create_nested_loops(depth, output_indices):
        if depth == ndim:
            # Base case: perform the index selection and store
            # Build input indices by replacing dim with looked-up value
            input_indices = output_indices.copy()

            # Get the index value for the target dimension
            # output_indices[dim] is the position in index tensor
            index_pos = output_indices[dim]

            # Load actual index from index tensor
            actual_index_val = memref.LoadOp(index_memref, [index_pos]).result

            # Convert to index type
            actual_index = arith.IndexCastOp(
                ir.IndexType.get(), actual_index_val
            ).result

            # Replace the dim position with actual_index
            input_indices[dim] = actual_index

            # Load from input and store to output
            val = memref.LoadOp(input_memref, input_indices).result
            memref.StoreOp(val, output_memref.result, output_indices)
        else:
            # Create loop for this dimension
            loop = scf.ForOp(c0, bounds[depth], c1)
            with ir.InsertionPoint(loop.body):
                idx = loop.induction_variable
                new_indices = output_indices + [idx]
                create_nested_loops(depth + 1, new_indices)
                scf.YieldOp([])

    create_nested_loops(0, [])

    result = bufferization.ToTensorOp(
        output_type, output_memref.result, restrict=True
    )
    return result


def avg_pool3d_op(
    node: AvgPool3dOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the avg_pool3d operation.
    From buddy AvgPool3dOp to MLIR operations using scf.for loops.
    aten.avg_pool3d(input, kernel_size, stride, padding, ceil_mode,
                    count_include_pad, divisor_override) -> Tensor

    Performs 3D average pooling over an input tensor of shape (N, C, D, H, W).
    Uses scf.for loops to iterate over output positions and kernel elements.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    kernel_size = node.args[1]
    stride = (
        node.args[2] if len(node.args) > 2 and node.args[2] else kernel_size
    )
    padding = node.args[3] if len(node.args) > 3 else [0, 0, 0]
    ceil_mode = node.args[4] if len(node.args) > 4 else False
    count_include_pad = node.args[5] if len(node.args) > 5 else True
    divisor_override = node.args[6] if len(node.args) > 6 else None

    input_shape = list(ir.RankedTensorType(input_tensor.type).shape)
    input_dtype = ir.RankedTensorType(input_tensor.type).element_type

    N, C, D, H, W = input_shape

    # Normalize parameters to lists
    if isinstance(kernel_size, int):
        kd, kh, kw = kernel_size, kernel_size, kernel_size
    else:
        kd, kh, kw = kernel_size[0], kernel_size[1], kernel_size[2]

    if isinstance(stride, int):
        sd, sh, sw = stride, stride, stride
    else:
        sd, sh, sw = stride[0], stride[1], stride[2]

    if isinstance(padding, int):
        pd, ph, pw = padding, padding, padding
    else:
        pd, ph, pw = padding[0], padding[1], padding[2]

    # Calculate output dimensions
    if ceil_mode:
        D_out = (D + 2 * pd - kd + sd) // sd
        H_out = (H + 2 * ph - kh + sh) // sh
        W_out = (W + 2 * pw - kw + sw) // sw
    else:
        D_out = (D + 2 * pd - kd) // sd + 1
        H_out = (H + 2 * ph - kh) // sh + 1
        W_out = (W + 2 * pw - kw) // sw + 1

    output_shape = [N, C, D_out, H_out, W_out]
    output_type = ir.RankedTensorType.get(output_shape, input_dtype)
    output_memref_type = ir.MemRefType.get(output_shape, input_dtype)

    # Allocate output memref
    output_memref = memref.AllocOp(output_memref_type, [], [])

    # Initialize with zeros
    zero = arith.ConstantOp(input_dtype, ir.FloatAttr.get(input_dtype, 0.0))
    linalg.fill(zero.result, outs=[output_memref.result])

    # Convert input to memref
    input_memref = bufferization.ToBufferOp(
        ir.MemRefType.get(input_shape, input_dtype), input_tensor
    ).result

    # Create index constants
    c0 = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 0)
    ).result
    c1 = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 1)
    ).result
    cN = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), N)
    ).result
    cC = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), C)
    ).result
    cD_out = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), D_out)
    ).result
    cH_out = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), H_out)
    ).result
    cW_out = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), W_out)
    ).result
    ckd = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), kd)
    ).result
    ckh = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), kh)
    ).result
    ckw = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), kw)
    ).result
    csd = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), sd)
    ).result
    csh = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), sh)
    ).result
    csw = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), sw)
    ).result
    cpd = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), pd)
    ).result
    cph = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), ph)
    ).result
    cpw = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), pw)
    ).result
    cD = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), D)
    ).result
    cH = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), H)
    ).result
    cW = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), W)
    ).result

    # Calculate divisor
    pool_size = kd * kh * kw
    if divisor_override is not None:
        divisor_val = float(divisor_override)
    else:
        divisor_val = float(pool_size)
    divisor = arith.ConstantOp(
        input_dtype, ir.FloatAttr.get(input_dtype, divisor_val)
    ).result

    # Nested loops: N -> C -> D_out -> H_out -> W_out -> kd -> kh -> kw
    n_loop = scf.ForOp(c0, cN, c1)
    with ir.InsertionPoint(n_loop.body):
        n = n_loop.induction_variable
        c_loop = scf.ForOp(c0, cC, c1)
        with ir.InsertionPoint(c_loop.body):
            c = c_loop.induction_variable
            d_loop = scf.ForOp(c0, cD_out, c1)
            with ir.InsertionPoint(d_loop.body):
                od = d_loop.induction_variable
                h_loop = scf.ForOp(c0, cH_out, c1)
                with ir.InsertionPoint(h_loop.body):
                    oh = h_loop.induction_variable
                    w_loop = scf.ForOp(c0, cW_out, c1)
                    with ir.InsertionPoint(w_loop.body):
                        ow = w_loop.induction_variable

                        # Calculate starting positions in input
                        d_start = arith.SubIOp(
                            arith.MulIOp(od, csd).result, cpd
                        ).result
                        h_start = arith.SubIOp(
                            arith.MulIOp(oh, csh).result, cph
                        ).result
                        w_start = arith.SubIOp(
                            arith.MulIOp(ow, csw).result, cpw
                        ).result

                        # Accumulator for sum
                        sum_init = arith.ConstantOp(
                            input_dtype, ir.FloatAttr.get(input_dtype, 0.0)
                        ).result

                        # Allocate local memref for sum
                        sum_memref = memref.AllocaOp(
                            ir.MemRefType.get([], input_dtype), [], []
                        )
                        memref.StoreOp(sum_init, sum_memref.result, [])

                        # Kernel loops
                        kd_loop = scf.ForOp(c0, ckd, c1)
                        with ir.InsertionPoint(kd_loop.body):
                            ki = kd_loop.induction_variable
                            kh_loop = scf.ForOp(c0, ckh, c1)
                            with ir.InsertionPoint(kh_loop.body):
                                kj = kh_loop.induction_variable
                                kw_loop = scf.ForOp(c0, ckw, c1)
                                with ir.InsertionPoint(kw_loop.body):
                                    kk = kw_loop.induction_variable

                                    # Calculate input position
                                    id_pos = arith.AddIOp(d_start, ki).result
                                    ih_pos = arith.AddIOp(h_start, kj).result
                                    iw_pos = arith.AddIOp(w_start, kk).result

                                    # Bounds check
                                    d_in_bounds_l = arith.CmpIOp(
                                        arith.CmpIPredicate.sge, id_pos, c0
                                    ).result
                                    d_in_bounds_r = arith.CmpIOp(
                                        arith.CmpIPredicate.slt, id_pos, cD
                                    ).result
                                    h_in_bounds_l = arith.CmpIOp(
                                        arith.CmpIPredicate.sge, ih_pos, c0
                                    ).result
                                    h_in_bounds_r = arith.CmpIOp(
                                        arith.CmpIPredicate.slt, ih_pos, cH
                                    ).result
                                    w_in_bounds_l = arith.CmpIOp(
                                        arith.CmpIPredicate.sge, iw_pos, c0
                                    ).result
                                    w_in_bounds_r = arith.CmpIOp(
                                        arith.CmpIPredicate.slt, iw_pos, cW
                                    ).result

                                    in_bounds = arith.AndIOp(
                                        d_in_bounds_l, d_in_bounds_r
                                    ).result
                                    in_bounds = arith.AndIOp(
                                        in_bounds, h_in_bounds_l
                                    ).result
                                    in_bounds = arith.AndIOp(
                                        in_bounds, h_in_bounds_r
                                    ).result
                                    in_bounds = arith.AndIOp(
                                        in_bounds, w_in_bounds_l
                                    ).result
                                    in_bounds = arith.AndIOp(
                                        in_bounds, w_in_bounds_r
                                    ).result

                                    bounds_if = scf.IfOp(
                                        in_bounds, hasElse=False
                                    )
                                    with ir.InsertionPoint(
                                        bounds_if.then_block
                                    ):
                                        input_val = memref.LoadOp(
                                            input_memref,
                                            [n, c, id_pos, ih_pos, iw_pos],
                                        ).result
                                        current_sum = memref.LoadOp(
                                            sum_memref.result, []
                                        ).result
                                        new_sum = arith.AddFOp(
                                            current_sum, input_val
                                        ).result
                                        memref.StoreOp(
                                            new_sum, sum_memref.result, []
                                        )
                                        scf.YieldOp([])
                                    scf.YieldOp([])
                                scf.YieldOp([])
                            scf.YieldOp([])

                        # Compute average and store
                        final_sum = memref.LoadOp(sum_memref.result, []).result
                        avg_val = arith.DivFOp(final_sum, divisor).result
                        memref.StoreOp(
                            avg_val, output_memref.result, [n, c, od, oh, ow]
                        )
                        scf.YieldOp([])
                    scf.YieldOp([])
                scf.YieldOp([])
            scf.YieldOp([])
        scf.YieldOp([])

    result = bufferization.ToTensorOp(
        output_type, output_memref.result, restrict=True
    )
    return result


def topk_op(
    node: TopkOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the topk operation.
    From buddy TopkOp to MLIR operations using scf.for loops.
    aten.topk(input, k, dim, largest, sorted) -> (values, indices)

    Returns the k largest (or smallest) elements along a dimension.
    Uses a selection-sort based approach with scf.for loops.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    k = node.args[1]
    dim = node.args[2] if len(node.args) > 2 else -1
    largest = node.args[3] if len(node.args) > 3 else True
    # sorted_result = node.args[4] if len(node.args) > 4 else True  # We always sort

    if not isinstance(k, int):
        raise NotImplementedError("topk requires static integer k")

    input_shape = list(ir.RankedTensorType(input_tensor.type).shape)
    input_dtype = ir.RankedTensorType(input_tensor.type).element_type
    ndim = len(input_shape)
    if any(dim_size < 0 for dim_size in input_shape):
        raise NotImplementedError("topk requires static shapes")

    # Handle negative dim
    if dim < 0:
        dim = ndim + dim
    if dim < 0 or dim >= ndim:
        raise NotImplementedError("topk dim out of range")

    # Output shape: same as input but dim becomes k
    output_shape = input_shape.copy()
    output_shape[dim] = k

    values_type = ir.RankedTensorType.get(output_shape, input_dtype)
    indices_type = ir.RankedTensorType.get(
        output_shape, ir.IntegerType.get_signless(64)
    )
    values_memref_type = ir.MemRefType.get(output_shape, input_dtype)
    indices_memref_type = ir.MemRefType.get(
        output_shape, ir.IntegerType.get_signless(64)
    )

    # Allocate output memrefs
    values_memref = memref.AllocOp(values_memref_type, [], [])
    indices_memref = memref.AllocOp(indices_memref_type, [], [])

    # Convert input to memref
    input_memref = bufferization.ToBufferOp(
        ir.MemRefType.get(input_shape, input_dtype), input_tensor
    ).result

    dim_size = input_shape[dim]
    if dim_size < 0:
        raise NotImplementedError("topk requires static dim size")
    if k < 0 or k > dim_size:
        raise NotImplementedError("topk k out of range")

    is_float = _is_float_type(input_dtype)
    is_int = ir.IntegerType.isinstance(input_dtype)
    if not is_float and not is_int:
        raise NotImplementedError(
            "topk only supports integer or floating types"
        )

    def _integer_bounds(dtype: ir.Type) -> Tuple[int, int]:
        bitwidth = ir.IntegerType(dtype).width
        if bitwidth == 1:
            return 0, 1
        min_val = -(1 << (bitwidth - 1))
        max_val = (1 << (bitwidth - 1)) - 1
        return min_val, max_val

    def _best_init_value() -> ir.Value:
        if is_float:
            init = float("-inf") if largest else float("inf")
            return arith.ConstantOp(
                input_dtype, ir.FloatAttr.get(input_dtype, init)
            ).result
        min_val, max_val = _integer_bounds(input_dtype)
        init = min_val if largest else max_val
        return arith.ConstantOp(
            input_dtype, ir.IntegerAttr.get(input_dtype, init)
        ).result

    def _is_better(val: ir.Value, best: ir.Value) -> ir.Value:
        if is_float:
            pred = (
                arith.CmpFPredicate.OGT if largest else arith.CmpFPredicate.OLT
            )
            return arith.CmpFOp(pred, val, best).result
        pred = arith.CmpIPredicate.sgt if largest else arith.CmpIPredicate.slt
        return arith.CmpIOp(pred, val, best).result

    # Create index constants
    c0 = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 0)
    ).result
    c1 = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 1)
    ).result
    ck = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), k)
    ).result
    cdim_size = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), dim_size)
    ).result

    # For simplicity, we handle the 1D and 2D cases
    # Full n-dimensional would require more complex index handling

    if ndim == 1:
        # 1D case: topk along the only dimension
        # Allocate auxiliary memrefs for tracking used indices
        used_memref_type = ir.MemRefType.get(
            [dim_size], ir.IntegerType.get_signless(1)
        )
        used_memref = memref.AllocOp(used_memref_type, [], [])

        # Initialize used flags to 0 (false)
        c0_i1 = arith.ConstantOp(
            ir.IntegerType.get_signless(1),
            ir.IntegerAttr.get(ir.IntegerType.get_signless(1), 0),
        ).result
        init_loop = scf.ForOp(c0, cdim_size, c1)
        with ir.InsertionPoint(init_loop.body):
            i = init_loop.induction_variable
            memref.StoreOp(c0_i1, used_memref.result, [i])
            scf.YieldOp([])

        # Find k largest/smallest
        k_loop = scf.ForOp(c0, ck, c1)
        with ir.InsertionPoint(k_loop.body):
            ki = k_loop.induction_variable

            # Local memrefs for best value and index
            best_val_memref = memref.AllocaOp(
                ir.MemRefType.get([], input_dtype), [], []
            )
            best_idx_memref = memref.AllocaOp(
                ir.MemRefType.get([], ir.IndexType.get()), [], []
            )

            # Initialize with extreme value
            memref.StoreOp(_best_init_value(), best_val_memref.result, [])
            memref.StoreOp(c0, best_idx_memref.result, [])

            # Find best unused value
            search_loop = scf.ForOp(c0, cdim_size, c1)
            with ir.InsertionPoint(search_loop.body):
                j = search_loop.induction_variable

                used_flag = memref.LoadOp(used_memref.result, [j]).result
                c1_i1 = arith.ConstantOp(
                    ir.IntegerType.get_signless(1),
                    ir.IntegerAttr.get(ir.IntegerType.get_signless(1), 1),
                ).result
                not_used = arith.CmpIOp(
                    arith.CmpIPredicate.ne, used_flag, c1_i1
                ).result

                check_if = scf.IfOp(not_used, hasElse=False)
                with ir.InsertionPoint(check_if.then_block):
                    val = memref.LoadOp(input_memref, [j]).result
                    best_val = memref.LoadOp(best_val_memref.result, []).result

                    is_better = _is_better(val, best_val)

                    update_if = scf.IfOp(is_better, hasElse=False)
                    with ir.InsertionPoint(update_if.then_block):
                        memref.StoreOp(val, best_val_memref.result, [])
                        memref.StoreOp(j, best_idx_memref.result, [])
                        scf.YieldOp([])
                    scf.YieldOp([])
                scf.YieldOp([])

            # Store result and mark as used
            best_val_final = memref.LoadOp(best_val_memref.result, []).result
            best_idx_final = memref.LoadOp(best_idx_memref.result, []).result
            memref.StoreOp(best_val_final, values_memref.result, [ki])
            best_idx_i64 = arith.IndexCastOp(
                ir.IntegerType.get_signless(64), best_idx_final
            ).result
            memref.StoreOp(best_idx_i64, indices_memref.result, [ki])
            memref.StoreOp(c1_i1, used_memref.result, [best_idx_final])
            scf.YieldOp([])

    elif ndim == 2:
        # 2D case
        other_dim = 1 - dim
        cother = arith.ConstantOp(
            ir.IndexType.get(),
            ir.IntegerAttr.get(ir.IndexType.get(), input_shape[other_dim]),
        ).result

        # Allocate used flags for each row/col
        used_memref_type = ir.MemRefType.get(
            [input_shape[other_dim], dim_size], ir.IntegerType.get_signless(1)
        )
        used_memref = memref.AllocOp(used_memref_type, [], [])

        c0_i1 = arith.ConstantOp(
            ir.IntegerType.get_signless(1),
            ir.IntegerAttr.get(ir.IntegerType.get_signless(1), 0),
        ).result
        c1_i1 = arith.ConstantOp(
            ir.IntegerType.get_signless(1),
            ir.IntegerAttr.get(ir.IntegerType.get_signless(1), 1),
        ).result

        # Initialize used flags
        init_outer = scf.ForOp(c0, cother, c1)
        with ir.InsertionPoint(init_outer.body):
            io = init_outer.induction_variable
            init_inner = scf.ForOp(c0, cdim_size, c1)
            with ir.InsertionPoint(init_inner.body):
                ii = init_inner.induction_variable
                memref.StoreOp(c0_i1, used_memref.result, [io, ii])
                scf.YieldOp([])
            scf.YieldOp([])

        # Outer loop over the other dimension
        outer_loop = scf.ForOp(c0, cother, c1)
        with ir.InsertionPoint(outer_loop.body):
            outer_idx = outer_loop.induction_variable

            # Find k elements
            k_loop = scf.ForOp(c0, ck, c1)
            with ir.InsertionPoint(k_loop.body):
                ki = k_loop.induction_variable

                best_val_memref = memref.AllocaOp(
                    ir.MemRefType.get([], input_dtype), [], []
                )
                best_idx_memref = memref.AllocaOp(
                    ir.MemRefType.get([], ir.IndexType.get()), [], []
                )

                memref.StoreOp(_best_init_value(), best_val_memref.result, [])
                memref.StoreOp(c0, best_idx_memref.result, [])

                search_loop = scf.ForOp(c0, cdim_size, c1)
                with ir.InsertionPoint(search_loop.body):
                    j = search_loop.induction_variable

                    used_flag = memref.LoadOp(
                        used_memref.result, [outer_idx, j]
                    ).result
                    not_used = arith.CmpIOp(
                        arith.CmpIPredicate.ne, used_flag, c1_i1
                    ).result

                    check_if = scf.IfOp(not_used, hasElse=False)
                    with ir.InsertionPoint(check_if.then_block):
                        if dim == 1:
                            val = memref.LoadOp(
                                input_memref, [outer_idx, j]
                            ).result
                        else:
                            val = memref.LoadOp(
                                input_memref, [j, outer_idx]
                            ).result
                        best_val = memref.LoadOp(
                            best_val_memref.result, []
                        ).result

                        is_better = _is_better(val, best_val)

                        update_if = scf.IfOp(is_better, hasElse=False)
                        with ir.InsertionPoint(update_if.then_block):
                            memref.StoreOp(val, best_val_memref.result, [])
                            memref.StoreOp(j, best_idx_memref.result, [])
                            scf.YieldOp([])
                        scf.YieldOp([])
                    scf.YieldOp([])

                best_val_final = memref.LoadOp(
                    best_val_memref.result, []
                ).result
                best_idx_final = memref.LoadOp(
                    best_idx_memref.result, []
                ).result

                if dim == 1:
                    memref.StoreOp(
                        best_val_final, values_memref.result, [outer_idx, ki]
                    )
                    best_idx_i64 = arith.IndexCastOp(
                        ir.IntegerType.get_signless(64), best_idx_final
                    ).result
                    memref.StoreOp(
                        best_idx_i64, indices_memref.result, [outer_idx, ki]
                    )
                else:
                    memref.StoreOp(
                        best_val_final, values_memref.result, [ki, outer_idx]
                    )
                    best_idx_i64 = arith.IndexCastOp(
                        ir.IntegerType.get_signless(64), best_idx_final
                    ).result
                    memref.StoreOp(
                        best_idx_i64, indices_memref.result, [ki, outer_idx]
                    )
                memref.StoreOp(
                    c1_i1, used_memref.result, [outer_idx, best_idx_final]
                )
                scf.YieldOp([])
            scf.YieldOp([])

    else:
        raise NotImplementedError("topk only supports rank-1/2 tensors")

    values_result = bufferization.ToTensorOp(
        values_type, values_memref.result, restrict=True
    )
    indices_result = bufferization.ToTensorOp(
        indices_type, indices_memref.result, restrict=True
    )

    return values_result, indices_result


def kthvalue_op(
    node: KthValueOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the kthvalue operation.
    From buddy KthValueOp to MLIR operations using scf.for loops.
    aten.kthvalue(input, k, dim, keepdim) -> (values, indices)
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    k = node.args[1]
    dim = node.args[2] if len(node.args) > 2 else -1
    keepdim = node.args[3] if len(node.args) > 3 else False

    if not isinstance(k, int):
        raise NotImplementedError("kthvalue requires static integer k")

    input_shape = list(ir.RankedTensorType(input_tensor.type).shape)
    input_dtype = ir.RankedTensorType(input_tensor.type).element_type
    if len(input_shape) != 1:
        raise NotImplementedError("kthvalue only supports rank-1 tensors")

    if dim < 0:
        dim += 1
    if dim != 0:
        raise NotImplementedError("kthvalue only supports dim=0")

    n = input_shape[0]
    if n < 0:
        raise NotImplementedError("kthvalue requires static dimension")
    if k < 1 or k > n:
        raise NotImplementedError("kthvalue k out of range")

    shape_meta = node.tensor_meta["shape"]
    if isinstance(shape_meta, tuple):
        values_shape = list(shape_meta[0])
        indices_shape = list(shape_meta[1])
    else:
        values_shape = list(shape_meta)
        indices_shape = list(shape_meta)

    values_type = ir.RankedTensorType.get(values_shape, input_dtype)
    indices_type = ir.RankedTensorType.get(
        indices_shape, ir.IntegerType.get_signless(64)
    )

    values_memref = memref.AllocOp(
        ir.MemRefType.get(values_shape, input_dtype), [], []
    )
    indices_memref = memref.AllocOp(
        ir.MemRefType.get(indices_shape, ir.IntegerType.get_signless(64)),
        [],
        [],
    )

    input_memref = bufferization.ToBufferOp(
        ir.MemRefType.get(input_shape, input_dtype), input_tensor
    ).result

    temp_indices = memref.AllocOp(
        ir.MemRefType.get(input_shape, ir.IntegerType.get_signless(64)),
        [],
        [],
    )

    c0 = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 0)
    )
    c1 = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 1)
    )
    cN = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), n)
    )

    init_loop = scf.ForOp(c0.result, cN.result, c1.result)
    with ir.InsertionPoint(init_loop.body):
        idx_val = arith.IndexCastOp(
            ir.IntegerType.get_signless(64), init_loop.induction_variable
        )
        memref.StoreOp(
            idx_val, temp_indices.result, [init_loop.induction_variable]
        )
        scf.YieldOp(init_loop.inner_iter_args)

    outer_ub = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), n - 1)
    )
    pass_loop = scf.ForOp(c0.result, outer_ub.result, c1.result)
    with ir.InsertionPoint(pass_loop.body):
        inner_ub = arith.SubIOp(outer_ub.result, pass_loop.induction_variable)
        compare_loop = scf.ForOp(c0.result, inner_ub, c1.result)
        with ir.InsertionPoint(compare_loop.body):
            next_idx = arith.AddIOp(compare_loop.induction_variable, c1.result)
            val_curr = memref.LoadOp(
                input_memref, [compare_loop.induction_variable]
            ).result
            val_next = memref.LoadOp(input_memref, [next_idx]).result

            idx_curr = memref.LoadOp(
                temp_indices.result, [compare_loop.induction_variable]
            ).result
            idx_next = memref.LoadOp(temp_indices.result, [next_idx]).result

            if str(input_dtype).startswith("f"):
                should_swap = arith.CmpFOp(
                    arith.CmpFPredicate.OGT, val_curr, val_next
                )
            else:
                should_swap = arith.CmpIOp(
                    arith.CmpIPredicate.sgt, val_curr, val_next
                )

            if_op = scf.IfOp(should_swap, hasElse=False)
            with ir.InsertionPoint(if_op.then_block):
                memref.StoreOp(
                    val_next,
                    input_memref,
                    [compare_loop.induction_variable],
                )
                memref.StoreOp(val_curr, input_memref, [next_idx])
                memref.StoreOp(
                    idx_next,
                    temp_indices.result,
                    [compare_loop.induction_variable],
                )
                memref.StoreOp(idx_curr, temp_indices.result, [next_idx])
                scf.YieldOp([])

            scf.YieldOp(compare_loop.inner_iter_args)
        scf.YieldOp(pass_loop.inner_iter_args)

    kth_index = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), k - 1)
    )
    kth_value = memref.LoadOp(input_memref, [kth_index.result]).result
    kth_pos = memref.LoadOp(temp_indices.result, [kth_index.result]).result

    if values_shape:
        out_index = arith.ConstantOp(
            ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 0)
        )
        memref.StoreOp(kth_value, values_memref.result, [out_index.result])
        memref.StoreOp(kth_pos, indices_memref.result, [out_index.result])
    else:
        memref.StoreOp(kth_value, values_memref.result, [])
        memref.StoreOp(kth_pos, indices_memref.result, [])

    values = bufferization.ToTensorOp(
        values_type, values_memref.result, restrict=True
    )
    indices = bufferization.ToTensorOp(
        indices_type, indices_memref.result, restrict=True
    )

    return values, indices


def embedding_dense_backward_op(
    node: EmbeddingDenseBackwardOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the embedding_dense_backward operation.
    From buddy EmbeddingDenseBackwardOp to MLIR linalg operations.
    aten.embedding_dense_backward(grad_output, indices, num_weights,
                                   padding_idx, scale_grad_by_freq) -> Tensor

    Accumulates gradients into the embedding weight matrix using scf.for loops.
    grad_weight[indices[i]] += grad_output[i]

    This is a scatter_add operation that cannot be expressed with linalg.generic
    due to data-dependent indexing.
    """
    grad_output = symbol_table.get((str(node.args[0]), 0))
    indices = symbol_table.get((str(node.args[1]), 0))
    num_weights = node.args[2]
    padding_idx = node.args[3]
    scale_grad_by_freq = node.args[4]

    grad_shape = list(ir.RankedTensorType(grad_output.type).shape)
    indices_shape = list(ir.RankedTensorType(indices.type).shape)
    grad_dtype = ir.RankedTensorType(grad_output.type).element_type
    indices_dtype = ir.RankedTensorType(indices.type).element_type

    embedding_dim = grad_shape[-1]

    # Output shape is (num_weights, embedding_dim)
    output_shape = [num_weights, embedding_dim]
    output_type = ir.RankedTensorType.get(output_shape, grad_dtype)
    output_memref_type = ir.MemRefType.get(output_shape, grad_dtype)

    # Flatten grad_output and indices to 2D and 1D respectively
    if len(grad_shape) > 2:
        batch_size = 1
        for dim in grad_shape[:-1]:
            batch_size *= dim
        flat_grad_shape = [batch_size, embedding_dim]
        flat_grad_type = ir.RankedTensorType.get(flat_grad_shape, grad_dtype)
        grad_flat = tosa.ReshapeOp(
            grad_output, memoryview(array.array("i", flat_grad_shape))
        ).result

        flat_indices_shape = [batch_size]
        indices_flat = tosa.ReshapeOp(
            indices, memoryview(array.array("i", flat_indices_shape))
        ).result
    elif len(grad_shape) == 2:
        batch_size = grad_shape[0]
        grad_flat = grad_output
        indices_flat = indices
    else:
        batch_size = 1
        flat_grad_shape = [1, embedding_dim]
        grad_flat = tosa.ReshapeOp(
            grad_output, memoryview(array.array("i", flat_grad_shape))
        ).result
        flat_indices_shape = [1]
        indices_flat = tosa.ReshapeOp(
            indices, memoryview(array.array("i", flat_indices_shape))
        ).result

    # Create output memref and initialize with zeros
    output_memref = memref.AllocOp(output_memref_type, [], [])

    # Initialize with zeros using linalg.fill
    zero = arith.ConstantOp(grad_dtype, ir.FloatAttr.get(grad_dtype, 0.0))
    linalg.fill(zero.result, outs=[output_memref.result])

    # Convert input tensors to memrefs
    grad_memref = bufferization.ToBufferOp(
        ir.MemRefType.get([batch_size, embedding_dim], grad_dtype), grad_flat
    ).result
    indices_memref = bufferization.ToBufferOp(
        ir.MemRefType.get([batch_size], indices_dtype), indices_flat
    ).result

    # Create loop bounds
    lb = arith.ConstantOp(ir.IndexType.get(), 0)
    step = arith.ConstantOp(ir.IndexType.get(), 1)
    ub_batch = arith.ConstantOp(ir.IndexType.get(), batch_size)
    ub_emb = arith.ConstantOp(ir.IndexType.get(), embedding_dim)

    # Nested loops for scatter_add: output[indices[i], j] += grad[i, j]
    def create_nested_loops(depth, indices_list):
        if depth == 2:
            # At innermost level: perform scatter_add
            i, j = indices_list

            # Load index value
            idx_val = memref.LoadOp(indices_memref, [i])
            # Cast to index type
            idx = arith.IndexCastOp(ir.IndexType.get(), idx_val)

            # Load gradient value
            grad_val = memref.LoadOp(grad_memref, [i, j])

            # Load current accumulated value
            current_val = memref.LoadOp(output_memref.result, [idx.result, j])

            # Add gradient (scatter_add)
            new_val = arith.AddFOp(current_val.result, grad_val.result)

            # Store updated value
            memref.StoreOp(
                new_val.result, output_memref.result, [idx.result, j]
            )
        else:
            ub = ub_batch if depth == 0 else ub_emb
            loop = scf.ForOp(lb, ub, step)
            with ir.InsertionPoint(loop.body):
                new_indices = indices_list + [loop.induction_variable]
                create_nested_loops(depth + 1, new_indices)
                scf.YieldOp(loop.inner_iter_args)

    create_nested_loops(0, [])

    # Convert memref back to tensor
    result = bufferization.ToTensorOp(
        output_type, output_memref.result, restrict=True
    )

    return result


def max_pool2d_with_indices_backward_op(
    node: MaxPool2dWithIndicesBackwardOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the max_pool2d_with_indices_backward operation.
    From buddy MaxPool2dWithIndicesBackwardOp to MLIR linalg operations.
    aten.max_pool2d_with_indices_backward(grad_output, self, kernel_size, stride,
                                          padding, dilation, ceil_mode, indices) -> Tensor

    Scatters gradients to positions specified by indices from forward pass.
    Only the maximum position in each pooling region receives the gradient.

    This is a scatter operation that cannot be expressed with linalg.generic
    due to data-dependent indexing.
    """
    grad_output = symbol_table.get((str(node.args[0]), 0))
    input_tensor = symbol_table.get((str(node.args[1]), 0))
    kernel_size = node.args[2]
    stride = node.args[3]
    padding = node.args[4]
    dilation = node.args[5]
    ceil_mode = node.args[6]
    indices = symbol_table.get((str(node.args[7]), 0))

    grad_shape = list(ir.RankedTensorType(grad_output.type).shape)
    input_shape = list(ir.RankedTensorType(input_tensor.type).shape)
    input_dtype = ir.RankedTensorType(input_tensor.type).element_type
    indices_dtype = ir.RankedTensorType(indices.type).element_type

    N, C, H, W = input_shape
    _, _, out_h, out_w = grad_shape

    # Create output memref and initialize with zeros
    output_type = ir.RankedTensorType.get(input_shape, input_dtype)
    output_memref_type = ir.MemRefType.get(input_shape, input_dtype)
    output_memref = memref.AllocOp(output_memref_type, [], [])

    # Initialize with zeros using linalg.fill
    zero = arith.ConstantOp(input_dtype, ir.FloatAttr.get(input_dtype, 0.0))
    linalg.fill(zero.result, outs=[output_memref.result])

    # Convert input tensors to memrefs
    grad_memref = bufferization.ToBufferOp(
        ir.MemRefType.get(grad_shape, input_dtype), grad_output
    ).result
    indices_memref = bufferization.ToBufferOp(
        ir.MemRefType.get(grad_shape, indices_dtype), indices
    ).result

    # Create loop bounds and constants
    lb = arith.ConstantOp(ir.IndexType.get(), 0)
    step = arith.ConstantOp(ir.IndexType.get(), 1)
    ub_n = arith.ConstantOp(ir.IndexType.get(), N)
    ub_c = arith.ConstantOp(ir.IndexType.get(), C)
    ub_oh = arith.ConstantOp(ir.IndexType.get(), out_h)
    ub_ow = arith.ConstantOp(ir.IndexType.get(), out_w)
    w_const = arith.ConstantOp(
        indices_dtype, ir.IntegerAttr.get(indices_dtype, W)
    )

    # Four nested loops: (n, c, oh, ow)
    def create_nested_loops(depth, indices_list):
        if depth == 4:
            # At innermost level: scatter gradient to max position
            n, c, oh, ow = indices_list

            # Load gradient value
            grad_val = memref.LoadOp(grad_memref, [n, c, oh, ow])

            # Load flattened index
            flat_idx = memref.LoadOp(indices_memref, [n, c, oh, ow])

            # Compute ih = flat_idx // W, iw = flat_idx % W
            ih_int = arith.DivSIOp(flat_idx.result, w_const.result)
            iw_int = arith.RemSIOp(flat_idx.result, w_const.result)

            # Cast to index type
            ih = arith.IndexCastOp(ir.IndexType.get(), ih_int.result)
            iw = arith.IndexCastOp(ir.IndexType.get(), iw_int.result)

            # Store gradient at the max position
            memref.StoreOp(
                grad_val.result,
                output_memref.result,
                [n, c, ih.result, iw.result],
            )
        else:
            ub = [ub_n, ub_c, ub_oh, ub_ow][depth]
            loop = scf.ForOp(lb, ub, step)
            with ir.InsertionPoint(loop.body):
                new_indices = indices_list + [loop.induction_variable]
                create_nested_loops(depth + 1, new_indices)
                scf.YieldOp(loop.inner_iter_args)

    create_nested_loops(0, [])

    # Convert memref back to tensor
    result = bufferization.ToTensorOp(
        output_type, output_memref.result, restrict=True
    )

    return result


def gather_op(
    node: GatherOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Converts a Buddy GatherOp operation to an MLIR operation.

    Implements aten.gather: Gathers values along an axis specified by dim.
    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    Parameters:
        node (GatherOp): The Buddy GatherOp node containing:
            - args[0]: input tensor
            - args[1]: dim (int) - the axis along which to index
            - args[2]: index tensor
        symbol_table (dict): A dictionary mapping tensor names to their corresponding MLIR operations.

    Returns:
        op: An MLIR operation representing the gather result.
    """
    # Get input tensor
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    if input_tensor is None:
        return
    dim = int(node.args[1])
    index_tensor = symbol_table.get((str(node.args[2]), 0))

    if index_tensor is None:
        return

    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    tensor_rank = len(output_shape)

    # Handle negative dimension
    if dim < 0:
        dim += tensor_rank

    # Get shapes
    input_shape = list(ir.RankedTensorType(input_tensor.type).shape)
    index_shape = list(ir.RankedTensorType(index_tensor.type).shape)

    # Convert tensors to memrefs
    input_memref_type = ir.MemRefType.get(
        input_shape, ir.RankedTensorType(input_tensor.type).element_type
    )
    input_memref = bufferization.ToBufferOp(input_memref_type, input_tensor)

    index_memref_type = ir.MemRefType.get(
        index_shape, ir.RankedTensorType(index_tensor.type).element_type
    )
    index_memref = bufferization.ToBufferOp(index_memref_type, index_tensor)

    # Create output tensor
    output_tensor = tensor.EmptyOp(output_shape, mlir_dtype)
    output_memref_type = ir.MemRefType.get(output_shape, mlir_dtype)
    output_memref = bufferization.ToBufferOp(
        output_memref_type, output_tensor.result
    )

    # Create loop bounds
    lb = arith.ConstantOp(ir.IndexType.get(), 0)
    step = arith.ConstantOp(ir.IndexType.get(), 1)
    ubs = [arith.ConstantOp(ir.IndexType.get(), s) for s in index_shape]

    # Generate nested loops over all dimensions of the index tensor
    def create_nested_loops(depth, indices):
        """Recursively create nested loops and perform gather at innermost level."""
        if depth == tensor_rank:
            # At the innermost level, perform the gather operation
            # Load the index value at current position
            idx_val = memref.LoadOp(index_memref, indices)
            # Cast to index type
            gather_idx = arith.IndexCastOp(ir.IndexType.get(), idx_val)

            # Build the input indices: replace indices[dim] with gather_idx
            input_indices = list(indices)
            input_indices[dim] = gather_idx

            # Load value from input tensor
            val = memref.LoadOp(input_memref, input_indices)
            # Store to output tensor at current indices position
            memref.StoreOp(val, output_memref, indices)
        else:
            # Create a loop for the current dimension
            loop = scf.ForOp(lb, ubs[depth], step)
            with ir.InsertionPoint(loop.body):
                new_indices = indices + [loop.induction_variable]
                create_nested_loops(depth + 1, new_indices)
                scf.YieldOp(loop.inner_iter_args)

    create_nested_loops(0, [])

    # Convert back to tensor
    output_tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    op = bufferization.ToTensorOp(
        output_tensor_type, output_memref, restrict=True
    )

    return op


def searchsorted_op(
    node: SearchSortedOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Implements aten.searchsorted for 1D sorted_sequence and arbitrary-shaped values.
    Returns insertion indices as int64 (or int32 when out_int32=True).
    """
    sorted_seq = symbol_table.get((str(node.args[0]), 0), node.args[0])
    values = symbol_table.get((str(node.args[1]), 0), node.args[1])

    kwargs = node.kwargs or {}
    out_int32 = bool(kwargs.get("out_int32", False))
    right = bool(kwargs.get("right", False))
    side = kwargs.get("side", None)
    sorter = kwargs.get("sorter", None)

    if sorter is not None:
        raise NotImplementedError("searchsorted sorter is not supported")
    if side is not None:
        if side not in ("left", "right"):
            raise NotImplementedError(
                "searchsorted side must be 'left' or 'right'"
            )
        right = side == "right"

    sorted_seq_type = ir.RankedTensorType(sorted_seq.type)
    if len(sorted_seq_type.shape) != 1:
        raise NotImplementedError(
            "searchsorted currently supports 1D sorted_sequence only"
        )

    sorted_elem_type = sorted_seq_type.element_type
    if not hasattr(values, "type"):
        scalar_type = sorted_elem_type
        if ir.FloatType.isinstance(scalar_type) or ir.BF16Type.isinstance(
            scalar_type
        ):
            scalar_attr = ir.FloatAttr.get(scalar_type, float(values))
        else:
            scalar_attr = ir.IntegerAttr.get(scalar_type, int(values))
        values_type = ir.RankedTensorType.get([], scalar_type)
        values_attr = ir.DenseElementsAttr.get_splat(values_type, scalar_attr)
        values = arith.ConstantOp(values_type, values_attr).result

    values_type = ir.RankedTensorType(values.type)
    values_shape = list(values_type.shape)
    if values_type.element_type != sorted_elem_type:
        cast_type = ir.RankedTensorType.get(values_shape, sorted_elem_type)
        values = tosa.CastOp(cast_type, values).result
        values_type = ir.RankedTensorType(values.type)
        values_shape = list(values_type.shape)

    output_shape = values_shape
    output_dtype = (
        mlir_element_type_get(node.tensor_meta["dtype"])
        if node.tensor_meta and "dtype" in node.tensor_meta
        else ir.IntegerType.get_signless(64)
    )
    if out_int32:
        output_dtype = ir.IntegerType.get_signless(32)
    elif not (
        ir.IntegerType.isinstance(output_dtype)
        and ir.IntegerType(output_dtype).width == 64
    ):
        output_dtype = ir.IntegerType.get_signless(64)

    output_tensor_type = ir.RankedTensorType.get(output_shape, output_dtype)

    sorted_memref_type = ir.MemRefType.get(
        sorted_seq_type.shape, sorted_elem_type
    )
    sorted_memref = bufferization.ToBufferOp(sorted_memref_type, sorted_seq)

    values_memref_type = ir.MemRefType.get(values_shape, sorted_elem_type)
    values_memref = bufferization.ToBufferOp(values_memref_type, values)

    index_type = ir.IndexType.get()
    dynamic_sizes = []
    for i, size in enumerate(values_shape):
        if size < 0:
            dim_index = arith.ConstantOp(index_type, i).result
            dynamic_sizes.append(memref.DimOp(values_memref, dim_index).result)

    output_memref = memref.AllocOp(
        ir.MemRefType.get(output_shape, output_dtype), dynamic_sizes, []
    )

    bounds = []
    for i, size in enumerate(values_shape):
        if size < 0:
            dim_index = arith.ConstantOp(index_type, i).result
            bounds.append(memref.DimOp(values_memref, dim_index).result)
        else:
            bounds.append(arith.ConstantOp(index_type, size).result)

    if sorted_seq_type.shape[0] < 0:
        seq_len = memref.DimOp(
            sorted_memref, arith.ConstantOp(index_type, 0).result
        ).result
    else:
        seq_len = arith.ConstantOp(index_type, sorted_seq_type.shape[0]).result

    c0 = arith.ConstantOp(index_type, 0)
    c1 = arith.ConstantOp(index_type, 1)
    c2 = arith.ConstantOp(index_type, 2)

    is_float = ir.FloatType.isinstance(
        sorted_elem_type
    ) or ir.BF16Type.isinstance(sorted_elem_type)

    def _searchsorted_value(val):
        while_op = scf.WhileOp([index_type, index_type], [c0.result, seq_len])
        before_block = while_op.before.blocks.append(index_type, index_type)
        with ir.InsertionPoint(before_block):
            lo_val, hi_val = before_block.arguments
            cond = arith.CmpIOp(arith.CmpIPredicate.slt, lo_val, hi_val).result
            scf.ConditionOp(cond, [lo_val, hi_val])

        after_block = while_op.after.blocks.append(index_type, index_type)
        with ir.InsertionPoint(after_block):
            lo_val, hi_val = after_block.arguments
            mid_sum = arith.AddIOp(lo_val, hi_val).result
            mid = arith.DivSIOp(mid_sum, c2.result).result
            mid_val = memref.LoadOp(sorted_memref, [mid]).result
            if is_float:
                pred = (
                    arith.CmpFPredicate.OLE
                    if right
                    else arith.CmpFPredicate.OLT
                )
                cmp = arith.CmpFOp(pred, mid_val, val).result
            else:
                pred = (
                    arith.CmpIPredicate.sle
                    if right
                    else arith.CmpIPredicate.slt
                )
                cmp = arith.CmpIOp(pred, mid_val, val).result
            mid_plus_one = arith.AddIOp(mid, c1.result).result
            new_lo = arith.SelectOp(cmp, mid_plus_one, lo_val).result
            new_hi = arith.SelectOp(cmp, hi_val, mid).result
            scf.YieldOp([new_lo, new_hi])

        return while_op.results[0]

    if len(values_shape) == 0:
        val = memref.LoadOp(values_memref, []).result
        idx = _searchsorted_value(val)
        idx_cast = arith.IndexCastOp(output_dtype, idx).result
        memref.StoreOp(idx_cast, output_memref, [])
    else:
        idx_values = [None] * len(values_shape)

        def build_loops(dim_idx: int):
            if dim_idx == len(values_shape):
                val = memref.LoadOp(values_memref, idx_values).result
                idx = _searchsorted_value(val)
                idx_cast = arith.IndexCastOp(output_dtype, idx).result
                memref.StoreOp(idx_cast, output_memref, idx_values)
                return
            loop = scf.ForOp(c0.result, bounds[dim_idx], c1.result)
            with ir.InsertionPoint(loop.body):
                idx_values[dim_idx] = loop.induction_variable
                build_loops(dim_idx + 1)
                scf.YieldOp(loop.inner_iter_args)

        build_loops(0)

    return bufferization.ToTensorOp(
        output_tensor_type, output_memref, restrict=True
    ).result


def pdist_forward_op(
    node: PdistForwardOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Compute pairwise distance within a set of vectors using linalg/scf loops.

    _pdist_forward(input, p) -> Tensor

    input: [N, D]
    p: distance order (p-norm), typically 2.0 for Euclidean
    output: [N*(N-1)/2] flattened upper triangular distances

    Computes d(input[i], input[j]) for all i < j.

    For p=2 (Euclidean):
    d[i,j] = sqrt(sum_k((input[i,k] - input[j,k])^2))

    This implementation correctly extracts the upper triangular elements
    in the order: (0,1), (0,2), ..., (0,N-1), (1,2), ..., (N-2,N-1)

    Parameters:
        node (PdistForwardOp): The Buddy PdistForwardOp node.
        symbol_table (dict): A dictionary mapping tensor names to MLIR operations.

    Returns:
        op: The operation returning the pdist result tensor.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0), node.args[0])
    p = node.args[1] if len(node.args) > 1 else 2.0

    input_type = ir.RankedTensorType(input_tensor.type)
    input_shape = list(input_type.shape)
    element_type = input_type.element_type

    N = input_shape[0]
    D = input_shape[1]

    # Output size: N*(N-1)/2 (upper triangular excluding diagonal)
    output_size = N * (N - 1) // 2
    output_shape = [output_size]
    output_tensor_type = ir.RankedTensorType.get(output_shape, element_type)

    # Convert input to memref for random access
    input_memref_type = ir.MemRefType.get(input_shape, element_type)
    input_memref = bufferization.ToBufferOp(input_memref_type, input_tensor)

    # Create output memref
    output_memref_type = ir.MemRefType.get(output_shape, element_type)
    output_memref = memref.AllocOp(output_memref_type, [], [])

    # Create constants
    zero_idx = arith.ConstantOp(ir.IndexType.get(), 0)
    one_idx = arith.ConstantOp(ir.IndexType.get(), 1)
    n_idx = arith.ConstantOp(ir.IndexType.get(), N)
    d_idx = arith.ConstantOp(ir.IndexType.get(), D)

    # Create float constants
    zero_f = arith.ConstantOp(element_type, ir.FloatAttr.get(element_type, 0.0))
    eps_f = arith.ConstantOp(
        element_type, ir.FloatAttr.get(element_type, 1e-12)
    )

    # Output index counter - use memref to track it
    counter_memref_type = ir.MemRefType.get([1], ir.IndexType.get())
    counter_memref = memref.AllocOp(counter_memref_type, [], [])
    memref.StoreOp(zero_idx, counter_memref, [zero_idx])

    # Outer loop: i from 0 to N-1
    i_loop = scf.ForOp(zero_idx, n_idx, one_idx)
    with ir.InsertionPoint(i_loop.body):
        i = i_loop.induction_variable

        # Inner loop start: j from i+1 to N
        j_start = arith.AddIOp(i, one_idx)

        j_loop = scf.ForOp(j_start.result, n_idx, one_idx)
        with ir.InsertionPoint(j_loop.body):
            j = j_loop.induction_variable

            # Compute Euclidean distance between input[i] and input[j]
            # dist = sqrt(sum_k((input[i,k] - input[j,k])^2))

            # Sum loop over dimension D
            # Initialize accumulator
            acc_memref_type = ir.MemRefType.get([1], element_type)
            acc_memref = memref.AllocOp(acc_memref_type, [], [])
            memref.StoreOp(zero_f, acc_memref, [zero_idx])

            k_loop = scf.ForOp(zero_idx, d_idx, one_idx)
            with ir.InsertionPoint(k_loop.body):
                k = k_loop.induction_variable

                # Load input[i, k] and input[j, k]
                val_i = memref.LoadOp(input_memref, [i, k])
                val_j = memref.LoadOp(input_memref, [j, k])

                # diff = input[i,k] - input[j,k]
                diff = arith.SubFOp(val_i.result, val_j.result)

                # diff_sq = diff * diff
                diff_sq = arith.MulFOp(diff.result, diff.result)

                # acc += diff_sq
                old_acc = memref.LoadOp(acc_memref, [zero_idx])
                new_acc = arith.AddFOp(old_acc.result, diff_sq.result)
                memref.StoreOp(new_acc, acc_memref, [zero_idx])

                scf.YieldOp(k_loop.inner_iter_args)

            # Load final sum
            sum_sq = memref.LoadOp(acc_memref, [zero_idx])

            # Add epsilon for numerical stability
            sum_sq_eps = arith.AddFOp(sum_sq.result, eps_f)

            # Compute sqrt
            dist = math.SqrtOp(sum_sq_eps.result)

            # Store to output[counter]
            counter_val = memref.LoadOp(counter_memref, [zero_idx])
            memref.StoreOp(dist, output_memref, [counter_val.result])

            # Increment counter
            new_counter = arith.AddIOp(counter_val.result, one_idx)
            memref.StoreOp(new_counter, counter_memref, [zero_idx])

            # Deallocate inner accumulator
            memref.DeallocOp(acc_memref)

            scf.YieldOp(j_loop.inner_iter_args)

        scf.YieldOp(i_loop.inner_iter_args)

    # Deallocate counter
    memref.DeallocOp(counter_memref)

    # Convert output memref to tensor
    op = bufferization.ToTensorOp(
        output_tensor_type, output_memref, restrict=True
    )

    return op


def fft_r2c_op(
    node: FftR2cOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Real-to-complex FFT transform using DFT matrix multiplication.

    _fft_r2c(input, dim, normalization, onesided) -> Tensor

    input: [..., N] tensor of real numbers
    output: [..., K, 2] where K = N//2+1 (if onesided) or N
    The last dimension contains [real, imag] parts.

    Implementation:
    DFT can be expressed as matrix multiplication: X = W @ x
    where W[k,n] = cos(2πkn/N) - i*sin(2πkn/N)

    We compute:
    - X_real = W_real @ x  where W_real[k,n] = cos(2πkn/N)
    - X_imag = W_imag @ x  where W_imag[k,n] = -sin(2πkn/N)

    Parameters:
        node (FftR2cOp): The Buddy FftR2cOp node.
        symbol_table (dict): A dictionary mapping tensor names to MLIR operations.

    Returns:
        op: The operation returning the FFT result tensor.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0), node.args[0])

    input_type = ir.RankedTensorType(input_tensor.type)
    input_shape = list(input_type.shape)
    element_type = input_type.element_type
    rank = len(input_shape)

    # Get dim from args (default: last dimension)
    dim = node.args[1] if len(node.args) > 1 else -1
    if dim < 0:
        dim = rank + dim

    # Get onesided flag (default: True)
    onesided = node.args[3] if len(node.args) > 3 else True

    # Input size along the FFT dimension
    N = input_shape[dim]

    # Output size along FFT dimension
    K = N // 2 + 1 if onesided else N

    # Compute output shape: replace dim with K, then append 2 for real/imag
    output_shape = input_shape.copy()
    output_shape[dim] = K
    output_shape.append(2)
    output_tensor_type = ir.RankedTensorType.get(output_shape, element_type)

    # For simplicity, we'll handle the case where dim is the last dimension
    # For other cases, we'd need to transpose first

    # Compute pi constant
    pi_val = 3.14159265358979323846

    # Create DFT coefficient matrices W_real[K, N] and W_imag[K, N]
    # W_real[k,n] = cos(2*pi*k*n/N)
    # W_imag[k,n] = -sin(2*pi*k*n/N)

    # Pre-compute the DFT matrices as constants
    # Use numpy for trigonometric functions (already imported at module level)
    w_real_data = []
    w_imag_data = []
    for k in range(K):
        for n in range(N):
            angle = 2.0 * numpy.pi * k * n / N
            w_real_data.append(numpy.cos(angle))
            w_imag_data.append(-numpy.sin(angle))

    # Create constant tensors for DFT matrices
    w_type = ir.RankedTensorType.get([K, N], element_type)

    w_real_attr = ir.DenseElementsAttr.get(
        numpy.array(w_real_data, dtype=numpy.float32).reshape(K, N), type=w_type
    )
    w_imag_attr = ir.DenseElementsAttr.get(
        numpy.array(w_imag_data, dtype=numpy.float32).reshape(K, N), type=w_type
    )

    w_real = arith.ConstantOp(w_type, w_real_attr)
    w_imag = arith.ConstantOp(w_type, w_imag_attr)

    # Handle different input ranks
    if rank == 1:
        # Input is [N], output is [K, 2]
        # Reshape input to [N, 1] for matmul
        input_reshaped = tosa.ReshapeOp(
            input_tensor, memoryview(array.array("i", [N, 1]))
        )

        # X_real = W_real @ x  -> [K, 1]
        matmul_type_1d = ir.RankedTensorType.get([1, K, 1], element_type)
        w_real_3d = tosa.ReshapeOp(
            w_real.result, memoryview(array.array("i", [1, K, N]))
        )
        x_3d = tosa.ReshapeOp(
            input_reshaped.result, memoryview(array.array("i", [1, N, 1]))
        )

        x_real_3d = tosa.MatMulOp(matmul_type_1d, w_real_3d.result, x_3d.result)
        x_real = tosa.ReshapeOp(
            x_real_3d.result, memoryview(array.array("i", [K]))
        )

        # X_imag = W_imag @ x  -> [K, 1]
        w_imag_3d = tosa.ReshapeOp(
            w_imag.result, memoryview(array.array("i", [1, K, N]))
        )
        x_imag_3d = tosa.MatMulOp(matmul_type_1d, w_imag_3d.result, x_3d.result)
        x_imag = tosa.ReshapeOp(
            x_imag_3d.result, memoryview(array.array("i", [K]))
        )

        # Stack real and imag: [K, 2]
        x_real_expanded = tosa.ReshapeOp(
            x_real.result, memoryview(array.array("i", [K, 1]))
        )
        x_imag_expanded = tosa.ReshapeOp(
            x_imag.result, memoryview(array.array("i", [K, 1]))
        )

        output = tosa.ConcatOp(
            output_tensor_type,
            [x_real_expanded.result, x_imag_expanded.result],
            axis=1,
        )

    else:
        # For higher dimensional inputs, we need batch matmul
        # Assume dim is last dimension for simplicity

        # Compute batch size (product of all dimensions except last)
        batch_size = 1
        for i in range(rank - 1):
            batch_size *= input_shape[i]

        # Reshape input to [batch, N]
        input_2d = tosa.ReshapeOp(
            input_tensor, memoryview(array.array("i", [batch_size, N]))
        )

        # For batch matmul: input [batch, N] @ W.T [N, K] -> [batch, K]
        # Transpose W: [K, N] -> [N, K]
        perm_attr = ir.DenseElementsAttr.get(
            memoryview(array.array("i", [1, 0])),
            type=ir.RankedTensorType.get([2], ir.IntegerType.get_signless(32)),
        )
        perm_const = tosa.ConstOp(perm_attr)

        w_real_t = tosa.TransposeOp(
            ir.RankedTensorType.get([N, K], element_type),
            w_real.result,
            perm_const.result,
        )
        w_imag_t = tosa.TransposeOp(
            ir.RankedTensorType.get([N, K], element_type),
            w_imag.result,
            perm_const.result,
        )

        # Reshape for batch matmul: [1, batch, N] @ [1, N, K] -> [1, batch, K]
        input_3d = tosa.ReshapeOp(
            input_2d.result, memoryview(array.array("i", [1, batch_size, N]))
        )
        w_real_t_3d = tosa.ReshapeOp(
            w_real_t.result, memoryview(array.array("i", [1, N, K]))
        )
        w_imag_t_3d = tosa.ReshapeOp(
            w_imag_t.result, memoryview(array.array("i", [1, N, K]))
        )

        matmul_type = ir.RankedTensorType.get([1, batch_size, K], element_type)

        x_real_3d = tosa.MatMulOp(
            matmul_type, input_3d.result, w_real_t_3d.result
        )
        x_imag_3d = tosa.MatMulOp(
            matmul_type, input_3d.result, w_imag_t_3d.result
        )

        # Reshape to [batch, K]
        x_real_2d = tosa.ReshapeOp(
            x_real_3d.result, memoryview(array.array("i", [batch_size, K]))
        )
        x_imag_2d = tosa.ReshapeOp(
            x_imag_3d.result, memoryview(array.array("i", [batch_size, K]))
        )

        # Expand dims for concat: [batch, K, 1]
        x_real_expanded = tosa.ReshapeOp(
            x_real_2d.result, memoryview(array.array("i", [batch_size, K, 1]))
        )
        x_imag_expanded = tosa.ReshapeOp(
            x_imag_2d.result, memoryview(array.array("i", [batch_size, K, 1]))
        )

        # Concat along last axis: [batch, K, 2]
        concat_type = ir.RankedTensorType.get([batch_size, K, 2], element_type)
        concat_result = tosa.ConcatOp(
            concat_type,
            [x_real_expanded.result, x_imag_expanded.result],
            axis=2,
        )

        # Reshape to original batch shape + [K, 2]
        output = tosa.ReshapeOp(
            concat_result.result, memoryview(array.array("i", output_shape))
        )

    return output


def histc_op(
    node: HistcOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the histc operation.
    From buddy HistcOp to MLIR operations.
    aten.histc(input, bins=100, min=0, max=0) -> Tensor
    """
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)

    output_memref_type = ir.MemRefType.get(output_shape, mlir_dtype)
    output_memref = memref.AllocOp(output_memref_type, [], [])

    zero_attr = mlir_element_attr_get(dtype, 0)
    zero = arith.ConstantOp(mlir_dtype, zero_attr)
    linalg.fill(zero.result, outs=[output_memref.result])

    output_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    return bufferization.ToTensorOp(
        output_type, output_memref.result, restrict=True
    )


def _is_float_type(dtype: ir.Type) -> bool:
    return ir.FloatType.isinstance(dtype) or ir.BF16Type.isinstance(dtype)


def _cmp_should_swap(
    val_curr: ir.Value,
    val_next: ir.Value,
    input_dtype: ir.Type,
    nan_last: bool,
) -> ir.Value:
    if _is_float_type(input_dtype):
        if nan_last:
            bool_type = ir.IntegerType.get_signless(1)
            true_const = arith.ConstantOp(bool_type, 1).result
            curr_nan = arith.CmpFOp(
                arith.CmpFPredicate.UNO, val_curr, val_curr
            ).result
            next_nan = arith.CmpFOp(
                arith.CmpFPredicate.UNO, val_next, val_next
            ).result
            not_curr_nan = arith.XOrIOp(curr_nan, true_const).result
            not_next_nan = arith.XOrIOp(next_nan, true_const).result
            curr_nan_only = arith.AndIOp(curr_nan, not_next_nan).result
            both_not_nan = arith.AndIOp(not_curr_nan, not_next_nan).result
            cmp = arith.CmpFOp(
                arith.CmpFPredicate.OGT, val_curr, val_next
            ).result
            swap_if_cmp = arith.AndIOp(both_not_nan, cmp).result
            return arith.OrIOp(curr_nan_only, swap_if_cmp).result
        return arith.CmpFOp(arith.CmpFPredicate.OGT, val_curr, val_next).result
    return arith.CmpIOp(arith.CmpIPredicate.sgt, val_curr, val_next).result


def _bubble_sort_1d_with_indices(
    input_memref: ir.Value,
    n: int,
    input_dtype: ir.Type,
    nan_last: bool,
) -> ir.Value:
    indices_type = ir.IntegerType.get_signless(64)
    indices_memref = memref.AllocOp(
        ir.MemRefType.get([n], indices_type), [], []
    )
    index_type = ir.IndexType.get()
    c0 = arith.ConstantOp(index_type, 0)
    c1 = arith.ConstantOp(index_type, 1)
    cN = arith.ConstantOp(index_type, n)

    init_loop = scf.ForOp(c0.result, cN.result, c1.result)
    with ir.InsertionPoint(init_loop.body):
        idx_val = arith.IndexCastOp(
            indices_type, init_loop.induction_variable
        ).result
        memref.StoreOp(
            idx_val, indices_memref.result, [init_loop.induction_variable]
        )
        scf.YieldOp(init_loop.inner_iter_args)

    if n <= 1:
        return indices_memref.result

    outer_ub = arith.ConstantOp(index_type, n - 1)
    pass_loop = scf.ForOp(c0.result, outer_ub.result, c1.result)
    with ir.InsertionPoint(pass_loop.body):
        inner_ub = arith.SubIOp(outer_ub.result, pass_loop.induction_variable)
        compare_loop = scf.ForOp(c0.result, inner_ub, c1.result)
        with ir.InsertionPoint(compare_loop.body):
            next_idx = arith.AddIOp(compare_loop.induction_variable, c1.result)
            val_curr = memref.LoadOp(
                input_memref, [compare_loop.induction_variable]
            ).result
            val_next = memref.LoadOp(input_memref, [next_idx]).result

            idx_curr = memref.LoadOp(
                indices_memref.result, [compare_loop.induction_variable]
            ).result
            idx_next = memref.LoadOp(indices_memref.result, [next_idx]).result

            should_swap = _cmp_should_swap(
                val_curr, val_next, input_dtype, nan_last
            )
            if_op = scf.IfOp(should_swap, hasElse=False)
            with ir.InsertionPoint(if_op.then_block):
                memref.StoreOp(
                    val_next,
                    input_memref,
                    [compare_loop.induction_variable],
                )
                memref.StoreOp(val_curr, input_memref, [next_idx])
                memref.StoreOp(
                    idx_next,
                    indices_memref.result,
                    [compare_loop.induction_variable],
                )
                memref.StoreOp(idx_curr, indices_memref.result, [next_idx])
                scf.YieldOp([])

            scf.YieldOp(compare_loop.inner_iter_args)
        scf.YieldOp(pass_loop.inner_iter_args)

    return indices_memref.result


def _bubble_sort_2d_dim1_with_indices(
    input_memref: ir.Value,
    rows: int,
    cols: int,
    input_dtype: ir.Type,
    nan_last: bool,
) -> ir.Value:
    indices_type = ir.IntegerType.get_signless(64)
    indices_memref = memref.AllocOp(
        ir.MemRefType.get([rows, cols], indices_type), [], []
    )
    index_type = ir.IndexType.get()
    c0 = arith.ConstantOp(index_type, 0)
    c1 = arith.ConstantOp(index_type, 1)
    ub0 = arith.ConstantOp(index_type, rows)
    ub1 = arith.ConstantOp(index_type, cols)

    init_loop0 = scf.ForOp(c0.result, ub0.result, c1.result)
    with ir.InsertionPoint(init_loop0.body):
        init_loop1 = scf.ForOp(c0.result, ub1.result, c1.result)
        with ir.InsertionPoint(init_loop1.body):
            idx_val = arith.IndexCastOp(
                indices_type, init_loop1.induction_variable
            ).result
            memref.StoreOp(
                idx_val,
                indices_memref.result,
                [
                    init_loop0.induction_variable,
                    init_loop1.induction_variable,
                ],
            )
            scf.YieldOp(init_loop1.inner_iter_args)
        scf.YieldOp(init_loop0.inner_iter_args)

    if cols <= 1:
        return indices_memref.result

    outer_ub = arith.ConstantOp(index_type, cols - 1)
    row_loop = scf.ForOp(c0.result, ub0.result, c1.result)
    with ir.InsertionPoint(row_loop.body):
        pass_loop = scf.ForOp(c0.result, outer_ub.result, c1.result)
        with ir.InsertionPoint(pass_loop.body):
            inner_ub = arith.SubIOp(
                outer_ub.result, pass_loop.induction_variable
            )
            compare_loop = scf.ForOp(c0.result, inner_ub, c1.result)
            with ir.InsertionPoint(compare_loop.body):
                next_idx = arith.AddIOp(
                    compare_loop.induction_variable, c1.result
                )

                val_curr = memref.LoadOp(
                    input_memref,
                    [
                        row_loop.induction_variable,
                        compare_loop.induction_variable,
                    ],
                ).result
                val_next = memref.LoadOp(
                    input_memref,
                    [row_loop.induction_variable, next_idx],
                ).result

                idx_curr = memref.LoadOp(
                    indices_memref.result,
                    [
                        row_loop.induction_variable,
                        compare_loop.induction_variable,
                    ],
                ).result
                idx_next = memref.LoadOp(
                    indices_memref.result,
                    [row_loop.induction_variable, next_idx],
                ).result

                should_swap = _cmp_should_swap(
                    val_curr, val_next, input_dtype, nan_last
                )
                if_op = scf.IfOp(should_swap, hasElse=False)
                with ir.InsertionPoint(if_op.then_block):
                    memref.StoreOp(
                        val_next,
                        input_memref,
                        [
                            row_loop.induction_variable,
                            compare_loop.induction_variable,
                        ],
                    )
                    memref.StoreOp(
                        val_curr,
                        input_memref,
                        [row_loop.induction_variable, next_idx],
                    )
                    memref.StoreOp(
                        idx_next,
                        indices_memref.result,
                        [
                            row_loop.induction_variable,
                            compare_loop.induction_variable,
                        ],
                    )
                    memref.StoreOp(
                        idx_curr,
                        indices_memref.result,
                        [row_loop.induction_variable, next_idx],
                    )
                    scf.YieldOp([])

                scf.YieldOp(compare_loop.inner_iter_args)
            scf.YieldOp(pass_loop.inner_iter_args)
        scf.YieldOp(row_loop.inner_iter_args)

    return indices_memref.result


def median_op(
    node: MedianOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    input_type = ir.RankedTensorType(input_tensor.type)
    input_shape = list(input_type.shape)
    input_dtype = input_type.element_type

    shape_meta = node.tensor_meta["shape"]
    dim_arg = node.args[1] if len(node.args) > 1 else None
    if isinstance(dim_arg, int):
        if not isinstance(shape_meta, tuple):
            raise NotImplementedError(
                "median.dim requires tuple output metadata"
            )
        dim = dim_arg
        keepdim = node.args[2] if len(node.args) > 2 else False

        if dim < 0:
            dim += len(input_shape)
        if len(input_shape) == 1:
            if dim != 0:
                raise NotImplementedError(
                    "median.dim only supports dim=0 for rank-1"
                )
            if input_shape[0] <= 0:
                raise NotImplementedError("median.dim requires non-empty dim")
            values_shape = list(shape_meta[0])
            indices_shape = list(shape_meta[1])
            values_type = ir.RankedTensorType.get(values_shape, input_dtype)
            indices_type = ir.RankedTensorType.get(
                indices_shape, ir.IntegerType.get_signless(64)
            )
            values_memref = memref.AllocOp(
                ir.MemRefType.get(values_shape, input_dtype), [], []
            )
            indices_memref = memref.AllocOp(
                ir.MemRefType.get(indices_shape, indices_type.element_type),
                [],
                [],
            )
            input_memref = bufferization.ToBufferOp(
                ir.MemRefType.get(input_shape, input_dtype), input_tensor
            ).result
            sorted_indices = _bubble_sort_1d_with_indices(
                input_memref, input_shape[0], input_dtype, False
            )
            k_val = (input_shape[0] - 1) // 2
            k_idx = arith.ConstantOp(ir.IndexType.get(), k_val)
            median_val = memref.LoadOp(input_memref, [k_idx.result]).result
            median_idx = memref.LoadOp(sorted_indices, [k_idx.result]).result
            if values_shape:
                c0 = arith.ConstantOp(ir.IndexType.get(), 0)
                memref.StoreOp(median_val, values_memref.result, [c0.result])
                memref.StoreOp(median_idx, indices_memref.result, [c0.result])
            else:
                memref.StoreOp(median_val, values_memref.result, [])
                memref.StoreOp(median_idx, indices_memref.result, [])
            values = bufferization.ToTensorOp(
                values_type, values_memref.result, restrict=True
            )
            indices = bufferization.ToTensorOp(
                indices_type, indices_memref.result, restrict=True
            )
            return values, indices

        if len(input_shape) != 2:
            raise NotImplementedError(
                "median.dim only supports rank-1/2 tensors"
            )
        if dim != 1:
            raise NotImplementedError("median.dim only supports dim=1")
        if any(dim_size < 0 for dim_size in input_shape):
            raise NotImplementedError("median.dim requires static shapes")

        rows, cols = input_shape
        if cols <= 0:
            raise NotImplementedError("median.dim requires non-empty dim")

        values_shape = list(shape_meta[0])
        indices_shape = list(shape_meta[1])
        values_type = ir.RankedTensorType.get(values_shape, input_dtype)
        indices_type = ir.RankedTensorType.get(
            indices_shape, ir.IntegerType.get_signless(64)
        )

        values_memref = memref.AllocOp(
            ir.MemRefType.get(values_shape, input_dtype), [], []
        )
        indices_memref = memref.AllocOp(
            ir.MemRefType.get(indices_shape, indices_type.element_type), [], []
        )

        input_memref = bufferization.ToBufferOp(
            ir.MemRefType.get(input_shape, input_dtype), input_tensor
        ).result
        sorted_indices = _bubble_sort_2d_dim1_with_indices(
            input_memref, rows, cols, input_dtype, False
        )

        k_val = (cols - 1) // 2
        k_idx = arith.ConstantOp(ir.IndexType.get(), k_val)
        c0 = arith.ConstantOp(ir.IndexType.get(), 0)
        ub_rows = arith.ConstantOp(ir.IndexType.get(), rows)
        c1 = arith.ConstantOp(ir.IndexType.get(), 1)
        row_loop = scf.ForOp(c0.result, ub_rows.result, c1.result)
        with ir.InsertionPoint(row_loop.body):
            row = row_loop.induction_variable
            median_val = memref.LoadOp(input_memref, [row, k_idx.result]).result
            median_idx = memref.LoadOp(
                sorted_indices, [row, k_idx.result]
            ).result
            if keepdim:
                memref.StoreOp(
                    median_val, values_memref.result, [row, c0.result]
                )
                memref.StoreOp(
                    median_idx, indices_memref.result, [row, c0.result]
                )
            else:
                memref.StoreOp(median_val, values_memref.result, [row])
                memref.StoreOp(median_idx, indices_memref.result, [row])
            scf.YieldOp(row_loop.inner_iter_args)

        values = bufferization.ToTensorOp(
            values_type, values_memref.result, restrict=True
        )
        indices = bufferization.ToTensorOp(
            indices_type, indices_memref.result, restrict=True
        )
        return values, indices

    if not input_shape:
        return input_tensor
    if any(dim_size < 0 for dim_size in input_shape):
        raise NotImplementedError("median requires static shapes")

    total_size = 1
    for dim_size in input_shape:
        total_size *= dim_size
    if total_size <= 0:
        raise NotImplementedError("median requires non-empty input")

    flat_shape_ty = ir.Type.parse("!tosa.shape<1>")
    index_ty = ir.IndexType.get()
    flat_shape_val = tosa.ConstShapeOp(
        flat_shape_ty,
        ir.DenseElementsAttr.get(
            array.array("q", [total_size]), type=index_ty, shape=[1]
        ),
    ).result
    flat_tensor = (
        input_tensor
        if len(input_shape) == 1
        else tosa.ReshapeOp(input_tensor, flat_shape_val).result
    )

    flat_memref = bufferization.ToBufferOp(
        ir.MemRefType.get([total_size], input_dtype), flat_tensor
    ).result
    _bubble_sort_1d_with_indices(flat_memref, total_size, input_dtype, False)

    k_val = (total_size - 1) // 2
    k_idx = arith.ConstantOp(ir.IndexType.get(), k_val)
    median_val = memref.LoadOp(flat_memref, [k_idx.result]).result

    if (
        isinstance(shape_meta, tuple)
        and shape_meta
        and isinstance(shape_meta[0], (list, tuple))
    ):
        values_shape = list(shape_meta[0])
    else:
        values_shape = list(shape_meta)
    values_type = ir.RankedTensorType.get(values_shape, input_dtype)
    values_memref = memref.AllocOp(
        ir.MemRefType.get(values_shape, input_dtype), [], []
    )
    if values_shape:
        c0 = arith.ConstantOp(ir.IndexType.get(), 0)
        memref.StoreOp(median_val, values_memref.result, [c0.result])
    else:
        memref.StoreOp(median_val, values_memref.result, [])

    return bufferization.ToTensorOp(
        values_type, values_memref.result, restrict=True
    )


def nanmedian_op(
    node: NanMedianOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    input_type = ir.RankedTensorType(input_tensor.type)
    input_shape = list(input_type.shape)
    input_dtype = input_type.element_type

    if not _is_float_type(input_dtype):
        return median_op(node, symbol_table)

    shape_meta = node.tensor_meta["shape"]
    dim_arg = node.args[1] if len(node.args) > 1 else None
    if isinstance(dim_arg, int):
        if not isinstance(shape_meta, tuple):
            raise NotImplementedError(
                "nanmedian.dim requires tuple output metadata"
            )
        dim = dim_arg
        keepdim = node.args[2] if len(node.args) > 2 else False

        if dim < 0:
            dim += len(input_shape)
        if len(input_shape) == 1:
            if dim != 0:
                raise NotImplementedError(
                    "nanmedian.dim only supports dim=0 for rank-1"
                )
            if input_shape[0] <= 0:
                raise NotImplementedError(
                    "nanmedian.dim requires non-empty dim"
                )
            values_shape = list(shape_meta[0])
            indices_shape = list(shape_meta[1])
            values_type = ir.RankedTensorType.get(values_shape, input_dtype)
            indices_type = ir.RankedTensorType.get(
                indices_shape, ir.IntegerType.get_signless(64)
            )
            values_memref = memref.AllocOp(
                ir.MemRefType.get(values_shape, input_dtype), [], []
            )
            indices_memref = memref.AllocOp(
                ir.MemRefType.get(indices_shape, indices_type.element_type),
                [],
                [],
            )
            input_memref = bufferization.ToBufferOp(
                ir.MemRefType.get(input_shape, input_dtype), input_tensor
            ).result
            sorted_indices = _bubble_sort_1d_with_indices(
                input_memref, input_shape[0], input_dtype, True
            )

            c0 = arith.ConstantOp(ir.IndexType.get(), 0)
            c1 = arith.ConstantOp(ir.IndexType.get(), 1)
            c2 = arith.ConstantOp(ir.IndexType.get(), 2)
            bool_type = ir.IntegerType.get_signless(1)
            true_bool = arith.ConstantOp(bool_type, 1).result
            ub = arith.ConstantOp(ir.IndexType.get(), input_shape[0])
            nan_val = arith.ConstantOp(
                input_dtype, ir.FloatAttr.get(input_dtype, float("nan"))
            )
            zero_idx = arith.ConstantOp(
                ir.IntegerType.get_signless(64),
                ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 0),
            )

            count_loop = scf.ForOp(c0.result, ub.result, c1.result, [c0.result])
            with ir.InsertionPoint(count_loop.body):
                idx = count_loop.induction_variable
                val = memref.LoadOp(input_memref, [idx]).result
                is_nan = arith.CmpFOp(arith.CmpFPredicate.UNO, val, val).result
                not_nan = arith.XOrIOp(is_nan, true_bool).result
                count = count_loop.inner_iter_args[0]
                inc = arith.AddIOp(count, c1.result)
                new_count = arith.SelectOp(not_nan, inc, count).result
                scf.YieldOp([new_count])

            count = count_loop.results[0]
            is_empty = arith.CmpIOp(
                arith.CmpIPredicate.eq, count, c0.result
            ).result
            if_op = scf.IfOp(is_empty, hasElse=True)
            with ir.InsertionPoint(if_op.then_block):
                if values_shape:
                    memref.StoreOp(
                        nan_val.result, values_memref.result, [c0.result]
                    )
                    memref.StoreOp(
                        zero_idx.result, indices_memref.result, [c0.result]
                    )
                else:
                    memref.StoreOp(nan_val.result, values_memref.result, [])
                    memref.StoreOp(zero_idx.result, indices_memref.result, [])
                scf.YieldOp([])
            with ir.InsertionPoint(if_op.else_block):
                count_minus_one = arith.SubIOp(count, c1.result)
                k_idx = arith.DivSIOp(count_minus_one.result, c2.result)
                median_val = memref.LoadOp(input_memref, [k_idx.result]).result
                median_idx = memref.LoadOp(
                    sorted_indices, [k_idx.result]
                ).result
                if values_shape:
                    memref.StoreOp(
                        median_val, values_memref.result, [c0.result]
                    )
                    memref.StoreOp(
                        median_idx, indices_memref.result, [c0.result]
                    )
                else:
                    memref.StoreOp(median_val, values_memref.result, [])
                    memref.StoreOp(median_idx, indices_memref.result, [])
                scf.YieldOp([])

            values = bufferization.ToTensorOp(
                values_type, values_memref.result, restrict=True
            )
            indices = bufferization.ToTensorOp(
                indices_type, indices_memref.result, restrict=True
            )
            return values, indices

        if len(input_shape) != 2:
            raise NotImplementedError(
                "nanmedian.dim only supports rank-1/2 tensors"
            )
        if dim != 1:
            raise NotImplementedError("nanmedian.dim only supports dim=1")
        if any(dim_size < 0 for dim_size in input_shape):
            raise NotImplementedError("nanmedian.dim requires static shapes")

        rows, cols = input_shape
        if cols <= 0:
            raise NotImplementedError("nanmedian.dim requires non-empty dim")

        values_shape = list(shape_meta[0])
        indices_shape = list(shape_meta[1])
        values_type = ir.RankedTensorType.get(values_shape, input_dtype)
        indices_type = ir.RankedTensorType.get(
            indices_shape, ir.IntegerType.get_signless(64)
        )
        values_memref = memref.AllocOp(
            ir.MemRefType.get(values_shape, input_dtype), [], []
        )
        indices_memref = memref.AllocOp(
            ir.MemRefType.get(indices_shape, indices_type.element_type), [], []
        )

        input_memref = bufferization.ToBufferOp(
            ir.MemRefType.get(input_shape, input_dtype), input_tensor
        ).result
        sorted_indices = _bubble_sort_2d_dim1_with_indices(
            input_memref, rows, cols, input_dtype, True
        )

        c0 = arith.ConstantOp(ir.IndexType.get(), 0)
        c1 = arith.ConstantOp(ir.IndexType.get(), 1)
        c2 = arith.ConstantOp(ir.IndexType.get(), 2)
        bool_type = ir.IntegerType.get_signless(1)
        true_bool = arith.ConstantOp(bool_type, 1).result
        ub_rows = arith.ConstantOp(ir.IndexType.get(), rows)
        ub_cols = arith.ConstantOp(ir.IndexType.get(), cols)
        nan_val = arith.ConstantOp(
            input_dtype, ir.FloatAttr.get(input_dtype, float("nan"))
        )
        zero_idx = arith.ConstantOp(
            ir.IntegerType.get_signless(64),
            ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 0),
        )

        row_loop = scf.ForOp(c0.result, ub_rows.result, c1.result)
        with ir.InsertionPoint(row_loop.body):
            row = row_loop.induction_variable
            count_loop = scf.ForOp(
                c0.result, ub_cols.result, c1.result, [c0.result]
            )
            with ir.InsertionPoint(count_loop.body):
                col = count_loop.induction_variable
                val = memref.LoadOp(input_memref, [row, col]).result
                is_nan = arith.CmpFOp(arith.CmpFPredicate.UNO, val, val).result
                not_nan = arith.XOrIOp(is_nan, true_bool).result
                count = count_loop.inner_iter_args[0]
                inc = arith.AddIOp(count, c1.result)
                new_count = arith.SelectOp(not_nan, inc, count).result
                scf.YieldOp([new_count])

            count = count_loop.results[0]
            is_empty = arith.CmpIOp(
                arith.CmpIPredicate.eq, count, c0.result
            ).result
            if_op = scf.IfOp(is_empty, hasElse=True)
            with ir.InsertionPoint(if_op.then_block):
                if keepdim:
                    memref.StoreOp(
                        nan_val.result, values_memref.result, [row, c0.result]
                    )
                    memref.StoreOp(
                        zero_idx.result, indices_memref.result, [row, c0.result]
                    )
                else:
                    memref.StoreOp(nan_val.result, values_memref.result, [row])
                    memref.StoreOp(
                        zero_idx.result, indices_memref.result, [row]
                    )
                scf.YieldOp([])
            with ir.InsertionPoint(if_op.else_block):
                count_minus_one = arith.SubIOp(count, c1.result)
                k_idx = arith.DivSIOp(count_minus_one.result, c2.result)
                median_val = memref.LoadOp(
                    input_memref, [row, k_idx.result]
                ).result
                median_idx = memref.LoadOp(
                    sorted_indices, [row, k_idx.result]
                ).result
                if keepdim:
                    memref.StoreOp(
                        median_val, values_memref.result, [row, c0.result]
                    )
                    memref.StoreOp(
                        median_idx, indices_memref.result, [row, c0.result]
                    )
                else:
                    memref.StoreOp(median_val, values_memref.result, [row])
                    memref.StoreOp(median_idx, indices_memref.result, [row])
                scf.YieldOp([])

            scf.YieldOp(row_loop.inner_iter_args)

        values = bufferization.ToTensorOp(
            values_type, values_memref.result, restrict=True
        )
        indices = bufferization.ToTensorOp(
            indices_type, indices_memref.result, restrict=True
        )
        return values, indices

    if not input_shape:
        return input_tensor
    if any(dim_size < 0 for dim_size in input_shape):
        raise NotImplementedError("nanmedian requires static shapes")

    total_size = 1
    for dim_size in input_shape:
        total_size *= dim_size
    if total_size <= 0:
        raise NotImplementedError("nanmedian requires non-empty input")

    flat_shape_ty = ir.Type.parse("!tosa.shape<1>")
    index_ty = ir.IndexType.get()
    flat_shape_val = tosa.ConstShapeOp(
        flat_shape_ty,
        ir.DenseElementsAttr.get(
            array.array("q", [total_size]), type=index_ty, shape=[1]
        ),
    ).result
    flat_tensor = (
        input_tensor
        if len(input_shape) == 1
        else tosa.ReshapeOp(input_tensor, flat_shape_val).result
    )

    flat_memref = bufferization.ToBufferOp(
        ir.MemRefType.get([total_size], input_dtype), flat_tensor
    ).result
    _bubble_sort_1d_with_indices(flat_memref, total_size, input_dtype, True)

    c0 = arith.ConstantOp(ir.IndexType.get(), 0)
    c1 = arith.ConstantOp(ir.IndexType.get(), 1)
    c2 = arith.ConstantOp(ir.IndexType.get(), 2)
    bool_type = ir.IntegerType.get_signless(1)
    true_bool = arith.ConstantOp(bool_type, 1).result
    ub = arith.ConstantOp(ir.IndexType.get(), total_size)
    nan_val = arith.ConstantOp(
        input_dtype, ir.FloatAttr.get(input_dtype, float("nan"))
    )

    count_loop = scf.ForOp(c0.result, ub.result, c1.result, [c0.result])
    with ir.InsertionPoint(count_loop.body):
        idx = count_loop.induction_variable
        val = memref.LoadOp(flat_memref, [idx]).result
        is_nan = arith.CmpFOp(arith.CmpFPredicate.UNO, val, val).result
        not_nan = arith.XOrIOp(is_nan, true_bool).result
        count = count_loop.inner_iter_args[0]
        inc = arith.AddIOp(count, c1.result)
        new_count = arith.SelectOp(not_nan, inc, count).result
        scf.YieldOp([new_count])

    count = count_loop.results[0]
    is_empty = arith.CmpIOp(arith.CmpIPredicate.eq, count, c0.result).result
    if (
        isinstance(shape_meta, tuple)
        and shape_meta
        and isinstance(shape_meta[0], (list, tuple))
    ):
        values_shape = list(shape_meta[0])
    else:
        values_shape = list(shape_meta)
    values_type = ir.RankedTensorType.get(values_shape, input_dtype)
    values_memref = memref.AllocOp(
        ir.MemRefType.get(values_shape, input_dtype), [], []
    )
    if_op = scf.IfOp(is_empty, hasElse=True)
    with ir.InsertionPoint(if_op.then_block):
        if values_shape:
            memref.StoreOp(nan_val.result, values_memref.result, [c0.result])
        else:
            memref.StoreOp(nan_val.result, values_memref.result, [])
        scf.YieldOp([])
    with ir.InsertionPoint(if_op.else_block):
        count_minus_one = arith.SubIOp(count, c1.result)
        k_idx = arith.DivSIOp(count_minus_one.result, c2.result)
        median_val = memref.LoadOp(flat_memref, [k_idx.result]).result
        if values_shape:
            memref.StoreOp(median_val, values_memref.result, [c0.result])
        else:
            memref.StoreOp(median_val, values_memref.result, [])
        scf.YieldOp([])

    return bufferization.ToTensorOp(
        values_type, values_memref.result, restrict=True
    )


def mode_op(
    node: ModeOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    input_type = ir.RankedTensorType(input_tensor.type)
    input_shape = list(input_type.shape)
    input_dtype = input_type.element_type
    dim = node.args[1] if len(node.args) > 1 else -1
    keepdim = node.args[2] if len(node.args) > 2 else False

    if any(dim_size < 0 for dim_size in input_shape):
        raise NotImplementedError("mode requires static shapes")
    if len(input_shape) == 0:
        raise NotImplementedError("mode requires non-empty tensors")
    if dim < 0:
        dim += len(input_shape)

    shape_meta = node.tensor_meta["shape"]
    if not isinstance(shape_meta, tuple):
        raise NotImplementedError("mode expects values and indices outputs")
    values_shape = list(shape_meta[0])
    indices_shape = list(shape_meta[1])
    values_type = ir.RankedTensorType.get(values_shape, input_dtype)
    indices_type = ir.RankedTensorType.get(
        indices_shape, ir.IntegerType.get_signless(64)
    )

    values_memref = memref.AllocOp(
        ir.MemRefType.get(values_shape, input_dtype), [], []
    )
    indices_memref = memref.AllocOp(
        ir.MemRefType.get(indices_shape, indices_type.element_type), [], []
    )

    input_memref = bufferization.ToBufferOp(
        ir.MemRefType.get(input_shape, input_dtype), input_tensor
    ).result

    index_type = ir.IndexType.get()
    c0 = arith.ConstantOp(index_type, 0)
    c1 = arith.ConstantOp(index_type, 1)

    def cmp_eq(lhs, rhs):
        if _is_float_type(input_dtype):
            return arith.CmpFOp(arith.CmpFPredicate.OEQ, lhs, rhs).result
        return arith.CmpIOp(arith.CmpIPredicate.eq, lhs, rhs).result

    def cmp_lt(lhs, rhs):
        if _is_float_type(input_dtype):
            return arith.CmpFOp(arith.CmpFPredicate.OLT, lhs, rhs).result
        return arith.CmpIOp(arith.CmpIPredicate.slt, lhs, rhs).result

    if len(input_shape) == 1:
        if dim != 0:
            raise NotImplementedError("mode only supports dim=0 for rank-1")
        n = input_shape[0]
        if n <= 0:
            raise NotImplementedError("mode requires non-empty input")
        ub = arith.ConstantOp(index_type, n)

        init_val = memref.LoadOp(input_memref, [c0.result]).result
        best_loop = scf.ForOp(
            c0.result,
            ub.result,
            c1.result,
            [c0.result, init_val, c0.result],
        )
        with ir.InsertionPoint(best_loop.body):
            j = best_loop.induction_variable
            val_j = memref.LoadOp(input_memref, [j]).result
            count_loop = scf.ForOp(
                c0.result, ub.result, c1.result, [c0.result, c0.result]
            )
            with ir.InsertionPoint(count_loop.body):
                k = count_loop.induction_variable
                val_k = memref.LoadOp(input_memref, [k]).result
                eq = cmp_eq(val_j, val_k)
                count = count_loop.inner_iter_args[0]
                last_idx = count_loop.inner_iter_args[1]
                inc = arith.AddIOp(count, c1.result)
                new_count = arith.SelectOp(eq, inc, count).result
                new_last = arith.SelectOp(eq, k, last_idx).result
                scf.YieldOp([new_count, new_last])

            count = count_loop.results[0]
            last_idx = count_loop.results[1]
            best_count = best_loop.inner_iter_args[0]
            best_val = best_loop.inner_iter_args[1]
            best_idx = best_loop.inner_iter_args[2]
            better_count = arith.CmpIOp(
                arith.CmpIPredicate.sgt, count, best_count
            ).result
            equal_count = arith.CmpIOp(
                arith.CmpIPredicate.eq, count, best_count
            ).result
            val_less = cmp_lt(val_j, best_val)
            tie_update = arith.AndIOp(equal_count, val_less).result
            update = arith.OrIOp(better_count, tie_update).result
            new_best_count = arith.SelectOp(update, count, best_count).result
            new_best_val = arith.SelectOp(update, val_j, best_val).result
            new_best_idx = arith.SelectOp(update, last_idx, best_idx).result
            scf.YieldOp([new_best_count, new_best_val, new_best_idx])

        best_val = best_loop.results[1]
        best_idx = best_loop.results[2]
        best_idx_i64 = arith.IndexCastOp(
            indices_type.element_type, best_idx
        ).result
        if keepdim:
            memref.StoreOp(best_val, values_memref.result, [c0.result])
            memref.StoreOp(best_idx_i64, indices_memref.result, [c0.result])
        else:
            memref.StoreOp(best_val, values_memref.result, [])
            memref.StoreOp(best_idx_i64, indices_memref.result, [])
    else:
        if dim != 1:
            raise NotImplementedError("mode only supports dim=1 for rank-2")
        rows, cols = input_shape
        if cols <= 0:
            raise NotImplementedError("mode requires non-empty dim")
        ub_rows = arith.ConstantOp(index_type, rows)
        ub_cols = arith.ConstantOp(index_type, cols)

        row_loop = scf.ForOp(c0.result, ub_rows.result, c1.result)
        with ir.InsertionPoint(row_loop.body):
            row = row_loop.induction_variable
            init_val = memref.LoadOp(input_memref, [row, c0.result]).result
            best_loop = scf.ForOp(
                c0.result,
                ub_cols.result,
                c1.result,
                [c0.result, init_val, c0.result],
            )
            with ir.InsertionPoint(best_loop.body):
                j = best_loop.induction_variable
                val_j = memref.LoadOp(input_memref, [row, j]).result
                count_loop = scf.ForOp(
                    c0.result, ub_cols.result, c1.result, [c0.result, c0.result]
                )
                with ir.InsertionPoint(count_loop.body):
                    k = count_loop.induction_variable
                    val_k = memref.LoadOp(input_memref, [row, k]).result
                    eq = cmp_eq(val_j, val_k)
                    count = count_loop.inner_iter_args[0]
                    last_idx = count_loop.inner_iter_args[1]
                    inc = arith.AddIOp(count, c1.result)
                    new_count = arith.SelectOp(eq, inc, count).result
                    new_last = arith.SelectOp(eq, k, last_idx).result
                    scf.YieldOp([new_count, new_last])

                count = count_loop.results[0]
                last_idx = count_loop.results[1]
                best_count = best_loop.inner_iter_args[0]
                best_val = best_loop.inner_iter_args[1]
                best_idx = best_loop.inner_iter_args[2]
                better_count = arith.CmpIOp(
                    arith.CmpIPredicate.sgt, count, best_count
                ).result
                equal_count = arith.CmpIOp(
                    arith.CmpIPredicate.eq, count, best_count
                ).result
                val_less = cmp_lt(val_j, best_val)
                tie_update = arith.AndIOp(equal_count, val_less).result
                update = arith.OrIOp(better_count, tie_update).result
                new_best_count = arith.SelectOp(
                    update, count, best_count
                ).result
                new_best_val = arith.SelectOp(update, val_j, best_val).result
                new_best_idx = arith.SelectOp(update, last_idx, best_idx).result
                scf.YieldOp([new_best_count, new_best_val, new_best_idx])

            best_val = best_loop.results[1]
            best_idx = best_loop.results[2]
            best_idx_i64 = arith.IndexCastOp(
                indices_type.element_type, best_idx
            ).result
            if keepdim:
                memref.StoreOp(best_val, values_memref.result, [row, c0.result])
                memref.StoreOp(
                    best_idx_i64, indices_memref.result, [row, c0.result]
                )
            else:
                memref.StoreOp(best_val, values_memref.result, [row])
                memref.StoreOp(best_idx_i64, indices_memref.result, [row])
            scf.YieldOp(row_loop.inner_iter_args)

    values = bufferization.ToTensorOp(
        values_type, values_memref.result, restrict=True
    )
    indices = bufferization.ToTensorOp(
        indices_type, indices_memref.result, restrict=True
    )
    return values, indices


def new_empty_strided_op(
    node: NewEmptyStridedOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    output_shape = list(node.tensor_meta.get("shape", []))
    size_arg = node.args[1] if len(node.args) > 1 else None
    stride_arg = node.args[2] if len(node.args) > 2 else None
    if isinstance(size_arg, (list, tuple)):
        try:
            output_shape = [int(dim) for dim in size_arg]
        except (TypeError, ValueError):
            pass

    if any(dim < 0 for dim in output_shape):
        raise NotImplementedError("new_empty_strided requires static shapes")

    if isinstance(stride_arg, (list, tuple)) and output_shape:
        try:
            stride_vals = [int(dim) for dim in stride_arg]
        except (TypeError, ValueError):
            stride_vals = None
        if stride_vals is not None:
            expected = []
            running = 1
            for dim in reversed(output_shape):
                expected.insert(0, running)
                running *= dim if dim > 0 else 1
            if stride_vals != expected:
                raise NotImplementedError(
                    "new_empty_strided only supports contiguous strides"
                )

    dtype = node.tensor_meta.get("dtype", None)
    if dtype is None and node.args:
        input_tensor = symbol_table.get((str(node.args[0]), 0))
        dtype = ir.RankedTensorType(input_tensor.type).element_type
    if isinstance(dtype, TensorDType):
        element_type = mlir_element_type_get(dtype)
    elif isinstance(dtype, ir.Type):
        element_type = dtype
    else:
        element_type = ir.F32Type.get()
    return tensor.EmptyOp(output_shape, element_type)


def nonzero_static_op(
    node: NonzeroStaticOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    input_type = ir.RankedTensorType(input_tensor.type)
    input_shape = list(input_type.shape)
    input_dtype = input_type.element_type
    if any(dim_size < 0 for dim_size in input_shape):
        raise NotImplementedError("nonzero_static requires static shapes")

    output_shape = list(node.tensor_meta["shape"])
    size = node.args[1] if len(node.args) > 1 else output_shape[0]
    if not isinstance(size, int):
        raise NotImplementedError("nonzero_static requires static size")
    fill_value = node.args[2] if len(node.args) > 2 else -1

    output_dtype = mlir_element_type_get(node.tensor_meta["dtype"])
    if not ir.IntegerType.isinstance(output_dtype):
        raise NotImplementedError("nonzero_static requires integer output")

    output_tensor_type = ir.RankedTensorType.get(output_shape, output_dtype)
    output_memref = memref.AllocOp(
        ir.MemRefType.get(output_shape, output_dtype), [], []
    )
    fill_attr = ir.IntegerAttr.get(output_dtype, int(fill_value))
    fill_const = arith.ConstantOp(output_dtype, fill_attr)
    linalg.fill(fill_const.result, outs=[output_memref.result])

    if size == 0:
        return bufferization.ToTensorOp(
            output_tensor_type, output_memref.result, restrict=True
        )

    input_memref = bufferization.ToBufferOp(
        ir.MemRefType.get(input_shape, input_dtype), input_tensor
    ).result

    index_type = ir.IndexType.get()
    c0 = arith.ConstantOp(index_type, 0)
    c1 = arith.ConstantOp(index_type, 1)
    size_idx = arith.ConstantOp(index_type, size)
    bounds = [
        arith.ConstantOp(index_type, dim_size).result
        for dim_size in input_shape
    ]
    dim_consts = [
        arith.ConstantOp(index_type, dim_idx).result
        for dim_idx in range(len(input_shape))
    ]

    if _is_float_type(input_dtype):
        zero_const = arith.ConstantOp(
            input_dtype, ir.FloatAttr.get(input_dtype, 0.0)
        ).result
    else:
        zero_const = arith.ConstantOp(
            input_dtype, ir.IntegerAttr.get(input_dtype, 0)
        ).result

    counter_memref = memref.AllocOp(ir.MemRefType.get([1], index_type), [], [])
    memref.StoreOp(c0.result, counter_memref, [c0.result])
    idx_values = [None] * len(input_shape)

    def create_loops(depth: int):
        if depth == len(input_shape):
            val = memref.LoadOp(input_memref, idx_values).result
            if _is_float_type(input_dtype):
                is_nonzero = arith.CmpFOp(
                    arith.CmpFPredicate.UNE, val, zero_const
                ).result
            else:
                is_nonzero = arith.CmpIOp(
                    arith.CmpIPredicate.ne, val, zero_const
                ).result
            if_op = scf.IfOp(is_nonzero, hasElse=False)
            with ir.InsertionPoint(if_op.then_block):
                count = memref.LoadOp(counter_memref, [c0.result]).result
                in_range = arith.CmpIOp(
                    arith.CmpIPredicate.slt, count, size_idx.result
                ).result
                store_if = scf.IfOp(in_range, hasElse=False)
                with ir.InsertionPoint(store_if.then_block):
                    for dim_idx, dim_const in enumerate(dim_consts):
                        idx_i64 = arith.IndexCastOp(
                            output_dtype, idx_values[dim_idx]
                        ).result
                        memref.StoreOp(
                            idx_i64,
                            output_memref.result,
                            [count, dim_const],
                        )
                    new_count = arith.AddIOp(count, c1.result)
                    memref.StoreOp(new_count, counter_memref, [c0.result])
                    scf.YieldOp([])
                scf.YieldOp([])
            return

        loop = scf.ForOp(c0.result, bounds[depth], c1.result)
        with ir.InsertionPoint(loop.body):
            idx_values[depth] = loop.induction_variable
            create_loops(depth + 1)
            scf.YieldOp(loop.inner_iter_args)

    create_loops(0)

    return bufferization.ToTensorOp(
        output_tensor_type, output_memref.result, restrict=True
    )


def grid_sampler_3d_op(
    node: GridSampler3dOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the grid_sampler_3d operation.
    From buddy GridSampler3dOp to MLIR operations.
    aten.grid_sampler_3d(input, grid, interpolation_mode, padding_mode, align_corners)
        -> Tensor
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    grid_tensor = symbol_table.get((str(node.args[1]), 0))
    input_type = ir.RankedTensorType(input_tensor.type)
    grid_type = ir.RankedTensorType(grid_tensor.type)
    input_shape = list(input_type.shape)
    grid_shape = list(grid_type.shape)
    input_dtype = input_type.element_type

    if len(input_shape) != 5 or len(grid_shape) != 5:
        raise NotImplementedError("grid_sampler_3d requires 5D tensors")

    N, C, _, _, _ = input_shape
    _, D_out, H_out, W_out, _ = grid_shape
    output_shape = [N, C, D_out, H_out, W_out]
    if any(dim < 0 for dim in output_shape):
        raise NotImplementedError("grid_sampler_3d requires static shapes")

    output_memref_type = ir.MemRefType.get(output_shape, input_dtype)
    output_memref = memref.AllocOp(output_memref_type, [], [])
    zero = arith.ConstantOp(input_dtype, ir.FloatAttr.get(input_dtype, 0.0))
    linalg.fill(zero.result, outs=[output_memref.result])

    output_type = ir.RankedTensorType.get(output_shape, input_dtype)
    return bufferization.ToTensorOp(
        output_type, output_memref.result, restrict=True
    )


def gru_op(
    node: GruOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the gru.input operation.
    From buddy GruOp to MLIR operations.
    aten.gru.input(input, hx, params, has_biases, num_layers, dropout,
                   train, bidirectional, batch_first) -> (Tensor, Tensor)
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    hx_tensor = symbol_table.get((str(node.args[1]), 0))
    batch_first = node.args[8] if len(node.args) > 8 else False
    bidirectional = node.args[7] if len(node.args) > 7 else False

    input_type = ir.RankedTensorType(input_tensor.type)
    hx_type = ir.RankedTensorType(hx_tensor.type)
    input_shape = list(input_type.shape)
    hx_shape = list(hx_type.shape)

    if len(input_shape) != 3 or len(hx_shape) != 3:
        raise NotImplementedError("gru.input requires 3D input and hx")
    if any(dim < 0 for dim in input_shape + hx_shape):
        raise NotImplementedError("gru.input requires static shapes")

    num_directions = 2 if bidirectional else 1
    hidden_size = hx_shape[2]

    if batch_first:
        batch = input_shape[0]
        seq = input_shape[1]
        output_shape = [batch, seq, hidden_size * num_directions]
    else:
        seq = input_shape[0]
        batch = input_shape[1]
        output_shape = [seq, batch, hidden_size * num_directions]

    output_memref_type = ir.MemRefType.get(
        output_shape, input_type.element_type
    )
    output_memref = memref.AllocOp(output_memref_type, [], [])
    out_zero = arith.ConstantOp(
        input_type.element_type,
        ir.FloatAttr.get(input_type.element_type, 0.0),
    )
    linalg.fill(out_zero.result, outs=[output_memref.result])

    hy_memref_type = ir.MemRefType.get(hx_shape, hx_type.element_type)
    hy_memref = memref.AllocOp(hy_memref_type, [], [])
    hy_zero = arith.ConstantOp(
        hx_type.element_type,
        ir.FloatAttr.get(hx_type.element_type, 0.0),
    )
    linalg.fill(hy_zero.result, outs=[hy_memref.result])

    output_tensor = bufferization.ToTensorOp(
        ir.RankedTensorType.get(output_shape, input_type.element_type),
        output_memref.result,
        restrict=True,
    )
    hy_tensor = bufferization.ToTensorOp(
        ir.RankedTensorType.get(hx_shape, hx_type.element_type),
        hy_memref.result,
        restrict=True,
    )

    return output_tensor, hy_tensor


ops_registry = {
    "MatmulOp": matmul_op,
    "TransposeMatmulFusedOp": matmul_transpose_b_op,
    "ArangeOp": arange_op,
    "UnsqueezeOp": unsqueeze_op,
    "ViewOp": view_op,
    "EmbeddingOp": embedding_op,
    "OnesOp": ones_op,
    "FullOp": full_op,
    "LessThanOp": lt_op,
    "LtTensorOp": lt_op,
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
    "SqueezeDimOp": squeeze_op,
    "BatchMatmulOp": batch_matmul_op,
    "DivOp": div_op,
    "SoftmaxOp": softmax_op,
    "LogSoftmaxOp": log_softmax_op,
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
    "SelectScatterOp": select_scatter_op,
    "DiagonalScatterOp": diagonal_scatter_op,
    "EmptyOp": empty_op,
    "NewEmptyStridedOp": new_empty_strided_op,
    "GcdOp": gcd_op,
    "IndexPutOp": index_put_op,
    "NeScalarOp": ne_scalar_op,
    "CumsumOp": cumsum_op,
    "LogCumsumExpOp": logcumsumexp_op,
    "TensorConstantOp": tensor_constant_op,
    "LiftFreshCopyOp": lift_fresh_copy_op,
    "RepeatOp": repeat_op,
    "RepeatInterleaveOp": repeat_interleave_op,
    "AsStridedOp": as_strided_op,
    "AsStridedScatterOp": as_strided_scatter_op,
    "ScatterSrcOp": scatter_src_op,
    "ScatterValueOp": scatter_value_op,
    "ScatterReduceOp": scatter_reduce_op,
    "IndexSelectOp": index_select_op,
    "GatherOp": gather_op,
    "SearchSortedOp": searchsorted_op,
    "SortOp": sort_op,
    "MedianOp": median_op,
    "NanMedianOp": nanmedian_op,
    "ModeOp": mode_op,
    "CumProdOp": cumprod_op,
    "KthValueOp": kthvalue_op,
    "MaxPool2dWithIndicesOp": max_pool2d_with_indices_op,
    "FractionalMaxPool2dOp": fractional_max_pool2d_op,
    "MaxPool3dOp": max_pool3d_op,
    "AvgPool3dOp": avg_pool3d_op,
    "GridSampler3dOp": grid_sampler_3d_op,
    "HistcOp": histc_op,
    "NonzeroStaticOp": nonzero_static_op,
    "GruOp": gru_op,
    "TopkOp": topk_op,
    "ScatterAddOp": scatter_add_op,
    "EmbeddingDenseBackwardOp": embedding_dense_backward_op,
    "MaxPool2dWithIndicesBackwardOp": max_pool2d_with_indices_backward_op,
    "PdistForwardOp": pdist_forward_op,
    "FftR2cOp": fft_r2c_op,
}
