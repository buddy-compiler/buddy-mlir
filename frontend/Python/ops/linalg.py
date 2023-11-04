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
# The registry of mappings from Torch node to MLIR linalg dialect operations.
#
# ===---------------------------------------------------------------------------

from typing import Dict, Tuple, List

import torch

import mlir.ir as ir
from mlir.dialects import tosa, linalg, arith, tensor, math
import copy
import numpy
import functools


def arange_op(
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import tensor arange operation.
    From PyTorch `aten.arange.default` and `aten.arange.start` operator to MLIR
    arith `constant` operation.

    Note: this function init an output tensor according input range.

    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation representing the result tensor of ranging the start
        and end from input node.
    """
    if node.target.__name__ == "arange.start":
        start = int(node.args[0])
        end = int(node.args[1])
        stride = int(node.meta["tensor_meta"].stride[0])
        dtype = str(node.meta["tensor_meta"].dtype)
        shape = list(node.meta["tensor_meta"].shape)
        dtype = ir.IntegerType.get_signless(64)
        tensor_type = ir.RankedTensorType.get(shape, dtype)
        attr = ir.DenseElementsAttr.get(
            numpy.array([i for i in range(start, end, stride)]),
            signless=True,
            type=tensor_type,
        )
        op = arith.ConstantOp(tensor_type, attr)

    elif node.target.__name__ == "arange.default":
        start = 0
        end = int(node.args[0])
        stride = int(node.meta["tensor_meta"].stride[0])
        dtype = str(node.meta["tensor_meta"].dtype)
        shape = list(node.meta["tensor_meta"].shape)
        dtype = ir.IntegerType.get_signless(64)
        tensor_type = ir.RankedTensorType.get(shape, dtype)
        attr = ir.DenseElementsAttr.get(
            numpy.array([i for i in range(start, end, stride)]),
            signless=True,
            type=tensor_type,
        )
        op = arith.ConstantOp(tensor_type, attr)

    return op


def unsqueeze_op(
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the unsqueeze operation.
    From PyTorch `aten.unsqueeze.default` operator to MLIR TOSA `reshape`
    operation.

    Note: "unsqueeze" means inserting a new dimension of size 1 at the specified
          position. For more information, please refer to
          https://pytorch.org/docs/stable/generated/torch.unsqueeze.html

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
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor view operation.
    From PyTorch `aten.view.default` operator to MLIR TOSA `reshape` operation.

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
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the embedding operation.
    From PyTorch `aten.embedding.default` operator to MLIR linalg `generic`
    operation.

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
    output_shape = list(node.meta["tensor_meta"].shape)
    dtype = str(node.meta["tensor_meta"].dtype)
    if dtype == "torch.float32":
        tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
        output = tensor.EmptyOp(output_shape, ir.F32Type.get())
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
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor ones operation.
    From PyTorch `aten.ones.default` operator to MLIR arith `constant`
    operation.

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
    dtype = str(node.meta["tensor_meta"].dtype)
    if dtype == "torch.bool":
        element = ir.BoolAttr.get(1)
        tensor_type = ir.RankedTensorType.get(output_shape, element.type)
        attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
    elif dtype == "torch.int64":
        dtype = ir.IntegerType.get_signless(64)
        tensor_type = ir.RankedTensorType.get(output_shape, dtype)
        attr = ir.DenseElementsAttr.get(
            numpy.ones(output_shape), signless=True, type=tensor_type
        )
    op = arith.ConstantOp(tensor_type, attr)

    return op


def full_op(
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor full operation.
    From PyTorch `aten.full.default` operator to MLIR arith `constant`
    operation.

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
    dtype = str(node.meta["tensor_meta"].dtype)
    if dtype == "torch.bool":
        element = ir.BoolAttr.get(bool(value))
        tensor_type = ir.RankedTensorType.get(output_shape, element.type)
        attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
    elif dtype == "torch.int64":
        dtype = ir.IntegerType.get_signless(64)
        tensor_type = ir.RankedTensorType.get(output_shape, dtype)
        attr = ir.DenseElementsAttr.get(
            numpy.full(output_shape, value, dtype=numpy.int64),
            signless=True,
            type=tensor_type,
        )
    elif dtype == "torch.float32":
        dtype = ir.F32Type.get()
        tensor_type = ir.RankedTensorType.get(output_shape, dtype)
        attr = ir.DenseElementsAttr.get(
            numpy.full(output_shape, value, dtype=numpy.float32),
            signless=True,
            type=tensor_type,
        )
    op = arith.ConstantOp(tensor_type, attr)

    return op


def lt_op(
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor less than operation.
    From PyTorch `aten.lt.Tensor` operator to MLIR arith `constant` operation.

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
    output_shape = list(node.meta["tensor_meta"].shape)
    dtype = str(node.meta["tensor_meta"].dtype)
    value = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 2)
    shp1 = list(ir.RankedTensorType(ir.Value(input1).type).shape)
    shp2 = list(ir.RankedTensorType(ir.Value(input2).type).shape)
    if dtype == "torch.bool":
        tensor_type = ir.RankedTensorType.get(
            output_shape, ir.IntegerType.get_signless(1)
        )
        output = tensor.EmptyOp(output_shape, ir.IntegerType.get_signless(1))
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
                        + [
                            ir.Attribute.parse(
                                "#linalg.iterator_type<reduction>"
                            )
                        ]
                    ),
                )
                block = ir.Block.create_at_start(
                    op.region,
                    [
                        ir.RankedTensorType(input2.type).element_type,
                        ir.RankedTensorType(input2.type).element_type,
                        ir.IntegerType.get_signless(1),
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
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor masked fill operation.
    From PyTorch `aten.masked_fill.Scalar` operator to MLIR linalg `generic`
    operation.

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
    if str(node.args[0].meta["tensor_meta"].dtype) == "torch.float32":
        value = float(node.args[2])
        attr = ir.FloatAttr.get(ir.F32Type.get(), value)
        value = arith.ConstantOp(ir.F32Type.get(), attr)
    output_shape = list(node.meta["tensor_meta"].shape)
    dtype = str(node.meta["tensor_meta"].dtype)
    if dtype == "torch.float32":
        tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
        output = tensor.EmptyOp(output_shape, ir.F32Type.get())
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
        select_op = arith.SelectOp(
            block.arguments[1], value, block.arguments[0]
        )
        block.append(select_op)
        block.append(linalg.YieldOp([select_op.result]))

    return op


def slice_op(
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor slice operation.
    From PyTorch `aten.slice.Tensor` operator to MLIR tensor `extract_slice`
    operation.

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
    output_shape = list(node.meta["tensor_meta"].shape)
    size_attr = ir._denseI64ArrayAttr(output_shape, None)
    stride = [1 for x in output_shape]
    stride[dim] = step
    stride_attr = ir._denseI64ArrayAttr(stride, None)
    dtype = str(node.meta["tensor_meta"].dtype)
    if dtype == "torch.float32":
        tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
    if dtype == "torch.bool":
        tensor_type = ir.RankedTensorType.get(
            output_shape, ir.IntegerType.get_signless(1)
        )

    op = tensor.ExtractSliceOp(
        tensor_type, input1, [], [], [], offset_attr, size_attr, stride_attr
    )

    return op


def expand_op(
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor expand operation.
    From PyTorch `aten.expand.default` operator to MLIR tensor `extract_slice`
    operation.

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
    output_shape = list(node.meta["tensor_meta"].shape)
    dtype = str(node.meta["tensor_meta"].dtype)
    if dtype == "torch.bool":
        empty_tensor = tensor.EmptyOp(
            output_shape, ir.IntegerType.get_signless(1)
        )
    elif dtype == "torch.float32":
        empty_tensor = tensor.EmptyOp(output_shape, ir.F32Type.get())
    if list(input_shape) == list(node.args[1]):
        offset_attr = ir._denseI64ArrayAttr([0 for x in input_shape], None)
        size_attr = ir._denseI64ArrayAttr(output_shape, None)
        stride_attr = ir._denseI64ArrayAttr([1 for x in input_shape], None)
        if dtype == "torch.bool":
            tensor_type = ir.RankedTensorType.get(
                output_shape, ir.IntegerType.get_signless(1)
            )
        elif dtype == "torch.float32":
            tensor_type = ir.RankedTensorType.get(
                output_shape, ir.F32Type.get()
            )
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
                    if dtype == "torch.bool":
                        tensor_type = ir.RankedTensorType.get(
                            [1] * (i + 1) + [x for x in output_shape[i + 1 :]],
                            ir.IntegerType.get_signless(1),
                        )
                    elif dtype == "torch.float32":
                        tensor_type = ir.RankedTensorType.get(
                            [1] * (i + 1) + [x for x in output_shape[i + 1 :]],
                            ir.F32Type.get(),
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
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor copy operation.
    From PyTorch `aten._to_copy.default` operator to MLIR linalg `generic`
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
    output_shape = list(node.meta["tensor_meta"].shape)
    dtype = str(node.meta["tensor_meta"].dtype)

    if dtype == "torch.bool":
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
    elif dtype == "torch.float32":
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
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor rsub operation.
    From PyTorch `aten.rsub.Scalar` operator to MLIR linalg `generic` operation.

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
    output_shape = list(node.meta["tensor_meta"].shape)
    dtype = str(node.meta["tensor_meta"].dtype)
    if not isinstance(value, torch.fx.Node):
        if dtype == "torch.float32":
            value = arith.ConstantOp(
                ir.F32Type.get(), ir.FloatAttr.get(ir.F32Type.get(), value)
            )
            generic_map = ir.AffineMap.get_permutation(
                [i for i in range(len(output_shape))]
            )
            tensor_type = ir.RankedTensorType.get(
                output_shape, ir.F32Type.get()
            )
            output = tensor.EmptyOp(output_shape, ir.F32Type.get())
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
            subf_op = arith.SubFOp(value.result, block.arguments[0])
            block.append(subf_op)
            block.append(linalg.YieldOp([subf_op.result]))

    return op


def pow_op(
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor copy operation.
    From PyTorch `aten.pow.Tensor_Scalar` operator to MLIR linalg `generic`
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
    output_shape = list(node.meta["tensor_meta"].shape)
    dtype = str(node.meta["tensor_meta"].dtype)
    if not isinstance(value, torch.fx.Node):
        if dtype == "torch.float32":
            generic_map = ir.AffineMap.get_permutation(
                [i for i in range(len(output_shape))]
            )
            tensor_type = ir.RankedTensorType.get(
                output_shape, ir.F32Type.get()
            )
            output = tensor.EmptyOp(output_shape, ir.F32Type.get())
            if abs(int(value) - float(value)) < 1e-6:
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
                fpowi_op = math.FPowIOp(block.arguments[0], value.result)
                block.append(fpowi_op)
                block.append(linalg.YieldOp([fpowi_op.result]))

    return op


def mean_op(
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor copy operation.
    From PyTorch `aten.mean.dim` operator to MLIR linalg `generic` operation.

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
    output_shape = list(node.meta["tensor_meta"].shape)
    dtype = str(node.meta["tensor_meta"].dtype)
    if dtype == "torch.float32":
        tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
        element = ir.FloatAttr.get(ir.F32Type.get(), 0.0)
        attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
        output = arith.ConstantOp(tensor_type, attr)

        assert len(dims) == 1

        for dim in dims:
            if dim == -1:
                dim = len(list(ir.RankedTensorType(input1.type).shape)) - 1
            if keep_dim:
                generic_map = ir.AffineMap.get_permutation(
                    [i for i in range(len(output_shape) + 1)]
                )
                tensor_type = ir.RankedTensorType.get(
                    output_shape, ir.F32Type.get()
                )
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
                    ir.F32Type.get(),
                    ir.FloatAttr.get(
                        ir.F32Type.get(),
                        list(ir.RankedTensorType(input1.type).shape)[dim],
                    ),
                )
                divf_op = arith.DivFOp(block.arguments[0], value.result)
                addf_op = arith.AddFOp(divf_op.result, block.arguments[1])
                block.append(value)
                block.append(divf_op)
                block.append(addf_op)
                block.append(linalg.YieldOp([addf_op.result]))

    return op


def rsqrt_op(
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor rsqrt operation.
    From PyTorch `aten.rsqrt.default` operator to MLIR linalg `generic`
    operation.

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

    output_shape = list(node.meta["tensor_meta"].shape)
    dtype = str(node.meta["tensor_meta"].dtype)

    if dtype == "torch.float32":
        tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
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
        math_rsqrt_op = math.RsqrtOp(block.arguments[0])
        block.append(math_rsqrt_op)
        block.append(linalg.YieldOp([math_rsqrt_op.result]))

    return op


def mul_op(
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor mul operation.
    From PyTorch `aten.mul.Tensor` operator to MLIR linalg `generic` operation.

    Note: This op, compute input node's mul result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    assert len(node.args) == 2
    if isinstance(node.args[0], torch.fx.Node):
        input1 = symbol_table.get((str(node.args[0]), 0))
    else:
        input1 = node.args[0]

    if isinstance(node.args[1], torch.fx.Node):
        input2 = symbol_table.get((str(node.args[1]), 0))
    else:
        input2 = node.args[1]

    if input1 is None or input2 is None:
        return

    output_shape = list(node.meta["tensor_meta"].shape)
    dtype = str(node.meta["tensor_meta"].dtype)

    if isinstance(node.args[0], torch.fx.Node):
        if dtype == "torch.float32":
            if not isinstance(node.args[1], torch.fx.Node):
                input2 = arith.ConstantOp(
                    ir.F32Type.get(), ir.FloatAttr.get(ir.F32Type.get(), input2)
                )
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
                mulf_op = arith.MulFOp(block.arguments[0], input2.result)
                block.append(mulf_op)
                block.append(linalg.YieldOp([mulf_op.result]))
            else:
                tensor_type = ir.RankedTensorType.get(
                    output_shape, ir.F32Type.get()
                )
                output = tensor.EmptyOp(output_shape, ir.F32Type.get())
                input1_shape = list(ir.RankedTensorType(input1.type).shape)
                if input1_shape != output_shape:
                    dims = []
                    for i in range(len(input1_shape) - 1, -1, -1):
                        if (
                            input1_shape[i]
                            != output_shape[
                                len(output_shape) - (len(input1_shape) - i)
                            ]
                        ):
                            dims.append(i)
                    output1 = tensor.EmptyOp(output_shape, ir.F32Type.get())
                    generic_map = ir.AffineMap.get_permutation(
                        [i for i in range(len(output_shape) + len(dims))]
                    )
                    input1_map = [
                        i
                        for i in range(
                            len(output_shape) - len(input1_shape),
                            len(output_shape),
                        )
                    ]
                    for index, i in enumerate(dims):
                        input1_map[i] = len(output_shape) + index
                    input1_map = generic_map.get_submap(input1_map)
                    input1_op = linalg.GenericOp(
                        [tensor_type],
                        [input1],
                        [output1],
                        ir.ArrayAttr.get(
                            [
                                ir.AffineMapAttr.get(input1_map),
                                ir.AffineMapAttr.get(
                                    generic_map.get_submap(
                                        [i for i in range(len(output_shape))]
                                    )
                                ),
                            ]
                        ),
                        ir.ArrayAttr.get(
                            [
                                ir.Attribute.parse(
                                    "#linalg.iterator_type<parallel>"
                                )
                            ]
                            * len(output_shape)
                            + [
                                ir.Attribute.parse(
                                    "#linalg.iterator_type<reduction>"
                                )
                            ]
                            * len(dims)
                        ),
                    )
                    block = ir.Block.create_at_start(
                        input1_op.region,
                        [
                            ir.RankedTensorType(input1.type).element_type,
                            ir.RankedTensorType(
                                output.result.type
                            ).element_type,
                        ],
                    )
                    block.append(linalg.YieldOp([block.arguments[0]]))
                    input1 = input1_op.result

                input2_shape = list(ir.RankedTensorType(input2.type).shape)
                if input2_shape != output_shape:
                    dims = []
                    for i in range(len(input2_shape) - 1, -1, -1):
                        if (
                            input2_shape[i]
                            != output_shape[
                                len(output_shape) - (len(input2_shape) - i)
                            ]
                        ):
                            dims.append(i)
                    output2 = tensor.EmptyOp(output_shape, ir.F32Type.get())
                    generic_map = ir.AffineMap.get_permutation(
                        [i for i in range(len(output_shape) + len(dims))]
                    )
                    input2_map = [
                        i
                        for i in range(
                            len(output_shape) - len(input2_shape),
                            len(output_shape),
                        )
                    ]
                    for index, i in enumerate(dims):
                        input2_map[i] = len(output_shape) + index
                    input2_map = generic_map.get_submap(input2_map)
                    input2_op = linalg.GenericOp(
                        [tensor_type],
                        [input2],
                        [output2],
                        ir.ArrayAttr.get(
                            [
                                ir.AffineMapAttr.get(input2_map),
                                ir.AffineMapAttr.get(
                                    generic_map.get_submap(
                                        [i for i in range(len(output_shape))]
                                    )
                                ),
                            ]
                        ),
                        ir.ArrayAttr.get(
                            [
                                ir.Attribute.parse(
                                    "#linalg.iterator_type<parallel>"
                                )
                            ]
                            * len(output_shape)
                            + [
                                ir.Attribute.parse(
                                    "#linalg.iterator_type<reduction>"
                                )
                            ]
                            * len(dims)
                        ),
                    )
                    block = ir.Block.create_at_start(
                        input2_op.region,
                        [
                            ir.RankedTensorType(input2.type).element_type,
                            ir.RankedTensorType(
                                output.result.type
                            ).element_type,
                        ],
                    )
                    block.append(linalg.YieldOp([block.arguments[0]]))
                    input2 = input2_op.result
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
                mulf_op = arith.MulFOp(block.arguments[0], block.arguments[1])
                block.append(mulf_op)
                block.append(linalg.YieldOp([mulf_op.result]))

    return op


def t_op(
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor tanspose operation.
    From PyTorch `aten.t.default` operator to MLIR linalg `generic` operation.

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

    input_shape = list(ir.RankedTensorType(input1.type).shape)
    output_shape = list(node.meta["tensor_meta"].shape)
    dtype = str(node.meta["tensor_meta"].dtype)
    if len(input_shape) == 2:
        if dtype == "torch.float32":
            tensor_type = ir.RankedTensorType.get(
                output_shape, ir.F32Type.get()
            )
            output = tensor.EmptyOp(output_shape, ir.F32Type.get())
            generic_map = ir.AffineMap.get_permutation([0, 1])
            op = linalg.GenericOp(
                [tensor_type],
                [input1],
                [output],
                ir.ArrayAttr.get(
                    [
                        ir.AffineMapAttr.get(generic_map.get_submap([0, 1])),
                        ir.AffineMapAttr.get(generic_map.get_submap([1, 0])),
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


def matmul_op(
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor matmul operation.
    From PyTorch `aten.mm.default` operator to MLIR linalg `matmul` operation.

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

    output_shape = list(node.meta["tensor_meta"].shape)
    dtype = str(node.meta["tensor_meta"].dtype)
    if dtype == "torch.float32":
        tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
        f32 = ir.F32Type.get()
        element = ir.FloatAttr.get(f32, 0.0)
        attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
        matmul_result_buffer = arith.ConstantOp(tensor_type, attr).result
        op = linalg.matmul(input1, input2, outs=[matmul_result_buffer])
    return op


def transpose_op(
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor transpose operation.
    From PyTorch `aten.transpose.int` operator to MLIR linalg `generic`
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
    output_shape = list(node.meta["tensor_meta"].shape)
    dtype = str(node.meta["tensor_meta"].dtype)
    if dtype == "torch.float32":
        tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
        output = tensor.EmptyOp(output_shape, ir.F32Type.get())
        generic_map = ir.AffineMap.get_permutation(
            [i for i in range(len(output_shape))]
        )
        input1_map = [i for i in range(len(output_shape))]
        input1_map[dim1], input1_map[dim2] = input1_map[dim2], input1_map[dim1]
        output_map = [i for i in range(len(output_shape))]
        op = linalg.GenericOp(
            [tensor_type],
            [input1],
            [output],
            ir.ArrayAttr.get(
                [
                    ir.AffineMapAttr.get(generic_map.get_submap(input1_map)),
                    ir.AffineMapAttr.get(generic_map.get_submap(output_map)),
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


def index_op(
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor index operation.
    From PyTorch `aten.index.Tensor` operator to MLIR linalg `generic`
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
    output_shape = list(node.meta["tensor_meta"].shape)
    dtype = str(node.meta["tensor_meta"].dtype)
    if len(input2) < len(input1_shape):
        if dtype == "torch.float32":
            tensor_type = ir.RankedTensorType.get(
                output_shape, ir.F32Type.get()
            )
            output = tensor.EmptyOp(output_shape, ir.F32Type.get())
            loops = ir.RankedTensorType(
                symbol_table.get((str(input2[0]), 0)).type
            ).shape
            generic_map = ir.AffineMap.get_permutation(
                [i for i in range(len(output_shape))]
            )
            input_map = [
                ir.AffineMapAttr.get(
                    generic_map.get_submap([j for j in range(len(loops))])
                )
                for i in range(len(input2))
            ] + [
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [j for j in range(len(output_shape))]
                    )
                )
            ]
            operands = [symbol_table.get((str(i), 0)) for i in input2]
            op = linalg.GenericOp(
                [tensor_type],
                operands,
                [output],
                ir.ArrayAttr.get(input_map),
                ir.ArrayAttr.get(
                    [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
                    * len(output_shape)
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
            for i in range(len(loops), len(output_shape) - len(input2) + 1):
                index_op = linalg.IndexOp(ir._i64Attr(i, None))
                block.append(index_op)
                index.append(index_op.result)
            value = tensor.ExtractOp(input1, index)
            block.append(value)
            block.append(linalg.YieldOp([value.result]))

    return op


def neg_op(
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor neg operation.
    From PyTorch `aten.neg.default` operator to MLIR linalg `matmul` operation.

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

    output_shape = list(node.meta["tensor_meta"].shape)
    dtype = str(node.meta["tensor_meta"].dtype)
    if dtype == "torch.float32":
        tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
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
        negf_op = arith.NegFOp(block.arguments[0])
        block.append(negf_op)
        block.append(linalg.YieldOp([negf_op.result]))

    return op


def cat_op(
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor concate operation.
    From PyTorch `aten.cat.default` operator to MLIR tensor `insert_slice`
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

    output_shape = list(node.meta["tensor_meta"].shape)
    if dim < 0:
        dim = len(output_shape) + dim
    dtype = str(node.meta["tensor_meta"].dtype)
    if dtype == "torch.float32":
        output = tensor.EmptyOp(output_shape, ir.F32Type.get())
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
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor squeeze operation.
    From PyTorch `aten.squeeze.dim` operator to MLIR linalg `generic` operation.

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

    output_shape = list(node.meta["tensor_meta"].shape)
    input1_shape = ir.RankedTensorType(input1.type).shape
    if dim < 0:
        dim = len(input1_shape) + dim
    dtype = str(node.meta["tensor_meta"].dtype)
    if dtype == "torch.float32":
        tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
        output = tensor.EmptyOp(output_shape, ir.F32Type.get())
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
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor batch matmul operation.
    From PyTorch `aten.bmm.default` operator to MLIR linalg `batch_matmul`
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

    output_shape = list(node.meta["tensor_meta"].shape)
    dtype = str(node.meta["tensor_meta"].dtype)
    if dtype == "torch.float32":
        tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
        output = tensor.EmptyOp(output_shape, ir.F32Type.get())
        # use linalg.generic implementation
        generic_map = ir.AffineMap.get_permutation([0, 1, 2])
        zero_fill = linalg.GenericOp(
            [tensor_type],
            [],
            [output],
            ir.ArrayAttr.get(
                [ir.AffineMapAttr.get(generic_map.get_submap([0, 1, 2]))]
            ),
            ir.ArrayAttr.get(
                [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * 3
            ),
        )
        block = ir.Block.create_at_start(
            zero_fill.region,
            [ir.RankedTensorType(output.result.type).element_type],
        )
        zero_op = arith.ConstantOp(
            ir.F32Type.get(), ir.FloatAttr.get(ir.F32Type.get(), 0)
        )
        block.append(zero_op)
        block.append(linalg.YieldOp([zero_op.result]))
        op = linalg.batch_matmul(input1, input2, outs=[zero_fill.result])

    return op


def div_op(
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor divsion operation.
    From PyTorch `aten.div.Tensor` operator to MLIR linalg `generic` operation.

    Note: This op, compute input node's division result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    assert len(node.args) == 2
    if isinstance(node.args[0], torch.fx.Node):
        input1 = symbol_table.get((str(node.args[0]), 0))
    else:
        input1 = node.args[0]

    if isinstance(node.args[1], torch.fx.Node):
        input2 = symbol_table.get((str(node.args[1]), 0))
    else:
        input2 = node.args[1]

    if input1 is None or input2 is None:
        return

    output_shape = list(node.meta["tensor_meta"].shape)
    dtype = str(node.meta["tensor_meta"].dtype)

    if isinstance(node.args[0], torch.fx.Node):
        if dtype == "torch.float32":
            if not isinstance(node.args[1], torch.fx.Node):
                input2 = arith.ConstantOp(
                    ir.F32Type.get(), ir.FloatAttr.get(ir.F32Type.get(), input2)
                )
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
                divf_op = arith.DivFOp(block.arguments[0], input2.result)
                block.append(divf_op)
                block.append(linalg.YieldOp([divf_op.result]))
            else:
                tensor_type = ir.RankedTensorType.get(
                    output_shape, ir.F32Type.get()
                )
                output = tensor.EmptyOp(output_shape, ir.F32Type.get())
                input1_shape = list(ir.RankedTensorType(input1.type).shape)
                if input1_shape != output_shape:
                    dims = []
                    for i in range(len(input1_shape) - 1, -1, -1):
                        if (
                            input1_shape[i]
                            != output_shape[
                                len(output_shape) - (len(input1_shape) - i)
                            ]
                        ):
                            dims.append(i)
                    output1 = tensor.EmptyOp(output_shape, ir.F32Type.get())
                    generic_map = ir.AffineMap.get_permutation(
                        [i for i in range(len(output_shape) + len(dims))]
                    )
                    input1_map = [
                        i
                        for i in range(
                            len(output_shape) - len(input1_shape),
                            len(output_shape),
                        )
                    ]
                    for index, i in enumerate(dims):
                        input1_map[i] = len(output_shape) + index
                    input1_map = generic_map.get_submap(input1_map)
                    input1_op = linalg.GenericOp(
                        [tensor_type],
                        [input1],
                        [output1],
                        ir.ArrayAttr.get(
                            [
                                ir.AffineMapAttr.get(input1_map),
                                ir.AffineMapAttr.get(
                                    generic_map.get_submap(
                                        [i for i in range(len(output_shape))]
                                    )
                                ),
                            ]
                        ),
                        ir.ArrayAttr.get(
                            [
                                ir.Attribute.parse(
                                    "#linalg.iterator_type<parallel>"
                                )
                            ]
                            * len(output_shape)
                            + [
                                ir.Attribute.parse(
                                    "#linalg.iterator_type<reduction>"
                                )
                            ]
                            * len(dims)
                        ),
                    )
                    block = ir.Block.create_at_start(
                        input1_op.region,
                        [
                            ir.RankedTensorType(input1.type).element_type,
                            ir.RankedTensorType(
                                output.result.type
                            ).element_type,
                        ],
                    )
                    block.append(linalg.YieldOp([block.arguments[0]]))
                    input1 = input1_op.result

                input2_shape = list(ir.RankedTensorType(input2.type).shape)
                if input2_shape != output_shape:
                    dims = []
                    for i in range(len(input2_shape) - 1, -1, -1):
                        if (
                            input2_shape[i]
                            != output_shape[
                                len(output_shape) - (len(input2_shape) - i)
                            ]
                        ):
                            dims.append(i)
                    output2 = tensor.EmptyOp(output_shape, ir.F32Type.get())
                    generic_map = ir.AffineMap.get_permutation(
                        [i for i in range(len(output_shape) + len(dims))]
                    )
                    input2_map = [
                        i
                        for i in range(
                            len(output_shape) - len(input2_shape),
                            len(output_shape),
                        )
                    ]
                    for index, i in enumerate(dims):
                        input2_map[i] = len(output_shape) + index
                    input2_map = generic_map.get_submap(input2_map)
                    input2_op = linalg.GenericOp(
                        [tensor_type],
                        [input2],
                        [output2],
                        ir.ArrayAttr.get(
                            [
                                ir.AffineMapAttr.get(input2_map),
                                ir.AffineMapAttr.get(
                                    generic_map.get_submap(
                                        [i for i in range(len(output_shape))]
                                    )
                                ),
                            ]
                        ),
                        ir.ArrayAttr.get(
                            [
                                ir.Attribute.parse(
                                    "#linalg.iterator_type<parallel>"
                                )
                            ]
                            * len(output_shape)
                            + [
                                ir.Attribute.parse(
                                    "#linalg.iterator_type<reduction>"
                                )
                            ]
                            * len(dims)
                        ),
                    )
                    block = ir.Block.create_at_start(
                        input2_op.region,
                        [
                            ir.RankedTensorType(input2.type).element_type,
                            ir.RankedTensorType(
                                output.result.type
                            ).element_type,
                        ],
                    )
                    block.append(linalg.YieldOp([block.arguments[0]]))
                    input2 = input2_op.result
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
                divf_op = arith.DivFOp(block.arguments[0], block.arguments[1])
                block.append(divf_op)
                block.append(linalg.YieldOp([divf_op.result]))

    return op


def softmax_op(
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor softmax operation.
    From PyTorch `aten._softmax.default` operator to MLIR linalg `generic`
    operation.

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
    output_shape = list(node.meta["tensor_meta"].shape)
    dtype = str(node.meta["tensor_meta"].dtype)
    if dim < 0:
        dim += len(output_shape)
    if dtype == "torch.float32":
        max_tensor_shape = copy.deepcopy(output_shape)
        max_tensor_shape[dim] = 1
        max_tensor_type = ir.RankedTensorType.get(
            max_tensor_shape, ir.F32Type.get()
        )
        max_tensor = tensor.EmptyOp(max_tensor_shape, ir.F32Type.get())
        max_tensor_map = [
            ir.AffineExpr.get_dim(i) for i in range(len(max_tensor_shape))
        ]
        max_tensor_map = ir.AffineMap.get(
            len(max_tensor_shape), 0, max_tensor_map
        )
        neg_inf_fill = linalg.GenericOp(
            [max_tensor_type],
            [],
            [max_tensor],
            ir.ArrayAttr.get([ir.AffineMapAttr.get(max_tensor_map)]),
            ir.ArrayAttr.get(
                [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
                * len(max_tensor_shape)
            ),
        )
        block = ir.Block.create_at_start(
            neg_inf_fill.region,
            [ir.RankedTensorType(max_tensor.result.type).element_type],
        )
        neg_inf_op = arith.ConstantOp(
            ir.F32Type.get(), ir.FloatAttr.get(ir.F32Type.get(), float("-inf"))
        )
        block.append(neg_inf_op)
        block.append(linalg.YieldOp([neg_inf_op.result]))

        input1_map = [
            ir.AffineExpr.get_dim(i) for i in range(len(output_shape))
        ]
        input1_map = ir.AffineMap.get(len(output_shape), 0, input1_map)
        max_tensor_map = [
            ir.AffineExpr.get_dim(i) for i in range(len(output_shape))
        ]
        max_tensor_map[dim] = ir.AffineExpr.get_constant(0)
        max_tensor_map = ir.AffineMap.get(len(output_shape), 0, max_tensor_map)
        loop_type = [
            ir.Attribute.parse("#linalg.iterator_type<parallel>")
        ] * len(output_shape)
        loop_type[dim] = ir.Attribute.parse("#linalg.iterator_type<reduction>")
        max_tensor_op = linalg.GenericOp(
            [max_tensor_type],
            [input1],
            [neg_inf_fill],
            ir.ArrayAttr.get(
                [
                    ir.AffineMapAttr.get(input1_map),
                    ir.AffineMapAttr.get(max_tensor_map),
                ]
            ),
            ir.ArrayAttr.get(loop_type),
        )
        block = ir.Block.create_at_start(
            max_tensor_op.region,
            [
                ir.RankedTensorType(input1.type).element_type,
                ir.RankedTensorType(neg_inf_fill.result.type).element_type,
            ],
        )
        max_op = arith.MaximumFOp(block.arguments[0], block.arguments[1])
        block.append(max_op)
        block.append(linalg.YieldOp([max_op.result]))

        exp_tensor = tensor.EmptyOp(output_shape, ir.F32Type.get())
        exp_tensor_type = ir.RankedTensorType.get(
            output_shape, ir.F32Type.get()
        )
        input1_map = [
            ir.AffineExpr.get_dim(i) for i in range(len(output_shape))
        ]
        input1_map = ir.AffineMap.get(len(output_shape), 0, input1_map)
        max_tensor_map = [
            ir.AffineExpr.get_dim(i) for i in range(len(output_shape))
        ]
        max_tensor_map[dim] = ir.AffineExpr.get_constant(0)
        max_tensor_map = ir.AffineMap.get(len(output_shape), 0, max_tensor_map)
        exp_tensor_map = [
            ir.AffineExpr.get_dim(i) for i in range(len(output_shape))
        ]
        exp_tensor_map = ir.AffineMap.get(len(output_shape), 0, exp_tensor_map)
        exp_tensor_op = linalg.GenericOp(
            [exp_tensor_type],
            [input1, max_tensor_op.result],
            [exp_tensor],
            ir.ArrayAttr.get(
                [
                    ir.AffineMapAttr.get(input1_map),
                    ir.AffineMapAttr.get(max_tensor_map),
                    ir.AffineMapAttr.get(exp_tensor_map),
                ]
            ),
            ir.ArrayAttr.get(
                [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
                * len(output_shape)
            ),
        )
        block = ir.Block.create_at_start(
            exp_tensor_op.region,
            [
                ir.RankedTensorType(input1.type).element_type,
                ir.RankedTensorType(max_tensor_op.result.type).element_type,
                ir.RankedTensorType(exp_tensor.result.type).element_type,
            ],
        )
        sub_op = arith.SubFOp(block.arguments[0], block.arguments[1])
        exp_op = math.ExpOp(sub_op.result)
        block.append(sub_op)
        block.append(exp_op)
        block.append(linalg.YieldOp([exp_op.result]))

        reduce_sum_tensor_shape = copy.deepcopy(output_shape)
        reduce_sum_tensor_shape[dim] = 1
        reduce_sum_tensor = tensor.EmptyOp(
            reduce_sum_tensor_shape, ir.F32Type.get()
        )
        reduce_sum_tensor_type = ir.RankedTensorType.get(
            reduce_sum_tensor_shape, ir.F32Type.get()
        )
        reduce_sum_tensor_map = [
            ir.AffineExpr.get_dim(i) for i in range(len(output_shape))
        ]
        reduce_sum_tensor_map = ir.AffineMap.get(
            len(output_shape), 0, reduce_sum_tensor_map
        )
        zero_fill_op = linalg.GenericOp(
            [reduce_sum_tensor_type],
            [],
            [reduce_sum_tensor.result],
            ir.ArrayAttr.get([ir.AffineMapAttr.get(reduce_sum_tensor_map)]),
            ir.ArrayAttr.get(
                [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
                * len(output_shape)
            ),
        )
        block = ir.Block.create_at_start(
            zero_fill_op.region,
            [ir.RankedTensorType(reduce_sum_tensor.result.type).element_type],
        )
        zero_op = arith.ConstantOp(
            ir.F32Type.get(), ir.FloatAttr.get(ir.F32Type.get(), 0)
        )
        block.append(zero_op)
        block.append(linalg.YieldOp([zero_op.result]))

        reduce_sum_tensor_shape = copy.deepcopy(output_shape)
        reduce_sum_tensor_shape[dim] = 1
        reduce_sum_tensor_type = ir.RankedTensorType.get(
            reduce_sum_tensor_shape, ir.F32Type.get()
        )
        exp_tensor_map = [
            ir.AffineExpr.get_dim(i) for i in range(len(output_shape))
        ]
        exp_tensor_map = ir.AffineMap.get(len(output_shape), 0, exp_tensor_map)
        reduce_sum_tensor_map = [
            ir.AffineExpr.get_dim(i) for i in range(len(output_shape))
        ]
        reduce_sum_tensor_map[dim] = ir.AffineExpr.get_constant(0)
        reduce_sum_tensor_map = ir.AffineMap.get(
            len(output_shape), 0, reduce_sum_tensor_map
        )
        loop_type = [
            ir.Attribute.parse("#linalg.iterator_type<parallel>")
        ] * len(output_shape)
        loop_type[dim] = ir.Attribute.parse("#linalg.iterator_type<reduction>")
        reduce_sum_tensor_op = linalg.GenericOp(
            [reduce_sum_tensor_type],
            [exp_tensor_op.result],
            [zero_fill_op.result],
            ir.ArrayAttr.get(
                [
                    ir.AffineMapAttr.get(exp_tensor_map),
                    ir.AffineMapAttr.get(reduce_sum_tensor_map),
                ]
            ),
            ir.ArrayAttr.get(loop_type),
        )
        block = ir.Block.create_at_start(
            reduce_sum_tensor_op.region,
            [
                ir.RankedTensorType(exp_tensor_op.result.type).element_type,
                ir.RankedTensorType(zero_fill_op.result.type).element_type,
            ],
        )
        add_op = arith.AddFOp(block.arguments[0], block.arguments[1])
        block.append(add_op)
        block.append(linalg.YieldOp([add_op.result]))

        reduce_sum_tensor_shape = copy.deepcopy(output_shape)
        reduce_sum_tensor_shape[dim] = 1
        result_tensor_type = ir.RankedTensorType.get(
            output_shape, ir.F32Type.get()
        )
        result_tensor = tensor.EmptyOp(output_shape, ir.F32Type.get())
        exp_tensor_map = [
            ir.AffineExpr.get_dim(i) for i in range(len(output_shape))
        ]
        exp_tensor_map = ir.AffineMap.get(len(output_shape), 0, exp_tensor_map)
        reduce_sum_tensor_map = [
            ir.AffineExpr.get_dim(i) for i in range(len(output_shape))
        ]
        reduce_sum_tensor_map[dim] = ir.AffineExpr.get_constant(0)
        reduce_sum_tensor_map = ir.AffineMap.get(
            len(output_shape), 0, reduce_sum_tensor_map
        )
        result_tensor_map = [
            ir.AffineExpr.get_dim(i) for i in range(len(output_shape))
        ]
        result_tensor_map = ir.AffineMap.get(
            len(output_shape), 0, result_tensor_map
        )
        op = linalg.GenericOp(
            [result_tensor_type],
            [exp_tensor_op.result, reduce_sum_tensor_op.result],
            [result_tensor.result],
            ir.ArrayAttr.get(
                [
                    ir.AffineMapAttr.get(exp_tensor_map),
                    ir.AffineMapAttr.get(reduce_sum_tensor_map),
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
                ir.RankedTensorType(exp_tensor_op.result.type).element_type,
                ir.RankedTensorType(
                    reduce_sum_tensor_op.result.type
                ).element_type,
                ir.RankedTensorType(result_tensor.result.type).element_type,
            ],
        )
        div_op = arith.DivFOp(block.arguments[0], block.arguments[1])
        block.append(div_op)
        block.append(linalg.YieldOp([div_op.result]))

    return op


def clone_op(
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor clone operation.
    From PyTorch `aten.clone.default` operator to MLIR tensor `extract_slice`
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

    output_shape = list(node.meta["tensor_meta"].shape)
    dtype = str(node.meta["tensor_meta"].dtype)
    if dtype == "torch.float32":
        offset = [0 for x in output_shape]
        offset_attr = ir._denseI64ArrayAttr(offset, None)
        size_attr = ir._denseI64ArrayAttr(output_shape, None)
        stride = [1 for x in output_shape]
        stride_attr = ir._denseI64ArrayAttr(stride, None)
        tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())

        op = tensor.ExtractSliceOp(
            tensor_type, input1, [], [], [], offset_attr, size_attr, stride_attr
        )

    return op


def silu_op(
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor silu activation operation.
    From PyTorch `aten.silu.default` operator to MLIR linalg `generic`
    operation.

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

    output_shape = list(node.meta["tensor_meta"].shape)
    dtype = str(node.meta["tensor_meta"].dtype)
    if dtype == "torch.float32":
        tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
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
        neg_op = arith.NegFOp(block.arguments[0])
        exp_op = math.ExpOp(neg_op.result)
        one_op = arith.ConstantOp(
            ir.F32Type.get(), ir.FloatAttr.get(ir.F32Type.get(), 1)
        )
        add_op = arith.AddFOp(one_op.result, exp_op.result)
        div_op = arith.DivFOp(block.arguments[0], add_op.result)
        block.append(neg_op)
        block.append(exp_op)
        block.append(one_op)
        block.append(add_op)
        block.append(div_op)
        block.append(linalg.YieldOp([div_op.result]))

    return op


def param_extract(
    node: torch.fx.Node,
    offset,
    params_mlir_node,
):
    """
    Extract param from packed params.

    Note: This function, extract slice from packed params tensor, and expand
    shape by param node shape.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the tensor.expand_shape op.
    """
    dtype_mapping = {
        torch.float32: ir.F32Type.get(),
        torch.int64: ir.IntegerType.get_signless(64),
    }
    tensor_element_type = dtype_mapping[node.meta["tensor_meta"].dtype]
    output_shape = list(node.meta["tensor_meta"].shape)
    extract_size = functools.reduce(lambda x, y: x * y, output_shape)
    offset_attr = ir._denseI64ArrayAttr([offset], None)
    size_attr = ir._denseI64ArrayAttr([extract_size], None)
    stride = [1]
    stride_attr = ir._denseI64ArrayAttr(stride, None)
    tensor_type = ir.RankedTensorType.get([extract_size], tensor_element_type)
    extract_slice_op = tensor.ExtractSliceOp(
        tensor_type,
        params_mlir_node,
        [],
        [],
        [],
        offset_attr,
        size_attr,
        stride_attr,
    )
    if len(output_shape) == 1:
        return extract_slice_op
    tensor_type = ir.RankedTensorType.get(output_shape, tensor_element_type)
    axis = ir.ArrayAttr.get(
        [
            ir.IntegerAttr.get(ir.IntegerType.get_signless(64), i)
            for i in range(len(output_shape))
        ],
        None,
    )
    axis = ir.ArrayAttr.get([axis], None)
    return tensor.ExpandShapeOp(tensor_type, extract_slice_op.result, axis)


ops_registry = {
    "arange.start": arange_op,
    "arange.default": arange_op,
    "unsqueeze.default": unsqueeze_op,
    "view.default": view_op,
    "ones.default": ones_op,
    "full.default": full_op,
    "lt.Tensor": lt_op,
    "embedding.default": embedding_op,
    "masked_fill.Scalar": masked_fill_op,
    "slice.Tensor": slice_op,
    "expand.default": expand_op,
    "_to_copy.default": to_copy_op,
    "rsub.Scalar": rsub_op,
    "pow.Tensor_Scalar": pow_op,
    "mean.dim": mean_op,
    "rsqrt.default": rsqrt_op,
    "mul.Tensor": mul_op,
    "t.default": t_op,
    "mm.default": matmul_op,
    "transpose.int": transpose_op,
    "index.Tensor": index_op,
    "neg.default": neg_op,
    "cat.default": cat_op,
    "squeeze.dim": squeeze_op,
    "bmm.default": batch_matmul_op,
    "div.Tensor": div_op,
    "_softmax.default": softmax_op,
    "clone.default": clone_op,
    "silu.default": silu_op,
    "param.extract": param_extract,
}
