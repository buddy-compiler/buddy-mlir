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
# The registry of mappings from Buddy Graph to MLIR tosa dialect operations.
#
# ===---------------------------------------------------------------------------

import array, copy
from typing import Dict, List, Tuple, Union
import numpy
import sys

import mlir.ir as ir
from mlir.dialects import tensor, tosa, arith, linalg, math

from ..graph import TensorDType
from ..graph import (
    AddOp,
    PermuteOp,
    AddMMOp,
    BatchMatmulOp,
    SubOp,
    MulOp,
    DivOp,
    TanhOp,
    ExpOp,
    RsqrtOp,
    AmaxOp,
    ReshapeOp,
    UnsqueezeOp,
    SelectOp,
    SliceOp,
    ConvertElementTypeOp,
    CloneOp,
    VarMeanOp,
    EmbeddingOp,
    ExpandOp,
    SumDimOp,
    TOp,
    TransposeOp,
    MaxPool2dOp,
    Conv2dOp,
    ReluOp,
    IotaOp,
    SigmoidOp,
    ReciprocalOp,
    MeanOp,
    ClampMinOp,
    ClampMaxOp,
    RandIntLowOp,
    ArgMaxOp,
    ScaledDotProductFlashAttentionForCpuOp,
    MatmulOp,
)
from .utils import *


def _normalize_binary_operator_shape(shp1, shp2):
    """Normalize the shape of two input tensors according to the broadcasting
    rule"""
    shp1 = list(shp1)
    shp2 = list(shp2)
    while len(shp1) < len(shp2):
        shp1.insert(0, 1)
    while len(shp2) < len(shp1):
        shp2.insert(0, 1)

    return shp1, shp2


def _gen_arith_binary_op(input1, input2, op_func):
    """Generate arithmetic binary operation. Most binary operations follow the
    same pattern.
    So we can use one function to generate them, avoiding code duplication."""
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
        index_ty = ir.IndexType.get()
        shape_ty = ir.Type.parse(f"!tosa.shape<{len(norm_input1_shape)}>")
        shape_val = tosa.ConstShapeOp(
            shape_ty,
            ir.DenseElementsAttr.get(
                array.array("q", norm_input1_shape),
                type=index_ty,
                shape=[len(norm_input1_shape)],
            ),
        ).result
        input1 = tosa.ReshapeOp(input1, shape_val).result
    if input2_shape != norm_input2_shape:
        index_ty = ir.IndexType.get()
        shape_ty = ir.Type.parse(f"!tosa.shape<{len(norm_input2_shape)}>")
        shape_val = tosa.ConstShapeOp(
            shape_ty,
            ir.DenseElementsAttr.get(
                array.array("q", norm_input2_shape),
                type=index_ty,
                shape=[len(norm_input2_shape)],
            ),
        ).result
        input2 = tosa.ReshapeOp(input2, shape_val).result

    result_element_type = ir.RankedTensorType(input1.type).element_type
    result_tensor_type = ir.RankedTensorType.get(
        broadcasted_result_shp, result_element_type
    )
    op = op_func(result_tensor_type, input1, input2)
    return op


def _scalar_to_tensor(
    scalar: Union[float, int], element_type: ir.Type, shape: List[int]
):
    """Convert scalers to cooresponding tensors since MLIR
    doesn't support operation between scalers and tensors."""
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
    """Normalize the types of binary operator arguments."""
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


def addmm_op(
    node: AddMMOp, symbol_table: Dict[Tuple[str, int], ir.Operation]
) -> ir.Operation:
    """
    Import matrix multiplication operation.
    From buddy graph ir's `AddMMOp` operator to MLIR TOSA `matmul` operation.

    Note: this function first reshapes the input matrices to 3D tensors
    (since tosa.MatMulOp requires it). Then it multiplies these reshaped
    matrices.
    Finally, it adds the input tensor to the matrix multiplication result.

    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation representing the result of adding the matrix
        multiplication to the input tensor.
    """
    # get input
    input_ = symbol_table.get((str(node.args[0]), 0))
    mat1 = symbol_table.get((str(node.args[1]), 0))
    mat2 = symbol_table.get((str(node.args[2]), 0))
    # get input shape
    mat1_shp = ir.RankedTensorType(mat1.type).shape
    mat2_shp = ir.RankedTensorType(mat2.type).shape
    shape1_ty = ir.Type.parse(f"!tosa.shape<{len([1, *mat1_shp])}>")
    index_ty = ir.IndexType.get()
    shape1_val = tosa.ConstShapeOp(
        shape1_ty,
        ir.DenseElementsAttr.get(
            array.array("q", [1, *mat1_shp]),
            type=index_ty,
            shape=[len([1, *mat1_shp])],
        ),
    ).result
    shape2_ty = ir.Type.parse(f"!tosa.shape<{len([1, *mat2_shp])}>")
    shape2_val = tosa.ConstShapeOp(
        shape2_ty,
        ir.DenseElementsAttr.get(
            array.array("q", [1, *mat2_shp]),
            type=index_ty,
            shape=[len([1, *mat2_shp])],
        ),
    ).result
    # append index because tosa.MatMulOp doesn't accept 2D tensor
    mat1_reshape_op = tosa.ReshapeOp(mat1, shape1_val)
    mat2_reshape_op = tosa.ReshapeOp(mat2, shape2_val)
    # do matmul
    result_element_type = ir.RankedTensorType(mat1.type).element_type
    matmul_result_shp = [1, mat1_shp[0], mat2_shp[1]]
    matmul_result_type = ir.RankedTensorType.get(
        matmul_result_shp, result_element_type
    )
    ty = ir.RankedTensorType.get([1], result_element_type)
    attr = ir.FloatAttr.get(result_element_type, 0.0)
    a_zp = tosa.ConstOp(ir.DenseElementsAttr.get_splat(ty, attr)).result
    b_zp = tosa.ConstOp(ir.DenseElementsAttr.get_splat(ty, attr)).result
    matmul_op = tosa.MatMulOp(
        matmul_result_type,
        mat1_reshape_op.result,
        mat2_reshape_op.result,
        a_zp,
        b_zp,
    )
    # restore the shape
    final_result_shape = [mat1_shp[0], mat2_shp[1]]
    shape_ty = ir.Type.parse(f"!tosa.shape<{len(final_result_shape)}>")
    shape_val = tosa.ConstShapeOp(
        shape_ty,
        ir.DenseElementsAttr.get(
            array.array("q", final_result_shape),
            type=index_ty,
            shape=[len(final_result_shape)],
        ),
    ).result
    matmul_result_reshape_op = tosa.ReshapeOp(matmul_op.result, shape_val)

    op = _gen_arith_binary_op(
        input_, matmul_result_reshape_op.result, tosa.AddOp
    )
    return op


def bmm_op(node: BatchMatmulOp, symbol_table) -> ir.Operation:
    """
    Import batch matrix multiplication operation.
    From buddy graph ir's `BatchMatmulOp` operator to MLIR TOSA `matmul`
    operation.
    """
    input_ = symbol_table.get((str(node.args[0]), 0))
    mat2 = symbol_table.get((str(node.args[1]), 0))
    input_shp = ir.RankedTensorType(input_.type).shape
    mat2_shp = ir.RankedTensorType(mat2.type).shape
    sizes = [input_shp[0], input_shp[1], mat2_shp[2]]
    result_element_type = ir.RankedTensorType(input_.type).element_type
    result_type = ir.RankedTensorType.get(sizes, result_element_type)
    ty = ir.RankedTensorType.get([1], result_element_type)
    attr = ir.FloatAttr.get(result_element_type, 0.0)
    a_zp = tosa.ConstOp(ir.DenseElementsAttr.get_splat(ty, attr)).result
    b_zp = tosa.ConstOp(ir.DenseElementsAttr.get_splat(ty, attr)).result
    op = tosa.MatMulOp(
        result_type,
        input_,
        mat2,
        a_zp,
        b_zp,
    )
    return op


def add_op(node: AddOp, symbol_table):
    """
    Import tensor addition operation.
    From buddy graph ir's `AddOp` operator to MLIR TOSA `add` operation.
    """
    input1 = symbol_table.get((str(node.args[0]), 0), node.args[0])
    input2 = symbol_table.get((str(node.args[1]), 0), node.args[1])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    if isinstance(node.args[0], str) and isinstance(node.args[1], str):
        input1_dtype = ir.RankedTensorType(input1.type).element_type
        input2_dtype = ir.RankedTensorType(input2.type).element_type
        if input1_dtype != mlir_dtype:
            input1 = tosa.CastOp(
                ir.RankedTensorType.get(
                    ir.RankedTensorType(input1.type).shape,
                    mlir_dtype,
                ),
                input1,
            ).result
        if input2_dtype != mlir_dtype:
            input2 = tosa.CastOp(
                ir.RankedTensorType.get(
                    ir.RankedTensorType(input2.type).shape,
                    mlir_dtype,
                ),
                input2,
            ).result
    return _gen_arith_binary_op(input1, input2, tosa.AddOp)


def sub_op(node: SubOp, symbol_table):
    """
    Import tensor subtraction operation.
    From buddy graph ir's `SubOp` operator to MLIR TOSA `sub` operation.
    """
    input1 = symbol_table.get((str(node.args[0]), 0), node.args[0])
    input2 = symbol_table.get((str(node.args[1]), 0), node.args[1])
    return _gen_arith_binary_op(input1, input2, tosa.SubOp)


def mul_op(node: MulOp, symbol_table):
    """
    Import tensor division operation.
    From buddy graph ir's `DivOp` operator to MLIR TOSA `div` operation.
    """

    def _inner_op(result_type, input1, input2):
        shift = tosa.ConstOp(
            ir.DenseElementsAttr.get_splat(
                ir.Type.parse("tensor<1xi8>"),
                ir.IntegerAttr.get(ir.IntegerType.get_signless(8), 0),
            )
        ).result
        return tosa.MulOp(result_type, input1, input2, shift)

    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)

    if isinstance(node.args[0], str):
        input1 = symbol_table.get((str(node.args[0]), 0), node.args[0])
    else:
        data = [node.args[0]]
        input1_shape = numpy.array(data).shape
        tensor_type = ir.RankedTensorType.get(input1_shape, mlir_dtype)
        element = mlir_element_attr_get(dtype, node.args[0])
        attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
        input2 = arith.ConstantOp(tensor_type, attr).result

    if isinstance(node.args[1], str):
        input2 = symbol_table.get((str(node.args[1]), 0), node.args[1])
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
        ).result
    if input2_dtype != mlir_dtype:
        input2 = tosa.CastOp(
            ir.RankedTensorType.get(
                ir.RankedTensorType(input2.type).shape,
                mlir_dtype,
            ),
            input2,
        ).result

    return _gen_arith_binary_op(input1, input2, _inner_op)


def div_op(node: DivOp, symbol_table):
    """
    Import tensor division operation.
    From buddy graph ir's `DivOp` operator to MLIR TOSA `div` operation.
    """

    def _inner_op(result_type, input1, input2):
        shift = tosa.ConstOp(
            ir.DenseElementsAttr.get_splat(
                ir.Type.parse("tensor<1xi8>"),
                ir.IntegerAttr.get(ir.IntegerType.get_signless(8), 0),
            )
        ).result
        return tosa.MulOp(
            result_type,
            input1,
            tosa.ReciprocalOp(input2.type, input2).result,
            shift,
        )

    input1 = symbol_table.get((str(node.args[0]), 0), node.args[0])
    input2 = symbol_table.get((str(node.args[1]), 0), node.args[1])

    return _gen_arith_binary_op(input1, input2, _inner_op)


def tanh_op(node: TanhOp, symbol_table):
    """
    Import elementwise tanh operation.
    From buddy graph ir's `TanhOp` operator to MLIR TOSA `tanh` operation.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    sizes = ir.RankedTensorType(input1.type).shape
    result_element_type = ir.RankedTensorType(input1.type).element_type
    tanhResultTensorType = ir.RankedTensorType.get(sizes, result_element_type)
    op = tosa.TanhOp(tanhResultTensorType, input1)
    return op


def exp_op(node: ExpOp, symbol_table):
    """
    Import elementwise exponential operation.
    From buddy graph ir's `ExpOp` operator to MLIR TOSA `exp` operation.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    sizes = ir.RankedTensorType(input1.type).shape
    result_element_type = ir.RankedTensorType(input1.type).element_type
    expResultTensorType = ir.RankedTensorType.get(sizes, result_element_type)
    op = tosa.ExpOp(expResultTensorType, input1)
    return op


def rsqrt_op(node: RsqrtOp, symbol_table):
    """
    Import elementwise reciprocal square root operation.
    From buddy graph ir's `RsqrtOp` operator to MLIR TOSA `rsqrt` operation.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    sizes = ir.RankedTensorType(input1.type).shape
    result_element_type = ir.RankedTensorType(input1.type).element_type
    rsqrt_result_tensor_type = ir.RankedTensorType.get(
        sizes, result_element_type
    )
    op = tosa.RsqrtOp(rsqrt_result_tensor_type, input1)
    return op


def amax_op(node: AmaxOp, symbol_table):
    """
    Import the amax operation.
    From buddy graph ir's `AmaxOp` operator to MLIR TOSA `reduce_max`
    operation.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    dim_val = node.args[1][0]
    if dim_val < 0:
        dim_val += len(ir.RankedTensorType(input1.type).shape)
    signless_type = ir.IntegerType.get_signless(32)
    dim_attr = ir.IntegerAttr.get(signless_type, dim_val)
    op = tosa.ReduceMaxOp(input1, dim_attr)
    return op


def reshape_op(node: ReshapeOp, symbol_table):
    """
    Import the reshape operation.
    From buddy graph ir's `ReshapeOp` operator to MLIR TOSA `reshape`
    operation.

    Note: If the new shape contains one and only one `-1`, the size of the new
    shape will be inferred automatically.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    new_shape = []
    for i in node.args[1]:
        new_shape.append(i)
    total_size = 1
    now_shape = ir.RankedTensorType(input1.type).shape
    for dim_siz in now_shape:
        total_size *= dim_siz

    neg_one_cnt = 0
    rest_size = 1
    for dim_siz in new_shape:
        if dim_siz == -1:
            neg_one_cnt += 1
            continue
        rest_size *= dim_siz

    if neg_one_cnt != 0:
        if neg_one_cnt > 1 or total_size % rest_size != 0:
            raise ValueError("Can not infer the new shape!")
        infer_dim_size = total_size // rest_size
        for i, _ in enumerate(new_shape):
            if new_shape[i] == -1:
                new_shape[i] = infer_dim_size
    shape_ty = ir.Type.parse(f"!tosa.shape<{len(new_shape)}>")
    index_ty = ir.IndexType.get()
    shape_val = tosa.ConstShapeOp(
        shape_ty,
        ir.DenseElementsAttr.get(
            array.array("q", new_shape), type=index_ty, shape=[len(new_shape)]
        ),
    ).result
    op = tosa.ReshapeOp(input1, shape_val)

    return op


def unsqueeze_op(node: UnsqueezeOp, symbol_table):
    """
    Import the unsqueeze operation.
    From buddy graph ir's `UnsqueezeOp` operator to MLIR TOSA `reshape`
    operation.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    dim = node.args[1]
    sizes = ir.RankedTensorType(input_tensor.type).shape
    if dim == -1:
        sizes.append(1)
    else:
        sizes.insert(dim, 1)
    shape_ty = ir.Type.parse(f"!tosa.shape<{len(sizes)}>")
    index_ty = ir.IndexType.get()
    shape_val = tosa.ConstShapeOp(
        shape_ty,
        ir.DenseElementsAttr.get(
            array.array("q", sizes), type=index_ty, shape=[len(sizes)]
        ),
    ).result
    op = tosa.ReshapeOp(input_tensor, shape_val)
    return op


def select_op(node: SelectOp, symbol_table):
    """
    Import the select operation.
    From buddy graph ir's `SelectOp` operator to MLIR TOSA `reshape`
    operation.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    dim = node.args[1]
    index = node.args[2]

    sizes = ir.RankedTensorType(input_tensor.type).shape

    new_sizes = sizes[:dim] + [1] + sizes[dim + 1 :]
    index_ty = ir.IndexType.get()
    shape_ty_new = ir.Type.parse(f"!tosa.shape<{len(new_sizes)}>")
    shape_val_new = tosa.ConstShapeOp(
        shape_ty_new,
        ir.DenseElementsAttr.get(
            array.array("q", new_sizes), type=index_ty, shape=[len(new_sizes)]
        ),
    ).result

    start = [0] * len(sizes)
    start[dim] = index
    shape_ty_start = ir.Type.parse(f"!tosa.shape<{len(start)}>")
    shape_val_start = tosa.ConstShapeOp(
        shape_ty_start,
        ir.DenseElementsAttr.get(
            array.array("q", start), type=index_ty, shape=[len(start)]
        ),
    ).result

    result_element_type = ir.RankedTensorType(input_tensor.type).element_type
    output_type = ir.RankedTensorType.get(new_sizes, result_element_type)
    op = tosa.SliceOp(output_type, input_tensor, shape_val_start, shape_val_new)

    reshape_sizes = sizes[:dim] + sizes[dim + 1 :]
    shape_ty = ir.Type.parse(f"!tosa.shape<{len(reshape_sizes)}>")
    index_ty = ir.IndexType.get()
    shape_val = tosa.ConstShapeOp(
        shape_ty,
        ir.DenseElementsAttr.get(
            array.array("q", reshape_sizes),
            type=index_ty,
            shape=[len(reshape_sizes)],
        ),
    ).result
    op = tosa.ReshapeOp(op.results[0], shape_val)

    return op


def slice_op(node: SliceOp, symbol_table):
    """
    Import the slice operation.
    From buddy graph ir's `SliceOp` operator to MLIR TOSA `extract_slice`
    operation.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    dim = node.args[1]
    start_idx = node.args[2]
    end_idx = node.args[3]

    sizes = ir.RankedTensorType(input_tensor.type).shape

    if start_idx < 0:
        start_idx += sizes[dim]

    if end_idx < 0:
        end_idx += sizes[dim]

    if start_idx < 0:
        start_idx = 0
    elif start_idx >= sizes[dim]:
        start_idx = sizes[dim]

    if end_idx < start_idx:
        end_idx = start_idx
    elif end_idx >= sizes[dim]:
        end_idx = sizes[dim]

    new_sizes = [x for x in sizes]
    new_sizes[dim] = end_idx - start_idx
    new_sizes_attr = ir._denseI64ArrayAttr(new_sizes, None)

    offsets = [0] * len(sizes)
    offsets[dim] = start_idx
    offsets_attr = ir._denseI64ArrayAttr(offsets, None)

    strides = [1] * len(sizes)
    strides_attr = ir._denseI64ArrayAttr(strides, None)

    result_element_type = ir.RankedTensorType(input_tensor.type).element_type
    extract_slice_result_type = ir.RankedTensorType.get(
        new_sizes, result_element_type
    )
    op = tensor.ExtractSliceOp(
        extract_slice_result_type,
        input_tensor,
        [],
        [],
        [],
        offsets_attr,
        new_sizes_attr,
        strides_attr,
    )

    return op


def convert_element_type_op(node: ConvertElementTypeOp, symbol_table):
    """
    Import the element type conversion operation.
    From buddy graph ir's `ConvertElementTypeOp` operator to MLIR TOSA
    `cast` operation.
    """
    # maintain a mapping of buddy dtype to mlir types
    types_mapping = {
        TensorDType.Float64: ir.F64Type.get(),
        TensorDType.Float32: ir.F32Type.get(),
        TensorDType.Float16: ir.F16Type.get(),
        TensorDType.Int64: ir.IntegerType.get_signless(64),
        TensorDType.Int32: ir.IntegerType.get_signless(32),
        TensorDType.Bool: ir.IntegerType.get_signless(1),
    }
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    to_cast_type = types_mapping[node.args[1]]
    input_type = ir.RankedTensorType(input_tensor.type).element_type
    # When converting float to int, tosa.cast lowers to math.roundeven, but we don't need rounding.
    if str(to_cast_type).find("i") != -1 and str(input_type).find("f") != -1:
        output_shape = list(node.tensor_meta["shape"])
        tensor_type = ir.RankedTensorType.get(output_shape, to_cast_type)
        output = tensor.EmptyOp(output_shape, to_cast_type)

        if str(to_cast_type) == "i1":
            false_val = arith.ConstantOp(to_cast_type, 0)
            true_val = arith.ConstantOp(to_cast_type, 1)
            zero_val = arith.ConstantOp(input_type, 0.0)

        generic_map = ir.AffineMap.get_permutation(
            [i for i in range(len(output_shape))]
        )
        op = linalg.GenericOp(
            [tensor_type],
            [input_tensor],
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
                input_type,
                to_cast_type,
            ],
        )
        if str(to_cast_type) == "i1":
            is_zero = arith.CmpFOp(1, block.arguments[0], zero_val)
            result = arith.SelectOp(is_zero, false_val, true_val)
            block.append(is_zero)
            block.append(result)
            block.append(linalg.YieldOp([result.result]))
        else:
            fptosi_op = arith.FPToSIOp(to_cast_type, block.arguments[0])
            block.append(fptosi_op)
            block.append(linalg.YieldOp([fptosi_op.result]))
    else:
        sizes = ir.RankedTensorType(input_tensor.type).shape
        output_type = ir.RankedTensorType.get(sizes, to_cast_type)
        op = tosa.CastOp(output_type, input_tensor)

    return op


def clone_op(node: CloneOp, symbol_table):
    """
    Import the clone operation.
    From buddy graph ir's `CloneOp` operator to MLIR TOSA `identity`
    operation.

    Note: Since MLIR follows the SSA form, when using the `identity` operation,
    we actually deep-copies the original tensor.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    sizes = ir.RankedTensorType(input_tensor.type).shape
    result_element_type = ir.RankedTensorType(input_tensor.type).element_type
    output_type = ir.RankedTensorType.get(sizes, result_element_type)

    return tosa.IdentityOp(output_type, input_tensor)


def var_mean_op(node: VarMeanOp, symbol_table):
    """
    Import the variance & mean operation.
    From buddy graph ir's `VarMeanOp` operator to two MLIR TOSA `mul`
    operation.

    Note: By now, this conversion function follows PyTorch's `var_mean`
    semantic.

          The conversion procedure can be splited into two steps:
          1. In the first part, we calculate the mean value along the given
          dimension(s) in `mean_dim_op` function. We first reduce the input
          tensor along the given dimension(s) using tosa's `reduce_sum`
          operation. Then we calculate the mean value by multiplying the
          reciprocal of the total size of the reduced dimension(s).
          2. In the second part, we calculate the variance value. We follow the
          formula in this link:
          https://pytorch.org/docs/stable/generated/torch.var_mean.html. We
          first calculate (\bar{x} - x_i), where \bar{x} is the mean value we
          calculated in the first step. By applying tosa's `mul` operation, we
          get (\bar{x} - x_i) ^ 2. Then we reduce the multiplication result to
          get \sum_{i=0}^{N}(\bar{x} - x_i) ^ 2. Finally, we divide the
          reduction sum result by the total size of the reduced dimension(s)
          minus the correction.

          `keepdim` argument is supported. It's handled by the applying a
          `reshape` operation.

    """

    def _inner_op(result_type, input1, input2):
        shift = tosa.ConstOp(
            ir.DenseElementsAttr.get_splat(
                ir.Type.parse("tensor<1xi8>"),
                ir.IntegerAttr.get(ir.IntegerType.get_signless(8), 0),
            )
        ).result
        return tosa.MulOp(result_type, input1, input2, shift)

    def mean_dim_op(_input_tensor: ir.Value, _dim) -> ir.Operation:
        if isinstance(_dim, int):
            _dim = [_dim]

        # `_input_tensor` is the first tensor we need to reduce
        reduce_sum_result = _input_tensor

        # reduce along each dimension in `_dim`
        for _dim_item in _dim:
            reduce_dim_attr = ir.IntegerAttr.get(
                ir.IntegerType.get_signless(32), _dim_item
            )
            reduce_sum_op: ir.Operation = tosa.ReduceSumOp(
                reduce_sum_result, reduce_dim_attr
            )
            # Next reduction is executed based on this time's reduction result
            reduce_sum_result = reduce_sum_op.results[0]

        tensor_shp = ir.RankedTensorType(_input_tensor.type).shape
        dim_size = 1
        # calculate the total size on all reduction dimensions to get the
        # denominator
        for _dim_item in _dim:
            dim_size *= tensor_shp[_dim_item]

        denominator_const_op: ir.Operation = tosa.ConstOp(
            ir.DenseElementsAttr.get(memoryview(array.array("f", [dim_size])))
        )

        reciprocal_op: ir.Operation = tosa.ReciprocalOp(
            denominator_const_op.results[0].type,
            denominator_const_op.results[0],
        )
        return _gen_arith_binary_op(
            reciprocal_op.results[0], reduce_sum_op.results[0], _inner_op
        )

    def var_dim_op(
        _input_tensor: ir.Value, _mean_tensor: ir.Value, _dim, _correction
    ) -> ir.Operation:
        if isinstance(_dim, int):
            _dim = [_dim]
        # get (\bar{x} - x_i)
        sub_op: ir.Operation = tosa.SubOp(
            _input_tensor.type, _input_tensor, _mean_tensor
        )

        # get (\bar{x} - x_i) ^ 2
        shift = tosa.ConstOp(
            ir.DenseElementsAttr.get_splat(
                ir.Type.parse("tensor<1xi8>"),
                ir.IntegerAttr.get(ir.IntegerType.get_signless(8), 0),
            )
        ).result
        mul_op: ir.Operation = tosa.MulOp(
            _input_tensor.type,
            sub_op.results[0],
            sub_op.results[0],
            shift,
        )

        # the result of `mul_op` is the first tensor we need to reduce
        reduce_sum_op = mul_op
        for _dim_item in _dim:
            reduce_dim_attr = ir.IntegerAttr.get(
                ir.IntegerType.get_signless(32), _dim_item
            )
            reduce_sum_op: ir.Operation = tosa.ReduceSumOp(
                reduce_sum_op.results[0], reduce_dim_attr
            )

        tensor_shp = ir.RankedTensorType(_input_tensor.type).shape
        dim_size = 1
        # calculate the denominator
        for _dim_item in _dim:
            dim_size *= tensor_shp[_dim_item]
        biased_denominator_const_op: ir.Operation = tosa.ConstOp(
            ir.DenseElementsAttr.get(
                memoryview(array.array("f", [dim_size - _correction]))
            )
        )
        reciprocal_op: ir.Operation = tosa.ReciprocalOp(
            biased_denominator_const_op.results[0].type,
            biased_denominator_const_op.results[0],
        )
        return _gen_arith_binary_op(
            reciprocal_op.results[0], reduce_sum_op.results[0], _inner_op
        )

    mean_input_tensor = symbol_table.get((str(node.args[0]), 0))
    var_input_tensor = symbol_table.get((str(node.args[0]), 0))

    kwargs = node.kwargs
    keepdim = kwargs.get("keepdim", False)
    correction = kwargs.get("correction", 1.0)

    mean_op = None
    var_op = None
    if len(node.args) == 1:
        calc_dims = range(
            len(ir.RankedTensorType(mean_input_tensor.type).shape)
        )
    else:
        calc_dims = node.args[1]

    mean_op = mean_dim_op(mean_input_tensor, calc_dims)
    var_op = var_dim_op(
        var_input_tensor, mean_op.results[0], calc_dims, correction
    )
    mean_input_tensor = mean_op.results[0]
    var_input_tensor = var_op.results[0]

    if not keepdim:
        result_shp = ir.RankedTensorType(var_op.results[0].type).shape
        result_shp = [siz for siz in result_shp if siz != 1]
        shape_ty = ir.Type.parse(f"!tosa.shape<{len(result_shp)}>")
        index_ty = ir.IndexType.get()
        shape_val = tosa.ConstShapeOp(
            shape_ty,
            ir.DenseElementsAttr.get(
                array.array("q", result_shp),
                type=index_ty,
                shape=[len(result_shp)],
            ),
        ).result
        var_op = tosa.ReshapeOp(var_op.results[0], shape_val)
        mean_op = tosa.ReshapeOp(mean_op.results[0], shape_val)

    return var_op, mean_op


def permute_op(node: PermuteOp, symbol_table):
    """
    Import the permute operation.
    From buddy graph ir's `PermuteOp` operator to MLIR TOSA `transpose`
    operation.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    perm = node.args[1]
    result_element_type = ir.RankedTensorType(input_tensor.type).element_type
    init_shape = ir.RankedTensorType(input_tensor.type).shape
    new_shape = []
    for perm_item in perm:
        new_shape.append(init_shape[perm_item])

    permute_result_type = ir.RankedTensorType.get(
        new_shape, result_element_type
    )
    perms_attr = ir.DenseI32ArrayAttr.get([int(v) for v in perm])
    permute_op = tosa.TransposeOp(permute_result_type, input_tensor, perms_attr)
    return permute_op


def embedding_op(node: EmbeddingOp, symbol_table):
    """
    Import the embedding operation.
    From buddy graph ir's `EmbeddingOp` operator to MLIR TOSA `reshape`
    operation.

    Note: Althought this conversion function will finally return a `reshape`
    operation, the core is the `gather` operation. It can generate a tensor for
    which each element in the output is a slice of the values tensor based on
    the value of indices. In this case, we use `gather` to extract elements from
    the weight tensor based on the `indices` argument.
    """
    indices = symbol_table.get((str(node.args[1]), 0))
    weight = symbol_table.get((str(node.args[0]), 0))

    indices_size = ir.RankedTensorType(indices.type).shape
    weight_size = ir.RankedTensorType(weight.type).shape
    result_element_type = ir.RankedTensorType(weight.type).element_type
    assert len(indices_size) == 2

    if indices_size[0] != 1:
        total_size = 1
        for x in indices_size:
            total_size *= x
        shape_ty = ir.Type.parse(f"!tosa.shape<{len([1, total_size])}>")
        index_ty = ir.IndexType.get()
        shape_val = tosa.ConstShapeOp(
            shape_ty,
            ir.DenseElementsAttr.get(
                array.array("q", [1, total_size]),
                type=index_ty,
                shape=[len([1, total_size])],
            ),
        ).result
        indices_reshape_op = tosa.ReshapeOp(indices, shape_val)
        indices = indices_reshape_op.result
        gather_result_type = ir.RankedTensorType.get(
            [1, total_size, weight_size[1]], result_element_type
        )
    else:
        gather_result_type = ir.RankedTensorType.get(
            [*indices_size, weight_size[1]], result_element_type
        )

    # tosa.gather doesn't support i64, so we need to cast it to i32
    if str(ir.RankedTensorType(indices.type).element_type) != "i32":
        indices = tosa.CastOp(
            ir.RankedTensorType.get(
                ir.RankedTensorType(indices.type).shape,
                ir.IntegerType.get_signless(32),
            ),
            indices,
        )
    shape_ty = ir.Type.parse(f"!tosa.shape<{len([1, *weight_size])}>")
    index_ty = ir.IndexType.get()
    shape_val = tosa.ConstShapeOp(
        shape_ty,
        ir.DenseElementsAttr.get(
            array.array("q", [1, *weight_size]),
            type=index_ty,
            shape=[len([1, *weight_size])],
        ),
    ).result
    weight_reshape_op = tosa.ReshapeOp(weight, shape_val)

    gather_op = tosa.GatherOp(
        gather_result_type, weight_reshape_op.result, indices
    )
    shape_ty = ir.Type.parse(
        f"!tosa.shape<{len([*indices_size, weight_size[1]])}>"
    )
    shape_val = tosa.ConstShapeOp(
        shape_ty,
        ir.DenseElementsAttr.get(
            array.array("q", [*indices_size, weight_size[1]]),
            type=index_ty,
            shape=[len([*indices_size, weight_size[1]])],
        ),
    ).result
    op = tosa.ReshapeOp(gather_op.output, shape_val)

    return op


def expand_op(node: ExpandOp, symbol_table) -> ir.Operation:
    """
    Import the expand operation.
    From buddy graph ir's `ExpandOp` operator to MLIR TOSA `add` operation.

    Note: This conversion is implemented using the broadcast machanism of TOSA
          `add` operation. We allocate a tensor with the shape to expand and
          elements in this tensor is all zero. Then we add the original tensor
          to this all-zero tensor. After the applying the broadcasting, we get
          the result.
    """
    to_expand_tensor = symbol_table.get((str(node.args[0]), 0))
    original_size = to_expand_tensor.type.shape
    new_size = node.args[1]
    result_element_type = ir.RankedTensorType(
        to_expand_tensor.type
    ).element_type
    if result_element_type in (
        ir.IntegerType.get_signless(1),
        ir.IntegerType.get_signless(64),
    ):
        element = ir.IntegerAttr.get(result_element_type, 0)
    elif result_element_type == ir.F32Type.get():
        element = ir.FloatAttr.get(result_element_type, 0.0)
    elif result_element_type == ir.F16Type.get():
        element = ir.FloatAttr.get(result_element_type, 0.0)
    else:
        raise NotImplementedError("Unsupported element type!")
    expanded_size = []
    for dim, size in zip(original_size, new_size):
        if size == -1:
            expanded_size.append(dim)
        else:
            expanded_size.append(size)
    new_size_tensor_type = ir.RankedTensorType.get(
        expanded_size, result_element_type
    )
    new_size_attr = ir.DenseElementsAttr.get_splat(
        new_size_tensor_type, element
    )
    new_size_tensor = tosa.ConstOp(new_size_attr).results[0]
    op = _gen_arith_binary_op(to_expand_tensor, new_size_tensor, tosa.AddOp)
    return op


def sum_op(node: SumDimOp, symbol_table):
    """
    Import the sum operation.
    From buddy graph ir's `SumDimOp` operator to MLIR TOSA `reduce_sum`
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    reduce_sum_dims = node.args[1]
    dim_cnt = len(ir.RankedTensorType(input_tensor.type).shape)
    reduce_sum_dims = [
        dim if dim >= 0 else dim_cnt + dim for dim in reduce_sum_dims
    ]
    _reduce_sum_input_tensor = input_tensor
    reduce_sum_op = None
    for dim in reduce_sum_dims:
        reduce_dim_attr = ir.IntegerAttr.get(
            ir.IntegerType.get_signless(32), dim
        )
        reduce_sum_op = tosa.ReduceSumOp(
            _reduce_sum_input_tensor, reduce_dim_attr
        )
        _reduce_sum_input_tensor = reduce_sum_op.results[0]

    return reduce_sum_op


def t_op(node: TOp, symbol_table):
    """
    Import the tensor transpose operation.
    From buddy graph ir's `TOp` operator to MLIR TOSA `transpose` operation
    """
    assert len(node.args) == 1
    input1 = symbol_table.get((str(node.args[0]), 0))
    assert input1 is not None

    input_shape = list(ir.RankedTensorType(input1.type).shape)
    output_shape = list(node.tensor_meta["shape"])
    assert len(input_shape) == 2, "Input tensor must be 2D"
    result_element_type = ir.RankedTensorType(input1.type).element_type
    permute_result_type = ir.RankedTensorType.get(
        output_shape, result_element_type
    )
    perms_attr = ir.DenseI32ArrayAttr.get([1, 0])
    op = tosa.TransposeOp(permute_result_type, input1, perms_attr)

    return op


def transpose_op(node: TransposeOp, symbol_table):
    """
    Import the tensor permute operation based on input dims.
    From buddy graph ir's `TransposeOp` operator to MLIR TOSA `transpose`
    operation.
    """
    assert len(node.args) == 3, "Input tensor must be 3D"
    input1 = symbol_table.get((str(node.args[0]), 0))
    if input1 is None:
        return
    dim1 = int(node.args[1])
    dim2 = int(node.args[2])
    input_shape = list(ir.RankedTensorType(input1.type).shape)
    perm_list = [i for i in range(len(input_shape))]
    temp = perm_list[dim1]
    perm_list[dim1] = perm_list[dim2]
    perm_list[dim2] = temp
    output_shape = list(node.tensor_meta["shape"])
    perm_const_op = tosa.ConstOp(
        ir.DenseElementsAttr.get(memoryview(array.array("i", perm_list)))
    )
    result_element_type = ir.RankedTensorType(input1.type).element_type
    permute_result_type = ir.RankedTensorType.get(
        output_shape, result_element_type
    )
    op = tosa.TransposeOp(permute_result_type, input1, perm_const_op.results[0])

    return op


def maxpool2d_op(node: MaxPool2dOp, symbol_table):
    """
    Import the maxpool2d operation.
    From Buddy MaxPool2dOp to MLIR TOSA max_pool2d operation.
    """
    if len(node.args) == 5:
        raise NotImplementedError
    input1 = symbol_table.get((str(node.args[0]), 0))
    kernel = node.args[1]
    stride = node.args[2]
    if len(node.args) > 3:
        pad = node.args[3]
    else:
        pad = [0 for _ in kernel]
    dtype = node.tensor_meta["dtype"]
    result_element_type = mlir_element_type_get(dtype)
    if node._layout.find("NCHW") != -1:
        perm_list = [0, 2, 3, 1]
        perms_attr = ir.DenseI32ArrayAttr.get([int(v) for v in perm_list])
        out_shape = list(ir.RankedTensorType(input1.type).shape)
        perm_shape = []
        perm_shape.append(out_shape[0])
        perm_shape.append(out_shape[2])
        perm_shape.append(out_shape[3])
        perm_shape.append(out_shape[1])
        permute_result_type = ir.RankedTensorType.get(
            perm_shape, result_element_type
        )
        input1 = tosa.TransposeOp(
            permute_result_type, input1, perms_attr
        ).result

    if len(pad) == 1:
        pad = [pad[0]] * 4
    elif len(pad) == 2:
        pad = [pad[0]] * 2 + [pad[1]] * 2

    def _ceil_div(a, b):
        return (a + b - 1) // b

    in_n, in_h, in_w, in_c = list(ir.RankedTensorType(input1.type).shape)
    k_h, k_w = kernel[0], kernel[1]
    s_h, s_w = stride[0], stride[1]
    pt, pb, pl, pr = pad

    def _divisible(i, p0, p1, k, s):
        return ((i + p0 + p1 - k) % s) == 0

    if not _divisible(in_h, pt, pb, k_h, s_h):
        out_h_same = _ceil_div(in_h, s_h)
        pad_total_h = max((out_h_same - 1) * s_h + k_h - in_h, 0)
        pt = pad_total_h // 2
        pb = pad_total_h - pt
    if not _divisible(in_w, pl, pr, k_w, s_w):
        out_w_same = _ceil_div(in_w, s_w)
        pad_total_w = max((out_w_same - 1) * s_w + k_w - in_w, 0)
        pl = pad_total_w // 2
        pr = pad_total_w - pl

    out_h = (in_h + pt + pb - k_h) // s_h + 1
    out_w = (in_w + pl + pr - k_w) // s_w + 1
    out_shape_nhwc = [in_n, out_h, out_w, in_c]

    kernel_attr = ir._denseI64ArrayAttr(kernel, None)
    stride_attr = ir._denseI64ArrayAttr(stride, None)
    pad_attr = ir._denseI64ArrayAttr([pt, pb, pl, pr], None)

    output = ir.RankedTensorType.get(out_shape_nhwc, result_element_type)
    op = tosa.MaxPool2dOp(output, input1, kernel_attr, stride_attr, pad_attr)

    if node._layout.find("NCHW") != -1:
        perm_list = [0, 3, 1, 2]
        perms_attr = ir.DenseI32ArrayAttr.get([int(v) for v in perm_list])
        perm_shape = []
        perm_shape.append(out_shape_nhwc[0])
        perm_shape.append(out_shape_nhwc[3])
        perm_shape.append(out_shape_nhwc[1])
        perm_shape.append(out_shape_nhwc[2])
        permute_result_type = ir.RankedTensorType.get(
            perm_shape, result_element_type
        )
        op = tosa.TransposeOp(permute_result_type, op.result, perms_attr)
    return op


# TODO: Rename convolution2d_op -> convolution_op
def convolution2d_op(node: Conv2dOp, symbol_table):
    """
    Import the convolution operation.
    From Buddy Conv2dOp to MLIR TOSA `conv2d` operation.
    arg[0]: Tensor input
    arg[1]: Tensor weight
    arg[2]: Tensor? bias
    arg[3]: SymInt[] stride
    arg[4]: SymInt[] padding
    arg[5]: SymInt[] dilation
    arg[6]: bool transposed
    arg[7]: SymInt[] output_padding
    arg[8]: SymInt groups
    """
    # Get arguments from convolution node.
    assert len(node.args) == 9
    input = node.args[0]
    weight = node.args[1]
    bias = node.args[2]
    stride = node.args[3]
    input_padding = node.args[4]
    dilation = node.args[5]
    is_kernel_transposed = node.args[6]
    out_padding = node.args[7]
    groups = node.args[8]

    # Prepare input, weight, and output information.
    input_val = symbol_table.get((str(input), 0))
    input_shape = list(ir.RankedTensorType(input_val.type).shape)
    weight_val = symbol_table.get((str(weight), 0))
    weight_shape = ir.RankedTensorType(weight_val.type).shape
    dtype = node.tensor_meta["dtype"]
    result_element_type = mlir_element_type_get(dtype)
    out_shape = node.tensor_meta["shape"]
    acc_type = ir.TypeAttr.get(result_element_type)

    # Prepare Depthwise Conv2D information
    is_grouped = (list(weight_shape)[1] == 1) and (groups != 1)
    is_depthwise = (groups == list(weight_shape)[0]) and is_grouped

    # Prepare input channel and output channel.
    if is_kernel_transposed:
        in_channels = list(weight_shape)[0]
        out_channels = list(weight_shape)[1] * groups
    else:
        in_channels = list(weight_shape)[1] * groups
        out_channels = list(weight_shape)[0]

    # Prepare bias tensor.
    if len(node._parents) == 2:
        new_size_tensor_type = ir.RankedTensorType.get(
            [out_channels], result_element_type
        )
        element = mlir_element_attr_get(dtype, 0)
        new_size_attr = ir.DenseElementsAttr.get_splat(
            new_size_tensor_type, element
        )
        bias_tensor = tosa.ConstOp(new_size_attr).results[0]
    else:
        bias_tensor = symbol_table.get((str(bias), 0))

    # Prepare attributes.
    dilation_attr = ir._denseI64ArrayAttr(dilation, None)
    stride_attr = ir._denseI64ArrayAttr(stride, None)

    # Convolution 2D
    if len(weight_shape) == 4:
        # Prepare input padding.
        if len(input_padding) == 1:
            input_padding = [input_padding[0]] * 4
        elif len(input_padding) == 2:
            input_padding = [input_padding[0]] * 2 + [input_padding[1]] * 2
        t, b, l, r = input_padding
        sy, sx = int(stride[0]), int(stride[1])
        dy, dx = int(dilation[0]), int(dilation[1])
        KH = int(list(weight_shape)[2])
        KW = int(list(weight_shape)[3])

        if node._layout.find("NCHW") != -1:
            H_in = int(input_shape[2])
            W_in = int(input_shape[3])
        else:
            H_in = int(input_shape[1])
            W_in = int(input_shape[2])

        def legalize_1d(In, K, D, S, p0, p1):
            base = In - 1 - (K - 1) * D
            rem = (base + p0 + p1) % S
            if rem == 0:
                return p0, p1
            take = min(rem, p1)
            p1 -= take
            rem = (base + p0 + p1) % S
            if rem == 0:
                return p0, p1
            take2 = min(rem, p0)
            p0 -= take2
            return p0, p1

        t, b = legalize_1d(H_in, KH, dy, sy, int(t), int(b))
        l, r = legalize_1d(W_in, KW, dx, sx, int(l), int(r))
        input_padding = [t, b, l, r]

        # Prepare input_padding attributes.
        input_padding_attr = ir._denseI64ArrayAttr(input_padding, None)

        # If the input layout is NCHW, then convert to NHWC.
        if node._layout.find("NCHW") != -1:
            perm_list = [0, 2, 3, 1]
            perms_attr = ir.DenseI32ArrayAttr.get([int(v) for v in perm_list])
            perm_shape = []
            perm_shape.append(input_shape[0])
            perm_shape.append(input_shape[2])
            perm_shape.append(input_shape[3])
            perm_shape.append(input_shape[1])
            permute_result_type = ir.RankedTensorType.get(
                perm_shape, result_element_type
            )
            input_val = tosa.TransposeOp(
                permute_result_type, input_val, perms_attr
            ).result

        # If the output layout is NCHW, then convert to NHWC
        if node._layout.find("NCHW") != -1:
            perm_shape = []
            perm_shape.append(out_shape[0])
            perm_shape.append(out_shape[2])
            perm_shape.append(out_shape[3])
            perm_shape.append(out_shape[1])
            out_shape = perm_shape
            output_type = ir.RankedTensorType.get(
                out_shape, result_element_type
            )
        else:
            output_type = ir.RankedTensorType.get(
                out_shape, result_element_type
            )

        # Depthwise Conv2D Operation.
        if is_depthwise is True:
            if node._layout.find("FCHW") != -1:
                perm_list = [2, 3, 0, 1]
                perms_attr = ir.DenseI32ArrayAttr.get(
                    [int(v) for v in perm_list]
                )
                perm_shape = []
                perm_shape.append(weight_shape[2])
                perm_shape.append(weight_shape[3])
                perm_shape.append(weight_shape[0])
                perm_shape.append(weight_shape[1])
                permute_result_type = ir.RankedTensorType.get(
                    perm_shape, result_element_type
                )
                weight_depthwise = tosa.TransposeOp(
                    permute_result_type, weight_val, perms_attr
                ).result
                ty = ir.Type.parse("tensor<1xf32>")
                input_zp = tosa.ConstOp(
                    ir.DenseElementsAttr.get_splat(
                        ty, ir.FloatAttr.get_f32(0.0)
                    )
                ).result
                weight_zp = tosa.ConstOp(
                    ir.DenseElementsAttr.get_splat(
                        ty, ir.FloatAttr.get_f32(0.0)
                    )
                ).result
                op = tosa.DepthwiseConv2DOp(
                    output_type,
                    input_val,
                    weight_depthwise,
                    bias_tensor,
                    input_zp,
                    weight_zp,
                    input_padding_attr,
                    stride_attr,
                    dilation_attr,
                    acc_type,
                )
            else:
                ty = ir.Type.parse("tensor<1xf32>")
                input_zp = tosa.ConstOp(
                    ir.DenseElementsAttr.get_splat(
                        ty, ir.FloatAttr.get_f32(0.0)
                    )
                ).result
                weight_zp = tosa.ConstOp(
                    ir.DenseElementsAttr.get_splat(
                        ty, ir.FloatAttr.get_f32(0.0)
                    )
                ).result
                op = tosa.DepthwiseConv2DOp(
                    output_type,
                    input_val,
                    weight_val,
                    bias_tensor,
                    input_zp,
                    weight_zp,
                    input_padding_attr,
                    stride_attr,
                    dilation_attr,
                    acc_type,
                )

        # Transpose Conv2D Operation.
        elif is_kernel_transposed:
            if sum(input_padding) > 0 or sum(dilation) > len(dilation):
                raise NotImplementedError
            for i in range(len(out_padding), 4):
                out_padding = [0] + out_padding
            out_padding_attr = ir._denseI64ArrayAttr(out_padding, None)
            out_shape_attr = ir._denseI64ArrayAttr(out_shape, None)
            ty = ir.Type.parse("tensor<1xf32>")
            input_zp = tosa.ConstOp(
                ir.DenseElementsAttr.get_splat(ty, ir.FloatAttr.get_f32(0.0))
            ).result
            weight_zp = tosa.ConstOp(
                ir.DenseElementsAttr.get_splat(ty, ir.FloatAttr.get_f32(0.0))
            ).result
            op = tosa.TransposeConv2DOp(
                output_type,
                input_val,
                weight_val,
                bias_tensor,
                input_zp,
                weight_zp,
                out_padding_attr,
                stride_attr,
                out_shape_attr,
            )

        # Generic Conv2D Operation.
        else:
            if node._layout.find("FCHW") != -1:
                perm_list = [0, 2, 3, 1]  # O, H, W, C
                perms_attr = ir.DenseI32ArrayAttr.get(
                    [int(v) for v in perm_list]
                )
                perm_shape = []
                perm_shape.append(weight_shape[0])
                perm_shape.append(weight_shape[2])
                perm_shape.append(weight_shape[3])
                perm_shape.append(weight_shape[1])
                permute_result_type = ir.RankedTensorType.get(
                    perm_shape, result_element_type
                )
                weight_val = tosa.TransposeOp(
                    permute_result_type, weight_val, perms_attr
                ).result

            if sum(input_padding) == 0:
                perm_list_hwcf = [1, 2, 3, 0]
                perms_attr_hwcf = ir.DenseI32ArrayAttr.get(
                    [int(v) for v in perm_list_hwcf]
                )
                w_shape = list(ir.RankedTensorType(weight_val.type).shape)
                weight_hwcf_shape = [
                    w_shape[1],
                    w_shape[2],
                    w_shape[3],
                    w_shape[0],
                ]
                weight_hwcf_type = ir.RankedTensorType.get(
                    weight_hwcf_shape, result_element_type
                )
                weight_hwcf = tosa.TransposeOp(
                    weight_hwcf_type, weight_val, perms_attr_hwcf
                ).result

                output_type = ir.RankedTensorType.get(
                    out_shape, result_element_type
                )
                output_init = tensor.EmptyOp(
                    list(out_shape), result_element_type
                )

                conv2d = linalg.conv_2d_nhwc_hwcf(
                    input_val,
                    weight_hwcf,
                    outs=[output_init],
                    strides=stride_attr,
                    dilations=dilation_attr,
                )

                output_after_bias = tensor.EmptyOp(
                    list(out_shape), result_element_type
                )
                rank = len(out_shape)
                map_nhwc = ir.AffineMap.get_identity(rank)
                map_c = ir.AffineMap.get(
                    rank, 0, [ir.AffineDimExpr.get(rank - 1)]
                )
                op = linalg.GenericOp(
                    [output_type],
                    [conv2d, bias_tensor],
                    [output_after_bias],
                    ir.ArrayAttr.get(
                        [
                            ir.AffineMapAttr.get(map_nhwc),
                            ir.AffineMapAttr.get(map_c),
                            ir.AffineMapAttr.get(map_nhwc),
                        ]
                    ),
                    ir.ArrayAttr.get(
                        [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
                        * rank
                    ),
                )
                block = ir.Block.create_at_start(
                    op.region,
                    [
                        result_element_type,
                        ir.RankedTensorType(bias_tensor.type).element_type,
                        result_element_type,
                    ],
                )
                add_op = arith.AddFOp(block.arguments[0], block.arguments[1])
                block.append(add_op)
                block.append(linalg.YieldOp([add_op.result]))

            else:
                ty = ir.Type.parse("tensor<1xf32>")
                input_zp = tosa.ConstOp(
                    ir.DenseElementsAttr.get_splat(
                        ty, ir.FloatAttr.get_f32(0.0)
                    )
                ).result
                weight_zp = tosa.ConstOp(
                    ir.DenseElementsAttr.get_splat(
                        ty, ir.FloatAttr.get_f32(0.0)
                    )
                ).result
                op = tosa.Conv2DOp(
                    output_type,
                    input_val,
                    weight_val,
                    bias_tensor,
                    input_zp,
                    weight_zp,
                    input_padding_attr,
                    stride_attr,
                    dilation_attr,
                    acc_type,
                )

        # Output transpose
        if node._layout.find("NCHW") != -1:
            perm_list = [0, 3, 1, 2]
            perms_attr = ir.DenseI32ArrayAttr.get([int(v) for v in perm_list])
            perm_shape = [
                out_shape[0],
                out_shape[3],
                out_shape[1],
                out_shape[2],
            ]
            permute_result_type = ir.RankedTensorType.get(
                perm_shape, result_element_type
            )
            op = tosa.TransposeOp(permute_result_type, op.result, perms_attr)

    # Convolution 1D
    elif len(weight_shape) == 3:
        # Prepare input with padding.
        if input_padding[0] != 0:
            input_shape = list(ir.RankedTensorType(input_val.type).shape)
            padded_type = ir.RankedTensorType.get(
                [
                    input_shape[0],
                    input_shape[1],
                    input_shape[2] + 2 * input_padding[0],
                ],
                result_element_type,
            )
            pad_values = ir.DenseElementsAttr.get(
                numpy.array(
                    [0, 0, 0, 0, input_padding[0], input_padding[0]],
                    dtype=numpy.int64,
                )
            )
            pad_tensor_type = ir.RankedTensorType.get([6], ir.IndexType.get())
            pad_values_attr = ir.DenseElementsAttr.get(
                pad_values, type=pad_tensor_type
            )
            shape_type = ir.Type.parse("!tosa.shape<6>")
            pad_constant = tosa.const_shape(shape_type, pad_values_attr)
            ty = ir.Type.parse("tensor<1xf32>")
            pad_zp = tosa.ConstOp(
                ir.DenseElementsAttr.get_splat(ty, ir.FloatAttr.get_f32(0.0))
            ).result
            input_val = tosa.PadOp(padded_type, input_val, pad_constant, pad_zp)

        output_type = ir.RankedTensorType.get(out_shape, result_element_type)
        output_conv = tensor.EmptyOp(list(out_shape), result_element_type)
        assert groups == 1, "only support one group"

        # Con1D Operation Without Bias
        conv_op = linalg.conv_1d_ncw_fcw(
            input_val,
            weight_val,
            outs=[output_conv],
            strides=stride_attr,
            dilations=dilation_attr,
        )

        output = tensor.EmptyOp(list(out_shape), result_element_type)
        generic_map = ir.AffineMap.get_permutation(
            [i for i in range(len(list(out_shape)))]
        )
        loop_type = [
            ir.Attribute.parse("#linalg.iterator_type<parallel>")
        ] * len(list(out_shape))
        loop_type[1] = ir.Attribute.parse("#linalg.iterator_type<reduction>")

        # Add Bias To Conv2d.
        op = linalg.GenericOp(
            [output_type],
            [conv_op, bias_tensor],
            [output],
            ir.ArrayAttr.get(
                [
                    ir.AffineMapAttr.get(
                        generic_map.get_submap(
                            [i for i in range(len(list(out_shape)))]
                        )
                    ),
                    ir.AffineMapAttr.get(generic_map.get_submap([1])),
                    ir.AffineMapAttr.get(
                        generic_map.get_submap(
                            [i for i in range(len(list(out_shape)))]
                        )
                    ),
                ]
            ),
            ir.ArrayAttr.get(loop_type),
        )
        block = ir.Block.create_at_start(
            op.region,
            [
                result_element_type,
                ir.RankedTensorType(bias_tensor.type).element_type,
                result_element_type,
            ],
        )
        add_op = arith.AddFOp(block.arguments[1], block.arguments[0])
        block.append(add_op)
        block.append(linalg.YieldOp([add_op.result]))

    return op


def relu_op(node: ReluOp, symbol_table):
    """
    Import the tensor relu operation.
    From Buddy ReluOp to MLIR TOSA `maximum` operation.
    """
    assert len(node.args) == 1
    input1 = symbol_table.get((str(node.args[0]), 0))
    if input1 is None:
        return
    dtype = node.tensor_meta["dtype"]
    result_element_type = mlir_element_type_get(dtype)
    in_shape = list(ir.RankedTensorType(input1.type).shape)
    tensor_type = ir.RankedTensorType.get(in_shape, result_element_type)
    element = mlir_element_attr_get(dtype, 0)
    attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
    zero_val = tosa.ConstOp(attr).result
    op = tosa.MaximumOp(tensor_type, input1, zero_val)
    return op


def iota_op(node: IotaOp, symbol_table):
    """
    Import the tensor iota operation.
    From Buddy IotaOp to MLIR TOSA `ConstOp` operation.
    """
    assert len(node.args) == 1
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    start = node.kwargs["start"]
    end = node.args[0]
    step = node.kwargs["step"]
    mlir_dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    attr = ir.DenseElementsAttr.get(
        numpy.arange(start, end, step),
        type=tensor_type,
    )
    op = tosa.ConstOp(attr)

    return op


def sigmoid_op(node: SigmoidOp, symbol_table):
    """
    Import the tensor sigmoid operation.
    From Buddy SigmoidOp to MLIR TOSA `SigmoidOp` operation.
    """
    assert len(node.args) == 1
    input1 = symbol_table.get((str(node.args[0]), 0))
    if input1 is None:
        return
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    op = tosa.SigmoidOp(tensor_type, input1)

    return op


def reciprocal_op(node: ReciprocalOp, symbol_table):
    """
    Import the buddy ReciprocalOp.
    From Buddy ReciprocalOp to MLIR TOSA `ReciprocalOp` operation.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    return tosa.ReciprocalOp(input_tensor.type, input_tensor)


def mean_op(node: MeanOp, symbol_table):
    """
    Import the buddy MeanOp.
    From Buddy MeanOp to MLIR TOSA operation.
    """

    def _inner_op(result_type, input1, input2):
        shift = tosa.ConstOp(
            ir.DenseElementsAttr.get_splat(
                ir.Type.parse("tensor<1xi8>"),
                ir.IntegerAttr.get(ir.IntegerType.get_signless(8), 0),
            )
        ).result
        return tosa.MulOp(
            result_type,
            input1,
            input2,
            shift,
        )

    input_tensor = symbol_table.get((str(node.args[0]), 0))
    keepdim = node.args[2]
    dims = [x for x in node.args[1]]
    if isinstance(dims, int):
        dims = [dims]

    for dim_item_idx, _ in enumerate(dims):
        if dims[dim_item_idx] < 0:
            dims[dim_item_idx] += len(
                ir.RankedTensorType(input_tensor.type).shape
            )

    reduce_sum_result = input_tensor
    for dim_item in dims:
        reduce_dim_attr = ir.IntegerAttr.get(
            ir.IntegerType.get_signless(32), dim_item
        )
        reduce_sum_op = tosa.ReduceSumOp(reduce_sum_result, reduce_dim_attr)
        reduce_sum_result = reduce_sum_op.results[0]

    tensor_shp = ir.RankedTensorType(input_tensor.type).shape
    dim_size = 1

    for dim_item in dims:
        dim_size *= tensor_shp[dim_item]

    denominator_const_op = tosa.ConstOp(
        ir.DenseElementsAttr.get(memoryview(array.array("f", [dim_size])))
    )
    reciprocal_op = tosa.ReciprocalOp(
        denominator_const_op.results[0].type, denominator_const_op
    )
    ret = _gen_arith_binary_op(
        reciprocal_op.results[0], reduce_sum_op.results[0], _inner_op
    )

    if not keepdim:
        result_shp = ir.RankedTensorType(ret.results[0].type).shape
        result_shp = [siz for siz in result_shp if siz != 1]
        shape_ty = ir.Type.parse(f"!tosa.shape<{len(result_shp)}>")
        index_ty = ir.IndexType.get()
        shape_val = tosa.ConstShapeOp(
            shape_ty,
            ir.DenseElementsAttr.get(
                array.array("q", result_shp),
                type=index_ty,
                shape=[len(result_shp)],
            ),
        ).result
        ret = tosa.ReshapeOp(ret.results[0], shape_val)

    return ret


def clamp_min_op(node: ClampMinOp, symbol_table):
    """
    Creates a TOSA clamp operation to set a minimum value for a tensor.

    Retrieves the input tensor and its minimum clamp value from the symbol table,
    setting the maximum clamp value to the highest possible for the data type.
    The operation ensures no values are below the specified minimum.

    Parameters:
    - node (ClampMinOp): Node with tensor and minimum value details.
    - symbol_table (dict): Dictionary mapping identifiers to values or nodes.

    Returns:
    - tosa.ClampOp: Configured TOSA clamp operation with minimum clamping.
    """
    input1 = symbol_table.get((str(node.args[0]), 0), node.args[0])
    min_value = symbol_table.get((str(node.args[1]), 0), node.args[1])
    tensor_type = input1.type
    min_fp = ir.FloatAttr.get(ir.F32Type.get(), min_value)
    max_fp = ir.FloatAttr.get(ir.F32Type.get(), float("inf"))
    op = tosa.ClampOp(tensor_type, input1, min_fp, max_fp)
    return op


def clamp_max_op(node: ClampMaxOp, symbol_table):
    """
    Creates a TOSA clamp operation to set a maximum value for a tensor.

    Retrieves the input tensor and its maximum clamp value from the symbol table,
    setting the minimum clamp value to the lowest possible for the data type.
    The operation ensures no values exceed the specified maximum.

    Parameters:
    - node (ClampMaxOp): Node with tensor and maximum value details.
    - symbol_table (dict): Dictionary mapping identifiers to values or nodes.

    Returns:
    - tosa.ClampOp: Configured TOSA clamp operation with maximum clamping.
    """
    input1 = symbol_table.get((str(node.args[0]), 0), node.args[0])
    max_value = symbol_table.get((str(node.args[1]), 0), node.args[1])
    tensor_type = input1.type
    min_value_int = round(max_value)
    min_fp = ir.FloatAttr.get(ir.F32Type.get(), -float("inf"))
    max_fp = ir.FloatAttr.get(ir.F32Type.get(), max_value)
    op = tosa.ClampOp(tensor_type, input1, min_fp, max_fp)
    return op


def randint_low_op(node: RandIntLowOp, symbol_table):
    """
    Generates a tensor of random integers within a specified range.

    Parameters:
    - node (RandIntLowOp): Node containing the range and shape.
    - symbol_table (dict): Maps identifiers to values.

    Returns:
    - tosa.ConstOp: Tensor with random integers.
    """
    min_value = symbol_table.get((str(node.args[0]), 0), node.args[0])
    max_value = symbol_table.get((str(node.args[1]), 0), node.args[1])
    shape = symbol_table.get((str(node.args[2]), 0), node.args[2])
    output = ir.DenseElementsAttr.get(
        numpy.random.randint(min_value, max_value, size=shape)
    )
    op = tosa.ConstOp(output)
    return op


def argmax_op(node: ArgMaxOp, symbol_table):
    """
    Compute the indices of the maximum values along the specified axis.

    Args:
        node (ArgMaxOp): The ArgMax operation node with metadata.
        symbol_table: Mapping of variable names to tensor references.

    Returns:
        op: The constructed ArgMax operation.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0), node.args[0])
    axis = symbol_table.get((str(node.args[1]), 0), node.args[1])
    input_shape = list(ir.RankedTensorType(input_tensor.type).shape)

    if axis < 0:
        axis += len(input_shape)

    result_shape = input_shape[:axis] + input_shape[axis + 1 :]
    result_type = ir.IntegerType.get_signless(64)
    result = ir.RankedTensorType.get(result_shape, result_type)
    op = tosa.ArgMaxOp(result, input_tensor, axis)
    return op


def scaled_dot_product_flash_attention_for_cpu_op(
    node: ScaledDotProductFlashAttentionForCpuOp, symbol_table
):
    """
    Perform scaled dot-product attention computation.
    Args:
        node (ScaledDotProductFlashAttentionForCpuOp): The scaled dot-product attention operation node with metadata.
        symbol_table: Mapping of variable names to tensor references.
    Returns:
        result_reshape_op: Reshaped result tensor of the attention operation.
        log_sumexp_op: Log-sum-exp constant operation.
    """
    query = symbol_table.get((str(node.args[0]), 0), node.args[0])
    key = symbol_table.get((str(node.args[1]), 0), node.args[1])
    value = symbol_table.get((str(node.args[2]), 0), node.args[2])

    if len(node.args) == 4:
        dropout_p = node.args[3]
        assert dropout_p != 0.0
    if len(node.args) == 5:
        dropout_p = node.args[3]
        is_causal = node.args[4]
        assert dropout_p != 0.0
        assert is_causal == True

    attn_mask = node.kwargs.get("attn_mask", None)
    scale = node.kwargs.get("scale", None)

    query_shape = query.type.shape
    key_shape = key.type.shape
    value_shape = value.type.shape
    output_shape = list(node.tensor_meta["shape"])
    L, S = query_shape[-2], key_shape[-2]
    scale_factor = (
        1 / numpy.sqrt(query.type.shape[-1]) if scale is None else scale
    )

    # Initialize attention bias
    dtype = node.tensor_meta["dtype"][0]
    attn_bias_shape = [L, S]
    mlir_dtype = mlir_element_type_get(dtype)
    attn_bias_type = ir.RankedTensorType.get(attn_bias_shape, mlir_dtype)
    zero_constant = arith.ConstantOp(mlir_dtype, 0.0)
    attn_bias = tensor.SplatOp(attn_bias_type, zero_constant, [])
    if attn_mask is not None:
        attn_mask = symbol_table.get((str(attn_mask), 0), attn_mask)
        if attn_mask.type.element_type == ir.IntegerType.get_signless(1):
            assert attn_mask.type.element_type == ir.IntegerType.get_signless(1)
            tensor_type = ir.RankedTensorType.get(
                attn_mask.type.shape, ir.IntegerType.get_signless(1)
            )
            true_tensor = arith.ConstantOp(
                tensor_type,
                ir.DenseElementsAttr.get_splat(
                    tensor_type, ir.BoolAttr.get(True)
                ),
            )
            attn_mask = arith.XOrIOp(attn_mask, true_tensor)
            minus_inf_tensor = arith.ConstantOp(
                attn_mask.type,
                ir.DenseElementsAttr.get_splat(
                    attn_mask.type, ir.FloatAttr.get(f32_type, float("-inf"))
                ),
            )
            attn_bias = tensor.SelectOp(attn_mask, minus_inf_tensor, attn_bias)
        else:
            if attn_mask.type.shape != attn_bias.result.type.shape:
                shape = attn_bias.result.type.shape
                shape_ty = ir.Type.parse(f"!tosa.shape<{len(shape)}>")
                index_ty = ir.IndexType.get()
                shape_val = tosa.ConstShapeOp(
                    shape_ty,
                    ir.DenseElementsAttr.get(
                        array.array("q", shape),
                        type=index_ty,
                        shape=[len(shape)],
                    ),
                ).result
                attn_mask = tosa.ReshapeOp(attn_mask, shape_val)
            attn_bias = tosa.AddOp(attn_bias.result.type, attn_bias, attn_mask)

    # Transpose key tensor
    key_shape = list(key.type.shape)
    perm_list = list(range(len(key_shape)))
    perm_list[-1], perm_list[-2] = perm_list[-2], perm_list[-1]
    perms_attr = ir.DenseI32ArrayAttr.get([int(v) for v in perm_list])
    perm_shape = []
    perm_shape.append(key_shape[0])
    perm_shape.append(key_shape[1])
    perm_shape.append(key_shape[3])
    perm_shape.append(key_shape[2])
    permute_result_type = ir.RankedTensorType.get(perm_shape, mlir_dtype)
    key = tosa.TransposeOp(permute_result_type, key, perms_attr).result

    # Matrix multiplication of query and key
    shape = [query_shape[0] * query_shape[1], query_shape[2], query_shape[3]]
    shape_ty = ir.Type.parse(f"!tosa.shape<{len(shape)}>")
    index_ty = ir.IndexType.get()
    shape_val = tosa.ConstShapeOp(
        shape_ty,
        ir.DenseElementsAttr.get(
            array.array("q", shape), type=index_ty, shape=[len(shape)]
        ),
    ).result
    query_reshape_op = tosa.ReshapeOp(query, shape_val)
    shape = [key_shape[0] * key_shape[1], key_shape[3], key_shape[2]]
    shape_ty = ir.Type.parse(f"!tosa.shape<{len(shape)}>")
    shape_val = tosa.ConstShapeOp(
        shape_ty,
        ir.DenseElementsAttr.get(
            array.array("q", shape), type=index_ty, shape=[len(shape)]
        ),
    ).result
    key_reshape_op = tosa.ReshapeOp(key, shape_val)
    matmul_result_shp = [
        key_shape[0] * key_shape[1],
        query_shape[2],
        key_shape[2],
    ]
    matmul_result_type = ir.RankedTensorType.get(matmul_result_shp, mlir_dtype)
    ty = ir.RankedTensorType.get([1], mlir_dtype)
    attr = ir.FloatAttr.get(mlir_dtype, 0.0)
    a_zp = tosa.ConstOp(ir.DenseElementsAttr.get_splat(ty, attr)).result
    b_zp = tosa.ConstOp(ir.DenseElementsAttr.get_splat(ty, attr)).result
    matmul_op = tosa.MatMulOp(
        matmul_result_type,
        query_reshape_op.result,
        key_reshape_op.result,
        a_zp,
        b_zp,
    )
    if mlir_dtype == ir.F16Type.get():
        f16_max_val = 65504.0
        f16_min_val = -65504.0
        min_fp_attr = ir.FloatAttr.get(ir.F16Type.get(), f16_min_val)
        max_fp_attr = ir.FloatAttr.get(ir.F16Type.get(), f16_max_val)

        matmul_op = tosa.ClampOp(
            matmul_op.result.type,
            matmul_op,
            min_fp_attr,
            max_fp_attr,
        )
    # Multiply result by scale factor
    scale_factor_constant = arith.ConstantOp(mlir_dtype, scale_factor)
    scale_factor = tensor.SplatOp(matmul_result_type, scale_factor_constant, [])
    shift = tosa.ConstOp(
        ir.DenseElementsAttr.get_splat(
            ir.Type.parse("tensor<1xi8>"),
            ir.IntegerAttr.get(ir.IntegerType.get_signless(8), 0),
        )
    ).result
    mul_op = tosa.MulOp(
        matmul_result_type,
        matmul_op,
        scale_factor,
        shift,
    )

    # Add attention bias to the result
    add_op = _gen_arith_binary_op(mul_op.result, attn_bias.result, tosa.AddOp)
    # add_op = tosa.AddOp(matmul_result_type, mul_op.result, attn_bias)
    # Apply softmax to the result
    softmax_output_shape = list(add_op.result.type.shape)
    softmax_dim = len(softmax_output_shape) - 1

    # Subtract the maximum value along the dimension where softmax is applied to
    # prevent overflow during the exp operation.
    max_vals = tosa.ReduceMaxOp(add_op.result, softmax_dim)
    sub_op = tosa.SubOp(add_op.result.type, add_op, max_vals)
    exp_op = math.ExpOp(sub_op.result)
    reduce_sum_op = tosa.ReduceSumOp(exp_op, softmax_dim)
    log_op = tosa.LogOp(reduce_sum_op.result.type, reduce_sum_op)
    log_sumexp = tosa.AddOp(max_vals.result.type, max_vals, log_op)
    log_weights = tosa.SubOp(add_op.result.type, add_op, log_sumexp)
    softmax_result = math.ExpOp(log_weights.result)
    shape_ty = ir.Type.parse(f"!tosa.shape<{len(output_shape[1])}>")
    index_ty = ir.IndexType.get()
    shape_val = tosa.ConstShapeOp(
        shape_ty,
        ir.DenseElementsAttr.get(
            array.array("q", output_shape[1]),
            type=index_ty,
            shape=[len(output_shape[1])],
        ),
    ).result
    log_sumexp = tosa.ReshapeOp(log_sumexp, shape_val)

    # This step includes dropout during training.
    # Multiply the result by the value tensor.
    shape = [key_shape[0] * key_shape[1], key_shape[2], key_shape[3]]
    shape_ty = ir.Type.parse(f"!tosa.shape<{len(shape)}>")
    shape_val = tosa.ConstShapeOp(
        shape_ty,
        ir.DenseElementsAttr.get(
            array.array("q", shape), type=index_ty, shape=[len(shape)]
        ),
    ).result
    value_reshape_op = tosa.ReshapeOp(value, shape_val)
    matmul_result_shp = matmul_result_shp = [
        key_shape[0] * key_shape[1],
        query_shape[2],
        value_shape[3],
    ]
    matmul_result_type = ir.RankedTensorType.get(matmul_result_shp, mlir_dtype)
    ty = ir.RankedTensorType.get([1], mlir_dtype)
    attr = ir.FloatAttr.get(mlir_dtype, 0.0)
    a_zp = tosa.ConstOp(ir.DenseElementsAttr.get_splat(ty, attr)).result
    b_zp = tosa.ConstOp(ir.DenseElementsAttr.get_splat(ty, attr)).result
    matmul_op = tosa.MatMulOp(
        matmul_result_type,
        softmax_result.result,
        value_reshape_op.result,
        a_zp,
        b_zp,
    )
    shape = [key_shape[0], key_shape[1], query_shape[2], value_shape[3]]
    shape_ty = ir.Type.parse(f"!tosa.shape<{len(shape)}>")
    index_ty = ir.IndexType.get()
    shape_val = tosa.ConstShapeOp(
        shape_ty,
        ir.DenseElementsAttr.get(
            array.array("q", shape), type=index_ty, shape=[len(shape)]
        ),
    ).result
    query_reshape_op = tosa.ReshapeOp(query, shape_val)
    result_reshape_op = tosa.ReshapeOp(matmul_op.result, shape_val)

    return result_reshape_op, log_sumexp


def matmul_op(
    node: MatmulOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    #     """
    #     Import the tensor matmul operation.
    #     From Buddy MatmulOp to MLIR TOSA `matmul` operation.

    #     Note: This op, compute input node's matrix multiplication result.
    #     Args:
    #         node: Containing information from the input graph node.
    #         symbol_table: A dictionary mapping symbols to their corresponding
    #         operations.

    #     Returns:
    #         op: The operation return the tosa.matmul op.
    #     """

    assert len(node.args) == 2
    mat1 = symbol_table.get((str(node.args[0]), 0))
    mat2 = symbol_table.get((str(node.args[1]), 0))
    # get input shape
    mat1_shp = ir.RankedTensorType(mat1.type).shape
    mat2_shp = ir.RankedTensorType(mat2.type).shape
    # append index because tosa.MatMulOp doesn't accept 2D tensor
    shape1_ty = ir.Type.parse(f"!tosa.shape<{len([1, *mat1_shp])}>")
    index_ty = ir.IndexType.get()
    shape1_val = tosa.ConstShapeOp(
        shape1_ty,
        ir.DenseElementsAttr.get(
            array.array("q", [1, *mat1_shp]),
            type=index_ty,
            shape=[len([1, *mat1_shp])],
        ),
    ).result
    shape2_ty = ir.Type.parse(f"!tosa.shape<{len([1, *mat2_shp])}>")
    shape2_val = tosa.ConstShapeOp(
        shape2_ty,
        ir.DenseElementsAttr.get(
            array.array("q", [1, *mat2_shp]),
            type=index_ty,
            shape=[len([1, *mat2_shp])],
        ),
    ).result
    mat1_reshape_op = tosa.ReshapeOp(mat1, shape1_val)
    mat2_reshape_op = tosa.ReshapeOp(mat2, shape2_val)
    # do matmul
    result_element_type = ir.RankedTensorType(mat1.type).element_type
    matmul_result_shp = [1, mat1_shp[0], mat2_shp[1]]
    matmul_result_type = ir.RankedTensorType.get(
        matmul_result_shp, result_element_type
    )
    ty = ir.RankedTensorType.get([1], result_element_type)
    attr = ir.FloatAttr.get(result_element_type, 0.0)
    a_zp = tosa.ConstOp(ir.DenseElementsAttr.get_splat(ty, attr)).result
    b_zp = tosa.ConstOp(ir.DenseElementsAttr.get_splat(ty, attr)).result
    matmul_op = tosa.MatMulOp(
        matmul_result_type,
        mat1_reshape_op.result,
        mat2_reshape_op.result,
        a_zp,
        b_zp,
    )
    final_result_shape = [mat1_shp[0], mat2_shp[1]]
    shape_ty = ir.Type.parse(f"!tosa.shape<{len(final_result_shape)}>")
    shape_val = tosa.ConstShapeOp(
        shape_ty,
        ir.DenseElementsAttr.get(
            array.array("q", final_result_shape),
            type=index_ty,
            shape=[len(final_result_shape)],
        ),
    ).result
    op = tosa.ReshapeOp(matmul_op.result, shape_val)
    return op


ops_registry = {
    "AddOp": add_op,
    "MulOp": mul_op,
    "SubOp": sub_op,
    "SumDimOp": sum_op,
    "TanhOp": tanh_op,
    "AmaxOp": amax_op,
    "RsqrtOp": rsqrt_op,
    "BatchMatmulOp": bmm_op,
    "CloneOp": clone_op,
    "DivOp": div_op,
    "ExpOp": exp_op,
    "ExpandOp": expand_op,
    "VarMeanOp": var_mean_op,
    "AddMMOp": addmm_op,
    "ReshapeOp": reshape_op,
    "ViewOp": reshape_op,
    "SelectOp": select_op,
    "SliceOp": slice_op,
    "EmbeddingOp": embedding_op,
    "ConvertElementTypeOp": convert_element_type_op,
    "PermuteOp": permute_op,
    "UnsqueezeOp": unsqueeze_op,
    "TOp": t_op,
    "TransposeOp": transpose_op,
    "MaxPool2dOp": maxpool2d_op,
    "Conv2dOp": convolution2d_op,
    "ReluOp": relu_op,
    "IotaOp": iota_op,
    "SigmoidOp": sigmoid_op,
    "ReciprocalOp": reciprocal_op,
    "MeanOp": mean_op,
    "ClampMinOp": clamp_min_op,
    "ClampMaxOp": clamp_max_op,
    "RandIntLowOp": randint_low_op,
    "ArgMaxOp": argmax_op,
    "ScaledDotProductFlashAttentionForCpuOp": scaled_dot_product_flash_attention_for_cpu_op,
    "MatmulOp": matmul_op,
}
