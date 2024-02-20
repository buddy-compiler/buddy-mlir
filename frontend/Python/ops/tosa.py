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

import array
from typing import Dict, List, Tuple, Union
import numpy

import mlir.ir as ir
from mlir.dialects import tensor, tosa

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
    # append index because tosa.MatMulOp doesn't accept 2D tensor
    mat1_reshape_op = tosa.ReshapeOp(
        mat1, memoryview(array.array("i", [1, *mat1_shp]))
    )
    mat2_reshape_op = tosa.ReshapeOp(
        mat2, memoryview(array.array("i", [1, *mat2_shp]))
    )
    # do matmul
    result_element_type = ir.RankedTensorType(mat1.type).element_type
    matmul_result_shp = [1, mat1_shp[0], mat2_shp[1]]
    matmul_result_type = ir.RankedTensorType.get(
        matmul_result_shp, result_element_type
    )
    matmul_op = tosa.MatMulOp(
        matmul_result_type, mat1_reshape_op.result, mat2_reshape_op.result
    )
    # restore the shape
    final_result_shape = [mat1_shp[0], mat2_shp[1]]
    matmul_result_reshape_op = tosa.ReshapeOp(
        matmul_op.c, memoryview(array.array("i", final_result_shape))
    )

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
    op = tosa.MatMulOp(result_type, input_, mat2)
    return op


def add_op(node: AddOp, symbol_table):
    """
    Import tensor addition operation.
    From buddy graph ir's `AddOp` operator to MLIR TOSA `add` operation.
    """
    input1 = symbol_table.get((str(node.args[0]), 0), node.args[0])
    input2 = symbol_table.get((str(node.args[1]), 0), node.args[1])
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
        return tosa.MulOp(
            result_type,
            input1,
            input2,
            ir.IntegerAttr.get(ir.IntegerType.get_signless(8), 0),
        )

    input1 = symbol_table.get((str(node.args[0]), 0), node.args[0])
    input2 = symbol_table.get((str(node.args[1]), 0), node.args[1])

    return _gen_arith_binary_op(input1, input2, _inner_op)


def div_op(node: DivOp, symbol_table):
    """
    Import tensor division operation.
    From buddy graph ir's `DivOp` operator to MLIR TOSA `div` operation.
    """

    def _inner_op(result_type, input1, input2):
        return tosa.MulOp(
            result_type,
            input1,
            tosa.ReciprocalOp(input2.type, input2).result,
            ir.IntegerAttr.get(ir.IntegerType.get_signless(8), 0),
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

    new_shape_content = array.array("i", new_shape)
    new_shape_content = memoryview(new_shape_content)
    op = tosa.ReshapeOp(input1, new_shape_content)

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
    new_shape_content = array.array("i", sizes)
    new_shape_content = memoryview(new_shape_content)
    op = tosa.ReshapeOp(input_tensor, new_shape_content)
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
    new_sizes_attr = ir._denseI64ArrayAttr(new_sizes, None)

    start = [0] * len(sizes)
    start[dim] = index
    start_attr = ir._denseI64ArrayAttr(start, None)

    result_element_type = ir.RankedTensorType(input_tensor.type).element_type
    output_type = ir.RankedTensorType.get(new_sizes, result_element_type)
    op = tosa.SliceOp(output_type, input_tensor, start_attr, new_sizes_attr)

    reshape_sizes = sizes[:dim] + sizes[dim + 1 :]
    reshape_sizes_content = array.array("i", reshape_sizes)
    reshape_sizes_content = memoryview(reshape_sizes_content)
    op = tosa.ReshapeOp(op.results[0], reshape_sizes_content)

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
        TensorDType.Int32: ir.IntegerType.get_signless(32),
        TensorDType.Bool: ir.IntegerType.get_signless(1),
    }
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    to_cast_type = types_mapping[node.args[1]]
    sizes = ir.RankedTensorType(input_tensor.type).shape
    output_type = ir.RankedTensorType.get(sizes, to_cast_type)
    return tosa.CastOp(output_type, input_tensor)


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

        return tosa.MulOp(
            reduce_sum_op.results[0].type,
            reciprocal_op.results[0],
            reduce_sum_op.results[0],
            ir.IntegerAttr.get(ir.IntegerType.get_signless(8), 0),
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
        mul_op: ir.Operation = tosa.MulOp(
            _input_tensor.type,
            sub_op.results[0],
            sub_op.results[0],
            ir.IntegerAttr.get(ir.IntegerType.get_signless(8), 0),
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

        return tosa.MulOp(
            reduce_sum_op.results[0].type,
            reciprocal_op.results[0],
            reduce_sum_op.results[0],
            ir.IntegerAttr.get(ir.IntegerType.get_signless(8), 0),
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
        var_op = tosa.ReshapeOp(
            var_op.results[0], memoryview(array.array("i", result_shp))
        )
        mean_op = tosa.ReshapeOp(
            mean_op.results[0], memoryview(array.array("i", result_shp))
        )

    return var_op, mean_op


def permute_op(node: PermuteOp, symbol_table):
    """
    Import the permute operation.
    From buddy graph ir's `PermuteOp` operator to MLIR TOSA `transpose`
    operation.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    perm = node.args[1]
    perm_const_op = tosa.ConstOp(
        ir.DenseElementsAttr.get(memoryview(array.array("i", perm)))
    )
    result_element_type = ir.RankedTensorType(input_tensor.type).element_type
    init_shape = ir.RankedTensorType(input_tensor.type).shape
    new_shape = []
    for perm_item in perm:
        new_shape.append(init_shape[perm_item])

    permute_result_type = ir.RankedTensorType.get(
        new_shape, result_element_type
    )
    permute_op = tosa.TransposeOp(
        permute_result_type, input_tensor, perm_const_op.results[0]
    )
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
        indices_reshape_op = tosa.ReshapeOp(
            indices, memoryview(array.array("i", [1, total_size]))
        )
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

    weight_reshape_op = tosa.ReshapeOp(
        weight, memoryview(array.array("i", [1, *weight_size]))
    )

    gather_op = tosa.GatherOp(
        gather_result_type, weight_reshape_op.result, indices
    )
    op = tosa.ReshapeOp(
        gather_op.output,
        memoryview(array.array("i", [*indices_size, weight_size[1]])),
    )

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
    new_size = node.args[1]
    result_element_type = ir.RankedTensorType(
        to_expand_tensor.type
    ).element_type
    if result_element_type == ir.IntegerType.get_signless(1):
        element = ir.IntegerAttr.get(result_element_type, 0)
    elif result_element_type == ir.F32Type.get():
        element = ir.FloatAttr.get(result_element_type, 0.0)
    else:
        raise NotImplementedError("Unsupported element type!")
    new_size_tensor_type = ir.RankedTensorType.get(
        new_size, result_element_type
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
    perm_const_op = tosa.ConstOp(
        ir.DenseElementsAttr.get(memoryview(array.array("i", [1, 0])))
    )
    result_element_type = ir.RankedTensorType(input1.type).element_type
    permute_result_type = ir.RankedTensorType.get(
        output_shape, result_element_type
    )
    op = tosa.TransposeOp(permute_result_type, input1, perm_const_op.results[0])

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
    From Buddy MaxPool2dOp to MLIR TOSA `max_pool2d` operation.
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
        perm_const_op = tosa.ConstOp(
            ir.DenseElementsAttr.get(memoryview(array.array("i", perm_list)))
        )
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
            permute_result_type, input1, perm_const_op.results[0]
        ).result
    out_shape = node.tensor_meta["shape"]
    if len(pad) == 1:
        pad = [pad[0]] * 4
    elif len(pad) == 2:
        pad = [pad[0]] * 2 + [pad[1]] * 2
    kernel_attr = ir._denseI64ArrayAttr(kernel, None)
    stride_attr = ir._denseI64ArrayAttr(stride, None)
    pad_attr = ir._denseI64ArrayAttr(pad, None)
    if node._layout.find("NCHW") != -1:
        perm_shape = []
        perm_shape.append(out_shape[0])
        perm_shape.append(out_shape[2])
        perm_shape.append(out_shape[3])
        perm_shape.append(out_shape[1])
        out_shape = perm_shape
    output = ir.RankedTensorType.get(out_shape, result_element_type)
    op = tosa.MaxPool2dOp(output, input1, kernel_attr, stride_attr, pad_attr)
    if node._layout.find("NCHW") != -1:
        perm_list = [0, 3, 1, 2]
        perm_const_op = tosa.ConstOp(
            ir.DenseElementsAttr.get(memoryview(array.array("i", perm_list)))
        )
        perm_shape = []
        perm_shape.append(out_shape[0])
        perm_shape.append(out_shape[3])
        perm_shape.append(out_shape[1])
        perm_shape.append(out_shape[2])
        permute_result_type = ir.RankedTensorType.get(
            perm_shape, result_element_type
        )
        op = tosa.TransposeOp(
            permute_result_type, op.result, perm_const_op.results[0]
        )
    return op

def convolution2d_op(node: Conv2dOp, symbol_table):
    """
    Import the convolution operation.
    From Buddy Conv2dOp to MLIR TOSA `conv2d` operation.
    """
    assert len(node.args) == 9
    input1 = symbol_table.get((str(node.args[0]), 0))
    weight = symbol_table.get((str(node.args[1]), 0))
    is_kernel_transposed = node.args[6]
    dtype = node.tensor_meta["dtype"]
    result_element_type = mlir_element_type_get(dtype)
    if node._layout.find("NCHW") != -1:
        perm_list = [0, 2, 3, 1]
        perm_const_op = tosa.ConstOp(
            ir.DenseElementsAttr.get(memoryview(array.array("i", perm_list)))
        )
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
            permute_result_type, input1, perm_const_op.results[0]
        ).result
    if node._layout.find("FCHW") != -1:
        perm_list = [0, 2, 3, 1]
        perm_const_op = tosa.ConstOp(
            ir.DenseElementsAttr.get(memoryview(array.array("i", perm_list)))
        )
        out_shape = list(ir.RankedTensorType(weight.type).shape)
        perm_shape = []
        perm_shape.append(out_shape[0])
        perm_shape.append(out_shape[2])
        perm_shape.append(out_shape[3])
        perm_shape.append(out_shape[1])
        permute_result_type = ir.RankedTensorType.get(
            perm_shape, result_element_type
        )
        weight = tosa.TransposeOp(
            permute_result_type, weight, perm_const_op.results[0]
        ).result
    if is_kernel_transposed:
        in_channels = list(ir.RankedTensorType(weight.type).shape)[0]
        out_channels = list(ir.RankedTensorType(weight.type).shape)[1]
    else:
        in_channels = list(ir.RankedTensorType(weight.type).shape)[1]
        out_channels = list(ir.RankedTensorType(weight.type).shape)[0]
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
        bias_tensor = symbol_table.get((str(node.args[2]), 0))
    assert input1 != None and weight != None and bias_tensor != None
    stride = node.args[3]
    input_padding = node.args[4]
    if len(input_padding) == 1:
        input_padding = [input_padding[0]] * 4
    elif len(input_padding) == 2:
        input_padding = [input_padding[0]] * 2 + [input_padding[1]] * 2
    dilation = node.args[5]
    groups = node.args[8]
    out_shape = node.tensor_meta["shape"]
    if node._layout.find("NCHW") != -1:
        perm_shape = []
        perm_shape.append(out_shape[0])
        perm_shape.append(out_shape[2])
        perm_shape.append(out_shape[3])
        perm_shape.append(out_shape[1])
        out_shape = perm_shape
    output = ir.RankedTensorType.get(out_shape, result_element_type)
    stride_attr = ir._denseI64ArrayAttr(stride, None)
    assert groups == 1, 'tosa.conv2d only support one group'
    if is_kernel_transposed:
        if sum(input_padding) > 0 or sum(dilation) > len(dilation):
            raise NotImplementedError
        out_padding = node.args[7]
        for i in range(len(out_padding), 4):
            out_padding = [0] + out_padding
        out_padding_attr = ir._denseI64ArrayAttr(out_padding, None)
        out_shape_attr = ir._denseI64ArrayAttr(out_shape, None)
        op = tosa.TransposeConv2DOp(
            output,
            input1,
            weight,
            bias_tensor,
            out_padding_attr,
            stride_attr,
            out_shape_attr,
        )
    else:
        input_padding_attr = ir._denseI64ArrayAttr(input_padding, None)
        dilation_attr = ir._denseI64ArrayAttr(dilation, None)
        op = tosa.Conv2DOp(
            output,
            input1,
            weight,
            bias_tensor,
            input_padding_attr,
            stride_attr,
            dilation_attr,
        )
    if node._layout.find("NCHW") != -1:
        perm_list = [0, 3, 1, 2]
        perm_const_op = tosa.ConstOp(
            ir.DenseElementsAttr.get(memoryview(array.array("i", perm_list)))
        )
        perm_shape = []
        perm_shape.append(out_shape[0])
        perm_shape.append(out_shape[3])
        perm_shape.append(out_shape[1])
        perm_shape.append(out_shape[2])
        permute_result_type = ir.RankedTensorType.get(
            perm_shape, result_element_type
        )
        op = tosa.TransposeOp(
            permute_result_type, op.result, perm_const_op.results[0]
        )
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
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    element = mlir_element_attr_get(dtype, 0)
    tensor_type = ir.RankedTensorType.get(output_shape, element.type)
    attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
    zero_op = tosa.ConstOp(attr)
    result_element_type = mlir_element_type_get(dtype)
    op = tosa.MaximumOp(tensor_type, input1, zero_op)

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

    ret = tosa.MulOp(
        reduce_sum_op.results[0].type,
        reciprocal_op.results[0],
        reduce_sum_op.results[0],
        ir.IntegerAttr.get(ir.IntegerType.get_signless(8), 0),
    )

    if not keepdim:
        result_shp = ir.RankedTensorType(ret.results[0].type).shape
        result_shp = [siz for siz in result_shp if siz != 1]
        ret = tosa.ReshapeOp(
            ret.results[0], memoryview(array.array("i", result_shp))
        )

    return ret


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
}
