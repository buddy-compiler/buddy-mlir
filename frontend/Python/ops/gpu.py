# ===- gpu.py -----------------------------------------------------------------
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
# The registry of mappings from Buddy node to MLIR GPU kernel.
#
# ===---------------------------------------------------------------------------


from typing import Tuple
import mlir.ir as ir
from mlir.dialects import gpu, memref, arith, scf

from ..graph import TensorDType
from ..graph import (
    ReluOp,
    ReshapeOp,
    PermuteOp
)
from .utils import *

def relu_op(node: ReluOp, symbol_table: Dict[Tuple[str, int], ir.Operation]):
    """
    Import the buddy ReluOp.
    From Buddy ReluOp to MLIR Relu GPU kernel.
    """
    assert len(node.args) == 1
    input = symbol_table.get((str(node.args[0]), 0))
    if input is None:
        return
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    element_type = mlir_element_type_get(dtype)

    c0 = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 0))
    c1 = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 1))
    kernels = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 512))

    # Flatten the input into a one-dimensional format 
    output_size = tensor_shape_size(output_shape)
    size = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), output_size))
    shape = memref.AllocOp(ir.MemRefType.get([1], ir.IndexType.get()), [], [])
    memref.StoreOp(size, shape, [c0])
    memref_reshape_type = ir.MemRefType.get([output_size], element_type)
    input_reshape = memref.ReshapeOp(memref_reshape_type, input, shape)

    unranked_memref_type = ir.UnrankedMemRefType.get(element_type, ir.IntegerAttr.get(ir.IndexType.get(), 0))
    input_cast = memref.CastOp(unranked_memref_type, input)
    gpu.HostRegisterOp(input_cast)
    gpu_kernel = gpu.LaunchOp(
        asyncToken=None,
        asyncDependencies=[],
        gridSizeX=c1.result, gridSizeY=c1.result, gridSizeZ=c1.result,
        blockSizeX=kernels.result, blockSizeY=c1.result, blockSizeZ=c1.result,
    )
    gpu_kernel_block = ir.Block.create_at_start(
        gpu_kernel.body,
        [
            ir.IndexType.get(), ir.IndexType.get(), ir.IndexType.get(),     # block_idx, block_idy, block_idz 
            ir.IndexType.get(), ir.IndexType.get(), ir.IndexType.get(),     # thread_idx , thread_idy, thread_idz 
            ir.IndexType.get(), ir.IndexType.get(), ir.IndexType.get(),     # grid_size x, grid_size y, grid_size z 
            ir.IndexType.get(), ir.IndexType.get(), ir.IndexType.get(),     # block_size x, block_size y, block_size z
        ]
    )

    with ir.InsertionPoint(gpu_kernel_block):
        tIdX = gpu_kernel_block.arguments[3]
        cst_0 = arith.ConstantOp(ir.F32Type.get(), ir.FloatAttr.get(ir.F32Type.get(), 0.0))
        loop = scf.ForOp(
            lower_bound=tIdX,
            upper_bound=size,
            step=gpu_kernel.blockSizeX
        )
        with ir.InsertionPoint(loop.body):
            load = memref.LoadOp(input_reshape, [loop.induction_variable])
            result = arith.MaxNumFOp(load, cst_0)
            memref.StoreOp(result, input_reshape, [loop.induction_variable])
            scf.YieldOp([])
        
        gpu.TerminatorOp()
    output = memref.AllocOp(ir.MemRefType.get(output_shape, element_type), [], [])
    memref.CopyOp(input, output)
    return output

# TODO: Implement Reshape Operation on GPU in future revisions.

def reshape_op(node: ReshapeOp, symbol_table):
    """
    Import the reshape operation.
    From buddy graph ir's `ReshapeOp` operator to MLIR Memref `reshape`
    operation.

    Note: If the new shape contains one and only one `-1`, the size of the new
    shape will be inferred automatically.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    new_shape = []
    for i in node.args[1]:
        new_shape.append(i)
    output_shape = list(node.tensor_meta["shape"])
    total_size = tensor_shape_size(output_shape)

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

    shape = memref.AllocOp(ir.MemRefType.get([len(new_shape)], ir.IndexType.get()), [], [])
    for i, _ in enumerate(new_shape):
        c = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), i))
        size = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), new_shape[i]))
        memref.StoreOp(size, shape, [c])

    dtype = node.tensor_meta["dtype"]
    element_type = mlir_element_type_get(dtype)
    output_type = ir.MemRefType.get(new_shape, element_type)
    op = memref.ReshapeOp(output_type, input1, shape)

    return op

# TODO: Implement Permute Operation on GPU in future revisions.

def permute_op(node: PermuteOp, symbol_table):
    """
    Import the permute operation.
    From buddy graph ir's `PermuteOp` operator to MLIR Memref `transpose`
    operation.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    perm = node.args[1]
    perm_attr = ir.AffineMapAttr.get(ir.AffineMap.get_permutation(perm))

    output_shape = list(node.tensor_meta["shape"])
    element_type = mlir_element_type_get(node.tensor_meta["dtype"])
    input_shape = [0] * len(output_shape)
    for i, p in enumerate(perm):
        input_shape[p] = output_shape[i]

    offset = 0
    input_stride = generate_strides(input_shape)
    output_stride = transpose_strides(input_stride, perm)
    result_type = ir.MemRefType.get(
        shape=output_shape,
        element_type=element_type,
        layout=ir.StridedLayoutAttr.get(offset, output_stride)
    )
    permute_op = memref.TransposeOp(
        result=result_type,
        in_=input1,
        permutation=perm_attr
    )
    return permute_op

ops_registry = {
    "ReluOp": relu_op,
    "ViewOp": reshape_op,
    "PermuteOp": permute_op
}
