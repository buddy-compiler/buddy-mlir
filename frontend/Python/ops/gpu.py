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
    PermuteOp,
    Conv2dOp,
    MaxPool2dOp
)
from .utils import *

TILE_WIDTH = 16

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
        gridSizeX=c1.result, 
        gridSizeY=c1.result, 
        gridSizeZ=c1.result,
        blockSizeX=kernels.result, 
        blockSizeY=c1.result, 
        blockSizeZ=c1.result,
    )
    gpu_kernel_block = ir.Block.create_at_start(
        gpu_kernel.body,
        [
            ir.IndexType.get(),     # block_id x
            ir.IndexType.get(),     # block_id y 
            ir.IndexType.get(),     # block_id z 
            ir.IndexType.get(),     # thread_id x
            ir.IndexType.get(),     # thread_id y  
            ir.IndexType.get(),     # thread_id z
            ir.IndexType.get(),     # grid_size x
            ir.IndexType.get(),     # grid_size y
            ir.IndexType.get(),     # grid_size z
            ir.IndexType.get(),     # block_size x
            ir.IndexType.get(),     # block_size y
            ir.IndexType.get(),     # block_size z
        ]
    )

    with ir.InsertionPoint(gpu_kernel_block):
        thread_local_idx = gpu_kernel_block.arguments[3]
        element_attr = mlir_element_attr_get(dtype, 0.0)
        cst_0 = arith.ConstantOp(element_type, element_attr)
        loop = scf.ForOp(
            lower_bound=thread_local_idx,
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


# TODO: Consider the cases where the arguments take different values.
def convolution2d_op(node: Conv2dOp, symbol_table):
    """
    Import the convolution operation.
    From Buddy Conv2dOp to MLIR GPU `conv2d` kernel.
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
    filter = node.args[1]
    bias = node.args[2]
    stride = node.args[3]
    input_padding = node.args[4]
    dilation = node.args[5]
    is_kernel_transposed = node.args[6]
    out_padding = node.args[7]
    groups = node.args[8]

    # TODO: Consider the cases where the variables take different values.
    assert input_padding[0] == input_padding[1] == 0
    assert dilation[0] == dilation[1] == 1
    assert is_kernel_transposed == False
    assert out_padding[0] == out_padding[1] == 0
    assert groups == 1

    # Prepare input, filter, and output information.
    input_val = symbol_table.get((str(input), 0))
    input_shape = list(ir.MemRefType(input_val.type).shape)
    filter_val = symbol_table.get((str(filter), 0))
    filter_shape = ir.MemRefType(filter_val.type).shape
    bias_val = symbol_table.get((str(bias), 0))
    dtype = node.tensor_meta["dtype"]
    element_type = mlir_element_type_get(dtype)
    output_shape = list(node.tensor_meta["shape"])

    batch_size = input_shape[0]
    in_channels = input_shape[1]
    out_channels = output_shape[0]
    H_in = input_shape[2]
    W_in = input_shape[3]
    H_out = output_shape[2]
    W_out = output_shape[3]
    H_filter = filter_shape[2]
    W_filter = filter_shape[3]

    output_val = memref.AllocOp(ir.MemRefType.get(output_shape, element_type), [], [])
    unranked_memref_type = ir.UnrankedMemRefType.get(element_type, ir.IntegerAttr.get(ir.IndexType.get(), 0))
    input_cast = memref.CastOp(unranked_memref_type, input_val)
    filter_cast = memref.CastOp(unranked_memref_type, filter_val)
    output_cast = memref.CastOp(unranked_memref_type, output_val)

    gpu.HostRegisterOp(input_cast)
    gpu.HostRegisterOp(filter_cast)
    gpu.HostRegisterOp(output_cast)

    # Tile the input_val into Grids
    block_z = ((H_out + TILE_WIDTH - 1) // TILE_WIDTH) * ((W_out + TILE_WIDTH - 1) // TILE_WIDTH)
    batch_size_val = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), batch_size))
    in_channels_val = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), in_channels))
    out_channels_val = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), out_channels))
    block_z_val = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), block_z))
    tile_width_val = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), TILE_WIDTH))
    H_filter_val = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), H_filter))
    W_filter_val = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), W_filter))
    c0 = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 0))
    c1 = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 1))
    
    # threadsPerBlock(TILE_WIDTH, TILE_WIDTH, 1)        numBlocks(N, K, block_z)
    
    gpu_kernel = gpu.LaunchOp(
        asyncToken=None,
        asyncDependencies=[],
        gridSizeX=batch_size_val.result,
        gridSizeY=out_channels_val.result,
        gridSizeZ=block_z_val.result,
        blockSizeX=tile_width_val.result,
        blockSizeY=tile_width_val.result,
        blockSizeZ=c1.result,
    )

    gpu_kernel_block = ir.Block.create_at_start(
        gpu_kernel.body,
        [
            ir.IndexType.get(),     # block_id x
            ir.IndexType.get(),     # block_id y 
            ir.IndexType.get(),     # block_id z 
            ir.IndexType.get(),     # thread_id x
            ir.IndexType.get(),     # thread_id y  
            ir.IndexType.get(),     # thread_id z
            ir.IndexType.get(),     # grid_size x
            ir.IndexType.get(),     # grid_size y
            ir.IndexType.get(),     # grid_size z
            ir.IndexType.get(),     # block_size x
            ir.IndexType.get(),     # block_size y
            ir.IndexType.get(),     # block_size z
        ]
    )

    with ir.InsertionPoint(gpu_kernel_block):
        batch_id = gpu_kernel_block.arguments[0]    
        out_channel_id = gpu_kernel_block.arguments[1]    
        tile_id = gpu_kernel_block.arguments[2] 
        thread_local_idx = gpu_kernel_block.arguments[3]  
        thread_local_idy = gpu_kernel_block.arguments[4]

        # Calculate the convolution element at (h, w) for this thread
        tile_num = (W_out + TILE_WIDTH - 1) // TILE_WIDTH
        tile_num_val = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), tile_num))
        
        t0 = arith.divui(tile_id, tile_num_val)
        t1 = arith.muli(t0, tile_width_val)
        thread_global_idx = arith.addi(t1, thread_local_idx)

        t2 = arith.remui(tile_id, tile_num_val)
        t3 = arith.muli(t2, tile_width_val)
        thread_global_idy = arith.addi(t3, thread_local_idy)

        stride_h = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), stride[0]))
        stride_w = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), stride[1]))
        t4 = arith.muli(thread_global_idx, stride_h)
        t5 = arith.muli(thread_global_idy, stride_w)

        # Check if the (h, w) is out of the output bounds
        ult = 6
        H_out_val = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), H_out))
        W_out_val = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), W_out))
        isHInBounds = arith.cmpi(ult, thread_global_idx, H_out_val)
        isWInBounds = arith.cmpi(ult, thread_global_idy, W_out_val)
        isInBounds = arith.andi(isHInBounds, isWInBounds)
        
        cst_0 = arith.ConstantOp(element_type, mlir_element_attr_get(dtype, 0.0))
        branch0 = scf.IfOp(isInBounds)
        with ir.InsertionPoint(branch0.then_block):
            loop0 = scf.ForOp(
                lower_bound=c0.result,
                upper_bound=in_channels_val.result,
                step=c1.result,
                iter_args=[cst_0.result]
            )
            with ir.InsertionPoint(loop0.body):
                loop1 = scf.ForOp(
                    lower_bound=c0.result,
                    upper_bound=H_filter_val.result,
                    step=c1.result,
                    iter_args=[cst_0.result]
                )
                with ir.InsertionPoint(loop1.body):
                    loop2 = scf.ForOp(
                        lower_bound=c0.result,
                        upper_bound=W_filter_val.result,
                        step=c1.result,
                        iter_args=[cst_0.result]
                    )
                    with ir.InsertionPoint(loop2.body):
                        # TODO : loop body
                        in_channel_id = loop0.body.arguments[0]
                        filter_ele_idx = loop1.body.arguments[0]
                        filter_ele_idy = loop2.body.arguments[0]
                        input_ele_idx = arith.addi(t4, filter_ele_idx)
                        input_ele_idy = arith.addi(t5, filter_ele_idy)
                        input_ele = memref.LoadOp(input_val, [batch_id, in_channel_id, input_ele_idx, input_ele_idy])
                        filter_ele = memref.LoadOp(filter_val, [out_channel_id, in_channel_id, filter_ele_idx, filter_ele_idy])
                        t6 = arith.mulf(input_ele, filter_ele)
                        iter_arg2 = loop2.body.arguments[1]
                        iter_res2 = arith.addf(iter_arg2, t6)
                        scf.YieldOp([iter_res2])

                    iter_arg1 = loop1.body.arguments[1]
                    iter_res1 = arith.addf(loop2, iter_arg1)
                    scf.YieldOp([iter_res1])

                iter_arg0 = loop0.body.arguments[1]
                iter_res0 = arith.addf(loop1, iter_arg0)
                scf.YieldOp([iter_res0])

            # Add bias data for any out_channel.
            bias_ele = memref.LoadOp(bias_val, [out_channel_id])
            result = arith.addf(loop0, bias_ele)
            memref.StoreOp(result, output_val, [batch_id, out_channel_id, thread_global_idx, thread_global_idy])
            scf.YieldOp([])
                
        gpu.TerminatorOp()

    return output_val


ops_registry = {
    "ReluOp": relu_op,
    "ViewOp": reshape_op,
    "PermuteOp": permute_op,
    "Conv2dOp": convolution2d_op,
}
