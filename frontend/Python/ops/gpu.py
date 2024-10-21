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
from mlir.dialects import gpu, memref, arith, scf, vector

from ..graph import TensorDType
from ..graph import (
    ReluOp,
    ReshapeOp,
    PermuteOp,
    Conv2dOp,
    MaxPool2dOp,
    AddMMOp
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

    gpu.HostUnregisterOp(input_cast)
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
    perm_map = node.args[1]
    perm_map_attr = ir.AffineMapAttr.get(ir.AffineMap.get_permutation(perm_map))

    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    
    element_type = mlir_element_type_get(dtype)
    element_attr = mlir_element_attr_get(dtype, 0.0)
    
    c0 = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 0))
    f0 = arith.ConstantOp(element_type, element_attr)

    v0 = vector.transfer_read(
        vector=ir.VectorType.get(output_shape, element_type),
        source=input1,
        indices=[c0]*len(output_shape),
        permutation_map=perm_map_attr,
        padding=f0
    )
    
    transpose = memref.AllocOp(ir.MemRefType.get(output_shape, element_type), [], [])

    vector.transfer_write(
        result=None,
        vector=v0,
        source=transpose,
        indices=[c0]*len(output_shape),
        permutation_map=ir.AffineMapAttr.get(
            ir.AffineMap.get_permutation([i for i in range(len(output_shape))])
        )
    )
    return transpose


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
    out_channels = output_shape[1]
    in_size_h = input_shape[2]
    in_size_w = input_shape[3]
    out_size_h = output_shape[2]
    out_size_w = output_shape[3]
    H_filter = filter_shape[2]
    W_filter = filter_shape[3]

    output_val = memref.AllocOp(ir.MemRefType.get(output_shape, element_type), [], [])
    unranked_memref_type = ir.UnrankedMemRefType.get(element_type, ir.IntegerAttr.get(ir.IndexType.get(), 0))
    input_cast = memref.CastOp(unranked_memref_type, input_val)
    filter_cast = memref.CastOp(unranked_memref_type, filter_val)
    bias_cast = memref.CastOp(unranked_memref_type, bias_val)
    output_cast = memref.CastOp(unranked_memref_type, output_val)

    gpu.HostRegisterOp(input_cast)
    gpu.HostRegisterOp(filter_cast)
    gpu.HostRegisterOp(bias_cast)
    gpu.HostRegisterOp(output_cast)

    # Tile the input_val into Grids
    block_z = ((out_size_h + TILE_WIDTH - 1) // TILE_WIDTH) * ((out_size_w + TILE_WIDTH - 1) // TILE_WIDTH)
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
        tile_num = (out_size_w + TILE_WIDTH - 1) // TILE_WIDTH
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
        out_size_h_val = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), out_size_h))
        out_size_w_val = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), out_size_w))
        isHInBounds = arith.cmpi(ult, thread_global_idx, out_size_h_val)
        isWInBounds = arith.cmpi(ult, thread_global_idy, out_size_w_val)
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
    
    gpu.HostUnregisterOp(input_cast)
    gpu.HostUnregisterOp(filter_cast)
    gpu.HostUnregisterOp(bias_cast)
    gpu.HostUnregisterOp(output_cast)

    return output_val


# TODO: Consider the cases where the maxpool2d operation needs padding.
def maxpool2d_op(node: MaxPool2dOp, symbol_table):
    """
    Import the maxpool2d operation.
    From Buddy MaxPool2dOp to MLIR GPU `max_pool2d` kernel.
    """
    if len(node.args) == 5:
        raise NotImplementedError
    input1 = node.args[0]
    kernel = node.args[1]
    stride = node.args[2]

    # Prepare padding data
    if len(node.args) > 3:
        pad = node.args[3]
    else:
        pad = [0 for _ in kernel]

    dtype = node.tensor_meta["dtype"]
    element_type = mlir_element_type_get(dtype)
    output_shape = node.tensor_meta["shape"]

    batch_size = output_shape[0]
    in_channels = output_shape[1]
    out_size_h = output_shape[2]
    out_size_w = output_shape[3]

    input_val = symbol_table.get((str(input1), 0))
    output_val = memref.AllocOp(ir.MemRefType.get(output_shape, element_type), [], [])
    unranked_memref_type = ir.UnrankedMemRefType.get(element_type, ir.IntegerAttr.get(ir.IndexType.get(), 0))
    input_cast = memref.CastOp(unranked_memref_type, input_val)
    output_cast = memref.CastOp(unranked_memref_type, output_val)

    gpu.HostRegisterOp(input_cast)
    gpu.HostRegisterOp(output_cast)

    # Tile the input_val into Grids
    block_z = ((out_size_h + TILE_WIDTH - 1) // TILE_WIDTH) * ((out_size_w + TILE_WIDTH - 1) // TILE_WIDTH)
    batch_size_val = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), batch_size))
    in_channels_val = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), in_channels))
    block_z_val = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), block_z))
    tile_width_val = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), TILE_WIDTH))
    c0 = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 0))
    c1 = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 1))
    
    # threadsPerBlock(TILE_WIDTH, TILE_WIDTH, 1)        numBlocks(N, K, block_z)
    
    gpu_kernel = gpu.LaunchOp(
        asyncToken=None,
        asyncDependencies=[],
        gridSizeX=batch_size_val.result,
        gridSizeY=in_channels_val.result,
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
        in_channel_id = gpu_kernel_block.arguments[1]    
        tile_id = gpu_kernel_block.arguments[2] 
        thread_local_idx = gpu_kernel_block.arguments[3]  
        thread_local_idy = gpu_kernel_block.arguments[4]

        # Calculate the convolution element at (h, w) for this thread
        tile_num = (out_size_w + TILE_WIDTH - 1) // TILE_WIDTH
        tile_num_val = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), tile_num))
        
        t0 = arith.divui(tile_id, tile_num_val)
        t1 = arith.muli(t0, tile_width_val)
        thread_global_idx = arith.addi(t1, thread_local_idx)

        t2 = arith.remui(tile_id, tile_num_val)
        t3 = arith.muli(t2, tile_width_val)
        thread_global_idy = arith.addi(t3, thread_local_idy)

        kernel_size_h = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), kernel[0]))
        kernel_size_w = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), kernel[1]))
        stride_h = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), stride[0]))
        stride_w = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), stride[1]))
        init_ele_idx = arith.muli(thread_global_idx, stride_h)
        init_ele_idy = arith.muli(thread_global_idy, stride_w)

        # Check if the (h, w) is out of the output bounds
        ult = 6
        out_size_h_val = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), out_size_h))
        out_size_w_val = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), out_size_w))
        isHInBounds = arith.cmpi(ult, thread_global_idx, out_size_h_val)
        isWInBounds = arith.cmpi(ult, thread_global_idy, out_size_w_val)
        isInBounds = arith.andi(isHInBounds, isWInBounds)
        
        branch0 = scf.IfOp(isInBounds)
        with ir.InsertionPoint(branch0.then_block):
            first_ele = memref.LoadOp(input_val, [batch_id, in_channel_id, init_ele_idx, init_ele_idy])
            loop0 = scf.ForOp(
                lower_bound=c0.result,
                upper_bound=kernel_size_h.result,
                step=c1.result,
                iter_args=[first_ele.result]
            )
            with ir.InsertionPoint(loop0.body):
                loop1 = scf.ForOp(
                    lower_bound=c0.result,
                    upper_bound=kernel_size_w.result,
                    step=c1.result,
                    iter_args=[first_ele.result]
                )
                with ir.InsertionPoint(loop1.body):
                    # TODO : loop body
                    kernel_ele_idx = loop0.body.arguments[0]
                    kernel_ele_idy = loop1.body.arguments[0]
                    input_ele_idx = arith.addi(init_ele_idx, kernel_ele_idx)
                    input_ele_idy = arith.addi(init_ele_idy, kernel_ele_idy)
                    input_ele = memref.LoadOp(input_val, [batch_id, in_channel_id, input_ele_idx, input_ele_idy])
                    iter_arg1 = loop1.body.arguments[1]
                    iter_res1 = arith.maxnumf(iter_arg1, input_ele)
                    scf.YieldOp([iter_res1])

                iter_arg0 = loop0.body.arguments[1]
                iter_res0 = arith.maxnumf(loop1, iter_arg0)
                scf.YieldOp([iter_res0])

            memref.StoreOp(loop0, output_val, [batch_id, in_channel_id, thread_global_idx, thread_global_idy])
            scf.YieldOp([])
                
        gpu.TerminatorOp()

    gpu.HostUnregisterOp(input_cast)
    gpu.HostUnregisterOp(output_cast)

    return output_val


def addmm_op(
    node: AddMMOp, symbol_table: Dict[Tuple[str, int], ir.Operation]
):
    dtype = node.tensor_meta["dtype"]
    element_type = mlir_element_type_get(dtype)
    c0 = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 0))
    c1 = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 1))
    kernels = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 512))

    # TODO: Reverse the order of the mat2 before multiplication to optimize the cache hit rate

    input_data = symbol_table.get((str(node.args[1]), 0), node.args[1])
    weight = symbol_table.get((str(node.args[2]), 0), node.args[2])
    bias = symbol_table.get((str(node.args[0]), 0), node.args[0])
    # print("input_data: "+str(input_data))
    # print("weight: "+str(weight))
    # print("bias: "+str(bias))

    # TODO: Transpose of the mat2 before multiplication to optimize the cache hit rate

    output_shape = list(node.tensor_meta["shape"])
    input_shape = input_data.type.shape
    weight_shape = weight.type.shape
    # print("output_shape: "+str(output_shape))
    # print("output_shape: "+str())
    # print("input_shape: "+str(input_shape))
    # print("weight_shape: "+str(weight_shape))
    # print("bias shape: "+str(bias.type.shape))

    # Flatten the input into a one-dimensional format 
    input_size = tensor_shape_size(input_shape)
    weight_size = tensor_shape_size(weight_shape)
    output_size = tensor_shape_size(output_shape)
    # print("input_size: "+str(input_size))
    # print("weight_size: "+str(weight_size))
    # print("output_size: "+str(output_size))

    input_size_c = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), input_size))
    weight_size_c = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), weight_size))
    output_size_c = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), output_size))
    # print("input_size_c: "+str(input_size_c))
    # print("weight_size_c: "+str(weight_size_c))
    # print("output_size_c: "+str(output_size_c))

    input_shape_1d = memref.AllocOp(ir.MemRefType.get([1], ir.IndexType.get()), [], [])
    weight_shape_1d = memref.AllocOp(ir.MemRefType.get([1], ir.IndexType.get()), [], [])
    bias_shape_1d = memref.AllocOp(ir.MemRefType.get([1], ir.IndexType.get()), [], [])
    # print("input_shape_1d: "+str(input_shape_1d))
    # print("weight_shape_1d: "+str(weight_shape_1d))
    # print("bias_shape_1d: "+str(bias_shape_1d))

    memref.StoreOp(input_size_c, input_shape_1d, [c0])
    memref.StoreOp(weight_size_c, weight_shape_1d, [c0])
    memref.StoreOp(output_size_c, bias_shape_1d, [c0])

    input_reshape_type = ir.MemRefType.get([input_size], element_type)
    weight_reshape_type = ir.MemRefType.get([weight_size], element_type)
    bias_reshape_type = ir.MemRefType.get([output_size], element_type)
    output_type = ir.MemRefType.get(output_shape, element_type)
    # print("input_reshape_type: "+str(input_reshape_type))
    # print("weight_reshape_type: "+str(weight_reshape_type))
    # print("bias_reshape_type: "+str(bias_reshape_type))
    # print("output_type: "+str(output_type))

    input_reshape_1d = memref.ReshapeOp(input_reshape_type, input_data, input_shape_1d)
    weight_reshape_1d = memref.ReshapeOp(weight_reshape_type, weight, weight_shape_1d)
    bias_reshape_1d = memref.ReshapeOp(bias_reshape_type, bias, bias_shape_1d)
    # print("input_reshape: "+str(input_reshape_1d))
    # print("weight_reshape: "+str(weight_reshape_1d))
    # print("bias_reshape: "+str(bias_reshape_1d))


    unranked_memref_type = ir.UnrankedMemRefType.get(element_type, ir.IntegerAttr.get(ir.IndexType.get(), 0))
    gpu.HostRegisterOp(memref.CastOp(unranked_memref_type, input_reshape_1d))
    gpu.HostRegisterOp(memref.CastOp(unranked_memref_type, weight_reshape_1d))
    gpu.HostRegisterOp(memref.CastOp(unranked_memref_type, bias_reshape_1d))

    row = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), input_shape[0]))
    col = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), weight_shape[1]))
    inner_dim = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), input_shape[1]))

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

    # TODO: optimize to one dimension
    with ir.InsertionPoint(gpu_kernel_block):
        tIdX = gpu_kernel_block.arguments[3]
        tIdY = gpu_kernel_block.arguments[4]
        otter_loop = scf.ForOp(
            lower_bound=tIdX,
            upper_bound=row,
            step=gpu_kernel.blockSizeX
        )
        with ir.InsertionPoint(otter_loop.body):
            inner_loop = scf.ForOp(
                lower_bound=tIdY,
                upper_bound=col,
                step=gpu_kernel.blockSizeY
            )
            with ir.InsertionPoint(inner_loop.body):
                initial_sum = arith.ConstantOp(ir.F32Type.get(), ir.FloatAttr.get(ir.F32Type.get(), 0.0))

                mul_loop = scf.ForOp(
                    lower_bound=c0.result,
                    upper_bound=inner_dim,
                    step=c1.result,
                    iter_args=[initial_sum]
                )
                with ir.InsertionPoint(mul_loop.body):
                    sum = mul_loop.inner_iter_args[0]
                    mat1_load = memref.LoadOp(input_reshape_1d, [arith.AddIOp(arith.MulIOp(otter_loop.induction_variable, inner_dim).result, mul_loop.induction_variable)])
                    mat2_load = memref.LoadOp(weight_reshape_1d, [arith.AddIOp(arith.MulIOp(mul_loop.induction_variable, col).result, inner_loop.induction_variable)])
                    res = arith.MulFOp(mat1_load, mat2_load)
                    res = arith.AddFOp(sum, res)
                    scf.YieldOp([res])
                
                sum = mul_loop.result
                bias_load = memref.LoadOp(bias_reshape_1d, [arith.AddIOp(arith.MulIOp(otter_loop.induction_variable, col).result, inner_loop.induction_variable)])
                res = arith.AddFOp(sum, bias_load)
                memref.StoreOp(res, bias_reshape_1d, [arith.AddIOp(arith.MulIOp(otter_loop.induction_variable, col).result, inner_loop.induction_variable)])
                scf.YieldOp([])
            scf.YieldOp([])
        gpu.TerminatorOp()


    output = memref.AllocOp(ir.MemRefType.get(output_shape, element_type), [], [])

    # FIXME: Dialect `memref' not found for custom op 'memref.expand_shape' 
    # axis = ir.ArrayAttr.get(
    #     [
    #         ir.IntegerAttr.get(ir.IntegerType.get_signless(64), i)
    #         for i in range(len(output_shape))
    #     ],
    #     None,
    # )
    # axis = ir.ArrayAttr.get([axis], None)
    # bias_reshape = memref.ExpandShapeOp(output_type, bias, axis)

    bias_shape = memref.AllocOp(ir.MemRefType.get([len(output_shape)], ir.IndexType.get()), [], [])
    # print("bias_shape: "+str(bias_shape))
    for i in range(len(output_shape)):
        memref.StoreOp(arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), output_shape[i])), bias_shape, [arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), i))])

    bias_reshape = memref.ReshapeOp(output_type, bias, bias_shape)
    memref.CopyOp(bias_reshape, output)
    return output


ops_registry = {
    "ReluOp": relu_op,
    "ViewOp": reshape_op,
    "PermuteOp": permute_op,
    "Conv2dOp": convolution2d_op,
    "MaxPool2dOp": maxpool2d_op,
    "AddMMOp": addmm_op
}
