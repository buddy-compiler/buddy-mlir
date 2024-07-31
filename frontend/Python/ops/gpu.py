# ===- func.py -----------------------------------------------------------------
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
    ReluOp
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
    element = mlir_element_attr_get(dtype, 0)
    memref_type = ir.MemrefType.get(output_shape, element.type)
    unranked_memref_type = ir.UnrankedMemRefType.get(dtype, ir.IntegerAttr.get(ir.IndexType.get(), 0))
    input_cast = memref.CastOp(unranked_memref_type, input)
    gpu.HostRegisterOp(input_cast)

    c1 = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 1))
    c512 = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 512))
    size = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 1024))

    gpu_kernel = gpu.LaunchOp(
        asyncToken=None,
        asyncDependencies=[],
        gridSizeX=c1.result, gridSizeY=c1.result, gridSizeZ=c1.result,
        blockSizeX=c512.result, blockSizeY=c1.result, blockSizeZ=c1.result,
    )
    # Create a GPU kernel block and define grid and block dimensions for GPU execution
    gpu_kernel_block = ir.Block.create_at_start(
        gpu_kernel.body,
        [
            ir.IndexType.get(),  # %bx : index, Block index X
            ir.IndexType.get(),  # %by : index, Block index Y
            ir.IndexType.get(),  # %bz : index, Block index Z
            ir.IndexType.get(),  # %tx : index, Thread index X
            ir.IndexType.get(),  # %ty : index, Thread index Y
            ir.IndexType.get(),  # %tz : index, Thread index Z
            ir.IndexType.get(),  # %num_bx : index, Grid size X
            ir.IndexType.get(),  # %num_by : index, Grid size Y
            ir.IndexType.get(),  # %num_bz : index, Grid size Z
            ir.IndexType.get(),  # %num_tx : index, Block size X
            ir.IndexType.get(),  # %num_ty : index, Block size Y
            ir.IndexType.get(),  # %num_tz : index, Block size Z
        ]
    )

    with ir.InsertionPoint(gpu_kernel_block):
        tIdX = gpu_kernel_block.arguments[3]
        cst_0 = arith.ConstantOp(ir.F32Type.get(), ir.FloatAttr.get(ir.F32Type.get(), 0.0))
        for1 = scf.ForOp(
            lower_bound=tIdX,
            upper_bound=size,
            step=gpu_kernel.blockSizeX
        )
        with ir.InsertionPoint(for1.body):
            load = memref.LoadOp(arg0, [for1.induction_variable])
            result = arith.MaxNumFOp(load, cst_0)
            memref.StoreOp(result, arg0, [for1.induction_variable])
            scf.YieldOp([])
        
        gpu.TerminatorOp()
    return op

ops_registry = {
    ReluOp: relu_op
}
