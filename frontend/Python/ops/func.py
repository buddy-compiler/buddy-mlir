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
# The registry of mappings from Buddy node to MLIR func dialect operations.
#
# ===---------------------------------------------------------------------------

from mlir.dialects import func, memref
from mlir import ir
from ..graph import FuncOp, CallOp
from .utils import *


def func_op(node: FuncOp, symbol_table):
    arguments = []
    for arg in node.args:
        shape = arg.shape
        mlir_dtype = mlir_element_type_get(arg.dtype)
        arguments.append(ir.MemRefType.get(shape, mlir_dtype))
    results = []
    for i, shape in enumerate(node.tensor_meta["shape"]):
        mlir_dtype = mlir_element_type_get(node.tensor_meta["dtype"][i])
        results.append(shape, mlir_dtype)
    function_type = ir.FunctionType.get(
        inputs=arguments, results=results
    )
    func_op = func.FuncOp(name=node.name, type=function_type)
    return func_op

def call_op(node: CallOp, symbol_table):
    arguments = []
    for arg in node.args:
        shape = arg.shape
        mlir_dtype = mlir_element_type_get(arg.dtype)
        arguments.append(ir.MemRefType.get(shape, mlir_dtype))
    results = []
    for i, shape in enumerate(node.tensor_meta["shape"]):
        mlir_dtype = mlir_element_type_get(node.tensor_meta["dtype"][i])
        results.append(shape, mlir_dtype)
    function_type = ir.FunctionType.get(
        inputs=arguments, results=results
    )
    func_op = func.FuncOp(name=node.name, type=function_type)
    return func_op

ops_registry = {
    "FuncOp": func_op,
    "CallOp": call_op,
}