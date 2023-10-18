# ===- frontend.py -------------------------------------------------------------
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
# This is the entry of the Buddy Compiler frontend.
#
# ===---------------------------------------------------------------------------

import operator
from typing import Any, List, Union, Optional
import os
import shutil

import mlir.dialects.func as func
import mlir.ir as ir
import torch
from mlir.passmanager import PassManager
from torch._functorch.aot_autograd import aot_module_simplified
import torch._functorch.aot_autograd as aot_autograd
import torch.utils._pytree as pytree
from functorch.compile import aot_function

from buddy.operators.linalg_operators import (
    operators_registry as linalg_operators_registry,
)

def from_dynamo(model, data):
    mod = None
    params = None
    
    def compile_fx(gm: torch.fx.GraphModule, inputs: List[torch.Tensor]) -> Any:
        
        def _compiler(_gm: torch.fx.GraphModule, inputs: List[torch.Tensor]):
            nonlocal mod
            mod = _gm

            return _gm.forward
        
        nonlocal params
        params = {
            **dict(gm.named_parameters(remove_duplicate=False)),
            **dict(gm.named_buffers(remove_duplicate=False)),
        }
        params_flat, params_spec = pytree.tree_flatten(params)
        params = params_flat
        
        return aot_module_simplified(
            gm,
            inputs,
            fw_compiler=_compiler,
        )
    
    model_opt = torch.compile(model, backend=compile_fx)
    model_opt(data)
    return mod, params

class BuddyDynamoCompiler:
    def __init__(
        self,
        func_name: str = "main",
        aot_autograd_decomposition: Optional[dict] = None,
    ) -> None:
        self.func_name = func_name
        self.aot_autograd_decoposition = aot_autograd_decomposition
        self._bufferize_pipelines = [
            "func.func(tosa-to-linalg-named)",
            "func.func(tosa-to-linalg)",
            "func.func(tosa-to-tensor)",
            "func.func(tosa-to-arith)",
            "empty-tensor-to-alloc-tensor",
            "convert-elementwise-to-linalg",
            "arith-bufferize",
            "func.func(linalg-bufferize)",
            "func.func(tensor-bufferize)",
            "func-bufferize",
        ]
        self._llvm_lower_pipelines = [
            "func.func(buffer-deallocation)",
            "func.func(convert-linalg-to-loops)",
            "convert-math-to-llvm",
            "convert-math-to-libm",
            "convert-scf-to-cf",
            "convert-linalg-to-llvm",
            "convert-arith-to-llvm",
            "expand-strided-metadata",
            "finalize-memref-to-llvm",
            "func.func(llvm-request-c-wrappers)",
            "convert-func-to-llvm",
            "reconcile-unrealized-casts",
        ]

    def __call__(
        self, gm: torch.fx.GraphModule, inputs: List[torch.Tensor]
    ) -> Any:
        def _compiler(_gm: torch.fx.GraphModule, _inputs: List[torch.Tensor]):
            """Compile a FX graph in Aten/Prims IR to MLIR."""
            # Initialize the MLIR context.
            ctx = ir.Context()
            with ir.Location.unknown(ctx):
                fx_importer = FXGraphImporter(_gm, _inputs)
                llvm_lowerer = LLVMLowerer(
                    self._bufferize_pipelines, self._llvm_lower_pipelines
                )
                module = fx_importer.import_graph()
                module = llvm_lowerer.lower(module)

            return _gm.forward
        return aot_module_simplified(
            gm,
            inputs,
            fw_compiler=_compiler,
            decompositions=self.aot_autograd_decoposition,
        )


class FXGraphImporter:
    """The FX graph importer class."""

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        inputs: List[torch.Tensor],
        func_name: str = "forward",
    ):
        """
        Args:
          gm (torch.fx.GraphModule): The FX graph module that will be imported.
          inputs (List[torch.Tensor]): Input tensor(s) of the FX graph.
          func_name (str): Name of the generated MLIR func.

        """
        self._symbol_table = {}
        self._gm = gm
        self._func_name = func_name
        self._inputs = inputs
        self._num_input_visited = 0
        self._module = ir.Module.create()
        self._operators_registry = {
            **linalg_operators_registry
        }

    def import_graph(self) -> ir.Module:
        """Import the FX graph, generate an MLIR module in high-level dialects.

        Returns:
          mlir.ir.Module: An MLIR module in high-level dialects.

        """
        with ir.InsertionPoint(self._module.body):
            arguments = []
            for arg in self._inputs:
                shape_list = list(arg.shape)
                dtype = arg.dtype
                match dtype:
                    case torch.int32:
                        mlir_dtype = ir.IntegerType.get_signless(32)
                    case torch.int64:
                        mlir_dtype = ir.IntegerType.get_signless(64)
                    case torch.float32:
                        mlir_dtype = ir.F32Type.get()
                    case torch.bool:
                        mlir_dtype = ir.IntegerType.get_signless(1)
                    case _:
                        raise NotImplementedError(
                            f"Unsupported dtype {dtype} for argument {arg}"
                        )
                tensor_arg = ir.RankedTensorType.get(shape_list, mlir_dtype)
                arguments.append(tensor_arg)

            @func.FuncOp.from_py_func(*arguments, name=self._func_name)
            def generated_func(*args):
                args_list = list(args)
                for node in self._gm.graph.nodes:
                    if node.op == "output":
                        output_node_args = node.args[0]
                        returns = []
                        for output_arg in output_node_args:
                            op = self._symbol_table.get((str(output_arg), 0))
                            returns.append(op)

                        self._symbol_table[("output", 0)] = returns
                    elif node.op == "placeholder":
                        self._import_placeholder(node, args_list)
                    else:
                        if node.target is operator.getitem:
                            self._symbol_table[
                                (str(node.name), 0)
                            ] = self._symbol_table[
                                (str(node.args[0]), node.args[1])
                            ]
                        else:
                            self._import_op(node)

                return self._symbol_table.get(("output", 0))

        return self._module

    def _import_placeholder(self, node: torch.fx.Node, args_list):
        placeholder_name = args_list[self._num_input_visited]
        self._symbol_table[(str(node.name), 0)] = placeholder_name
        self._num_input_visited += 1

    def _import_op(self, node: torch.fx.Node):
        op_name = node.target.__name__

        op_ret: Union[ir.Operation.result, tuple] = self._operators_registry[op_name](
            node, self._symbol_table
        )
        if isinstance(op_ret, tuple):
            for i, operation in enumerate(op_ret):
                self._symbol_table[(str(node.name), i)] = operation
        else:
            self._symbol_table[(str(node.name), 0)] = op_ret


class LLVMLowerer:
    def __init__(
        self, bufferizing_pipelines: List[str], llvm_lower_pipelines: List[str]
    ) -> None:
        self._bufferizing_pipelines = bufferizing_pipelines
        self._llvm_lower_pipelines = llvm_lower_pipelines

    def lower(self, module: ir.Module) -> Any:
        """Lower an MLIR module to LLVM dialect.

        Args:
        module (mlir.ir.Module): An MLIR module that need to be lowered.

        Returns:
        mlir.ir.Module: An MLIR module in LLVM dialect.

        """
        pm = PassManager("builtin.module")
        # bufferize
        for pipeline in self._bufferizing_pipelines:
            pm.add(pipeline)
        pm.run(module.operation)

        # lower to LLVM dialect
        for pipeline in self._llvm_lower_pipelines:
            pm.add(pipeline)
        pm.run(module.operation)
        print(module)

        return module