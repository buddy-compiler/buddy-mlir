import operator
from typing import List

import mlir.dialects.func as func
from mlir.dialects import arith
import torch

import mlir.ir as ir
from mlir.passmanager import *

from .operators_gen import operation_func


def DynamoCompiler(gm: torch.fx.GraphModule, inputs: List[torch.Tensor]):
    print("Custom Compiler from FX Graph to MLIR:")
    print("-------------------------------------------------------------------")
    gm.graph.print_tabular()
    # Initialize the MLIR context.
    ctx = ir.Context()
    with ir.Location.unknown(ctx):
        fx_importer = _FXGraphImporter(gm, inputs)
        module = fx_importer.import_graph()
        module = Lowering(module)
    return gm.forward


class _FXGraphImporter:
    def __init__(self, gm: torch.fx.GraphModule, inputs: List[torch.Tensor], func_name: str = "main"):
        self._symbol_table = {}
        self._gm = gm
        self._func_name = func_name
        self._inputs = inputs
        self._num_input_visited = 0
        self._module = ir.Module.create()

    def import_graph(self):
        with ir.InsertionPoint(self._module.body):
            arguments = []
            for arg in self._inputs:
                shape_list = list(arg.shape)
                f32 = ir.F32Type.get()
                tensor_arg = ir.RankedTensorType.get(shape_list, f32)
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
                            self._symbol_table[(str(node.name), 0)] = self._symbol_table[(node.args[0], node.args[1])]
                        else:
                            self._import_op(node)

                return self._symbol_table.get(("output", 0))

        print("Printing the generated MLIR")
        print(self._module)
        return self._module

    def _import_placeholder(self, node: torch.fx.Node, args_list):
        placeholder_name = args_list[self._num_input_visited]
        self._symbol_table[(str(node.name), 0)] = placeholder_name
        self._num_input_visited += 1

    def _import_op(self, node: torch.fx.Node):
        op_code = node.target.__name__

        operation: ir.Operation = operation_func[op_code](node, self._symbol_table)
        for i, result in enumerate(operation.results):
            self._symbol_table[(str(node.name), i)] = result


def Lowering(module: ir.Module):
    print("-------------------------------------------------------------------")
    print("Bufferizing the module ...")
    pm = PassManager("builtin.module")
    pm.add("func.func(tosa-to-linalg)")
    pm.add("func.func(tosa-to-tensor)")
    pm.add("empty-tensor-to-alloc-tensor")
    pm.add("convert-elementwise-to-linalg")
    pm.add("arith-bufferize")
    pm.add("func.func(linalg-bufferize)")
    pm.add("func.func(tensor-bufferize)")
    pm.add("func-bufferize")
    pm.run(module.operation)
    print(module)
    print("-------------------------------------------------------------------")
    print("Lowering the module to LLVM dialect ...")
    pm.add("func.func(buffer-deallocation)")
    pm.add("func.func(convert-linalg-to-loops)")
    pm.add("convert-scf-to-cf")
    pm.add("convert-linalg-to-llvm")
    pm.add("convert-arith-to-llvm")
    pm.add("expand-strided-metadata")
    pm.add("finalize-memref-to-llvm")
    pm.add("convert-func-to-llvm")
    pm.add("reconcile-unrealized-casts")
    pm.run(module.operation)
    print(module)
    return module
