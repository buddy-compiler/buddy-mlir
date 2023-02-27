from mlir.ir import *
from mlir.dialects import arith
import mlir.dialects.func as func
from mlir.passmanager import *
import torch
from typing import List

def DynamoCompiler(gm: torch.fx.GraphModule, inputs: List[torch.Tensor]):
  print("Custom Compiler from FX Graph to MLIR:")
  print("-------------------------------------------------------------------")
  gm.graph.print_tabular()
  # Initialize the MLIR context.
  ctx = Context()
  with Location.unknown(ctx):
    module = Importer(gm, inputs)
    module = Lowering(module)
  return gm.forward

def Importer(gm: torch.fx.GraphModule, inputs: List[torch.Tensor]):
  # Initialize the symbol table.
  symbolTable = {}
  # Create a module and build the operations into the module.
  module = Module.create()
  with InsertionPoint(module.body):
    # Parse the arguments.
    arguments = []
    for arg in inputs:
      shapeList = list(arg.shape)
      f32 = F32Type.get()
      tensorArg = RankedTensorType.get(shapeList, f32)
      arguments.append(tensorArg)
    # Generate the function.
    @func.FuncOp.from_py_func(*arguments)
    def generated_func(*args):
      # Convert arguments tuple into a list.
      argsList = list(args)
      # Traverse the graph and generate IR.
      for node in gm.graph.nodes:
        CodeGen(node, symbolTable, argsList)
      return symbolTable.get("output")
  print("-------------------------------------------------------------------")
  print("Printing the symbol table ...")
  for symbol, op in symbolTable.items():
    print(symbol, ": ", op)
  print("-------------------------------------------------------------------")
  print("Printing the generated MLIR ...")
  print(module)
  return(module)

def CodeGen(node, symbolTable, argsList):
  if node.op == "placeholder" :
    # Bind the placeholder with args.
    symbolTable[str(node.name)] = argsList[0]
    argsList.pop(0)
  if node.op == "call_function" :
    # Parse a call_function operation.
    if node.target.__name__ == "add":
      # Generate add operation.
      input1 = symbolTable.get(str(node._args[0]))
      input2 = symbolTable.get(str(node._args[1]))
      op = arith.AddFOp(input1, input2)
      symbolTable[str(node.name)] = op
  if node.op == "output" :
    # Generating return operation.
    ret = symbolTable.get(str(node._args[0][0]))
    symbolTable["output"] = ret

def Lowering(module):
  print("-------------------------------------------------------------------")
  print("Bufferizing the module ...")
  pm = PassManager('builtin.module')
  pm.add("convert-elementwise-to-linalg")
  pm.add("arith-bufferize")
  pm.add("func.func(linalg-bufferize)")
  pm.add("func.func(tensor-bufferize)")
  pm.add("func-bufferize")
  pm.run(module)
  print(module)
  print("-------------------------------------------------------------------")
  print("Lowering the module to LLVM dialect ...")
  pm.add("func.func(buffer-deallocation)")
  pm.add("func.func(convert-linalg-to-loops)")
  pm.add("convert-scf-to-cf")
  pm.add("convert-linalg-to-llvm")
  pm.add("convert-memref-to-llvm")
  pm.add("convert-func-to-llvm")
  pm.add("reconcile-unrealized-casts")
  pm.run(module)
  print(module)
  return module
