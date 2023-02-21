from mlir.ir import *
from mlir.dialects import arith
import torch
import torch._dynamo as dynamo
from typing import List

def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
  print("Custom Backend from FX Graph to MLIR:")
  print("-------------------------------------------------------------------")
  gm.graph.print_tabular()
  print("-------------------------------------------------------------------")
  compile(gm)
  return gm.forward

def compile(gm: torch.fx.GraphModule):
  # Initialize the MLIR context and allow the unregistered dialects.
  ctx = Context()
  ctx.allow_unregistered_dialects = True
  # Initialize the symbol table.
  symbolTable = {}
  # Create a module and build the operations into the module.
  with Location.unknown(ctx):
    module = Module.create()
    with InsertionPoint(module.body):
      for node in gm.graph.nodes:
        code_gen(node, symbolTable)
    print("-------------------------------------------------------------------")
    print("Printing the symbol table ...")
    for symbol, op in symbolTable.items():
      print(symbol, ": ", op)
    print("-------------------------------------------------------------------")
    print("Printing the generated MLIR ...")
    print(module)

def code_gen(node, symbolTable):
  if node.op == "placeholder" :
    print("Generating placeholder operation...")
    f32 = F32Type.get()
    op = Operation.create("placeholder", results=[f32])
    symbolTable[str(node.name)] = op
  if node.op == "call_function" :
    print("Parsing a call_function operation...")
    if node.target.__name__ == "add":
      print("Generating add operation...")
      f32 = F32Type.get()
      input1 = symbolTable.get(str(node._args[0]))
      input2 = symbolTable.get(str(node._args[1]))
      op = arith.AddFOp(input1, input2)
      symbolTable[str(node.name)] = op
  if node.op == "output" :
    print("Generating return operation...")

def foo(x, y):
  return x + y

foo_mlir = dynamo.optimize(custom_backend)(foo)
in1 = torch.randn(10)
in2 = torch.randn(10)
foo_mlir(in1, in2)
