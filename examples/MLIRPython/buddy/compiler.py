from mlir.ir import *
from mlir.dialects import arith, linalg, tosa
import mlir.dialects.func as func
from mlir.passmanager import *
import torch
from typing import List
import array

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
    if node.target.__name__ == "matmul":
      # Only support 2D matmul now.
      # Get two input values.
      input1 = symbolTable.get(str(node._args[0]))
      input2 = symbolTable.get(str(node._args[1]))
      shp1 = RankedTensorType(input1.type).shape
      shp2 = RankedTensorType(input2.type).shape
      assert len(shp1) == len(shp2)
      f32 = F32Type.get()
      zero_element = FloatAttr.get(f32, 0.0)
      if len(shp1) == 2:
        # Infer the output sizes.
        size1 = shp1[0]
        size2 = shp2[1]
        sizes = [size1, size2]
        # Generate an output tensor for matmul operation.
        # For example:
        # `arith.constant dense<0.000000e+00> : tensor<3x3xf32>`
        tensor_type = RankedTensorType.get(sizes, f32)
        attr = DenseElementsAttr.get_splat(tensor_type, zero_element)
        init_result = arith.ConstantOp(tensor_type, attr)
        # Generate matmul operation.
        op = linalg.matmul(input1, input2, outs=[init_result.result])
        symbolTable[str(node.name)] = op
      elif len(shp1) == 3:
        size0 = shp1[0]
        size1 = shp1[1]
        size2 = shp2[2]
        sizes = [size0, size1, size2]
        tensor_type = RankedTensorType.get(sizes, f32)
        attr = DenseElementsAttr.get_splat(tensor_type, zero_element)
        init_result = arith.ConstantOp(tensor_type, attr)
        op = linalg.batch_matmul(input1, input2, outs=[init_result.result])
        symbolTable[str(node.name)] = op
      else:
        raise NotImplementedError
    if node.target.__name__ == "transpose":
      input_tensor = symbolTable.get(str(node._args[0]))
      size1 = RankedTensorType(input_tensor.type).shape[0]
      size2 = RankedTensorType(input_tensor.type).shape[1]
      sizes = [size2, size1]

      f32 = F32Type.get()
      trans_result_tensor_type = RankedTensorType.get(sizes, f32)
      perm_tensor_type = RankedTensorType.get([2], f32)
      zero = FloatAttr.get(f32, 0.0)
      # one = FloatAttr.get(f32, 1.0)
      trans_result_attr = DenseElementsAttr.get_splat(trans_result_tensor_type, zero)
      trans_result = arith.ConstantOp(trans_result_tensor_type, trans_result_attr)
      perm_content = memoryview(array.array('i', [1, 0]))
      perm_attr = DenseElementsAttr.get(perm_content)
      perm = arith.ConstantOp(perm_tensor_type, perm_attr)
      op = tosa.TransposeOp(trans_result_tensor_type, input_tensor, perm)
      symbolTable[str(node.name)] = op

  if node.op == "output" :
    # Generating return operation.
    ret = symbolTable.get(str(node._args[0][0]))
    symbolTable["output"] = ret

def Lowering(module: Module):
  print("-------------------------------------------------------------------")
  print("Bufferizing the module ...")
  pm = PassManager('builtin.module')
  pm.add("func.func(tosa-to-arith)")
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
  # pm.add("func.func(tosa-to-linalg)")
  pm.add("memref-expand")
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
