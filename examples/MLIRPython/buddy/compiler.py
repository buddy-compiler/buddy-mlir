from mlir.ir import *
from mlir.dialects import arith, linalg, tosa
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
    if node.target.__name__ == "sub":
      # Generate sub operation.
      input1 = symbolTable.get(str(node._args[0]))
      input2 = symbolTable.get(str(node._args[1]))
      op = arith.SubFOp(input1, input2)
      symbolTable[str(node.name)] = op
    if node.target.__name__ == "matmul":
      # Only support 2D matmul now.
      # Get two input values.
      input1 = symbolTable.get(str(node._args[0]))
      input2 = symbolTable.get(str(node._args[1]))
      # Infer the output sizes.
      size1 = RankedTensorType(input1.type).shape[0]
      size2 = RankedTensorType(input2.type).shape[1]
      sizes = [size1, size2]
      # Generate an output tensor for matmul operation.
      # For example:
      # `arith.constant dense<0.000000e+00> : tensor<3x3xf32>`
      f32 = F32Type.get()
      element = FloatAttr.get(f32, 0.0)
      tensor_type = RankedTensorType.get(sizes, f32)
      attr = DenseElementsAttr.get_splat(tensor_type, element)
      init_result = arith.ConstantOp(tensor_type, attr)
      # Generate matmul operation.
      op = linalg.matmul(input1, input2, outs=[init_result.result])
      symbolTable[str(node.name)] = op
    if node.target.__name__ == "softmax":
      # Generate softmax operation.
      f32 = F32Type.get()
      
      input = symbolTable.get(str(node._args[0]))
      axis = node._args[1]
      
      input_size = RankedTensorType(input.type).shape
      sum_size = input_size[:axis] + [1] + input_size[axis+1:]
      sum_result_tensor_type = RankedTensorType.get(sum_size, f32)
      
      exp_op = tosa.ExpOp(input.type, input)
      sum_exp_op = tosa.ReduceSumOp(exp_op.result, axis)
      
      # `float` is not supported in tosa.DivOp
      # div_op = tosa.DivOp(softmax_result_tensor_type, input, sum_op.result)
      
      # `broadcast` is not supported in arith.DivFOp
      # div_op = arith.DivFOp(exp_op.result, sum_op.result)
      
      # e^(lna - lnb)= a/b 
      log_sum_exp_op = tosa.LogOp(sum_result_tensor_type, sum_exp_op.result)
      sub_op = tosa.SubOp(input.type, input, log_sum_exp_op.result)
      div_op = tosa.ExpOp(input.type, sub_op.result)
      
      symbolTable[str(node.name)] = div_op
      
  if node.op == "output" :
    # Generating return operation.
    ret = symbolTable.get(str(node._args[0][0]))
    symbolTable["output"] = ret

def Lowering(module: Module):
  print("-------------------------------------------------------------------")
  print("Bufferizing the module ...")
  pm = PassManager('builtin.module')
  pm.add("func.func(tosa-to-linalg)")
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
  
  # pm.add("reconcile-unrealized-casts")
  # %141 and %142 should be removed but I `reconcile-unrealized-casts` is not working, 
  # meanwhile, it set the op to illegal.
  # %141 = builtin.unrealized_conversion_cast %140 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<1x3xf32>
  # %142 = builtin.unrealized_conversion_cast %141 : memref<1x3xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  
  pm.run(module.operation)
  print(module)
  return module
