from mlir import ir
from mlir.dialects import tosa, linalg, arith, tensor
import numpy
from mlir.passmanager import PassManager

ctx = ir.Context()
with ir.Location.unknown(ctx):
  _module = ir.Module.create()
  with ir.InsertionPoint(_module.body):
      # tensor_type = ir.RankedTensorType.get([1, 13], ir.IntegerType.get_signed(64))
      # attr = ir.DenseElementsAttr.get(numpy.array([i for i in range(0, 13, 1)]), signless=True, type=tensor_type)
      # op = arith.ConstantOp(tensor_type, attr)
      # print(op)
      # op1 = op.result
      # input_shape = ir.RankedTensorType(op1.type).shape
      # input_shape.insert(0, 1)
      # print(input_shape)
      # dtype = ir.RankedTensorType(op1.type).element_type
      # tensor_type = ir._denseI64ArrayAttr(numpy.array(input_shape), ctx)
      # op = tosa.ReshapeOp(op, tensor_type)
      # print(op)
      # op = tosa.ReshapeOp(op, tensor_type)
      # print(op)
      # tensor_type = ir.RankedTensorType.get([1, 13], ir.F32Type.get())
      # attr1 = ir.DenseElementsAttr.get(numpy.array([i for i in range(0, 13, 1)], dtype=numpy.float32), signless=True, type=tensor_type)
      # attr2 = ir.DenseElementsAttr.get(numpy.array([i for i in range(0, 13, 1)], dtype=numpy.float32), signless=True, type=tensor_type)
      # op1 = arith.ConstantOp(tensor_type, attr1)
      # op2 = arith.ConstantOp(tensor_type, attr2)
      # print(linalg.GenericOp([tensor_type], [op1], [op2],
      #                        ir.ArrayAttr.get([ir.AffineMapAttr.get(ir.AffineMap.get_permutation([0, 1])),
      #                           ir.AffineMapAttr.get(ir.AffineMap.get_permutation([0, 1]))]),
      #                         ir.ArrayAttr.get([ir.StringAttr.get("parallel")]*2)))
      
      # element = ir.BoolAttr.get(1)
      # tensor_type = ir.RankedTensorType.get(output_shape, element.type)
      # attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
      # op = arith.ConstantOp(tensor_type, attr)
      
      # print(op)
      value = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 3)
      # op3 = arith.CmpIOp(value, op1, op2)
      # print(op3)
      output_shape = [1, 13, 13]
      tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
    #   op1 = tensor.EmptyOp([1, 13, 13], ir.F32Type.get())
    #   op2 = tensor.EmptyOp([1, 13, 13], ir.F32Type.get())
    #   op3 = tensor.EmptyOp([1, 13, 13], ir.F32Type.get())
    #   op4 = linalg.BatchMatmulOp([op1.result, op2.result], [op3.result], [tensor_type])
      value = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 3)
      op1 = arith.ConstantOp(ir.IntegerType.get_signless(32), value)
      op2 = arith.SIToFPOp(ir.F32Type.get(), op1.result)
      print(_module)
      exit()
      print(ir.RankedTensorType(op1.result.type).element_type)
      print(op2)
      op = linalg.GenericOp([tensor_type], [op1, op3], [op2],
                             ir.ArrayAttr.get([ir.AffineMapAttr.get(ir.AffineMap.get_permutation([0, 1, 2]).get_submap([0, 2])),
                                ir.AffineMapAttr.get(ir.AffineMap.get_permutation([0, 1])), ir.AffineMapAttr.get(ir.AffineMap.get_permutation([0, 1]))]),
                              ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*2))
      print(op)
      body = op.region.blocks
      block = ir.Block.create_at_start(op.region, [ir.F16Type.get() ,ir.F16Type.get(), ir.IntegerType.get_signless(1)])
      cmpop = arith.CmpFOp(value, block.arguments[0], block.arguments[1])
      block.append(cmpop)
      block.append(linalg.YieldOp([cmpop.result]))
  print(_module)
  pm = PassManager("builtin.module")
  pm.add("func.func(tosa-to-linalg)")
  pm.add("func.func(tosa-to-tensor)")
  pm.add("func.func(tosa-to-arith)")
  pm.add("empty-tensor-to-alloc-tensor")
  pm.add("convert-elementwise-to-linalg")
  pm.add("arith-bufferize")
  pm.add("func.func(linalg-bufferize)")
  pm.add("func.func(tensor-bufferize)")
  pm.add("func-bufferize")
  pm.run(_module.operation)
  print(_module)
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
  pm.run(_module.operation)
  print(_module)