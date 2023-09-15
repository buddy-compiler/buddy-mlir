"""Generate the MLIR operations for the operators in the FX graph.
"""
from typing import Dict, Tuple, List

import torch

import mlir.ir as ir
from mlir.dialects import tosa, linalg, arith, tensor, math
import copy
import numpy
from .global_var import *
import functools


def _broadcast_shape(tensor_input1: ir.Value,
                     tensor_input2: ir.Value) -> List[int]:
  """Calculate the broadcast shape of two tensors with broadcastable shapes 
  according to PyTorch's broadcast semantics: https://pytorch.org/docs/stable/notes/broadcasting.html"""
  shp1 = ir.RankedTensorType(tensor_input1.type).shape
  shp2 = ir.RankedTensorType(tensor_input2.type).shape
  if len(shp1) < len(shp2):
    shp1, shp2 = shp2, shp1
  while len(shp2) < len(shp1):
    shp2.insert(0, 1)
  for idx, (dim1, dim2) in enumerate(zip(shp1, shp2)):
    shp1[idx] = shp2[idx] = max(dim1, dim2)

  return shp1


def AddOp(node: torch.fx.Node,
          symbol_table: Dict[Tuple[str, int], ir.Operation],
          ctx: ir.Context) -> ir.Operation:
  """Map aten.add.Tensor to tosa.add.

  Args:
    node: A FX graph containing the aten.add.Tensor operator and its parameter.
    symbol_table: The symbol table that records the mapping between symbols and operations.

  Returns:
    ir.Operation: The generated tosa.add operation.

  """
  input1 = symbol_table.get((str(node.args[0]), 0))
  dtype = str(node.meta['tensor_meta'].dtype)
  if isinstance(node.args[1], torch.fx.Node):
    input2 = symbol_table.get((str(node.args[1]), 0))
    if dtype == "torch.int64":
      dtype = type_dict[dtype].get_signless(64)
    elif dtype == "torch.float32":
      dtype = type_dict[dtype].get()
  else:
    if dtype == "torch.int64":
      data = numpy.array(node.args[1], dtype=numpy.int64)
      dtype = type_dict[dtype].get_signless(64)
      tensor_type = ir.RankedTensorType.get(list(data.shape), dtype)
      attr = ir.DenseElementsAttr.get(data, signless=True, type=tensor_type)
    elif dtype == "torch.float32":
      data = numpy.array(node.args[1], dtype=numpy.float32)
      dtype = type_dict[dtype].get()
      tensor_type = ir.RankedTensorType.get(list(data.shape), dtype)
      attr = ir.DenseElementsAttr.get(data, signless=True, type=tensor_type)
    input2 = arith.ConstantOp(tensor_type, attr).result
  if input1 is None or input2 is None:
    return
  broadcasted_shp = _broadcast_shape(input1, input2)
  sizes = broadcasted_shp
  addResultTensorType = ir.RankedTensorType.get(sizes, dtype)
  op = tosa.AddOp(addResultTensorType, input1, input2)
  return op


def AddMMOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation]) -> ir.Operation:
  """Map aten.addmm.default to MLIR operation.

  Args:
    node (torch.fx.Node): A FX graph containing the aten.addmm.default operator and its parameter.
    symbol_table (Dict[Tuple[str, int], ir.Operation]): The symbol table that records the mapping between symbols and operations.

  Returns:
    ir.Operation: The generated MLIR operation representing aten.addmm.default

  """
  input_ = symbol_table.get((str(node.args[0]), 0))
  mat1 = symbol_table.get((str(node.args[1]), 0))
  mat2 = symbol_table.get((str(node.args[2]), 0))
  mat1_shp = ir.RankedTensorType(mat1.type).shape
  mat2_shp = ir.RankedTensorType(mat2.type).shape
  result_shp = [mat1_shp[0], mat2_shp[1]]
  f32 = ir.F32Type.get()
  element = ir.FloatAttr.get(f32, 0.0)
  tensor_type = ir.RankedTensorType.get(result_shp, f32)
  attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
  matmul_result_buffer = arith.ConstantOp(tensor_type, attr).result
  # Generate matmul operation.
  matmul_op_result = linalg.matmul(mat1, mat2, outs=[matmul_result_buffer])

  add_result_tensor_type = ir.RankedTensorType.get(result_shp, f32)
  op = tosa.AddOp(add_result_tensor_type, input_, matmul_op_result)
  return op

def ArangeOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  if node.target.__name__ == "arange.start":
    start = int(node.args[0])
    end = int(node.args[1])
    stride = int(node.meta['tensor_meta'].stride[0])
    dtype = str(node.meta['tensor_meta'].dtype)
    shape = list(node.meta['tensor_meta'].shape)
    dtype = type_dict[dtype].get_signless(64)
    tensor_type = ir.RankedTensorType.get(shape, dtype)
    attr = ir.DenseElementsAttr.get(numpy.array([i for i in range(start, end, stride)]), signless=True, type=tensor_type)
    op = arith.ConstantOp(tensor_type, attr)
  
  elif node.target.__name__ == "arange.default":
    start = 0
    end = int(node.args[0])
    stride = int(node.meta['tensor_meta'].stride[0])
    dtype = str(node.meta['tensor_meta'].dtype)
    shape = list(node.meta['tensor_meta'].shape)
    dtype = type_dict[dtype].get_signless(64)
    tensor_type = ir.RankedTensorType.get(shape, dtype)
    attr = ir.DenseElementsAttr.get(numpy.array([i for i in range(start, end, stride)]), signless=True, type=tensor_type)
    op = arith.ConstantOp(tensor_type, attr)

  return op

def UnsqueezeOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:

  input_node = symbol_table.get((str(node.args[0]), 0))
  if input_node is None:
    return
  axis = int(node.args[1])
  input_shape = ir.RankedTensorType(input_node.type).shape
  input_shape.insert(axis, 1)
  tensor_type = ir._denseI64ArrayAttr(numpy.array(input_shape, dtype=numpy.int64), ctx)
  op = tosa.ReshapeOp(input_node, tensor_type)

  return op

def ViewOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  input_node = symbol_table.get((str(node.args[0]), 0))
  if input_node is None:
    return
  output_shape = list(node.args[1])
  input_shape = list(ir.RankedTensorType(input_node.type).shape)
  
  nums = 1
  for i in input_shape:
    nums *= i
  for i in output_shape:
    if i != -1:
      nums //= i
  for i, s in enumerate(output_shape):
    if s == -1:
      output_shape[i] = nums
  
  tensor_type = ir._denseI64ArrayAttr(numpy.array(output_shape, dtype=numpy.int64), ctx)
  op = tosa.ReshapeOp(input_node, tensor_type)

  return op

def EmbeddingOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  input1 = symbol_table.get((str(node.args[0]), 0))
  input2 = symbol_table.get((str(node.args[1]), 0))
  output_shape = list(node.meta['tensor_meta'].shape)
  dtype = str(node.meta['tensor_meta'].dtype)
  if dtype == "torch.float32":
    tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
    output = tensor.EmptyOp(output_shape, ir.F32Type.get())
    generic_map = ir.AffineMap.get_permutation([0, 1, 2])
    op = linalg.GenericOp([tensor_type], [input2], [output],
                         ir.ArrayAttr.get([ir.AffineMapAttr.get(generic_map.get_submap([0, 1])), ir.AffineMapAttr.get(generic_map.get_submap([0, 1, 2]))]),
                          ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*3))
    block = ir.Block.create_at_start(op.region, [ir.RankedTensorType(input2.type).element_type, ir.RankedTensorType(output.result.type).element_type])
    index1 = arith.IndexCastOp(ir.IndexType.get(), block.arguments[0])
    index2 = linalg.IndexOp(ir._i64Attr(2, ctx))
    value = tensor.ExtractOp(input1, [index1.result, index2.result])
    block.append(index1)
    block.append(index2)
    block.append(value)
    block.append(linalg.YieldOp([value.result]))

  return op

def OnesOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  output_shape = list(node.args[0])
  dtype = str(node.meta['tensor_meta'].dtype)
  if dtype == "torch.bool":
    element = ir.BoolAttr.get(1)
    tensor_type = ir.RankedTensorType.get(output_shape, element.type)
    attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
  elif dtype == "torch.int64":
    dtype = type_dict[dtype].get_signless(64)
    tensor_type = ir.RankedTensorType.get(output_shape, dtype)
    attr = ir.DenseElementsAttr.get(numpy.ones(output_shape), signless=True, type=tensor_type)
  op = arith.ConstantOp(tensor_type, attr)

  return op

def FullOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  output_shape = list(node.args[0])
  value = node.args[1]
  dtype = str(node.meta['tensor_meta'].dtype)
  if dtype == "torch.bool":
    element = ir.BoolAttr.get(bool(value))
    tensor_type = ir.RankedTensorType.get(output_shape, element.type)
    attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
  elif dtype == "torch.int64":
    dtype = type_dict[dtype].get_signless(64)
    tensor_type = ir.RankedTensorType.get(output_shape, dtype)
    attr = ir.DenseElementsAttr.get(numpy.full(output_shape, value, dtype=numpy.int64), signless=True, type=tensor_type)
  elif dtype == "torch.float32":
    dtype = type_dict[dtype].get()
    tensor_type = ir.RankedTensorType.get(output_shape, dtype)
    attr = ir.DenseElementsAttr.get(numpy.full(output_shape, value, dtype=numpy.float32), signless=True, type=tensor_type)
  op = arith.ConstantOp(tensor_type, attr)

  return op

def LtOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  input1 = symbol_table.get((str(node.args[0]), 0))
  input2 = symbol_table.get((str(node.args[1]), 0))
  output_shape = list(node.meta['tensor_meta'].shape)
  dtype = str(node.meta['tensor_meta'].dtype)
  value = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 2)
  shp1 = list(ir.RankedTensorType(ir.Value(input1).type).shape)
  shp2 = list(ir.RankedTensorType(ir.Value(input2).type).shape)
  if dtype == "torch.bool":
    tensor_type = ir.RankedTensorType.get(output_shape, ir.IntegerType.get_signless(1))
    output = tensor.EmptyOp(output_shape, ir.IntegerType.get_signless(1))
    if len(shp1) < len(shp2):
      if int(shp1[-1]) > 1 and shp2[-1] == 1:
        generic_map = ir.AffineMap.get_permutation([i for i in range(len(shp2)+1)])
        op = linalg.GenericOp([tensor_type], [input1, input2], [output],
                             ir.ArrayAttr.get([ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(shp2)-len(shp1), len(shp2))])), ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(0, len(shp2)-1)]+[len(shp2)])), ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(0, len(shp2))]))]),
                              ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(shp2)+[ir.Attribute.parse('#linalg.iterator_type<reduction>')]))
        block = ir.Block.create_at_start(op.region, [ir.RankedTensorType(input2.type).element_type, ir.RankedTensorType(input2.type).element_type, ir.IntegerType.get_signless(1)])
        if str(ir.RankedTensorType(input2.type).element_type).find('i') != -1:
          cmpop = arith.CmpIOp(value, block.arguments[0], block.arguments[1])
        else:
          cmpop = arith.CmpFOp(value, block.arguments[0], block.arguments[1])
        block.append(cmpop)
        block.append(linalg.YieldOp([cmpop.result]))
  
  return op

def MaskedFillOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  input1 = symbol_table.get((str(node.args[0]), 0))
  input2 = symbol_table.get((str(node.args[1]), 0))
  if input1 is None or input2 is None:
    return
  if str(node.args[0].meta['tensor_meta'].dtype) == "torch.float32":
    value = float(node.args[2])
    attr = ir.FloatAttr.get(ir.F32Type.get(), value)
    value = arith.ConstantOp(ir.F32Type.get(), attr)
  output_shape = list(node.meta['tensor_meta'].shape)
  dtype = str(node.meta['tensor_meta'].dtype)
  if dtype == "torch.float32":
    tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
    output = tensor.EmptyOp(output_shape, ir.F32Type.get())
    generic_map = ir.AffineMap.get_permutation([i for i in range(len(output_shape))])
    op = linalg.GenericOp([tensor_type], [input1, input2], [output],
                         ir.ArrayAttr.get([ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))])), ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))])), ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))]))]),
                          ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(output_shape)))
    block = ir.Block.create_at_start(op.region, [ir.RankedTensorType(input1.type).element_type, ir.RankedTensorType(input2.type).element_type, ir.RankedTensorType(output.result.type).element_type])
    select_op = arith.SelectOp(block.arguments[1], value, block.arguments[0])
    block.append(select_op)
    block.append(linalg.YieldOp([select_op.result]))
  
  return op

def SliceOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  input1 = symbol_table.get((str(node.args[0]), 0))
  if input1 is None:
    return
  dim = int(node.args[1])
  start = int(node.args[2])
  end = int(node.args[3])
  input_shape = ir.RankedTensorType(input1.type).shape
  if end > input_shape[dim]:
    end = input_shape[dim]
  if len(node.args) < 5:
    step = 1
  else:
    step = node.args[4]
  offset = [0 for x in input_shape]
  offset[dim] = start
  offset_attr = ir._denseI64ArrayAttr(offset, ctx)
  output_shape = list(node.meta['tensor_meta'].shape)
  size_attr = ir._denseI64ArrayAttr(output_shape, ctx)
  stride = [1 for x in output_shape]
  stride[dim] = step
  stride_attr = ir._denseI64ArrayAttr(stride, ctx)
  dtype = str(node.meta['tensor_meta'].dtype)
  if dtype == "torch.float32":
    tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
  if dtype == "torch.bool":
    tensor_type = ir.RankedTensorType.get(output_shape, ir.IntegerType.get_signless(1))
  
  op = tensor.ExtractSliceOp(tensor_type, input1, [], [], [], offset_attr, size_attr, stride_attr)

  return op

def ExpandOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  input1 = symbol_table.get((str(node.args[0]), 0))
  assert isinstance(node.args[1], list)

  if input1 is None:
    return
  input_shape = ir.RankedTensorType(input1.type).shape
  output_shape = list(node.meta['tensor_meta'].shape)
  dtype = str(node.meta['tensor_meta'].dtype)
  if dtype == "torch.bool":
    empty_tensor = tensor.EmptyOp(output_shape, ir.IntegerType.get_signless(1))
  elif dtype == "torch.float32":
    empty_tensor = tensor.EmptyOp(output_shape, ir.F32Type.get())
  if list(input_shape) == list(node.args[1]):
    offset_attr = ir._denseI64ArrayAttr([0 for x in input_shape], ctx)
    size_attr = ir._denseI64ArrayAttr(output_shape, ctx)
    stride_attr = ir._denseI64ArrayAttr([1 for x in input_shape], ctx)
    if dtype == "torch.bool":
      tensor_type = ir.RankedTensorType.get(output_shape, ir.IntegerType.get_signless(1))
    elif dtype == "torch.float32":
      tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
    extract_tensor = tensor.ExtractSliceOp(tensor_type, input1, [], [], [], offset_attr, size_attr, stride_attr)
    op = tensor.InsertSliceOp(extract_tensor.result, empty_tensor.result, [], [], [], offset_attr, size_attr, stride_attr)
  else:
    for i in range(len(input_shape)-1, -1, -1):
      if input_shape[i] != output_shape[i]:
        for j in range(output_shape[i]):
          offset = [0 for x in input_shape]
          offset_attr = ir._denseI64ArrayAttr(offset, ctx)
          size_attr = ir._denseI64ArrayAttr([1]*(i+1)+[x for x in output_shape[i+1:]], ctx)
          stride_attr = ir._denseI64ArrayAttr([1]*len(offset), ctx)
          if dtype == "torch.bool":
            tensor_type = ir.RankedTensorType.get([1]*(i+1)+[x for x in output_shape[i+1:]], ir.IntegerType.get_signless(1))
          elif dtype == "torch.float32":
            tensor_type = ir.RankedTensorType.get([1]*(i+1)+[x for x in output_shape[i+1:]], ir.F32Type.get())
          extract_tensor = tensor.ExtractSliceOp(tensor_type, input1, [], [], [], offset_attr, size_attr, stride_attr)
          offset[i] = j
          offset_attr = ir._denseI64ArrayAttr(offset, ctx)
          op = tensor.InsertSliceOp(extract_tensor.result, empty_tensor.result, [], [], [], offset_attr, size_attr, stride_attr)
          empty_tensor = op
  return op

def ToCopyOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  input1 = symbol_table.get((str(node.args[0]), 0))
  if input1 is None:
    return
  output_shape = list(node.meta['tensor_meta'].shape)
  dtype = str(node.meta['tensor_meta'].dtype)

  if dtype == "torch.bool":
    if str(ir.RankedTensorType(input1.type).element_type) == "f32":
      tensor_type = ir.RankedTensorType.get(output_shape, ir.IntegerType.get_signless(1))
      output = tensor.EmptyOp(output_shape, ir.IntegerType.get_signless(1))
      generic_map = ir.AffineMap.get_permutation([i for i in range(len(output_shape))])
      op = linalg.GenericOp([tensor_type], [input1], [output],
                            ir.ArrayAttr.get([ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))])), ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))]))]),
                            ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(output_shape)))
      block = ir.Block.create_at_start(op.region, [ir.RankedTensorType(input1.type).element_type, ir.RankedTensorType(output.result.type).element_type])
      fptosi_op = arith.FPToSIOp(ir.IntegerType.get_signless(32), block.arguments[0])
      trunc_op = arith.TruncIOp(ir.IntegerType.get_signless(1), fptosi_op.result)
      block.append(fptosi_op)
      block.append(trunc_op)
      block.append(linalg.YieldOp([trunc_op.result]))
  elif dtype == "torch.float32":
    if str(ir.RankedTensorType(input1.type).element_type) == "i1":
      tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
      output = tensor.EmptyOp(output_shape, ir.F32Type.get())
      generic_map = ir.AffineMap.get_permutation([i for i in range(len(output_shape))])
      op = linalg.GenericOp([tensor_type], [input1], [output],
                            ir.ArrayAttr.get([ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))])), ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))]))]),
                            ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(output_shape)))
      block = ir.Block.create_at_start(op.region, [ir.RankedTensorType(input1.type).element_type, ir.RankedTensorType(output.result.type).element_type])
      exti_op = arith.ExtUIOp(ir.IntegerType.get_signless(32), block.arguments[0])
      sitofp_op = arith.SIToFPOp(ir.F32Type.get(), exti_op.result)
      block.append(exti_op)
      block.append(sitofp_op)
      block.append(linalg.YieldOp([sitofp_op.result]))
  
  return op

def RSubOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  input1 = symbol_table.get((str(node.args[0]), 0))
  value = node.args[1]
  output_shape = list(node.meta['tensor_meta'].shape)
  dtype = str(node.meta['tensor_meta'].dtype)
  if not isinstance(value, torch.fx.Node):
    if dtype == "torch.float32":
      value = arith.ConstantOp(ir.F32Type.get(), ir.FloatAttr.get(ir.F32Type.get(), value))
      generic_map = ir.AffineMap.get_permutation([i for i in range(len(output_shape))])
      tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
      output = tensor.EmptyOp(output_shape, ir.F32Type.get())
      op = linalg.GenericOp([tensor_type], [input1], [output],
                           ir.ArrayAttr.get([ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))])), ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))]))]),
                            ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(output_shape)))
      block = ir.Block.create_at_start(op.region, [ir.RankedTensorType(input1.type).element_type, ir.RankedTensorType(output.result.type).element_type])
      subf_op = arith.SubFOp(value.result, block.arguments[0])
      block.append(subf_op)
      block.append(linalg.YieldOp([subf_op.result]))
  
  return op

def PowOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  input1 = symbol_table.get((str(node.args[0]), 0))
  if input1 is None:
    return
  value = node.args[1]
  output_shape = list(node.meta['tensor_meta'].shape)
  dtype = str(node.meta['tensor_meta'].dtype)
  if not isinstance(value, torch.fx.Node):
    if dtype == "torch.float32":
      generic_map = ir.AffineMap.get_permutation([i for i in range(len(output_shape))])
      tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
      output = tensor.EmptyOp(output_shape, ir.F32Type.get())
      if abs(int(value)-float(value)) < 1e-6:
        value = arith.ConstantOp(ir.IntegerType.get_signless(32), ir.IntegerAttr.get(ir.IntegerType.get_signless(32), value))
        op = linalg.GenericOp([tensor_type], [input1], [output],
                             ir.ArrayAttr.get([ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))])), ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))]))]),
                              ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(output_shape)))
        block = ir.Block.create_at_start(op.region, [ir.RankedTensorType(input1.type).element_type, ir.RankedTensorType(output.result.type).element_type])
        fpowi_op = math.FPowIOp(block.arguments[0], value.result)
        block.append(fpowi_op)
        block.append(linalg.YieldOp([fpowi_op.result]))
  
  return op

def MeanOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  input1 = symbol_table.get((str(node.args[0]), 0))
  if input1 is None:
    return
  dims = list(node.args[1])
  keep_dim = bool(node.args[2])
  output_shape = list(node.meta['tensor_meta'].shape)
  dtype = str(node.meta['tensor_meta'].dtype)
  if dtype == "torch.float32":
    tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
    element = ir.FloatAttr.get(ir.F32Type.get(), 0.0)
    attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
    output = arith.ConstantOp(tensor_type, attr)

    assert len(dims) == 1

    for dim in dims:
      if dim == -1:
        dim = len(list(ir.RankedTensorType(input1.type).shape))-1
      if keep_dim:
        generic_map = ir.AffineMap.get_permutation([i for i in range(len(output_shape)+1)])
        tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
        output_map = [i for i in range(len(output_shape))]
        output_map[dim] = len(output_shape)
        loop_type = [ir.Attribute.parse('#linalg.iterator_type<parallel>')]*(len(output_shape)+1)
        loop_type[dim] = ir.Attribute.parse('#linalg.iterator_type<reduction>')
        op = linalg.GenericOp([tensor_type], [input1], [output],
                              ir.ArrayAttr.get([ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))])), ir.AffineMapAttr.get(generic_map.get_submap(output_map))]),
                              ir.ArrayAttr.get(loop_type))
        block = ir.Block.create_at_start(op.region, [ir.RankedTensorType(input1.type).element_type, ir.RankedTensorType(output.result.type).element_type])
        value = arith.ConstantOp(ir.F32Type.get(), ir.FloatAttr.get(ir.F32Type.get(), list(ir.RankedTensorType(input1.type).shape)[dim]))
        divf_op = arith.DivFOp(block.arguments[0], value.result)
        addf_op = arith.AddFOp(divf_op.result, block.arguments[1])
        block.append(value)
        block.append(divf_op)
        block.append(addf_op)
        block.append(linalg.YieldOp([addf_op.result]))
  
  return op

def RSqrtOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  assert len(node.args) == 1
  input1 = symbol_table.get((str(node.args[0]), 0))
  if input1 is None:
    return

  output_shape = list(node.meta['tensor_meta'].shape)
  dtype = str(node.meta['tensor_meta'].dtype)

  if dtype == "torch.float32":
    tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
    output = tensor.EmptyOp(output_shape, ir.F32Type.get())
    generic_map = ir.AffineMap.get_permutation([i for i in range(len(output_shape))])
    op = linalg.GenericOp([tensor_type], [input1], [output],
                         ir.ArrayAttr.get([ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))])), ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))]))]),
                          ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(output_shape)))
    block = ir.Block.create_at_start(op.region, [ir.RankedTensorType(input1.type).element_type, ir.RankedTensorType(output.result.type).element_type])
    rsqrt_op = math.RsqrtOp(block.arguments[0])
    block.append(rsqrt_op)
    block.append(linalg.YieldOp([rsqrt_op.result]))
  
  return op

def MulOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  assert len(node.args) == 2
  if isinstance(node.args[0], torch.fx.Node):
    input1 = symbol_table.get((str(node.args[0]), 0))
  else:
    input1 = node.args[0]
  
  if isinstance(node.args[1], torch.fx.Node):
    input2 = symbol_table.get((str(node.args[1]), 0))
  else:
    input2 = node.args[1]
  
  if input1 is None or input2 is None:
    return

  output_shape = list(node.meta['tensor_meta'].shape)
  dtype = str(node.meta['tensor_meta'].dtype)

  if isinstance(node.args[0], torch.fx.Node):
    if dtype == "torch.float32":
      if not isinstance(node.args[1], torch.fx.Node):
        input2 = arith.ConstantOp(ir.F32Type.get(), ir.FloatAttr.get(ir.F32Type.get(), input2))
        tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
        output = tensor.EmptyOp(output_shape, ir.F32Type.get())
        generic_map = ir.AffineMap.get_permutation([i for i in range(len(output_shape))])
        op = linalg.GenericOp([tensor_type], [input1], [output],
                             ir.ArrayAttr.get([ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))])), ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))]))]),
                              ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(output_shape)))
        block = ir.Block.create_at_start(op.region, [ir.RankedTensorType(input1.type).element_type, ir.RankedTensorType(output.result.type).element_type])
        mul_op = arith.MulFOp(block.arguments[0], input2.result)
        block.append(mul_op)
        block.append(linalg.YieldOp([mul_op.result]))
      else:
        tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
        output = tensor.EmptyOp(output_shape, ir.F32Type.get())
        input1_shape = list(ir.RankedTensorType(input1.type).shape)
        if input1_shape != output_shape:
          dims = []
          for i in range(len(input1_shape)-1, -1, -1):
            if input1_shape[i] != output_shape[len(output_shape)-(len(input1_shape)-i)]:
              dims.append(i)
          output1 = tensor.EmptyOp(output_shape, ir.F32Type.get())
          generic_map = ir.AffineMap.get_permutation([i for i in range(len(output_shape)+len(dims))])
          input1_map = [i for i in range(len(output_shape)-len(input1_shape), len(output_shape))]
          for index, i in enumerate(dims):
            input1_map[i] = len(output_shape)+index
          input1_map = generic_map.get_submap(input1_map)
          input1_op = linalg.GenericOp([tensor_type], [input1], [output1],
                               ir.ArrayAttr.get([ir.AffineMapAttr.get(input1_map), ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))]))]),
                                ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(output_shape)+[ir.Attribute.parse('#linalg.iterator_type<reduction>')]*len(dims)))
          block = ir.Block.create_at_start(input1_op.region, [ir.RankedTensorType(input1.type).element_type, ir.RankedTensorType(output.result.type).element_type])
          block.append(linalg.YieldOp([block.arguments[0]]))
          input1 = input1_op.result

        input2_shape = list(ir.RankedTensorType(input2.type).shape)
        if input2_shape != output_shape:
          dims = []
          for i in range(len(input2_shape)-1, -1, -1):
            if input2_shape[i] != output_shape[len(output_shape)-(len(input2_shape)-i)]:
              dims.append(i)
          output2 = tensor.EmptyOp(output_shape, ir.F32Type.get())
          generic_map = ir.AffineMap.get_permutation([i for i in range(len(output_shape)+len(dims))])
          input2_map = [i for i in range(len(output_shape)-len(input2_shape), len(output_shape))]
          for index, i in enumerate(dims):
            input2_map[i] = len(output_shape)+index
          input2_map = generic_map.get_submap(input2_map)
          input2_op = linalg.GenericOp([tensor_type], [input2], [output2],
                               ir.ArrayAttr.get([ir.AffineMapAttr.get(input2_map), ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))]))]),
                                ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(output_shape)+[ir.Attribute.parse('#linalg.iterator_type<reduction>')]*len(dims)))
          block = ir.Block.create_at_start(input2_op.region, [ir.RankedTensorType(input2.type).element_type, ir.RankedTensorType(output.result.type).element_type])
          block.append(linalg.YieldOp([block.arguments[0]]))
          input2 = input2_op.result
        generic_map = ir.AffineMap.get_permutation([i for i in range(len(output_shape))])
        op = linalg.GenericOp([tensor_type], [input1, input2], [output],
                              ir.ArrayAttr.get([ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))])), ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))])), ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))]))]),
                              ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(output_shape)))
        block = ir.Block.create_at_start(op.region, [ir.RankedTensorType(input1.type).element_type, ir.RankedTensorType(input2.type).element_type, ir.RankedTensorType(output.result.type).element_type])
        mul_op = arith.MulFOp(block.arguments[0], block.arguments[1])
        block.append(mul_op)
        block.append(linalg.YieldOp([mul_op.result]))
  
  return op

def TOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  assert len(node.args) == 1
  input1 = symbol_table.get((str(node.args[0]), 0))
  if input1 is None:
    return
  
  input_shape = list(ir.RankedTensorType(input1.type).shape)
  output_shape = list(node.meta['tensor_meta'].shape)
  dtype = str(node.meta['tensor_meta'].dtype)
  if len(input_shape) == 2:
    if dtype == "torch.float32":
      tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
      output = tensor.EmptyOp(output_shape, ir.F32Type.get())
      generic_map = ir.AffineMap.get_permutation([0, 1])
      op = linalg.GenericOp([tensor_type], [input1], [output],
                            ir.ArrayAttr.get([ir.AffineMapAttr.get(generic_map.get_submap([0, 1])), ir.AffineMapAttr.get(generic_map.get_submap([1, 0]))]),
                            ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(output_shape)))
      block = ir.Block.create_at_start(op.region, [ir.RankedTensorType(input1.type).element_type, ir.RankedTensorType(output.result.type).element_type])
      block.append(linalg.YieldOp([block.arguments[0]]))

  return op

def MMOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  assert len(node.args) == 2
  input1 = symbol_table.get((str(node.args[0]), 0))
  input2 = symbol_table.get((str(node.args[1]), 0))
  if input1 is None or input2 is None:
    return
  
  output_shape = list(node.meta['tensor_meta'].shape)
  dtype = str(node.meta['tensor_meta'].dtype)
  if dtype == "torch.float32":
    tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
    output = tensor.EmptyOp(output_shape, ir.F32Type.get())
    generic_map = ir.AffineMap.get_permutation([0, 1])
    zero_fill = linalg.GenericOp([tensor_type], [], [output],
                                      ir.ArrayAttr.get([ir.AffineMapAttr.get(generic_map.get_submap([0, 1]))]),
                                      ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*2))
    block = ir.Block.create_at_start(zero_fill.region, [ir.RankedTensorType(output.result.type).element_type])
    zero_op = arith.ConstantOp(ir.F32Type.get(), ir.FloatAttr.get(ir.F32Type.get(), 0))
    block.append(zero_op)
    block.append(linalg.YieldOp([zero_op.result]))
    generic_map = ir.AffineMap.get_permutation([0, 1, 2])
    op = linalg.GenericOp([tensor_type], [input1, input2], [zero_fill],
                          ir.ArrayAttr.get([ir.AffineMapAttr.get(generic_map.get_submap([0, 2])), ir.AffineMapAttr.get(generic_map.get_submap([2, 1])), ir.AffineMapAttr.get(generic_map.get_submap([0, 1]))]),
                          ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*2+[ir.Attribute.parse('#linalg.iterator_type<reduction>')]))
    block = ir.Block.create_at_start(op.region, [ir.RankedTensorType(input1.type).element_type, ir.RankedTensorType(input2.type).element_type, ir.RankedTensorType(output.result.type).element_type])
    mul_op = arith.MulFOp(block.arguments[0], block.arguments[1])
    add_op = arith.AddFOp(mul_op.result, block.arguments[2])
    block.append(mul_op)
    block.append(add_op)
    block.append(linalg.YieldOp([add_op.result]))

  return op

def TransposeOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  assert len(node.args) == 3
  input1 = symbol_table.get((str(node.args[0]), 0))
  if input1 is None:
    return
  dim1 = int(node.args[1])
  dim2 = int(node.args[2])
  output_shape = list(node.meta['tensor_meta'].shape)
  dtype = str(node.meta['tensor_meta'].dtype)
  if dtype == "torch.float32":
    tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
    output = tensor.EmptyOp(output_shape, ir.F32Type.get())
    generic_map = ir.AffineMap.get_permutation([i for i in range(len(output_shape))])
    input1_map = [i for i in range(len(output_shape))]
    input1_map[dim1], input1_map[dim2] = input1_map[dim2], input1_map[dim1]
    output_map = [i for i in range(len(output_shape))]
    op = linalg.GenericOp([tensor_type], [input1], [output],
                          ir.ArrayAttr.get([ir.AffineMapAttr.get(generic_map.get_submap(input1_map)), ir.AffineMapAttr.get(generic_map.get_submap(output_map))]),
                          ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(output_shape)))
    block = ir.Block.create_at_start(op.region, [ir.RankedTensorType(input1.type).element_type, ir.RankedTensorType(output.result.type).element_type])
    block.append(linalg.YieldOp([block.arguments[0]]))

  return op

def IndexOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  assert len(node.args) == 2
  input1 = symbol_table.get((str(node.args[0]), 0))
  if input1 is None:
    return
  input1_shape = ir.RankedTensorType(input1.type).shape
  input2 = node.args[1]
  output_shape = list(node.meta['tensor_meta'].shape)
  dtype = str(node.meta['tensor_meta'].dtype)
  if len(input2) < len(input1_shape):
    if dtype == "torch.float32":
      tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
      output = tensor.EmptyOp(output_shape, ir.F32Type.get())
      loops = ir.RankedTensorType(symbol_table.get((str(input2[0]), 0)).type).shape
      generic_map = ir.AffineMap.get_permutation([i for i in range(len(output_shape))])
      input_map = [ir.AffineMapAttr.get(generic_map.get_submap([j for j in range(len(loops))])) for i in range(len(input2))] + [ir.AffineMapAttr.get(generic_map.get_submap([j for j in range(len(output_shape))]))]
      operands = [symbol_table.get((str(i), 0)) for i in input2]
      op = linalg.GenericOp([tensor_type], operands, [output],
                            ir.ArrayAttr.get(input_map),
                            ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(output_shape)))
      arguments = [ir.RankedTensorType(i.type).element_type for i in operands] + [ir.RankedTensorType(output.result.type).element_type]
      block = ir.Block.create_at_start(op.region, arguments)
      index = []
      for i in block.arguments[:-1]:
        indexcast_op = arith.IndexCastOp(ir.IndexType.get(), i)
        block.append(indexcast_op)
        index.append(indexcast_op.result)
      for i in range(len(loops), len(output_shape)-len(input2)+1):
        index_op = linalg.IndexOp(ir._i64Attr(i, ctx))
        block.append(index_op)
        index.append(index_op.result)
      value = tensor.ExtractOp(input1, index)
      block.append(value)
      block.append(linalg.YieldOp([value.result]))

  return op

def NegOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  assert len(node.args) == 1
  input1 = symbol_table.get((str(node.args[0]), 0))
  if input1 is None:
    return

  output_shape = list(node.meta['tensor_meta'].shape)
  dtype = str(node.meta['tensor_meta'].dtype)
  if dtype == "torch.float32":
    tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
    output = tensor.EmptyOp(output_shape, ir.F32Type.get())
    generic_map = ir.AffineMap.get_permutation([i for i in range(len(output_shape))])
    op = linalg.GenericOp([tensor_type], [input1], [output],
                          ir.ArrayAttr.get([ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))])), ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))]))]),
                          ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(output_shape)))
    block = ir.Block.create_at_start(op.region, [ir.RankedTensorType(input1.type).element_type, ir.RankedTensorType(output.result.type).element_type])
    neg_op = arith.NegFOp(block.arguments[0])
    block.append(neg_op)
    block.append(linalg.YieldOp([neg_op.result]))

  return op

def CatOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  assert len(node.args) == 2
  input1 = symbol_table.get((str(node.args[0][0]), 0))
  input2 = symbol_table.get((str(node.args[0][1]), 0))
  dim = int(node.args[1])
  if input1 is None or input2 is None:
    return

  output_shape = list(node.meta['tensor_meta'].shape)
  if dim < 0:
    dim = len(output_shape) + dim
  dtype = str(node.meta['tensor_meta'].dtype)
  if dtype == "torch.float32":
    output = tensor.EmptyOp(output_shape, ir.F32Type.get())
    offset = [0 for x in output_shape]
    offset_attr = ir._denseI64ArrayAttr(offset, ctx)
    input1_shape = ir.RankedTensorType(input1.type).shape
    size_attr = ir._denseI64ArrayAttr(input1_shape, ctx)
    stride_attr = ir._denseI64ArrayAttr([1]*len(offset), ctx)
    insert_input1 = tensor.InsertSliceOp(input1, output.result, [], [], [], offset_attr, size_attr, stride_attr)
    offset[dim] += input1_shape[dim]
    offset_attr = ir._denseI64ArrayAttr(offset, ctx)
    input2_shape = ir.RankedTensorType(input2.type).shape
    size_attr = ir._denseI64ArrayAttr(input2_shape, ctx)
    insert_input2 = tensor.InsertSliceOp(input2, insert_input1.result, [], [], [], offset_attr, size_attr, stride_attr)

  return insert_input2

def SqueezeOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  assert len(node.args) == 2
  input1 = symbol_table.get((str(node.args[0]), 0))
  dim = int(node.args[1])
  if input1 is None:
    return

  output_shape = list(node.meta['tensor_meta'].shape)
  input1_shape = ir.RankedTensorType(input1.type).shape
  if dim < 0:
    dim = len(input1_shape) + dim
  dtype = str(node.meta['tensor_meta'].dtype)
  if dtype == "torch.float32":
    tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
    output = tensor.EmptyOp(output_shape, ir.F32Type.get())
    if input1_shape[dim] != 1:
      offset = [0 for x in output_shape]
      offset_attr = ir._denseI64ArrayAttr(offset, ctx)
      size_attr = ir._denseI64ArrayAttr(input1_shape, ctx)
      stride_attr = ir._denseI64ArrayAttr([1]*len(offset), ctx)
      op = tensor.InsertSliceOp(input1, output.result, [], [], [], offset_attr, size_attr, stride_attr)
    else:
      output_map = ir.AffineMap.get(len(output_shape), 0, [ir.AffineExpr.get_dim(i) for i in range(len(output_shape))])
      input1_map = []
      loop_index = 0
      for i in range(len(input1_shape)):
        if len(input1_map) == dim:
          input1_map.append(ir.AffineExpr.get_constant(0))
        else:
          input1_map.append(ir.AffineExpr.get_dim(loop_index))
          loop_index += 1
      input1_map = ir.AffineMap.get(len(output_shape), 0, input1_map)
      op = linalg.GenericOp([tensor_type], [input1], [output],
                            ir.ArrayAttr.get([ir.AffineMapAttr.get(input1_map), ir.AffineMapAttr.get(output_map)]),
                            ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(output_shape)))
      block = ir.Block.create_at_start(op.region, [ir.RankedTensorType(input1.type).element_type, ir.RankedTensorType(output.result.type).element_type])
      block.append(linalg.YieldOp([block.arguments[0]]))

  return op

def BMMOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context):
  assert len(node.args) == 2
  input1 = symbol_table.get((str(node.args[0]), 0))
  input2 = symbol_table.get((str(node.args[1]), 0))
  if input1 is None or input2 is None:
    return
  
  output_shape = list(node.meta['tensor_meta'].shape)
  dtype = str(node.meta['tensor_meta'].dtype)
  if dtype == "torch.float32":
    tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
    output = tensor.EmptyOp(output_shape, ir.F32Type.get())
    # use linalg.generic implementation
    # generic_map = ir.AffineMap.get_permutation([0, 1, 2])
    # zero_fill = linalg.GenericOp([tensor_type], [], [output],
    #                                   ir.ArrayAttr.get([ir.AffineMapAttr.get(generic_map.get_submap([0, 1, 2]))]),
    #                                   ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*3))
    # block = ir.Block.create_at_start(zero_fill.region, [ir.RankedTensorType(output.result.type).element_type])
    # zero_op = arith.ConstantOp(ir.F32Type.get(), ir.FloatAttr.get(ir.F32Type.get(), 0))
    # block.append(zero_op)
    # block.append(linalg.YieldOp([zero_op.result]))
    # generic_map = ir.AffineMap.get_permutation([0, 1, 2, 3])
    # op = linalg.GenericOp([tensor_type], [input1, input2], [zero_fill],
    #                       ir.ArrayAttr.get([ir.AffineMapAttr.get(generic_map.get_submap([0, 1, 3])), ir.AffineMapAttr.get(generic_map.get_submap([0, 3, 2])), ir.AffineMapAttr.get(generic_map.get_submap([0, 1, 2]))]),
    #                       ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*3+[ir.Attribute.parse('#linalg.iterator_type<reduction>')]))
    # block = ir.Block.create_at_start(op.region, [ir.RankedTensorType(input1.type).element_type, ir.RankedTensorType(input2.type).element_type, ir.RankedTensorType(output.result.type).element_type])
    # mul_op = arith.MulFOp(block.arguments[0], block.arguments[1])
    # add_op = arith.AddFOp(mul_op.result, block.arguments[2])
    # block.append(mul_op)
    # block.append(add_op)
    # block.append(linalg.YieldOp([add_op.result]))
    # linalg.BatchMatmulOp()
    op = linalg.batch_matmul(input1, input2, outs=[output])

  return op

def DivOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  assert len(node.args) == 2
  if isinstance(node.args[0], torch.fx.Node):
    input1 = symbol_table.get((str(node.args[0]), 0))
  else:
    input1 = node.args[0]
  
  if isinstance(node.args[1], torch.fx.Node):
    input2 = symbol_table.get((str(node.args[1]), 0))
  else:
    input2 = node.args[1]
  
  if input1 is None or input2 is None:
    return

  output_shape = list(node.meta['tensor_meta'].shape)
  dtype = str(node.meta['tensor_meta'].dtype)

  if isinstance(node.args[0], torch.fx.Node):
    if dtype == "torch.float32":
      if not isinstance(node.args[1], torch.fx.Node):
        input2 = arith.ConstantOp(ir.F32Type.get(), ir.FloatAttr.get(ir.F32Type.get(), input2))
        tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
        output = tensor.EmptyOp(output_shape, ir.F32Type.get())
        generic_map = ir.AffineMap.get_permutation([i for i in range(len(output_shape))])
        op = linalg.GenericOp([tensor_type], [input1], [output],
                             ir.ArrayAttr.get([ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))])), ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))]))]),
                              ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(output_shape)))
        block = ir.Block.create_at_start(op.region, [ir.RankedTensorType(input1.type).element_type, ir.RankedTensorType(output.result.type).element_type])
        div_op = arith.DivFOp(block.arguments[0], input2.result)
        block.append(div_op)
        block.append(linalg.YieldOp([div_op.result]))
      else:
        tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
        output = tensor.EmptyOp(output_shape, ir.F32Type.get())
        input1_shape = list(ir.RankedTensorType(input1.type).shape)
        if input1_shape != output_shape:
          dims = []
          for i in range(len(input1_shape)-1, -1, -1):
            if input1_shape[i] != output_shape[len(output_shape)-(len(input1_shape)-i)]:
              dims.append(i)
          output1 = tensor.EmptyOp(output_shape, ir.F32Type.get())
          generic_map = ir.AffineMap.get_permutation([i for i in range(len(output_shape)+len(dims))])
          input1_map = [i for i in range(len(output_shape)-len(input1_shape), len(output_shape))]
          for index, i in enumerate(dims):
            input1_map[i] = len(output_shape)+index
          input1_map = generic_map.get_submap(input1_map)
          input1_op = linalg.GenericOp([tensor_type], [input1], [output1],
                               ir.ArrayAttr.get([ir.AffineMapAttr.get(input1_map), ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))]))]),
                                ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(output_shape)+[ir.Attribute.parse('#linalg.iterator_type<reduction>')]*len(dims)))
          block = ir.Block.create_at_start(input1_op.region, [ir.RankedTensorType(input1.type).element_type, ir.RankedTensorType(output.result.type).element_type])
          block.append(linalg.YieldOp([block.arguments[0]]))
          input1 = input1_op.result

        input2_shape = list(ir.RankedTensorType(input2.type).shape)
        if input2_shape != output_shape:
          dims = []
          for i in range(len(input2_shape)-1, -1, -1):
            if input2_shape[i] != output_shape[len(output_shape)-(len(input2_shape)-i)]:
              dims.append(i)
          output2 = tensor.EmptyOp(output_shape, ir.F32Type.get())
          generic_map = ir.AffineMap.get_permutation([i for i in range(len(output_shape)+len(dims))])
          input2_map = [i for i in range(len(output_shape)-len(input2_shape), len(output_shape))]
          for index, i in enumerate(dims):
            input2_map[i] = len(output_shape)+index
          input2_map = generic_map.get_submap(input2_map)
          input2_op = linalg.GenericOp([tensor_type], [input2], [output2],
                               ir.ArrayAttr.get([ir.AffineMapAttr.get(input2_map), ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))]))]),
                                ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(output_shape)+[ir.Attribute.parse('#linalg.iterator_type<reduction>')]*len(dims)))
          block = ir.Block.create_at_start(input2_op.region, [ir.RankedTensorType(input2.type).element_type, ir.RankedTensorType(output.result.type).element_type])
          block.append(linalg.YieldOp([block.arguments[0]]))
          input2 = input2_op.result
        generic_map = ir.AffineMap.get_permutation([i for i in range(len(output_shape))])
        op = linalg.GenericOp([tensor_type], [input1, input2], [output],
                              ir.ArrayAttr.get([ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))])), ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))])), ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))]))]),
                              ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(output_shape)))
        block = ir.Block.create_at_start(op.region, [ir.RankedTensorType(input1.type).element_type, ir.RankedTensorType(input2.type).element_type, ir.RankedTensorType(output.result.type).element_type])
        div_op = arith.DivFOp(block.arguments[0], block.arguments[1])
        block.append(div_op)
        block.append(linalg.YieldOp([div_op.result]))
  
  return op

def SoftmaxOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  assert len(node.args) == 3
  assert node.args[2] == False
  input1 = symbol_table.get((str(node.args[0]), 0))
  dim = int(node.args[1])
  if input1 is None:
    return
  output_shape = list(node.meta['tensor_meta'].shape)
  dtype = str(node.meta['tensor_meta'].dtype)
  if dim < 0:
    dim += len(output_shape)
  if dtype == "torch.float32":
    max_tensor_shape = copy.deepcopy(output_shape)
    max_tensor_shape[dim] = 1
    max_tensor_type = ir.RankedTensorType.get(max_tensor_shape, ir.F32Type.get())
    max_tensor = tensor.EmptyOp(max_tensor_shape, ir.F32Type.get())
    max_tensor_map = [ir.AffineExpr.get_dim(i) for i in range(len(max_tensor_shape))]
    max_tensor_map = ir.AffineMap.get(len(max_tensor_shape), 0, max_tensor_map)
    neg_inf_fill = linalg.GenericOp([max_tensor_type], [], [max_tensor],
                                    ir.ArrayAttr.get([ir.AffineMapAttr.get(max_tensor_map)]),
                                    ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(max_tensor_shape)))
    block = ir.Block.create_at_start(neg_inf_fill.region, [ir.RankedTensorType(max_tensor.result.type).element_type])
    neg_inf_op = arith.ConstantOp(ir.F32Type.get(), ir.FloatAttr.get(ir.F32Type.get(), float("-inf")))
    block.append(neg_inf_op)
    block.append(linalg.YieldOp([neg_inf_op.result]))

    input1_map = [ir.AffineExpr.get_dim(i) for i in range(len(output_shape))]
    input1_map = ir.AffineMap.get(len(output_shape), 0, input1_map)
    max_tensor_map = [ir.AffineExpr.get_dim(i) for i in range(len(output_shape))]
    max_tensor_map[dim] = ir.AffineExpr.get_constant(0)
    max_tensor_map = ir.AffineMap.get(len(output_shape), 0, max_tensor_map)
    loop_type = [ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(output_shape)
    loop_type[dim] = ir.Attribute.parse('#linalg.iterator_type<reduction>')
    max_tensor_op = linalg.GenericOp([max_tensor_type], [input1], [max_tensor],
                                    ir.ArrayAttr.get([ir.AffineMapAttr.get(input1_map), ir.AffineMapAttr.get(max_tensor_map)]),
                                    ir.ArrayAttr.get(loop_type))
    block = ir.Block.create_at_start(max_tensor_op.region, [ir.RankedTensorType(input1.type).element_type, ir.RankedTensorType(max_tensor.result.type).element_type])
    max_op = arith.MaxFOp(block.arguments[0], block.arguments[1])
    block.append(max_op)
    block.append(linalg.YieldOp([max_op.result]))

    exp_tensor = tensor.EmptyOp(output_shape, ir.F32Type.get())
    exp_tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
    input1_map = [ir.AffineExpr.get_dim(i) for i in range(len(output_shape))]
    input1_map = ir.AffineMap.get(len(output_shape), 0, input1_map)
    max_tensor_map = [ir.AffineExpr.get_dim(i) for i in range(len(output_shape))]
    max_tensor_map[dim] = ir.AffineExpr.get_constant(0)
    max_tensor_map = ir.AffineMap.get(len(output_shape), 0, max_tensor_map)
    exp_tensor_map = [ir.AffineExpr.get_dim(i) for i in range(len(output_shape))]
    exp_tensor_map = ir.AffineMap.get(len(output_shape), 0, exp_tensor_map)
    exp_tensor_op = linalg.GenericOp([exp_tensor_type], [input1, max_tensor_op.result], [exp_tensor],
                                    ir.ArrayAttr.get([ir.AffineMapAttr.get(input1_map), ir.AffineMapAttr.get(max_tensor_map), ir.AffineMapAttr.get(exp_tensor_map)]),
                                    ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(output_shape)))
    block = ir.Block.create_at_start(exp_tensor_op.region, [ir.RankedTensorType(input1.type).element_type, ir.RankedTensorType(max_tensor_op.result.type).element_type, ir.RankedTensorType(exp_tensor.result.type).element_type])
    sub_op = arith.SubFOp(block.arguments[0], block.arguments[1])
    exp_op = math.ExpOp(sub_op.result)
    block.append(sub_op)
    block.append(exp_op)
    block.append(linalg.YieldOp([exp_op.result]))
    
    reduce_sum_tensor_shape = copy.deepcopy(output_shape)
    reduce_sum_tensor_shape[dim] = 1
    reduce_sum_tensor = tensor.EmptyOp(reduce_sum_tensor_shape, ir.F32Type.get())
    reduce_sum_tensor_type = ir.RankedTensorType.get(reduce_sum_tensor_shape, ir.F32Type.get())
    reduce_sum_tensor_map = [ir.AffineExpr.get_dim(i) for i in range(len(output_shape))]
    reduce_sum_tensor_map = ir.AffineMap.get(len(output_shape), 0, reduce_sum_tensor_map)
    zero_fill_op = linalg.GenericOp([reduce_sum_tensor_type], [], [reduce_sum_tensor.result],
                                    ir.ArrayAttr.get([ir.AffineMapAttr.get(reduce_sum_tensor_map)]),
                                    ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(output_shape)))
    block = ir.Block.create_at_start(zero_fill_op.region, [ir.RankedTensorType(reduce_sum_tensor.result.type).element_type])
    zero_op = arith.ConstantOp(ir.F32Type.get(), ir.FloatAttr.get(ir.F32Type.get(), 0))
    block.append(zero_op)
    block.append(linalg.YieldOp([zero_op.result]))

    reduce_sum_tensor_shape = copy.deepcopy(output_shape)
    reduce_sum_tensor_shape[dim] = 1
    reduce_sum_tensor_type = ir.RankedTensorType.get(reduce_sum_tensor_shape, ir.F32Type.get())
    exp_tensor_map = [ir.AffineExpr.get_dim(i) for i in range(len(output_shape))]
    exp_tensor_map = ir.AffineMap.get(len(output_shape), 0, exp_tensor_map)
    reduce_sum_tensor_map = [ir.AffineExpr.get_dim(i) for i in range(len(output_shape))]
    reduce_sum_tensor_map[dim] = ir.AffineExpr.get_constant(0)
    reduce_sum_tensor_map = ir.AffineMap.get(len(output_shape), 0, reduce_sum_tensor_map)
    loop_type = [ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(output_shape)
    loop_type[dim] = ir.Attribute.parse('#linalg.iterator_type<reduction>')
    reduce_sum_tensor_op = linalg.GenericOp([reduce_sum_tensor_type], [exp_tensor_op.result], [zero_fill_op.result],
                                    ir.ArrayAttr.get([ir.AffineMapAttr.get(exp_tensor_map), ir.AffineMapAttr.get(reduce_sum_tensor_map)]),
                                    ir.ArrayAttr.get(loop_type))
    block = ir.Block.create_at_start(reduce_sum_tensor_op.region, [ir.RankedTensorType(exp_tensor_op.result.type).element_type, ir.RankedTensorType(zero_fill_op.result.type).element_type])
    add_op = arith.AddFOp(block.arguments[0], block.arguments[1])
    block.append(add_op)
    block.append(linalg.YieldOp([add_op.result]))

    reduce_sum_tensor_shape = copy.deepcopy(output_shape)
    reduce_sum_tensor_shape[dim] = 1
    result_tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
    result_tensor = tensor.EmptyOp(output_shape, ir.F32Type.get())
    exp_tensor_map = [ir.AffineExpr.get_dim(i) for i in range(len(output_shape))]
    exp_tensor_map = ir.AffineMap.get(len(output_shape), 0, exp_tensor_map)
    reduce_sum_tensor_map = [ir.AffineExpr.get_dim(i) for i in range(len(output_shape))]
    reduce_sum_tensor_map[dim] = ir.AffineExpr.get_constant(0)
    reduce_sum_tensor_map = ir.AffineMap.get(len(output_shape), 0, reduce_sum_tensor_map)
    result_tensor_map = [ir.AffineExpr.get_dim(i) for i in range(len(output_shape))]
    result_tensor_map = ir.AffineMap.get(len(output_shape), 0, result_tensor_map)
    op = linalg.GenericOp([result_tensor_type], [exp_tensor_op.result, reduce_sum_tensor_op.result], [result_tensor.result],
                          ir.ArrayAttr.get([ir.AffineMapAttr.get(exp_tensor_map), ir.AffineMapAttr.get(reduce_sum_tensor_map), ir.AffineMapAttr.get(result_tensor_map)]),
                          ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(output_shape)))
    block = ir.Block.create_at_start(op.region, [ir.RankedTensorType(exp_tensor_op.result.type).element_type, ir.RankedTensorType(reduce_sum_tensor_op.result.type).element_type, ir.RankedTensorType(result_tensor.result.type).element_type])
    div_op = arith.DivFOp(block.arguments[0], block.arguments[1])
    block.append(div_op)
    block.append(linalg.YieldOp([div_op.result]))

  return op

def CloneOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  assert len(node.args) == 1
  input1 = symbol_table.get((str(node.args[0]), 0))
  if input1 is None:
    return
  
  output_shape = list(node.meta['tensor_meta'].shape)
  dtype = str(node.meta['tensor_meta'].dtype)
  if dtype == "torch.float32":
    offset = [0 for x in output_shape]
    offset_attr = ir._denseI64ArrayAttr(offset, ctx)
    size_attr = ir._denseI64ArrayAttr(output_shape, ctx)
    stride = [1 for x in output_shape]
    stride_attr = ir._denseI64ArrayAttr(stride, ctx)
    tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())

    op = tensor.ExtractSliceOp(tensor_type, input1, [], [], [], offset_attr, size_attr, stride_attr)

  return op

def SiluOp(node: torch.fx.Node,
            symbol_table: Dict[Tuple[str, int], ir.Operation],
            ctx: ir.Context) -> ir.Operation:
  assert len(node.args) == 1
  input1 = symbol_table.get((str(node.args[0]), 0))
  if input1 is None:
    return

  output_shape = list(node.meta['tensor_meta'].shape)
  dtype = str(node.meta['tensor_meta'].dtype)
  if dtype == "torch.float32":
    tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
    output = tensor.EmptyOp(output_shape, ir.F32Type.get())
    generic_map = ir.AffineMap.get_permutation([i for i in range(len(output_shape))])
    op = linalg.GenericOp([tensor_type], [input1], [output],
                          ir.ArrayAttr.get([ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))])), ir.AffineMapAttr.get(generic_map.get_submap([i for i in range(len(output_shape))]))]),
                          ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*len(output_shape)))
    block = ir.Block.create_at_start(op.region, [ir.RankedTensorType(input1.type).element_type, ir.RankedTensorType(output.result.type).element_type])
    neg_op = arith.NegFOp(block.arguments[0])
    exp_op = math.ExpOp(neg_op.result)
    one_op = arith.ConstantOp(ir.F32Type.get(), ir.FloatAttr.get(ir.F32Type.get(), 1))
    add_op = arith.AddFOp(one_op.result, exp_op.result)
    div_op = arith.DivFOp(block.arguments[0], add_op.result)
    block.append(neg_op)
    block.append(exp_op)
    block.append(one_op)
    block.append(add_op)
    block.append(div_op)
    block.append(linalg.YieldOp([div_op.result]))

  return op

def ParamToConstantOp(node: torch.fx.Node,
                      index: int,
                      ) -> ir.Operation:
  dtype = str(node.meta['tensor_meta'].dtype)
  if dtype == "torch.float32":
    param_data = numpy.fromfile(global_var_get_value("params-write-path")+"/params_data/arg"+str(index)+".data", dtype=numpy.float32)
  output_shape = list(node.meta['tensor_meta'].shape)
  param_data = param_data.reshape(output_shape)
  if dtype == "torch.float32":
    tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
  attr = ir.DenseFPElementsAttr.get(param_data, signless=True, type=tensor_type)
  op = arith.ConstantOp(tensor_type, attr)
  return op

def ParamExtract(node: torch.fx.Node,
                offset,
                params_mlir_node,
                ctx: ir.Context) -> ir.Operation:
  dtype = str(node.meta['tensor_meta'].dtype)
  output_shape = list(node.meta['tensor_meta'].shape)
  extract_size = functools.reduce(lambda x, y : x*y, output_shape)
  if dtype == "torch.float32":
    offset_attr = ir._denseI64ArrayAttr([offset], ctx)
    size_attr = ir._denseI64ArrayAttr([extract_size], ctx)
    stride = [1]
    stride_attr = ir._denseI64ArrayAttr(stride, ctx)
    tensor_type = ir.RankedTensorType.get([extract_size], ir.F32Type.get())
    extract_slice_op = tensor.ExtractSliceOp(tensor_type, params_mlir_node, [], [], [], offset_attr, size_attr, stride_attr)
    if len(output_shape) == 1:
      return extract_slice_op
    tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
    axis = ir.ArrayAttr.get([ir.IntegerAttr.get(ir.IntegerType.get_signless(64), i) for i in range(len(output_shape))], ctx)
    axis = ir.ArrayAttr.get([axis], ctx)
    expand_shape_op = tensor.ExpandShapeOp(tensor_type, extract_slice_op.result, axis)
  
  return expand_shape_op

operation_func = {"arange.start": ArangeOp, "arange.default": ArangeOp, "unsqueeze.default": UnsqueezeOp, "view.default": ViewOp,
                  "ones.default": OnesOp, "full.default": FullOp, "add.Tensor": AddOp, "lt.Tensor": LtOp, "embedding.default": EmbeddingOp,
                  "masked_fill.Scalar": MaskedFillOp, "slice.Tensor": SliceOp, "expand.default": ExpandOp, "_to_copy.default": ToCopyOp,
                  "rsub.Scalar": RSubOp, "pow.Tensor_Scalar": PowOp, "mean.dim": MeanOp, "rsqrt.default": RSqrtOp, "mul.Tensor": MulOp,
                  "t.default": TOp, "mm.default": MMOp, "transpose.int": TransposeOp, "index.Tensor": IndexOp, "neg.default": NegOp,
                  "cat.default": CatOp, "squeeze.dim": SqueezeOp, "bmm.default": BMMOp, "div.Tensor": DivOp, "_softmax.default": SoftmaxOp,
                  "clone.default": CloneOp, "silu.default": SiluOp}
# operation_func = {"arange.start": ArangeOp, "arange.default": ArangeOp, "unsqueeze.default": UnsqueezeOp, "view.default": ViewOp,
#                   "ones.default": OnesOp, "full.default": FullOp, "add.Tensor": AddOp, "lt.Tensor": LtOp, "embedding.default": EmbeddingOp,
#                   "masked_fill.Scalar": MaskedFillOp, "slice.Tensor": SliceOp, "squeeze.dim": SqueezeOp, "expand.default": ExpandOp, 
#                   "_to_copy.default": ToCopyOp, "rsub.Scalar": RSubOp, "pow.Tensor_Scalar": PowOp, "mean.dim": MeanOp, "rsqrt.default": RSqrtOp, 
#                   "mul.Tensor": MulOp, "t.default": TOp, "mm.default": MMOp}
type_dict = {"torch.int64": ir.IntegerType, "torch.float32": ir.F32Type}