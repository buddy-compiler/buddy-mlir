"""Generate the MLIR operations for the operators in the FX graph.
"""
from typing import Dict, Tuple, List

import torch

import mlir.ir as ir
from mlir.dialects import tosa, linalg, arith, tensor
import numpy


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
  value = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 3)
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
    generic_map = ir.AffineMap.get_permutation([0, 1])
    op = linalg.GenericOp([tensor_type], [input1, input2], [output],
                         ir.ArrayAttr.get([ir.AffineMapAttr.get(generic_map.get_submap([0, 1])), ir.AffineMapAttr.get(generic_map.get_submap([0, 1])), ir.AffineMapAttr.get(generic_map.get_submap([0, 1]))]),
                          ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*2))
    block = ir.Block.create_at_start(op.region, [ir.RankedTensorType(input1.type).element_type, ir.RankedTensorType(input2.type).element_type, ir.RankedTensorType(output.result.type).element_type])
    select_op = arith.SelectOp(block.arguments[1], value, block.arguments[0])
    block.append(select_op)
    block.append(linalg.YieldOp([select_op.result]))
  
  return op

def SliceOp(node: torch.fx.Node,
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
    generic_map = ir.AffineMap.get_permutation([0, 1])
    op = linalg.GenericOp([tensor_type], [input1, input2], [output],
                         ir.ArrayAttr.get([ir.AffineMapAttr.get(generic_map.get_submap([0, 1])), ir.AffineMapAttr.get(generic_map.get_submap([0, 1])), ir.AffineMapAttr.get(generic_map.get_submap([0, 1]))]),
                          ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*2))
    block = ir.Block.create_at_start(op.region, [ir.RankedTensorType(input1.type).element_type, ir.RankedTensorType(input2.type).element_type, ir.RankedTensorType(output.result.type).element_type])
    select_op = arith.SelectOp(block.arguments[1], value, block.arguments[0])
    block.append(select_op)
    block.append(linalg.YieldOp([select_op.result]))
  
  return op

operation_func = {"arange.start": ArangeOp, "arange.default": ArangeOp, "unsqueeze.default": UnsqueezeOp, "view.default": ViewOp,
                  "ones.default": OnesOp, "full.default": FullOp, "add.Tensor": AddOp, "lt.Tensor": LtOp, "embedding.default": EmbeddingOp,
                  "masked_fill.Scalar": MaskedFillOp}
type_dict = {"torch.int64": ir.IntegerType, "torch.float32": ir.F32Type}