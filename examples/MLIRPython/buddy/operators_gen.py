"""Generate the MLIR operations for the operators in the FX graph.
"""
from typing import Dict, Tuple, List

import torch

import mlir.ir as ir
from mlir.dialects import tosa, linalg, arith


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
          symbol_table: Dict[Tuple[str, int], ir.Operation]) -> ir.Operation:
  """Map aten.add.Tensor to tosa.add.

  Args:
    node: A FX graph containing the aten.add.Tensor operator and its parameter.
    symbol_table: The symbol table that records the mapping between symbols and operations.

  Returns:
    ir.Operation: The generated tosa.add operation.

  """
  input1 = symbol_table.get((str(node.args[0]), 0))
  input2 = symbol_table.get((str(node.args[1]), 0))
  broadcasted_shp = _broadcast_shape(input1, input2)
  sizes = broadcasted_shp
  f32 = ir.F32Type.get()
  addResultTensorType = ir.RankedTensorType.get(sizes, f32)
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


operation_func = {"add.Tensor": AddOp, "addmm.default": AddMMOp}
