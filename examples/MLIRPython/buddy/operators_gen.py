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


def add_op(node: torch.fx.Node,
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
  result_element_type = ir.RankedTensorType(input1.type).element_type
  add_result_tensor_type = ir.RankedTensorType.get(sizes, result_element_type)
  op = tosa.AddOp(add_result_tensor_type, input1, input2)
  return op


def addmm_op(node: torch.fx.Node,
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
  mat1 = tosa.ReshapeOp(mat1, [1, *mat1_shp]).output
  mat2 = tosa.ReshapeOp(mat2, [1, *mat2_shp]).output

  matmul_result_shp = [1, mat1_shp[0], mat2_shp[1]]
  result_element_type = ir.RankedTensorType(input_.type).element_type
  matmul_result_type = ir.RankedTensorType.get(matmul_result_shp,
                                               result_element_type)
  matmul_op = tosa.MatMulOp(matmul_result_type, mat1, mat2)
  matmul_result = tosa.ReshapeOp(matmul_op.c, matmul_result_shp[1:])

  add_result_shp = [mat1_shp[0], mat2_shp[1]]
  add_result_tensor_type = ir.RankedTensorType.get(add_result_shp,
                                                   result_element_type)
  op = tosa.AddOp(add_result_tensor_type, input_, matmul_result)
  return op


def sub_op(node, symbol_table):
  """Map aten.sub.Tensor to MLIR operation.

  Args:
    node (torch.fx.Node): A FX graph containing the aten.addmm.default operator and its parameter.
    symbol_table (Dict[Tuple[str, int], ir.Operation]): The symbol table that records the mapping between symbols and operations.

  Returns:
    ir.Operation: The generated MLIR operation representing aten.addmm.default

  """
  input1 = symbol_table.get((str(node.args[0]), 0))
  input2 = symbol_table.get((str(node.args[1]), 0))
  broadcasted_shp = _broadcast_shape(input1, input2)
  sizes = broadcasted_shp
  result_element_type = ir.RankedTensorType(input1.type).element_type
  sub_result_tensor_type = ir.RankedTensorType.get(sizes, result_element_type)
  op = tosa.SubOp(sub_result_tensor_type, input1, input2)
  return op


def mul_op(node, symbol_table):
  """Map aten.mul.Tensor to MLIR operation.

  Args:
    node (torch.fx.Node): A FX graph containing the aten.addmm.default operator and its parameter.
    symbol_table (Dict[Tuple[str, int], ir.Operation]): The symbol table that records the mapping between symbols and operations.

  Returns:
    ir.Operation: The generated MLIR operation representing aten.addmm.default

  """
  input1 = symbol_table.get((str(node.args[0]), 0))
  input2 = symbol_table.get((str(node.args[1]), 0))
  broadcasted_shp = _broadcast_shape(input1, input2)
  sizes = broadcasted_shp
  result_element_type = ir.RankedTensorType(input1.type).element_type
  mul_result_tensor_type = ir.RankedTensorType.get(sizes, result_element_type)
  op = tosa.MulOp(mul_result_tensor_type, input1, input2,
                  ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0))
  return op


operation_func = {
    "add.Tensor": add_op,
    "addmm.default": addmm_op,
    "sub.Tensor": sub_op,
    "mul.Tensor": mul_op
}
