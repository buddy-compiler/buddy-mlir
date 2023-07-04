import array

from mlir.ir import (
    RankedTensorType,
    F32Type,
)
from mlir.dialects import tosa


def _broadcast_shape(tensor_input1, tensor_input2):
    shp1 = RankedTensorType(tensor_input1.type).shape
    shp2 = RankedTensorType(tensor_input2.type).shape
    if len(shp1) < len(shp2):
        shp1, shp2 = shp2, shp1
    while len(shp2) < len(shp1):
        shp2.insert(0, 1)
    for idx, (dim1, dim2) in enumerate(zip(shp1, shp2)):
        shp1[idx] = shp2[idx] = max(dim1, dim2)

    return shp1


def AddOp(node, symbol_table):
    input1 = symbol_table.get((str(node.args[0]), 0))
    input2 = symbol_table.get((str(node.args[1]), 0))
    broadcasted_shp = _broadcast_shape(input1, input2)
    sizes = broadcasted_shp
    f32 = F32Type.get()
    addResultTensorType = RankedTensorType.get(sizes, f32)
    op = tosa.AddOp(addResultTensorType, input1, input2)
    return op


operation_func = {"add": AddOp}
