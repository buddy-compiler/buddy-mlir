# RUN: %PYTHON %s

import importlib.util
from pathlib import Path

from buddy.compiler.graph import MegaMaxPool2dOp
from buddy_mlir import ir
from buddy_mlir.dialects import func

source = Path(__file__).resolve().parents[2] / "frontend/Python/ops/linalg.py"
spec = importlib.util.spec_from_file_location(
    "buddy.compiler.ops.pool_geometry", source
)
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)

for size, kernel, stride, padding in (
    (55, 3, 2, 0),
    (27, 3, 2, 0),
    (13, 3, 2, 0),
    (56, 3, 2, 1),
    (24, 2, 2, 0),
    (8, 2, 2, 0),
    (112, 3, 2, 1),
    (20, 5, 1, 2),
):
    for final in (False, True):
        pooled = (size + 2 * padding - kernel) // stride + 1
        with ir.Context(), ir.Location.unknown():
            module = ir.Module.create()
            i8 = ir.IntegerType.get_signless(8)
            input_type = ir.RankedTensorType.get([1, size, size, 64], i8)
            shape = (
                [1, 64, pooled, pooled] if final else [1, pooled, pooled, 64]
            )
            output_type = ir.RankedTensorType.get(shape, i8)
            with ir.InsertionPoint(module.body):
                function = func.FuncOp("pool", ([input_type], [output_type]))
                block = function.add_entry_block()
                with ir.InsertionPoint(block):
                    node = MegaMaxPool2dOp()
                    node.name = "pool"
                    node.add_argument("input")
                    node._output_shape = [1, 64, pooled, pooled]
                    node._kernel, node._stride, node._padding = (
                        kernel,
                        stride,
                        padding,
                    )
                    node._final_output = final
                    value = ops.mega_max_pool2d_op(
                        node, {("input", 0): block.arguments[0]}
                    )
                    func.ReturnOp([value])
            assert module.operation.verify()
            assert ir.RankedTensorType(value.type).shape == shape
            assert f"stride = {stride} : i64" in str(module)
            assert f"kernel = {kernel} : i64" in str(module)

print("Mega MaxPool odd-size and padded geometry checks passed")
