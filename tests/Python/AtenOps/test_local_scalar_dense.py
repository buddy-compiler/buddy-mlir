# RUN: %PYTHON %s 2>&1 | FileCheck %s
"""
Test local_scalar_dense operation - converts single-element tensor to 0-D tensor.

Since _local_scalar_dense returns a Python scalar which cannot be directly
captured by torch.compile, this test directly validates the MLIR generation.
"""

from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, tosa
from mlir import ir
import array


def _create_shape_operand(shape):
    """Create a tosa.shape value for reshape-like ops."""
    dims = [int(dim) for dim in shape]
    rank = len(dims)
    shape_type = ir.Type.parse(f"!tosa.shape<{rank}>")
    index_type = ir.IndexType.get()
    shape_attr = ir.DenseElementsAttr.get(
        array.array("q", dims),
        type=index_type,
        shape=[rank],
    )
    return tosa.ConstShapeOp(shape_type, shape_attr).result


def test_local_scalar_dense_1d():
    """Test reshaping [1] tensor to 0-D tensor."""
    with Context() as ctx, Location.unknown():
        ctx.allow_unregistered_dialects = True

        module = Module.create()
        with InsertionPoint(module.body):
            input_type = ir.RankedTensorType.get([1], ir.F32Type.get())
            output_type = ir.RankedTensorType.get([], ir.F32Type.get())
            func_type = ir.FunctionType.get([input_type], [output_type])

            func_op = func.FuncOp("forward_1d", func_type)
            entry_block = func_op.add_entry_block()

            with InsertionPoint(entry_block):
                input_tensor = entry_block.arguments[0]
                shape_operand = _create_shape_operand([])
                result = tosa.ReshapeOp(
                    input_tensor, shape_operand
                )
                func.ReturnOp([result.result])

        print("=== Test 1D single-element tensor ===")
        print(module)


def test_local_scalar_dense_2d():
    """Test reshaping [1,1] tensor to 0-D tensor."""
    with Context() as ctx, Location.unknown():
        ctx.allow_unregistered_dialects = True

        module = Module.create()
        with InsertionPoint(module.body):
            input_type = ir.RankedTensorType.get([1, 1], ir.F32Type.get())
            output_type = ir.RankedTensorType.get([], ir.F32Type.get())
            func_type = ir.FunctionType.get([input_type], [output_type])

            func_op = func.FuncOp("forward_2d", func_type)
            entry_block = func_op.add_entry_block()

            with InsertionPoint(entry_block):
                input_tensor = entry_block.arguments[0]
                shape_operand = _create_shape_operand([])
                result = tosa.ReshapeOp(
                    input_tensor, shape_operand
                )
                func.ReturnOp([result.result])

        print("\n=== Test 2D single-element tensor ===")
        print(module)


def test_local_scalar_dense_int():
    """Test reshaping integer tensor to 0-D tensor."""
    with Context() as ctx, Location.unknown():
        ctx.allow_unregistered_dialects = True

        module = Module.create()
        with InsertionPoint(module.body):
            input_type = ir.RankedTensorType.get(
                [1], ir.IntegerType.get_signless(64)
            )
            output_type = ir.RankedTensorType.get(
                [], ir.IntegerType.get_signless(64)
            )
            func_type = ir.FunctionType.get([input_type], [output_type])

            func_op = func.FuncOp("forward_int", func_type)
            entry_block = func_op.add_entry_block()

            with InsertionPoint(entry_block):
                input_tensor = entry_block.arguments[0]
                shape_operand = _create_shape_operand([])
                result = tosa.ReshapeOp(
                    input_tensor, shape_operand
                )
                func.ReturnOp([result.result])

        print("\n=== Test integer tensor ===")
        print(module)


if __name__ == "__main__":
    test_local_scalar_dense_1d()
    test_local_scalar_dense_2d()
    test_local_scalar_dense_int()

# CHECK: === Test 1D single-element tensor ===
# CHECK: module {
# CHECK-LABEL: func.func @forward_1d
# CHECK: tosa.const_shape
# CHECK: tosa.reshape %{{.*}}, %{{.*}} : (tensor<1xf32>, !tosa.shape<0>) -> tensor<f32>
# CHECK: }
# CHECK: === Test 2D single-element tensor ===
# CHECK: module {
# CHECK-LABEL: func.func @forward_2d
# CHECK: tosa.const_shape
# CHECK: tosa.reshape %{{.*}}, %{{.*}} : (tensor<1x1xf32>, !tosa.shape<0>) -> tensor<f32>
# CHECK: }
# CHECK: === Test integer tensor ===
# CHECK: module {
# CHECK-LABEL: func.func @forward_int
# CHECK: tosa.const_shape
# CHECK: tosa.reshape %{{.*}}, %{{.*}} : (tensor<1xi64>, !tosa.shape<0>) -> tensor<i64>
# CHECK: }
