# RUN: %PYTHON %s

import importlib.util
from pathlib import Path

import buddy_mlir.ir as ir
from buddy.compiler.graph import MegaConv2dDepthwiseOp, MegaConv2dOp, ReshapeOp
from buddy_mlir.dialects import func, tensor


def source_ops(name):
    path = (
        Path(__file__).resolve().parents[2]
        / "frontend/Python/ops"
        / f"{name}.py"
    )
    spec = importlib.util.spec_from_file_location(
        f"buddy.compiler.ops.layout_test_{name}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


conv_ops = source_ops("linalg")
reshape_ops = source_ops("tosa")


def check(depthwise, final, reshape_shape, expected_transposes):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        i8, f32 = ir.IntegerType.get_signless(8), ir.F32Type.get()
        activation_type = ir.RankedTensorType.get([1, 4, 4, 4], i8)
        weight_shape = [3, 3, 4, 1] if depthwise else [1, 4, 16, 16]
        weight_type = ir.RankedTensorType.get(weight_shape, i8)
        output_type = ir.RankedTensorType.get(
            reshape_shape or [1, 4, 4, 4], f32 if final else i8
        )
        with ir.InsertionPoint(module.body):
            function = func.FuncOp(
                "test", ([activation_type, weight_type], [output_type])
            )
            block = function.add_entry_block()
            with ir.InsertionPoint(block):
                node = MegaConv2dDepthwiseOp() if depthwise else MegaConv2dOp()
                node.name = "conv"
                node.add_argument("input")
                node.add_argument("weight")
                node._input_shape = node._output_shape = [1, 4, 4, 4]
                node._weight_shape = [4, 1, 3, 3] if depthwise else [4, 4, 1, 1]
                node._final_output, node._activation = final, 0
                node._stride, node._padding = 1, 1 if depthwise else 0
                node._bias_i32 = [0] * 4
                node._dequant_scale = node._requant_scale = [1.0] * 4
                node._lut_i8, node._output_scale = [0], 1.0
                node.tensor_meta = {"shape": [1, 4, 4, 4]}
                symbols = {
                    ("input", 0): block.arguments[0],
                    ("weight", 0): block.arguments[1],
                }
                value = conv_ops.mega_conv2d_op(node, symbols)
                if reshape_shape:
                    reshape = ReshapeOp()
                    reshape.add_argument("conv")
                    reshape._newshape = reshape_shape
                    symbols[("conv", 0)] = value
                    symbols[("__buddy_ops_by_name__", 0)] = {"conv": node}
                    value = reshape_ops.reshape_op(reshape, symbols)
                    if not isinstance(value, ir.Value):
                        value = value.result
                func.ReturnOp([value])
        assert module.operation.verify()
        text = str(module)
        assert text.count("tosa.transpose") == expected_transposes, text
        if expected_transposes:
            assert "perms = array<i32: 0, 3, 1, 2>" in text, text


for depthwise in (False, True):
    check(depthwise, False, None, 0)
    for shape in ([1, 4, 16], [1, 2, 2, 16]):
        check(depthwise, True, shape, 0)
        check(depthwise, False, shape, 1)

# Ordinary FP32 tensors retain sticky-NHWC handling: dtype alone is not proof
# that a producer follows the final Mega Conv NCHW contract.
with ir.Context(), ir.Location.unknown():
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        value = tensor.EmptyOp([1, 4, 4, 4], ir.F32Type.get()).result
        producer = MegaConv2dOp()
        producer.tensor_meta = {"shape": [1, 4, 4, 4]}
        reshape = ReshapeOp()
        reshape.add_argument("ordinary")
        reshape._newshape = [1, 4, 16]
        reshape_ops.reshape_op(
            reshape,
            {
                ("ordinary", 0): value,
                ("__buddy_ops_by_name__", 0): {"ordinary": producer},
            },
        )
    assert str(module).count("tosa.transpose") == 1, str(module)

print("equal-dimension Mega Conv/DW and reshape layout checks passed")
