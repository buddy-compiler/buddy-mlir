# RUN: %PYTHON %s
"""CPU mean, stack and scalar clamp semantics across shapes and dtypes."""

from types import SimpleNamespace

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy_mlir import ir
from buddy_mlir.dialects import arith


class Case(torch.nn.Module):
    def __init__(self, op, args, kwargs=None):
        super().__init__()
        self.op, self.args, self.kwargs = op, args, kwargs or {}

    def forward(self, x, y=None):
        if self.op == "stack":
            return torch.ops.aten.stack.default(
                [x, y], *self.args, **self.kwargs
            )
        return self.op(x, *self.args, **self.kwargs)


def check(model, x):
    inputs = (x, x + 1) if model.op == "stack" else (x,)
    expected = model(*inputs)
    exported = torch.export.export(model, inputs, strict=True)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry, enable_external_calls=False
    )
    compiler._compile_fx(exported.graph_module, list(inputs))
    actual = compiler.dynamo_run()(*inputs)[0]
    torch.testing.assert_close(
        actual, expected, rtol=1e-4, atol=1e-5, equal_nan=True
    )


torch.set_num_threads(1)
torch.manual_seed(0)
count = 0
for dtype in (torch.float32, torch.float64):
    for shape, dims, keep in (
        ((), [0], False),
        ((), None, False),
        ((3, 5), [-1], False),
        ((1, 3, 1), [1], False),
        ((2, 1, 3), [0, 2], False),
        ((2, 3), [0, 1], True),
        ((2, 3), [], False),
        ((2, 3), None, True),
        ((2, 0, 3), [1], False),
        ((0, 3), [1], False),
    ):
        check(
            Case(
                torch.ops.aten.mean.dim, (dims,) if not keep else (dims, keep)
            ),
            torch.randn(shape, dtype=dtype),
        )
        count += 1
for dtype in (torch.float32, torch.float64, torch.int32, torch.int64):
    for shape in ((), (2, 3), (0, 3)):
        x = torch.zeros(shape, dtype=dtype)
        for dim in (0, -1):
            check(Case("stack", (dim,)), x)
            count += 1
    for bounds in ((-1, 2), (None, 2), (-1, None), (2, -1)):
        x = torch.tensor([-4, -1, 0, 1, 5], dtype=dtype)
        if dtype.is_floating_point:
            x = torch.cat(
                (
                    x,
                    torch.tensor(
                        [float("nan"), float("inf"), float("-inf")], dtype=dtype
                    ),
                )
            )
        check(Case(torch.ops.aten.clamp.default, bounds), x)
        count += 1
    for shape in ((), (0, 3)):
        check(
            Case(torch.ops.aten.clamp.default, (), {"min": -1, "max": 1}),
            torch.zeros(shape, dtype=dtype),
        )
        count += 1
print(f"Mean, stack and clamp: {count} CPU JIT cases passed")

# Reject overflowing bounds before constructing an integer attribute that wraps.

with ir.Context(), ir.Location.unknown():
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        for width in (32, 64):
            dtype = ir.IntegerType.get_signless(width)
            typ = ir.RankedTensorType.get([1], dtype)
            value = arith.ConstantOp(
                typ,
                ir.DenseElementsAttr.get_splat(
                    typ, ir.IntegerAttr.get(dtype, 0)
                ),
            ).result
            for bound in (1 << (width - 1), -(1 << (width - 1)) - 1):
                node = SimpleNamespace(args=["x", bound], kwargs={})
                try:
                    tosa.clamp_op(node, {("x", 0): value})
                except OverflowError:
                    pass
                else:
                    raise AssertionError("Overflowing clamp bound was accepted")
print("Clamp: 4 overflowing-bound lowering checks passed")
