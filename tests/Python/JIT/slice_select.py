# RUN: %PYTHON %s

"""CPU slicing: strides, offsets, clipped bounds, empty results and rank reduction."""

import math

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import linalg, tosa


class Case(torch.nn.Module):
    def __init__(self, operation, arguments, chained=False):
        super().__init__()
        self.operation, self.arguments, self.chained = (
            operation,
            arguments,
            chained,
        )

    def forward(self, x):
        if self.chained:
            x = torch.ops.aten.slice.Tensor(x, -1, 1, None, 2)
        return self.operation(x, *self.arguments)


def check(model, shape, dtype, registry):
    x = (
        torch.arange(math.prod(shape), dtype=torch.float64)
        .reshape(shape)
        .to(dtype)
    )
    if dtype == torch.bool:
        x = torch.arange(math.prod(shape)).reshape(shape) % 3 == 0
    original = x.clone()
    exported = torch.export.export(model, (x,), strict=True)
    compiler = DynamoCompiler(
        primary_registry=registry, enable_external_calls=False
    )
    compiler._compile_fx(exported.graph_module, [x])
    outputs = compiler.dynamo_run()(x)
    assert len(outputs) == 1
    torch.testing.assert_close(outputs[0], model(x), rtol=0, atol=0)
    torch.testing.assert_close(x, original, rtol=0, atol=0)


torch.set_num_threads(1)
count = 0
for dtype in (torch.float32, torch.float64, torch.int64, torch.bool):
    for registry in (tosa.ops_registry, linalg.ops_registry):
        for shape, arguments in (
            ((3, 5), (1, 1, 5, 2)),
            ((3, 5), (-1, -4, -1, 2)),
            ((3, 5), (1, -100, 100, 3)),
            ((3, 5), (0, 1, 3, 1)),
            ((3, 5), (1, 4, 2, 1)),
            ((3, 5), (1, 100, None, 1)),
            ((3, 0), (1, None, None, 2)),
            ((7,), (0, None, None, 3)),
            ((2, 3, 5), (1, 0, 3, 2)),
        ):
            check(
                Case(torch.ops.aten.slice.Tensor, arguments),
                shape,
                dtype,
                registry,
            )
            count += 1
    for shape, arguments in (
        ((3, 5), (1, 4)),
        ((3, 5), (-1, -1)),
        ((3, 5), (0, -2)),
        ((2, 3, 5), (1, 1)),
        ((5,), (0, -1)),
        ((1, 3, 1), (0, 0)),
        ((0, 3), (1, 2)),
    ):
        check(
            Case(torch.ops.aten.select.int, arguments),
            shape,
            dtype,
            tosa.ops_registry,
        )
        count += 1
    check(
        Case(torch.ops.aten.select.int, (-1, -1), chained=True),
        (3, 7),
        dtype,
        tosa.ops_registry,
    )
    count += 1


class MatmulCase(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x, y):
        if self.mode == "strided":
            y = y[:, 1::2]
        elif self.mode == "offset":
            y = y[1:, :]
        elif self.mode == "materialized":
            y = y + 1
        return x @ y


for dtype in (torch.float32, torch.float64):
    for mode in (
        "direct",
        "external_strided",
        "strided",
        "offset",
        "materialized",
    ):
        torch.manual_seed(0)
        x = torch.randn(3, 4 if mode == "offset" else 5, dtype=dtype)
        # Exercise both a 32-element vector and its scalar tail when eligible.
        y = torch.randn(5, 37, dtype=dtype)
        if mode == "external_strided":
            # The JIT adapter must fulfill its contiguous-input ABI for callers.
            x = torch.randn(3, 10, dtype=dtype)[:, ::2]
            y = torch.randn(10, 74, dtype=dtype)[::2, ::2]
        model = MatmulCase(mode)
        exported = torch.export.export(model, (x, y), strict=True)
        compiler = DynamoCompiler(
            primary_registry=tosa.ops_registry, enable_external_calls=False
        )
        compiler._compile_fx(exported.graph_module, [x, y])
        actual = compiler.dynamo_run()(x, y)
        torch.testing.assert_close(actual[0], model(x, y), rtol=1e-4, atol=1e-5)
        module = str(compiler.imported_graphs[0]._imported_module)
        if mode != "strided":
            # Direct inputs and row-offset views retain the vectorized path.
            assert "vector<32x" in module, module
        else:
            assert "vector<32x" not in module, module
        count += 1
print(f"Slice/select and matmul layouts: {count} CPU JIT cases passed")
