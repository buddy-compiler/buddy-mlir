# RUN: %PYTHON %s
"""Pixel rearrangements preserve exact values across ranks and strided views."""

import math

import torch
from buddy.compiler.export import export
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


class Rearrange(torch.nn.Module):
    def __init__(self, inverse, factor, sliced):
        super().__init__()
        self.inverse = inverse
        self.factor = factor
        self.sliced = sliced

    def forward(self, x):
        value = x[..., 1::2] if self.sliced else x
        op = (
            torch.ops.aten.pixel_unshuffle.default
            if self.inverse
            else torch.ops.aten.pixel_shuffle.default
        )
        result = op(value, self.factor)
        return result, result.transpose(-1, -2), x


def check(shape, dtype, inverse, factor, sliced):
    shape = list(shape)
    if sliced:
        shape[-1] = shape[-1] * 2 + 1
    x = (torch.arange(math.prod(shape)).reshape(shape) % 61 - 30).to(dtype)
    original = x.clone()
    model = Rearrange(inverse, factor, sliced)
    expected = model(x)
    exported = export(model, (x,), strict=True)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry, enable_external_calls=False
    )
    compiler._compile_fx(exported.graph_module, [x])
    actual = compiler.dynamo_run()(x)
    assert len(actual) == len(expected) == 3
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(
            result,
            reference,
            rtol=0,
            atol=0,
            msg=f"shape={shape}, dtype={dtype}, inverse={inverse}, factor={factor}, sliced={sliced}",
        )
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    return True


torch.set_num_threads(1)
count = 0
for dtype in (torch.float32, torch.float64, torch.int32, torch.int64):
    for inverse in (False, True):
        for factor in (1, 2, 3):
            spatial = (
                (2, 2 * factor, 3 * factor)
                if inverse
                else (2 * factor * factor, 2, 3)
            )
            for batch in ((), (2,), (2, 1), (0,)):
                for sliced in (False, True):
                    check(batch + spatial, dtype, inverse, factor, sliced)
                    count += 1
for dtype in (torch.float32, torch.int64):
    for shape in ((2, 0, 4, 6), (2, 2, 0, 6), (2, 2, 4, 0)):
        check(shape, dtype, True, 2, False)
        count += 1

# The raw PyTorch export path must not silently accept contradictory metadata.
x = torch.empty((0, 2, 4, 6))
model = Rearrange(True, 2, False)
raw = torch.export.export(model, (x,), strict=True)
compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry, enable_external_calls=False
)
compiler._compile_fx(raw.graph_module, [x])
try:
    compiler.dynamo_run()
except ValueError as error:
    assert "output metadata mismatch" in str(error)
else:
    raise AssertionError("Conflicting raw metadata was accepted")

for shape, factor in (((0, 2, 3, 4), 2), ((0, 2, 4, 4), 0), ((0, 4), 2)):
    try:
        export(
            Rearrange(True, factor, False), (torch.empty(shape),), strict=True
        )
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("Invalid empty input was accepted")
print(
    f"Pixel rearrangements: {count} numerical cases and 4 metadata/input guards passed"
)
