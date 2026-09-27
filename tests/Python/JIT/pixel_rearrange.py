# RUN: %PYTHON %s
"""Pixel rearrangements preserve exact values across ranks and strided views."""

import math

import torch
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
        return op(value, self.factor), x


def check(shape, dtype, inverse, factor, sliced):
    shape = list(shape)
    if sliced:
        shape[-1] = shape[-1] * 2 + 1
    x = (torch.arange(math.prod(shape)).reshape(shape) % 61 - 30).to(dtype)
    original = x.clone()
    model = Rearrange(inverse, factor, sliced)
    expected = model(x)
    exported = torch.export.export(model, (x,), strict=True)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry, enable_external_calls=False
    )
    compiler._compile_fx(exported.graph_module, [x])
    if inverse and 0 in shape:
        try:
            compiler.dynamo_run()
        except NotImplementedError as error:
            assert "Empty pixel unshuffle" in str(error)
        else:
            raise AssertionError(
                "Empty pixel unshuffle must reject inconsistent metadata"
            )
        return False
    actual = compiler.dynamo_run()(x)
    assert len(actual) == len(expected) == 2
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
guards = 0
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
                    if check(batch + spatial, dtype, inverse, factor, sliced):
                        count += 1
                    else:
                        guards += 1
print(
    f"Pixel rearrangements: {count} cases and {guards} empty-input guards passed"
)
