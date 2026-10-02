# RUN: %PYTHON %s
"""CPU dtype conversion, ranges, broadcasting and strided slice updates."""

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


class Case(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


def check(fn, args, *, rtol=1e-4, atol=1e-5):
    originals = [x.clone() for x in args]
    model = Case(fn)
    expected = model(*args)
    exported = torch.export.export(model, args, strict=True)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry, enable_external_calls=False
    )
    compiler._compile_fx(exported.graph_module, list(args))
    actual = compiler.dynamo_run()(*args)[0]
    torch.testing.assert_close(
        actual, expected, rtol=rtol, atol=atol, equal_nan=True
    )
    if actual.is_floating_point():
        torch.testing.assert_close(
            torch.signbit(actual[expected == 0]),
            torch.signbit(expected[expected == 0]),
        )
    for arg, original in zip(args, originals):
        torch.testing.assert_close(
            arg, original, rtol=0, atol=0, equal_nan=True
        )


torch.set_num_threads(1)
torch.manual_seed(0)
count = 0
for source, target in (
    (torch.float32, torch.float64),
    (torch.float64, torch.float32),
):
    for x in (
        torch.tensor(
            [0.0, -0.0, 1.25, -2.75, float("nan"), float("inf")], dtype=source
        ),
        torch.tensor(1.125, dtype=source),
        torch.empty(0, 3, dtype=source),
        torch.randn(3, 8, dtype=source)[:, ::2],
    ):
        check(
            lambda x, target=target: torch.ops.aten._to_copy.default(
                x, dtype=target
            ),
            (x,),
        )
        count += 1
for dtype in (torch.float32, torch.float64, torch.int32, torch.int64):
    for start, end, step in ((2, 12, 2), (7, -2, -2), (2, 2, 1)):
        check(
            lambda start=start, end=end, step=step, dtype=dtype: (
                torch.ops.aten.arange.start_step(
                    start, end, step, dtype=dtype, device="cpu"
                )
            ),
            (),
        )
        count += 1
    if dtype.is_floating_point:
        check(
            lambda dtype=dtype: torch.ops.aten.arange.start_step(
                0.5, 3.25, 0.5, dtype=dtype, device="cpu"
            ),
            (),
        )
        count += 1
for dtype in (torch.float32, torch.float64, torch.int64, torch.bool):
    for shape, target_shape in (
        ((2, 1, 3), (2, 4, 3)),
        ((1, 3), (2, 4, 3)),
        ((), (2, 3)),
        ((0, 1), (0, 3)),
    ):
        x = torch.zeros(shape, dtype=dtype)
        values = torch.arange(x.numel()).reshape(shape)
        x = (values % 2 if dtype == torch.bool else values).to(dtype)
        if dtype.is_floating_point and x.numel():
            x.reshape(-1)[0] = -0.0
        check(
            lambda x, target_shape=target_shape: torch.ops.aten.expand.default(
                x, target_shape
            ),
            (x,),
        )
        count += 1
    x = torch.arange(15).reshape(3, 5).to(dtype)
    for dim, start, end, step in (
        (1, 1, 5, 2),
        (-1, -4, -1, 2),
        (0, 1, 3, 1),
        (1, -100, 100, 3),
        (1, 4, 2, 1),
    ):
        slices = [slice(None)] * 2
        slices[dim] = slice(start, end, step)
        selected = x[tuple(slices)]
        source = (
            ~selected if dtype == torch.bool else torch.full_like(selected, 1)
        )
        check(
            lambda a, b, dim=dim, start=start, end=end, step=step: (
                torch.ops.aten.slice_scatter.default(
                    a, b, dim, start, end, step
                )
            ),
            (x, source),
        )
        count += 1
# Early f32 rounding would duplicate or skip values near 2**24.
check(
    lambda: torch.ops.aten.arange.start_step(
        16777217, 16777225, 1, dtype=torch.float32, device="cpu"
    ),
    (),
    rtol=0,
    atol=0,
)
count += 1
print(f"Dtypes, ranges, expand and slice-scatter: {count} CPU JIT cases passed")
