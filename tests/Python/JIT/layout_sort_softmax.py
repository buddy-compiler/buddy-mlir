# RUN: %PYTHON %s
"""Logical reshape/copy, sort axes and softmax dtype/axis regressions."""

import math

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from torch.utils._pytree import tree_flatten


class Case(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x), x


def check(fn, x):
    original = x.clone()
    model = Case(fn)
    expected = tree_flatten(model(x))[0]
    exported = torch.export.export(model, (x,), strict=True)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry, enable_external_calls=False
    )
    compiler._compile_fx(exported.graph_module, [x])
    actual = tree_flatten(compiler.dynamo_run()(x))[0]
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5, equal_nan=True)
    torch.testing.assert_close(x, original, rtol=0, atol=0, equal_nan=True)


torch.set_num_threads(1)
count = 0
for dtype in (torch.float32, torch.float64, torch.int32, torch.int64):
    x = torch.arange(30, dtype=dtype).reshape(5, 6)
    for fn in (
        lambda a: a.T.reshape(3, 10),
        lambda a: a[:, 1::2].reshape(-1),
        lambda a: a.T.contiguous().reshape(-1),
        lambda a: a[:, 1::2].contiguous().reshape(-1),
        lambda a: a[:, :0].reshape(0, 2),
        lambda a: a[:, :0].contiguous(),
        lambda a: a[2, 3].reshape(1, 1),
        lambda a: a[2:3, 3:4].reshape(()),
    ):
        check(fn, x)
        count += 1
    for shape in ((7,), (4, 7), (2, 3, 5), (0, 3), ()):
        size = math.prod(shape)
        x = torch.arange(size, dtype=dtype).reshape(shape)
        x = ((x * 7) % max(size, 1)) - 3
        for dim in range(-max(len(shape), 1), max(len(shape), 1)):
            for descending in (False, True):
                check(
                    lambda a, d=dim, desc=descending: torch.sort(
                        a, dim=d, descending=desc, stable=True
                    ),
                    x,
                )
                count += 1
    x = torch.arange(30, dtype=dtype).reshape(5, 6)
    for dim in (0, 1, -1):
        for descending in (False, True):
            check(
                lambda a, d=dim, desc=descending: torch.argsort(
                    a.T, dim=d, descending=desc
                ),
                x,
            )
            count += 1

for dtype in (torch.float32, torch.float64):
    x = torch.tensor(
        [
            [float("nan"), -1.0, 3.0, float("inf")],
            [2.0, 2.0, -float("inf"), 0.0],
        ],
        dtype=dtype,
    )
    for dim in (0, 1):
        for descending in (False, True):
            check(
                lambda a, d=dim, desc=descending: torch.sort(
                    a, dim=d, descending=desc, stable=True
                ),
                x,
            )
            count += 1
    x = torch.arange(30, dtype=dtype).reshape(5, 6).sin() * 100
    for dim in (0, 1, -1):
        for target in (None, torch.float32, torch.float64):
            check(
                lambda a, d=dim, t=target: torch.softmax(a.T, dim=d, dtype=t), x
            )
            count += 1
print(f"Layout, sort and softmax: {count} CPU cases passed")
