# RUN: %PYTHON %s
"""Scatter reductions preserve self and handle duplicate target indices."""

import signal
import subprocess
import sys

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


class Case(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, index, src):
        result = self.fn(x, index, src)
        return result, x + 1


def check(fn, args):
    original = args[0].clone()
    model = Case(fn)
    expected = model(*args)
    exported = torch.export.export(model, args, strict=True)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry, enable_external_calls=False
    )
    compiler._compile_fx(exported.graph_module, list(args))
    actual = compiler.dynamo_run()(*args)
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, equal_nan=True)
    torch.testing.assert_close(
        args[0], original, rtol=0, atol=0, equal_nan=True
    )


torch.set_num_threads(1)
if len(sys.argv) == 2:
    args = (
        torch.zeros(3, 5),
        torch.zeros(3, 2, dtype=torch.int64),
        torch.ones(3, 2),
    )
    model = Case(
        lambda x, i, s: torch.ops.aten.scatter_reduce.two(
            x, 1, i, s, "sum", include_self=False
        )
    )
    exported = torch.export.export(model, args, strict=True)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry, enable_external_calls=False
    )
    compiler._compile_fx(exported.graph_module, list(args))
    run = compiler.dynamo_run()
    args[1][0, 0] = int(sys.argv[1])
    print("Executing invalid index", flush=True)
    run(*args)
    raise AssertionError("Invalid index did not trigger a runtime assertion")

count = 0
for dtype in (torch.float32, torch.float64, torch.int32, torch.int64):
    x = torch.arange(15).reshape(3, 5).to(dtype) - 7
    src = torch.tensor([[2, -3, 4], [5, 2, -1], [-2, 3, 2]], dtype=dtype)
    index = torch.tensor([[0, 0, 2], [3, 1, 3], [2, 2, 2]])
    for reduction in ("sum", "prod", "amax", "amin"):
        for include in (True, False):
            for empty in (False, True):
                args = (x, index[:, :0] if empty else index, src)
                check(
                    lambda x, i, s, r=reduction, inc=include: (
                        torch.ops.aten.scatter_reduce.two(
                            x, -1, i, s, r, include_self=inc
                        )
                    ),
                    args,
                )
                count += 1
    for reduction in ("add", "multiply"):
        check(
            lambda x, i, s, r=reduction: torch.ops.aten.scatter.reduce(
                x, 1, i, s, reduce=r
            ),
            (x, index, src),
        )
        check(
            lambda x, i, s, r=reduction: torch.ops.aten.scatter.value_reduce(
                x, 1, i, 2, reduce=r
            ),
            (x, index, src),
        )
        count += 2
    for reduction in ("prod", "amax", "amin"):
        for include in (True, False):
            check(
                lambda x, i, s, r=reduction, inc=include: (
                    torch.ops.aten.index_reduce.default(
                        x, 1, i, s, r, include_self=inc
                    )
                ),
                (x, torch.tensor([0, 0, 2]), src),
            )
            count += 1
    # Transposition exercises dimension zero and noncontiguous caller inputs.
    for reduction in ("sum", "prod", "amax", "amin"):
        check(
            lambda x, i, s, r=reduction: torch.ops.aten.scatter_reduce.two(
                x, 0, i, s, r, include_self=False
            ),
            (x.T, index.T, src.T),
        )
        count += 1
    if dtype.is_floating_point:
        src[0, 0] = float("nan")
        for reduction in ("sum", "prod", "amax", "amin"):
            check(
                lambda x, i, s, r=reduction: torch.ops.aten.scatter_reduce.two(
                    x, 1, i, s, r, include_self=False
                ),
                (x, index, src),
            )
            count += 1
print(f"Scatter reductions: {count} cases passed")

# Unsupported reductions must never silently fall back to summation.
model = Case(
    lambda x, i, s: torch.ops.aten.scatter_reduce.two(x, 1, i, s, "mean")
)
args = (
    torch.zeros(3, 5),
    torch.zeros(3, 2, dtype=torch.int64),
    torch.ones(3, 2),
)
exported = torch.export.export(model, args, strict=True)
compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry, enable_external_calls=False
)
compiler._compile_fx(exported.graph_module, list(args))
try:
    compiler.dynamo_run()
except NotImplementedError as error:
    assert "Unsupported scatter reduction: mean" in str(error)
else:
    raise AssertionError("Unsupported mean reduction was accepted")

for index_value in (-1, 5):
    child = subprocess.run(
        [sys.executable, __file__, str(index_value)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "Executing invalid index" in child.stdout, child.stderr
    assert child.returncode == -signal.SIGABRT, (child.returncode, child.stderr)
print("Scatter reductions: mean rejection and 2 runtime bounds checks passed")
