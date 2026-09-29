# RUN: %PYTHON %s
"""CPU numeric copy conversions, strided views and checked integer casts."""

import signal
import subprocess
import sys

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


class Copy(torch.nn.Module):
    def __init__(self, dtype, view=False):
        super().__init__()
        self.dtype = dtype
        self.view = view

    def forward(self, x):
        if self.view:
            x = x[1::2]
        return torch.ops.aten._to_copy.default(x, dtype=self.dtype)


def compile_model(model, x):
    ep = torch.export.export(model, (x,), strict=True)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry, enable_external_calls=False
    )
    compiler._compile_fx(ep.graph_module, [x])
    return compiler.dynamo_run()


def check(model, x):
    original = x.clone()
    actual = compile_model(model, x)(x)
    assert len(actual) == 1
    torch.testing.assert_close(
        actual[0], model(x), rtol=0, atol=0, equal_nan=True
    )
    actual[0].fill_(1)
    torch.testing.assert_close(x, original, rtol=0, atol=0, equal_nan=True)


torch.set_num_threads(1)
if len(sys.argv) > 1:
    run = compile_model(
        Copy(torch.int8), torch.tensor([0.0], dtype=torch.float64)
    )
    print("Executing invalid cast", flush=True)
    run(torch.tensor([float(sys.argv[1])], dtype=torch.float64))
    raise AssertionError("Invalid float-to-integer cast was accepted")

count = 0
types = (
    torch.bool,
    torch.int8,
    torch.int32,
    torch.int64,
    torch.float32,
    torch.float64,
)
for source in types:
    for target in types:
        for shape in ((), (6,), (2, 3), (0,), (2, 0)):
            x = torch.arange(torch.Size(shape).numel()).reshape(shape) - 3
            x = x.to(source)
            if source.is_floating_point:
                x += 0.75
            check(Copy(target), x)
            count += 1
        check(Copy(target, view=True), torch.arange(12).to(source))
        count += 1
for source in (torch.float32, torch.float64):
    x = torch.tensor(
        [0.0, -0.0, -0.25, 0.25, float("nan"), float("inf"), -float("inf")],
        dtype=source,
    )
    check(Copy(torch.bool), x)
    check(
        Copy(torch.int8),
        torch.tensor([-128.9, -128.0, -0.9, 0.9, 127.9], dtype=source),
    )
    count += 2
print(f"Numeric copies: {count} cases passed")

for invalid in ("nan", "inf", "128", "-129"):
    result = subprocess.run(
        [sys.executable, __file__, invalid],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "Executing invalid cast" in result.stdout, result.stderr
    assert result.returncode == -signal.SIGABRT, (
        result.returncode,
        result.stderr,
    )
print("Numeric copies: 4 runtime range checks passed")
