# RUN: %PYTHON %s
"""Bincount runtime extents, weighted accumulation, views and index checks."""

import signal
import subprocess
import sys

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


class Bincount(torch.nn.Module):
    def __init__(self, minimum, view=False):
        super().__init__()
        self.minimum = minimum
        self.view = view

    def forward(self, x, weights=None):
        if self.view:
            x = x[1::2]
            if weights is not None:
                weights = weights[1::2]
        return torch.bincount(x, weights, minlength=self.minimum)


def compile_model(model, args):
    exported = torch.export.export(model, args, strict=True)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry, enable_external_calls=False
    )
    compiler._compile_fx(
        exported.graph_module,
        list(args),
        tracing_inputs=[
            n.meta["val"] for n in exported.graph.nodes if n.op == "placeholder"
        ],
    )
    return compiler.dynamo_run()


torch.set_num_threads(1)
if len(sys.argv) > 1:
    execute = compile_model(Bincount(0), (torch.tensor([0, 1]),))
    print("Executing invalid bin", flush=True)
    execute(torch.tensor([0, int(sys.argv[1])]))
    raise AssertionError("Invalid bin was accepted")

count = 0
for dtype in (torch.int8, torch.int32, torch.int64):
    for length in (0, 1, 5):
        x = torch.arange(length).remainder(3).to(dtype)
        for minimum in (0, 3, 10):
            for weight_dtype in (
                None,
                torch.float32,
                torch.float64,
                torch.int64,
                torch.bool,
            ):
                weights = (
                    None
                    if weight_dtype is None
                    else (torch.arange(length) - 2).to(weight_dtype)
                )
                args = (x,) if weights is None else (x, weights)
                model = Bincount(minimum)
                execute = compile_model(model, args)
                for value in (x, torch.zeros_like(x), x + 7):
                    args = (value,) if weights is None else (value, weights)
                    originals = tuple(a.clone() for a in args)
                    actual = execute(*args)
                    assert len(actual) == 1
                    torch.testing.assert_close(
                        actual[0], model(*args), rtol=0, atol=0
                    )
                    for a, before in zip(args, originals):
                        torch.testing.assert_close(a, before, rtol=0, atol=0)
                    count += 1
    for weighted in (False, True):
        model = Bincount(2, view=True)
        x = torch.arange(10).remainder(4).to(dtype)
        weights = torch.arange(10, dtype=torch.float64) * -0.25
        args = (x, weights) if weighted else (x,)
        execute = compile_model(model, args)
        for value in (x, x + 5, torch.zeros_like(x)):
            args = (value, weights) if weighted else (value,)
            torch.testing.assert_close(
                execute(*args)[0], model(*args), rtol=0, atol=0
            )
            count += 1
print(f"Bincount: {count} numerical cases passed")

for invalid in (-1, 2**63 - 1):
    result = subprocess.run(
        [sys.executable, __file__, str(invalid)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "Executing invalid bin" in result.stdout, result.stderr
    assert result.returncode == -signal.SIGABRT, (
        result.returncode,
        result.stderr,
    )
print("Bincount: 2 runtime index checks passed")
