# RUN: %PYTHON %s
"""Checked one-hot, runtime class extents and invalid-label CPU regressions."""

import signal
import subprocess
import sys

import torch
from buddy.compiler.export import export
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


class Model(torch.nn.Module):
    def __init__(self, classes, view=False):
        super().__init__()
        self.classes, self.view = classes, view

    def forward(self, x):
        if self.view:
            x = x[::2]
        return torch.nn.functional.one_hot(x, self.classes)


def compile_model(model, x):
    ep = export(model, (x,), strict=True)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry, enable_external_calls=False
    )
    compiler._compile_fx(
        ep.graph_module,
        [x],
        tracing_inputs=[
            n.meta["val"] for n in ep.graph.nodes if n.op == "placeholder"
        ],
    )
    return compiler.dynamo_run()


torch.set_num_threads(1)
if len(sys.argv) > 1:
    execute = compile_model(Model(4), torch.tensor([0, 1, 2]))
    print("Executing invalid class", flush=True)
    execute(torch.tensor([0, int(sys.argv[1]), 2]))
    raise AssertionError("Invalid class accepted")

count = 0
for classes in (-1, 4):
    for shape in ((), (3,), (2, 3), (0,), (2, 0)):
        if classes == -1 and 0 in shape:
            continue
        model = Model(classes)
        execute = compile_model(model, torch.zeros(shape, dtype=torch.int64))
        for label in (0, 2, 3 if classes == 4 else 7):
            x = torch.full(shape, label, dtype=torch.int64)
            torch.testing.assert_close(execute(x)[0], model(x), rtol=0, atol=0)
            count += 1
for classes in (-1, 4):
    model = Model(classes, True)
    execute = compile_model(model, torch.arange(6).remainder(3))
    for x in (torch.arange(6).remainder(3), torch.zeros(6, dtype=torch.int64)):
        torch.testing.assert_close(execute(x)[0], model(x), rtol=0, atol=0)
        count += 1
for label in (-1, 4):
    result = subprocess.run(
        [sys.executable, __file__, str(label)], capture_output=True, text=True
    )
    assert result.returncode == -signal.SIGABRT, result.stderr
    assert "Executing invalid class" in result.stdout
print(
    f"Checked one_hot: {count} comparisons and 2 runtime guards passed",
    flush=True,
)
