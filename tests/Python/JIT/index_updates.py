# RUN: %PYTHON %s

"""CPU index updates: dimensions, repeated indices, alpha and empty inputs."""

import signal
import subprocess
import sys

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


class Update(torch.nn.Module):
    def __init__(self, mode, dim, alpha):
        super().__init__()
        self.mode, self.dim, self.alpha = mode, dim, alpha

    def forward(self, base, index, source):
        if self.mode == "add":
            return torch.ops.aten.index_add.default(
                base, self.dim, index, source, alpha=self.alpha
            )
        return torch.ops.aten.index_copy.default(base, self.dim, index, source)


def check(
    mode, shape, dim, dtype, alpha=1, empty=False, index_dtype=torch.int64
):
    base = torch.arange(torch.tensor(shape).prod().item(), dtype=dtype).reshape(
        shape
    )
    indices = (
        []
        if empty
        else ([0, 0, shape[dim] - 1] if mode == "add" else [0, shape[dim] - 1])
    )
    index = torch.tensor(indices, dtype=index_dtype)
    src_shape = list(shape)
    src_shape[dim] = len(indices)
    source = (
        torch.arange(
            torch.tensor(src_shape).prod().item(), dtype=dtype
        ).reshape(src_shape)
        + 1
    )
    args = (base, index, source)
    originals = [value.clone() for value in args]
    model = Update(mode, dim, alpha)
    exported = torch.export.export(model, args, strict=True)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry, enable_external_calls=False
    )
    compiler._compile_fx(exported.graph_module, list(args))
    assert len(compiler.imported_graphs) == 1
    actual = compiler.dynamo_run()(*args)
    assert len(actual) == 1
    torch.testing.assert_close(actual[0], model(*args), rtol=0, atol=0)
    for value, original in zip(args, originals):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


def check_invalid_index(mode, index_value):
    base = torch.zeros(3, 5)
    index = torch.tensor([0, 2])
    source = torch.ones(2, 5)
    args = (base, index, source)
    exported = torch.export.export(Update(mode, 0, 1), args, strict=True)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry, enable_external_calls=False
    )
    compiler._compile_fx(exported.graph_module, list(args))
    run = compiler.dynamo_run()
    index[0] = index_value
    print("Executing invalid index", flush=True)
    run(*args)
    raise AssertionError("Invalid index reached execution without an assertion")


torch.set_num_threads(1)
if len(sys.argv) == 4 and sys.argv[1] == "--invalid-index":
    check_invalid_index(sys.argv[2], int(sys.argv[3]))

count = 0
for dtype in (torch.float32, torch.float64, torch.int32, torch.int64):
    for mode in ("add", "copy"):
        for shape, dim in (
            ((5,), 0),
            ((3, 5), 0),
            ((3, 5), -1),
            ((2, 3, 4), 1),
        ):
            for empty in (False, True):
                check(mode, shape, dim, dtype, alpha=-2, empty=empty)
                count += 1
for alpha in (0, 0.5, -0.5):
    for dtype in (torch.float32, torch.float64):
        check("add", (3, 5), -1, dtype, alpha=alpha, index_dtype=torch.int32)
        count += 1
print(f"Index updates: {count} CPU JIT cases passed")
for mode in ("add", "copy"):
    for index_value in (-1, 3):
        child = subprocess.run(
            [
                sys.executable,
                __file__,
                "--invalid-index",
                mode,
                str(index_value),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert "Executing invalid index" in child.stdout, child.stderr
        # cf.assert lowers to abort; distinguish it from a segmentation fault.
        assert child.returncode == -signal.SIGABRT, (
            child.returncode,
            child.stdout,
            child.stderr,
        )
print("Index updates: 4 invalid-index runtime checks passed")
