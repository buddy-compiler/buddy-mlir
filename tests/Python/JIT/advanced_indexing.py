# RUN: %PYTHON %s
"""Advanced indexing: broadcast placement, negative indices and dynamic masks."""

import signal
import subprocess
import sys

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


class Gather(torch.nn.Module):
    def __init__(self, positions, view=False):
        super().__init__()
        self.positions = positions
        self.view = view

    def forward(self, x, *indices):
        if self.view:
            x = x[..., ::2]
        selection = [None if i is None else indices[i] for i in self.positions]
        return torch.ops.aten.index.Tensor(x, selection)


def compile_model(model, args):
    ep = torch.export.export(model, args, strict=True)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry, enable_external_calls=False
    )
    compiler._compile_fx(
        ep.graph_module,
        list(args),
        tracing_inputs=[
            n.meta["val"] for n in ep.graph.nodes if n.op == "placeholder"
        ],
    )
    return compiler.dynamo_run()


def check(model, args, variants=()):
    run = compile_model(model, args)
    for values in (args, *variants):
        originals = [value.clone() for value in values]
        actual = run(*values)
        assert len(actual) == 1
        torch.testing.assert_close(actual[0], model(*values), rtol=0, atol=0)
        for value, original in zip(values, originals):
            torch.testing.assert_close(value, original, rtol=0, atol=0)


class Put(torch.nn.Module):
    def __init__(self, positions, accumulate, view=False):
        super().__init__()
        self.positions = positions
        self.accumulate = accumulate
        self.view = view

    def forward(self, x, values, *indices):
        if self.view:
            x = x[..., ::2]
        selection = [None if i is None else indices[i] for i in self.positions]
        return torch.ops.aten.index_put.default(
            x, selection, values, self.accumulate
        )


if len(sys.argv) > 1:
    x = torch.zeros(3, 4)
    index = torch.tensor([0])
    is_put = sys.argv[1] == "put"
    model = Put((0,), True) if is_put else Gather((0,))
    args = (x, torch.tensor(1.0), index) if is_put else (x, index)
    run = compile_model(model, args)
    print("Executing invalid index", flush=True)
    args[-1].fill_(int(sys.argv[2]))
    run(*args)
    raise AssertionError("Out-of-bounds index was accepted")


torch.set_num_threads(1)
count = 0
for dtype in (torch.float32, torch.float64, torch.int64):
    for index_dtype in (torch.int32, torch.int64):
        for view in (False, True):
            x = torch.arange(
                3 * 4 * 5 * (2 if view else 1), dtype=dtype
            ).reshape(3, 4, -1)
            for positions, indices in (
                ((0,), ([0, -1, 1],)),
                ((None, 0), ([0, -1, 1],)),
                ((None, None, 0), ([0, -1, 1],)),
                ((0, None, 1), ([[0], [-1]], [[0, 2, -1]])),
                ((None, 0, 1), ([[0], [-1]], [[0, 2, -1]])),
                ((0, 1), (1, [0, -1])),
                ((0, None, 1), (1, [0, -1])),
                ((None, 0), ([],)),
            ):
                args = (
                    x,
                    *(torch.tensor(i, dtype=index_dtype) for i in indices),
                )
                check(Gather(positions, view), args)
                count += 1
print(f"Integer advanced indexing: {count} cases passed", flush=True)
count = 0
for dtype in (torch.float32, torch.float64, torch.int64):
    x = torch.arange(60, dtype=dtype).reshape(3, 4, 5)
    for positions, mask_shape in (
        ((0,), (3,)),
        ((None, 0), (4,)),
        ((0,), (3, 4)),
        ((None, 0), (4, 5)),
    ):
        mask = torch.ones(mask_shape, dtype=torch.bool)
        mixed = (
            torch.arange(mask.numel()).reshape(mask_shape).remainder(2).bool()
        )
        check(
            Gather(positions),
            (x, mask),
            ((x, torch.zeros_like(mask)), (x, mixed)),
        )
        count += 3
print(
    f"Boolean advanced indexing: {count} reused-graph cases passed", flush=True
)

count = 0
for dtype in (torch.float32, torch.float64, torch.int64, torch.bool):
    for accumulate in (False, True):
        for view in (False, True):
            x = (
                torch.arange(3 * 4 * 5 * (2 if view else 1))
                .reshape(3, 4, -1)
                .to(dtype)
            )
            for positions, raw in (
                ((0,), ([0, -1],)),
                ((None, 0), ([0, -1],)),
                ((0, None, 1), ([[0], [-1]], [[0, 2, -1]])),
                ((None, 0, 1), ([[0], [-1]], [[0, 2, -1]])),
                ((0, 1), (1, [0, -1])),
                ((None, 0), ([],)),
            ):
                indices = tuple(torch.tensor(i, dtype=torch.int64) for i in raw)
                selected = Gather(positions, view)(x, *indices)
                for values in (
                    torch.tensor(2).to(dtype),
                    torch.ones_like(selected),
                ):
                    check(
                        Put(positions, accumulate, view), (x, values, *indices)
                    )
                    count += 1
            if accumulate:
                check(
                    Put((0,), True, view),
                    (x, torch.tensor(2).to(dtype), torch.tensor([0, 0, -1])),
                )
                count += 1
print(f"Integer advanced updates: {count} cases passed", flush=True)
count = 0
for dtype in (torch.float32, torch.float64, torch.int64, torch.bool):
    for accumulate in (False, True):
        x = torch.arange(60).reshape(3, 4, 5).to(dtype)
        value = torch.tensor(2).to(dtype)
        for positions, mask_shape in (
            ((0,), (3,)),
            ((None, 0), (4,)),
            ((0,), (3, 4)),
        ):
            mask = torch.ones(mask_shape, dtype=torch.bool)
            mixed = (
                torch.arange(mask.numel())
                .reshape(mask_shape)
                .remainder(2)
                .bool()
            )
            check(
                Put(positions, accumulate),
                (x, value, mask),
                ((x, value, torch.zeros_like(mask)), (x, value, mixed)),
            )
            count += 3
print(
    f"Boolean advanced updates: {count} reused-graph cases passed", flush=True
)
for mode in ("gather", "put"):
    for invalid in (-4, 3):
        result = subprocess.run(
            [sys.executable, __file__, mode, str(invalid)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == -signal.SIGABRT, result.stderr
        assert "Executing invalid index" in result.stdout
print("Advanced indexing: 4 runtime bounds checks passed", flush=True)


class MixedMask(torch.nn.Module):
    def __init__(self, update=False):
        super().__init__()
        self.update = update

    def forward(self, x, mask, index):
        mask = mask[::2]
        if self.update:
            return torch.ops.aten.index_put.default(
                x, [mask, None, index], torch.full((), 2.0, dtype=x.dtype), True
            )
        return torch.ops.aten.index.Tensor(x, [mask, None, index])


for update in (False, True):
    x = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5)
    mask = torch.tensor([True, False, False, True, True, False])
    index = torch.tensor([[0], [-1]])
    check(
        MixedMask(update),
        (x, mask, index),
        ((x, torch.zeros_like(mask), index), (x, torch.ones_like(mask), index)),
    )
print(
    "Mixed strided boolean/integer indices: 6 reused-graph cases passed",
    flush=True,
)
