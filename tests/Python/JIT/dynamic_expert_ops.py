# RUN: %PYTHON %s
"""Runtime selection, strided reshape, expert GEMM and sequential accumulation."""

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


class Expert(torch.nn.Module):
    def __init__(self, copy=False):
        super().__init__()
        self.copy = copy

    def forward(self, x, selected, weights):
        ids = torch.nonzero(selected).unbind(1)[0]
        rows = x[ids].transpose(0, 1).reshape(-1, 4)
        activated = torch.nn.functional.silu(rows @ weights)
        restored = activated.reshape(4, -1).transpose(0, 1)
        base = torch.zeros_like(x)
        if self.copy:
            return torch.index_copy(base, 0, ids, restored)
        return torch.index_add(base, 0, ids, restored)


torch.manual_seed(0)
torch.set_num_threads(1)
count = 0
for dtype in (torch.float32, torch.float64):
    for copy in (False, True):
        model = Expert(copy)
        x = torch.randn(6, 4, dtype=dtype)
        weights = torch.randn(4, 4, dtype=dtype)
        mask = torch.tensor([True, False, True, False, True, False])
        ep = torch.export.export(model, (x, mask, weights), strict=True)
        compiler = DynamoCompiler(
            primary_registry=tosa.ops_registry, enable_external_calls=False
        )
        compiler._compile_fx(
            ep.graph_module,
            [x, mask, weights],
            tracing_inputs=[
                n.meta["val"] for n in ep.graph.nodes if n.op == "placeholder"
            ],
        )
        execute = compiler.dynamo_run()
        for mask in (
            torch.zeros(6, dtype=torch.bool),
            torch.ones(6, dtype=torch.bool),
            torch.tensor([False, True, False, False, False, False]),
            torch.tensor([True, False, True, True, False, True]),
        ):
            for scale in (0.0, 1.0, -2.0):
                args = (x * scale, mask, weights)
                original = args[0].clone()
                torch.testing.assert_close(
                    execute(*args)[0], model(*args), rtol=1e-4, atol=1e-5
                )
                torch.testing.assert_close(args[0], original, rtol=0, atol=0)
                count += 1
print(f"Dynamic expert operations: {count} comparisons passed")


class Gram(torch.nn.Module):
    def forward(self, x, selected):
        rows = x[torch.nonzero(selected).unbind(1)[0]]
        return rows.transpose(0, 1) @ rows, rows @ rows.transpose(0, 1)


# Exercise both a runtime contraction size and two runtime output dimensions.
for dtype in (torch.float32, torch.float64):
    model = Gram()
    x = torch.randn(6, 4, dtype=dtype)
    mask = torch.ones(6, dtype=torch.bool)
    ep = torch.export.export(model, (x, mask), strict=True)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry, enable_external_calls=False
    )
    compiler._compile_fx(
        ep.graph_module,
        [x, mask],
        tracing_inputs=[
            n.meta["val"] for n in ep.graph.nodes if n.op == "placeholder"
        ],
    )
    execute = compiler.dynamo_run()
    for length in (0, 1, 4, 6):
        mask = torch.arange(6) < length
        actual = execute(x, mask)
        expected = model(x, mask)
        assert len(actual) == len(expected)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=1e-4, atol=1e-5)
print("Dynamic matmul: 8 input sets passed")
