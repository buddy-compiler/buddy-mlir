# RUN: %PYTHON %s
"""Reuse nonzero and masked-select JIT outputs of different lengths."""

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


class Nonzero(torch.nn.Module):
    def forward(self, x, mask):
        return torch.nonzero(x), torch.masked_select(x, mask)


torch.set_num_threads(1)
count = 0
for dtype in (torch.float32, torch.float64, torch.int64, torch.bool):
    for shape in ((6,), (2, 3), (0,), (2, 0), ()):
        initial = torch.ones(shape, dtype=dtype)
        initial_mask = torch.ones(shape, dtype=torch.bool)
        model = Nonzero()
        exported = torch.export.export(
            model, (initial, initial_mask), strict=True
        )
        compiler = DynamoCompiler(
            primary_registry=tosa.ops_registry, enable_external_calls=False
        )
        compiler._compile_fx(
            exported.graph_module,
            [initial, initial_mask],
            tracing_inputs=[
                n.meta["val"]
                for n in exported.graph.nodes
                if n.op == "placeholder"
            ],
        )
        execute = compiler.dynamo_run()
        mixed = (
            torch.arange(initial.numel()).reshape(shape).remainder(2).to(dtype)
        )
        for value in (initial, torch.zeros_like(initial), mixed):
            original = value.clone()
            mask = value != 0
            actual = execute(value, mask)
            assert len(actual) == 2
            for out, expected in zip(actual, model(value, mask)):
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
            torch.testing.assert_close(value, original, rtol=0, atol=0)
            count += 1
print(
    f"Dynamic selection: {count} cases, two outputs each, across 20 compiled graphs"
)
