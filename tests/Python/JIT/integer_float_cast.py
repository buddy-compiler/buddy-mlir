# RUN: %PYTHON %s
"""Integer/boolean casts preserve values and feed floating mask arithmetic."""

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


class Cast(torch.nn.Module):
    def __init__(self, dtype, mask=False, view=False):
        super().__init__()
        self.dtype = dtype
        self.mask = mask
        self.view = view

    def forward(self, x):
        if self.view:
            x = x[:, 1::2].T
        result = x.to(self.dtype)
        return (1.0 - result) * -10000.0 if self.mask else result


def check(model, x):
    original = x.clone()
    expected = model(x)
    exported = torch.export.export(model, (x,), strict=True)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry, enable_external_calls=False
    )
    compiler._compile_fx(exported.graph_module, [x])
    actual = compiler.dynamo_run()(x)
    assert len(actual) == 1
    torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(x, original, rtol=0, atol=0)


torch.set_num_threads(1)
count = 0
for source in (torch.bool, torch.int8, torch.int32, torch.int64):
    for target in (torch.float32, torch.float64):
        for shape in ((), (6,), (2, 3), (0,), (2, 0)):
            x = (
                torch.tensor(-1, dtype=source)
                if not shape
                else (
                    (
                        torch.arange(torch.Size(shape).numel()).reshape(shape)
                        % 7
                        - 3
                    ).to(source)
                )
            )
            check(Cast(target), x)
            count += 1
        x = torch.arange(24).reshape(3, 8).to(source)
        check(Cast(target, view=True), x)
        check(Cast(target, mask=True), torch.tensor([[1, 0, 1]], dtype=source))
        count += 2
        if source != torch.bool:
            limits = torch.iinfo(source)
            check(
                Cast(target),
                torch.tensor([limits.min, limits.max], dtype=source),
            )
            count += 1
print(f"Integer/boolean to float casts: {count} cases passed")
