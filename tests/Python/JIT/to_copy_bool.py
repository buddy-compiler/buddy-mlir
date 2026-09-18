# RUN: %PYTHON %s

"""Check boolean-to-integer casts and the one-hot decomposition on CPU JIT."""

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def check(module, value):
    exported = torch.export.export(module, (value,), strict=True)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry, enable_external_calls=False
    )
    compiler._compile_fx(exported.graph_module, [value])
    assert len(compiler.imported_graphs) == 1
    actual = compiler.dynamo_run()(value)
    assert len(actual) == 1
    torch.testing.assert_close(actual[0], module(value), rtol=0, atol=0)


class Cast(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, value):
        return torch.ops.aten._to_copy.default(value, dtype=self.dtype)


class OneHot(torch.nn.Module):
    def forward(self, value):
        return torch.ops.aten.one_hot.default(value, 4)


torch.set_num_threads(1)
for dtype in (torch.int8, torch.int32, torch.int64):
    for value in (
        torch.tensor(True),
        torch.tensor(False),
        torch.tensor([False, True, True, False]),
        torch.tensor([[True, False, True], [False, True, False]]),
    ):
        check(Cast(dtype), value)
for value in (torch.tensor([0, 3, 1, 0]), torch.tensor([[0, 2], [3, 1]])):
    check(OneHot(), value)
print("Boolean casts and one_hot: 14 CPU JIT cases passed")
