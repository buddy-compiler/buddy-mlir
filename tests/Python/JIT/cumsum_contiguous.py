# RUN: %PYTHON %s
"""Cumsum preserves sequential rounding, signed zeros and fallback behavior."""

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


class Scan(torch.nn.Module):
    def __init__(self, dim, transpose=False):
        super().__init__()
        self.dim, self.transpose = dim, transpose

    def forward(self, x):
        if self.transpose:
            x = x.T
        return torch.cumsum(x, self.dim, dtype=x.dtype)


def sequential(x, dim):
    out = torch.empty_like(x)
    acc = torch.zeros_like(x.sum(dim=dim))
    for i in range(x.shape[dim]):
        acc = acc + x.select(dim, i)
        out.select(dim, i).copy_(acc)
    return out


torch.set_num_threads(1)
torch.manual_seed(17)
count = 0
for dtype in (torch.float32, torch.int64, torch.float64, torch.int32):
    for shape, dim, transpose in (
        ((1, 64), 0, False),
        ((65, 65), 0, False),
        ((65, 63), 0, False),
        ((65, 64), -2, False),
        ((0, 64), 0, False),
        ((64, 0), 0, False),
        ((65, 64), 1, False),
        ((2, 65, 3), 1, False),
        ((64, 65), 0, True),
    ):
        x = torch.randint(-8, 9, shape).to(dtype)
        model = Scan(dim, transpose)
        ep = torch.export.export(model, (x,), strict=True)
        compiler = DynamoCompiler(
            primary_registry=tosa.ops_registry, enable_external_calls=False
        )
        compiler._compile_fx(ep.graph_module, [x])
        run = compiler.dynamo_run()
        cases = [x]
        if dtype.is_floating_point:
            cases.append(torch.randn(shape, dtype=dtype))
            special = torch.tensor(
                [
                    float("nan"),
                    float("inf"),
                    -float("inf"),
                    -0.0,
                    0.0,
                    1e20,
                    -1e20,
                    1.0,
                ],
                dtype=dtype,
            )
            cases.append(
                special.repeat((x.numel() + 7) // 8)[: x.numel()].reshape(shape)
            )
        for value in cases:
            before = value.numpy().tobytes()
            source = value.T if transpose else value
            expected = sequential(source, dim)
            actual = run(value)[0]
            assert (
                actual.shape == expected.shape
                and actual.dtype == expected.dtype
            )
            # Framework NaN signs/payloads can differ; compare their positions.
            if dtype.is_floating_point:
                assert torch.equal(torch.isnan(actual), torch.isnan(expected))
                actual = actual.masked_fill(torch.isnan(actual), 0)
                expected = expected.masked_fill(torch.isnan(expected), 0)
            assert actual.numpy().tobytes() == expected.numpy().tobytes(), (
                dtype,
                shape,
                dim,
                transpose,
            )
            assert value.numpy().tobytes() == before
            count += 1
print(f"Cumsum contiguous: {count} cases passed")
