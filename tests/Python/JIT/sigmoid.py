# RUN: %PYTHON %s | FileCheck %s

import torch
from buddy.compiler.frontend import dynamo_compiler


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


class Sigmoid(torch.nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)


@run
def test_sigmoid():
    torch.manual_seed(0)

    x = torch.randn((1024,), dtype=torch.float32)

    model = Sigmoid()
    compiled = torch.compile(model, backend=dynamo_compiler)

    actual = compiled(x)
    expect = model(x)

    # CHECK: Is MLIR equal to Torch? True
    print(
        f"Is MLIR equal to Torch? {torch.allclose(actual, expect, atol=1e-05, rtol=1e-05)}"
    )
    assert torch.allclose(actual, expect, atol=1e-05, rtol=1e-05)
