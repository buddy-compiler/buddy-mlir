# RUN: %PYTHON %s | FileCheck %s

import torch
import torch.nn.functional as F
from buddy.compiler.frontend import dynamo_compiler


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


class Conv2dRelu(torch.nn.Module):
    def forward(self, x, w):
        y = F.conv2d(x, w, bias=None, stride=1, padding=1)
        return torch.relu(y)


@run
def test_conv2d_relu():
    torch.manual_seed(0)

    x = torch.randn((1, 3, 32, 32), dtype=torch.float32)
    w = torch.randn((8, 3, 3, 3), dtype=torch.float32)

    model = Conv2dRelu()
    compiled = torch.compile(model, backend=dynamo_compiler)

    actual = compiled(x, w)
    expect = model(x, w)

    # CHECK: Is MLIR equal to Torch? True
    print(
        f"Is MLIR equal to Torch? {torch.allclose(actual, expect, atol=1e-05, rtol=1e-05)}"
    )
    assert torch.allclose(actual, expect, atol=1e-05, rtol=1e-05)
