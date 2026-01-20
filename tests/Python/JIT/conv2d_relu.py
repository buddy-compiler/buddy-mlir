# RUN: %PYTHON %s | FileCheck %s

import torch
import torch.nn.functional as F
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


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

    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )

    x = torch.randn((1, 3, 32, 32), dtype=torch.float32)
    w = torch.randn((8, 3, 3, 3), dtype=torch.float32)

    dynamo_compiler.importer_by_export(Conv2dRelu(), x, w)
    exec_func = dynamo_compiler.dynamo_run()

    actual = exec_func(x, w)[0]
    expect = Conv2dRelu().forward(x, w)

    # CHECK: Is MLIR equal to Torch? True
    print(
        f"Is MLIR equal to Torch? {torch.allclose(actual, expect, atol=1e-05, rtol=1e-05)}"
    )
