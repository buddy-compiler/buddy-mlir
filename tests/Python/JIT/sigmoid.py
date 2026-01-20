# RUN: %PYTHON %s | FileCheck %s

import torch
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


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

    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )

    x = torch.randn((1024,), dtype=torch.float32)

    dynamo_compiler.importer_by_export(Sigmoid(), x)
    exec_func = dynamo_compiler.dynamo_run()

    actual = exec_func(x)[0]
    expect = Sigmoid().forward(x)

    # CHECK: Is MLIR equal to Torch? True
    print(
        f"Is MLIR equal to Torch? {torch.allclose(actual, expect, atol=1e-05, rtol=1e-05)}"
    )
