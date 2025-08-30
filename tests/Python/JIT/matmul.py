# RUN: %PYTHON %s | FileCheck %s

import torch
from torch._inductor.decomposition import decompositions as inductor_decomp
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


class MatrixMultiply(torch.nn.Module):
    def forward(self, a, b):
        return torch.matmul(a, b)


@run
def test_matrix_multiply():
    # Initialize the dynamo compiler.
    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp
    )

    a = torch.rand([2048, 2048], dtype=torch.float32)
    b = torch.rand([2048, 2048], dtype=torch.float32)

    dynamo_compiler.importer_by_export(MatrixMultiply(), a, b)
    exec_func = dynamo_compiler.dynamo_run()

    actual = exec_func(a, b)[0]
    expect = MatrixMultiply().forward(a, b)

    # CHECK: Is MLIR equal to Torch? True
    print(f"Is MLIR equal to Torch? {torch.allclose(actual, expect, atol=1e-05, rtol=1e-05)}")
