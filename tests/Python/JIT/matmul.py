# RUN: %PYTHON %s | FileCheck %s

import torch
from buddy.compiler.frontend import dynamo_compiler


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


class MatrixMultiply(torch.nn.Module):
    def forward(self, a, b):
        return torch.matmul(a, b)


@run
def test_matrix_multiply():
    torch.manual_seed(0)
    a = torch.rand([2048, 2048], dtype=torch.float32)
    b = torch.rand([2048, 2048], dtype=torch.float32)

    model = MatrixMultiply()
    compiled = torch.compile(model, backend=dynamo_compiler)

    actual = compiled(a, b)
    expect = model(a, b)

    # CHECK: Is MLIR equal to Torch? True
    print(
        f"Is MLIR equal to Torch? {torch.allclose(actual, expect, atol=1e-05, rtol=1e-05)}"
    )
    assert torch.allclose(actual, expect, atol=1e-05, rtol=1e-05)
