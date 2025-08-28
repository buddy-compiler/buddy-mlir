# RUN: %PYTHON %s | FileCheck %s
import time

import torch
from torch._inductor.decomposition import decompositions as inductor_decomp
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


class MatrixMultiply(torch.nn.Module):
    def forward(self, a, b):
        return torch.matmul(a, b)


def execute(a, b):
    # Initialize the dynamo compiler.
    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp
    )
    dynamo_compiler.importer_by_export(MatrixMultiply(), a, b)
    exec_func = dynamo_compiler.dynamo_run()

    # Return is a tensor list
    return exec_func(a, b)[0]

c = torch.rand(2048, 2048, dtype=torch.float32)
d = torch.rand(2048, 2048, dtype=torch.float32)

start_time = time.process_time()
actual = execute(c, d)
end_time = time.process_time()
mlir_time = end_time - start_time

start_time = time.process_time()
expect = MatrixMultiply().forward(c, d)
end_time = time.process_time()
torch_time = end_time - start_time

# CHECK: Is MLIR equal to Torch? True
print(f"Is MLIR equal to Torch? {torch.allclose(actual, expect, atol=1e-03, rtol=1e-03)}")
print(f"MLIR time: {mlir_time * 1000:.2f}ms, Torch time: {torch_time * 1000:.2f}ms")
