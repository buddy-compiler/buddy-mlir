import time

import torch
from torch._inductor.decomposition import decompositions as inductor_decomp
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


class MatrixMultiplex(torch.nn.Module):
    def forward(self, a, b):
        return torch.matmul(a, b)


def print_high_ir(a, b):
    # Initialize the dynamo compiler.
    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )
    dynamo_compiler.importer(MatrixMultiplex(), a, b)
    graph = dynamo_compiler._imported_graphs[0]
    graph.lower_to_top_level_ir()
    print(graph._imported_module)


def execute(a, b):
    # Initialize the dynamo compiler.
    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp
    )
    dynamo_compiler.importer_by_export(MatrixMultiplex(), a, b)
    exec_func = dynamo_compiler.dynamo_run()

    # Return is a tensor list
    return exec_func(a, b)[0]


a = torch.tensor([[2, 0, 3], [1, 1, 1]], dtype=torch.float32)
b = torch.tensor([[4, 4], [3, 2], [1, 0]], dtype=torch.float32)

print_high_ir(a, b)

actual = execute(a, b)
print("a:\n", a)
print("b:\n", b)
print("Actual output:\n", actual)
print("Expect output:\n", MatrixMultiplex().forward(a, b))

c = torch.rand(2048, 2048, dtype=torch.float32)
d = torch.rand(2048, 2048, dtype=torch.float32)

start_time = time.process_time()
actual = execute(c, d)
end_time = time.process_time()
mlir_time = end_time - start_time

start_time = time.process_time()
expect = MatrixMultiplex().forward(c, d)
end_time = time.process_time()
torch_time = end_time - start_time

print(f"Is MLIR equal to Torch? {torch.allclose(actual, expect, atol=1e-03, rtol=1e-03)}")
print(f"MLIR time: {mlir_time * 1000}ms, Torch time: {torch_time * 1000}ms")
