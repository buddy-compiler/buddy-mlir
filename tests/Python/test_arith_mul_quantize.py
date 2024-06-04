# RUN: %PYTHON %s 2>&1 | FileCheck %s
import os
import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import linalg


def foo(x, y):
    return x * y

in1 = torch.rand(1, dtype=torch.float16)
in2 = torch.rand(1, dtype=torch.float16)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=linalg.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler.importer(foo, in1, in2)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()

path_prefix = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(path_prefix, "test_mul.mlir"), "w") as module_file:
     module_file.write(str(graph._imported_module))

print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = arith.constant {{.*}} : tensor<{{.*}}xf16>
# CHECK: %{{.*}} = tosa.mul %{{.*}}, %{{.*}} {{.*}} : (tensor<{{.*}}xf16>, tensor<{{.*}}xf16>) -> tensor<{{.*}}xf16> 
# CHECK: return %{{.*}} : tensor<{{.*}}xf16>
# CHECK: }
# CHECK: }

