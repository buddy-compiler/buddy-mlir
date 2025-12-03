# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo

try:
    from torch._decomp import aot_autograd_decompositions as decomp
except ImportError:
    from torch._inductor.decomposition import decompositions as decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(x):
    return torch.ops.aten.bitwise_or.Scalar(x, 5)


in1 = torch.tensor([1, 2, 3, 4], dtype=torch.int64)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=decomp,
)

graphs = dynamo_compiler.importer(foo, in1)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = "tosa.const"
# CHECK: %{{.*}} = arith.ori
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
