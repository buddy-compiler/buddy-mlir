# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp
from torch._functorch.aot_autograd import aot_autograd_decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import linalg
from buddy.compiler.graph.transform import simply_fuse, apply_classic_fusion

def foo(m1, m2,map):
    tmp = torch.ops.aten.permute(m2,map)
    return torch.matmul(m1,tmp)

m1 = torch.ones([3, 4], dtype=torch.float32)
m2 = torch.ones([3, 4], dtype=torch.float32)
map = (1,0)
# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=linalg.ops_registry,
    aot_autograd_decomposition=aot_autograd_decompositions,
)

graphs = dynamo_compiler.importer(foo, m1,m2,map)
assert len(graphs) == 1
graph = graphs[0]
pattern_list = [apply_classic_fusion]
graphs[0].fuse_ops(pattern_list)

graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = arith.constant
# CHECK: %{{.*}} = linalg.matmul_transpose_b
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
