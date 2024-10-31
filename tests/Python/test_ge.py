# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import linalg


def foo(x, y):
    return torch.ops.aten.ge(x, y)


in1 = torch.ones([13, 5], dtype=torch.int64)
in2 = 0
# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=linalg.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler.importer(foo, in1, in2)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: "builtin.module"() ({
# CHECK-LABEL: "func.func"() <{function_type = ({{.*}} -> {{.*}}, sym_name = "forward"}
# CHECK: %{{.*}} = "arith.constant"
# CHECK: %{{.*}} = "tensor.empty"
# CHECK: %{{.*}} = "linalg.generic"
# CHECK: "func.return"(%{{.*}}) : {{.*}} -> ()
# CHECK:   }) : () -> ()
# CHECK: }) : () -> ()
