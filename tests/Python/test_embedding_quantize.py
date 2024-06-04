# RUN: %PYTHON %s 2>&1 | FileCheck %s
import os
import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(weight, indices):
    return torch.ops.aten.embedding(weight, indices)


# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

# test trivial case
weight = torch.randn(10, 5, dtype=torch.float16)
indices = torch.randint(10, (3, 3)).to(torch.int32)

graphs = dynamo_compiler.importer(foo, weight, indices)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
path_prefix = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(path_prefix, "test_embedding.mlir"), "w") as module_file:
     module_file.write(str(graph._imported_module))

print(graph._imported_module)


# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = tosa.reshape %{{.*}} {new_shape = array<{{.*}}>} : (tensor<{{.*}}>) -> tensor<{{.*}}>
# CHECK: %{{.*}} = tosa.reshape %{{.*}} {new_shape = array<{{.*}}>} : (tensor<{{.*}}xf16>) -> tensor<{{.*}}xf16>
# CHECK: %{{.*}} = tosa.gather %{{.*}}, %{{.*}} : (tensor<{{.*}}xf16>, tensor<{{.*}}>) -> tensor<{{.*}}xf16>
# CHECK: %{{.*}} = tosa.reshape %{{.*}} {new_shape = array<{{.*}}>} : (tensor<{{.*}}xf16>) -> tensor<{{.*}}xf16>
# CHECK: return %{{.*}} : tensor<{{.*}}xf16>
# CHECK: }
# CHECK: }

# test cast case
weight = torch.randn(10, 5, dtype=torch.float16)
indices = torch.randint(10, (3, 3)).to(torch.int64)

graphs = dynamo_compiler.importer(foo, weight, indices)
print(graphs)
assert len(graphs) == 2
graphs[0].lower_to_top_level_ir()
print(graphs[0]._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = tosa.reshape %{{.*}} {new_shape = array<{{.*}}>} : (tensor<{{.*}}>) -> tensor<{{.*}}>
# CHECK: %{{.*}} = tosa.reshape %{{.*}} {new_shape = array<{{.*}}>} : (tensor<{{.*}}xf16>) -> tensor<{{.*}}xf16>
# CHECK: %{{.*}} = tosa.gather %{{.*}}, %{{.*}} : (tensor<{{.*}}xf16>, tensor<{{.*}}>) -> tensor<{{.*}}xf16>
# CHECK: %{{.*}} = tosa.reshape %{{.*}} {new_shape = array<{{.*}}>} : (tensor<{{.*}}xf16>) -> tensor<{{.*}}xf16>
# CHECK: return %{{.*}} : tensor<{{.*}}xf16>
# CHECK: }
# CHECK: }


graphs[1].lower_to_top_level_ir()
print(graphs[1]._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = tosa.reshape %{{.*}} {new_shape = array<{{.*}}>} :(tensor<{{.*}}>) -> tensor<{{.*}}>
# CHECK: %{{.*}} = tosa.cast %{{.*}} : (tensor<{{.*}}>) -> tensor<{{.*}}>
# CHECK: %{{.*}} = tosa.reshape %{{.*}} {new_shape = array<{{.*}}>} :(tensor<{{.*}}xf16>) -> tensor<{{.*}}xf16>
# CHECK: %{{.*}} = tosa.gather %{{.*}}, %{{.*}} : (tensor<{{.*}}xf16>,tensor<{{.*}}>) -> tensor<{{.*}}xf16>
# CHECK: %{{.*}} = tosa.reshape %{{.*}} {new_shape = array<{{.*}}>} :(tensor<{{.*}}xf16>) -> tensor<{{.*}}xf16>
# CHECK: return %{{.*}} : tensor<{{.*}}xf16>
# CHECK: }
# CHECK: }
