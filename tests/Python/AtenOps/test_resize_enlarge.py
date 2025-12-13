# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(x):
    """
    Test resize_ operation with enlarging.
    resize_(tensor, size) resizes the tensor to the specified size.
    When enlarging, new elements are filled with zeros.
    """
    # Use torch.ops.aten.resize_ directly
    return torch.ops.aten.resize_(x.clone(), [3, 3])


# Input: 2x2 tensor -> resize to 3x3 (enlarging)
in1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler.importer(foo, in1)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: tosa.reshape
# CHECK: tosa.const
# CHECK: tosa.concat
# CHECK: tosa.reshape
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
