# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp


def foo(x):
    """
    Test resize operation with enlarging.
    resize(tensor, size) resizes the tensor to the specified size.
    When enlarging, new elements are filled with zeros.
    """
    # Use the out-of-place torch.ops.aten.resize directly: the in-place
    # torch.ops.aten.resize_ can no longer enlarge a tensor under torch 2.15,
    # where AOT autograd functionalization rejects resizing a view tensor to a
    # larger size before the graph reaches the importer. Both variants lower
    # through the same ResizeOp handler.
    return torch.ops.aten.resize(x.clone(), [3, 3])


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
