# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(size, stride, dtype, device):
    # Create an empty strided tensor
    return torch.ops.aten.empty_strided(
        size, stride, dtype=dtype, device=device
    )


# Initialize the dynamo compiler without decomposition to preserve the op
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=None,
)

# Test with typical parameters
size = [2, 3, 4]
stride = [12, 4, 1]  # Row-major strides

graphs = dynamo_compiler.importer(foo, size, stride, torch.float32, "cpu")
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module
# CHECK: func.func
# CHECK: tosa.const
# CHECK: return
