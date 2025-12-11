# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch.nn.functional as F
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)


# Define a simple function using softshrink
def softshrink_func(x):
    return F.softshrink(x, lambd=0.5)


# Generate input data
input_data = torch.randn(3, 4, dtype=torch.float32)

# Compile the function
graphs = dynamo_compiler.importer(softshrink_func, input_data)

# Print the generated MLIR module
assert len(graphs) == 1
graphs[0].lower_to_top_level_ir()
print(graphs[0]._imported_module)

# CHECK: module {
# CHECK: func.func @forward
# CHECK: tosa.abs
# CHECK: tosa.greater
# CHECK: tosa.select
# CHECK: return
# CHECK: }
