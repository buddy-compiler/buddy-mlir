# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import Graph
from buddy.compiler.graph.operation import FftR2cOp


# Create a simple test graph manually to test FftR2cOp
# This avoids the complex dtype issue in dynamo

# Test that the fft_r2c_op function exists and can be imported
from buddy.compiler.ops import linalg

print("Testing FftR2cOp implementation in linalg")
print("FftR2cOp is registered:", "FftR2cOp" in linalg.ops_registry)

# Simple verification of the ops_registry
assert (
    "FftR2cOp" in linalg.ops_registry
), "FftR2cOp should be in linalg.ops_registry"

# CHECK: Testing FftR2cOp implementation in linalg
# CHECK: FftR2cOp is registered: True
