# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from buddy.compiler.frontend import DynamoCompiler

# Test min.dim operation
# Returns both the minimum values and their indices along a dimension


def test_min_dim():
    """Test min.dim operator"""
    dynamo_compiler = DynamoCompiler()

    def fn(x):
        min_vals, min_indices = torch.min(x, dim=0)
        return min_vals, min_indices

    x = torch.randn(3, 5)
    graphs = dynamo_compiler.importer(fn, x)
    graph = graphs[0]
    graph.lower_to_top_level_ir()
    print(graph._imported_module)


# CHECK: module
# CHECK: func.func @forward
# CHECK: tosa.reduce_min
# CHECK: tosa.negate
# CHECK: tosa.argmax
# CHECK: return

test_min_dim()
