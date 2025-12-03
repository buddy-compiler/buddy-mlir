# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from buddy.compiler.frontend import DynamoCompiler

# Test argmin operation
# Returns the indices of the minimum values along a dimension


def test_argmin():
    """Test argmin operator"""
    dynamo_compiler = DynamoCompiler()

    def fn(x):
        return torch.argmin(x, dim=0)

    x = torch.randn(3, 5)
    graphs = dynamo_compiler.importer(fn, x)
    graph = graphs[0]
    graph.lower_to_top_level_ir()
    print(graph._imported_module)


# CHECK: module
# CHECK: func.func @forward
# CHECK: tosa.negate
# CHECK: tosa.argmax
# CHECK: return

test_argmin()
