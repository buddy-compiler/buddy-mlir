# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from buddy.compiler.frontend import DynamoCompiler

# Test arange.start_step operation
# Creates a 1D tensor with values from start to end with step
# This test uses arange in a computation context to ensure it's traced


def test_arange_start_step():
    """Test arange.start_step operator used in computation"""
    dynamo_compiler = DynamoCompiler()

    def fn(x):
        indices = torch.arange(0, 5, 1, dtype=torch.int64)  # [0, 1, 2, 3, 4]
        return x + indices.float()  # Add to input

    x = torch.zeros(5)
    graphs = dynamo_compiler.importer(fn, x)
    graph = graphs[0]
    graph.lower_to_top_level_ir()
    print(graph._imported_module)


# CHECK: module
# CHECK: func.func @forward
# CHECK: tosa.const
# CHECK: return

test_arange_start_step()
