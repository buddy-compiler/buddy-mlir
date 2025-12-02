# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from buddy.compiler.frontend import DynamoCompiler

# Test lt.Scalar operation
# lt.Scalar: Less than scalar comparison


def test_lt_scalar():
    """Test lt.Scalar operator"""
    dynamo_compiler = DynamoCompiler()

    def fn(x):
        return x < 0.5  # lt.Scalar: compare each element with 0.5

    x = torch.randn(5)
    graphs = dynamo_compiler.importer(fn, x)
    graph = graphs[0]
    graph.lower_to_top_level_ir()
    print(graph._imported_module)


# CHECK: module
# CHECK: func.func @forward
# CHECK: tosa.const
# CHECK: arith.cmpf
# CHECK: return

test_lt_scalar()
