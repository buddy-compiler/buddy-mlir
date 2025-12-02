# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from buddy.compiler.frontend import DynamoCompiler

# Test le.Scalar operation
# le.Scalar: Less than or equal to scalar comparison


def test_le_scalar():
    """Test le.Scalar operator"""
    dynamo_compiler = DynamoCompiler()

    def fn(x):
        return x <= 0.5  # le.Scalar: compare each element with 0.5

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

test_le_scalar()
