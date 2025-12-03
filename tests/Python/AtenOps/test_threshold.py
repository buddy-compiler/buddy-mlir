# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa

# Test threshold activation function
# threshold(x, threshold, value) = x if x > threshold else value


def test_threshold():
    """Test threshold activation function"""
    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )

    def fn(x):
        return torch.nn.functional.threshold(x, 0.5, 0.0)

    x = torch.randn(5)
    graphs = dynamo_compiler.importer(fn, x)
    graph = graphs[0]
    graph.lower_to_top_level_ir()
    print(graph._imported_module)


# CHECK: func.func
# CHECK: arith.cmpf
# CHECK: tosa.select
# CHECK: return

test_threshold()
