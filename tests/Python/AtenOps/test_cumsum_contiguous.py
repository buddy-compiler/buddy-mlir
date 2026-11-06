# RUN: %PYTHON %s | FileCheck %s
"""Eligible scans read the previous output row; fallback keeps iter_args."""

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import linalg


def scan(x):
    return torch.cumsum(x, 0, dtype=x.dtype)


for label, width, dtype in (
    ("CONTIGUOUS", 64, torch.float32),
    ("SHORT", 63, torch.float32),
    ("F64", 64, torch.float64),
):
    compiler = DynamoCompiler(primary_registry=linalg.ops_registry)
    graph = compiler.importer(scan, torch.zeros(65, width, dtype=dtype))[0]
    graph.lower_to_top_level_ir()
    print(label)
    print(graph._imported_module)

# CHECK-LABEL: CONTIGUOUS
# CHECK: arith.subi
# CHECK: memref.load
# CHECK: memref.load
# CHECK: arith.addf
# CHECK-LABEL: SHORT
# CHECK: iter_args
# CHECK-LABEL: F64
# CHECK: iter_args
