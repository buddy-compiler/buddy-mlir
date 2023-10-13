# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo

from buddy.compiler import frontend


def foo(x, y):
    return x + y


foo_mlir = dynamo.optimize(frontend.dynamo_importer)(foo)
in1 = torch.randn(10)
in2 = torch.randn(10)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = "tosa.add"
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
foo_mlir(in1, in2)
