# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._functorch.aot_autograd import aot_autograd_decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import linalg


def foo(query, key, value):
    return torch.nn.functional.scaled_dot_product_attention(query, key, value)


# 模拟输入 (batch=2, heads=4, seq_len=8, head_dim=16)
q = torch.randn(2, 4, 8, 16, dtype=torch.float32)
k = torch.randn(2, 4, 8, 16, dtype=torch.float32)
v = torch.randn(2, 4, 8, 16, dtype=torch.float32)

# 初始化 dynamo 编译器
dynamo_compiler = DynamoCompiler(
    primary_registry=linalg.ops_registry,
    aot_autograd_decomposition=aot_autograd_decompositions,
)

graphs = dynamo_compiler.importer(foo, q, k, v)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()

print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# === Step1: QK^T MatMul ===
# CHECK: %{{.*}} = tosa.matmul
# === Step2: Scale factor multiply ===
# CHECK: %{{.*}} = arith.constant
# CHECK: %{{.*}} = tosa.mul
# === Step3: Add attention bias ===
# CHECK: %{{.*}} = tosa.add
# === Step4: Softmax sequence ===
# CHECK: %{{.*}} = tosa.reduce_max
# CHECK: %{{.*}} = tosa.sub
# CHECK: %{{.*}} = math.exp
# CHECK: %{{.*}} = tosa.reduce_sum
# CHECK: %{{.*}} = tosa.log
# CHECK: %{{.*}} = tosa.add
# CHECK: %{{.*}} = tosa.sub
# CHECK: %{{.*}} = math.exp
# === Step5: Multiply softmax result with V ===
# CHECK: %{{.*}} = tosa.matmul
# === Step6: Reshape to final output ===
# CHECK: %{{.*}} = tosa.reshape
# CHECK: return %{{.*}}
# CHECK: }
