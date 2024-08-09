# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from torch._inductor.decomposition import decompositions as inductor_decomp
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa

def foo(x, y):
    return torch.matmul(x, y)

# 创建输入张量
in1 = torch.ones([13, 13], dtype=torch.float32)
in2 = torch.full([13, 13], 2.0, dtype=torch.float32)  # 每个元素为2.0的张量

# 初始化Dynamo编译器
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

# 使用Dynamo编译器编译foo函数
foo_mlir = torch.compile(foo, backend=dynamo_compiler)

# 验证MLIR编译结果与PyTorch结果一致
assert torch.allclose(
    foo_mlir(in1, in2), foo(in1, in2), equal_nan=True
)

# 导入计算图并转换为顶层IR
graphs = dynamo_compiler.importer(foo, in1, in2)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()

# 打印导入的模块
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = linalg.matmul
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
