# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph.transform.quantization import w8a8_channel_wise
from buddy.compiler.graph.operation import QuantizedMatmulOp
from buddy.compiler.ops import tosa

torch.manual_seed(42)


class TinyMatMul(torch.nn.Module):
    def __init__(self, K, N):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(K, N))

    def forward(self, x):
        return torch.matmul(x, self.weight)


model = TinyMatMul(K=17, N=13).eval()
x = torch.randn(3, 17)

dynamo = DynamoCompiler(primary_registry=tosa.ops_registry, func_name="forward")
with torch.no_grad():
    graphs = dynamo.importer(model, x)

assert len(graphs) == 1
graph = graphs[0]

w8a8_channel_wise(graph)
assert any(isinstance(op, QuantizedMatmulOp) for op in graph.body), (
    "QuantizedMatmulOp not found in the graph"
)

graph.lower_to_top_level_ir()
print(graph._imported_module)
# CHECK-LABEL: func.func @forward
# CHECK: tosa.reduce_max %{{.*}} {axis = 0 : i32}
# CHECK: tosa.reduce_max %{{.*}} {axis = 1 : i32}

W = model.weight.data
amax = W.abs().amax(dim=0, keepdim=True)
s_W = (amax / 127.0).clamp(min=1e-10)
W_i8 = torch.clamp(torch.round(W / s_W), -128, 127).to(torch.int8)

exec_fn = dynamo.dynamo_run()
out = exec_fn(W_i8, s_W.float(), x)[0]

out_fp32 = x @ W

amax_a = x.abs().max()
s_A = max(amax_a / 127.0, 1e-10)
x_i8 = torch.clamp(torch.round(x / s_A), -128, 127).to(torch.int8)
out_sim = (x_i8.float() @ W_i8.float()) * s_A * s_W

assert (out - out_sim).abs().max().item() < 1e-5, (
    f"max |compiled - sim| = {(out - out_sim).abs().max().item()}"
)
