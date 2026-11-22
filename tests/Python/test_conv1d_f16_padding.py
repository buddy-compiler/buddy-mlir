# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp


class Conv1d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(4, 8, 3, padding=1, bias=False).half()

    def forward(self, x):
        return self.conv(x)


compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)
with torch.no_grad():
    graph = compiler.importer(
        Conv1d(), torch.zeros(1, 4, 16, dtype=torch.float16)
    )[0]
graph.lower_to_top_level_ir()
module = graph._imported_module
assert module.operation.verify()
assert "tosa.pad" in str(module)
assert "tensor<1xf16>" in str(module)
print("FP16 Conv1d TOSA padding verifies")
# CHECK: FP16 Conv1d TOSA padding verifies
