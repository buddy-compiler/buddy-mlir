# RUN: %PYTHON %s

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph.operation import MatmulOp, PermuteOp, TOp
from buddy.compiler.graph.transform.fuse_ops import (
    decompose_addmm_to_mm_add,
    transpose_matmul_fusion,
)
from buddy.compiler.ops import tosa


class BufferedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.buffer_weight = torch.nn.Parameter(torch.randn(11, 17))
        self.register_buffer("bias", torch.randn(11))

    def forward(self, image):
        return torch.nn.functional.linear(image, self.buffer_weight, self.bias)


torch.manual_seed(0)
torch.set_num_threads(4)
for fused in (False, True):
    for batch, features, outputs in ((1, 256, 120), (3, 17, 11)):
        model = torch.nn.Linear(features, outputs).eval()
        image = torch.randn(batch, features)
        compiler = DynamoCompiler(
            primary_registry=tosa.ops_registry, trace=None
        )
        with torch.no_grad():
            (graph,) = compiler.importer(model, image)
        params = [value.detach() for value in compiler.imported_params[graph]]
        assert len(params) == 2
        assert len(graph.inputs_shapes) == 1
        assert torch.equal(params[0], model.weight)
        assert torch.equal(params[1], model.bias)
        if fused:
            for node in [node for node in graph.body if isinstance(node, TOp)]:
                permute = PermuteOp()
                permute.name = node.name
                graph.displace_node(node, permute)
                permute.add_argument([1, 0])
            decompose_addmm_to_mm_add(graph)
            for node in [
                node for node in graph.body if isinstance(node, MatmulOp)
            ]:
                transpose_matmul_fusion(
                    graph,
                    node,
                    graph.node_table[node.args[1]],
                    [graph.node_table[parent] for parent in node.parents],
                    "transpose_matmul_fusion",
                )
        run = compiler._dynamo_run_for_graph(graph)
        try:
            run(image)
        except ValueError as error:
            assert "expects 3 inputs, got 1" in str(error)
        else:
            raise AssertionError(
                "Missing graph parameters must raise ValueError"
            )
        for _ in range(3):
            image = torch.randn(batch, features)
            actual = run(*params, image)[0]
            with torch.no_grad():
                expected = model(image)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

model = BufferedLinear().eval()
image = torch.randn(2, 17)
compiler = DynamoCompiler(primary_registry=tosa.ops_registry, trace=None)
with torch.no_grad():
    (graph,) = compiler.importer(model, image)
params = [value.detach() for value in compiler.imported_params[graph]]
assert len(params) == 2
assert len(graph.inputs_shapes) == 1
assert torch.equal(params[0], model.buffer_weight)
assert torch.equal(params[1], model.bias)
actual = compiler._dynamo_run_for_graph(graph)(*params, image)[0]
with torch.no_grad():
    expected = model(image)
torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
