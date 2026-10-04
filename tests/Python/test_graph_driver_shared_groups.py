# RUN: %PYTHON %s
import importlib.util
from pathlib import Path

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.ops import tosa


class Shared(torch.nn.Module):
    def forward(self, x):
        value = x + x
        return value * 2 + value * 3


with torch.no_grad():
    graph = DynamoCompiler(primary_registry=tosa.ops_registry).importer(
        Shared(), torch.ones(2)
    )[0]
ops = {n.name: n for n in graph.body}
print("ops", [(n.name, type(n).__name__) for n in graph.body])
graph.op_groups = {
    "subgraph0": [ops["add"]],
    "subgraph1": [ops["mul"]],
    "subgraph2": [ops["mul_1"], ops["add_1"]],
}
graph.group_map_device = dict.fromkeys(graph.op_groups, DeviceType.CPU)
path = (
    Path(__file__).resolve().parents[2]
    / "frontend/Python/graph/graph_driver.py"
)
spec = importlib.util.spec_from_file_location(
    "buddy.compiler.graph.driver_test", path
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
driver = m.GraphDriver(graph)
assert driver._subgraphs_outputs["subgraph0"] == ["add"]
assert len(driver._subgraphs_inputs["subgraph0"]) == 1
for sub in driver.subgraphs:
    sub.lower_to_top_level_ir()
    assert sub._imported_module.operation.verify()
assert driver.construct_main_graph(True).operation.verify()
print("Shared-input and shared-output graph grouping checks passed")
