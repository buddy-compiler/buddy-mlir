# RUN: %PYTHON %s buddy-opt
# ===- source_lineage.py ---------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# Source access history survives storage rewrites independently of graph IDs.
#
# Lit Python test. Source-access lineage (contract.provenance / origin lists)
# must survive storage rewrites independently of task-graph value IDs, and empty
# or negative origin lists must fail closed under annotate.
#
# ===---------------------------------------------------------------------------

import re
import subprocess
import sys
from pathlib import Path

tool = sys.argv[1]


def annotate(source):
    return subprocess.run(
        [tool, "--kernel-contract-annotate", "--kernel-contract-verify"],
        input=source,
        text=True,
        capture_output=True,
        check=True,
    ).stdout


def graph_complete(output, source_complete):
    graph = re.search(r"contract.compute_graph = \{([^}]*)\}", output)[1]
    assert re.search(r"\bcomplete = true\b", graph), graph
    expected = "true" if source_complete else "false"
    assert f"source_lineage_complete = {expected}" in graph, graph


def dot_provenance(output):
    dot = next(line for line in output.splitlines() if "linalg.matmul" in line)
    return re.search(r"contract.provenance = \{([^}]*)\}", dot)[1]


def reject_aggregate(output, actual):
    graph = re.search(r"contract.compute_graph = \{([^}]*)\}", output)
    expected = "true" if actual else "false"
    wrong = "false" if actual else "true"
    mutated = (
        output[: graph.start()]
        + graph[0].replace(
            f"source_lineage_complete = {expected}",
            f"source_lineage_complete = {wrong}",
        )
        + output[graph.end() :]
    )
    assert mutated != output
    result = subprocess.run(
        [tool, "--kernel-contract-verify"],
        input=mutated,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0, result.stdout
    assert "kernel contract V12: source lineage completeness" in result.stderr


facts = """
contract.provenance = {
  source_load_origin_ids = array<i64: 42, 75, 99>,
  a_source_load_origin_ids = array<i64: 42>,
  b_source_load_origin_ids = array<i64: 75>,
  accumulator_source_load_origin_ids = array<i64: 99>,
  a_source_lineage_complete = true, b_source_lineage_complete = true,
  accumulator_source_lineage_complete = true}
"""


def direct(facts, constant_accumulator=False):
    accumulator_arg = "" if constant_accumulator else ", %c: memref<4x4xi32>"
    initialize = ""
    if constant_accumulator:
        initialize = """
    %zero = arith.constant 0 : i32
    %c = memref.alloc() : memref<4x4xi32>
    linalg.fill ins(%zero : i32) outs(%c : memref<4x4xi32>)
"""
    comma = "," if facts else ""
    return f"""
module attributes {{
  contract.abi = {{args_aligned_bytes = 64 : i64, args_disjoint = true}}
}} {{
  func.func @direct(%a: memref<4x4xi8>, %b: memref<4x4xi8>{accumulator_arg}) {{
    {initialize}
    linalg.matmul {{cast = #linalg.type_fn<cast_signed>{comma} {facts}}}
      ins(%a, %b : memref<4x4xi8>, memref<4x4xi8>) outs(%c : memref<4x4xi32>)
    return
  }}
}}
"""


# All source snapshots were forwarded. The surviving direct operation still
# knows the three source events, whose IDs differ from its new task ID 0.
output = annotate(direct(facts))
graph_complete(output, True)
reject_aggregate(output, True)
assert "source_load_origin_ids = array<i64: 42, 75, 99>" in dot_provenance(
    output
)
assert "id = 0 : i64" in output

# An ordinary buffer-form computation can have a complete dependency graph
# while source-language events are unknown. No origin is invented.
output = annotate(direct(""))
graph_complete(output, False)
reject_aggregate(output, False)
assert "origin_id" not in output

# A known constant initialization is different from unknown provenance.
# Former accumulator facts must not replace the actual zero-fill lineage.
output = annotate(direct(facts, constant_accumulator=True))
graph_complete(output, True)
reject_aggregate(output, True)
provenance = dot_provenance(output)
assert "source_load_origin_ids = array<i64: 42, 75>" in provenance
assert "accumulator_source_load_origin_ids" not in provenance
assert "accumulator_source_lineage_complete" not in provenance

# A generated physical staging event retains the complete operand history,
# including multiple sources, without inventing a source event for the copy.
output = annotate("""
module {
  func.func @staging(%source: memref<4xi32>) {
    %snapshot = memref.alloc() : memref<4xi32>
    memref.copy %source, %snapshot {
      contract.access = {kind = "load", boundary = "external_ddr",
        mask = {kind = "none"}},
      contract.provenance = {generated_by = "test-staging",
        source_load_origin_ids = array<i64: 42, 75>, source_lineage_complete = true}}
      : memref<4xi32> to memref<4xi32>
    return
  }
}
""")
graph_complete(output, True)
reject_aggregate(output, True)
assert "source_load_origin_ids = array<i64: 42, 75>" in output
assert not re.search(r"\borigin_id\s*=", output)

# Accumulator source history follows the loop slot's initial C load, rather
# than expanding the preceding iteration's A/B arithmetic into a source set.
reference = (
    Path(__file__).resolve().parent / "Inputs/kernel.contract.mlir"
).read_text()
output = annotate(reference)
graph_complete(output, True)
for line in output.splitlines():
    if "linalg.matmul" in line:
        assert "accumulator_source_load_origin_ids = array<i64: 0>" in line
        assert "accumulator_source_lineage_complete = true" in line
