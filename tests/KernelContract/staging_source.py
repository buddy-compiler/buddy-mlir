# RUN: %PYTHON %s buddy-opt
# ===- staging_source.py ---------------------------------------------------
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
# Buddy's synthetic-transfer policy is shared by annotate and verify.
#
# Lit Python test. Buddy planner staging inserts synthetic memref copies; the
# same SourcePolicy must be shared by annotate and verify so staging cannot
# invent a missing source-language boundary event.
#
# ===---------------------------------------------------------------------------

import subprocess
import sys

source = """module {func.func @f(%source: memref<4xi32>) {
  %snapshot = memref.alloc() : memref<4xi32>
  memref.copy %source, %snapshot {
    contract.access = {kind = "load", boundary = "external_ddr",
      mask = {kind = "none"}, origin_id = 999 : i64},
    contract.provenance = {generated_by = "planner-staging",
      source_load_origin_ids = array<i64: 42>, source_lineage_complete = false}}
      : memref<4xi32> to memref<4xi32>
  return
}}"""
result = subprocess.run(
    [sys.argv[1], "--kernel-contract-annotate", "--kernel-contract-verify"],
    input=source,
    text=True,
    capture_output=True,
    check=True,
)
graph = next(
    line
    for line in result.stdout.splitlines()
    if "contract.compute_graph" in line
)
assert "source_lineage_complete = false" in graph, graph
mutated = result.stdout.replace(
    "source_lineage_complete = false", "source_lineage_complete = true", 1
)
rejected = subprocess.run(
    [sys.argv[1], "--kernel-contract-verify"],
    input=mutated,
    text=True,
    capture_output=True,
)
assert rejected.returncode != 0 and "V12" in rejected.stderr, rejected.stderr
