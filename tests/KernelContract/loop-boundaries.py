# RUN: %PYTHON %s buddy-opt
# ===- loop-boundaries.py --------------------------------------------------
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
# Reject valid MLIR whose declared content history crosses a loop incorrectly.
#
# Lit Python test. Feeds syntactically valid MLIR whose declared content-version
# history crosses scf.for incorrectly (init/backedge/exit mismatch) and expects
# kernel-contract-annotate or verify to reject with the loop-boundary diagnostics.
#
# ===---------------------------------------------------------------------------

import subprocess
import sys
from pathlib import Path

tool = sys.argv[1]
reference = (
    Path(__file__).resolve().parent / "Inputs/kernel.contract.mlir"
).read_text()


def reject(before, after, rule):
    assert before in reference, before
    mutated = reference.replace(before, after, 1)
    result = subprocess.run(
        [tool, "--kernel-contract-verify", "-o", "/dev/null"],
        input=mutated,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0, f"accepted invalid loop history: {after}"
    assert rule in result.stderr, result.stderr


# A merely resolvable body version is not a valid C backedge: value 4 is A.
reject("yielded_value_id = 6", "yielded_value_id = 4", "V11/V20/V26")
# Entry initialization must be the current private C version, not external C.
reject("init_value_id = 3", "init_value_id = 2", "V20/V26")
# Body reads bind to iter, rather than the parent's one-time initializer.
reject("array<i64: 4, 5, 7>", "array<i64: 4, 5, 3>", "V22/V23")
# A parent consumer must read result, not a static body version.
reject(
    "input_values = array<i64: 8>", "input_values = array<i64: 6>", "V22/V23"
)

# A full-kill loop needs boundary WAW ordering, but has no semantic value users.
source = """
module {
  func.func @kills() {
    %zero = arith.constant 0 : i32
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    %c = memref.alloc() : memref<4xi32>
    linalg.fill ins(%zero : i32) outs(%c : memref<4xi32>)
    scf.for %i = %lb to %ub step %step {
      linalg.fill ins(%zero : i32) outs(%c : memref<4xi32>)
    }
    linalg.fill ins(%zero : i32) outs(%c : memref<4xi32>)
    return
  }
}
"""
annotated = subprocess.run(
    [tool, "--kernel-contract-annotate"],
    input=source,
    text=True,
    capture_output=True,
    check=True,
).stdout
assert "linalg.fill {" in annotated
for users, should_pass in (("array<i64>", True), ("array<i64: 1>", False)):
    modified = annotated.replace(
        "linalg.fill {", f"linalg.fill {{contract.value_users = [{users}], ", 1
    )
    result = subprocess.run(
        [tool, "--kernel-contract-verify", "-o", "/dev/null"],
        input=modified,
        text=True,
        capture_output=True,
        check=False,
    )
    assert (result.returncode == 0) == should_pass, result.stderr
    if not should_pass:
        assert "kernel contract V8" in result.stderr, result.stderr
