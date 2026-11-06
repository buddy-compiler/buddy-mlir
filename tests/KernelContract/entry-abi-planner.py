# RUN: %python %s buddy-opt
# ===- entry-abi-planner.py ------------------------------------------------
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
# Module entry ABI promises do not propagate to called/address-taken helpers.
#
# Lit Python test driving buddy-opt. Asserts that contract.abi promises attached
# to a true module entry do not leak into called or address-taken helpers when
# the planner (--bosc-ame-plan) consumes the contract. Pair with entry-abi.py for
# annotate/verify-only coverage.
#
# ===---------------------------------------------------------------------------

import re
import subprocess
import sys

TOOL = sys.argv[1]
ABI = "contract.abi = {args_disjoint = true, args_aligned_bytes = 64 : i64}"


def run(source, *passes):
    return subprocess.run(
        [TOOL, *passes], input=source, text=True, capture_output=True
    )


def war(name, visibility=""):
    return f"""
  func.func {visibility}@{name}(%left: memref<16xf32>, %right: memref<16xf32>, %value: f32) -> f32 {{
    %index = arith.constant 0 : index
    %captured = memref.load %left[%index] : memref<16xf32>
    memref.store %value, %right[%index] : memref<16xf32>
    return %captured : f32
  }}
"""


def module(helper, use):
    return f"""
module attributes {{{ABI}}} {{
  {war("uncalled_entry")}
  {helper}
  func.func @caller(%actual: memref<16xf32>, %value: f32) {{
    {use}
    return
  }}
}}
"""


def assert_war(source, helper_expected=True):
    result = run(
        source, "--kernel-contract-annotate", "--kernel-contract-verify"
    )
    assert result.returncode == 0, result.stderr
    functions = re.split(r"func.func (?:private )?@", result.stdout)
    for name, expected in (
        ("uncalled_entry", False),
        ("helper", helper_expected),
    ):
        body = next(body for body in functions if body.startswith(name + "("))
        tasks = re.findall(r"contract.task = \{([^}]*)\}", body)
        assert len(tasks) == 2, (name, body)
        predecessors = "array<i64: 0>" if expected else "array<i64>"
        assert f"predecessors = {predecessors}" in tasks[1], (name, body)
    return result.stdout


direct = "%result = func.call @helper(%actual, %actual, %value) : (memref<16xf32>, memref<16xf32>, f32) -> f32"
for visibility in ("", "private "):
    output = assert_war(module(war("helper", visibility), direct))
    # A claimed noalias edge omission in the helper must fail verification.
    begin = output.index("func.func " + visibility + "@helper")
    head, tail = output[:begin], output[begin:]
    mutated = head + tail.replace(
        "predecessors = array<i64: 0>", "predecessors = array<i64>", 1
    )
    result = run(mutated, "--kernel-contract-verify")
    assert result.returncode != 0
    assert (
        "V23/V24: missing RAW, WAR, WAW or scalar SSA dependency"
        in result.stderr
    )

address = "%callee = func.constant @helper : (memref<16xf32>, memref<16xf32>, f32) -> f32"
assert_war(module(war("helper"), address))
assert_war(
    module(
        war("helper"),
        address
        + "\n%result = func.call_indirect %callee(%actual, %actual, %value) : (memref<16xf32>, memref<16xf32>, f32) -> f32",
    )
)

# The same helper body has entry guarantees only when it is truly unreferenced.
assert_war(module(war("helper"), ""), helper_expected=False)

pointwise = """
  func.func @helper(%left: memref<16xf32>, %right: memref<16xf32>, %value: f32) -> f32 {
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%left : memref<16xf32>) outs(%right : memref<16xf32>) {
    ^bb0(%x: f32, %old: f32):
      %next = arith.addf %x, %x : f32
      linalg.yield %next : f32
    }
    return %value : f32
  }
"""
result = run(module(pointwise, direct), "--kernel-contract-annotate")
assert result.returncode != 0
assert (
    "V25: overlapping views must have identical element addresses"
    in result.stderr
)
result = run(
    module(pointwise, ""),
    "--kernel-contract-annotate",
    "--kernel-contract-verify",
)
assert result.returncode == 0, result.stderr

# A caller can pass a view whose aligned pointer is one byte past its entry
# argument. The helper's plain descriptor hides that offset, so only an actual
# entry may use the module's alignment promise for direct/DMA decisions.
matrix = """
  func.func @matrix_helper(%a: memref<16x64xi8>,
      %b: memref<64x16xi8, strided<[1, 64]>>) {
    %zero = arith.constant 0 : i32
    %c = memref.alloc() alignment = 64 : memref<16x16xi32>
    linalg.fill ins(%zero : i32) outs(%c : memref<16x16xi32>)
    linalg.matmul {cast = #linalg.type_fn<cast_signed>}
      ins(%a, %b : memref<16x64xi8>, memref<64x16xi8, strided<[1, 64]>>)
      outs(%c : memref<16x16xi32>)
    return
  }
"""
unaligned_caller = """
  func.func @matrix_entry(%bytes: memref<1025xi8>,
      %b: memref<64x16xi8, strided<[1, 64]>>) {
    %one = arith.constant 1 : index
    %shifted = memref.view %bytes[%one][] : memref<1025xi8> to memref<16x64xi8>
    func.call @matrix_helper(%shifted, %b)
      : (memref<16x64xi8>, memref<64x16xi8, strided<[1, 64]>>) -> ()
    return
  }
"""
for caller, accepted in (("", True), (unaligned_caller, False)):
    source = f"module attributes {{{ABI}}} {{\n{matrix}{caller}\n}}"
    result = run(
        source,
        "--kernel-contract-annotate",
        "--bosc-ame-plan",
        "--bosc-ame-plan-verify",
    )
    if accepted:
        assert result.returncode == 0, result.stderr
        assert "plan.route" in result.stdout
    else:
        assert result.returncode != 0
        assert (
            "no legal direct or staging route: external operand"
            in result.stderr
        )

print(
    "PASS entry ABI isolation for direct calls, function addresses, indirect calls and pointwise alias safety"
)
