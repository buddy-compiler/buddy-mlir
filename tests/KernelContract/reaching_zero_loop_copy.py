# RUN: %PYTHON %s buddy-opt
# ===- reaching_zero_loop_copy.py ------------------------------------------
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
# Fold a loop's zero snapshot without hiding writes in a shared lifetime.
#
# Lit Python test. Folds a loop-carried zero snapshot into a copy when lifetimes
# prove no intervening write, and rejects folds that would hide stores sharing
# storage with the zeroed view.
#
# ===---------------------------------------------------------------------------

import re
import subprocess
import sys


def annotate(source):
    result = subprocess.run(
        [sys.argv[1], "--kernel-contract-annotate", "--kernel-contract-verify"],
        input=source,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def module(body, element="i32", zero="0", changed="7"):
    return f"""
module {{
  func.func @scoped(%observation: memref<16x{element}>, %out: memref<16x{element}>) {{
    %zero = arith.constant {zero} : {element}
    %changed = arith.constant {changed} : {element}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %source = memref.alloc() : memref<16x{element}>
    linalg.fill ins(%zero : {element}) outs(%source : memref<16x{element}>)
    {body}
    return
  }}
}}
"""


def product_body(alias=False):
    later = (
        "%view = memref.subview %source[4] [8] [1] : memref<16xi32> "
        "to memref<8xi32, strided<[1], offset: 4>>\n"
        "linalg.fill ins(%changed : i32) "
        "outs(%view : memref<8xi32, strided<[1], offset: 4>>)"
        if alias
        else "linalg.fill ins(%changed : i32) outs(%source : memref<16xi32>)"
    )
    return (
        """
    %acc = memref.alloc() : memref<16xi32>
    linalg.fill ins(%zero : i32) outs(%acc : memref<16xi32>)
    // Preserve an independent observation of the original zero storage.
    memref.copy %source, %observation : memref<16xi32> to memref<16xi32>
    scf.for %i = %c0 to %c3 step %c1 {
      %product = memref.alloc() : memref<16xi32>
      memref.copy %source, %product : memref<16xi32> to memref<16xi32>
      linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%acc, %product : memref<16xi32>, memref<16xi32>) outs(%acc : memref<16xi32>) {
      ^bb0(%old: i32, %value: i32, %unused: i32):
        %sum = arith.addi %old, %value : i32
        linalg.yield %sum : i32
      }
    }
    """
        + later
        + """
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%acc, %source : memref<16xi32>, memref<16xi32>) outs(%acc : memref<16xi32>) {
    ^bb0(%old: i32, %value: i32, %unused: i32):
      %sum = arith.addi %old, %value : i32
      linalg.yield %sum : i32
    }
    memref.copy %acc, %out : memref<16xi32> to memref<16xi32>
    """
    )


for alias in (False, True):
    output = annotate(module(product_body(alias)))
    # The three-iteration reduction remains a complete single-slot graph;
    # its result is consumed by the following tail add in the parent scope.
    assert "contract.compute_graph = {complete = true" in output
    assert "trip_count = 3 : i64" in output
    assert re.search(r"exit_consumer_compute_ids = array<i64: \d+>", output)
    assert re.search(r"backedge_producer_compute_ids = array<i64: \d+>", output)
    loop = output[output.index("scf.for") : output.index("contract.cf")]
    assert "memref.copy" not in loop
    assert "linalg.fill" in loop
    # Only the loop snapshot disappears. The original observer, final output,
    # zero storage and post-loop mutation retain their normal graph events.
    assert output.count("memref.copy") == 2
    assert "arith.constant 7 : i32" in output
    assert output.count("memref.alloc") == 3

copy = """
    %product = memref.alloc() : memref<16xi32>
    memref.copy %source, %product : memref<16xi32> to memref<16xi32>
"""
write = "linalg.fill ins(%changed : i32) outs(%source : memref<16xi32>)"

# Following the copy lexically in the same loop does not suffice: the write
# precedes the next copy. The real loop-carried source must remain visible.
output = annotate(
    module("scf.for %i = %c0 to %c3 step %c1 {" + copy + write + "}")
)
assert "memref.copy" in output
assert "contract.loop_carried_slots" in output

# Lift through every enclosing repeated scope sharing the source allocation,
# not just the innermost loop. Its next outer iteration observes the mutation.
output = annotate(
    module(
        "scf.for %outer = %c0 to %c2 step %c1 {"
        "scf.for %inner = %c0 to %c3 step %c1 {" + copy + "}" + write + "}"
    )
)
assert "memref.copy" in output

# If each outer iteration creates fresh source storage, the inner reduction's
# scope ends before its local mutation. A previous outer instance cannot leak.
body = copy + "\n"
fresh = module(
    "scf.for %outer = %c0 to %c2 step %c1 {"
    "%fresh = memref.alloc() : memref<16xi32>"
    "linalg.fill ins(%zero : i32) outs(%fresh : memref<16xi32>)"
    "scf.for %inner = %c0 to %c3 step %c1 {"
    + body.replace("%source", "%fresh")
    + "}"
    + write.replace("%source", "%fresh")
    + "}"
)
assert "memref.copy" not in annotate(fresh)

# Reuse the exact scalar bits, including a floating negative-zero initializer.
floating = module("scf.for %i = %c0 to %c3 step %c1 {" + copy + "}" + write)
floating = (
    floating.replace("i32", "f32")
    .replace("0 : f32", "-0.0 : f32")
    .replace("7 : f32", "7.0 : f32")
)
output = annotate(floating)
assert "memref.copy" not in output
assert "-0.000000e+00" in output

print("PASS scoped reaching-zero snapshots and loop boundary dependencies")
