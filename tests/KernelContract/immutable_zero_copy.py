# RUN: %PYTHON %s buddy-opt
# ===- immutable_zero_copy.py ----------------------------------------------
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
# Prove zero rematerialization before creating loop content versions.
#
# Lit Python test. Ensures a proved immutable zero can be rematerialized only
# before loop content versions are created, and that annotation/verify reject
# histories that would invent a zero after a conflicting write.
#
# ===---------------------------------------------------------------------------

import subprocess
import sys


def annotate(source):
    result = subprocess.run(
        [sys.argv[1], "--kernel-contract-annotate", "--kernel-contract-verify"],
        input=source,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def source(initial="0", before="", after="", loop=False, partial=False):
    initialize = (
        "%nonzero = arith.constant 7 : i32\n"
        "linalg.fill ins(%nonzero : i32) outs(%zero : memref<16xi32>)\n"
        "%part = memref.subview %zero[0] [8] [1] : "
        "memref<16xi32> to memref<8xi32, strided<[1]>>\n"
        "linalg.fill ins(%value : i32) outs(%part : "
        "memref<8xi32, strided<[1]>>)"
        if partial
        else "linalg.fill ins(%value : i32) outs(%zero : memref<16xi32>)"
    )
    copy = """
    %target = memref.alloc() : memref<16xi32>
    memref.copy %zero, %target : memref<16xi32> to memref<16xi32>
    """
    if loop:
        copy = "scf.for %i = %c0 to %c2 step %c1 {" + copy + "}"
    return f"""
module {{
  func.func @zero() {{
    %value = arith.constant {initial} : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero = memref.alloc() : memref<16xi32>
    {initialize}
    {before}
    {copy}
    {after}
    return
  }}
}}
"""


# The unchanged complete zero can be rematerialized in every loop iteration.
# There is no cross-scope read to misclassify as a loop accumulator boundary.
output = annotate(source(loop=True))
assert "memref.copy" not in output
assert output.count("memref.alloc") == 1
assert "complete = true" in output
assert "invariant" not in output

# Shape equality alone cannot prove the content remains zero.
store = """
    %changed = arith.constant 7 : i32
    linalg.fill ins(%changed : i32) outs(%zero : memref<16xi32>)
"""
for text in (
    source(initial="1"),
    source(before=store),
    source(partial=True),
    source(
        before="""
    %alias = memref.subview %zero[4] [8] [1] : memref<16xi32>
      to memref<8xi32, strided<[1], offset: 4>>
    linalg.fill ins(%value : i32) outs(%alias : memref<8xi32, strided<[1], offset: 4>>)
    """
    ),
):
    assert "memref.copy" in annotate(text)

# A write after the complete copy scope cannot change its reaching zero.
# Keep the later write and its allocation: only the earlier read is folded.
output = annotate(source(after=store))
assert "memref.copy" not in output
assert "arith.constant 7 : i32" in output
assert output.count("memref.alloc") == 2

# An unknown callee may change a passed allocation. The normalizer cannot
# turn that uncertainty into a fresh zero value or hide the original read.
escaped = source(
    before="func.call @touch(%zero) : (memref<16xi32>) -> ()"
).replace("module {", "module {\n  func.func private @touch(memref<16xi32>)", 1)
assert "memref.copy" in annotate(escaped)

# Copying negative zero preserves the actual scalar bits; it is not silently
# replaced with the additive identity positive zero.
negative = source().replace("0 : i32", "-0.0 : f32").replace("i32", "f32")
output = annotate(negative)
assert "memref.copy" not in output
assert "-0.000000e+00" in output

# Dynamic scalar splats folded into a lane expression become explicit scalar
# inputs. Mixed element types and intermediate predicates stay inside it.
output = annotate(
    """
module {
  func.func @expression(%scale: f32) {
    %zero = arith.constant 0.0 : f32
    %input = memref.alloc() : memref<16xf32>
    %output = memref.alloc() : memref<16xi8>
    linalg.fill ins(%zero : f32) outs(%input : memref<16xf32>)
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%input : memref<16xf32>) outs(%output : memref<16xi8>) {
    ^bb0(%x: f32, %old: i8):
      %value = arith.divf %x, %scale : f32
      %positive = arith.cmpf oge, %value, %zero : f32
      %selected = arith.select %positive, %value, %zero : f32
      %integer = arith.fptosi %selected : f32 to i32
      %byte = arith.trunci %integer : i32 to i8
      linalg.yield %byte : i8
    }
    return
  }
}
"""
)
assert "affine_map<(d0) -> ()>" in output
assert "memref<16xf32>, f32" in output
assert "complete = true" in output
