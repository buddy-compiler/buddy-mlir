# RUN: %PYTHON %s %t
# ===- scalar_abs_execution.py ---------------------------------------------
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
# Opaque scalar abs clears only the sign bit and needs no libm call.
#
# Lit Python test. Confirms the opaque scalar abs lowering clears only the sign
# bit (IEEE-754) and does not emit a libm call. Uses the temporary path from lit
# for intermediate artifacts when encoding/running the snippet.
#
# ===---------------------------------------------------------------------------

import ctypes
import os
import random
import subprocess
import sys
from pathlib import Path
from shutil import which

ROOT = Path(__file__).resolve().parents[3]
WORK = Path(sys.argv[1])
WORK.mkdir(parents=True, exist_ok=True)


def tool(name, default=None):
    override = os.environ.get("BUDDY_OPT")
    if override and name.startswith("buddy-"):
        sibling = Path(override).with_name(name)
        if sibling.is_file():
            return sibling
    found = which(name)
    if found:
        return Path(found)
    if default and Path(default).is_file():
        return Path(default)
    raise FileNotFoundError(name)


OPT = tool("buddy-opt", ROOT / "build/bin/buddy-opt")
SOURCE = r"""
module {
  func.func @abs_bits(%word: i32) -> i32 {
    %x = arith.bitcast %word : i32 to f32
    %abs = math.absf %x : f32
    %bits = arith.bitcast %abs : f32 to i32
    return %bits : i32
  }
  func.func @abs_loop(%word: i32, %n: index) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %x = arith.bitcast %word : i32 to f32
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%a = %x) -> f32 {
      %abs = math.absf %a : f32
      scf.yield %abs : f32
    }
    %bits = arith.bitcast %result : f32 to i32
    return %bits : i32
  }
}
"""


def run(command):
    result = subprocess.run(
        list(map(str, command)), capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


source = WORK / "abs.mlir"
lowered = WORK / "abs.llvm.mlir"
ll = WORK / "abs.ll"
source.write_text(SOURCE)
run([OPT, source, "--lower-bosc-ame", "-o", lowered])
assert "fabsf" not in lowered.read_text(), lowered.read_text()
run([OPT.with_name("buddy-translate"), lowered, "--buddy-to-llvmir", "-o", ll])
assembly = run(
    [
        tool("llc"),
        ll,
        "-O2",
        "-mtriple=riscv64",
        "-mattr=+m,+a,+f,+d,+c",
        "-o",
        "-",
    ]
)
assert "fabs" not in assembly, assembly
assert "call" not in assembly, assembly
library = WORK / "abs.so"
run([tool("clang"), ll, "-O2", "-shared", "-fPIC", "-o", library])
native = ctypes.CDLL(str(library.resolve()))
native.abs_bits.argtypes = [ctypes.c_uint32]
native.abs_bits.restype = ctypes.c_uint32
native.abs_loop.argtypes = [ctypes.c_uint32, ctypes.c_uint64]
native.abs_loop.restype = ctypes.c_uint32
special = [
    0,
    0x80000000,
    1,
    0x80000001,
    0x007FFFFF,
    0x807FFFFF,
    0x00800000,
    0x80800000,
    0x7F7FFFFF,
    0xFF7FFFFF,
    0x7F800000,
    0xFF800000,
    0x7FC01234,
    0xFFC01234,
    0x7F800001,
    0xFF800001,
]
rng = random.Random(112)
words = special + [rng.getrandbits(32) for _ in range(10000)]
for word in words:
    expected = word & 0x7FFFFFFF
    assert native.abs_bits(word) == expected, hex(word)
    assert native.abs_loop(word, 3) == expected, hex(word)
    assert native.abs_loop(word, 0) == word, hex(word)
print(
    f"PASS scalar abs: {len(words)} f32 bit patterns, opaque loop and no libcall"
)
