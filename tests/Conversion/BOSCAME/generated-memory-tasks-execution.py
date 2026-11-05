# RUN: %PYTHON %s %t
# ===- generated-memory-tasks-execution.py ---------------------------------
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
# Generated private DDR fills/copies preserve striding, padding and bits.
#
# Lit Python test. Exercises planner-generated private DDR fill/copy tasks and
# checks that strided layouts, padding bytes, and bit patterns survive lowering
# and the functional buffer simulator. Failures must name the mismatched region.
#
# ===---------------------------------------------------------------------------

import ctypes
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORK = Path(sys.argv[1])
WORK.mkdir(parents=True, exist_ok=True)
OPT = Path(os.environ.get("BUDDY_OPT", ROOT / "build/bin/buddy-opt"))
LLVM = ROOT / "llvm/build/bin"


def run(command):
    result = subprocess.run(
        list(map(str, command)), capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def task(body, kind):
    return (
        "scf.execute_region {\n"
        + body
        + "\nscf.yield\n} "
        + '{contract.task = {kind = "'
        + kind
        + '"}}'
    )


def loops(operation, rows, cols):
    return f"""
      %z = arith.constant 0 : index
      %one = arith.constant 1 : index
      %rows = arith.constant {rows} : index
      scf.for %i = %z to %rows step %one {{
        %zz = arith.constant 0 : index
        %step = arith.constant 1 : index
        %cols = arith.constant {cols} : index
        scf.for %j = %zz to %cols step %step {{
          {operation}
        }}
      }}
    """


def function(name, element, transposed=False):
    # Include a non-power-of-two valid extent and preserve untouched padding.
    rows, cols = (24, 2) if transposed else (2, 24)
    strides = [1, 29] if transposed else [29, 1]
    ty = f"memref<{rows}x{cols}x{element}, strided<{strides}>>"
    contiguous_ty = f"memref<2x64x{element}>"
    tile_ty = (
        f"memref<64x2x{element}, strided<[1, 64]>>"
        if transposed
        else contiguous_ty
    )
    size = 128 * (1 if element == "i8" else 4)
    transpose = (
        f"%tile = memref.transpose %storage (d0, d1) -> (d1, d0) "
        f": {contiguous_ty} to {tile_ty}"
        if transposed
        else ""
    )
    tile = "%tile" if transposed else "%storage"
    fill = task(
        loops(f"memref.store %zero, %storage[%i, %j] : {contiguous_ty}", 2, 64),
        "fill",
    )
    copy = task(
        loops(
            f"%v = memref.load %input[%i, %j] : {ty}\n"
            f"memref.store %v, {tile}[%i, %j] : {tile_ty}",
            rows,
            cols,
        ),
        "memory-load",
    )
    store = task(
        loops(
            f"%v = memref.load {tile}[%i, %j] : {tile_ty}\n"
            f"memref.store %v, %output[%i, %j] : {ty}",
            rows,
            cols,
        ),
        "memory-store",
    )
    zero = "0.0" if element == "f32" else "0"
    return (
        f"""
      func.func @{name}(%input: {ty}, %output: {ty}) attributes {{llvm.emit_c_interface}} {{
        %frame = bosc_ame.ddr.frame : memref<{size}xi8>
        %offset = arith.constant 0 : index
        %storage = memref.view %frame[%offset][] : memref<{size}xi8> to {contiguous_ty}
        {transpose}
        %zero = arith.constant {zero} : {element}
        {fill}
        {copy}
        {store}
        bosc_ame.ddr.release %frame : memref<{size}xi8>
        return
      }}
    """,
        (rows, cols),
        strides,
    )


functions = []
cases = []
for element in ("i8", "i32", "f32"):
    for transpose in (False, True):
        name = element + ("_column" if transpose else "_row")
        source, shape, strides = function(name, element, transpose)
        functions.append(source)
        cases.append((name, element, shape, strides))

# Two public arguments have no private-frame proof: even a claimed generated
# copy must keep scalar traversal, including when their descriptors alias.
ty = "memref<2x24xi8>"
functions.append(f"""
  func.func @external_alias(%input: {ty}, %output: {ty}) attributes {{llvm.emit_c_interface}} {{
    {task(loops(f"%v = memref.load %input[%i, %j] : {ty}" + chr(10) + f"memref.store %v, %output[%i, %j] : {ty}", 2, 24), "memory-load")}
    return
  }}
""")
functions.append(f"""
  func.func @same_private_root() {{
    %frame = bosc_ame.ddr.frame : memref<128xi8>
    %offset = arith.constant 0 : index
    %shift = arith.constant 8 : index
    %input = memref.view %frame[%offset][] : memref<128xi8> to {ty}
    %output = memref.view %frame[%shift][] : memref<128xi8> to {ty}
    {task(loops(f"%v = memref.load %input[%i, %j] : {ty}" + chr(10) + f"memref.store %v, %output[%i, %j] : {ty}", 2, 24), "memory-load")}
    bosc_ame.ddr.release %frame : memref<128xi8>
    return
  }}
  func.func private @side_effect()
  func.func @preserve_side_effect() {{
    %frame = bosc_ame.ddr.frame : memref<128xi8>
    %offset = arith.constant 0 : index
    %output = memref.view %frame[%offset][] : memref<128xi8> to {ty}
    %zero = arith.constant 0 : i8
    {task(loops("func.call @side_effect() : () -> ()" + chr(10) + f"memref.store %zero, %output[%i, %j] : {ty}", 2, 24), "fill")}
    bosc_ame.ddr.release %frame : memref<128xi8>
    return
  }}
""")
source = (
    'module attributes {bosc_ame.target = "nr-fpga", bosc_ame.cpu_ddr_policy, '
    "contract.abi = {args_disjoint = true}} {\n" + "\n".join(functions) + "\n}"
)
source_path = WORK / "memory.mlir"
lowered_path = WORK / "memory.llvm.mlir"
source_path.write_text(source)
run([OPT, source_path, "--lower-bosc-ame", "-o", lowered_path])
lowered = lowered_path.read_text()
for name, element, _, _ in cases:
    body = lowered.split("llvm.func @" + name + "(", 1)[1].split("\n  }", 1)[0]
    lanes = 24 if element == "i8" else 12
    assert f"vector<{lanes}x{element}>" in body, name
    assert body.count("llvm.inline_asm") == 5, name
alias = lowered.split("llvm.func @external_alias(", 1)[1].split("\n  }", 1)[0]
assert "vector<" not in alias
for name in ("same_private_root", "preserve_side_effect"):
    body = lowered.split("llvm.func @" + name + "(", 1)[1].split("\n  }", 1)[0]
    assert "vector<" not in body, name
    if name == "preserve_side_effect":
        assert "llvm.call @side_effect" in body

# The target policies are MMIO/cache transitions. Remove only their zero-result
# inline assembly for native execution; all transformed loads/stores stay intact.
native = WORK / "memory.native.mlir"
native.write_text(
    "\n".join(
        line for line in lowered.splitlines() if "llvm.inline_asm" not in line
    )
)
ll = WORK / "memory.ll"
run([OPT.with_name("buddy-translate"), native, "--buddy-to-llvmir", "-o", ll])
runtime = WORK / "runtime.c"
runtime.write_text("""
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
void side_effect(void) {}
uint64_t bosc_ame_private_ddr_acquire(uint64_t n) {
  void *p = aligned_alloc(64, (n + 63) & ~63ull);
  if (!p) abort();
  memset(p, 0xa7, n);
  return (uintptr_t)p;
}
void bosc_ame_private_ddr_release(uint64_t p, uint64_t n) {
  const unsigned char *s = (const unsigned char *)(uintptr_t)p;
  /* Every private tile padding element must still contain the explicit zero. */
  for (uint64_t row = 0; row < 2; ++row)
    for (uint64_t k = row * n / 2 + 24 * (n / 128); k < (row + 1) * n / 2; ++k)
      if (s[k] != 0) abort();
  free((void *)(uintptr_t)p);
}
""")
library = WORK / "memory.so"
run([LLVM / "clang", ll, runtime, "-O2", "-shared", "-fPIC", "-o", library])
native_library = ctypes.CDLL(str(library.resolve()))


class Descriptor(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64 * 2),
        ("strides", ctypes.c_int64 * 2),
    ]


checked = 0
for name, element, shape, strides in cases:
    width = 1 if element == "i8" else 4
    count = 29 + 24
    for seed in (0, 19, 231):
        input_bytes = bytes((i * 37 + seed) % 256 for i in range(count * width))
        source_buffer = (ctypes.c_ubyte * (len(input_bytes) + 128))()
        output_buffer = (ctypes.c_ubyte * (len(input_bytes) + 128))()
        ctypes.memset(output_buffer, 0xE7, ctypes.sizeof(output_buffer))
        ctypes.memmove(
            ctypes.addressof(source_buffer) + 64, input_bytes, len(input_bytes)
        )
        before = bytes(source_buffer)
        expected = bytearray(output_buffer)
        for i in range(shape[0]):
            for j in range(shape[1]):
                pos = (i * strides[0] + j * strides[1]) * width
                expected[64 + pos : 64 + pos + width] = input_bytes[
                    pos : pos + width
                ]

        def descriptor(buffer, shape=shape, strides=strides):
            return Descriptor(
                ctypes.addressof(buffer),
                ctypes.addressof(buffer) + 64,
                0,
                (ctypes.c_int64 * 2)(*shape),
                (ctypes.c_int64 * 2)(*strides),
            )

        left, right = descriptor(source_buffer), descriptor(output_buffer)
        fn = getattr(native_library, "_mlir_ciface_" + name)
        fn.argtypes = [ctypes.POINTER(Descriptor), ctypes.POINTER(Descriptor)]
        fn(ctypes.byref(left), ctypes.byref(right))
        assert bytes(source_buffer) == before, name
        assert bytes(output_buffer) == expected, name
        checked += shape[0] * shape[1]

print(
    f"PASS generated memory tasks: {len(cases)} strided copies, {checked} bit-exact elements and guards"
)
