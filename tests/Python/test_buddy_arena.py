# RUN: %PYTHON %s buddy-opt 2>&1 | FileCheck %s
#
# runtime/arena/BuddyArena.c, the arena of models built with "arena": true
# (docs/ModelMemoryOptions.md). A function that allocates is lowered with the
# arena passes of compile_pipeline.py, linked with the arena and a C driver,
# and run: its buffers are bumped from one range, 64-byte aligned or more,
# frees do nothing, and buddy_arena_reset() makes the next call reuse the
# same memory.

import os
import shutil
import subprocess
import sys
import tempfile

SRC = os.environ["BUDDY_SRC_ROOT"]
sys.path.insert(0, os.path.join(SRC, "tools", "buddy-codegen"))
import compile_pipeline  # noqa: E402

# f(n) returns a buffer of n floats filled with 1.0, after allocating (and
# freeing) a scratch buffer of 3 floats aligned to 256 bytes.
MLIR = """
func.func @f(%n: index) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %one = arith.constant 1.0 : f32
  %t = memref.alloc() alignment = 256 : memref<3xf32>
  memref.store %one, %t[%c0] : memref<3xf32>
  memref.dealloc %t : memref<3xf32>
  %b = memref.alloc(%n) : memref<?xf32>
  scf.for %i = %c0 to %n step %c1 {
    memref.store %one, %b[%i] : memref<?xf32>
  }
  return %b : memref<?xf32>
}
"""

DRIVER = r"""
#include <stdint.h>
#include <stdio.h>

typedef struct {
  float *allocated, *aligned;
  intptr_t offset, size, stride;
} Buf;
extern void _mlir_ciface_f(Buf *result, intptr_t n);
extern void buddy_arena_reset(void);

static Buf call(intptr_t n) {
  Buf b;
  _mlir_ciface_f(&b, n);
  float sum = 0;
  for (intptr_t i = 0; i < n; ++i)
    sum += b.aligned[i];
  printf("n=%ld sum=%g aligned64=%d\n", (long)n, sum,
         (int)((uintptr_t)b.aligned % 64 == 0));
  return b;
}

int main(void) {
  Buf a = call(1000);
  Buf b = call(10);  // same call: after a
  long gap = (long)((char *)b.aligned - (char *)a.aligned);
  printf("second buffer after the first: %d\n", gap >= 1000 * 4);
  buddy_arena_reset();
  Buf c = call(1000);  // next call: reuses a's memory
  printf("after reset, same address: %d\n", c.aligned == a.aligned);
  return 0;
}
"""

work = tempfile.mkdtemp()


def path(name):
    return os.path.join(work, name)


with open(path("f.mlir"), "w") as f:
    f.write(MLIR)
with open(path("driver.c"), "w") as f:
    f.write(DRIVER)

passes = [
    "-convert-scf-to-cf",
    *compile_pipeline.lower_to_llvm(arena=True),
]
llvm_ir = subprocess.run(
    [sys.argv[1], path("f.mlir"), *passes],
    capture_output=True,
    text=True,
    check=True,
).stdout
llvm_ir = subprocess.run(
    [shutil.which("mlir-translate"), "-mlir-to-llvmir"],
    input=llvm_ir,
    capture_output=True,
    text=True,
    check=True,
).stdout
with open(path("f.ll"), "w") as f:
    f.write(llvm_ir)
# CHECK: _mlir_memref_to_llvm_alloc 1, _mlir_memref_to_llvm_free 1, malloc 0, free 0
print(
    ", ".join(
        f"{fn} {int(f'@{fn}(' in llvm_ir)}"
        for fn in (
            "_mlir_memref_to_llvm_alloc",
            "_mlir_memref_to_llvm_free",
            "malloc",
            "free",
        )
    )
)

subprocess.run(
    [
        shutil.which("llc"),
        "-filetype=obj",
        "-relocation-model=pic",
        path("f.ll"),
        "-o",
        path("f.o"),
    ],
    check=True,
)
cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
subprocess.run(
    [
        cc,
        "-O2",
        path("driver.c"),
        path("f.o"),
        os.path.join(SRC, "runtime", "arena", "BuddyArena.c"),
        "-o",
        path("run"),
    ],
    check=True,
)
env = dict(os.environ, BUDDY_ARENA_RESERVE_MB="64", BUDDY_ARENA_STATS="1")
run = subprocess.run([path("run")], capture_output=True, text=True, env=env)
print(run.stdout + run.stderr)
# CHECK: n=1000 sum=1000 aligned64=1
# CHECK-NEXT: n=10 sum=10 aligned64=1
# CHECK-NEXT: second buffer after the first: 1
# CHECK-NEXT: n=1000 sum=1000 aligned64=1
# CHECK-NEXT: after reset, same address: 1
# CHECK-NEXT: [BuddyArena] largest call: 0 MiB of 64 MiB reserved

# A call that needs more than the reserved range stops with a message.
with open(path("big.c"), "w") as f:
    f.write(DRIVER.replace("Buf a = call(1000);", "Buf a = call(20 << 20);"))
subprocess.run(
    [
        cc,
        "-O2",
        path("big.c"),
        path("f.o"),
        os.path.join(SRC, "runtime", "arena", "BuddyArena.c"),
        "-o",
        path("big"),
    ],
    check=True,
)
run = subprocess.run([path("big")], capture_output=True, text=True, env=env)
print("exit", "abort" if run.returncode != 0 else run.returncode)
print(run.stderr.strip())
# CHECK: exit abort
# CHECK-NEXT: [BuddyArena] out of space: a forward call needs more than 64 MiB (set BUDDY_ARENA_RESERVE_MB)
