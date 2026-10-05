# RUN: %PYTHON %s buddy-opt 2>&1 | FileCheck %s
#
# runtime/threadpool/BuddyThreadPool.c, the OpenMP runtime of models built
# with "thread_pool": true (docs/ModelThreadPool.md). Parallel loops lowered
# like compile_pipeline.py lowers them (scf.parallel -> OpenMP -> LLVM) are
# linked with the pool and without libomp, which only links if the pool has
# every entry point they call, and run: each iteration runs once, on several
# threads; BUDDY_THREAD_POOL_THREADS limits them; a nested parallel loop runs
# serially inside the outer one.

import os
import shutil
import subprocess
import sys
import tempfile

SRC = os.environ["BUDDY_SRC_ROOT"]
sys.path.insert(0, os.path.join(SRC, "tools", "buddy-codegen"))
import compile_pipeline  # noqa: E402

THREADS = 4

# @squares(out): out[i] = i * i, recording the thread of each iteration.
# @table(out): out[i][j] = i * 1000 + j, a parallel loop in a parallel loop.
MLIR = """
func.func private @record(index)
func.func @squares(%out: memref<?xi64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = memref.dim %out, %c0 : memref<?xi64>
  scf.parallel (%i) = (%c0) to (%n) step (%c1) {
    func.call @record(%i) : (index) -> ()
    %v = arith.index_cast %i : index to i64
    %sq = arith.muli %v, %v : i64
    memref.store %sq, %out[%i] : memref<?xi64>
    scf.reduce
  }
  return
}
func.func @table(%out: memref<?x?xi64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %k = arith.constant 1000 : i64
  %n = memref.dim %out, %c0 : memref<?x?xi64>
  %m = memref.dim %out, %c1 : memref<?x?xi64>
  scf.parallel (%i) = (%c0) to (%n) step (%c1) {
    scf.parallel (%j) = (%c0) to (%m) step (%c1) {
      %vi = arith.index_cast %i : index to i64
      %vj = arith.index_cast %j : index to i64
      %a = arith.muli %vi, %k : i64
      %v = arith.addi %a, %vj : i64
      memref.store %v, %out[%i, %j] : memref<?x?xi64>
      scf.reduce
    }
    scf.reduce
  }
  return
}
"""

DRIVER = r"""
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int64_t *allocated, *aligned;
  intptr_t offset, size, stride;
} Vec;
typedef struct {
  int64_t *allocated, *aligned;
  intptr_t offset, sizes[2], strides[2];
} Mat;
extern void _mlir_ciface_squares(Vec *out);
extern void _mlir_ciface_table(Mat *out);

#define N 1000
static pthread_t who[N];
static int calls[N];

// @record, through its C interface (-llvm-request-c-wrappers)
void _mlir_ciface_record(intptr_t i) {
  who[i] = pthread_self();
  __atomic_fetch_add(&calls[i], 1, __ATOMIC_RELAXED);
}

int main(void) {
  static int64_t v[N];
  for (int round = 0; round < 3; ++round) {
    Vec out = {v, v, 0, N, 1};
    _mlir_ciface_squares(&out);
  }
  int right = 1, once = 1, threads = 0;
  for (int i = 0; i < N; ++i) {
    right &= v[i] == (int64_t)i * i;
    once &= calls[i] == 3;
    int seen = 0;
    for (int j = 0; j < i && !seen; ++j)
      seen = pthread_equal(who[i], who[j]);
    threads += !seen;
  }
  printf("squares: right %d, each iteration once per call %d, threads %d\n",
         right, once, threads);

  static int64_t t[37][53];
  Mat m = {&t[0][0], &t[0][0], 0, {37, 53}, {53, 1}};
  _mlir_ciface_table(&m);
  right = 1;
  for (int i = 0; i < 37; ++i)
    for (int j = 0; j < 53; ++j)
      right &= t[i][j] == i * 1000 + j;
  printf("nested: right %d\n", right);
  return 0;
}
"""

work = tempfile.mkdtemp()


def path(name):
    return os.path.join(work, name)


passes = [
    f"-convert-scf-to-openmp=num-threads={THREADS}",
    *compile_pipeline.lower_to_llvm(arena=False),
]
llvm_ir = subprocess.run(
    [sys.argv[1], *passes],
    input=MLIR,
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
subprocess.run(
    [shutil.which("llc"), "-filetype=obj", "-relocation-model=pic"]
    + [path("f.ll"), "-o", path("f.o")],
    check=True,
)
with open(path("driver.c"), "w") as f:
    f.write(DRIVER)
cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
# No libomp: the pool must provide every OpenMP entry point.
subprocess.run(
    [
        cc,
        "-O2",
        path("driver.c"),
        path("f.o"),
        os.path.join(SRC, "runtime", "threadpool", "BuddyThreadPool.c"),
        "-lpthread",
        "-o",
        path("run"),
    ],
    check=True,
)
print("linked without libomp")
# CHECK: linked without libomp

cpus = len(os.sched_getaffinity(0))
for limit in (None, 2, 1):
    env = dict(os.environ)
    env.pop("BUDDY_THREAD_POOL_THREADS", None)
    if limit:
        env["BUDDY_THREAD_POOL_THREADS"] = str(limit)
    run = subprocess.run([path("run")], capture_output=True, text=True, env=env)
    # the pool has num_threads threads, or fewer CPUs or the limit
    want = min(THREADS, cpus, limit or THREADS)
    out = run.stdout.replace(f"threads {want}", "threads as expected")
    print(f"limit {limit}:", out.strip().replace("\n", " | "), run.stderr)
# CHECK: limit None: squares: right 1, each iteration once per call 1, threads as expected | nested: right 1
# CHECK: limit 2: squares: right 1, each iteration once per call 1, threads as expected | nested: right 1
# CHECK: limit 1: squares: right 1, each iteration once per call 1, threads as expected | nested: right 1
