# RUN: %PYTHON %s buddy-opt 2>&1 | FileCheck %s
#
# runtime/threadpool/BuddyThreadPool.c, the OpenMP runtime of models built
# with "thread_pool": true (docs/ModelThreadPool.md). Parallel loops lowered
# like compile_pipeline.py lowers them (scf.parallel -> OpenMP -> LLVM) are
# linked with the pool and without libomp, which only links if the pool has
# every entry point they call, and run: each iteration runs once, on several
# threads; BUDDY_THREAD_POOL_THREADS limits them; a nested parallel loop runs
# serially inside the outer one; two application threads that run parallel
# loops of different thread counts at the same time each get their own count.

import os
import shutil
import subprocess
import sys
import tempfile

from Inputs.runtime_test_utils import llc_target_args

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
    + llc_target_args()
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


# Two application threads at once, with loops lowered for 2 and for 3
# threads: every call must run on its own count (at most one per CPU), the
# pool growing to the larger one.
TEAM = """
func.func private @record{n}(index)
func.func @team{n}(%out: memref<?xi64>) {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %len = memref.dim %out, %c0 : memref<?xi64>
  scf.parallel (%i) = (%c0) to (%len) step (%c1) {{
    func.call @record{n}(%i) : (index) -> ()
    scf.reduce
  }}
  return
}}
"""

CONCURRENT = r"""
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
  int64_t *allocated, *aligned;
  intptr_t offset, size, stride;
} Vec;
extern void _mlir_ciface_team2(Vec *out);
extern void _mlir_ciface_team3(Vec *out);

#define N 60
#define CALLS 2000
static pthread_t who2[N], who3[N];
void _mlir_ciface_record2(intptr_t i) { who2[i] = pthread_self(); }
void _mlir_ciface_record3(intptr_t i) { who3[i] = pthread_self(); }

static int distinct(pthread_t *who) {
  int count = 0;
  for (int i = 0; i < N; ++i) {
    int seen = 0;
    for (int j = 0; j < i && !seen; ++j)
      seen = pthread_equal(who[i], who[j]);
    count += !seen;
  }
  return count;
}

struct Caller {
  void (*team)(Vec *);
  pthread_t *who;
  int expected, wrong;
};

static void *caller(void *arg) {
  struct Caller *c = arg;
  static int64_t v[2][N];
  for (int k = 0; k < CALLS; ++k) {
    Vec out = {v[c->expected == 3], v[c->expected == 3], 0, N, 1};
    c->team(&out);
    c->wrong += distinct(c->who) != c->expected;
  }
  return NULL;
}

// The interleaving of the review of this runtime, forced: A pushes 2
// threads, B pushes 3, then A forks, then B. Each region must get the count
// its own thread pushed.
typedef struct ident ident_t;
typedef void (*microtask_t)(int32_t *, int32_t *, ...);
extern void __kmpc_push_num_threads(ident_t *, int32_t, int32_t);
extern void __kmpc_fork_call(ident_t *, int32_t, microtask_t, ...);
static void count(int32_t *gtid, int32_t *btid, int *team) {
  (void)gtid, (void)btid;
  __atomic_add_fetch(team, 1, __ATOMIC_RELAXED);
}
static int pushedA, pushedB, forkedA, teamA, teamB;
static void wait(int *flag) {
  while (!__atomic_load_n(flag, __ATOMIC_ACQUIRE))
    ;
}
static void *threadA(void *arg) {
  (void)arg;
  __kmpc_push_num_threads(NULL, 0, 2);
  __atomic_store_n(&pushedA, 1, __ATOMIC_RELEASE);
  wait(&pushedB);
  __kmpc_fork_call(NULL, 1, (microtask_t)count, &teamA);
  __atomic_store_n(&forkedA, 1, __ATOMIC_RELEASE);
  return NULL;
}
static void *threadB(void *arg) {
  (void)arg;
  wait(&pushedA);
  __kmpc_push_num_threads(NULL, 0, 3);
  __atomic_store_n(&pushedB, 1, __ATOMIC_RELEASE);
  wait(&forkedA);
  __kmpc_fork_call(NULL, 1, (microtask_t)count, &teamB);
  return NULL;
}

int main(int argc, char **argv) {
  int cpus = argc > 1 ? atoi(argv[1]) : 1;
  pthread_t pa, pb;
  pthread_create(&pa, NULL, threadA, NULL);
  pthread_create(&pb, NULL, threadB, NULL);
  pthread_join(pa, NULL);
  pthread_join(pb, NULL);
  printf("interleaved: A pushed 2, team as expected %d; B pushed 3, team as "
         "expected %d\n",
         teamA == (cpus < 2 ? cpus : 2), teamB == (cpus < 3 ? cpus : 3));

  struct Caller a = {_mlir_ciface_team2, who2, cpus < 2 ? cpus : 2, 0};
  struct Caller b = {_mlir_ciface_team3, who3, cpus < 3 ? cpus : 3, 0};
  pthread_t ta, tb;
  pthread_create(&ta, NULL, caller, &a);
  pthread_create(&tb, NULL, caller, &b);
  pthread_join(ta, NULL);
  pthread_join(tb, NULL);
  printf("concurrent: 2-thread calls with a wrong team %d, 3-thread calls "
         "with a wrong team %d\n", a.wrong, b.wrong);
  return 0;
}
"""


def lower(mlir, threads, name):
    passes = [
        f"-convert-scf-to-openmp=num-threads={threads}",
        *compile_pipeline.lower_to_llvm(arena=False),
    ]
    ir = subprocess.run(
        [sys.argv[1], *passes],
        input=mlir,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    ir = subprocess.run(
        [shutil.which("mlir-translate"), "-mlir-to-llvmir"],
        input=ir,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    with open(path(f"{name}.ll"), "w") as f:
        f.write(ir)
    subprocess.run(
        [shutil.which("llc"), "-filetype=obj", "-relocation-model=pic"]
        + llc_target_args()
        + [path(f"{name}.ll"), "-o", path(f"{name}.o")],
        check=True,
    )
    return path(f"{name}.o")


objs = [lower(TEAM.format(n=n), n, f"team{n}") for n in (2, 3)]
with open(path("concurrent.c"), "w") as f:
    f.write(
        CONCURRENT.replace(
            "#include <stdio.h>", "#include <stdio.h>\n#include <stdlib.h>"
        )
    )
pool = os.environ.get(
    "POOL_SOURCE",
    os.path.join(SRC, "runtime", "threadpool", "BuddyThreadPool.c"),
)
subprocess.run(
    [
        cc,
        "-O2",
        path("concurrent.c"),
        *objs,
        pool,
        "-lpthread",
        "-o",
        path("concurrent"),
    ],
    check=True,
)
env = dict(os.environ)
env.pop("BUDDY_THREAD_POOL_THREADS", None)
run = subprocess.run(
    [path("concurrent"), str(cpus)], capture_output=True, text=True, env=env
)
print(run.stdout.strip(), run.stderr)
# CHECK: interleaved: A pushed 2, team as expected 1; B pushed 3, team as expected 1
# CHECK-NEXT: concurrent: 2-thread calls with a wrong team 0, 3-thread calls with a wrong team 0
