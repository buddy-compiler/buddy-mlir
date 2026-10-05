//===- BuddyThreadPool.c - Pinned thread pool behind the OpenMP calls ----===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// Linked into the model library of a buddy-codegen model whose spec sets
// "thread_pool": true (docs/ModelThreadPool.md), in place of libomp.
//
// The model's parallel loops are lowered scf.parallel -> omp.parallel /
// omp.wsloop -> calls to the libomp entry points below, which is all the
// generated code calls. They run the parallel regions on one pool of threads,
// pinned one per allowed CPU, that wait for work spinning: a region costs a
// few microseconds less than with libomp, which counts when a decode step
// runs hundreds of short regions. The symbols are hidden, so the model
// library binds to them when it is linked and does not need libomp.
//
//   - The pool has as many threads as the largest parallel region so far
//     asked for (num_threads of the lowering), and at most one per CPU the
//     process may run on: the first region starts it, a larger one adds
//     threads. The caller of a region is its thread 0; the first caller is
//     pinned too.
//   - Static schedules only (what omp.wsloop without a schedule uses).
//   - Nested parallel regions run serially, as with libomp by default.
//   - One parallel region at a time: regions forked by different application
//     threads wait for each other. The thread count a thread pushes applies
//     to its own next region.
//
// Environment:
//   BUDDY_THREAD_POOL_THREADS  at most this many threads
//   OMP_STACKSIZE              stack size of the pool threads (default 256M,
//                              virtual)
//
//===----------------------------------------------------------------------===//

#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define API __attribute__((visibility("hidden")))
#define MAX_ARGS 24

typedef struct ident ident_t;
typedef void (*microtask_t)(int32_t *, int32_t *, ...);

// The pool; changed with forkLock held and no region running.
static pthread_mutex_t forkLock = PTHREAD_MUTEX_INITIALIZER;
static int poolSize;  // threads in the pool, thread 0 (a region's caller) too
static int poolLimit; // at most this many: allowed CPUs, environment
static int *poolCpus; // the CPU of each thread (poolLimit entries)
static pthread_t *poolThreads;
static unsigned *poolStart; // the job generation a worker starts after
static size_t poolStack;

static _Thread_local int pushedThreads; // num_threads of this thread's next
                                        // region

static _Thread_local int threadNum; // in the current team
static _Thread_local int teamSize = 1;
static _Thread_local int inRegion;

static struct {
  microtask_t fn;
  int argc, threads;
  void *args[MAX_ARGS];
} job;
static _Atomic unsigned jobGeneration;
static _Atomic int quitting;
static _Atomic int finished;
static _Atomic int barrierCount;
static _Atomic unsigned barrierGeneration;

static void invoke(microtask_t fn, int argc, void **a) {
  int32_t gtid = threadNum, btid = threadNum;
  switch (argc) {
  case 0:
    fn(&gtid, &btid);
    break;
  case 1:
    fn(&gtid, &btid, a[0]);
    break;
  case 2:
    fn(&gtid, &btid, a[0], a[1]);
    break;
  case 3:
    fn(&gtid, &btid, a[0], a[1], a[2]);
    break;
  case 4:
    fn(&gtid, &btid, a[0], a[1], a[2], a[3]);
    break;
  case 5:
    fn(&gtid, &btid, a[0], a[1], a[2], a[3], a[4]);
    break;
  case 6:
    fn(&gtid, &btid, a[0], a[1], a[2], a[3], a[4], a[5]);
    break;
  case 7:
    fn(&gtid, &btid, a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
    break;
  case 8:
    fn(&gtid, &btid, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
    break;
  default: {
    // Up to MAX_ARGS pointers: pass them all, the microtask reads argc.
    void *x[MAX_ARGS] = {0};
    memcpy(x, a, sizeof(void *) * argc);
    fn(&gtid, &btid, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9],
       x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18], x[19],
       x[20], x[21], x[22], x[23]);
  }
  }
}

static void pin(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  sched_setaffinity(0, sizeof set, &set);
}

static void *worker(void *arg) {
  int id = (int)(intptr_t)arg;
  pin(poolCpus[id]);
  unsigned seen = poolStart[id];
  for (;;) {
    unsigned generation;
    long spins = 0;
    while ((generation = atomic_load_explicit(&jobGeneration,
                                              memory_order_acquire)) == seen)
      if (++spins > (1L << 24)) // idle (e.g. between requests): back off
        usleep(50);
    seen = generation;
    if (atomic_load_explicit(&quitting, memory_order_relaxed))
      return NULL;
    if (id < job.threads) {
      threadNum = id, teamSize = job.threads, inRegion = 1;
      invoke(job.fn, job.argc, job.args);
      inRegion = 0;
    }
    // Every worker reports, so that the next region does not rewrite job
    // while a worker that sat this one out still reads it.
    atomic_fetch_add_explicit(&finished, 1, memory_order_release);
  }
}

static void initPool(void) {
  cpu_set_t allowed;
  sched_getaffinity(0, sizeof allowed, &allowed);
  int limit = CPU_SETSIZE;
  const char *env = getenv("BUDDY_THREAD_POOL_THREADS");
  if (env && atoi(env) > 0)
    limit = atoi(env);
  poolCpus = malloc(sizeof(int) * CPU_SETSIZE);
  for (int c = 0; c < CPU_SETSIZE && poolLimit < limit; c++)
    if (CPU_ISSET(c, &allowed))
      poolCpus[poolLimit++] = c;
  if (poolLimit == 0)
    poolCpus[poolLimit++] = sched_getcpu();
  poolThreads = malloc(sizeof(pthread_t) * poolLimit);
  poolStart = malloc(sizeof(unsigned) * poolLimit);
  pin(poolCpus[0]);
  poolSize = 1;
  // Virtual size: only the touched pages take memory.
  poolStack = 256ull << 20;
  env = getenv("OMP_STACKSIZE");
  if (env) {
    char *end;
    poolStack = strtoull(env, &end, 10);
    poolStack <<= (*end == 'G' || *end == 'g')   ? 30
                  : (*end == 'K' || *end == 'k') ? 10
                  : (*end == 'B' || *end == 'b') ? 0
                                                 : 20;
  }
}

// Threads up to n (at most poolLimit). With forkLock held and no region
// running: a new worker runs the jobs after the current generation.
static void growPool(int n) {
  if (n > poolLimit)
    n = poolLimit;
  if (n <= poolSize)
    return;
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, poolStack);
  unsigned generation =
      atomic_load_explicit(&jobGeneration, memory_order_relaxed);
  for (int i = poolSize; i < n; i++) {
    poolStart[i] = generation;
    if (pthread_create(&poolThreads[i], &attr, worker, (void *)(intptr_t)i)) {
      perror("[BuddyThreadPool] pthread_create");
      abort();
    }
  }
  pthread_attr_destroy(&attr);
  poolSize = n;
}

// The model library may be unloaded (dlclose): stop the workers before their
// code is unmapped.
__attribute__((destructor)) static void stopPool(void) {
  if (poolSize < 2)
    return;
  atomic_store_explicit(&quitting, 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&jobGeneration, 1, memory_order_release);
  for (int i = 1; i < poolSize; i++)
    pthread_join(poolThreads[i], NULL);
  poolSize = 0;
}

API int32_t __kmpc_global_thread_num(ident_t *loc) {
  (void)loc;
  return threadNum;
}

API void __kmpc_push_num_threads(ident_t *loc, int32_t gtid, int32_t n) {
  (void)loc, (void)gtid;
  if (!inRegion)
    pushedThreads = n;
}

API void __kmpc_fork_call(ident_t *loc, int32_t argc, microtask_t fn, ...) {
  (void)loc;
  if (argc > MAX_ARGS) {
    fprintf(stderr, "[BuddyThreadPool] %d parallel region arguments (max %d)\n",
            argc, MAX_ARGS);
    abort();
  }
  void *args[MAX_ARGS];
  va_list ap;
  va_start(ap, fn);
  for (int i = 0; i < argc; i++)
    args[i] = va_arg(ap, void *);
  va_end(ap);
  if (inRegion) { // nested: serially, on this thread
    int savedNum = threadNum, savedSize = teamSize;
    threadNum = 0, teamSize = 1;
    invoke(fn, argc, args);
    threadNum = savedNum, teamSize = savedSize;
    return;
  }
  int n = pushedThreads > 0 ? pushedThreads : CPU_SETSIZE;
  pushedThreads = 0;
  pthread_mutex_lock(&forkLock);
  if (!poolSize)
    initPool();
  growPool(n);
  if (n > poolSize)
    n = poolSize;
  job.fn = fn, job.argc = argc, job.threads = n;
  memcpy(job.args, args, sizeof(void *) * argc);
  atomic_store_explicit(&finished, 0, memory_order_relaxed);
  atomic_fetch_add_explicit(&jobGeneration, 1, memory_order_release);
  threadNum = 0, teamSize = n, inRegion = 1;
  invoke(fn, argc, args);
  inRegion = 0, teamSize = 1;
  while (atomic_load_explicit(&finished, memory_order_acquire) != poolSize - 1)
    ;
  pthread_mutex_unlock(&forkLock);
}

API void __kmpc_barrier(ident_t *loc, int32_t gtid) {
  (void)loc, (void)gtid;
  if (teamSize <= 1)
    return;
  unsigned generation =
      atomic_load_explicit(&barrierGeneration, memory_order_relaxed);
  if (atomic_fetch_add_explicit(&barrierCount, 1, memory_order_acq_rel) ==
      teamSize - 1) {
    atomic_store_explicit(&barrierCount, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&barrierGeneration, 1, memory_order_release);
  } else {
    while (atomic_load_explicit(&barrierGeneration, memory_order_acquire) ==
           generation)
      ;
  }
}

enum { kSchedStaticChunked = 33, kSchedStatic = 34 };

API void __kmpc_for_static_init_8u(ident_t *loc, int32_t gtid, int32_t sched,
                                   int32_t *plast, uint64_t *plower,
                                   uint64_t *pupper, int64_t *pstride,
                                   int64_t incr, int64_t chunk) {
  (void)loc, (void)gtid;
  int n = teamSize, id = threadNum;
  uint64_t lower = *plower, upper = *pupper;
  if (incr != 1) {
    fprintf(stderr, "[BuddyThreadPool] unsupported loop increment %ld\n",
            (long)incr);
    abort();
  }
  uint64_t trip = upper - lower + 1; // 0 for an empty loop (upper = lower - 1)
  if (sched == kSchedStatic) {
    uint64_t each = trip / n, extra = trip % n;
    uint64_t first = id * each + ((uint64_t)id < extra ? (uint64_t)id : extra);
    uint64_t count = each + ((uint64_t)id < extra);
    *plower = lower + first;
    *pupper = lower + first + count - 1; // count 0: no iterations
    if (plast)
      *plast = trip <= (uint64_t)n ? (uint64_t)id == trip - 1 : id == n - 1;
    *pstride = trip;
  } else if (sched == kSchedStaticChunked) {
    if (chunk < 1)
      chunk = 1;
    *plower = lower + (uint64_t)id * chunk;
    *pupper = *plower + chunk - 1;
    *pstride = chunk * n;
    if (plast)
      *plast = (int32_t)(((trip - 1) / chunk) % n) == id;
  } else {
    fprintf(stderr, "[BuddyThreadPool] unsupported schedule %d\n", sched);
    abort();
  }
}

API void __kmpc_for_static_fini(ident_t *loc, int32_t gtid) {
  (void)loc, (void)gtid;
}
