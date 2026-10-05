//===- BuddyArena.c - Per-call bump arena for model libraries -------------===//
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
// "arena": true (docs/ModelMemoryOptions.md). The model is then lowered with
// -finalize-memref-to-llvm=use-generic-functions=true and without the buffer
// deallocation passes, so every buffer it allocates comes from the functions
// below: allocations bump a pointer in one reserved address range, frees do
// nothing, and the generated ModelSession calls buddy_arena_reset() before
// each forward call, after it has copied out the results of the previous one.
//
// The arena holds all the buffers of one call at once: its size is the sum of
// the allocations of a call, not their peak. Only the pages that are touched
// take memory; they stay mapped and are reused by the next calls.
//
// Environment:
//   BUDDY_ARENA_RESERVE_MB   address range reserved (default 16384)
//   BUDDY_ARENA_PREFAULT_MB  pages faulted in when the library is loaded
//                            instead of during the first call (default 512)
//   BUDDY_ARENA_STATS=1      print the largest call's usage at exit
//
//===----------------------------------------------------------------------===//

#define _GNU_SOURCE
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

#define HIDDEN __attribute__((visibility("hidden")))

static char *arenaBase;
static size_t arenaSize;
static _Atomic size_t arenaUsed;
static size_t arenaHigh;

static size_t envMiB(const char *name, size_t fallback) {
  const char *e = getenv(name);
  return (e && *e ? (size_t)strtoull(e, NULL, 10) : fallback) << 20;
}

static void arenaStats(void) {
  size_t u = atomic_load(&arenaUsed);
  if (u > arenaHigh)
    arenaHigh = u;
  fprintf(stderr, "[BuddyArena] largest call: %zu MiB of %zu MiB reserved\n",
          arenaHigh >> 20, arenaSize >> 20);
}

// Reserves the range when the library is loaded, before any forward call
// (and so before any OpenMP thread allocates).
__attribute__((constructor)) static void arenaInit(void) {
  arenaSize = envMiB("BUDDY_ARENA_RESERVE_MB", 16384);
  void *p = mmap(NULL, arenaSize, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (p == MAP_FAILED) {
    perror("[BuddyArena] mmap");
    abort();
  }
#ifdef MADV_HUGEPAGE
  madvise(p, arenaSize, MADV_HUGEPAGE);
#endif
  // Otherwise the first call faults its pages in, and the page faults of
  // its threads serialize in the kernel (SpacemiT K3, int4 DeepSeek R1: the
  // first 64-token prefill call takes 0.12 s longer, of 0.57 s). 512 MiB
  // covers the calls of the int4 models; more is faulted in when needed.
  size_t prefault = envMiB("BUDDY_ARENA_PREFAULT_MB", 512);
  if (prefault > arenaSize)
    prefault = arenaSize;
#ifdef MADV_POPULATE_WRITE
  if (prefault)
    madvise(p, prefault, MADV_POPULATE_WRITE); // best effort
#endif
  const char *stats = getenv("BUDDY_ARENA_STATS");
  if (stats && stats[0] == '1')
    atexit(arenaStats);
  arenaBase = (char *)p;
}

// Called by the session before each forward call: the buffers of the
// previous call are dead.
void buddy_arena_reset(void) {
  size_t u = atomic_exchange(&arenaUsed, 0);
  if (u > arenaHigh)
    arenaHigh = u;
}

HIDDEN void *_mlir_memref_to_llvm_aligned_alloc(size_t alignment, size_t size) {
  if (alignment < 64)
    alignment = 64;
  // Room for the alignment, in multiples of 64 bytes so that the next
  // allocation starts 64-byte aligned.
  size_t need = (size + alignment + 63) & ~(size_t)63;
  size_t offset = atomic_fetch_add(&arenaUsed, need);
  if (offset + need > arenaSize) {
    fprintf(stderr,
            "[BuddyArena] out of space: a forward call needs more than %zu "
            "MiB (set BUDDY_ARENA_RESERVE_MB)\n",
            arenaSize >> 20);
    abort();
  }
  uintptr_t p = (uintptr_t)(arenaBase + offset);
  return (void *)((p + alignment - 1) & ~(uintptr_t)(alignment - 1));
}

HIDDEN void *_mlir_memref_to_llvm_alloc(size_t size) {
  return _mlir_memref_to_llvm_aligned_alloc(64, size);
}

HIDDEN void _mlir_memref_to_llvm_free(void *ptr) { (void)ptr; }
