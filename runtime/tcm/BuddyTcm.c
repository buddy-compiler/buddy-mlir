//===- BuddyTcm.c - Core-pair TCM of the SpacemiT K3 A100 cores -----------===//
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
// "prefill_ime": true (docs/K3DeepSeekR1.md). The IME prefill tiles of
// graph/transform/k3_w4.py read their activations from the TCM of the core
// pair they run on, which the K3 kernel exposes as /dev/tcm
// (drivers/misc/tcm.c): blocks of SRAM, each with the CPUs it belongs to.
// On the K3 the 8 A100 cores have 8 blocks of 384 KiB, two per core pair
// (CPUs 8-9, 10-11, 12-13, 14-15). A core loads 1 KiB from its pair's TCM in
// ~9 ns whatever the other cores do, while cached loads share a cluster path
// that takes ~40 ns per KiB when 3 or 4 cores of a cluster load; the TCM of
// another pair is uncached (microseconds per KiB).
//
// A "pair region" is the TCM of one core pair: its blocks, contiguous in the
// mapping. The kernels copy the activations of a call into every region
// (buddy_tcm_pair) and each tile reads the copy of its own pair
// (buddy_tcm_here). Both return 0 when there is no TCM, it is not big
// enough, or another process holds it: the kernels then read the
// activations from memory, with the same results.
//
// The TCM is not shared between processes: the first model library that maps
// it takes an exclusive flock on /dev/tcm; others go without. Programs that
// use the TCM otherwise (e.g. SpacemiT's spine runtime) must not run at the
// same time.
//
// Environment:
//   BUDDY_TCM=0   do not use the TCM
//
//===----------------------------------------------------------------------===//

#define _GNU_SOURCE
#include <fcntl.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#define API __attribute__((visibility("hidden")))
#define MAX_REGIONS 16
#define MAX_CPUS 64

// drivers/misc/tcm.c
typedef struct {
  void *base;
  size_t block_size;
  size_t block_num;
} TcmInfo;
typedef struct {
  uint32_t block_id;
  uint32_t reserved;
  uint64_t phys;
  uint64_t size;
  uint64_t cpu_affinity_mask;
} TcmBlockInfo;
#define TCM_INFO_GET _IOR('c', 7, int)
#define TCM_BLOCK_INFO_GET _IOWR('c', 9, int)

static char *regionBase[MAX_REGIONS];
static size_t regionBytes[MAX_REGIONS];
static int numRegions;
static int cpuRegion[MAX_CPUS]; // -1: no TCM for this CPU

__attribute__((constructor)) static void tcmInit(void) {
  for (int c = 0; c < MAX_CPUS; c++)
    cpuRegion[c] = -1;
  const char *env = getenv("BUDDY_TCM");
  if (env && env[0] == '0')
    return;
  int fd = open("/dev/tcm", O_RDWR | O_CLOEXEC);
  if (fd < 0)
    return;
  TcmInfo info;
  if (ioctl(fd, TCM_INFO_GET, &info) || !info.block_num ||
      flock(fd, LOCK_EX | LOCK_NB)) {
    close(fd);
    return;
  }
  // Block i is at i * block_size in the mapping. Consecutive blocks of the
  // same CPUs form a region.
  size_t total = info.block_size * info.block_num;
  char *map = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (map == MAP_FAILED) {
    close(fd);
    return;
  }
  uint64_t lastMask = 0;
  for (uint32_t b = 0; b < info.block_num; b++) {
    TcmBlockInfo block = {.block_id = b};
    if (ioctl(fd, TCM_BLOCK_INFO_GET, &block) || !block.cpu_affinity_mask ||
        block.size != info.block_size)
      goto fail;
    if (b > 0 && block.cpu_affinity_mask == lastMask) {
      regionBytes[numRegions - 1] += block.size;
    } else {
      if (numRegions == MAX_REGIONS)
        goto fail;
      regionBase[numRegions] = map + b * info.block_size;
      regionBytes[numRegions] = block.size;
      for (int c = 0; c < MAX_CPUS; c++)
        if (block.cpu_affinity_mask >> c & 1)
          cpuRegion[c] = numRegions;
      numRegions++;
    }
    lastMask = block.cpu_affinity_mask;
  }
  // The fd stays open: it holds the lock.
  return;
fail:
  munmap(map, total);
  close(fd);
  numRegions = 0;
  for (int c = 0; c < MAX_CPUS; c++)
    cpuRegion[c] = -1;
}

// The kernels call buddy_tcm_pair / buddy_tcm_here; their lowering
// (-llvm-request-c-wrappers) makes those call the _mlir_ciface_ functions.

// The address of pair region `region` if it exists, the kernels copy into at
// most `regions` regions and it has `bytes` bytes; else 0.
API int64_t _mlir_ciface_buddy_tcm_pair(int64_t region, int64_t regions,
                                        int64_t bytes) {
  if (region < 0 || region >= numRegions || region >= regions ||
      (size_t)bytes > regionBytes[region])
    return 0;
  return (int64_t)(intptr_t)regionBase[region];
}

// buddy_tcm_pair of the region of the CPU the caller runs on.
API int64_t _mlir_ciface_buddy_tcm_here(int64_t regions, int64_t bytes) {
  int cpu = sched_getcpu();
  if (cpu < 0 || cpu >= MAX_CPUS)
    return 0;
  return _mlir_ciface_buddy_tcm_pair(cpuRegion[cpu], regions, bytes);
}
