//===- nr_memory.c - NR runtime memory and cache helpers ------------------===//
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
// NR RA-side memory support.
//
// Split out of nr_runtime.c so the platform runtime (NH/RA scheduling, console
// forwarding, traps, result report) is separate from the runtime library the
// lowered operators call.  This file owns:
//
//   - the low-DDR bump heap and its mark/reset contract,
//   - the scalar memcpy/memmove/memset/memcmp helpers,
//   - the MLIR memrefCopy ABI (rank <= 8, arbitrary strided element copies).
//
// The exported names and their semantics are unchanged: free() does not
// reclaim, aligned_alloc() stops with a FAIL report on exhaustion, and both are
// distinct from the bare backend's allocator, which is not unified with this
// one.  Fatal reports go through nr_finish()/abort() from nr_runtime.c.
//

#include "nr_runtime.h"
#include "nr_runtime_internal.h"

extern unsigned char __heap_start[], __heap_end[];
static uintptr_t heap_cursor;

// Ordinary scalar memory operations avoid unsupported vector spills/CSRs.
typedef uint64_t CopyWord __attribute__((may_alias));
void *memcpy(void *destination, const void *source, size_t count) {
  unsigned char *out = destination;
  const unsigned char *in = source;
  if ((((uintptr_t)out | (uintptr_t)in) & 7u) == 0) {
    while (count >= 8) {
      *(CopyWord *)out = *(const CopyWord *)in;
      out += 8;
      in += 8;
      count -= 8;
    }
  }
  while (count--)
    *out++ = *in++;
  return destination;
}
void *memmove(void *destination, const void *source, size_t count) {
  unsigned char *out = destination;
  const unsigned char *in = source;
  if ((uintptr_t)out > (uintptr_t)in &&
      (uintptr_t)out - (uintptr_t)in < count) {
    while (count) {
      --count;
      out[count] = in[count];
    }
    return destination;
  }
  return memcpy(destination, source, count);
}
void *memset(void *destination, int value, size_t count) {
  unsigned char *out = destination;
  while (count && ((uintptr_t)out & 7u)) {
    *out++ = (unsigned char)value;
    --count;
  }
  uint64_t word = (uint64_t)(unsigned char)value * UINT64_C(0x0101010101010101);
  while (count >= 8) {
    *(CopyWord *)out = word;
    out += 8;
    count -= 8;
  }
  while (count--)
    *out++ = (unsigned char)value;
  return destination;
}
int memcmp(const void *lhs, const void *rhs, size_t count) {
  const unsigned char *left = lhs, *right = rhs;
  while (count--) {
    if (*left != *right)
      return (int)*left - (int)*right;
    ++left;
    ++right;
  }
  return 0;
}
uintptr_t nr_heap_mark(void) {
  return heap_cursor ? heap_cursor : (uintptr_t)__heap_start;
}
void nr_heap_reset(uintptr_t mark) {
  if (mark < (uintptr_t)__heap_start || mark > nr_heap_mark()) {
    nr_puts("[nr] invalid heap mark: FAIL\r\n");
    nr_finish(1);
  }
  heap_cursor = mark;
}
static __attribute__((noreturn)) void heap_exhausted(size_t size) {
  nr_puts("[nr] heap exhausted: FAIL bytes=0x");
  nr_hex64(size);
  nr_puts(" cursor=0x");
  nr_hex64(nr_heap_mark());
  nr_puts(" limit=0x");
  nr_hex64((uintptr_t)__heap_end);
  nr_puts("\r\n");
  nr_finish(1);
}
void *aligned_alloc(size_t alignment, size_t size) {
  if (!alignment || (alignment & (alignment - 1)))
    return 0;
  uintptr_t cursor = heap_cursor ? heap_cursor : (uintptr_t)__heap_start;
  if (cursor > UINTPTR_MAX - (alignment - 1))
    heap_exhausted(size);
  cursor = (cursor + alignment - 1) & ~(uintptr_t)(alignment - 1);
  if (cursor > (uintptr_t)__heap_end || size > (uintptr_t)__heap_end - cursor)
    heap_exhausted(size);
  heap_cursor = cursor + size;
  return (void *)cursor;
}
void *malloc(size_t size) { return aligned_alloc(64, size ? size : 1); }
void free(void *pointer) { (void)pointer; }
void *calloc(size_t count, size_t size) {
  if (count && size > SIZE_MAX / count)
    return 0;
  void *result = malloc(count * size);
  if (result)
    memset(result, 0, count * size);
  return result;
}

// Standard MLIR CRunner ABI; rank <= 8, arbitrary strided element copies.
typedef struct {
  int64_t rank;
  void *descriptor;
} UnrankedMemRef;
typedef struct {
  char *base, *data;
  int64_t offset;
  int64_t dimensions[];
} RankedMemRef;
void memrefCopy(int64_t element_size, UnrankedMemRef *source,
                UnrankedMemRef *destination) {
  if (!source || !destination || source->rank != destination->rank ||
      source->rank < 0 || source->rank > 8 || element_size <= 0)
    abort();
  int64_t rank = source->rank;
  RankedMemRef *src = source->descriptor, *dst = destination->descriptor;
  if (!src || !dst || !src->data || !dst->data)
    abort();
  uint64_t elements = 1;
  for (int64_t i = 0; i < rank; ++i) {
    if (src->dimensions[i] != dst->dimensions[i] || src->dimensions[i] < 0)
      abort();
    if (!src->dimensions[i])
      return;
    if (elements > (uint64_t)INT64_MAX / (uint64_t)src->dimensions[i])
      abort();
    elements *= (uint64_t)src->dimensions[i];
  }
  if (elements > (uint64_t)INT64_MAX / (uint64_t)element_size)
    abort();
  int contiguous = 1;
  int64_t stride = 1;
  for (int64_t d = rank; d-- > 0;) {
    if (src->dimensions[d] > 1 && (src->dimensions[rank + d] != stride ||
                                   dst->dimensions[rank + d] != stride))
      contiguous = 0;
    stride *= src->dimensions[d];
  }
  if (contiguous) {
    memcpy(dst->data + dst->offset * element_size,
           src->data + src->offset * element_size, elements * element_size);
    return;
  }
  for (uint64_t i = 0; i < elements; ++i) {
    uint64_t remaining = i;
    int64_t src_index = src->offset, dst_index = dst->offset;
    for (int64_t d = rank; d-- > 0;) {
      uint64_t coordinate = remaining % (uint64_t)src->dimensions[d];
      remaining /= (uint64_t)src->dimensions[d];
      src_index += (int64_t)coordinate * src->dimensions[rank + d];
      dst_index += (int64_t)coordinate * dst->dimensions[rank + d];
    }
    memcpy(dst->data + dst_index * element_size,
           src->data + src_index * element_size, element_size);
  }
}
