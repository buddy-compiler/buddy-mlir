//===- bare_runtime_memory.c - Bare-runtime heap and string helpers -------===//
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
// Bare single-hart memory support.
//
// Owns the private bump heap (optional reclaiming free list), the allocator
// entry points and the scalar memory/string helpers used by the freestanding
// samples.  `_init` and the startup hooks stay in bare_runtime.c; sharing the
// heap state between the two would only hide the module boundary, so the entry
// point reaches the allocator through the public bare_runtime.h API.
//

#include "bare_runtime_internal.h"
#include <stddef.h>
#include <stdint.h>

#if BARE_RUNTIME_TRACE_MALLOC
#define TRACE_MALLOC(msg) print_uart(msg)
#else
#define TRACE_MALLOC(msg) ((void)0)
#endif

#if BARE_RUNTIME_TRACE_MEMCPY
#define TRACE_MEMCPY(msg) print_uart(msg)
#else
#define TRACE_MEMCPY(msg) ((void)0)
#endif

static uint8_t heap_buf[BARE_RUNTIME_HEAP_SIZE] __attribute__((
    aligned(BARE_RUNTIME_HEAP_ALIGN), section(".bare_runtime_heap")));
static size_t heap_offset = 0;
static size_t heap_floor = 0;
#if BARE_RUNTIME_RECLAIM_FREE
typedef struct {
  size_t size;
  size_t is_free;
} BareHeapBlock;
#define BARE_HEAP_HEADER_SIZE                                                  \
  ((sizeof(BareHeapBlock) + BARE_RUNTIME_HEAP_ALIGN - 1u) &                    \
   ~(size_t)(BARE_RUNTIME_HEAP_ALIGN - 1u))
#endif

void *bare_runtime_heap_base(void) { return heap_buf; }

size_t bare_runtime_heap_offset(void) { return heap_offset; }

void bare_runtime_preserve_heap(void) { heap_floor = heap_offset; }

void bare_runtime_reset_heap(void) { heap_offset = heap_floor; }

void *malloc(size_t size) {
#if BARE_RUNTIME_TRACE_MALLOC
  TRACE_MALLOC("rt: malloc size=");
  bare_trace_hex_u64((uint64_t)size);
#endif
  size = (size + (BARE_RUNTIME_HEAP_ALIGN - 1u)) &
         ~(size_t)(BARE_RUNTIME_HEAP_ALIGN - 1u);
#if BARE_RUNTIME_RECLAIM_FREE
  size_t block_offset = heap_floor;
  while (block_offset < heap_offset) {
    BareHeapBlock *block = (BareHeapBlock *)&heap_buf[block_offset];
    if (block->is_free && block->size >= size) {
      size_t remainder = block->size - size;
      if (remainder >= BARE_HEAP_HEADER_SIZE + BARE_RUNTIME_HEAP_ALIGN) {
        size_t next_offset = block_offset + BARE_HEAP_HEADER_SIZE + size;
        BareHeapBlock *next = (BareHeapBlock *)&heap_buf[next_offset];
        next->size = remainder - BARE_HEAP_HEADER_SIZE;
        next->is_free = 1;
        block->size = size;
      }
      block->is_free = 0;
      return &heap_buf[block_offset + BARE_HEAP_HEADER_SIZE];
    }
    block_offset += BARE_HEAP_HEADER_SIZE + block->size;
  }
  if (heap_offset + BARE_HEAP_HEADER_SIZE + size > sizeof(heap_buf))
    return (void *)0;
  BareHeapBlock *block = (BareHeapBlock *)&heap_buf[heap_offset];
  block->size = size;
  block->is_free = 0;
  void *ptr = &heap_buf[heap_offset + BARE_HEAP_HEADER_SIZE];
  heap_offset += BARE_HEAP_HEADER_SIZE + size;
  return ptr;
#else
  if (heap_offset + size > sizeof(heap_buf)) {
#if BARE_RUNTIME_TRACE_MALLOC
    TRACE_MALLOC(" failed off=");
    bare_trace_hex_u64((uint64_t)heap_offset);
    TRACE_MALLOC("\r\n");
#endif
    return (void *)0;
  }

  void *ptr = &heap_buf[heap_offset];
  heap_offset += size;
#if BARE_RUNTIME_TRACE_MALLOC
  TRACE_MALLOC(" ptr=");
  bare_trace_hex_u64((uint64_t)(uintptr_t)ptr);
  TRACE_MALLOC(" next=");
  bare_trace_hex_u64((uint64_t)heap_offset);
  TRACE_MALLOC("\r\n");
#endif
  return ptr;
#endif
}

void free(void *ptr) {
#if BARE_RUNTIME_RECLAIM_FREE
  if (!ptr || (uint8_t *)ptr < heap_buf + heap_floor + BARE_HEAP_HEADER_SIZE ||
      (uint8_t *)ptr >= heap_buf + heap_offset)
    return;
  size_t target = (size_t)((uint8_t *)ptr - heap_buf) - BARE_HEAP_HEADER_SIZE;
  size_t current = heap_floor;
  size_t previous = (size_t)-1;
  while (current < heap_offset && current != target) {
    BareHeapBlock *block = (BareHeapBlock *)&heap_buf[current];
    previous = current;
    current += BARE_HEAP_HEADER_SIZE + block->size;
  }
  if (current != target)
    return;
  BareHeapBlock *block = (BareHeapBlock *)&heap_buf[current];
  block->is_free = 1;
  size_t next_offset = current + BARE_HEAP_HEADER_SIZE + block->size;
  if (next_offset < heap_offset) {
    BareHeapBlock *next = (BareHeapBlock *)&heap_buf[next_offset];
    if (next->is_free)
      block->size += BARE_HEAP_HEADER_SIZE + next->size;
  }
  if (previous != (size_t)-1) {
    BareHeapBlock *prev = (BareHeapBlock *)&heap_buf[previous];
    if (prev->is_free) {
      prev->size += BARE_HEAP_HEADER_SIZE + block->size;
      current = previous;
      block = prev;
    }
  }
  if (current + BARE_HEAP_HEADER_SIZE + block->size == heap_offset)
    heap_offset = current;
#else
  (void)ptr;
#endif
}

void *memcpy(void *dst, const void *src, size_t size) {
#if BARE_RUNTIME_TRACE_MEMCPY
  TRACE_MEMCPY("rt: memcpy size=");
  bare_trace_hex_u64((uint64_t)size);
  TRACE_MEMCPY(" dst=");
  bare_trace_hex_u64((uint64_t)(uintptr_t)dst);
  TRACE_MEMCPY(" src=");
  bare_trace_hex_u64((uint64_t)(uintptr_t)src);
  TRACE_MEMCPY("\r\n");
#endif
  bare_byte_copy((char *)dst, (const char *)src, (int64_t)size);
  return dst;
}

void *memmove(void *dst, const void *src, size_t size) {
  char *d = (char *)dst;
  const char *s = (const char *)src;
  if (d <= s) {
    bare_byte_copy(d, s, (int64_t)size);
  } else {
    for (size_t i = size; i > 0; --i)
      d[i - 1] = s[i - 1];
  }
  return dst;
}

void *memset(void *dst, int value, size_t size) {
  char *d = (char *)dst;
  for (size_t i = 0; i < size; ++i)
    d[i] = (char)value;
  return dst;
}

size_t strlen(const char *text) {
  const char *end = text;
  while (*end)
    ++end;
  return (size_t)(end - text);
}

int strcmp(const char *lhs, const char *rhs) {
  while (*lhs && *lhs == *rhs) {
    ++lhs;
    ++rhs;
  }
  return (int)(unsigned char)*lhs - (int)(unsigned char)*rhs;
}
