//===- bare_runtime_memref.c - Bare-runtime memrefCopy implementation -----===//
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
// MLIR memrefCopy ABI of the bare single-hart runtime.
//
// The lowered kernels call `memrefCopy(elemSize, UnrankedMemRef*, ...)` with
// the standard runner descriptor layout.  The walker below is a separate
// responsibility from the scalar memory helpers, so it lives in its own
// translation unit and keeps the descriptor layout, the offset/stride handling
// and the bounds checks unchanged.
//

#include "bare_runtime_internal.h"
#include <stddef.h>
#include <stdint.h>

#if BARE_RUNTIME_TRACE_MEMREF_COPY
#define TRACE_MEMREF_COPY(msg) print_uart(msg)
#else
#define TRACE_MEMREF_COPY(msg) ((void)0)
#endif

typedef struct {
  int64_t rank;
  void *descriptor;
} UnrankedMemRef;

typedef struct {
  char *base;
  char *data;
  int64_t offset;
  int64_t sizes_and_strides[];
} RankedMemRef;

#if BARE_RUNTIME_TRACE_MEMREF_COPY
static void trace_memref_i64(const char *name, int64_t value) {
  TRACE_MEMREF_COPY(name);
  bare_trace_hex_u64((uint64_t)value);
}

static void trace_memref_ptr(const char *name, const void *ptr) {
  TRACE_MEMREF_COPY(name);
  bare_trace_hex_u64((uint64_t)(uintptr_t)ptr);
}

static void trace_ranked_memref(const char *label, const RankedMemRef *memref,
                                int64_t rank) {
  TRACE_MEMREF_COPY(label);
  trace_memref_ptr(" base=", memref->base);
  trace_memref_ptr(" data=", memref->data);
  trace_memref_i64(" off=", memref->offset);
  for (int64_t dim = 0; dim < rank; ++dim) {
    trace_memref_i64(" size=", memref->sizes_and_strides[dim]);
    trace_memref_i64(" stride=", memref->sizes_and_strides[rank + dim]);
  }
  TRACE_MEMREF_COPY("\r\n");
}
#endif

#if BARE_RUNTIME_ENABLE_MEMREF_COPY
// The aligned case covers the FP32 buffers produced by MLIR.  may_alias keeps
// these bitwise word copies valid for float, integer, and byte allocations.
typedef uint32_t BareMemRefCopyWord __attribute__((__may_alias__));

static void contiguous_memref_copy(char *dst, const char *src, int64_t bytes) {
  if ((((uintptr_t)dst | (uintptr_t)src) & 3u) == 0) {
    BareMemRefCopyWord *out = (BareMemRefCopyWord *)dst;
    const BareMemRefCopyWord *in = (const BareMemRefCopyWord *)src;
    int64_t words = bytes / 4;
    for (int64_t i = 0; i < words; ++i)
      out[i] = in[i];
    dst += words * 4;
    src += words * 4;
    bytes -= words * 4;
  }
  bare_byte_copy(dst, src, bytes);
}

void memrefCopy(int64_t elemSize, UnrankedMemRef *srcArg,
                UnrankedMemRef *dstArg) {
  TRACE_MEMREF_COPY("rt: copy begin elem=");
#if BARE_RUNTIME_TRACE_MEMREF_COPY
  bare_trace_hex_u64((uint64_t)elemSize);
  TRACE_MEMREF_COPY("\r\n");
#endif

  if (!srcArg || !dstArg || srcArg->rank != dstArg->rank || elemSize <= 0) {
    TRACE_MEMREF_COPY("rt: copy skipped\r\n");
    return;
  }

  int64_t rank = srcArg->rank;
  if (rank < 0 || rank > BARE_RUNTIME_MEMREF_COPY_MAX_RANK) {
    TRACE_MEMREF_COPY("rt: copy skipped rank\r\n");
    return;
  }

#if BARE_RUNTIME_TRACE_MEMREF_COPY
  trace_memref_i64("rt: copy rank=", rank);
  TRACE_MEMREF_COPY("\r\n");
#endif

  RankedMemRef *src = (RankedMemRef *)srcArg->descriptor;
  RankedMemRef *dst = (RankedMemRef *)dstArg->descriptor;
  if (!src || !dst || !src->data || !dst->data) {
    TRACE_MEMREF_COPY("rt: copy skipped\r\n");
    return;
  }

#if BARE_RUNTIME_TRACE_MEMREF_COPY
  trace_ranked_memref("rt: copy src", src, rank);
  trace_ranked_memref("rt: copy dst", dst, rank);
#endif

  int64_t *src_sizes = src->sizes_and_strides;
  int64_t *src_strides = src_sizes + rank;
  int64_t *dst_sizes = dst->sizes_and_strides;
  int64_t *dst_strides = dst_sizes + rank;
  uint64_t elements = 1;
  for (int64_t dim = 0; dim < rank; ++dim) {
    if (src_sizes[dim] != dst_sizes[dim] || src_sizes[dim] < 0) {
      TRACE_MEMREF_COPY("rt: copy skipped shape\r\n");
      return;
    }
    if (src_sizes[dim] == 0) {
      TRACE_MEMREF_COPY("rt: copy end empty\r\n");
      return;
    }
    if (elements > (uint64_t)BARE_RUNTIME_MEMREF_COPY_MAX_ELEMS /
                       (uint64_t)src_sizes[dim]) {
      TRACE_MEMREF_COPY("rt: copy skipped too large\r\n");
      return;
    }
    elements *= (uint64_t)src_sizes[dim];
  }

  if (elements > (uint64_t)INT64_MAX / (uint64_t)elemSize) {
    TRACE_MEMREF_COPY("rt: copy skipped byte size\r\n");
    return;
  }

  // Singleton dimensions address only coordinate zero, so their strides do
  // not affect contiguity.  Negative strides on non-singleton dimensions
  // retain the generic path below, as do transposes and slices with gaps.
  int contiguous = 1;
  int64_t expected_stride = 1;
  for (int64_t dim = rank; dim > 0; --dim) {
    int64_t current = dim - 1;
    if (src_sizes[current] > 1 && (src_strides[current] != expected_stride ||
                                   dst_strides[current] != expected_stride)) {
      contiguous = 0;
      break;
    }
    expected_stride *= src_sizes[current];
  }
  if (contiguous) {
    contiguous_memref_copy(dst->data + dst->offset * elemSize,
                           src->data + src->offset * elemSize,
                           (int64_t)elements * elemSize);
    TRACE_MEMREF_COPY("rt: copy end contiguous\r\n");
    return;
  }

  for (uint64_t linear = 0; linear < elements; ++linear) {
    uint64_t remaining = linear;
    int64_t src_index = src->offset;
    int64_t dst_index = dst->offset;
    for (int64_t dim = rank; dim > 0; --dim) {
      int64_t current = dim - 1;
      uint64_t coordinate = remaining % (uint64_t)src_sizes[current];
      remaining /= (uint64_t)src_sizes[current];
      src_index += (int64_t)coordinate * src_strides[current];
      dst_index += (int64_t)coordinate * dst_strides[current];
    }
    bare_byte_copy(dst->data + dst_index * elemSize,
                   src->data + src_index * elemSize, elemSize);
  }

  TRACE_MEMREF_COPY("rt: copy end\r\n");
}
#endif
