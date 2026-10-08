#include "support.h"
#ifdef HOST_TEST
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

static const float guard_value = 12345.25f;

/* Host counterpart of common/nr/nr_runtime.c's rank-0..8 memrefCopy ABI.
 * Elementwise masks use rank-1 subviews; Linear masks also use rank-2.
 * This is data movement only, with no tensor arithmetic or allocation. */
#ifdef HOST_TEST
typedef struct { void *allocated, *aligned; int64_t offset, dimensions[]; } Ranked;
typedef struct { int64_t rank; Ranked *descriptor; } Unranked;
static void copy_bytes(void *destination, const void *source, size_t bytes) {
  memcpy(destination, source, bytes);
}
void memrefCopy(int64_t element_bytes, Unranked *source, Unranked *target) {
  if (!source || !target || source->rank != target->rank ||
      source->rank < 0 || source->rank > 8 || element_bytes <= 0)
    __builtin_trap();
  int64_t rank = source->rank;
  Ranked *s = source->descriptor, *d = target->descriptor;
  if (!s || !d || !s->aligned || !d->aligned) __builtin_trap();
  uint64_t elements = 1;
  for (int64_t i = 0; i < rank; ++i) {
    if (s->dimensions[i] != d->dimensions[i] || s->dimensions[i] < 0) __builtin_trap();
    if (!s->dimensions[i]) return;
    if (elements > (uint64_t)INT64_MAX / (uint64_t)s->dimensions[i]) __builtin_trap();
    elements *= (uint64_t)s->dimensions[i];
  }
  if (elements > (uint64_t)INT64_MAX / (uint64_t)element_bytes) __builtin_trap();
  int contiguous = 1;
  int64_t stride = 1;
  for (int64_t i = rank; i-- > 0;) {
    if (s->dimensions[i] > 1 && (s->dimensions[rank + i] != stride ||
                                d->dimensions[rank + i] != stride)) contiguous = 0;
    stride *= s->dimensions[i];
  }
  char *src = (char *)s->aligned + s->offset * element_bytes;
  char *dst = (char *)d->aligned + d->offset * element_bytes;
  if (contiguous) {
    copy_bytes(dst, src, elements * element_bytes);
  } else {
    for (uint64_t i = 0; i < elements; ++i) {
      uint64_t remaining = i;
      int64_t si = 0, di = 0;
      for (int64_t dim = rank; dim-- > 0;) {
        uint64_t coordinate = remaining % (uint64_t)s->dimensions[dim];
        remaining /= (uint64_t)s->dimensions[dim];
        si += (int64_t)coordinate * s->dimensions[rank + dim];
        di += (int64_t)coordinate * d->dimensions[rank + dim];
      }
      copy_bytes(dst + di * element_bytes, src + si * element_bytes, (size_t)element_bytes);
    }
  }
}
#endif

int buffer_init(GuardedBuffer *buffer, size_t count, float *nr_storage) {
  buffer->count = count;
#ifdef HOST_TEST
  (void)nr_storage;
  long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) return 1;
  size_t page = (size_t)page_size;
  size_t writable = ((count + GUARD_FLOATS) * sizeof(float) + page - 1) / page * page;
  buffer->mapping_bytes = writable + page;
  buffer->mapping = mmap(NULL, buffer->mapping_bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (buffer->mapping == MAP_FAILED) return 1;
  char *end = (char *)buffer->mapping + writable;
  if (mprotect(end, page, PROT_NONE)) {
    munmap(buffer->mapping, buffer->mapping_bytes);
    return 1;
  }
  /* The first invalid element is on an inaccessible page: detect tail loads
   * as well as stores, even when LLVM IR is compiled without sanitizers. */
  buffer->data = (float *)end - count;
#else
  buffer->data = nr_storage + GUARD_FLOATS;
  buffer->mapping = NULL;
  buffer->mapping_bytes = 0;
  for (unsigned i = 0; i < GUARD_FLOATS; ++i)
    buffer->data[count + i] = guard_value;
#endif
  for (unsigned i = 1; i <= GUARD_FLOATS; ++i)
    buffer->data[-(int)i] = guard_value;
  return 0;
}

unsigned buffer_check(const GuardedBuffer *buffer) {
  unsigned errors = 0;
  for (unsigned i = 1; i <= GUARD_FLOATS; ++i)
    errors += buffer->data[-(int)i] != guard_value;
#ifndef HOST_TEST
  for (unsigned i = 0; i < GUARD_FLOATS; ++i)
    errors += buffer->data[buffer->count + i] != guard_value;
#endif
  return errors;
}

void buffer_free(GuardedBuffer *buffer) {
#ifdef HOST_TEST
  munmap(buffer->mapping, buffer->mapping_bytes);
#else
  (void)buffer;
#endif
}

int report_check(const char *name, unsigned errors, float maximum) {
  union { float f; uint32_t u; } bits = {maximum};
#ifdef HOST_TEST
  printf("verify %s: %s errors=%08x max_abs_error_f32_bits=%08x\n",
         name, errors ? "FAIL" : "PASS", errors, bits.u);
#else
  nr_puts("verify "); nr_puts(name);
  nr_puts(errors ? ": FAIL errors=" : ": PASS errors=");
  nr_hex32(errors); nr_puts(" max_abs_error_f32_bits="); nr_hex32(bits.u);
  nr_puts("\r\n");
#endif
  return errors != 0;
}
