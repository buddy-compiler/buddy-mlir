#ifndef MOBILENET_OPERATOR_SUPPORT_H
#define MOBILENET_OPERATOR_SUPPORT_H
#include <stddef.h>
#include <stdint.h>
#ifdef HOST_TEST
#define WORKSPACE __attribute__((aligned(64)))
#else
#include "nr_runtime.h"
#define WORKSPACE NR_WORKSPACE
#endif

#define GUARD_FLOATS 128
typedef struct {
  float *data;
  size_t count;
  void *mapping;
  size_t mapping_bytes;
} GuardedBuffer;
int buffer_init(GuardedBuffer *buffer, size_t count, float *nr_storage);
unsigned buffer_check(const GuardedBuffer *buffer);
void buffer_free(GuardedBuffer *buffer);
int report_check(const char *name, unsigned errors, float maximum);
#endif
