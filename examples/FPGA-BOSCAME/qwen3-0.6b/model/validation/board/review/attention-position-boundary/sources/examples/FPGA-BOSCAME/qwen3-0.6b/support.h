#ifndef QWEN_OPERATOR_SUPPORT_H
#define QWEN_OPERATOR_SUPPORT_H
#include <stddef.h>
#include <stdint.h>
#ifdef HOST_TEST
#include <math.h>
#else
#include "nr_runtime.h"
#endif
#define MEMREF(R) typedef struct { void *allocated, *aligned; int64_t offset; int64_t sizes[R], strides[R]; } MemRef##R
MEMREF(1); MEMREF(2); MEMREF(3); MEMREF(4);
static inline MemRef1 make_1(void *p, int64_t a) { return (MemRef1){p,p,0,{a},{1}}; }
static inline MemRef2 make_2(void *p, int64_t a,int64_t b) { return (MemRef2){p,p,0,{a,b},{b,1}}; }
static inline MemRef3 make_3(void *p, int64_t a,int64_t b,int64_t c) { return (MemRef3){p,p,0,{a,b,c},{b*c,c,1}}; }
static inline MemRef4 make_4(void *p, int64_t a,int64_t b,int64_t c,int64_t d) { return (MemRef4){p,p,0,{a,b,c,d},{b*c*d,c*d,d,1}}; }
void *workspace(size_t byte_offset);
void nr_copy_bytes(void *destination, const void *source, size_t bytes);
int check_close(float actual, float expected, float atol, float rtol);
int print_check(const char *name, unsigned errors, float max_error);
void nr_puts(const char *s);
void nr_hex32(uint32_t x);
void nr_hex64(uint64_t x);
uint64_t nr_cycles(void);
#endif
