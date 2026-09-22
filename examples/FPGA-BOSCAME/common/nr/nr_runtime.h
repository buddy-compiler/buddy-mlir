#ifndef BUDDY_FPGA_NR_RUNTIME_H
#define BUDDY_FPGA_NR_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

/* The application implements this entry point; it runs on RA, not NH. */
int launch(void);
void nr_puts(const char *text);
void nr_write(const void *bytes, size_t length);
/* UART input, read on RA. The NH core owns the UART, so this consumes the ring NH
 * fills rather than touching the device. Returns -1 when nothing is waiting. */
int nr_getchar(void);
void nr_hex32(uint32_t value);
void nr_hex64(uint64_t value);
uint64_t nr_cycles(void);
/*
 * Optional RA-side cache probes for AME experiments.  These are deliberately
 * disabled unless NR_RA_AME_CACHE_DIAGNOSTIC=1 is supplied at compile time.
 * The v0.5 software contract names the E6 driver's SYNC_MEM path as the
 * production mechanism; it does not define RA-side CBO as an ABI.  Therefore
 * callers must treat these hooks as an experiment, never as a correctness
 * guarantee.  The range is byte-based and the implementation rounds it out to
 * NR_RA_AME_CACHE_LINE_BYTES (64 by default).
 */
void nr_ame_cache_clean(const void *address, size_t bytes);
void nr_ame_cache_invalidate(const void *address, size_t bytes);
#ifdef NR_HANG_DIAGNOSTICS
/* Diagnostic observations, not a cache-coherence or AME completion primitive.
 * NH samples these RA-owned records and prints directly through its UART. */
enum {
  NR_DIAG_GRAPH_BEGIN = 1, NR_DIAG_KERNEL_ENTER = 2,
  NR_DIAG_KERNEL_RETURN = 3, NR_DIAG_GRAPH_RETURN = 4,
  NR_DIAG_SYNC_DONE = 5, NR_DIAG_COLLECT_BEGIN = 6,
  NR_DIAG_COLLECT_DONE = 7
};
void nr_diag_reset(uint64_t position);
void nr_diag_mark(uint64_t stage, uint64_t detail);
void nr_diag_memref(unsigned operand, uintptr_t descriptor, uintptr_t aligned,
                    int64_t offset, int64_t rows, int64_t cols,
                    int64_t stride0, int64_t stride1);
#endif
/* RA-only scoped scratch allocator. free() is a no-op. A reset invalidates
 * EVERY allocation after mark: first copy graph results/KV into persistent
 * caller-owned buffers and complete asynchronous kernel accesses. */
uintptr_t nr_heap_mark(void);
void nr_heap_reset(uintptr_t mark);
void ame_fence(void);
/* Non-overlapping memcpy semantics. Aligned words use verified NR RVV copy;
 * unaligned addresses and the last 0..3 bytes use scalar loads/stores. */
void nr_copy_bytes(void *destination, const void *source, size_t bytes);

/* Linker-owned, uninitialized workspace in RA-accessible high DDR. Arrays
 * placed in .workspace must be explicitly initialized by launch(). */
extern unsigned char __workspace_start[], __workspace_end[];
#define NR_WORKSPACE __attribute__((section(".workspace"), aligned(64)))

float expf(float);
float logf(float);
float powf(float, float);
float tanhf(float);
float erff(float);
float sqrtf(float);
float sinf(float);
float cosf(float);

void *memcpy(void *destination, const void *source, size_t count);
void *memmove(void *destination, const void *source, size_t count);
void *memset(void *destination, int value, size_t count);
int memcmp(const void *lhs, const void *rhs, size_t count);
void *malloc(size_t size);
void *aligned_alloc(size_t alignment, size_t size);
void *calloc(size_t count, size_t size);
void free(void *pointer);

#endif
