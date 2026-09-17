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
