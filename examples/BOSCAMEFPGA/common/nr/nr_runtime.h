//===- nr_runtime.h - NH/RA operator runtime interface --------------------===//
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
// C API for NR operator images linked with crt.S / nr.ld / nr_runtime.c.
// Applications implement launch() on RA. NH owns UART MMIO and relays the
// console ring; RA uses nr_puts / nr_write / nr_getchar.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_FPGA_NR_RUNTIME_H
#define BUDDY_FPGA_NR_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

// Operator entry point. Runs on RA after BSS clear; return 0 means PASS.
int launch(void);

// Append a NUL-terminated string to the RA->NH console ring (no automatic
// CRLF).
void nr_puts(const char *text);

// Append raw bytes (may contain NUL) to the console ring.
void nr_write(const void *bytes, size_t length);

// One UART input byte forwarded by NH into the RX ring, or -1 if empty.
// RA must not call uart_getc(); only NH touches the device.
int nr_getchar(void);

// Hex printers used by launch diagnostics and print_check().
void nr_hex32(uint32_t value);
void nr_hex64(uint64_t value);

// RA cycle counter (rdcycle).
uint64_t nr_cycles(void);

// RA bump-heap markers. free() is a no-op. After nr_heap_reset(mark), every
// allocation made after mark is invalid — copy live results out first.
uintptr_t nr_heap_mark(void);
void nr_heap_reset(uintptr_t mark);

// 1x1 AME resync with non-transposed B load; call only at operator boundaries.
void ame_fence(void);

// Non-overlapping memcpy. Aligned words may use nr_copy.S (RVV); else scalar.
void nr_copy_bytes(void *destination, const void *source, size_t bytes);

// High-DDR NOLOAD workspace bounds from nr.ld. Place arrays with NR_WORKSPACE;
// launch() must initialize every element the kernel may read.
extern unsigned char __workspace_start[], __workspace_end[];
#define NR_WORKSPACE __attribute__((section(".workspace"), aligned(64)))

// Freestanding math (nr_math.c). Not a full IEEE libm.
float expf(float);
float logf(float);
float powf(float, float);
float tanhf(float);
float erff(float);
float sqrtf(float);
float sinf(float);
float cosf(float);

// Scalar libc subset used by MLIR-generated kernels and launch code.
void *memcpy(void *destination, const void *source, size_t count);
void *memmove(void *destination, const void *source, size_t count);
void *memset(void *destination, int value, size_t count);
int memcmp(const void *lhs, const void *rhs, size_t count);
void *malloc(size_t size);
void *aligned_alloc(size_t alignment, size_t size);
void *calloc(size_t count, size_t size);
void free(void *pointer);

#endif // BUDDY_FPGA_NR_RUNTIME_H
