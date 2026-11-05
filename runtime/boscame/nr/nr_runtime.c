//===- nr_runtime.c - NR NH/RA dual-core bare-metal runtime ---------------===//
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
// NR platform runtime: the NH/RA execution environment shared by the NR
// operator images.
//
// Owned here:
//   - NH boot (UART init, console init, RA release through the 0x50000000
//     reset/entry registers) and the NH completion-mailbox poll at
//     0x80010000,
//   - the inter-core console (nr_console.inc) and the optional hang watch
//     (nr_hang_watch.inc), including the fence/flush/invalidate hooks they
//     call and the RA-side input ring behind nr_getchar(),
//   - the RA/NH trap entries, abort/stack-check paths and the terminal result
//     report (nr_finish),
//   - the ModelZoo-compatible print_uart* log-forwarding entries.
//
// Not owned here: the heap, the scalar memory helpers and memrefCopy, which
// live in nr_memory.c behind the public nr_runtime.h ABI.  The RA diagnostic
// cache hooks stay with the platform because they emit platform CBOs.
//

#include "nr_runtime.h"
#include "../uart/uart.h"
#include "nr_runtime_internal.h"

//
// The AME v0.5 document specifies a platform DMA SYNC_MEM operation for
// CPU/NH <-> AME ownership transfers, but gives no RA CBO ABI.  Keep the CBO
// probe opt-in so ordinary NR images retain the validated NH-only cache path.
//
#ifndef NR_RA_AME_CACHE_DIAGNOSTIC
#define NR_RA_AME_CACHE_DIAGNOSTIC 0
#endif
#ifndef NR_RA_AME_CACHE_LINE_BYTES
#define NR_RA_AME_CACHE_LINE_BYTES 64u
#endif
#if NR_RA_AME_CACHE_DIAGNOSTIC != 0 && NR_RA_AME_CACHE_DIAGNOSTIC != 1
#error "NR_RA_AME_CACHE_DIAGNOSTIC must be 0 or 1"
#endif
#if NR_RA_AME_CACHE_LINE_BYTES < 16 ||                                         \
    (NR_RA_AME_CACHE_LINE_BYTES & (NR_RA_AME_CACHE_LINE_BYTES - 1u)) != 0
#error "NR_RA_AME_CACHE_LINE_BYTES must be a power of two >= 16"
#endif

// In the production path RA does not use cache-management instructions. NH
// invalidates each shared line before reading it; RA publishes data with
// volatile + fences, matching ModelZoo's validated NR console and completion
// protocol. The opt-in AME CBO walker above is diagnostic only.
#define RA_SIGNAL ((volatile uint8_t *)0x80010000UL)
#define RA_REGISTER(offset) (*(volatile uint32_t *)(0x50000000UL + (offset)))
extern void _ra_start(void);

// Defined further down; forward-declared so the diagnostic sampling below can
// use them without depending on where it happens to sit in the file.
static void host_puts(const char *text);
static void host_hex(uint64_t value);

static void fence(void) { __asm__ volatile("fence rw, rw" ::: "memory"); }
static void flush(const volatile void *pointer) {
  __asm__ volatile("cbo.flush (%0)" ::"r"(pointer) : "memory");
}
static void invalidate(const volatile void *pointer) {
  __asm__ volatile("cbo.inval (%0)" ::"r"(pointer) : "memory");
}

//
// Diagnostic-only RA range walk.  CBO instructions are intentionally not
// emitted in the default build.  A caller must pass the complete byte range
// that the next AME operation will read or that the CPU will read after an
// AME store; this helper does not inspect memref descriptors or infer strides.
// A zero/overflowing range is ignored so a malformed diagnostic cannot turn
// into an unbounded walk.
//
static void ra_ame_cache_range(const void *address, size_t bytes, int drop) {
#if NR_RA_AME_CACHE_DIAGNOSTIC
  if (!address || !bytes)
    return;
  uintptr_t begin = (uintptr_t)address;
  if (bytes > UINTPTR_MAX - begin)
    return;
  uintptr_t end = begin + bytes;
  const uintptr_t line = (uintptr_t)NR_RA_AME_CACHE_LINE_BYTES;
  uintptr_t cursor = begin & ~(line - 1u);
  uintptr_t last = (end - 1u) | (line - 1u);
  if (last > UINTPTR_MAX - (line - 1u))
    return;
  last += 1u;
  fence();
  for (; cursor < last; cursor += line) {
    if (drop)
      __asm__ volatile("cbo.inval (%0)" ::"r"(cursor) : "memory");
    else
      __asm__ volatile("cbo.flush (%0)" ::"r"(cursor) : "memory");
  }
  fence();
#else
  (void)address;
  (void)bytes;
  (void)drop;
#endif
}

void nr_ame_cache_clean(const void *address, size_t bytes) {
  ra_ame_cache_range(address, bytes, 0);
}

void nr_ame_cache_invalidate(const void *address, size_t bytes) {
  ra_ame_cache_range(address, bytes, 1);
}

// UART input has the same ownership problem as output, in the other direction:
// the NH core owns the UART, so uart_getc() called from RA sees nothing. NH
// polls the receive line in the same loop that drains the console and
// republishes what it finds here, and RA consumes it with nr_getchar(). Each
// writer owns a separate cache line: NH flushes head/data and invalidates the
// RA-owned tail. RA uses ordered volatile loads; it does not issue CBO
// instructions because the current platform only validates cache management on
// the NH side.
#define RX_CAPACITY (4u * 1024u)
static volatile struct {
  uint32_t head; // advanced by NH
  unsigned char head_padding[60];
  uint32_t tail; // advanced by RA
  unsigned char tail_padding[60];
  char data[RX_CAPACITY];
} input __attribute__((section(".bss.nr_console"), aligned(64)));

// NH side: move anything the UART has received into the ring.
static void pump_input(void) {
  for (unsigned i = 0; i < 64 && uart_rx_ready(); ++i) {
    // RBR reads consume bytes: debug output must reuse this one read.
    unsigned char value = (unsigned char)UART_RBR_THR_DLL;
#ifdef NR_UART_DEBUG
    host_puts("[n] received=");
    host_hex(value);
    host_puts("\r\n");
#endif
    uint32_t head = input.head;
    invalidate(&input.tail);
    fence();
    uint32_t tail = input.tail;
    if (head - tail >= RX_CAPACITY)
      continue; // full: drop, do not block
    input.data[head & (RX_CAPACITY - 1u)] = value;
    flush(&input.data[head & (RX_CAPACITY - 1u)]);
    fence();
    input.head = head + 1;
    flush(&input.head);
    fence();
  }
}

// RA side: one character, or -1 when nothing is waiting.
int nr_getchar(void) {
  // NH publishes head and data after flushing them.  The platform's validated
  // RA contract uses ordered volatile loads here; CBO is NH-owned.
  fence();
  uint32_t head = input.head;
  uint32_t tail = input.tail;
  if (head == tail)
    return -1;
  uint32_t slot = tail & (RX_CAPACITY - 1u);
  fence();
  char value = input.data[slot];
  input.tail = tail + 1;
  fence();
  return (unsigned char)value;
}

static __attribute__((noreturn)) void halt(void) {
  for (;;)
    __asm__ volatile("wfi");
}
static void host_puts(const char *text) {
  while (*text)
    uart_putc(*text++);
}
static void host_hex(uint64_t value) {
  const char *digits = "0123456789ABCDEF";
  for (int shift = 60; shift >= 0; shift -= 4)
    uart_putc(digits[(value >> shift) & 15]);
}

#include "nr_console.inc"

#ifdef NR_HANG_DIAGNOSTICS
#include "nr_hang_watch.inc"
#if !NR_CONSOLE_APPEND_ONLY
static void nr_console_wait_enter(void) {
  if (!nr_diag_console[0]) {
    nr_diag_console[0] = 1;
    ++nr_diag_console[2];
    fence();
  }
}
static void nr_console_wait_leave(void) {
  if (nr_diag_console[0]) {
    nr_diag_console[0] = 0;
    fence();
  }
}
#ifdef NR_HANG_CONSOLE_BOUNDED
static void nr_console_drop(void) {
  ++nr_diag_console[1];
  nr_diag_console[0] = 0;
  fence();
}
#endif
#endif
#endif

#ifdef NR_UART_DEBUG
// Diagnostic only, compiled in with -DNR_UART_DEBUG. This runs on NH -- the
// core that owns the UART -- because a probe running on RA reads zeros from
// every register and therefore says nothing.
static void uart_debug_sample(void) {
  static uint32_t last_lsr = 0xffffffffu;
  static uint32_t ticks = 0;
  uint32_t lsr = UART_LSR;
  if (lsr != last_lsr) {
    last_lsr = lsr;
    host_puts("[n] lsr=");
    host_hex(lsr);
    host_puts("\r\n");
  } else if (((++ticks) & 0xfffffu) == 0) {
    host_puts("[n] lsr=");
    host_hex(lsr);
    host_puts(" (idle)\r\n");
  }
}
#endif

void nr_puts(const char *text) {
  while (*text)
    write_serial(*text++);
}
void nr_write(const void *bytes, size_t length) {
  const unsigned char *data = bytes;
  while (length--)
    write_serial((char)*data++);
}
void nr_hex32(uint32_t value) {
  const char *digits = "0123456789ABCDEF";
  for (int shift = 28; shift >= 0; shift -= 4)
    write_serial(digits[(value >> shift) & 15]);
}
void nr_hex64(uint64_t value) {
  nr_hex32((uint32_t)(value >> 32));
  nr_hex32((uint32_t)value);
}
uint64_t nr_cycles(void) {
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value));
  return value;
}
// Compatibility with ModelZoo standalone operator diagnostics. Only NH calls
// the real uart_init: application code always runs on RA.
void init_uart(uint32_t frequency, uint32_t baud) {
  (void)frequency;
  (void)baud;
}
void print_uart(const char *text) { nr_puts(text); }
void print_uart_int(uint32_t value) { nr_hex32(value); }
void print_uart_addr(uint64_t value) { nr_hex64(value); }

__attribute__((noreturn)) void nr_nh_main(void) {
  uart_init();
  host_puts("\r\n[nr] NH boot; RA operator runtime\r\n[nr] RA entry=0x");
  host_hex((uintptr_t)&_ra_start);
  host_puts("\r\n");
  nr_console_init();
  // .nr_console is NOLOAD and excluded from RA's BSS initialization.
  input.head = 0;
  input.tail = 0;
  flush(&input.head);
  flush(&input.tail);
#ifdef NR_HANG_DIAGNOSTICS
  nr_diag_init();
  uint64_t diag_last = nr_cycles(), diag_sample = 0;
  host_puts("[nh-watch] enabled; stable samples may still be stale\r\n");
#endif
  *RA_SIGNAL = 0;
  flush(RA_SIGNAL);
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  RA_REGISTER(0x100) = (uint32_t)(uintptr_t)&_ra_start;
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  RA_REGISTER(0x60) = 0;
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  RA_REGISTER(0x50) = 1;
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  RA_REGISTER(0x60) = 1;
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  uint32_t consumed = 0;
  uint8_t result;
  do {
    // Deferred mode leaves only the existing completion-mailbox poll active.
    consumed = nr_console_poll_running(consumed);
#ifdef NR_HANG_DIAGNOSTICS
    nr_diag_poll(&diag_last, &diag_sample, 0);
#endif
#ifdef NR_UART_DEBUG
    uart_debug_sample();
#endif
    invalidate(RA_SIGNAL);
    fence();
    result = *RA_SIGNAL;
  } while (!result);
  do {
    consumed = drain(consumed);
#ifdef NR_HANG_DIAGNOSTICS
    nr_diag_poll(&diag_last, &diag_sample, 0);
#endif
  } while (consumed != console.count);
#ifdef NR_HANG_DIAGNOSTICS
  nr_diag_poll(&diag_last, &diag_sample, 1);
#endif
  nr_console_report(result);
  halt();
}

// Terminal platform report: publish the RA result and stop.  nr_memory.c
// reports heap exhaustion and ABI violations through this entry point.
__attribute__((noreturn)) void nr_finish(int status) {
  fence();
  *RA_SIGNAL = status == 0 ? 1 : 2;
  fence();
  halt();
}
void nr_ra_entered(void) {
  nr_puts("[nr] RA entered; clearing application BSS\r\n");
}
__attribute__((noreturn)) void nr_ra_main(void) {
  uintptr_t status;
  __asm__ volatile("csrr %0, mstatus" : "=r"(status));
  nr_puts("[nr] mstatus=0x");
  nr_hex64(status);
  nr_puts("; launch BEGIN\r\n");
  uint64_t begin = nr_cycles();
  int result = launch();
  uint64_t elapsed = nr_cycles() - begin;
  nr_puts("[nr] launch cycles=0x");
  nr_hex64(elapsed);
  nr_puts(" status=0x");
  nr_hex32((uint32_t)result);
  nr_puts("\r\n");
  nr_finish(result);
}
__attribute__((noreturn)) void nr_ra_trap(uintptr_t cause, uintptr_t pc,
                                          uintptr_t value,
                                          uintptr_t return_address,
                                          uintptr_t stack_pointer) {
  nr_puts("\r\n[nr] RA TRAP mcause=0x");
  nr_hex64(cause);
  nr_puts(" mepc=0x");
  nr_hex64(pc);
  nr_puts(" mtval=0x");
  nr_hex64(value);
  nr_puts(" ra=0x");
  nr_hex64(return_address);
  nr_puts(" sp=0x");
  nr_hex64(stack_pointer);
  nr_puts("\r\n");
  nr_finish(1);
}
__attribute__((noreturn)) void nr_nh_trap(uintptr_t cause, uintptr_t pc,
                                          uintptr_t value,
                                          uintptr_t return_address,
                                          uintptr_t stack_pointer) {
  host_puts("\r\n[nr] NH TRAP mcause=0x");
  host_hex(cause);
  host_puts(" mepc=0x");
  host_hex(pc);
  host_puts(" mtval=0x");
  host_hex(value);
  host_puts(" ra=0x");
  host_hex(return_address);
  host_puts(" sp=0x");
  host_hex(stack_pointer);
  host_puts("\r\nverify NH: FAIL\r\n");
  halt();
}
__attribute__((noreturn)) void abort(void) {
  nr_puts("[nr] abort: FAIL\r\n");
  nr_finish(1);
}
__attribute__((noreturn)) void __stack_chk_fail(void) {
  nr_puts("[nr] stack check: FAIL\r\n");
  nr_finish(1);
}
uintptr_t __stack_chk_guard = (uintptr_t)0x9e3779b97f4a7c15ULL;
