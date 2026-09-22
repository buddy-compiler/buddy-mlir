#include "nr_runtime.h"
#include "../uart/uart.h"

/*
 * The AME v0.5 document specifies a platform DMA SYNC_MEM operation for
 * CPU/NH <-> AME ownership transfers, but gives no RA CBO ABI.  Keep the CBO
 * probe opt-in so ordinary NR images retain the validated NH-only cache path.
 */
#ifndef NR_RA_AME_CACHE_DIAGNOSTIC
#define NR_RA_AME_CACHE_DIAGNOSTIC 0
#endif
#ifndef NR_RA_AME_CACHE_LINE_BYTES
#define NR_RA_AME_CACHE_LINE_BYTES 64u
#endif
#if NR_RA_AME_CACHE_DIAGNOSTIC != 0 && NR_RA_AME_CACHE_DIAGNOSTIC != 1
#error "NR_RA_AME_CACHE_DIAGNOSTIC must be 0 or 1"
#endif
#if NR_RA_AME_CACHE_LINE_BYTES < 16 || \
    (NR_RA_AME_CACHE_LINE_BYTES & (NR_RA_AME_CACHE_LINE_BYTES - 1u)) != 0
#error "NR_RA_AME_CACHE_LINE_BYTES must be a power of two >= 16"
#endif

/* In the production path RA does not use cache-management instructions. NH
 * invalidates each shared line before reading it; RA publishes data with
 * volatile + fences, matching ModelZoo's validated NR console and completion
 * protocol. The opt-in AME CBO walker above is diagnostic only. */
#define RA_SIGNAL ((volatile uint8_t *)0x80010000UL)
#define RA_REGISTER(offset) (*(volatile uint32_t *)(0x50000000UL + (offset)))
extern void _ra_start(void);
extern unsigned char __heap_start[], __heap_end[];
static uintptr_t heap_cursor;

/* Defined further down; forward-declared so the diagnostic sampling below can use
 * them without depending on where it happens to sit in the file. */
static void host_puts(const char *text);
static void host_hex(uint64_t value);

static void fence(void) {
  __asm__ volatile("fence rw, rw" ::: "memory");
}
static void flush(const volatile void *pointer) {
  __asm__ volatile("cbo.flush (%0)" :: "r"(pointer) : "memory");
}
static void invalidate(const volatile void *pointer) {
  __asm__ volatile("cbo.inval (%0)" :: "r"(pointer) : "memory");
}

/*
 * Diagnostic-only RA range walk.  CBO instructions are intentionally not
 * emitted in the default build.  A caller must pass the complete byte range
 * that the next AME operation will read or that the CPU will read after an
 * AME store; this helper does not inspect memref descriptors or infer strides.
 * A zero/overflowing range is ignored so a malformed diagnostic cannot turn
 * into an unbounded walk.
 */
static void ra_ame_cache_range(const void *address, size_t bytes, int drop) {
#if NR_RA_AME_CACHE_DIAGNOSTIC
  if (!address || !bytes) return;
  uintptr_t begin = (uintptr_t)address;
  if (bytes > UINTPTR_MAX - begin) return;
  uintptr_t end = begin + bytes;
  const uintptr_t line = (uintptr_t)NR_RA_AME_CACHE_LINE_BYTES;
  uintptr_t cursor = begin & ~(line - 1u);
  uintptr_t last = (end - 1u) | (line - 1u);
  if (last > UINTPTR_MAX - (line - 1u)) return;
  last += 1u;
  fence();
  for (; cursor < last; cursor += line) {
    if (drop)
      __asm__ volatile("cbo.inval (%0)" :: "r"(cursor) : "memory");
    else
      __asm__ volatile("cbo.flush (%0)" :: "r"(cursor) : "memory");
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

/* UART input has the same ownership problem as output, in the other direction:
 * the NH core owns the UART, so uart_getc() called from RA sees nothing. NH polls
 * the receive line in the same loop that drains the console and republishes what
 * it finds here, and RA consumes it with nr_getchar(). Each writer owns a
 * separate cache line: NH flushes head/data and invalidates the RA-owned tail.
 * RA uses ordered volatile loads; it does not issue CBO instructions because
 * the current platform only validates cache management on the NH side. */
#define RX_CAPACITY (4u * 1024u)
static volatile struct {
  uint32_t head;                  /* advanced by NH */
  unsigned char head_padding[60];
  uint32_t tail;                  /* advanced by RA */
  unsigned char tail_padding[60];
  char data[RX_CAPACITY];
} input __attribute__((section(".bss.nr_console"), aligned(64)));


/* NH side: move anything the UART has received into the ring. */
static void pump_input(void) {
  for (unsigned i = 0; i < 64 && uart_rx_ready(); ++i) {
    /* RBR reads consume bytes: debug output must reuse this one read. */
    unsigned char value = (unsigned char)UART_RBR_THR_DLL;
#ifdef NR_UART_DEBUG
    host_puts("[n] received="); host_hex(value);
    host_puts("\r\n");
#endif
    uint32_t head = input.head;
    invalidate(&input.tail);
    fence();
    uint32_t tail = input.tail;
    if (head - tail >= RX_CAPACITY) continue;   /* full: drop, do not block */
    input.data[head & (RX_CAPACITY - 1u)] = value;
    flush(&input.data[head & (RX_CAPACITY - 1u)]);
    fence();
    input.head = head + 1;
    flush(&input.head);
    fence();
  }
}

/* RA side: one character, or -1 when nothing is waiting. */
int nr_getchar(void) {
  /* NH publishes head and data after flushing them.  The platform's validated
   * RA contract uses ordered volatile loads here; CBO is NH-owned. */
  fence();
  uint32_t head = input.head;
  uint32_t tail = input.tail;
  if (head == tail) return -1;
  uint32_t slot = tail & (RX_CAPACITY - 1u);
  fence();
  char value = input.data[slot];
  input.tail = tail + 1;
  fence();
  return (unsigned char)value;
}

static __attribute__((noreturn)) void halt(void) {
  for (;;) __asm__ volatile("wfi");
}
static void host_puts(const char *text) {
  while (*text) uart_putc(*text++);
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
  if (nr_diag_console[0]) { nr_diag_console[0] = 0; fence(); }
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
/* Diagnostic only, compiled in with -DNR_UART_DEBUG. This runs on NH -- the core
 * that owns the UART -- because a probe running on RA reads zeros from every
 * register and therefore says nothing. */
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
    host_puts("[n] lsr="); host_hex(lsr);
    host_puts(" (idle)\r\n");
  }
}
#endif

void nr_puts(const char *text) {
  while (*text) write_serial(*text++);
}
void nr_write(const void *bytes, size_t length) {
  const unsigned char *data = bytes;
  while (length--) write_serial((char)*data++);
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
/* Compatibility with ModelZoo standalone operator diagnostics. Only NH calls
 * the real uart_init: application code always runs on RA. */
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
  /* .nr_console is NOLOAD and excluded from RA's BSS initialization. */
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
    /* Deferred mode leaves only the existing completion-mailbox poll active. */
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

static __attribute__((noreturn)) void finish(int status) {
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
  finish(result);
}
__attribute__((noreturn)) void nr_ra_trap(uintptr_t cause, uintptr_t pc,
                                       uintptr_t value, uintptr_t return_address,
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
  finish(1);
}
__attribute__((noreturn)) void nr_nh_trap(uintptr_t cause, uintptr_t pc,
                                       uintptr_t value, uintptr_t return_address,
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
  finish(1);
}
__attribute__((noreturn)) void __stack_chk_fail(void) {
  nr_puts("[nr] stack check: FAIL\r\n");
  finish(1);
}
uintptr_t __stack_chk_guard = (uintptr_t)0x9e3779b97f4a7c15ULL;

/* Ordinary scalar memory operations avoid unsupported vector spills/CSRs. */
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
  while (count--) *out++ = *in++;
  return destination;
}
void *memmove(void *destination, const void *source, size_t count) {
  unsigned char *out = destination;
  const unsigned char *in = source;
  if ((uintptr_t)out > (uintptr_t)in &&
      (uintptr_t)out - (uintptr_t)in < count) {
    while (count) { --count; out[count] = in[count]; }
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
  while (count--) *out++ = (unsigned char)value;
  return destination;
}
int memcmp(const void *lhs, const void *rhs, size_t count) {
  const unsigned char *left = lhs, *right = rhs;
  while (count--) {
    if (*left != *right) return (int)*left - (int)*right;
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
    finish(1);
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
  finish(1);
}
void *aligned_alloc(size_t alignment, size_t size) {
  if (!alignment || (alignment & (alignment - 1))) return 0;
  uintptr_t cursor = heap_cursor ? heap_cursor : (uintptr_t)__heap_start;
  if (cursor > UINTPTR_MAX - (alignment - 1)) heap_exhausted(size);
  cursor = (cursor + alignment - 1) & ~(uintptr_t)(alignment - 1);
  if (cursor > (uintptr_t)__heap_end ||
      size > (uintptr_t)__heap_end - cursor) heap_exhausted(size);
  heap_cursor = cursor + size;
  return (void *)cursor;
}
void *malloc(size_t size) { return aligned_alloc(64, size ? size : 1); }
void free(void *pointer) { (void)pointer; }
void *calloc(size_t count, size_t size) {
  if (count && size > SIZE_MAX / count) return 0;
  void *result = malloc(count * size);
  if (result) memset(result, 0, count * size);
  return result;
}

/* Standard MLIR CRunner ABI; rank <= 8, arbitrary strided element copies. */
typedef struct { int64_t rank; void *descriptor; } UnrankedMemRef;
typedef struct {
  char *base, *data;
  int64_t offset;
  int64_t dimensions[];
} RankedMemRef;
void memrefCopy(int64_t element_size, UnrankedMemRef *source,
                UnrankedMemRef *destination) {
  if (!source || !destination || source->rank != destination->rank ||
      source->rank < 0 || source->rank > 8 || element_size <= 0) abort();
  int64_t rank = source->rank;
  RankedMemRef *src = source->descriptor, *dst = destination->descriptor;
  if (!src || !dst || !src->data || !dst->data) abort();
  uint64_t elements = 1;
  for (int64_t i = 0; i < rank; ++i) {
    if (src->dimensions[i] != dst->dimensions[i] || src->dimensions[i] < 0)
      abort();
    if (!src->dimensions[i]) return;
    if (elements > (uint64_t)INT64_MAX / (uint64_t)src->dimensions[i]) abort();
    elements *= (uint64_t)src->dimensions[i];
  }
  if (elements > (uint64_t)INT64_MAX / (uint64_t)element_size) abort();
  int contiguous = 1;
  int64_t stride = 1;
  for (int64_t d = rank; d-- > 0;) {
    if (src->dimensions[d] > 1 &&
        (src->dimensions[rank+d] != stride ||
         dst->dimensions[rank+d] != stride)) contiguous = 0;
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
      src_index += (int64_t)coordinate * src->dimensions[rank+d];
      dst_index += (int64_t)coordinate * dst->dimensions[rank+d];
    }
    memcpy(dst->data + dst_index * element_size,
           src->data + src_index * element_size, element_size);
  }
}
