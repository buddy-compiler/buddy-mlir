/* Shared NR console. Platform hooks are defined by nr_runtime.c; tests replace
 * only those hooks. NH initializes/flushes before releasing RA. RA owns the first
 * control line and data, NH owns the second line. Both remain outside RA BSS.
 */
#ifndef NR_CONSOLE_CAPACITY
#define NR_CONSOLE_CAPACITY (64u * 1024u)
#endif
#ifndef NR_CONSOLE_APPEND_ONLY
#define NR_CONSOLE_APPEND_ONLY 0
#endif
#ifndef NR_CONSOLE_DRAIN_AFTER_COMPLETION
#define NR_CONSOLE_DRAIN_AFTER_COMPLETION 0
#endif
#if NR_CONSOLE_CAPACITY < 64 || NR_CONSOLE_CAPACITY > 0x40000000u || \
    (NR_CONSOLE_CAPACITY & (NR_CONSOLE_CAPACITY - 1u)) != 0
#error "NR_CONSOLE_CAPACITY must be a power of two between 64 and 2^30 bytes"
#endif
#if NR_CONSOLE_APPEND_ONLY != 0 && NR_CONSOLE_APPEND_ONLY != 1
#error "NR_CONSOLE_APPEND_ONLY must be 0 or 1"
#endif
#if NR_CONSOLE_APPEND_ONLY && defined(NR_HANG_CONSOLE_BOUNDED)
#error "NR_HANG_CONSOLE_BOUNDED applies only to the ring console"
#endif
#if NR_CONSOLE_DRAIN_AFTER_COMPLETION != 0 && NR_CONSOLE_DRAIN_AFTER_COMPLETION != 1
#error "NR_CONSOLE_DRAIN_AFTER_COMPLETION must be 0 or 1"
#endif
#if NR_CONSOLE_DRAIN_AFTER_COMPLETION && !NR_CONSOLE_APPEND_ONLY
#error "Deferred console draining requires the finite append-only console"
#endif
#if NR_CONSOLE_DRAIN_AFTER_COMPLETION && \
    (defined(NR_HANG_DIAGNOSTICS) || defined(NR_UART_DEBUG))
#error "Deferred console draining cannot run concurrent NH diagnostic sampling"
#endif

static volatile struct {
  uint32_t count;
  uint32_t overflow;
  unsigned char count_padding[56];
  uint32_t consumed;
  unsigned char consumed_padding[60];
  char data[NR_CONSOLE_CAPACITY];
} console __attribute__((section(".bss.nr_console"), aligned(64)));

#if defined(NR_HANG_DIAGNOSTICS) && !NR_CONSOLE_APPEND_ONLY
static void nr_console_wait_enter(void);
static void nr_console_wait_leave(void);
#ifdef NR_HANG_CONSOLE_BOUNDED
#ifndef NR_HANG_CONSOLE_SPINS
#define NR_HANG_CONSOLE_SPINS 1000000u
#endif
static void nr_console_drop(void);
#endif
#endif

static void nr_console_init(void) {
  console.count = 0;
  console.overflow = 0;
  console.consumed = 0;
  flush(&console.count);
  flush(&console.consumed);
  fence();
}

void write_serial(char value) {
  uint32_t count = console.count;
#if NR_CONSOLE_APPEND_ONLY
  /* Fixed validation runs must not wait on the NH-to-RA acknowledgement. A
   * full log is an explicit failure, not permission to overwrite earlier data.
   */
  if (count >= NR_CONSOLE_CAPACITY) {
    console.overflow = 1;
    fence();
    return;
  }
  console.data[count] = value;
#else
#if defined(NR_HANG_DIAGNOSTICS) && defined(NR_HANG_CONSOLE_BOUNDED)
  unsigned spins = 0;
#endif
  while ((uint32_t)(count - console.consumed) >= NR_CONSOLE_CAPACITY) {
#ifdef NR_HANG_DIAGNOSTICS
    nr_console_wait_enter();
#ifdef NR_HANG_CONSOLE_BOUNDED
    if (++spins >= NR_HANG_CONSOLE_SPINS) {
      nr_console_drop();
      return;
    }
#endif
#endif
    fence();
  }
#ifdef NR_HANG_DIAGNOSTICS
  nr_console_wait_leave();
#endif
  console.data[count & (NR_CONSOLE_CAPACITY - 1u)] = value;
#endif
  fence();
  console.count = count + 1u;
  fence();
}

static uint32_t drain(uint32_t consumed) {
  invalidate(&console.count);
  fence();
  uint32_t end = console.count;
  /* Bounded batches keep completion, input, and hang-watch polling responsive. */
  if ((uint32_t)(end - consumed) > 256u) end = consumed + 256u;
  while (consumed != end) {
    uint32_t slot = consumed & (NR_CONSOLE_CAPACITY - 1u);
    invalidate(&console.data[slot]);
    fence();
    uint32_t amount = 64u - (slot & 63u);
    if (amount > end - consumed) amount = end - consumed;
    while (amount--) { uart_putc(console.data[slot++]); ++consumed; }
    console.consumed = consumed;
#if !NR_CONSOLE_APPEND_ONLY
    flush(&console.consumed);
    fence();
#endif
  }
  return consumed;
}

static uint32_t nr_console_poll_running(uint32_t consumed) {
#if !NR_CONSOLE_DRAIN_AFTER_COMPLETION
  consumed = drain(consumed);
  pump_input();
#endif
  return consumed;
}

static void nr_console_report(uint8_t result) {
  invalidate(&console.count);
  fence();
  if (console.overflow) {
    host_puts("[nr] console overflow: FAIL (output truncated)\r\n");
    result = 2;
  }
  host_puts(result == 1 ? "[nr] RA returned: PASS\r\nverify NR runtime: PASS\r\n" :
                         "[nr] RA returned: FAIL\r\nverify NR runtime: FAIL\r\n");
}
