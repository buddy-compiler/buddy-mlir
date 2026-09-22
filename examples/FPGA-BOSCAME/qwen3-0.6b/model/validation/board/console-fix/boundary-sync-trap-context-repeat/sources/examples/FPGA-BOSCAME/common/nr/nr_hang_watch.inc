/* Optional NH/RA hang observations. Included by nr_runtime.c only in diagnostic
 * builds. No AME instruction, RA CBO, cache-control CSR, or hardware reset here.
 * RA publication still depends on the platform's DDR visibility: stable stale
 * samples do NOT prove RA has stopped. A sequence check detects some torn reads;
 * it cannot establish cross-line visibility on a non-coherent memory system.
 */
#ifndef NR_HANG_DIAG_PERIOD_CYCLES
#define NR_HANG_DIAG_PERIOD_CYCLES 73728000ULL
#endif
#ifndef NR_HANG_CONSOLE_SPINS
#define NR_HANG_CONSOLE_SPINS 1000000u
#endif

/* One RA writer, NH reader; NH initializes/flushes before releasing RA. Keep
 * these outside RA-cleared BSS, and never share a line with NH-owned state.
 * Header: sequence, phase, position, detail, last target phase/detail, reserved.
 * Three operand records: descriptor, aligned, offset, rows, cols, strides, valid.
 */
static volatile uint64_t nr_diag_record[32]
    __attribute__((section(".bss.nr_console"), aligned(64)));
/* in_console_wait, dropped_bytes, wait_entries; separate RA-owned cache line. */
static volatile uint64_t nr_diag_console[8]
    __attribute__((section(".bss.nr_console"), aligned(64)));

static uint64_t nr_diag_begin(void) {
  uint64_t sequence = nr_diag_record[0];
  nr_diag_record[0] = sequence + 1;
  fence();
  return sequence + 2;
}
static void nr_diag_end(uint64_t sequence) {
  fence();
  nr_diag_record[0] = sequence;
  fence();
}
void nr_diag_reset(uint64_t position) {
  uint64_t sequence = nr_diag_begin();
  for (unsigned i = 1; i < 32; ++i) nr_diag_record[i] = 0;
  nr_diag_record[1] = NR_DIAG_GRAPH_BEGIN;
  nr_diag_record[2] = position;
  nr_diag_end(sequence);
}
void nr_diag_mark(uint64_t stage, uint64_t detail) {
  uint64_t sequence = nr_diag_begin();
  nr_diag_record[1] = stage;
  nr_diag_record[3] = detail;
  if (stage == NR_DIAG_KERNEL_ENTER || stage == NR_DIAG_KERNEL_RETURN) {
    nr_diag_record[4] = stage;
    nr_diag_record[5] = detail;
  }
  nr_diag_end(sequence);
}
void nr_diag_memref(unsigned operand, uintptr_t descriptor, uintptr_t aligned,
                    int64_t offset, int64_t rows, int64_t cols,
                    int64_t stride0, int64_t stride1) {
  if (operand >= 3) return;
  uint64_t sequence = nr_diag_begin();
  unsigned base = 8 + operand * 8;
  nr_diag_record[base] = descriptor;
  nr_diag_record[base + 1] = aligned;
  nr_diag_record[base + 2] = (uint64_t)offset;
  nr_diag_record[base + 3] = (uint64_t)rows;
  nr_diag_record[base + 4] = (uint64_t)cols;
  nr_diag_record[base + 5] = (uint64_t)stride0;
  nr_diag_record[base + 6] = (uint64_t)stride1;
  nr_diag_record[base + 7] = descriptor != 0;
  nr_diag_end(sequence);
}
static void nr_diag_init(void) {
  for (unsigned i = 0; i < 32; ++i) nr_diag_record[i] = 0;
  for (unsigned i = 0; i < 8; ++i) nr_diag_console[i] = 0;
  for (unsigned i = 0; i < 32; i += 8) flush(&nr_diag_record[i]);
  flush(nr_diag_console);
  fence();
}
static void nr_diag_field(const char *key, uint64_t value) {
  host_puts(key);
  host_hex(value);
}
/* Called only by NH. UART output does not depend on RA's console ring. The
 * snapshot is bounded: a busy writer cannot trap NH in a retry loop. */
static void nr_diag_poll(uint64_t *last, uint64_t *sample, int force) {
  uint64_t now = nr_cycles();
  if (!force && now - *last < NR_HANG_DIAG_PERIOD_CYCLES) return;
  *last = now;
  uint64_t values[32];
  for (unsigned i = 0; i < 32; i += 8) invalidate(&nr_diag_record[i]);
  invalidate(nr_diag_console);
  invalidate(&console.count);
  fence();
  values[0] = nr_diag_record[0];
  fence();
  for (unsigned i = 1; i < 32; ++i) values[i] = nr_diag_record[i];
  uint64_t waiting = nr_diag_console[0], dropped = nr_diag_console[1];
  uint64_t wait_entries = nr_diag_console[2];
  uint32_t count = console.count, consumed = console.consumed;
  invalidate(nr_diag_record);
  fence();
  uint64_t after = nr_diag_record[0];
  int stable = !(after & 1) && after == values[0];
  /* RA output may currently end midway through a line. Binary framing lets
   * decode_hang_watch.py remove this whole NH insertion byte-for-byte. */
  host_puts("\036NRWATCH1\n\r\n[nh-watch]");
  nr_diag_field(" sample=", ++*sample);
  nr_diag_field(" nh_cycles=", now);
  nr_diag_field(" stable=", stable);
  nr_diag_field(" seq=", values[0]);
  nr_diag_field(" seq_after=", after);
  if (stable) {
    nr_diag_field(" phase=", values[1]);
    nr_diag_field(" position=", values[2]);
    nr_diag_field(" detail=", values[3]);
    nr_diag_field(" target_phase=", values[4]);
    nr_diag_field(" target_call=", values[5]);
  }
  nr_diag_field(" count=", count);
  nr_diag_field(" consumed=", consumed);
  nr_diag_field(" pending=", (uint32_t)(count - consumed));
  nr_diag_field(" ra_console_wait=", waiting);
  nr_diag_field(" dropped=", dropped);
  nr_diag_field(" waits=", wait_entries);
  host_puts("\r\n");
  if (stable && values[4]) {
    for (unsigned operand = 0; operand < 3; ++operand) {
      unsigned base = 8 + operand * 8;
      host_puts("[nh-watch-memref]");
      nr_diag_field(" arg=", operand);
      nr_diag_field(" valid=", values[base + 7]);
      nr_diag_field(" descriptor=", values[base]);
      nr_diag_field(" aligned=", values[base + 1]);
      nr_diag_field(" offset=", values[base + 2]);
      nr_diag_field(" rows=", values[base + 3]);
      nr_diag_field(" cols=", values[base + 4]);
      nr_diag_field(" stride0=", values[base + 5]);
      nr_diag_field(" stride1=", values[base + 6]);
      host_puts("\r\n");
    }
  }
  host_puts("\037");
}
