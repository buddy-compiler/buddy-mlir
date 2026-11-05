//===- bare_runtime_trace.c - Bare-runtime buddy cycle-trace hooks --------===//
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
// Buddy cycle-trace support for bare single-hart targets.
//
// The compiler-instrumented kernels call the `buddyTraceCycleStartPath` /
// `buddyTraceCycleEndPath` ABI; the summary is printed through the shared UART
// driver.  The trace records are private to this module, so the sample never
// links trace storage unless it also calls one of these functions.
//

#include "bare_runtime_internal.h"
#include "uart.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
  uint64_t start;
  uint64_t total;
  uint64_t minimum;
  uint64_t maximum;
  uint64_t count;
  int64_t depth;
  int64_t path[4];
} BareTraceRecord;

static BareTraceRecord trace_records[BARE_RUNTIME_TRACE_MAX_ID];
static uint64_t trace_invalid_ids = 0;

static inline uint64_t bare_trace_read_cycle(void) {
#if defined(__riscv)
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value));
  return value;
#else
  return 0;
#endif
}

static BareTraceRecord *bare_trace_record(int64_t id) {
  if (id < 0 || id >= BARE_RUNTIME_TRACE_MAX_ID) {
    ++trace_invalid_ids;
    return (BareTraceRecord *)0;
  }
  return &trace_records[id];
}

void _mlir_ciface_buddyTraceCycleStartPath(int64_t id, int64_t depth,
                                           int64_t path0, int64_t path1,
                                           int64_t path2, int64_t path3) {
  BareTraceRecord *record = bare_trace_record(id);
  if (!record)
    return;
  record->depth = depth;
  record->path[0] = path0;
  record->path[1] = path1;
  record->path[2] = path2;
  record->path[3] = path3;
  record->start = bare_trace_read_cycle();
#if BARE_RUNTIME_TRACE_LIVE
  print_uart("[buddy-trace-live] start id=0x");
  print_uart_addr((uint64_t)id);
  print_uart(" path0=0x");
  print_uart_addr((uint64_t)path0);
  print_uart("\r\n");
#endif
}

void _mlir_ciface_buddyTraceCycleEndPath(int64_t id, int64_t depth,
                                         int64_t path0, int64_t path1,
                                         int64_t path2, int64_t path3) {
  uint64_t end = bare_trace_read_cycle();
  BareTraceRecord *record = bare_trace_record(id);
  uint64_t elapsed;
  (void)depth;
  (void)path0;
  (void)path1;
  (void)path2;
  (void)path3;
  if (!record)
    return;
  elapsed = end - record->start;
  record->total += elapsed;
  if (record->count == 0 || elapsed < record->minimum)
    record->minimum = elapsed;
  if (elapsed > record->maximum)
    record->maximum = elapsed;
  ++record->count;
#if BARE_RUNTIME_TRACE_LIVE
  // Read the counter before UART output so live diagnostics do not become
  // part of the measured region. This mode is intended for locating a slow
  // or stuck region and is disabled in normal performance images.
  print_uart("[buddy-trace-live] id=0x");
  print_uart_addr((uint64_t)id);
  print_uart(" elapsed=0x");
  print_uart_addr(elapsed);
  print_uart(" count=0x");
  print_uart_addr(record->count);
  print_uart("\r\n");
#endif
}

void bare_trace_reset(void) {
  for (int64_t id = 0; id < BARE_RUNTIME_TRACE_MAX_ID; ++id) {
    trace_records[id].start = 0;
    trace_records[id].total = 0;
    trace_records[id].minimum = 0;
    trace_records[id].maximum = 0;
    trace_records[id].count = 0;
    trace_records[id].depth = 0;
    for (int64_t level = 0; level < 4; ++level)
      trace_records[id].path[level] = -1;
  }
  trace_invalid_ids = 0;
}

void bare_trace_print(void) {
  int printed_header = 0;
  for (int64_t id = 0; id < BARE_RUNTIME_TRACE_MAX_ID; ++id) {
    BareTraceRecord *record = &trace_records[id];
    if (record->count == 0)
      continue;
    if (!printed_header) {
      print_uart("[buddy-trace] cycle summary\r\n");
      printed_header = 1;
    }
    print_uart("[buddy-trace] id=0x");
    print_uart_addr((uint64_t)id);
    print_uart(" path=");
    for (int64_t level = 0; level < record->depth && level < 4; ++level) {
      if (level)
        write_serial('.');
      print_uart_addr((uint64_t)record->path[level]);
    }
    print_uart(" count=0x");
    print_uart_addr(record->count);
    print_uart(" total=0x");
    print_uart_addr(record->total);
    print_uart(" min=0x");
    print_uart_addr(record->minimum);
    print_uart(" max=0x");
    print_uart_addr(record->maximum);
    print_uart("\r\n");
  }
  if (trace_invalid_ids) {
    print_uart("[buddy-trace] invalid_ids=0x");
    print_uart_addr(trace_invalid_ids);
    print_uart("\r\n");
  }
}
