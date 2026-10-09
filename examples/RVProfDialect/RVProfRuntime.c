//===- RVProfRuntime.c - RVProf Runtime Library -----------------*- C -*-===//
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
// This file implements the RVProf runtime library for RISC-V profiling.
// It uses the rdcycle instruction to measure execution cycles and outputs
// Perfetto trace format with cycle counts (no time conversion).
//
//===----------------------------------------------------------------------===//

#include "RVProfRuntime.h"
#include <stdio.h>
#include <stdlib.h>

// Compile-time check: only support RISC-V
#ifndef __riscv
#error "RVProf only supports RISC-V platform. Use -march=rv64gc or similar."
#endif

#define MAX_EVENTS 10000

typedef struct {
  const char *name;
  uint64_t start_cycles;
  uint64_t end_cycles;
} rvprof_event_t;

static rvprof_event_t events[MAX_EVENTS];
static int event_count = 0;
static uint64_t total_start_cycles = 0;
static uint64_t total_end_cycles = 0;

static inline uint64_t rdcycle(void) {
  uint64_t cycles;
  asm volatile("rdcycle %0" : "=r"(cycles));
  return cycles;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

void __rvprof_init(void) {
  event_count = 0;
  total_start_cycles = rdcycle();
}

void __rvprof_region_begin(const char *name) {
  if (event_count >= MAX_EVENTS) {
    fprintf(stderr, "ERROR: RVProf event buffer overflow (max %d events).\n",
            MAX_EVENTS);
    exit(1);
  }
  events[event_count].name = name;
  events[event_count].start_cycles = rdcycle();
}

void __rvprof_region_end(const char *name) {
  if (event_count >= MAX_EVENTS) {
    fprintf(stderr, "ERROR: RVProf event buffer overflow (max %d events).\n",
            MAX_EVENTS);
    exit(1);
  }
  events[event_count].end_cycles = rdcycle();
  event_count++;
}

void __rvprof_dump(const char *output_file) {
  total_end_cycles = rdcycle();

  FILE *f = fopen(output_file, "w");
  if (!f) {
    fprintf(stderr, "ERROR: Failed to open %s for writing.\n", output_file);
    exit(1);
  }

  // Calculate statistics
  uint64_t total_cycles = total_end_cycles - total_start_cycles;
  uint64_t profiled_cycles = 0;
  for (int i = 0; i < event_count; i++) {
    profiled_cycles += events[i].end_cycles - events[i].start_cycles;
  }

  double coverage = 100.0 * profiled_cycles / total_cycles;

  // Write Perfetto trace format (using cycles directly, no time conversion)
  fprintf(f, "{\n");
  fprintf(f, "  \"traceEvents\": [\n");

  for (int i = 0; i < event_count; i++) {
    uint64_t start_cycles = events[i].start_cycles - total_start_cycles;
    uint64_t dur_cycles = events[i].end_cycles - events[i].start_cycles;

    // Use cycles directly as "microseconds" for Perfetto
    // Perfetto will display these as time units, but they're actually cycles
    fprintf(f,
            "    "
            "{\"name\":\"%s\",\"ph\":\"X\",\"ts\":%llu,\"dur\":%llu,\"pid\":1,"
            "\"tid\":1}",
            events[i].name, (unsigned long long)start_cycles,
            (unsigned long long)dur_cycles);

    if (i < event_count - 1) {
      fprintf(f, ",\n");
    } else {
      fprintf(f, "\n");
    }
  }

  fprintf(f, "  ],\n");
  fprintf(f, "  \"displayTimeUnit\": \"ns\",\n");
  fprintf(f, "  \"metadata\": {\n");
  fprintf(f, "    \"unit\": \"cycles\",\n");
  fprintf(f, "    \"total_cycles\": %llu,\n", (unsigned long long)total_cycles);
  fprintf(f, "    \"profiled_cycles\": %llu,\n",
          (unsigned long long)profiled_cycles);
  fprintf(f, "    \"coverage_percent\": %.2f,\n", coverage);
  fprintf(f, "    \"event_count\": %d\n", event_count);
  fprintf(f, "  }\n");
  fprintf(f, "}\n");

  if (fclose(f) != 0) {
    fprintf(stderr, "ERROR: Failed to close %s.\n", output_file);
    exit(1);
  }

  // Print summary to stderr
  fprintf(stderr, "\n=== RVProf Summary ===\n");
  fprintf(stderr, "Total cycles:     %llu\n", (unsigned long long)total_cycles);
  fprintf(stderr, "Profiled cycles:  %llu\n",
          (unsigned long long)profiled_cycles);
  fprintf(stderr, "Coverage:         %.2f%%\n", coverage);
  fprintf(stderr, "Events:           %d\n", event_count);
  fprintf(stderr, "Output:           %s\n", output_file);
  fprintf(
      stderr,
      "Note: Perfetto displays cycles as time units (ignore the time scale)\n");
}
