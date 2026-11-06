//===- bare_runtime.h - Bare-metal C runtime public interface -------------===//
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
// Public ABI of the bare single-hart runtime used by BOSCAME examples.
//
// Implementation is split by responsibility:
//   bare_runtime.c         C entry, weak hooks, trap handler
//   bare_runtime_memory.c  bump heap, malloc/free, memory/string helpers
//   bare_runtime_memref.c  MLIR memrefCopy ABI
//   bare_runtime_trace.c   buddyTraceCycle* / bare_trace_* summary
//   bare_runtime_ame.c     ame_fence and AME resync
//
// Samples include this header only.  Compile-time knobs and private helpers
// live in bare_runtime_internal.h.  See runtime/boscame/README.md for the
// module table.
//
//===----------------------------------------------------------------------===//

#ifndef EXAMPLES_TOOLS_BARE_RUNTIME_H
#define EXAMPLES_TOOLS_BARE_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

// ---- bump heap (bare_runtime_memory.c) ----

// Base address of the bump-heap arena (linker / BSS-backed storage).
void *bare_runtime_heap_base(void);

// Bytes consumed from the arena so far (current bump offset).
size_t bare_runtime_heap_offset(void);

// Snapshot the current bump offset so a later reset can restore it.
void bare_runtime_preserve_heap(void);

// Restore the bump offset saved by bare_runtime_preserve_heap (or zero).
void bare_runtime_reset_heap(void);

// ---- weak lifecycle hooks (bare_runtime.c; samples may override) ----

// Print a one-line boot banner on the console UART.
void bare_runtime_print_banner(void);

// Called once after CRT init and before the application main/launch.
void bare_runtime_before_main(void);

// Called once after the application returns, before the terminal halt.
void bare_runtime_after_main(void);

// ---- cycle-trace summary (bare_runtime_trace.c) ----

// Clear recorded buddyTraceCycle* samples for this run.
void bare_trace_reset(void);

// Print the accumulated cycle-trace summary to the UART.
void bare_trace_print(void);

// ---- AME host-side fence (bare_runtime_ame.c) ----

// Order AME register / memory visibility for the host core after tile work.
void ame_fence(void);

#endif
