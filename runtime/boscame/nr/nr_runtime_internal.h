//===- nr_runtime_internal.h - Internal NR runtime contracts and hooks ----===//
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

#ifndef BUDDY_FPGA_NR_RUNTIME_INTERNAL_H
#define BUDDY_FPGA_NR_RUNTIME_INTERNAL_H

//
// Private interface between the NR platform runtime and its memory support
// module.
//
// nr_runtime.c owns the platform side: NH boot and RA release, the completion
// mailbox at 0x80010000, the inter-core console forwarding (nr_console.inc),
// the optional hang watch (nr_hang_watch.inc), the cache/fence hooks the
// includes need, the RA/NH trap entries and the final result report.
//
// nr_memory.c owns the runtime side: the RA bump heap, the scalar memory
// helpers and the MLIR memrefCopy ABI.  The exported memory ABI is declared in
// the public "nr_runtime.h"; this header only carries what both translation
// units need to share.  Applications never include it.
//
// Both modules are linked into every NR image; the object list lives in
// nr.mk / the sample common.mk.
//

#include <stdint.h>

//
// Terminal report of the RA result mailbox (PASS/FAIL), owned by nr_runtime.c.
// Called by the platform trap entries and, on a fatal runtime error such as an
// exhausted heap or an ABI violation in memrefCopy, by nr_memory.c.
//
__attribute__((noreturn)) void nr_finish(int status);

// Defined by nr_runtime.c; publishes FAIL and stops the RA core.
__attribute__((noreturn)) void abort(void);

#endif
