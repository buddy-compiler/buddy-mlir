//===- RVProfRuntime.h - RVProf Runtime Library -----------------*- C -*-===//
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
// This file declares the RVProf runtime library for RISC-V profiling.
//
//===----------------------------------------------------------------------===//

#ifndef RVPROF_RUNTIME_H
#define RVPROF_RUNTIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize RVProf runtime
void __rvprof_init(void);

// Mark the beginning of a profiled region
void __rvprof_region_begin(const char *name);

// Mark the end of a profiled region
void __rvprof_region_end(const char *name);

// Dump profiling data to a Perfetto trace file
void __rvprof_dump(const char *output_file);

#ifdef __cplusplus
}
#endif

#endif // RVPROF_RUNTIME_H
