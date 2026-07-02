//===- ResidentModelPlugin.h - Resident model plugin C ABI ------*- C++ -*-===//
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
// Resident model plugins export C symbols that create and destroy a concrete
// ResidentModel instance. The returned object uses buddy-mlir's C++ runtime
// ABI, so plugins should be built against the same BuddyRuntime as
// buddy-server.
//
// Required exports:
//   extern "C" buddy::runtime::ResidentModel *
//   buddy_create_resident_model_v1();
//
//   extern "C" void
//   buddy_destroy_resident_model_v1(buddy::runtime::ResidentModel *);
//
// Optional export:
//   extern "C" const char *buddy_resident_model_type_v1();
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_CORE_RESIDENTMODELPLUGIN_H
#define BUDDY_RUNTIME_CORE_RESIDENTMODELPLUGIN_H

#include "buddy/runtime/core/ResidentModel.h"

namespace buddy {
namespace runtime {

using CreateResidentModelFn = ResidentModel *(*)();
using DestroyResidentModelFn = void (*)(ResidentModel *);
using ResidentModelTypeFn = const char *(*)();

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_CORE_RESIDENTMODELPLUGIN_H
