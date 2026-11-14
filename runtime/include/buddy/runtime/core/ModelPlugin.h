//===- ModelPlugin.h - Unified model plugin discovery ABI -------*- C++ -*-===//
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

#ifndef BUDDY_RUNTIME_CORE_MODELPLUGIN_H
#define BUDDY_RUNTIME_CORE_MODELPLUGIN_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace buddy {
namespace runtime {

enum class ModelServiceKind : uint32_t {
  Invalid = 0,
  Resident = 1,
  Embedding = 2,
  MaskedLM = 3,
  Transcription = 4,
};

inline constexpr uint32_t ModelPluginDescriptorAbiVersionV1 = 1;
inline constexpr const char ModelPluginDescriptorSymbolV1[] =
    "buddy_model_plugin_descriptor_v1";

// Keep this structure C-compatible. Plugins return a pointer to storage they
// own; the server copies and validates it before loading a typed factory.
struct ModelPluginDescriptorV1 {
  uint32_t structSize;
  uint32_t abiVersion;
  ModelServiceKind serviceKind;
  uint32_t reserved;
};

using ModelPluginDescriptorFn = const ModelPluginDescriptorV1 *(*)();

static_assert(sizeof(ModelPluginDescriptorV1) == 16,
              "ModelPluginDescriptorV1 ABI layout changed");
static_assert(alignof(ModelPluginDescriptorV1) == alignof(uint32_t),
              "ModelPluginDescriptorV1 alignment changed");
static_assert(std::is_standard_layout_v<ModelPluginDescriptorV1>);
static_assert(std::is_trivially_copyable_v<ModelPluginDescriptorV1>);
static_assert(offsetof(ModelPluginDescriptorV1, structSize) == 0);
static_assert(offsetof(ModelPluginDescriptorV1, abiVersion) == 4);
static_assert(offsetof(ModelPluginDescriptorV1, serviceKind) == 8);
static_assert(offsetof(ModelPluginDescriptorV1, reserved) == 12);
static_assert(static_cast<uint32_t>(ModelServiceKind::Invalid) == 0);
static_assert(static_cast<uint32_t>(ModelServiceKind::Resident) == 1);
static_assert(static_cast<uint32_t>(ModelServiceKind::Embedding) == 2);
static_assert(static_cast<uint32_t>(ModelServiceKind::MaskedLM) == 3);
static_assert(static_cast<uint32_t>(ModelServiceKind::Transcription) == 4);

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_CORE_MODELPLUGIN_H
