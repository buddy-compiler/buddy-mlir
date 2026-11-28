//===----------------------------------------------------------------------===//
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

#ifndef BUDDY_RUNTIME_PIPELINE_STAGEMANIFEST_H
#define BUDDY_RUNTIME_PIPELINE_STAGEMANIFEST_H

#include <charconv>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace buddy::runtime {

struct StageManifest {
  int startLayer;
  int endLayer;
  int totalLayers;
  std::string activationType;
  int activationWidth;
};

inline StageManifest
parseStageV1(const std::unordered_map<std::string, std::string> &moduleAttrs) {
  auto require = [&](const char *field) -> const std::string & {
    auto it = moduleAttrs.find(field);
    if (it == moduleAttrs.end())
      throw std::runtime_error(std::string("missing stage field ") + field);
    return it->second;
  };
  auto parseInt = [&](const char *field) {
    const std::string &value = require(field);
    int number;
    const char *end = value.data() + value.size();
    auto result = std::from_chars(value.data(), end, number);
    if (result.ec != std::errc{} || result.ptr != end)
      throw std::runtime_error(std::string(field) + " must be an integer");
    return number;
  };

  if (require("pp_stage_version") != "1")
    throw std::runtime_error("pp_stage_version must be 1");
  StageManifest stage{parseInt("pp_start_layer"), parseInt("pp_end_layer"),
                      parseInt("pp_total_layers"),
                      require("pp_activation_type"),
                      parseInt("pp_activation_width")};
  if (stage.startLayer < 0)
    throw std::runtime_error("pp_start_layer must be non-negative");
  if (stage.endLayer <= stage.startLayer)
    throw std::runtime_error("pp_end_layer must exceed pp_start_layer");
  if (stage.totalLayers < stage.endLayer)
    throw std::runtime_error("pp_total_layers must be at least pp_end_layer");
  if (stage.activationType != "f16")
    throw std::runtime_error("pp_activation_type must be f16");
  if (stage.activationWidth <= 0)
    throw std::runtime_error("pp_activation_width must be positive");
  return stage;
}

} // namespace buddy::runtime

#endif // BUDDY_RUNTIME_PIPELINE_STAGEMANIFEST_H
