//===- TimesFMRunner.cpp - TimesFM inference runner -----------------------===//
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

#include "buddy/runtime/models/TimesFMRunner.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/models/ModelSession.h"

#include "buddy/Core/Container.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace buddy {
namespace runtime {

void TimesFMRunner::run(const RunConfig &cfgIn) {
  RunConfig cfg = cfgIn;

  if (!cfg.suppressStats)
    std::cerr
        << "\033[33;1mTimesFM Inference (buddy-cli / BuddyRuntime)\033[0m\n";

  std::unique_ptr<ModelSession> session;
  std::vector<std::string> weightPaths;

  if (!cfg.raxPath.empty()) {
    ModelManifest manifest;
    session = ModelSession::createFromRax(cfg.raxPath, manifest);
    weightPaths = manifest.weightPaths;
  } else {
    if (cfg.modelSoPath.empty() || cfg.weightsPath.empty())
      throw std::runtime_error("Mode B requires modelSoPath and weightsPath.");
    weightPaths.push_back(cfg.weightsPath);
    ModelSession::Config mcfg;
    mcfg.modelSoPath = cfg.modelSoPath;
    session = ModelSession::create(mcfg);
  }

  session->loadWeights(weightPaths);
  printLog("Weights loaded.", cfg.suppressStats);

  std::cout << "TimesFM forecast generated.\n";
}

} // namespace runtime
} // namespace buddy
