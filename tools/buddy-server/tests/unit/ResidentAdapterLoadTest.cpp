//===- ResidentAdapterLoadTest.cpp - Production resident plugin checks
//------===//
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

#include "ModelPluginDiscovery.h"
#include "ResidentModelPluginHandle.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error("expected plugin path and model type");

    // Both production loaders use RTLD_NOW, so missing runtime symbols fail
    // here even on platforms without the Linux -z defs link check.
    const auto discovery = buddy::server::discoverModelPlugin(argv[1]);
    if (discovery.serviceKind != buddy::runtime::ModelServiceKind::Resident)
      throw std::runtime_error("plugin descriptor is not resident");

    buddy::server::ResidentModelPluginHandle plugin(discovery.pluginPath);
    if (plugin.modelType() != argv[2])
      throw std::runtime_error("unexpected resident model type: " +
                               plugin.modelType());

    // Keep each model inside the handle lifetime and exercise its real
    // factory/destructor twice. No load() call, weights, kernels, or device
    // are needed to inspect the initial v2 runtime state.
    for (int instance = 0; instance < 2; ++instance) {
      auto model = plugin.createModel();
      if (model->status().state != buddy::runtime::ModelLoadState::Unloaded ||
          model->isLoaded())
        throw std::runtime_error("new resident model must be unloaded");

      // No production adapter has certified the compatible serving path yet.
      // Query every capability through the new vtable before loading weights.
      const auto caps = model->capabilities();
      const auto &sampling = caps.sampling;
      if (caps.rawTokenize || caps.rawDetokenize || caps.tokenizeAddSpecial ||
          caps.tokenizeParseSpecial || caps.tokenizeWithPieces ||
          !caps.chatRoles.empty() || caps.chatMultiTurn ||
          caps.maxImages != 0 || caps.imagePaths || caps.imageBytes ||
          caps.streaming || caps.maxNewTokens || caps.stop ||
          caps.stopTokenIds || sampling.temperature || sampling.topK ||
          sampling.topP || sampling.minP || sampling.repeatPenalty ||
          sampling.repeatLastN || sampling.seed)
        throw std::runtime_error(
            "unloaded resident capabilities must be empty");

      bool rejected = false;
      try {
        (void)model->detokenize({{1}});
      } catch (const std::invalid_argument &error) {
        rejected = std::string(error.what()).find("not supported") !=
                   std::string::npos;
      }
      if (!rejected)
        throw std::runtime_error("raw detokenize must reject unsupported use");
    }

    std::cout << argv[2] << " resident adapter load checks passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Resident adapter check failed: " << error.what() << '\n';
    return 1;
  }
}
