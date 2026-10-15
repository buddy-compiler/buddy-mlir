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
    // are needed to inspect the initial v1 runtime state.
    for (int instance = 0; instance < 2; ++instance) {
      auto model = plugin.createModel();
      if (model->status().state != buddy::runtime::ModelLoadState::Unloaded ||
          model->isLoaded())
        throw std::runtime_error("new resident model must be unloaded");
    }

    std::cout << argv[2] << " resident adapter load checks passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Resident adapter check failed: " << error.what() << '\n';
    return 1;
  }
}
