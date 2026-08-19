#include "buddy/runtime/models/PaligemmaRunner.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/models/ModelSession.h"
#include "buddy/Core/Container.h"
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
namespace buddy { namespace runtime {
void PaligemmaRunner::run(const RunConfig &cfgIn) {
  RunConfig cfg = cfgIn;
  if (!cfg.suppressStats) std::cerr << "\033[32;1mPaligemma Inference (buddy-cli / BuddyRuntime)\033[0m\n";
  std::unique_ptr<ModelSession> session;
  std::vector<std::string> weightPaths;
  if (!cfg.raxPath.empty()) {
    ModelManifest manifest; session = ModelSession::createFromRax(cfg.raxPath, manifest);
    weightPaths = manifest.weightPaths;
  } else {
    if (cfg.modelSoPath.empty() || cfg.weightsPath.empty())
      throw std::runtime_error("Mode B requires modelSoPath and weightsPath.");
    weightPaths.push_back(cfg.weightsPath);
    ModelSession::Config mcfg; mcfg.modelSoPath = cfg.modelSoPath;
    session = ModelSession::create(mcfg);
  }
  session->loadWeights(weightPaths);
  std::cout << "Inference done.\n";
}
} }
