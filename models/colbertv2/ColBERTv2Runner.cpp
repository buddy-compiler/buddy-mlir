#include "buddy/runtime/models/ColBERTv2Runner.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/models/ModelSession.h"
#include "buddy/Core/Container.h"
#include <filesystem><iostream><string><vector>
namespace buddy { namespace runtime {
void ColBERTv2Runner::run(const RunConfig &cfgIn) {
  RunConfig cfg = cfgIn;
  if (!cfg.suppressStats) std::cerr << "\033[32;1mColBERTv2 Inference (buddy-cli)\033[0m\n";
  std::unique_ptr<ModelSession> session; std::vector<std::string> weightPaths;
  if (!cfg.raxPath.empty()) { ModelManifest manifest; session = ModelSession::createFromRax(cfg.raxPath, manifest); weightPaths = manifest.weightPaths; }
  else { if (cfg.modelSoPath.empty() || cfg.weightsPath.empty()) throw std::runtime_error("Mode B requires modelSoPath and weightsPath."); weightPaths.push_back(cfg.weightsPath); ModelSession::Config mcfg; mcfg.modelSoPath = cfg.modelSoPath; session = ModelSession::create(mcfg); }
  session->loadWeights(weightPaths);
  std::cout << "ColBERTv2 inference done.\n";
} } }
