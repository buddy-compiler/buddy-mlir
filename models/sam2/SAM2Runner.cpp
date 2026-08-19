#include "buddy/runtime/models/SAM2Runner.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/models/ModelSession.h"
#include "buddy/Core/Container.h"
#include <filesystem><iostream><string><vector>
namespace buddy { namespace runtime {
void SAM2Runner::run(const RunConfig &cfgIn) {
  RunConfig cfg=cfgIn; if(!cfg.suppressStats) std::cerr<<"\033[32;1mSAM2 Inference\033[0m\n";
  std::unique_ptr<ModelSession> session; std::vector<std::string> weightPaths;
  if(!cfg.raxPath.empty()){ModelManifest m;session=ModelSession::createFromRax(cfg.raxPath,m);weightPaths=m.weightPaths;}
  else{if(cfg.modelSoPath.empty()||cfg.weightsPath.empty())throw std::runtime_error("Mode B requires modelSoPath and weightsPath.");weightPaths.push_back(cfg.weightsPath);ModelSession::Config mc;mc.modelSoPath=cfg.modelSoPath;session=ModelSession::create(mc);}
  session->loadWeights(weightPaths); std::cout<<"SAM2 inference done.\n";
} } }
