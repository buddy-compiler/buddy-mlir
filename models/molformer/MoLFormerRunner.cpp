#include "buddy/runtime/models/MoLFormerRunner.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/models/ModelSession.h"
#include "buddy/Core/Container.h"
#include <filesystem><iostream><string><vector>
namespace buddy { namespace runtime {
void MoLFormerRunner::run(const RunConfig &cfgIn){RunConfig cfg=cfgIn;if(!cfg.suppressStats)std::cerr<<"\033[32;1mMoLFormer Inference\033[0m\n";std::unique_ptr<ModelSession> s;std::vector<std::string> w;if(!cfg.raxPath.empty()){ModelManifest m;s=ModelSession::createFromRax(cfg.raxPath,m);w=m.weightPaths;}else{if(cfg.modelSoPath.empty()||cfg.weightsPath.empty())throw std::runtime_error("need paths");w.push_back(cfg.weightsPath);ModelSession::Config mc;mc.modelSoPath=cfg.modelSoPath;s=ModelSession::create(mc);}s->loadWeights(w);std::cout<<"done.\n";}
} }
