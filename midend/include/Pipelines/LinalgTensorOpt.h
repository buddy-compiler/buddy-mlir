//===- LinalgTensorOpt.h --------------------------------------------------===//
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

#ifndef PIPELINES_GPU_LINALGTENSOROPT_H
#define PIPELINES_GPU_LINALGTENSOROPT_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <memory>
#include <string>

namespace mlir {
namespace buddy {

struct LinalgTensorOptPipelineOptions
    : public PassPipelineOptions<LinalgTensorOptPipelineOptions> {
  Option<std::string> target{
      *this, "target",
      llvm::cl::desc("An optional attribute to speicify target."),
      llvm::cl::init("gpu")};
  Option<std::string> arch{
      *this, "arch", llvm::cl::desc("An optional attribute to speicify arch."),
      llvm::cl::init("nv_sm_80")};
};

void createLinalgTensorOptPassPipeline(
    OpPassManager &pm, const LinalgTensorOptPipelineOptions &options);

void registerLinalgTensorOptPassPipeline();

} // namespace buddy
} // namespace mlir

#endif
