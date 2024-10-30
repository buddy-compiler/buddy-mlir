//===- GemmCodegenTransform.h ---------------------------------------------===//
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

#ifndef PIPELINES_GPU_GEMM_CODEGEN_TRANSOFRM_H
#define PIPELINES_GPU_GEMM_CODEGEN_TRANSOFRM_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace buddy {
struct GPUGemmCodegenConfigOptions
    : public PassPipelineOptions<GPUGemmCodegenConfigOptions> {
  Option<std::string> funcAnchor{
      *this, "func-anchor",
      llvm::cl::desc(
          "An optional Unit attribute anchoring on target functions."),
      llvm::cl::init("")};
  Option<std::string> annotatePrefix{
      *this, "annotate-prefix",
      llvm::cl::desc("An optional annotate prefix attribute on target ops."),
      llvm::cl::init("__buddy_gpu_gemm__")};
  ListOption<int64_t> tileConfig{
      *this, "tile-config",
      llvm::cl::desc("An optional tile config for matmul op")};
  ListOption<int64_t> workGroup{
      *this, "work-group",
      llvm::cl::desc("An optional workgroup size config for matmul op")};
  Option<int64_t> stages{
      *this, "stages",
      llvm::cl::desc("An optional stages config for matmul op")};
};

struct GPUGemmGeneralOptions
    : public PassPipelineOptions<GPUGemmGeneralOptions> {
  Option<std::string> funcAnchor{
      *this, "func-anchor",
      llvm::cl::desc(
          "An optional Unit attribute anchoring on target functions."),
      llvm::cl::init("")};
  Option<std::string> annotatePrefix{
      *this, "annotate-prefix",
      llvm::cl::desc("An optional annotate prefix attribute on target ops."),
      llvm::cl::init("__buddy_gpu_gemm__")};
};

void createGemmTileConfigInsertTransform(
    OpPassManager &pm, const GPUGemmCodegenConfigOptions &options);

void createGemmTileTransform(OpPassManager &pm,
                             const GPUGemmGeneralOptions &options);

} // namespace buddy
} // namespace mlir

#endif
