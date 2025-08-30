//===- Transforms.h - RVV Dialect Transformation Entrypoints -*- C++ -*-===//
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

#ifndef RVV_TRANSFORMS_H
#define RVV_TRANSFORMS_H

namespace mlir {

class LLVMConversionTarget;
class LLVMTypeConverter;
class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

/// Collect a set of patterns to lower RVV ops to ops that map to LLVM
/// intrinsics.
void populateRVVLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                              RewritePatternSet &patterns,
                                              int64_t RVVIndexBitwidth);

/// Configure the target to support lowering RVV ops to ops that map to LLVM
/// intrinsics.
void configureRVVLegalizeForExportTarget(LLVMConversionTarget &target);

} // namespace mlir

#endif // RVV_TRANSFORMS_H
