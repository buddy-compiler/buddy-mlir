//===- Transform.h - AME Dialect Transformation Passes ----------*- C++ -*-===//
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

#ifndef AME_TRANSFORM_H
#define AME_TRANSFORM_H

#include "mlir/Pass/Pass.h"

namespace mlir {

class LLVMConversionTarget;
class LLVMTypeConverter;
class RewritePatternSet;

void populateAMELegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                              RewritePatternSet &patterns);
void configureAMELegalizeForExportTarget(LLVMConversionTarget &target);

} // namespace mlir

namespace buddy {
namespace ame {

std::unique_ptr<mlir::Pass> createLegalizeForLLVMExportPass();

} // namespace ame
} // namespace buddy

#endif // AME_TRANSFORM_H
