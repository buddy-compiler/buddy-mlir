//====- Transform.h - IME Transform Passes -----===//
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

#ifndef IME_TRANSFORM_H
#define IME_TRANSFORM_H

#include "mlir/Pass/Pass.h"

namespace mlir {

class LLVMConversionTarget;
class LLVMTypeConverter;
class RewritePatternSet;

void populateIMELegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                              RewritePatternSet &patterns);
void configureIMELegalizeForExportTarget(LLVMConversionTarget &target);

} // namespace mlir

namespace buddy {
namespace ime {

std::unique_ptr<mlir::Pass> createLegalizeForLLVMExportPass();

} // namespace ime
} // namespace buddy

#endif // IME_TRANSFORM_H
