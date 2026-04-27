//===- Transform.h - MLIR Dialect for RISC-V Buckyball extension ---------===//
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

#ifndef BUCKYBALL_TRANSLATE_H
#define BUCKYBALL_TRANSLATE_H

#include <cstdint>

namespace mlir {

class LLVMConversionTarget;
class LLVMTypeConverter;
class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

void populateBuckyballLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    int64_t lane, int64_t warp, int64_t bankDepth, int64_t bankNum);
void configureBuckyballLegalizeForExportTarget(LLVMConversionTarget &target);

} // namespace mlir

#endif // BUCKYBALL_TRANSLATE_H
