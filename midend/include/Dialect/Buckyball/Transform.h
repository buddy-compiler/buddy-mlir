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

typedef uint32_t acc_scale_t_bits;
typedef float acc_scale_t;
typedef uint32_t scale_t_bits;
typedef float scale_t;
typedef int32_t scale_acc_t;

namespace mlir {

class LLVMConversionTarget;
class LLVMTypeConverter;
class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

// defined in midend/lib/Dialect/Buckyball/Transforms/LegalizeForLLVMExport.cpp
void populateBuckyballLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, int64_t dim,
    int64_t memAddrLen, int64_t spAddrLen, int64_t accRows, int64_t bankRows, size_t sizeOfElemT,
    size_t sizeOfAccT, int64_t warp, int64_t lane, int64_t hartId);
void configureBuckyballLegalizeForExportTarget(LLVMConversionTarget &target);

} // namespace mlir

#endif // BUCKYBALL_TRANSLATE_H
