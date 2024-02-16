//====- Passes.h - Include the declaration and registration of passes ---===//
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
//
// This file includes the declaration and registration of passes
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_TRANSFORMS_PASSES_H
#define BUDDY_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace buddy {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Transforms/Passes.h.inc"

void populateGemminiLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, int64_t dim,
    int64_t addrLen, int64_t accRows, int64_t bankRows, size_t sizeOfElemT,
    size_t sizeOfAccT);

void configureGemminiLegalizeForExportTarget(LLVMConversionTarget &target);

} // namespace buddy
} // namespace mlir

#endif // BUDDY_TRANSFORMS_PASSES_H
