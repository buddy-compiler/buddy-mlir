//===- Transform.h - BOSC AME Dialect Transformation Passes ----===//
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
// Pass factories and LLVM-export pattern hooks for the `bosc_ame` dialect.
// Covers unified `-lower-bosc-ame`, FPGA `-lower-bosc-ame-tile`, and standalone
// SPM/DMA contract checking. Pattern population is shared with the tile path
// so both entry points keep the same LLVM dialect boundary.
//
//===----------------------------------------------------------------------===//

#ifndef BOSCAME_TRANSFORM_H
#define BOSCAME_TRANSFORM_H

#include "mlir/Pass/Pass.h"

namespace mlir {

class LLVMConversionTarget;
class LLVMTypeConverter;
class ModuleOp;
class RewritePatternSet;

// Add rewrite patterns that lower BOSCAME ops to LLVM dialect / runtime calls.
void populateBOSCAMELegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                  RewritePatternSet &patterns);

// Mark which ops remain illegal / legal on the LLVM export conversion target.
void configureBOSCAMELegalizeForExportTarget(LLVMConversionTarget &target);

} // namespace mlir

namespace buddy {
namespace boscame {

// The unified BOSCAME lowering pass (`-lower-bosc-ame`).  It verifies the
// SPM/DMA contract, lowers AME matrix operations to `bosc_ame.intr.*`, lowers
// `bosc_ame.spm.view` to an LLVM memref descriptor anchored at the fixed bank
// base address, lowers DMA requests to LLVM calls into the runtime ABI and
// finishes the module with the standard memref/func/arith/cf/vector/math
// lowering so that the pass leaves a module ready for LLVM translation.
std::unique_ptr<mlir::Pass> createLegalizeForLLVMExportPass();

// The FPGA tile entry point (`-lower-bosc-ame-tile`).  It runs the same
// lowering body as `-lower-bosc-ame` - one `LLVMTypeConverter`, one conversion
// target, one pattern set, the same LLVM-dialect boundary - and only adds the
// requirement that the module selects an AME hardware contract this build
// implements.  See `Transforms/BOSCAMETileLowering.h` for the tile pattern
// population interface and the supported contract.
std::unique_ptr<mlir::Pass> createLowerBOSCAMETilePass();

// Verify the cross-operation SPM/DMA contract: token lifetime, channel
// occupancy and source/destination aliasing.  Exposed so that the unified
// lowering pass can run the same checks before the SPM types are converted.
mlir::LogicalResult verifySpmDmaContract(mlir::ModuleOp module);

// Verify the cross-operation SPM/DMA contract as a standalone pass, kept for
// debugging a pipeline in isolation.
std::unique_ptr<mlir::Pass> createSpmDmaCheckPass();

} // namespace boscame
} // namespace buddy

#endif // BOSCAME_TRANSFORM_H
