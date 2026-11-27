//===- Transform.h - XSAIAME transformation passes --------------*- C++ -*-===//
//
// Part of the Buddy Compiler project, under the Apache License v2.0.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_DIALECT_XSAIAME_TRANSFORM_H
#define BUDDY_DIALECT_XSAIAME_TRANSFORM_H

#include <memory>

namespace mlir {
class LLVMConversionTarget;
class LLVMTypeConverter;
class Pass;
class RewritePatternSet;

/// Add native XSAIAME type/intrinsic and func/CF/SCF structural conversions.
/// Function-valued types are converted recursively; other conversions remain
/// under the caller's control. Patterns always emit safe configuration setters.
void populateXSAIAMELegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                  RewritePatternSet &patterns);
void configureXSAIAMELegalizeForExportTarget(
    LLVMConversionTarget &target, const LLVMTypeConverter &converter);

namespace buddy {
void registerLowerXSAIAMEPass();
} // namespace buddy
} // namespace mlir

namespace buddy::xsaiame {
/// Lower XSAIAME, retaining non-XSAIAME types and func/CF/SCF structure, then
/// deterministically deduplicate configuration within each block.
/// Native cross-function tile/sync calling conventions are not supplied.
std::unique_ptr<mlir::Pass> createLegalizeForLLVMExportPass();
} // namespace buddy::xsaiame

#endif // BUDDY_DIALECT_XSAIAME_TRANSFORM_H
