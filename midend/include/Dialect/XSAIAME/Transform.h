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

/// Add the native XSAIAME type and intrinsic conversions and the SCF structural
/// conversions that carry tiles and sync handles through SCF regions. Patterns
/// always emit safe configuration setters.
void populateXSAIAMELegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                  RewritePatternSet &patterns);
/// Mark XSAIAME illegal. A func or cf operation that carries a tile or sync
/// handle stays illegal: the native target has no calling convention for them.
void configureXSAIAMELegalizeForExportTarget(
    LLVMConversionTarget &target, const LLVMTypeConverter &converter);

namespace buddy {
void registerLowerXSAIAMEPass();
} // namespace buddy
} // namespace mlir

namespace buddy::xsaiame {
/// Lower XSAIAME, retaining non-XSAIAME types, then deterministically
/// deduplicate configuration within each block. Tiles and sync handles may flow
/// through SCF regions but not across functions or cf branches.
std::unique_ptr<mlir::Pass> createLegalizeForLLVMExportPass();
} // namespace buddy::xsaiame

#endif // BUDDY_DIALECT_XSAIAME_TRANSFORM_H
