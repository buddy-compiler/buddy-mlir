//===- BOSCAMEConversionUtils.h - Shared LLVM conversion helpers ------===//
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
// Helpers shared by the BOSCAME conversion patterns
// (`LegalizeForLLVMExport.cpp` and `BOSCAMETileLowering.cpp`).  They are
// internal to the transform library: the point is that the generic BOSCAME
// lowering and the tile lowering decode a converted memref descriptor and build
// a `bosc_ame.intr.*` operation in the same way, instead of each owning a
// slightly different copy.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_LIB_DIALECT_BOSCAME_TRANSFORMS_BOSCAMECONVERSIONUTILS_H
#define BUDDY_LIB_DIALECT_BOSCAME_TRANSFORMS_BOSCAMECONVERSIONUTILS_H

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace buddy {
namespace boscame {
namespace detail {

// The scalable register type the backend uses for one AME matrix value.
//
// The 2-D vector shape of a matrix value is the *logical* tile shape; the
// physical matrix register has a fixed storage capacity, expressed here as
// `vscale x <lanes> x <element type>`.  The mapping is element-type only, so
// `vector<64x64xi8>` and `vector<4x4xi8>` describe the same register, while
// the tile configuration (`msettilem/n/k`) carries the shape.
inline mlir::Type getBOSCAMERegisterType(mlir::Type type) {
  auto matrixType = mlir::dyn_cast<mlir::VectorType>(type);
  if (!matrixType)
    return mlir::Type();

  mlir::Type elementType = matrixType.getElementType();
  unsigned lanes = 0;
  if (auto integerType = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
    switch (integerType.getWidth()) {
    case 8:
      lanes = 128;
      break;
    case 16:
      lanes = 64;
      break;
    case 32:
      lanes = 32;
      break;
    case 64:
      lanes = 16;
      break;
    default:
      return mlir::Type();
    }
  } else {
    auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType);
    if (!floatType)
      return mlir::Type();
    switch (floatType.getWidth()) {
    case 16:
      lanes = 64;
      break;
    case 32:
      lanes = 32;
      break;
    case 64:
      lanes = 16;
      break;
    default:
      return mlir::Type();
    }
  }

  // i32 + 32 + scalable produces <vscale x 32 x i32> in LLVM IR, i.e.
  // RISCV MVT::nxv32i32, registered to TileReg/AccReg by the backend.
  return mlir::LLVM::getVectorType(elementType, lanes, true);
}

// Decode the effective base pointer of a converted memref descriptor:
//
//   effective = alignedPtr + descriptorOffset * elementBytes
//
// `MemRefDescriptor::bufferPtr` folds in the descriptor offset (dynamic
// layouts) or the type's static offset (static layouts), so arbitrarily
// nested subview/reinterpret_cast chains and strided layouts are resolved
// exactly once.  This is the only place a descriptor becomes a pointer, which
// is what guarantees that the descriptor offset is applied exactly once.
inline mlir::Value extractPointerFromDescriptor(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value descriptor,
    mlir::Type originalMemRefType, const mlir::LLVMTypeConverter &converter) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(originalMemRefType);
  if (!memrefType || !descriptor)
    return mlir::Value();
  mlir::MemRefDescriptor memref(descriptor);
  return memref.bufferPtr(builder, loc, converter, memrefType);
}

// `bosc_ame.<suffix>` becomes the TableGen-defined `bosc_ame.intr.<suffix>`.
inline mlir::Operation *
createBOSCAMEIntrinsicOp(mlir::ConversionPatternRewriter &rewriter,
                         mlir::Location loc, llvm::StringRef sourceName,
                         mlir::TypeRange resultTypes,
                         mlir::ValueRange operands) {
  static const llvm::StringRef dialectPrefix = "bosc_ame.";
  llvm::StringRef suffix = sourceName.drop_front(dialectPrefix.size());
  std::string intrinsicName = "bosc_ame.intr.";
  intrinsicName += suffix;

  mlir::OperationState state(loc, intrinsicName);
  state.addOperands(operands);
  state.addTypes(resultTypes);
  return rewriter.create(state);
}

// Create the `bosc_ame.intr.<name>` configuration intrinsic with one i64
// operand and one i64 result, i.e. the register form of
// `msettilem/msettilen/msettilek/msettype`.
inline void createConfigIntrinsic(mlir::ConversionPatternRewriter &rewriter,
                                  mlir::Location loc, llvm::StringRef name,
                                  int64_t value) {
  mlir::Type i64Type = rewriter.getI64Type();
  mlir::Value constant = mlir::LLVM::ConstantOp::create(
      rewriter, loc, i64Type, rewriter.getI64IntegerAttr(value));
  std::string intrinsicName = "bosc_ame.intr.";
  intrinsicName += name.str();
  mlir::OperationState state(loc, intrinsicName);
  state.addOperands(constant);
  state.addTypes(i64Type);
  rewriter.create(state);
}

// Emit an `llvm.mlir.constant` of type i64.
inline mlir::Value createI64Constant(mlir::OpBuilder &builder,
                                     mlir::Location loc, int64_t value) {
  return mlir::LLVM::ConstantOp::create(builder, loc, builder.getI64Type(),
                                        builder.getI64IntegerAttr(value));
}

// Emit an `llvm.mlir.constant` of type i32 (value truncated to 32 bits).
inline mlir::Value createI32Constant(mlir::OpBuilder &builder,
                                     mlir::Location loc, int64_t value) {
  return mlir::LLVM::ConstantOp::create(
      builder, loc, builder.getI32Type(),
      builder.getI32IntegerAttr(static_cast<int32_t>(value)));
}

// True when `type` is a ranked memref in an SPM memory space.  The memory
// space attribute itself lives in the dialect; callers pass the predicate in
// to keep this header independent of the attribute definition.
template <typename SpmAttrT> inline bool hasSpmSpace(mlir::Type type) {
  auto memref = mlir::dyn_cast_or_null<mlir::MemRefType>(type);
  return memref && mlir::isa_and_nonnull<SpmAttrT>(memref.getMemorySpace());
}

} // namespace detail
} // namespace boscame
} // namespace buddy

#endif // BUDDY_LIB_DIALECT_BOSCAME_TRANSFORMS_BOSCAMECONVERSIONUTILS_H
