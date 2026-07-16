//===- MatMulVectorizationDecodePacked.cpp -------------------------------===//
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
// Variant of matmul-vectorization-decode (m==1 GEMV) for weight operands
// that have been physically repacked offline into N-tile panels:
//
//   Bpacked[nt][k][v] == B[k, nt*vecSize + v]   for nt in [0, N/vecSize)
//
// stored as a flat row-major buffer of K*N elements. Repacking removes the
// N-stride the plain MatMulVectorizationDecode kernel pays for on the
// inner-K loop (see MatMulVectorizationDecode.cpp): consecutive k now read
// vecSize contiguous elements apart instead of N apart.
//
// This pass only rewrites matmuls whose static (K, N) operand-B shape is
// listed in the `packed-shapes` option, so it is a no-op unless a caller
// explicitly opts a given (K, N) pair in. Physically repacking the weight
// bytes at that shape is the caller's responsibility (done offline, since
// decode weights are static across calls); this pass only changes how the
// kernel addresses them.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;

namespace {

static bool hasDefaultMatmulIndexingMaps(linalg::MatmulOp op) {
  SmallVector<AffineMap, 3> maps = op.getIndexingMapsArray();
  if (maps.size() != 3)
    return false;

  MLIRContext *context = op.getContext();
  AffineExpr m = getAffineDimExpr(0, context);
  AffineExpr n = getAffineDimExpr(1, context);
  AffineExpr k = getAffineDimExpr(2, context);
  SmallVector<AffineMap, 3> expected = {
      AffineMap::get(3, 0, {m, k}, context),
      AffineMap::get(3, 0, {k, n}, context),
      AffineMap::get(3, 0, {m, n}, context),
  };

  return maps[0] == expected[0] && maps[1] == expected[1] &&
         maps[2] == expected[2];
}

static bool isZeroAttribute(Attribute attr) {
  if (auto floatAttr = dyn_cast<FloatAttr>(attr))
    return floatAttr.getValue().isZero();
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getValue().isZero();
  return false;
}

static bool isZeroGlobal(Value value, ModuleOp module) {
  auto getGlobal = value.getDefiningOp<memref::GetGlobalOp>();
  if (!getGlobal)
    return false;

  auto global = module.lookupSymbol<memref::GlobalOp>(getGlobal.getNameAttr());
  if (!global || !global.getInitialValue().has_value())
    return false;

  auto dense = dyn_cast<DenseElementsAttr>(*global.getInitialValue());
  if (!dense || !dense.isSplat())
    return false;

  return isZeroAttribute(dense.getSplatValue<Attribute>());
}

static memref::CopyOp findZeroInitCopy(Value output, linalg::MatmulOp matmulOp,
                                       ModuleOp module) {
  for (Operation *user : output.getUsers()) {
    auto copyOp = dyn_cast<memref::CopyOp>(user);
    if (!copyOp || copyOp.getTarget() != output)
      continue;
    if (copyOp->getBlock() != matmulOp->getBlock() ||
        !copyOp->isBeforeInBlock(matmulOp))
      continue;
    if (isZeroGlobal(copyOp.getSource(), module))
      return copyOp;
  }
  return nullptr;
}

/// Parses a `packed-shapes` option value like "1536x8960,8960x1536" into a
/// set of (K, N) pairs.
static llvm::SmallDenseSet<std::pair<int64_t, int64_t>, 4>
parsePackedShapes(StringRef spec) {
  llvm::SmallDenseSet<std::pair<int64_t, int64_t>, 4> result;
  SmallVector<StringRef, 4> pairs;
  spec.split(pairs, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef pair : pairs) {
    StringRef kStr, nStr;
    std::tie(kStr, nStr) = pair.split('x');
    int64_t k, n;
    if (kStr.trim().getAsInteger(10, k) || nStr.trim().getAsInteger(10, n))
      continue;
    result.insert({k, n});
  }
  return result;
}

class MatMulVectorizationDecodePackedPattern : public ConversionPattern {
public:
  MatMulVectorizationDecodePackedPattern(
      MLIRContext *ctx, ModuleOp module, int64_t vecSize,
      llvm::SmallDenseSet<std::pair<int64_t, int64_t>, 4> packedShapes)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), 1, ctx),
        module(module), vecSize(vecSize),
        packedShapes(std::move(packedShapes)) {}

  bool matchesPackedShape(linalg::MatmulOp op) const {
    auto aType = dyn_cast<MemRefType>(op.getInputs()[0].getType());
    auto bType = dyn_cast<MemRefType>(op.getInputs()[1].getType());
    auto cType = dyn_cast<MemRefType>(op.getOutputs()[0].getType());
    if (!aType || !bType || !cType)
      return false;
    if (!hasDefaultMatmulIndexingMaps(op))
      return false;
    if (!aType.hasStaticShape() || aType.getRank() != 2)
      return false;
    if (aType.getDimSize(0) != 1)
      return false;
    if (cType.getRank() != 2 || cType.getDimSize(0) != 1)
      return false;
    if (!bType.hasStaticShape() || bType.getRank() != 2)
      return false;
    int64_t k = bType.getDimSize(0);
    int64_t n = bType.getDimSize(1);
    if (n % vecSize != 0)
      return false;
    // Empty `packed-shapes` means every m==1 matmul weight in this module is
    // packed -- what pack_decode_matmul_weights guarantees -- so take them all.
    // With no exceptions there is no opt-in list to fall out of step with.
    if (packedShapes.empty())
      return true;
    return packedShapes.contains({k, n});
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto matmulOp = cast<linalg::MatmulOp>(op);
    if (!matchesPackedShape(matmulOp))
      return failure();

    auto loc = matmulOp.getLoc();

    Value A = matmulOp.getInputs()[0];
    Value B = matmulOp.getInputs()[1];
    Value C = matmulOp.getOutputs()[0];

    auto bType = cast<MemRefType>(B.getType());
    int64_t K = bType.getDimSize(0);
    int64_t N = bType.getDimSize(1);
    Type elementType = cast<MemRefType>(C.getType()).getElementType();
    auto vectorType = VectorType::get({vecSize}, elementType);

    // Reinterpret B's underlying buffer as a flat [K*N] row-major view,
    // regardless of whatever layout/offset it currently carries. The
    // physical bytes are expected (by construction, offline) to already be
    // in panel-packed order: flat[n*K + k*vecSize + v] == B[k, n + v].
    auto metadata = memref::ExtractStridedMetadataOp::create(rewriter, loc, B);
    auto flatLayout = StridedLayoutAttr::get(
        rewriter.getContext(), ShapedType::kDynamic, ArrayRef<int64_t>{1});
    auto flatType = MemRefType::get({K * N}, elementType, flatLayout);
    Value flatB = memref::ReinterpretCastOp::create(
        rewriter, loc, flatType, metadata.getBaseBuffer(), metadata.getOffset(),
        /*sizes=*/ArrayRef<OpFoldResult>{rewriter.getIndexAttr(K * N)},
        /*strides=*/ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)});

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value cK = arith::ConstantIndexOp::create(rewriter, loc, K);
    Value step = arith::ConstantIndexOp::create(rewriter, loc, vecSize);

    Value n = memref::DimOp::create(rewriter, loc, C, c1);
    Value k = memref::DimOp::create(rewriter, loc, A, c1);
    memref::CopyOp zeroInitCopy = findZeroInitCopy(C, matmulOp, module);
    bool zeroInitialized = zeroInitCopy != nullptr;

    scf::ParallelOp::create(
        rewriter, loc,
        /*lowerBounds=*/ValueRange{c0},
        /*upperBounds=*/ValueRange{n},
        /*steps=*/ValueRange{step},
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          Value nIdx = ivs.front();
          Value cVec;
          if (zeroInitialized) {
            Value zero = arith::ConstantOp::create(
                builder, loc, elementType, builder.getZeroAttr(elementType));
            cVec = vector::BroadcastOp::create(builder, loc, vectorType, zero);
          } else {
            cVec = vector::LoadOp::create(builder, loc, vectorType, C,
                                          ValueRange{c0, nIdx});
          }

          // Panel base offset: nt*K*vecSize == nIdx*K, since nIdx is always
          // an exact multiple of vecSize (the scf.parallel step).
          Value panelBase = arith::MulIOp::create(builder, loc, nIdx, cK);

          auto sumIter = scf::ForOp::create(
              builder, loc, c0, k, c1, ValueRange{cVec},
              [&](OpBuilder &builder, Location loc, Value kIdx,
                  ValueRange iterArgs) {
                Value aElem = memref::LoadOp::create(builder, loc, A,
                                                     ValueRange{c0, kIdx});
                Value aVec = vector::BroadcastOp::create(builder, loc,
                                                         vectorType, aElem);
                Value kOffset = arith::MulIOp::create(builder, loc, kIdx, step);
                Value linOffset =
                    arith::AddIOp::create(builder, loc, panelBase, kOffset);
                Value bVec = vector::LoadOp::create(
                    builder, loc, vectorType, flatB, ValueRange{linOffset});
                Value res = vector::FMAOp::create(builder, loc, aVec, bVec,
                                                  iterArgs.front());
                scf::YieldOp::create(builder, loc, res);
              });

          vector::StoreOp::create(builder, loc, sumIter.getResult(0), C,
                                  ValueRange{c0, nIdx});
        });

    rewriter.eraseOp(op);
    if (zeroInitCopy) {
      auto getGlobal =
          zeroInitCopy.getSource().getDefiningOp<memref::GetGlobalOp>();
      rewriter.eraseOp(zeroInitCopy);
      if (getGlobal && getGlobal->use_empty())
        rewriter.eraseOp(getGlobal);
    }

    return success();
  }

private:
  ModuleOp module;
  int64_t vecSize;
  llvm::SmallDenseSet<std::pair<int64_t, int64_t>, 4> packedShapes;
};

class MatMulVectorizationDecodePackedPass
    : public PassWrapper<MatMulVectorizationDecodePackedPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MatMulVectorizationDecodePackedPass)

  StringRef getArgument() const final {
    return "matmul-vectorization-decode-packed";
  }
  StringRef getDescription() const final {
    return "Vectorize linalg.matmul with m==1 for decode workloads whose B "
           "operand has been offline-repacked into N-tile panels. Only "
           "matmuls whose static (K, N) shape is listed in `packed-shapes` "
           "are rewritten; all others are left for a later pass (e.g. plain "
           "matmul-vectorization-decode) to handle.";
  }

  MatMulVectorizationDecodePackedPass() = default;
  MatMulVectorizationDecodePackedPass(
      const MatMulVectorizationDecodePackedPass &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    vector::VectorDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    auto packedShapes = parsePackedShapes(packedShapesOpt);

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                           scf::SCFDialect, vector::VectorDialect,
                           func::FuncDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
    target.addDynamicallyLegalOp<linalg::MatmulOp>(
        [&](linalg::MatmulOp op) -> bool {
          MatMulVectorizationDecodePackedPattern matcher(
              context, module, vectorSize, packedShapes);
          return !matcher.matchesPackedShape(op);
        });

    RewritePatternSet patterns(context);
    patterns.add<MatMulVectorizationDecodePackedPattern>(
        context, module, vectorSize, packedShapes);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }

  Option<int64_t> vectorSize{
      *this, "vector-size",
      llvm::cl::desc("Panel width used both by the offline repacking and by "
                     "this pass's addressing math. Must match the width "
                     "the weight file was packed with."),
      llvm::cl::init(32)};
  Option<std::string> packedShapesOpt{
      *this, "packed-shapes",
      llvm::cl::desc("Comma-separated KxN pairs (e.g. '1536x8960,8960x1536') "
                     "identifying which matmul B operands are pre-packed. "
                     "Matmuls whose (K,N) shape isn't listed are left "
                     "untouched."),
      llvm::cl::init("")};
};

} // namespace

namespace mlir {
namespace buddy {
void registerMatMulVectorizationDecodePackedPass() {
  PassRegistration<MatMulVectorizationDecodePackedPass>();
}
} // namespace buddy
} // namespace mlir
