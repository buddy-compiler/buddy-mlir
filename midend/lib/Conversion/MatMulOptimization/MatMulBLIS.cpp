//===- MatMulVectorization.cpp --------------------------------------------===//
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
// This file implements the matmul vectorization.
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/InliningUtils.h>

#include "Utils/Utils.h"

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

void createBlisParams(ModuleOp module, PatternRewriter &rewriter) {
  auto loc = rewriter.getUnknownLoc();
  auto indexType = rewriter.getIndexType();

  // TODO: Change struct array to pair array
  struct Param {
    StringRef name;
    int64_t value;
  };
  Param params[] = {{"get_NC", 1024},
                    {"get_MC", 256},
                    {"get_KC", 128},
                    {"get_MR", 4},
                    {"get_NR", 8}};

  rewriter.setInsertionPointToStart(module.getBody());

  for (auto &p : params) {
    // Function type: () -> index.
    auto funcType = rewriter.getFunctionType({}, indexType);
    auto funcOp = rewriter.create<func::FuncOp>(loc, p.name, funcType);

    Block *entry = funcOp.addEntryBlock();
    OpBuilder innerBuilder(entry, entry->begin());
    Value c = innerBuilder.create<arith::ConstantIndexOp>(loc, p.value);
    innerBuilder.create<func::ReturnOp>(loc, c);

    rewriter.setInsertionPointAfter(funcOp);
  }
}

void createMicroKernel(ModuleOp module, PatternRewriter &rewriter) {
  auto loc = rewriter.getUnknownLoc();

  auto indexType = rewriter.getIndexType();
  auto f32Type = rewriter.getF32Type();
  auto memref1DType = MemRefType::get({ShapedType::kDynamic}, f32Type);
  auto memref2DType =
      MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32Type);

  auto funcType =
      rewriter.getFunctionType({indexType, memref1DType, memref1DType,
                                memref2DType, indexType, indexType, indexType},
                               {});

  auto funcOp = rewriter.create<func::FuncOp>(loc, "micro_kernel", funcType);
  Block *entryBlock = funcOp.addEntryBlock();
  OpBuilder innerBuilder(entryBlock, entryBlock->begin());

  Value k_c = entryBlock->getArgument(0);
  Value a_sliver = entryBlock->getArgument(1);
  Value b_sliver = entryBlock->getArgument(2);

  Value c = entryBlock->getArgument(3);
  Value i_start = entryBlock->getArgument(4);
  Value j_start = entryBlock->getArgument(5);
  Value n_dim = entryBlock->getArgument(6);

  Value c0 = innerBuilder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = innerBuilder.create<arith::ConstantIndexOp>(loc, 1);

  auto callOpMR =
      innerBuilder.create<func::CallOp>(loc, indexType, "get_MR", ValueRange{});
  auto callOpNR =
      innerBuilder.create<func::CallOp>(loc, indexType, "get_NR", ValueRange{});

  Value MR = callOpMR.getResult(0);
  Value NR = callOpNR.getResult(0);

  // Outer i loop (0..MR).
  // for (i = 0; i < MR; i++)
  auto iLoop = innerBuilder.create<scf::ForOp>(loc, c0, MR, c1);
  innerBuilder.setInsertionPointToStart(iLoop.getBody());
  Value i = iLoop.getInductionVar();

  Value c_i = innerBuilder.create<arith::AddIOp>(loc, i_start, i);
  Value row_acc =
      innerBuilder.create<memref::AllocaOp>(loc, memref1DType, ValueRange{NR});

  auto jInitLoop = innerBuilder.create<scf::ForOp>(loc, c0, NR, c1);
  {
    OpBuilder::InsertionGuard guard2(innerBuilder);
    innerBuilder.setInsertionPointToStart(jInitLoop.getBody());
    Value j = jInitLoop.getInductionVar();
    Value c_j_init = innerBuilder.create<arith::AddIOp>(loc, j_start, j);
    Value c_val_init =
        innerBuilder.create<memref::LoadOp>(loc, c, ValueRange{c_i, c_j_init});
    innerBuilder.create<memref::StoreOp>(loc, c_val_init, row_acc,
                                         ValueRange{j});
  }

  // K loop.
  // for (l = 0; l < k_c; l++)
  auto lLoop = innerBuilder.create<scf::ForOp>(loc, c0, k_c, c1);
  innerBuilder.setInsertionPointToStart(lLoop.getBody());
  Value l = lLoop.getInductionVar();

  // Final j loop.
  auto jStoreLoop = innerBuilder.create<scf::ForOp>(loc, c0, NR, c1);
  innerBuilder.setInsertionPointToStart(jStoreLoop.getBody());
  Value j_final = jStoreLoop.getInductionVar();
  Value c_j = innerBuilder.create<arith::AddIOp>(loc, j_start, j_final);
  Value acc_val_final =
      innerBuilder.create<memref::LoadOp>(loc, row_acc, ValueRange{j_final});
  innerBuilder.create<memref::StoreOp>(loc, acc_val_final, c,
                                       ValueRange{c_i, c_j});

  innerBuilder.setInsertionPointAfter(iLoop);
  innerBuilder.create<func::ReturnOp>(loc);

  rewriter.setInsertionPointAfter(funcOp);
}

void createMacroKernel(ModuleOp module, PatternRewriter &rewriter) {
  auto loc = rewriter.getUnknownLoc();
  auto indexType = rewriter.getIndexType();
  auto f32Type = rewriter.getF32Type();
  auto memref1DType = MemRefType::get({ShapedType::kDynamic}, f32Type);
  auto memref2DType =
      MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32Type);

  auto funcType = rewriter.getFunctionType(
      {indexType, indexType, indexType, memref2DType, indexType, indexType,
       indexType, memref1DType, memref1DType},
      {});

  auto funcOp = rewriter.create<func::FuncOp>(loc, "macro_kernel", funcType);
  Block *entryBlock = funcOp.addEntryBlock();
  OpBuilder innerBuilder(entryBlock, entryBlock->begin());

  Value m_c = entryBlock->getArgument(0);
  Value n_c = entryBlock->getArgument(1);
  Value k_c = entryBlock->getArgument(2);
  Value c = entryBlock->getArgument(3);
  Value i_c = entryBlock->getArgument(4);
  Value j_c = entryBlock->getArgument(5);
  Value n_dim = entryBlock->getArgument(6);
  Value a_tilde = entryBlock->getArgument(7);
  Value b_tilde = entryBlock->getArgument(8);

  Value c0 = innerBuilder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = innerBuilder.create<arith::ConstantIndexOp>(loc, 1);

  // MR, NR
  auto callMR =
      innerBuilder.create<func::CallOp>(loc, indexType, "get_MR", ValueRange{});
  Value MR = callMR.getResult(0);
  auto callNR =
      innerBuilder.create<func::CallOp>(loc, indexType, "get_NR", ValueRange{});
  Value NR = callNR.getResult(0);

  // jr loop: 0..n_c step NR.
  auto jrLoop = innerBuilder.create<scf::ForOp>(loc, c0, n_c, NR);
  innerBuilder.setInsertionPointToStart(jrLoop.getBody());
  Value j_r = jrLoop.getInductionVar();

  // ir loop: 0..m_c step MR.
  auto irLoop = innerBuilder.create<scf::ForOp>(loc, c0, m_c, MR);
  innerBuilder.setInsertionPointToStart(irLoop.getBody());
  Value i_r = irLoop.getInductionVar();

  // Starting indices for C tile.
  Value c_i_start = innerBuilder.create<arith::AddIOp>(loc, i_c, i_r);
  Value c_j_start = innerBuilder.create<arith::AddIOp>(loc, j_c, j_r);

  // Offsets for packed A/B panels.
  Value a_offset_index = innerBuilder.create<arith::MulIOp>(loc, i_r, k_c);
  Value b_offset_index = innerBuilder.create<arith::MulIOp>(loc, j_r, k_c);

  // memref.subview + cast
  auto a_sliver = innerBuilder.create<memref::SubViewOp>(
      loc, a_tilde, ValueRange{a_offset_index}, ValueRange{MR}, ValueRange{c1});
  auto a_sliver_cast =
      innerBuilder.create<memref::CastOp>(loc, memref1DType, a_sliver);
  auto b_sliver = innerBuilder.create<memref::SubViewOp>(
      loc, b_tilde, ValueRange{b_offset_index}, ValueRange{NR}, ValueRange{c1});
  auto b_sliver_cast =
      innerBuilder.create<memref::CastOp>(loc, memref1DType, b_sliver);

  // Call the micro kernel.
  innerBuilder.create<func::CallOp>(loc, TypeRange{}, "micro_kernel",
                                    ValueRange{k_c, a_sliver_cast,
                                               b_sliver_cast, c, c_i_start,
                                               c_j_start, n_dim});

  innerBuilder.setInsertionPointAfter(jrLoop);
  innerBuilder.create<func::ReturnOp>(loc);

  rewriter.setInsertionPointAfter(funcOp);
}

template <bool ColMajor>
void createPackFunc(ModuleOp module, PatternRewriter &rewriter,
                    StringRef funcName) {
  auto loc = rewriter.getUnknownLoc();

  // Function type: (rows, cols, memref<?x?xf32>, row_offset, col_offset,
  // memref<?xf32>) -> ().
  auto indexType = rewriter.getIndexType();
  auto f32Type = rewriter.getF32Type();
  auto memref2DType =
      MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32Type);
  auto memref1DType = MemRefType::get({ShapedType::kDynamic}, f32Type);
  auto funcType = rewriter.getFunctionType(
      {indexType, indexType, memref2DType, indexType, indexType, memref1DType},
      {});

  // Create the function operation.
  auto funcOp = rewriter.create<func::FuncOp>(loc, funcName, funcType);

  // Add the entry block.
  Block *entryBlock = funcOp.addEntryBlock();
  OpBuilder innerBuilder(entryBlock, entryBlock->begin());

  // Function arguments.
  Value rows = entryBlock->getArgument(0);
  Value cols = entryBlock->getArgument(1);
  Value mat = entryBlock->getArgument(2);
  Value row_offset = entryBlock->getArgument(3);
  Value col_offset = entryBlock->getArgument(4);
  Value mat_tilde = entryBlock->getArgument(5);

  // Constants.
  Value c0 = innerBuilder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = innerBuilder.create<arith::ConstantIndexOp>(loc, 1);

  // Outer loop: iterate j in [0, cols).
  auto jLoop = innerBuilder.create<scf::ForOp>(loc, c0, cols, c1);
  {
    OpBuilder::InsertionGuard guard2(innerBuilder);
    innerBuilder.setInsertionPointToStart(jLoop.getBody());
    Value j = jLoop.getInductionVar();

    // Inner loop: iterate i in [0, rows).
    auto iLoop = innerBuilder.create<scf::ForOp>(loc, c0, rows, c1);
    innerBuilder.setInsertionPointToStart(iLoop.getBody());
    Value i = iLoop.getInductionVar();

    // Indices in the original matrix.
    Value row_idx = innerBuilder.create<arith::AddIOp>(loc, row_offset, i);
    Value col_idx = innerBuilder.create<arith::AddIOp>(loc, col_offset, j);

    // Load from the source matrix.
    Value src_val = innerBuilder.create<memref::LoadOp>(
        loc, mat, ValueRange{row_idx, col_idx});

    // Compute destination index.
    Value dst_idx;
    if constexpr (ColMajor) {
      // Column-major layout: j * rows + i.
      Value dst_idx_base = innerBuilder.create<arith::MulIOp>(loc, j, rows);
      dst_idx = innerBuilder.create<arith::AddIOp>(loc, dst_idx_base, i);
    } else {
      // Row-major layout: i * cols + j.
      Value dst_idx_base = innerBuilder.create<arith::MulIOp>(loc, i, cols);
      dst_idx = innerBuilder.create<arith::AddIOp>(loc, dst_idx_base, j);
    }

    // Store into the packed buffer.
    innerBuilder.create<memref::StoreOp>(loc, src_val, mat_tilde,
                                         ValueRange{dst_idx});
  }

  innerBuilder.setInsertionPointAfter(jLoop);
  innerBuilder.create<func::ReturnOp>(loc);
}

void createPackFunctions(ModuleOp module, PatternRewriter &rewriter) {
  createPackFunc<true>(module, rewriter, "pack_a");
  createPackFunc<false>(module, rewriter, "pack_b");

  // module->print(llvm::outs());
}

void createSgemmBlis32(ModuleOp module, PatternRewriter &rewriter) {
  auto loc = rewriter.getUnknownLoc();
  auto indexType = rewriter.getIndexType();
  auto f32Type = rewriter.getF32Type();
  auto memref2DType =
      MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32Type);
  auto memref1DType = MemRefType::get({ShapedType::kDynamic}, f32Type);

  auto funcType =
      rewriter.getFunctionType({memref2DType, memref2DType, memref2DType}, {});
  auto funcOp = rewriter.create<func::FuncOp>(loc, "sgemm_blis_32", funcType);
  Block *entry = funcOp.addEntryBlock();
  OpBuilder innerBuilder(entry, entry->begin());

  Value a = entry->getArgument(0);
  Value b = entry->getArgument(1);
  Value c = entry->getArgument(2);

  Value c0 = innerBuilder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = innerBuilder.create<arith::ConstantIndexOp>(loc, 1);

  Value M = innerBuilder.create<memref::DimOp>(loc, a, c0);
  Value N = innerBuilder.create<memref::DimOp>(loc, c, c1);
  Value K = innerBuilder.create<memref::DimOp>(loc, a, c1);

  auto callNC =
      innerBuilder.create<func::CallOp>(loc, indexType, "get_NC", ValueRange{});
  Value NC = callNC.getResult(0);
  auto callMC =
      innerBuilder.create<func::CallOp>(loc, indexType, "get_MC", ValueRange{});
  Value MC = callMC.getResult(0);
  auto callKC =
      innerBuilder.create<func::CallOp>(loc, indexType, "get_KC", ValueRange{});
  Value KC = callKC.getResult(0);

  Value mc_kc_size = innerBuilder.create<arith::MulIOp>(loc, MC, KC);
  Value a_tilde = innerBuilder.create<memref::AllocOp>(loc, memref1DType,
                                                       ValueRange{mc_kc_size});

  Value kc_nc_size = innerBuilder.create<arith::MulIOp>(loc, KC, NC);
  Value b_tilde = innerBuilder.create<memref::AllocOp>(loc, memref1DType,
                                                       ValueRange{kc_nc_size});

  // Loop 5: jc in [0, N) with step NC.
  auto jcLoop = innerBuilder.create<scf::ForOp>(loc, c0, N, NC);
  innerBuilder.setInsertionPointToStart(jcLoop.getBody());
  Value j_c = jcLoop.getInductionVar();

  // n_c = min(N - j_c, NC).
  Value n_remain = innerBuilder.create<arith::SubIOp>(loc, N, j_c);
  Value n_c = innerBuilder.create<arith::MinSIOp>(loc, n_remain, NC);

  // Loop 4: pc in [0, K) with step KC.
  auto pcLoop = innerBuilder.create<scf::ForOp>(loc, c0, K, KC);
  innerBuilder.setInsertionPointToStart(pcLoop.getBody());
  Value p_c = pcLoop.getInductionVar();

  Value k_remain = innerBuilder.create<arith::SubIOp>(loc, K, p_c);
  Value k_c = innerBuilder.create<arith::MinSIOp>(loc, k_remain, KC);

  // Pack matrix B to b_tilde.
  innerBuilder.create<func::CallOp>(loc, TypeRange{}, "pack_b",
                                    ValueRange{k_c, n_c, b, p_c, j_c, b_tilde});

  auto icLoop = innerBuilder.create<scf::ForOp>(loc, c0, M, MC);
  innerBuilder.setInsertionPointToStart(icLoop.getBody());
  Value i_c = icLoop.getInductionVar();

  Value m_remain = innerBuilder.create<arith::SubIOp>(loc, M, i_c);
  Value m_c = innerBuilder.create<arith::MinSIOp>(loc, m_remain, MC);

  // Pack matrix A to a_tilde.
  innerBuilder.create<func::CallOp>(loc, TypeRange{}, "pack_a",
                                    ValueRange{m_c, k_c, a, i_c, p_c, a_tilde});

  // Invoke the macro kernel.
  innerBuilder.create<func::CallOp>(
      loc, TypeRange{}, "macro_kernel",
      ValueRange{m_c, n_c, k_c, c, i_c, j_c, N, a_tilde, b_tilde});

  innerBuilder.setInsertionPointToEnd(entry);

  innerBuilder.create<memref::DeallocOp>(loc, a_tilde);
  innerBuilder.create<memref::DeallocOp>(loc, b_tilde);

  innerBuilder.create<func::ReturnOp>(loc);
}

void createInlineCall(ModuleOp module, func::CallOp call) {
  llvm::errs() << "inlining failed" << call << "\n";
  call.emitError("inlining failed");
  return;
}

class MatMulBLISPattern : public ConversionPattern {
public:
  explicit MatMulBLISPattern(MLIRContext *context, int64_t NC, int64_t MC,
                             int64_t KC, int64_t MR, int64_t NR)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), 1, context),
        NC(NC), MC(MC), KC(KC), MR(MR), NR(NR) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto module = op->getParentOfType<ModuleOp>();

    createBlisParams(module, rewriter);

    createMicroKernel(module, rewriter);
    createMacroKernel(module, rewriter);
    createPackFunctions(module, rewriter);

    createSgemmBlis32(module, rewriter);

    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);

    rewriter.setInsertionPointAfter(op);

    // TODO: Make float type independent
    auto call = rewriter.create<func::CallOp>(loc, TypeRange{}, "sgemm_blis_32",
                                              ValueRange{A, B, C});

    /*
    InlinerInterface interface(module->getContext());

    auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());

    inlineCall(interface, call, callee, &callee.getBody());

    rewriter.eraseOp(call);
    */

    rewriter.eraseOp(op);

    return success();
  }

private:
  int64_t NC;
  int64_t MC;
  int64_t KC;
  int64_t MR;
  int64_t NR;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// MatMulBLISPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg matmul operations to mixture of
/// Affine + Vector operations.
namespace {
class MatMulBLISPass
    : public PassWrapper<MatMulBLISPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulBLISPass)
  StringRef getArgument() const final { return "matmul-blis"; }
  StringRef getDescription() const final { return "MatMul BLIS."; }
  MatMulBLISPass() = default;
  MatMulBLISPass(const MatMulBLISPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, VectorDialect>();
  }
  Option<int64_t> NC{*this, "n_c", llvm::cl::desc("Cache block size NC"),
                     llvm::cl::init(1024)};

  Option<int64_t> MC{*this, "m_c", llvm::cl::desc("L2 cache block size MC"),
                     llvm::cl::init(256)};

  Option<int64_t> KC{*this, "k_c", llvm::cl::desc("L1 cache block size KC"),
                     llvm::cl::init(128)};

  Option<int64_t> MR{*this, "m_r", llvm::cl::desc("Register block size MR"),
                     llvm::cl::init(4)};

  Option<int64_t> NR{*this, "n_r", llvm::cl::desc("Register block size NR"),
                     llvm::cl::init(8)};
};
} // end anonymous namespace.

void MatMulBLISPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                       scf::SCFDialect, memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::CallOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<MatMulBLISPattern>(context, NC, MC, KC, MR, NR);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerMatMulBLISPass() { PassRegistration<MatMulBLISPass>(); }
} // namespace buddy
} // namespace mlir
