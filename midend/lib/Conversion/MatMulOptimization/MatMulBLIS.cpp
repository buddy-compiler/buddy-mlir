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

typedef llvm::SmallDenseMap<llvm::StringRef, int64_t> ParamsType;

//===----------------------------------------------------------------------===//
// pack_a: (A, A_packed, i_c, p_c) -> () with zero padding
// Layout: A_packed[i*KC + j], i in [0,MC), j in [0,KC)
// OOB: if (i_c+i>=M || p_c+j>=K) -> store 0
//===----------------------------------------------------------------------===//

static void createPackA(ModuleOp module, PatternRewriter &rewriter,
                        int64_t MC_val, int64_t KC_val) {
  auto loc = rewriter.getUnknownLoc();

  auto indexType = rewriter.getIndexType();
  auto f32Type = rewriter.getF32Type();
  auto memref2DType =
      MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32Type);
  auto memref1DType = MemRefType::get({ShapedType::kDynamic}, f32Type);

  auto funcType = rewriter.getFunctionType(
      {memref2DType, memref1DType, indexType, indexType}, {});

  auto funcOp = rewriter.create<func::FuncOp>(loc, "pack_a", funcType);
  Block *entry = funcOp.addEntryBlock();
  OpBuilder b(entry, entry->begin());

  Value A = entry->getArgument(0);
  Value A_packed = entry->getArgument(1);
  Value i_c = entry->getArgument(2);
  Value p_c = entry->getArgument(3);

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);

  // dims
  Value M = b.create<memref::DimOp>(loc, A, c0);
  Value K = b.create<memref::DimOp>(loc, A, c1);

  // Params
  Value MC = b.create<arith::ConstantIndexOp>(loc, MC_val);
  Value KC = b.create<arith::ConstantIndexOp>(loc, KC_val);

  // for j in [0, KC)
  auto jFor = b.create<scf::ForOp>(loc, c0, KC, c1);
  {
    OpBuilder::InsertionGuard g1(b);
    b.setInsertionPointToStart(jFor.getBody());
    Value j = jFor.getInductionVar();

    // for i in [0, MC)
    auto iFor = b.create<scf::ForOp>(loc, c0, MC, c1);
    OpBuilder::InsertionGuard g2(b);
    b.setInsertionPointToStart(iFor.getBody());
    Value i = iFor.getInductionVar();

    // globals
    Value iGlobal = b.create<arith::AddIOp>(loc, i_c, i);
    Value jGlobal = b.create<arith::AddIOp>(loc, p_c, j);

    // idx = i*KC + j
    Value iMulKC = b.create<arith::MulIOp>(loc, i, KC);
    Value idx = b.create<arith::AddIOp>(loc, iMulKC, j);

    // cond = (iGlobal < M) && (jGlobal < K)
    Value condI =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, iGlobal, M);
    Value condJ =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, jGlobal, K);
    Value cond = b.create<arith::AndIOp>(loc, condI, condJ);

    auto ifOp =
        b.create<scf::IfOp>(loc, TypeRange{}, cond, /*withElseRegion*/ true);
    {
      // Then region: insert before the already-existing yield.
      auto &thenBlock = ifOp.getThenRegion().front();
      OpBuilder tb(&thenBlock, thenBlock.getTerminator()->getIterator());
      Value v = tb.create<memref::LoadOp>(loc, A, ValueRange{iGlobal, jGlobal});
      tb.create<memref::StoreOp>(loc, v, A_packed, ValueRange{idx});
      // No explicit scf.yield here; it's already present.
    }
    {
      // Else region: insert before the already-existing yield.
      auto &elseBlock = ifOp.getElseRegion().front();
      OpBuilder eb(&elseBlock, elseBlock.getTerminator()->getIterator());
      Value f0 =
          eb.create<arith::ConstantOp>(loc, f32Type, eb.getF32FloatAttr(0.0f));
      eb.create<memref::StoreOp>(loc, f0, A_packed, ValueRange{idx});
    }
  }

  // Return after the outer loop.
  b.setInsertionPointAfter(jFor);
  b.create<func::ReturnOp>(loc);

  rewriter.setInsertionPointAfter(funcOp);

  // module->print(llvm::outs());
}

static void createPackB(ModuleOp module, PatternRewriter &rewriter,
                        int64_t NC_val, int64_t KC_val) {
  auto loc = rewriter.getUnknownLoc();

  auto indexType = rewriter.getIndexType();
  auto f32Type = rewriter.getF32Type();
  auto memref2DType =
      MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32Type);
  auto memref1DType = MemRefType::get({ShapedType::kDynamic}, f32Type);

  auto funcType = rewriter.getFunctionType(
      {memref2DType, memref1DType, indexType, indexType}, {});

  auto funcOp = rewriter.create<func::FuncOp>(loc, "pack_b", funcType);
  Block *entry = funcOp.addEntryBlock();
  OpBuilder b(entry, entry->begin());

  Value B = entry->getArgument(0);
  Value Bpacked = entry->getArgument(1);
  Value j_c = entry->getArgument(2);
  Value p_c = entry->getArgument(3);

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);

  // dims of B: K x N
  Value K = b.create<memref::DimOp>(loc, B, c0);
  Value N = b.create<memref::DimOp>(loc, B, c1);

  // Params
  Value KC = b.create<arith::ConstantIndexOp>(loc, KC_val);
  Value NC = b.create<arith::ConstantIndexOp>(loc, NC_val);

  // for i in [0, KC)
  auto iFor = b.create<scf::ForOp>(loc, c0, KC, c1);
  {
    OpBuilder::InsertionGuard g1(b);
    b.setInsertionPointToStart(iFor.getBody());
    Value i = iFor.getInductionVar();

    // for j in [0, NC)
    auto jFor = b.create<scf::ForOp>(loc, c0, NC, c1);
    OpBuilder::InsertionGuard g2(b);
    b.setInsertionPointToStart(jFor.getBody());
    Value j = jFor.getInductionVar();

    Value iGlobal = b.create<arith::AddIOp>(loc, p_c, i);
    Value jGlobal = b.create<arith::AddIOp>(loc, j_c, j);

    // idx = i + j*KC
    Value jMulKC = b.create<arith::MulIOp>(loc, j, KC);
    Value idx = b.create<arith::AddIOp>(loc, i, jMulKC);

    // cond = (iGlobal < K) && (jGlobal < N)
    Value condI =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, iGlobal, K);
    Value condJ =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, jGlobal, N);
    Value cond = b.create<arith::AndIOp>(loc, condI, condJ);

    auto ifOp =
        b.create<scf::IfOp>(loc, TypeRange{}, cond, /*withElseRegion*/ true);
    {
      auto &thenBlock = ifOp.getThenRegion().front();
      OpBuilder tb(&thenBlock, thenBlock.getTerminator()->getIterator());
      Value v = tb.create<memref::LoadOp>(loc, B, ValueRange{iGlobal, jGlobal});
      tb.create<memref::StoreOp>(loc, v, Bpacked, ValueRange{idx});
    }
    {
      auto &elseBlock = ifOp.getElseRegion().front();
      OpBuilder eb(&elseBlock, elseBlock.getTerminator()->getIterator());
      Value f0 =
          eb.create<arith::ConstantOp>(loc, f32Type, eb.getF32FloatAttr(0.0f));
      eb.create<memref::StoreOp>(loc, f0, Bpacked, ValueRange{idx});
    }
  }

  // Return after the outermost loop.
  b.setInsertionPointAfter(iFor);
  b.create<func::ReturnOp>(loc);

  rewriter.setInsertionPointAfter(funcOp);

  // module->print(llvm::outs());
}

//===----------------------------------------------------------------------===//
// micro_kernel: vectorized 4xNR FMA into c_sub
// Signature: (memref<?xf32> A_packed, memref<?xf32> B_packed,
//             memref<?xf32> c_sub, index a_offset, index b_offset) -> ()
// A layout: a_offset + i*KC + p
// B layout: b_offset + j*KC + p
// c_sub rows: each row length = NR, stored as 4 vectors
//===----------------------------------------------------------------------===//

static void createMicroKernel(ModuleOp module, PatternRewriter &rewriter,
                              int64_t MR, int64_t NR, int64_t KC_val) {
  auto loc = rewriter.getUnknownLoc();

  auto indexType = rewriter.getIndexType();
  auto f32Type = rewriter.getF32Type();
  auto memref1DType = MemRefType::get({ShapedType::kDynamic}, f32Type);

  auto funcType = rewriter.getFunctionType(
      {memref1DType, memref1DType, memref1DType, indexType, indexType}, {});

  auto funcOp = rewriter.create<func::FuncOp>(loc, "micro_kernel", funcType);
  Block *entry = funcOp.addEntryBlock();
  OpBuilder b(entry, entry->begin());

  Value Apacked = entry->getArgument(0);
  Value Bpacked = entry->getArgument(1);
  Value Csub = entry->getArgument(2);
  Value Aoff = entry->getArgument(3);
  Value Boff = entry->getArgument(4);

  // index constants 0..max(MR,NR)
  auto c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  auto c1 = b.create<arith::ConstantIndexOp>(loc, 1);

  SmallVector<Value, 8> laneIdx;
  for (int64_t t = 0; t < NR; ++t)
    laneIdx.push_back(b.create<arith::ConstantIndexOp>(loc, t));

  // Params: KC from getter; NR is compile-time for vector type

  Value KC = b.create<arith::ConstantIndexOp>(loc, KC_val);

  // vector type: <NR x f32>
  auto vType = VectorType::get({NR}, f32Type);

  // z = splat(0.0)
  Value f0 = b.create<arith::ConstantOp>(loc, f32Type, b.getF32FloatAttr(0.0f));
  Value z = b.create<vector::SplatOp>(loc, vType, f0);

  // scf.for p = 0..KC with iter_args (acc0..acc{MR-1})
  SmallVector<Value, 4> initAcc;
  for (int64_t i = 0; i < MR; ++i)
    initAcc.push_back(z);

  auto pFor = b.create<scf::ForOp>(loc, c0, KC, c1, initAcc);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(pFor.getBody());
    Value p = pFor.getInductionVar();
    auto accArgs = pFor.getRegionIterArgs(); // current accs

    // Build bvec by scalar loads + vector.insert for NR lanes
    Value bvec = b.create<vector::SplatOp>(loc, vType, f0); // init zeros
    for (int64_t j = 0; j < NR; ++j) {
      // b_idx = Boff + j*KC + p
      Value jVal = laneIdx[j];
      Value jMulKC = b.create<arith::MulIOp>(loc, jVal, KC);
      Value idx = b.create<arith::AddIOp>(loc, Boff, jMulKC);
      idx = b.create<arith::AddIOp>(loc, idx, p);
      Value bv = b.create<memref::LoadOp>(loc, Bpacked, idx);
      bvec = b.create<vector::InsertOp>(loc, bv, bvec, j);
    }

    // For i=0..MR-1: load a(i,p), broadcast, fma with bvec
    SmallVector<Value, 4> newAcc;
    for (int64_t i = 0; i < MR; ++i) {
      Value iIdx = b.create<arith::ConstantIndexOp>(loc, i);
      // a_idx = Aoff + i*KC + p
      Value iMulKC = b.create<arith::MulIOp>(loc, iIdx, KC);
      Value aidx = b.create<arith::AddIOp>(loc, Aoff, iMulKC);
      aidx = b.create<arith::AddIOp>(loc, aidx, p);
      Value aval = b.create<memref::LoadOp>(loc, Apacked, aidx);
      Value av = b.create<vector::BroadcastOp>(loc, vType, aval);
      Value acci = accArgs[i];
      Value accNew = b.create<vector::FMAOp>(loc, av, bvec, acci);
      newAcc.push_back(accNew);
    }

    b.create<scf::YieldOp>(loc, newAcc);
  }

  // Store final accumulators to c_sub rows: base = i*NR
  auto accResults = pFor.getResults();
  for (int64_t i = 0; i < MR; ++i) {
    Value base = b.create<arith::ConstantIndexOp>(loc, i * NR);
    b.create<vector::StoreOp>(loc, accResults[i], Csub, ValueRange{base});
  }

  b.create<func::ReturnOp>(loc);
  rewriter.setInsertionPointAfter(funcOp);
}

//===----------------------------------------------------------------------===//
// sgemm_blis_32: drives blocking, packing, micro-kernel, and C update
// Loops:
//   for j_c in [0,N) step NC
//     for p_c in [0,K) step KC
//       for i_c in [0,M) step MC
//         pack_a(A, A_packed, i_c, p_c)
//         pack_b(B, B_packed, j_c, p_c)
//         for j_r in [0,NC) step NR
//           for i_r in [0,MC) step MR
//             call micro_kernel(A_packed, B_packed, c_sub, i_r*KC, j_r*KC)
//             scatter-add c_sub into C with bounds checks
//===----------------------------------------------------------------------===//

static void createSgemmBlis32(ModuleOp module, PatternRewriter &rewriter,
                              ParamsType params) {
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
  OpBuilder b(entry, entry->begin());

  Value A = entry->getArgument(0);
  Value B = entry->getArgument(1);
  Value C = entry->getArgument(2);

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);

  Value M = b.create<memref::DimOp>(loc, A, c0);
  Value N = b.create<memref::DimOp>(loc, C, c1);
  Value K = b.create<memref::DimOp>(loc, A, c1);

  int64_t MR_val = params["MR"];
  int64_t NR_val = params["NR"];

  // Params
  Value NC = b.create<arith::ConstantIndexOp>(loc, params["NC"]);
  Value MC = b.create<arith::ConstantIndexOp>(loc, params["MC"]);
  Value KC = b.create<arith::ConstantIndexOp>(loc, params["KC"]);
  Value MR = b.create<arith::ConstantIndexOp>(loc, MR_val);
  Value NR = b.create<arith::ConstantIndexOp>(loc, NR_val);

  // Allocate packed buffers and c_sub
  Value aSize = b.create<arith::MulIOp>(loc, MC, KC); // MC*KC
  Value bSize = b.create<arith::MulIOp>(loc, KC, NC); // KC*NC
  Value csubSize =
      b.create<arith::ConstantIndexOp>(loc, MR_val * NR_val); // MR*NR

  Value Apacked =
      b.create<memref::AllocOp>(loc, memref1DType, ValueRange{aSize});
  Value Bpacked =
      b.create<memref::AllocOp>(loc, memref1DType, ValueRange{bSize});
  Value Csub =
      b.create<memref::AllocOp>(loc, memref1DType, ValueRange{csubSize});

  // for j_c in [0,N) step NC
  auto jcFor = b.create<scf::ForOp>(loc, c0, N, NC);
  {
    OpBuilder::InsertionGuard g1(b);
    b.setInsertionPointToStart(jcFor.getBody());
    Value j_c = jcFor.getInductionVar();

    // for p_c in [0,K) step KC
    auto pcFor = b.create<scf::ForOp>(loc, c0, K, KC);
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(pcFor.getBody());
      Value p_c = pcFor.getInductionVar();

      // for i_c in [0,M) step MC
      auto icFor = b.create<scf::ForOp>(loc, c0, M, MC);
      {
        OpBuilder::InsertionGuard g3(b);
        b.setInsertionPointToStart(icFor.getBody());
        Value i_c = icFor.getInductionVar();

        // pack
        b.create<func::CallOp>(loc, TypeRange{}, "pack_a",
                               ValueRange{A, Apacked, i_c, p_c});
        b.create<func::CallOp>(loc, TypeRange{}, "pack_b",
                               ValueRange{B, Bpacked, j_c, p_c});

        // Macro-tile loops: j_r, i_r
        auto jrFor = b.create<scf::ForOp>(loc, c0, NC, NR);
        {
          OpBuilder::InsertionGuard g4(b);
          b.setInsertionPointToStart(jrFor.getBody());
          Value j_r = jrFor.getInductionVar();

          auto irFor = b.create<scf::ForOp>(loc, c0, MC, MR);
          {
            OpBuilder::InsertionGuard g5(b);
            b.setInsertionPointToStart(irFor.getBody());
            Value i_r = irFor.getInductionVar();

            // a_offset = i_r * KC
            Value aOff = b.create<arith::MulIOp>(loc, i_r, KC);
            // b_offset = j_r * KC
            Value bOff = b.create<arith::MulIOp>(loc, j_r, KC);

            // micro-kernel fills Csub (MRxNR)
            b.create<func::CallOp>(
                loc, TypeRange{}, "micro_kernel",
                ValueRange{Apacked, Bpacked, Csub, aOff, bOff});

            // Scatter-add Csub into C with bounds checks
            // for i in [0,MRConst)
            auto iLoop = b.create<scf::ForOp>(loc, c0, MR, c1);
            {
              OpBuilder::InsertionGuard gi(b);
              b.setInsertionPointToStart(iLoop.getBody());
              Value i = iLoop.getInductionVar();

              auto jLoop = b.create<scf::ForOp>(loc, c0, NR, c1);
              OpBuilder::InsertionGuard gj(b);
              b.setInsertionPointToStart(jLoop.getBody());
              Value j = jLoop.getInductionVar();

              // c_sub_idx = i*NR + j
              Value iNR = b.create<arith::MulIOp>(loc, i, NR);
              Value subIdx = b.create<arith::AddIOp>(loc, iNR, j);
              Value subVal = b.create<memref::LoadOp>(loc, Csub, subIdx);

              // i_global = i_c + i_r + i
              Value iTmp = b.create<arith::AddIOp>(loc, i_r, i);
              Value iGlobal = b.create<arith::AddIOp>(loc, i_c, iTmp);
              // j_global = j_c + j_r + j
              Value jTmp = b.create<arith::AddIOp>(loc, j_r, j);
              Value jGlobal = b.create<arith::AddIOp>(loc, j_c, jTmp);

              // bounds check
              Value inI = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::ult, iGlobal, M);
              Value inJ = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::ult, jGlobal, N);
              Value inB = b.create<arith::AndIOp>(loc, inI, inJ);

              auto ifOp =
                  b.create<scf::IfOp>(loc, TypeRange{}, inB, /*withElse*/ true);
              {
                auto &thenBlk = ifOp.getThenRegion().front();
                OpBuilder tb(&thenBlk, thenBlk.getTerminator()->getIterator());
                Value cVal = tb.create<memref::LoadOp>(
                    loc, C, ValueRange{iGlobal, jGlobal});
                Value cNew = tb.create<arith::AddFOp>(loc, cVal, subVal);
                tb.create<memref::StoreOp>(loc, cNew, C,
                                           ValueRange{iGlobal, jGlobal});
              }
              {
                auto &elseBlk = ifOp.getElseRegion().front();
                OpBuilder eb(&elseBlk, elseBlk.getTerminator()->getIterator());
                // no-op
              }
            }
          }
        }
      }
    }
  }

  // Dealloc
  b.setInsertionPointAfter(jcFor);
  b.create<memref::DeallocOp>(loc, Csub);
  b.create<memref::DeallocOp>(loc, Apacked);
  b.create<memref::DeallocOp>(loc, Bpacked);

  b.create<func::ReturnOp>(loc);
  rewriter.setInsertionPointAfter(funcOp);
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

    // Parameter getters
    // HACK: max(NC, MC, KC) must be greater than min(MR, NR) to prevent
    // out-of-bounds access.
    ParamsType params = {
        {"NC", NC}, {"MC", MC}, {"KC", KC}, {"MR", MR}, {"NR", NR}};

    rewriter.setInsertionPointToStart(module.getBody());

    // packers
    createPackA(module, rewriter, MC, KC);
    createPackB(module, rewriter, NC, KC);
    // micro-kernel (vectorized 4xNR)
    createMicroKernel(module, rewriter, MR, NR, KC);
    // sgemm driver
    createSgemmBlis32(module, rewriter, params);

    // Replace linalg.matmul with a call to sgemm_blis_32(A, B, C)
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);

    rewriter.setInsertionPointAfter(op);
    rewriter.create<func::CallOp>(loc, TypeRange{}, "sgemm_blis_32",
                                  ValueRange{A, B, C});

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t NC, MC, KC, MR, NR;
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
  StringRef getDescription() const final {
    return "Performs BLIS-style matrix multiplication with hierarchical "
           "blocking.";
  }
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
