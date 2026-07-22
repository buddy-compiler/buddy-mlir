//===- MatMulGPUTiling.cpp ------------------------------------------------===//
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
// Tile linalg.matmul for GPU execution. Produces a two-level parallel
// structure so that gpu-map-parallel-loops maps the outer level to GPU blocks
// and the inner level to GPU threads.
//
// Before:
//   linalg.matmul ins(%A, %B) outs(%C)  // M×K × K×N
//
// After:
//   scf.parallel (bm, bn) step (TM, TN) {        // → GPU blocks
//     linalg.matmul ins(subview %A, subview %B)
//                   outs(subview %C)               // TM×K × K×TN
//   }
//
// Then convert-linalg-to-parallel-loops expands the inner matmul:
//   scf.parallel (bm, bn) {                        // → GPU blocks
//     scf.parallel (tm, tn) {                      // → GPU threads
//       scf.for (k) { ... }                        // reduction
//     }
//   }
//
// Result: blocks=(M/TM, N/TN), threads=(TM, TN), each thread does K MACs.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

class MatMulGPUTilingPass
    : public PassWrapper<MatMulGPUTilingPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulGPUTilingPass)
  StringRef getArgument() const final { return "matmul-gpu-tiling"; }
  StringRef getDescription() const final {
    return "Tile linalg.matmul for GPU block/thread mapping.";
  }

  MatMulGPUTilingPass() = default;
  MatMulGPUTilingPass(const MatMulGPUTilingPass &) {}

  Option<int64_t> tileM{*this, "tile-m",
                        llvm::cl::desc("Block tile size for M dimension"),
                        llvm::cl::init(16)};
  Option<int64_t> tileN{*this, "tile-n",
                        llvm::cl::desc("Block tile size for N dimension"),
                        llvm::cl::init(16)};
  Option<int64_t> tileK{*this, "tile-k",
                        llvm::cl::desc("Shared memory tile size for K (0=off)"),
                        llvm::cl::init(16)};

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, memref::MemRefDialect, arith::ArithDialect,
                    gpu::GPUDialect>();
  }
};

void MatMulGPUTilingPass::runOnOperation() {
  auto funcOp = getOperation();
  SmallVector<linalg::MatmulOp> matmuls;
  funcOp->walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });

  for (auto matmulOp : matmuls) {
    // Skip ops with non-default indexing maps (e.g. transpose-B matmuls
    // emitted as linalg.matmul with custom indexing_maps). This pass assumes
    // the standard C[m,n] += A[m,k] * B[k,n] layout; tiling a transposed op as
    // if it were standard would slice the wrong operand dimensions and produce
    // incorrect results. Leave such ops untiled (correct, just not optimized).
    if (matmulOp.hasUserDefinedMaps())
      continue;
    // Only handle buffer semantics (memref, not tensor).
    auto outType = dyn_cast<MemRefType>(matmulOp.getDpsInits()[0].getType());
    if (!outType)
      continue;
    auto inAType = dyn_cast<MemRefType>(matmulOp.getDpsInputs()[0].getType());
    auto inBType = dyn_cast<MemRefType>(matmulOp.getDpsInputs()[1].getType());
    if (!inAType || !inBType)
      continue;

    // Get static dimensions M, K, N.
    auto shapeA = inAType.getShape(); // [M, K]
    auto shapeB = inBType.getShape(); // [K, N]
    if (shapeA.size() != 2 || shapeB.size() != 2)
      continue;
    int64_t M = shapeA[0], K = shapeA[1], N = shapeB[1];
    if (ShapedType::isDynamic(M) || ShapedType::isDynamic(K) ||
        ShapedType::isDynamic(N))
      continue;

    int64_t tm = tileM;
    int64_t tn = tileN;
    int64_t tk = tileK;
    if (tm > M)
      tm = M;
    if (tn > N)
      tn = N;
    if (M % tm != 0 || N % tn != 0)
      continue;

    // GEMV path: M==1 with K-parallel reduction in shared memory.
    // Each block computes one output element C[0,n]. Threads split K.
    if (M == 1 && tk > 0) {
      int64_t numThreads = 256;
      if (numThreads > K)
        numThreads = K;
      // Round down to power of 2 for tree reduction.
      int64_t nt = 1;
      while (nt * 2 <= numThreads)
        nt *= 2;
      numThreads = nt;

      OpBuilder builder(matmulOp);
      auto loc = matmulOp.getLoc();
      Value A = matmulOp.getDpsInputs()[0];
      Value B = matmulOp.getDpsInputs()[1];
      Value C = matmulOp.getDpsInits()[0];
      auto elemType = outType.getElementType();

      Value gridX = arith::ConstantIndexOp::create(builder, loc, N);
      Value gridY = arith::ConstantIndexOp::create(builder, loc, 1);
      Value gridZ = arith::ConstantIndexOp::create(builder, loc, 1);
      Value blockX = arith::ConstantIndexOp::create(builder, loc, numThreads);
      Value blockY = arith::ConstantIndexOp::create(builder, loc, 1);
      Value blockZ = arith::ConstantIndexOp::create(builder, loc, 1);

      auto launchOp = gpu::LaunchOp::create(builder, loc, gridX, gridY, gridZ,
                                            blockX, blockY, blockZ);
      builder.setInsertionPointToStart(&launchOp.getBody().front());

      Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
      Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);
      Value cK = arith::ConstantIndexOp::create(builder, loc, K);
      Value cNT = arith::ConstantIndexOp::create(builder, loc, numThreads);

      Value bid = launchOp.getBlockIds().x;  // output column index
      Value tid = launchOp.getThreadIds().x; // thread within block

      // Shared memory for reduction.
      auto smemType = MemRefType::get(
          {numThreads}, elemType, AffineMap(),
          gpu::AddressSpaceAttr::get(builder.getContext(),
                                     gpu::AddressSpace::Workgroup));
      Value smem = memref::AllocOp::create(builder, loc, smemType);

      // Each thread: partial sum over K with stride numThreads.
      Value zero = arith::ConstantOp::create(
          builder, loc, builder.getFloatAttr(elemType, 0.0));
      auto sumLoop =
          scf::ForOp::create(builder, loc, tid, cK, cNT, ValueRange{zero});
      {
        builder.setInsertionPointToStart(sumLoop.getBody());
        Value k = sumLoop.getInductionVar();
        Value acc = sumLoop.getRegionIterArg(0);
        Value a = memref::LoadOp::create(builder, loc, A, ValueRange{c0, k});
        Value b = memref::LoadOp::create(builder, loc, B, ValueRange{k, bid});
        Value prod = arith::MulFOp::create(builder, loc, a, b);
        Value newAcc = arith::AddFOp::create(builder, loc, acc, prod);
        scf::YieldOp::create(builder, loc, ValueRange{newAcc});
      }
      builder.setInsertionPointAfter(sumLoop);

      // Store partial sum to shared memory.
      memref::StoreOp::create(builder, loc, sumLoop.getResult(0), smem,
                              ValueRange{tid});
      gpu::BarrierOp::create(builder, loc);

      // Tree reduction in shared memory.
      int64_t stride = numThreads / 2;
      while (stride > 0) {
        Value cStride = arith::ConstantIndexOp::create(builder, loc, stride);
        Value cond = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::ult, tid, cStride);
        auto ifOp = scf::IfOp::create(builder, loc, cond, /*else=*/false);
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        Value partner = arith::AddIOp::create(builder, loc, tid, cStride);
        Value myVal =
            memref::LoadOp::create(builder, loc, smem, ValueRange{tid});
        Value otherVal =
            memref::LoadOp::create(builder, loc, smem, ValueRange{partner});
        Value reduced = arith::AddFOp::create(builder, loc, myVal, otherVal);
        memref::StoreOp::create(builder, loc, reduced, smem, ValueRange{tid});
        builder.setInsertionPointAfter(ifOp);
        gpu::BarrierOp::create(builder, loc);
        stride /= 2;
      }

      // Thread 0 writes result.
      Value isTid0 = arith::CmpIOp::create(builder, loc,
                                           arith::CmpIPredicate::eq, tid, c0);
      auto writeIf = scf::IfOp::create(builder, loc, isTid0, /*else=*/false);
      builder.setInsertionPointToStart(&writeIf.getThenRegion().front());
      Value result = memref::LoadOp::create(builder, loc, smem, ValueRange{c0});
      Value cOld = memref::LoadOp::create(builder, loc, C, ValueRange{c0, bid});
      Value cNew = arith::AddFOp::create(builder, loc, cOld, result);
      memref::StoreOp::create(builder, loc, cNew, C, ValueRange{c0, bid});

      builder.setInsertionPointAfter(writeIf);
      gpu::TerminatorOp::create(builder, loc);
      matmulOp->erase();
      continue;
    }

    // When tile-k > 0: generate gpu.launch with shared memory tiling.
    // When tile-k == 0: fall back to scf.parallel (no shared memory).
    if (tk > 0 && K % tk == 0 && tm >= tk && tn >= tk) {
      OpBuilder builder(matmulOp);
      auto loc = matmulOp.getLoc();
      Value A = matmulOp.getDpsInputs()[0];
      Value B = matmulOp.getDpsInputs()[1];
      Value C = matmulOp.getDpsInits()[0];
      auto elemType = outType.getElementType();

      // Grid/block sizes must be defined OUTSIDE the launch.
      Value gridX = arith::ConstantIndexOp::create(builder, loc, M / tm);
      Value gridY = arith::ConstantIndexOp::create(builder, loc, N / tn);
      Value gridZ = arith::ConstantIndexOp::create(builder, loc, 1);
      Value blockX = arith::ConstantIndexOp::create(builder, loc, tm);
      Value blockY = arith::ConstantIndexOp::create(builder, loc, tn);
      Value blockZ = arith::ConstantIndexOp::create(builder, loc, 1);

      auto launchOp = gpu::LaunchOp::create(builder, loc, gridX, gridY, gridZ,
                                            blockX, blockY, blockZ);
      builder.setInsertionPointToStart(&launchOp.getBody().front());

      // ALL constants inside the launch body so they don't become kernel args.
      Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
      Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);
      Value cTM = arith::ConstantIndexOp::create(builder, loc, tm);
      Value cTN = arith::ConstantIndexOp::create(builder, loc, tn);
      Value cKval = arith::ConstantIndexOp::create(builder, loc, K);
      Value cTKval = arith::ConstantIndexOp::create(builder, loc, tk);

      Value bx = launchOp.getBlockIds().x;
      Value by = launchOp.getBlockIds().y;
      Value tx = launchOp.getThreadIds().x;
      Value ty = launchOp.getThreadIds().y;

      Value globalM = arith::AddIOp::create(
          builder, loc, arith::MulIOp::create(builder, loc, bx, cTM), tx);
      Value globalN = arith::AddIOp::create(
          builder, loc, arith::MulIOp::create(builder, loc, by, cTN), ty);

      // Shared memory (workgroup address space).
      auto smemAType = MemRefType::get(
          {tm, tk}, elemType, AffineMap(),
          gpu::AddressSpaceAttr::get(builder.getContext(),
                                     gpu::AddressSpace::Workgroup));
      auto smemBType = MemRefType::get(
          {tk, tn}, elemType, AffineMap(),
          gpu::AddressSpaceAttr::get(builder.getContext(),
                                     gpu::AddressSpace::Workgroup));
      Value smemA = memref::AllocOp::create(builder, loc, smemAType);
      Value smemB = memref::AllocOp::create(builder, loc, smemBType);

      Value zero = arith::ConstantOp::create(
          builder, loc, builder.getFloatAttr(elemType, 0.0));

      // K-tile loop.
      auto kLoop =
          scf::ForOp::create(builder, loc, c0, cKval, cTKval, ValueRange{zero});
      builder.setInsertionPointToStart(kLoop.getBody());
      Value kt = kLoop.getInductionVar();
      Value acc = kLoop.getRegionIterArg(0);

      // Cooperative load: each thread loads one element.
      Value aCol = arith::AddIOp::create(builder, loc, kt, ty);
      Value aVal =
          memref::LoadOp::create(builder, loc, A, ValueRange{globalM, aCol});
      memref::StoreOp::create(builder, loc, aVal, smemA, ValueRange{tx, ty});

      Value bRow = arith::AddIOp::create(builder, loc, kt, tx);
      Value bVal =
          memref::LoadOp::create(builder, loc, B, ValueRange{bRow, globalN});
      memref::StoreOp::create(builder, loc, bVal, smemB, ValueRange{tx, ty});

      gpu::BarrierOp::create(builder, loc);

      // Inner accumulation from shared memory.
      auto kkLoop =
          scf::ForOp::create(builder, loc, c0, cTKval, c1, ValueRange{acc});
      builder.setInsertionPointToStart(kkLoop.getBody());
      Value kk = kkLoop.getInductionVar();
      Value accIn = kkLoop.getRegionIterArg(0);
      Value sa =
          memref::LoadOp::create(builder, loc, smemA, ValueRange{tx, kk});
      Value sb =
          memref::LoadOp::create(builder, loc, smemB, ValueRange{kk, ty});
      Value prod = arith::MulFOp::create(builder, loc, sa, sb);
      Value sum = arith::AddFOp::create(builder, loc, accIn, prod);
      scf::YieldOp::create(builder, loc, ValueRange{sum});

      builder.setInsertionPointAfter(kkLoop);
      gpu::BarrierOp::create(builder, loc);
      scf::YieldOp::create(builder, loc, ValueRange{kkLoop.getResult(0)});

      // Store result.
      builder.setInsertionPointAfter(kLoop);
      Value finalAcc = kLoop.getResult(0);
      Value cOld =
          memref::LoadOp::create(builder, loc, C, ValueRange{globalM, globalN});
      Value cNew = arith::AddFOp::create(builder, loc, cOld, finalAcc);
      memref::StoreOp::create(builder, loc, cNew, C,
                              ValueRange{globalM, globalN});

      gpu::TerminatorOp::create(builder, loc);
      matmulOp->erase();
      continue;
    }

    // Fallback: scf.parallel tiling (no shared memory).
    OpBuilder builder(matmulOp);
    auto loc = matmulOp.getLoc();

    Value A = matmulOp.getDpsInputs()[0];
    Value B = matmulOp.getDpsInputs()[1];
    Value C = matmulOp.getDpsInits()[0];

    auto elemType = outType.getElementType();

    Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
    Value cM = arith::ConstantIndexOp::create(builder, loc, M);
    Value cN = arith::ConstantIndexOp::create(builder, loc, N);
    Value cTM = arith::ConstantIndexOp::create(builder, loc, tm);
    Value cTN = arith::ConstantIndexOp::create(builder, loc, tn);

    auto ploop =
        scf::ParallelOp::create(builder, loc, ValueRange{c0, c0},
                                ValueRange{cM, cN}, ValueRange{cTM, cTN});

    builder.setInsertionPointToStart(ploop.getBody());
    Value bm = ploop.getInductionVars()[0];
    Value bn = ploop.getInductionVars()[1];

    auto subA = memref::SubViewOp::create(
        builder, loc, A, SmallVector<OpFoldResult>{bm, c0},
        SmallVector<OpFoldResult>{builder.getIndexAttr(tm),
                                  builder.getIndexAttr(K)},
        SmallVector<OpFoldResult>{builder.getIndexAttr(1),
                                  builder.getIndexAttr(1)});

    auto subB = memref::SubViewOp::create(
        builder, loc, B, SmallVector<OpFoldResult>{c0, bn},
        SmallVector<OpFoldResult>{builder.getIndexAttr(K),
                                  builder.getIndexAttr(tn)},
        SmallVector<OpFoldResult>{builder.getIndexAttr(1),
                                  builder.getIndexAttr(1)});

    auto subC = memref::SubViewOp::create(
        builder, loc, C, SmallVector<OpFoldResult>{bm, bn},
        SmallVector<OpFoldResult>{builder.getIndexAttr(tm),
                                  builder.getIndexAttr(tn)},
        SmallVector<OpFoldResult>{builder.getIndexAttr(1),
                                  builder.getIndexAttr(1)});

    linalg::MatmulOp::create(builder, loc,
                             ValueRange{subA.getResult(), subB.getResult()},
                             ValueRange{subC.getResult()});

    matmulOp->erase();
  }

  // --- Tile linalg.batch_matmul: [B,M,K] x [B,K,N] -> [B,M,N] ---
  // Outer parallel over B (kept as-is), tile M and N for block/thread mapping.
  SmallVector<linalg::BatchMatmulOp> batchMatmuls;
  funcOp->walk([&](linalg::BatchMatmulOp op) { batchMatmuls.push_back(op); });

  for (auto bmOp : batchMatmuls) {
    // Skip transpose-B (and other non-default) batch matmuls: the attention
    // Detect the operand layout from the indexing maps. The standard op is
    // ins(A:[B,M,K], B:[B,K,N]); the attention QK^T is emitted as a
    // transpose-B op with custom maps rhs=(b,n,k), i.e. B laid out [B,N,K].
    // Both are tiled here; any other custom-map op is left untiled (correct,
    // just unoptimized).
    bool isTransB = false;
    if (bmOp.hasUserDefinedMaps()) {
      MLIRContext *ctx = bmOp.getContext();
      AffineExpr d0, d1, d2, d3;
      bindDims(ctx, d0, d1, d2, d3);                      // (b, m, n, k)
      auto mA = AffineMap::get(4, 0, {d0, d1, d3}, ctx);  // A: (b, m, k)
      auto mBt = AffineMap::get(4, 0, {d0, d2, d3}, ctx); // B: (b, n, k)
      auto mC = AffineMap::get(4, 0, {d0, d1, d2}, ctx);  // C: (b, m, n)
      auto maps = bmOp.getIndexingMapsArray();
      if (maps.size() == 3 && maps[0] == mA && maps[1] == mBt && maps[2] == mC)
        isTransB = true;
      else
        continue;
    }

    auto outType = dyn_cast<MemRefType>(bmOp.getDpsInits()[0].getType());
    if (!outType)
      continue;
    auto inAType = dyn_cast<MemRefType>(bmOp.getDpsInputs()[0].getType());
    auto inBType = dyn_cast<MemRefType>(bmOp.getDpsInputs()[1].getType());
    if (!inAType || !inBType)
      continue;

    auto shapeA = inAType.getShape(); // [B, M, K]
    // Standard B is [B, K, N]; transpose-B is [B, N, K]. Read N from the
    // output ([B, M, N]) so it is correct for both layouts.
    auto shapeC = outType.getShape();
    if (shapeA.size() != 3 || shapeC.size() != 3)
      continue;
    int64_t B = shapeA[0], M = shapeA[1], K = shapeA[2], N = shapeC[2];
    if (ShapedType::isDynamic(B) || ShapedType::isDynamic(M) ||
        ShapedType::isDynamic(K) || ShapedType::isDynamic(N))
      continue;

    int64_t tm = tileM, tn = tileN;
    if (tm > M)
      tm = M;
    if (tn > N)
      tn = N;
    if (M % tm != 0 || N % tn != 0)
      continue;

    OpBuilder builder(bmOp);
    auto loc = bmOp.getLoc();

    Value A = bmOp.getDpsInputs()[0];
    Value Bv = bmOp.getDpsInputs()[1];
    Value C = bmOp.getDpsInits()[0];

    Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
    Value cB = arith::ConstantIndexOp::create(builder, loc, B);
    Value cM = arith::ConstantIndexOp::create(builder, loc, M);
    Value cN = arith::ConstantIndexOp::create(builder, loc, N);
    Value c1step = arith::ConstantIndexOp::create(builder, loc, 1);
    Value cTM = arith::ConstantIndexOp::create(builder, loc, tm);
    Value cTN = arith::ConstantIndexOp::create(builder, loc, tn);

    // scf.parallel (b, bm, bn) = (0,0,0) to (B,M,N) step (1,TM,TN)
    auto ploop = scf::ParallelOp::create(builder, loc, ValueRange{c0, c0, c0},
                                         ValueRange{cB, cM, cN},
                                         ValueRange{c1step, cTM, cTN});

    builder.setInsertionPointToStart(ploop.getBody());
    Value bidx = ploop.getInductionVars()[0];
    Value bm = ploop.getInductionVars()[1];
    Value bn = ploop.getInductionVars()[2];

    auto i1 = builder.getIndexAttr(1);
    auto iK = builder.getIndexAttr(K);
    auto iTM = builder.getIndexAttr(tm);
    auto iTN = builder.getIndexAttr(tn);

    // subview A[b, bm:bm+TM, 0:K]
    auto subA = memref::SubViewOp::create(
        builder, loc, A, SmallVector<OpFoldResult>{bidx, bm, c0},
        SmallVector<OpFoldResult>{i1, iTM, iK},
        SmallVector<OpFoldResult>{i1, i1, i1});
    // subview B: standard [b, 0:K, bn:bn+TN] vs transpose-B [b, bn:bn+TN, 0:K]
    memref::SubViewOp subB =
        isTransB
            ? memref::SubViewOp::create(builder, loc, Bv,
                                        SmallVector<OpFoldResult>{bidx, bn, c0},
                                        SmallVector<OpFoldResult>{i1, iTN, iK},
                                        SmallVector<OpFoldResult>{i1, i1, i1})
            : memref::SubViewOp::create(builder, loc, Bv,
                                        SmallVector<OpFoldResult>{bidx, c0, bn},
                                        SmallVector<OpFoldResult>{i1, iK, iTN},
                                        SmallVector<OpFoldResult>{i1, i1, i1});
    // subview C[b, bm:bm+TM, bn:bn+TN]
    auto subC = memref::SubViewOp::create(
        builder, loc, C, SmallVector<OpFoldResult>{bidx, bm, bn},
        SmallVector<OpFoldResult>{i1, iTM, iTN},
        SmallVector<OpFoldResult>{i1, i1, i1});

    auto newBmm = linalg::BatchMatmulOp::create(
        builder, loc, ValueRange{subA.getResult(), subB.getResult()},
        ValueRange{subC.getResult()});
    // Preserve the transpose-B semantics on the tiled op so that
    // convert-linalg-to-parallel-loops lowers B as B[b, n, k].
    if (isTransB) {
      MLIRContext *ctx = bmOp.getContext();
      AffineExpr d0, d1, d2, d3;
      bindDims(ctx, d0, d1, d2, d3);
      SmallVector<Attribute> mapAttrs = {
          AffineMapAttr::get(AffineMap::get(4, 0, {d0, d1, d3}, ctx)),
          AffineMapAttr::get(AffineMap::get(4, 0, {d0, d2, d3}, ctx)),
          AffineMapAttr::get(AffineMap::get(4, 0, {d0, d1, d2}, ctx))};
      newBmm->setAttr("indexing_maps", builder.getArrayAttr(mapAttrs));
    }

    bmOp->erase();
  }
}

} // namespace

namespace mlir {
namespace buddy {
void registerMatMulGPUTilingPass() { PassRegistration<MatMulGPUTilingPass>(); }
} // namespace buddy
} // namespace mlir
