//===- MatMulOptimizeGPU.cpp -------------------------------------------------===//
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
// This file implements the matmul optimization.
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class MatMulOptimizeGPUPattern : public ConversionPattern {
public:
  explicit MatMulOptimizeGPUPattern(MLIRContext *context, SmallVector<int64_t, 3> tile_M_param,
                                    SmallVector<int64_t, 3> tile_N_param, int64_t tile_K_param)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), 1, context) {
//    vecSize = 16;
//    kernelM = 4;
//    kernelN = 2;
    tileM = tile_M_param;
    tileN = tile_N_param;
    tileK = tile_K_param;

  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // Get input A, B, C.
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);
    // Get shape of input and output
    ShapedType ATy = A.getType().cast<ShapedType>();
    auto eleTy = ATy.getElementType();
    MemRefType memRefTy = A.getType().cast<MemRefType>();
    // ShapedType BTy = B.getType().cast<ShapedType>();
    // ShapedType CTy = C.getType().cast<ShapedType>();

    // Some constants.
    const Value c0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value c1 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    const AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
    const AffineExpr s1 = rewriter.getAffineSymbolExpr(1);
    const AffineExpr s2 = rewriter.getAffineSymbolExpr(2);
    const AffineExpr s3 = rewriter.getAffineSymbolExpr(3);
    const AffineExpr d0 = rewriter.getAffineDimExpr(0);
    const AffineExpr d1 = rewriter.getAffineDimExpr(1);
    const AffineMap mapBroadcast =
        AffineMap::get(2, 0, rewriter.getAffineConstantExpr(0));
    const VectorType vTy = VectorType::get(16, ATy.getElementType());

    const Value BlockTileM = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(tileM[0]));
    const Value BlockTileN = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(tileN[0]));
    const Value BlockTileK = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(tileK));
    auto zeroAttr =
        rewriter.getZeroAttr(getElementTypeOrSelf(eleTy));
    Value zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
//    const Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getAttr<eleTy>(0));
    // Configs
//    int64_t kNLen = vecSize * kernelN;
    SmallVector<int64_t> smemA_shape = {tileM[0], tileK};
    SmallVector<int64_t> smemB_shape = {tileN[0], tileK};
    SmallVector<int64_t> compute_local_shape = {tileM[1], tileN[1]};
    SmallVector<int64_t> A_shared_local_shape = {tileM[1]};
    SmallVector<int64_t> B_shared_local_shape = {tileN[1]};

    gpu::AddressSpace workgroup_addressSpace = gpu::GPUDialect::getWorkgroupAddressSpace();
    gpu::AddressSpace private_addressSpace = gpu::GPUDialect::getPrivateAddressSpace();


    // Dims
    Value M = rewriter.create<memref::DimOp>(loc, A, 0);
    Value N = rewriter.create<memref::DimOp>(loc, B, 1);
    Value K = rewriter.create<memref::DimOp>(loc, A, 1);

    affine::buildAffineLoopNest(
        rewriter, loc, {c0, c0}, {M, N}, {tileM[0], tileN[0]},
        [&](OpBuilder &builder, Location loc,
            ValueRange ivRange) { // thread_idx, thread_idy
          auto smemA_type = MemRefType::get(
              smemA_shape, eleTy, MemRefLayoutAttrInterface{},
              gpu::AddressSpaceAttr::get(builder.getContext(),
                                         workgroup_addressSpace));
          auto smemB_type = MemRefType::get(
              smemB_shape, eleTy, MemRefLayoutAttrInterface{},
              gpu::AddressSpaceAttr::get(builder.getContext(),
                                         workgroup_addressSpace));
          auto smemA = builder.create<memref::AllocOp>(loc, smemA_type);
          auto smemB = builder.create<memref::AllocOp>(loc, smemB_type);
          auto compute_local_type = MemRefType::get(
              compute_local_shape, eleTy, MemRefLayoutAttrInterface{},
              gpu::AddressSpaceAttr::get(builder.getContext(),
                                         private_addressSpace));
          auto compute_local =
              builder.create<memref::AllocOp>(loc, compute_local_type);
          auto A_shared_local_type = MemRefType::get(
              A_shared_local_shape, eleTy, MemRefLayoutAttrInterface{},
              gpu::AddressSpaceAttr::get(builder.getContext(),
                                         private_addressSpace));
          auto B_shared_local_type = MemRefType::get(
              B_shared_local_shape, eleTy, MemRefLayoutAttrInterface{},
              gpu::AddressSpaceAttr::get(builder.getContext(),
                                         private_addressSpace));
          auto A_shared_local =
              builder.create<memref::AllocOp>(loc, A_shared_local_type);
          auto B_shared_local =
              builder.create<memref::AllocOp>(loc, B_shared_local_type);

          affine::buildAffineLoopNest(
              builder, loc, {c0, c0}, {BlockTileM, BlockTileN},
              {tileM[1], tileN[1]},
              [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
                auto threadTileM = builder.create<arith::ConstantOp>(
                    loc, rewriter.getIndexAttr(tileM[1]));
                auto threadTileN = builder.create<arith::ConstantOp>(
                    loc, rewriter.getIndexAttr(tileN[1]));
                affine::buildAffineLoopNest(
                    builder, loc, {c0, c0}, {threadTileM, threadTileN}, {1, 1},
                    [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
                      builder.create<memref::StoreOp>(loc, zero, compute_local,
                                                      ivRange);
                    });

                Value blockIdx =
                    builder.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
                Value blockIdy =
                    builder.create<gpu::BlockIdOp>(loc, gpu::Dimension::y);
                Value threadIdx =
                    builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
                Value threadIdy =
                    builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::y);
                Value blockDimx =
                    builder.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
                Value blockDimy =
                    builder.create<gpu::BlockIdOp>(loc, gpu::Dimension::y);
                affine::buildAffineLoopNest(
                    builder, loc, {c0}, {K}, tileK,
                    [&](OpBuilder builder, Location loc, ValueRange ivRange) {
                      builder.create<gpu::BarrierOp>(loc);
                      //                builder.create<>(loc, )
                      int64_t m_steps =
                          tileM[0] /
                          tileM[1]; // which should equal to blockDimx
                      int64_t n_steps =
                          tileN[0] /
                          tileN[1]; // which should equal to blockDimy

                      //                auto A_global_to_shared =
                      //                    builder.create<memref::SubViewOp>(loc,
                      //                    {})
                      Value shareAMemPerBlockOffsetX =
                          builder.create<affine::AffineApplyOp>(
                              loc,
                              AffineMap::get(1, 0, {d0 * tileM[0]},
                                             builder.getContext()),
                              SmallVector<Value>{blockIdx});
                      Value shareBMemPerBlockOffsetY =
                          builder.create<affine::AffineApplyOp>(
                              loc,
                              AffineMap::get(1, 0, {d0 * tileN[0]},
                                             builder.getContext()),
                              SmallVector<Value>{blockIdy});
                      MemRefType resTy = MemRefType::get(
                          ATy.getShape(), ATy.getElementType(),
                          AffineMap::get(2, 3, {d0 * s1 + d1 * s2 + s0}));
                      auto A_global_to_shared =
                          builder.create<memref::SubViewOp>(
                              loc, resTy, A,
                              SmallVector<OpFoldResult>{
                                  shareAMemPerBlockOffsetX, ivRange[0]},
                              SmallVector<OpFoldResult>{BlockTileM,
                                                        BlockTileK},
                              SmallVector<OpFoldResult>{c1, c1});

                      auto B_global_to_shared =
                          builder.create<memref::SubViewOp>(
                              loc, resTy, B,
                              SmallVector<OpFoldResult>{
                                  ivRange[0], shareBMemPerBlockOffsetY},
                              SmallVector<OpFoldResult> {
                                  BlockTileK, BlockTileN
                              },
                              SmallVector<OpFoldResult>{c1, c1}
                              );

                      affine::buildAffineLoopNest(
                          // BlockTileM = 128
                          // BlockTileN = 64
                          // BlockTileK = 32
                          builder, loc, {c0, c0}, {BlockTileM, BlockTileK},
                          {m_steps, n_steps},
                          [&](OpBuilder builder, Location loc,
                              ValueRange ivRange) {

                            Value thread_loc_x= builder.create<affine::AffineApplyOp>(loc, AffineMap::get(1, 1, {d0 * m_steps + s0}, builder.getContext()),
                                                                                     SmallVector<Value>{ivRange[0], threadIdx});

                            Value thread_loc_y= builder.create<affine::AffineApplyOp>(loc, AffineMap::get(1, 1, {d0 * n_steps + s0}, builder.getContext()),
                                                                                      SmallVector<Value>{ivRange[1], threadIdy});
                            auto global_value = builder.create<memref::LoadOp>(loc, A_global_to_shared, ValueRange{thread_loc_x, thread_loc_y});
                            auto store_value = builder.create<memref::StoreOp>(loc, global_value, smemA, ValueRange{thread_loc_x, thread_loc_y});
                          });
                      affine::buildAffineLoopNest(
                          builder, loc, {c0, c0}, {BlockTileK, BlockTileN},
                          {m_steps, n_steps},
                          [&](OpBuilder builder, Location loc,
                              ValueRange ivRange) {

                            Value thread_loc_x= builder.create<affine::AffineApplyOp>(loc, AffineMap::get(1, 1, {d0 * m_steps + s0}, builder.getContext()),
                                                                                       SmallVector<Value>{ivRange[0], threadIdx});
                            Value thread_loc_y= builder.create<affine::AffineApplyOp>(loc, AffineMap::get(1, 1, {d0 * n_steps + s0}, builder.getContext()),
                                                                                       SmallVector<Value>{ivRange[1], threadIdy});

                            auto global_value = builder.create<memref::LoadOp>(loc, B_global_to_shared, ValueRange{thread_loc_x, thread_loc_y});

                            auto store_value = builder.create<memref::StoreOp>(loc, global_value, smemB, ValueRange{thread_loc_y, thread_loc_x});
                          });
                      builder.create<gpu::BarrierOp>(loc);
                      auto c32 = builder.create<arith::ConstantIndexOp>(loc, 32);
                      auto thread_to_shm_idx_map_A = AffineMap::get(1, 1, {d0 * tileM[1] + s0}, builder.getContext());
                      auto thread_to_shm_idx_map_B = AffineMap::get(1, 1, {d0 * tileN[1] + s0}, builder.getContext());
                      affine::buildAffineLoopNest(
                          builder, loc, {c0}, {c32},1,
                          [&](OpBuilder builder, Location loc, ValueRange ivRange) {
                            for (int m_step = 0; m_step < m_steps; m_step++) {
                              auto index = builder.create<arith::ConstantIndexOp>(loc, m_step);
                              auto shm_A_index_y = ivRange[0];
                              auto shm_A_index_x = builder.create<affine::AffineApplyOp>(loc, thread_to_shm_idx_map_A, SmallVector<Value>{index, threadIdx});
                              auto value_in_shm_A = builder.create<memref::LoadOp>(loc, smemA, ValueRange{shm_A_index_x, shm_A_index_y});
                              builder.create<memref::StoreOp>(loc, value_in_shm_A, A_shared_local, ValueRange{index});
                            }
                            for (int n_step = 0; n_step < n_steps; n_step++) {
                              auto index = builder.create<arith::ConstantIndexOp>(loc, n_step);
                              auto shm_B_index_x = ivRange[0];
                              auto shm_B_index_y = builder.create<affine::AffineApplyOp>(loc, thread_to_shm_idx_map_B, SmallVector<Value>{index, threadIdx});
                              auto value_in_shm_B = builder.create<memref::LoadOp>(loc, smemA, ValueRange{shm_B_index_x, shm_B_index_y});
                              builder.create<memref::StoreOp>(loc, value_in_shm_B, B_shared_local, ValueRange{index});
                            }

                            for (int n_step = 0; n_step < n_steps; n_step++) {
                              for (int m_step = 0; m_step < m_steps; m_step++) {
                                auto index_m = builder.create<arith::ConstantIndexOp>(loc, m_step);
                                auto index_n = builder.create<arith::ConstantIndexOp>(loc, n_step);
//                                auto index_n_m = builder.create<arith::ConstantIndexOp>(loc, m_step * n_steps + n_)
                                auto a_local = builder.create<memref::LoadOp>(loc, A_shared_local, ValueRange{index_m});
                                auto b_local = builder.create<memref::LoadOp>(loc, B_shared_local, ValueRange{index_n});
                                auto mul_local = builder.create<arith::MulFOp>(loc, a_local, b_local);
                                auto com_local = builder.create<memref::LoadOp>(loc, compute_local, ValueRange{index_m, index_n});
                                auto res_local = builder.create<arith::AddFOp>(loc, com_local, mul_local);
                                builder.create<memref::StoreOp>(loc, res_local, compute_local, ValueRange{index_m, index_n});
                              }
                            }
                          }


                          );
                      // s0 = block_idx s1 = block_idy s2 = thread_idx s3 = thread_idy
                      // d0 = m_step d1 = n_step
                      auto global_mem_map = AffineMap::get(2, 4, {tileM[0] * s0 + d0 * tileM[1] + s2, tileN[0] * s1 + d1 * tileN[1] + s3}, builder.getContext());
                      auto global_mem_map_x = AffineMap::get(1, 2, {tileM[0] * s0 + d0 * tileM[1] + s1}, builder.getContext());
                      auto global_mem_map_y = AffineMap::get(1, 2, {tileN[0] * s0 + d0 * tileN[1] + s1}, builder.getContext());
                      for (int n_step = 0; n_step < n_steps; n_step++) {
                        for (int m_step = 0; m_step < m_steps; m_step++) {
                          auto index_m = builder.create<arith::ConstantIndexOp>(loc, m_step);
                          auto index_n = builder.create<arith::ConstantIndexOp>(loc, n_step);
                          auto com_local = builder.create<memref::LoadOp>(loc, compute_local, ValueRange{index_m, index_n});
                          auto glb_index_x = builder.create<affine::AffineApplyOp>(loc, global_mem_map_x, SmallVector<Value>{index_m, blockIdx, threadIdx});
                          auto glb_index_y = builder.create<affine::AffineApplyOp>(loc, global_mem_map_y, SmallVector<Value>{index_n, blockIdy, threadIdy});
                          builder.create<memref::StoreOp>(loc, com_local, C, ValueRange{glb_index_x, glb_index_y});
                        }
                      }

                    });
              });
          //          builder.create<arith::ConstantOp>()
        });

    rewriter.eraseOp(op);
    return success();
  }

private:
//  int64_t vecSize;
//  int64_t kernelM;
//  int64_t kernelN;
  SmallVector<int64_t, 3> tileM;
  SmallVector<int64_t, 3> tileN;
  int64_t tileK;

};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// MatMulOptimizeGPUPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg pooling operations to mixture of
/// Affine + Vector operations.
namespace {
class MatMulOptimizeGPUPass
    : public PassWrapper<MatMulOptimizeGPUPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulOptimizeGPUPass)
  StringRef getArgument() const final { return "matmul-optimize-gpu"; }
  StringRef getDescription() const final { return "MatMul Optimization on GPU."; }
  MatMulOptimizeGPUPass() = default;
  MatMulOptimizeGPUPass(const MatMulOptimizeGPUPass &) {}
  explicit MatMulOptimizeGPUPass(SmallVector<int64_t> tileMParam, SmallVector<int64_t> tileNParam,
                              int64_t tileKParam) {
    tileM = tileMParam;
    tileN = tileNParam;
    tileK = tileKParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, VectorDialect, gpu::GPUDialect>();
  }

  ListOption<int64_t> tileM{*this, "tile-m",
                          llvm::cl::desc("Tile size of axis M")};

  ListOption<int64_t> tileN{*this, "tile-n",
                            llvm::cl::desc("Tile size of axis N")};

  Option<int64_t> tileK{*this, "tile-k",
                          llvm::cl::desc("Tile size of axis K"),
                          llvm::cl::init(4)};
};
} // end anonymous namespace.

void MatMulOptimizeGPUPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                       scf::SCFDialect, memref::MemRefDialect, VectorDialect, gpu::GPUDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  SmallVector<int64_t, 3> tileMparam(tileM.begin(), tileM.end());
  SmallVector<int64_t, 3> tileNparam(tileN.begin(), tileN.end());
  patterns.add<MatMulOptimizeGPUPattern>(context, tileMparam, tileNparam, tileK);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerMatMulOptimizeGPUPass() { PassRegistration<MatMulOptimizeGPUPass>(); }
} // namespace buddy
} // namespace mlir
