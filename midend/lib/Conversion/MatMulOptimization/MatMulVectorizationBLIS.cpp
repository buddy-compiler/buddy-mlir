//===- MatMulVectorizationBLIS.cpp ----------------------------------------===//
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
// This file implements the BLIS-style matmul vectorization.
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
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

class MatMulVectorizationBLISPattern : public ConversionPattern {
public:
  explicit MatMulVectorizationBLISPattern(MLIRContext *context)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // Create constant indices
    const Value c0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value c1 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    const Value c2 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(2));
    const Value c3 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(3));
    const Value c4 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(4));
    const Value c5 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(5));
    const Value c6 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(6));
    const Value c7 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(7));

    // Fixed BLIS blocking parameters from txt file
    const Value nc = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(256));  // nc = 256
    const Value kc = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(128));  // kc = 128
    const Value mc = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(64));   // mc = 64
    const Value mr = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(8));    // mr = 8
    const Value nr = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(32));   // nr = 32

    // Get input A, B, C
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);

    // Get dimensions
    const Value m = rewriter.create<memref::DimOp>(loc, A, c0);
    const Value n = rewriter.create<memref::DimOp>(loc, C, c1);
    const Value k = rewriter.create<memref::DimOp>(loc, A, c1);

    // Get element type and create vector type
    ShapedType ATy = cast<ShapedType>(A.getType());
    Type eleTy = ATy.getElementType();
    VectorType vectorTy = VectorType::get({32}, eleTy);  // Fixed vector size 32

    // BLIS 5-loop structure
    // Loop 1: jc - column blocking
        rewriter.create<scf::ParallelOp>(
    loc, c0, n, nc, 
    [&](OpBuilder &builder, Location loc, ValueRange ivs) {
      Value jc = ivs[0];
          // Compute actual nc for this block
          auto jcEnd = builder.create<arith::AddIOp>(loc, jc, nc);
          auto jcBound = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, jcEnd, n);
          auto jcActualEnd = builder.create<arith::SelectOp>(loc, jcBound, jcEnd, n);
          auto ncActual = builder.create<arith::SubIOp>(loc, jcActualEnd, jc);

          // Loop 2: pc - k blocking
          builder.create<scf::ForOp>(
              loc, c0, k, kc, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value pc, ValueRange) {
                // Compute actual kc for this block
                auto pcEnd = builder.create<arith::AddIOp>(loc, pc, kc);
                auto pcBound = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, pcEnd, k);
                auto pcActualEnd = builder.create<arith::SelectOp>(loc, pcBound, pcEnd, k);
                auto kcActual = builder.create<arith::SubIOp>(loc, pcActualEnd, pc);

                // Check if we should allocate B_packed (avoid allocation for empty blocks)
                auto kcActualPositive = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, kcActual, c0);
                auto ncActualPositive = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, ncActual, c0);
                auto shouldAllocB = builder.create<arith::AndIOp>(loc, kcActualPositive, ncActualPositive);

                builder.create<scf::IfOp>(loc, shouldAllocB,
                  [&](OpBuilder &builder, Location loc) {
                    // Allocate and pack B block
                    auto B_packedType = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, 
                                                eleTy,AffineMap(),  nullptr);
                    Value B_packed = builder.create<memref::AllocOp>(loc, B_packedType, 
                                                                   ValueRange{kcActual, ncActual});

                    // Pack B block
                    builder.create<scf::ForOp>(
                        loc, c0, kcActual, c1, ValueRange{},
                        [&](OpBuilder &builder, Location loc, Value kp, ValueRange) {
                          builder.create<scf::ForOp>(
                              loc, c0, ncActual, c1, ValueRange{},
                              [&](OpBuilder &builder, Location loc, Value j, ValueRange) {
                                auto bRowIdx = builder.create<arith::AddIOp>(loc, pc, kp);
                                auto bColIdx = builder.create<arith::AddIOp>(loc, jc, j);
                                auto bVal = builder.create<memref::LoadOp>(loc, B, 
                                                                         ValueRange{bRowIdx, bColIdx});
                                builder.create<memref::StoreOp>(loc, bVal, B_packed, 
                                                              ValueRange{kp, j});
                                builder.create<scf::YieldOp>(loc);
                              });
                          builder.create<scf::YieldOp>(loc);
                        });

                    // Loop 3: ic - row blocking
                    builder.create<scf::ParallelOp>(
                            loc, c0, m, mc,
                            [&](OpBuilder &builder, Location loc, ValueRange ivs) {
                            Value ic = ivs[0];
                          // Compute actual mc for this block
                          auto icEnd = builder.create<arith::AddIOp>(loc, ic, mc);
                          auto icBound = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, icEnd, m);
                          auto icActualEnd = builder.create<arith::SelectOp>(loc, icBound, icEnd, m);
                          auto mcActual = builder.create<arith::SubIOp>(loc, icActualEnd, ic);

                          // Check if we should allocate A_packed
                          auto mcActualPositive = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, mcActual, c0);
                          auto shouldAllocA = builder.create<arith::AndIOp>(loc, mcActualPositive, kcActualPositive);

                          builder.create<scf::IfOp>(loc, shouldAllocA,
                            [&](OpBuilder &builder, Location loc) {
                              // Allocate and pack A block
                              auto A_packedType = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, 
                                                eleTy,AffineMap(),  nullptr);
                              Value A_packed = builder.create<memref::AllocOp>(loc, A_packedType, 
                                                                             ValueRange{mcActual, kcActual});

                              // Pack A block
                              builder.create<scf::ForOp>(
                                  loc, c0, mcActual, c1, ValueRange{},
                                  [&](OpBuilder &builder, Location loc, Value i, ValueRange) {
                                    builder.create<scf::ForOp>(
                                        loc, c0, kcActual, c1, ValueRange{},
                                        [&](OpBuilder &builder, Location loc, Value kp, ValueRange) {
                                          auto aRowIdx = builder.create<arith::AddIOp>(loc, ic, i);
                                          auto aColIdx = builder.create<arith::AddIOp>(loc, pc, kp);
                                          auto aVal = builder.create<memref::LoadOp>(loc, A, 
                                                                                   ValueRange{aRowIdx, aColIdx});
                                          builder.create<memref::StoreOp>(loc, aVal, A_packed, 
                                                                        ValueRange{i, kp});
                                          builder.create<scf::YieldOp>(loc);
                                        });
                                    builder.create<scf::YieldOp>(loc);
                                  });

                              // Loop 4: jr - micro column blocking
                              builder.create<scf::ForOp>(
                                  loc, c0, ncActual, nr, ValueRange{},
                                  [&](OpBuilder &builder, Location loc, Value jr, ValueRange) {
                                    auto jrEnd = builder.create<arith::AddIOp>(loc, jr, nr);
                                    auto jrBound = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, jrEnd, ncActual);
                                    auto jrActualEnd = builder.create<arith::SelectOp>(loc, jrBound, jrEnd, ncActual);
                                    auto nrActual = builder.create<arith::SubIOp>(loc, jrActualEnd, jr);

                                    // Process micro blocks
                                    builder.create<scf::ForOp>(
                                        loc, c0, nrActual, nr, ValueRange{},
                                        [&](OpBuilder &builder, Location loc, Value nIdx, ValueRange) {
                                          auto nIdxEnd = builder.create<arith::AddIOp>(loc, nIdx, nr);
                                          auto nIdxBound = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, nIdxEnd, nrActual);
                                          auto nIdxActualEnd = builder.create<arith::SelectOp>(loc, nIdxBound, nIdxEnd, nrActual);
                                          auto colsToProcess = builder.create<arith::SubIOp>(loc, nIdxActualEnd, nIdx);

                                          // Check if we can vectorize (at least 32 columns)
                                          auto canVectorize = builder.create<arith::CmpIOp>(
                                              loc, arith::CmpIPredicate::sge, colsToProcess, nr);

                                          builder.create<scf::IfOp>(
                                              loc, canVectorize,
                                              [&](OpBuilder &builder, Location loc) {
                                                // Vectorized path 32列
                                                builder.create<scf::ForOp>(
                                                    loc, c0, mcActual, mr, ValueRange{},
                                                    [&](OpBuilder &builder, Location loc, Value ir, ValueRange) {
                                                      auto irEnd = builder.create<arith::AddIOp>(loc, ir, mr);
                                                      auto irBound = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, irEnd, mcActual);
                                                      auto irActualEnd = builder.create<arith::SelectOp>(loc, irBound, irEnd, mcActual);
                                                      auto mrActual = builder.create<arith::SubIOp>(loc, irActualEnd, ir);

                                                      // Check if we have full 8 rows - FIXED: use mr instead of c8
                                                      auto hasFullRows = builder.create<arith::CmpIOp>(
                                                          loc, arith::CmpIPredicate::sge, mrActual, mr);

                                                      builder.create<scf::IfOp>(
                                                          loc, hasFullRows,//8行
                                                          [&](OpBuilder &builder, Location loc) {
                                                            // Full rows vectorized processing
                                                            auto ir0 = builder.create<arith::AddIOp>(loc, ir, c0);
                                                            auto ir1 = builder.create<arith::AddIOp>(loc, ir, c1);
                                                            auto ir2 = builder.create<arith::AddIOp>(loc, ir, c2);
                                                            auto ir3 = builder.create<arith::AddIOp>(loc, ir, c3);
                                                            auto ir4 = builder.create<arith::AddIOp>(loc, ir, c4);
                                                            auto ir5 = builder.create<arith::AddIOp>(loc, ir, c5);
                                                            auto ir6 = builder.create<arith::AddIOp>(loc, ir, c6);
                                                            auto ir7 = builder.create<arith::AddIOp>(loc, ir, c7);

                                                            auto sumInit = builder.create<arith::ConstantOp>(
                                                                loc, vectorTy, builder.getZeroAttr(vectorTy));

                                                            auto sumIterVecs = builder.create<scf::ForOp>(
                                                                loc, c0, kcActual, c1,
                                                                ValueRange{sumInit, sumInit, sumInit, sumInit,
                                                                         sumInit, sumInit, sumInit, sumInit},
                                                                [&](OpBuilder &builder, Location loc, Value kInner, 
                                                                    ValueRange iterArgs) {
                                                                  // Load A values for 8 rows
                                                                  auto aVal0 = builder.create<memref::LoadOp>(
                                                                      loc, A_packed, ValueRange{ir0, kInner});
                                                                  auto aVal1 = builder.create<memref::LoadOp>(
                                                                      loc, A_packed, ValueRange{ir1, kInner});
                                                                  auto aVal2 = builder.create<memref::LoadOp>(
                                                                      loc, A_packed, ValueRange{ir2, kInner});
                                                                  auto aVal3 = builder.create<memref::LoadOp>(
                                                                      loc, A_packed, ValueRange{ir3, kInner});
                                                                  auto aVal4 = builder.create<memref::LoadOp>(
                                                                      loc, A_packed, ValueRange{ir4, kInner});
                                                                  auto aVal5 = builder.create<memref::LoadOp>(
                                                                      loc, A_packed, ValueRange{ir5, kInner});
                                                                  auto aVal6 = builder.create<memref::LoadOp>(
                                                                      loc, A_packed, ValueRange{ir6, kInner});
                                                                  auto aVal7 = builder.create<memref::LoadOp>(
                                                                      loc, A_packed, ValueRange{ir7, kInner});

                                                                  // Broadcast A values to vectors
                                                                  auto aVec0 = builder.create<vector::BroadcastOp>(
                                                                      loc, vectorTy, aVal0);
                                                                  auto aVec1 = builder.create<vector::BroadcastOp>(
                                                                      loc, vectorTy, aVal1);
                                                                  auto aVec2 = builder.create<vector::BroadcastOp>(
                                                                      loc, vectorTy, aVal2);
                                                                  auto aVec3 = builder.create<vector::BroadcastOp>(
                                                                      loc, vectorTy, aVal3);
                                                                  auto aVec4 = builder.create<vector::BroadcastOp>(
                                                                      loc, vectorTy, aVal4);
                                                                  auto aVec5 = builder.create<vector::BroadcastOp>(
                                                                      loc, vectorTy, aVal5);
                                                                  auto aVec6 = builder.create<vector::BroadcastOp>(
                                                                      loc, vectorTy, aVal6);
                                                                  auto aVec7 = builder.create<vector::BroadcastOp>(
                                                                      loc, vectorTy, aVal7);

                                                                  // Load B vector
                                                                  auto bColIdx = builder.create<arith::AddIOp>(loc, jr, nIdx);
                                                                  auto bVec = builder.create<vector::LoadOp>(
                                                                      loc, vectorTy, B_packed, 
                                                                      ValueRange{kInner, bColIdx});

                                                                  // Fused multiply-add
                                                                  auto resSumVec0 = builder.create<vector::FMAOp>(
                                                                      loc, aVec0, bVec, iterArgs[0]);
                                                                  auto resSumVec1 = builder.create<vector::FMAOp>(
                                                                      loc, aVec1, bVec, iterArgs[1]);
                                                                  auto resSumVec2 = builder.create<vector::FMAOp>(
                                                                      loc, aVec2, bVec, iterArgs[2]);
                                                                  auto resSumVec3 = builder.create<vector::FMAOp>(
                                                                      loc, aVec3, bVec, iterArgs[3]);
                                                                  auto resSumVec4 = builder.create<vector::FMAOp>(
                                                                      loc, aVec4, bVec, iterArgs[4]);
                                                                  auto resSumVec5 = builder.create<vector::FMAOp>(
                                                                      loc, aVec5, bVec, iterArgs[5]);
                                                                  auto resSumVec6 = builder.create<vector::FMAOp>(
                                                                      loc, aVec6, bVec, iterArgs[6]);
                                                                  auto resSumVec7 = builder.create<vector::FMAOp>(
                                                                      loc, aVec7, bVec, iterArgs[7]);

                                                                  builder.create<scf::YieldOp>(
                                                                      loc, ValueRange{resSumVec0, resSumVec1, resSumVec2,
                                                                                      resSumVec3, resSumVec4, resSumVec5,
                                                                                      resSumVec6, resSumVec7});
                                                                });

                                                            // Store results with accumulation
                                                            auto cRow0 = builder.create<arith::AddIOp>(loc, ic, ir0);
                                                            auto cRow1 = builder.create<arith::AddIOp>(loc, ic, ir1);
                                                            auto cRow2 = builder.create<arith::AddIOp>(loc, ic, ir2);
                                                            auto cRow3 = builder.create<arith::AddIOp>(loc, ic, ir3);
                                                            auto cRow4 = builder.create<arith::AddIOp>(loc, ic, ir4);
                                                            auto cRow5 = builder.create<arith::AddIOp>(loc, ic, ir5);
                                                            auto cRow6 = builder.create<arith::AddIOp>(loc, ic, ir6);
                                                            auto cRow7 = builder.create<arith::AddIOp>(loc, ic, ir7);
                                                            auto cColBase = builder.create<arith::AddIOp>(loc, jc, jr);
                                                            auto cColIdx = builder.create<arith::AddIOp>(loc, cColBase, nIdx);

                                                            // Load current C values - FIXED: add vectorTy parameter
                                                            auto cVec0 = builder.create<vector::LoadOp>(
                                                                loc, vectorTy, C, ValueRange{cRow0, cColIdx});
                                                            auto cVec1 = builder.create<vector::LoadOp>(
                                                                loc, vectorTy, C, ValueRange{cRow1, cColIdx});
                                                            auto cVec2 = builder.create<vector::LoadOp>(
                                                                loc, vectorTy, C, ValueRange{cRow2, cColIdx});
                                                            auto cVec3 = builder.create<vector::LoadOp>(
                                                                loc, vectorTy, C, ValueRange{cRow3, cColIdx});
                                                            auto cVec4 = builder.create<vector::LoadOp>(
                                                                loc, vectorTy, C, ValueRange{cRow4, cColIdx});
                                                            auto cVec5 = builder.create<vector::LoadOp>(
                                                                loc, vectorTy, C, ValueRange{cRow5, cColIdx});
                                                            auto cVec6 = builder.create<vector::LoadOp>(
                                                                loc, vectorTy, C, ValueRange{cRow6, cColIdx});
                                                            auto cVec7 = builder.create<vector::LoadOp>(
                                                                loc, vectorTy, C, ValueRange{cRow7, cColIdx});

                                                            // Accumulate: C = C + A*B
                                                            auto finalVec0 = builder.create<arith::AddFOp>(
                                                                loc, cVec0, sumIterVecs.getResult(0));
                                                            auto finalVec1 = builder.create<arith::AddFOp>(
                                                                loc, cVec1, sumIterVecs.getResult(1));
                                                            auto finalVec2 = builder.create<arith::AddFOp>(
                                                                loc, cVec2, sumIterVecs.getResult(2));
                                                            auto finalVec3 = builder.create<arith::AddFOp>(
                                                                loc, cVec3, sumIterVecs.getResult(3));
                                                            auto finalVec4 = builder.create<arith::AddFOp>(
                                                                loc, cVec4, sumIterVecs.getResult(4));
                                                            auto finalVec5 = builder.create<arith::AddFOp>(
                                                                loc, cVec5, sumIterVecs.getResult(5));
                                                            auto finalVec6 = builder.create<arith::AddFOp>(
                                                                loc, cVec6, sumIterVecs.getResult(6));
                                                            auto finalVec7 = builder.create<arith::AddFOp>(
                                                                loc, cVec7, sumIterVecs.getResult(7));

                                                            // Store back to C
                                                            builder.create<vector::StoreOp>(
                                                                loc, finalVec0, C, ValueRange{cRow0, cColIdx});
                                                            builder.create<vector::StoreOp>(
                                                                loc, finalVec1, C, ValueRange{cRow1, cColIdx});
                                                            builder.create<vector::StoreOp>(
                                                                loc, finalVec2, C, ValueRange{cRow2, cColIdx});
                                                            builder.create<vector::StoreOp>(
                                                                loc, finalVec3, C, ValueRange{cRow3, cColIdx});
                                                            builder.create<vector::StoreOp>(
                                                                loc, finalVec4, C, ValueRange{cRow4, cColIdx});
                                                            builder.create<vector::StoreOp>(
                                                                loc, finalVec5, C, ValueRange{cRow5, cColIdx});
                                                            builder.create<vector::StoreOp>(
                                                                loc, finalVec6, C, ValueRange{cRow6, cColIdx});
                                                            builder.create<vector::StoreOp>(
                                                                loc, finalVec7, C, ValueRange{cRow7, cColIdx});

                                                            builder.create<scf::YieldOp>(loc);
                                                          },
                                                          [&](OpBuilder &builder, Location loc) {
                                                            // Scalar path for incomplete rows
                                                            builder.create<scf::ForOp>(
                                                                loc, c0, nr, c1, ValueRange{},
                                                                [&](OpBuilder &builder, Location loc, Value jj, ValueRange) {
                                                                  builder.create<scf::ForOp>(
                                                                      loc, ir, irActualEnd, c1, ValueRange{},
                                                                      [&](OpBuilder &builder, Location loc, Value ii, ValueRange) {
                                                                        auto sumInit = builder.create<arith::ConstantOp>(
                                                                            loc, eleTy, builder.getFloatAttr(eleTy, 0.0));
                                                                        auto sumIter = builder.create<scf::ForOp>(
                                                                            loc, c0, kcActual, c1,
                                                                            ValueRange{sumInit},
                                                                            [&](OpBuilder &builder, Location loc, Value kInner, 
                                                                                ValueRange iterArgs) {
                                                                              auto aVal = builder.create<memref::LoadOp>(
                                                                                  loc, A_packed, ValueRange{ii, kInner});
                                                                              auto bColBase = builder.create<arith::AddIOp>(loc, jr, nIdx);
                                                                              auto bColIdx = builder.create<arith::AddIOp>(loc, bColBase, jj);
                                                                              auto bVal = builder.create<memref::LoadOp>(
                                                                                  loc, B_packed, ValueRange{kInner, bColIdx});
                                                                              auto prod = builder.create<arith::MulFOp>(loc, aVal, bVal);
                                                                              auto newSum = builder.create<arith::AddFOp>(loc, iterArgs[0], prod);
                                                                              builder.create<scf::YieldOp>(loc, ValueRange{newSum});
                                                                            });
                                                                        auto cRowIdx = builder.create<arith::AddIOp>(loc, ic, ii);
                                                                        auto cColBase = builder.create<arith::AddIOp>(loc, jc, jr);
                                                                        auto cColBase2 = builder.create<arith::AddIOp>(loc, cColBase, nIdx);
                                                                        auto cColIdx = builder.create<arith::AddIOp>(loc, cColBase2, jj);
                                                                        auto currentVal = builder.create<memref::LoadOp>(
                                                                            loc, C, ValueRange{cRowIdx, cColIdx});
                                                                        auto finalSum = builder.create<arith::AddFOp>(loc, currentVal, 
                                                                                                                    sumIter.getResult(0));
                                                                        builder.create<memref::StoreOp>(
                                                                            loc, finalSum, C, ValueRange{cRowIdx, cColIdx});
                                                                        builder.create<scf::YieldOp>(loc);
                                                                      });
                                                                  builder.create<scf::YieldOp>(loc);
                                                                });
                                                            builder.create<scf::YieldOp>(loc);
                                                          });

                                                      builder.create<scf::YieldOp>(loc);
                                                    });
                                                builder.create<scf::YieldOp>(loc);
                                              },
                                              [&](OpBuilder &builder, Location loc) {
                                                // Scalar path for incomplete columns
                                                builder.create<scf::ForOp>(
                                                    loc, nIdx, nIdxActualEnd, c1, ValueRange{},
                                                    [&](OpBuilder &builder, Location loc, Value nIdxTail, ValueRange) {
                                                      builder.create<scf::ForOp>(
                                                          loc, c0, mcActual, mr, ValueRange{},
                                                          [&](OpBuilder &builder, Location loc, Value ir, ValueRange) {
                                                            auto irEnd = builder.create<arith::AddIOp>(loc, ir, mr);
                                                            auto irBound = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, irEnd, mcActual);
                                                            auto irActualEnd = builder.create<arith::SelectOp>(loc, irBound, irEnd, mcActual);

                                                            builder.create<scf::ForOp>(
                                                                loc, ir, irActualEnd, c1, ValueRange{},
                                                                [&](OpBuilder &builder, Location loc, Value ii, ValueRange) {
                                                                  auto sumInit = builder.create<arith::ConstantOp>(
                                                                      loc, eleTy, builder.getFloatAttr(eleTy, 0.0));
                                                                  auto sumIter = builder.create<scf::ForOp>(
                                                                      loc, c0, kcActual, c1,
                                                                      ValueRange{sumInit},
                                                                      [&](OpBuilder &builder, Location loc, Value kInner, 
                                                                          ValueRange iterArgs) {
                                                                        auto aVal = builder.create<memref::LoadOp>(
                                                                            loc, A_packed, ValueRange{ii, kInner});
                                                                        auto bColBase = builder.create<arith::AddIOp>(loc, jr, nIdxTail);
                                                                        auto bVal = builder.create<memref::LoadOp>(
                                                                            loc, B_packed, ValueRange{kInner, bColBase});
                                                                        auto prod = builder.create<arith::MulFOp>(loc, aVal, bVal);
                                                                        auto newSum = builder.create<arith::AddFOp>(loc, iterArgs[0], prod);
                                                                        builder.create<scf::YieldOp>(loc, ValueRange{newSum});
                                                                      });
                                                                  auto cRowIdx = builder.create<arith::AddIOp>(loc, ic, ii);
                                                                  auto cColBase = builder.create<arith::AddIOp>(loc, jc, jr);
                                                                  auto cColIdx = builder.create<arith::AddIOp>(loc, cColBase, nIdxTail);
                                                                  auto currentVal = builder.create<memref::LoadOp>(
                                                                      loc, C, ValueRange{cRowIdx, cColIdx});
                                                                  auto finalSum = builder.create<arith::AddFOp>(loc, currentVal, 
                                                                                                              sumIter.getResult(0));
                                                                  builder.create<memref::StoreOp>(
                                                                      loc, finalSum, C, ValueRange{cRowIdx, cColIdx});
                                                                  builder.create<scf::YieldOp>(loc);
                                                                });
                                                            builder.create<scf::YieldOp>(loc);
                                                          });
                                                      builder.create<scf::YieldOp>(loc);
                                                    });
                                                builder.create<scf::YieldOp>(loc);
                                              });

                                          builder.create<scf::YieldOp>(loc);
                                        });
                                    builder.create<scf::YieldOp>(loc);
                                  });

                              // Deallocate A_packed
                              builder.create<memref::DeallocOp>(loc, A_packed);
                              builder.create<scf::YieldOp>(loc);
                            },
                            [&](OpBuilder &builder, Location loc) {
                              // Skip A_packed allocation for empty blocks
                              builder.create<scf::YieldOp>(loc);
                            });
                        });

                    // Deallocate B_packed
                    builder.create<memref::DeallocOp>(loc, B_packed);
                    builder.create<scf::YieldOp>(loc);
                  },
                  [&](OpBuilder &builder, Location loc) {
                    // Skip B_packed allocation for empty blocks
                    builder.create<scf::YieldOp>(loc);
                  });

                builder.create<scf::YieldOp>(loc);
              });
        });

    rewriter.eraseOp(op);
    return success();
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// MatMulVectorizationBLISPass
//===----------------------------------------------------------------------===//

namespace {
class MatMulVectorizationBLISPass
    : public PassWrapper<MatMulVectorizationBLISPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulVectorizationBLISPass)
  StringRef getArgument() const final { return "matmul-vectorization-blis"; }
  StringRef getDescription() const final { 
    return "BLIS-style MatMul Vectorization with fixed blocking parameters."; 
  }
  MatMulVectorizationBLISPass() = default;
  MatMulVectorizationBLISPass(const MatMulVectorizationBLISPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, VectorDialect, 
                    memref::MemRefDialect, arith::ArithDialect>();
  }
};
} // end anonymous namespace.

void MatMulVectorizationBLISPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                       scf::SCFDialect, memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<MatMulVectorizationBLISPattern>(context);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerMatMulVectorizationBLISPass() {
  PassRegistration<MatMulVectorizationBLISPass>();
}
} // namespace buddy
} // namespace mlir
