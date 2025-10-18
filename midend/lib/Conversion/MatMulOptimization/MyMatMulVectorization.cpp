//===- MyMatMulVectorization.cpp - My MatMul Vectorization Pass ----------===//
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
// This file implements a MLIR pass that converts linalg.matmul operations
// to BLIS-style hand-written implementations with vectorization and 
// multi-level blocking, based on the analysis of high-performance matrix 
// multiplication techniques from the paper
// "Anatomy of High-Performance Many-Threaded Matrix Multiplication".
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

/// Pattern to convert linalg.matmul operations to BLIS-style hand-written
/// implementations with multi-level blocking and vectorization.
class MyMatMulVectorizationPattern : public ConversionPattern {
public:
  explicit MyMatMulVectorizationPattern(MLIRContext *context,
                                       int64_t vecSizeParam)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), 1, context) {
    vecSize = vecSizeParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    
    // Get input matrices A, B, and output matrix C
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);
    
    // Get matrix dimensions
    const Value c0 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(0));
    const Value c1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(1));
    
    const Value m = rewriter.create<memref::DimOp>(loc, A, c0);
    const Value n = rewriter.create<memref::DimOp>(loc, C, c1);
    const Value k = rewriter.create<memref::DimOp>(loc, A, c1);
    
    // BLIS-style cache-aware blocking parameters
    const Value nc = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(4096)); // L3 cache blocking
    const Value kc = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(256));  // L2 cache blocking
    const Value mc = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(256));  // L2 cache blocking
    const Value nr = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(32));   // Register blocking
    const Value mr = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(8));    // Register blocking
    
    // BLIS Loop Nesting Structure: jc -> pc -> ic [micro-kernel: jr × ir]
    // Level 1: jc loop - Block columns of C and B (L3 cache blocking)
    auto jcLoop = rewriter.create<scf::ForOp>(
        loc, c0, n, nc, ValueRange{c0},
        [&](OpBuilder &builder, Location loc, Value jc, ValueRange /*iterArgs*/) {
          
          // Calculate remaining columns
          Value n_remaining = builder.create<arith::SubIOp>(loc, n, jc);
          Value nc_actual = builder.create<arith::MinSIOp>(loc, nc, n_remaining);
          
          // Level 2: pc loop - Block panels of A and B (L2 cache blocking)
          auto pcLoop = rewriter.create<scf::ForOp>(
              loc, c0, k, kc, ValueRange{c0},
              [&](OpBuilder &builder, Location loc, Value pc, ValueRange /*iterArgs*/) {
                
                // Calculate remaining k dimension
                Value k_remaining = builder.create<arith::SubIOp>(loc, k, pc);
                Value kc_actual = builder.create<arith::MinSIOp>(loc, kc, k_remaining);
                
                // Level 3: ic loop - Block rows of A and C (L2 cache blocking)
                auto icLoop = rewriter.create<scf::ForOp>(
                    loc, c0, m, mc, ValueRange{c0},
                    [&](OpBuilder &builder, Location loc, Value ic, ValueRange /*iterArgs*/) {
                      
                      // Calculate remaining rows
                      Value m_remaining = builder.create<arith::SubIOp>(loc, m, ic);
                      Value mc_actual = builder.create<arith::MinSIOp>(loc, mc, m_remaining);
                      
                      // BLIS Data Packing Strategy: Pack A and B blocks into contiguous buffers
                      // This is the key innovation for cache optimization
                      
                      // Pack A block (mc_actual × kc_actual) into contiguous memory
                      Value packed_A = builder.create<memref::AllocOp>(loc, 
                          MemRefType::get({-1, -1}, rewriter.getF32Type()), 
                          ValueRange{mc_actual, kc_actual});
                      
                      auto packALoop = rewriter.create<scf::ForOp>(
                          loc, c0, mc_actual, c1, ValueRange{c0},
                          [&](OpBuilder &builder, Location loc, Value i_pack, ValueRange /*iterArgs*/) {
                            auto packKLoop = rewriter.create<scf::ForOp>(
                                loc, c0, kc_actual, c1, ValueRange{c0},
                                [&](OpBuilder &builder, Location loc, Value k_pack, ValueRange /*iterArgs*/) {
                                  Value a_i = builder.create<arith::AddIOp>(loc, ic, i_pack);
                                  Value a_k = builder.create<arith::AddIOp>(loc, pc, k_pack);
                                  Value a_val = builder.create<memref::LoadOp>(loc, A, ValueRange{a_i, a_k});
                                  builder.create<memref::StoreOp>(loc, a_val, packed_A, ValueRange{i_pack, k_pack});
                                  builder.create<scf::YieldOp>(loc, ValueRange{k_pack});
                                });
                            builder.create<scf::YieldOp>(loc, ValueRange{i_pack});
                          });
                      
                      // Pack B block (kc_actual × nc_actual) into contiguous memory
                      Value packed_B = builder.create<memref::AllocOp>(loc, 
                          MemRefType::get({-1, -1}, rewriter.getF32Type()), 
                          ValueRange{kc_actual, nc_actual});
                      
                      auto packBLoop = rewriter.create<scf::ForOp>(
                          loc, c0, kc_actual, c1, ValueRange{c0},
                          [&](OpBuilder &builder, Location loc, Value k_pack, ValueRange /*iterArgs*/) {
                            auto packJLoop = rewriter.create<scf::ForOp>(
                                loc, c0, nc_actual, c1, ValueRange{c0},
                                [&](OpBuilder &builder, Location loc, Value j_pack, ValueRange /*iterArgs*/) {
                                  Value b_k = builder.create<arith::AddIOp>(loc, pc, k_pack);
                                  Value b_j = builder.create<arith::AddIOp>(loc, jc, j_pack);
                                  Value b_val = builder.create<memref::LoadOp>(loc, B, ValueRange{b_k, b_j});
                                  builder.create<memref::StoreOp>(loc, b_val, packed_B, ValueRange{k_pack, j_pack});
                                  builder.create<scf::YieldOp>(loc, ValueRange{j_pack});
                                });
                            builder.create<scf::YieldOp>(loc, ValueRange{k_pack});
                          });
                      
                      // BLIS Micro-Kernel: jr × ir loops operating on packed data
                      // This is the true micro-kernel that operates on contiguous memory
                      auto jrLoop = rewriter.create<scf::ForOp>(
                          loc, c0, nc_actual, nr, ValueRange{c0},
                          [&](OpBuilder &builder, Location loc, Value jr, ValueRange /*iterArgs*/) {
                            
                            Value n_remaining_inner = builder.create<arith::SubIOp>(loc, nc_actual, jr);
                            Value nr_actual = builder.create<arith::MinSIOp>(loc, nr, n_remaining_inner);
                            
                            auto irLoop = rewriter.create<scf::ForOp>(
                                loc, c0, mc_actual, mr, ValueRange{c0},
                                [&](OpBuilder &builder, Location loc, Value ir, ValueRange /*iterArgs*/) {
                                  
                                  Value m_remaining_inner = builder.create<arith::SubIOp>(loc, mc_actual, ir);
                                  Value mr_actual = builder.create<arith::MinSIOp>(loc, mr, m_remaining_inner);
                                  
                                  // True BLIS Micro-Kernel: Vectorized computation on packed data
                                  // This operates on contiguous memory blocks for optimal cache performance
                                  Value sum_init = builder.create<arith::ConstantOp>(
                                      loc, rewriter.getF32Type(), rewriter.getF32FloatAttr(0.0));
                                  Value sum_init_vec = builder.create<vector::SplatOp>(
                                      loc, VectorType::get({32}, rewriter.getF32Type()), sum_init);
                                  
                                  // Micro-kernel computation: mr_actual rows × nr_actual columns
                                  auto krLoop = rewriter.create<scf::ForOp>(
                                      loc, c0, kc_actual, c1, 
                                      ValueRange{sum_init_vec, sum_init_vec, sum_init_vec, sum_init_vec,
                                                sum_init_vec, sum_init_vec, sum_init_vec, sum_init_vec},
                                      [&](OpBuilder &builder, Location loc, Value kr, ValueRange iterArgs) {
                                        
                                        // Load from packed A (contiguous memory access)
                                        Value a_row_0 = builder.create<memref::LoadOp>(loc, packed_A, ValueRange{ir, kr});
                                        Value a_row_1 = builder.create<memref::LoadOp>(loc, packed_A, 
                                            ValueRange{builder.create<arith::AddIOp>(loc, ir, c1), kr});
                                        // Create constant indices for row offsets
                                        Value c2 = builder.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(2));
                                        Value c3 = builder.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(3));
                                        Value c4 = builder.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(4));
                                        Value c5 = builder.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(5));
                                        Value c6 = builder.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(6));
                                        Value c7 = builder.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(7));
                                        
                                        Value a_row_2 = builder.create<memref::LoadOp>(loc, packed_A, 
                                            ValueRange{builder.create<arith::AddIOp>(loc, ir, c2), kr});
                                        Value a_row_3 = builder.create<memref::LoadOp>(loc, packed_A, 
                                            ValueRange{builder.create<arith::AddIOp>(loc, ir, c3), kr});
                                        Value a_row_4 = builder.create<memref::LoadOp>(loc, packed_A, 
                                            ValueRange{builder.create<arith::AddIOp>(loc, ir, c4), kr});
                                        Value a_row_5 = builder.create<memref::LoadOp>(loc, packed_A, 
                                            ValueRange{builder.create<arith::AddIOp>(loc, ir, c5), kr});
                                        Value a_row_6 = builder.create<memref::LoadOp>(loc, packed_A, 
                                            ValueRange{builder.create<arith::AddIOp>(loc, ir, c6), kr});
                                        Value a_row_7 = builder.create<memref::LoadOp>(loc, packed_A, 
                                            ValueRange{builder.create<arith::AddIOp>(loc, ir, c7), kr});
                                        
                                        // Broadcast A elements
                                        Value a_vec_0 = builder.create<vector::BroadcastOp>(
                                            loc, VectorType::get({32}, rewriter.getF32Type()), a_row_0);
                                        Value a_vec_1 = builder.create<vector::BroadcastOp>(
                                            loc, VectorType::get({32}, rewriter.getF32Type()), a_row_1);
                                        Value a_vec_2 = builder.create<vector::BroadcastOp>(
                                            loc, VectorType::get({32}, rewriter.getF32Type()), a_row_2);
                                        Value a_vec_3 = builder.create<vector::BroadcastOp>(
                                            loc, VectorType::get({32}, rewriter.getF32Type()), a_row_3);
                                        Value a_vec_4 = builder.create<vector::BroadcastOp>(
                                            loc, VectorType::get({32}, rewriter.getF32Type()), a_row_4);
                                        Value a_vec_5 = builder.create<vector::BroadcastOp>(
                                            loc, VectorType::get({32}, rewriter.getF32Type()), a_row_5);
                                        Value a_vec_6 = builder.create<vector::BroadcastOp>(
                                            loc, VectorType::get({32}, rewriter.getF32Type()), a_row_6);
                                        Value a_vec_7 = builder.create<vector::BroadcastOp>(
                                            loc, VectorType::get({32}, rewriter.getF32Type()), a_row_7);
                                        
                                        // Load from packed B (contiguous memory access)
                                        Value b_vec = builder.create<vector::LoadOp>(
                                            loc, VectorType::get({32}, rewriter.getF32Type()), packed_B, ValueRange{kr, jr});
                                        
                                        // Fused multiply-add operations
                                        Value res_sum_vec_0 = builder.create<vector::FMAOp>(loc, a_vec_0, b_vec, iterArgs[0]);
                                        Value res_sum_vec_1 = builder.create<vector::FMAOp>(loc, a_vec_1, b_vec, iterArgs[1]);
                                        Value res_sum_vec_2 = builder.create<vector::FMAOp>(loc, a_vec_2, b_vec, iterArgs[2]);
                                        Value res_sum_vec_3 = builder.create<vector::FMAOp>(loc, a_vec_3, b_vec, iterArgs[3]);
                                        Value res_sum_vec_4 = builder.create<vector::FMAOp>(loc, a_vec_4, b_vec, iterArgs[4]);
                                        Value res_sum_vec_5 = builder.create<vector::FMAOp>(loc, a_vec_5, b_vec, iterArgs[5]);
                                        Value res_sum_vec_6 = builder.create<vector::FMAOp>(loc, a_vec_6, b_vec, iterArgs[6]);
                                        Value res_sum_vec_7 = builder.create<vector::FMAOp>(loc, a_vec_7, b_vec, iterArgs[7]);
                                        
                                        builder.create<scf::YieldOp>(loc, 
                                            ValueRange{res_sum_vec_0, res_sum_vec_1, res_sum_vec_2, res_sum_vec_3,
                                                      res_sum_vec_4, res_sum_vec_5, res_sum_vec_6, res_sum_vec_7});
                                      });
                                  
                                  // Store results back to C with proper indexing
                                  Value c_j = builder.create<arith::AddIOp>(loc, jc, jr);
                                  Value c_i_0 = builder.create<arith::AddIOp>(loc, ic, ir);
                                  Value c_i_1 = builder.create<arith::AddIOp>(loc, c_i_0, c1);
                                  Value c_i_2 = builder.create<arith::AddIOp>(loc, c_i_0, c2);
                                  Value c_i_3 = builder.create<arith::AddIOp>(loc, c_i_0, c3);
                                  Value c_i_4 = builder.create<arith::AddIOp>(loc, c_i_0, c4);
                                  Value c_i_5 = builder.create<arith::AddIOp>(loc, c_i_0, c5);
                                  Value c_i_6 = builder.create<arith::AddIOp>(loc, c_i_0, c6);
                                  Value c_i_7 = builder.create<arith::AddIOp>(loc, c_i_0, c7);
                                  
                                  builder.create<vector::StoreOp>(loc, krLoop.getResult(0), C, ValueRange{c_i_0, c_j});
                                  builder.create<vector::StoreOp>(loc, krLoop.getResult(1), C, ValueRange{c_i_1, c_j});
                                  builder.create<vector::StoreOp>(loc, krLoop.getResult(2), C, ValueRange{c_i_2, c_j});
                                  builder.create<vector::StoreOp>(loc, krLoop.getResult(3), C, ValueRange{c_i_3, c_j});
                                  builder.create<vector::StoreOp>(loc, krLoop.getResult(4), C, ValueRange{c_i_4, c_j});
                                  builder.create<vector::StoreOp>(loc, krLoop.getResult(5), C, ValueRange{c_i_5, c_j});
                                  builder.create<vector::StoreOp>(loc, krLoop.getResult(6), C, ValueRange{c_i_6, c_j});
                                  builder.create<vector::StoreOp>(loc, krLoop.getResult(7), C, ValueRange{c_i_7, c_j});
                                  
                                  builder.create<scf::YieldOp>(loc, ValueRange{ir});
                                });
                            builder.create<scf::YieldOp>(loc, ValueRange{jr});
                          });
                      
                      // Clean up packed buffers
                      builder.create<memref::DeallocOp>(loc, packed_B);
                      builder.create<memref::DeallocOp>(loc, packed_A);
                      
                      builder.create<scf::YieldOp>(loc, ValueRange{ic});
                    });
                builder.create<scf::YieldOp>(loc, ValueRange{pc});
              });
          builder.create<scf::YieldOp>(loc, ValueRange{jc});
        });
    
    // Erase the original linalg.matmul operation
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t vecSize;
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

/// This pass converts linalg.matmul operations to hand-written implementations
/// that incorporate BLIS-style multi-level blocking and vectorization techniques
/// based on the paper "Anatomy of High-Performance Many-Threaded Matrix Multiplication".
class MyMatMulVectorizationPass
    : public PassWrapper<MyMatMulVectorizationPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyMatMulVectorizationPass)
  
  MyMatMulVectorizationPass() = default;
  MyMatMulVectorizationPass(const MyMatMulVectorizationPass &) {}

  StringRef getArgument() const final { return "my-matmul-vectorization"; }
  StringRef getDescription() const final {
    return "Convert linalg.matmul to BLIS-style hand-written implementation";
  }

  Option<int64_t> vectorSize{
      *this, "vector-size", llvm::cl::desc("Vector size for optimization"),
      llvm::cl::init(32)};

  void runOnOperation() override 
  {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    // Set up conversion target
    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, scf::SCFDialect, 
                          memref::MemRefDialect, vector::VectorDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
    target.addLegalOp<linalg::FillOp>();

    // Add our pattern to convert linalg.matmul operations
    RewritePatternSet patterns(context);
    patterns.add<MyMatMulVectorizationPattern>(context, vectorSize);
    
    // Apply partial conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<vector::VectorDialect>();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace buddy {

void registerMyMatMulVectorizationPass() {
  PassRegistration<MyMatMulVectorizationPass>();
}

} // namespace buddy
} // namespace mlir
