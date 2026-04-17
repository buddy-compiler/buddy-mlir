//====- DAPUtils.cpp ------------------------------------------------------===//
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
// This file implements DAP dialect specific utility functions for the buddy
// compiler ecosystem.
//
//===----------------------------------------------------------------------===//

#ifndef UTILS_DAPUTILS_DEF
#define UTILS_DAPUTILS_DEF

#include <cassert>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <vector>

#include "DAP/DAPDialect.h"
#include "DAP/DAPOps.h"
#include "Utils/DAPUtils.h"
#include "Utils/Utils.h"

using namespace mlir;

namespace buddy {
namespace dap {

// Generate 5 vector params from SOS matrices
SmallVector<Value, 5> generateSOSParams(OpBuilder &rewriter, Location loc,
                                        VectorType vectorTy, Value f0, Value f1,
                                        Value c0, Value c1, Value c2, Value c4,
                                        Value c5, Value filterSize,
                                        Value kernel) {
  Value initB0 = vector::BroadcastOp::create(rewriter, loc, vectorTy, f1);
  Value initB1 = vector::BroadcastOp::create(rewriter, loc, vectorTy, f0);
  Value initB2 = vector::BroadcastOp::create(rewriter, loc, vectorTy, f0);
  Value initA1 = vector::BroadcastOp::create(rewriter, loc, vectorTy, f0);
  Value initA2 = vector::BroadcastOp::create(rewriter, loc, vectorTy, f0);

  // Distribute all params into 5 param vectors
  auto vecDistribute = scf::ForOp::create(rewriter, 
      loc, c0, filterSize, c1,
      ValueRange{initB0, initB1, initB2, initA1, initA2},
      [&](OpBuilder &builder, Location loc, Value iv, ValueRange iargs) {
        Value b0 =
            memref::LoadOp::create(builder, loc, kernel, ValueRange{iv, c0});
        Value b1 =
            memref::LoadOp::create(builder, loc, kernel, ValueRange{iv, c1});
        Value b2 =
            memref::LoadOp::create(builder, loc, kernel, ValueRange{iv, c2});
        Value a1 =
            memref::LoadOp::create(builder, loc, kernel, ValueRange{iv, c4});
        Value a2 =
            memref::LoadOp::create(builder, loc, kernel, ValueRange{iv, c5});

        Value b0Next = vector::InsertOp::create(builder, loc, b0, iargs[0], iv);
        Value b1Next = vector::InsertOp::create(builder, loc, b0, iargs[1], iv);
        Value b2Next = vector::InsertOp::create(builder, loc, b0, iargs[2], iv);
        Value a1Next = vector::InsertOp::create(builder, loc, b0, iargs[3], iv);
        Value a2Next = vector::InsertOp::create(builder, loc, b0, iargs[4], iv);

        scf::YieldOp::create(builder, 
            loc, std::vector<Value>{b0Next, b1Next, b2Next, a1Next, a2Next});
      });

  return SmallVector<Value, 5>{vecDistribute.getResults()};
}

// Processing iir operation, result are stored in output MemRef
void biquadProcess(OpBuilder &rewriter, Location loc, VectorType vectorTy,
                   Value f0, Value c0, Value c1, Value cUpperBound,
                   Value iUpperBound, SmallVector<Value, 5> SOSParams,
                   ArrayRef<int64_t> arrayRef, Value N, Value input,
                   Value output) {
  Value vecB0 = SOSParams[0];
  Value vecB1 = SOSParams[1];
  Value vecB2 = SOSParams[2];
  Value vecA1 = SOSParams[3];
  Value vecA2 = SOSParams[4];

  Value vecOut = vector::BroadcastOp::create(rewriter, loc, vectorTy, f0);
  Value vecS1 = vector::BroadcastOp::create(rewriter, loc, vectorTy, f0);
  Value vecS2 = vector::BroadcastOp::create(rewriter, loc, vectorTy, f0);

  // Injection stage for iir operation, no output produced
  auto injectionResult = scf::ForOp::create(rewriter, 
      loc, c0, cUpperBound, c1, ValueRange{vecOut, vecS1, vecS2},
      [&](OpBuilder &builder, Location loc, Value iv, ValueRange iargs) {
        Value inElem = memref::LoadOp::create(builder, loc, input, iv);
        Value vecInMoveRight = vector::ShuffleOp::create(builder, 
            loc, iargs[0], iargs[0], arrayRef);
        Value vecInNext =
            vector::InsertOp::create(builder, loc, inElem, vecInMoveRight, c0);
        Value vecOutNext =
            vector::FMAOp::create(builder, loc, vecB0, vecInNext, iargs[1]);

        Value vecS1Lhs =
            vector::FMAOp::create(builder, loc, vecB1, vecInNext, iargs[2]);
        Value vecS1Rhs = arith::MulFOp::create(builder, loc, vecA1, vecOutNext);
        Value vecS1Next =
            arith::SubFOp::create(builder, loc, vecS1Lhs, vecS1Rhs);

        Value vecS2Lhs = arith::MulFOp::create(builder, loc, vecB2, vecInNext);
        Value vecS2Rhs = arith::MulFOp::create(builder, loc, vecA2, vecOutNext);
        Value vecS2Next =
            arith::SubFOp::create(builder, loc, vecS2Lhs, vecS2Rhs);

        scf::YieldOp::create(builder, 
            loc, std::vector<Value>{vecOutNext, vecS1Next, vecS2Next});
      });

  Value upperBound = arith::SubIOp::create(rewriter, loc, N, cUpperBound);

  // Processing stage for iir operation, start to produce ouput
  auto processResult = scf::ForOp::create(rewriter, 
      loc, c0, upperBound, c1, ValueRange{injectionResult.getResults()},
      [&](OpBuilder &builder, Location loc, Value iv, ValueRange iargs) {
        Value index = arith::AddIOp::create(builder, loc, iv, cUpperBound);
        Value inElem = memref::LoadOp::create(builder, loc, input, index);
        Value vecInMoveRight = vector::ShuffleOp::create(builder, 
            loc, iargs[0], iargs[0], arrayRef);
        Value vecInNext =
            vector::InsertOp::create(builder, loc, inElem, vecInMoveRight, c0);
        Value vecOutNext =
            vector::FMAOp::create(builder, loc, vecB0, vecInNext, iargs[1]);
        Value outElem =
            vector::ExtractOp::create(builder, loc, vecOutNext, iUpperBound);
        memref::StoreOp::create(builder, loc, outElem, output, iv);

        Value vecS1Lhs =
            vector::FMAOp::create(builder, loc, vecB1, vecInNext, iargs[2]);
        Value vecS1Rhs = arith::MulFOp::create(builder, loc, vecA1, vecOutNext);
        Value vecS1Next =
            arith::SubFOp::create(builder, loc, vecS1Lhs, vecS1Rhs);

        Value vecS2Lhs = arith::MulFOp::create(builder, loc, vecB2, vecInNext);
        Value vecS2Rhs = arith::MulFOp::create(builder, loc, vecA2, vecOutNext);
        Value vecS2Next =
            arith::SubFOp::create(builder, loc, vecS2Lhs, vecS2Rhs);

        scf::YieldOp::create(builder, 
            loc, std::vector<Value>{vecOutNext, vecS1Next, vecS2Next});
      });

  // Tail ending stafe for iir operation, generate rest ouput
  scf::ForOp::create(rewriter, 
      loc, upperBound, N, c1, ValueRange{processResult.getResults()},
      [&](OpBuilder &builder, Location loc, Value iv, ValueRange iargs) {
        Value vecInMoveRight = vector::ShuffleOp::create(builder, 
            loc, iargs[0], iargs[0], arrayRef);
        Value vecInNext =
            vector::InsertOp::create(builder, loc, f0, vecInMoveRight, c0);
        Value vecOutNext =
            vector::FMAOp::create(builder, loc, vecB0, vecInNext, iargs[1]);
        Value outElem =
            vector::ExtractOp::create(builder, loc, vecOutNext, iUpperBound);
        memref::StoreOp::create(builder, loc, outElem, output, iv);

        Value vecS1Lhs =
            vector::FMAOp::create(builder, loc, vecB1, vecInNext, iargs[2]);
        Value vecS1Rhs = arith::MulFOp::create(builder, loc, vecA1, vecOutNext);
        Value vecS1Next =
            arith::SubFOp::create(builder, loc, vecS1Lhs, vecS1Rhs);

        Value vecS2Lhs = arith::MulFOp::create(builder, loc, vecB2, vecInNext);
        Value vecS2Rhs = arith::MulFOp::create(builder, loc, vecA2, vecOutNext);
        Value vecS2Next =
            arith::SubFOp::create(builder, loc, vecS2Lhs, vecS2Rhs);

        scf::YieldOp::create(builder, 
            loc, std::vector<Value>{vecOutNext, vecS1Next, vecS2Next});
      });
}

// Total process for a specific vector length iir vectorization process
void iirVectorizationProcess(OpBuilder &rewriter, Location loc, uint64_t vecLen,
                             FloatType floatType, Value f0, Value f1, Value c0,
                             Value c1, Value c2, Value c4, Value c5,
                             Value filterSize, Value kernel,
                             ArrayRef<int64_t> arrayRef, Value N, Value input,
                             Value output) {
  VectorType vectorTy = VectorType::get(vecLen, floatType);
  uint64_t vecLenMinusOne = vecLen - 1;
  Value cUpperBound =
      arith::ConstantIndexOp::create(rewriter, loc, vecLenMinusOne);
  Value iUpperBound = arith::ConstantIntOp::create(rewriter, 
      loc,
      /*value=*/vecLenMinusOne, /*width=*/64);

  auto SOSParams = dap::generateSOSParams(rewriter, loc, vectorTy, f0, f1, c0,
                                          c1, c2, c4, c5, filterSize, kernel);
  dap::biquadProcess(rewriter, loc, vectorTy, f0, c0, c1, cUpperBound,
                     iUpperBound, SOSParams, arrayRef, N, input, output);
}

} // namespace dap
} // namespace buddy
#endif // UTILS_DAPUTILS_DEF
