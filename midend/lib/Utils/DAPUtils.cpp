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
  Value initB0 = rewriter.create<vector::SplatOp>(loc, vectorTy, f1);
  Value initB1 = rewriter.create<vector::SplatOp>(loc, vectorTy, f0);
  Value initB2 = rewriter.create<vector::SplatOp>(loc, vectorTy, f0);
  Value initA1 = rewriter.create<vector::SplatOp>(loc, vectorTy, f0);
  Value initA2 = rewriter.create<vector::SplatOp>(loc, vectorTy, f0);

  // Distribute all params into 5 param vectors
  auto vecDistribute = rewriter.create<scf::ForOp>(
      loc, c0, filterSize, c1,
      ValueRange{initB0, initB1, initB2, initA1, initA2},
      [&](OpBuilder &builder, Location loc, Value iv, ValueRange iargs) {
        Value b0 =
            builder.create<memref::LoadOp>(loc, kernel, ValueRange{iv, c0});
        Value b1 =
            builder.create<memref::LoadOp>(loc, kernel, ValueRange{iv, c1});
        Value b2 =
            builder.create<memref::LoadOp>(loc, kernel, ValueRange{iv, c2});
        Value a1 =
            builder.create<memref::LoadOp>(loc, kernel, ValueRange{iv, c4});
        Value a2 =
            builder.create<memref::LoadOp>(loc, kernel, ValueRange{iv, c5});

        Value b0Next =
            builder.create<vector::InsertElementOp>(loc, b0, iargs[0], iv);
        Value b1Next =
            builder.create<vector::InsertElementOp>(loc, b1, iargs[1], iv);
        Value b2Next =
            builder.create<vector::InsertElementOp>(loc, b2, iargs[2], iv);
        Value a1Next =
            builder.create<vector::InsertElementOp>(loc, a1, iargs[3], iv);
        Value a2Next =
            builder.create<vector::InsertElementOp>(loc, a2, iargs[4], iv);

        builder.create<scf::YieldOp>(
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

  Value vecOut = rewriter.create<vector::SplatOp>(loc, vectorTy, f0);
  Value vecS1 = rewriter.create<vector::SplatOp>(loc, vectorTy, f0);
  Value vecS2 = rewriter.create<vector::SplatOp>(loc, vectorTy, f0);

  // Injection stage for iir operation, no output produced
  auto injectionResult = rewriter.create<scf::ForOp>(
      loc, c0, cUpperBound, c1, ValueRange{vecOut, vecS1, vecS2},
      [&](OpBuilder &builder, Location loc, Value iv, ValueRange iargs) {
        Value inElem = builder.create<memref::LoadOp>(loc, input, iv);
        Value vecInMoveRight = builder.create<vector::ShuffleOp>(
            loc, iargs[0], iargs[0], arrayRef);
        Value vecInNext = builder.create<vector::InsertElementOp>(
            loc, inElem, vecInMoveRight, c0);
        Value vecOutNext =
            builder.create<vector::FMAOp>(loc, vecB0, vecInNext, iargs[1]);

        Value vecS1Lhs =
            builder.create<vector::FMAOp>(loc, vecB1, vecInNext, iargs[2]);
        Value vecS1Rhs = builder.create<arith::MulFOp>(loc, vecA1, vecOutNext);
        Value vecS1Next =
            builder.create<arith::SubFOp>(loc, vecS1Lhs, vecS1Rhs);

        Value vecS2Lhs = builder.create<arith::MulFOp>(loc, vecB2, vecInNext);
        Value vecS2Rhs = builder.create<arith::MulFOp>(loc, vecA2, vecOutNext);
        Value vecS2Next =
            builder.create<arith::SubFOp>(loc, vecS2Lhs, vecS2Rhs);

        builder.create<scf::YieldOp>(
            loc, std::vector<Value>{vecOutNext, vecS1Next, vecS2Next});
      });

  Value upperBound = rewriter.create<arith::SubIOp>(loc, N, cUpperBound);

  // Processing stage for iir operation, start to produce ouput
  auto processResult = rewriter.create<scf::ForOp>(
      loc, c0, upperBound, c1, ValueRange{injectionResult.getResults()},
      [&](OpBuilder &builder, Location loc, Value iv, ValueRange iargs) {
        Value index = builder.create<arith::AddIOp>(loc, iv, cUpperBound);
        Value inElem = builder.create<memref::LoadOp>(loc, input, index);
        Value vecInMoveRight = builder.create<vector::ShuffleOp>(
            loc, iargs[0], iargs[0], arrayRef);
        Value vecInNext = builder.create<vector::InsertElementOp>(
            loc, inElem, vecInMoveRight, c0);
        Value vecOutNext =
            builder.create<vector::FMAOp>(loc, vecB0, vecInNext, iargs[1]);
        Value outElem = builder.create<vector::ExtractElementOp>(
            loc, vecOutNext, iUpperBound);
        builder.create<memref::StoreOp>(loc, outElem, output, iv);

        Value vecS1Lhs =
            builder.create<vector::FMAOp>(loc, vecB1, vecInNext, iargs[2]);
        Value vecS1Rhs = builder.create<arith::MulFOp>(loc, vecA1, vecOutNext);
        Value vecS1Next =
            builder.create<arith::SubFOp>(loc, vecS1Lhs, vecS1Rhs);

        Value vecS2Lhs = builder.create<arith::MulFOp>(loc, vecB2, vecInNext);
        Value vecS2Rhs = builder.create<arith::MulFOp>(loc, vecA2, vecOutNext);
        Value vecS2Next =
            builder.create<arith::SubFOp>(loc, vecS2Lhs, vecS2Rhs);

        builder.create<scf::YieldOp>(
            loc, std::vector<Value>{vecOutNext, vecS1Next, vecS2Next});
      });

  // Tail ending stafe for iir operation, generate rest ouput
  rewriter.create<scf::ForOp>(
      loc, upperBound, N, c1, ValueRange{processResult.getResults()},
      [&](OpBuilder &builder, Location loc, Value iv, ValueRange iargs) {
        Value vecInMoveRight = builder.create<vector::ShuffleOp>(
            loc, iargs[0], iargs[0], arrayRef);
        Value vecInNext = builder.create<vector::InsertElementOp>(
            loc, f0, vecInMoveRight, c0);
        Value vecOutNext =
            builder.create<vector::FMAOp>(loc, vecB0, vecInNext, iargs[1]);
        Value outElem = builder.create<vector::ExtractElementOp>(
            loc, vecOutNext, iUpperBound);
        builder.create<memref::StoreOp>(loc, outElem, output, iv);

        Value vecS1Lhs =
            builder.create<vector::FMAOp>(loc, vecB1, vecInNext, iargs[2]);
        Value vecS1Rhs = builder.create<arith::MulFOp>(loc, vecA1, vecOutNext);
        Value vecS1Next =
            builder.create<arith::SubFOp>(loc, vecS1Lhs, vecS1Rhs);

        Value vecS2Lhs = builder.create<arith::MulFOp>(loc, vecB2, vecInNext);
        Value vecS2Rhs = builder.create<arith::MulFOp>(loc, vecA2, vecOutNext);
        Value vecS2Next =
            builder.create<arith::SubFOp>(loc, vecS2Lhs, vecS2Rhs);

        builder.create<scf::YieldOp>(
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
      rewriter.create<arith::ConstantIndexOp>(loc, vecLenMinusOne);
  Value iUpperBound = rewriter.create<arith::ConstantIntOp>(
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
