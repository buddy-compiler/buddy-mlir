//====- PerspectiveTransformUtils.cpp -------------------------------------===//
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
// This file implements perspective transform utility functions for image
// processing.
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <optional>

#include "Utils/Utils.h"
using namespace mlir;

namespace buddy {
namespace dip {
void perspectiveTransformTail(OpBuilder &builder, Location loc, Value XY,
                              Value yStart, Value blockHeightIndex,
                              Value xStart, Value blockWidthIndex, Value stride,
                              SmallVector<SmallVector<Value, 3>, 3> &h) {
  Value c0Index = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1Index = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value c1F64 =
      builder.create<arith::ConstantOp>(loc, builder.getF64FloatAttr(1.0));

  VectorType vectorTyF64 = VectorType::get({16}, builder.getF64Type());
  VectorType vectorTyI32 = VectorType::get({16}, builder.getI32Type());
  Value deltaInit = iotaVec0F64(builder, loc, 16);
  Value c1Vec = builder.create<vector::SplatOp>(loc, vectorTyF64, c1F64);
  Value strideI32Vec = builder.create<vector::SplatOp>(
      loc, vectorTyF64, indexToF64(builder, loc, stride));

  // block y loop start
  builder.create<scf::ForOp>(
      loc, c0Index, blockHeightIndex, c1Index, std::nullopt,
      [&](OpBuilder &builder, Location loc, Value yyiv, ValueRange) {
        Value deltaYI = builder.create<arith::AddIOp>(loc, yStart, yyiv);
        Value deltaYF64 = indexToF64(builder, loc, deltaYI);
        Value xivF64 = indexToF64(builder, loc, xStart);
        auto ReduceAdd = [&](std::vector<Value> values) {
          Value res;
          for (auto i = values.begin(); i != values.end(); i++) {
            if (i == values.begin())
              res = *i;
            else
              res = builder.create<arith::AddFOp>(loc, res, *i);
          }
          return res;
        };
        Value X0 = ReduceAdd(std::vector<Value>{
            builder.create<arith::MulFOp>(loc, h[0][0], xivF64),
            builder.create<arith::MulFOp>(loc, h[0][1], deltaYF64), h[0][2]});
        Value Y0 = ReduceAdd(std::vector<Value>{
            builder.create<arith::MulFOp>(loc, h[1][0], xivF64),
            builder.create<arith::MulFOp>(loc, h[1][1], deltaYF64), h[1][2]});
        Value W0 = ReduceAdd(std::vector<Value>{
            builder.create<arith::MulFOp>(loc, h[2][0], xivF64),
            builder.create<arith::MulFOp>(loc, h[2][1], deltaYF64), h[2][2]});

        Value M00Vec =
            builder.create<vector::SplatOp>(loc, vectorTyF64, h[0][0]);
        Value M10Vec =
            builder.create<vector::SplatOp>(loc, vectorTyF64, h[1][0]);
        Value M20Vec =
            builder.create<vector::SplatOp>(loc, vectorTyF64, h[2][0]);
        Value X0Vec = builder.create<vector::SplatOp>(loc, vectorTyF64, X0);
        Value Y0Vec = builder.create<vector::SplatOp>(loc, vectorTyF64, Y0);
        Value W0Vec = builder.create<vector::SplatOp>(loc, vectorTyF64, W0);
        Value upperBound =
            builder.create<arith::SubIOp>(loc, blockWidthIndex, stride);
        // block x loop start
        auto blockForOp = builder.create<scf::ForOp>(
            loc, c0Index, upperBound, stride, ValueRange{c0Index, deltaInit},
            [&](OpBuilder &builder, Location loc, Value xxiv,
                ValueRange iterArgs) {
              Value delta = iterArgs[1];
              Value WVec = builder.create<arith::AddFOp>(
                  loc, builder.create<arith::MulFOp>(loc, M20Vec, delta),
                  W0Vec);

              // TODO: Division by zero may occur here
              WVec = builder.create<arith::DivFOp>(loc, c1Vec, WVec);
              Value XVec = builder.create<arith::MulFOp>(
                  loc,
                  builder.create<arith::AddFOp>(
                      loc, X0Vec,
                      builder.create<arith::MulFOp>(loc, M00Vec, delta)),
                  WVec);

              Value YVec = builder.create<arith::MulFOp>(
                  loc,
                  builder.create<arith::AddFOp>(
                      loc, Y0Vec,
                      builder.create<arith::MulFOp>(loc, M10Vec, delta)),
                  WVec);
              Value XVecI32 =
                  builder.create<arith::FPToSIOp>(loc, vectorTyI32, XVec);
              Value YVecI32 =
                  builder.create<arith::FPToSIOp>(loc, vectorTyI32, YVec);
              // store coodinate
              builder.create<vector::StoreOp>(loc, XVecI32, XY,
                                              ValueRange{c0Index, yyiv, xxiv});
              builder.create<vector::StoreOp>(loc, YVecI32, XY,
                                              ValueRange{c1Index, yyiv, xxiv});
              Value xNext =
                  builder.create<arith::AddIOp>(loc, iterArgs[0], xxiv);
              Value deltaNext =
                  builder.create<arith::AddFOp>(loc, iterArgs[1], strideI32Vec);
              builder.create<scf::YieldOp>(loc, ValueRange{xNext, deltaNext});
            });
        // block x loop end
        // remaining part of x loop start
        builder.create<scf::ForOp>(
            loc, blockForOp.getResult(0), blockWidthIndex, c1Index,
            std::nullopt,
            [&](OpBuilder &builder, Location loc, Value xxiv, ValueRange) {
              Value delta = indexToF64(builder, loc, xxiv);
              Value W = builder.create<arith::AddFOp>(
                  loc, builder.create<arith::MulFOp>(loc, h[2][0], delta), W0);
              W = builder.create<arith::DivFOp>(loc, c1F64, W);
              Value X = builder.create<arith::MulFOp>(
                  loc,
                  builder.create<arith::AddFOp>(
                      loc, X0,
                      builder.create<arith::MulFOp>(loc, h[0][0], delta)),
                  W);
              Value Y = builder.create<arith::MulFOp>(
                  loc,
                  builder.create<arith::AddFOp>(
                      loc, Y0,
                      builder.create<arith::MulFOp>(loc, h[1][0], delta)),
                  W);
              Value XI32 =
                  builder.create<arith::FPToSIOp>(loc, builder.getI32Type(), X);
              Value YI32 =
                  builder.create<arith::FPToSIOp>(loc, builder.getI32Type(), Y);
              builder.create<memref::StoreOp>(loc, XI32, XY,
                                              ValueRange{c0Index, yyiv, xxiv});
              builder.create<memref::StoreOp>(loc, YI32, XY,
                                              ValueRange{c1Index, yyiv, xxiv});
              builder.create<scf::YieldOp>(loc);
            });
        // remaining part of x loop end
        builder.create<scf::YieldOp>(loc);
      });
  // block y loop end
}

void perspectiveTransform3dTail(OpBuilder &builder, Location loc, Value XY,
                                Value yStart, Value blockHeight, Value xStart,
                                Value blockWidth, Value stride, Value Z0,
                                Value Z1, Value Z3,
                                SmallVector<SmallVector<Value, 4>, 4> &mvp) {
  Value c0Index = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1Index = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value c1F64 =
      builder.create<arith::ConstantOp>(loc, builder.getF64FloatAttr(1.0));

  VectorType vectorTyF64 = VectorType::get({16}, builder.getF64Type());
  VectorType vectorTyI32 = VectorType::get({16}, builder.getI32Type());
  Value deltaInit = iotaVec0F64(builder, loc, 16);
  Value c1Vec = builder.create<vector::SplatOp>(loc, vectorTyF64, c1F64);
  Value strideI32Vec = builder.create<vector::SplatOp>(
      loc, vectorTyF64, indexToF64(builder, loc, stride));

  // X = [x, y, z, 1], where (y, z, 1) are constants
  // block y loop start
  builder.create<scf::ForOp>(
      loc, c0Index, blockHeight, c1Index, std::nullopt,
      [&](OpBuilder &builder, Location loc, Value yyiv, ValueRange) {
        Value deltaYI = builder.create<arith::AddIOp>(loc, yStart, yyiv);
        Value deltaYF64 = indexToF64(builder, loc, deltaYI);
        Value xivF64 = indexToF64(builder, loc, xStart);
        auto ReduceAdd = [&](std::vector<Value> values) {
          Value res;
          for (auto i = values.begin(); i != values.end(); i++) {
            if (i == values.begin())
              res = *i;
            else
              res = builder.create<arith::AddFOp>(loc, res, *i);
          }
          return res;
        };
        Value X0 = ReduceAdd(std::vector<Value>{
            builder.create<arith::MulFOp>(loc, mvp[0][0], xivF64),
            builder.create<arith::MulFOp>(loc, mvp[0][1], deltaYF64), Z0,
            mvp[0][3]});
        Value Y0 = ReduceAdd(std::vector<Value>{
            builder.create<arith::MulFOp>(loc, mvp[1][0], xivF64),
            builder.create<arith::MulFOp>(loc, mvp[1][1], deltaYF64), Z1,
            mvp[1][3]});
        Value W0 = ReduceAdd(std::vector<Value>{
            builder.create<arith::MulFOp>(loc, mvp[3][0], xivF64),
            builder.create<arith::MulFOp>(loc, mvp[3][1], deltaYF64), Z3,
            mvp[3][3]});

        Value M00Vec =
            builder.create<vector::SplatOp>(loc, vectorTyF64, mvp[0][0]);
        Value M10Vec =
            builder.create<vector::SplatOp>(loc, vectorTyF64, mvp[1][0]);
        Value M30Vec =
            builder.create<vector::SplatOp>(loc, vectorTyF64, mvp[3][0]);
        Value X0Vec = builder.create<vector::SplatOp>(loc, vectorTyF64, X0);
        Value Y0Vec = builder.create<vector::SplatOp>(loc, vectorTyF64, Y0);
        Value W0Vec = builder.create<vector::SplatOp>(loc, vectorTyF64, W0);
        Value upperBound =
            builder.create<arith::SubIOp>(loc, blockWidth, stride);
        // block x loop start
        auto blockForOp = builder.create<scf::ForOp>(
            loc, c0Index, upperBound, stride, ValueRange{c0Index, deltaInit},
            [&](OpBuilder &builder, Location loc, Value xxiv,
                ValueRange iterArgs) {
              Value delta = iterArgs[1];
              Value WVec = builder.create<arith::AddFOp>(
                  loc, builder.create<arith::MulFOp>(loc, M30Vec, delta),
                  W0Vec);

              // TODO: Division by zero may occur here
              WVec = builder.create<arith::DivFOp>(loc, c1Vec, WVec);
              Value XVec = builder.create<arith::MulFOp>(
                  loc,
                  builder.create<arith::AddFOp>(
                      loc, X0Vec,
                      builder.create<arith::MulFOp>(loc, M00Vec, delta)),
                  WVec);

              Value YVec = builder.create<arith::MulFOp>(
                  loc,
                  builder.create<arith::AddFOp>(
                      loc, Y0Vec,
                      builder.create<arith::MulFOp>(loc, M10Vec, delta)),
                  WVec);
              Value XVecI32 =
                  builder.create<arith::FPToSIOp>(loc, vectorTyI32, XVec);
              Value YVecI32 =
                  builder.create<arith::FPToSIOp>(loc, vectorTyI32, YVec);
              // store coodinate
              builder.create<vector::StoreOp>(loc, XVecI32, XY,
                                              ValueRange{c0Index, yyiv, xxiv});
              builder.create<vector::StoreOp>(loc, YVecI32, XY,
                                              ValueRange{c1Index, yyiv, xxiv});
              Value xNext =
                  builder.create<arith::AddIOp>(loc, iterArgs[0], xxiv);
              Value deltaNext =
                  builder.create<arith::AddFOp>(loc, iterArgs[1], strideI32Vec);
              builder.create<scf::YieldOp>(loc, ValueRange{xNext, deltaNext});
            });
        // block x loop end
        // remaining part of x loop start
        builder.create<scf::ForOp>(
            loc, blockForOp.getResult(0), blockWidth, c1Index, std::nullopt,
            [&](OpBuilder &builder, Location loc, Value xxiv, ValueRange) {
              Value delta = indexToF64(builder, loc, xxiv);
              Value W = builder.create<arith::AddFOp>(
                  loc, builder.create<arith::MulFOp>(loc, mvp[3][0], delta),
                  W0);
              W = builder.create<arith::DivFOp>(loc, c1F64, W);
              Value X = builder.create<arith::MulFOp>(
                  loc,
                  builder.create<arith::AddFOp>(
                      loc, X0,
                      builder.create<arith::MulFOp>(loc, mvp[0][0], delta)),
                  W);
              Value Y = builder.create<arith::MulFOp>(
                  loc,
                  builder.create<arith::AddFOp>(
                      loc, Y0,
                      builder.create<arith::MulFOp>(loc, mvp[1][0], delta)),
                  W);
              Value XI32 =
                  builder.create<arith::FPToSIOp>(loc, builder.getI32Type(), X);
              Value YI32 =
                  builder.create<arith::FPToSIOp>(loc, builder.getI32Type(), Y);
              builder.create<memref::StoreOp>(loc, XI32, XY,
                                              ValueRange{c0Index, yyiv, xxiv});
              builder.create<memref::StoreOp>(loc, YI32, XY,
                                              ValueRange{c1Index, yyiv, xxiv});
              builder.create<scf::YieldOp>(loc);
            });
        // remaining part of x loop end
        builder.create<scf::YieldOp>(loc);
      });
  // block y loop end
}

/// yStart, xStart, blockHeight, blockWidth are of the index type.
void forwardRemap(OpBuilder &builder, Location loc, Value input, Value output,
                  Value XY, Value yStart, Value blockHeight, Value xStart,
                  Value blockWidth) {
  Value c0Index = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1Index = builder.create<arith::ConstantIndexOp>(loc, 1);

  Value outputHeight = builder.create<memref::DimOp>(loc, output, c0Index);
  Value outputWidth = builder.create<memref::DimOp>(loc, output, c1Index);

  builder.create<scf::ForOp>(
      loc, c0Index, blockHeight, c1Index, std::nullopt,
      [&](OpBuilder &yBuilder, Location yLoc, Value yyiv, ValueRange) {
        Value srcY = yBuilder.create<arith::AddIOp>(yLoc, yStart, yyiv);
        yBuilder.create<scf::ForOp>(
            yLoc, c0Index, blockWidth, c1Index, std::nullopt,
            [&](OpBuilder &builder, Location loc, Value xxiv, ValueRange) {
              Value srcX = builder.create<arith::AddIOp>(loc, xStart, xxiv);
              Value dstX = builder.create<memref::LoadOp>(
                  loc, XY, ValueRange{c0Index, yyiv, xxiv});
              Value dstY = builder.create<memref::LoadOp>(
                  loc, XY, ValueRange{c1Index, yyiv, xxiv});
              Value xInBound = inBound(builder, loc,
                                       builder.create<arith::IndexCastOp>(
                                           loc, builder.getIndexType(), dstX),
                                       c0Index, outputWidth);
              Value yInBound = inBound(builder, loc,
                                       builder.create<arith::IndexCastOp>(
                                           loc, builder.getIndexType(), dstY),
                                       c0Index, outputHeight);
              Value pixelInBound =
                  builder.create<arith::AndIOp>(loc, xInBound, yInBound);

              builder.create<scf::IfOp>(
                  loc, pixelInBound, [&](OpBuilder &ifBuilder, Location ifLoc) {
                    Value dstXIndex = ifBuilder.create<arith::IndexCastOp>(
                        ifLoc, ifBuilder.getIndexType(), dstX);
                    Value dstYIndex = ifBuilder.create<arith::IndexCastOp>(
                        ifLoc, ifBuilder.getIndexType(), dstY);
                    Value pixel = ifBuilder.create<memref::LoadOp>(
                        ifLoc, input, ValueRange{srcX, srcY});
                    ifBuilder.create<memref::StoreOp>(
                        ifLoc, pixel, output, ValueRange{dstXIndex, dstYIndex});

                    ifBuilder.create<scf::YieldOp>(ifLoc);
                  });

              builder.create<scf::YieldOp>(loc);
            });

        yBuilder.create<scf::YieldOp>(yLoc);
      });
}
} // namespace dip
} // namespace buddy
