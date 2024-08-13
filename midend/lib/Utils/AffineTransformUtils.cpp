//====- AffineTransformUtils.cpp ------------------------------------------===//
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
// This file implements affine transform utility functions for image
// processing
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/IR/AffineValueMap.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>

#include "DIP/DIPDialect.h"
#include "DIP/DIPOps.h"
#include "Utils/AffineTransformUtils.h"
#include "Utils/Utils.h"

using namespace mlir;

namespace buddy {
// compute core(tiled)
void affineTransformCoreTiled(OpBuilder &builder, Location loc,
                              Value resIntPart, Value yStart,
                              Value yEnd, Value xStart, Value xEnd, Value m1,
                              Value m4, Value xAddr1, Value xAddr2,
                              Value rsvValVec, Value strideVal, Value c0,
                              Value c1, Value c_rsv, int64_t stride) {
  VectorType vectorTyI32 =
      VectorType::get({stride}, IntegerType::get(builder.getContext(), 32));
  VectorType vectorTyI16 =
      VectorType::get({stride}, IntegerType::get(builder.getContext(), 16));

  builder.create<scf::ForOp>(
      loc, yStart, yEnd, c1, std::nullopt,
      [&](OpBuilder &yBuilder, Location yLoc, Value yiv, ValueRange) {
        Value yOffset = yBuilder.create<arith::SubIOp>(yLoc, yiv, yStart);

        Value yF32 = indexToF32(yBuilder, yLoc, yiv);
        Value yF32_0 = yBuilder.create<arith::MulFOp>(yLoc, yF32, m1);
        Value yF32_1 = yBuilder.create<arith::MulFOp>(yLoc, yF32, m4);
        Value yF32_0_rsv = yBuilder.create<arith::MulFOp>(yLoc, yF32_0, c_rsv);
        Value yF32_1_rsv = yBuilder.create<arith::MulFOp>(yLoc, yF32_1, c_rsv);
        Value y0 = yBuilder.create<arith::FPToSIOp>(yLoc, yBuilder.getI32Type(),
                                                    yF32_0_rsv);
        Value y1 = yBuilder.create<arith::FPToSIOp>(yLoc, yBuilder.getI32Type(),
                                                    yF32_1_rsv);
        
        Value y0Vec = yBuilder.create<vector::SplatOp>(yLoc, vectorTyI32, y0);
        Value y1Vec = yBuilder.create<vector::SplatOp>(yLoc, vectorTyI32, y1);

        yBuilder.create<scf::ForOp>(
            yLoc, xStart, xEnd, strideVal, std::nullopt,
            [&](OpBuilder &xBuilder, Location xLoc, Value xiv, ValueRange) {
              Value xOffset = xBuilder.create<arith::SubIOp>(xLoc, xiv, xStart);
              Value x0Vec = xBuilder.create<vector::LoadOp>(xLoc, vectorTyI32,
                                                            xAddr1, xiv);
              Value x1Vec = xBuilder.create<vector::LoadOp>(xLoc, vectorTyI32,
                                                            xAddr2, xiv);

              Value srcXVec =
                  xBuilder.create<arith::AddIOp>(xLoc, x0Vec, y0Vec);
              Value srcYVec =
                  xBuilder.create<arith::AddIOp>(xLoc, x1Vec, y1Vec);

              Value srcXVecShifted =
                  xBuilder.create<arith::ShRSIOp>(xLoc, srcXVec, rsvValVec);
              Value srcYVecShifted =
                  xBuilder.create<arith::ShRSIOp>(xLoc, srcYVec, rsvValVec);
              Value srcXVecInt = xBuilder.create<arith::TruncIOp>(
                  xLoc, vectorTyI16, srcXVecShifted);
              Value srcYVecInt = xBuilder.create<arith::TruncIOp>(
                  xLoc, vectorTyI16, srcYVecShifted);

              SmallVector<int64_t> maskVec;
              for (int i = 0; i < stride; i++) {
                  maskVec.push_back(i);
                  maskVec.push_back(i + stride);
              }
              Value res2Store = xBuilder.create<vector::ShuffleOp>(
                  loc, srcXVecInt, srcYVecInt, maskVec);
              xBuilder.create<vector::StoreOp>(
                  loc, res2Store, resIntPart, ValueRange{yOffset, xOffset, c0});

              xBuilder.create<scf::YieldOp>(xLoc);
            });

        yBuilder.create<scf::YieldOp>(yLoc);
      });
}

void affineTransformCore(OpBuilder &builder, Location loc, MLIRContext *ctx,  Value input,
                         Value output, Value yStart, Value yEnd, Value xStart,
                         Value xEnd, Value m1, Value m4, Value xAddr1,
                         Value xAddr2, int64_t stride, const int &RSV_BITS,
                         int interp_type, dip::ImageFormat format) {
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value c_rsv = builder.create<arith::ConstantOp>(
      loc, builder.getF32FloatAttr((float)(1 << RSV_BITS)));
  Value rsvVal = builder.create<arith::ConstantOp>(
      loc, builder.getI32IntegerAttr(RSV_BITS));
  Value strideVal = builder.create<arith::ConstantIndexOp>(loc, stride);
  VectorType vectorTyI32 =
      VectorType::get({stride}, IntegerType::get(builder.getContext(), 32));
  Value rsvValVec = builder.create<vector::SplatOp>(loc, vectorTyI32, rsvVal);

  // create memref to store compute result for remap use
  // TODO: auto config BLOCK_SZ by input type. float->32, uchar->64
#define BLOCK_SZ 32
  MemRefType resIntPartType =
      MemRefType::get({BLOCK_SZ / 2, BLOCK_SZ * 2, 2},
                      IntegerType::get(builder.getContext(), 16));

  Value resIntPart = builder.create<memref::AllocOp>(loc, resIntPartType);

  Value rowStride = builder.create<arith::ConstantIndexOp>(loc, BLOCK_SZ / 2);
  Value colStride = builder.create<arith::ConstantIndexOp>(loc, BLOCK_SZ * 2);
#undef BLOCK_SZ

  builder.create<scf::ForOp>(
      loc, yStart, yEnd, rowStride, std::nullopt,
      [&](OpBuilder &yBuilder, Location yLoc, Value yiv, ValueRange) {

        Value realYEnd = yBuilder.create<arith::MinUIOp>(
            yLoc, yEnd, yBuilder.create<arith::AddIOp>(yLoc, yiv, rowStride));
        Value rows = yBuilder.create<arith::SubIOp>(yLoc, realYEnd, yiv);

        yBuilder.create<scf::ForOp>(
            yLoc, xStart, xEnd, colStride, std::nullopt,
            [&](OpBuilder &xBuilder, Location xLoc, Value xiv, ValueRange) {

              Value realXEnd = xBuilder.create<arith::MinUIOp>(
                  xLoc, xEnd,
                  xBuilder.create<arith::AddIOp>(xLoc, xiv, colStride));
              Value cols = xBuilder.create<arith::SubIOp>(xLoc, realXEnd, xiv);

              affineTransformCoreTiled(xBuilder, xLoc, resIntPart,
                                       yiv, realYEnd, xiv, realXEnd, m1, m4,
                                       xAddr1, xAddr2, rsvValVec, strideVal, c0,
                                       c1, c_rsv, stride);

              // remap
              if (format == dip::ImageFormat::HW) {
                remapNearest2D(xBuilder, xLoc, ctx, input, output, resIntPart, yiv, xiv,
                               rows, cols);
              } else {
                remapNearest4D(xBuilder, xLoc, ctx, input, output, resIntPart, yiv, xiv,
                               rows, cols, format);
              }

              xBuilder.create<scf::YieldOp>(xLoc);
            });
        yBuilder.create<scf::YieldOp>(yLoc);
      });

  builder.create<memref::DeallocOp>(loc, resIntPart);
}

void remapNearest2D(OpBuilder &builder, Location loc, MLIRContext *ctx, Value input, Value output,
                    Value mapInt, Value yStart, Value xStart, Value rows,
                    Value cols) {
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value inputRow = builder.create<memref::DimOp>(loc, input, c0);
  Value inputCol = builder.create<memref::DimOp>(loc, input, c1);
  builder.create<scf::ForOp>(
      loc, c0, rows, c1, std::nullopt,
      [&](OpBuilder &yBuilder, Location yLoc, Value yiv, ValueRange) {

        Value dstY = yBuilder.create<arith::AddIOp>(yLoc, yiv, yStart);

        yBuilder.create<scf::ForOp>(
            yLoc, c0, cols, c1, std::nullopt,
            [&](OpBuilder &xBuilder, Location xLoc, Value xiv, ValueRange) {
              Value dstX = xBuilder.create<arith::AddIOp>(xLoc, xiv, xStart);
              Value srcXI16 = xBuilder.create<memref::LoadOp>(
                  xLoc, mapInt, ValueRange{yiv, xiv, c0});
              Value srcYI16 = xBuilder.create<memref::LoadOp>(
                  xLoc, mapInt, ValueRange{yiv, xiv, c1});

              Value srcX = xBuilder.create<arith::IndexCastOp>(
                  xLoc, IndexType::get(xBuilder.getContext()), srcXI16);
              Value srcY = xBuilder.create<arith::IndexCastOp>(
                  xLoc, IndexType::get(xBuilder.getContext()), srcYI16);
              Value xInBound = inBound(xBuilder, xLoc, srcX, c0, inputCol);
              Value yInBound = inBound(xBuilder, xLoc, srcY, c0, inputRow);
              Value pixelInBound =
                  xBuilder.create<arith::AndIOp>(xLoc, xInBound, yInBound);
              xBuilder.create<scf::IfOp>(
                  xLoc, pixelInBound,
                  [&](OpBuilder &thenBuilder, Location thenLoc) {
                    Value pixel = thenBuilder.create<memref::LoadOp>(
                        thenLoc, input, ValueRange{srcY, srcX});
                    thenBuilder.create<memref::StoreOp>(thenLoc, pixel, output,
                                                      ValueRange{dstY, dstX});
                    thenBuilder.create<scf::YieldOp>(thenLoc);
                  },
                  [&](OpBuilder &elseBuilder, Location elseLoc) {
                    auto inElemTy = input.getType().cast<MemRefType>().getElementType();
                    Value pixel = insertZeroConstantOp(ctx, elseBuilder, elseLoc, inElemTy);
                    elseBuilder.create<memref::StoreOp>(elseLoc, pixel, output,
                                                      ValueRange{dstY, dstX});
                    elseBuilder.create<scf::YieldOp>(elseLoc);
                  });

              xBuilder.create<scf::YieldOp>(xLoc);
            });

        yBuilder.create<scf::YieldOp>(yLoc);
      });
}

void remapNearest4D(OpBuilder &builder, Location loc, MLIRContext *ctx, Value input, Value output,
                    Value mapInt, Value yStart, Value xStart, Value rows,
                    Value cols, dip::ImageFormat format) {
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value c2 = builder.create<arith::ConstantIndexOp>(loc, 2);
  Value c3 = builder.create<arith::ConstantIndexOp>(loc, 3);
  if (format == dip::ImageFormat::NHWC) {
    Value inputBatch = builder.create<memref::DimOp>(loc, input, c0);
    Value inputRow = builder.create<memref::DimOp>(loc, input, c1);
    Value inputCol = builder.create<memref::DimOp>(loc, input, c2);
    Value inputChannel = builder.create<memref::DimOp>(loc, input, c3);
    builder.create<scf::ForOp>(
      loc, c0, inputBatch, c1, std::nullopt,
      [&](OpBuilder &nBuilder, Location nLoc, Value niv, ValueRange) {
        nBuilder.create<scf::ForOp>(
          nLoc, c0, rows, c1, std::nullopt,
          [&](OpBuilder &yBuilder, Location yLoc, Value yiv, ValueRange) {
            Value dstY = yBuilder.create<arith::AddIOp>(yLoc, yiv, yStart);
            yBuilder.create<scf::ForOp>(
              yLoc, c0, cols, c1, std::nullopt,
              [&](OpBuilder &xBuilder, Location xLoc, Value xiv, ValueRange) {
                Value dstX = xBuilder.create<arith::AddIOp>(xLoc, xiv, xStart);
                Value srcXI16 = xBuilder.create<memref::LoadOp>(
                    xLoc, mapInt, ValueRange{yiv, xiv, c0});
                Value srcYI16 = xBuilder.create<memref::LoadOp>(
                    xLoc, mapInt, ValueRange{yiv, xiv, c1});

                Value srcX = xBuilder.create<arith::IndexCastOp>(
                    xLoc, IndexType::get(xBuilder.getContext()), srcXI16);
                Value srcY = xBuilder.create<arith::IndexCastOp>(
                    xLoc, IndexType::get(xBuilder.getContext()), srcYI16);
                Value xInBound = inBound(xBuilder, xLoc, srcX, c0, inputCol);
                Value yInBound = inBound(xBuilder, xLoc, srcY, c0, inputRow);
                Value pixelInBound =
                    xBuilder.create<arith::AndIOp>(xLoc, xInBound, yInBound);

                xBuilder.create<scf::IfOp>(
                  xLoc, pixelInBound,
                  [&](OpBuilder &thenBuilder, Location thenLoc) {
                    thenBuilder.create<scf::ForOp>(
                      thenLoc, c0, inputChannel, c1, std::nullopt,
                      [&](OpBuilder &cBuilder, Location cLoc, Value civ, ValueRange) {
                        Value srcC = cBuilder.create<memref::LoadOp>(
                            cLoc, input, ValueRange{niv, srcY, srcX, civ});
                        cBuilder.create<memref::StoreOp>(cLoc, srcC, output,
                                                      ValueRange{niv, dstY, dstX, civ});
                        cBuilder.create<scf::YieldOp>(cLoc);
                      });
                    thenBuilder.create<scf::YieldOp>(thenLoc);
                  },
                  [&](OpBuilder &elseBuilder, Location elseLoc) {
                    auto inElemTy = input.getType().cast<MemRefType>().getElementType();
                    Value pixel = insertZeroConstantOp(ctx, elseBuilder, elseLoc, inElemTy);
                    elseBuilder.create<scf::ForOp>(
                      elseLoc, c0, inputChannel, c1, std::nullopt,
                      [&](OpBuilder &cBuilder, Location cLoc, Value civ, ValueRange) {
                        cBuilder.create<memref::StoreOp>(cLoc, pixel, output,
                                                      ValueRange{niv, dstY, dstX, civ});
                        cBuilder.create<scf::YieldOp>(cLoc);
                      });
                    elseBuilder.create<scf::YieldOp>(elseLoc);
                  });

                xBuilder.create<scf::YieldOp>(xLoc);
              });
            yBuilder.create<scf::YieldOp>(yLoc);
          });
        nBuilder.create<scf::YieldOp>(nLoc);
    });
  } else if (format == dip::ImageFormat::NCHW) {

  }

}
} // namespace buddy
