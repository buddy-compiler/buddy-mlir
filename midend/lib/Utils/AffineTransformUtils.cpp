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
                              Value resIntPart, Value yStart, Value yEnd,
                              Value xStart, Value xEnd, Value m1, Value m4,
                              Value xAddr1, Value xAddr2, Value rsvValVec,
                              Value strideVal, Value c0, Value c1, Value c_rsv,
                              int64_t stride) {
  VectorType vectorTyI32 =
      VectorType::get({stride}, IntegerType::get(builder.getContext(), 32));
  VectorType vectorTyI16 =
      VectorType::get({stride}, IntegerType::get(builder.getContext(), 16));

  scf::ForOp::create(builder, 
      loc, yStart, yEnd, c1, ValueRange{},
      [&](OpBuilder &yBuilder, Location yLoc, Value yiv, ValueRange) {
        Value yOffset = arith::SubIOp::create(yBuilder, yLoc, yiv, yStart);

        Value yF32 = indexToF32(yBuilder, yLoc, yiv);
        Value yF32_0 = arith::MulFOp::create(yBuilder, yLoc, yF32, m1);
        Value yF32_1 = arith::MulFOp::create(yBuilder, yLoc, yF32, m4);
        Value yF32_0_rsv = arith::MulFOp::create(yBuilder, yLoc, yF32_0, c_rsv);
        Value yF32_1_rsv = arith::MulFOp::create(yBuilder, yLoc, yF32_1, c_rsv);
        Value y0 = arith::FPToSIOp::create(yBuilder, yLoc, yBuilder.getI32Type(),
                                                    yF32_0_rsv);
        Value y1 = arith::FPToSIOp::create(yBuilder, yLoc, yBuilder.getI32Type(),
                                                    yF32_1_rsv);

        Value y0Vec = vector::BroadcastOp::create(yBuilder, yLoc, vectorTyI32, y0);
        Value y1Vec = vector::BroadcastOp::create(yBuilder, yLoc, vectorTyI32, y1);

        scf::ForOp::create(yBuilder, 
            yLoc, xStart, xEnd, strideVal, ValueRange{},
            [&](OpBuilder &xBuilder, Location xLoc, Value xiv, ValueRange) {
              Value xOffset = arith::SubIOp::create(xBuilder, xLoc, xiv, xStart);
              Value x0Vec = vector::LoadOp::create(xBuilder, xLoc, vectorTyI32,
                                                            xAddr1, xiv);
              Value x1Vec = vector::LoadOp::create(xBuilder, xLoc, vectorTyI32,
                                                            xAddr2, xiv);

              Value srcXVec =
                  arith::AddIOp::create(xBuilder, xLoc, x0Vec, y0Vec);
              Value srcYVec =
                  arith::AddIOp::create(xBuilder, xLoc, x1Vec, y1Vec);

              Value srcXVecShifted =
                  arith::ShRSIOp::create(xBuilder, xLoc, srcXVec, rsvValVec);
              Value srcYVecShifted =
                  arith::ShRSIOp::create(xBuilder, xLoc, srcYVec, rsvValVec);
              Value srcXVecInt = arith::TruncIOp::create(xBuilder, 
                  xLoc, vectorTyI16, srcXVecShifted);
              Value srcYVecInt = arith::TruncIOp::create(xBuilder, 
                  xLoc, vectorTyI16, srcYVecShifted);

              SmallVector<int64_t> maskVec;
              for (int i = 0; i < stride; i++) {
                maskVec.push_back(i);
                maskVec.push_back(i + stride);
              }
              Value res2Store = vector::ShuffleOp::create(xBuilder, 
                  loc, srcXVecInt, srcYVecInt, maskVec);
              vector::StoreOp::create(xBuilder, 
                  loc, res2Store, resIntPart, ValueRange{yOffset, xOffset, c0});

              scf::YieldOp::create(xBuilder, xLoc);
            });

        scf::YieldOp::create(yBuilder, yLoc);
      });
}

void affineTransformCore(OpBuilder &builder, Location loc, MLIRContext *ctx,
                         Value input, Value output, Value yStart, Value yEnd,
                         Value xStart, Value xEnd, Value m1, Value m4,
                         Value xAddr1, Value xAddr2, int64_t stride,
                         const int &RSV_BITS, int interp_type,
                         dip::ImageFormat format) {
  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);
  Value c_rsv = arith::ConstantOp::create(builder, 
      loc, builder.getF32FloatAttr((float)(1 << RSV_BITS)));
  Value rsvVal = arith::ConstantOp::create(builder, 
      loc, builder.getI32IntegerAttr(RSV_BITS));
  Value strideVal = arith::ConstantIndexOp::create(builder, loc, stride);
  VectorType vectorTyI32 =
      VectorType::get({stride}, IntegerType::get(builder.getContext(), 32));
  Value rsvValVec = vector::BroadcastOp::create(builder, loc, vectorTyI32, rsvVal);

  // create memref to store compute result for remap use
  // TODO: auto config BLOCK_SZ by input type. float->32, uchar->64
#define BLOCK_SZ 32
  MemRefType resIntPartType =
      MemRefType::get({BLOCK_SZ / 2, BLOCK_SZ * 2, 2},
                      IntegerType::get(builder.getContext(), 16));

  Value resIntPart = memref::AllocOp::create(builder, loc, resIntPartType);
  Value rowStride = arith::ConstantIndexOp::create(builder, loc, BLOCK_SZ / 2);
  Value colStride = arith::ConstantIndexOp::create(builder, loc, BLOCK_SZ * 2);
#undef BLOCK_SZ

  if (format == dip::ImageFormat::HW) {
    scf::ForOp::create(builder, 
        loc, yStart, yEnd, rowStride, ValueRange{},
        [&](OpBuilder &yBuilder, Location yLoc, Value yiv, ValueRange) {
          Value realYEnd = arith::MinUIOp::create(yBuilder, 
              yLoc, yEnd, arith::AddIOp::create(yBuilder, yLoc, yiv, rowStride));
          Value rows = arith::SubIOp::create(yBuilder, yLoc, realYEnd, yiv);

          scf::ForOp::create(yBuilder, 
              yLoc, xStart, xEnd, colStride, ValueRange{},
              [&](OpBuilder &xBuilder, Location xLoc, Value xiv, ValueRange) {
                Value realXEnd = arith::MinUIOp::create(xBuilder, 
                    xLoc, xEnd,
                    arith::AddIOp::create(xBuilder, xLoc, xiv, colStride));
                Value cols =
                    arith::SubIOp::create(xBuilder, xLoc, realXEnd, xiv);

                affineTransformCoreTiled(xBuilder, xLoc, resIntPart, yiv,
                                         realYEnd, xiv, realXEnd, m1, m4,
                                         xAddr1, xAddr2, rsvValVec, strideVal,
                                         c0, c1, c_rsv, stride);
                // remap
                remapNearest2D(xBuilder, xLoc, ctx, input, output, resIntPart,
                               yiv, xiv, rows, cols);

                scf::YieldOp::create(xBuilder, xLoc);
              });
          scf::YieldOp::create(yBuilder, yLoc);
        });

  } else if (format == dip::ImageFormat::NCHW ||
             format == dip::ImageFormat::NHWC) {
    Value inputBatch = memref::DimOp::create(builder, loc, input, c0);
    scf::ForOp::create(builder, 
        loc, c0, inputBatch, c1, ValueRange{},
        [&](OpBuilder &nBuilder, Location nLoc, Value niv, ValueRange) {
          scf::ForOp::create(nBuilder, 
              loc, yStart, yEnd, rowStride, ValueRange{},
              [&](OpBuilder &yBuilder, Location yLoc, Value yiv, ValueRange) {
                Value realYEnd = arith::MinUIOp::create(yBuilder, 
                    yLoc, yEnd,
                    arith::AddIOp::create(yBuilder, yLoc, yiv, rowStride));
                Value rows =
                    arith::SubIOp::create(yBuilder, yLoc, realYEnd, yiv);

                scf::ForOp::create(yBuilder, 
                    yLoc, xStart, xEnd, colStride, ValueRange{},
                    [&](OpBuilder &xBuilder, Location xLoc, Value xiv,
                        ValueRange) {
                      Value realXEnd = arith::MinUIOp::create(xBuilder, 
                          xLoc, xEnd,
                          arith::AddIOp::create(xBuilder, xLoc, xiv, colStride));
                      Value cols =
                          arith::SubIOp::create(xBuilder, xLoc, realXEnd, xiv);

                      affineTransformCoreTiled(
                          xBuilder, xLoc, resIntPart, yiv, realYEnd, xiv,
                          realXEnd, m1, m4, xAddr1, xAddr2, rsvValVec,
                          strideVal, c0, c1, c_rsv, stride);
                      // remap
                      remapNearest3D(xBuilder, xLoc, ctx, input, output,
                                     resIntPart, yiv, xiv, rows, cols, format,
                                     niv);

                      scf::YieldOp::create(xBuilder, xLoc);
                    });
                scf::YieldOp::create(yBuilder, yLoc);
              });
          scf::YieldOp::create(nBuilder, nLoc);
        });
  }

  memref::DeallocOp::create(builder, loc, resIntPart);
}

void remapNearest2D(OpBuilder &builder, Location loc, MLIRContext *ctx,
                    Value input, Value output, Value mapInt, Value yStart,
                    Value xStart, Value rows, Value cols) {
  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);
  Value inputRow = memref::DimOp::create(builder, loc, input, c0);
  Value inputCol = memref::DimOp::create(builder, loc, input, c1);
  scf::ForOp::create(builder, 
      loc, c0, rows, c1, ValueRange{},
      [&](OpBuilder &yBuilder, Location yLoc, Value yiv, ValueRange) {
        Value dstY = arith::AddIOp::create(yBuilder, yLoc, yiv, yStart);

        scf::ForOp::create(yBuilder, 
            yLoc, c0, cols, c1, ValueRange{},
            [&](OpBuilder &xBuilder, Location xLoc, Value xiv, ValueRange) {
              Value dstX = arith::AddIOp::create(xBuilder, xLoc, xiv, xStart);
              Value srcXI16 = memref::LoadOp::create(xBuilder, 
                  xLoc, mapInt, ValueRange{yiv, xiv, c0});
              Value srcYI16 = memref::LoadOp::create(xBuilder, 
                  xLoc, mapInt, ValueRange{yiv, xiv, c1});

              Value srcX = arith::IndexCastOp::create(xBuilder, 
                  xLoc, IndexType::get(xBuilder.getContext()), srcXI16);
              Value srcY = arith::IndexCastOp::create(xBuilder, 
                  xLoc, IndexType::get(xBuilder.getContext()), srcYI16);
              Value xInBound = inBound(xBuilder, xLoc, srcX, c0, inputCol);
              Value yInBound = inBound(xBuilder, xLoc, srcY, c0, inputRow);
              Value pixelInBound =
                  arith::AndIOp::create(xBuilder, xLoc, xInBound, yInBound);
              scf::IfOp::create(xBuilder, 
                  xLoc, pixelInBound,
                  [&](OpBuilder &thenBuilder, Location thenLoc) {
                    Value pixel = memref::LoadOp::create(thenBuilder, 
                        thenLoc, input, ValueRange{srcY, srcX});
                    memref::StoreOp::create(thenBuilder, thenLoc, pixel, output,
                                                        ValueRange{dstY, dstX});
                    scf::YieldOp::create(thenBuilder, thenLoc);
                  },
                  [&](OpBuilder &elseBuilder, Location elseLoc) {
                    auto inElemTy =
                        mlir::cast<mlir::MemRefType>(input.getType())
                            .getElementType();
                    Value pixel = insertZeroConstantOp(ctx, elseBuilder,
                                                       elseLoc, inElemTy);
                    memref::StoreOp::create(elseBuilder, elseLoc, pixel, output,
                                                        ValueRange{dstY, dstX});
                    scf::YieldOp::create(elseBuilder, elseLoc);
                  });

              scf::YieldOp::create(xBuilder, xLoc);
            });

        scf::YieldOp::create(yBuilder, yLoc);
      });
}

void remapNearest3D(OpBuilder &builder, Location loc, MLIRContext *ctx,
                    Value input, Value output, Value mapInt, Value yStart,
                    Value xStart, Value rows, Value cols,
                    dip::ImageFormat format, Value niv) {
  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);
  Value c2 = arith::ConstantIndexOp::create(builder, loc, 2);
  Value c3 = arith::ConstantIndexOp::create(builder, loc, 3);

  Value inputRow, inputCol, inputChannel;
  if (format == dip::ImageFormat::NHWC) {
    inputRow = memref::DimOp::create(builder, loc, input, c1);
    inputCol = memref::DimOp::create(builder, loc, input, c2);
    inputChannel = memref::DimOp::create(builder, loc, input, c3);
  } else if (format == dip::ImageFormat::NCHW) {
    inputRow = memref::DimOp::create(builder, loc, input, c2);
    inputCol = memref::DimOp::create(builder, loc, input, c3);
    inputChannel = memref::DimOp::create(builder, loc, input, c1);
  }
  Value is3Channel = arith::CmpIOp::create(builder, 
      loc, arith::CmpIPredicate::eq, inputChannel, c3);

  scf::ForOp::create(builder, 
      loc, c0, rows, c1, ValueRange{},
      [&](OpBuilder &yBuilder, Location yLoc, Value yiv, ValueRange) {
        Value dstY = arith::AddIOp::create(yBuilder, yLoc, yiv, yStart);

        scf::IfOp::create(yBuilder, 
            yLoc, is3Channel,
            [&](OpBuilder &thenBuilder, Location thenLoc) {
              //  3 channels, make common case fast
              scf::ForOp::create(thenBuilder, 
                  thenLoc, c0, cols, c1, ValueRange{},
                  [&](OpBuilder &xBuilder, Location xLoc, Value xiv,
                      ValueRange) {
                    Value dstX =
                        arith::AddIOp::create(xBuilder, xLoc, xiv, xStart);
                    Value srcXI16 = memref::LoadOp::create(xBuilder, 
                        xLoc, mapInt, ValueRange{yiv, xiv, c0});
                    Value srcYI16 = memref::LoadOp::create(xBuilder, 
                        xLoc, mapInt, ValueRange{yiv, xiv, c1});

                    Value srcX = arith::IndexCastOp::create(xBuilder, 
                        xLoc, IndexType::get(xBuilder.getContext()), srcXI16);
                    Value srcY = arith::IndexCastOp::create(xBuilder, 
                        xLoc, IndexType::get(xBuilder.getContext()), srcYI16);
                    Value xInBound =
                        inBound(xBuilder, xLoc, srcX, c0, inputCol);
                    Value yInBound =
                        inBound(xBuilder, xLoc, srcY, c0, inputRow);
                    Value pixelInBound = arith::AndIOp::create(xBuilder, 
                        xLoc, xInBound, yInBound);
                    scf::IfOp::create(xBuilder, 
                        xLoc, pixelInBound,
                        [&](OpBuilder &thenBuilder, Location thenLoc) {
                          if (format == dip::ImageFormat::NCHW) {
                            Value srcC0 = memref::LoadOp::create(thenBuilder, 
                                thenLoc, input,
                                ValueRange{niv, c0, srcY, srcX});
                            Value srcC1 = memref::LoadOp::create(thenBuilder, 
                                thenLoc, input,
                                ValueRange{niv, c1, srcY, srcX});
                            Value srcC2 = memref::LoadOp::create(thenBuilder, 
                                thenLoc, input,
                                ValueRange{niv, c2, srcY, srcX});
                            memref::StoreOp::create(thenBuilder, 
                                thenLoc, srcC0, output,
                                ValueRange{niv, c0, dstY, dstX});
                            memref::StoreOp::create(thenBuilder, 
                                thenLoc, srcC1, output,
                                ValueRange{niv, c1, dstY, dstX});
                            memref::StoreOp::create(thenBuilder, 
                                thenLoc, srcC2, output,
                                ValueRange{niv, c2, dstY, dstX});
                          } else if (format == dip::ImageFormat::NHWC) {
                            Value srcC0 = memref::LoadOp::create(thenBuilder, 
                                thenLoc, input,
                                ValueRange{niv, srcY, srcX, c0});
                            Value srcC1 = memref::LoadOp::create(thenBuilder, 
                                thenLoc, input,
                                ValueRange{niv, srcY, srcX, c1});
                            Value srcC2 = memref::LoadOp::create(thenBuilder, 
                                thenLoc, input,
                                ValueRange{niv, srcY, srcX, c2});
                            memref::StoreOp::create(thenBuilder, 
                                thenLoc, srcC0, output,
                                ValueRange{niv, dstY, dstX, c0});
                            memref::StoreOp::create(thenBuilder, 
                                thenLoc, srcC1, output,
                                ValueRange{niv, dstY, dstX, c1});
                            memref::StoreOp::create(thenBuilder, 
                                thenLoc, srcC2, output,
                                ValueRange{niv, dstY, dstX, c2});
                          }
                          scf::YieldOp::create(thenBuilder, thenLoc);
                        },
                        [&](OpBuilder &elseBuilder, Location elseLoc) {
                          auto inElemTy =
                              mlir::cast<mlir::MemRefType>(input.getType())
                                  .getElementType();
                          Value pixel = insertZeroConstantOp(ctx, elseBuilder,
                                                             elseLoc, inElemTy);
                          if (format == dip::ImageFormat::NCHW) {
                            memref::StoreOp::create(elseBuilder, 
                                elseLoc, pixel, output,
                                ValueRange{niv, c0, dstY, dstX});
                            memref::StoreOp::create(elseBuilder, 
                                elseLoc, pixel, output,
                                ValueRange{niv, c1, dstY, dstX});
                            memref::StoreOp::create(elseBuilder, 
                                elseLoc, pixel, output,
                                ValueRange{niv, c2, dstY, dstX});
                          } else if (format == dip::ImageFormat::NHWC) {
                            memref::StoreOp::create(elseBuilder, 
                                elseLoc, pixel, output,
                                ValueRange{niv, dstY, dstX, c0});
                            memref::StoreOp::create(elseBuilder, 
                                elseLoc, pixel, output,
                                ValueRange{niv, dstY, dstX, c1});
                            memref::StoreOp::create(elseBuilder, 
                                elseLoc, pixel, output,
                                ValueRange{niv, dstY, dstX, c2});
                          }
                          scf::YieldOp::create(elseBuilder, elseLoc);
                        });
                    scf::YieldOp::create(thenBuilder, thenLoc);
                  });
              scf::YieldOp::create(thenBuilder, thenLoc);
            },
            [&](OpBuilder &elseBuilder, Location elseLoc) {
              scf::ForOp::create(elseBuilder, 
                  elseLoc, c0, cols, c1, ValueRange{},
                  [&](OpBuilder &xBuilder, Location xLoc, Value xiv,
                      ValueRange) {
                    Value dstX =
                        arith::AddIOp::create(xBuilder, xLoc, xiv, xStart);
                    Value srcXI16 = memref::LoadOp::create(xBuilder, 
                        xLoc, mapInt, ValueRange{yiv, xiv, c0});
                    Value srcYI16 = memref::LoadOp::create(xBuilder, 
                        xLoc, mapInt, ValueRange{yiv, xiv, c1});

                    Value srcX = arith::IndexCastOp::create(xBuilder, 
                        xLoc, IndexType::get(xBuilder.getContext()), srcXI16);
                    Value srcY = arith::IndexCastOp::create(xBuilder, 
                        xLoc, IndexType::get(xBuilder.getContext()), srcYI16);
                    Value xInBound =
                        inBound(xBuilder, xLoc, srcX, c0, inputCol);
                    Value yInBound =
                        inBound(xBuilder, xLoc, srcY, c0, inputRow);
                    Value pixelInBound = arith::AndIOp::create(xBuilder, 
                        xLoc, xInBound, yInBound);
                    scf::IfOp::create(xBuilder, 
                        xLoc, pixelInBound,
                        [&](OpBuilder &thenBuilder, Location thenLoc) {
                          scf::ForOp::create(thenBuilder, 
                              thenLoc, c0, inputChannel, c1, ValueRange{},
                              [&](OpBuilder &cBuilder, Location cLoc, Value civ,
                                  ValueRange) {
                                if (format == dip::ImageFormat::NCHW) {
                                  Value srcC = memref::LoadOp::create(cBuilder, 
                                      cLoc, input,
                                      ValueRange{niv, civ, srcY, srcX});
                                  memref::StoreOp::create(cBuilder, 
                                      cLoc, srcC, output,
                                      ValueRange{niv, civ, dstY, dstX});
                                } else if (format == dip::ImageFormat::NHWC) {
                                  Value srcC = memref::LoadOp::create(cBuilder, 
                                      cLoc, input,
                                      ValueRange{niv, srcY, srcX, civ});
                                  memref::StoreOp::create(cBuilder, 
                                      cLoc, srcC, output,
                                      ValueRange{niv, dstY, dstX, civ});
                                }
                                scf::YieldOp::create(cBuilder, cLoc);
                              });
                          scf::YieldOp::create(thenBuilder, elseLoc);
                        },
                        [&](OpBuilder &elseBuilder, Location elseLoc) {
                          auto inElemTy =
                              mlir::cast<mlir::MemRefType>(input.getType())
                                  .getElementType();
                          Value pixel = insertZeroConstantOp(ctx, elseBuilder,
                                                             elseLoc, inElemTy);
                          scf::ForOp::create(elseBuilder, 
                              elseLoc, c0, inputChannel, c1, ValueRange{},
                              [&](OpBuilder &cBuilder, Location cLoc, Value civ,
                                  ValueRange) {
                                if (format == dip::ImageFormat::NCHW) {
                                  memref::StoreOp::create(cBuilder, 
                                      cLoc, pixel, output,
                                      ValueRange{niv, civ, dstY, dstX});
                                } else if (format == dip::ImageFormat::NHWC) {
                                  memref::StoreOp::create(cBuilder, 
                                      cLoc, pixel, output,
                                      ValueRange{niv, dstY, dstX, civ});
                                }
                                scf::YieldOp::create(cBuilder, cLoc);
                              });
                          scf::YieldOp::create(elseBuilder, elseLoc);
                        });
                    scf::YieldOp::create(xBuilder, xLoc);
                  });
              scf::YieldOp::create(elseBuilder, elseLoc);
            });

        scf::YieldOp::create(yBuilder, yLoc);
      });
}

} // namespace buddy
