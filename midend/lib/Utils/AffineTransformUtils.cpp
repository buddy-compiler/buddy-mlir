//
// Created by flagerlee on 4/27/23.
//

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/IR/AffineValueMap.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>

#include "Utils/AffineTransformUtils.h"
#include "Utils/Utils.h"

using namespace mlir;

namespace buddy {
void affineTransformCore(OpBuilder &builder, Location loc, Value input,
                         Value output, Value yStart, Value yEnd, Value xStart,
                         Value xEnd, Value m1, Value m4, Value xAddr1,
                         Value xAddr2, int64_t stride, int interp_type) {
#define RSV_BITS 10
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value c_rsv = builder.create<arith::ConstantOp>(
      loc, builder.getF32FloatAttr((float)(1 << RSV_BITS)));
  Value rsvVal = builder.create<arith::ConstantOp>(
      loc, builder.getI32IntegerAttr(RSV_BITS));
  Value strideVal = builder.create<arith::ConstantIndexOp>(loc, stride);
  Value inputRow = builder.create<memref::DimOp>(loc, input, c0);
  Value inputCol = builder.create<memref::DimOp>(loc, input, c1);
  VectorType vectorTyI32 =
      VectorType::get({stride}, IntegerType::get(builder.getContext(), 32));

  builder.create<scf::ForOp>(
      loc, yStart, yEnd, c1, std::nullopt,
      [&](OpBuilder &yBuilder, Location yLoc, Value yiv, ValueRange) {
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
              Value x0Vec = xBuilder.create<vector::LoadOp>(xLoc, vectorTyI32,
                                                            xAddr1, xiv);
              Value x1Vec = xBuilder.create<vector::LoadOp>(xLoc, vectorTyI32,
                                                            xAddr2, xiv);
              Value srcXVec =
                  xBuilder.create<arith::AddIOp>(xLoc, x0Vec, y0Vec);
              Value srcYVec =
                  xBuilder.create<arith::AddIOp>(xLoc, x1Vec, y1Vec);

              // remapping
              xBuilder.create<scf::ForOp>(
                  xLoc, c0, strideVal, c1, std::nullopt,
                  [&](OpBuilder &rBuilder, Location rLoc, Value riv,
                      ValueRange) {
                    Value srcPixelXInt_rsv = rBuilder.create<vector::ExtractElementOp>(
                        rLoc, srcXVec, riv
                    );
                    Value srcPixelYInt_rsv = rBuilder.create<vector::ExtractElementOp>(
                        rLoc, srcYVec, riv
                    );
                    Value srcPixelXInt = rBuilder.create<arith::ShRSIOp>(
                        rLoc, srcPixelXInt_rsv, rsvVal
                    );
                    Value srcPixelYInt = rBuilder.create<arith::ShRSIOp>(
                        rLoc, srcPixelYInt_rsv, rsvVal
                    );
                    Value srcPixelXIndex = rBuilder.create<arith::IndexCastOp>(
                        rLoc, rBuilder.getIndexType(), srcPixelXInt
                    );
                    Value srcPixelYIndex = rBuilder.create<arith::IndexCastOp>(
                        rLoc, rBuilder.getIndexType(), srcPixelYInt
                    );
                    Value pixelInImage = rBuilder.create<arith::AndIOp>(
                        rLoc,
                        inBound(rBuilder, rLoc, srcPixelYIndex, c0, inputRow),
                        inBound(rBuilder, rLoc, srcPixelXIndex, c0, inputCol)
                    );
                    rBuilder.create<scf::IfOp>(rLoc, pixelInImage, [&](OpBuilder &ifBuilder, Location ifLoc) {
                      Value pixel = ifBuilder.create<memref::LoadOp>(ifLoc, input, ValueRange{srcPixelYIndex, srcPixelXIndex});
                      Value dstPixelXIndex = ifBuilder.create<arith::AddIOp>(ifLoc, riv, xiv);
                      Value dstPixelYIndex = yiv;
                      ifBuilder.create<memref::StoreOp>(ifLoc, pixel, output, ValueRange{dstPixelYIndex, dstPixelXIndex});
                      ifBuilder.create<scf::YieldOp>(ifLoc);
                    });
                    rBuilder.create<scf::YieldOp>(rLoc);
                  });

              xBuilder.create<scf::YieldOp>(xLoc);
            });

        yBuilder.create<scf::YieldOp>(yLoc);
      });

#undef RSV_BITS
}
} // namespace buddy