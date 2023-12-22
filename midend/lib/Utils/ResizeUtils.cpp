//====- ResizeUtils.cpp ---------------------------------------------------===//
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
// This file implements resize utility functions for image processing
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

#include "Utils/ResizeUtils.h"
#include "Utils/Utils.h"

namespace buddy {
void remapNearest(OpBuilder &builder, Location loc, Value input, Value output,
                  Value yMapVec, Value xMapVec, Value yStart, Value xStart,
                  Value rows, Value cols) {
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
              Value srcX = xBuilder.create<vector::ExtractElementOp>(
                  xLoc, IndexType::get(xBuilder.getContext()), xMapVec, xiv);
              Value srcY = xBuilder.create<vector::ExtractElementOp>(
                  xLoc, IndexType::get(xBuilder.getContext()), yMapVec, yiv);
              Value xInBound = inBound(xBuilder, xLoc, srcX, c0, inputCol);
              Value yInBound = inBound(xBuilder, xLoc, srcY, c0, inputRow);
              Value pixelInBound =
                  xBuilder.create<arith::AndIOp>(xLoc, xInBound, yInBound);
              xBuilder.create<scf::IfOp>(
                  xLoc, pixelInBound,
                  [&](OpBuilder &ifBuilder, Location ifLoc) {
                    Value pixel = ifBuilder.create<memref::LoadOp>(
                        ifLoc, input, ValueRange{srcY, srcX});
                    ifBuilder.create<memref::StoreOp>(ifLoc, pixel, output,
                                                      ValueRange{dstY, dstX});
                    ifBuilder.create<scf::YieldOp>(ifLoc);
                  });

              xBuilder.create<scf::YieldOp>(xLoc);
            });
        yBuilder.create<scf::YieldOp>(yLoc);
      });
}
} // namespace buddy
