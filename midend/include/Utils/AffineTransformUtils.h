//====- AffineTransformUtils.h --------------------------------------------===//
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
// This file defines affine transform utility functions for image processing
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_MLIR_AFFINETRANSFORMUTILS_H
#define BUDDY_MLIR_AFFINETRANSFORMUTILS_H

using namespace mlir;

namespace buddy {
// Given x*m0+m2(and x*m3+m5) and m1(and m4), compute new x and y, then remap
// origin pixels to new pixels
void affineTransformCore(OpBuilder &builder, Location loc, MLIRContext *ctx,
                         Value input, Value output, Value yStart, Value yEnd,
                         Value xStart, Value xEnd, Value m1, Value m4,
                         Value xAddr1, Value xAddr2, int64_t stride,
                         const int &RSV_BITS, int interp_type,
                         dip::ImageFormat format);

// remap using nearest neighbor interpolation
void remapNearest2D(OpBuilder &builder, Location loc, MLIRContext *ctx,
                    Value input, Value output, Value mapInt, Value yStart,
                    Value xStart, Value rows, Value cols);

void remapNearest3D(OpBuilder &builder, Location loc, MLIRContext *ctx,
                    Value input, Value output, Value mapInt, Value yStart,
                    Value xStart, Value rows, Value cols,
                    dip::ImageFormat format, Value niv);

// remap using bilinear interpolation
void remapBilinear(OpBuilder &builder, Location loc, Value input, Value output,
                   Value mapInt, Value mapFrac);

// remap using bicubic interpolation
void remapBicubic(OpBuilder &builder, Location loc, Value input, Value output,
                  Value mapInt, Value mapFrac);

// remap using lancoz interpolation
void remapLancoz(OpBuilder &builder, Location loc, Value input, Value output,
                 Value mapInt, Value mapFrac);
} // namespace buddy

#endif // BUDDY_MLIR_AFFINETRANSFORMUTILS_H
