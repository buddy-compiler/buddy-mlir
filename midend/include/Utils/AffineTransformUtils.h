//
// Created by flagerlee on 4/27/23.
//

#ifndef BUDDY_MLIR_AFFINETRANSFORMUTILS_H
#define BUDDY_MLIR_AFFINETRANSFORMUTILS_H

using namespace mlir;

namespace buddy {
// Given x*m0+m2(and x*m3+m5) and m1(and m4), compute new x and y, then remap
// origin pixels to new pixels
void affineTransformCore(OpBuilder &builder, Location loc, Value input,
                         Value output, Value yStart, Value yEnd, Value xStart,
                         Value xEnd, Value m1, Value m4, Value xAddr1,
                         Value xAddr2, int64_t stride, int interp_type);

// remap using nearest neighbor interpolation
void remapNearest(OpBuilder &builder, Location loc, Value input, Value output,
                  Value mapInt, Value yStart, Value xStart, Value rows,
                  Value cols);

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
