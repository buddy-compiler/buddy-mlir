//====- Utils.h -----------------------------------------------------------===//
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
// This file defines generic utility functions for the buddy compiler ecosystem.
//
//===----------------------------------------------------------------------===//

#ifndef INCLUDE_UTILS_UTILS_H
#define INCLUDE_UTILS_UTILS_H

using namespace mlir;

namespace buddy {

// Function to test whether a value is equivalent to zero or not.
Value zeroCond(OpBuilder &builder, Location loc, Type elemType, Value value,
               Value zeroElem);

// Create an inverted mask having all 1's shifted to right side.
Value createInvertedMask(OpBuilder &builder, Location loc, Value strideVal,
                         VectorType vectorMaskTy, Value leftIndex);

// Cast a value from index type to f32 type.
Value indexToF32(OpBuilder &builder, Location loc, Value val);

// Cast a value from f32 type to index type.
Value F32ToIndex(OpBuilder &builder, Location loc, Value val);

// Round off floating point value to nearest integer type value.
Value roundOff(OpBuilder &builder, Location loc, Value val);

// Bound values to permissible range of allocatable values w.r.t output image.
Value valBound(OpBuilder &builder, Location loc, Value val, Value lastElemF32,
               Value c0F32);

// Equivalent of std::iota.
Value iotaVec(OpBuilder &builder, Location loc, MLIRContext *ctx,
              Value indexStart, Value strideVal, VectorType vecType, Value c0,
              int64_t stride);

// Cast index type value to f32 type and then expand it in a vector.
Value castAndExpand(OpBuilder &builder, Location loc, Value val,
                    VectorType vecType);

// Function for calculating complex addition of 2 input 1D complex vectors.
// Separate vectors for real and imaginary parts are expected.
inline std::vector<Value> complexVecAddI(OpBuilder &builder, Location loc,
                                         Value vec1Real, Value vec1Imag,
                                         Value vec2Real, Value vec2Imag);

// Function for calculating complex subtraction of 2 input 1D complex vectors.
// Separate vectors for real and imaginary parts are expected.
inline std::vector<Value> complexVecSubI(OpBuilder &builder, Location loc,
                                         Value vec1Real, Value vec1Imag,
                                         Value vec2Real, Value vec2Imag);

// Function for calculating complex product of 2 input 1D complex vectors.
// Separate vectors for real and imaginary parts are expected.
std::vector<Value> complexVecMulI(OpBuilder &builder, Location loc,
                                  Value vec1Real, Value vec1Imag,
                                  Value vec2Real, Value vec2Imag);

// Function for calculating Transpose of 2D input MemRef.
void scalar2DMemRefTranspose(OpBuilder &builder, Location loc, Value memref1,
                             Value memref2, Value memref1NumRows,
                             Value memref1NumCols, Value memref2NumRows,
                             Value memref2NumCols, Value c0);

// Function for calculating Hadamard product of complex type 2D MemRefs.
// Separate MemRefs for real and imaginary parts are expected.
void vector2DMemRefMultiply(OpBuilder &builder, Location loc, Value memRef1Real,
                            Value memRef1Imag, Value memRef2Real,
                            Value memRef2Imag, Value memRef3Real,
                            Value memRef3Imag, Value memRefNumRows,
                            Value memRefNumCols, Value c0, VectorType vecType);

// Function for implementing Cooley Tukey Butterfly algortihm for calculating
// inverse of discrete Fourier transform of invidiual 1D components of 2D input
// MemRef. Separate MemRefs for real and imaginary parts are expected.
void idft1DCooleyTukeyButterfly(OpBuilder &builder, Location loc,
                                Value memRefReal2D, Value memRefImag2D,
                                Value memRefLength, Value strideVal,
                                VectorType vecType, Value rowIndex, Value c0,
                                Value c1, int64_t step);

// Function for implementing Gentleman Sande Butterfly algortihm for calculating
// discrete Fourier transform of invidiual 1D components of 2D input MemRef.
// Separate MemRefs for real and imaginary parts are expected.
void dft1DGentlemanSandeButterfly(OpBuilder &builder, Location loc,
                                  Value memRefReal2D, Value memRefImag2D,
                                  Value memRefLength, Value strideVal,
                                  VectorType vecType, Value rowIndex, Value c0,
                                  Value c1, int64_t step);

// Function for applying inverse of discrete fourier transform on a 2D MemRef.
// Separate MemRefs for real and imaginary parts are expected.
void idft2D(OpBuilder &builder, Location loc, Value container2DReal,
            Value container2DImag, Value container2DRows, Value container2DCols,
            Value intermediateReal, Value intermediateImag, Value c0, Value c1,
            Value strideVal, VectorType vecType);

// Function for applying discrete fourier transform on a 2D MemRef. Separate
// MemRefs for real and imaginary parts are expected.
void dft2D(OpBuilder &builder, Location loc, Value container2DReal,
           Value container2DImag, Value container2DRows, Value container2DCols,
           Value intermediateReal, Value intermediateImag, Value c0, Value c1,
           Value strideVal, VectorType vecType);

} // namespace buddy

#endif // INCLUDE_UTILS_UTILS_H
