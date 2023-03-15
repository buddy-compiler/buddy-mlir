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

} // namespace buddy

#endif // INCLUDE_UTILS_UTILS_H
