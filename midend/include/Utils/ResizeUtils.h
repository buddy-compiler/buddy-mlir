//====- ResizeUtils.h ----------------------------------------------------===//
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
// This file defines resize utility functions for image processing
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_MLIR_RESIZEUTILS_H
#define BUDDY_MLIR_RESIZEUTILS_H

using namespace mlir;

namespace buddy {
/**
 * @brief remap pixels from input to output using Nearest Neighbor Interpolation
 *
 * @param builder reference of rewriter/OpBuilder
 * @param loc Location
 * @param input input image with its type = Memref<?x?xu8/i26/i32/i64/f16/f32/f64>
 * @param output output image with its type = Memref<?x?xu8/i26/i32/i64/f16/f32/f64>
 * @param yMapVec mapping vector which maps row i in output image to row j in input image. yMapVec's type is Vector<?xIndex>
 * @param xMapVec mapping vector which maps col i in output image to col j in input image. xMapVec's type is Vector<?xIndex>
 * @param yStart remap region [yStart, yStart + rows]
 * @param xStart remap region [xStart, xStart + cols]
 * @param rows same as yStart
 * @param cols same as xStart
 */
void remapNearest(OpBuilder &builder, Location loc, Value input, Value output,
                  Value yMapVec, Value xMapVec, Value yStart, Value xStart,
                  Value rows, Value cols);

}

#endif // BUDDY_MLIR_RESIZEUTILS_H
