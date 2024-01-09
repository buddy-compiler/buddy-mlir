//====- PerspectiveTransformUtils.h ---------------------------------------===//
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
// This file defines perspective transform utility functions for image
// processing.
//
//===----------------------------------------------------------------------===//
#ifndef BUDDY_MLIR_PERSPECTIVETRANSFORMUTILS_H
#define BUDDY_MLIR_PERSPECTIVETRANSFORMUTILS_H

using namespace mlir;

namespace buddy {
namespace dip {
void perspectiveTransformTail(OpBuilder &builder, Location loc, Value XY,
                              Value yiv, Value blockHeightIndex, Value xiv,
                              Value blockWidthIndex, Value stride,
                              SmallVector<SmallVector<Value, 3>, 3> &h);

void perspectiveTransform3dTail(OpBuilder &builder, Location loc, Value XY,
                                Value yStart, Value blockHeightIndex,
                                Value xStart, Value blockWidthIndex,
                                Value stride, Value Z0, Value Z1, Value Z3,
                                SmallVector<SmallVector<Value, 4>, 4> &mvp);

// forward mapping
void forwardRemap(OpBuilder &builder, Location loc, Value input, Value output,
                  Value XY, Value yStart, Value blockHeight, Value xStart,
                  Value blockWidth);
} // namespace dip
} // namespace buddy

#endif // BUDDY_MLIR_PERSPECTIVETRANSFORMUTILS_H