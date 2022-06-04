//===- dip.h --------------------------------------------------------------===//
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
// Header file for DIP dialect specific operations and other entities.
//
//===----------------------------------------------------------------------===//

#ifndef INCLUDE_DIP
#define INCLUDE_DIP

#include <Interface/buddy/dip/memref.h>

namespace dip {
namespace detail {
// Functions present inside dip::detail are not meant to be called by users
// directly.
// Declare the Corr2D C interface.
extern "C" {
void _mlir_ciface_corr_2d_constant_padding(
    MemRef_descriptor input, MemRef_descriptor kernel, MemRef_descriptor output,
    unsigned int centerX, unsigned int centerY, float constantValue);

void _mlir_ciface_corr_2d_replicate_padding(
    MemRef_descriptor input, MemRef_descriptor kernel, MemRef_descriptor output,
    unsigned int centerX, unsigned int centerY, float constantValue);
}
} // namespace detail

enum class BOUNDARY_OPTION { CONSTANT_PADDING, REPLICATE_PADDING };

void Corr2D(MemRef_descriptor input, MemRef_descriptor kernel,
            MemRef_descriptor output, unsigned int centerX,
            unsigned int centerY, BOUNDARY_OPTION option,
            float constantValue = 0) {
  if (option == BOUNDARY_OPTION::CONSTANT_PADDING) {
    detail::_mlir_ciface_corr_2d_constant_padding(
        input, kernel, output, centerX, centerY, constantValue);
  } else if (option == BOUNDARY_OPTION::REPLICATE_PADDING) {
    detail::_mlir_ciface_corr_2d_replicate_padding(input, kernel, output,
                                                   centerX, centerY, 0);
  }
}
} // namespace dip

#endif
