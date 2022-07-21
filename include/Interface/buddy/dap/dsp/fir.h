//===- fir.h --------------------------------------------------------------===//
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
// Header file for Fir operation and other entities in DAP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef INCLUDE_DAP_DSP_FIR
#define INCLUDE_DAP_DSP_FIR

#include "Interface/buddy/core/Container.h"
#include "Interface/buddy/dap/AudioContainer.h"
#include "Interface/buddy/dap/dsp/window.h"

namespace dap {
namespace detail {
// Declare the Fir C interface.
extern "C" {
// TODO: support both float and double.
void _mlir_ciface_conv1d_buddy(MemRef<float, 1> *inputBuddyConv1D,
                               MemRef<float, 1> *kernelBuddyConv1D,
                               MemRef<float, 1> *outputBuddyConv1D);

void _mlir_ciface_conv1d_linalg(MemRef<float, 1> *inputBuddyConv1D,
                                MemRef<float, 1> *kernelBuddyConv1D,
                                MemRef<float, 1> *outputBuddyConv1D);
}
} // namespace detail

// type: see WINDOW_TYPE
// len: filter length
// cutoff: Lowpass cutoff frequency
// args: filter-specific arguments, size is limited using WINDOW_TYPE
template <typename T, size_t N>
void firLowpass(MemRef<T, N> &input, WINDOW_TYPE type, size_t len, T cutoff,
                T *args) {
  // TODO: setup a return enum for error handling?
  // TODO: check memref rank and apply filter onto each ranks.
  // only N=1 is supported for now.

  // TODO: check lowpass input range.
  bool normalize = true;
  T t, h1, h2;
  auto window = detail::_bind_window(type, args);
  T sum = 0;
  for (size_t i = 0; i < len; ++i) {
    t = (T)i - (T)(len - 1) / (T)2;
    h1 = sinc((T)2 * cutoff * t);
    h2 = window(i, len);
    input[i] = h1 * h2;
    sum += input[i];
  }
  if (normalize) {
    for (size_t i = 0; i < len; ++i) {
      input[i] /= sum;
    }
  }
}

template <typename T, size_t N>
void fir(MemRef<float, N> *input, MemRef<T, N> *filter,
         MemRef<float, N> *output) {
  if (N != 1)
    assert(0 && "Only mono audio is supported for now.");
  detail::_mlir_ciface_conv1d_buddy(input, filter, output);
}
} // namespace dap

#endif // INCLUDE_DAP_DSP_FIR
