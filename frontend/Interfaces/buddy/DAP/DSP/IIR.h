//===- IIR.h --------------------------------------------------------------===//
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
// Header file for IIR operation and other entities in DAP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef FRONTEND_INTERFACES_BUDDY_DAP_DSP_IIR
#define FRONTEND_INTERFACES_BUDDY_DAP_DSP_IIR

#include "buddy/Core/Container.h"
#include "buddy/DAP/AudioContainer.h"
#include "buddy/DAP/DSP/IIRDesign.h"

namespace dap {
namespace detail {
// Declare the Iir C interface.
extern "C" {
// TODO: support both float and double.
void _mlir_ciface_mlir_iir(MemRef<double, 1> *inputBuddyConv1D,
                           MemRef<double, 2> *kernelBuddyConv1D,
                           MemRef<double, 1> *outputBuddyConv1D);

void _mlir_ciface_mlir_iir_simd(MemRef<double, 1> *inputBuddyConv1D,
                                MemRef<double, 2> *kernelBuddyConv1D,
                                MemRef<double, 1> *outputBuddyConv1D);

void _mlir_ciface_buddy_iir(MemRef<double, 1> *inputBuddyConv1D,
                            MemRef<double, 2> *kernelBuddyConv1D,
                            MemRef<double, 1> *outputBuddyConv1D);

void _mlir_ciface_buddy_iir_simd(MemRef<double, 1> *inputBuddyConv1D,
                                 MemRef<double, 2> *kernelBuddyConv1D,
                                 MemRef<double, 1> *outputBuddyConv1D);
}
} // namespace detail

// filter: lowpass filter, Supports butterworth filter upto order 12 for now.
// frequency: cutoff frequency
// fs: frequency at which data is sampled
template <typename T, size_t N>
void iirLowpass(MemRef<T, N> &input, const zpk<T> &filter, T frequency, T fs) {
  // only N=2 is supported for now .
  // TODO: check input range.

  T warped = detail::warp_freq(frequency, fs);

  zpk<T> result = filter;
  result = detail::lp2lp_zpk(result, warped);
  result = detail::bilinear<float>(result, 2.0);
  auto bqs = detail::to_sos(result);
  int M = bqs[0].size();
  for (size_t i = 0; i < bqs.size(); i++) {
    for (size_t j = 0; j < M; j++) {
      input[i * M + j] = bqs[i][j];
    }
  }
}

template <typename T, size_t N, size_t M>
void iir(MemRef<float, N> *input, MemRef<T, M> *filter,
         MemRef<float, N> *output) {
  if (N != 1)
    assert(0 && "Only mono audio is supported for now.");
  if (M != 2)
    assert(0 && "Second Order Section (SOS) filter is only supported for now.");
  detail::_mlir_ciface_buddy_iir(input, filter, output);
}
} // namespace dap

#endif // FRONTEND_INTERFACES_BUDDY_DAP_DSP_IIR
