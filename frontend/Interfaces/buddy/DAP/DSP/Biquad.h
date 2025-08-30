//===- Biquad.h -----------------------------------------------------------===//
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
// Header file for Biquad operation and other entities in DAP dialect.
//===----------------------------------------------------------------------===//

#ifndef FRONTEND_INTERFACES_BUDDY_DAP_DSP_BIQUAD
#define FRONTEND_INTERFACES_BUDDY_DAP_DSP_BIQUAD

#include "buddy/Core/Container.h"
#include "buddy/DAP/AudioContainer.h"

#include <cmath>

namespace dap {
namespace detail {
extern "C" {
// TODO: support both float and double.
void _mlir_ciface_buddy_biquad(MemRef<float, 1> *input,
                               MemRef<float, 1> *kernel,
                               MemRef<float, 1> *output);
}
} // namespace detail

// frequency: Normalized frequency (frequency_Hz / samplerate_Hz)
// Q: Q-factor
template <typename T, size_t N>
void biquadLowpass(MemRef<T, N> &input, T frequency, T Q) {

  const T K = tan(M_PI * frequency);
  const T K2 = K * K;
  const T norm = 1 / (1 + K / Q + K2);
  const T a0 = K2 * norm;
  const T a1 = 2 * a0;
  const T a2 = a0;
  const T b0 = 1;
  const T b1 = 2 * (K2 - 1) * norm;
  const T b2 = (1 - K / Q + K2) * norm;

  input[0] = a0;
  input[1] = a1;
  input[2] = a2;
  input[3] = b0;
  input[4] = b1;
  input[5] = b2;
}

template <typename T, size_t N>
void biquad(MemRef<float, N> *input, MemRef<T, N> *filter,
            MemRef<float, N> *output) {
  if (N != 1)
    assert(0 && "Only mono audio is supported for now.");
  detail::_mlir_ciface_buddy_biquad(input, filter, output);
}

} // namespace dap

#endif // FRONTEND_INTERFACES_BUDDY_DAP_DSP_BIQUAD
