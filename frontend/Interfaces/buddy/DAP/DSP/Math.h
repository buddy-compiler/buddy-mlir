//===- Math.h -------------------------------------------------------------===//
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
// Header file for mathematical operations needed in DAP operations.
//
//===----------------------------------------------------------------------===//

#ifndef FRONTEND_INTERFACES_BUDDY_DAP_DSP_MATH
#define FRONTEND_INTERFACES_BUDDY_DAP_DSP_MATH

#include <cmath>

namespace dap {
// Basic math functions
template <typename T> T sinc(T x) { return sin(x) / x; }

template <typename T> T besseli0(T x) {
  assert(0 && "Not implemented.");
  // return besseli(0,x);
  return 0;
}

// More math functions with higher complexity
} // namespace dap

#endif // FRONTEND_INTERFACES_BUDDY_DAP_DSP_MATH
