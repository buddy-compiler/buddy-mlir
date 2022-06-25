//===- math.h
//--------------------------------------------------------------===//
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

#ifndef INCLUDE_DAP_DSP_MATH
#define INCLUDE_DAP_DSP_MATH

#include <cmath>

namespace dap {
// Basic math functions
template <typename T> T sinc(T x) { return sin(x) / x; }

template <typename T> T besseli0(T x) {
  assert(0 && "Not implemented.");
  // return besseli(0,x);
  return 0;
}

//  float besseli(float _nu,
//                        float _z)
//  {
//    // special case: check for zeros; besseli_nu(0) = (nu = 0 ? 1 : 0)
//    if (_z == 0) {
//      return _nu == 0.0f ? 1.0f : 0.0f;
//    }
//
//    // special case: _nu = 1/2, besseli(z) = sqrt(2/pi*z)*sinh(z)
//    if (_nu == 0.5f) {
//      return sqrt(2.0f/(M_PI*_z)) * sinh(_z);
//    }
//
//    // low signal approximation
//    if (_z < 1e-3f*sqrtf(_nu + 1.0f)) {
//      return pow(0.5f*_z,_nu) / gamma(_nu + 1.0f);
//    }
//
//    // derive from logarithmic expansion
//    return exp( lnbesseli(_nu, _z) );
//  }

// More math functions with higher complexity
} // namespace dap

#endif // INCLUDE_DAP_DSP_MATH