//===- window.h
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
// Header file for Window operations and other entities in DAP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef INCLUDE_DAP_DSP_WINDOW
#define INCLUDE_DAP_DSP_WINDOW

#include "Interface/buddy/core/Container.h"
#include "Interface/buddy/dap/dsp/math.h"

#include <cassert>

// TODO: implement input checks and error handling

namespace dap {
enum class WINDOW_TYPE {
  KBD,
  KAISER,
  HAMMING,
  HANN,
  BLACKMANHARRIS,
  BLACKMANHARRIS7,
  FLATTOP,
  TRIANGULAR
};

namespace detail {
// Window functions for internal calculations.
// References: liquid-dsp

template <typename T> T _window_kaiser(size_t i, size_t len, T beta){
  float t = (float)i - (float)(len-1)/2;
  float r = 2.0f*t/(float)(len-1);
  float a = besseli0(beta*sqrt(1-r*r));
  float b = besseli0(beta);
  //printf("kaiser(%3u,%u3,%6.3f) t:%8.3f, r:%8.3f, a:%8.3f, b:%8.3f\n", _i,_wlen,_beta,t,r,a,b);
  return a / b;
}

template <typename T> T _window_kbd(size_t i, size_t len, T beta){
  unsigned int M = len / 2;
  if (i >= M)
    return _window_kbd(len-i-1,len,beta);

  float w0 = 0.0f;
  float w1 = 0.0f;
  float w;
  for (auto j=0; j<=M; j++) {
    // compute Kaiser window
    w = _window_kaiser(j,M+1,beta);

    // accumulate window sums
    w1 += w;
    if (j <= i) w0 += w;
  }
  return sqrt(w0 / w1);
}

template <typename T> T _window_hamming(size_t i, size_t len) {
  return 0.53836 - 0.46164 * cosf((2 * M_PI * (T)i) / ((T)(len - 1)));
}

template <typename T> T _window_hann(size_t i, size_t len){
  return 0.5f - 0.5f*cos( (2*M_PI*(float)i) / ((float)(len-1)) );
}

template <typename T> T _window_blackmanharris(size_t i, size_t len){
  float a0 = 0.35875f;
  float a1 = 0.48829f;
  float a2 = 0.14128f;
  float a3 = 0.01168f;
  float t = 2*M_PI*(float)i / ((float)(len-1));

  return a0 - a1*cos(t) + a2*cos(2*t) - a3*cos(3*t);
}

template <typename T> T _window_blackmanharris7(size_t i, size_t len){
  float a0 = 0.27105f;
  float a1 = 0.43329f;
  float a2 = 0.21812f;
  float a3 = 0.06592f;
  float a4 = 0.01081f;
  float a5 = 0.00077f;
  float a6 = 0.00001f;
  float t = 2*M_PI*(float)i / ((float)(len-1));

  return a0 - a1*cos(  t) + a2*cos(2*t) - a3*cos(3*t)
         + a4*cos(4*t) - a5*cos(5*t) + a6*cos(6*t);
}

template <typename T> T _window_flattop(size_t i, size_t len){
  T a0 = 1.000f;
  T a1 = 1.930f;
  T a2 = 1.290f;
  T a3 = 0.388f;
  T a4 = 0.028f;
  T t = 2*M_PI*(T)i / ((float)(len-1));

  return a0 - a1*cos(t) + a2*cos(2*t) - a3*cos(3*t) + a4*cos(4*t);
}

template <typename T> T _window_triangular(size_t i, size_t len, T n){
  T v0 = (T)i - (T)((len-1)/2.0f);
  T v1 = n/2.0f;
  return 1.0 - abs(v0 / v1);
}

// TODO: performance improvement
template <typename T>
T _apply_window(WINDOW_TYPE type, size_t i, size_t len, T *args) {
  switch (type) {
  case WINDOW_TYPE::HAMMING:
    return _window_hamming<T>(i, len);
  case WINDOW_TYPE::KAISER:
    if(!args) assert(0 && "Argument not provided.");
    return _window_kaiser<T>(i,len,args[0]);
  case WINDOW_TYPE::KBD:
    if(!args) assert(0 && "Argument not provided.");
    return _window_kbd<T>(i,len,args[0]);
  case WINDOW_TYPE::HANN:
    return _window_hann<T>(i,len);
  case WINDOW_TYPE::BLACKMANHARRIS:
    return _window_blackmanharris<T>(i,len);
  case WINDOW_TYPE::BLACKMANHARRIS7:
    return _window_blackmanharris7<T>(i,len);
  case WINDOW_TYPE::FLATTOP:
    return _window_flattop<T>(i,len);
  case WINDOW_TYPE::TRIANGULAR:
    if(!args) assert(0 && "Argument not provided.");
    return _window_triangular<T>(i,len,args[0]);
  }
}
} // namespace detail
} // namespace dap

#endif // INCLUDE_DAP_DSP_WINDOW