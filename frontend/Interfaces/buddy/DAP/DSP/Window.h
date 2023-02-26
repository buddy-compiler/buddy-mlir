//===- Window.h -----------------------------------------------------------===//
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

#ifndef FRONTEND_INTERFACES_BUDDY_DAP_DSP_WINDOW
#define FRONTEND_INTERFACES_BUDDY_DAP_DSP_WINDOW

#include "buddy/Core/Container.h"
#include "buddy/DAP/DSP/Math.h"

#include <cassert>
#include <functional>

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

template <typename T> T _window_kaiser(size_t i, size_t len, T beta) {
  T t = (T)i - (T)(len - 1) / 2;
  T r = 2.0f * t / (T)(len - 1);
  T a = besseli0(beta * sqrt(1 - r * r));
  T b = besseli0(beta);
  // printf("kaiser(%3u,%u3,%6.3f) t:%8.3f, r:%8.3f, a:%8.3f, b:%8.3f\n",
  // _i,_wlen,_beta,t,r,a,b);
  return a / b;
}

template <typename T> T _window_kbd(size_t i, size_t len, T beta) {
  unsigned int M = len / 2;
  if (i >= M)
    return _window_kbd(len - i - 1, len, beta);

  T w0 = 0.0f;
  T w1 = 0.0f;
  T w;
  for (size_t j = 0; j <= M; j++) {
    // compute Kaiser window
    w = _window_kaiser(j, M + 1, beta);

    // accumulate window sums
    w1 += w;
    if (j <= i)
      w0 += w;
  }
  return sqrt(w0 / w1);
}

template <typename T> T _window_hamming(size_t i, size_t len) {
  return 0.53836 - 0.46164 * cosf((2 * M_PI * (T)i) / ((T)(len - 1)));
}

template <typename T> T _window_hann(size_t i, size_t len) {
  return 0.5f - 0.5f * cos((2 * M_PI * (T)i) / ((T)(len - 1)));
}

template <typename T> T _window_blackmanharris(size_t i, size_t len) {
  T a0 = 0.35875f;
  T a1 = 0.48829f;
  T a2 = 0.14128f;
  T a3 = 0.01168f;
  T t = 2 * M_PI * (T)i / ((T)(len - 1));

  return a0 - a1 * cos(t) + a2 * cos(2 * t) - a3 * cos(3 * t);
}

template <typename T> T _window_blackmanharris7(size_t i, size_t len) {
  T a0 = 0.27105f;
  T a1 = 0.43329f;
  T a2 = 0.21812f;
  T a3 = 0.06592f;
  T a4 = 0.01081f;
  T a5 = 0.00077f;
  T a6 = 0.00001f;
  T t = 2 * M_PI * (T)i / ((T)(len - 1));

  return a0 - a1 * cos(t) + a2 * cos(2 * t) - a3 * cos(3 * t) +
         a4 * cos(4 * t) - a5 * cos(5 * t) + a6 * cos(6 * t);
}

template <typename T> T _window_flattop(size_t i, size_t len) {
  T a0 = 1.000f;
  T a1 = 1.930f;
  T a2 = 1.290f;
  T a3 = 0.388f;
  T a4 = 0.028f;
  T t = 2 * M_PI * (T)i / ((T)(len - 1));

  return a0 - a1 * cos(t) + a2 * cos(2 * t) - a3 * cos(3 * t) + a4 * cos(4 * t);
}

template <typename T> T _window_triangular(size_t i, size_t len, T n) {
  T v0 = (T)i - (T)((len - 1) / 2.0f);
  T v1 = n / 2.0f;
  return 1.0 - abs(v0 / v1);
}

template <typename T>
std::function<T(size_t, size_t)> _bind_window(WINDOW_TYPE type, T *args) {
  using namespace std;
  using namespace std::placeholders;
  switch (type) {
  case WINDOW_TYPE::HAMMING:
    return bind(_window_hamming<T>, _1, _2);
  case WINDOW_TYPE::KAISER:
    if (!args)
      assert(0 && "Argument not provided.");
    return bind(_window_kaiser<T>, _1, _2, args[0]);
  case WINDOW_TYPE::KBD:
    if (!args)
      assert(0 && "Argument not provided.");
    return bind(_window_kbd<T>, _1, _2, args[0]);
  case WINDOW_TYPE::HANN:
    return bind(_window_hann<T>, _1, _2);
  case WINDOW_TYPE::BLACKMANHARRIS:
    return bind(_window_blackmanharris<T>, _1, _2);
  case WINDOW_TYPE::BLACKMANHARRIS7:
    return bind(_window_blackmanharris7<T>, _1, _2);
  case WINDOW_TYPE::FLATTOP:
    return bind(_window_flattop<T>, _1, _2);
  case WINDOW_TYPE::TRIANGULAR:
    if (!args)
      assert(0 && "Argument not provided.");
    return bind(_window_triangular<T>, _1, _2, args[0]);
  }
}
} // namespace detail
} // namespace dap

#endif // FRONTEND_INTERFACES_BUDDY_DAP_DSP_WINDOW
