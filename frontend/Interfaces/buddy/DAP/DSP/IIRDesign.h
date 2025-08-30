//===- IIRDesign.h --------------------------------------------------------===//
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
// Header file for IIR filter design operations and other entities in
// DAP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef FRONTEND_INTERFACES_BUDDY_DAP_DSP_IIRDESIGN
#define FRONTEND_INTERFACES_BUDDY_DAP_DSP_IIRDESIGN

#include <cmath>
#include <complex>
#include <functional>
#include <numeric>
#include <vector>

namespace dap {
// Conversion and transformation functions for internal IIR coefficients
// calculations.
// References: kfr library: https://github.com/kfrlib/kfr
template <typename T> struct zpk {
  std::vector<std::complex<T>> z;
  std::vector<std::complex<T>> p;
  T k;
};

// TODO: Add bessel
// TODO: Add chebyshev1, chebyshev2
template <typename T> zpk<T> butterworth(int N) {
  switch (N) {
  case 1:
    return {{}, {std::complex<T>(-1., +0.)}, 1};
  case 2:
    return {{},
            {std::complex<T>(-0.7071067811865476, -0.7071067811865476),
             std::complex<T>(-0.7071067811865476, +0.7071067811865476)},
            1};
  case 3:
    return {{},
            {std::complex<T>(-0.5000000000000001, -0.8660254037844386),
             std::complex<T>(-1., +0.),
             std::complex<T>(-0.5000000000000001, +0.8660254037844386)},
            1};
  case 4:
    return {{},
            {std::complex<T>(-0.38268343236508984, -0.9238795325112867),
             std::complex<T>(-0.9238795325112867, -0.3826834323650898),
             std::complex<T>(-0.9238795325112867, +0.3826834323650898),
             std::complex<T>(-0.38268343236508984, +0.9238795325112867)},
            1};
  case 5:
    return {{},
            {std::complex<T>(-0.30901699437494745, -0.9510565162951535),
             std::complex<T>(-0.8090169943749475, -0.5877852522924731),
             std::complex<T>(-1., +0.),
             std::complex<T>(-0.8090169943749475, +0.5877852522924731),
             std::complex<T>(-0.30901699437494745, +0.9510565162951535)},
            1};
  case 6:
    return {{},
            {std::complex<T>(-0.25881904510252096, -0.9659258262890682),
             std::complex<T>(-0.7071067811865476, -0.7071067811865476),
             std::complex<T>(-0.9659258262890683, -0.25881904510252074),
             std::complex<T>(-0.9659258262890683, +0.25881904510252074),
             std::complex<T>(-0.7071067811865476, +0.7071067811865476),
             std::complex<T>(-0.25881904510252096, +0.9659258262890682)},
            1};
  case 7:
    return {{},
            {std::complex<T>(-0.22252093395631445, -0.9749279121818236),
             std::complex<T>(-0.6234898018587336, -0.7818314824680298),
             std::complex<T>(-0.9009688679024191, -0.4338837391175581),
             std::complex<T>(-1., +0.),
             std::complex<T>(-0.9009688679024191, +0.4338837391175581),
             std::complex<T>(-0.6234898018587336, +0.7818314824680298),
             std::complex<T>(-0.22252093395631445, +0.9749279121818236)},
            1};
  case 8:
    return {{},
            {std::complex<T>(-0.19509032201612833, -0.9807852804032304),
             std::complex<T>(-0.5555702330196023, -0.8314696123025452),
             std::complex<T>(-0.8314696123025452, -0.5555702330196022),
             std::complex<T>(-0.9807852804032304, -0.19509032201612825),
             std::complex<T>(-0.9807852804032304, +0.19509032201612825),
             std::complex<T>(-0.8314696123025452, +0.5555702330196022),
             std::complex<T>(-0.5555702330196023, +0.8314696123025452),
             std::complex<T>(-0.19509032201612833, +0.9807852804032304)},
            1};
  case 9:
    return {{},
            {std::complex<T>(-0.17364817766693041, -0.984807753012208),
             std::complex<T>(-0.5000000000000001, -0.8660254037844386),
             std::complex<T>(-0.766044443118978, -0.6427876096865393),
             std::complex<T>(-0.9396926207859084, -0.3420201433256687),
             std::complex<T>(-1., +0.),
             std::complex<T>(-0.9396926207859084, +0.3420201433256687),
             std::complex<T>(-0.766044443118978, +0.6427876096865393),
             std::complex<T>(-0.5000000000000001, +0.8660254037844386),
             std::complex<T>(-0.17364817766693041, +0.984807753012208)},
            1};
  case 10:
    return {{},
            {std::complex<T>(-0.15643446504023092, -0.9876883405951378),
             std::complex<T>(-0.4539904997395468, -0.8910065241883678),
             std::complex<T>(-0.7071067811865476, -0.7071067811865476),
             std::complex<T>(-0.8910065241883679, -0.45399049973954675),
             std::complex<T>(-0.9876883405951378, -0.15643446504023087),
             std::complex<T>(-0.9876883405951378, +0.15643446504023087),
             std::complex<T>(-0.8910065241883679, +0.45399049973954675),
             std::complex<T>(-0.7071067811865476, +0.7071067811865476),
             std::complex<T>(-0.4539904997395468, +0.8910065241883678),
             std::complex<T>(-0.15643446504023092, +0.9876883405951378)},
            1};
  case 11:
    return {{},
            {std::complex<T>(-0.14231483827328512, -0.9898214418809327),
             std::complex<T>(-0.41541501300188644, -0.9096319953545183),
             std::complex<T>(-0.654860733945285, -0.7557495743542583),
             std::complex<T>(-0.8412535328311812, -0.5406408174555976),
             std::complex<T>(-0.9594929736144974, -0.28173255684142967),
             std::complex<T>(-1., +0.),
             std::complex<T>(-0.9594929736144974, +0.28173255684142967),
             std::complex<T>(-0.8412535328311812, +0.5406408174555976),
             std::complex<T>(-0.654860733945285, +0.7557495743542583),
             std::complex<T>(-0.41541501300188644, +0.9096319953545183),
             std::complex<T>(-0.14231483827328512, +0.9898214418809327)},
            1};
  case 12:
    return {{},
            {std::complex<T>(-0.13052619222005193, -0.9914448613738104),
             std::complex<T>(-0.38268343236508984, -0.9238795325112867),
             std::complex<T>(-0.6087614290087207, -0.7933533402912352),
             std::complex<T>(-0.7933533402912353, -0.6087614290087205),
             std::complex<T>(-0.9238795325112867, -0.3826834323650898),
             std::complex<T>(-0.9914448613738104, -0.13052619222005157),
             std::complex<T>(-0.9914448613738104, +0.13052619222005157),
             std::complex<T>(-0.9238795325112867, +0.3826834323650898),
             std::complex<T>(-0.7933533402912353, +0.6087614290087205),
             std::complex<T>(-0.6087614290087207, +0.7933533402912352),
             std::complex<T>(-0.38268343236508984, +0.9238795325112867),
             std::complex<T>(-0.13052619222005193, +0.9914448613738104)},
            1};
  default:
    return {{}, {}, 1.0};
  }
}

namespace detail {
template <typename T> zpk<T> bilinear(const zpk<T> &filter, T fs) {
  const T fs2 = 2.0 * fs;
  zpk<T> result;

  // result.z = (fs2 + filter.z) / (fs2 - filter.z);
  std::vector<std::complex<T>> z_num = filter.z;
  std::vector<std::complex<T>> z_deno = filter.z;
  result.z = filter.z;
  std::for_each(z_num.begin(), z_num.end(),
                [&fs2](auto &element) { return element = fs2 + element; });
  std::for_each(z_deno.begin(), z_deno.end(),
                [&fs2](auto &element) { return element = fs2 - element; });
  std::transform(z_num.begin(), z_num.end(), z_deno.begin(), result.z.begin(),
                 std::divides<std::complex<T>>());

  // result.p = (fs2 + filter.p) / (fs2 - filter.p);
  std::vector<std::complex<T>> p_num = filter.p;
  std::vector<std::complex<T>> p_deno = filter.p;
  result.p = filter.p;
  std::for_each(p_num.begin(), p_num.end(),
                [&fs2](auto &element) { return element = fs2 + element; });
  std::for_each(p_deno.begin(), p_deno.end(),
                [&fs2](auto &element) { return element = fs2 - element; });
  std::transform(p_num.begin(), p_num.end(), p_deno.begin(), result.p.begin(),
                 std::divides<std::complex<T>>());

  result.z.resize(result.p.size(), std::complex<T>(-1));

  // result.k = filter.k * real(product(fs2 - filter.z) / product(fs2 -
  // filter.p));
  auto k_num = accumulate(z_deno.begin(), z_deno.end(), std::complex<T>(1),
                          std::multiplies<std::complex<T>>());
  auto k_den = accumulate(p_deno.begin(), p_deno.end(), std::complex<T>(1),
                          std::multiplies<std::complex<T>>());
  result.k = filter.k * (k_num / k_den).real();
  return result;
}

template <typename T> struct zero_pole_pairs {
  std::complex<T> p1, p2, z1, z2;
};

template <typename T>
std::vector<T> zpk2tf_poly(const std::complex<T> &x, const std::complex<T> &y) {
  return {T(1), -(x.real() + y.real()),
          x.real() * y.real() - x.imag() * y.imag()};
}

template <typename T>
std::vector<T> zpk2tf(const zero_pole_pairs<T> &pairs, T k) {
  // std::vector<T> zz = k * zpk2tf_poly(pairs.z1, pairs.z2);
  std::vector<T> zz = zpk2tf_poly(pairs.z1, pairs.z2);
  std::for_each(zz.begin(), zz.end(),
                [&k](auto &element) { return element = k * element; });

  std::vector<T> pp = zpk2tf_poly(pairs.p1, pairs.p2);
  return {zz[0], zz[1], zz[2], pp[0], pp[1], pp[2]};
}

template <typename T> bool isreal(const std::complex<T> &x) {
  return x.imag() == 0;
}

template <typename T>
std::vector<std::complex<T>>
cplxreal(const std::vector<std::complex<T>> &list) {
  std::vector<std::complex<T>> x = list;
  std::sort(x.begin(), x.end(),
            [](const std::complex<T> &a, const std::complex<T> &b) {
              return a.real() < b.real();
            });
  T tol = std::numeric_limits<T>::epsilon() * 100;
  std::vector<std::complex<T>> result = x;
  for (size_t i = result.size(); i > 1; i--) {
    if (!isreal(result[i - 1]) && !isreal(result[i - 2])) {

      if (std::abs(result[i - 1].real() - result[i - 2].real()) < tol &&
          std::abs(result[i - 1].imag() + result[i - 2].imag()) < tol) {
        result.erase(result.begin() + i - 1);

        result[i - 2].imag(std::abs(result[i - 2].imag()));
      }
    }
  }

  return result;
}

template <typename T>
size_t nearest_real_or_complex(const std::vector<std::complex<T>> &list,
                               const std::complex<T> &val,
                               bool mustbereal = true) {
  std::vector<std::complex<T>> filtered;
  for (std::complex<T> v : list) {
    if (isreal(v) == mustbereal) {
      filtered.push_back(v);
    }
  }

  if (filtered.empty())
    return std::numeric_limits<size_t>::max();

  size_t minidx = 0;
  T minval = std::abs(val - filtered[0]);
  for (size_t i = 1; i < list.size(); i++) {
    T newminval = std::abs(val - filtered[i]);
    if (newminval < minval) {
      minval = newminval;
      minidx = i;
    }
  }
  return minidx;
}

template <typename T> int countreal(const std::vector<std::complex<T>> &list) {
  int nreal = 0;
  for (std::complex<T> c : list) {
    if (c.imag() == 0)
      nreal++;
  }
  return nreal;
}

template <typename T> T warp_freq(T frequency, T fs) {
  frequency = 2 * frequency / fs;
  fs = 2.0;
  T warped = 2 * fs * tan(M_PI * frequency / fs);
  return warped;
}

template <typename T> zpk<T> lp2lp_zpk(const zpk<T> &filter, T wo) {
  zpk<T> result;
  // result.z = wo * filter.z;
  // result.p = wo * filter.p;
  result.z = filter.z;
  std::for_each(result.z.begin(), result.z.end(),
                [&wo](auto &element) { return element = wo * element; });
  result.p = filter.p;
  std::for_each(result.p.begin(), result.p.end(),
                [&wo](auto &element) { return element = wo * element; });
  result.k = filter.k * pow(wo, filter.p.size() - filter.z.size());
  return result;
}

template <typename T> std::vector<std::vector<T>> to_sos(const zpk<T> &filter) {
  if (filter.p.empty() && filter.z.empty())
    return {{filter.k, 0., 0., 1., 0., 0}};

  zpk<T> filt = filter;
  size_t length = std::max(filter.p.size(), filter.z.size());
  filt.p.resize(length, std::complex<T>(0));
  filt.z.resize(length, std::complex<T>(0));

  size_t n_sections = (length + 1) / 2;
  if (length & 1) {
    filt.z.push_back(std::complex<T>(0));
    filt.p.push_back(std::complex<T>(0));
  }

  filt.z = cplxreal(filt.z);
  filt.p = cplxreal(filt.p);

  std::vector<zero_pole_pairs<T>> pairs(n_sections);

  for (size_t si = 0; si < n_sections; si++) {
    size_t worstidx = 0;
    T worstval = std::abs(1.0 - std::abs(filt.p[0]));

    for (size_t i = 1; i < filt.p.size(); i++) {
      T val = std::abs(1 - std::abs(filt.p[i]));

      if (val < worstval) {
        worstidx = i;
        worstval = val;
      }
    }

    std::complex<T> p1 = filt.p[worstidx];

    filt.p.erase(filt.p.begin() + worstidx);

    std::complex<T> z1, p2, z2;
    if (isreal(p1) && countreal(filt.p) == 0) {
      size_t z1_idx = nearest_real_or_complex(filt.z, p1, true);
      z1 = filt.z[z1_idx];
      filt.z.erase(filt.z.begin() + z1_idx);
      p2 = z2 = 0;
    } else {
      size_t z1_idx;
      if (!isreal(p1) && countreal(filt.z) == 1) {
        z1_idx = nearest_real_or_complex(filt.z, p1, false);
      } else {
        size_t minidx = 0;
        T minval = std::abs(p1 - filt.z[0]);
        for (size_t i = 1; i < filt.z.size(); i++) {
          T newminval = std::abs(p1 - filt.z[i]);
          if (newminval < minval) {
            minidx = i;
            minval = newminval;
          }
        }
        z1_idx = minidx;
      }
      z1 = filt.z[z1_idx];
      filt.z.erase(filt.z.begin() + z1_idx);
      if (!isreal(p1)) {
        if (!isreal(z1)) {
          p2 = std::conj(p1);
          z2 = std::conj(z1);
        } else {
          p2 = std::conj(p1);
          size_t z2_idx = nearest_real_or_complex(filt.z, p1, true);
          z2 = filt.z[z2_idx];
          // TESTO_ASSERT(isreal(z2));
          filt.z.erase(filt.z.begin() + z2_idx);
        }
      } else {
        size_t p2_idx;
        size_t z2_idx;
        if (!isreal(z1)) {
          z2 = std::conj(z1);
          p2_idx = nearest_real_or_complex(filt.z, p1, true);
          p2 = filt.p[p2_idx];
          // TESTO_ASSERT(isreal(p2));
        } else {
          size_t worstidx = 0;
          T worstval = std::abs(std::abs(filt.p[0]) - 1);
          for (size_t i = 1; i < filt.p.size(); i++) {
            T val = std::abs(std::abs(filt.p[i]) - 1);
            if (val < worstval) {
              worstidx = i;
              worstval = val;
            }
          }
          p2_idx = worstidx;
          p2 = filt.p[p2_idx];

          // TESTO_ASSERT(isreal(p2));
          z2_idx = nearest_real_or_complex(filt.z, p2, true);
          z2 = filt.z[z2_idx];
          // TESTO_ASSERT(isreal(z2));
          filt.z.erase(filt.z.begin() + z2_idx);
        }
        filt.p.erase(filt.p.begin() + p2_idx);
      }
    }
    pairs[si].p1 = p1;
    pairs[si].p2 = p2;
    pairs[si].z1 = z1;
    pairs[si].z2 = z2;
  }

  std::vector<std::vector<T>> result(n_sections);
  for (size_t si = 0; si < n_sections; si++) {
    result[si] = zpk2tf(pairs[n_sections - 1 - si], si == 0 ? filt.k : T(1));
  }
  return result;
}
} // namespace detail

} // namespace dap
#endif // FRONTEND_INTERFACES_BUDDY_DAP_DSP_IIRDESIGN
