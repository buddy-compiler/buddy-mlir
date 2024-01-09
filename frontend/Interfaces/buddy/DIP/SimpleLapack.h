//====- SimpleLapack.h ---------------------------------------------------===//
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
// This file implements simple linear algebra functions for image
// processing.
//
//===----------------------------------------------------------------------===//
#ifndef INCLUDE_UTILS_SIMPLELAPACK_H
#define INCLUDE_UTILS_SIMPLELAPACK_H
#include <assert.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace buddy {
namespace dip {
template <typename T>
std::tuple<bool, std::vector<std::vector<T>>, std::vector<std::size_t>>
luDecomposition(std::vector<std::vector<T>> mat) {
  auto n = mat.size();
  std::vector<std::size_t> P(n); // 置换矩阵, P[i] = j, 即 Pij = 1
  for (size_t i = 0; i < n; i++) {
    P[i] = i;
  }
  for (size_t k = 0; k < n; k++) {
    auto max = std::numeric_limits<T>::min();
    size_t maxIndex = k;
    for (size_t i = k; i < n; i++) {
      if (std::abs(mat[i][k]) > max) {
        max = std::abs(mat[i][k]);
        maxIndex = i;
      }
    }
    if (max < std::numeric_limits<T>::epsilon()) {
      return {false, {}, {}};
    }
    std::swap(P[k], P[maxIndex]);
    std::swap(mat[k], mat[maxIndex]);
    for (size_t i = k + 1; i < n; i++) {
      mat[i][k] /= mat[k][k];
      for (size_t j = k + 1; j < n; j++) {
        mat[i][j] -= mat[i][k] * mat[k][j];
      }
    }
  }
  return {true, mat, P};
}

template <typename T>
std::vector<T> forwardSubstitution(const std::vector<std::vector<T>> &lower,
                                   const std::vector<T> &b) {
  auto n = lower.size();
  std::vector<T> y(n, 0);
  for (size_t i = 0; i < n; i++) {
    T sum = 0;
    for (size_t j = 0; j < i; j++) {
      sum += lower[i][j] * y[j];
    }
    y[i] = b[i] - sum;
  }
  return y;
}

template <typename T>
std::vector<T> backwardSubstitution(const std::vector<std::vector<T>> &upper,
                                    const std::vector<T> &y) {
  auto n = upper.size();
  std::vector<T> x(n, 0);
  for (int i = n - 1; i >= 0; i--) {
    T sum = 0;
    for (size_t j = i + 1; j < n; j++) {
      sum += upper[i][j] * x[j];
    }
    x[i] = (y[i] - sum) / upper[i][i];
  }
  return x;
}

template <typename T>
std::tuple<bool, std::vector<T>> luSolve(const std::vector<std::vector<T>> &mat,
                                         std::vector<T> b) {
  auto [successful, LU, P] = luDecomposition<T>(mat);
  if (!successful)
    return {false, {}};
  // Pb
  std::vector<T> tmp(b.size());
  for (size_t i = 0; i < b.size(); i++) {
    tmp[i] = b[P[i]];
  }
  auto y = forwardSubstitution(LU, tmp);
  auto x = backwardSubstitution(LU, y);
  return {true, x};
}

template <typename T>
std::tuple<bool, std::vector<std::vector<T>>>
invert(const std::vector<std::vector<T>> &mat) {
  auto n = mat.size();
  std::vector<std::vector<T>> inv(n, std::vector<T>(n, 0));
  auto [successful, LU, P] = luDecomposition<T>(mat);
  if (!successful)
    return {false, {}};
  for (size_t i = 0; i < n; i++) {
    std::vector<T> b(n, 0);
    b[i] = 1;
    std::vector<T> tmp(b.size());
    for (size_t i = 0; i < b.size(); i++) {
      tmp[i] = b[P[i]];
    }
    auto y = forwardSubstitution(LU, tmp);
    auto x = backwardSubstitution(LU, y);
    for (size_t j = 0; j < n; j++) {
      inv[j][i] = x[j];
    }
  }
  return {true, inv};
}

template <typename T>
std::vector<std::vector<T>> operator*(const std::vector<std::vector<T>> &a,
                                      const std::vector<std::vector<T>> &b) {
  auto n = a.size();
  auto m = b.size();
  auto p = b[0].size();
  std::vector<std::vector<T>> c(n, std::vector<T>(p, 0));
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < p; j++) {
      for (size_t k = 0; k < m; k++) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return c;
}

template <typename T>
std::vector<std::vector<T>> getModelMatrix(T x, T y, T z, T t_x, T t_y, T t_z) {
  std::vector<std::vector<T>> xRotation = {{1, 0, 0, 0},
                                           {0, cos(x), -sin(x), 0},
                                           {0, sin(x), cos(x), 0},
                                           {0, 0, 0, 1}};
  std::vector<std::vector<T>> yRotation = {{cos(y), 0, sin(y), 0},
                                           {0, 1, 0, 0},
                                           {-sin(y), 0, cos(y), 0},
                                           {0, 0, 0, 1}};
  std::vector<std::vector<T>> zRotation = {{cos(z), -sin(z), 0, 0},
                                           {sin(z), cos(z), 0, 0},
                                           {0, 0, 1, 0},
                                           {0, 0, 0, 1}};
  std::vector<std::vector<T>> translationMatrix = {
      {1, 0, 0, -t_x}, {0, 1, 0, -t_y}, {0, 0, 1, -t_z}, {0, 0, 0, 1}};
  std::vector<std::vector<T>> translation_matrix_ivt = {
      {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, t_z}, {0, 0, 0, 1}};
  return translation_matrix_ivt * zRotation * yRotation * xRotation *
         translationMatrix;
}
template <typename T>
std::vector<std::vector<T>> getViewportMatrix(T width, T height) {
  return {{width / 2, 0, 0, width / 2},
          {0, height / 2, 0, height / 2},
          {0, 0, 1, 0},
          {0, 0, 0, 1}};
}

template <typename T>
std::vector<std::vector<T>> getCameraMatrix(T eyeX, T eyeY, T eyeZ) {
  return {{1, 0, 0, -eyeX}, {0, 1, 0, -eyeY}, {0, 0, 1, -eyeZ}, {0, 0, 0, 1}};
}

template <typename T>
std::vector<std::vector<T>> getProjectionMatrix(T fov, T aspectRatio, T zNear,
                                                T zFar) {
  auto n = -zNear;
  auto f = -zFar;
  auto t = std::abs(n) * tan(fov / 2);
  auto b = -t;
  auto r = t * aspectRatio;
  auto l = -r;

  std::vector<std::vector<T>> perspToOrtho = {
      {n, 0, 0, 0}, {0, n, 0, 0}, {0, 0, n + f, -n * f}, {0, 0, 1, 0}};
  std::vector<std::vector<T>> ortho = {{2 / (r - l), 0, 0, -(r + l) / (r - l)},
                                       {0, 2 / (t - b), 0, -(t + b) / (t - b)},
                                       {0, 0, 2 / (n - f), -(n + f) / (n - f)},
                                       {0, 0, 0, 1}};
  return ortho * perspToOrtho;
}

template <class T>
std::vector<std::vector<T>>
getPerspectiveTransform(const std::vector<std::pair<intptr_t, intptr_t>> &src,
                        const std::vector<std::pair<intptr_t, intptr_t>> &dst) {
  assert(src.size() == dst.size() && "The number of points must be equal.");
  assert(src.size() >= 4 && "The number of points must be greater than 4.");
  // stack all points
  std::vector<std::vector<T>> A;
  std::vector<T> b;
  b.reserve(8);
  for (intptr_t i = 0; i < src.size(); i++) {
    T x = src[i].first;
    T y = src[i].second;
    T u = dst[i].first;
    T v = dst[i].second;
    A.push_back({x, y, 1, 0, 0, 0, -x * u, -y * u});
    A.push_back({0, 0, 0, x, y, 1, -x * v, -y * v});
    b.push_back(u);
    b.push_back(v);
  }
  auto [successful, x] = luSolve(A, b);
  assert(successful && "Failed to solve the linear equations.");
  std::vector<std::vector<T>> h = {
      {x[0], x[1], x[2]}, {x[3], x[4], x[5]}, {x[6], x[7], 1}};
  return h;
}
} // namespace dip
} // namespace buddy

#endif