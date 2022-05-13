//===- Container.cpp ------------------------------------------------------===//
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
// This file implements the container descriptor.
//
//===----------------------------------------------------------------------===//

#ifndef CORE_CONTAINER_DEF
#define CORE_CONTAINER_DEF

#include <algorithm>
#include <memory>
#include <numeric>
#include <stdexcept>

#include "Interface/buddy/core/Container.h"

template <typename T, std::size_t N>
MemRef<T, N>::MemRef(intptr_t sizes[N], T init) {
  static_assert(N >= 1 && N <= 4, "MemRef size not supported.");

  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = sizes[i];
  }
  setStrides();
  size = product(sizes);
  T *data = new T[size];
  aligned = data;
  allocated = data;
  std::fill(data, data + size, init);
}

template <typename T, std::size_t N>
MemRef<T, N>::MemRef(const MemRef<T, N> &other)
    : offset(other.offset), size(other.size) {
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = other.sizes[i];
  }
  setStrides();
  T *ptr = new T[size];
  for (size_t i = 0; i < size; i++) {
    ptr[i] = other.aligned[i];
  }
  aligned = ptr;
  allocated = ptr;
}

// Overloading assignment operator.
// - Check if they are the same object.
// - Copy `offset` and `size` directly.
// - Elementwise copy `sizes`.
// - Calculate the `strides`.
// - Free the original data space to avoid memory leaks.
// - Allocate new space and deep copy.
template <typename T, std::size_t N>
MemRef<T, N> &MemRef<T, N>::operator=(const MemRef<T, N> &other) {
  if (this != &other) {
    this->offset = other.offset;
    this->size = other.size;
    for (size_t i = 0; i < N; i++) {
      this->sizes[i] = other.sizes[i];
    }
    setStrides();
    // Free the original aligned and allocated space.
    delete[] allocated;
    // Allocate new space and deep copy.
    T *ptr = new T[size];
    for (size_t i = 0; i < size; i++) {
      ptr[i] = other.aligned[i];
    }
    aligned = ptr;
    allocated = ptr;
  }
  return *this;
}

template <typename T, std::size_t N>
MemRef<T, N>::MemRef(MemRef<T, N> &&other) noexcept
    : allocated(other.allocated), aligned(other.aligned), offset(other.offset),
      size(other.size) {
  std::swap(this->sizes, other.sizes);
  std::swap(this->strides, other.strides);
  other.allocated = other.aligned = nullptr;
}

template <typename T, std::size_t N>
MemRef<T, N> &MemRef<T, N>::operator=(MemRef<T, N> &&other) noexcept {
  if (this != &other) {
    std::swap(strides, other.strides);
    std::swap(offset, other.offset);
    std::swap(sizes, other.sizes);
    std::swap(size, other.size);
    std::swap(allocated, other.allocated);
    std::swap(aligned, other.aligned);
    other.allocated = other.aligned = nullptr;
  }
  return *this;
}

template <typename T, std::size_t N> MemRef<T, N>::~MemRef() {
  delete[] allocated;
}

template <typename T, std::size_t N> void MemRef<T, N>::setStrides() {
  strides[N - 1] = 1;
  for (int i = N - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * sizes[i + 1];
  }
}

template <typename T, std::size_t N>
size_t MemRef<T, N>::product(intptr_t sizes[N]) const {
  size_t size = 1;
  for (size_t i = 0; i < N; i++)
    size *= sizes[i];
  return size;
}

#endif // CORE_CONTAINER_DEF
