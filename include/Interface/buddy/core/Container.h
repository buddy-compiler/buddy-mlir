//===- Container.h --------------------------------------------------------===//
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
// Container descriptor.
//
//===----------------------------------------------------------------------===//

#ifndef INTERFACE_BUDDY_CORE_CONTAINER
#define INTERFACE_BUDDY_CORE_CONTAINER

#include <cstdint>
#include <memory>
#include <vector>

// MemRef descriptor.
// - T represents the type of the elements.
// - N represents the number of dimensions.
// - The storage order is NCHW.
template <typename T, size_t N> class MemRef {
public:
  // Constructor from shape.
  MemRef(intptr_t sizes[N], T init = T(0));
  MemRef(std::vector<size_t> sizes, T init = T(0));
  // Constructor from data.
  MemRef(const T *data, intptr_t sizes[N], intptr_t offset = 0);
  // Constructor from a unique_ptr, taking over.
  MemRef(std::unique_ptr<T> &uptr, intptr_t sizes[N], intptr_t offset = 0);
  // Copy constructor.
  MemRef(const MemRef<T, N> &other);
  // Copy assignment operator.
  MemRef<T, N> &operator=(const MemRef<T, N> &other);
  // Move constructor.
  MemRef(MemRef<T, N> &&other) noexcept;
  // Move assignment operator.
  MemRef<T, N> &operator=(MemRef<T, N> &&other) noexcept;
  // Desctrutor.
  ~MemRef();
  // Get the data pointer.
  T *getData();
  // Get the sizes (shape).
  const intptr_t *getSizes() { return sizes; }
  // Get the strides.
  const intptr_t *getStrides() { return strides; }
  // Get the rank of the memref.
  size_t getRank() const { return N; }
  // Get the size (number of elements).
  size_t getSize() const { return size; }
  // Get the element at index.
  const T &operator[](size_t index) const;
  T &operator[](size_t index);
  // release the pointer
  T *release();

protected:
  // Default constructor.
  // This constructor is designed for derived domain-specific constructor.
  MemRef(){};
  // Set the strides.
  // Computes the strides of the transposed tensor for transpose=true.
  void setStrides();
  // Compute the product of array elements.
  size_t product(intptr_t sizes[N]) const;

  // Data.
  // The `aligned` and `allocated` members point to the same address, `aligned`
  // member is responsible for handling data, and `allocated` member is
  // resposible for handling the memory space.
  T *allocated = nullptr;
  T *aligned = nullptr;
  // Offset.
  intptr_t offset = 0;
  // Shape.
  intptr_t sizes[N];
  // Strides.
  intptr_t strides[N];
  // Number of elements.
  size_t size;
};

#include "Interface/core/Container.cpp"

#endif // INTERFACE_BUDDY_CORE_CONTAINER
