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

#ifndef FRONTEND_INTERFACES_BUDDY_CORE_CONTAINER
#define FRONTEND_INTERFACES_BUDDY_CORE_CONTAINER

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

// MemRef descriptor.
// - T represents the type of the elements.
// - N represents the number of dimensions.
// - The storage order is NCHW.
template <typename T, size_t N> class MemRef {
public:
  // Constructor from shape.
  MemRef(intptr_t sizes[N]);
  MemRef(std::vector<size_t> sizes);
  MemRef(intptr_t sizes[N], T init);
  MemRef(intptr_t sizes[N], bool needMalloc, intptr_t offset);
  MemRef(std::vector<size_t> sizes, T init);
  MemRef(std::vector<size_t> sizes, bool needMalloc, intptr_t offset);
  // Constructor from data.
  MemRef(const T *data, intptr_t sizes[N], intptr_t offset = 0);
  MemRef(const T *data, std::vector<size_t> sizes, intptr_t offset = 0);
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
  // Concat two MemRefs into a MemRef.
  void concatenateMemRefs(MemRef<T, N> &other0, MemRef<T, N> &other1, MemRef<T, N> &other2, size_t concatDim);
  // Split a MemRef into two MemRefs.
  void splitMemRef(MemRef<T, N> &&other0, MemRef<T, N> &other1, MemRef<T, N> &other2, size_t splitDim, size_t splitIndex);
  // Add two MemRef
  void addMemRef(MemRef<T, N>& a, MemRef<T, N>& b);
  // Get the data pointer.
  T *getData();
  // Get the data.
  std::vector<T> getDataVector();
  // Get the sizes (shape).
  const intptr_t *getSizes() { return sizes; }
  // Get the strides.
  const intptr_t *getStrides() { return strides; }
  // Get the rank of the memref.
  size_t getRank() const { return N; }
  // Get the size (number of elements).
  size_t getSize() const { return product(this->sizes); }
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
  size_t product(const intptr_t sizes[N]) const;

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
};

// MemRef Shape Constructor.
// Construct a MemRef object from the data shape and (optional)initial value.
template <typename T, std::size_t N> MemRef<T, N>::MemRef(intptr_t sizes[N]) {
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = sizes[i];
  }
  setStrides();
  size_t size = product(sizes);
  allocated = (T *)malloc(sizeof(T) * size);
  aligned = allocated;
}

template <typename T, std::size_t N>
MemRef<T, N>::MemRef(std::vector<size_t> sizes) {
  if (sizes.size() != N) {
    throw std::runtime_error("Invalid number of dimensions.");
  }
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = sizes[i];
  }
  setStrides();
  size_t size = product(this->sizes);
  allocated = (T *)malloc(sizeof(T) * size);
  aligned = allocated;
}

template <typename T, std::size_t N>
MemRef<T, N>::MemRef(intptr_t sizes[N], T init) : MemRef(sizes) {
  size_t size = product(sizes);
  std::fill(aligned, aligned + size, init);
}

template <typename T, std::size_t N>
MemRef<T, N>::MemRef(intptr_t sizes[N], bool needMalloc, intptr_t offset)
    : allocated(nullptr), aligned(nullptr), offset(offset) {
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = sizes[i];
  }
  setStrides();
  if (needMalloc) {
    size_t size = product(sizes);
    allocated = (T *)malloc(sizeof(T) * size);
  }
}

template <typename T, std::size_t N>
MemRef<T, N>::MemRef(std::vector<size_t> sizes, T init) : MemRef(sizes) {
  size_t size = product(this->sizes);
  std::fill(aligned, aligned + size, init);
}

template <typename T, std::size_t N>
MemRef<T, N>::MemRef(std::vector<size_t> sizes, bool needMalloc,
                     intptr_t offset)
    : allocated(nullptr), aligned(nullptr), offset(offset) {
  if (sizes.size() != N) {
    throw std::runtime_error("Invalid number of dimensions.");
  }
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = sizes[i];
  }
  setStrides();
  if (needMalloc) {
    size_t size = product(this->sizes);
    allocated = (T *)malloc(sizeof(T) * size);
  }
}

// MemRef Array Constructor.
// Construct a MemRef object from the data pointer, sizes, and offset.
// The default offset is 0.
template <typename T, std::size_t N>
MemRef<T, N>::MemRef(const T *data, intptr_t sizes[N], intptr_t offset) {
  this->offset = offset;
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = sizes[i];
  }
  setStrides();
  size_t size = product(sizes);
  allocated = (T *)malloc(sizeof(T) * size);
  aligned = allocated;
  for (size_t i = 0; i < size; i++) {
    aligned[i] = data[i];
  }
}

template <typename T, std::size_t N>
MemRef<T, N>::MemRef(const T *data, std::vector<size_t> sizes, intptr_t offset) {
  if (sizes.size() != N) {
    throw std::runtime_error("Invalid number of dimensions.");
  }
  this->offset = offset;
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = sizes[i];
  }
  setStrides();
  size_t size = product(this->sizes);
  allocated = (T *)malloc(sizeof(T) * size);
  aligned = allocated;
  for (size_t i = 0; i < size; i++) {
    aligned[i] = data[i];
  }
}

// Copy Constructor.
// This constructor is used to initialize a MemRef object with another MemRef
// object.
// - Copy `offset` and `size` directly.
// - Elementwise copy `sizes` array.
// - Calculate `strides`.
// - Allocate new space.
// - Deep copy the data from the original object.
template <typename T, std::size_t N>
MemRef<T, N>::MemRef(const MemRef<T, N> &other) : offset(other.offset) {
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = other.sizes[i];
  }
  setStrides();
  size_t size = product(this->sizes);
  allocated = (T *)malloc(sizeof(T) * size);
  aligned = allocated;
  for (size_t i = 0; i < size; i++) {
    aligned[i] = other.aligned[i];
  }
}

// Copy Assignment Operator.
// - Check if they are the same object.
// - Copy `offset` and `size` directly.
// - Elementwise copy `sizes`.
// - Calculate the `strides`.
// - Free the data space of this object to avoid memory leaks.
// - Allocate new space and deep copy.
template <typename T, std::size_t N>
MemRef<T, N> &MemRef<T, N>::operator=(const MemRef<T, N> &other) {
  if (this != &other) {
    this->offset = other.offset;
    for (size_t i = 0; i < N; i++) {
      this->sizes[i] = other.sizes[i];
    }
    setStrides();
    // Free the original aligned and allocated space.
    free(allocated);
    // Allocate new space and deep copy.
    size_t size = product(this->sizes);
    T *ptr = (T *)malloc(sizeof(T) * size);
    for (size_t i = 0; i < size; i++) {
      ptr[i] = other.aligned[i];
    }
    aligned = ptr;
    allocated = ptr;
  }
  return *this;
}

// Move Constructor.
// This constructor is used to initialize a MemRef object from a rvalue.
// The move constructor steals the resources of the original object.
// Note that the original object no longer owns the members and spaces.
// - Steal members from the original object.
// - Assign the NULL pointer to the original aligned and allocated members to
//   avoid the double free error.
template <typename T, std::size_t N>
MemRef<T, N>::MemRef(MemRef<T, N> &&other) noexcept
    : allocated(other.allocated), aligned(other.aligned), offset(other.offset) {
  std::swap(this->sizes, other.sizes);
  std::swap(this->strides, other.strides);
  // Assign the NULL pointer to the original aligned and allocated members to
  // avoid the double free error.
  other.allocated = other.aligned = nullptr;
}

// Move Assignment Operator.
// Note that the original object no longer owns the members and spaces.
// - Check if they are the same object.
// - Free the data space of this object to avoid memory leaks.
// - Steal members from the original object.
// - Assign the NULL pointer to the original aligned and allocated members to
//   avoid the double free error.
template <typename T, std::size_t N>
MemRef<T, N> &MemRef<T, N>::operator=(MemRef<T, N> &&other) noexcept {
  if (this != &other) {
    // Free the original aligned and allocated space.
    free(allocated);
    // Steal members of the original object.
    std::swap(strides, other.strides);
    std::swap(offset, other.offset);
    std::swap(sizes, other.sizes);
    std::swap(allocated, other.allocated);
    std::swap(aligned, other.aligned);
    // Assign the NULL pointer to the original aligned and allocated members to
    // avoid the double free error.
    other.allocated = other.aligned = nullptr;
  }
  return *this;
}

// MemRef Destructor.
// Note that the `allocated` and `aligned` point to the same address, so it is
// enough to release the space of the `allocated` pointer in the destructor.
template <typename T, std::size_t N> MemRef<T, N>::~MemRef() {
  if (allocated)
    free(allocated);
}

// Get the data pointer.
// Return the `aligned` pointer if the container data size is greater than zero.
// If the data size is negative or zero, which means no space is allocated for
// the container data pointer, the function does not allow to return the data
// pointer.
template <typename T, std::size_t N> T *MemRef<T, N>::getData() {
  size_t size = product(this->sizes);
  assert((size > 0) && "Invalid container data size.");
  return aligned;
}

// Get the element at index.
// Return the specific element if the container data size is greater than zero.
// If the data size is negative or zero, which means no space is allocated for
// the container data pointer, this operator does not allow to return the data
// element.
template <typename T, std::size_t N>
const T &MemRef<T, N>::operator[](size_t index) const {
  size_t size = product(this->sizes);
  assert((size > 0) && "Invalid container data size.");
  return aligned[index + offset];
}
template <typename T, std::size_t N> T &MemRef<T, N>::operator[](size_t index) {
  size_t size = product(this->sizes);
  assert((size > 0) && "Invalid container data size.");
  return aligned[index + offset];
}

// Calculate the stride values for each dimension based on the sizes.
template <typename T, std::size_t N> void MemRef<T, N>::setStrides() {
  assert((N > 0) && "Invalid container number of dims");
  strides[N - 1] = 1;
  if (N < 2)
    return;
  // Prevent implicit conversions between unsigned and signed
  for (std::size_t i = N - 1; i > 0; i--) {
    strides[i - 1] = strides[i] * sizes[i];
  }
}

// Calculate the total number of elements in the MemRef container.
template <typename T, std::size_t N>
size_t MemRef<T, N>::product(const intptr_t sizes[N]) const {
  size_t size = 1;
  for (size_t i = 0; i < N; i++)
    size *= sizes[i];
  return size;
}

template <typename T, size_t N>
MemRef<T, N>::MemRef(std::unique_ptr<T> &uptr, intptr_t *sizes,
                     intptr_t offset) {
  if (!uptr)
    assert(0 && "Taking over an empty unique pointer.");
  T *data = uptr.release();
  this->aligned = data;
  this->allocated = data;
  this->offset = offset;
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = sizes[i];
  }
  setStrides();
}

template <typename T, size_t N> T *MemRef<T, N>::release() {
  T *temp = allocated;
  aligned = nullptr;
  allocated = nullptr;
  return temp;
}

template <typename T, std::size_t N>
void MemRef<T, N>::concatenateMemRefs(MemRef<T, N> &other0, MemRef<T, N> &other1,
                                      MemRef<T, N> &other2, size_t concatDim) {
  if (concatDim >= N) {
    throw std::runtime_error("Invalid concatenation dimension.");
  }

  for (size_t i = 0; i < N; i++) {
    if (i != concatDim && other0.getSizes()[i] != other1.getSizes()[i]) {
      throw std::runtime_error("Shapes are not compatible for concatenation.");
    }
  }

  intptr_t *sizes2 = const_cast<intptr_t *>(other2.getSizes());
  for (size_t i = 0; i < N; i++) {
    if (i == concatDim) {
      sizes2[i] = other0.getSizes()[i] + other1.getSizes()[i];
    } else {
      sizes2[i] = other0.getSizes()[i];
    }
  }

  other2.setStrides();

  intptr_t concatOffset0 = other0.getStrides()[concatDim] * other0.getSizes()[concatDim];
  intptr_t concatOffset1 = other1.getStrides()[concatDim] * other1.getSizes()[concatDim];

  size_t offset0 = 0;
  size_t offset1 = 0;
  size_t offset2 = 0;
  while (offset2 < other2.getSize()) {
    size_t tmp = 0;
    while (tmp < concatOffset0) {
      other2[offset2 + tmp] = other0[offset0 + tmp];
      tmp++;
    }
    offset0 += concatOffset0;
    offset2 += concatOffset0;
    tmp = 0;
    while (tmp < concatOffset1) {
      other2[offset2 + tmp] = other1[offset1 + tmp];
      tmp++;
    }
    offset1 += concatOffset1;
    offset2 += concatOffset1;
  }
}

template <typename T, std::size_t N>
void MemRef<T, N>::splitMemRef(MemRef<T, N> &&other0, MemRef<T, N> &other1,
                 MemRef<T, N> &other2, size_t splitDim, size_t splitIndex) {
  if (splitDim >= N) {
    throw std::runtime_error("Invalid split dimension.");
  }

  if (splitIndex > static_cast<size_t>(other0.getSizes()[splitDim])) {
    throw std::runtime_error("Split index out of bounds.");
  }

  intptr_t *sizes1 = const_cast<intptr_t *>(other1.getSizes());
  const intptr_t *sizes0 = other0.getSizes();
  for (size_t i = 0; i < N; i++) {
    sizes1[i] = sizes0[i];
  }
  sizes1[splitDim] = splitIndex;

  intptr_t *sizes2 = const_cast<intptr_t *>(other2.getSizes());
  for (size_t i = 0; i < N; i++) {
    sizes2[i] = sizes0[i];
  }
  sizes2[splitDim] = sizes0[splitDim] - splitIndex;

  other1.setStrides();
  other2.setStrides();

  other1.allocated = other0.allocated;
  other1.aligned = other0.aligned;
  other1.offset = other0.offset;

  other2.allocated = nullptr;
  other2.aligned = other0.aligned + other1.getSize();
  other2.offset = other0.offset;

  other0.allocated = nullptr;
  other0.aligned = nullptr;
}

/// Add two MemRef
template <typename T, size_t N>
void MemRef<T, N>::addMemRef(MemRef<T, N>& a, MemRef<T, N>& b) {
  const intptr_t * aSizes = a.getSizes();
  const intptr_t * bSizes = b.getSizes();
  for (size_t i = 0; i < N; i++) {
    if (aSizes[i] != bSizes[i]) {
      throw std::runtime_error("Shapes are not compatible for concatenation.");
    }
  }

  for (size_t i = 0; i < a.getSize(); i++) {
    a[i] += b[i];
  }
}

/// Get the data.
template <typename T, std::size_t N>
std::vector<T> MemRef<T, N>::getDataVector() {
  size_t size = product(this->sizes);
  assert((size > 0) && "Invalid container data size.");
  std::vector<T> dataVector(size);
  for (size_t i = 0; i < size; i++) {
    dataVector[i] = aligned[i + offset];
  }
  return dataVector;
}

#endif // FRONTEND_INTERFACES_BUDDY_CORE_CONTAINER
