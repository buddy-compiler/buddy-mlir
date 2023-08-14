//===- ImageContainer.h ---------------------------------------------------===//
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
// Image container descriptor.
//
//===----------------------------------------------------------------------===//

#ifndef FRONTEND_INTERFACES_BUDDY_DIP_IMAGECONTAINER
#define FRONTEND_INTERFACES_BUDDY_DIP_IMAGECONTAINER

#include "buddy/Core/Container.h"
#include "buddy/DIP/imgcodecs/Replenishment.hpp"
#include <cassert>
//  Image container.
//  - T represents the type of the elements.
//  - N represents the number of dimensions.
//  - image represents the OpenCV Mat object.
//  - norm indicates whether to perform normalization, and the normalization is
//    disabled by default.
template <typename T, size_t N> class Img : public MemRef<T, N> {
public:
  /*
    These are various constructors that form a matrix. As noted in the
    AutomaticAllocation, often the default constructor is enough, and the proper
    matrix will be allocated by an OpenCV function. The constructed matrix can
    further be assigned to another matrix or matrix expression or can be
    allocated with Mat::create . In the former case, the old content is
    de-referenced.
  */
  Img();
  /*
    @rows:rows Number of rows in a 2D array.
    @cols:cols Number of columns in a 2D array.
    @type:type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel
    matrices, or CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to
    CV_CN_MAX channels) matrices.
  */
  Img(int rows, int cols, int type);
  /** @overload
   @param rows Number of rows in a 2D array.
   @param cols Number of columns in a 2D array.
   @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel
    matrices, or CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to
    CV_CN_MAX channels) matrices.
   @param data Pointer to the user data. Matrix constructors that take data and
    step parameters do not allocate matrix data. Instead, they just initialize
   the matrix header that points to the specified data, which means that no data
   is copied. This operation is very efficient and can be used to process
   external data using OpenCV functions. The external data is not automatically
    deallocated, so you should take care of it.*/

  Img(int rows, int cols, int type, T *get_data);
  /*
    @ndims:ndims Array dimensionality.
    @sizes sizes Array of integers specifying an n-dimensional array shape.
    @type:type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel
    matrices, or CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to
    CV_CN_MAX channels) matrices.
  */
  Img(int ndims, const int *sizes, int type);
  /*
    @m:m Array that (as a whole or partly) is assigned to the constructed
    matrix. No data is copied by these constructors. If you want to have an
    independent copy of the sub-array, use Mat::clone()
  */
  Img(const Img<T, N> &m);

  // destructor - calls release()
  //~Img();

  /*
    @m:m Assigned, right-hand-side matrix. Matrix assignment is an O(1)
    operation. This means that no data is copied but the data is shared and the
    reference counter, if any, is incremented. Before assigning new data, the
    old data is de-referenced via Mat::release .
  */
  Img &operator=(const Img<T, N> &m);
  /*
    Allocates new array data if needed.
    @rows:New number of rows.
    @cols:New number of columns.
    @type:New matrix type.
  */

  // Img(cv::Mat image, intptr_t sizes[N] = nullptr, bool norm = false);

  // Move constructor.
  Img(Img<T, N> &&m);
  // Move assignment operator.
  Img &operator=(Img<T, N> &&other);

  void create(int rows, int cols, int type);
  /*
    @mdims: ndims New array dimensionality.
    @sizes: sizes Array of integers specifying a new array shape.
    @type: type New matrix type.
  */
  void create(int ndims, int *sizes, int _type);

  // The method increments the reference counter associated with the matrix data
  // TODO gc : void addref();
  // The method decrements the reference counter associated with the matrix data
  // TODO gc : void release();

  //  deallocates the matrix data
  void deallocate();

  // static bool load(const String &filename, int flags, Img &img);
  // static bool save(const String &filename);

  int depth() const;
  int channels() const;
  int _cols() const;
  int _rows() const;
  bool empty() const;
  size_t total();

  size_t elemsize() const;

  // The template methods return a reference to the specified array element.
  // param row Index along the dimension 0
  // param col Index along the dimension 1
  template <typename _Tp> _Tp at(int row, int col);

  int _type;

  int flags;
  //! the matrix dimensionality, >= 2
  int dims;
  //! the number of rows and columns or (-1, -1) when the matrix has more than 2
  //! dimensions
  int rows, cols;

  int sz[N];

  T *data;
};

// Image Constructor from Img
/*
  These are various constructors that form a matrix. As noted in the
  AutomaticAllocation, often the default constructor is enough, and the proper
  matrix will be allocated by an OpenCV function. The constructed matrix can
  further be assigned to another matrix or matrix expression or can be
  allocated with Mat::create . In the former case, the old content is
  de-referenced.
*/
template <typename T, size_t N>
Img<T, N>::Img() : MemRef<T, N>(), flags(0), dims(0), rows(0), cols(0) {}

/*
  @rows:rows Number of rows in a 2D array.
  @cols:cols Number of columns in a 2D array.
  @type:type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel
  matrices, or CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to
  CV_CN_MAX channels) matrices.
*/
template <typename T, size_t N>
Img<T, N>::Img(int _rows, int _cols, int type)
    : MemRef<T, N>(), flags(0), dims(0), rows(0), cols(0) {
  create(rows, cols, type);
}
/*
  @ndims:ndims Array dimensionality.
  @sizes:sizes Array of integers specifying an n-dimensional array shape.
  @type:type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel
  matrices, or CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to
  CV_CN_MAX channels) matrices.
*/
template <typename T, size_t N>
Img<T, N>::Img(int ndims, const int *sizes, int type)
    : MemRef<T, N>(), flags(0), dims(0), rows(0), cols(0) {
  create(ndims, sizes, type);
}
/*
  @m:m Array that (as a whole or partly) is assigned to the constructed
  matrix. No data is copied by these constructors. If you want to have an
  independent copy of the sub-array, use Mat::clone()
*/
template <typename T, size_t N>
Img<T, N>::Img(const Img<T, N> &m)
    : MemRef<T, N>(), flags(m.flags), dims(m.dims), rows(m.rows), cols(m.cols),
      _type(m._type) {

  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = m.sizes[i];
  }
  this->setStrides();
  this->size = total();
  this->allocated = new T[total()];
  this->aligned = this->allocated;
  this->data = this->allocated;
  for (size_t i = 0; i < this->size; i++) {
    this->aligned[i] = m.aligned[i];
  }
}
/*
  Allocates new array data if needed.
  @rows:New number of rows.
  @cols:New number of columns.
  @type:New matrix type.
*/
template <typename T, size_t N>
void Img<T, N>::create(int rows, int cols, int type) {

  this->_type = type;
  this->cols = cols;
  this->rows = rows;
  this->sizes[0] = cols;
  this->sizes[1] = rows;
  for (int i = 0; i < N; i++) {
    sz[i] = this->sizes[i];
  }
  create(2, sz, _type);
}
/*
  @mdims: ndims New array dimensionality.
  @sizes: sizes Array of integers specifying a new array shape.
  @type: type New matrix type.
*/
template <typename T, size_t N>
void Img<T, N>::create(int ndims, int *sizes, int type) {
  int i;
  this->_type = type;
  this->dims = ndims;
  this->size = total();
  this->setStrides();
  if (total() > 0) {
    this->allocated = new T[total()];
    this->aligned = this->allocated;
    this->data = this->allocated;
  }
}

/*
@m:m Assigned, right-hand-side matrix. Matrix assignment is an O(1)
operation. This means that no data is copied but the data is shared and the
reference counter, if any, is incremented. Before assigning new data, the
old data is de-referenced via Mat::release .
*/
template <typename T, size_t N>
Img<T, N> &Img<T, N>::operator=(const Img<T, N> &m) {
  if (this == &m) {
    return *this;
  }
  this->flags = m.flags;
  this->_type = m._type;
  if (this->dims <= 2 && m.dims <= 2) {
    this->dims = m.dims;
    this->rows = m.rows;
    this->cols = m.cols;
    for (int i = 0; i < this->dims; i++) {
      this->sizes[i] = m.sizes[i];
    }
    this->size = total();
  } else {
  }
  // Allocate new space and deep copy.
  this->setStrides();
  T *ptr = new T[total()];
  for (size_t i = 0; i < total(); i++) {
    ptr[i] = m.aligned[i];
  }
  this->allocated = ptr;
  this->aligned = ptr;
  this->data = ptr;
  return *this;
}

// Move Constructor.
// This constructor is used to initialize a MemRef object from a rvalue.
// The move constructor steals the resources of the original object.
// Note that the original object no longer owns the members and spaces.
// - Steal members from the original object.
// - Assign the NULL pointer to the original aligned and allocated members to
//   avoid the double free error.
template <typename T, size_t N>
Img<T, N>::Img(Img<T, N> &&m)
    : flags(m.flags), dims(m.dims), rows(m.rows), cols(m.cols), _type(m._type) {
  this->allocated = m.allocated;
  this->aligned = m.aligned;
  this->data = m.data;
  this->size = m.size;
  std::swap(this->sizes, m.sizes);
  std::swap(this->strides, m.strides);
  // Assign the NULL pointer to the original aligned and allocated members to
  // avoid the double free error.
  m.allocated = m.aligned = m.data = nullptr;
}

// Move Assignment Operator.
// Note that the original object no longer owns the members and spaces.
// - Check if they are the same object.
// - Free the data space of this object to avoid memory leaks.
// - Steal members from the original object.
// - Assign the NULL pointer to the original aligned and allocated members to
//   avoid the double free error.
template <typename T, size_t N> Img<T, N> &Img<T, N>::operator=(Img<T, N> &&m) {
  if (this != &m) {
    // Free the original aligned and allocated space.
    delete[] this->allocated;
    // Steal members of the original object.
    std::swap(this->flags, m.flags);
    std::swap(this->dims, m.dims);
    std::swap(this->rows, m.rows);
    std::swap(this->cols, m.cols);
    std::swap(this->_type, m._type);
    std::swap(this->size, m.size);
    std::swap(this->allocated, m.allocated);
    std::swap(this->aligned, m.aligned);
    std::swap(this->data, m.data);
    std::swap(this->strides, m.strides);
    std::swap(this->sizes, m.sizes);
    // Assign the NULL pointer to the original aligned and allocated members to
    // avoid the double free error.
    m.allocated = m.aligned = m.data = nullptr;
  }
}
template <typename T, size_t N>
Img<T, N>::Img(int rows, int cols, int type, T *get_data)
    : MemRef<T, N>(), dims(2), rows(rows), cols(cols), _type(type) {

  this->data = get_data;
  this->aligned = get_data;
  this->sizes[0] = rows;
  this->sizes[1] = cols;
  this->size = total();
  this->setStrides();
}

template <typename T, std::size_t N> int Img<T, N>::channels() const {
  return CV_MAT_CN(_type);
}
template <typename T, size_t N> int Img<T, N>::depth() const {}

template <typename T, size_t N> int Img<T, N>::_rows() const {
  return this->rows;
}
template <typename T, size_t N> int Img<T, N>::_cols() const {
  return this->cols;
}

template <typename T, size_t N> size_t Img<T, N>::elemsize() const {
  // return CV_ELEM_SIZE(_type);
  return sizeof(T) * channels();
}

template <typename T, size_t N> size_t Img<T, N>::total() {
  if (dims <= 2) {
    return (size_t)rows * cols * channels();
  }
  size_t p = 1;
  for (int i = 0; i < dims; i++)
    p *= this->sizes[i];
  return p;
}

template <typename T, size_t N>
template <typename _Tp>
_Tp Img<T, N>::at(int row, int col) {
  if (row < 0 || col < 0) {
    throw std::out_of_range("Index out of bounds");
  }

  return this->data + (row * cols * channels() + col * channels());
}

#endif // FRONTEND_INTERFACES_BUDDY_DIP_IMAGECONTAINER
