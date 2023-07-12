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

#include "Container.h"
#include <cassert>
#include <opencv2/opencv.hpp>
// Image container.
// - T represents the type of the elements.
// - N represents the number of dimensions.
// - image represents the OpenCV Mat object.
// - norm indicates whether to perform normalization, and the normalization is
//   disabled by default.
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
  /*
    @ndims:ndims Array dimensionality.
    @sizes：sizes Array of integers specifying an n-dimensional array shape.
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
    @m：m Assigned, right-hand-side matrix. Matrix assignment is an O(1)
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

  Img(cv::Mat image, intptr_t sizes[N] = nullptr, bool norm = false);

  // Move constructor.
  Img(Img<T, N> &&m);
  // Move assignment operator.
  Img &operator=(const Img<T, N> &&other);

  void create(int rows, int cols, int type);
  /*
    @mdims: ndims New array dimensionality.
    @sizes: sizes Array of integers specifying a new array shape.
    @type: type New matrix type.
  */
  void create(int ndims, const int *sizes, int _type);

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
  int type() const;
  bool empty() const;
  size_t total();
  T *_getData();
  // The template methods return a reference to the specified array element.
  // param row Index along the dimension 0
  // param col Index along the dimension 1
  template <typename _Tp> _Tp &at(int row, int col);
  // param i0 Index along the dimension 0
  // param i1 Index along the dimension 1
  // param i2 Index along the dimension 2
  template <typename _Tp> _Tp &at(int i0, int i1, int i2);

  int flags;
  //! the matrix dimensionality, >= 2
  int dims;
  //! the number of rows and columns or (-1, -1) when the matrix has more than 2
  //! dimensions
  int rows, cols;
  // store the size of each dimension.
  size_t *_size;

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
  @sizes：sizes Array of integers specifying an n-dimensional array shape.
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
    : MemRef<T, N>(), flags(m.flags), dims(m.dims), rows(m.rows), cols(m.cols) {
  if (m._size == nullptr) {
    return;
  } else {
    this->_size = new size_t[m.dims];
    for (size_t i = 0; i < N; i++) {
      this->_size[i] = m._size[i];
    }
  }
  this->size = total();
  this->allocated = new T[this->size];
  this->aligned = this->allocated;
  this->data = this->allocated;
  for (size_t i = 0; i < this->size; i++) {
    this->aligned[i] = m.aligned[i];
  }
}

// destructor - calls release()
// template <typename T, size_t N> Img<T, N>::~Img(); // release()
/*
  Allocates new array data if needed.
  @rows:New number of rows.
  @cols:New number of columns.
  @type:New matrix type.
*/
template <typename T, size_t N>
void Img<T, N>::create(int rows, int cols, int type) {
  // type &= TYPE_MASK;
  // if (dims <= 2 && rows == this->rows && cols == this->cols &&
  //    type() == this->type && this->data)
  //  return;
  this->cols = cols;
  this->rows = rows;
  int sz[] = {rows, cols};
  create(2, sz, type);
}
/*
  @mdims: ndims New array dimensionality.
  @sizes: sizes Array of integers specifying a new array shape.
  @type: type New matrix type.
*/
template <typename T, size_t N>
void Img<T, N>::create(int ndims, const int *sizes, int _type) {
  int i;
  if ((ndims == dims || (ndims == 1 && ndims <= 2))) {
    if (dims == 1 && (ndims == 1 && sizes[0] == _size[0]))
      return;
    if (ndims == 2 && rows == sizes[0] && cols == sizes[1])
      return;
    for (i = 0; i < ndims; i++)
      if (sizes[i] != _size[i])
        break;
    if (i == ndims && (ndims > 1 || _size[1] == 1))
      return;
  }
  this->dims = ndims;
  this->_size = new size_t[ndims];
  this->size = total();
  for (int i = 0; i < ndims; i++) {
    this->_size[i] = sizes[i];
  }
  if (total() > 0) {
    this->allocated = new T[total()];
    this->aligned = this->allocated;
    this->data = this->allocated;
  }
}

/*
@m：m Assigned, right-hand-side matrix. Matrix assignment is an O(1)
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
  if (this->dims <= 2 && m.dims <= 2) {
    this->dims = m.dims;
    this->rows = m.rows;
    this->cols = m.cols;
    this->_size = m._size;
    this->size = total();
  } else {
  }
  // Allocate new space and deep copy.
  T *ptr = new T[this->size];
  for (size_t i = 0; i < this->size; i++) {
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
    : flags(m.flags), dims(m.dims), rows(m.rows), cols(m.cols) {
  this->aligned = m.aligned;
  this->allocated = m.allocated;
  this->data = m.data;
  this->size = m.size;
  std::swap(this->_size, m._size);
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
template <typename T, size_t N>
Img<T, N> &Img<T, N>::operator=(const Img<T, N> &&m) {
  if (this != &m) {
    // Free the original aligned and allocated space.
    delete[] this->allocated;
    // Steal members of the original object.
    std::swap(this->_sizes, m._sizes);
    std::swap(this->size, m.size);
    std::swap(this->allocated, m.allocated);
    std::swap(this->aligned, m.aligned);
    std::swap(this->data, m.data);
    // Assign the NULL pointer to the original aligned and allocated members to
    // avoid the double free error.
    m.allocated = m.aligned = m.data = nullptr;
  }
}

// Image Constructor from OpenCV Mat.
template <typename T, size_t N>
Img<T, N>::Img(cv::Mat image, intptr_t sizes[N], bool norm) : MemRef<T, N>() {
  if (image.channels() == 1) {
    assert((N == 2) && "For gray images, the number of dimensions must be 2.");
  } else if (image.channels() == 3) {
    assert((N == 4) && "For RGB images, the number of dimensions must be 4, "
                       "either in NHWC or NCHW layout.");
  } else {
    std::cerr << "Only 2-channel gray images and 3-channel RGB images are "
                 "supported, but got images' channel equal to "
              << image.channels() << "." << std::endl;
  }
  // Use default layout setting.
  if (sizes == nullptr) {
    // The size of the gray image is represented by height and width by default.
    if (N == 2) {
      this->sizes[0] = image.rows;
      this->sizes[1] = image.cols;
      this->rows = image.rows;
      this->cols = image.cols;
      this->dims = N;
      this->_size = new size_t[N];
      for (int i = 0; i < N; i++) {
        this->_size[i] = this->sizes[i];
      }
    }
    // For RGB images, use NHWC layout by default.
    else if (N == 4) {
      this->sizes[0] = 1;
      this->sizes[1] = image.rows;
      this->sizes[2] = image.cols;
      this->sizes[3] = 3;
    }
  } else {
    // Use custom layout setting.
    for (size_t i = 0; i < N; i++) {
      this->sizes[i] = sizes[i];
    }
  }
  this->size = this->product(this->sizes);
  this->setStrides();
  this->allocated = new T[this->size];
  this->aligned = this->allocated;

  if (N == 2) {
    size_t k = 0;
    for (int i = 0; i < this->sizes[0]; i++) {
      for (int j = 0; j < this->sizes[1]; j++) {
        if (norm) {
          this->aligned[k] = (T)image.at<uchar>(i, j) / 255;
        } else {
          this->aligned[k] = (T)image.at<uchar>(i, j);
        }
        k++;
      }
    }
  } else {
    std::cerr << "RGB images must be arranged in either NHWC or NCHW layout."
              << std::endl;
  }
}

template <typename T, std::size_t N> int Img<T, N>::channels() const {
  return dims <= 2 ? 1 : 3;
}
//template <typename T, size_t N> int Img<T, N>::depth() const {}

template <typename T, size_t N> int Img<T, N>::_rows() const {
  return this->rows;
}
template <typename T, size_t N> int Img<T, N>::_cols() const {
  return this->cols;
}

template <typename T, size_t N> size_t Img<T, N>::total() {
  if (dims <= 2) {
    return (size_t)rows * cols;
  }
  size_t p = 1;
  for (int i = 0; i < dims; i++)
    p *= this->_size[i];
  return p;
}

#endif // FRONTEND_INTERFACES_BUDDY_DIP_IMAGECONTAINER
