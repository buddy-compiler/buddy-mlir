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
#include <opencv2/opencv.hpp>

using namespace dip;
// Image container.
// - T represents the type of the elements.
// - N represents the number of dimensions.
template <typename T, size_t N> class Img : public MemRef<T, N> {
public:
  Img();

  /**
   * @brief overload
   * @param rows Number of rows in a 2D array.
   * @param cols Number of columns in a 2D array.
   * @param type Array type. Use IMG_8UC1, ..., IMG_64FC4 to create 1-4 channel
   * matrices.
   */
  Img(int rows, int cols, int type);

  /**
   * @brief overload
   * @param rows Number of rows in a 2D array.
   * @param cols Number of columns in a 2D array.
   * @param type Array type. Use IMG_8UC1, ..., IMG_64FC4 to create 1-4 channel
   * matrices.
   * @param data Pointer to the user data.
   * they just initialize the matrix header that points to the specified data.
   */
  Img(int rows, int cols, int type, T *data);

  /**
   * @brief overload
   * @param ndims Array dimensionality.
   * @param sizes Array of integers specifying an n-dimensional array shape.
   * @param type Use IMG_8UC1, ..., IMG_64FC4 to create 1-4 channel
   * matrices.
   */
  Img(int ndims, intptr_t *sizes, int type);

  /**
   * @brief overload
   * @param m Array that (as a whole or partly) is assigned to the constructed
   * matrix.
   */
  Img(const Img<T, N> &m);

  /**
   * @brief assignment operators
   * @param m Assigned, right-hand-side matrix.
   * matrix.
   */
  Img &operator=(const Img<T, N> &m);

  /**
   * @brief Load image data from OpenCV Mat.
   * @param image represents the OpenCV Mat object.
   * @param norm indicates whether to perform.
   */
  Img(cv::Mat image, intptr_t sizes[N] = nullptr, bool norm = false);

  /**
   * @brief Allocates new array data if needed.
   * @param rows Number of rows in a 2D array.
   * @param cols Number of columns in a 2D array.
   * @param type Array type. Use IMG_8UC1, ..., IMG_64FC4 to create 1-4 channel
   * matrices.
   */
  void create(int rows, int cols, int type);

  /**
   * @brief overload
   * @param ndims Array dimensionality.
   * @param sizes Array of integers specifying an n-dimensional array shape.
   * @param type Use IMG_8UC1, ..., IMG_64FC4 to create 1-4 channel
   * matrices.
   */
  void create(int ndims, intptr_t *sizes, int type);

  int channels() const;

  int _cols() const;

  int _rows() const;

  size_t total();

  int flags;

  // the matrix dimensionality, >= 2
  int dims;

  // the number of rows and columns or (-1, -1) when the matrix has more than 2
  // dimensions
  int rows, cols;

  // Use IMG_8UC1, ..., IMG_64FC4 to create 1-4 channel matrices.
  int _type;

  // Used to assign addresses to image data
  // A Memref::aligned member is a protected member that is not directly
  // accessible from other classes or functions
  T *_data;
};

// Image Constructor from Img.
template <typename T, size_t N>
Img<T, N>::Img() : MemRef<T, N>(), flags(0), dims(0), rows(0), cols(0) {}

/**
 * @brief overload
 * @param rows Number of rows in a 2D array.
 * @param cols Number of columns in a 2D array.
 * @param type Array type. Use IMG_8UC1, ..., IMG_64FC4 to create 1-4 channel
 * matrices.
 */
template <typename T, size_t N>
Img<T, N>::Img(int rows, int cols, int type)
    : MemRef<T, N>(), flags(0), dims(0), rows(0), cols(0) {
  create(rows, cols, type);
}

/**
 * @brief overload
 * @param ndims Array dimensionality.
 * @param sizes Array of integers specifying an n-dimensional array shape.
 * @param type Use IMG_8UC1, ..., IMG_64FC4 to create 1-4 channel
 * matrices.
 */
template <typename T, size_t N>
Img<T, N>::Img(int ndims, intptr_t *sizes, int type)
    : MemRef<T, N>(), flags(0), dims(0), rows(0), cols(0) {
  create(ndims, sizes, type);
}

/**
 * @brief overload
 * @param m Array that (as a whole or partly) is assigned to the constructed
 * matrix.
 */
template <typename T, size_t N>
Img<T, N>::Img(const Img<T, N> &m)
    : MemRef<T, N>(), flags(m.flags), dims(m.dims), rows(m.rows), cols(m.cols),
      _type(m._type) {
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = m.sizes[i];
    this->strides[i] = m.strides[i];
  }
  this->size = m.size;
  this->allocated = new T[this->size];
  this->aligned = this->allocated;
  this->_data = this->allocated;
  for (size_t i = 0; i < this->size; i++) {
    this->aligned[i] = m.aligned[i];
  }
}

/**
 * @brief Allocates new array data if needed.
 * @param rows Number of rows in a 2D array.
 * @param cols Number of columns in a 2D array.
 * @param type Array type. Use IMG_8UC1, ..., IMG_64FC4 to create 1-4 channel
 * matrices.
 */
template <typename T, size_t N>
void Img<T, N>::create(int rows, int cols, int type) {
  this->_type = type;
  this->cols = cols;
  this->rows = rows;
  this->sizes[0] = cols;
  this->sizes[1] = rows;
  if (N <= 2) {
    create(2, this->sizes, _type);
  }
}

/**
 * @brief overload
 * @param ndims Array dimensionality.
 * @param sizes Array of integers specifying an n-dimensional array shape.
 * @param type Use IMG_8UC1, ..., IMG_64FC4 to create 1-4 channel
 * matrices.
 */
template <typename T, size_t N>
void Img<T, N>::create(int ndims, intptr_t *sizes, int type) {
  this->dims = ndims;
  this->setStrides();
  this->size = total();
  if (total() > 0) {
    this->allocated = new T[total()];
    this->aligned = this->allocated;
    this->_data = this->allocated;
  }
}

/**
 * @brief assignment operators
 * @param m Assigned, right-hand-side matrix.
 * matrix.
 */
template <typename T, size_t N>
Img<T, N> &Img<T, N>::operator=(const Img<T, N> &m) {
  if (this == &m) {
    return *this;
  } else {
    this->flags = m.flags;
    this->_type = m._type;
    this->dims = m.dims;
    this->rows = m.rows;
    this->cols = m.cols;
    for (int i = 0; i < this->dims; i++) {
      this->sizes[i] = m.sizes[i];
      this->strides[i] = m.strides[i];
    }
    this->size = total();
    // Allocate new space and deep copy.
    T *ptr = new T[total()];
    for (size_t i = 0; i < total(); i++) {
      ptr[i] = m.aligned[i];
    }
    this->allocated = ptr;
    this->aligned = ptr;
    this->_data = ptr;
  }
  return *this;
}

/**
 * @brief overload
 * @param rows Number of rows in a 2D array.
 * @param cols Number of columns in a 2D array.
 * @param type Array type. Use IMG_8UC1, ..., IMG_64FC4 to create 1-4 channel
 * matrices.
 * @param data Pointer to the user data.
 * they just initialize the matrix header that points to the specified data.
 */
template <typename T, size_t N>
Img<T, N>::Img(int rows, int cols, int type, T *data)
    : MemRef<T, N>(), dims(2), rows(rows), cols(cols), _type(type) {
  this->aligned = data;
  this->_data = data;
  this->sizes[0] = rows;
  this->sizes[1] = cols;
  this->size = total();
  this->setStrides();
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
    }
    this->size = this->product(this->sizes);
    this->setStrides();
    this->allocated = new T[this->size];
    this->aligned = this->allocated;
    // Load gray image data from OpenCV Mat.
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
    }
  }
}

template <typename T, size_t N> int Img<T, N>::channels() const {
  return IMG_MAT_CN(_type);
}

template <typename T, size_t N> int Img<T, N>::_rows() const {
  return this->rows;
}

template <typename T, size_t N> int Img<T, N>::_cols() const {
  return this->cols;
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
#endif // FRONTEND_INTERFACES_BUDDY_DIP_IMAGECONTAINER
