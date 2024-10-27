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
#include "buddy/DIP/imgcodecs/replenishment.h"
#include <cassert>

#ifdef BUDDY_ENABLE_OPENCV
#include <opencv2/opencv.hpp>
#endif

using namespace dip;
// Image container.
// - T represents the type of the elements.
// - N represents the number of dimensions.
template <typename T, size_t N> class Img : public MemRef<T, N> {
public:
  Img(){};

  /**
   * @brief overload
   * @param sizes Array of integers specifying an n-dimensional array shape.
   * @param data Pointer to the user data.
   * they just initialize the matrix header that points to the specified data.
   */
  Img(T *data, intptr_t sizes[N]);

  /**
   * @brief overload
   * @param sizes Array of integers specifying an n-dimensional array shape.
   */
  Img(intptr_t sizes[N]);

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

  // Move constructor.
  Img(Img<T, N> &&m);

  // Move assignment operator.
  Img &operator=(Img<T, N> &&m);

#ifdef BUDDY_ENABLE_OPENCV
  /**
   * @brief Load image data from OpenCV Mat.
   * @param image represents the OpenCV Mat object.
   * @param norm indicates whether to perform.
   */
  Img(cv::Mat image, intptr_t sizes[N] = nullptr, bool norm = false);
#endif

  int channels();
};

/**
 * @brief overload
 * @param sizes Array of integers specifying an n-dimensional array shape.
 */
template <typename T, size_t N>
Img<T, N>::Img(intptr_t sizes[N]) : MemRef<T, N>(sizes) {}

/**
 * @brief overload
 * @param m Array that (as a whole or partly) is assigned to the constructed
 * matrix.
 */
template <typename T, size_t N>
Img<T, N>::Img(const Img<T, N> &m) : MemRef<T, N>(m) {}

// Move Constructor.
// This constructor is used to initialize a MemRef object from a rvalue.
// The move constructor steals the resources of the original object.
// Note that the original object no longer owns the members and spaces.
// Steal members from the original object.
// Assign the NULL pointer to the original aligned and allocated members to
// avoid the double free error.
template <typename T, size_t N>
Img<T, N>::Img(Img<T, N> &&m) : MemRef<T, N>(m) {}

// Move Assignment Operator.
// Note that the original object no longer owns the members and spaces.
// Check if they are the same object.
// Free the data space of this object to avoid memory leaks.
// Steal members from the original object.
// Assign the NULL pointer to the original aligned and allocated members to
// avoid the double free error.
template <typename T, size_t N> Img<T, N> &Img<T, N>::operator=(Img<T, N> &&m) {
  MemRef<T, N>::operator=(m);
}

/**
 * @brief assignment operators
 * @param m Assigned, right-hand-side matrix.
 * matrix.
 */
template <typename T, size_t N>
Img<T, N> &Img<T, N>::operator=(const Img<T, N> &m) {
  MemRef<T, N>::operator=(m);
  return *this;
}

/**
 * @brief overload
 * @param sizes Array of integers specifying an n-dimensional array shape.
 * @param data Pointer to the user data.
 * they just initialize the matrix header that points to the specified data.
 */
template <typename T, size_t N>
Img<T, N>::Img(T *data, intptr_t sizes[N]) : MemRef<T, N>(data, sizes) {}

#ifdef BUDDY_ENABLE_OPENCV
// Image Constructor from OpenCV Mat.

template <typename T, size_t N>
Img<T, N>::Img(cv::Mat image, intptr_t sizes[N], bool norm) : MemRef<T, N>() {
  if (image.channels() == 1) {
    assert((N == 2 || N == 4) &&
           "For gray images, the number of dimensions can be 2 or 4.");
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
    this->setStrides();
    this->allocated = new T[this->product(this->sizes)];
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
  } else {
    // Use custom layout setting.
    // Only support gray scale image and NCHW now.
    for (size_t i = 0; i < N; i++) {
      this->sizes[i] = sizes[i];
    }
    size_t size = this->product(this->sizes);
    this->setStrides();
    this->allocated = new T[size];
    this->aligned = this->allocated;
    size_t k = 0;
    //NCHW Layout
    for (int batch = 0; batch < this->sizes[0]; batch++) {
      for (int channel = 0; channel < this->sizes[1]; channel++) {
        T *chandata = image.ptr<T>(batch, channel);
        for (int row = 0; row < this->sizes[2]; row++) {
          for (int col = 0; col < this->sizes[3]; col++) {
            if (norm) {
              this->aligned[k] = chandata[row * this->sizes[3] + col] / 255;
            } else {
              this->aligned[k] = chandata[row * this->sizes[3] + col];
            }
            k++;
          }
        }
      }
    }
  }
}

#endif

template <typename T, size_t N> int Img<T, N>::channels() {
  if (N == 2) {
    return 1;
  }
  return this->sizes[2];
}

#endif // FRONTEND_INTERFACES_BUDDY_DIP_IMAGECONTAINER
