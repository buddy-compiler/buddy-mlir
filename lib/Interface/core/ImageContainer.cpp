//===- ImageContainer.cpp -------------------------------------------------===//
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
// This file implements the image container descriptor.
//
//===----------------------------------------------------------------------===//

#ifndef CORE_IMAGE_CONTAINER_DEF
#define CORE_IMAGE_CONTAINER_DEF

#include <cassert>

#include "Interface/buddy/core/Container.h"
#include "Interface/buddy/core/ImageContainer.h"
#include "Interface/buddy/core/Mat.h"

// Image Constructor from file path.
template <typename T, size_t N>
Img<T, N>::Img(const std::string &file_path) : MemRef<T, N>() {
  Mat image(file_path);
  if (image._channel == 1) {
    assert((N == 2) &&
           "Input image type does not match the selected dimension.");
    this->sizes[0] = image._height;
    this->sizes[1] = image._width;
    this->size = image._height * image._width;
    this->allocated = new T[this->size];
    this->aligned = this->allocated;
    int k = 0;
    for (int i = 0; i < image._height; i++) {
      for (int j = 0; j < image._width; j++) {
        if (typeid(T) == typeid(uint8_t)) {
          this->aligned[k] = image._data.u8 + i * image._width + j;
        } else if (typeid(T) == typeid(int8_t)) {
          this->aligned[k] = image._data.u8 + i * image._width + j;
        } else if (typeid(T) == typeid(uint16_t)) {
          this->aligned[k] = image._data.u16 + i * image._width + j;
        } else if (typeid(T) == typeid(int16_t)) {
          this->aligned[k] = image._data.i16 + i * image._width + j;
        } else if (typeid(T) == typeid(uint32_t)) {
          this->aligned[k] = image._data.u32 + i * image._width + j;
        } else if (typeid(T) == typeid(int32_t)) {
          this->aligned[k] = image._data.i32 + i * image._width + j;
        } else if (typeid(T) == typeid(float)) {
          this->aligned[k] = image._data.f32 + i * image._width + j;
        } else if (typeid(T) == typeid(double)) {
          this->aligned[k] = image._data.f64 + i * image._width + j;
        }
        k++;
      }
    }
    this->setStrides();
  }
}

// Image Constructor from OpenCV Mat.
template <typename T, size_t N> Img<T, N>::Img(cv::Mat image) : MemRef<T, N>() {
  if (image.channels() == 1) {
    assert((N == 2) &&
           "Input image type does not match the selected dimension.");
    this->sizes[0] = image.rows;
    this->sizes[1] = image.cols;
    this->size = image.rows * image.cols;
    this->allocated = new T[this->size];
    this->aligned = this->allocated;
    int k = 0;
    for (int i = 0; i < image.rows; i++) {
      for (int j = 0; j < image.cols; j++) {
        this->aligned[k] = (T)image.at<uchar>(i, j);
        k++;
      }
    }
    this->setStrides();
  } else {
    // TODO: Add more image channels in this constructor.
    assert((N != 2) && "This image channels is not supported.");
  }
}

#endif // CORE_IMAGE_CONTAINER_DEF
