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
    if (image.channels() == 1) {
      this->sizes[0] = image.rows;
      this->sizes[1] = image.cols;
    }
    // For RGB images, use NHWC layout by default.
    else if (image.channels() == 3) {
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
  this->loadImg(image, norm);
}

template <typename T, size_t N>
void Img<T, N>::loadImg(cv::Mat image, bool norm) {
  if (image.channels() == 1) {
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
  } else if (image.channels() == 3) {
    // Detect NHWC layout.
    if (this->sizes[1] == image.rows && this->sizes[2] == image.cols &&
        this->sizes[3] == 3) {
      size_t k = 0;
      for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
          for (int color = 0; color < 3; color++) {
            if (norm) {
              this->aligned[k] = (T)image.at<cv::Vec3b>(i, j)[2 - color] / 255;
            } else {
              this->aligned[k] = (T)image.at<cv::Vec3b>(i, j)[2 - color];
            }
            k++;
          }
        }
      }
    }
    // Detect NCHW layout.
    else if (this->sizes[2] == image.rows && this->sizes[3] == image.cols &&
             this->sizes[1] == 3) {
      size_t k = 0;
      for (int color = 0; color < 3; color++) {
        for (int i = 0; i < image.rows; i++) {
          for (int j = 0; j < image.cols; j++) {
            if (norm) {
              this->aligned[k] = (T)image.at<cv::Vec3b>(i, j)[2 - color] / 255;
            } else {
              this->aligned[k] = (T)image.at<cv::Vec3b>(i, j)[2 - color];
            }
            k++;
          }
        }
      }
    } else {
      std::cerr << "RGB images must be arranged in either NHWC or NCHW layout."
                << std::endl;
    }
  }
}

#endif // CORE_IMAGE_CONTAINER_DEF
