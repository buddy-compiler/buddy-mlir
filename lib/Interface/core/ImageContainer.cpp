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
Img<T, N>::Img(cv::Mat image, bool norm) : MemRef<T, N>() {
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
  }
  // Use NHWC layout by default.
  else if (image.channels() == 3) {
    assert((N == 4) &&
           "Input image type does not match the selected dimension.");
    this->sizes[0] = 1;
    this->sizes[1] = image.rows;
    this->sizes[2] = image.cols;
    this->sizes[3] = 3;
    this->size = image.rows * image.cols * 3;
    this->allocated = new T[this->size];
    this->aligned = this->allocated;
    int k = 0;
    for (int i = 0; i < image.rows; i++) {
      for (int j = 0; j < image.cols; j++) {
        for (int color = 0; color < 3; color++) {
          // Reorder to RGB layout.
          if (norm) {
            this->aligned[k] = (T)image.at<cv::Vec3b>(i, j)[2 - color] / 255;
          } else {
            this->aligned[k] = (T)image.at<cv::Vec3b>(i, j)[2 - color];
          }
          k++;
        }
      }
    }
    this->setStrides();
  } else {
    // TODO: Add more image channels in this constructor.
    std::cerr << "This image channels is not supported." << std::endl;
  }
}

#endif // CORE_IMAGE_CONTAINER_DEF
