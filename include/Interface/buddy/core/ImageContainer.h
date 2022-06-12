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

#ifndef INTERFACE_BUDDY_CORE_IMAGECONTAINER
#define INTERFACE_BUDDY_CORE_IMAGECONTAINER

#include "Interface/buddy/core/Container.h"
#include <iostream>
#include <opencv2/opencv.hpp>

// Image container.
// - T represents the type of the elements.
// - N represents the number of dimensions.
template <typename T, size_t N> class Img : public MemRef<T, N> {
public:
  Img(const std::string &file_path); // Default image decoder.
  Img(cv::Mat image);                // OpenCV data stracture.
};

#include "Interface/core/ImageContainer.cpp"

#endif // INTERFACE_BUDDY_CORE_IMAGECONTAINER
