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
#include <cassert>

// Image container.
// - T represents the type of the elements.
// - N represents the number of dimensions.
template <typename T, size_t N> class Img : public MemRef<T, N> {
public:
  Img();

  Img(int rows, int cols, int type);

  Img(int rows, int cols, int type, T *get_data);

  Img(int ndims, const int *sizes, int type);

  Img(const Img<T, N> &m);

  Img &operator=(const Img<T, N> &m);

  Img(cv::Mat image, intptr_t sizes[N] = nullptr, bool norm = false);

  Img(Img<T, N> &&m);

  Img &operator=(Img<T, N> &&other);
}
#endif // FRONTEND_INTERFACES_BUDDY_DIP_IMAGECONTAINER
