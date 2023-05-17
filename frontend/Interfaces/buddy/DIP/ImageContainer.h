
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
#include <opencv2/opencv.hpp>

// Image container.
// - T represents the type of the elements.
// - N represents the number of dimensions.
// - image represents the OpenCV Mat object.
// - norm indicates whether to perform normalization, and the normalization is
//   disabled by default.
template <typename T, size_t N> class Img : public MemRef<T, N> {
public:
  Img();
  /*
      @rows:二维数组的行数
      @cols:二维数组的列数
      @type:数组类型 使用CV_8UC1，…， CV_64FC4创建1-4通道矩阵，或CV_8UC (n),……，
     CV_64FC(n)创建多通道(最多CV_CN_MAX通道)矩阵。
  */
  Img(int rows, int cols, int type);
  /*
      @ndims:数组的维度
      @sizes：用来指定n维数组的形状
      @type:数组类型 使用CV_8UC1，…， CV_64FC4创建1-4通道矩阵，或CV_8UC (n),……，
     CV_64FC(n)创建多通道(最多CV_CN_MAX通道)矩阵。
  */
  Img(int ndims, const int *sizes, int type);
  /*
      @sizes:使用vector容器存储n维数组的形状
      @type:数组类型 使用CV_8UC1，…， CV_64FC4创建1-4通道矩阵，或CV_8UC (n),……，
     CV_64FC(n)创建多通道(最多CV_CN_MAX通道)矩阵。
  */
  Img(const std::vector<int> &sizes, int type);
  /*
      @m:赋值构造函数，并不发生拷贝。当用户使用这个构造函数进行构造时，改变该矩阵同时也会改变用来构建该矩阵的源矩阵。
      如果用户想拥有独立的内存空间用来拷贝新矩阵，可以使用Img::clone()
  */
  Img(const Img &m);

  ~Img();

  Img &operator=(const Img &m);

  // 成员函数。

  void create(int rows, int cols, int type);
  void create(int ndims, const int *sizes, int type);
  void create(const std::vector<int> &sizes, int type);
  // TODO gc : void addref();
  void release();
  void deallocate();
  Img clone() const;

  void static bool load(const String &filename, int flags, Img &img);
  static bool save(const String &filename);

  int depth() const;
  int channels() const;
  int cols() const;
  int rows() const;
  int type() const;
  bool empty() const;
  /*
      成员数据。
  */
  int flags;
  //! 矩阵维度, >= 2
  int dims;
  //! 当矩阵有多个维度时，行数及列数或者(-1, -1)
  int rows, cols;
  //! 指向数据的指针
  T *data;
};
#endif // FRONTEND_INTERFACES_BUDDY_DIP_IMAGECONTAINER