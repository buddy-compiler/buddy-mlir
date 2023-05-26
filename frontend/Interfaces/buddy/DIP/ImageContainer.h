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
  /*
    These are various constructors that form a matrix. As noted in the AutomaticAllocation, often
    the default constructor is enough, and the proper matrix will be allocated by an OpenCV function.
    The constructed matrix can further be assigned to another matrix or matrix expression or can be
    allocated with Mat::create . In the former case, the old content is de-referenced.
  */
  Img();
  /*
    @rows:rows Number of rows in a 2D array.
    @cols:cols Number of columns in a 2D array.
    @type:type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
  */
  Img(int rows, int cols, int type);
  /*
    @ndims:ndims Array dimensionality.
    @sizes：sizes Array of integers specifying an n-dimensional array shape.
    @type:type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
  */
  Img(int ndims, const int *sizes, int type);
  /*
    @m:m Array that (as a whole or partly) is assigned to the constructed matrix. No data is copied by these constructors. 
    If you want to have an independent copy of the sub-array, use Mat::clone() 
  */
  Img(const Img &m);
  
  //destructor - calls release()
  ~Img();
  
  /*
    @m：m Assigned, right-hand-side matrix. Matrix assignment is an O(1) operation. This means that
    no data is copied but the data is shared and the reference counter, if any, is incremented. Before
    assigning new data, the old data is de-referenced via Mat::release . 
  */
  Img &operator=(const Img &m);
  /*
    Allocates new array data if needed.
    @rows:New number of rows.
    @cols:New number of columns.
    @type:New matrix type.
  */
  void create(int rows, int cols, int type);
  /*
    @mdims: ndims New array dimensionality.
    @sizes: sizes Array of integers specifying a new array shape.
    @type: type New matrix type.
  */
  void create(int ndims, const int *sizes, int type);
  /*
    @sizes: sizes Array of integers specifying a new array shape.
    @type: type New matrix type.
  */
  void create(const std::vector<int> &sizes, int type);
  
  // The method increments the reference counter associated with the matrix data
  // TODO gc : void addref();
  // The method decrements the reference counter associated with the matrix data
  // TODO gc : void release();

  //  deallocates the matrix data
  void deallocate();
  
  /*
      Creates a full copy of the array and the underlying data.
  */
  Img clone() const;

  void static bool load(const String &filename, int flags, Img &img);
  static bool save(const String &filename);

  int depth() const;
  int channels() const;
  int cols() const;
  int rows() const;
  int type() const;
  bool empty() const;

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
  //! the number of rows and columns or (-1, -1) when the matrix has more than 2 dimensions
  int rows, cols;
  //! pointer to the data
  T *data;
};
#endif // FRONTEND_INTERFACES_BUDDY_DIP_IMAGECONTAINER
