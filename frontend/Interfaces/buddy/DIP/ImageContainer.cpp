#include "buddy/DIP/ImageContainer.h"
#include "buddy/Core/Container.h"
#include <cassert>

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
Img<T, N>::Img()
    : MemRef<T, N>(), flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0) {}

/*
  @rows:rows Number of rows in a 2D array.
  @cols:cols Number of columns in a 2D array.
  @type:type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel
  matrices, or CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to
  CV_CN_MAX channels) matrices.
*/
template <typename T, size_t N>
Img<T, N>::Img(int rows, int cols, int type)
    : MemRef<T, N>(), flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0) {
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
    : MemRef<T, N>(), flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0) {
  create(ndims, sizes, type);
}
/*
  @m:m Array that (as a whole or partly) is assigned to the constructed
  matrix. No data is copied by these constructors. If you want to have an
  independent copy of the sub-array, use Mat::clone()
*/
template <typename T, size_t N>
Img<T, N>::Img(const Img &m)
    : MemRef<T, N>(), flags(m.flags), dims(m.dims), rows(m.rows), cols(m.cols),
      data(m.data) {
  // TODO refcount gc
}

// destructor - calls release()
template <typename T, size_t N> Img<T, N>::~Img() { release(); }
/*
  Allocates new array data if needed.
  @rows:New number of rows.
  @cols:New number of columns.
  @type:New matrix type.
*/
template <typename T, size_t N>
void Img<T, N>::create(int rows, int _cols, int type) {
  type &= TYPE_MASK;
  if (dims <= 2 && rows == this->rows && cols == this->cols &&
      type() == this->type && this->data)
    return;
  int sz[] = {rows, cols};
  create(2, sz, type);
}
/*
  @mdims: ndims New array dimensionality.
  @sizes: sizes Array of integers specifying a new array shape.
  @type: type New matrix type.
*/
template <typename T, size_t N>
void Img<T, N>::create(int ndims, const int *sizes, int type) {
  int i;
  // CV_MAX_DIM是一个宏。它表示维度的最大值。
  assert(0 <= ndims && ndims <= MAX_DIM && sizes);
  // CV_MAT_TYPE是一个宏。它用于将数据类型编码为整数值。每个整数值与一种特定的数据类型对应，例如，CV_8UC1表示8位无符号单通道图像，CV_32FC3表示32位浮点型三通道图像等。

  type = CV_MAT_TYPE(type);

  // CV_ELEM_SIZE是一个宏定义，用于计算指定数据类型的每个元素所占的字节数。
  int s = CV_ELEM_SIZE(type);

  // Compute the size and allocate memory for the matrix
  int total = rows * cols * s;
  if (data && (d == dims || (d == 1 && dims <= 2)) && _type == type()) {
    if (dims == 1 && (d == 1 && sizes[0] == rows))
      return;
    if (d == 2 && rows == sizes[0] && cols == sizes[1])
      return;
    if (i == d && (d > 1 || cols == 1))
      return;
  }
  if (d == 0)
    return;
  // CV_MAT_TYPE_MASK是OpenCV中的一个宏定义，用于提取矩阵类型的掩码。掩码是指在二进制编码中，用特定位数的二进制表示来标识某个属性或状态的方式。CV_MAT_TYPE_MASK提取矩阵类型掩码，可以用于判断矩阵的数据类型（例如8位、16位、32位浮点、64位浮点等）。
  flags = (type & CV_MAT_TYPE_MASK) | MAGIC_VAL;
  this->rows = rows;
  this->cols = cols;

  // Allocate memory for the matrix data
  allocated = new T[total];
  data = allocated;
  aligned = allocated;
  std::fill(aligned, aligned + size, init);
}