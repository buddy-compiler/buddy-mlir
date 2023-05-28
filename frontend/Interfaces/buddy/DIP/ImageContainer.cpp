#include "buddy/DIP/ImageContainer.h"
#include "buddy/Core/Container.h"
#include <cassert>

// Image Constructor from Img
Img::Img() : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0) {}

Img::Img(int rows, int cols, int type)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0) {
  create(rows, cols, type);
}

Img::Img(int ndims, const int *sizes, int type)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0) {
  create(ndims, sizes, type);
}

Img::Img(const std::vector<int> &sizes, int type)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0) {
  create(sizes, type);
}

Img::Img(const Img &m)
    : flags(m.flags), dims(m.dims), rows(m.rows), cols(m.cols), data(m.data) {
  // TODO refcount gc
}

Img::~Img() { release(); }

void Img::create(int rows, int _cols, int type) {
  type &= TYPE_MASK;
  if (dims <= 2 && rows == this->rows && cols == this->cols &&
      type() == this->type && this->data)
    return;
  int sz[] = {rows, cols};
  create(2, sz, type);
}

void Img::create(int ndims, const int *sizes, int type) {

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
