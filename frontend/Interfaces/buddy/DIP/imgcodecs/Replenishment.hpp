
#ifndef REPLEISHMENT
#define REPLEISHMENT

#include <cassert>
#include <memory> // std::shared_ptr
#include <string>
#include <type_traits> // std::enable_if
using namespace std;
typedef unsigned long ulong;
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef string String;



#define CV_CN_MAX 512
#define CV_CN_SHIFT 3
#define CV_DEPTH_MAX (1 << CV_CN_SHIFT)

#define CV_8U 0
#define CV_8S 1
#define CV_16U 2
#define CV_16S 3
#define CV_32S 4
#define CV_32F 5
#define CV_64F 6
#define CV_16F 7

#define CV_MAT_DEPTH_MASK (CV_DEPTH_MAX - 1)
#define CV_MAT_DEPTH(flags) ((flags)&CV_MAT_DEPTH_MASK)

#define CV_MAKETYPE(depth, cn) (CV_MAT_DEPTH(depth) + (((cn)-1) << CV_CN_SHIFT))
#define CV_MAKE_TYPE CV_MAKETYPE

#define CV_8UC1 CV_MAKETYPE(CV_8U, 1)
#define CV_8UC2 CV_MAKETYPE(CV_8U, 2)
#define CV_8UC3 CV_MAKETYPE(CV_8U, 3)
#define CV_8UC4 CV_MAKETYPE(CV_8U, 4)
#define CV_8UC(n) CV_MAKETYPE(CV_8U, (n))

#define CV_8SC1 CV_MAKETYPE(CV_8S, 1)
#define CV_8SC2 CV_MAKETYPE(CV_8S, 2)
#define CV_8SC3 CV_MAKETYPE(CV_8S, 3)
#define CV_8SC4 CV_MAKETYPE(CV_8S, 4)
#define CV_8SC(n) CV_MAKETYPE(CV_8S, (n))

#define CV_16UC1 CV_MAKETYPE(CV_16U, 1)
#define CV_16UC2 CV_MAKETYPE(CV_16U, 2)
#define CV_16UC3 CV_MAKETYPE(CV_16U, 3)
#define CV_16UC4 CV_MAKETYPE(CV_16U, 4)
#define CV_16UC(n) CV_MAKETYPE(CV_16U, (n))

#define CV_16SC1 CV_MAKETYPE(CV_16S, 1)
#define CV_16SC2 CV_MAKETYPE(CV_16S, 2)
#define CV_16SC3 CV_MAKETYPE(CV_16S, 3)
#define CV_16SC4 CV_MAKETYPE(CV_16S, 4)
#define CV_16SC(n) CV_MAKETYPE(CV_16S, (n))

#define CV_32SC1 CV_MAKETYPE(CV_32S, 1)
#define CV_32SC2 CV_MAKETYPE(CV_32S, 2)
#define CV_32SC3 CV_MAKETYPE(CV_32S, 3)
#define CV_32SC4 CV_MAKETYPE(CV_32S, 4)
#define CV_32SC(n) CV_MAKETYPE(CV_32S, (n))

#define CV_32FC1 CV_MAKETYPE(CV_32F, 1)
#define CV_32FC2 CV_MAKETYPE(CV_32F, 2)
#define CV_32FC3 CV_MAKETYPE(CV_32F, 3)
#define CV_32FC4 CV_MAKETYPE(CV_32F, 4)
#define CV_32FC(n) CV_MAKETYPE(CV_32F, (n))

#define CV_64FC1 CV_MAKETYPE(CV_64F, 1)
#define CV_64FC2 CV_MAKETYPE(CV_64F, 2)
#define CV_64FC3 CV_MAKETYPE(CV_64F, 3)
#define CV_64FC4 CV_MAKETYPE(CV_64F, 4)
#define CV_64FC(n) CV_MAKETYPE(CV_64F, (n))

#define CV_16FC1 CV_MAKETYPE(CV_16F, 1)
#define CV_16FC2 CV_MAKETYPE(CV_16F, 2)
#define CV_16FC3 CV_MAKETYPE(CV_16F, 3)
#define CV_16FC4 CV_MAKETYPE(CV_16F, 4)
#define CV_16FC(n) CV_MAKETYPE(CV_16F, (n))


class _Size {
public:
  _Size(){};
  _Size(int _width, int _height) : width(_width), height(_height) {}
  inline _Size &operator=(const _Size &rhs) {
    this->width = rhs.width;
    this->height = rhs.height;
    return *this;
  }
  _Size &operator+=(const _Size &rhs) {
    width += rhs.width;
    height += rhs.height;
    return *this;
  }
  bool operator==(const _Size &rhs) {
    return width == rhs.width && height == rhs.height;
  }
  bool operator!=(const _Size &rhs) { return !(*this == rhs); }
  int width = 0;
  int height = 0;
};

#endif // Repleishment_HPP