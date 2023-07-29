#ifndef _LOADSAVE_HPP_
#define _LOADSAVE_HPP_

#include"buddy/DIP/ImageContainer.h"
#include"buddy/DIP/imgcodecs/Replenishment.hpp"
#include"buddy/DIP/imgcodecs/grfmt_bmp.hpp"

template <typename T, size_t N>
Img<T, N> imread(const String &filename, int flags) {
  //定义一个bmpDecoder类
  BmpDecoder<uchar, 2> bmp;
  //定义图像是否缩放
  int scale_denom = 1;
  bmp.setScale(scale_denom);
  //设置图像路径
  bmp.setSource(filename);
  //读取图像头
  bmp.readHeader();
  _Size size(bmp.width(), bmp.height());
  // grab the decoded type
  int type = bmp.type();
  if ((flags & IMREAD_ANYDEPTH) == 0) {
    type = CV_MAKETYPE(CV_8U, CV_MAT_CN(type));
  }
  if ((flags & IMREAD_COLOR) != 0 ||
      ((flags & IMREAD_ANYCOLOR) != 0 && CV_MAT_CN(type) > 1)) {
    type = CV_MAKETYPE(CV_MAT_DEPTH(type), 3);
  } else {
    type = CV_MAKETYPE(CV_MAT_DEPTH(type), 1);
  }
  //创建一个Img类
  Img<uchar, 2> Image;
  Image.create(size.width, size.height, type);
  //读取图像数据
  bmp.readData(Image);
  return Image;
}













#endif

