#ifndef _LOADSAVE_HPP_
#define _LOADSAVE_HPP_

#include"buddy/DIP/ImageContainer.h"
#include"buddy/DIP/imgcodecs/Replenishment.hpp"
#include"buddy/DIP/imgcodecs/grfmt_bmp.hpp"

template <typename T, size_t N>
Img<T, N> imread(const String &filename, int flags) {
  //定义一个bmpDecoder类
  BmpDecoder<uchar, 2> bmp;
  int scale_denom = 2;
  bmp.setScale(scale_denom);
  //设置图像路径
  bmp.setSource(filename);
  //读取图像头
  bmp.readHeader();
  _Size size(bmp.width(), bmp.height());
  // TODO type的定义
  //创建一个Img类
  Img<uchar, 2> Image;
  Image.create(size.width, size.height, 1);
  //读取图像数据
  bmp.readData(Image);
  return Image;
}












#endif

