#ifndef _LOADSAVE_HPP_
#define _LOADSAVE_HPP_

#include"buddy/DIP/ImageContainer.h"
#include"buddy/DIP/imgcodecs/Replenishment.hpp"
#include"buddy/DIP/imgcodecs/grfmt_bmp.hpp"

template <typename T, size_t N>
Img<T, N> imread(const String& filename, int flags) {
  
  //����һ��bmpDecoder��
  BmpDecoder<uchar, 2> bmp;
  int scale_denom = 2;
  bmp.setScale(scale_denom);
  //����ͼ��·��
  // bmp.setSource("TestGrayImage_8.bmp");//8λ�Ҷ�ͼ
  // bmp.setSource("TestGrayImage_24.bmp"); // 24λRGBͼ��
  bmp.setSource(filename);
  //��ȡͼ��ͷ
  bmp.readHeader();
  _Size size(bmp.width(), bmp.height());
  // TODO type�Ķ���
  //����һ��Img��
  Img<uchar, 2> Image;
  Image.create(size.width, size.height, 1);
  //��ȡͼ������
  bmp.readData(Image);
  return Image;
}












#endif

