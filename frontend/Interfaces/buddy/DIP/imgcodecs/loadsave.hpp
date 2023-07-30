#ifndef _LOADSAVE_HPP_
#define _LOADSAVE_HPP_

#include"buddy/DIP/ImageContainer.h"
#include"buddy/DIP/imgcodecs/Replenishment.hpp"
#include"buddy/DIP/imgcodecs/grfmt_bmp.hpp"

/**
 * @struct ImageCodecInitializer
 *
 * Container which stores the registered codecs to be used by OpenCV
 */

template <typename T, size_t N> struct ImageCodecInitializer {
  /**
   * Default Constructor for the ImageCodecInitializer
   */
  ImageCodecInitializer() {
    /// BMP Support
    decoders.push_back(std::make_unique<BmpDecoder<T, N>>());
    // encoders.push_back(std::make_unique<BmpEncoder<T, N>>());
  }

  std::vector<std::unique_ptr<BaseImageDecoder<T, N>>> decoders;
  // std::vector<std::unique_ptr<BaseImageEncoder<T, N>>> encoders;
};

template <typename T, size_t N>
static ImageCodecInitializer<T, N> &getCodecs() {
#ifdef CV_CXX11
  static ImageCodecInitializer<T,N> g_codecs;
  return g_codecs;
#else
  // C++98 doesn't guarantee correctness of multi-threaded initialization of
  // static global variables (memory leak here is not critical, use C++11 to
  // avoid that)
  static ImageCodecInitializer<T, N> *g_codecs =
      new ImageCodecInitializer<T, N>();
  return *g_codecs;
#endif
}


/**
 * Find the decoders
 *
 * @param[in] filename File to search
 *
 * @return Image decoder to parse image file.
 */
template <typename T, size_t N>
static std::unique_ptr<BaseImageDecoder<T, N>> findDecoder(const String &filename) {

  size_t i, maxlen = 0;

  /// iterate through list of registered codecs
  ImageCodecInitializer<T,N> &codecs = getCodecs<T,N>();
  for (i = 0; i < codecs.decoders.size(); i++) {
    size_t len = codecs.decoders[i]->signatureLength();
    maxlen = std::max(maxlen, len);
  }

  /// Open the file
  FILE *f = fopen(filename.c_str(), "rb");

  /// in the event of a failure, return an empty image decoder
  if (!f) {
    return nullptr;
  }
  // read the file signature
  String signature(maxlen, ' ');
  maxlen = fread((void *)signature.c_str(), 1, maxlen, f);
  fclose(f);
  signature = signature.substr(0, maxlen);

  /// compare signature against all decoders
  for (i = 0; i < codecs.decoders.size(); i++) {
    if (codecs.decoders[i]->checkSignature(signature))
      return codecs.decoders[i]->newDecoder();
  }
  /// If no decoder was found, return base type
  return nullptr;
}


template <typename T, size_t N>
Img<T, N> imread(const String &filename, int flags) {

  std::unique_ptr<BaseImageDecoder<uchar, 2>> decoder = findDecoder<uchar, 2>("TestGrayImage_24.bmp");

  if (decoder) {
    // 转换为 BmpDecoder<T, N> 的指针
    BmpDecoder<uchar, 2> *bmpDecoderPtr = dynamic_cast<BmpDecoder<uchar, 2> *>(decoder.get());
    if (bmpDecoderPtr) {
      // 创建 BmpDecoder<T, N> 实例后，进行相关操作
      // 例如：调用成员函数，解码图像等
      //定义图像是否缩放
      int scale_denom = 1;
      bmpDecoderPtr->setScale(scale_denom);
      //设置图像路径
      bmpDecoderPtr->setSource(filename);
      //读取图像头
      bmpDecoderPtr->readHeader();
      _Size size(bmpDecoderPtr->width(), bmpDecoderPtr->height());
      // grab the decoded type
      int type = bmpDecoderPtr->type();
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
      bmpDecoderPtr->readData(Image);

      return Image;

    } else {

      // 无法转换为 BmpDecoder<T, N> 的指针，可能是其他类型的解码器
      // 处理其他类型的解码器
    }
  } else {
    // 找不到合适的解码器
    // 处理找不到解码器的情况
  }
}



//  //定义图像是否缩放
//  int scale_denom = 1;
//  bmp.setScale(scale_denom);
//  //设置图像路径
//  bmp.setSource(filename);
//  //读取图像头
//  bmp.readHeader();
//  _Size size(bmp.width(), bmp.height());
//  // grab the decoded type
//  int type = bmp.type();
//  if ((flags & IMREAD_ANYDEPTH) == 0) {
//    type = CV_MAKETYPE(CV_8U, CV_MAT_CN(type));
//  }
//  if ((flags & IMREAD_COLOR) != 0 ||
//      ((flags & IMREAD_ANYCOLOR) != 0 && CV_MAT_CN(type) > 1)) {
//    type = CV_MAKETYPE(CV_MAT_DEPTH(type), 3);
//  } else {
//    type = CV_MAKETYPE(CV_MAT_DEPTH(type), 1);
//  }
//  //创建一个Img类
//  Img<uchar, 2> Image;
//  Image.create(size.width, size.height, type);
//  //读取图像数据
//  bmp.readData(Image);
//
//  return Image;
//}











#endif

