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
    encoders.push_back(std::make_unique<BmpEncoder<T, N>>());

    ///JPEG Support
    decoders.push_back(std::make_unique<JpegDecoder<T, N>>());
    //encoders.push_back(std::make_unique<JpegEncoder<T, N>>());
  }

  std::vector<std::unique_ptr<BaseImageDecoder<T, N>>> decoders;
  std::vector<std::unique_ptr<BaseImageEncoder<T, N>>> encoders;
};

template <typename T, size_t N>
static ImageCodecInitializer<T, N> &getCodecs() {
#ifdef CV_CXX11
  static ImageCodecInitializer<T, N> g_codecs;
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
static std::unique_ptr<BaseImageDecoder<T, N>>
findDecoder(const String &filename) {

  size_t i, maxlen = 0;

  /// iterate through list of registered codecs
  ImageCodecInitializer<T, N> &codecs = getCodecs<T, N>();
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

    std::cout << "imread:" << filename << std::endl;
  std::unique_ptr<BaseImageDecoder<uchar, 2>> decoder =
      findDecoder<uchar, 2>(filename);

  if (decoder) {

    // 转换为 BmpDecoder<T, N> 的指针
    BmpDecoder<uchar, 2> *bmpDecoderPtr =
        dynamic_cast<BmpDecoder<uchar, 2> *>(decoder.get());

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
    }

    // 转换为 JpegDecoder<T, N> 的指针
    JpegDecoder<uchar, 2> *JpegDecoderPtr =
        dynamic_cast<JpegDecoder<uchar, 2> *>(decoder.get());

    if (JpegDecoderPtr) {
      // 创建 JpegDecoder<T, N> 实例后，进行相关操作
      // 例如：调用成员函数，解码图像等
      //定义图像是否缩放
      int scale_denom = 1;
      JpegDecoderPtr->setScale(scale_denom);
      //设置图像路径
      JpegDecoderPtr->setSource(filename);
      //读取图像头
      JpegDecoderPtr->readHeader();
      _Size size(JpegDecoderPtr->width(), JpegDecoderPtr->height());
      // grab the decoded type
      int type = JpegDecoderPtr->type();
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
      JpegDecoderPtr->readData(Image);
      return Image;
    }

  }


}
template <typename T, size_t N>
static std::unique_ptr<BaseImageEncoder<T, N>> findEncoder(const String &_ext) {
  if (_ext.size() <= 1)
    return nullptr;

  const char *ext = strrchr(_ext.c_str(), '.');
  if (!ext)
    return nullptr;
  int len = 0;
  for (ext++; len < 128 && isalnum(ext[len]); len++)
    ;

  ImageCodecInitializer<T,N> &codecs = getCodecs<T,N>();
  for (size_t i = 0; i < codecs.encoders.size(); i++) {
    String description = codecs.encoders[i]->getDescription();
    const char *descr = strchr(description.c_str(), '(');

    while (descr) {
      descr = strchr(descr + 1, '.');
      if (!descr)
        break;
      int j = 0;
      for (descr++; j < len && isalnum(descr[j]); j++) {
        int c1 = tolower(ext[j]);
        int c2 = tolower(descr[j]);
        if (c1 != c2)
          break;
      }
      if (j == len && !isalnum(descr[j]))
        return codecs.encoders[i]->newEncoder();
      descr += j;
    }
  }

  return nullptr;
}


template <typename T, size_t N>
static bool imwrite_(const String &filename, const Img<T, N> &img_vec,
                     bool flipv) {
  // bool isMultiImg = img_vec.size() > 1; 存储多张图片
  std::vector<Img<T, N>> write_vec;

  std::unique_ptr<BaseImageEncoder<uchar, 2>> encoder =
      findEncoder<uchar, 2>(filename);

  if (encoder) {

    // 转换为 BmpDecoder<T, N> 的指针
    BmpEncoder<uchar, 2> *bmpEncoderPtr =
        dynamic_cast<BmpEncoder<uchar, 2> *>(encoder.get());

    if (bmpEncoderPtr) {

        std::vector<int> params;
      bmpEncoderPtr->setDestination(filename);
        bool code = false;
      code = bmpEncoderPtr->write(img_vec, params);
        return code;
    }
    
  }
}

#endif

