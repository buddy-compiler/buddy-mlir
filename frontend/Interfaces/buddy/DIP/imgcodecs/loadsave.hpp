#ifndef _LOADSAVE_HPP_
#define _LOADSAVE_HPP_

#include "buddy/DIP/ImageContainer.h"
#include "buddy/DIP/imgcodecs/Replenishment.hpp"
#include "buddy/DIP/imgcodecs/grfmt_bmp.hpp"

template <typename T, size_t N> struct ImageCodecInitializer {
  /**
   * Default Constructor for the ImageCodecInitializer
   */
  ImageCodecInitializer() {
    /// BMP Support
    decoders.push_back(std::make_unique<BmpDecoder<T, N>>());
    encoders.push_back(std::make_unique<BmpEncoder<T, N>>());

    /// JPEG Support
    // decoders.push_back(std::make_unique<JpegDecoder<T, N>>());
    // encoders.push_back(std::make_unique<JpegEncoder<T, N>>());
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

  // std::cout << "imread:" << filename << std::endl;
  std::unique_ptr<BaseImageDecoder<T, N>> decoder = findDecoder<T, N>(filename);

  if (decoder) {

    // Convert to a pointer of BmpDecoder<T, N>
    BmpDecoder<T, N> *bmpDecoderPtr =
        dynamic_cast<BmpDecoder<T, N> *>(decoder.get());

    if (bmpDecoderPtr) {
      // After creating an instance of BmpDecoder<T, N>, perform relevant
      // operations. Define whether the image is scaled
      int scale_denom = 1;
      bmpDecoderPtr->setScale(scale_denom);
      // Set the image path.
      bmpDecoderPtr->setSource(filename);
      // Read the image header
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
      // Create a class named Img.
      Img<T, N> Image;
      Image.create(size.height, size.width, type);
      // Read image data.
      bmpDecoderPtr->readData(Image);

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

  ImageCodecInitializer<T, N> &codecs = getCodecs<T, N>();
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
static bool imwrite_(const String &filename, const Img<T, N> &img_vec) {
  // bool isMultiImg = img_vec.size() > 1;
  std::vector<Img<T, N>> write_vec;

  std::unique_ptr<BaseImageEncoder<T, N>> encoder = findEncoder<T, N>(filename);

  if (encoder) {

    // Convert to a pointer of BmpEncoder<T, N>
    BmpEncoder<T, N> *bmpEncoderPtr =
        dynamic_cast<BmpEncoder<T, N> *>(encoder.get());

    if (bmpEncoderPtr) {

      bmpEncoderPtr->setDestination(filename);
      bool code = false;
      std::vector<int> params;
      code = bmpEncoderPtr->write(img_vec, params);
      return code;
    }
  }
}
#endif
