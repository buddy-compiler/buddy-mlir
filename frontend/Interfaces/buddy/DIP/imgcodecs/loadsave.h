//===- Loadsave.h ---------------------------------------------------===//
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this
//  license. If you do not agree to this license, do not download, install, copy
//  or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright
//   notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote
//   products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is"
// and any express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular purpose
// are disclaimed. In no event shall the Intel Corporation or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or business
// interruption) however caused and on any theory of liability, whether in
// contract, strict liability, or tort (including negligence or otherwise)
// arising in any way out of the use of this software, even if advised of the
// possibility of such damage.
//
//
//===----------------------------------------------------------------------===//
//
// This file is modified from opencv's modules/imgcodecs/src/loadsave.hpp file
//
//===----------------------------------------------------------------------===//

#ifndef _LOADSAVE_H_
#define _LOADSAVE_H_

#include "buddy/DIP/imgcodecs/grfmt_bmp.h"
#include "buddy/DIP/imgcodecs/grfmt_jpeg.h"
#include "buddy/DIP/imgcodecs/grfmt_png.h"
#include "buddy/DIP/imgcodecs/replenishment.h"

namespace dip {
template <typename T, size_t N> struct ImageCodecInitializer {
  /**
   * Default Constructor for the ImageCodecInitializer
   */
  ImageCodecInitializer() {
    // BMP Support
    decoders.push_back(std::make_unique<BmpDecoder<T, N>>());
    encoders.push_back(std::make_unique<BmpEncoder<T, N>>());

    // JPEG Support
    decoders.push_back(std::make_unique<JpegDecoder<T, N>>());
    encoders.push_back(std::make_unique<JpegEncoder<T, N>>());

    // PNG Support
    decoders.push_back(std::make_unique<PngDecoder<T, N>>());
    encoders.push_back(std::make_unique<PngEncoder<T, N>>());
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
  std::unique_ptr<BaseImageDecoder<T, N>> decoder = findDecoder<T, N>(filename);

  if (!decoder) {
    throw std::runtime_error("Decoder not found for the given image.");
  }

  if (decoder) {
    // Converts a pointer to BmpDecoder<T, N>
    BmpDecoder<T, N> *bmpDecoderPtr =
        dynamic_cast<BmpDecoder<T, N> *>(decoder.get());
    if (bmpDecoderPtr) {
      // After creating the BmpDecoder<T, N> instance, perform related
      // operations Defines whether the image is scaled or not
      int scale_denom = 1;
      bmpDecoderPtr->setScale(scale_denom);
      // Set image path
      bmpDecoderPtr->setSource(filename);
      // Read image head
      bmpDecoderPtr->readHeader();
      int channels = bmpDecoderPtr->channels();
      if ((flags & IMGRD_COLOR) != 0 ||
          ((flags & IMGRD_ANYCOLOR) != 0 && channels > 1)) {
        channels = 3;
      } else {
        channels = 1;
      }
      // Create an Img instance
      intptr_t sizes[3] = {bmpDecoderPtr->height(), bmpDecoderPtr->width(),
                           channels};
      Img<T, N> Image(sizes);
      bmpDecoderPtr->readData(Image);
      return Image;
    }
    // Converts a pointer to JpegDecoder<T, N>
    JpegDecoder<T, N> *JpegDecoderPtr =
        dynamic_cast<JpegDecoder<T, N> *>(decoder.get());
    if (JpegDecoderPtr) {
      // After creating the JpegDecoder<T, N> instance, perform related
      // operations Defines whether the image is scaled or not
      int scale_denom = 1;
      JpegDecoderPtr->setScale(scale_denom);
      // Set image path
      JpegDecoderPtr->setSource(filename);
      // Read image head
      JpegDecoderPtr->readHeader();
      int channels = JpegDecoderPtr->channels();
      if ((flags & IMGRD_COLOR) != 0 ||
          ((flags & IMGRD_ANYCOLOR) != 0 && channels > 1)) {
        channels = 3;
      } else {
        channels = 1;
      }
      // Create an Img instance
      intptr_t sizes[3] = {JpegDecoderPtr->height(), JpegDecoderPtr->width(),
                           channels};
      Img<T, N> Image(sizes);
      JpegDecoderPtr->readData(Image);
      return Image;
    }

    // Converts a pointer to BmpDecoder<T, N>
    PngDecoder<T, N> *PngDecoderPtr =
        dynamic_cast<PngDecoder<T, N> *>(decoder.get());
    if (PngDecoderPtr) {
      // After creating the BmpDecoder<T, N> instance, perform related
      // operations Defines whether the image is scaled or not
      int scale_denom = 1;
      PngDecoderPtr->setScale(scale_denom);
      // Set image path
      PngDecoderPtr->setSource(filename);
      // Read image head
      PngDecoderPtr->readHeader();
      int channels = PngDecoderPtr->channels();
      if ((flags & IMGRD_COLOR) != 0 ||
          ((flags & IMGRD_ANYCOLOR) != 0 && channels > 1)) {
        channels = 3;
      } else {
        channels = 1;
      }
      // Create an Img instance
      intptr_t sizes[3] = {PngDecoderPtr->height(), PngDecoderPtr->width(),
                           channels};
      Img<T, N> Image(sizes);
      PngDecoderPtr->readData(Image);
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
static bool imwrite(const String &filename, Img<T, N> &img_vec) {
  std::unique_ptr<BaseImageEncoder<T, N>> encoder = findEncoder<T, N>(filename);

  if (!encoder) {
    throw std::runtime_error("Encoder not found for the given image.");
  }

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

    // Convert to a pointer of JpegEncoder<T, N>
    JpegEncoder<T, N> *jpegEncoderPtr =
        dynamic_cast<JpegEncoder<T, N> *>(encoder.get());
    if (jpegEncoderPtr) {
      jpegEncoderPtr->setDestination(filename);
      bool code = false;
      std::vector<int> params;
      code = jpegEncoderPtr->write(img_vec, params);
      return code;
    }

    // Convert to a pointer of PngEncoder<T, N>
    PngEncoder<T, N> *pngEncoderPtr =
        dynamic_cast<PngEncoder<T, N> *>(encoder.get());
    if (pngEncoderPtr) {
      pngEncoderPtr->setDestination(filename);
      bool code = false;
      std::vector<int> params;
      code = pngEncoderPtr->write(img_vec, params);
      return code;
    }
  }

  return true;
}
} // namespace dip
#endif
