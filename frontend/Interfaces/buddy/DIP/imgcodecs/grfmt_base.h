//===- Grfmt_base.h ---------------------------------------------------===//
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
// This file is modified from opencv's modules/imgcodecs/src/grfmt_base.hpp
//
//===----------------------------------------------------------------------===//

#ifndef _GRFMT_BASE_H_
#define _GRFMT_BASE_H_

#include "buddy/DIP/imgcodecs/bitstrm.h"
#include "buddy/DIP/imgcodecs/utils.h"

namespace dip {
///////////////////////////////// base class for decoders
///////////////////////////
template <typename T, size_t N> class BaseImageDecoder {
public:
  BaseImageDecoder();
  virtual ~BaseImageDecoder() {}

  int width() const { return m_width; }
  int height() const { return m_height; }
  virtual int channels() { return m_channels; }

  // ExifEntry_t getExifTag(const ExifTagName tag) const;
  virtual bool setSource(const String &filename);
  virtual bool setSource(const Img<T, N> &buf);
  virtual int setScale(const int &scale_denom);
  virtual bool readHeader() = 0;
  virtual bool readData(Img<T, N> &img) = 0;

  /// Called after readData to advance to the next page, if any.
  virtual bool nextPage() { return false; }

  virtual size_t signatureLength() const;
  virtual bool checkSignature(const String &signature) const;
  virtual std::unique_ptr<BaseImageDecoder<T, N>> newDecoder() const = 0;

protected:
  int m_width;  // width  of the image ( filled by readHeader )
  int m_height; // height of the image ( filled by readHeader )
  int m_channels;
  int m_scale_denom;
  String m_filename;
  String m_signature;
  Img<T, N> m_buf;
  bool m_buf_supported;
  // ExifReader m_exif;
};

/////////////////////////// base class for encoders
/////////////////////////////
template <typename T, size_t N> class BaseImageEncoder {
public:
  BaseImageEncoder();
  virtual ~BaseImageEncoder() {}
  virtual bool setDestination(const String &filename);
  virtual bool setDestination(std::vector<uchar> &buf);
  virtual bool write(Img<T, N> &img, const std::vector<int> &params) = 0;
  virtual bool writemulti(const std::vector<Img<T, N>> &img_vec,
                          const std::vector<int> &params);
  virtual String getDescription() const;
  virtual std::unique_ptr<BaseImageEncoder<T, N>> newEncoder() const = 0;

protected:
  String m_description;
  String m_filename;
  std::vector<uchar> *m_buf;
  bool m_buf_supported;
  String m_last_error;
};

template <typename T, size_t N> BaseImageDecoder<T, N>::BaseImageDecoder() {
  m_width = m_height = 0;
  m_channels = -1;
  m_buf_supported = false;
  m_scale_denom = 1;
}

template <typename T, size_t N>
bool BaseImageDecoder<T, N>::setSource(const String &filename) {
  m_filename = filename;
  m_buf.release();
  return true;
}

template <typename T, size_t N>
bool BaseImageDecoder<T, N>::setSource(const Img<T, N> &buf) {
  if (!m_buf_supported)
    return false;
  m_filename = String();
  m_buf = buf;
  return true;
}

template <typename T, size_t N>
size_t BaseImageDecoder<T, N>::signatureLength() const {
  return m_signature.size();
}

template <typename T, size_t N>
bool BaseImageDecoder<T, N>::checkSignature(const String &signature) const {
  size_t len = signatureLength();
  return signature.size() >= len &&
         memcmp(signature.c_str(), m_signature.c_str(), len) == 0;
}

template <typename T, size_t N>
int BaseImageDecoder<T, N>::setScale(const int &scale_denom) {
  int temp = m_scale_denom;
  m_scale_denom = scale_denom;
  return temp;
}

template <typename T, size_t N> BaseImageEncoder<T, N>::BaseImageEncoder() {
  m_buf = 0;
  m_buf_supported = false;
}

template <typename T, size_t N>
String BaseImageEncoder<T, N>::getDescription() const {
  return m_description;
}

template <typename T, size_t N>
bool BaseImageEncoder<T, N>::setDestination(const String &filename) {
  m_filename = filename;
  m_buf = 0;
  return true;
}

template <typename T, size_t N>
bool BaseImageEncoder<T, N>::setDestination(std::vector<uchar> &buf) {
  if (!m_buf_supported)
    return false;
  m_buf = &buf;
  m_buf->clear();
  m_filename = String();
  return true;
}

template <typename T, size_t N>
bool BaseImageEncoder<T, N>::writemulti(const std::vector<Img<T, N>> &,
                                        const std::vector<int> &) {
  return false;
}
} // namespace dip
#endif /*_GRFMT_BASE_H_*/
