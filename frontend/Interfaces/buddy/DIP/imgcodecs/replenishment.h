//===- Replenishment.h ----------------------------------------------===//
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
//===----------------------------------------------------------------------===//
//
// This file is modified from opencv's
// modules/imgcodecs/include/opencv2/imgcodecs.hpp file
//
//===----------------------------------------------------------------------===//

#ifndef _REPLEISHMENT_H
#define _REPLEISHMENT_H

#include <cassert>
#include <memory>
#include <string>
#include <type_traits>

using namespace std;
typedef unsigned long ulong;
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef string String;

namespace dip {
// Imread flags
enum ImreadModes {
  // If set, return the loaded image as is (with alpha channel,otherwise it gets
  // cropped).
  IMGRD_UNCHANGED = -1,
  // If set, always convert image to the single channel grayscale image (codec
  // internal conversion).
  IMGRD_GRAYSCALE = 0,
  // If set, always convert image to the 3 channel BGR color image.
  IMGRD_COLOR = 1,
  // If set, the image is read in any possible color format.
  IMGRD_ANYCOLOR = 4,
};

// Imwrite flags
enum ImwriteFlags {
  // For JPEG, it can be a quality from 0 to 100 (the higher is the
  // better). Default value is 95.
  IMWRITE_JPEG_QUALITY = 1,
  // Enable JPEG features, 0 or 1, default is False.
  IMWRITE_JPEG_PROGRESSIVE = 2,
  // Enable JPEG features, 0 or 1, default is False.
  IMWRITE_JPEG_OPTIMIZE = 3,
  // JPEG restart interval, 0 - 65535, default is 0 - no restart.
  IMWRITE_JPEG_RST_INTERVAL = 4,
  // Separate luma quality level, 0 - 100, default is 0 - don't use.
  IMWRITE_JPEG_LUMA_QUALITY = 5,
  // Separate chroma quality level, 0 - 100, default is 0 - don't  use.
  IMWRITE_JPEG_CHROMA_QUALITY = 6,
  // Use this value for normal data.
  IMWRITE_PNG_STRATEGY_DEFAULT = 0,
  // Use this value for data produced by a filter (or predictor).Filtered data
  // consists mostly of small values with a somewhat random //distribution. In
  // this case, the compression algorithm is tuned to compress them better.
  IMWRITE_PNG_STRATEGY_FILTERED = 1,
  // Use this value to force Huffman encoding only (no string match).
  IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY = 2,
  // Use this value to limit match distances to one (run-length encoding).
  IMWRITE_PNG_STRATEGY_RLE = 3,
  // Using this value prevents the use of dynamic Huffman codes, allowing for a
  // simpler decoder for special applications.
  IMWRITE_PNG_STRATEGY_FIXED = 4,
};

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
} // namespace dip
#endif // Repleishment_HPP
