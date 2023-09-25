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
