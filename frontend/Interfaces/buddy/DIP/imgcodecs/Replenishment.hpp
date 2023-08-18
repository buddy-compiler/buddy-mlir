//===- Replenishment.hpp ---------------------------------------------------===//
////////////////////////////////////////////////////////////////////////////////////////
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
#define IMG_CN_MAX 512
#define IMG_CN_SHIFT 3
#define IMG_DEPTH_MAX (1 << IMG_CN_SHIFT)
#define IMG_MAT_CN_MASK ((IMG_CN_MAX - 1) << IMG_CN_SHIFT)
#define IMG_MAT_CN(flags) ((((flags)&IMG_MAT_CN_MASK) >> IMG_CN_SHIFT) + 1)

#define IMG_8U 0
#define IMG_8S 1
#define IMG_16U 2
#define IMG_16S 3
#define IMG_32S 4
#define IMG_32F 5
#define IMG_64F 6
#define IMG_16F 7

#define IMG_MAT_DEPTH_MASK (IMG_DEPTH_MAX - 1)
#define IMG_MAT_DEPTH(flags) ((flags)&IMG_MAT_DEPTH_MASK)

#define IMG_MAKETYPE(depth, cn)                                                \
  (IMG_MAT_DEPTH(depth) + (((cn)-1) << IMG_CN_SHIFT))
#define IMG_MAKE_TYPE IMG_MAKETYPE

#define IMG_8UC1 IMG_MAKETYPE(IMG_8U, 1)
#define IMG_8UC2 IMG_MAKETYPE(IMG_8U, 2)
#define IMG_8UC3 IMG_MAKETYPE(IMG_8U, 3)
#define IMG_8UC4 IMG_MAKETYPE(IMG_8U, 4)
#define IMG_8UC(n) IMG_MAKETYPE(IMG_8U, (n))

#define IMG_8SC1 IMG_MAKETYPE(IMG_8S, 1)
#define IMG_8SC2 IMG_MAKETYPE(IMG_8S, 2)
#define IMG_8SC3 IMG_MAKETYPE(IMG_8S, 3)
#define IMG_8SC4 IMG_MAKETYPE(IMG_8S, 4)
#define IMG_8SC(n) IMG_MAKETYPE(IMG_8S, (n))

#define IMG_16UC1 IMG_MAKETYPE(IMG_16U, 1)
#define IMG_16UC2 IMG_MAKETYPE(IMG_16U, 2)
#define IMG_16UC3 IMG_MAKETYPE(IMG_16U, 3)
#define IMG_16UC4 IMG_MAKETYPE(IMG_16U, 4)
#define IMG_16UC(n) IMG_MAKETYPE(IMG_16U, (n))

#define IMG_16SC1 IMG_MAKETYPE(IMG_16S, 1)
#define IMG_16SC2 IMG_MAKETYPE(IMG_16S, 2)
#define IMG_16SC3 IMG_MAKETYPE(IMG_16S, 3)
#define IMG_16SC4 IMG_MAKETYPE(IMG_16S, 4)
#define IMG_16SC(n) IMG_MAKETYPE(IMG_16S, (n))

#define IMG_32SC1 IMG_MAKETYPE(IMG_32S, 1)
#define IMG_32SC2 IMG_MAKETYPE(IMG_32S, 2)
#define IMG_32SC3 IMG_MAKETYPE(IMG_32S, 3)
#define IMG_32SC4 IMG_MAKETYPE(IMG_32S, 4)
#define IMG_32SC(n) IMG_MAKETYPE(IMG_32S, (n))

#define IMG_32FC1 IMG_MAKETYPE(IMG_32F, 1)
#define IMG_32FC2 IMG_MAKETYPE(IMG_32F, 2)
#define IMG_32FC3 IMG_MAKETYPE(IMG_32F, 3)
#define IMG_32FC4 IMG_MAKETYPE(IMG_32F, 4)
#define IMG_32FC(n) IMG_MAKETYPE(IMG_32F, (n))

#define IMG_64FC1 IMG_MAKETYPE(IMG_64F, 1)
#define IMG_64FC2 IMG_MAKETYPE(IMG_64F, 2)
#define IMG_64FC3 IMG_MAKETYPE(IMG_64F, 3)
#define IMG_64FC4 IMG_MAKETYPE(IMG_64F, 4)
#define IMG_64FC(n) IMG_MAKETYPE(IMG_64F, (n))

#define IMG_16FC1 IMG_MAKETYPE(IMG_16F, 1)
#define IMG_16FC2 IMG_MAKETYPE(IMG_16F, 2)
#define IMG_16FC3 IMG_MAKETYPE(IMG_16F, 3)
#define IMG_16FC4 IMG_MAKETYPE(IMG_16F, 4)
#define IMG_16FC(n) IMG_MAKETYPE(IMG_16F, (n))

#define IMG_ELEM_SIZE1(type) ((0x28442211 >> IMG_MAT_DEPTH(type) * 4) & 15)
#define IMG_ELEM_SIZE(type) (IMG_MAT_CN(type) * IMG_ELEM_SIZE1(type))

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
  // If set, return 16-bit/32-bit image when the input has the corresponding
  // depth, otherwise convert it to 8-bit.
  IMGRD_ANYDEPTH = 2,
  // If set, the image is read in any possible color format.
  IMGRD_ANYCOLOR = 4,
  // If set, use the gdal driver for loading the image.
  IMGRD_LOAD_GDAL = 8,
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
