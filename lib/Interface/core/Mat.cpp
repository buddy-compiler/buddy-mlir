//===- Mat.cpp ------------------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Mat class.
//
//===----------------------------------------------------------------------===//
#ifndef CORE_MAT_DEF
#define CORE_MAT_DEF

#include <algorithm>
#include <cassert>
#include <memory>
#include <numeric>
#include <stdexcept>

#include "Interface/buddy/core/Mat.h"
#include "Interface/buddy/stb/stb_image_write.h"
#include "Interface/buddy/stb/std_image.h"

Mat::Mat(const Mat &mat) {
  this->_channel = mat._channel;
  this->_height = mat._height;
  this->_width = mat._width;
  this->_matType = mat._matType;
  this->_step = mat._step;

  this->_data.u8 = new uint8_t[this->_width * this->_height * this->_step]();

  if (mat.getData().u8 != nullptr) {
    memcpy(this->_data.u8, mat.getData().u8,
           this->_width * this->_height * this->_step);
  }
}

void Mat::setDataNull() {
  this->_width = 0;
  this->_height = 0;
  this->_channel = 0;
  this->_step = 0;
  this->_matType = MatType::MAT_RGB_U8;
  this->_data.u8 = nullptr;
}

Mat::Mat(Mat &&mat) {

  this->_channel = mat._channel;
  this->_height = mat._height;
  this->_width = mat._width;
  this->_matType = mat._matType;
  this->_step = mat._step;
  this->_data.u8 = mat._data.u8;
  mat.setDataNull();
}

void Mat::readImage(const std::string &path) {

  this->_data.u8 =
      stbi_load(path.data(), &this->_width, &this->_height, &this->_channel, 0);

  if (this->_data.u8 == nullptr) {
    throw Exception(1, "[Mat]: img empty, maybe path error! \n", __FILE__,
                    __LINE__, __FUNCTION__);
  }

  if (this->_channel == 1) {
    this->_matType = MatType::MAT_GRAY_U8;
    this->_step = this->_channel;
  } else if (this->_channel == 3) {
    this->_matType = MatType::MAT_RGB_U8;
    this->_step = this->_channel;
  } else if (this->_channel == 4) {
    this->_matType = MatType::MAT_RGBA_U8;
    this->_step = this->_channel;
  }
}

Mat::Mat(const std::string &path) { readImage(path); }

void Mat::saveImage(const std::string &path, const SaveImageType &saveImageType,
                    const int &quality) {
  if (this->isEmpty()) {
    throw Exception(1, "[Mat]: img empty! \n", __FILE__, __LINE__,
                    __FUNCTION__);
  }

  float *f32Val = nullptr;
  uint8_t *u8Val = nullptr;

  Mat tmpMat;

  if (saveImageType == SaveImageType::MAT_SAVE_HDR) {
    if (this->isF32Mat()) {
      f32Val = this->_data.f32;
    } else {
      this->convertTo(tmpMat, CVT_DATA_TO_F32);
      f32Val = tmpMat.getData().f32;
    }
  } else {
    if (this->isU8Mat()) {
      u8Val = this->_data.u8;
    } else {
      this->convertTo(tmpMat, CVT_DATA_TO_U8);
      u8Val = tmpMat.getData().u8;
    }
  }

  int ret;
  switch (saveImageType) {
  case SaveImageType::MAT_SAVE_BMP:
    ret = stbi_write_bmp(path.c_str(), this->_width, this->_height,
                         this->_channel, u8Val);
    break;
  case SaveImageType::MAT_SAVE_JPG:
    ret = stbi_write_jpg(path.c_str(), this->_width, this->_height,
                         this->_channel, u8Val, quality);
    break;
  case SaveImageType::MAT_SAVE_PNG:
    ret = stbi_write_png(path.c_str(), this->_width, this->_height,
                         this->_channel, u8Val, 0);
    break;
  case SaveImageType::MAT_SAVE_HDR:
    ret = stbi_write_hdr(path.c_str(), this->_width, this->_height,
                         this->_channel, f32Val);
    break;
  case SaveImageType::MAT_SAVE_TGA:
    ret = stbi_write_tga(path.c_str(), this->_width, this->_height,
                         this->_channel, u8Val);
    break;
  }

  if (ret < 1) {
    throw Exception(1, "[Mat]: save image error! \n", __FILE__, __LINE__,
                    __FUNCTION__);
  }
}

void Mat::saveImage(const std::string &path, const int &quality) {
  if (this->isEmpty()) {
    throw Exception(1, "[Mat]: img empty! \n", __FILE__, __LINE__,
                    __FUNCTION__);
  }
  std::vector<std::string> splits;
  std::string tmpPath = path;
  ExString::split(splits, tmpPath, ".");
  std::string imgType = splits[splits.size() - 1];
  if (imgType == "jpg" || imgType == "jpeg" || imgType == "JPG" ||
      imgType == "JPEG") {
    saveImage(path, SaveImageType::MAT_SAVE_JPG, quality);
  } else if (imgType == "png" || imgType == "PNG") {
    saveImage(path, SaveImageType::MAT_SAVE_PNG, quality);
  } else if (imgType == "bmp" || imgType == "BMP") {
    saveImage(path, SaveImageType::MAT_SAVE_BMP, quality);
  } else if (imgType == "tga" || imgType == "TGA") {
    saveImage(path, SaveImageType::MAT_SAVE_TGA, quality);
  } else if (imgType == "hdr" || imgType == "HDR") {
    saveImage(path, SaveImageType::MAT_SAVE_TGA, quality);
  } else {
    throw Exception(1, "[Mat]: unknown image type : " + imgType + "! \n",
                    __FILE__, __LINE__, __FUNCTION__);
  }
}

#endif // CORE_MAT_DEF
