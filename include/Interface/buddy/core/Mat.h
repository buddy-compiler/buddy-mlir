//===- Mat.h ---------------------------------------------------===//
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
// Buddy Mat Similar To OpenCV Mat.
//
//===----------------------------------------------------------------------===//

#ifndef INTERFACE_BUDDY_CORE_MAT
#define INTERFACE_BUDDY_CORE_MAT

#include "iostream"

enum MatType {
  MAT_GRAY_U8,
  MAT_GRAY_F32,
  MAT_GRAY_F64,
  MAT_RGB_U8,
  MAT_RGB_F32,
  MAT_RGB_F64,
  MAT_RGBA_U8,
  MAT_RGBA_F32,
  MAT_RGBA_F64
};

enum SaveImageType {
  MAT_SAVE_PNG,
  MAT_SAVE_BMP,
  MAT_SAVE_JPG,
  MAT_SAVE_HDR,
  MAT_SAVE_TGA,
};

union MatData {
  uint8_t *u8 = nullptr;
  int8_t *i8;
  uint16_t *u16;
  int16_t *i16;
  uint32_t *u32;
  int32_t *i32;
  float *f32;
  double *f64;
};

class Mat {
public:
  Mat(const std::string &path);
  // Copy constructor.
  Mat(const Mat &mat);
  // Move constructor.
  Mat(Mat &&mat);
  /* must with && or std::move */
  void setDataNull();
  void readImage(const std::string &path);
  void saveImage(const std::string &path, const SaveImageType &saveImageType,
                 const int &quality);
  void saveImage(const std::string &path, const int &quality = 100);

protected:
  int _width = 0;
  int _height = 0;
  int _channel = 0;
  int _step = 0;

  MatType _matType = MatType::MAT_RGB_F32;
  MatData _data;
};

#endif // INTERFACE_BUDDY_CORE_MAT
