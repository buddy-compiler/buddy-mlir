//===- Utilis.h ---------------------------------------------------===//
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
// This file is modified from opencv's modules/imgcodecs/src/utils.hpp file
//
//===----------------------------------------------------------------------===//

#ifndef _UTILS_H_
#define _UTILS_H_

#include "buddy/DIP/imgcodecs/replenishment.h"

namespace dip {
inline int validateToInt(size_t sz) {
  int valueInt = (int)sz;
  assert((size_t)valueInt == sz);
  return valueInt;
}

template <typename _Tp>
static inline size_t safeCastToSizeT(const _Tp v_origin, const char *msg) {
  const size_t value_cast = (size_t)v_origin;
  if ((_Tp)value_cast != v_origin)
    std::cout << "Can't cast value into size_t";
  return value_cast;
}

struct PaletteEntry {
  unsigned char b, g, r, a;
};

#define WRITE_PIX(ptr, clr)                                                    \
  (((uchar *)(ptr))[0] = (clr).b, ((uchar *)(ptr))[1] = (clr).g,               \
   ((uchar *)(ptr))[2] = (clr).r)

#define descale(x, n) (((x) + (1 << ((n)-1))) >> (n))
#define saturate(x) (uchar)(((x) & ~255) == 0 ? (x) : ~((x) >> 31))

inline void icvCvt_BGR2Gray_8u_C3C1R(const uchar *bgr, int bgr_step,
                                     uchar *gray, int gray_step, _Size size,
                                     int swap_rb = 0);

inline void FillGrayPalette(PaletteEntry *palette, int bpp,
                            bool negative = false);
inline bool IsColorPalette(PaletteEntry *palette, int bpp);
inline void CvtPaletteToGray(const PaletteEntry *palette, uchar *grayPalette,
                             int entries);
inline uchar *FillUniColor(uchar *data, uchar *&line_end, int step, int width3,
                           int &y, int height, int count3, PaletteEntry clr);
inline uchar *FillUniGray(uchar *data, uchar *&line_end, int step, int width3,
                          int &y, int height, int count3, uchar clr);
inline uchar *FillColorRow8(uchar *data, uchar *indices, int len,
                            PaletteEntry *palette);
inline uchar *FillGrayRow8(uchar *data, uchar *indices, int len,
                           uchar *palette);
inline uchar *FillColorRow4(uchar *data, uchar *indices, int len,
                            PaletteEntry *palette);
inline uchar *FillGrayRow4(uchar *data, uchar *indices, int len,
                           uchar *palette);
inline uchar *FillColorRow1(uchar *data, uchar *indices, int len,
                            PaletteEntry *palette);
inline uchar *FillGrayRow1(uchar *data, uchar *indices, int len,
                           uchar *palette);

#define SCALE 14
#define cR (int)(0.299 * (1 << SCALE) + 0.5)
#define cG (int)(0.587 * (1 << SCALE) + 0.5)
#define cB ((1 << SCALE) - cR - cG)

void icvCvt_BGR2Gray_8u_C3C1R(const uchar *bgr, int bgr_step, uchar *gray,
                              int gray_step, _Size size, int _swap_rb) {
  int i;
  for (; size.height--; gray += gray_step) {
    short cBGR0 = cB;
    short cBGR2 = cR;
    if (_swap_rb)
      std::swap(cBGR0, cBGR2);
    for (i = 0; i < size.width; i++, bgr += 3) {
      int t = descale(bgr[0] * cBGR0 + bgr[1] * cG + bgr[2] * cBGR2, SCALE);
      gray[i] = t;
    }

    bgr += bgr_step - size.width * 3;
  }
}

void CvtPaletteToGray(const PaletteEntry *palette, uchar *grayPalette,
                      int entries) {
  int i;
  for (i = 0; i < entries; i++) {
    icvCvt_BGR2Gray_8u_C3C1R((uchar *)(palette + i), 0, grayPalette + i, 0,
                             _Size(1, 1));
  }
}

void FillGrayPalette(PaletteEntry *palette, int bpp, bool negative) {
  int i, length = 1 << bpp;
  int xor_mask = negative ? 255 : 0;

  for (i = 0; i < length; i++) {
    int val = (i * 255 / (length - 1)) ^ xor_mask;
    palette[i].b = palette[i].g = palette[i].r = (uchar)val;
    palette[i].a = 0;
  }
}

bool IsColorPalette(PaletteEntry *palette, int bpp) {
  int i, length = 1 << bpp;
  for (i = 0; i < length; i++) {
    if (palette[i].b != palette[i].g || palette[i].b != palette[i].r)
      return true;
  }
  return false;
}

uchar *FillUniColor(uchar *data, uchar *&line_end, int step, int width3, int &y,
                    int height, int count3, PaletteEntry clr) {
  do {
    uchar *end = data + count3;
    if (end > line_end)
      end = line_end;
    count3 -= (int)(end - data);
    for (; data < end; data += 3) {
      WRITE_PIX(data, clr);
    }
    if (data >= line_end) {
      line_end += step;
      data = line_end - width3;
      if (++y >= height)
        break;
    }
  } while (count3 > 0);
  return data;
}

uchar *FillUniGray(uchar *data, uchar *&line_end, int step, int width, int &y,
                   int height, int count, uchar clr) {
  do {
    uchar *end = data + count;

    if (end > line_end)
      end = line_end;
    count -= (int)(end - data);
    for (; data < end; data++) {
      *data = clr;
    }
    if (data >= line_end) {
      line_end += step;
      data = line_end - width;
      if (++y >= height)
        break;
    }
  } while (count > 0);
  return data;
}

uchar *FillColorRow8(uchar *data, uchar *indices, int len,
                     PaletteEntry *palette) {
  uchar *end = data + len * 3;
  while ((data += 3) < end) {
    *((PaletteEntry *)(data - 3)) = palette[*indices++];
  }
  PaletteEntry clr = palette[indices[0]];
  WRITE_PIX(data - 3, clr);
  return data;
}

uchar *FillGrayRow8(uchar *data, uchar *indices, int len, uchar *palette) {
  int i;
  for (i = 0; i < len; i++) {
    data[i] = palette[indices[i]];
  }
  return data + len;
}

uchar *FillColorRow4(uchar *data, uchar *indices, int len,
                     PaletteEntry *palette) {
  uchar *end = data + len * 3;

  while ((data += 6) < end) {
    int idx = *indices++;
    *((PaletteEntry *)(data - 6)) = palette[idx >> 4];
    *((PaletteEntry *)(data - 3)) = palette[idx & 15];
  }

  int idx = indices[0];
  PaletteEntry clr = palette[idx >> 4];
  WRITE_PIX(data - 6, clr);

  if (data == end) {
    clr = palette[idx & 15];
    WRITE_PIX(data - 3, clr);
  }
  return end;
}

uchar *FillGrayRow4(uchar *data, uchar *indices, int len, uchar *palette) {
  uchar *end = data + len;
  while ((data += 2) < end) {
    int idx = *indices++;
    data[-2] = palette[idx >> 4];
    data[-1] = palette[idx & 15];
  }

  int idx = indices[0];
  uchar clr = palette[idx >> 4];
  data[-2] = clr;

  if (data == end) {
    clr = palette[idx & 15];
    data[-1] = clr;
  }
  return end;
}

uchar *FillColorRow1(uchar *data, uchar *indices, int len,
                     PaletteEntry *palette) {
  uchar *end = data + len * 3;

  const PaletteEntry p0 = palette[0], p1 = palette[1];

  while ((data += 24) < end) {
    int idx = *indices++;
    *((PaletteEntry *)(data - 24)) = (idx & 128) ? p1 : p0;
    *((PaletteEntry *)(data - 21)) = (idx & 64) ? p1 : p0;
    *((PaletteEntry *)(data - 18)) = (idx & 32) ? p1 : p0;
    *((PaletteEntry *)(data - 15)) = (idx & 16) ? p1 : p0;
    *((PaletteEntry *)(data - 12)) = (idx & 8) ? p1 : p0;
    *((PaletteEntry *)(data - 9)) = (idx & 4) ? p1 : p0;
    *((PaletteEntry *)(data - 6)) = (idx & 2) ? p1 : p0;
    *((PaletteEntry *)(data - 3)) = (idx & 1) ? p1 : p0;
  }

  int idx = indices[0];
  for (data -= 24; data < end; data += 3, idx += idx) {
    const PaletteEntry clr = (idx & 128) ? p1 : p0;
    WRITE_PIX(data, clr);
  }

  return data;
}

uchar *FillGrayRow1(uchar *data, uchar *indices, int len, uchar *palette) {
  uchar *end = data + len;

  const uchar p0 = palette[0], p1 = palette[1];

  while ((data += 8) < end) {
    int idx = *indices++;
    *((uchar *)(data - 8)) = (idx & 128) ? p1 : p0;
    *((uchar *)(data - 7)) = (idx & 64) ? p1 : p0;
    *((uchar *)(data - 6)) = (idx & 32) ? p1 : p0;
    *((uchar *)(data - 5)) = (idx & 16) ? p1 : p0;
    *((uchar *)(data - 4)) = (idx & 8) ? p1 : p0;
    *((uchar *)(data - 3)) = (idx & 4) ? p1 : p0;
    *((uchar *)(data - 2)) = (idx & 2) ? p1 : p0;
    *((uchar *)(data - 1)) = (idx & 1) ? p1 : p0;
  }

  int idx = indices[0];
  for (data -= 8; data < end; data++, idx += idx) {
    data[0] = (idx & 128) ? p1 : p0;
  }
  return data;
}
} // namespace dip
#endif /*_UTILS_H_*/
