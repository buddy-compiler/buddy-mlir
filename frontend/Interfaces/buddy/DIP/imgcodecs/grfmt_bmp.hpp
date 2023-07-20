/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this
license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without
modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright
notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote
products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is"
and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are
disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any
direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef _GRFMT_BMP_H_
#define _GRFMT_BMP_H_

#include "grfmt_base.hpp"

enum BmpCompression {
  BMP_RGB = 0,
  BMP_RLE8 = 1,
  BMP_RLE4 = 2,
  BMP_BITFIELDS = 3
};

// Windows Bitmap reader
template <typename T, size_t N>
class BmpDecoder : public BaseImageDecoder<T, N> {
public:
  BmpDecoder();
  ~BmpDecoder();

  bool readData(Img<T, N> &img) override;
  bool readHeader() override;
  void close();
  std::unique_ptr<BaseImageDecoder<T, N>> newDecoder() const override;

protected:
  void initMask();
  void maskBGRA(uchar *des, uchar *src, int num);

  enum Origin { ORIGIN_TL = 0, ORIGIN_BL = 1 };

  RLByteStream<T, N> m_strm;
  PaletteEntry m_palette[256];
  Origin m_origin;
  int m_bpp;
  int m_offset;
  BmpCompression m_rle_code;
  uint m_rgba_mask[4];
  int m_rgba_bit_offset[4];
};

// ... writer
// class BmpEncoder CV_FINAL : public BaseImageEncoder {
// public:
//  BmpEncoder();
//  ~BmpEncoder() CV_OVERRIDE;
//
//  bool write(const Mat &img, const std::vector<int> &params) CV_OVERRIDE;
//
//  ImageEncoder newEncoder() const CV_OVERRIDE;
//};

static const char *fmtSignBmp = "BM";

/************************ BMP decoder *****************************/

template <typename T, size_t N> BmpDecoder<T, N>::BmpDecoder() {
  this->m_signature = fmtSignBmp;
  m_offset = -1;
  this->m_buf_supported = true;
  m_origin = ORIGIN_TL;
  m_bpp = 0;
  m_rle_code = BMP_RGB;
  initMask();
}

template <typename T, size_t N> BmpDecoder<T, N>::~BmpDecoder() {}

template <typename T, size_t N> void BmpDecoder<T, N>::close() {
  m_strm.close();
}

template <typename T, size_t N>
std::unique_ptr<BaseImageDecoder<T, N>> BmpDecoder<T, N>::newDecoder() const {
  return std::make_unique<BmpDecoder<T, N>>();
}

template <typename T, size_t N> void BmpDecoder<T, N>::initMask() {
  memset(m_rgba_mask, 0, sizeof(m_rgba_mask));
  memset(m_rgba_bit_offset, -1, sizeof(m_rgba_bit_offset));
}

template <typename T, size_t N>
void BmpDecoder<T, N>::maskBGRA(uchar *des, uchar *src, int num) {
  for (int i = 0; i < num; i++, des += 4, src += 4) {
    uint data = *((uint *)src);
    des[0] = (uchar)((m_rgba_mask[2] & data) >> m_rgba_bit_offset[2]);
    des[1] = (uchar)((m_rgba_mask[1] & data) >> m_rgba_bit_offset[1]);
    des[2] = (uchar)((m_rgba_mask[0] & data) >> m_rgba_bit_offset[0]);
    if (m_rgba_bit_offset[3] >= 0)
      des[3] = (uchar)((m_rgba_mask[3] & data) >> m_rgba_bit_offset[3]);
    else
      des[3] = 255;
  }
}

template <typename T, size_t N> bool BmpDecoder<T, N>::readHeader() {
  bool result = false;
  bool iscolor = false;

  /*if (!this->m_buf.empty()) {
    if (!m_strm.open(m_buf))
      return false;
  } else if (!m_strm.open(m_filename))
    return false;*/
  m_strm.open(this->m_filename);

  try {
    m_strm.skip(10);
    m_offset = m_strm.getDWord();
    int size = m_strm.getDWord();
    assert(size > 0); // overflow, 2Gb limit
    initMask();
    if (size >= 36) {
      this->m_width = m_strm.getDWord();
      this->m_height = m_strm.getDWord();
      m_bpp = m_strm.getDWord() >> 16;
      int m_rle_code_ = m_strm.getDWord();
      assert(m_rle_code_ >= 0 && m_rle_code_ <= BMP_BITFIELDS);
      m_rle_code = (BmpCompression)m_rle_code_;
      m_strm.skip(12);
      int clrused = m_strm.getDWord();

      if (m_bpp == 32 && m_rle_code == BMP_BITFIELDS && size >= 56) {
        m_strm.skip(4); // important colors
        // 0 is Red channel bit mask, 1 is Green channel bit mask, 2 is Blue
        // channel bit mask, 3 is Alpha channel bit mask
        for (int index_rgba = 0; index_rgba < 4; ++index_rgba) {
          uint mask = m_strm.getDWord();
          m_rgba_mask[index_rgba] = mask;
          if (mask != 0) {
            int bit_count = 0;
            while (!(mask & 1)) {
              mask >>= 1;
              ++bit_count;
            }
            m_rgba_bit_offset[index_rgba] = bit_count;
          }
        }
        m_strm.skip(size - 56);
      } else
        m_strm.skip(size - 36);

      if (this->m_width > 0 && this->m_height != 0 &&
          (((m_bpp == 1 || m_bpp == 4 || m_bpp == 8 || m_bpp == 24 ||
             m_bpp == 32) &&
            m_rle_code == BMP_RGB) ||
           ((m_bpp == 16 || m_bpp == 32) &&
            (m_rle_code == BMP_RGB || m_rle_code == BMP_BITFIELDS)) ||
           (m_bpp == 4 && m_rle_code == BMP_RLE4) ||
           (m_bpp == 8 && m_rle_code == BMP_RLE8))) {
        iscolor = true;
        result = true;

        if (m_bpp <= 8) {
          assert(clrused >= 0 && clrused <= 256);
          memset(m_palette, 0, sizeof(m_palette));
          m_strm.getBytes(m_palette, (clrused == 0 ? 1 << m_bpp : clrused) * 4);
          iscolor = IsColorPalette(m_palette, m_bpp);
        } else if (m_bpp == 16 && m_rle_code == BMP_BITFIELDS) {
          int redmask = m_strm.getDWord();
          int greenmask = m_strm.getDWord();
          int bluemask = m_strm.getDWord();

          if (bluemask == 0x1f && greenmask == 0x3e0 && redmask == 0x7c00)
            m_bpp = 15;
          else if (bluemask == 0x1f && greenmask == 0x7e0 && redmask == 0xf800)
            ;
          else
            result = false;
        } else if (m_bpp == 32 && m_rle_code == BMP_BITFIELDS) {
          // 32bit BMP not require to check something - we can simply allow it
          // to use
          ;
        } else if (m_bpp == 16 && m_rle_code == BMP_RGB)
          m_bpp = 15;
      }
    } else if (size == 12) {
      this->m_width = m_strm.getWord();
      this->m_height = m_strm.getWord();
      m_bpp = m_strm.getDWord() >> 16;
      m_rle_code = BMP_RGB;

      if (this->m_width > 0 && this->m_height != 0 &&
          (m_bpp == 1 || m_bpp == 4 || m_bpp == 8 || m_bpp == 24 ||
           m_bpp == 32)) {
        if (m_bpp <= 8) {
          uchar buffer[256 * 3];
          int j, clrused = 1 << m_bpp;
          m_strm.getBytes(buffer, clrused * 3);
          for (j = 0; j < clrused; j++) {
            m_palette[j].b = buffer[3 * j + 0];
            m_palette[j].g = buffer[3 * j + 1];
            m_palette[j].r = buffer[3 * j + 2];
          }
        }
        result = true;
      }
    }
  } catch (...) {
    throw;
  }
  // in 32 bit case alpha channel is used - so require CV_8UC4 type
  this->m_type =
      iscolor ? ((m_bpp == 32 && m_rle_code != BMP_RGB) ? CV_8UC4 : CV_8UC3)
              : CV_8UC1;
  m_origin = this->m_height > 0 ? ORIGIN_BL : ORIGIN_TL;
  this->m_height = std::abs(this->m_height);
  if (!result) {
    m_offset = -1;
    this->m_width = this->m_height = -1;
    m_strm.close();
  }

  std::cout << this->m_filename << " m_bpp = " << m_bpp << std::endl;
  std::cout << " m_height = " << this->m_height << std::endl;
  std::cout << " m_width = " << this->m_width << std::endl;
  return result;
}

template <typename T, size_t N>
bool BmpDecoder<T, N>::readData(Img<T, N> &img) {
  uchar *data = img.data;
  // int step = validateToInt(img.step);
  int step = 4;
  bool color = false;
  uchar gray_palette[256] = {0};
  bool result = false;
  int src_pitch =
      ((this->m_width * (m_bpp != 15 ? m_bpp : 16) + 7) / 8 + 3) & -4;
  int nch = color ? 3 : 1;
  int y, width3 = this->m_width * nch;

  // FIXIT: use safe pointer arithmetic (avoid 'int'), use size_t, intptr_t, etc
  // assert(((uint64)m_height * m_width * nch < (CV_BIG_UINT(1) << 30)) &&"BMP
  // reader implementation doesn't support large images >= 1Gb");

  if (m_offset < 0 || !m_strm.isOpened())
    return false;

  if (m_origin == ORIGIN_BL) {
    data += (this->m_height - 1) * (size_t)step;
    step = -step;
  }

  /*AutoBuffer<uchar> _src, _bgr;
  _src.allocate(src_pitch + 32);*/
  uchar *_src = new uchar[src_pitch + 32];
  uchar *_bgr = NULL;

  if (!color) {
    if (m_bpp <= 8) {
      CvtPaletteToGray(m_palette, gray_palette, 1 << m_bpp);
    }
    _bgr = new uchar[this->m_width * 3 + 32];
  }

  uchar *src = _src;
  uchar *bgr = _bgr;
  m_strm.setPos(m_offset);

  switch (m_bpp) {
    /************************* 24 BPP ************************/
  case 24:
    for (y = 0; y < this->m_height; y++, data += step) {

      m_strm.getBytes(src, src_pitch);

      memcpy(data, src, this->m_width * 3);
    }
    result = true;
    break;

    /************************* 8 BPP ************************/
  case 8:
    if (m_rle_code == BMP_RGB) {
      for (y = 0; y < this->m_height; y++, data += step) {
        m_strm.getBytes(src, src_pitch);
        if (color)
          FillColorRow8(data, src, this->m_width, m_palette);
        else
          FillGrayRow8(data, src, this->m_width, gray_palette);
      }
      result = true;
    } else if (m_rle_code == BMP_RLE8) // rle8 compression
    {
      uchar *line_end = data + width3;
      int line_end_flag = 0;
      y = 0;

      for (;;) {
        int code = m_strm.getWord();
        int len = code & 255;
        code >>= 8;
        if (len != 0) // encoded mode
        {
          int prev_y = y;
          len *= nch;

          if (data + len > line_end)
            goto decode_rle8_bad;

          if (color)
            data = FillUniColor(data, line_end, step, width3, y, this->m_height,
                                len, m_palette[code]);
          else
            data = FillUniGray(data, line_end, step, width3, y, this->m_height,
                               len, gray_palette[code]);

          line_end_flag = y - prev_y;

          if (y >= this->m_height)
            break;
        } else if (code > 2) // absolute mode
        {
          int prev_y = y;
          int code3 = code * nch;

          if (data + code3 > line_end)
            goto decode_rle8_bad;
          int sz = (code + 1) & (~1);
          // assert((size_t)sz < _src.size());
          m_strm.getBytes(src, sz);
          if (color)
            data = FillColorRow8(data, src, code, m_palette);
          else
            data = FillGrayRow8(data, src, code, gray_palette);

          line_end_flag = y - prev_y;
        } else {
          int x_shift3 = (int)(line_end - data);
          int y_shift = this->m_height - y;

          if (code || !line_end_flag || x_shift3 < width3) {
            if (code == 2) {
              x_shift3 = m_strm.getByte() * nch;
              y_shift = m_strm.getByte();
            }

            x_shift3 += (y_shift * width3) & ((code == 0) - 1);

            if (y >= this->m_height)
              break;

            if (color)
              data = FillUniColor(data, line_end, step, width3, y,
                                  this->m_height, x_shift3, m_palette[0]);
            else
              data = FillUniGray(data, line_end, step, width3, y,
                                 this->m_height, x_shift3, gray_palette[0]);

            if (y >= this->m_height)
              break;
          }

          line_end_flag = 0;
          if (y >= this->m_height)
            break;
        }
      }

      result = true;
    decode_rle8_bad:;
    }
    break;
  }

  return result;
}

#endif /*_GRFMT_BMP_H_*/
