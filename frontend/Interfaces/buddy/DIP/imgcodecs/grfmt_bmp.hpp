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

#include "buddy/DIP/imgcodecs/grfmt_base.hpp"

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

//... writer
template <typename T, size_t N>
class BmpEncoder : public BaseImageEncoder<T, N> {
public:
  BmpEncoder();
  ~BmpEncoder();

  bool write(const Img<T, N> &img, const std::vector<int> &params);

  std::unique_ptr<BaseImageEncoder<T, N>> newEncoder() const override;
};

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
  return result;
}
template <typename T, size_t N>
bool BmpDecoder<T, N>::readData(Img<T, N> &img) {
  T *data = img.data;
  // int step = validateToInt(img.step);
  int step = this->m_width * img.channels();
  bool color = img.channels() > 1;
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

    /************************* 32 BPP ************************/
  case 32:
    for (y = 0; y < this->m_height; y++, data += step) {
      m_strm.getBytes(src, src_pitch);
      if (!color) {
        int rgba_step = 0;
        int gray_step = 0;
        _Size size(this->m_width, 1);
        int _swap_rb = 1;

        for (; size.height--; data += gray_step) {
          short cBGR0 = cB;
          short cBGR2 = cR;
          if (_swap_rb)
            std::swap(cBGR0, cBGR2);
          for (int i = 0; i < size.width; i++, src += 4) {
            int t =
                descale(src[0] * cBGR0 + src[1] * cG + src[2] * cBGR2, SCALE);
            data[i] = (T)t;
          }
          src += rgba_step - size.width * 4;
        }
      } else if (img.channels() == 3) {
        int swap_rb = 0;
        int bgra_step = 0;
        int bgr_step = 0;
        _Size size(this->m_width, 1);
        for (; size.height--;) {
          for (int i = 0; i < size.width; i++, data += 3, src += 4) {
            uchar t0 = src[swap_rb], t1 = src[1];
            data[0] = (T)t0;
            data[1] = (T)t1;
            t0 = src[swap_rb ^ 2];
            data[2] = (T)t0;
          }
          data += bgr_step - size.width * 3;
          src += bgra_step - size.width * 4;
        }
      } else if (img.channels() == 4) {
        bool has_bit_mask = (m_rgba_bit_offset[0] >= 0) &&
                            (m_rgba_bit_offset[1] >= 0) &&
                            (m_rgba_bit_offset[2] >= 0);
        if (has_bit_mask) {
          for (int i = 0; i < this->m_width; i++, data += 4, src += 4) {
            uint _data = *((uint *)src);
            data[0] = (T)((m_rgba_mask[2] & _data) >> m_rgba_bit_offset[2]);
            data[1] = (T)((m_rgba_mask[1] & _data) >> m_rgba_bit_offset[1]);
            data[2] = (T)((m_rgba_mask[0] & _data) >> m_rgba_bit_offset[0]);
            if (m_rgba_bit_offset[3] >= 0)
              data[3] = (T)((m_rgba_mask[3] & _data) >> m_rgba_bit_offset[3]);
            else
              data[3] = (T)255;
          }
        } else {
          for (int i = 0; i < this->m_width * 4; i++) {
            data[i] = (T)src[i];
          }
        }
      }
    }
    result = true;
    break;
    /************************* 24 BPP ************************/
  case 24:
    for (y = 0; y < this->m_height; y++, data += step) {
      m_strm.getBytes(src, src_pitch);
      if (!color) {
        int i;
        int gray_step = 0;
        int bgr_step = 0;
        int _swap_rb = 0;
        _Size size(this->m_width, 1);
        for (; size.height--; data += gray_step) {
          short cBGR0 = cB;
          short cBGR2 = cR;
          if (_swap_rb)
            std::swap(cBGR0, cBGR2);
          for (i = 0; i < size.width; i++, src += 3) {
            int t =
                descale(src[0] * cBGR0 + src[1] * cG + src[2] * cBGR2, SCALE);
            data[i] = (T)t;
          }
          src += bgr_step - size.width * 3;
        }
      } else {
        for (int k = 0; k < this->m_width * 3; k++) {
          data[k] = (T)src[k];
        }
      }
    }
    result = true;
    break;

    /************************* 8 BPP ************************/
  case 8:
    if (m_rle_code == BMP_RGB) {
      for (y = 0; y < this->m_height; y++, data += step) {
        m_strm.getBytes(src, src_pitch);
        if (!color) {
          for (int i = 0; i < this->m_width; i++) {
            data[i] = (T)gray_palette[src[i]];
          }
        }
      }
    }
    result = true;
    break;
  }
  return result;
}

//////////////////////////////////////////////////////
template <typename T, size_t N>
std::unique_ptr<BaseImageEncoder<T, N>> BmpEncoder<T, N>::newEncoder() const {
  return std::make_unique<BmpEncoder<T, N>>();
}

template <typename T, size_t N> BmpEncoder<T, N>::BmpEncoder() {
  this->m_description = "Windows bitmap (*.bmp;*.dib)";
  this->m_buf_supported = true;
}

template <typename T, size_t N> BmpEncoder<T, N>::~BmpEncoder() {}

template <typename T, size_t N>
bool BmpEncoder<T, N>::write(const Img<T, N> &img, const std::vector<int> &) {
  int width = img.cols, height = img.rows, channels = img.channels();
  int fileStep = (width * channels + 3) & -4;
  uchar zeropad[] = "\0\0\0\0";
  WLByteStream strm;

  strm.open(this->m_filename);

  int bitmapHeaderSize = 40;
  int paletteSize = channels > 1 ? 0 : 1024;
  int headerSize = 14 /* fileheader */ + bitmapHeaderSize + paletteSize;
  size_t fileSize = (size_t)fileStep * height + headerSize;
  PaletteEntry palette[256];

  // write signature 'BM'
  strm.putBytes(fmtSignBmp, (int)strlen(fmtSignBmp));

  // write file header
  strm.putDWord(validateToInt(fileSize)); // file size
  strm.putDWord(0);
  strm.putDWord(headerSize);

  // write bitmap header
  strm.putDWord(bitmapHeaderSize);
  strm.putDWord(width);
  strm.putDWord(height);
  strm.putWord(1);
  strm.putWord(channels << 3);
  strm.putDWord(BMP_RGB);
  strm.putDWord(0);
  strm.putDWord(0);
  strm.putDWord(0);
  strm.putDWord(0);
  strm.putDWord(0);

  if (channels == 1) {
    FillGrayPalette(palette, 8);
    strm.putBytes(palette, sizeof(palette));
  }

  width *= channels;

  for (int y = height - 1; y >= 0; y--) {
    T *data = img.data + (y * width);
    // strm.putBytes(img.data + (y * width), width);
    for (int i = 0; i < width; i++) {
      strm.putByte((uchar)data[i]);
    }
    if (fileStep > width)
      strm.putBytes(zeropad, fileStep - width);
  }

  strm.close();
  return true;
}

#endif /*_GRFMT_BMP_H_*/
