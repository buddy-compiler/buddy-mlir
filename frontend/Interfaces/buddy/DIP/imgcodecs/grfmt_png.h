//===-----------------------grfmt_png.h----------------------------===//
//
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
// By downloading, copying, installing or using the software you agree to
// this license. If you do not agree to this license, do not download,
// install, copy or use the software.
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
//  * Redistribution's of source code must retain the above copyright notice,
//  this list of conditions and the following disclaimer.
//
//  * Redistribution's in binary form must reproduce the above copyright
//  notice,
//  this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
//  * The name of Intel Corporation may not be used to endorse or promote
//  products
//  derived from this software without specific prior written permission.
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
//----------------------------------------------------------------------//
//
// This file is modified from opencv's
// modules/imgcodecs/src/grfmt_png.hpp file
//
//----------------------------------------------------------------------//

#ifndef _GRFMT_PNG_H_
#define _GRFMT_PNG_H_
#ifndef _LFS64_LARGEFILE
#define _LFS64_LARGEFILE 0
#endif
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 0
#endif

#include "buddy/DIP/imgcodecs/bitstrm.h"
#include "buddy/DIP/imgcodecs/grfmt_base.h"
#include <png.h>
#include <zlib.h>

#if defined _MSC_VER && _MSC_VER >= 1200
// interaction between '_setjmp' and C++ object destruction is non-portable
#pragma warning(disable : 4611)
#endif

// the following defines are a hack to avoid multiple problems with frame
// pointer handling and setjmp see
// http://gcc.gnu.org/ml/gcc/2011-10/msg00324.html for some details
#define mingw_getsp(...) 0
#define __builtin_frame_address(...) 0

namespace dip {
template <typename T, size_t N>
class PngDecoder : public BaseImageDecoder<T, N> {
public:
  PngDecoder();
  virtual ~PngDecoder();
  bool readData(Img<T, N> &img);
  bool readHeader();
  void close();
  std::unique_ptr<BaseImageDecoder<T, N>> newDecoder() const;

protected:
  int m_bit_depth;
  void *m_png_ptr;  // pointer to decompression structure
  void *m_info_ptr; // pointer to image information structure
  void *m_end_info; // pointer to one more image information structure
  FILE *m_f;
  int m_color_type;
  size_t m_buf_pos;
};

template <typename T, size_t N>
class PngEncoder : public BaseImageEncoder<T, N> {
public:
  PngEncoder();
  virtual ~PngEncoder();
  bool write(Img<T, N> &img, const std::vector<int> &params);
  std::unique_ptr<BaseImageEncoder<T, N>> newEncoder() const;

protected:
  static void writeDataToBuf(void *png_ptr, uchar *src, size_t size);
};

inline bool isBigEndian() {
  int num = 1;
  char *ptr = (char *)&num;
  return (*ptr == 0);
}

/////////////////////// PngDecoder ///////////////////
template <typename T, size_t N> PngDecoder<T, N>::PngDecoder() {
  this->m_signature = "\x89\x50\x4e\x47\xd\xa\x1a\xa";
  m_color_type = 0;
  m_png_ptr = 0;
  m_info_ptr = m_end_info = 0;
  m_f = 0;
  this->m_buf_supported = true;
  m_buf_pos = 0;
  m_bit_depth = 0;
}

template <typename T, size_t N> PngDecoder<T, N>::~PngDecoder() { close(); }

template <typename T, size_t N>
std::unique_ptr<BaseImageDecoder<T, N>> PngDecoder<T, N>::newDecoder() const {
  return std::make_unique<PngDecoder<T, N>>();
}

template <typename T, size_t N> void PngDecoder<T, N>::close() {
  if (m_f) {
    fclose(m_f);
    m_f = 0;
  }
  if (m_png_ptr) {
    png_structp png_ptr = (png_structp)m_png_ptr;
    png_infop info_ptr = (png_infop)m_info_ptr;
    png_infop end_info = (png_infop)m_end_info;
    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
    m_png_ptr = m_info_ptr = m_end_info = 0;
  }
}

template <typename T, size_t N> bool PngDecoder<T, N>::readHeader() {
  volatile bool result = false;
  close();
  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
  if (png_ptr) {
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_infop end_info = png_create_info_struct(png_ptr);
    m_png_ptr = png_ptr;
    m_info_ptr = info_ptr;
    m_end_info = end_info;
    m_buf_pos = 0;
    if (info_ptr && end_info) {
      if (setjmp(png_jmpbuf(png_ptr)) == 0) {
        m_f = fopen(this->m_filename.c_str(), "rb");
        if (m_f)
          png_init_io(png_ptr, m_f);
        if (m_f) {
          png_uint_32 wdth, hght;
          int bit_depth, color_type, num_trans = 0;
          png_bytep trans;
          png_color_16p trans_values;
          png_read_info(png_ptr, info_ptr);
          png_get_IHDR(png_ptr, info_ptr, &wdth, &hght, &bit_depth, &color_type,
                       0, 0, 0);
          this->m_width = (int)wdth;
          this->m_height = (int)hght;
          m_color_type = color_type;
          m_bit_depth = bit_depth;
          if (bit_depth <= 8 || bit_depth == 16) {
            switch (color_type) {
            case PNG_COLOR_TYPE_RGB:
            case PNG_COLOR_TYPE_PALETTE:
              png_get_tRNS(png_ptr, info_ptr, &trans, &num_trans,
                           &trans_values);
              if (num_trans > 0)
                this->m_channels = 4;
              else
                this->m_channels = 3;
              break;
            case PNG_COLOR_TYPE_GRAY_ALPHA:
            case PNG_COLOR_TYPE_RGB_ALPHA:
              this->m_channels = 4;
              break;
            default:
              this->m_channels = 1;
            }
            result = true;
          }
        }
      }
    }
  }
  if (!result)
    close();
  return result;
}

template <typename T, size_t N>
bool PngDecoder<T, N>::readData(Img<T, N> &img) {
  volatile bool result = false;
  uchar **_buffer = new uchar *[this->m_height];
  uchar **buffer = _buffer;
  bool color = img.channels() > 1;
  T *data = img.getData();
  png_structp png_ptr = (png_structp)m_png_ptr;
  png_infop info_ptr = (png_infop)m_info_ptr;
  png_infop end_info = (png_infop)m_end_info;
  if (m_png_ptr && m_info_ptr && m_end_info && this->m_width &&
      this->m_height) {
    if (setjmp(png_jmpbuf(png_ptr)) == 0) {
      int y;

      if (m_bit_depth == 16)
        png_set_strip_16(png_ptr);
      else if (!isBigEndian())
        png_set_swap(png_ptr);

      if (img.channels() < 4) {
        /* observation: png_read_image() writes 400 bytes beyond
         * end of data when reading a 400x118 color png
         * "mpplus_sand.png".  OpenCV crashes even with demo
         * programs.  Looking at the loaded image I'd say we get 4
         * bytes per pixel instead of 3 bytes per pixel.  Test
         * indicate that it is a good idea to always ask for
         * stripping alpha..  18.11.2004 Axel Walthelm
         */
        png_set_strip_alpha(png_ptr);
      } else
        png_set_tRNS_to_alpha(png_ptr);

      if (m_color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png_ptr);

      if ((m_color_type & PNG_COLOR_MASK_COLOR) == 0 && m_bit_depth < 8)
#if (PNG_LIBPNG_VER_MAJOR * 10000 + PNG_LIBPNG_VER_MINOR * 100 +               \
         PNG_LIBPNG_VER_RELEASE >=                                             \
     10209) ||                                                                 \
    (PNG_LIBPNG_VER_MAJOR == 1 && PNG_LIBPNG_VER_MINOR == 0 &&                 \
     PNG_LIBPNG_VER_RELEASE >= 18)
        png_set_expand_gray_1_2_4_to_8(png_ptr);
#else
        png_set_gray_1_2_4_to_8(png_ptr);
#endif

      if ((m_color_type & PNG_COLOR_MASK_COLOR) && color)
        png_set_bgr(png_ptr); // convert RGB to BGR
      else if (color)
        png_set_gray_to_rgb(png_ptr); // Gray->RGB
      else
        png_set_rgb_to_gray(png_ptr, 1, 0.299, 0.587); // RGB->Gray

      png_set_interlace_handling(png_ptr);
      png_read_update_info(png_ptr, info_ptr);

      size_t step = this->m_width * img.channels();
      uchar *myArry = new uchar[img.getSize()];
      for (int y = 0; y < this->m_height; y++)
        buffer[y] = myArry + y * step;

      png_read_image(png_ptr, buffer);
      png_read_end(png_ptr, end_info);
      for (int i = 0; i < img.getSize(); i++) {
        data[i] = (T)myArry[i];
      }
      delete[] myArry;
      delete[] _buffer;
#ifdef PNG_eXIf_SUPPORTED
      png_uint_32 num_exif = 0;
      png_bytep exif = 0;

      // Exif info could be in info_ptr (intro_info) or end_info per
      // specification
      if (png_get_valid(png_ptr, info_ptr, PNG_INFO_eXIf))
        png_get_eXIf_1(png_ptr, info_ptr, &num_exif, &exif);
      else if (png_get_valid(png_ptr, end_info, PNG_INFO_eXIf))
        png_get_eXIf_1(png_ptr, end_info, &num_exif, &exif);

#endif
      result = true;
    }
  }
  close();
  return result;
}

/////////////////////// PngEncoder ///////////////////
template <typename T, size_t N> PngEncoder<T, N>::PngEncoder() {
  this->m_description = "Portable Network Graphics files (*.png)";
  this->m_buf_supported = true;
}

template <typename T, size_t N> PngEncoder<T, N>::~PngEncoder() {}

template <typename T, size_t N>
std::unique_ptr<BaseImageEncoder<T, N>> PngEncoder<T, N>::newEncoder() const {
  return std::make_unique<PngEncoder<T, N>>();
}

template <typename T, size_t N>
void PngEncoder<T, N>::writeDataToBuf(void *_png_ptr, uchar *src, size_t size) {
  if (size == 0)
    return;
  png_structp png_ptr = (png_structp)_png_ptr;
  PngEncoder *encoder = (PngEncoder *)(png_get_io_ptr(png_ptr));
  size_t cursz = encoder->m_buf->size();
  encoder->m_buf->resize(cursz + size);
  memcpy(&(*encoder->m_buf)[cursz], src, size);
}

template <typename T, size_t N>
bool PngEncoder<T, N>::write(Img<T, N> &img, const std::vector<int> &params) {
  png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
  png_infop info_ptr = 0;
  FILE *volatile f = 0;
  int y, width = img.getSizes()[1], height = img.getSizes()[0];
  int channels = img.channels();
  volatile bool result = false;
  uchar **buffer = new uchar *[height];
  T *data = img.getData();
  if (png_ptr) {
    info_ptr = png_create_info_struct(png_ptr);

    if (info_ptr) {
      if (setjmp(png_jmpbuf(png_ptr)) == 0) {

        f = fopen(this->m_filename.c_str(), "wb");
        if (f)
          png_init_io(png_ptr, (png_FILE_p)f);

        int compression_level =
            -1; // Invalid value to allow setting 0-9 as valid
        int compression_strategy = 3; // Default strategy
        bool isBilevel = false;

        if (f) {
          if (compression_level >= 0) {
            png_set_compression_level(png_ptr, compression_level);
          } else {
            // tune parameters for speed
            // (see http://wiki.linuxquestions.org/wiki/Libpng)
            png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_SUB);
            png_set_compression_level(png_ptr, Z_BEST_SPEED);
          }
          png_set_compression_strategy(png_ptr, compression_strategy);

          png_set_IHDR(png_ptr, info_ptr, width, height,
                       1 ? isBilevel ? 1 : 8 : 16,
                       channels == 1   ? PNG_COLOR_TYPE_GRAY
                       : channels == 3 ? PNG_COLOR_TYPE_RGB
                                       : PNG_COLOR_TYPE_RGBA,
                       PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                       PNG_FILTER_TYPE_DEFAULT);

          png_write_info(png_ptr, info_ptr);

          if (isBilevel)
            png_set_packing(png_ptr);

          png_set_bgr(png_ptr);
          if (!isBigEndian())
            png_set_swap(png_ptr);

          size_t step = width * img.channels();
          uchar *myArry = new uchar[img.getSize()];
          for (int i = 0; i < img.getSize(); i++) {
            myArry[i] = (uchar)data[i];
          }

          for (int y = 0; y < height; y++)
            buffer[y] = myArry + y * step;

          png_write_image(png_ptr, buffer);
          png_write_end(png_ptr, info_ptr);
          delete[] myArry;
          delete[] buffer;
          result = true;
        }
      }
    }
  }

  png_destroy_write_struct(&png_ptr, &info_ptr);
  if (f)
    fclose((FILE *)f);

  return result;
}
} // namespace dip
#endif /*_GRFMT_PNG_H_*/