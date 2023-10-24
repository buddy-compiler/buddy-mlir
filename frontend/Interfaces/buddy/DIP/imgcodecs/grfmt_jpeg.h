//===-----------------------grfmt_jpeg.h----------------------------===//
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
// modules/imgcodecs/src/grfmt_jpeg.hpp file
//
//----------------------------------------------------------------------//

#ifndef _GRFMT_JPEG_H_
#define _GRFMT_JPEG_H_

#ifdef _MSC_VER
// interaction between '_setjmp' and C++ object destruction is non-portable
#pragma warning(disable : 4611)
#endif

#include <algorithm>
#include <setjmp.h>
#include <stdio.h>

// the following defines are a hack to avoid multiple problems with frame
// pointer handling and setjmp see
// http://gcc.gnu.org/ml/gcc/2011-10/msg00324.html for some details
#define mingw_getsp(...) 0
#define __builtin_frame_address(...) 0

#ifdef _WIN32

#define XMD_H // prevent redefinition of INT32
#undef FAR    // prevent FAR redefinition

#endif

#if defined _WIN32 && defined __GNUC__
typedef unsigned char boolean;
#endif

#undef FALSE
#undef TRUE

extern "C" {
#include "jpeglib.h"
}

#ifndef IMG_MANUAL_JPEG_STD_HUFF_TABLES
#if defined(LIBJPEG_TURBO_VERSION_NUMBER) &&                                   \
    LIBJPEG_TURBO_VERSION_NUMBER >= 1003090
#define IMG_MANUAL_JPEG_STD_HUFF_TABLES                                        \
  0 // libjpeg-turbo handles standard huffman tables itself (jstdhuff.c)
#else
#define IMG_MANUAL_JPEG_STD_HUFF_TABLES 1
#endif
#endif
#if IMG_MANUAL_JPEG_STD_HUFF_TABLES == 0
#undef IMG_MANUAL_JPEG_STD_HUFF_TABLES
#endif

#include "buddy/DIP/imgcodecs/bitstrm.h"
#include "buddy/DIP/imgcodecs/grfmt_base.h"

/* Miscellaneous useful macros */
#undef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#undef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
/**
 * @brief Jpeg markers that can be encountered in a Jpeg file
 */
enum AppMarkerTypes {
  SOI = 0xD8,
  SOF0 = 0xC0,
  SOF2 = 0xC2,
  DHT = 0xC4,
  DQT = 0xDB,
  DRI = 0xDD,
  SOS = 0xDA,
  RST0 = 0xD0,
  RST1 = 0xD1,
  RST2 = 0xD2,
  RST3 = 0xD3,
  RST4 = 0xD4,
  RST5 = 0xD5,
  RST6 = 0xD6,
  RST7 = 0xD7,
  APP0 = 0xE0,
  APP1 = 0xE1,
  APP2 = 0xE2,
  APP3 = 0xE3,
  APP4 = 0xE4,
  APP5 = 0xE5,
  APP6 = 0xE6,
  APP7 = 0xE7,
  APP8 = 0xE8,
  APP9 = 0xE9,
  APP10 = 0xEA,
  APP11 = 0xEB,
  APP12 = 0xEC,
  APP13 = 0xED,
  APP14 = 0xEE,
  APP15 = 0xEF,
  COM = 0xFE,
  EOI = 0xD9
};

namespace dip {
template <typename T, size_t N>
class JpegDecoder : public BaseImageDecoder<T, N> {
public:
  JpegDecoder();
  virtual ~JpegDecoder();

  bool readData(Img<T, N> &img);
  bool readHeader();
  void close();

  std::unique_ptr<BaseImageDecoder<T, N>> newDecoder() const;

protected:
  FILE *m_f;
  void *m_state;

private:
  JpegDecoder(const JpegDecoder &);            // copy disabled
  JpegDecoder &operator=(const JpegDecoder &); // assign disabled
};

template <typename T, size_t N>
class JpegEncoder : public BaseImageEncoder<T, N> {
public:
  JpegEncoder();
  virtual ~JpegEncoder();

  bool write(Img<T, N> &img, const std::vector<int> &params);
  std::unique_ptr<BaseImageEncoder<T, N>> newEncoder() const;
};

struct JpegErrorMgr {
  struct jpeg_error_mgr pub;
  jmp_buf setjmp_buffer;
};

struct JpegSource {
  struct jpeg_source_mgr pub;
  int skip;
};

struct JpegState {
  jpeg_decompress_struct cinfo; // IJG JPEG codec structure
  JpegErrorMgr jerr;            // error processing manager state
  JpegSource source;            // memory buffer source
};

/////////////////////// Error processing /////////////////////

METHODDEF(void)
stub(j_decompress_ptr) {}

METHODDEF(boolean)
fill_input_buffer(j_decompress_ptr) { return FALSE; }

// emulating memory input stream

METHODDEF(void)
skip_input_data(j_decompress_ptr cinfo, long num_bytes) {
  JpegSource *source = (JpegSource *)cinfo->src;

  if (num_bytes > (long)source->pub.bytes_in_buffer) {
    // We need to skip more data than we have in the buffer.
    // This will force the JPEG library to suspend decoding.
    source->skip = (int)(num_bytes - source->pub.bytes_in_buffer);
    source->pub.next_input_byte += source->pub.bytes_in_buffer;
    source->pub.bytes_in_buffer = 0;
  } else {
    // Skip portion of the buffer
    source->pub.bytes_in_buffer -= num_bytes;
    source->pub.next_input_byte += num_bytes;
    source->skip = 0;
  }
}

static void jpeg_buffer_src(j_decompress_ptr cinfo, JpegSource *source) {
  cinfo->src = &source->pub;

  // Prepare for suspending reader
  source->pub.init_source = stub;
  source->pub.fill_input_buffer = fill_input_buffer;
  source->pub.skip_input_data = skip_input_data;
  source->pub.resync_to_restart = jpeg_resync_to_restart;
  source->pub.term_source = stub;
  source->pub.bytes_in_buffer = 0; // forces fill_input_buffer on first read

  source->skip = 0;
}

METHODDEF(void)
error_exit(j_common_ptr cinfo) {
  JpegErrorMgr *err_mgr = (JpegErrorMgr *)(cinfo->err);

  /* Return control to the setjmp point */
  longjmp(err_mgr->setjmp_buffer, 1);
}

/************************ JPEG decoder *****************************/
template <typename T, size_t N> JpegDecoder<T, N>::JpegDecoder() {
  this->m_signature = "\xFF\xD8\xFF";
  m_state = 0;
  m_f = 0;
  this->m_buf_supported = true;
}

template <typename T, size_t N> JpegDecoder<T, N>::~JpegDecoder() { close(); }

template <typename T, size_t N> void JpegDecoder<T, N>::close() {
  if (m_state) {
    JpegState *state = (JpegState *)m_state;
    jpeg_destroy_decompress(&state->cinfo);
    delete state;
    m_state = 0;
  }

  if (m_f) {
    fclose(m_f);
    m_f = 0;
  }

  this->m_width = this->m_height = 0;
  this->m_channels = -1;
}

template <typename T, size_t N>
std::unique_ptr<BaseImageDecoder<T, N>> JpegDecoder<T, N>::newDecoder() const {
  return std::make_unique<JpegDecoder<T, N>>();
}

template <typename T, size_t N> bool JpegDecoder<T, N>::readHeader() {
  volatile bool result = false;
  close();

  JpegState *state = new JpegState;
  m_state = state;
  state->cinfo.err = jpeg_std_error(&state->jerr.pub);
  state->jerr.pub.error_exit = error_exit;

  if (setjmp(state->jerr.setjmp_buffer) == 0) {
    jpeg_create_decompress(&state->cinfo);

    m_f = fopen(this->m_filename.c_str(), "rb");
    if (m_f)
      jpeg_stdio_src(&state->cinfo, m_f);

    if (state->cinfo.src != 0) {
      jpeg_save_markers(&state->cinfo, APP1, 0xffff);
      jpeg_read_header(&state->cinfo, TRUE);

      state->cinfo.scale_num = 1;
      state->cinfo.scale_denom = this->m_scale_denom;
      this->m_scale_denom =
          1; // trick! to know which decoder used scale_denom see imread_
      jpeg_calc_output_dimensions(&state->cinfo);
      this->m_width = state->cinfo.output_width;
      this->m_height = state->cinfo.output_height;
      this->m_channels = state->cinfo.num_components > 1 ? 3 : 1;
      result = true;
    }
  }

  if (!result)
    close();

  return result;
}

#ifdef IMG_MANUAL_JPEG_STD_HUFF_TABLES
/***************************************************************************
 * following code is for supporting MJPEG image files
 * based on a message of Laurent Pinchart on the video4linux mailing list
 ***************************************************************************/

/* JPEG DHT Segment for YCrCb omitted from MJPEG data */
static unsigned char my_jpeg_odml_dht[0x1a4] = {
    0xff, 0xc4, 0x01, 0xa2,

    0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
    0x07, 0x08, 0x09, 0x0a, 0x0b,

    0x01, 0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
    0x07, 0x08, 0x09, 0x0a, 0x0b,

    0x10, 0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04,
    0x04, 0x00, 0x00, 0x01, 0x7d, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05,
    0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14,
    0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1,
    0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19,
    0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54,
    0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84,
    0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
    0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa,
    0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4,
    0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7,
    0xd8, 0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
    0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,

    0x11, 0x00, 0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04, 0x07, 0x05, 0x04,
    0x04, 0x00, 0x01, 0x02, 0x77, 0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05,
    0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32,
    0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52,
    0xf0, 0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1,
    0x17, 0x18, 0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37,
    0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53,
    0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67,
    0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82,
    0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95,
    0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8,
    0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2,
    0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5,
    0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8,
    0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa};

/*
 * Parse the DHT table.
 * This code comes from jpeg6b (jdmarker.c).
 */
static int my_jpeg_load_dht(struct jpeg_decompress_struct *info,
                            unsigned char *dht, JHUFF_TBL *ac_tables[],
                            JHUFF_TBL *dc_tables[]) {
  unsigned int length = (dht[2] << 8) + dht[3] - 2;
  unsigned int pos = 4;
  unsigned int count, i;
  int index;

  JHUFF_TBL **hufftbl;
  unsigned char bits[17];
  unsigned char huffval[256] = {0};

  while (length > 16) {
    bits[0] = 0;
    index = dht[pos++];
    count = 0;
    for (i = 1; i <= 16; ++i) {
      bits[i] = dht[pos++];
      count += bits[i];
    }
    length -= 17;

    if (count > 256 || count > length)
      return -1;

    for (i = 0; i < count; ++i)
      huffval[i] = dht[pos++];
    length -= count;

    if (index & 0x10) {
      index &= ~0x10;
      hufftbl = &ac_tables[index];
    } else
      hufftbl = &dc_tables[index];

    if (index < 0 || index >= NUM_HUFF_TBLS)
      return -1;

    if (*hufftbl == NULL)
      *hufftbl = jpeg_alloc_huff_table((j_common_ptr)info);
    if (*hufftbl == NULL)
      return -1;

    memcpy((*hufftbl)->bits, bits, sizeof(*hufftbl)->bits);
    memcpy((*hufftbl)->huffval, huffval, sizeof(*hufftbl)->huffval);
  }

  if (length != 0)
    return -1;

  return 0;
}

/***************************************************************************
 * end of code for supportting MJPEG image files
 * based on a message of Laurent Pinchart on the video4linux mailing list
 ***************************************************************************/
#endif // IMG_MANUAL_JPEG_STD_HUFF_TABLES

template <typename T, size_t N>
bool JpegDecoder<T, N>::readData(Img<T, N> &img) {
  volatile bool result = false;
  size_t step = this->m_width * img.channels();
  bool color = img.channels() > 1;

  if (m_state && this->m_width && this->m_height) {
    jpeg_decompress_struct *cinfo = &((JpegState *)m_state)->cinfo;
    JpegErrorMgr *jerr = &((JpegState *)m_state)->jerr;
    JSAMPARRAY buffer = 0;

    if (setjmp(jerr->setjmp_buffer) == 0) {
#ifdef IMG_MANUAL_JPEG_STD_HUFF_TABLES
      /* check if this is a mjpeg image format */
      if (cinfo->ac_huff_tbl_ptrs[0] == NULL &&
          cinfo->ac_huff_tbl_ptrs[1] == NULL &&
          cinfo->dc_huff_tbl_ptrs[0] == NULL &&
          cinfo->dc_huff_tbl_ptrs[1] == NULL) {
        /* yes, this is a mjpeg image format, so load the correct
        huffman table */
        my_jpeg_load_dht(cinfo, my_jpeg_odml_dht, cinfo->ac_huff_tbl_ptrs,
                         cinfo->dc_huff_tbl_ptrs);
      }
#endif

      if (color) {
        if (cinfo->num_components != 4) {
          cinfo->out_color_space = JCS_RGB;
          cinfo->out_color_components = 3;
        } else {
          cinfo->out_color_space = JCS_CMYK;
          cinfo->out_color_components = 4;
        }
      } else {
        if (cinfo->num_components != 4) {
          cinfo->out_color_space = JCS_GRAYSCALE;
          cinfo->out_color_components = 1;
        } else {
          cinfo->out_color_space = JCS_CMYK;
          cinfo->out_color_components = 4;
        }
      }
      // Check for Exif marker APP1
      jpeg_saved_marker_ptr exif_marker = NULL;
      jpeg_saved_marker_ptr cmarker = cinfo->marker_list;
      while (cmarker && exif_marker == NULL) {
        if (cmarker->marker == APP1)
          exif_marker = cmarker;

        cmarker = cmarker->next;
      }
      jpeg_start_decompress(cinfo);
      buffer = (*cinfo->mem->alloc_sarray)((j_common_ptr)cinfo, JPOOL_IMAGE,
                                           this->m_width * 4, 1);
      T *data = img.getData();
      for (; this->m_height--; data += step) {
        jpeg_read_scanlines(cinfo, buffer, 1);
        if (color) {
          if (cinfo->out_color_components == 3) {
            int bgr_step = 0;
            int rgb_step = 0;
            _Size size(this->m_width, 1);
            for (; size.height--;) {
              for (int i = 0; i < size.width; i++, buffer[0] += 3, data += 3) {
                uchar t0 = buffer[0][0], t1 = buffer[0][1], t2 = buffer[0][2];
                data[2] = (T)t0;
                data[1] = (T)t1;
                data[0] = (T)t2;
              }
              buffer[0] += bgr_step - size.width * 3;
              data += rgb_step - size.width * 3;
            }
          } else {
            int cmyk_step = 0;
            int bgr_step = 0;
            _Size size(this->m_width, 1);
            for (; size.height--;) {
              for (int i = 0; i < size.width; i++, data += 3, buffer[0] += 4) {
                int c = buffer[0][0], m = buffer[0][1], y = buffer[0][2],
                    k = buffer[0][3];
                c = k - ((255 - c) * k >> 8);
                m = k - ((255 - m) * k >> 8);
                y = k - ((255 - y) * k >> 8);
                data[2] = (T)c;
                data[1] = (T)m;
                data[0] = (T)y;
              }
              data += bgr_step - size.width * 3;
              buffer[0] += cmyk_step - size.width * 4;
            }
          }
        } else {
          if (cinfo->out_color_components == 1) {
            for (int i = 0; i < this->m_width; i++) {
              data[i] = (T)buffer[0][i];
            }
          } else {
            int cmyk_step = 0;
            int gray_step = 0;
            _Size size(this->m_width, 1);
            for (; size.height--;) {
              for (int i = 0; i < size.width; i++, buffer[0] += 4) {
                int c = buffer[0][0], m = buffer[0][1], y = buffer[0][2],
                    k = buffer[0][3];
                c = k - ((255 - c) * k >> 8);
                m = k - ((255 - m) * k >> 8);
                y = k - ((255 - y) * k >> 8);
                int t = descale(y * cB + m * cG + c * cR, SCALE);
                data[i] = (T)t;
              }
              data += gray_step;
              buffer[0] += cmyk_step - size.width * 4;
            }
          }
        }
      }
      result = true;
      jpeg_finish_decompress(cinfo);
    }
  }
  close();
  return result;
}

/////////////////////// JpegEncoder ///////////////////

struct JpegDestination {
  struct jpeg_destination_mgr pub;
  std::vector<uchar> *buf, *dst;
};

METHODDEF(void)
stub(j_compress_ptr) {}

METHODDEF(void)
term_destination(j_compress_ptr cinfo) {
  JpegDestination *dest = (JpegDestination *)cinfo->dest;
  size_t sz = dest->dst->size(),
         bufsz = dest->buf->size() - dest->pub.free_in_buffer;
  if (bufsz > 0) {
    dest->dst->resize(sz + bufsz);
    memcpy(&(*dest->dst)[0] + sz, &(*dest->buf)[0], bufsz);
  }
}

METHODDEF(boolean)
empty_output_buffer(j_compress_ptr cinfo) {
  JpegDestination *dest = (JpegDestination *)cinfo->dest;
  size_t sz = dest->dst->size(), bufsz = dest->buf->size();
  dest->dst->resize(sz + bufsz);
  memcpy(&(*dest->dst)[0] + sz, &(*dest->buf)[0], bufsz);

  dest->pub.next_output_byte = &(*dest->buf)[0];
  dest->pub.free_in_buffer = bufsz;
  return TRUE;
}

static void jpeg_buffer_dest(j_compress_ptr cinfo,
                             JpegDestination *destination) {
  cinfo->dest = &destination->pub;

  destination->pub.init_destination = stub;
  destination->pub.empty_output_buffer = empty_output_buffer;
  destination->pub.term_destination = term_destination;
}

template <typename T, size_t N> JpegEncoder<T, N>::JpegEncoder() {
  this->m_description = "JPEG files (*.jpeg;*.jpg;*.jpe)";
  this->m_buf_supported = true;
}

template <typename T, size_t N> JpegEncoder<T, N>::~JpegEncoder() {}

template <typename T, size_t N>
std::unique_ptr<BaseImageEncoder<T, N>> JpegEncoder<T, N>::newEncoder() const {
  return std::make_unique<JpegEncoder<T, N>>();
}

template <typename T, size_t N>
bool JpegEncoder<T, N>::write(Img<T, N> &img,
                              const std::vector<int> &params) {

  this->m_last_error.clear();

  struct fileWrapper {
    FILE *f;

    fileWrapper() : f(0) {}
    ~fileWrapper() {
      if (f)
        fclose(f);
    }
  };
  volatile bool result = false;
  fileWrapper fw;
  int width = img.getSizes()[1], height = img.getSizes()[0];
  std::vector<uchar> out_buf(1 << 12);

  uchar *_buffer = nullptr;
  uchar *buffer = nullptr;

  struct jpeg_compress_struct cinfo;
  JpegErrorMgr jerr;
  JpegDestination dest;

  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = error_exit;
  jpeg_create_compress(&cinfo);

  if (!this->m_buf) {
    fw.f = fopen(this->m_filename.c_str(), "wb");
    if (!fw.f)
      goto _exit_;
    jpeg_stdio_dest(&cinfo, fw.f);
  } else {
    dest.dst = this->m_buf;
    dest.buf = &out_buf;

    jpeg_buffer_dest(&cinfo, &dest);

    dest.pub.next_output_byte = &out_buf[0];
    dest.pub.free_in_buffer = out_buf.size();
  }

  if (setjmp(jerr.setjmp_buffer) == 0) {
    cinfo.image_width = width;
    cinfo.image_height = height;

    int _channels = img.channels();
    int channels = _channels > 1 ? 3 : 1;
    cinfo.input_components = channels;
    cinfo.in_color_space = channels > 1 ? JCS_RGB : JCS_GRAYSCALE;

    int quality = 95;
    int progressive = 0;
    int optimize = 0;
    int rst_interval = 0;
    int luma_quality = -1;
    int chroma_quality = -1;

    jpeg_set_defaults(&cinfo);
    cinfo.restart_interval = rst_interval;

    jpeg_set_quality(&cinfo, quality, TRUE /* limit to baseline-JPEG values */);
    if (progressive)
      jpeg_simple_progression(&cinfo);
    if (optimize)
      cinfo.optimize_coding = TRUE;

#if JPEG_LIB_VERSION >= 70
    if (luma_quality >= 0 && chroma_quality >= 0) {
      cinfo.q_scale_factor[0] = jpeg_quality_scaling(luma_quality);
      cinfo.q_scale_factor[1] = jpeg_quality_scaling(chroma_quality);
      if (luma_quality != chroma_quality) {
        /* disable subsampling - ref. Libjpeg.txt */
        cinfo.comp_info[0].v_samp_factor = 1;
        cinfo.comp_info[0].h_samp_factor = 1;
        cinfo.comp_info[1].v_samp_factor = 1;
        cinfo.comp_info[1].h_samp_factor = 1;
      }
      jpeg_default_qtables(&cinfo, TRUE);
    }
#endif // #if JPEG_LIB_VERSION >= 70

    jpeg_start_compress(&cinfo, TRUE);

    //_buffer.allocate(width * channels);
    _buffer = new uchar[width * channels];
    buffer = _buffer;
    int step = width * img.channels();

    for (int y = 0; y < height; y++) {
      T *data = img.getData() + step * y;
      uchar *ptr = nullptr;

      if (_channels == 3) {
        // icvCvt_BGR2RGB_8u_C3R(data, 0, buffer, 0, _Size(width, 1));
        int bgr_step = 0;
        int rgb_step = 0;
        _Size size(width, 1);
        for (; size.height--;) {
          for (int i = 0; i < size.width; i++, data += 3, buffer += 3) {
            uchar t0 = (uchar)data[0], t1 = (uchar)data[1], t2 = (uchar)data[2];
            buffer[2] = t0;
            buffer[1] = t1;
            buffer[0] = t2;
          }
          data += bgr_step - size.width * 3;
          buffer += rgb_step - size.width * 3;
        }
        ptr = buffer;

      } else if (_channels == 4) {
        // icvCvt_BGRA2BGR_8u_C4C3R(data, 0, buffer, 0, _Size(width, 1), 2);
        int swap_rb = 2;
        int bgra_step = 0;
        int bgr_step = 0;
        _Size size(width, 1);
        for (; size.height--;) {
          for (int i = 0; i < size.width; i++, buffer += 3, data += 4) {
            uchar t0 = (uchar)data[swap_rb], t1 = (uchar)data[1];
            buffer[0] = t0;
            buffer[1] = t1;
            t0 = (uchar)data[swap_rb ^ 2];
            buffer[2] = t0;
          }
          buffer += bgr_step - size.width * 3;
          data += bgra_step - size.width * 4;
        }
        ptr = buffer;
      } else if (_channels == 1) {
        for (int i = 0; i < width; i++) {
          buffer[i] = (uchar)data[i];
        }
        ptr = buffer;
      }

      jpeg_write_scanlines(&cinfo, &ptr, 1);
    }

    jpeg_finish_compress(&cinfo);
    result = true;
  }

_exit_:

  if (!result) {
    char jmsg_buf[JMSG_LENGTH_MAX];
    jerr.pub.format_message((j_common_ptr)&cinfo, jmsg_buf);
    this->m_last_error = jmsg_buf;
  }

  jpeg_destroy_compress(&cinfo);

  return result;
}
} // namespace dip
#endif /*_GRFMT_JPEG_H_*/