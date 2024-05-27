//===- Bitstrm.h ---------------------------------------------------===//
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
// This file is modified from opencv's modules/imgcodecs/src/bitstrm.hpp file
//
//===----------------------------------------------------------------------===//

#ifndef _BITSTRM_H_
#define _BITSTRM_H_

#include "buddy/DIP/ImageContainer.h"
#include "buddy/DIP/imgcodecs/replenishment.h"
#include "buddy/DIP/imgcodecs/utils.h"
#include <stdio.h>
#include <string.h>

namespace dip {
// class RBaseStream - base class for other reading streams.
template <typename T, size_t N> class RBaseStream {
public:
  RBaseStream();
  virtual ~RBaseStream();
  virtual bool open(const String &filename);
  virtual void close();
  bool isOpened();
  void setPos(int pos);
  int getPos();
  void skip(int bytes);

protected:
  bool m_allocated;
  uchar *m_start;
  uchar *m_end;
  uchar *m_current;
  FILE *m_file;
  int m_block_size;
  int m_block_pos;
  bool m_is_opened;

  virtual void readBlock();
  virtual void release();
  virtual void allocate();
};

// class RLByteStream - uchar-oriented stream.
// l in prefix means that the least significant uchar of a multi-uchar value
// goes first
template <typename T, size_t N> class RLByteStream : public RBaseStream<T, N> {
public:
  virtual ~RLByteStream();

  int getByte();
  int getBytes(void *buffer, int count);
  int getWord();
  int getDWord();
};

// class RMBitStream - uchar-oriented stream.
// m in prefix means that the most significant uchar of a multi-uchar value go
// first
template <typename T, size_t N> class RMByteStream : public RLByteStream<T, N> {
public:
  virtual ~RMByteStream();

  int getWord();
  int getDWord();
};

// WBaseStream - base class for output streams
class WBaseStream {
public:
  // methods
  inline WBaseStream();
  inline virtual ~WBaseStream();

  inline virtual bool open(const String &filename);
  inline virtual bool open(std::vector<uchar> &buf);
  inline virtual void close();
  inline bool isOpened();
  inline int getPos();

protected:
  uchar *m_start;
  uchar *m_end;
  uchar *m_current;
  int m_block_size;
  int m_block_pos;
  FILE *m_file;
  bool m_is_opened;
  std::vector<uchar> *m_buf;

  inline virtual void writeBlock();
  inline virtual void release();
  inline virtual void allocate();
};

// class WLByteStream - uchar-oriented stream.
// l in prefix means that the least significant uchar of a multi-byte value goes
// first
class WLByteStream : public WBaseStream {
public:
  inline virtual ~WLByteStream();

  inline void putByte(int val);
  inline void putBytes(const void *buffer, int count);
  inline void putWord(int val);
  inline void putDWord(int val);
};

// class WLByteStream - uchar-oriented stream.
// m in prefix means that the least significant uchar of a multi-byte value goes
// last
class WMByteStream : public WLByteStream {
public:
  inline virtual ~WMByteStream();
  inline void putWord(int val);
  inline void putDWord(int val);
};

inline unsigned BSWAP(unsigned v) {
  return (v << 24) | ((v & 0xff00) << 8) | ((v >> 8) & 0xff00) |
         ((unsigned)v >> 24);
}

const int BS_DEF_BLOCK_SIZE = 1 << 15;

inline bool bsIsBigEndian(void) {
  return (((const int *)"\0\x1\x2\x3\x4\x5\x6\x7")[0] & 255) != 0;
}

/////////////////////////  RBaseStream ////////////////////////////
template <typename T, size_t N> bool RBaseStream<T, N>::isOpened() {
  return m_is_opened;
}

template <typename T, size_t N> void RBaseStream<T, N>::allocate() {
  if (!m_allocated) {
    m_start = new uchar[m_block_size];
    m_end = m_start + m_block_size;
    m_current = m_end;
    m_allocated = true;
  }
}

template <typename T, size_t N> RBaseStream<T, N>::RBaseStream() {
  m_start = m_end = m_current = 0;
  m_file = 0;
  m_block_pos = 0;
  m_block_size = BS_DEF_BLOCK_SIZE;
  m_is_opened = false;
  m_allocated = false;
}

template <typename T, size_t N> RBaseStream<T, N>::~RBaseStream() {
  close();
  release();
}

template <typename T, size_t N> void RBaseStream<T, N>::readBlock() {
  setPos(getPos());

  if (m_file == 0) {
    if (m_block_pos == 0 && m_current < m_end)
      return;
  }

  fseek(m_file, m_block_pos, SEEK_SET);
  size_t readed = fread(m_start, 1, m_block_size, m_file);
  m_end = m_start + readed;

  if (readed == 0 || m_current >= m_end)
    return;
}

template <typename T, size_t N>
bool RBaseStream<T, N>::open(const String &filename) {
  close();
  allocate();

  m_file = fopen(filename.c_str(), "rb");
  if (m_file) {
    m_is_opened = true;
    setPos(0);
    readBlock();
  }
  return m_file != 0;
}

template <typename T, size_t N> void RBaseStream<T, N>::close() {
  if (m_file) {
    fclose(m_file);
    m_file = 0;
  }
  m_is_opened = false;
  if (!m_allocated)
    m_start = m_end = m_current = 0;
}

template <typename T, size_t N> void RBaseStream<T, N>::release() {
  if (m_allocated)
    delete[] m_start;
  m_start = m_end = m_current = 0;
  m_allocated = false;
}

template <typename T, size_t N> void RBaseStream<T, N>::setPos(int pos) {
  if (!m_file) {
    m_current = m_start + pos;
    m_block_pos = 0;
    return;
  }

  int offset = pos % m_block_size;
  int old_block_pos = m_block_pos;
  m_block_pos = pos - offset;
  m_current = m_start + offset;
  if (old_block_pos != m_block_pos)
    readBlock();
}

template <typename T, size_t N> int RBaseStream<T, N>::getPos() {
  int pos = dip::validateToInt((m_current - m_start) + m_block_pos);
  return pos;
}

template <typename T, size_t N> void RBaseStream<T, N>::skip(int bytes) {
  assert(bytes >= 0);
  uchar *old = m_current;
  m_current += bytes;
  assert(m_current >= old);
}

/////////////////////////  RLByteStream ////////////////////////////

template <typename T, size_t N> RLByteStream<T, N>::~RLByteStream() {}

template <typename T, size_t N> int RLByteStream<T, N>::getByte() {
  uchar *current = this->m_current;
  int val;

  if (current >= this->m_end) {
    this->readBlock();
    current = this->m_current;
  }
  val = *((uchar *)current);
  this->m_current = current + 1;
  return val;
}

template <typename T, size_t N>
int RLByteStream<T, N>::getBytes(void *buffer, int count) {
  uchar *data = (uchar *)buffer;
  int readed = 0;
  assert(count >= 0);
  while (count > 0) {
    int l;
    for (;;) {
      l = (int)(this->m_end - this->m_current);
      if (l > count)
        l = count;
      if (l > 0)
        break;
      this->readBlock();
    }
    memcpy(data, this->m_current, l);
    this->m_current += l;
    data += l;
    count -= l;
    readed += l;
  }
  return readed;
}

////////////  RLByteStream & RMByteStream <Get[d]word>s ////////////////
template <typename T, size_t N> RMByteStream<T, N>::~RMByteStream() {}

template <typename T, size_t N> int RLByteStream<T, N>::getWord() {
  uchar *current = this->m_current;
  int val;

  if (current + 1 < this->m_end) {
    val = current[0] + (current[1] << 8);
    this->m_current = current + 2;
  } else {
    val = getByte();
    val |= getByte() << 8;
  }
  return val;
}

template <typename T, size_t N> int RLByteStream<T, N>::getDWord() {
  uchar *current = this->m_current;
  int val;

  if (current + 3 < this->m_end) {
    val = current[0] + (current[1] << 8) + (current[2] << 16) +
          (current[3] << 24);
    this->m_current = current + 4;
  } else {

    val = getByte();
    val |= getByte() << 8;
    val |= getByte() << 16;
    val |= getByte() << 24;
  }
  return val;
}

template <typename T, size_t N> int RMByteStream<T, N>::getWord() {
  uchar *current = this->m_current;
  int val;

  if (current + 1 < this->m_end) {
    val = (current[0] << 8) + current[1];
    this->m_current = current + 2;
  } else {
    val = this->getByte() << 8;
    val |= this->getByte();
  }
  return val;
}

template <typename T, size_t N> int RMByteStream<T, N>::getDWord() {
  uchar *current = this->m_current;
  int val;

  if (current + 3 < this->m_end) {
    val = (current[0] << 24) + (current[1] << 16) + (current[2] << 8) +
          current[3];
    this->m_current = current + 4;
  } else {
    val = this->getByte() << 24;
    val |= this->getByte() << 16;
    val |= this->getByte() << 8;
    val |= this->getByte();
  }
  return val;
}

/////////////////////////// WBaseStream /////////////////////////////////
// WBaseStream - base class for output streams
WBaseStream::WBaseStream() {
  m_start = m_end = m_current = 0;
  m_file = 0;
  m_block_pos = 0;
  m_block_size = BS_DEF_BLOCK_SIZE;
  m_is_opened = false;
  m_buf = 0;
}

WBaseStream::~WBaseStream() {
  close();
  release();
}

bool WBaseStream::isOpened() { return m_is_opened; }

void WBaseStream::allocate() {
  if (!m_start)
    m_start = new uchar[m_block_size];

  m_end = m_start + m_block_size;
  m_current = m_start;
}

void WBaseStream::writeBlock() {
  int size = (int)(m_current - m_start);

  assert(isOpened());
  if (size == 0)
    return;

  if (m_buf) {
    size_t sz = m_buf->size();
    m_buf->resize(sz + size);
    memcpy(&(*m_buf)[sz], m_start, size);
  } else {
    fwrite(m_start, 1, size, m_file);
  }
  m_current = m_start;
  m_block_pos += size;
}

bool WBaseStream::open(const String &filename) {
  close();
  allocate();

  m_file = fopen(filename.c_str(), "wb");
  if (m_file) {
    m_is_opened = true;
    m_block_pos = 0;
    m_current = m_start;
  }
  return m_file != 0;
}

bool WBaseStream::open(std::vector<uchar> &buf) {
  close();
  allocate();

  m_buf = &buf;
  m_is_opened = true;
  m_block_pos = 0;
  m_current = m_start;

  return true;
}

void WBaseStream::close() {
  if (m_is_opened)
    writeBlock();
  if (m_file) {
    fclose(m_file);
    m_file = 0;
  }
  m_buf = 0;
  m_is_opened = false;
}

void WBaseStream::release() {
  if (m_start)
    delete[] m_start;
  m_start = m_end = m_current = 0;
}

int WBaseStream::getPos() {
  assert(isOpened());
  return m_block_pos + (int)(m_current - m_start);
}

///////////////////////////// WLByteStream ///////////////////////////////////
WLByteStream::~WLByteStream() {}

void WLByteStream::putByte(int val) {
  *m_current++ = (uchar)val;
  if (m_current >= m_end)
    writeBlock();
}

void WLByteStream::putBytes(const void *buffer, int count) {
  uchar *data = (uchar *)buffer;

  assert(data && m_current && count >= 0);

  while (count) {
    int l = (int)(m_end - m_current);
    if (l > count)
      l = count;
    if (l > 0) {
      memcpy(m_current, data, l);
      m_current += l;
      data += l;
      count -= l;
    }
    if (m_current == m_end)
      writeBlock();
  }
}

void WLByteStream::putWord(int val) {
  uchar *current = m_current;

  if (current + 1 < m_end) {
    current[0] = (uchar)val;
    current[1] = (uchar)(val >> 8);
    m_current = current + 2;
    if (m_current == m_end)
      writeBlock();
  } else {
    putByte(val);
    putByte(val >> 8);
  }
}

void WLByteStream::putDWord(int val) {
  uchar *current = m_current;

  if (current + 3 < m_end) {
    current[0] = (uchar)val;
    current[1] = (uchar)(val >> 8);
    current[2] = (uchar)(val >> 16);
    current[3] = (uchar)(val >> 24);
    m_current = current + 4;
    if (m_current == m_end)
      writeBlock();
  } else {
    putByte(val);
    putByte(val >> 8);
    putByte(val >> 16);
    putByte(val >> 24);
  }
}

///////////////////////////// WMByteStream ///////////////////////////////////
WMByteStream::~WMByteStream() {}

void WMByteStream::putWord(int val) {
  uchar *current = m_current;

  if (current + 1 < m_end) {
    current[0] = (uchar)(val >> 8);
    current[1] = (uchar)val;
    m_current = current + 2;
    if (m_current == m_end)
      writeBlock();
  } else {
    putByte(val >> 8);
    putByte(val);
  }
}

void WMByteStream::putDWord(int val) {
  uchar *current = m_current;

  if (current + 3 < m_end) {
    current[0] = (uchar)(val >> 24);
    current[1] = (uchar)(val >> 16);
    current[2] = (uchar)(val >> 8);
    current[3] = (uchar)val;
    m_current = current + 4;
    if (m_current == m_end)
      writeBlock();
  } else {
    putByte(val >> 24);
    putByte(val >> 16);
    putByte(val >> 8);
    putByte(val);
  }
}
} // namespace dip
#endif /*_BITSTRM_H_*/
