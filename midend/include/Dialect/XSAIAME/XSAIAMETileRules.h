//===- XSAIAMETileRules.h - Complete XSAIAME matrix tiles -------*- C++ -*-===//
//
// Part of the Buddy Compiler project, under the Apache License v2.0.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// The only tile shape the matrix unit computes correctly. The CUTE matrix unit
// traps only for extents above the maximum; smaller extents deadlock (K below
// 32 bytes, M/N = 0), corrupt other rows of C (N < 128) or compute wrong
// results (K = 32 bytes) without any exception. Every mma must therefore be a
// complete tile.
//
//===----------------------------------------------------------------------===//

#ifndef XSAIAME_XSAIAMETILERULES_H
#define XSAIAME_XSAIAMETILERULES_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

#include <cstdint>

namespace buddy::xsaiame {

inline constexpr int64_t kTileM = 128;
inline constexpr int64_t kTileN = 128;
// One A/B operand row: K * element bits.
inline constexpr int64_t kTileKBits = 512;

/// Bits of an A/B element token mma accepts, or 0.
inline unsigned getInputElementBits(llvm::StringRef token) {
  return llvm::StringSwitch<unsigned>(token)
      .Cases({"i8", "fp8e4m3", "fp8e5m2"}, 8)
      .Cases({"fp16", "bf16"}, 16)
      .Default(0);
}

/// K extent of a complete tile, or 0 for an unsupported input token.
inline int64_t getTileK(llvm::StringRef input) {
  unsigned bits = getInputElementBits(input);
  return bits ? kTileKBits / bits : 0;
}

} // namespace buddy::xsaiame

#endif // XSAIAME_XSAIAMETILERULES_H
