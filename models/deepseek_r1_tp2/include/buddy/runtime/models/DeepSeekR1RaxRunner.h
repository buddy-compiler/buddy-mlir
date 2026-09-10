//===- DeepSeekR1RaxRunner.h - DeepSeek RAX execution entry ----*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_MODELS_DEEPSEEKR1RAXRUNNER_H
#define BUDDY_RUNTIME_MODELS_DEEPSEEKR1RAXRUNNER_H

#include <functional>
#include <string>

namespace buddy {
namespace runtime {

class DeepSeekR1RaxSession;

using DeepSeekR1RaxSessionCallback =
    std::function<void(DeepSeekR1RaxSession &, int rank)>;

/// Run a callback while the rank-local RAX session and MPI world are alive.
/// Each process opens the supplied rank-local RAX path.
void runDeepSeekR1Rax(const std::string &raxPath,
                      const DeepSeekR1RaxSessionCallback &callback);

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_DEEPSEEKR1RAXRUNNER_H
