//===- WhisperPreprocess.h ------------------------------------------------===//
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
// Header file for whisper preprocess operation in DAP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef FRONTEND_INTERFACES_BUDDY_DAP_DSP_WHISPERPREPROCESS
#define FRONTEND_INTERFACES_BUDDY_DAP_DSP_WHISPERPREPROCESS

#include "buddy/Core/Container.h"
#include "buddy/DAP/AudioContainer.h"
#include "buddy/DAP/DSP/IIRDesign.h"

namespace dap {
namespace detail {
// Declare the whisper preprocess C interface.
extern "C" {
// The original MLIR function:
//    ```mlir
//    func.func @buddy_whisperPreprocess(%in : memref<?xf64>) ->
//        memref<1x80x3000xf32>
//    ```
//
// After applying the '-llvm-request-c-wrappers' pass:
//    The result of the function (memref<1x80x3000xf32>) is modified to be the
//    first operand.
void _mlir_ciface_buddy_whisperPreprocess(MemRef<float, 3> *outputFeatures,
                                          MemRef<double, 1> *inputRawSpeech);

void _mlir_ciface_buddy_RFFT(MemRef<double, 1> *inputRawSpeech);

}
} // namespace detail

// Function for Whisper preprocess
void whisperPreprocess(MemRef<double, 1> *inputRawSpeech,
                       MemRef<float, 3> *outputFeatures) {
  detail::_mlir_ciface_buddy_whisperPreprocess(outputFeatures, inputRawSpeech);
}


void RFFT(MemRef<double, 1> *inputRawSpeech) {
  detail::_mlir_ciface_buddy_RFFT(inputRawSpeech);
}
} // namespace dap

#endif // FRONTEND_INTERFACES_BUDDY_DAP_DSP_WHISPERPREPROCESS
