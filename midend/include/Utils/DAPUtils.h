//====- DAPUtils.h --------------------------------------------------------===//
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
// This file defines DAP dialect specific utility functions for the buddy
// compiler ecosystem.
//
//===----------------------------------------------------------------------===//

#ifndef INCLUDE_UTILS_DAPUTILS_H
#define INCLUDE_UTILS_DAPUTILS_H

#include "Utils/Utils.h"
#include <stdarg.h>

using namespace mlir;

namespace buddy {
namespace dap {

// Generate 5 vector params from SOS matrices
SmallVector<Value, 5> generateSOSParams(OpBuilder &rewriter, Location loc,
                                        VectorType vectorTy, Value f0, Value f1,
                                        Value c0, Value c1, Value c2, Value c4,
                                        Value c5, Value filterSize,
                                        Value kernel);

// Processing iir operation, result are stored in output MemRef
void biquadProcess(OpBuilder &rewriter, Location loc, VectorType vectorTy,
                   Value f0, Value c0, Value c1, Value cUpperBound,
                   Value iUpperBound, SmallVector<Value, 5> SOSParams,
                   ArrayRef<int64_t> arrayRef, Value N, Value input,
                   Value output);

// Total process for a specific vector length iir vectorization process
void iirVectorizationProcess(OpBuilder &rewriter, Location loc, uint64_t vecLen,
                             FloatType floatType, Value f0, Value f1, Value c0,
                             Value c1, Value c2, Value c4, Value c5,
                             Value filterSize, Value kernel,
                             ArrayRef<int64_t> arrayRef, Value N, Value input,
                             Value output);

} // namespace dap
} // namespace buddy

#endif // INCLUDE_UTILS_DAPUTILS_H
