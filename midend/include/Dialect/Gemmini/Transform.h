//===- Transform.h - MLIR Dialect for RISC-V Gemmmini extension ---------===//
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

#ifndef GEMMINI_TRANSLATE_H
#define GEMMINI_TRANSLATE_H
#define CONFIG_LD 1
#define CONFIG_ST 2
#define CONFIG_EX 0
#define DIM 16
#define ADDR_LEN 32
#define ACC_SCALE_IDENTITY 1.0
typedef uint32_t acc_scale_t_bits;
typedef float acc_scale_t;
typedef uint32_t scale_t_bits;
typedef float scale_t;

static acc_scale_t_bits acc_scale_t_to_acc_scale_t_bits(acc_scale_t x) {
  union {
    acc_scale_t_bits b;
    acc_scale_t f;
  } un;

  un.f = x;
  return un.b;
}

static scale_t_bits scale_t_to_scale_t_bits(scale_t x) {
  union {
    scale_t_bits b;
    scale_t f;
  } un;

  un.f = x;
  return un.b;
}

namespace mlir {

class LLVMConversionTarget;
class LLVMTypeConverter;
class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

void populateGemminiLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                  RewritePatternSet &patterns);
void configureGemminiegalizeForExportTarget(LLVMConversionTarget &target);

} // namespace mlir

#endif // GEMMINI_TRANSLATE_H
