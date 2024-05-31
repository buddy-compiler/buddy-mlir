//===- f16test-main.cpp -----------------------------------------------------===//
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

#include <buddy/Core/Container.h>
#include <string>
#include <iostream>
#include <filesystem>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cassert>

using fp16_t = uint16_t;
using bp16_t = uint16_t;

extern "C" void _mlir_ciface_forward(MemRef<float, 1> *, MemRef<bp16_t, 1> *, MemRef<fp16_t, 1> *);

// ----------------- FP16/BP16 <-> F32 conversion utils & builtins ----------------------
extern "C" fp16_t __gnu_f2h_ieee(float f);
extern "C" float __gnu_h2f_ieee(fp16_t hf);
extern "C" bp16_t __truncsfbf2(float f);

static fp16_t float2half(float f) {
  return __gnu_f2h_ieee(f);
}
static float half2float(fp16_t hf) {
   return __gnu_h2f_ieee(hf);
}

static bp16_t float2bfloat(float floatValue) {
  return __truncsfbf2(floatValue);
}

// Converts the 16 bit representation of a bfloat value to a float value. This
// implementation is adapted from Eigen.
union Float32Bits {
  uint32_t u;
  float f;
};
float bfloat2float(bp16_t bfloatBits) {
  Float32Bits floatBits;
  floatBits.u = static_cast<uint32_t>(bfloatBits) << 16;
  return floatBits.f;
}

static std::string fp16Container2string(const MemRef<fp16_t, 1>& container, int n) {
  std::ostringstream ss;
  for (int i = 0; i < n; i++) {
    ss << half2float(container[i]) << ((i < n - 1) ? " " : "");
  }
  return ss.str();
}

static std::string bp16Container2string(const MemRef<bp16_t, 1>& container, int n) {
  std::ostringstream ss;
  for (int i = 0; i < n; i++) {
    ss << bfloat2float(container[i]) << ((i < n - 1) ? " " : "");
  }
  return ss.str();
}

static std::string f32Container2string(const MemRef<float, 1>& container, int n) {
  std::ostringstream ss;
  for (int i = 0; i < n; i++) {
    ss << container[i] << ((i < n - 1) ? " " : "");
  }
  return ss.str();
}

// ---------------- 

int main() {

  constexpr int kSize = 5;

  /// Initialize data containers
  bp16_t indata1[kSize] = {0};
  bp16_t indata2[kSize] = {0};
  for (int i = 0; i < kSize; i++) {
    indata1[i] = float2bfloat(3.f);
    indata2[i] = float2half(4.f);
  }
  long sizes[1] = {kSize};
  MemRef<bp16_t, 1> inputContainer1((bp16_t*)indata1, sizes, 0l);
  MemRef<fp16_t, 1> inputContainer2((fp16_t*)indata2, sizes, 0l);
  MemRef<float, 1> resultContainer(sizes);

  // check input
  std::cout << "Input 1: " << bp16Container2string(inputContainer1, kSize) << std::endl;
  std::cout << "Input 2: " << fp16Container2string(inputContainer2, kSize) << std::endl;

  // Execute the forward pass of the model.
  std::cout << "Perform squared sum" << std::endl;
  _mlir_ciface_forward(&resultContainer, &inputContainer1, &inputContainer2);

  // check output
  std::cout << "Output: " << f32Container2string(resultContainer, kSize) << std::endl;
  // should be all 25.0
  for (int i = 0; i < kSize; i++) {
    double error = std::fabs(float(resultContainer[i]) - 25.0f);
    assert (error < 1e-5);
  }

  return 0;
}