// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: buddy-external-kernel-test | FileCheck %s

#include "buddy/runtime/core/ExternalKernel.h"

#include <algorithm>
#include <iostream>
#include <utility>

using buddy::runtime::ExternalKernel;
using buddy::runtime::KernelPath;
static int calls = 0;
static int32_t fallback(const BuddyKernelCallV1 *call) {
  ++calls;
  std::fill_n(static_cast<float *>(call->output.data), 8, 7.f);
  return BUDDY_KERNEL_OK;
}
static int32_t failedFallback(const BuddyKernelCallV1 *) {
  return BUDDY_KERNEL_FAILED;
}

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "failed line " << __LINE__ << ": " << #condition << '\n';   \
      return 1;                                                                \
    }                                                                          \
  } while (false)

int main() {
  float input[8] = {}, output[8] = {};
  int64_t shape[] = {2, 4}, strides[] = {4, 1}, axis = -1;
  BuddyKernelCallV1 call{BUDDY_KERNEL_ABI_V1,
                         sizeof(BuddyKernelCallV1),
                         BUDDY_KERNEL_SOFTMAX,
                         0,
                         {input, sizeof(input), 2, BUDDY_KERNEL_F32,
                          BUDDY_KERNEL_ROW_MAJOR, 0, shape, strides},
                         {output, sizeof(output), 2, BUDDY_KERNEL_F32,
                          BUDDY_KERNEL_ROW_MAJOR, 0, shape, strides},
                         &axis,
                         1,
                         nullptr,
                         0};
  ExternalKernel valid(TEST_VALID);
  CHECK(valid.diagnostic().empty());
  auto result = valid.dispatch(call, fallback);
  CHECK(result.status == 0 && result.path == KernelPath::External);
  CHECK(output[0] == 42 && calls == 0);

  // Serialized requirement tokens are normalized before the comparison: order,
  // case and surrounding whitespace must not change the decision, and tokens
  // match exactly (never as substrings).
  CHECK(valid.satisfiesRequirements(""));
  CHECK(valid.satisfiesRequirements("baseline"));
  CHECK(valid.satisfiesRequirements("  BASEline ; "));
  CHECK(!valid.satisfiesRequirements("baseline,avx2"));
  ExternalKernel features(TEST_FEATURE_PLUGIN);
  CHECK(features.descriptorInfo() != nullptr);
  CHECK(features.satisfiesRequirements("fma, avx2"));
  CHECK(features.satisfiesRequirements("AVX2;BMI;FMA"));
  CHECK(features.satisfiesRequirements("avx2,avx2,fma"));
  CHECK(!features.satisfiesRequirements("avx2,sse2"));
  CHECK(!features.satisfiesRequirements("avx"));
  result = features.dispatch(call, fallback);
  CHECK(result.path == KernelPath::External && output[0] == 42 && calls == 0);

  const char *rejected[] = {"",
                            TEST_VALID ".missing",
                            TEST_BAD_ABI,
                            TEST_BAD_TARGET,
                            TEST_BAD_HOST,
                            TEST_NO_SYMBOL};
  for (const char *path : rejected) {
    ExternalKernel kernel(path);
    CHECK(!kernel.diagnostic().empty());
    // A handle that failed to load cannot satisfy any recorded requirement.
    CHECK(!kernel.satisfiesRequirements(""));
    result = kernel.dispatch(call, fallback);
    CHECK(result.status == 0 && result.path == KernelPath::Fallback);
    CHECK(result.externalStatus == BUDDY_KERNEL_UNSUPPORTED && output[0] == 7);
  }
  CHECK(calls == 6);
  ExternalKernel failed(TEST_FAILED);
  result = failed.dispatch(call, fallback);
  CHECK(result.path == KernelPath::Fallback && output[0] == 7);
  CHECK(result.externalStatus == BUDDY_KERNEL_FAILED && calls == 7);
  CHECK(std::all_of(output, output + 8,
                    [](float value) { return value == 7.f; }));
  result = failed.dispatch(call, failedFallback);
  CHECK(result.status == BUDDY_KERNEL_FAILED);
  result = failed.dispatch(call, nullptr);
  CHECK(result.status == BUDDY_KERNEL_FAILED &&
        result.path == KernelPath::None);

  shape[0] = 1;
  result = valid.dispatch(call, fallback);
  CHECK(result.path == KernelPath::Fallback && calls == 8);
  shape[0] = 2;
  auto invalid = call;
  invalid.output.byteSize = 4;
  CHECK(valid.dispatch(invalid, fallback).path == KernelPath::None);
  invalid = call;
  invalid.output.data = input + 1;
  CHECK(valid.dispatch(invalid, fallback).status == BUDDY_KERNEL_INVALID);
  invalid = call;
  invalid.abiVersion = 999;
  CHECK(valid.dispatch(invalid, fallback).status == BUDDY_KERNEL_INVALID);
  strides[1] = 2;
  CHECK(valid.dispatch(call, fallback).status == BUDDY_KERNEL_INVALID);
  strides[1] = 1;
  axis = 0;
  CHECK(valid.dispatch(call, fallback).status == BUDDY_KERNEL_INVALID);
  CHECK(calls == 8);
  axis = -1;
  CHECK(!ExternalKernel::hasFeatureRequirements(" , ; "));
  CHECK(ExternalKernel::hasFeatureRequirements(" BASELINE ; "));
  for (const char *path :
       {TEST_NULL_QUERY, TEST_BAD_SIZE, TEST_NO_CALLBACK, TEST_NO_INVOKE}) {
    ExternalKernel rejected(path);
    CHECK(rejected.descriptorInfo() == nullptr);
    result = rejected.dispatch(call, fallback);
    CHECK(result.status == BUDDY_KERNEL_OK &&
          result.path == KernelPath::Fallback);
  }
  ExternalKernel shapeMismatch(TEST_SHAPE);
  CHECK(shapeMismatch.descriptorInfo() && !shapeMismatch.matches(call));
  result = shapeMismatch.dispatch(call, fallback);
  CHECK(result.path == KernelPath::Fallback);
  for (auto fixture : {std::pair{TEST_UNSUPPORTED, BUDDY_KERNEL_UNSUPPORTED},
                       std::pair{TEST_INVALID, BUDDY_KERNEL_INVALID}}) {
    ExternalKernel rejected(fixture.first);
    result = rejected.dispatch(call, fallback);
    CHECK(result.path == KernelPath::Fallback &&
          result.status == BUDDY_KERNEL_OK &&
          result.externalStatus == fixture.second);
  }
  // CHECK: External kernel dispatch tests passed
  std::cout << "External kernel dispatch tests passed\n";
}
