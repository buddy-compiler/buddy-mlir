//===- Run.cpp - Buddy computation, optional Triton dispatch and timings
//----===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "SoftmaxSession.h"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

using buddy::runtime::ExternalKernel;
using buddy::runtime::KernelPath;

using buddy::examples::softmax::compute;
using buddy::examples::softmax::fallbackCalls;
using buddy::examples::softmax::Session;

using Clock = std::chrono::steady_clock;
template <class F> static double timeCalls(F &&fn, int iterations) {
  auto start = Clock::now();
  for (int i = 0; i < iterations; ++i)
    fn();
  return std::chrono::duration<double, std::nano>(Clock::now() - start)
             .count() /
         iterations;
}

// An approximate timer-loop/call/status-check control, not a cost to subtract
// blindly: real paths have different branch and instruction dependencies.
__attribute__((noinline)) static int32_t
emptyInvoke(const BuddyKernelCallV1 *call) {
  asm volatile("" : : "r"(call) : "memory");
  return BUDDY_KERNEL_OK;
}

int main(int argc, char **argv) {
  if ((argc != 2 && argc != 3) ||
      (argc == 3 && std::strcmp(argv[2], "--benchmark") != 0)) {
    std::cerr << "usage: run <libtriton_softmax.so> [--benchmark]\n";
    return 1;
  }
  float input[8] = {1, 2, 3, 4, 0, 0, 0, 0};
  float external[8], original[8];
  int64_t shape[] = {2, 4}, strides[] = {4, 1}, axis = -1;
  BuddyKernelCallV1 call{BUDDY_KERNEL_ABI_V1,
                         sizeof(BuddyKernelCallV1),
                         BUDDY_KERNEL_SOFTMAX,
                         0,
                         {input, sizeof(input), 2, BUDDY_KERNEL_F32,
                          BUDDY_KERNEL_ROW_MAJOR, 0, shape, strides},
                         {external, sizeof(external), 2, BUDDY_KERNEL_F32,
                          BUDDY_KERNEL_ROW_MAJOR, 0, shape, strides},
                         &axis,
                         1,
                         nullptr,
                         0};
  const bool benchmark = argc == 3;
  auto loadStart = benchmark ? Clock::now() : Clock::time_point{};
  Session session(argv[1]);
  auto &kernel = session.kernel;
  auto loadNs =
      std::chrono::duration<double, std::nano>(
          (benchmark ? Clock::now() : Clock::time_point{}) - loadStart)
          .count();
  Session missing(std::string(argv[1]) + ".missing");
  const auto firstCallStart = benchmark ? Clock::now() : Clock::time_point{};
  auto result = compute(session, call);
  const auto firstCallNs =
      std::chrono::duration<double, std::nano>(
          (benchmark ? Clock::now() : Clock::time_point{}) - firstCallStart)
          .count();
  if (result.status || result.path != KernelPath::External || fallbackCalls) {
    std::cerr << "external dispatch failed: " << kernel.diagnostic() << '\n';
    return 2;
  }
  call.output.data = original;
  result = compute(missing, call);
  if (result.status || result.path != KernelPath::Fallback ||
      fallbackCalls != 1)
    return 3;
  double maxError = 0;
  for (int i = 0; i < 8; ++i) {
    std::cout << "index=" << i << " Triton=" << external[i]
              << " Buddy=" << original[i] << '\n';
    maxError = std::max(maxError, double(std::abs(external[i] - original[i])));
    if (!std::isfinite(external[i]) || !std::isfinite(original[i]) ||
        std::abs(external[i] - original[i]) >
            1e-6f + 1e-5f * std::abs(original[i]))
      return 4;
  }
  std::cout << "PASS path=external fallback_calls=0; path=fallback "
               "fallback_calls=1 max_abs_error="
            << maxError << " atol=1e-6 rtol=1e-5\n";

  // Deterministic finite corpus: offsets test stability; scales test
  // saturation.
  double corpusError = 0, oracleError = 0;
  uint64_t corpusHash = 14695981039346656037ULL;
  for (int sample = 0; sample < 1000; ++sample) {
    float scale = sample % 3 == 0 ? 100.f : (sample % 3 == 1 ? 1.f : .001f);
    for (int i = 0; i < 8; ++i)
      input[i] = std::sin(float(sample * 8 + i)) * scale + (sample % 7) * 100.f;
    if (sample == 0)
      std::fill_n(input, 8, 0.f);
    else if (sample == 1)
      for (int i = 0; i < 8; ++i)
        input[i] = -float(i + 1);
    else if (sample == 2)
      for (int i = 0; i < 8; ++i)
        input[i] = 1000.f + float(i);
    else if (sample == 3)
      for (int i = 0; i < 8; ++i)
        input[i] = i % 2 ? -1000.f : 1000.f;
    else if (sample == 4)
      for (int i = 0; i < 8; ++i)
        input[i] = float(i) * 1e-7f;
    call.output.data = external;
    if (compute(session, call).path != KernelPath::External)
      return 5;
    const auto *bytes = reinterpret_cast<const unsigned char *>(external);
    for (size_t i = 0; i < sizeof(external); ++i) {
      corpusHash ^= bytes[i];
      corpusHash *= 1099511628211ULL;
    }
    call.output.data = original;
    auto fallbackResult = compute(missing, call);
    if (fallbackResult.status || fallbackResult.path != KernelPath::Fallback)
      return 6;
    for (int row = 0; row < 2; ++row) {
      double maximum = *std::max_element(input + row * 4, input + row * 4 + 4);
      double denom = 0;
      for (int col = 0; col < 4; ++col)
        denom += std::exp(double(input[row * 4 + col]) - maximum);
      for (int col = 0; col < 4; ++col) {
        int i = row * 4 + col;
        double oracle = std::exp(double(input[i]) - maximum) / denom;
        if (!std::isfinite(external[i]) || !std::isfinite(original[i]) ||
            std::abs(external[i] - original[i]) >
                1e-6 + 1e-5 * std::abs(original[i]) ||
            std::abs(external[i] - oracle) > 1e-6 + 1e-5 * oracle)
          return 7;
        corpusError =
            std::max(corpusError, double(std::abs(external[i] - original[i])));
        oracleError = std::max(oracleError, std::abs(external[i] - oracle));
      }
    }
  }
  std::cout << "PASS finite_corpus=1000 max_abs_vs_buddy=" << corpusError
            << " max_abs_vs_double=" << oracleError
            << " output_bits_fnv1a=" << corpusHash << '\n';
  const float fixed[] = {1, 2, 3, 4, 0, 0, 0, 0};
  std::copy(fixed, fixed + 8, input);
  call.output.data = external;
  if (argc == 2)
    return 0;

  void *handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    return 8;
  auto query = reinterpret_cast<BuddyKernelQueryV1>(
      dlsym(handle, "buddy_external_kernel_query_v1"));
  const auto *descriptor = query ? query() : nullptr;
  if (!descriptor || !descriptor->invoke || !descriptor->isCompatibleHost) {
    dlclose(handle);
    return 8;
  }
  auto direct = descriptor->invoke;
  auto raw = reinterpret_cast<int32_t (*)(const BuddyKernelCallV1 *)>(
      dlsym(handle, "buddy_softmax_raw_measurement"));
  if (!raw) {
    std::cerr << "Benchmark requires the dedicated measurement_kernel.so; "
                 "production libraries export only the checked query.\n";
    dlclose(handle);
    return 8;
  }
  bool timingFailure = false;
  const float expected[] = {0.032058604f, 0.087144323f, 0.236882836f,
                            0.643914282f, 0.25f,        0.25f,
                            0.25f,        0.25f};
  auto checkOutput = [&] {
    for (int i = 0; i < 8; ++i)
      timingFailure |=
          !std::isfinite(external[i]) ||
          std::abs(external[i] - expected[i]) > 1e-6f + 1e-5f * expected[i];
    timingFailure |= std::memcmp(input, fixed, sizeof(input)) != 0;
  };
  for (int i = 0; i < 1000; ++i) {
    auto externalResult = compute(session, call);
    auto buddyResult = compute(missing, call);
    timingFailure |= externalResult.status != BUDDY_KERNEL_OK ||
                     externalResult.path != KernelPath::External ||
                     buddyResult.status != BUDDY_KERNEL_OK ||
                     buddyResult.path != KernelPath::Fallback;
  }
  // §6 metric-1 path B correctness: the raw kernel entry must match the Buddy
  // fallback and the double-precision oracle over the same deterministic corpus
  // used above (regenerated identically). Timing is never started on a
  // mismatch.
  double rawCorpusError = 0, rawOracleError = 0;
  {
    float rawOutput[8];
    for (int sample = 0; sample < 1000; ++sample) {
      float scale = sample % 3 == 0 ? 100.f : (sample % 3 == 1 ? 1.f : .001f);
      for (int i = 0; i < 8; ++i)
        input[i] =
            std::sin(float(sample * 8 + i)) * scale + (sample % 7) * 100.f;
      if (sample == 0)
        std::fill_n(input, 8, 0.f);
      else if (sample == 1)
        for (int i = 0; i < 8; ++i)
          input[i] = -float(i + 1);
      else if (sample == 2)
        for (int i = 0; i < 8; ++i)
          input[i] = 1000.f + float(i);
      else if (sample == 3)
        for (int i = 0; i < 8; ++i)
          input[i] = i % 2 ? -1000.f : 1000.f;
      else if (sample == 4)
        for (int i = 0; i < 8; ++i)
          input[i] = float(i) * 1e-7f;
      call.output.data = rawOutput;
      if (raw(&call) != BUDDY_KERNEL_OK) {
        timingFailure = true;
        break;
      }
      call.output.data = original;
      auto fallbackResult = compute(missing, call);
      if (fallbackResult.status ||
          fallbackResult.path != KernelPath::Fallback) {
        timingFailure = true;
        break;
      }
      for (int row = 0; row < 2; ++row) {
        double maximum =
            *std::max_element(input + row * 4, input + row * 4 + 4);
        double denom = 0;
        for (int col = 0; col < 4; ++col)
          denom += std::exp(double(input[row * 4 + col]) - maximum);
        for (int col = 0; col < 4; ++col) {
          int i = row * 4 + col;
          double oracle = std::exp(double(input[i]) - maximum) / denom;
          if (!std::isfinite(rawOutput[i]) ||
              std::abs(rawOutput[i] - original[i]) >
                  1e-6 + 1e-5 * std::abs(original[i]) ||
              std::abs(rawOutput[i] - oracle) > 1e-6 + 1e-5 * oracle)
            timingFailure = true;
          rawCorpusError = std::max(
              rawCorpusError, double(std::abs(rawOutput[i] - original[i])));
          rawOracleError =
              std::max(rawOracleError, std::abs(rawOutput[i] - oracle));
        }
      }
    }
    call.output.data = external;
    std::copy(fixed, fixed + 8, input);
  }
  std::cout << "PASS raw_entry_corpus=1000 max_abs_vs_buddy=" << rawCorpusError
            << " max_abs_vs_double=" << rawOracleError
            << " atol=1e-6 rtol=1e-5\n";
  volatile int32_t sink = 0;
  int iterations = 200000;
  if (const char *value = std::getenv("BUDDY_SOFTMAX_BENCH_ITERATIONS")) {
    char *end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (!end || *end || parsed < 1000 || parsed > 10000000)
      return 10;
    iterations = static_cast<int>(parsed);
  }
  std::cout << std::setprecision(12) << "first_external_call_ns=" << firstCallNs
            << " prepare_plus_first_call_ns=" << loadNs + firstCallNs << '\n';
  std::cout << std::setprecision(12) << "first_session_prepare_ns=" << loadNs
            << " layout_conversion=not_applicable threads=1 iterations="
            << iterations << " warmup=1000\n";
  std::cout << "round,order,buddy_ns,triton_direct_ns,triton_graph_ns,"
               "raw_kernel_ns,match_ns,runtime_dispatch_ns,host_check_ns,"
               "framework_control_ns\n";
  for (int round = 0; round < 10; ++round) {
    double buddyNs, directNs, dispatchNs, rawNs;
    auto buddyTime = [&] {
      buddyNs = timeCalls(
          [&] {
            auto result = compute(missing, call);
            sink = result.status;
            timingFailure |= result.status != BUDDY_KERNEL_OK ||
                             result.path != KernelPath::Fallback;
          },
          iterations);
    };
    auto directTime = [&] {
      directNs = timeCalls(
          [&] {
            sink = direct(&call);
            timingFailure |= sink != BUDDY_KERNEL_OK;
          },
          iterations);
      checkOutput();
    };
    auto tritonTime = [&] {
      dispatchNs = timeCalls(
          [&] {
            auto result = compute(session, call);
            sink = result.status;
            timingFailure |= result.status != BUDDY_KERNEL_OK ||
                             result.path != KernelPath::External;
          },
          iterations);
      checkOutput();
    };
    auto rawTime = [&] {
      rawNs = timeCalls(
          [&] {
            sink = raw(&call);
            timingFailure |= sink != BUDDY_KERNEL_OK;
          },
          iterations);
      checkOutput();
    };
    if (round % 2 == 0) {
      buddyTime();
      checkOutput();
      rawTime();
      tritonTime();
    } else {
      tritonTime();
      rawTime();
      buddyTime();
      checkOutput();
    }
    double runtimeNs = 0;
    auto runtimeTime = [&] {
      runtimeNs = timeCalls(
          [&] {
            // A missing fallback is intentional here: any failure invalidates
            // timing.
            auto result = kernel.dispatch(call, nullptr);
            sink = result.status;
            timingFailure |= result.status != BUDDY_KERNEL_OK ||
                             result.path != KernelPath::External;
          },
          iterations);
      checkOutput();
    };
    if (round % 2 == 0) {
      directTime();
      runtimeTime();
    } else {
      runtimeTime();
      directTime();
    }
    double lookupNs = timeCalls(
        [&] {
          sink = kernel.matches(call);
          timingFailure |= sink != 1;
        },
        iterations);
    double hostNs = timeCalls(
        [&] {
          sink = descriptor->isCompatibleHost();
          timingFailure |= sink != 1;
        },
        iterations);
    double controlNs = timeCalls(
        [&] {
          sink = emptyInvoke(&call);
          timingFailure |= sink != BUDDY_KERNEL_OK;
        },
        iterations);
    std::cout << round << ',' << (round % 2 == 0 ? "AB" : "BA") << ','
              << buddyNs << ',' << directNs << ',' << dispatchNs << ',' << rawNs
              << ',' << lookupNs << ',' << runtimeNs << ',' << hostNs << ','
              << controlNs << '\n';
  }
  dlclose(handle);
  if (timingFailure) {
    std::cerr << "INVALID benchmark: unexpected status or execution path\n";
    return 9;
  }
  std::cout << "PASS benchmark status_paths_outputs=valid\n";
  return sink == -999 ? 8 : 0;
}
