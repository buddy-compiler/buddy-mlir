// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "RiscvCapabilities.h"
#include "SoftmaxSession.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

using buddy::examples::softmax::compute;
using buddy::examples::softmax::fallbackCalls;
using buddy::examples::softmax::Session;
using buddy::runtime::ExternalKernel;
using buddy::runtime::KernelPath;

static bool withinTolerance(float expected, float actual) {
  return std::isfinite(expected) && std::isfinite(actual) &&
         std::abs(expected - actual) <= 1e-6f + 1e-5f * std::abs(expected);
}

static_assert(sizeof(BuddyKernelTensorV1) == 48);
static_assert(sizeof(BuddyKernelCallV1) == 144);
static_assert(sizeof(BuddyKernelDescriptorV1) == 80);
static_assert(offsetof(BuddyKernelCallV1, scalars) == 112);
static_assert(offsetof(BuddyKernelDescriptorV1, invoke) == 72);

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;
  try {
    float input[10] = {1, 2, 3, 4}, output[10];
    const std::array<float, 10> saved{1, 2, 3, 4};
    int64_t shape[] = {2, 4}, strides[] = {4, 1}, axis = -1;
    int64_t zero[] = {0, 4}, negative[] = {-1, 4};
    int64_t overflow[] = {std::numeric_limits<int64_t>::max(), 4};
    int64_t other[] = {1, 4}, badStride[] = {4, 2}, negativeStride[] = {-4, 1};
    int64_t badAxis = 0;
    BuddyKernelCallV1 base{1,
                           sizeof(BuddyKernelCallV1),
                           BUDDY_KERNEL_SOFTMAX,
                           0,
                           {input, 32, 2, BUDDY_KERNEL_F32,
                            BUDDY_KERNEL_ROW_MAJOR, 0, shape, strides},
                           {output, 32, 2, BUDDY_KERNEL_F32,
                            BUDDY_KERNEL_ROW_MAJOR, 0, shape, strides},
                           &axis,
                           1,
                           nullptr,
                           0};
    Session actual(argv[1]), missing("");
    if (!actual.kernel.matches(base))
      throw std::runtime_error("actual kernel unavailable: " +
                               actual.kernel.diagnostic());
    const auto direct = actual.kernel.descriptorInfo()->invoke;
    if (direct(nullptr) != BUDDY_KERNEL_INVALID)
      throw std::runtime_error("direct null call was not INVALID");
    unsigned checked = 0;
    auto invalid = [&](BuddyKernelCallV1 call, const std::string &name) {
      std::fill_n(output, 10, -77.f);
      auto before = fallbackCalls;
      for (Session *session : {&actual, &missing}) {
        auto result = compute(*session, call);
        if (result.status != BUDDY_KERNEL_INVALID ||
            result.path != KernelPath::None || fallbackCalls != before ||
            !std::all_of(output, output + 10,
                         [](float v) { return v == -77.f; }))
          throw std::runtime_error(
              "invalid call accepted or output modified: " + name);
      }
      if (direct(&call) != BUDDY_KERNEL_INVALID ||
          !std::all_of(output, output + 10,
                       [](float value) { return value == -77.f; }) ||
          std::memcmp(input, saved.data(), sizeof(input)))
        throw std::runtime_error("direct invalid call modified buffers: " +
                                 name);
      ++checked;
    };
    auto callCase = [&](const char *name, auto change) {
      auto call = base;
      change(call);
      invalid(call, name);
    };
    callCase("abi", [](auto &c) { c.abiVersion = 99; });
    callCase("size", [](auto &c) { --c.structSize; });
    callCase("reserved", [](auto &c) { c.reserved = 1; });
    callCase("operation", [](auto &c) { c.operation = 99; });
    callCase("null scalar", [](auto &c) { c.scalars = nullptr; });
    callCase("missing scalar", [](auto &c) { c.scalarCount = 0; });
    callCase("extra scalar", [](auto &c) { c.scalarCount = 2; });
    callCase("axis", [&](auto &c) { c.scalars = &badAxis; });
    callCase("workspace pointer", [&](auto &c) { c.workspace = input; });
    callCase("workspace size", [](auto &c) { c.workspaceBytes = 16; });
    callCase("workspace both", [&](auto &c) {
      c.workspace = input;
      c.workspaceBytes = 16;
    });
    callCase("exact overlap", [&](auto &c) { c.output.data = input; });
    callCase("partial overlap", [&](auto &c) { c.output.data = input + 1; });
    callCase("shape mismatch", [&](auto &c) { c.output.shape = other; });
    for (bool outputTensor : {false, true}) {
      using Mutation =
          std::pair<const char *, std::function<void(BuddyKernelTensorV1 &)>>;
      std::vector<Mutation> mutations{
          {"null data", [](auto &t) { t.data = nullptr; }},
          {"unaligned",
           [](auto &t) { t.data = static_cast<char *>(t.data) + 1; }},
          {"short span", [](auto &t) { t.byteSize = 31; }},
          {"rank", [](auto &t) { t.rank = 3; }},
          {"dtype", [](auto &t) { t.dtype = 99; }},
          {"layout", [](auto &t) { t.layout = 99; }},
          {"reserved", [](auto &t) { t.reserved = 1; }},
          {"null shape", [](auto &t) { t.shape = nullptr; }},
          {"null stride", [](auto &t) { t.strides = nullptr; }},
          {"zero dim", [&](auto &t) { t.shape = zero; }},
          {"negative dim", [&](auto &t) { t.shape = negative; }},
          {"overflow", [&](auto &t) { t.shape = overflow; }},
          {"bad stride", [&](auto &t) { t.strides = badStride; }},
          {"negative stride", [&](auto &t) { t.strides = negativeStride; }},
      };
      for (auto &[name, mutate] : mutations) {
        auto call = base;
        mutate(outputTensor ? call.output : call.input);
        invalid(call, name);
      }
    }
    auto outside = base;
    outside.input.shape = outside.output.shape = other;
    auto unsupported = compute(actual, outside);
    if (unsupported.status != BUDDY_KERNEL_UNSUPPORTED ||
        unsupported.path != KernelPath::None ||
        direct(&outside) != BUDDY_KERNEL_UNSUPPORTED ||
        !std::all_of(output, output + 10,
                     [](float value) { return value == -77.f; }))
      throw std::runtime_error("outside graph shape was not rejected");
    for (int64_t value : {-1, 1}) {
      axis = value;
      for (Session *session : {&actual, &missing}) {
        auto before = fallbackCalls;
        auto result = compute(*session, base);
        auto expectedPath =
            session == &actual ? KernelPath::External : KernelPath::Fallback;
        if (result.status || result.path != expectedPath ||
            fallbackCalls - before != unsigned(session == &missing) ||
            std::memcmp(input, saved.data(), sizeof(input)))
          throw std::runtime_error("valid axis/path/input ownership failed");
        std::array<float, 8> expected;
        std::copy_n(output, 8, expected.begin());
        std::fill_n(output, 10, -77.f);
        if (direct(&base) != BUDDY_KERNEL_OK ||
            !std::equal(expected.begin(), expected.end(), output,
                        withinTolerance) ||
            output[8] != -77.f || output[9] != -77.f ||
            std::memcmp(input, saved.data(), sizeof(input)))
          throw std::runtime_error("direct valid call/result/canary failed");
      }
    }
    auto offset = base;
    offset.input.data = input + 1;
    offset.output.data = output + 1;
    std::fill_n(output, 10, -77.f);
    if (compute(actual, offset).path != KernelPath::External)
      throw std::runtime_error("aligned logical offset rejected");
    std::array<float, 8> offsetExpected;
    std::copy_n(output + 1, 8, offsetExpected.begin());
    std::fill_n(output, 10, -77.f);
    if (direct(&offset) != BUDDY_KERNEL_OK ||
        !std::equal(offsetExpected.begin(), offsetExpected.end(), output + 1,
                    withinTolerance) ||
        output[0] != -77.f || output[9] != -77.f ||
        std::memcmp(input, saved.data(), sizeof(input)))
      throw std::runtime_error("direct offset result/canary failed");
    std::cout << "PASS ABI and compiled graph contract invalid_cases="
              << checked << " direct_invalid_cases=" << checked + 1
              << " paths=external,fallback axes=1,-1 input_unchanged=1 "
                 "direct_valid_canaries=1\n";

    using buddy::examples::softmax::allHartsSupport;
    std::vector<std::pair<std::string, bool>> records{
        {"processor: 0\nisa: rv64imafdcv\n", true},
        {"processor: 0\nisa: rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0\n", true},
        {"processor: 0\nisa: rv64imafdc_zve32f_zvfh\n", false},
        {"processor: 0\nisa: rv64imafdcv0p7\n", false},
        {"processor: 0\nisa: rv64imafdc_v0p7\n", false},
        {"processor: 0\nisa: rv64imafdcv\nprocessor: 1\nisa: rv64imafdc\n",
         false},
        {"processor: 0\nisa: rv64imafdcv\nprocessor: 1\n", false},
        {"processor: 0\nisa: rv32imafdcv\n", false},
        {"", false},
    };
    for (auto &[record, expected] : records) {
      std::istringstream stream(record);
      if (allHartsSupport(stream, "i,m,a,f,d,c,v") != expected)
        throw std::runtime_error("ISA capability record failed");
    }
    for (auto [isa, expected] : std::vector<std::pair<std::string, bool>>{
             {"rv64imafdcv_zba", true}, {"rv64imafdcv_zbb", false}}) {
      std::istringstream stream("processor: 0\nisa: " + isa + "\n");
      if (allHartsSupport(stream, "v,zba") != expected)
        throw std::runtime_error("required extension identity check failed");
    }
    if (ExternalKernel::hasFeatureRequirements(" , ; ") ||
        !ExternalKernel::hasFeatureRequirements(" V ; v "))
      throw std::runtime_error("empty requirement normalization failed");
    std::cout << "PASS ISA records=" << records.size() + 2
              << " normalized_requirements=2\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
