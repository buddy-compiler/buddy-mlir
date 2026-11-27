// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Minimal RAX runner: the shared object contains the compiled Buddy computation
// and fallback; Triton is an optional resource loaded through Buddy Runtime.

#include "SoftmaxSession.h"
#include "buddy/runtime/core/InferenceRunner.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/core/RaxDependenciesHandle.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_set>

// This runner targets riscv64 only; the RAX metadata's target must match.
#if !(defined(__riscv) && __riscv_xlen == 64)
#error "SoftmaxRunner targets riscv64 only"
#endif
#define SOFTMAX_HOST_TARGET "riscv64-unknown-linux-gnu"

namespace {
using buddy::examples::softmax::Session;
using buddy::runtime::ModelManifest;

const ModelManifest::ResolvedBuffer &findBuffer(const ModelManifest &manifest,
                                                const std::string &name) {
  const ModelManifest::ResolvedBuffer *found = nullptr;
  for (const auto &buffer : manifest.buffers) {
    if (buffer.name != name)
      continue;
    if (found)
      throw std::runtime_error("duplicate Softmax buffer: " + name);
    found = &buffer;
  }
  if (!found)
    throw std::runtime_error("missing Softmax buffer: " + name);
  return *found;
}

void requireSoftmaxBuffer(const ModelManifest::ResolvedBuffer &buffer) {
  if (buffer.dtype != rhal::rax::DType_F32 ||
      buffer.layout != rhal::rax::Layout_RowMajor ||
      buffer.space != rhal::rax::MemorySpace_Host ||
      buffer.shape != std::vector<int64_t>{2, 4} ||
      buffer.strides != std::vector<int64_t>{4, 1})
    throw std::runtime_error(
        "Softmax runner requires host row-major 2x4 FP32 buffer: " +
        buffer.name);
}

bool has(const ModelManifest::ResolvedCodeObject &code, const char *key,
         const char *value) {
  auto entry = code.attrs.find(key);
  return entry != code.attrs.end() && entry->second == value;
}

class SoftmaxRunner final : public buddy::runtime::InferenceRunner {
public:
  void run(const buddy::runtime::RunConfig &cfg) override {
    using Clock = std::chrono::steady_clock;
    const char *measurement = std::getenv("BUDDY_SOFTMAX_MEASURE_PREPARATION");
    const bool measure = measurement && std::string(measurement) == "1";
    auto preparationStart = measure ? Clock::now() : Clock::time_point{};
    auto manifest = ModelManifest::loadFromRax(cfg.raxPath);
    auto manifestEnd = measure ? Clock::now() : Clock::time_point{};
    if (manifest.soPath.empty())
      throw std::runtime_error(
          "Softmax RAX is missing its mandatory Buddy computation");
    // The MVP accepts one computation, not a silently truncated RAX program.
    std::unordered_set<uint32_t> bufferIds, codeIds;
    for (const auto &buffer : manifest.buffers)
      if (!buffer.id || !bufferIds.insert(buffer.id).second)
        throw std::runtime_error("invalid or duplicate Softmax buffer id");
    for (const auto &code : manifest.codeObjects)
      if (!code.id || !codeIds.insert(code.id).second)
        throw std::runtime_error("invalid or duplicate Softmax code object id");
    if (manifest.functions.size() > 1)
      throw std::runtime_error("Softmax runner accepts exactly one function");
    const ModelManifest::ResolvedFunction *function =
        manifest.functions.empty() ? nullptr : &manifest.functions.front();
    const ModelManifest::ResolvedDispatch *dispatch =
        (function && !function->dispatches.empty())
            ? &function->dispatches.front()
            : nullptr;
    if (function &&
        (function->hasUnsupportedOps || function->dispatches.size() != 1))
      throw std::runtime_error("Softmax function requires one dispatch only");

    const ModelManifest::ResolvedBuffer *inputMeta = nullptr;
    const ModelManifest::ResolvedBuffer *outputMeta = nullptr;
    int64_t axis = -1;
    if (dispatch) {
      if (!dispatch->supportedArguments || dispatch->bufferArgs.size() != 2 ||
          dispatch->scalars.size() != 1 ||
          dispatch->bufferArgs[0] == dispatch->bufferArgs[1] ||
          function->inputs != std::vector<uint32_t>{dispatch->bufferArgs[0]} ||
          function->outputs != std::vector<uint32_t>{dispatch->bufferArgs[1]} ||
          dispatch->grid != std::vector<uint32_t>{1, 1, 1} ||
          dispatch->block != std::vector<uint32_t>{1, 1, 1})
        throw std::runtime_error(
            "invalid Softmax dispatch bindings or launch dims");
      for (const auto &buffer : manifest.buffers) {
        if (buffer.id == dispatch->bufferArgs[0])
          inputMeta = &buffer;
        else if (buffer.id == dispatch->bufferArgs[1])
          outputMeta = &buffer;
      }
      axis = dispatch->scalars.front();
      if (!inputMeta || !outputMeta || (axis != 1 && axis != -1))
        throw std::runtime_error("invalid Softmax buffer reference or axis");
    } else {
      inputMeta = &findBuffer(manifest, "input");
      outputMeta = &findBuffer(manifest, "output");
    }
    requireSoftmaxBuffer(*inputMeta);
    requireSoftmaxBuffer(*outputMeta);

    std::array<float, 8> input{1, 2, 3, 4, 0, 0, 0, 0}, output{};
    BuddyKernelCallV1 call{BUDDY_KERNEL_ABI_V1,
                           sizeof(BuddyKernelCallV1),
                           BUDDY_KERNEL_SOFTMAX,
                           0,
                           {input.data(), sizeof(input), 2, BUDDY_KERNEL_F32,
                            BUDDY_KERNEL_ROW_MAJOR, 0, inputMeta->shape.data(),
                            inputMeta->strides.data()},
                           {output.data(), sizeof(output), 2, BUDDY_KERNEL_F32,
                            BUDDY_KERNEL_ROW_MAJOR, 0, outputMeta->shape.data(),
                            outputMeta->strides.data()},
                           &axis,
                           1,
                           nullptr,
                           0};

    if (!buddy::runtime::ExternalKernel::isValidCall(call))
      throw std::runtime_error("invalid Softmax call");
    auto bindingsEnd = measure ? Clock::now() : Clock::time_point{};
    std::vector<std::string> dependencies{manifest.soPath};
    dependencies.insert(dependencies.end(), manifest.dependentSoPaths.begin(),
                        manifest.dependentSoPaths.end());
    buddy::runtime::RaxDependenciesHandle primary(dependencies, "softmax");
    auto dependenciesEnd = measure ? Clock::now() : Clock::time_point{};

    // Prefer the explicitly referenced object, then try the remaining Softmax
    // candidates in package order. An invalid explicit reference is an error.
    const ModelManifest::ResolvedCodeObject *candidateCode = nullptr;
    if (dispatch) {
      for (const auto &code : manifest.codeObjects)
        if (code.id == dispatch->codeObjectId) {
          candidateCode = &code;
          break;
        }
      if (!candidateCode)
        throw std::runtime_error("unknown Softmax dispatch code object id");
    }
    std::vector<const ModelManifest::ResolvedCodeObject *> candidates;
    if (candidateCode)
      candidates.push_back(candidateCode);
    for (const auto &code : manifest.codeObjects)
      if (&code != candidateCode && code.optional &&
          code.backend == "triton_cpu" && has(code, "operation", "softmax"))
        candidates.push_back(&code);
    auto lookupEnd = measure ? Clock::now() : Clock::time_point{};
    std::unique_ptr<Session> session;
    for (const auto *candidate : candidates) {
      const auto &code = *candidate;
      auto requirements = code.attrs.find("required_isa");
      const bool metadataOk =
          code.optional && code.backend == "triton_cpu" &&
          code.kind == rhal::rax::CodeObjectKind_HostSharedLib &&
          code.entrySymbol == "buddy_external_kernel_query_v1" &&
          requirements != code.attrs.end() &&
          buddy::runtime::ExternalKernel::hasFeatureRequirements(
              requirements->second) &&
          has(code, "kernel_abi", "buddy.external.v1") &&
          has(code, "operation", "softmax") && has(code, "dtype", "f32") &&
          has(code, "layout", "row_major") && has(code, "shape", "2,4") &&
          has(code, "strides", "4,1") && has(code, "axis", "1") &&
          has(code, "profile", "softmax.f32.2x4.rvv1.vlen256.v1") &&
          has(code, "required_rvv_version", "1.0") &&
          has(code, "rvv_vlen_bits", "256") &&
          has(code, "required_matrix_features", "none") &&
          has(code, "target", SOFTMAX_HOST_TARGET);
      if (!metadataOk) {
        std::cout << "candidate=" << code.name << " rejected=metadata\n";
      } else {
        auto loaded = std::make_unique<Session>(code.path);
        const auto *descriptor = loaded->kernel.descriptorInfo();
        if (!descriptor) {
          std::cout << "candidate=" << code.name
                    << " rejected=load_or_host reason="
                    << loaded->kernel.diagnostic() << '\n';
        } else if (!loaded->kernel.matches(call)) {
          std::cout << "candidate=" << code.name
                    << " rejected=specialization\n";
        } else if (!has(code, "target", descriptor->target)) {
          std::cout << "candidate=" << code.name << " rejected=target\n";
        } else if (!loaded->kernel.satisfiesRequirements(
                       requirements->second)) {
          std::cout << "candidate=" << code.name
                    << " rejected=requirements reason=required_isa\n";
        } else {
          session = std::move(loaded);
          std::cout << "selected_kernel=" << code.name
                    << " code_object_id=" << code.id << '\n';
          break;
        }
      }
    }
    if (!session)
      session = std::make_unique<Session>(std::string());
    auto preparationEnd = measure ? Clock::now() : Clock::time_point{};
    if (measure)
      std::cout << "manifest_load_ns="
                << std::chrono::duration<double, std::nano>(manifestEnd -
                                                            preparationStart)
                       .count()
                << " selection_and_load_ns="
                << std::chrono::duration<double, std::nano>(preparationEnd -
                                                            manifestEnd)
                       .count()
                << " bindings_ns="
                << std::chrono::duration<double, std::nano>(bindingsEnd -
                                                            manifestEnd)
                       .count()
                << " mandatory_dependencies_ns="
                << std::chrono::duration<double, std::nano>(dependenciesEnd -
                                                            bindingsEnd)
                       .count()
                << " candidate_lookup_ns="
                << std::chrono::duration<double, std::nano>(lookupEnd -
                                                            dependenciesEnd)
                       .count()
                << " candidate_load_match_ns="
                << std::chrono::duration<double, std::nano>(preparationEnd -
                                                            lookupEnd)
                       .count()
                << '\n';

    auto before = buddy::examples::softmax::fallbackCalls;
    auto result = buddy::examples::softmax::compute(*session, call);
    if (result.status != BUDDY_KERNEL_OK)
      throw std::runtime_error("Softmax computation failed: " +
                               std::to_string(result.status));
    const float expected[] = {0.032058604f, 0.087144323f, 0.236882836f,
                              0.643914282f, 0.25f,        0.25f,
                              0.25f,        0.25f};
    std::cout << std::setprecision(9);
    for (size_t i = 0; i < output.size(); ++i) {
      if (!std::isfinite(output[i]) ||
          std::abs(output[i] - expected[i]) > 1e-6f + 1e-5f * expected[i])
        throw std::runtime_error("Softmax output failed numerical validation");
      std::cout << output[i] << (i % 4 == 3 ? '\n' : ' ');
    }
    std::cout << "PASS RAX Softmax path="
              << (result.path == buddy::runtime::KernelPath::External
                      ? "external"
                      : "fallback")
              << " fallback_calls="
              << buddy::examples::softmax::fallbackCalls - before
              << " external_status=" << result.externalStatus
              << " atol=1e-6 rtol=1e-5\n";
  }
};
} // namespace

extern "C" buddy::runtime::InferenceRunner *buddy_create_inference_runner_v1() {
  return new SoftmaxRunner;
}

extern "C" void
buddy_destroy_inference_runner_v1(buddy::runtime::InferenceRunner *runner) {
  delete runner;
}
