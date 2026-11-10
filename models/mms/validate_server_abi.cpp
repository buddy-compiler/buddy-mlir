//===- validate_server_abi.cpp - Real MMS plugin lifecycle test
//------------===//
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

#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/core/RaxDependenciesHandle.h"
#include "buddy/runtime/core/ServerModelABI.h"

#include <chrono>
#include <cmath>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
std::vector<float> readFloats(const char *path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || stream.tellg() <= 0 || stream.tellg() % sizeof(float))
    throw std::runtime_error("invalid real-data float32 file");
  std::vector<float> values(static_cast<size_t>(stream.tellg()) /
                            sizeof(float));
  stream.seekg(0);
  stream.read(reinterpret_cast<char *>(values.data()),
              values.size() * sizeof(float));
  if (!stream)
    throw std::runtime_error("failed to read real-data float32 file");
  return values;
}

BuddyServerStringViewV1 view(const std::string &s) {
  return {s.data(), s.size()};
}

struct Result {
  const std::vector<float> &reference;
  int calls = 0;
  bool matches = false;
};

BuddyServerEmitResultV1 collect(void *data, const BuddyServerEventV1 *event) {
  auto &result = *static_cast<Result *>(data);
  ++result.calls;
  if (!event || event->kind != BUDDY_SERVER_EVENT_FINAL ||
      event->operation != BUDDY_SERVER_OPERATION_VOCODER ||
      !event->vocoder.samples || event->vocoder.sampleRate != 16000 ||
      event->vocoder.sampleCount != result.reference.size())
    return BUDDY_SERVER_EMIT_ERROR;
  result.matches = true;
  for (size_t i = 0; i < result.reference.size(); ++i) {
    const float actual = event->vocoder.samples[i],
                expected = result.reference[i];
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > 1e-4 + 1e-3 * std::abs(expected))
      result.matches = false;
  }
  return result.matches ? BUDDY_SERVER_EMIT_CONTINUE : BUDDY_SERVER_EMIT_ERROR;
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 4)
      throw std::invalid_argument(
          "usage: mms-server-abi-test MODEL.rax LATENTS.f32 REFERENCE.f32");
    auto input = readFloats(argv[2]);
    auto reference = readFloats(argv[3]);
    const auto manifest = buddy::runtime::ModelManifest::loadFromRax(argv[1]);
    buddy::runtime::RaxDependenciesHandle dependencies(
        manifest.dependentSoPaths, "MMS ABI test");
    std::unique_ptr<void, decltype(&dlclose)> library(
        dlopen(manifest.modelLibraryPath.c_str(), RTLD_NOW | RTLD_LOCAL),
        &dlclose);
    if (!library)
      throw std::runtime_error(dlerror());
    auto describe = reinterpret_cast<BuddyServerModelDescriptorFnV1>(
        dlsym(library.get(), BUDDY_SERVER_MODEL_DESCRIPTOR_SYMBOL_V1));
    if (!describe)
      throw std::runtime_error("missing production model descriptor");
    const auto *descriptor = describe();
    int checks = 0;
    auto expect = [&](bool ok, const char *message) {
      if (!ok)
        throw std::runtime_error(message);
      ++checks;
    };
    expect(
        descriptor && descriptor->serviceKind == BUDDY_SERVER_SERVICE_VOCODER &&
            descriptor->resultKind == BUDDY_SERVER_RESULT_AUDIO &&
            descriptor->maxConcurrency == 1 && descriptor->capabilityMask == 0,
        "wrong production descriptor");
    const auto &ops = *descriptor->ops;
    BuddyServerErrorV1 error{};
    expect(!ops.create(nullptr, &error) &&
               error.code == BUDDY_SERVER_STATUS_INVALID_ARGUMENT,
           "null create accepted");
    BuddyServerCreateArgsV1 createArgs{sizeof(createArgs),
                                       view(manifest.modelName)};
    std::unique_ptr<void, BuddyServerDestroyFnV1> model(
        ops.create(&createArgs, &error), ops.destroy);
    expect(bool(model), "create failed");
    BuddyServerStatusV1 status{};
    status.structSize = sizeof(status);
    expect(ops.getStatus(model.get(), &status, &error) ==
                   BUDDY_SERVER_STATUS_OK &&
               status.state == BUDDY_SERVER_MODEL_UNLOADED,
           "initial status is wrong");
    BuddyServerRequestV1 request{};
    request.structSize = sizeof(request);
    request.operation = BUDDY_SERVER_OPERATION_VOCODER;
    request.vocoder = {view(manifest.modelName), input.data(), input.size()};
    BuddyServerCallV1 call{};
    call.structSize = sizeof(call);
    call.request = &request;
    Result result{reference};
    auto invoke = [&] {
      return ops.invoke(model.get(), &call, &collect, &result, &error);
    };
    expect(invoke() == BUDDY_SERVER_STATUS_MODEL_NOT_READY,
           "unloaded invoke accepted");
    std::string package = argv[1];
    BuddyServerLoadArgsV1 loadArgs{};
    loadArgs.structSize = sizeof(loadArgs);
    loadArgs.raxPath = view(package);
    expect(ops.load(model.get(), &loadArgs, &error) == BUDDY_SERVER_STATUS_OK,
           "load failed");
    expect(ops.getStatus(model.get(), &status, &error) ==
                   BUDDY_SERVER_STATUS_OK &&
               status.state == BUDDY_SERVER_MODEL_READY,
           "ready status is wrong");
    expect(invoke() == BUDDY_SERVER_STATUS_OK && result.matches &&
               result.calls == 1,
           "real waveform mismatch");
    const float referenceSample = reference.front();
    for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity()}) {
      reference.front() = invalid;
      expect(invoke() == BUDDY_SERVER_STATUS_INTERNAL_ERROR && !result.matches,
             "non-finite reference accepted");
    }
    reference.front() = referenceSample;
    const int emittedBeforeInvalidRequests = result.calls;
    request.structSize = 0;
    expect(invoke() == BUDDY_SERVER_STATUS_INVALID_ARGUMENT,
           "short request accepted");
    request.structSize = sizeof(request);
    request.operation = BUDDY_SERVER_OPERATION_COMPLETION;
    expect(invoke() == BUDDY_SERVER_STATUS_NOT_SUPPORTED,
           "unsupported operation accepted");
    request.operation = BUDDY_SERVER_OPERATION_VOCODER;
    request.vocoder.latents = nullptr;
    expect(invoke() == BUDDY_SERVER_STATUS_INVALID_ARGUMENT,
           "null latents accepted");
    request.vocoder.latents = input.data();
    --request.vocoder.latentCount;
    expect(invoke() == BUDDY_SERVER_STATUS_INVALID_ARGUMENT,
           "short latents accepted");
    ++request.vocoder.latentCount;
    const float original = input.front();
    input.front() = std::numeric_limits<float>::quiet_NaN();
    expect(invoke() == BUDDY_SERVER_STATUS_INVALID_ARGUMENT, "NaN accepted");
    input.front() = original;
    call.isCancelled = [](void *) -> uint32_t { return 1; };
    expect(invoke() == BUDDY_SERVER_STATUS_CANCELLED, "cancellation ignored");
    call.isCancelled = nullptr;
    call.deadlineUnixNs = 1;
    expect(invoke() == BUDDY_SERVER_STATUS_DEADLINE_EXCEEDED,
           "expired deadline ignored");
    call.deadlineUnixNs = 0;
    expect(result.calls == emittedBeforeInvalidRequests,
           "an invalid request emitted samples");
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    call.deadlineUnixNs =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count() +
        100000000;
    expect(invoke() == BUDDY_SERVER_STATUS_DEADLINE_EXCEEDED &&
               result.calls == emittedBeforeInvalidRequests,
           "inference emitted a result after its deadline");
    call.deadlineUnixNs = 0;
    expect(ops.invoke(
               model.get(), &call,
               [](void *, const BuddyServerEventV1 *) {
                 return BUDDY_SERVER_EMIT_ERROR;
               },
               nullptr, &error) == BUDDY_SERVER_STATUS_INTERNAL_ERROR,
           "callback failure ignored");
    expect(invoke() == BUDDY_SERVER_STATUS_OK && result.matches,
           "recovery failed");
    loadArgs.raxPath = {};
    expect(ops.load(model.get(), &loadArgs, &error) ==
               BUDDY_SERVER_STATUS_INVALID_ARGUMENT,
           "empty package accepted");
    expect(invoke() == BUDDY_SERVER_STATUS_MODEL_NOT_READY,
           "failed reload retained a stale model");
    expect(ops.getStatus(model.get(), &status, &error) ==
                   BUDDY_SERVER_STATUS_OK &&
               status.state == BUDDY_SERVER_MODEL_ERROR,
           "failed reload status is wrong");
    loadArgs.raxPath = view(package);
    expect(ops.load(model.get(), &loadArgs, &error) == BUDDY_SERVER_STATUS_OK,
           "reload failed");
    expect(invoke() == BUDDY_SERVER_STATUS_OK && result.matches,
           "reloaded waveform mismatch");
    model.reset();
    std::cout << "PASS: " << checks << " real-model ABI/lifecycle checks\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
