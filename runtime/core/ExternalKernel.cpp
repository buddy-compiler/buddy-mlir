//===- ExternalKernel.cpp - Optional synchronous host kernel dispatch -----===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "buddy/runtime/core/ExternalKernel.h"
#include "buddy/runtime/core/ExternalKernelValidation.h"

#include <dlfcn.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>

namespace buddy::runtime {
namespace {
const char *hostTarget() {
#if defined(__x86_64__)
  return "x86_64-unknown-linux-gnu";
#elif defined(__riscv) && __riscv_xlen == 64
  return "riscv64-unknown-linux-gnu";
#else
  return "unsupported";
#endif
}

// Serialized requirements are opaque feature tokens. Normalizing them here
// keeps RAX metadata independent of ordering, case and whitespace. This is
// deliberately not an LLVM feature-string interpreter: the plugin callback
// stays authoritative for what the host can actually run.
std::vector<std::string> parseFeatureTokens(const std::string &text) {
  std::vector<std::string> tokens;
  std::string current;
  for (unsigned char character : text) {
    if (character == ',' || character == ';' || std::isspace(character)) {
      if (!current.empty())
        tokens.push_back(current);
      current.clear();
      continue;
    }
    current.push_back(static_cast<char>(std::tolower(character)));
  }
  if (!current.empty())
    tokens.push_back(current);
  std::sort(tokens.begin(), tokens.end());
  tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
  return tokens;
}

} // namespace

ExternalKernel::ExternalKernel(const std::string &libraryPath) {
  if (libraryPath.empty()) {
    message = "no optional kernel library";
    return;
  }
  handle = dlopen(libraryPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    const char *error = dlerror();
    message = error ? error : "dlopen failed";
    return;
  }
  dlerror();
  auto query = reinterpret_cast<BuddyKernelQueryV1>(
      dlsym(handle, "buddy_external_kernel_query_v1"));
  const char *error = dlerror();
  if (!query || error) {
    message = error ? error : "missing kernel query";
    return;
  }
  const auto *candidate = query();
  if (!candidate || candidate->abiVersion != BUDDY_KERNEL_ABI_V1 ||
      candidate->structSize != sizeof(*candidate) || !candidate->target ||
      std::strcmp(candidate->target, hostTarget()) != 0 ||
      !candidate->requiredFeatures || !candidate->isCompatibleHost ||
      !candidate->invoke || !candidate->isCompatibleHost()) {
    message = "incompatible kernel ABI or host target";
    return;
  }
  descriptor = candidate;
}

ExternalKernel::~ExternalKernel() {
  if (handle)
    dlclose(handle);
}

bool ExternalKernel::matches(const BuddyKernelCallV1 &call) const {
  return detail::validSoftmaxCall(call) && descriptor &&
         descriptor->operation == call.operation &&
         descriptor->dtype == call.input.dtype &&
         descriptor->rank == call.input.rank &&
         descriptor->layout == call.input.layout &&
         descriptor->rows == call.input.shape[0] &&
         descriptor->cols == call.input.shape[1] && descriptor->axis == 1;
}

bool ExternalKernel::isValidCall(const BuddyKernelCallV1 &call) {
  return detail::validSoftmaxCall(call);
}

bool ExternalKernel::hasFeatureRequirements(const std::string &features) {
  return !parseFeatureTokens(features).empty();
}

bool ExternalKernel::satisfiesRequirements(
    const std::string &serializedFeatures) const {
  if (!descriptor || !descriptor->requiredFeatures)
    return false;
  const auto required = parseFeatureTokens(serializedFeatures);
  // An empty list records no requirement; the compatibility callback still
  // decides whether the host can run the library.
  if (required.empty())
    return true;
  const auto declared = parseFeatureTokens(descriptor->requiredFeatures);
  return std::includes(declared.begin(), declared.end(), required.begin(),
                       required.end());
}

KernelResult ExternalKernel::dispatch(const BuddyKernelCallV1 &call,
                                      BuddyKernelInvokeV1 fallback) const {
  if (!detail::validSoftmaxCall(call))
    return {BUDDY_KERNEL_INVALID, KernelPath::None, BUDDY_KERNEL_INVALID};
  int32_t status = BUDDY_KERNEL_UNSUPPORTED;
  if (matches(call)) {
    status = descriptor->invoke(&call);
    if (status == BUDDY_KERNEL_OK)
      return {status, KernelPath::External, status};
  }
  if (!fallback)
    return {status, KernelPath::None, status};
  return {fallback(&call), KernelPath::Fallback, status};
}
} // namespace buddy::runtime
