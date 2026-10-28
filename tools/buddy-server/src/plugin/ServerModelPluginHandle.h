//===- ServerModelPluginHandle.h - Server Model ABI loader ----------------===//
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

#ifndef BUDDY_TOOLS_BUDDY_SERVER_SERVERMODELPLUGINHANDLE_H
#define BUDDY_TOOLS_BUDDY_SERVER_SERVERMODELPLUGINHANDLE_H

#include "buddy/runtime/core/ServerModelABI.h"

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace buddy {
namespace server {

class ServerModelError : public std::runtime_error {
public:
  ServerModelError(BuddyServerStatusCodeV1 status, const std::string &message)
      : std::runtime_error(message), statusValue(status) {}

  BuddyServerStatusCodeV1 status() const noexcept { return statusValue; }

private:
  BuddyServerStatusCodeV1 statusValue;
};

struct ServerModelStatus {
  uint32_t state = BUDDY_SERVER_MODEL_UNLOADED;
  std::string modelName;
  std::string backend;
  std::string message;
  int32_t contextLength = 0;
};

class ServerModelPluginHandle {
public:
  explicit ServerModelPluginHandle(const std::string &path);
  ~ServerModelPluginHandle();

  ServerModelPluginHandle(const ServerModelPluginHandle &) = delete;
  ServerModelPluginHandle &operator=(const ServerModelPluginHandle &) = delete;
  ServerModelPluginHandle(ServerModelPluginHandle &&) = delete;
  ServerModelPluginHandle &operator=(ServerModelPluginHandle &&) = delete;

  void create(const std::string &modelName);
  void load(const BuddyServerLoadArgsV1 &args);
  ServerModelStatus status() const;
  void invoke(const BuddyServerCallV1 &call, BuddyServerEmitFnV1 emit,
              void *userData);

  uint32_t serviceKind() const { return descriptor->serviceKind; }
  uint64_t operationMask() const { return descriptor->operationMask; }
  uint64_t capabilityMask() const { return descriptor->capabilityMask; }
  uint64_t inputModalityMask() const { return descriptor->inputModalityMask; }
  uint32_t resultKind() const { return descriptor->resultKind; }
  uint32_t maxConcurrency() const { return descriptor->maxConcurrency; }
  const std::string &modelType() const { return pluginModelType; }

  static std::string errorMessage(const BuddyServerErrorV1 &error);

private:
  void close();
  void checkError(const BuddyServerErrorV1 &error,
                  BuddyServerStatusCodeV1 status) const;

  void acquireCall() const;
  void releaseCall() const;

  void *library = nullptr;
  const BuddyServerModelDescriptorV1 *descriptor = nullptr;
  void *model = nullptr;
  std::string pluginPath;
  std::string pluginModelType;
  mutable std::mutex callMutex;
  mutable std::condition_variable callCondition;
  mutable uint32_t activeCalls = 0;
  mutable uint32_t liveCalls = 0;
  mutable bool closing = false;
};

const char *serverServiceKindName(uint32_t kind);

} // namespace server
} // namespace buddy

#endif // BUDDY_TOOLS_BUDDY_SERVER_SERVERMODELPLUGINHANDLE_H
