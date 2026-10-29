//===- ModelError.h - Typed model serving errors ---------------*- C++ -*-===//
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

#ifndef BUDDY_RUNTIME_CORE_MODELERROR_H
#define BUDDY_RUNTIME_CORE_MODELERROR_H

#include <stdexcept>
#include <string>

namespace buddy {
namespace runtime {

/// Transport-independent failures reported by typed serving models.
enum class ModelErrorCode {
  InvalidArgument,
  NotSupported,
  ModelNotReady,
  ResourceExhausted,
  Cancelled,
  DeadlineExceeded,
  ModelError,
  InternalError,
};

/// Carries a stable error category across the typed model interface.
///
/// Server ABI adapters also recognize std::invalid_argument and std::bad_alloc
/// for the two common standard-library error categories. Other failures that
/// need a specific status should use ModelException explicitly.
class ModelException : public std::runtime_error {
public:
  ModelException(ModelErrorCode code, const std::string &message)
      : std::runtime_error(message), errorCode(code) {}

  ModelErrorCode code() const noexcept { return errorCode; }

private:
  ModelErrorCode errorCode;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_CORE_MODELERROR_H
