//===- ScheDialect.h - Sche Dialect Definition --------------------*- C++ -*-===//
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
// 
// This is the header file for the sche dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SCHE_SCHEDIALECT_H
#define SCHE_SCHEDIALECT_H

namespace buddy {
namespace sche {
class AsyncTokenType
    : public ::mlir::Type::TypeBase<AsyncTokenType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;
};

void addAsyncDependency(::mlir::Operation *op, ::mlir::Value token);
}
}

#include "mlir/IR/Dialect.h"

#include "Sche/ScheOpsDialect.h.inc"

#include "Sche/ScheOpInterfaces.h.inc"

#endif // SCHE_SCHEDIALECT_H
