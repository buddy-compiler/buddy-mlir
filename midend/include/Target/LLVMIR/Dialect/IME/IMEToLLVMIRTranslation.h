//===- IMEToLLVMIRTranslation.h - IME to LLVM IR ----------------*- C++ -*-===//
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
// This provides registration calls for IME dialect to LLVM IR translation.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LLVMIR_DIALECT_IME_IMETOLLVMIRTRANSLATION_H
#define TARGET_LLVMIR_DIALECT_IME_IMETOLLVMIRTRANSLATION_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

namespace buddy {
void registerIMEDialectTranslation(mlir::DialectRegistry &registry);
void registerIMEDialectTranslation(mlir::MLIRContext &context);
} // namespace buddy

#endif
