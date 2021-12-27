//=======- RVVToLLVMIRTranslation.h - RVV to LLVM IR ------------*- C++ -*-===//
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
// This provides registration calls for RVV dialect to LLVM IR translation.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LLVMIR_DIALECT_RVV_RVVTOLLVMIRTRANSLATION_H
#define TARGET_LLVMIR_DIALECT_RVV_RVVTOLLVMIRTRANSLATION_H

namespace buddy {

/// Register the RVV dialect and the translation from it to the LLVM IR in
/// the given registry.
void registerRVVDialectTranslation(mlir::DialectRegistry &registry);

/// Register the RVV dialect and the translation from it in the registry
/// associated with the given context.
void registerRVVDialectTranslation(mlir::MLIRContext &context);

} // namespace buddy

#endif // TARGET_LLVMIR_DIALECT_RVV_RVVTOLLVMIRTRANSLATION_H
