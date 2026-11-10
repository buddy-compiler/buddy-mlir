//===- MathAccuracy.h - Target-independent mathematical accuracy ----------===//
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
// Helpers that read `contract.math` accuracy promises from the enclosing
// func.func. Used by pointwise / approx lowering to decide whether a finite
// abs+rel error budget authorizes an approximate implementation.
//
// Accuracy is an explicit caller promise scoped to one function. Absence of a
// promise never authorizes approximation. For finite reference results the
// permitted error is abs + rel * |result|. Modes such as all-f32 / ieee are
// recorded on the attribute dictionary; NaN payloads and FP exception flags
// are outside this value contract.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_ANALYSIS_MATHACCURACY_H
#define BUDDY_ANALYSIS_MATHACCURACY_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace buddy::contract {

// Return the per-op math accuracy dictionary under the parent function's
// `contract.math` attribute, keyed by the op's dialect-stripped name.
// Empty DictionaryAttr when the function or key is missing.
inline mlir::DictionaryAttr mathAccuracy(mlir::Operation &operation) {
  auto function = operation.getParentOfType<mlir::func::FuncOp>();
  auto accuracy =
      function ? function->getAttrOfType<mlir::DictionaryAttr>("contract.math")
               : mlir::DictionaryAttr();
  return accuracy ? accuracy.getAs<mlir::DictionaryAttr>(
                        operation.getName().stripDialect())
                  : mlir::DictionaryAttr();
}

// True when the op's accuracy promise allows at least the given abs/rel
// budgets (both max_abs_error and max_rel_error must be present and large
// enough). Missing promises return false.
inline bool permitsMathError(mlir::Operation &operation, double absolute,
                             double relative) {
  auto accuracy = mathAccuracy(operation);
  auto abs = accuracy ? accuracy.getAs<mlir::FloatAttr>("max_abs_error")
                      : mlir::FloatAttr();
  auto rel = accuracy ? accuracy.getAs<mlir::FloatAttr>("max_rel_error")
                      : mlir::FloatAttr();
  return abs && rel && abs.getValueAsDouble() >= absolute &&
         rel.getValueAsDouble() >= relative;
}

} // namespace buddy::contract

#endif // BUDDY_ANALYSIS_MATHACCURACY_H
