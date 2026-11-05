//===- LinalgComputationFactsPrinter.h - SSA fact diagnostics -------------===//
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
// Diagnostic printer for LinalgComputationFacts. Emits a stable, line-oriented
// text form of the explicit SSA / Linalg graph for FileCheck and interactive
// debugging. Does not analyze or rewrite IR; pass a fresh facts snapshot after
// any mutation.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_ANALYSIS_LINALGCOMPUTATIONFACTSPRINTER_H
#define BUDDY_ANALYSIS_LINALGCOMPUTATIONFACTSPRINTER_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace buddy::boscame {
struct LinalgComputationFacts;

// Print `facts` to `out`. Every line is prefixed with `indent`. Only fields
// present in the snapshot are shown; absence is rendered as <none>/<unknown>
// rather than guessed from neighboring edges.
void printLinalgComputationFacts(const LinalgComputationFacts &facts,
                                 llvm::raw_ostream &out,
                                 llvm::StringRef indent = "");
} // namespace buddy::boscame

#endif // BUDDY_ANALYSIS_LINALGCOMPUTATIONFACTSPRINTER_H
