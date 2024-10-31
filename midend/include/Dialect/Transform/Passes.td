//===- Passes.td ---------------------------------------------------------===//
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
include "mlir/Pass/PassBase.td"

// TransformDialectInterpreter
def TransformDialectInterpreter : Pass<"transform-dialect-interpreter", "ModuleOp"> {
    let summary = "Apply transform dialect operations one by one";
    let constructor = "mlir::createTransformDialectInterpreter()";
    let options = [
        Option<"eraseAfter", "erase-after", "bool", "false", "erase Transform Ops after applied">
    ];
}
