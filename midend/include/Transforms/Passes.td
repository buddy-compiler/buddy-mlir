//===-- Passes.td - Declare all of buddy's passes ------------ tablegen --===//
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
// This file declares all the passes in buddy.
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def LowerGemminiPass : Pass<"lower-gemmini", "mlir::ModuleOp"> {
  let summary = "lower Gemmini Dialect high level op to intrinsic op. ";
  let options = [
    Option<"dim", "dim", "int64_t", "16","Size of systolic array.">,
    Option<"addrLen", "addr_len", "int64_t", "32", "The length of address.">,
    Option<"accRows", "acc_rows", "int64_t", "1024", "The row of acc.">,
    Option<"bankRows", "bank_rows", "int64_t", "4096", "The row of the bank.">,
    Option<"elemType", "elem_t", "std::string", "\"i8\"", "The type of elem_t.">,
    Option<"accType", "acc_t", "std::string", "\"i32\"", "The type of acc_t.">
  ];
}
