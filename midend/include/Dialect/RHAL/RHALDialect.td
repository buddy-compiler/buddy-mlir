//===- RHALDialect.td - RHAL Dialect Definition ------------*- tablegen -*-===//
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
// RuyiAI Hardware Abstraction Layer dialect.
// Describes model execution manifests: weights, code objects, buffers, and
// function dispatch tables.  An rhal.module can be lowered to a .rax
// FlatBuffer by rax-pack.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_MIDEND_RHAL_RHALDIALECT_TD
#define BUDDY_MIDEND_RHAL_RHALDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// RHAL Dialect
//===----------------------------------------------------------------------===//

def RHAL_Dialect : Dialect {
  let name = "rhal";
  let summary = "RuyiAI Hardware Abstraction Layer dialect.";
  let description = [{
    The `rhal` dialect describes model execution manifests.

    An `rhal.module` contains:
    - `rhal.constant`  — external weight tensors (file URIs)
    - `rhal.codeobj`   — compiled kernel libraries (shared objects)
    - `rhal.buffer`    — named I/O and state buffer descriptors
    - `rhal.func`      — function dispatch declarations

    Example:
    ```mlir
    rhal.module @deepseek_r1 attributes {version = "0.1.0"} {
      rhal.constant @weights id = 1 : i32, storage = "external",
                             type = tensor<1777088064xf32>,
                             uri = "file:arg0.data"
      rhal.codeobj @kernels id = 1 : i32, kind = "host_shared_lib",
                            backend = "cpu", uri = "file:model.so"
      rhal.buffer @tokens space = "host", type = tensor<1x1024xi64>
      rhal.func @forward_prefill inputs = ["tokens"],
                                 outputs = ["logits"],
                                 dispatch = "kernels",
                                 args = ["tokens", "logits"]
    }
    ```
  }];

  let cppNamespace = "::buddy::rhal";
}

#endif // BUDDY_MIDEND_RHAL_RHALDIALECT_TD
