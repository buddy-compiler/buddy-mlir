//===-- VIRDialect.td - Dynamic Vector IR Dialect Definition --------------===//
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
// This file defines the core dialect structure for the Dynamic Vector IR (VIR),
// including the dialect registration and namespace configuration.
//
//===----------------------------------------------------------------------===//

#ifndef VIR_VIRDIALECT_TD
#define VIR_VIRDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dynamic Vector IR Dialect Definition
//===----------------------------------------------------------------------===//

def VIR_Dialect : Dialect {
  let name = "vir";
  let summary = "The Dynamic Vector IR (VIR) Dialect";
  let description = [{
    The `VIR` dialect provides a dynamic vector IR that serves as a unified
    hardware abstraction layer for multiple hardware backends, enabling
    portable vector operations across different architectures.
  }];
  let cppNamespace = "::buddy::vir";
  let extraClassDeclaration = [{
      void registerTypes();
      void registerAttrs();
  }];

  // Add type and attribute parsing and printing hooks.
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // VIR_VIRDIALECT_TD
