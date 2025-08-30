//===-- VIRAttrs.td - Dynamic Vector IR Attribute Definitions -------------===//
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
// This file defines the attribute system for the VIR dialect,
// including scaling factor attributes.
//
//===----------------------------------------------------------------------===//

#ifndef VIR_ATTRS
#define VIR_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "VIR/VIRDialect.td"

def VIR_ScalingFactorAttr : AttrDef<VIR_Dialect, "ScalingFactor"> {
  let mnemonic = "sf";
  let hasCustomAssemblyFormat = 1;
  let summary = "Scaling factor attribute for dynamic vector types";
  let description = [{
    The scaling factor attribute encodes a scaling ratio and value for dynamic
    vector types.

    Syntax:
      scaling-factor-attr ::= scaling-ratio scaling-value
      scaling-ratio ::= "m" | "f"
      scaling-value ::= "2" | "4" | "8" | "16" | "32" | "64"

    Examples:
      m2, f4
  }];

  let parameters = (ins
    "llvm::StringRef":$ratio,   // "m" or "f"
    "int64_t":$value      // 2, 4, 8, 16, 32, 64
  );

  let builders = [
    AttrBuilder<(ins
        "mlir::MLIRContext *":$ctx,
        "llvm::StringRef":$ratio,
        "int64_t":$value
    ), [{
        return get(ctx, ratio, value);
    }]>
  ];

  let extraClassDeclaration = [{
    std::string str() const {
      return (getRatio() + std::to_string(getValue())).str();
    }
  }];
}

#endif // VIR_ATTRS
