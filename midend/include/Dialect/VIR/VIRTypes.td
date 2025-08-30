//===-- VIRTypes.td - Dynamic Vector IR Type Definitions ---*- tablegen -*-===//
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
// This file defines the type system for the Dynamic Vector IR (VIR) dialect,
// including dynamic vector types and their associated attributes.
//
//===----------------------------------------------------------------------===//

#ifndef VIR_TYPES
#define VIR_TYPES

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "VIR/VIRAttrs.td"

//===----------------------------------------------------------------------===//
// Dynamic Vector Type Definitions
//===----------------------------------------------------------------------===//

def VIR_DynamicVectorTypeElementType : AnyTypeOf<[AnyInteger, Index, AnyFloat]>
{
  let cppFunctionName = "isValidDynamicVectorTypeElementType";
}

def VIR_DynamicVectorType : TypeDef<VIR_Dialect, "DynamicVector",
    [ShapedTypeInterface, ValueSemantics]> {
  let mnemonic = "vec";
  let summary = "Dynamic Vector Type";
  let description = [{
    Syntax:
    ```
    dynamic-vector-type ::= `vec` `<` static-dim-list
                            dynamic-dim `x` vector-element-type
                            (`,` scaling-factor-attr)? `>`
    static-dim-list ::= (static-dim (`x` static-dim)* `x` )?
    static-dim ::= decimal-literal
    dynamic-dim ::= `?`
    vector-element-type ::= float-type | integer-type | index-type
    scaling-factor-attr ::= scaling-ratio scaling-value
    scaling-value ::= `2` | `4` | `8` | `16` | `32` | `64`
    scaling-ratio ::= `m` | `f`
    ```

    A dynamic vector type with a specified element type and an optional runtime
    length.

    Example:
    ```
    vec<?xf32>
    vec<2x?xi32>
    vec<?xf64, f2>
    vec<?xf32, m2>
    ```
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    VIR_DynamicVectorTypeElementType:$elementType,
    VIR_ScalingFactorAttr:$scalingFactor
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "::llvm::ArrayRef<int64_t>":$shape,
      "Type":$elementType,
      CArg<"ScalingFactorAttr", "{}">:$scalingFactor
    ), [{
      return $_get(elementType.getContext(), shape, elementType, scalingFactor);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Returns if this type is ranked (always true).
    bool hasRank() const { return true; }

    /// Clone this vector type with the given shape and element type. If the
    /// provided shape is `std::nullopt`, the current shape of the type is used.
    DynamicVectorType cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape,
                       Type elementType) const {
      return get(shape.value_or(getShape()), elementType);
    }
  }];
}

#endif // VIR_TYPES
