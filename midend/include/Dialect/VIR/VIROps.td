//===-- VIROps.td - Dynamic Vector IR Operation Definitions ---------------===//
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
// This file defines the operation set for the VIR dialect,
// including vector length management and dynamic vector operations.
//
//===----------------------------------------------------------------------===//

#ifndef VIR_VIROPS_TD
#define VIR_VIROPS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "VIR/VIRDialect.td"
include "VIR/VIRTypes.td"

//===----------------------------------------------------------------------===//
// Base VIR Operation Class
//===----------------------------------------------------------------------===//

class VIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<VIR_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Dynamic Vector Region Operation - set_vl
//===----------------------------------------------------------------------===//

def VIR_SetVLOp : VIR_Op<"set_vl"> {
  let summary = "Dynamic Vector Region SetVL Operation.";
  let description = [{
    The `set_vl` operation sets the vector length for subsequent dynamic vector
    operations within its region. This enables runtime vector length management
    for hardware-agnostic vector computations.
  }];
  let arguments = (ins Index:$vl);
  let results = (outs Variadic<AnyTypeOf<[AnyInteger, AnyFloat]>>:$results);
  // TODO: Determine the region limitation.
  let regions = (region AnyRegion:$region);

  let assemblyFormat = "$vl attr-dict `:` type($vl)"
      "$region (`->` type($results)^)?";
}

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

def VIR_LoadOp : VIR_Op<"load"> {
  let summary = "Load operation for dynamic vector IR.";
  let description = [{
    The `load` operation loads data from memory into a vector register.
    This operation supports dynamic vector lengths and can load various data types.
  }];
  let arguments = (ins
    Arg<AnyMemRef, "the reference to load from", [MemRead]>:$base,
    Variadic<Index>:$indices
  );
  let results = (outs VIR_DynamicVectorType:$result);

  let assemblyFormat = "$base `[`$indices`]` attr-dict"
                       "`:` type($base) `->` qualified(type($result))";
}

def VIR_StoreOp : VIR_Op<"store"> {
  let summary = "Store operation for dynamic vector IR.";
  let description = [{
    The `store` operation stores data from a vector register to memory.
    This operation supports dynamic vector lengths and can store various data types.
  }];
  let arguments = (ins
    VIR_DynamicVectorType:$value,
    Arg<AnyMemRef, "the reference to store to", [MemWrite]>:$base,
    Variadic<Index>:$indices
  );

  let assemblyFormat = "$value `,` $base `[` $indices `]` attr-dict "
                       "`:` qualified(type($value)) `->` type($base)";
}

//===----------------------------------------------------------------------===//
// Basic Operations
//===----------------------------------------------------------------------===//

def VIR_ConstantOp : VIR_Op<"constant"> {
  let summary = "Constant operation for dynamic vector IR.";
  let description = [{
    The `constant` operation creates a constant value or vector.
    This operation can create scalar constants or broadcast them to vectors.
  }];
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs VIR_DynamicVectorType:$result);

  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

def VIR_BroadcastOp : VIR_Op<"broadcast"> {
  let summary = "Broadcast operation for dynamic vector IR.";
  let description = [{
    The `broadcast` operation broadcasts a scalar value to a vector.
    This operation is useful for creating vectors with repeated values.
  }];
  let arguments = (ins AnyType:$value);
  let results = (outs VIR_DynamicVectorType:$result);

  let assemblyFormat = "$value attr-dict `:` type($value) `->` qualified(type($result))";
}

//===----------------------------------------------------------------------===//
// Arithmetic Operations
//===----------------------------------------------------------------------===//

def VIR_FMAOp : VIR_Op<"fma", [AllTypesMatch<["lhs", "rhs", "acc", "result"]>]> {
  let summary = "Fused Multiply-Add operation for dynamic vector IR.";
  let description = [{
    The `fma` operation performs fused multiply-add: result = lhs * rhs + acc.
    This operation is commonly used in vector computations for efficiency.
  }];
  let arguments = (ins
    VIR_DynamicVectorType:$lhs,
    VIR_DynamicVectorType:$rhs,
    VIR_DynamicVectorType:$acc
  );
  let results = (outs VIR_DynamicVectorType:$result);

  let assemblyFormat = "$lhs `,` $rhs `,` $acc attr-dict `:` qualified(type($lhs))";
}

#endif // VIR_VIROPS_TD
