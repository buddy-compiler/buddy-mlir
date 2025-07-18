//===- VectorExpOps.td - Vector Experiment Dialect Ops --------------------===//
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
// This is the top level file for operations in the vector experiment dialect.
//
//===----------------------------------------------------------------------===//

#ifndef VECTOREXP_VECTOREXPOPS_TD
#define VECTOREXP_VECTOREXPOPS_TD

include "VectorExp/VectorExpDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Vector Predication Operation
//===----------------------------------------------------------------------===//

def VectorExp_PredicationOp : VectorExp_Op<"predication"> {
  let summary = "Vector Experiment Predication Operation.";
  let arguments = (ins AnyVectorOfAnyRank:$mask,
                       AnyInteger:$vl
                  );
  let results = (outs AnyType:$result);
  let regions = (region AnyRegion:$region);

  let assemblyFormat = "$mask `,` $vl attr-dict `:` type($mask) `,` type($vl) "
      "$region `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Vector GetVL Operation
//===----------------------------------------------------------------------===//

def VectorExp_GetVLOp : VectorExp_Op<"get_vl"> {
  let summary = "Vector Experiment GetVL Operation.";
  let arguments = (ins TypeAttr:$dtype, IndexAttr:$lmul);
  let results = (outs Index:$result);
  let assemblyFormat = "$dtype `,` $lmul attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Vector SetVL Operation
//===----------------------------------------------------------------------===//

def VectorExp_SetVLOp : VectorExp_Op<"set_vl"> {
  let summary = "Vector Experiment SetVL Operation.";
  let arguments = (ins Index:$vl);
  // TODO: Add optional returns.
  // let results = (outs AnyType:$result);
  let regions = (region AnyRegion:$region);
  let assemblyFormat = "$vl attr-dict `:` type($vl) $region";
}

//===----------------------------------------------------------------------===//
// Vector Load Operation with Dynamic Length
//===----------------------------------------------------------------------===//

def VectorExp_LoadOp : VectorExp_Op<"load", [AttrSizedOperandSegments]> {
  let summary = "reads an n-D slice of memory into an n-D vector";
  let description = [{
    The 'vector.load' operation reads an n-D slice of memory into an n-D
    vector. It takes a 'base' memref, an index for each memref dimension and a
    result vector type as arguments. It returns a value of the result vector
    type. The 'base' memref and indices determine the start memory address from
    which to read. Each index provides an offset for each memref dimension
    based on the element type of the memref. The shape of the result vector
    type determines the shape of the slice read from the start memory address.
    The elements along each dimension of the slice are strided by the memref
    strides. Only unit strides are allowed along the most minor memref
    dimension. These constraints guarantee that elements read along the first
    dimension of the slice are contiguous in memory.

    The memref element type can be a scalar or a vector type. If the memref
    element type is a scalar, it should match the element type of the result
    vector. If the memref element type is vector, it should match the result
    vector type.

    Example 1: 1-D vector load on a scalar memref.
    ```mlir
    %result = vector.load %base[%i, %j] : memref<100x100xf32>, vector<8xf32>
    ```

    Example 2: 1-D vector load on a vector memref.
    ```mlir
    %result = vector.load %memref[%i, %j] : memref<200x100xvector<8xf32>>, vector<8xf32>
    ```

    Example 3:  2-D vector load on a scalar memref.
    ```mlir
    %result = vector.load %memref[%i, %j] : memref<200x100xf32>, vector<4x8xf32>
    ```

    Example 4:  2-D vector load on a vector memref.
    ```mlir
    %result = vector.load %memref[%i, %j] : memref<200x100xvector<4x8xf32>>, vector<4x8xf32>
    ```

    Representation-wise, the 'vector.load' operation permits out-of-bounds
    reads. Support and implementation of out-of-bounds vector loads is
    target-specific. No assumptions should be made on the value of elements
    loaded out of bounds. Not all targets may support out-of-bounds vector
    loads.

    Example 5:  Potential out-of-bound vector load.
    ```mlir
    %result = vector.load %memref[%index] : memref<?xf32>, vector<8xf32>
    ```

    Example 6:  Explicit out-of-bound vector load.
    ```mlir
    %result = vector.load %memref[%c0] : memref<7xf32>, vector<8xf32>
    ```
  }];

  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
      [MemRead]>:$base,
      Optional<AnyInteger>:$vl,
      Variadic<Index>:$indices
      );
  let results = (outs AnyVectorOfAnyRank:$result);

  let extraClassDeclaration = [{
    mlir::MemRefType getMemRefType() {
      return getBase().getType().cast<mlir::MemRefType>();
    }

    mlir::VectorType getVectorType() {
      return getResult().getType().cast<mlir::VectorType>();
    }
  }];

  let hasFolder = 0;
  let hasVerifier = 0;

  let assemblyFormat =
    "$base `[` $indices `]` (`(` `vl` `=` $vl^ `:` type($vl) `)`)? attr-dict `:` type($base) `,` type($result)";
}

#endif // VECTOREXP_VECTOREXPOPS_TD
