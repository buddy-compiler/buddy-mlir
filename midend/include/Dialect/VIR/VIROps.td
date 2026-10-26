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

    Without results, the region is lowered to a strip-mined loop: it runs
    once per vector of the target width (then once per remaining element),
    so no value flows from one part of the vectors to the next and the
    region computes what it stores.

    With results (`vector.yield %a, ... : f32, ...`), the region computes on
    whole vectors of length `vl`, which must be a constant: a `vir.reduce`
    reduces all `vl` lanes, and its result can be used in the region (also
    by `vir.broadcast`), carried by `scf.for` / `affine.for` loops in it, or
    yielded. Dynamic vectors may be loop-carried values of those loops.
    The lowering splits each dynamic vector into vl / w vectors of the
    target width w and vl % w scalars; a `vir.reduce` combines them in that
    order.

    The dynamic vectors of a region with results have one static layout,
    e.g. `!vir.vec<2x?xf32>`: a dynamic vector holds its rows of vl lanes
    one after the other (rows * vl lanes, split as above), and a
    `vir.reduce` reduces all of them. A `vir.load` / `vir.store` maps the
    rows to the dims before the last one of the memref, as `vector.load`
    does; the memref must hold the rows one after the other (its last dim
    has vl elements, the dims between match the layout), so that an access
    is one access of rows * vl lanes.
  }];
  let arguments = (ins Index:$vl);
  let results = (outs Variadic<AnyTypeOf<[AnyInteger, AnyFloat]>>:$results);
  // TODO: Determine the region limitation.
  let regions = (region AnyRegion:$region);
  // the lowering maps the values of the region, not entry block arguments
  let hasVerifier = 1;

  let assemblyFormat = "$vl attr-dict `:` type($vl)"
      "$region (`->` type($results)^)?";
}

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

def VIR_LoadOp : VIR_Op<"load"> {
  let summary = "Load operation for dynamic vector IR.";
  let description = [{
    The `load` operation loads a dynamic vector from memory, as
    `vector.load` does: its lanes (vl of them) are along the last dim of
    the memref, and the rows of a multi-dimensional dynamic vector (e.g.
    `!vir.vec<2x?xf32>`) along the dims before it. For a memref of rank n
    and the indices `[i0, ..., i(n-1)]` (one per dim, or all 0 when none
    are given), lane `c` of row `(r1, ..., rk)` is at
    `[i0, ..., i(n-k-1) + r1, ..., i(n-2) + rk, i(n-1) + c]`.
  }];
  let arguments = (ins
    Arg<AnyNon0RankedMemRef, "the reference to load from", [MemRead]>:$base,
    Variadic<Index>:$indices
  );
  let results = (outs VIR_DynamicVectorType:$result);
  let hasVerifier = 1;

  let assemblyFormat = "$base `[`$indices`]` attr-dict"
                       "`:` type($base) `->` qualified(type($result))";
}

def VIR_StoreOp : VIR_Op<"store"> {
  let summary = "Store operation for dynamic vector IR.";
  let description = [{
    The `store` operation stores a dynamic vector to memory, at the
    elements a `vir.load` of the same memref and indices reads.
  }];
  let arguments = (ins
    VIR_DynamicVectorType:$value,
    Arg<AnyNon0RankedMemRef, "the reference to store to", [MemWrite]>:$base,
    Variadic<Index>:$indices
  );
  let hasVerifier = 1;

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

def VIR_FromVectorOp : VIR_Op<"from_vector",
    [AllElementTypesMatch<["source", "result"]>]> {
  let summary = "A fixed-size vector as a dynamic vector.";
  let description = [{
    The `from_vector` operation gives the lanes of a 1-D vector, computed
    outside the region, as a dynamic vector of the region. It is only
    supported in a `vir.set_vl` with results, whose constant vl must be the
    number of lanes of the vector.
  }];
  let arguments = (ins VectorOfRank<[1]>:$source);
  let results = (outs VIR_DynamicVectorType:$result);

  let assemblyFormat = "$source attr-dict `:` type($source) `->` qualified(type($result))";
}

def VIR_ShuffleOp : VIR_Op<"shuffle",
    [Pure, AllTypesMatch<["v1", "v2", "result"]>]> {
  let summary = "Shuffle the rows of two dynamic vectors.";
  let description = [{
    The `shuffle` operation picks the rows (along the leading static dim)
    of its result from the rows of `v1` followed by those of `v2`, as
    `vector.shuffle` does with fixed indices. Row `i` of the result is row
    `mask[i]` of the concatenation of `v1` and `v2`. The mask has one index
    per row. RoPE's rotate_half of a head as two rows `x` is
    `vir.shuffle %neg_x, %x [1, 2]`, i.e. `[-x[1], x[0]]`.

    It is only supported in a `vir.set_vl` with results, whose dynamic
    vectors must lower to one vector each.
  }];
  let arguments = (ins
    VIR_DynamicVectorType:$v1,
    VIR_DynamicVectorType:$v2,
    DenseI64ArrayAttr:$mask
  );
  let results = (outs VIR_DynamicVectorType:$result);
  let hasVerifier = 1;

  let assemblyFormat = "$v1 `,` $v2 $mask attr-dict `:` qualified(type($v1))";
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

def VIR_ReduceOp : VIR_Op<"reduce"> {
  let summary = "Horizontal reduction for dynamic vector IR.";
  let description = [{
    The `reduce` operation reduces a dynamic vector into a scalar using a
    predefined combiner kind.

    v1 semantics:
    - `kind = \"add\"`    : floating-point sum reduction
    - `kind = \"maxnum\"` : floating-point maxnum reduction

    With `reassoc`, the lanes may be combined in any order (the
    `vector.reduction` it lowers to has `fastmath<reassoc>`: an unordered
    reduction, e.g. `vfredusum` on RVV, instead of an ordered one).
  }];
  let arguments = (ins
    VIR_DynamicVectorType:$input,
    AnyFloat:$acc,
    StrAttr:$kind,
    UnitAttr:$reassoc
  );
  let results = (outs AnyFloat:$result);

  let assemblyFormat = "$input `,` $acc attr-dict `:` qualified(type($input))"
                       " `,` type($acc) `->` type($result)";
}

def VIR_SelectOp : VIR_Op<"select",
    [AllTypesMatch<["trueValue", "falseValue", "result"]>]> {
  let summary = "Select operation for dynamic vector IR.";
  let description = [{
    The `select` operation selects elements from `trueValue` or `falseValue`
    based on the boolean `condition` vector.
  }];
  let arguments = (ins
    VIR_DynamicVectorType:$condition,
    VIR_DynamicVectorType:$trueValue,
    VIR_DynamicVectorType:$falseValue
  );
  let results = (outs VIR_DynamicVectorType:$result);

  let assemblyFormat = "$condition `,` $trueValue `,` $falseValue attr-dict "
                       "`:` qualified(type($condition)) `,` qualified(type($trueValue))";
}

def VIR_ExtFOp : VIR_Op<"extf"> {
  let summary = "Floating-point extend for dynamic vector IR.";
  let description = [{
    The `extf` operation extends floating-point element type width
    elementwise on a dynamic vector.
  }];
  let arguments = (ins VIR_DynamicVectorType:$in);
  let results = (outs VIR_DynamicVectorType:$result);

  let assemblyFormat = "$in attr-dict `:` qualified(type($in)) `->` qualified(type($result))";
}

def VIR_TruncFOp : VIR_Op<"truncf"> {
  let summary = "Floating-point truncate for dynamic vector IR.";
  let description = [{
    The `truncf` operation truncates floating-point element type width
    elementwise on a dynamic vector.
  }];
  let arguments = (ins VIR_DynamicVectorType:$in);
  let results = (outs VIR_DynamicVectorType:$result);

  let assemblyFormat = "$in attr-dict `:` qualified(type($in)) `->` qualified(type($result))";
}

#endif // VIR_VIROPS_TD
