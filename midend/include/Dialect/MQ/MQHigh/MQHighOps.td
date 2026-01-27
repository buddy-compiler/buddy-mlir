//===- MQHighOps.td - MQHigh dialect operation definitions --------*- tablegen -*-===//
//
// This file defines the operations for the MQHigh dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_MQHIGH_OPS
#define BUDDY_MQHIGH_OPS

include "MQ/MQHigh/MQHighDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// MatMul operation
//===----------------------------------------------------------------------===//

def MQHigh_MatMulOp : MQHigh_Op<"matmul", [Pure]> {
  let summary = "Matrix multiplication operation for MoonQuest accelerator";
  let description = [{ 
    Performs matrix multiplication on the MoonQuest accelerator.
    Supports different data types and layouts.
  }];

  let arguments = (ins
    AnyTensor:$A,
    AnyTensor:$B,
    Optional<AnyTensor>:$C
  );

  let results = (outs
    AnyTensor:$result
  );

  let assemblyFormat = "$A $B $C attr-dict `:` type($A) `,` type($B) `,` type($C) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Pack operation
//===----------------------------------------------------------------------===//

def MQHigh_PackOp : MQHigh_Op<"pack", [Pure]> {
  let summary = "Pack operation for MoonQuest accelerator";
  let description = [{ 
    Packs data into a format suitable for the MoonQuest accelerator.
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs
    AnyTensor:$result
  );

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Unpack operation
//===----------------------------------------------------------------------===//

def MQHigh_UnpackOp : MQHigh_Op<"unpack", [Pure]> {
  let summary = "Unpack operation for MoonQuest accelerator";
  let description = [{ 
    Unpacks data from the MoonQuest accelerator format to standard format.
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs
    AnyTensor:$result
  );

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

#endif // BUDDY_MQHIGH_OPS
