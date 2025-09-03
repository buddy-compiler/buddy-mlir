//===- TransformOps.td ----------------------------------------------------===//
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
// The process in this file references the IREE project,
// which is hereby acknowledged.
// For the license of the IREE project
// please see: https://github.com/iree-org/iree/blob/main/LICENSE
//
//===----------------------------------------------------------------------===//
//
// This file defines the transform operations of the gpu dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TRANSFORM_OPS_TD
#define TRANSFORM_OPS_TD

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"
include "mlir/IR/EnumAttr.td"

// From IREE Common Extension OPs
def HoistStaticAllocOp :  Op<Transform_Dialect, "buddy.hoist_static_alloc",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformEachOpTrait,
     TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Hoist static allocations";
  let description = [{
    Find static allocations and hoist them to the top level.

    #### Return modes
    This transform applies static alloc hoisting the whole region of the operand.

    It does not consume the target handle and always return success.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);

  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";
  let cppNamespace = "mlir::buddy::gpu";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::func::FuncOp funcOp,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def ApplyUnrollVectorsGpuMmaSyncPatternsOp : Op<Transform_Dialect,
    "apply_patterns.buddy.unroll_vectors_gpu_mma_sync",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Populate patterns that unroll vectors. TODO: better documentation.
  }];

  let cppNamespace = "mlir::buddy::gpu";
  let assemblyFormat = "attr-dict";
}

def VectorToMMAConversionOp : Op<Transform_Dialect, "buddy.vector.vector_to_mma_conversion",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformEachOpTrait,
     TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    This converts slices of operations containing vector.contract op into
    mma operations, targetting warp level tensorcore operations. If the vector
    operations are bigger than the native mma size it will first split up those
    vector operations.

    Exactly one of use_wmma or use_mma_sync must be specified.

    #### Return modes

    This transform consumes the target handle and produces a result handle.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       UnitAttr:$use_mma_sync,
                       UnitAttr:$use_wmma);
  let results = (outs);

  let assemblyFormat = [{
    $target
    attr-dict
    `:` functional-type($target, results)
  }];
  let cppNamespace = "mlir::buddy::gpu";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // TRANSFORM_OPS_TD
