
// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef BUDDYGPU_TRANSFORM_OPS_TD
#define BUDDYGPU_TRANSFORM_OPS_TD

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

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
  let cppNamespace = "mlir::buddy::buddygpu::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::func::FuncOp funcOp,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

// def ApplyBubbleCollapsePatternsOp : Op<Transform_Dialect,
//     "apply_patterns.buddy.bubble_collapse",
//     [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>,
//      ReportTrackingListenerFailuresOpTrait]> {
//   let description = [{
//     Populate patterns to fold an expanding tensor.expand_shape operation with
//     its producer generic operation by collapsing the dimensions of the generic
//     op.
//   }];

//   let cppNamespace = "mlir::buddy::buddygpu::transform_dialect";
//   let assemblyFormat = "attr-dict";
// }

// def ApplyBubbleExpandPatternsOp : Op<Transform_Dialect,
//     "apply_patterns.buddy.bubble_expand",
//     [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>,
//      ReportTrackingListenerFailuresOpTrait]> {
//   let description = [{
//     Populate patterns to fold an expanding (collapsing) tensor_reshape
//     operation with its producer (consumer) generic operation by expanding
//     the dimensionality of the loop in the generic op.
//   }];

//   let cppNamespace = "mlir::buddy::buddygpu::transform_dialect";
//   let assemblyFormat = "attr-dict";
// }

// def ApplyBubblePackUnpackPatternsOp : Op<Transform_Dialect,
//     "apply_patterns.buddy.bubble_pack_unpack",
//     [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>,
//      ReportTrackingListenerFailuresOpTrait]> {
//   let description = [{
//     Populate patterns to bubble up or down data layout ops across other
//     operations.
//   }];

//   let cppNamespace = "mlir::buddy::buddygpu::transform_dialect";
//   let assemblyFormat = "attr-dict";
// }

// def ApplyFoldFillIntoPadPatternsOp : Op<Transform_Dialect,
//     "apply_patterns.buddy.fold_fill_into_pad",
//     [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>,
//      ReportTrackingListenerFailuresOpTrait]> {
//   let description = [{
//     Populates a pattern that folds
//     "tensor.pad(cst, tensor.extract*(linalg.fill(cst)))" into
//     "linalg.fill(cst, empty)" when the padding constant and the fill constant
//     are the same.
//   }];

//   let cppNamespace = "mlir::buddy::buddygpu::transform_dialect";
//   let assemblyFormat = "attr-dict";
// }

// def ApplyFoldTensorSliceIntoTransferPatternsOp : Op<Transform_Dialect,
//     "apply_patterns.buddy.fold_tensor_slice_into_transfer",
//     [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
//   let description = [{
//     Indicates that tensor.extract_slice -> vector.transfer_read and
//     vector.transfer_write -> tensor.insert_slice op chains should be folded into
//     vector tranfer read and write ops
//   }];

//   let cppNamespace = "mlir::buddy::buddygpu::transform_dialect";
//   let assemblyFormat = "attr-dict";
// }

// def ApplyIreeLinalgElementwiseGreedyFusionPatternsOp : Op<Transform_Dialect,
//     "apply_patterns.buddy.linalg_elementwise_greedy_fusion",
//     [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>,
//      ReportTrackingListenerFailuresOpTrait]> {
//   let description = [{
//     Populate patterns to fuse `linalg.generic` -> `linalg.generic` operations
//     when both operations are fusable elementwise operations.

//     Note: This pattern set is parameterized for usage in IREE, therefore
//     it is called "buddy.linalg_elementwise_greedy_fusion".
//   }];

//   let cppNamespace = "mlir::buddy::buddygpu::transform_dialect";
//   let assemblyFormat = "attr-dict";
// }

// def ApplyPrepareVectorToMMAPatternsOp : Op<Transform_Dialect,
//     "apply_patterns.buddy.prepare_vector_to_mma",
//     [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>,
//      ReportTrackingListenerFailuresOpTrait]> {
//   let description = [{
//     Populate patterns that transform vector ops into a canonical form to
//     convert to MMA matrix operations. If `useNvGpu` is true, then the patterns
//     will populated will prepare for conversion to `nvgpu` mma operations
//     rather than the `gpu` dialect WMMA operations.
//   }];

//   let arguments = (ins DefaultValuedAttr<BoolAttr, "true">:$useNvGpu);
//   let cppNamespace = "mlir::buddy::buddygpu::transform_dialect";
//   let assemblyFormat = "attr-dict";
// }

def ApplyUnrollVectorsGpuMmaSyncPatternsOp : Op<Transform_Dialect,
    "apply_patterns.buddy.unroll_vectors_gpu_mma_sync",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Populate patterns that unroll vectors. TODO: better documentation.
  }];

  let cppNamespace = "mlir::buddy::buddygpu::transform_dialect";
  let assemblyFormat = "attr-dict";
}

def ApplyUnrollVectorsGpuWmmaSyncPatternsOp : Op<Transform_Dialect,
    "apply_patterns.buddy.unroll_vectors_gpu_wmma_sync",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Populate patterns that unroll vectors. TODO: better documentation.
  }];

  let cppNamespace = "mlir::buddy::buddygpu::transform_dialect";
  let assemblyFormat = "attr-dict";
}

// def ApplyCommonSubexpressionEliminationOp : Op<Transform_Dialect, "buddy.apply_cse",
//     [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
//      TransformEachOpTrait,
//      TransformOpInterface,
//      ReportTrackingListenerFailuresOpTrait]> {
//   let description = [{
//     Apply common subexpression elimination. This transform is applied to all
//     ops within the target that are isolated from above.

//     #### Return modes

//     This operation does not consume the target handle and does not produce any
//     handle.
//   }];

//   let arguments = (ins TransformHandleTypeInterface:$target);
//   let assemblyFormat = "$target attr-dict `:` type($target)";
//   let cppNamespace = "mlir::buddy::buddygpu::transform_dialect";

//   let extraClassDeclaration = [{
//     ::mlir::DiagnosedSilenceableFailure applyToOne(
//         ::mlir::transform::TransformRewriter &rewriter,
//         ::mlir::Operation *target,
//         ::mlir::transform::ApplyToEachResultList &results,
//         ::mlir::transform::TransformState &state);
//   }];
// }

// def ApplyLoopIndependentCodeMotionOp : Op<Transform_Dialect, "buddy.apply_licm",
//     [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
//      TransformEachOpTrait,
//      TransformOpInterface,
//      ReportTrackingListenerFailuresOpTrait]> {
//   let description = [{
//     Apply loop-independent code motion and single iteration loop promotion.
//     This transform is applied to all FuncOps within the target.

//     #### Return modes

//     This operation does not consume the target handle and does not produce any
//     handle.
//   }];

//   let arguments = (ins TransformHandleTypeInterface:$target);
//   let assemblyFormat = "$target attr-dict `:` type($target)";
//   let cppNamespace = "mlir::buddy::buddygpu::transform_dialect";

//   let extraClassDeclaration = [{
//     ::mlir::DiagnosedSilenceableFailure applyToOne(
//         ::mlir::transform::TransformRewriter &rewriter,
//         ::mlir::Operation *target,
//         ::mlir::transform::ApplyToEachResultList &results,
//         ::mlir::transform::TransformState &state);
//   }];
// }

// def IREEEliminateEmptyTensorsOp : Op<
//     Transform_Dialect, "buddy.eliminate_empty_tensors",
//     [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
//      TransformEachOpTrait,
//      TransformOpInterface,
//      ReportTrackingListenerFailuresOpTrait]> {
//   let description = [{
//     This is a pre-processing pass for iree.bufferize. It tries to remove
//     tensor.empty ops by replacing them with suitable destination tensors,
//     which can reduce the number of allocations when bufferizing.

//     This transform is not part of iree.bufferize because additional
//     canonicalization are sometimes possible after eliminate_empty_tensors but
//     before iree.bufferize.

//     #### Return modes

//     This transform does not consume the target handle and always return success.
//   }];

//   let arguments = (ins TransformHandleTypeInterface:$target);
//   let results = (outs);
//   let assemblyFormat = "attr-dict $target `:` functional-type($target, results)";
//   let cppNamespace = "mlir::buddy::buddygpu::transform_dialect";
//   let extraClassDeclaration = [{
//     ::mlir::DiagnosedSilenceableFailure applyToOne(
//         ::mlir::transform::TransformRewriter &rewriter,
//         ::mlir::Operation *target,
//         ::mlir::transform::ApplyToEachResultList &results,
//         ::mlir::transform::TransformState &state);
//   }];
// }

// def WorkgroupSwizzleOp :  Op<Transform_Dialect, "buddy.workgroup_swizzle",
//     [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
//      TransformEachOpTrait,
//      TransformOpInterface,
//      ReportTrackingListenerFailuresOpTrait]> {
//   let summary = "Swizzle workgroup ids";
//   let description = [{
//     Swizzle workgroup ids for better cache reuse.

//     This transform implements the following swizzling logic (x, y here are the
//     workgroup ids and tiledx, tiledy are the swizzled workgroup ids) :

//     t_tiledx = (x + (y % tile) * grid_size_x) / tile;
//     t_tiledy = (y / tile) * tile + (x + (y % tile) * grid_size_x) % tile;
//     c = grid_size_y % tile != 0 && ((y / tile) * tile + tile) > grid_size_y;
//     tiledx = c ? x : t_tiledx;
//     tiledy = c ? y : t_tiledy;

//     #### Return modes
//     This transform replaces the old workgroup ids inside the given operation's
//     region with swizzled workgroup ids.

//     It does not consume the target handle and always return success.
//   }];

//   let arguments = (
//       ins TransformHandleTypeInterface:$target,
//           I64Attr:$log_tile
//   );
//   let results = (outs);

//   let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";
//   let cppNamespace = "mlir::buddy::buddygpu::transform_dialect";

//   let extraClassDeclaration = [{
//     ::mlir::DiagnosedSilenceableFailure applyToOne(
//         ::mlir::transform::TransformRewriter &rewriter,
//         ::mlir::func::FuncOp funcOp,
//         ::mlir::transform::ApplyToEachResultList &results,
//         ::mlir::transform::TransformState &state);
//   }];
// }

// From IREE LLVM GPU Extension OPs

// def SynchronizeLoopOp : Op<
//     Transform_Dialect, "buddy.synchronize_loop", [
//       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
//       TransformEachOpTrait,
//       TransformOpInterface,
//       ReportTrackingListenerFailuresOpTrait]> {
//   let description = [{
//     This inserts a gpu.barrier after a given scf.for loop.

//     #### Return modes
//     This transform consumes the scf.for handle and produces a result handle
//     which points to the new scf.for loop generated. It will fail if the loop
//     cannot be pipelined or if there are no shared memory copies.
//   }];

//   let arguments = (
//       ins TransformHandleTypeInterface:$for_op);
//   let results = (outs);

//   let cppNamespace = "mlir::buddy::buddygpu::transform_dialect";

//   let assemblyFormat = [{ 
//     $for_op 
//     attr-dict 
//     `:` functional-type(operands, results)}];

//   let extraClassDeclaration = [{
//     ::mlir::DiagnosedSilenceableFailure applyToOne(
//         ::mlir::transform::TransformRewriter &rewriter,
//         ::mlir::scf::ForOp forOp,
//         ::mlir::transform::ApplyToEachResultList &results,
//         ::mlir::transform::TransformState &state);
//   }];
// }

// def CreateAsyncGroupsOp :
//   Op<Transform_Dialect, "buddy.create_async_groups",
//     [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
//      TransformEachOpTrait,
//      TransformOpInterface,
//      ReportTrackingListenerFailuresOpTrait]> {
//   let description = [{
//     Convert copies to shared memory to async copies. This creates groups
//     of consecutive copies and emit wait operation right after.
//     The input operation is a `func.func`.

//     `use_mma_sync` specifies whether `bypassL1` attributes should be added to the
//     async copies.

//     #### Return modes
//     This op returns a handle to the transformed function, even if nothing
//     changed.
//   }];

//   let arguments = (ins TransformHandleTypeInterface:$target,
//                    UnitAttr:$use_mma_sync);
//   let results = (outs);

//   let assemblyFormat = [{ 
//     $target 
//     attr-dict 
//     `:` functional-type(operands, results)}];
//   let cppNamespace = "mlir::buddy::buddygpu::transform_dialect";

//   let extraClassDeclaration = [{
//     ::mlir::DiagnosedSilenceableFailure applyToOne(
//         ::mlir::transform::TransformRewriter &rewriter,
//         ::mlir::func::FuncOp target,
//         ::mlir::transform::ApplyToEachResultList &results,
//         ::mlir::transform::TransformState &state);
//   }];
// }


def VectorToWarpExecuteOnLane0Op : Op<Transform_Dialect, "buddy.vector.to_warp_execute_on_lane_0",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Given an scf.if target predicated by `if (threadIdx.x == 0)`, rewrite its
    body to vector.execute_on_lane_0 running ***on a single warp***.

    The warp size is determined by the `warp_size` attribute (it is generally
    32 but we do not want to hardcode it).

    This rewrite only applies if it can be determined from the IR (i.e. from
    the surrounding IREE::HAL::ExecutableExportOp) that the number of threads
    along the warp dimension is a multiple of the warp size. The transformation
    bails on non-perfect multiples of the warp size that would not properly
    distribute.

    This is the first of two step towards apply vector distribution to a single
    warp.


    Return modes:
    =============
    This operation ignores non-scf::IfOp ops and drops them in the return.

    If all the operations referred to by the `target` operand are properly
    properly, the transform succeeds. Otherwise the transform silently fails.

    If the transform is anchored at a top-level that is not isolated from above,
    the transform definitely fails.

    If the transform cannot find a proper HAL::ExecutableExportOp with a
    well-formed workgroup_size 3-entry attribute such that the threadIdx.x
    component is a multiple of warp_size, the transform silently fails.
    If the scf::ForOp predicate does not predicate on threadIdx.x == 0, the
    transform silently fails.

    Otherwise the transformation succeeds and the returned handle points to the
    produced vector::WarpExecuteOnThread0Op.


    Example:
    ========

    ```
    hal.executable.export public @foo ... { workgroup_size = [64: index, 1: index, 1: index] }
    builtin.module {
      func.func @foo() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %0 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<128xf32>
        %1 = gpu.thread_id  x
        %2 = arith.cmpi ult, %1, %c1 : index
        scf.if %2 {
          %3 = arith.constant dense<1.0> : vector<128xf32>
          vector.transfer_write %3, %0[%c0] : vector<128xf32>, memref<128xf32>
        }
      }
    }
    ```

    rewrites to:

    ```
    hal.executable.export public @foo ... { workgroup_size = [64: index, 1: index, 1: index] }
    builtin.module {
      func.func @foo() {
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c32 = arith.constant 32 : index
        %cst = arith.constant dense<1.000000e+00> : vector<128xf32>
        %0 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<128xf32>
        %1 = gpu.thread_id  x
        %2 = arith.cmpi ult, %1, %c32 : index
        // Single-warp guard filters out threads 32-63.
        scf.if %2 {
          vector.warp_execute_on_lane_0(%1)[32] {
            %cst = arith.constant dense<1.000000e+00> : vector<128xf32>
            vector.transfer_write %cst, %0[%c0] : vector<128xf32>, memref<128xf32>
          }
        }
      }
    }
    ```
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                   DefaultValuedAttr<I64Attr, "{}">:$warp_size);
  let results = (outs TransformHandleTypeInterface:$result);

  let assemblyFormat = "$target attr-dict `:` functional-type($target, $result)";
  let cppNamespace = "mlir::buddy::buddygpu::transform_dialect";

  let builders = [
    OpBuilder<(ins "::mlir::Value":$target, "int64_t":$warpSize)>
  ];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::scf::IfOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def VectorWarpDistributionOp : Op<Transform_Dialect, "buddy.vector.warp_distribute",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformEachOpTrait,
     TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Given a vector.warp_execute_on_lane_0, apply the patterns to rewrite into
    distributed form with warp synchronization. This produces IR that runs
    ***on a single warp***.

    IR that cannot be distributed will be predicated by `if (threadIdx.x == 0)`.

    This is the second step of two for applying vector distribution to a single
    warp.


    Return modes:
    =============
    This operation applies a number of patterns to rewrite vector IR into
    distributed warp form. To apply these patterns, this operation must target
    an operation that is isolated from above, otherwise the transform definitely
    fails.

    Patterns sets are applied in the following order:
      - applyMultiReductionLoweringPatterns
      - applyVectorTransferWriteDistribution
      - applyPropagateVectorDistribution
      - applyWarpExecuteOnLane0ToScf

    If any of the pattern sets fail to apply, the transformation definitely
    fails.

    Otherwise the transformation is successful and no result is returned.


    Example:
    ========

    ```
    hal.executable.export public @foo ... { workgroup_size = [64: index, 1: index, 1: index] }
    builtin.module {
      func.func @foo() {
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c32 = arith.constant 32 : index
        %cst = arith.constant dense<1.000000e+00> : vector<128xf32>
        %0 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<128xf32>
        %1 = gpu.thread_id  x
        %2 = arith.cmpi ult, %1, %c32 : index
        // Single-warp guard filters out threads 32-63.
        scf.if %2 {
          vector.warp_execute_on_lane_0(%1)[32] {
            %cst = arith.constant dense<1.000000e+00> : vector<128xf32>
            vector.transfer_write %cst, %0[%c0] : vector<128xf32>, memref<128xf32>
          }
      }
      }
    }
    ```

    distributes to:

    ```
    hal.executable.export public @foo ... { workgroup_size = [64: index, 1: index, 1: index] }
    builtin.module {
      func.func @foo() {
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c32 = arith.constant 32 : index
        %cst = arith.constant dense<1.000000e+00> : vector<128xf32>
        %0 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<128xf32>
        %1 = gpu.thread_id  x
        %2 = arith.cmpi ult, %1, %c32 : index
        // Single-warp guard filters out threads 32-63.
        scf.if %2 {
          %3 = arith.cmpi eq, %1, %c0 : index
          %4 = memref.alloc() : memref<128xf32, 3>
          // Single-thread guard runs on thread 0 only.
          scf.if %3 {
            vector.store %cst, %4[%c0] : memref<128xf32, 3>, vector<128xf32>
          }
          %5 = arith.muli %1, %c4 : index
          %6 = vector.load %4[%5] : memref<128xf32, 3>, vector<4xf32>
          %7 = affine.apply #map()[%1]
          vector.transfer_write %6, %0[%7] {in_bounds = [true]} : vector<4xf32>, memref<128xf32>
        }
      }
    }
    ```
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);

  let assemblyFormat = [{
    $target 
    attr-dict 
    `:` functional-type($target, results)
  }];
  let cppNamespace = "mlir::buddy::buddygpu::transform_dialect";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::mlir::Value":$target)>
  ];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::func::FuncOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

// def VectorToMMAConversionOp : Op<Transform_Dialect, "buddy.vector.vector_to_mma_conversion",
//     [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
//      TransformEachOpTrait,
//      TransformOpInterface,
//      ReportTrackingListenerFailuresOpTrait]> {
//   let description = [{
//     This converts slices of operations containing vector.contract op into
//     mma operations, targetting warp level tensorcore operations. If the vector
//     operations are bigger than the native mma size it will first split up those
//     vector operations.

//     Exactly one of use_wmma or use_mma_sync must be specified.

//     #### Return modes

//     This transform consumes the target handle and produces a result handle.
//   }];

//   let arguments = (ins TransformHandleTypeInterface:$target,
//                        UnitAttr:$use_mma_sync,
//                        UnitAttr:$use_wmma);
//   let results = (outs);

//   let assemblyFormat = [{
//     $target
//     attr-dict
//     `:` functional-type($target, results)
//   }];
//   let cppNamespace = "mlir::buddy::buddygpu::transform_dialect";

//   let extraClassDeclaration = [{
//     ::mlir::DiagnosedSilenceableFailure applyToOne(
//         ::mlir::transform::TransformRewriter &rewriter,
//         ::mlir::Operation *target,
//         ::mlir::transform::ApplyToEachResultList &results,
//         ::mlir::transform::TransformState &state);
//   }];
// }

// def EliminateGpuBarriersOp :
//   Op<Transform_Dialect, "buddy.eliminate_gpu_barriers",
//     [FunctionalStyleTransformOpTrait,
//      MemoryEffectsOpInterface,
//      TransformEachOpTrait,
//      TransformOpInterface,
//      ReportTrackingListenerFailuresOpTrait]> {
//   let description = [{
//     Removes unnecessary GPU barriers from the function. If a barrier does not
//     enforce any conflicting pair of memory effects, including a pair that is
//     enforced by another barrier, it is unnecessary and can be removed.

//     #### Return modes

//     Consumes the operand handle and produces a new handle to the function after
//     rewriting.
//   }];

//   let arguments = (ins TransformHandleTypeInterface:$target);
//   let results = (outs TransformHandleTypeInterface:$result);

//   let assemblyFormat = [{ $target attr-dict `:` functional-type(operands, results)}];
//   let cppNamespace = "mlir::buddy::buddygpu::transform_dialect";

//   let builders = [
//     OpBuilder<(ins "Value":$target)>
//   ];

//   let extraClassDeclaration = [{
//     ::mlir::DiagnosedSilenceableFailure applyToOne(
//         ::mlir::transform::TransformRewriter &rewriter,
//         ::mlir::func::FuncOp target,
//         ::mlir::transform::ApplyToEachResultList &results,
//         ::mlir::transform::TransformState &state);
//   }];
// }

// def PipelineSharedMemoryCopiesOp : Op<
//     Transform_Dialect, "buddy.pipeline_shared_memory_copies", [
//       FunctionalStyleTransformOpTrait,
//       MemoryEffectsOpInterface,
//       TransformEachOpTrait,
//       TransformOpInterface,
//       ReportTrackingListenerFailuresOpTrait]> {
//   let description = [{
//     This applies software pipelining to a given scf.for loop. The pipelining
//     strategy will look for a copy to shared memory and pipeline it to overlap
//     it with the rest of the loop.
//     It is user responsability to ensure that there are no dependency between
//     `depth` iterations of the loop by using multi-buffering.

//     `depth` will indicate how many stages the software pipeline should have.
//     `peel_epilogue` allows to force the epilogue to be peeled out instead of
//     potentially using predicated operations for the epilogue phase.

//     #### Return modes
//     This transform consumes the scf.for handle and produces a result handle
//     which points to a) the new scf.for loop generated (success case) or b) the
//     existing scf.for loop (failure case).
//   }];

//   let arguments = (
//       ins TransformHandleTypeInterface:$for_op,
//           I64Attr:$depth,
//           UnitAttr:$peel_epilogue,
//           UnitAttr:$use_mma_sync);
//   let results = (outs TransformHandleTypeInterface:$result);

//   let cppNamespace = "mlir::buddy::buddygpu::transform_dialect";

//   let assemblyFormat = [{ 
//     $for_op 
//     attr-dict 
//     `:` functional-type(operands, results)}];

//   let extraClassDeclaration = [{
//     ::mlir::DiagnosedSilenceableFailure applyToOne(
//         ::mlir::transform::TransformRewriter &rewriter,
//         ::mlir::scf::ForOp forOp,
//         ::mlir::transform::ApplyToEachResultList &results,
//         ::mlir::transform::TransformState &state);
//   }];
// }

#endif // BUDDYGPU_TRANSFORM_OPS_TD
