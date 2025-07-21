// RUN: buddy-opt %s \
// RUN: 	-conv-vectorization \
// RUN: 	-convert-linalg-to-loops \
// RUN: 	-lower-affine \
// RUN: 	--one-shot-bufferize="bufferize-function-boundaries" \
// RUN: 	-convert-scf-to-cf \
// RUN: 	-convert-cf-to-llvm \
// RUN: 	-convert-vector-to-llvm \
// RUN: 	-convert-arith-to-llvm \
// RUN: 	-finalize-memref-to-llvm \
// RUN: 	-llvm-request-c-wrappers \
// RUN: 	-convert-func-to-llvm \
// RUN: 	-reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0 + d1 - 1)>

module {
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @conv_2d(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    linalg.conv_2d ins (%arg0, %arg1: memref<?x?xf32>, memref<?x?xf32>)
                  outs (%arg2: memref<?x?xf32>)
    return
  }

  func.func @alloc_f32(%arg0: index, %arg1: index, %arg2: f32) -> memref<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
    scf.for %arg3 = %c0 to %arg0 step %c1 {
      scf.for %arg4 = %c0 to %arg1 step %c1 {
        memref.store %arg2, %0[%arg3, %arg4] : memref<?x?xf32>
      }
    }
    return %0 : memref<?x?xf32>
  }

  func.func @main() {
    %c0 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1.000000e+00 : f32
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %current_v1 = arith.constant 3 : index
    %current_v2 = arith.constant 8 : index
    %current_v0 = affine.apply #map0(%current_v2, %current_v1)

    %v0 = call @alloc_f32(%current_v0, %current_v0, %c1) : (index, index, f32) -> memref<?x?xf32>
    %v1 = call @alloc_f32(%current_v1, %current_v1, %c1) : (index, index, f32) -> memref<?x?xf32>
    %v2 = call @alloc_f32(%current_v2, %current_v2, %c0) : (index, index, f32) -> memref<?x?xf32>

    call @conv_2d(%v0, %v1, %v2) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()

    %print_v2 = memref.cast %v2 : memref<?x?xf32> to memref<*xf32>

    // All the elements of the MemRef are the same,
    // only check the first line to verify the correctness.
    // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [8, 8] strides = [8, 1] data =
    // CHECK-NEXT: [
    // CHECK-SAME: [9{{(, 9)*}}],
    call @printMemrefF32(%print_v2) : (memref<*xf32>) -> ()

    memref.dealloc %v0 : memref<?x?xf32>
    memref.dealloc %v1 : memref<?x?xf32>
    memref.dealloc %v2 : memref<?x?xf32>
    return
  }
}
