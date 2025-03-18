// RUN: buddy-opt %s \
// RUN:     -convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
// RUN:     -convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
// RUN:     -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module {
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @alloc_3d_filled_f32(%arg0: index, %arg1: index, %arg2: index, %arg3: f32) -> memref<?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc(%arg0, %arg1, %arg2) : memref<?x?x?xf32>
    scf.for %arg4 = %c0 to %arg0 step %c1 {
      scf.for %arg5 = %c0 to %arg1 step %c1 {
        scf.for %arg6 = %c0 to %arg1 step %c1 {
          memref.store %arg3, %0[%arg4, %arg5, %arg6] : memref<?x?x?xf32>
        }
      }
    }
    return %0 : memref<?x?x?xf32>
  }

  func.func @generic_without_inputs(%arg0 : memref<?x?x?xf32>) {
    linalg.generic {indexing_maps = [#map0],
                    iterator_types = ["parallel", "parallel", "parallel"]}
                    outs(%arg0 : memref<?x?x?xf32>) {
      ^bb0(%arg3: f32):
        %cst = arith.constant 2.000000e+00 : f32
        linalg.yield %cst : f32
    }
    return
  }

  func.func @main() {
    // Define this sizes of the MemRef.
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    // Define the data of the MemRef.
    %cst = arith.constant 1.000000e+00 : f32
    // Generate a 3D MemRef.
    %mem = call @alloc_3d_filled_f32(%c1, %c2, %c3, %cst) : (index, index, index, f32) -> memref<?x?x?xf32>
    // Call the linalg generic function.
    call @generic_without_inputs(%mem) : (memref<?x?x?xf32>) -> ()
    // Print output.
    // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [1, 2, 3] strides = [6, 3, 1] data =
    // CHECK-NEXT: [
    // CHECK-SAME:   [
    // CHECK-SAME:     [2, 2, 2],
    // CHECK-NEXT:     [2, 2, 2]
    // CHECK-SAME:   ]
    // CHECK-SAME: ]
    %print_output = memref.cast %mem : memref<?x?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_output) : (memref<*xf32>) -> ()
    // Release the MemRef.
    memref.dealloc %mem : memref<?x?x?xf32>

    return
  }
}
