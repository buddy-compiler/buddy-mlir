// RUN: buddy-opt %s \
// RUN:     -convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
// RUN:     -convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
// RUN:     -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

module {
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @alloc_2d_filled_f32(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: f32) -> memref<?x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc(%arg0, %arg1, %arg2, %arg3) : memref<?x?x?x?xf32>
    scf.for %arg5 = %c0 to %arg0 step %c1 {
      scf.for %arg6 = %c0 to %arg1 step %c1 {
        scf.for %arg7 = %c0 to %arg2 step %c1 {
          scf.for %arg8 = %c0 to %arg3 step %c1 {
            %iarg8 = arith.index_cast %arg8 : index to i32
            %loopf = arith.sitofp %iarg8 : i32 to f32
            memref.store %loopf, %0[%arg5, %arg6, %arg7, %arg8] : memref<?x?x?x?xf32>
          }
        }
      }
    }
    return %0 : memref<?x?x?x?xf32>
  }

  // Definition for the batch matrix multiplication function
  func.func @buddy_batchmatmul_f32(%a: memref<?x?x?xf32>, %b: memref<?x?x?xf32>, %c: memref<?x?x?xf32>) {
    linalg.batch_matmul 
      ins(%a, %b: memref<?x?x?xf32>, memref<?x?x?xf32>)
      outs(%c: memref<?x?x?xf32>)
    return
  }

  func.func @main() {
    // Constants for matrix dimensions and values
    %cst = arith.constant 0.500000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32

    %batch_size = arith.constant 4 : index
    %mat_m = arith.constant 8 : index
    %mat_k = arith.constant 6 : index
    %mat_n = arith.constant 8 : index

    // Allocate and fill matrices A, B, and C for batch matmul
    %a = call @alloc_2d_filled_f32(%batch_size, %mat_m, %mat_k, %mat_n, %cst) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
    %b = call @alloc_2d_filled_f32(%batch_size, %mat_k, %mat_n, %mat_m, %cst) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
    %c = call @alloc_2d_filled_f32(%batch_size, %mat_m, %mat_n, %mat_n, %cst_0) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>

    // Call batch matrix multiplication function
    call @buddy_batchmatmul_f32(%a, %b, %c) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()

    %c_cast = memref.cast %c : memref<?x?x?x?xf32> to memref<*xf32>

    // Print the result of matrix multiplication
    call @printMemrefF32(%c_cast) : (memref<*xf32>) -> ()

    // Deallocate memory
    memref.dealloc %a : memref<?x?x?x?xf32>
    memref.dealloc %b : memref<?x?x?x?xf32>
    memref.dealloc %c : memref<?x?x?x?xf32>
    return
  }
}
