// RUN: buddy-opt %s \
// RUN:     -matmul-transpose-b-vectorization="vf=8 scalable=false" \
// RUN:     -convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
// RUN:     -convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
// RUN:     -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

module{
  func.func private @printMemrefF32(memref<*xf32>)
  func.func private @printMemrefF64(memref<*xf64>)

  func.func @matmul_f32(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>) {
    linalg.matmul_transpose_b
      ins(%a, %b: memref<?x?xf32>, memref<?x?xf32>)
      outs(%c:memref<?x?xf32>)
    return
  }

  func.func @matmul_f64(%a : memref<?x?xf64>, %b : memref<?x?xf64>, %c : memref<?x?xf64>) {
    linalg.matmul_transpose_b
      ins(%a, %b: memref<?x?xf64>, memref<?x?xf64>)
      outs(%c:memref<?x?xf64>)
    return
  }

  func.func @main(){
    // Set up dims.
    %cM = arith.constant 4 : index
    %cN = arith.constant 4 : index
    %cK = arith.constant 4 : index

    //--------------------------------------------------------------------------
    // Test f32 as element type.
    //--------------------------------------------------------------------------

    // Set Init Value.
    %cf1_32 = arith.constant 1.0 : f32

    %A_f32 = memref.alloc(%cM, %cK) : memref<?x?xf32>
    %B_f32 = memref.alloc(%cK, %cN) : memref<?x?xf32>
    %C_f32 = memref.alloc(%cM, %cN) : memref<?x?xf32>

    linalg.fill ins(%cf1_32 : f32) outs(%A_f32 : memref<?x?xf32>)
    linalg.fill ins(%cf1_32 : f32) outs(%B_f32 : memref<?x?xf32>)
    linalg.fill ins(%cf1_32 : f32) outs(%C_f32 : memref<?x?xf32>)

    call @matmul_f32(%A_f32, %B_f32, %C_f32) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()

    // Print output.
    // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
    // CHECK-NEXT: [
    // CHECK-SAME:  [5, 5, 5, 5],
    // CHECK-NEXT:  [5, 5, 5, 5],
    // CHECK-NEXT:  [5, 5, 5, 5],
    // CHECK-NEXT:  [5, 5, 5, 5]
    // CHECK-SAME: ]
    %print_C_f32 = memref.cast %C_f32 : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_C_f32) : (memref<*xf32>) -> ()

    memref.dealloc %C_f32 : memref<?x?xf32>
    memref.dealloc %B_f32 : memref<?x?xf32>
    memref.dealloc %A_f32 : memref<?x?xf32>

    //--------------------------------------------------------------------------
    // Test f64 as element type.
    //--------------------------------------------------------------------------

    // Set Init Value.
    %cf1_64 = arith.constant 1.0 : f64

    %A_f64 = memref.alloc(%cM, %cK) : memref<?x?xf64>
    %B_f64 = memref.alloc(%cK, %cN) : memref<?x?xf64>
    %C_f64 = memref.alloc(%cM, %cN) : memref<?x?xf64>

    linalg.fill ins(%cf1_64 : f64) outs(%A_f64 : memref<?x?xf64>)
    linalg.fill ins(%cf1_64 : f64) outs(%B_f64 : memref<?x?xf64>)
    linalg.fill ins(%cf1_64 : f64) outs(%C_f64 : memref<?x?xf64>)

    call @matmul_f64(%A_f64, %B_f64, %C_f64) : (memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) -> ()

    // Print output.
    // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
    // CHECK-NEXT: [
    // CHECK-SAME:  [5, 5, 5, 5],
    // CHECK-NEXT:  [5, 5, 5, 5],
    // CHECK-NEXT:  [5, 5, 5, 5],
    // CHECK-NEXT:  [5, 5, 5, 5]
    // CHECK-SAME: ]
    %print_C_f64 = memref.cast %C_f64 : memref<?x?xf64> to memref<*xf64>
    call @printMemrefF64(%print_C_f64) : (memref<*xf64>) -> ()

    memref.dealloc %C_f64 : memref<?x?xf64>
    memref.dealloc %B_f64 : memref<?x?xf64>
    memref.dealloc %A_f64 : memref<?x?xf64>

    return
  }
}
