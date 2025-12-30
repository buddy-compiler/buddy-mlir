// RUN: buddy-opt %s \
// RUN:     -matmul-vectorization-ggml \
// RUN:     -convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
// RUN:     -convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm -convert-cf-to-llvm \
// RUN:     -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

module{
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @matmul_blis_f32(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>) {
    linalg.matmul
      ins(%a, %b: memref<?x?xf32>, memref<?x?xf32>)
      outs(%c:memref<?x?xf32>)
    return
  }

  func.func @main(){
    // Set up dims.
    %cM = arith.constant 9 : index
    %cN = arith.constant 17 : index
    %cK = arith.constant 4: index

    // -------------------------------------------------------------------------
    // Test f32 as element type.
    // -------------------------------------------------------------------------

    // Set Init Value.
    %cf10_32 = arith.constant 10.0 : f32
    %cf1_32 = arith.constant 1.0 : f32

    %A_f32 = memref.alloc(%cM, %cK) : memref<?x?xf32>
    %B_f32 = memref.alloc(%cK, %cN) : memref<?x?xf32>
    %C_f32 = memref.alloc(%cM, %cN) : memref<?x?xf32>

    linalg.fill ins(%cf1_32 : f32) outs(%A_f32 : memref<?x?xf32>)
    linalg.fill ins(%cf1_32 : f32) outs(%B_f32 : memref<?x?xf32>)
    linalg.fill ins(%cf10_32 : f32) outs(%C_f32 : memref<?x?xf32>)

    call @matmul_blis_f32(%A_f32, %B_f32, %C_f32) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()

    // Print output.
    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}} rank = 2 offset = 0 sizes = [9, 17] strides = [17, 1] data =
    // CHECK-NEXT: [
    // CHECK-SAME:  [14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14], 
    // CHECK-NEXT:  [14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14], 
    // CHECK-NEXT:  [14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14], 
    // CHECK-NEXT:  [14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14], 
    // CHECK-NEXT:  [14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14], 
    // CHECK-NEXT:  [14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14], 
    // CHECK-NEXT:  [14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14], 
    // CHECK-NEXT:  [14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14], 
    // CHECK-NEXT:  [14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14]
    // CHECK-SAME: ]
    %print_C_f32 = memref.cast %C_f32 : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_C_f32) : (memref<*xf32>) -> ()

    memref.dealloc %C_f32 : memref<?x?xf32>
    memref.dealloc %B_f32 : memref<?x?xf32>
    memref.dealloc %A_f32 : memref<?x?xf32>

    return
  }
}
