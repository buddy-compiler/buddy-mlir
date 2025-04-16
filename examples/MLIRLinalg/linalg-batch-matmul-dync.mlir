// RUN: buddy-opt  %s -convert-linalg-to-loops -expand-strided-metadata -lower-affine \
// RUN:   -convert-scf-to-cf -convert-cf-to-llvm -convert-vector-to-llvm -finalize-memref-to-llvm \
// RUN:   -convert-arith-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

module {
  func.func private @printMemrefF32(memref<*xf32>)

  // Definition for the batch matrix multiplication function
  func.func @buddy_batchmatmul_f32(%A: memref<?x?x?xf32>, %B: memref<?x?x?xf32>, %C: memref<?x?x?xf32>) {
    linalg.batch_matmul
      ins(%A, %B: memref<?x?x?xf32>, memref<?x?x?xf32>)
      outs(%C: memref<?x?x?xf32>)
    return
  }

  func.func @main(){
      // Set up dims.
      %cBatch = arith.constant 2:index
      %cM = arith.constant 2 : index
      %cN = arith.constant 3 : index
      %cK = arith.constant 4 : index

      // Set Init Value.
      %cf1 = arith.constant 1.0 : f32
      %cf2 = arith.constant 2.0 : f32
      %c0 = arith.constant 0.0 : f32

      %A = memref.alloc(%cBatch,%cM, %cK) : memref<?x?x?xf32>
      %B = memref.alloc(%cBatch,%cK, %cN) : memref<?x?x?xf32>
      %C = memref.alloc(%cBatch,%cM, %cN) : memref<?x?x?xf32>

      linalg.fill
      ins(%cf1 : f32)
      outs(%A:memref<?x?x?xf32>)

      linalg.fill
      ins(%cf2 : f32)
      outs(%B:memref<?x?x?xf32>)

      linalg.fill
      ins(%c0 : f32)
      outs(%C:memref<?x?x?xf32>)

      call @buddy_batchmatmul_f32(%A, %B, %C) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()

      %print_C = memref.cast %C : memref<?x?x?xf32> to memref<*xf32>
      call @printMemrefF32(%print_C) : (memref<*xf32>) -> ()

      memref.dealloc %C : memref<?x?x?xf32>
      memref.dealloc %B : memref<?x?x?xf32>
      memref.dealloc %A : memref<?x?x?xf32>
      return
  }
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [2, 2, 3] strides = [6, 3, 1] data =
// CHECK{LITERAL}: [[[8,    8,    8],
// CHECK{LITERAL}:   [8,    8,    8]],
// CHECK{LITERAL}:  [[8,    8,    8],
// CHECK{LITERAL}:   [8,    8,    8]]]
