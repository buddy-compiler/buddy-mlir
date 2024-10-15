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

  // Definition for the batch matrix multiplication function
  func.func @buddy_batchmatmul_f32(%A: memref<?x?x?xf32>, %B: memref<?x?x?xf32>, %C: memref<?x?x?xf32>) {
    linalg.batch_matmul 
      ins(%A, %B: memref<?x?x?xf32>, memref<?x?x?xf32>)
      outs(%C: memref<?x?x?xf32>)
    return
  }

  func.func @main(){
      // Set up dims.
      %cBatch = arith.constant 10:index
      %cM = arith.constant 2 : index
      %cN = arith.constant 5 : index
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

      // CHECK: {{ Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 3 offset = 0 sizes = \[10, 2, 5\] strides = \[10, 5, 1\] data = }}
      // CHECK{LITERAL}: [[[8,    8,    8,    8,    8], 
      // CHECK{LITERAL}:   [8,    8,    8,    8,    8]], 
      // CHECK{LITERAL}:  [[8,    8,    8,    8,    8], 
      // CHECK{LITERAL}:   [8,    8,    8,    8,    8]], 
      // CHECK{LITERAL}:  [[8,    8,    8,    8,    8], 
      // CHECK{LITERAL}:   [8,    8,    8,    8,    8]], 
      // CHECK{LITERAL}:  [[8,    8,    8,    8,    8], 
      // CHECK{LITERAL}:   [8,    8,    8,    8,    8]], 
      // CHECK{LITERAL}:  [[8,    8,    8,    8,    8], 
      // CHECK{LITERAL}:   [8,    8,    8,    8,    8]], 
      // CHECK{LITERAL}:  [[8,    8,    8,    8,    8], 
      // CHECK{LITERAL}:   [8,    8,    8,    8,    8]], 
      // CHECK{LITERAL}:  [[8,    8,    8,    8,    8], 
      // CHECK{LITERAL}:   [8,    8,    8,    8,    8]], 
      // CHECK{LITERAL}:  [[8,    8,    8,    8,    8], 
      // CHECK{LITERAL}:   [8,    8,    8,    8,    8]], 
      // CHECK{LITERAL}:  [[8,    8,    8,    8,    8], 
      // CHECK{LITERAL}:   [8,    8,    8,    8,    8]], 
      // CHECK{LITERAL}:  [[8,    8,    8,    8,    8], 
      // CHECK{LITERAL}:   [8,    8,    8,    8,    8]]]

      %print_C = memref.cast %C : memref<?x?x?xf32> to memref<*xf32>
      call @printMemrefF32(%print_C) : (memref<*xf32>) -> ()

      memref.dealloc %C : memref<?x?x?xf32>
      memref.dealloc %B : memref<?x?x?xf32>
      memref.dealloc %A : memref<?x?x?xf32>
      return 
    }
}
