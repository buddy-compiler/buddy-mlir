// RUN: buddy-opt -convert-memcpy-to-gpu -canonicalize %s | FileCheck %s

// CHECK: %memref = gpu.alloc  () : memref<32x32xf32>
// CHECK: %memref_0 = gpu.alloc  () : memref<32x32xf32>
// CHECK: gpu.dealloc  %memref : memref<32x32xf32>
// CHECK: %alloc = memref.alloc() : memref<32x32xf32>
// CHECK: gpu.memcpy  %alloc, %memref_0 : memref<32x32xf32>, memref<32x32xf32>
// CHECK: gpu.dealloc  %memref_0 : memref<32x32xf32>
module attributes {gpu.container_module} {
  func.func @matmul(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>) -> memref<32x32xf32> {
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
    gpu.launch_func  @matmul_kernel::@matmul_kernel blocks in (%c1, %c1, %c1) threads in (%c64, %c2, %c1)  
    return %alloc : memref<32x32xf32>
  }
  gpu.module @matmul_kernel {
    gpu.func @matmul_kernel() kernel attributes {gpu.known_block_size = array<i32: 64, 2, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>} {
      gpu.return
    }
  }
}
