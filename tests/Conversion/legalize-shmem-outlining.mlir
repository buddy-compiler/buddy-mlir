// RUN: buddy-opt -legalize-shmem-outlining -canonicalize %s | FileCheck %s

// CHECK: module attributes {gpu.container_module}
// CHECK: gpu.launch_func  @matmul_kernel::@matmul_kernel blocks in (%c1, %c1, %c1) threads in (%c64, %c2, %c1)
// CHECK: return %alloc : memref<32x32xf32>
// CHECK: gpu.module @matmul_kernel {
// CHECK-NEXT:   gpu.func @matmul_kernel() kernel attributes {known_block_size = array<i32: 64, 2, 1>, known_grid_size = array<i32: 1, 1, 1>} {
// CHECK-NEXT:     gpu.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
module {
  func.func @matmul(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>) -> memref<32x32xf32> {
    %alloc = memref.alloc() : memref<16x32xf32, 3>
    %alloc_2 = memref.alloc() : memref<32x16xf32, 3>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c2 = arith.constant 2 : index
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %c1, %arg9 = %c1, %arg10 = %c1) threads(%arg5, %arg6, %arg7) in (%arg11 = %c64, %arg12 = %c2, %arg13 = %c1) {
        gpu.terminator
    }
    memref.dealloc %alloc_2 : memref<32x16xf32, 3>
    memref.dealloc %alloc : memref<16x32xf32, 3>
    return %alloc_3 : memref<32x32xf32>
  }
}
