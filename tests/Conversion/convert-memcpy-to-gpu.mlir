// RUN: buddy-opt -convert-memcpy-to-gpu="process-args=1" %s | FileCheck %s

#map = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
module attributes {gpu.container_module} {
  memref.global "private" constant @__constant_1x10x10xf32 : memref<1x10x10xf32> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @matmul(%arg0: memref<1x10x10xf32>, %arg1: memref<1x10x10xf32>) -> memref<1x10x10xf32> {
    // CHECK: %[[d_arg0:.*]] = gpu.alloc  () : memref<1x10x10xf32>
    // CHECK-NEXT: gpu.memcpy  %[[d_arg0]], %arg0 : memref<1x10x10xf32>, memref<1x10x10xf32>
    // CHECK: %[[d_arg1:.*]] = gpu.alloc  () : memref<1x10x10xf32>
    // CHECK-NEXT: gpu.memcpy  %[[d_arg1:.*]], %arg1 : memref<1x10x10xf32>, memref<1x10x10xf32>
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    // CHECK: %[[h_global_data:.*]] = memref.get_global @__constant_1x10x10xf32 : memref<1x10x10xf32>
    // CHECK: %[[d_global_data:.*]] = gpu.alloc  () : memref<1x10x10xf32>
    // CHECK: gpu.memcpy  %[[d_global_data]], %[[h_global_data]] : memref<1x10x10xf32>, memref<1x10x10xf32>
    %0 = memref.get_global @__constant_1x10x10xf32 : memref<1x10x10xf32>
    // CHECK: %[[d_alloc0:.*]] = gpu.alloc  () : memref<1x10x10xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x10x10xf32>
    // CHECK: gpu.launch_func
    gpu.launch_func  @kernel::@fill blocks in (%c10, %c10, %c1) threads in (%c1, %c1, %c1)  args(%c1 : index, %c0 : index, %cst : f32, %alloc : memref<1x10x10xf32>)
    // CHECK: gpu.launch_func
    // CHECK-SAME: %[[d_arg0]]
    // CHECK-SAME: %[[d_arg1]]
    // CHECK-SAME: %[[d_alloc0]]
    gpu.launch_func  @kernel::@matmul blocks in (%c10, %c10, %c1) threads in (%c1, %c1, %c1)  args(%c1 : index, %c0 : index, %arg0 : memref<1x10x10xf32>, %arg1 : memref<1x10x10xf32>, %alloc : memref<1x10x10xf32>, %c10 : index)
    // CHECK: %[[d_alloc1:.*]] = gpu.alloc  () : memref<1x10x10xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x10x10xf32>
    // CHECK: gpu.launch_func
    gpu.launch_func  @kernel::@fill blocks in (%c10, %c10, %c1) threads in (%c1, %c1, %c1)  args(%c1 : index, %c0 : index, %cst : f32, %alloc_0 : memref<1x10x10xf32>)
    // CHECK: gpu.launch_func
    // CHECK-SAME: %[[d_global_data]]
    // CHECK-SAME: %[[d_alloc0]]
    // CHECK-SAME: %[[d_alloc1]]
    gpu.launch_func  @kernel::@matmul blocks in (%c10, %c10, %c1) threads in (%c1, %c1, %c1)  args(%c1 : index, %c0 : index, %0 : memref<1x10x10xf32>, %alloc : memref<1x10x10xf32>, %alloc_0 : memref<1x10x10xf32>, %c10 : index)
    // CHECK: %[[d_result:.*]] = gpu.alloc  () : memref<1x10x10xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x10x10xf32>
    // CHECK: gpu.launch_func
    gpu.launch_func  @kernel::@fill blocks in (%c10, %c10, %c1) threads in (%c1, %c1, %c1)  args(%c1 : index, %c0 : index, %cst : f32, %alloc_1 : memref<1x10x10xf32>)
    // CHECK: gpu.launch_func
    // CHECK-SAME: %[[d_alloc0]]
    // CHECK-SAME: %[[d_alloc1]]
    // CHECK-SAME: %[[d_result]]
    gpu.launch_func  @kernel::@matmul blocks in (%c10, %c10, %c1) threads in (%c1, %c1, %c1)  args(%c1 : index, %c0 : index, %alloc : memref<1x10x10xf32>, %alloc_0 : memref<1x10x10xf32>, %alloc_1 : memref<1x10x10xf32>, %c10 : index)
    // CHECK: gpu.dealloc %[[d_alloc1]] : memref<1x10x10xf32>
    memref.dealloc %alloc_0 : memref<1x10x10xf32>
    // CHECK: gpu.dealloc  %[[d_alloc0]] : memref<1x10x10xf32>
    memref.dealloc %alloc : memref<1x10x10xf32>

    // CHECK: %[[h_alloc:.*]] = memref.alloc() : memref<1x10x10xf32>
    // CHECK-NEXT: gpu.memcpy  %[[h_alloc]], %[[d_result]] : memref<1x10x10xf32>, memref<1x10x10xf32>

    // CHECK: gpu.dealloc  %[[d_arg0]] : memref<1x10x10xf32>
    // CHECK: gpu.dealloc  %[[d_arg1]] : memref<1x10x10xf32>
    // CHECK: gpu.dealloc  %[[d_global_data]] : memref<1x10x10xf32>

    // CHECK: return %[[h_alloc]] : memref<1x10x10xf32>
    return %alloc_1 : memref<1x10x10xf32>
  }
  gpu.module @kernel {
    gpu.func @fill(%arg0: index, %arg1: index, %arg2: f32, %arg3: memref<1x10x10xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>} {
      gpu.return
    }
    gpu.func @matmul(%arg0: index, %arg1: index, %arg2: memref<1x10x10xf32>, %arg3: memref<1x10x10xf32>, %arg4: memref<1x10x10xf32>, %arg5: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>} {
      gpu.return
    }
  }
}
