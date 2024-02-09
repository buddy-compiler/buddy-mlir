module attributes {gpu.container_module} {
  memref.global "private" constant @__constant_16x16xf32 : memref<16x16xf32> = dense<0.000000e+00>
  func.func @forward(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>) -> memref<16x16xf32> {
    %0 = gpu.wait async
    %e1 = gpu.wait async
    %e2 = gpu.wait async
    %memref, %asyncToken = gpu.alloc async [%0] () : memref<16x16xf32>
    %1 = gpu.memcpy async [%asyncToken] %memref, %arg0 : memref<16x16xf32>, memref<16x16xf32>
    %memref_0, %asyncToken_1 = gpu.alloc async [%e1] () : memref<16x16xf32>
    %2 = gpu.memcpy async [%asyncToken_1] %memref_0, %arg1 : memref<16x16xf32>, memref<16x16xf32>
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %3 = memref.get_global @__constant_16x16xf32 : memref<16x16xf32>
    %memref_4, %asyncToken_5 = gpu.alloc async [%e2] () : memref<16x16xf32>
    %5 = gpu.memcpy async [%asyncToken_5] %memref_4, %memref_4 : memref<16x16xf32>, memref<16x16xf32>
    %gather = gpu.wait async [%1, %2, %5]
    %6 = gpu.launch_func async [%gather] @forward_kernel::@forward_kernel blocks in (%c16, %c16, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<16x16xf32>, %memref_0 : memref<16x16xf32>, %memref_4 : memref<16x16xf32>, %c0 : index, %c16 : index, %c1 : index)
    %8 = gpu.dealloc async [%6] %memref_0 : memref<16x16xf32>
    gpu.wait [%8]
    %alloc = memref.alloc() : memref<16x16xf32>
    %9 = gpu.wait async
    %10 = gpu.memcpy async [%9] %alloc, %memref_4 : memref<16x16xf32>, memref<16x16xf32>
    %11 = gpu.dealloc async [%10] %memref_4 : memref<16x16xf32>
    gpu.wait [%11]
    return %alloc : memref<16x16xf32>
  }
  gpu.module @forward_kernel {
    gpu.func @forward_kernel(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>, %arg2: memref<16x16xf32>, %arg3: index, %arg4: index, %arg5: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 16, 16, 1>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      scf.for %arg6 = %arg3 to %arg4 step %arg5 {
        %2 = memref.load %arg0[%0, %arg6] : memref<16x16xf32>
        %3 = memref.load %arg1[%arg6, %1] : memref<16x16xf32>
        %4 = memref.load %arg2[%0, %1] : memref<16x16xf32>
        %5 = arith.mulf %2, %3 : f32
        %6 = arith.addf %4, %5 : f32
        memref.store %6, %arg2[%0, %1] : memref<16x16xf32>
      }
      gpu.return
    }
  }
}

