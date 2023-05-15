module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @mutmal(%matA: memref<1024x1024xf32>, %matB: memref<1024x1024xf32>, %matC: memref<1024x1024xf32>) workgroup(%sharedA : memref<32x32xf32, #gpu.address_space<workgroup>>, %sharedB : memref<32x32xf32, #gpu.address_space<workgroup>>) kernel {
      %block_idx = gpu.block_id  x
      %block_idy = gpu.block_id  y
      %block_dimx = gpu.block_dim  x
      %block_dimy = gpu.block_dim  y
      %thread_idx = gpu.thread_id  x
      %thread_idy = gpu.thread_id  y

      %0 = arith.muli %block_idx, %block_dimx : index
      %global_idx = arith.addi %0, %thread_idx : index
      
      %1 = arith.muli %block_idy, %block_dimy : index
      %global_idy = arith.addi %1, %thread_idy : index

      %c1024 = arith.constant 1024 : index

      %tile_k = arith.constant 32 : index
      %step = arith.constant 32 : index

      %sum_0 = arith.constant 0.0 : f32
      %sum_res = affine.for %tile_k_iter = 0 to %step 
       iter_args(%sum_iter = %sum_0) -> (f32) {
        %k_index_start = arith.muli %tile_k_iter, %tile_k : index
        %A_k_index = arith.addi %thread_idx, %k_index_start : index
        %B_k_index = arith.addi %thread_idy, %k_index_start : index

        %2 = memref.load %matA[%global_idy, %A_k_index] : memref<1024x1024xf32>
        memref.store %2, %sharedA[%thread_idy, %thread_idx] : memref<32x32xf32, #gpu.address_space<workgroup>>
        %3 = memref.load %matB[%B_k_index, %global_idx] : memref<1024x1024xf32>
        memref.store %3, %sharedB[%thread_idy, %thread_idx] : memref<32x32xf32, #gpu.address_space<workgroup>>
        gpu.barrier
        %sum_res_inner = affine.for %k_iter = 0 to %tile_k
          iter_args(%sum_iter_inner = %sum_0) -> (f32) {
          %4 = memref.load %sharedA[%thread_idy, %k_iter] : memref<32x32xf32, #gpu.address_space<workgroup>>
          %5 = memref.load %sharedB[%k_iter, %thread_idx] : memref<32x32xf32, #gpu.address_space<workgroup>>
          %6 = arith.mulf %4, %5 : f32
          %7 = arith.addf %sum_iter_inner, %6 : f32
          affine.yield %7 : f32
        }

        %8 = arith.addf %sum_iter, %sum_res_inner : f32
        gpu.barrier
        affine.yield %8 : f32
      }

      memref.store %sum_res, %matC[%global_idy, %global_idx] : memref<1024x1024xf32>
      gpu.return
    }
  }

  func.func @main() {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %cst = arith.constant 1.000000e+00 : f32
    %A = memref.alloc() : memref<1024x1024xf32>
    %B = memref.alloc() : memref<1024x1024xf32>
    %C = memref.alloc() : memref<1024x1024xf32>
    %cast_a = memref.cast %A : memref<1024x1024xf32> to memref<*xf32>
    gpu.host_register %cast_a : memref<*xf32>
    %cast_b = memref.cast %B : memref<1024x1024xf32> to memref<*xf32>
    gpu.host_register %cast_b : memref<*xf32>
    %cast_c = memref.cast %C : memref<1024x1024xf32> to memref<*xf32>
    gpu.host_register %cast_c : memref<*xf32>

    affine.for %i = 0 to 1024 {
      affine.for %j = 0 to 1024 {
        %int_i = arith.index_cast %i : index to i32
        %int_j = arith.index_cast %j : index to i32
        %float_i = arith.sitofp %int_i : i32 to f32
        %float_j = arith.sitofp %int_j : i32 to f32
        memref.store %float_i, %A[%i, %j] : memref<1024x1024xf32>
        memref.store %float_j, %B[%i, %j] : memref<1024x1024xf32>
      }
    }
    gpu.launch_func  @kernels::@mutmal blocks in (%c32, %c32, %c1) threads in (%c32, %c32, %c1) args(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>)
    call @printMemrefF32(%cast_c) : (memref<*xf32>) -> ()
    return
  }

  func.func private @printMemrefF32(memref<*xf32>)
}
