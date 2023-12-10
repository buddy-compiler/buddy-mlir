// RUN: mlir-opt %s -convert-gpu-to-nvvm -cse -canonicalize | FileCheck %s

gpu.module @modules {
  llvm.mlir.global internal @__dynamic_shmem__0() {addr_space = 3 : i32, alignment = 4 : i64} : !llvm.array<0 x i8>
  llvm.mlir.global internal @__dynamic_shmem__1() {addr_space = 3 : i32, alignment = 4 : i64} : !llvm.array<0 x i8>  
  llvm.mlir.global internal @__dynamic_shmem__2() {alignment = 16 : i64} : !llvm.array<0 x i8>  
  gpu.func @dynamic_shared_memory_kernel(%d : index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>} {    
    %c1 = arith.constant 1 : index
    %c8192 = arith.constant 8192 : index
    %c16384 = arith.constant 16384 : index
    %shmem = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
    %shmem2 = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>

    %0 = memref.view %shmem[%c8192][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<32x64xf32, #gpu.address_space<workgroup>>
    "test.use.shared.memory"(%0) : (memref<32x64xf32, #gpu.address_space<workgroup>>) -> ()

    %1 = memref.view %shmem[%c16384][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<32x64xf32, #gpu.address_space<workgroup>>
    "test.use.shared.memory"(%1) : (memref<32x64xf32, #gpu.address_space<workgroup>>) -> ()
    gpu.return
  }

  gpu.func @gpu_device_function()  {    
    %c8192 = arith.constant 8192 : index
    %shmem = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
    %0 = memref.view %shmem[%c8192][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<32x64xf32, #gpu.address_space<workgroup>>
    "test.use.shared.memory"(%0) : (memref<32x64xf32, #gpu.address_space<workgroup>>) -> ()
    gpu.return
  }

  func.func @func_device_function()  {    
    %c8192 = arith.constant 8192 : index
    %shmem = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
    %0 = memref.view %shmem[%c8192][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<32x64xf32, #gpu.address_space<workgroup>>
    "test.use.shared.memory"(%0) : (memref<32x64xf32, #gpu.address_space<workgroup>>) -> ()
    func.return
  }
}
