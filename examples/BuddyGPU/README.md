# Buddy GPU operations

There are different components for this example.

## Async Matmul
This demonstrates the process of making serial operations asynchronous. 
1. We bufferize the operation.
2. We use a naive strategy to map the operations to the GPU.
3. We replace all CPU memory with GPU memory operations without Async.
4. We use official `gpu-async-region` pass to make the operations asynchronous. **NOTICE: A pass to automatically analyze the data flow is on it's way. A demonstration of what the result should look like is in `matmul-converted.mlir`**
5. We lower the operations and execute with MLIR's python wrapper. Which generates random data and compares the results with the GPU, then compare the results with numpy.

## Transform strategies
This demonstrates the process of optimizing the computations for the GPU, using the transform dialect.
Currently, a `memref.alloc` operation inside the GPU kernel is generated to use the shared memory. Yet it's not supported by the GPU backend.
```mlir
scf.for %arg15 = %c0 to %c2048 step %c16 {
%subview_2 = memref.subview %subview[0, %arg15] [128, 16] [1, 1] : memref<128x2048xf32, strided<[2048, 1], offset: ?>> to memref<128x16xf32, strided<[2048, 1], offset: ?>>
%subview_3 = memref.subview %subview_0[%arg15, 0] [16, 128] [1, 1] : memref<2048x128xf32, strided<[5376, 1], offset: ?>> to memref<16x128xf32, strided<[5376, 1], offset: ?>>
%alloc = memref.alloc() : memref<128x16xf32, 3>
%4 = gpu.thread_id  x
%5 = gpu.thread_id  y
```

One way to solve this is to use `gpu.dynamic_shared_memory` operation to replace the original allocation.
```mlir
%0 = arith.constant 0 : index
%1 = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
%2 = memref.view %1[%0][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<4xf32, #gpu.address_space<workgroup>>
memref.store %arg0, %2[%0] : memref<4xf32, #gpu.address_space<workgroup>>
gpu.return
```

The dynamic shared memory's size could be computed using all allocation informations, then it could be specified in the `gpu.launch_func` operation.
```mlir
%shmem = arith.constant 81920 : i32
gpu.launch_func  @matmul_kernel::@matmul_kernel blocks in (%c42, %c42, %c1) threads in (%c4, %c8, %c4) dynamic_shared_memory_size %shmem args(%c128 : index, %memref : memref<5376x2048xf32>, %memref_0 : memref<2048x5376xf32>, %memref_1 : memref<5376x5376xf32>, %c4 : index, %c32 : index, %c0 : index, %c8 : index, %c-1 : index, %c16 : index, %c-16 : index, %cst : f32, %c-128 : index, %c1 : index, %c2048 : index)
```

This operation is not yet supported in Buddy MLIR's upstream LLVM version.