func.func @main() {
  %data = memref.alloc() : memref<1x6xi32>
  %cst0 = arith.constant 0 : i32
  %cst1 = arith.constant 1 : i32
  %cst2 = arith.constant 2 : i32
  %cst4 = arith.constant 4 : i32
  %cst8 = arith.constant 8 : i32
  %cst16 = arith.constant 16 : i32

  %value = arith.constant 0 : i32

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index

  memref.store %cst0, %data[%c0, %c0] : memref<1x6xi32>
  memref.store %cst1, %data[%c0, %c1] : memref<1x6xi32>
  memref.store %cst2, %data[%c0, %c2] : memref<1x6xi32>
  memref.store %cst4, %data[%c0, %c3] : memref<1x6xi32>
  memref.store %cst8, %data[%c0, %c4] : memref<1x6xi32>
  memref.store %cst16, %data[%c0, %c5] : memref<1x6xi32>
  
  %cast_data = memref.cast %data : memref<1x6xi32> to memref<*xi32>
  gpu.host_register %cast_data : memref<*xi32>
  call @printMemrefI32(%cast_data) : (memref<*xi32>) -> ()

  %t0 = gpu.wait async
  %t1 = gpu.memset async [%t0] %data, %value : memref<1x6xi32>, i32 

  %cast_memset_data = memref.cast %data : memref<1x6xi32> to memref<*xi32>
  gpu.host_register %cast_memset_data : memref<*xi32>
  
  call @printMemrefI32(%cast_memset_data) : (memref<*xi32>) -> ()
  return
}

func.func private @printMemrefI32(memref<*xi32>)