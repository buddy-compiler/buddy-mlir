memref.global "private" @gv1 : memref<10xi32> = dense<[12, 13, 15, 16, 89, 90, 98, 77, 66, 17]>
memref.global "private" @gv2 : memref<6xi32> = dense<[12, 45, 67, 90, 123, 131]>

func.func private @printMemrefI32(memref<*xi32>)

func.func @main() -> i32 {
  %cons1 = arith.constant 1 : index
  %index0 = arith.constant dense<[1, 3, 7, 9, 0, 6, 5, 4, 2, 8]> : vector<10xi32> // index vector

  %mask0 = arith.constant dense<[0, 1, 0, 0, 1, 0, 1, 0, 0, 1]> : vector<10xi1> // mask vector
  %value0 = arith.constant dense<[122, 11, 23, 45, 67, 89, 21, 90, 88, 110]> : vector<10xi32>

  %base0 = memref.get_global @gv1 : memref<10xi32>
  vector.scatter %base0[%cons1][%index0], %mask0, %value0 : 
  memref<10xi32>, vector<10xi32>, vector<10xi1>, vector<10xi32>

  %res0 = memref.cast %base0 : memref<10xi32> to memref<*xi32>
  func.call @printMemrefI32(%res0) : (memref<*xi32>) -> ()

  %cons0 = arith.constant 0 : index
  %index1 = arith.constant dense<[0, 3, 5, 2, 1, 4]> : vector<6xi32> // index vector

  %mask1 = arith.constant dense<[1, 1, 1, 0, 0, 1]> : vector<6xi1> // mask vector
  %value1 = arith.constant dense<[123, 45, 67, 78, 90, 23]> : vector<6xi32> 

  %base1 = memref.get_global @gv2 : memref<6xi32>
  vector.scatter %base1[%cons0][%index1], %mask1, %value1 : 
  memref<6xi32>, vector<6xi32>, vector<6xi1>, vector<6xi32>

  %res1 = memref.cast %base1 : memref<6xi32> to memref<*xi32>
  func.call @printMemrefI32(%res0) : (memref<*xi32>) -> ()

  %ret = arith.constant 0 : i32
  return %ret : i32
}
