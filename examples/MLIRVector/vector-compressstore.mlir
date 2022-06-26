memref.global "private" @gv0 : memref<4x4xi32> = dense<[[12, 13, 56, 67],
                                                         [122, 233, 14, 51],
                                                         [11, 787, 65, 32],
                                                         [90, 88, 77, 66]]>

memref.global "private" @gv1 : memref<8xi32> = dense<[12, 13, 14, 16, 17, 10, 45, 65]>

func.func private @printMemrefI32(memref<*xi32>)

func.func @main() -> i32 {
  %cons1 = arith.constant 1 : index
  %cons0 = arith.constant 0 : index

  %base0 = memref.get_global @gv0 : memref<4x4xi32>
  %mask0 = arith.constant dense<[1, 0, 0, 1]> : vector<4xi1>

  %value0 = arith.constant dense<[111, 34, 56, 29]> : vector<4xi32>
  vector.compressstore %base0[%cons0, %cons1], %mask0, %value0 : 
  memref<4x4xi32>, vector<4xi1>, vector<4xi32>

  %res0 = memref.cast %base0 : memref<4x4xi32> to memref<*xi32>
  func.call @printMemrefI32(%res0) : (memref<*xi32>) -> ()

  %base1 = memref.get_global @gv1 : memref<8xi32>
  %mask1 = arith.constant dense<[1, 0, 1]> : vector<3xi1>
  %value1 = arith.constant dense<[73, 83, 90]> : vector<3xi32>

  vector.compressstore %base1[%cons0], %mask1, %value1 : 
  memref<8xi32> , vector<3xi1>,vector<3xi32>

  %res1 = memref.cast %base1 : memref<8xi32> to memref<*xi32>
  func.call @printMemrefI32(%res1) : (memref<*xi32>) -> ()

  %ret = arith.constant 0 : i32
  return %ret : i32
}
