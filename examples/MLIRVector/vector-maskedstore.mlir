memref.global "private" @gv : memref<3x3x3xi32> = dense<[[[12, 1,13 ],[30, 100, 5],[90, 76, 84]],
                                                         [[98, 100, 101], [121, 134, 145], [156, 167, 172]],
                                                        [[184, 191, 300], [22, 455, 500], [23, 27, 12]]]>     
func.func private @printMemrefI32(memref<*xi32>)

func.func @main() -> i32 {
  %mask0 = arith.constant dense<[0, 1, 0, 1, 0, 1, 0, 1, 1]> : vector<9xi1>
  %value0 = arith.constant dense<[12, 78, 54, 32, 34, 36, 39, 90, 21]> : vector<9xi32>

  %base = memref.get_global @gv : memref<3x3x3xi32> 
  %c0 = arith.constant 0 : index

  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index

  %mask1 = arith.constant dense<[1, 0, 0, 0, 0, 1, 1, 1, 1]> : vector<9xi1>
  %value1 = arith.constant dense<[35, 89, 90, 78, 67, 54, 53, 21, 90]> : vector<9xi32>

  vector.maskedstore %base[%c0, %c0, %c2], %mask0, %value0
  :memref<3x3x3xi32>, vector<9xi1>, vector<9xi32>

  %print_out0 = memref.cast %base : memref<3x3x3xi32> to memref<*xi32> 
  func.call @printMemrefI32(%print_out0) : (memref<*xi32>) -> ()  

  vector.maskedstore %base[%c0, %c2, %c1], %mask1, %value1
  :memref<3x3x3xi32>, vector<9xi1>, vector<9xi32>

  %print_out1 = memref.cast %base : memref<3x3x3xi32> to memref<*xi32>
  func.call @printMemrefI32(%print_out1) : (memref<*xi32>) -> ()

  %ret = arith.constant 0 : i32
  return %ret : i32
}
