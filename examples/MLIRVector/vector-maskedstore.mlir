memref.global "private" @gv : memref<3x3x3xi32> = dense<[[[0, 1, 2],[3, 4, 5],[6, 7, 8]],
                                                         [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
                                                        [[18, 19, 20], [21, 22, 23], [24, 25, 26]]]>     
func.func private @printMemrefI32(memref<*xi32>)          

func.func @main() {
  %mask0 = arith.constant dense<[0, 1, 0, 1, 0, 1, 0, 1, 0]> : vector<9xi1>
  %value0 = arith.constant dense<[27, 28, 29, 30, 31, 32, 33, 34, 35]> : vector<9xi32>
  %base = memref.get_global @gv : memref<3x3x3xi32> 
  %c0 = arith.constant 0 : index 
  %c1 = arith.constant 1 : index 
  vector.maskedstore %base[%c0, %c0, %c0], %mask0, %value0
  :memref<3x3x3xi32>, vector<9xi1>, vector<9xi32>
  %print_out0 = memref.cast %base : memref<3x3x3xi32> to memref<*xi32> 
  func.call @printMemrefI32(%print_out0) : (memref<*xi32>) -> ()  

  vector.maskedstore %base[%c0, %c1, %c0], %mask0, %value0
  :memref<3x3x3xi32>, vector<9xi1>, vector<9xi32>
  %print_out1 = memref.cast %base : memref<3x3x3xi32> to memref<*xi32>
  func.call @printMemrefI32(%print_out1) : (memref<*xi32>) -> ()
  func.return 

}