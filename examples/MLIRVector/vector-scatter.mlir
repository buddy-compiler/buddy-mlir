memref.global "private" @gv : memref<8xi32> = dense<[1, 2, 3, 4, 5, 6, 7, 8]>

func.func private @printMemrefI32(memref<*xi32>) 

func.func @main() {
  %c0 = arith.constant 0 : index 
  %index_vec0  = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xi32>
  %mask0 = arith.constant dense<[1, 0, 1, 0, 1, 0, 1, 0]> : vector<8xi1>
  %value0 = arith.constant dense<[9, 10, 11, 12, 13, 14, 15, 16]> : vector<8xi32>
  %base0 = memref.get_global @gv : memref<8xi32>
  vector.scatter %base0[%c0][%index_vec0], %mask0, %value0
  : memref<8xi32>, vector<8xi32>, vector<8xi1>, vector<8xi32>  
  %print_out0 = memref.cast %base0 : memref<8xi32> to memref<*xi32>
  func.call @printMemrefI32(%print_out0) : (memref<*xi32>) -> ()

  %index_vec1 = arith.constant dense<[7, 6, 5, 4, 3, 2, 1, 0]> : vector<8xi32>
  vector.scatter %base0[%c0][%index_vec1], %mask0, %value0
  : memref<8xi32>, vector<8xi32>, vector<8xi1>, vector<8xi32>
  %print_out1 = memref.cast %base0 : memref<8xi32> to memref<*xi32>
  func.call @printMemrefI32(%print_out1) : (memref<*xi32>) -> ()
  func.return 

}