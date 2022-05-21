memref.global "private" @gv0 : memref<5xf32> = dense<[0. , 1. , 2. , 3. , 4.]>
memref.global "private" @gv1 : memref<4x4xf32> = dense<[[0., 1., 2., 3.],
                                                        [4., 5., 6., 7.],
                                                        [8., 9., 10., 11.],
                                                        [12., 13., 14., 15.]]>
func.func private @printMemrefF32(memref<*xf32>)

func.func @main() {
  %c0 = arith.constant 0 : index 
  %base0 = memref.get_global @gv0 : memref<5xf32>
  %mask0 = arith.constant dense<[1 , 0, 1]> : vector<3xi1>
  %value0 = arith.constant dense<[5.0, 6.0, 7.0]> : vector<3xf32>
  vector.compressstore %base0[%c0], %mask0, %value0 
  : memref<5xf32>, vector<3xi1>, vector<3xf32>
  %print_out0 = memref.cast %base0 : memref<5xf32> to memref<*xf32>
  func.call @printMemrefF32(%print_out0) : (memref<*xf32>) -> ()

  %base1 = memref.get_global @gv1 : memref<4x4xf32>
  %mask1 = arith.constant dense<[1, 0, 1, 0, 1, 0, 1, 0]> : vector<8xi1>
  %value1 = arith.constant dense<[16., 17., 18., 19., 20., 21., 22., 23.]> : vector<8xf32>
  vector.compressstore %base1[%c0, %c0], %mask1, %value1
  :memref<4x4xf32>, vector<8xi1>, vector<8xf32>
  %print_out1 = memref.cast %base1 : memref<4x4xf32> to memref<*xf32>
  func.call @printMemrefF32(%print_out1) : (memref<*xf32>) -> ()
  func.return 

}
