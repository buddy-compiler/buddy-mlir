memref.global "private" @gv : memref<5x5xf32> = dense<[[5. , 6. , 7. , 8. , 9.],
                                                         [10., 11., 12., 13. , 14.],
                                                         [20., 21., 22., 23. , 24.],
                                                         [30., 31., 32., 33. , 34.],
                                                         [40., 41., 42., 43. , 44.]]>
func.func @main() {
  %c0 = arith.constant 0 : index 
  %c1 = arith.constant 1 : index 
  %c2 = arith.constant 2 : index  
  %v0 = arith.constant dense<[0., 1., 2., 3. ,4. , 5., 6., 7.]> : vector<8xf32>
  %base = memref.get_global @gv : memref<5x5xf32>
  %mask = arith.constant dense<[1,0,1,0,1,0,1,0]> : vector<8xi1>
  %v1 = vector.expandload %base[%c0, %c0], %mask, %v0
  :memref<5x5xf32>, vector<8xi1>, vector<8xf32> into vector<8xf32>     
  vector.print %v1 : vector<8xf32>  
  func.return 
}
