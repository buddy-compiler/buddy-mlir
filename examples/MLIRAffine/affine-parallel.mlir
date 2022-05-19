module {
  memref.global "private" @gv0 : memref<2x5xf32> = dense<[[0., 1., 2., 3. ,4.],
                                                          [5., 6., 7., 8. ,9.]]>

  memref.global "private" @gv1 : memref<5x2xf32> = dense<[[0., 0.],
                                                          [0., 0.],
                                                          [0., 0.],
                                                          [0., 0.],
                                                          [0., 0.]]>
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @main() {
    %mem0 = memref.get_global @gv0 : memref<2x5xf32>
    %mem1 = memref.get_global @gv1 : memref<5x2xf32>
    %c0 = arith.constant 1 : index
    affine.parallel (%i, %j) = (0, 0) to (2, 5) {
      %0 = affine.load %mem0[%i, %j] : memref<2x5xf32>
      affine.store %0, %mem1[%j, %i] : memref<5x2xf32>
    }
    %print_output0 = memref.cast %mem1 : memref<5x2xf32> to memref<*xf32>
    func.call @printMemrefF32(%print_output0) : (memref<*xf32>) -> ()
    func.return 

    }
}
