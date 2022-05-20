module {
  memref.global "private" @gv : memref<4x4xf32> = dense<[[0., 1., 2., 3.],
                                                        [4., 5., 6., 7.],
                                                        [8., 9., 10., 12.],
                                                        [13., 14., 15., 16.]]>
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @main() {
    %mem0 = memref.get_global @gv : memref<4x4xf32>
    %mem1 = memref.cast %mem0 : memref<4x4xf32> to memref<?x?xf32>
    %mem2 = memref.expand_shape %mem1 [[0],[1,2,3]] : memref<?x?xf32> into memref<?x2x?x2xf32>
    %print_output0 = memref.cast %mem2 : memref<?x2x?x2xf32> to memref<*xf32>
    func.call @printMemrefF32(%print_output0) : (memref<*xf32>) -> ()
    %mem3 = memref.expand_shape %mem1 [[0,1],[2,3]] : memref<?x?xf32> into memref<?x2x?x2xf32>
    %print_output1 = memref.cast %mem3 : memref<?x2x?x2xf32> to memref<*xf32>
    func.call @printMemrefF32(%print_output1) : (memref<*xf32>) -> ()
    %mem4 = memref.expand_shape %mem1 [[0,1],[2,3]] : memref<?x?xf32> into memref<1x?x?x2xf32>
    %print_output2 = memref.cast %mem4 : memref<1x?x?x2xf32> to memref<*xf32>
    func.call @printMemrefF32(%print_output2) : (memref<*xf32>) -> ()
    func.return

    }
}
