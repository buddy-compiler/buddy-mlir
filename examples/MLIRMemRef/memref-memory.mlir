module {
  memref.global "private" @gv : memref<4x4xf32> = dense<[[0., 1., 2., 3.],
                                                         [4., 5., 6., 7.],
                                                         [8., 9., 10., 12.],
                                                         [13., 14., 15., 16.]]>
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %mem0 = memref.get_global @gv : memref<4x4xf32>
    %mem1 = memref.cast %mem0 : memref<4x4xf32> to memref<?x?xf32>
    %ele = memref.load %mem1[%c0, %c1] : memref<?x?xf32>
    vector.print %ele : f32

    memref.store %ele, %mem1[%c3, %c1] : memref<?x?xf32>
    %print_mem =  memref.cast %mem1 : memref<?x?xf32> to memref<*xf32>
    func.call @printMemrefF32(%print_mem) : (memref<*xf32>) -> ()

    func.return
  }
}
