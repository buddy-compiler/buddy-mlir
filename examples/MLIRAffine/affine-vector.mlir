module {
  memref.global "private" @gv : memref<5x5xf32> = dense<[[0. , 1. , 2. , 3. , 4.],
                                                        [10., 11., 12., 13. , 14.],
                                                        [20., 21., 22., 23. , 24.],
                                                        [30., 31., 32., 33. , 34.],
                                                        [40., 41., 42., 43. , 44.]]>

  func.func private @printMemrefF32(memref<*xf32>)

  func.func @main() {
    %i0 = arith.constant 1 : index
    %j0 = arith.constant 1 : index
    %i1 = arith.constant 0 : index
    %j1 = arith.constant 0 : index
    %mem = memref.get_global @gv : memref<5x5xf32>
    // Load.
    // Method one.
    %v0 = affine.vector_load %mem[%i0, %j0] : memref<5x5xf32>, vector<2xf32>
    vector.print %v0 : vector<2xf32>
    // Method two.
    %v1 = vector.load %mem[%i0, %j0] : memref<5x5xf32>, vector<4xf32>
    vector.print %v1 : vector<4xf32>
    // Store.
    // Method one.
    affine.vector_store %v0, %mem[%i1, %j1] : memref<5x5xf32>, vector<2xf32>
    %print_out1 = memref.cast %mem : memref<5x5xf32> to memref<*xf32>
    func.call @printMemrefF32(%print_out1) : (memref<*xf32>) -> ()
    // Method two.
    affine.vector_store %v1, %mem[%i1, %j1] : memref<5x5xf32>, vector<4xf32>
    %print_out2 = memref.cast %mem : memref<5x5xf32> to memref<*xf32>
    func.call @printMemrefF32(%print_out2) : (memref<*xf32>) -> () 
    func.return

    }
}
