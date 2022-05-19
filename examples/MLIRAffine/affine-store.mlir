memref.global "private" @gv : memref<4xf32> = dense<[0. , 0. , 0. , 0. ]>
#map0 = affine_map<(d0, d1) -> (d0 + d1)>

func.func private @printMemrefF32(memref<*xf32>)

func.func @main() {
  %mem = memref.get_global @gv : memref<4xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 1. : f32
  %c3 = arith.constant 2. : f32
  %c4 = arith.constant 3. : f32
  affine.for %idx0 = 0 to 2 {
    affine.for %idx1 = 0 to 2 {
      // Method one.
      affine.store %c2,%mem[%idx0 + %idx1] : memref<4xf32>
      %print_out1 = memref.cast %mem : memref<4xf32> to  memref<*xf32>
      func.call @printMemrefF32(%print_out1) : (memref<*xf32>) -> ()
      // Method two.
      %idx = affine.apply #map0(%idx0, %idx1)
      affine.store %c3, %mem[%idx] : memref<4xf32>
      %print_cout2 = memref.cast %mem : memref<4xf32> to memref<*xf32>
      func.call @printMemrefF32(%print_cout2) : (memref<*xf32>) -> ()
    }
  }
  // Method three.
  %idx = arith.addi %c0, %c1 : index
  affine.store %c4, %mem[symbol(%idx)] : memref<4xf32>
  %print_out3 = memref.cast %mem : memref<4xf32> to memref<*xf32>
  func.call @printMemrefF32(%print_out3) : (memref<*xf32>) -> ()
  func.return

}
