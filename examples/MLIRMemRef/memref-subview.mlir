#map0 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
module {
  memref.global "private" @gv : memref<5x5xf32> = dense<[[0. , 1. , 2. , 3. , 4.],
                                                         [10., 11., 12., 13. , 14.],
                                                         [20., 21., 22., 23. , 24.],
                                                         [30., 31., 32., 33. , 34.],
                                                         [40., 41., 42., 43. , 44.]]>
  func private @print_memref_f32(memref<*xf32>)

  func @main() {
    %mem = memref.get_global @gv : memref<5x5xf32>

    %offset = arith.constant 0 : index
    %sub_size = arith.constant 3 : index
    %stride = arith.constant 2 : index

    %result = memref.subview %mem[%offset, %offset] [%sub_size, %sub_size] [%stride, %stride]
        : memref<5x5xf32> to memref<?x?xf32, #map0>

    // Print output.
    %print_output = memref.cast %result : memref<?x?xf32, #map0> to memref<*xf32>
    call @print_memref_f32(%print_output) : (memref<*xf32>) -> ()

    return
  }
}
