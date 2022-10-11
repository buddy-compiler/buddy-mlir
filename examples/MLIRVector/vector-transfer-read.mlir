#map0 = affine_map<(d0, d1) -> (d1, d0)>
memref.global "private" @gv : memref<4x4xf32> = dense<[[0. , 1. , 2. , 3. ],
                                                       [10., 11., 12., 13.],
                                                       [20., 21., 22., 23.],
                                                       [30., 31., 32., 33.]]>

func.func @main() -> (i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index  
  %f0 = arith.constant 0. : f32
  %f1 = arith.constant 1. : f32
  %cst0 = arith.constant 0 : i32
  %mem = memref.get_global @gv : memref<4x4xf32>
  %v0 = vector.transfer_read %mem[%c0, %c0], %f0 { permutation_map = #map0 } : memref<4x4xf32>, vector<4x4xf32>
  %v1 = vector.transfer_read %mem[%c1, %c1], %f0 { permutation_map = #map0 } : memref<4x4xf32>, vector<2x3xf32>
  %v2 = vector.transfer_read %mem[%c0, %c0], %f0 {} : memref<4x4xf32>, vector<5x5xf32>
  %v3 = vector.transfer_read %mem[%c0, %c0], %f1 {} : memref<4x4xf32>, vector<5x5xf32>
  vector.print %v0 : vector<4x4xf32>
  vector.print %v1 : vector<2x3xf32>
  vector.print %v2 : vector<5x5xf32>
  vector.print %v3 : vector<5x5xf32>
  return  %cst0 : i32
}

