memref.global "private" @gv_f32 : memref<16xf32> = dense<[0. , 1. , 2. , 3. , 4. , 5. , 6. , 7. , 8. , 9. ,
                                                          10., 11., 12., 13., 14., 15.]>

func.func private @printMemrefF32(memref<*xf32>)

func.func @alloc_mem_f32() -> memref<16xf32> {
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %mem = memref.alloc() : memref<16xf32>
  %dim = memref.dim %mem, %c0 : memref<16xf32>
  scf.for %idx = %c0 to %dim step %c1 {
    memref.store %f0, %mem[%idx] : memref<16xf32>
  }
  return %mem : memref<16xf32>
}

func.func @main() -> i32 {
  %mem_f32 = memref.get_global @gv_f32 : memref<16xf32>
  %dst = call @alloc_mem_f32() : () -> memref<16xf32>

  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %size = arith.constant 16 : index
  %vs = vector.vscale
  vector.print %vs : index
  %step = arith.muli %c4, %vs : index

  // %step is a multiple of `vscale`
  scf.for %i0 = %c0 to %size step %step {
    %0 = vector.load %mem_f32[%i0] : memref<16xf32>, vector<[4]xf32>
    vector.store %0, %dst[%i0] : memref<16xf32>, vector<[4]xf32>
  }

  %print_dst = memref.cast %dst : memref<16xf32> to memref<*xf32>
  call @printMemrefF32(%print_dst) : (memref<*xf32>) -> ()

  %ret = arith.constant 0 : i32
  return %ret : i32
}
