func.func private @printMemrefF32(memref<*xf32>)

func.func @test(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>) {
    linalg.matmul_transpose_b 
      ins(%a, %b: memref<?x?xf32>, memref<?x?xf32>)
      outs(%c: memref<?x?xf32>)
    return
  }

func.func @alloc_f32(%arg0: index, %arg1: index, %arg4: f32) -> memref<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
  scf.for %idx0 = %c0 to %arg0 step %c1 {
    scf.for %idx1 = %c0 to %arg1 step %c1 {
        memref.store %arg4, %0[%idx0, %idx1] : memref<?x?xf32>
    }
  }
  return %0 : memref<?x?xf32>
}


func.func @main(){
  %c32 = arith.constant 32 : index
  %c1024 = arith.constant 1024 : index
  %f0 = arith.constant 0.0 : f32
  %f1 = arith.constant 1.0 : f32

  %m0 = call @alloc_f32(%c32,%c1024, %f1) : (index, index, f32) -> memref<?x?xf32>
  %m1 = call @alloc_f32(%c32,%c1024, %f1) : (index, index, f32) -> memref<?x?xf32>
  %m2 = call @alloc_f32(%c32,%c32, %f0) : (index, index, f32) -> memref<?x?xf32>

  call @test(%m0, %m1, %m2) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()

  %printed_m2 = memref.cast %m2 : memref<?x?xf32> to memref<*xf32>

  call @printMemrefF32(%printed_m2) : (memref<*xf32>) -> ()

  return
}
