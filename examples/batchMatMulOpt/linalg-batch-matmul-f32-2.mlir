func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

func.func @conv_2d(%arg0: memref<1x1x1024xf32>, %arg1: memref<1x1024x1000xf32>, %arg2: memref<1x1x1000xf32>) {
  linalg.batch_matmul 
    ins(%arg0, %arg1 : memref<1x1x1024xf32>, memref<1x1024x1000xf32>) 
    outs(%arg2 : memref<1x1x1000xf32>)
  return
}

func.func @main(){

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c576 = arith.constant 576 : index
  %c1024 = arith.constant 1024 : index
  %c1000 = arith.constant 1000 : index
  %f0 = arith.constant 0.0 : f32
  %f2 = arith.constant 2.0 : f32
  %f3 = arith.constant 3.0 : f32

  %a = memref.alloc() : memref<1x1x1024xf32>
  scf.for %arg0 = %c0 to %c1 step %c1 {
    scf.for %arg1 = %c0 to %c1 step %c1 {
      scf.for %arg2 = %c0 to %c1024 step %c1 {
        memref.store %f3, %a[%arg0, %arg1, %arg2] : memref<1x1x1024xf32>
      }
    }
  }

  %b = memref.alloc() : memref<1x1024x1000xf32>
  scf.for %arg0 = %c0 to %c1 step %c1 {
    scf.for %arg1 = %c0 to %c1024 step %c1 {
      scf.for %arg2 = %c0 to %c1000 step %c1 {
        memref.store %f2, %b[%arg0, %arg1, %arg2] : memref<1x1024x1000xf32>
      }
    }
  }

  %c = memref.alloc() : memref<1x1x1000xf32>
  scf.for %arg0 = %c0 to %c1 step %c1 {
    scf.for %arg1 = %c0 to %c1 step %c1 {
      scf.for %arg2 = %c0 to %c1000 step %c1 {
        memref.store %f0, %c[%arg0, %arg1, %arg2] : memref<1x1x1000xf32>
      }
    }
  }
  
  linalg.batch_matmul 
    ins(%a, %b : memref<1x1x1024xf32>, memref<1x1024x1000xf32>) 
    outs(%c : memref<1x1x1000xf32>)

  %printed_c = memref.cast %c : memref<1x1x1000xf32> to memref<*xf32>
  call @printMemrefF32(%printed_c) : (memref<*xf32>) -> ()
  return
}
