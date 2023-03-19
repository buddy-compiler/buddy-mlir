memref.global "private" @gv : memref<4x4xf32> = dense<[[1., 2., 3., 4.],
                                                       [5., 6., 7., 8.],
                                                       [9., 10., 11., 12.],
                                                       [13., 14., 15., 16.]]>

func.func private@printMemrefF32(memref<*xf32>)

func.func @main() {
  %cf1 = arith.constant 1.0 : f32 
  %a = memref.alloc() : memref<4x4xf32> 
  %b = memref.get_global @gv : memref<4x4xf32> 
  %c = memref.alloc() : memref<4x4xf32>
  linalg.fill 
    ins(%cf1 : f32)
  outs(%a:memref<4x4xf32>)
  linalg.matmul_transpose_b 
    ins(%a, %b) 
  outs(%c:memref<4x4xf32>)
  %castC = memref.cast %c : memref<4x4xf32> to memref<*xf32> 
  call @printMemrefF32(%castC) : (memref<*xf32>) -> ()
  memref.dealloc %a : memref<4x4xf32> 
  memref.dealloc %b : memref<4x4xf32> 
  memref.dealloc %c : memref<4x4xf32> 
  return 
}