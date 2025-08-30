module{
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @matmul(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>) {
    linalg.matmul
      ins(%a, %b: memref<?x?xf32>, memref<?x?xf32>)
      outs(%c:memref<?x?xf32>)
    return
  }

  func.func @main(){
    // Set up dims.
    %cM = arith.constant 64 : index
    %cN = arith.constant 64 : index
    %cK = arith.constant 64 : index

    // Set Init Value.
    %cf1 = arith.constant 1.0 : f32

    %A = memref.alloc(%cM, %cK) : memref<?x?xf32>
    %B = memref.alloc(%cK, %cN) : memref<?x?xf32>
    %C = memref.alloc(%cM, %cN) : memref<?x?xf32>

    linalg.fill
    ins(%cf1 : f32)
    outs(%A:memref<?x?xf32>)

    linalg.fill
    ins(%cf1 : f32)
    outs(%B:memref<?x?xf32>)

    linalg.fill
    ins(%cf1 : f32)
    outs(%C:memref<?x?xf32>)

    call @matmul(%A, %B, %C) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()

    %print_C = memref.cast %C : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_C) : (memref<*xf32>) -> ()

    memref.dealloc %C : memref<?x?xf32>
    memref.dealloc %B : memref<?x?xf32>
    memref.dealloc %A : memref<?x?xf32>
    return 
  }
}
