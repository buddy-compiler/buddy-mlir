module {
  memref.global "private" @matA : memref<4x8xi8> = dense<[[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]]>
  memref.global "private" @matB : memref<8x4xi8> = dense<1>
  func.func @main() -> i32 {
    %0 = memref.get_global @matA : memref<4x8xi8>
    %1 = memref.get_global @matB : memref<8x4xi8>
    %alloc = memref.alloc() : memref<4x4xi32>
    %c0_i32 = arith.constant 0 : i32
    linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<4x4xi32>)
    ime.vmadot %alloc, %0, %1 : memref<4x4xi32>, memref<4x8xi8>, memref<8x4xi8>
    %c0_i32_0 = arith.constant 0 : i32
    return %c0_i32_0 : i32
  }
}

