// XtameBench: linalg.matmul on i8 inputs with an i32 accumulator, M=8 K=20 N=8.
// K is larger than one 16-byte register row, so the reduction is chunked.
module {
  func.func @matmul_8x20x8(%a: memref<8x20xi8>, %b: memref<20x8xi8>,
                           %c: memref<8x8xi32>) {
    linalg.matmul ins(%a, %b : memref<8x20xi8>, memref<20x8xi8>)
                  outs(%c : memref<8x8xi32>)
    return
  }
}
