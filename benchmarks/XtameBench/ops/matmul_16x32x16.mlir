// XtameBench: linalg.matmul on i8 inputs with an i32 accumulator, M=16 K=32 N=16.
module {
  func.func @matmul_16x32x16(%a: memref<16x32xi8>, %b: memref<32x16xi8>,
                             %c: memref<16x16xi32>) {
    linalg.matmul ins(%a, %b : memref<16x32xi8>, memref<32x16xi8>)
                  outs(%c : memref<16x16xi32>)
    return
  }
}
