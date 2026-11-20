// XtameBench: linalg.matmul on i8 inputs with an i32 accumulator, M=32 K=64 N=32.
// Too many column blocks for the interchange path, so the column loop stays outer.
module {
  func.func @matmul_32x64x32(%a: memref<32x64xi8>, %b: memref<64x32xi8>,
                             %c: memref<32x32xi32>) {
    linalg.matmul ins(%a, %b : memref<32x64xi8>, memref<64x32xi8>)
                  outs(%c : memref<32x32xi32>)
    return
  }
}
