// XtameBench: linalg.matmul on i8 inputs with an i32 accumulator, M=16 K=256
// N=16. The plain matmul counterpart of quantized_matmul_16x256x16, used to
// compare the zero-point correction against the matrix product it corrects.
module {
  func.func @matmul_16x256x16(%a: memref<16x256xi8>, %b: memref<256x16xi8>,
                              %c: memref<16x16xi32>) {
    linalg.matmul ins(%a, %b : memref<16x256xi8>, memref<256x16xi8>)
                  outs(%c : memref<16x16xi32>)
    return
  }
}
