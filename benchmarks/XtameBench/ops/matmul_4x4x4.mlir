// XtameBench: linalg.matmul on i8 inputs with an i32 accumulator, M=4 K=4 N=4.
module {
  func.func @matmul_4x4x4(%a: memref<4x4xi8>, %b: memref<4x4xi8>,
                          %c: memref<4x4xi32>) {
    linalg.matmul ins(%a, %b : memref<4x4xi8>, memref<4x4xi8>)
                  outs(%c : memref<4x4xi32>)
    return
  }
}
