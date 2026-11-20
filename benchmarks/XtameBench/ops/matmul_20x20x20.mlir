// XtameBench: linalg.matmul on i8 inputs with an i32 accumulator, M=20 K=20 N=20.
// Five column blocks and five 4-element K chunks: the interchange path.
module {
  func.func @matmul_20x20x20(%a: memref<20x20xi8>, %b: memref<20x20xi8>,
                             %c: memref<20x20xi32>) {
    linalg.matmul ins(%a, %b : memref<20x20xi8>, memref<20x20xi8>)
                  outs(%c : memref<20x20xi32>)
    return
  }
}
