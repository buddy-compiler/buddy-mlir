// XtameBench: linalg.matmul on i8 inputs with an i32 accumulator, M=33 K=20 N=13.
// Exercises the remainder of the row (16+16+1), column (4+4+4+1) and K partition.
module {
  func.func @matmul_33x20x13(%a: memref<33x20xi8>, %b: memref<20x13xi8>,
                             %c: memref<33x13xi32>) {
    linalg.matmul ins(%a, %b : memref<33x20xi8>, memref<20x13xi8>)
                  outs(%c : memref<33x13xi32>)
    return
  }
}
