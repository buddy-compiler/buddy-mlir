// XtameBench: linalg.matmul on i8 inputs with an i32 accumulator, M=5 K=3 N=2.
// Every extent is a remainder: one row tile of 5, a 3-element K tail and a
// 2-column N tail, so a wrong tile extent shows up as a wrong element.
module {
  func.func @matmul_5x3x2(%a: memref<5x3xi8>, %b: memref<3x2xi8>,
                          %c: memref<5x2xi32>) {
    linalg.matmul ins(%a, %b : memref<5x3xi8>, memref<3x2xi8>)
                  outs(%c : memref<5x2xi32>)
    return
  }
}
