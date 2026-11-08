// XtameBench: the smallest shape the pass accepts, linalg.matmul on i8 inputs
// with an i32 accumulator, M=4 K=1 N=1. K is below one 4-element chunk and N is
// a single column, so the whole reduction and the store are tail paths.
module {
  func.func @matmul_4x1x1(%a: memref<4x1xi8>, %b: memref<1x1xi8>,
                          %c: memref<4x1xi32>) {
    linalg.matmul ins(%a, %b : memref<4x1xi8>, memref<1x1xi8>)
                  outs(%c : memref<4x1xi32>)
    return
  }
}
