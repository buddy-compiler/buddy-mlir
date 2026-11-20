// XtameBench: linalg.batch_matmul on i8 inputs with an i32 accumulator.
// Batch 4, M=8 K=16 N=8. The XTAME lowering does not handle the batch dimension
// yet, so the driver reports this case as no-xtame.
module {
  func.func @batch_matmul_4x8x16x8(%a: memref<4x8x16xi8>,
                                   %b: memref<4x16x8xi8>,
                                   %c: memref<4x8x8xi32>) {
    linalg.batch_matmul ins(%a, %b : memref<4x8x16xi8>, memref<4x16x8xi8>)
                        outs(%c : memref<4x8x8xi32>)
    return
  }
}
