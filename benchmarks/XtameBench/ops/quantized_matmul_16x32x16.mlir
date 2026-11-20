// XtameBench: linalg.quantized_matmul on i8 inputs with an i32 accumulator and
// i32 zero points. M=16 K=32 N=16. The XTAME lowering does not handle the zero
// point operands yet, so the driver reports this case as no-xtame.
module {
  func.func @quantized_matmul_16x32x16(%a: memref<16x32xi8>,
                                       %b: memref<32x16xi8>,
                                       %a_zero: i32, %b_zero: i32,
                                       %c: memref<16x16xi32>) {
    linalg.quantized_matmul ins(%a, %b, %a_zero, %b_zero :
                                  memref<16x32xi8>, memref<32x16xi8>, i32, i32)
                            outs(%c : memref<16x16xi32>)
    return
  }
}
