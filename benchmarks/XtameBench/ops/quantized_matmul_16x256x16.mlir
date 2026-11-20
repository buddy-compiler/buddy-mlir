// XtameBench: linalg.quantized_matmul on i8 inputs with an i32 accumulator and
// i32 zero points, M=16 K=256 N=16. K is eight times larger than in
// quantized_matmul_16x32x16, which separates the K-proportional part of the
// zero-point correction from the O(M*N) part.
module {
  func.func @quantized_matmul_16x256x16(%a: memref<16x256xi8>,
                                        %b: memref<256x16xi8>,
                                        %a_zero: i32, %b_zero: i32,
                                        %c: memref<16x16xi32>) {
    linalg.quantized_matmul ins(%a, %b, %a_zero, %b_zero :
                                  memref<16x256xi8>, memref<256x16xi8>, i32, i32)
                            outs(%c : memref<16x16xi32>)
    return
  }
}
