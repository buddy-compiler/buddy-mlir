// FP32 linear operator; Buddy transpose-B RVV lowering, no quantization.
// Physical [N,K] B permits contiguous K-vector loads without a copy.
module {
  func.func @kernel_matmul_16x2048x1024_f32(%a: memref<16x1024xf32>,
      %b: memref<2048x1024xf32>,
      %c: memref<16x2048xf32>) attributes {llvm.emit_c_interface} {
    linalg.matmul indexing_maps = [affine_map<(m,n,k)->(m,k)>, affine_map<(m,n,k)->(n,k)>, affine_map<(m,n,k)->(m,n)>] ins(%a, %b : memref<16x1024xf32>, memref<2048x1024xf32>)
      outs(%c : memref<16x2048xf32>)
    return
  }
}
