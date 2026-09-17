// Signed W8A8 deployment dot product; raw NR i32 accumulator ABI.
// B is stored [N,K], exposed as logical [K,N] without a data transpose.
module {
  func.func @kernel_matmul_16x1024x2048(%a: memref<16x2048xi8>,
      %b: memref<2048x1024xi8, strided<[1, 2048]>>,
      %c: memref<16x1024xi32>) attributes {llvm.emit_c_interface} {
    linalg.matmul ins(%a, %b : memref<16x2048xi8>, memref<2048x1024xi8, strided<[1, 2048]>>)
      outs(%c : memref<16x1024xi32>)
    return
  }
}
