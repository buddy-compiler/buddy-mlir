// Signed W8A8 deployment dot product; raw NR i32 accumulator ABI.
// B is stored [N,K], exposed as logical [K,N] without a data transpose.
module {
  func.func @kernel_matmul_1x2048x1024(%a: memref<1x1024xi8>,
      %b: memref<1024x2048xi8, strided<[1, 1024]>>,
      %c: memref<1x2048xi32>) attributes {llvm.emit_c_interface} {
    linalg.matmul ins(%a, %b : memref<1x1024xi8>, memref<1024x2048xi8, strided<[1, 1024]>>)
      outs(%c : memref<1x2048xi32>)
    return
  }
}
