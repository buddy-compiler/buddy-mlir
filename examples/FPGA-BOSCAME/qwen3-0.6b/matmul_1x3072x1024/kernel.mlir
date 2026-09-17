// Signed W8A8 deployment dot product; raw NR i32 accumulator ABI.
// B is stored [N,K], exposed as logical [K,N] without a data transpose.
module {
  func.func @kernel_matmul_1x3072x1024(%a: memref<1x1024xi8>,
      %b: memref<1024x3072xi8, strided<[1, 1024]>>,
      %c: memref<1x3072xi32>) attributes {llvm.emit_c_interface} {
    linalg.matmul ins(%a, %b : memref<1x1024xi8>, memref<1024x3072xi8, strided<[1, 1024]>>)
      outs(%c : memref<1x3072xi32>)
    return
  }
}
