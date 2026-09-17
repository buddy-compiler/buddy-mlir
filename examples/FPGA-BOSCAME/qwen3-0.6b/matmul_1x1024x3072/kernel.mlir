// Signed W8A8 deployment dot product; raw NR i32 accumulator ABI.
// B is stored [N,K], exposed as logical [K,N] without a data transpose.
module {
  func.func @kernel_matmul_1x1024x3072(%a: memref<1x3072xi8>,
      %b: memref<3072x1024xi8, strided<[1, 3072]>>,
      %c: memref<1x1024xi32>) attributes {llvm.emit_c_interface} {
    linalg.matmul ins(%a, %b : memref<1x3072xi8>, memref<3072x1024xi8, strided<[1, 3072]>>)
      outs(%c : memref<1x1024xi32>)
    return
  }
}
