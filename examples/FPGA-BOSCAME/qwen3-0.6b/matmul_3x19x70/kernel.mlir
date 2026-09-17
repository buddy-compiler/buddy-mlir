// Signed W8A8 deployment dot product; raw NR i32 accumulator ABI.
// B is stored [N,K], exposed as logical [K,N] without a data transpose.
module {
  func.func @kernel_matmul_3x19x70(%a: memref<3x70xi8>,
      %b: memref<70x19xi8, strided<[1, 70]>>,
      %c: memref<3x19xi32>) attributes {llvm.emit_c_interface} {
    linalg.matmul ins(%a, %b : memref<3x70xi8>, memref<70x19xi8, strided<[1, 70]>>)
      outs(%c : memref<3x19xi32>)
    return
  }
}
