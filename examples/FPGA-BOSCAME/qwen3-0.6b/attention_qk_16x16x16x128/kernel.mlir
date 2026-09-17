// FP32 attention, no integer quantization of probabilities or Q/K/V.
module {
  func.func @kernel_attention_qk_16x16x16x128(%a: memref<16x16x128xf32>, %b: memref<16x128x16xf32>, %c: memref<16x16x16xf32>) attributes {llvm.emit_c_interface} {
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%c : memref<16x16x16xf32>)
    linalg.batch_matmul ins(%a, %b : memref<16x16x128xf32>, memref<16x128x16xf32>) outs(%c : memref<16x16x16xf32>)
    return
  }
}
